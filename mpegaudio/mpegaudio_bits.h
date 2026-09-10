/*
	Bitstream reader and Huffman decoder for the MPEG audio port.

	Replaces libavcodec/get_bits.h and libavcodec/vlc.c with the parts this decoder
	actually uses. Both are deliberately small:

	  - GetBits is MSB-first with an explicit end, and reads past the end return zero
	    bits. FFmpeg gets that behavior from AV_INPUT_BUFFER_PADDING_SIZE on the input
	    buffer; we pad the decoder's own frame buffers instead, and clamp anyway.

	  - The Huffman tables are built as a two-level lookup rather than through
	    FFmpeg's VLC machinery. The code ASSIGNMENT still has to match FFmpeg exactly,
	    because the tables are stored as code LENGTHS: ff_vlc_init_from_lengths walks
	    the length list in order with a 32-bit left-justified accumulator, giving each
	    symbol the top `len` bits of the accumulator and then advancing it by
	    1 << (32 - len). Lengths of zero are skipped without advancing. That rule is
	    reproduced verbatim in vlc_build(); the codes it produces are NOT contiguous
	    per length, which is why this is a prefix-tree lookup rather than the usual
	    canonical first-code/count decoder.
*/

#ifndef MPEGAUDIO_BITS_H
#define MPEGAUDIO_BITS_H

#include <stdint.h>
#include <string.h>

// ─── bit reader ──────────────────────────────────────────────────────────────

struct GetBits {
	const uint8_t* buf;
	int size_in_bits;
	int index;
};

static inline void init_get_bits(GetBits* g, const uint8_t* buf, int bit_size)
{
	g->buf = buf;
	g->size_in_bits = bit_size;
	g->index = 0;
}

static inline void init_get_bits8(GetBits* g, const uint8_t* buf, int byte_size)
{
	init_get_bits(g, buf, byte_size * 8);
}

static inline int get_bits_count(const GetBits* g) { return g->index; }
static inline int get_bits_left(const GetBits* g) { return g->size_in_bits - g->index; }
static inline void skip_bits_long(GetBits* g, int n) { g->index += n; }
static inline void align_get_bits(GetBits* g) { g->index = (g->index + 7) & ~7; }

/* Next 32 bits, MSB-first, zero-padded past the end (never reads out of bounds). */
static inline uint32_t peek32(const GetBits* g)
{
	const int idx = g->index;
	if (idx < 0 || idx >= g->size_in_bits) return 0;

	const int total_bytes = (g->size_in_bits + 7) >> 3;
	const int byte = idx >> 3;
	const int sh   = idx & 7;

	uint8_t t[5];
	for (int i = 0; i < 5; i++) {
		const int b = byte + i;
		t[i] = (b < total_bytes) ? g->buf[b] : 0;
	}
	uint32_t v = ((uint32_t)t[0] << 24) | ((uint32_t)t[1] << 16) |
	             ((uint32_t)t[2] << 8)  |  (uint32_t)t[3];
	if (sh) v = (v << sh) | (uint32_t)(t[4] >> (8 - sh));
	return v;
}

static inline unsigned show_bits(const GetBits* g, int n)
{
	if (n <= 0) return 0;
	return peek32(g) >> (32 - n);
}

static inline unsigned get_bits(GetBits* g, int n)
{
	if (n <= 0) return 0;
	unsigned v = peek32(g) >> (32 - n);
	g->index += n;
	return v;
}

static inline unsigned get_bits1(GetBits* g)
{
	const int idx = g->index;
	unsigned bit = 0;
	if (idx < g->size_in_bits)
		bit = (g->buf[idx >> 3] >> (7 - (idx & 7))) & 1;
	g->index = idx + 1;
	return bit;
}

static inline void skip_bits(GetBits* g, int n) { g->index += n; }

/* get_bitsz: n may be 0, in which case nothing is consumed (used for linbits). */
static inline unsigned get_bitsz(GetBits* g, int n) { return n ? get_bits(g, n) : 0; }

// ─── Huffman ────────────────────────────────────────────────────────────────

#define VLC_ROOT_BITS 9
#define VLC_MAX_BITS  24

/*
	Two-level lookup. A root entry is either

	  leaf:     bit31 clear, length in bits 0..4, symbol in bits 8..23
	  subtable: bit31 set,   sub-table bit count in bits 20..24, offset in bits 0..19

	A sub-table is indexed by the bits following the root prefix and holds leaves in
	the same encoding (the length stored is the TOTAL code length).
*/
struct Vlc {
	uint32_t root[1 << VLC_ROOT_BITS];
	uint32_t* sub;
	int sub_size;
};

#define VLC_LEAF(len, sym) (((uint32_t)(sym) << 8) | (uint32_t)(len))
#define VLC_IS_SUB(e)      ((e) & 0x80000000u)
#define VLC_SUB_BITS(e)    (((e) >> 20) & 0x1F)
#define VLC_SUB_OFF(e)     ((e) & 0xFFFFF)
#define VLC_LEN(e)         ((e) & 0x1F)
#define VLC_SYM(e)         (((e) >> 8) & 0xFFFF)

/**
 * Build a VLC from a length list, assigning codes exactly as
 * ff_vlc_init_from_lengths does.
 *
 * `sub_pool` is caller-owned scratch for the second level; sub_pool_size counts
 * uint32_t entries. Returns 0 on success, -1 if the pool was too small.
 */
static int vlc_build(Vlc* v, const uint8_t* lens, const uint16_t* syms, int nb_codes,
                     uint32_t* sub_pool, int sub_pool_size)
{
	uint32_t codes[600];
	uint8_t  lengths[600];
	int n = 0;

	if (nb_codes > 600) return -1;

	// FFmpeg's canonical assignment: top `len` bits of a left-justified accumulator.
	uint64_t code = 0;
	for (int i = 0; i < nb_codes; i++) {
		int len = lens[i];
		if (len <= 0) continue;
		if (len > VLC_MAX_BITS) return -1;
		codes[n]   = (uint32_t)(code >> (32 - len)) << (32 - len);
		lengths[n] = (uint8_t)len;
		n++;
		code += 1ULL << (32 - len);
	}

	for (int i = 0; i < (1 << VLC_ROOT_BITS); i++) v->root[i] = VLC_LEAF(0, 0);
	v->sub = sub_pool;
	v->sub_size = 0;

	// Level 1: everything that fits in VLC_ROOT_BITS.
	int k = 0;
	for (int i = 0; i < nb_codes; i++) {
		if (lens[i] <= 0) continue;
		const int len = lengths[k];
		const uint32_t c = codes[k];
		const uint16_t sym = syms ? syms[i] : (uint16_t)i;
		k++;
		if (len > VLC_ROOT_BITS) continue;
		const int prefix = c >> (32 - VLC_ROOT_BITS);
		const int fill = 1 << (VLC_ROOT_BITS - len);
		for (int j = 0; j < fill; j++) v->root[prefix + j] = VLC_LEAF(len, sym);
	}

	// Level 2: one sub-table per root prefix that has longer codes.
	for (int prefix = 0; prefix < (1 << VLC_ROOT_BITS); prefix++) {
		int maxlen = 0;
		k = 0;
		for (int i = 0; i < nb_codes; i++) {
			if (lens[i] <= 0) continue;
			const int len = lengths[k];
			const uint32_t c = codes[k];
			k++;
			if (len <= VLC_ROOT_BITS) continue;
			if ((int)(c >> (32 - VLC_ROOT_BITS)) != prefix) continue;
			if (len > maxlen) maxlen = len;
		}
		if (!maxlen) continue;

		const int sub_bits = maxlen - VLC_ROOT_BITS;
		const int sub_n = 1 << sub_bits;
		if (v->sub_size + sub_n > sub_pool_size) return -1;
		const int off = v->sub_size;
		v->sub_size += sub_n;
		for (int j = 0; j < sub_n; j++) v->sub[off + j] = VLC_LEAF(0, 0);

		k = 0;
		for (int i = 0; i < nb_codes; i++) {
			if (lens[i] <= 0) continue;
			const int len = lengths[k];
			const uint32_t c = codes[k];
			const uint16_t sym = syms ? syms[i] : (uint16_t)i;
			k++;
			if (len <= VLC_ROOT_BITS) continue;
			if ((int)(c >> (32 - VLC_ROOT_BITS)) != prefix) continue;
			const int idx = (c << VLC_ROOT_BITS) >> (32 - sub_bits);
			const int fill = 1 << (maxlen - len);
			for (int j = 0; j < fill; j++) v->sub[off + idx + j] = VLC_LEAF(len, sym);
		}
		v->root[prefix] = 0x80000000u | ((uint32_t)sub_bits << 20) | (uint32_t)off;
	}
	return 0;
}

/**
 * Build a VLC from EXPLICIT codes plus lengths, for the two Layer III count1
 * ("quad") tables. FFmpeg initializes those with vlc_init() from mpa_quad_codes /
 * mpa_quad_bits rather than from lengths alone, and the symbol is the table index.
 * Codes are right-justified in the source table, so they are shifted up here.
 */
static int vlc_build_codes(Vlc* v, const uint8_t* lens, const uint8_t* codes, int nb_codes,
                           uint32_t* sub_pool, int sub_pool_size)
{
	for (int i = 0; i < (1 << VLC_ROOT_BITS); i++) v->root[i] = VLC_LEAF(0, 0);
	v->sub = sub_pool;
	v->sub_size = 0;

	for (int i = 0; i < nb_codes; i++) {
		const int len = lens[i];
		if (len <= 0) continue;
		if (len > VLC_ROOT_BITS) return -1;	// quad codes are <= 6 bits
		const int prefix = ((uint32_t)codes[i] << (VLC_ROOT_BITS - len)) & ((1 << VLC_ROOT_BITS) - 1);
		const int fill = 1 << (VLC_ROOT_BITS - len);
		for (int j = 0; j < fill; j++) v->root[prefix + j] = VLC_LEAF(len, i);
	}
	(void)sub_pool_size;
	return 0;
}

/**
 * Decode one symbol. Consumes the code's bits. An invalid code (a hole in the
 * table, only reachable on corrupt input) consumes VLC_ROOT_BITS and returns 0,
 * which is what FFmpeg's get_vlc2 effectively does for an unmatched prefix.
 */
static inline int get_vlc(GetBits* g, const Vlc* v)
{
	const uint32_t bits = peek32(g);
	uint32_t e = v->root[bits >> (32 - VLC_ROOT_BITS)];

	if (VLC_IS_SUB(e)) {
		const int sub_bits = VLC_SUB_BITS(e);
		const int off = VLC_SUB_OFF(e);
		const uint32_t idx = (bits << VLC_ROOT_BITS) >> (32 - sub_bits);
		e = v->sub[off + idx];
	}
	const int len = VLC_LEN(e);
	if (!len) { g->index += VLC_ROOT_BITS; return 0; }
	g->index += len;
	return (int)VLC_SYM(e);
}

#endif /* MPEGAUDIO_BITS_H */
