/*
	Vorbis I audio decoder -- WASM C++.

	A port of FFmpeg n8.1's Vorbis decoder (LGPL-2.1+): libavcodec/vorbisdec.c, vorbis.c,
	vorbis_data.c and the C path of vorbisdsp.c, with vector_fmul_window from libavutil/float_dsp.c.
	The headers, the codebooks, floors 0 and 1, residues 0, 1 and 2, the inverse coupling, the
	overlap and the channel order are FFmpeg's; the libavcodec framework around them (AVCodecContext,
	AVFrame, the DSP pointers) is replaced by one handle that owns planar float output.

	A packet is one whole Vorbis packet: vorbis_decode_packet gives the samples the packet completes,
	(previous block + this block) / 4 of them, as FFmpeg's decoder does. It drops nothing, so the first
	packet's output is there and the container drops it, where FFmpeg's avctx->internal->skip_samples
	does.

	Departures from FFmpeg:
	  - The IMDCT runs its own complex FFT, in double, in place of libavutil/tx; the twiddles and the
	    pre- and post-rotation follow ff_tx_mdct_inv and ff_tx_mdct_gen_exp for a scale of -1. FFmpeg
	    computes in float, so the PCM differs from `ffmpeg -f f32le` by float round-off.
	  - The VLCs are built here rather than by ff_vlc_init, with the same table bits. A symbol keeps all
	    its 16 bits, where FFmpeg's int16_t table takes one past 32767 for an invalid code.
	  - After a reset, a long block takes the previous block's size from its previous_window flag, the
	    first of its two bits. FFmpeg takes the second, next_window, there. Only the first packet of a
	    stream or a seek's preroll sees this, and neither is heard; with the flag, a seek's preroll
	    gives the sample counts of the whole read.
	  - A header packet inside the stream gives no samples and changes nothing: FFmpeg starts over
	    from an identification header there. The headers come once, to vorbis_set_headers.
	  - The extradata framing of the headers (avpriv_split_xiph_headers) is gone.

	-fwrapv is REQUIRED, as in FFmpeg's build.
*/

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include <emscripten/emscripten.h>

#include "vorbis_tables.h"

#define V_NB_BITS 8
#define V_NB_BITS2 11
#define V_MAX_VLCS (1 << 16)
#define V_MAX_PARTITIONS (1 << 20)
#define VORBIS_MAX_BLOCK_LENGTH 8192
#define MAX_VLC_DEPTH 3
// Bytes the input must hold free past its end: a bit read loads 8.
#define INPUT_PADDING 16

#define ERROR_INVALID (-1)
#define ERROR_MEMORY (-2)

///////////////////////////////////////////////////////////////////////////////
// Bit reader: libavcodec/get_bits.h with BITSTREAM_READER_LE, checked at the end of the packet.

struct GetBitContext {
	const uint8_t* buffer;
	int index;
	int size_in_bits;
	int size_in_bits_plus8;
};

static inline void init_get_bits(GetBitContext* gb, const uint8_t* buffer, int bytes)
{
	gb->buffer = buffer;
	gb->index = 0;
	gb->size_in_bits = bytes * 8;
	gb->size_in_bits_plus8 = bytes * 8 + 8;
}

static inline uint32_t load_le32(const uint8_t* p)
{
	uint32_t value;
	memcpy(&value, p, 4);
	return value;
}

static inline uint32_t show_bits(const GetBitContext* gb, int n)
{
	if (!n) return 0;
	return (load_le32(gb->buffer + (gb->index >> 3)) >> (gb->index & 7)) & (0xFFFFFFFFu >> (32 - n));
}

static inline void skip_bits(GetBitContext* gb, int n)
{
	const int index = gb->index + n;
	gb->index = index < gb->size_in_bits_plus8 ? index : gb->size_in_bits_plus8;
}

/** At most 25 bits. */
static inline unsigned get_bits(GetBitContext* gb, int n)
{
	const unsigned value = show_bits(gb, n);
	skip_bits(gb, n);
	return value;
}

static inline unsigned get_bits1(GetBitContext* gb) { return get_bits(gb, 1); }

static inline uint32_t get_bits_long(GetBitContext* gb, int n)
{
	if (n <= 25) return get_bits(gb, n);
	const uint32_t low = get_bits(gb, 16);
	return low | (uint32_t)get_bits(gb, n - 16) << 16;
}

static inline uint64_t get_bits64(GetBitContext* gb, int n)
{
	if (n <= 32) return get_bits_long(gb, n);
	const uint64_t low = get_bits_long(gb, 32);
	return low | (uint64_t)get_bits_long(gb, n - 32) << 32;
}

static inline int get_bits_left(const GetBitContext* gb) { return gb->size_in_bits - gb->index; }

static inline int av_log2(unsigned v) { return 31 - __builtin_clz(v | 1); }
static inline int ilog(unsigned i) { return av_log2(2 * i); }

static inline unsigned clip_uint8(int a) { return a & ~0xFF ? (~a >> 31) & 0xFF : a; }
static inline unsigned clip_uint16(int a) { return a & ~0xFFFF ? (~a >> 31) & 0xFFFF : a; }

///////////////////////////////////////////////////////////////////////////////
// VLC: a table of `bits` index bits per level, LE codes. A negative len is a subtable of -len bits at sym.

struct VLCElem { int32_t sym; int32_t len; };

struct VLC { VLCElem* table; int table_size; int allocated; };

struct VLCCode { uint32_t code; uint16_t symbol; uint8_t bits; };

static int vlc_alloc(VLC* vlc, int entries)
{
	const int index = vlc->table_size;
	if (index + entries > vlc->allocated) {
		int allocated = vlc->allocated ? vlc->allocated * 2 : 1024;
		while (allocated < index + entries) allocated *= 2;
		VLCElem* table = (VLCElem*)realloc(vlc->table, allocated * sizeof(VLCElem));
		if (!table) return ERROR_MEMORY;
		vlc->table = table;
		vlc->allocated = allocated;
	}
	vlc->table_size = index + entries;
	for (int i = 0; i < entries; i++) {
		vlc->table[index + i].sym = -1;
		vlc->table[index + i].len = 0;
	}
	return index;
}

/** `shift` low bits of each code are the levels above; `codes` come as compare_vlc_codes sorts them. */
static int vlc_build_level(VLC* vlc, VLCCode* codes, int count, int shift, int table_bits, int nb_bits)
{
	const int index = vlc_alloc(vlc, 1 << table_bits);
	if (index < 0) return index;
	const unsigned mask = (1u << table_bits) - 1;

	for (int i = 0; i < count;) {
		const int rest = codes[i].bits - shift;
		const unsigned low = (codes[i].code >> shift) & mask;
		if (rest <= table_bits) {
			for (unsigned j = low; j < (1u << table_bits); j += 1u << rest) {
				vlc->table[index + j].sym = codes[i].symbol;
				vlc->table[index + j].len = rest;
			}
			i++;
		} else {
			int end = i;
			int deepest = 0;
			while (end < count && codes[end].bits - shift > table_bits && ((codes[end].code >> shift) & mask) == low) {
				if (codes[end].bits - shift - table_bits > deepest) deepest = codes[end].bits - shift - table_bits;
				end++;
			}
			const int sub_bits = deepest < nb_bits ? deepest : nb_bits;
			const int sub = vlc_build_level(vlc, codes + i, end - i, shift + table_bits, sub_bits, nb_bits);
			if (sub < 0) return sub;
			vlc->table[index + low].sym = sub;
			vlc->table[index + low].len = -sub_bits;
			i = end;
		}
	}
	return index;
}

static int compare_vlc_codes(const void* a, const void* b)
{
	const VLCCode* x = (const VLCCode*)a;
	const VLCCode* y = (const VLCCode*)b;
	// By the bits read first, so that codes of one prefix are one run at every level.
	const uint32_t rx = __builtin_bitreverse32(x->code);
	const uint32_t ry = __builtin_bitreverse32(y->code);
	return rx < ry ? -1 : rx > ry ? 1 : 0;
}

static int vlc_init(VLC* vlc, int nb_bits, int count, const uint8_t* bits, const uint32_t* codes)
{
	VLCCode* list = (VLCCode*)malloc((count + 1) * sizeof(VLCCode));
	if (!list) return ERROR_MEMORY;
	int used = 0;
	for (int i = 0; i < count; i++) {
		if (bits[i]) {
			list[used].code = codes[i];
			list[used].symbol = (uint16_t)i;
			list[used].bits = bits[i];
			used++;
		}
	}
	qsort(list, used, sizeof(VLCCode), compare_vlc_codes);
	const int ret = vlc_build_level(vlc, list, used, 0, nb_bits, nb_bits);
	free(list);
	return ret < 0 ? ret : 0;
}

static inline int get_vlc2(GetBitContext* gb, const VLCElem* table, int bits)
{
	int index = show_bits(gb, bits);
	int code = table[index].sym;
	int n = table[index].len;
	while (n < 0) {
		skip_bits(gb, bits);
		bits = -n;
		index = show_bits(gb, bits) + code;
		code = table[index].sym;
		n = table[index].len;
	}
	skip_bits(gb, n);
	return code;
}

///////////////////////////////////////////////////////////////////////////////
// Context

struct vorbis_floor1_entry {
	uint16_t x;
	uint16_t sort;
	uint16_t low;
	uint16_t high;
};

struct vorbis_codebook {
	uint8_t dimensions;
	uint8_t lookup_type;
	uint8_t maxdepth;
	VLC vlc;
	float* codevectors;
	unsigned nb_bits;
};

struct vorbis_floor0 {
	uint8_t order;
	uint16_t rate;
	uint16_t bark_map_size;
	int32_t* map[2];
	uint32_t map_size[2];
	uint8_t amplitude_bits;
	uint8_t amplitude_offset;
	uint8_t num_books;
	uint8_t* book_list;
	float* lsp;
};

struct vorbis_floor1 {
	uint8_t partitions;
	uint8_t partition_class[32];
	uint8_t class_dimensions[16];
	uint8_t class_subclasses[16];
	uint8_t class_masterbook[16];
	int16_t subclass_books[16][8];
	uint8_t multiplier;
	uint16_t x_list_dim;
	vorbis_floor1_entry* list;
};

struct vorbis_floor {
	uint8_t floor_type;
	vorbis_floor0 t0;
	vorbis_floor1 t1;
};

struct vorbis_residue {
	uint16_t type;
	uint32_t begin;
	uint32_t end;
	unsigned partition_size;
	uint8_t classifications;
	uint8_t classbook;
	int16_t books[64][8];
	uint8_t maxpass;
	uint16_t ptns_to_read;
	uint8_t* classifs;
};

struct vorbis_mapping {
	uint8_t submaps;
	uint16_t coupling_steps;
	uint8_t* magnitude;
	uint8_t* angle;
	uint8_t* mux;
	uint8_t submap_floor[16];
	uint8_t submap_residue[16];
};

struct vorbis_mode {
	uint8_t blockflag;
	uint16_t windowtype;
	uint16_t transformtype;
	uint8_t mapping;
};

struct Cpx { double re, im; };

/** One block size's half IMDCT: `len` coefficients in, `len` samples out. */
struct Imdct {
	int len;
	Cpx* exp;
	Cpx* twiddles;
	Cpx* z;
};

struct vorbis_context {
	GetBitContext gb;

	Imdct mdct[2];

	uint32_t version;
	uint8_t audio_channels;
	uint32_t audio_samplerate;
	uint32_t blocksize[2];
	const float* win[2];
	uint16_t codebook_count;
	vorbis_codebook* codebooks;
	uint8_t floor_count;
	vorbis_floor* floors;
	uint8_t residue_count;
	vorbis_residue* residues;
	uint8_t mapping_count;
	vorbis_mapping* mappings;
	uint8_t mode_count;
	vorbis_mode* modes;
	uint8_t mode_number;	// mode number for the current packet
	int8_t previous_window;
	float* channel_residues;
	float* saved;
	float* output;			// blocksize[1] / 2 samples per channel, in FFmpeg's channel order
};

#define BARK(x) \
	(13.1f * atan((double)(0.00074f * (x))) + 2.24f * atan((double)(1.85e-8f * (x) * (x))) + 1e-4f * (x))

#define VALIDATE_INDEX(idx, limit) \
	if ((unsigned)(idx) >= (unsigned)(limit)) return ERROR_INVALID;
#define GET_VALIDATED_INDEX(idx, bits, limit) \
	{ idx = get_bits(gb, bits); VALIDATE_INDEX(idx, limit) }

static float vorbisfloat2float(unsigned val)
{
	float mant = val & 0x1fffff;
	const int exp = (val & 0x7fe00000) >> 21;
	if (val & 0x80000000) mant = -mant;
	return ldexpf(mant, exp - 20 - 768);
}

///////////////////////////////////////////////////////////////////////////////
// libavcodec/vorbis.c

// x^(1/n)
static unsigned ff_vorbis_nth_root(unsigned x, unsigned n)
{
	unsigned ret = 0, i, j;
	do {
		++ret;
		for (i = 0, j = ret; i < n - 1; i++) j *= ret;
	} while (j <= x);
	return ret - 1;
}

// Generate vlc codes from vorbis huffman code lengths
static int ff_vorbis_len2vlc(uint8_t* bits, uint32_t* codes, unsigned num)
{
	uint32_t exit_at_level[33] = { 404 };
	unsigned i, j, p, code;

	for (p = 0; (p < num) && (bits[p] == 0); ++p);
	if (p == num) return 0;

	codes[p] = 0;
	if (bits[p] > 32) return ERROR_INVALID;
	for (i = 0; i < bits[p]; ++i) exit_at_level[i + 1] = 1u << i;

	++p;

	for (i = p; (i < num) && (bits[i] == 0); ++i);
	if (i == num) return 0;

	for (; p < num; ++p) {
		if (bits[p] > 32) return ERROR_INVALID;
		if (bits[p] == 0) continue;
		// find corresponding exit(node which the tree can grow further from)
		for (i = bits[p]; i > 0; --i) {
			if (exit_at_level[i]) break;
		}
		if (!i) return ERROR_INVALID;	// overspecified tree
		code = exit_at_level[i];
		exit_at_level[i] = 0;
		// construct code (append 0s to end) and introduce new exits
		for (j = i + 1; j <= bits[p]; ++j) exit_at_level[j] = code + (1u << (j - 1));
		codes[p] = code;
	}

	// no exits should be left (underspecified tree - ie. unused valid vlcs - not allowed by SPEC)
	for (p = 1; p < 33; p++) {
		if (exit_at_level[p]) return ERROR_INVALID;
	}
	return 0;
}

static int ff_vorbis_ready_floor1_list(vorbis_floor1_entry* list, int values)
{
	list[0].sort = 0;
	list[1].sort = 1;
	for (int i = 2; i < values; i++) {
		list[i].low = 0;
		list[i].high = 1;
		list[i].sort = i;
		for (int j = 2; j < i; j++) {
			const int tmp = list[j].x;
			if (tmp < list[i].x) {
				if (tmp > list[list[i].low].x) list[i].low = j;
			} else {
				if (tmp < list[list[i].high].x) list[i].high = j;
			}
		}
	}
	for (int i = 0; i < values - 1; i++) {
		for (int j = i + 1; j < values; j++) {
			if (list[i].x == list[j].x) return ERROR_INVALID;
			if (list[list[i].sort].x > list[list[j].sort].x) {
				const int tmp = list[i].sort;
				list[i].sort = list[j].sort;
				list[j].sort = tmp;
			}
		}
	}
	return 0;
}

static inline void render_line_unrolled(intptr_t x, int y, int x1, intptr_t sy, int ady, int adx, float* buf)
{
	int err = -adx;
	x -= x1 - 1;
	buf += x1 - 1;
	while (++x < 0) {
		err += ady;
		if (err >= 0) {
			err += ady - adx;
			y += sy;
			buf[x++] = ff_vorbis_floor1_inverse_db_table[clip_uint8(y)];
		}
		buf[x] = ff_vorbis_floor1_inverse_db_table[clip_uint8(y)];
	}
	if (x <= 0) {
		if (err + ady >= 0) y += sy;
		buf[x] = ff_vorbis_floor1_inverse_db_table[clip_uint8(y)];
	}
}

static void render_line(int x0, int y0, int x1, int y1, float* buf)
{
	const int dy = y1 - y0;
	const int adx = x1 - x0;
	int ady = abs(dy);
	const int sy = dy < 0 ? -1 : 1;
	buf[x0] = ff_vorbis_floor1_inverse_db_table[clip_uint8(y0)];
	if (ady * 2 <= adx) {	// optimized common case
		render_line_unrolled(x0, y0, x1, sy, ady, adx, buf);
	} else {
		const int base = dy / adx;
		int x = x0;
		int y = y0;
		int err = -adx;
		ady -= abs(base) * adx;
		while (++x < x1) {
			y += base;
			err += ady;
			if (err >= 0) {
				err -= adx;
				y += sy;
			}
			buf[x] = ff_vorbis_floor1_inverse_db_table[clip_uint8(y)];
		}
	}
}

static void ff_vorbis_floor1_render_list(vorbis_floor1_entry* list, int values, uint16_t* y_list, int* flag,
	int multiplier, float* out, int samples)
{
	int lx = 0;
	int ly = y_list[0] * multiplier;
	for (int i = 1; i < values; i++) {
		const int pos = list[i].sort;
		if (flag[pos]) {
			const int x1 = list[pos].x;
			const int y1 = y_list[pos] * multiplier;
			if (lx < samples) render_line(lx, ly, x1 < samples ? x1 : samples, y1, out);
			lx = x1;
			ly = y1;
		}
		if (lx >= samples) break;
	}
	if (lx < samples) render_line(lx, ly, samples, ly, out);
}

///////////////////////////////////////////////////////////////////////////////
// IMDCT: ff_tx_mdct_inv over a radix-2 inverse FFT.

static int imdct_init(Imdct* mdct, int len)
{
	const int len2 = len >> 1;
	mdct->len = len;
	mdct->exp = (Cpx*)malloc(len2 * sizeof(Cpx));
	mdct->twiddles = (Cpx*)malloc(len2 * sizeof(Cpx));
	mdct->z = (Cpx*)malloc(len2 * sizeof(Cpx));
	if (!mdct->exp || !mdct->twiddles || !mdct->z) return ERROR_MEMORY;

	// ff_tx_mdct_gen_exp for a scale of -1: its root is 1, and a negative scale adds len4 to theta.
	const double theta = len2 + 1.0 / 8.0;
	for (int i = 0; i < len2; i++) {
		const double alpha = M_PI_2 * (i + theta) / len2;
		mdct->exp[i].re = cos(alpha);
		mdct->exp[i].im = sin(alpha);
		const double angle = 2.0 * M_PI * i / len2;
		mdct->twiddles[i].re = cos(angle);
		mdct->twiddles[i].im = sin(angle);
	}
	return 0;
}

static void imdct_free(Imdct* mdct)
{
	free(mdct->exp);
	free(mdct->twiddles);
	free(mdct->z);
}

static void fft_inverse(Cpx* z, const Cpx* twiddles, int n)
{
	for (int i = 1, j = 0; i < n; i++) {
		int bit = n >> 1;
		for (; j & bit; bit >>= 1) j ^= bit;
		j ^= bit;
		if (i < j) { const Cpx t = z[i]; z[i] = z[j]; z[j] = t; }
	}
	for (int size = 2; size <= n; size <<= 1) {
		const int half = size >> 1;
		const int stride = n / size;
		for (int i = 0; i < n; i += size) {
			for (int j = 0; j < half; j++) {
				const Cpx w = twiddles[j * stride];
				Cpx* a = &z[i + j];
				Cpx* b = &z[i + j + half];
				const double vr = b->re * w.re - b->im * w.im;
				const double vi = b->re * w.im + b->im * w.re;
				b->re = a->re - vr;
				b->im = a->im - vi;
				a->re += vr;
				a->im += vi;
			}
		}
	}
}

static void imdct_half(Imdct* mdct, float* dst, const float* src)
{
	const int len = mdct->len;
	const Cpx* exp = mdct->exp;
	Cpx* z = mdct->z;
	const int len2 = len >> 1;
	const int len4 = len >> 2;

	for (int i = 0; i < len2; i++) {
		const double re = src[len - 1 - 2 * i];
		const double im = src[2 * i];
		z[i].re = re * exp[i].re - im * exp[i].im;
		z[i].im = re * exp[i].im + im * exp[i].re;
	}

	fft_inverse(z, mdct->twiddles, len2);

	for (int i = 0; i < len4; i++) {
		const int i0 = len4 + i, i1 = len4 - i - 1;
		const Cpx src1 = { z[i1].im, z[i1].re };
		const Cpx src0 = { z[i0].im, z[i0].re };

		z[i1].re = src1.re * exp[i1].im - src1.im * exp[i1].re;
		z[i0].im = src1.re * exp[i1].re + src1.im * exp[i1].im;
		z[i0].re = src0.re * exp[i0].im - src0.im * exp[i0].re;
		z[i1].im = src0.re * exp[i0].re + src0.im * exp[i0].im;
	}

	for (int i = 0; i < len2; i++) {
		dst[2 * i] = (float)z[i].re;
		dst[2 * i + 1] = (float)z[i].im;
	}
}

///////////////////////////////////////////////////////////////////////////////
// Free all allocated memory

static void vorbis_free(vorbis_context* vc)
{
	free(vc->channel_residues);
	free(vc->saved);
	free(vc->output);
	imdct_free(&vc->mdct[0]);
	imdct_free(&vc->mdct[1]);

	if (vc->residues) {
		for (int i = 0; i < vc->residue_count; i++) free(vc->residues[i].classifs);
	}
	free(vc->residues);
	free(vc->modes);

	if (vc->codebooks) {
		for (int i = 0; i < vc->codebook_count; ++i) {
			free(vc->codebooks[i].codevectors);
			free(vc->codebooks[i].vlc.table);
		}
	}
	free(vc->codebooks);

	if (vc->floors) {
		for (int i = 0; i < vc->floor_count; ++i) {
			if (vc->floors[i].floor_type == 0) {
				free(vc->floors[i].t0.map[0]);
				free(vc->floors[i].t0.map[1]);
				free(vc->floors[i].t0.book_list);
				free(vc->floors[i].t0.lsp);
			} else {
				free(vc->floors[i].t1.list);
			}
		}
	}
	free(vc->floors);

	if (vc->mappings) {
		for (int i = 0; i < vc->mapping_count; ++i) {
			free(vc->mappings[i].magnitude);
			free(vc->mappings[i].angle);
			free(vc->mappings[i].mux);
		}
	}
	free(vc->mappings);
}

///////////////////////////////////////////////////////////////////////////////
// Parse setup header

// Process codebooks part. The three buffers hold V_MAX_VLCS entries.
static int parse_codebooks(vorbis_context* vc, uint8_t* tmp_vlc_bits, uint32_t* tmp_vlc_codes, uint16_t* codebook_multiplicands)
{
	GetBitContext* gb = &vc->gb;

	vc->codebook_count = get_bits(gb, 8) + 1;
	vc->codebooks = (vorbis_codebook*)calloc(vc->codebook_count, sizeof(*vc->codebooks));
	if (!vc->codebooks) return ERROR_MEMORY;

	for (unsigned cb = 0; cb < vc->codebook_count; ++cb) {
		vorbis_codebook* codebook_setup = &vc->codebooks[cb];
		unsigned ordered, t, entries, used_entries = 0;

		if (get_bits(gb, 24) != 0x564342) return ERROR_INVALID;

		codebook_setup->dimensions = get_bits(gb, 16);
		if (codebook_setup->dimensions > 16 || codebook_setup->dimensions == 0) return ERROR_INVALID;
		entries = get_bits(gb, 24);
		if (entries > V_MAX_VLCS) return ERROR_INVALID;

		ordered = get_bits1(gb);

		if (!ordered) {
			const unsigned sparse = get_bits1(gb);
			if (sparse) {
				used_entries = 0;
				for (unsigned ce = 0; ce < entries; ++ce) {
					if (get_bits1(gb)) {
						tmp_vlc_bits[ce] = get_bits(gb, 5) + 1;
						++used_entries;
					} else {
						tmp_vlc_bits[ce] = 0;
					}
				}
			} else {
				used_entries = entries;
				for (unsigned ce = 0; ce < entries; ++ce) tmp_vlc_bits[ce] = get_bits(gb, 5) + 1;
			}
		} else {
			unsigned current_entry = 0;
			unsigned current_length = get_bits(gb, 5) + 1;

			used_entries = entries;
			for (; current_entry < used_entries && current_length <= 32; ++current_length) {
				const unsigned number = get_bits(gb, ilog(entries - current_entry));
				for (unsigned i = current_entry; i < number + current_entry; ++i) {
					if (i < used_entries) tmp_vlc_bits[i] = current_length;
				}
				current_entry += number;
			}
			if (current_entry > used_entries) return ERROR_INVALID;
		}

		codebook_setup->lookup_type = get_bits(gb, 4);

		// If the codebook is used for (inverse) VQ, calculate codevectors.
		if (codebook_setup->lookup_type == 1) {
			unsigned i, j, k;
			const unsigned codebook_lookup_values = ff_vorbis_nth_root(entries, codebook_setup->dimensions);

			const float codebook_minimum_value = vorbisfloat2float(get_bits_long(gb, 32));
			const float codebook_delta_value = vorbisfloat2float(get_bits_long(gb, 32));
			const unsigned codebook_value_bits = get_bits(gb, 4) + 1;
			const unsigned codebook_sequence_p = get_bits1(gb);

			if (!std::isfinite(codebook_minimum_value) || !std::isfinite(codebook_delta_value)) return ERROR_INVALID;

			for (i = 0; i < codebook_lookup_values; ++i) codebook_multiplicands[i] = get_bits(gb, codebook_value_bits);

			// Weed out unused vlcs and build codevector vector
			if (used_entries) {
				codebook_setup->codevectors = (float*)calloc(used_entries, codebook_setup->dimensions * sizeof(float));
				if (!codebook_setup->codevectors) return ERROR_MEMORY;
			} else {
				codebook_setup->codevectors = nullptr;
			}

			for (j = 0, i = 0; i < entries; ++i) {
				const unsigned dim = codebook_setup->dimensions;

				if (tmp_vlc_bits[i]) {
					float last = 0.0;
					unsigned lookup_offset = i;

					for (k = 0; k < dim; ++k) {
						const unsigned multiplicand_offset = lookup_offset % codebook_lookup_values;
						codebook_setup->codevectors[j * dim + k] =
							codebook_multiplicands[multiplicand_offset] * codebook_delta_value + codebook_minimum_value + last;
						if (codebook_sequence_p) last = codebook_setup->codevectors[j * dim + k];
						lookup_offset /= codebook_lookup_values;
					}
					tmp_vlc_bits[j] = tmp_vlc_bits[i];
					++j;
				}
			}
			if (j != used_entries) return ERROR_INVALID;
			entries = used_entries;
		} else if (codebook_setup->lookup_type >= 2) {
			return ERROR_INVALID;
		}

		// Initialize VLC table
		if (ff_vorbis_len2vlc(tmp_vlc_bits, tmp_vlc_codes, entries)) return ERROR_INVALID;
		codebook_setup->maxdepth = 0;
		for (t = 0; t < entries; ++t) {
			if (tmp_vlc_bits[t] >= codebook_setup->maxdepth) codebook_setup->maxdepth = tmp_vlc_bits[t];
		}

		codebook_setup->nb_bits = codebook_setup->maxdepth > 3 * V_NB_BITS ? V_NB_BITS2 : V_NB_BITS;
		codebook_setup->maxdepth = (codebook_setup->maxdepth + codebook_setup->nb_bits - 1) / codebook_setup->nb_bits;

		const int ret = vlc_init(&codebook_setup->vlc, codebook_setup->nb_bits, entries, tmp_vlc_bits, tmp_vlc_codes);
		if (ret) return ret;
	}
	return 0;
}

static int vorbis_parse_setup_hdr_codebooks(vorbis_context* vc)
{
	uint8_t* tmp_vlc_bits = (uint8_t*)calloc(V_MAX_VLCS, sizeof(uint8_t));
	uint32_t* tmp_vlc_codes = (uint32_t*)calloc(V_MAX_VLCS, sizeof(uint32_t));
	uint16_t* codebook_multiplicands = (uint16_t*)malloc(V_MAX_VLCS * sizeof(uint16_t));
	const int ret = tmp_vlc_bits && tmp_vlc_codes && codebook_multiplicands
		? parse_codebooks(vc, tmp_vlc_bits, tmp_vlc_codes, codebook_multiplicands) : ERROR_MEMORY;
	free(tmp_vlc_bits);
	free(tmp_vlc_codes);
	free(codebook_multiplicands);
	return ret;
}

// Process time domain transforms part (unused in Vorbis I)
static int vorbis_parse_setup_hdr_tdtransforms(vorbis_context* vc)
{
	GetBitContext* gb = &vc->gb;
	const unsigned vorbis_time_count = get_bits(gb, 6) + 1;

	for (unsigned i = 0; i < vorbis_time_count; ++i) {
		if (get_bits(gb, 16)) return ERROR_INVALID;
	}
	return 0;
}

static int create_map(vorbis_context* vc, unsigned floor_number)
{
	vorbis_floor0* vf = &vc->floors[floor_number].t0;

	for (int blockflag = 0; blockflag < 2; ++blockflag) {
		const int n = vc->blocksize[blockflag] / 2;
		int32_t* map = (int32_t*)malloc((n + 1) * sizeof(int32_t));	// n + sentinel
		if (!map) return ERROR_MEMORY;
		vf->map[blockflag] = map;

		for (int idx = 0; idx < n; ++idx) {
			map[idx] = (int32_t)floor(BARK((vf->rate * idx) / (2.0f * n)) * (vf->bark_map_size / BARK(vf->rate / 2.0f)));
			if (vf->bark_map_size - 1 < map[idx]) map[idx] = vf->bark_map_size - 1;
		}
		map[n] = -1;
		vf->map_size[blockflag] = n;
	}
	return 0;
}

// Process floors part
static int vorbis_parse_setup_hdr_floors(vorbis_context* vc)
{
	GetBitContext* gb = &vc->gb;
	int i, j, k, ret;

	vc->floor_count = get_bits(gb, 6) + 1;

	vc->floors = (vorbis_floor*)calloc(vc->floor_count, sizeof(*vc->floors));
	if (!vc->floors) return ERROR_MEMORY;

	for (i = 0; i < vc->floor_count; ++i) {
		vorbis_floor* floor_setup = &vc->floors[i];

		floor_setup->floor_type = get_bits(gb, 16);

		if (floor_setup->floor_type == 1) {
			vorbis_floor1* t1 = &floor_setup->t1;
			int maximum_class = -1;
			unsigned rangebits, rangemax, floor1_values = 2;

			t1->partitions = get_bits(gb, 5);

			for (j = 0; j < t1->partitions; ++j) {
				t1->partition_class[j] = get_bits(gb, 4);
				if (t1->partition_class[j] > maximum_class) maximum_class = t1->partition_class[j];
			}

			for (j = 0; j <= maximum_class; ++j) {
				t1->class_dimensions[j] = get_bits(gb, 3) + 1;
				t1->class_subclasses[j] = get_bits(gb, 2);

				if (t1->class_subclasses[j]) {
					GET_VALIDATED_INDEX(t1->class_masterbook[j], 8, vc->codebook_count)
				}

				for (k = 0; k < (1 << t1->class_subclasses[j]); ++k) {
					const int16_t bits = get_bits(gb, 8) - 1;
					if (bits != -1) {
						VALIDATE_INDEX(bits, vc->codebook_count)
					}
					t1->subclass_books[j][k] = bits;
				}
			}

			t1->multiplier = get_bits(gb, 2) + 1;
			t1->x_list_dim = 2;

			for (j = 0; j < t1->partitions; ++j) t1->x_list_dim += t1->class_dimensions[t1->partition_class[j]];

			t1->list = (vorbis_floor1_entry*)calloc(t1->x_list_dim, sizeof(*t1->list));
			if (!t1->list) return ERROR_MEMORY;

			rangebits = get_bits(gb, 4);
			if (!rangebits && t1->partitions) return ERROR_INVALID;
			rangemax = 1 << rangebits;
			if (rangemax > vc->blocksize[1] / 2) return ERROR_INVALID;
			t1->list[0].x = 0;
			t1->list[1].x = rangemax;

			for (j = 0; j < t1->partitions; ++j) {
				for (k = 0; k < t1->class_dimensions[t1->partition_class[j]]; ++k, ++floor1_values) {
					t1->list[floor1_values].x = get_bits(gb, rangebits);
				}
			}

			// Precalculate order of x coordinates - needed for decode
			if (ff_vorbis_ready_floor1_list(t1->list, t1->x_list_dim)) return ERROR_INVALID;
		} else if (floor_setup->floor_type == 0) {
			vorbis_floor0* t0 = &floor_setup->t0;
			unsigned max_codebook_dim = 0;

			t0->order = get_bits(gb, 8);
			if (!t0->order) return ERROR_INVALID;
			t0->rate = get_bits(gb, 16);
			if (!t0->rate) return ERROR_INVALID;
			t0->bark_map_size = get_bits(gb, 16);
			if (!t0->bark_map_size) return ERROR_INVALID;
			t0->amplitude_bits = get_bits(gb, 6);
			t0->amplitude_offset = get_bits(gb, 8);
			t0->num_books = get_bits(gb, 4) + 1;

			t0->book_list = (uint8_t*)malloc(t0->num_books);
			if (!t0->book_list) return ERROR_MEMORY;
			for (int idx = 0; idx < t0->num_books; ++idx) {
				unsigned book_idx;
				GET_VALIDATED_INDEX(book_idx, 8, vc->codebook_count)
				t0->book_list[idx] = book_idx;
				if (vc->codebooks[book_idx].dimensions > max_codebook_dim) max_codebook_dim = vc->codebooks[book_idx].dimensions;
			}

			if ((ret = create_map(vc, i)) < 0) return ret;

			// codebook dim is for padding if codebook dim doesn't divide order+1 then we need to read more data
			t0->lsp = (float*)malloc((t0->order + 1 + max_codebook_dim) * sizeof(float));
			if (!t0->lsp) return ERROR_MEMORY;
		} else {
			return ERROR_INVALID;
		}
	}
	return 0;
}

// Process residues part
static int vorbis_parse_setup_hdr_residues(vorbis_context* vc)
{
	GetBitContext* gb = &vc->gb;
	unsigned i, j, k;

	vc->residue_count = get_bits(gb, 6) + 1;
	vc->residues = (vorbis_residue*)calloc(vc->residue_count, sizeof(*vc->residues));
	if (!vc->residues) return ERROR_MEMORY;

	for (i = 0; i < vc->residue_count; ++i) {
		vorbis_residue* res_setup = &vc->residues[i];
		uint8_t cascade[64];
		unsigned high_bits, low_bits;

		res_setup->type = get_bits(gb, 16);

		res_setup->begin = get_bits(gb, 24);
		res_setup->end = get_bits(gb, 24);
		res_setup->partition_size = get_bits(gb, 24) + 1;
		// Validations to prevent a buffer overflow later.
		if (res_setup->begin > res_setup->end
			|| (res_setup->end - res_setup->begin) / res_setup->partition_size > (V_MAX_PARTITIONS < 65535 ? V_MAX_PARTITIONS : 65535)) {
			return ERROR_INVALID;
		}

		res_setup->classifications = get_bits(gb, 6) + 1;
		GET_VALIDATED_INDEX(res_setup->classbook, 8, vc->codebook_count)

		res_setup->ptns_to_read = (res_setup->end - res_setup->begin) / res_setup->partition_size;
		res_setup->classifs = (uint8_t*)malloc(res_setup->ptns_to_read * vc->audio_channels + 1);
		if (!res_setup->classifs) return ERROR_MEMORY;

		for (j = 0; j < res_setup->classifications; ++j) {
			high_bits = 0;
			low_bits = get_bits(gb, 3);
			if (get_bits1(gb)) high_bits = get_bits(gb, 5);
			cascade[j] = (high_bits << 3) + low_bits;
		}

		res_setup->maxpass = 0;
		for (j = 0; j < res_setup->classifications; ++j) {
			for (k = 0; k < 8; ++k) {
				if (cascade[j] & (1 << k)) {
					GET_VALIDATED_INDEX(res_setup->books[j][k], 8, vc->codebook_count)
					if (k > res_setup->maxpass) res_setup->maxpass = k;
				} else {
					res_setup->books[j][k] = -1;
				}
			}
		}
	}
	return 0;
}

// Process mappings part
static int vorbis_parse_setup_hdr_mappings(vorbis_context* vc)
{
	GetBitContext* gb = &vc->gb;
	unsigned i, j;

	vc->mapping_count = get_bits(gb, 6) + 1;
	vc->mappings = (vorbis_mapping*)calloc(vc->mapping_count, sizeof(*vc->mappings));
	if (!vc->mappings) return ERROR_MEMORY;

	for (i = 0; i < vc->mapping_count; ++i) {
		vorbis_mapping* mapping_setup = &vc->mappings[i];

		if (get_bits(gb, 16)) return ERROR_INVALID;
		mapping_setup->submaps = get_bits1(gb) ? get_bits(gb, 4) + 1 : 1;

		if (get_bits1(gb)) {
			mapping_setup->coupling_steps = get_bits(gb, 8) + 1;
			if (vc->audio_channels < 2) return ERROR_INVALID;
			mapping_setup->magnitude = (uint8_t*)calloc(mapping_setup->coupling_steps, 1);
			mapping_setup->angle = (uint8_t*)calloc(mapping_setup->coupling_steps, 1);
			if (!mapping_setup->angle || !mapping_setup->magnitude) return ERROR_MEMORY;

			for (j = 0; j < mapping_setup->coupling_steps; ++j) {
				GET_VALIDATED_INDEX(mapping_setup->magnitude[j], ilog(vc->audio_channels - 1), vc->audio_channels)
				GET_VALIDATED_INDEX(mapping_setup->angle[j], ilog(vc->audio_channels - 1), vc->audio_channels)
			}
		} else {
			mapping_setup->coupling_steps = 0;
		}

		if (get_bits(gb, 2)) return ERROR_INVALID;	// following spec.

		if (mapping_setup->submaps > 1) {
			mapping_setup->mux = (uint8_t*)calloc(vc->audio_channels, 1);
			if (!mapping_setup->mux) return ERROR_MEMORY;
			for (j = 0; j < vc->audio_channels; ++j) mapping_setup->mux[j] = get_bits(gb, 4);
		}

		for (j = 0; j < mapping_setup->submaps; ++j) {
			skip_bits(gb, 8);
			GET_VALIDATED_INDEX(mapping_setup->submap_floor[j], 8, vc->floor_count)
			GET_VALIDATED_INDEX(mapping_setup->submap_residue[j], 8, vc->residue_count)
		}
	}
	return 0;
}

// Process modes part
static int vorbis_parse_setup_hdr_modes(vorbis_context* vc)
{
	GetBitContext* gb = &vc->gb;

	vc->mode_count = get_bits(gb, 6) + 1;
	vc->modes = (vorbis_mode*)calloc(vc->mode_count, sizeof(*vc->modes));
	if (!vc->modes) return ERROR_MEMORY;

	for (unsigned i = 0; i < vc->mode_count; ++i) {
		vorbis_mode* mode_setup = &vc->modes[i];

		mode_setup->blockflag = get_bits1(gb);
		mode_setup->windowtype = get_bits(gb, 16);
		mode_setup->transformtype = get_bits(gb, 16);
		GET_VALIDATED_INDEX(mode_setup->mapping, 8, vc->mapping_count);
	}
	return 0;
}

// Process the whole setup header using the functions above
static int vorbis_parse_setup_hdr(vorbis_context* vc)
{
	GetBitContext* gb = &vc->gb;
	int ret;

	if (get_bits(gb, 8) != 'v' || get_bits(gb, 8) != 'o' || get_bits(gb, 8) != 'r'
		|| get_bits(gb, 8) != 'b' || get_bits(gb, 8) != 'i' || get_bits(gb, 8) != 's') {
		return ERROR_INVALID;
	}

	if ((ret = vorbis_parse_setup_hdr_codebooks(vc))) return ret;
	if ((ret = vorbis_parse_setup_hdr_tdtransforms(vc))) return ret;
	if ((ret = vorbis_parse_setup_hdr_floors(vc))) return ret;
	if ((ret = vorbis_parse_setup_hdr_residues(vc))) return ret;
	if ((ret = vorbis_parse_setup_hdr_mappings(vc))) return ret;
	if ((ret = vorbis_parse_setup_hdr_modes(vc))) return ret;
	if (!get_bits1(gb)) return ERROR_INVALID;	// framing flag bit unset error

	return 0;
}

// Process the identification header
static int vorbis_parse_id_hdr(vorbis_context* vc)
{
	GetBitContext* gb = &vc->gb;
	unsigned bl0, bl1;
	int ret;

	if (get_bits(gb, 8) != 'v' || get_bits(gb, 8) != 'o' || get_bits(gb, 8) != 'r'
		|| get_bits(gb, 8) != 'b' || get_bits(gb, 8) != 'i' || get_bits(gb, 8) != 's') {
		return ERROR_INVALID;
	}

	vc->version = get_bits_long(gb, 32);
	vc->audio_channels = get_bits(gb, 8);
	if (vc->audio_channels <= 0) return ERROR_INVALID;
	vc->audio_samplerate = get_bits_long(gb, 32);
	if (vc->audio_samplerate <= 0) return ERROR_INVALID;
	get_bits_long(gb, 32);	// bitrate_maximum
	get_bits_long(gb, 32);	// bitrate_nominal
	get_bits_long(gb, 32);	// bitrate_minimum
	bl0 = get_bits(gb, 4);
	bl1 = get_bits(gb, 4);
	if (bl0 > 13 || bl0 < 6 || bl1 > 13 || bl1 < 6 || bl1 < bl0) return ERROR_INVALID;
	vc->blocksize[0] = 1 << bl0;
	vc->blocksize[1] = 1 << bl1;
	vc->win[0] = ff_vorbis_vwin[bl0 - 6];
	vc->win[1] = ff_vorbis_vwin[bl1 - 6];

	if (get_bits1(gb) == 0) return ERROR_INVALID;

	vc->channel_residues = (float*)malloc((vc->blocksize[1] / 2) * vc->audio_channels * sizeof(float));
	vc->saved = (float*)calloc(vc->blocksize[1] / 4, vc->audio_channels * sizeof(float));
	vc->output = (float*)malloc((vc->blocksize[1] / 2) * vc->audio_channels * sizeof(float));
	if (!vc->channel_residues || !vc->saved || !vc->output) return ERROR_MEMORY;

	vc->previous_window = -1;

	if ((ret = imdct_init(&vc->mdct[0], vc->blocksize[0] >> 1))) return ret;
	if ((ret = imdct_init(&vc->mdct[1], vc->blocksize[1] >> 1))) return ret;

	return 0;
}

///////////////////////////////////////////////////////////////////////////////
// Decode audiopackets

// Read and decode floor
static int vorbis_floor0_decode(vorbis_context* vc, vorbis_floor0* vf, float* vec)
{
	float* lsp = vf->lsp;
	unsigned book_idx;
	uint64_t amplitude;
	const unsigned blockflag = vc->modes[vc->mode_number].blockflag;

	if (!vf->amplitude_bits) return 1;

	amplitude = get_bits64(&vc->gb, vf->amplitude_bits);
	if (amplitude > 0) {
		float last = 0;
		unsigned idx, lsp_len = 0;

		book_idx = get_bits(&vc->gb, ilog(vf->num_books));
		if (book_idx >= vf->num_books) book_idx = 0;
		const vorbis_codebook* codebook = &vc->codebooks[vf->book_list[book_idx]];
		// Invalid codebook!
		if (!codebook->codevectors) return ERROR_INVALID;

		while (lsp_len < vf->order) {
			int vec_off;

			// read temp vector
			vec_off = get_vlc2(&vc->gb, codebook->vlc.table, codebook->nb_bits);
			if (vec_off < 0) return ERROR_INVALID;
			vec_off *= codebook->dimensions;
			// copy each vector component and add last to it
			for (idx = 0; idx < codebook->dimensions; ++idx) lsp[lsp_len + idx] = codebook->codevectors[vec_off + idx] + last;
			last = lsp[lsp_len + idx - 1];	// set last to last vector component

			lsp_len += codebook->dimensions;
		}

		// synthesize floor output vector
		{
			int i;
			const int order = vf->order;
			const float wstep = M_PI / vf->bark_map_size;

			for (i = 0; i < order; i++) lsp[i] = 2.0f * cos((double)lsp[i]);

			i = 0;
			while ((unsigned)i < vf->map_size[blockflag]) {
				int j;
				const int iter_cond = vf->map[blockflag][i];
				float p = 0.5f;
				float q = 0.5f;
				const float two_cos_w = 2.0f * cos((double)(wstep * iter_cond));	// needed all times

				// similar part for the q and p products
				for (j = 0; j + 1 < order; j += 2) {
					q *= lsp[j] - two_cos_w;
					p *= lsp[j + 1] - two_cos_w;
				}
				if (j == order) {	// even order
					p *= p * (2.0f - two_cos_w);
					q *= q * (2.0f + two_cos_w);
				} else {			// odd order
					q *= two_cos_w - lsp[j];	// one more time for q

					// final step and square
					p *= p * (4.f - two_cos_w * two_cos_w);
					q *= q;
				}

				if (p + q == 0.0) return ERROR_INVALID;

				// calculate linear floor value
				q = exp((((amplitude * vf->amplitude_offset) / (((1ULL << vf->amplitude_bits) - 1) * sqrt((double)(p + q))))
					- vf->amplitude_offset) * .11512925f);

				// fill vector
				do {
					vec[i] = q;
					++i;
				} while (vf->map[blockflag][i] == iter_cond);
			}
		}
	} else {
		// this channel is unused
		return 1;
	}
	return 0;
}

static int vorbis_floor1_decode(vorbis_context* vc, vorbis_floor1* vf, float* vec)
{
	GetBitContext* gb = &vc->gb;
	static const uint16_t range_v[4] = { 256, 128, 86, 64 };
	const unsigned range = range_v[vf->multiplier - 1];
	uint16_t floor1_Y[258];
	uint16_t floor1_Y_final[258];
	int floor1_flag[258];
	unsigned partition_class, cdim, cbits, csub, cval, offset, i, j;
	int book, adx, ady, dy, off, predicted, err;

	if (!get_bits1(gb)) return 1;	// silence

	// Read values (or differences) for the floor's points
	floor1_Y[0] = get_bits(gb, ilog(range - 1));
	floor1_Y[1] = get_bits(gb, ilog(range - 1));

	offset = 2;
	for (i = 0; i < vf->partitions; ++i) {
		partition_class = vf->partition_class[i];
		cdim = vf->class_dimensions[partition_class];
		cbits = vf->class_subclasses[partition_class];
		csub = (1 << cbits) - 1;
		cval = 0;

		if (cbits) {	// this reads all subclasses for this partition's class
			const vorbis_codebook* master = &vc->codebooks[vf->class_masterbook[partition_class]];
			cval = get_vlc2(gb, master->vlc.table, master->nb_bits);
		}

		for (j = 0; j < cdim; ++j) {
			book = vf->subclass_books[partition_class][cval & csub];

			cval = cval >> cbits;
			if (book > -1) {
				const int v = get_vlc2(gb, vc->codebooks[book].vlc.table, vc->codebooks[book].nb_bits);
				if (v < 0) return ERROR_INVALID;
				floor1_Y[offset + j] = v;
			} else {
				floor1_Y[offset + j] = 0;
			}
		}
		offset += cdim;
	}

	// Amplitude calculation from the differences
	floor1_flag[0] = 1;
	floor1_flag[1] = 1;
	floor1_Y_final[0] = floor1_Y[0];
	floor1_Y_final[1] = floor1_Y[1];

	for (i = 2; i < vf->x_list_dim; ++i) {
		unsigned val, highroom, lowroom, room, high_neigh_offs, low_neigh_offs;

		low_neigh_offs = vf->list[i].low;
		high_neigh_offs = vf->list[i].high;
		dy = floor1_Y_final[high_neigh_offs] - floor1_Y_final[low_neigh_offs];	// render_point begin
		adx = vf->list[high_neigh_offs].x - vf->list[low_neigh_offs].x;
		ady = abs(dy);
		err = ady * (vf->list[i].x - vf->list[low_neigh_offs].x);
		off = err / adx;
		predicted = dy < 0 ? floor1_Y_final[low_neigh_offs] - off : floor1_Y_final[low_neigh_offs] + off;	// render_point end

		val = floor1_Y[i];
		highroom = range - predicted;
		lowroom = predicted;
		room = highroom < lowroom ? highroom * 2 : lowroom * 2;
		if (val) {
			floor1_flag[low_neigh_offs] = 1;
			floor1_flag[high_neigh_offs] = 1;
			floor1_flag[i] = 1;
			if (val >= room) {
				if (highroom > lowroom) {
					floor1_Y_final[i] = clip_uint16(val - lowroom + predicted);
				} else {
					floor1_Y_final[i] = clip_uint16(predicted - val + highroom - 1);
				}
			} else {
				if (val & 1) {
					floor1_Y_final[i] = clip_uint16(predicted - (val + 1) / 2);
				} else {
					floor1_Y_final[i] = clip_uint16(predicted + val / 2);
				}
			}
		} else {
			floor1_flag[i] = 0;
			floor1_Y_final[i] = clip_uint16(predicted);
		}
	}

	// Curve synth - connect the calculated dots and convert from dB scale
	ff_vorbis_floor1_render_list(vf->list, vf->x_list_dim, floor1_Y_final, floor1_flag, vf->multiplier, vec, vf->list[1].x);

	return 0;
}

static inline int setup_classifs(vorbis_context* vc, vorbis_residue* vr, uint8_t* do_not_decode, unsigned ch_used,
	int partition_count, int ptns_to_read)
{
	const vorbis_codebook* codebook = vc->codebooks + vr->classbook;
	int p, j, i;
	const unsigned c_p_c = codebook->dimensions;
	int temp, temp2;
	for (p = 0, j = 0; (unsigned)j < ch_used; ++j) {
		if (!do_not_decode[j]) {
			temp = get_vlc2(&vc->gb, codebook->vlc.table, codebook->nb_bits);
			if (temp < 0) return ERROR_INVALID;

			if (vr->classifications == 1) {
				for (i = partition_count + c_p_c - 1; i >= partition_count; i--) {
					if (i < ptns_to_read) vr->classifs[p + i] = 0;
				}
			} else {
				for (i = partition_count + c_p_c - 1; i >= partition_count; i--) {
					temp2 = temp / vr->classifications;
					if (i < ptns_to_read) vr->classifs[p + i] = temp - temp2 * vr->classifications;
					temp = temp2;
				}
			}
		}
		p += ptns_to_read;
	}
	return 0;
}

// Read and decode residue
static inline int vorbis_residue_decode_internal(vorbis_context* vc, vorbis_residue* vr, unsigned ch, uint8_t* do_not_decode,
	float* vec, unsigned vlen, unsigned ch_left, int vr_type)
{
	GetBitContext* gb = &vc->gb;
	const unsigned c_p_c = vc->codebooks[vr->classbook].dimensions;
	uint8_t* classifs = vr->classifs;
	unsigned pass, ch_used, i, j, k, l;
	unsigned max_output = (ch - 1) * vlen;
	int ptns_to_read = vr->ptns_to_read;
	int libvorbis_bug = 0;

	if (vr_type == 2) {
		for (j = 1; j < ch; ++j) do_not_decode[0] &= do_not_decode[j];
		if (do_not_decode[0]) return 0;
		ch_used = 1;
		max_output += vr->end / ch;
	} else {
		ch_used = ch;
		max_output += vr->end;
	}

	if (max_output > ch_left * vlen) {
		if (max_output <= ch_left * vlen + vr->partition_size * ch_used / ch) {
			ptns_to_read--;
			libvorbis_bug = 1;
		} else {
			return ERROR_INVALID;
		}
	}

	for (pass = 0; pass <= vr->maxpass; ++pass) {
		int voffset, partition_count, j_times_ptns_to_read;

		voffset = vr->begin;
		for (partition_count = 0; partition_count < ptns_to_read;) {
			if (!pass) {
				const int ret = setup_classifs(vc, vr, do_not_decode, ch_used, partition_count, ptns_to_read);
				if (ret < 0) return ret;
			}
			for (i = 0; (i < c_p_c) && (partition_count < ptns_to_read); ++i) {
				for (j_times_ptns_to_read = 0, j = 0; j < ch_used; ++j) {
					unsigned voffs;

					if (!do_not_decode[j]) {
						const unsigned vqclass = classifs[j_times_ptns_to_read + partition_count];
						const int vqbook = vr->books[vqclass][pass];

						if (vqbook >= 0 && vc->codebooks[vqbook].codevectors) {
							int coffs;
							const vorbis_codebook* codebook = &vc->codebooks[vqbook];
							const unsigned dim = codebook->dimensions;
							const unsigned step = vr->partition_size / dim;
							const VLCElem* table = codebook->vlc.table;
							const int nb_bits = codebook->nb_bits;
							const float* codevectors = codebook->codevectors;

							if (get_bits_left(gb) < 0) return 0;	// Overread

							if (vr_type == 0) {
								voffs = voffset + j * vlen;
								for (k = 0; k < step; ++k) {
									coffs = get_vlc2(gb, table, nb_bits);
									if (coffs < 0) return coffs;
									coffs *= dim;
									for (l = 0; l < dim; ++l) vec[voffs + k + l * step] += codevectors[coffs + l];
								}
							} else if (vr_type == 1) {
								voffs = voffset + j * vlen;
								for (k = 0; k < step; ++k) {
									coffs = get_vlc2(gb, table, nb_bits);
									if (coffs < 0) return coffs;
									coffs *= dim;
									for (l = 0; l < dim; ++l, ++voffs) vec[voffs] += codevectors[coffs + l];
								}
							} else if (vr_type == 2 && ch == 2 && (voffset & 1) == 0 && (dim & 1) == 0) {	// most frequent case optimized
								voffs = voffset >> 1;

								if (dim == 2) {
									for (k = 0; k < step; ++k) {
										coffs = get_vlc2(gb, table, nb_bits);
										if (coffs < 0) return coffs;
										coffs *= 2;
										vec[voffs + k] += codevectors[coffs];
										vec[voffs + k + vlen] += codevectors[coffs + 1];
									}
								} else if (dim == 4) {
									for (k = 0; k < step; ++k, voffs += 2) {
										coffs = get_vlc2(gb, table, nb_bits);
										if (coffs < 0) return coffs;
										coffs *= 4;
										vec[voffs] += codevectors[coffs];
										vec[voffs + 1] += codevectors[coffs + 2];
										vec[voffs + vlen] += codevectors[coffs + 1];
										vec[voffs + vlen + 1] += codevectors[coffs + 3];
									}
								} else {
									for (k = 0; k < step; ++k) {
										coffs = get_vlc2(gb, table, nb_bits);
										if (coffs < 0) return coffs;
										coffs *= dim;
										for (l = 0; l < dim; l += 2, voffs++) {
											vec[voffs] += codevectors[coffs + l];
											vec[voffs + vlen] += codevectors[coffs + l + 1];
										}
									}
								}
							} else if (vr_type == 2) {
								unsigned voffs_div = ch == 1 ? voffset : voffset / ch;
								unsigned voffs_mod = voffset - voffs_div * ch;

								for (k = 0; k < step; ++k) {
									coffs = get_vlc2(gb, table, nb_bits);
									if (coffs < 0) return coffs;
									coffs *= dim;
									for (l = 0; l < dim; ++l) {
										vec[voffs_div + voffs_mod * vlen] += codevectors[coffs + l];

										if (++voffs_mod == ch) {
											voffs_div++;
											voffs_mod = 0;
										}
									}
								}
							}
						}
					}
					j_times_ptns_to_read += ptns_to_read;
				}
				++partition_count;
				voffset += vr->partition_size;
			}
		}
		if (libvorbis_bug && !pass) {
			for (j = 0; j < ch_used; ++j) {
				if (!do_not_decode[j]) {
					get_vlc2(&vc->gb, vc->codebooks[vr->classbook].vlc.table, vc->codebooks[vr->classbook].nb_bits);
				}
			}
		}
	}
	return 0;
}

static inline int vorbis_residue_decode(vorbis_context* vc, vorbis_residue* vr, unsigned ch, uint8_t* do_not_decode,
	float* vec, unsigned vlen, unsigned ch_left)
{
	if (vr->type == 2) return vorbis_residue_decode_internal(vc, vr, ch, do_not_decode, vec, vlen, ch_left, 2);
	if (vr->type == 1) return vorbis_residue_decode_internal(vc, vr, ch, do_not_decode, vec, vlen, ch_left, 1);
	if (vr->type == 0) return vorbis_residue_decode_internal(vc, vr, ch, do_not_decode, vec, vlen, ch_left, 0);
	return ERROR_INVALID;
}

static void vorbis_inverse_coupling(float* mag, float* ang, int blocksize)
{
	for (int i = 0; i < blocksize; i++) {
		const float angi = ang[i], magi = mag[i];

		if (magi > 0.f) {
			if (angi > 0.f) {
				ang[i] = magi - angi;
			} else {
				ang[i] = magi;
				mag[i] = magi + angi;
			}
		} else {
			if (angi > 0.f) {
				ang[i] = magi + angi;
			} else {
				ang[i] = magi;
				mag[i] = magi - angi;
			}
		}
	}
}

static void vector_fmul_window(float* dst, const float* src0, const float* src1, const float* win, int len)
{
	int i, j;

	dst += len;
	win += len;
	src0 += len;

	for (i = -len, j = len - 1; i < 0; i++, j--) {
		const float s0 = src0[i];
		const float s1 = src1[j];
		const float wi = win[i];
		const float wj = win[j];
		dst[i] = s0 * wj - s1 * wi;
		dst[j] = s0 * wi + s1 * wj;
	}
}

// Decode the audio packet using the functions above
static int vorbis_parse_audio_packet(vorbis_context* vc, float** floor_ptr)
{
	GetBitContext* gb = &vc->gb;
	int previous_window = vc->previous_window;
	unsigned mode_number, blockflag, blocksize;
	int i, j;
	uint8_t no_residue[255];
	uint8_t do_not_decode[255];
	vorbis_mapping* mapping;
	float* ch_res_ptr = vc->channel_residues;
	uint8_t res_chan[255];
	unsigned res_num = 0;
	int retlen = 0;
	unsigned ch_left = vc->audio_channels;
	unsigned vlen;

	if (get_bits1(gb)) return ERROR_INVALID;	// packet type not audio

	if (vc->mode_count == 1) {
		mode_number = 0;
	} else {
		GET_VALIDATED_INDEX(mode_number, ilog(vc->mode_count - 1), vc->mode_count)
	}
	vc->mode_number = mode_number;
	mapping = &vc->mappings[vc->modes[mode_number].mapping];

	blockflag = vc->modes[mode_number].blockflag;
	blocksize = vc->blocksize[blockflag];
	vlen = blocksize / 2;
	if (blockflag) {
		const int code = get_bits(gb, 2);
		if (previous_window < 0) previous_window = code & 1;
	} else if (previous_window < 0) {
		previous_window = 0;
	}

	memset(ch_res_ptr, 0, sizeof(float) * vc->audio_channels * vlen);
	for (i = 0; i < vc->audio_channels; ++i) memset(floor_ptr[i], 0, vlen * sizeof(floor_ptr[0][0]));

	// Decode floor
	for (i = 0; i < vc->audio_channels; ++i) {
		vorbis_floor* floor = mapping->submaps > 1
			? &vc->floors[mapping->submap_floor[mapping->mux[i]]] : &vc->floors[mapping->submap_floor[0]];

		const int ret = floor->floor_type == 1
			? vorbis_floor1_decode(vc, &floor->t1, floor_ptr[i])
			: vorbis_floor0_decode(vc, &floor->t0, floor_ptr[i]);

		if (ret < 0) return ERROR_INVALID;
		no_residue[i] = ret;
	}

	// Nonzero vector propagate
	for (i = mapping->coupling_steps - 1; i >= 0; --i) {
		if (!(no_residue[mapping->magnitude[i]] & no_residue[mapping->angle[i]])) {
			no_residue[mapping->magnitude[i]] = 0;
			no_residue[mapping->angle[i]] = 0;
		}
	}

	// Decode residue
	for (i = 0; i < mapping->submaps; ++i) {
		vorbis_residue* residue;
		unsigned ch = 0;

		for (j = 0; j < vc->audio_channels; ++j) {
			if ((mapping->submaps == 1) || (i == mapping->mux[j])) {
				res_chan[j] = res_num;
				do_not_decode[ch] = no_residue[j] ? 1 : 0;
				++ch;
				++res_num;
			}
		}
		residue = &vc->residues[mapping->submap_residue[i]];
		if (ch_left < ch) return ERROR_INVALID;
		if (ch) {
			const int ret = vorbis_residue_decode(vc, residue, ch, do_not_decode, ch_res_ptr, vlen, ch_left);
			if (ret < 0) return ret;
		}

		ch_res_ptr += ch * vlen;
		ch_left -= ch;
	}

	if (ch_left > 0) return ERROR_INVALID;

	// Inverse coupling
	for (i = mapping->coupling_steps - 1; i >= 0; --i) {
		float* mag = vc->channel_residues + res_chan[mapping->magnitude[i]] * blocksize / 2;
		float* ang = vc->channel_residues + res_chan[mapping->angle[i]] * blocksize / 2;
		vorbis_inverse_coupling(mag, ang, blocksize / 2);
	}

	// Dotproduct, MDCT
	for (j = vc->audio_channels - 1; j >= 0; j--) {
		ch_res_ptr = vc->channel_residues + res_chan[j] * blocksize / 2;
		if (no_residue[j]) {
			memset(ch_res_ptr, 0, (blocksize / 2) * sizeof(float));
			continue;
		}
		float* floor_channel = floor_ptr[j];
		for (unsigned n = 0; n < blocksize / 2; n++) floor_channel[n] *= ch_res_ptr[n];
		imdct_half(&vc->mdct[blockflag], ch_res_ptr, floor_channel);
	}

	// Overlap/add, save data for next overlapping
	retlen = (blocksize + vc->blocksize[previous_window]) / 4;
	for (j = 0; j < vc->audio_channels; j++) {
		const unsigned bs0 = vc->blocksize[0];
		const unsigned bs1 = vc->blocksize[1];
		float* residue = vc->channel_residues + res_chan[j] * blocksize / 2;
		float* saved = vc->saved + j * bs1 / 4;
		float* ret = floor_ptr[j];
		float* buf = residue;
		const float* win = vc->win[blockflag & previous_window];

		if (blockflag == (unsigned)previous_window) {
			vector_fmul_window(ret, saved, buf, win, blocksize / 4);
		} else if (blockflag > (unsigned)previous_window) {
			vector_fmul_window(ret, saved, buf, win, bs0 / 4);
			memcpy(ret + bs0 / 2, buf + bs0 / 4, ((bs1 - bs0) / 4) * sizeof(float));
		} else {
			memcpy(ret, saved, ((bs1 - bs0) / 4) * sizeof(float));
			vector_fmul_window(ret + (bs1 - bs0) / 4, saved + (bs1 - bs0) / 4, buf, win, bs0 / 4);
		}
		memcpy(saved, buf + blocksize / 4, blocksize / 4 * sizeof(float));
	}

	vc->previous_window = blockflag;
	return retlen;
}

/** Channel `ch` of the output, in FFmpeg's order; its buffer holds blocksize[1] / 2 samples. */
static inline float* output_channel(vorbis_context* vc, int ch)
{
	return vc->output + ch * (vc->blocksize[1] / 2);
}

///////////////////////////////////////////////////////////////////////////////
// Boundary

extern "C" {

EMSCRIPTEN_KEEPALIVE vorbis_context* vorbis_open()
{
	return (vorbis_context*)calloc(1, sizeof(vorbis_context));
}

EMSCRIPTEN_KEEPALIVE void vorbis_close(vorbis_context* vc)
{
	vorbis_free(vc);
	free(vc);
}

/**
 * The identification and setup header packets, whole, with INPUT_PADDING zero bytes free past each. 1, 0 for a header
 * FFmpeg's decoder refuses.
 */
EMSCRIPTEN_KEEPALIVE int vorbis_set_headers(vorbis_context* vc, uint8_t* id, int id_size, uint8_t* setup, int setup_size)
{
	GetBitContext* gb = &vc->gb;

	memset(id + id_size, 0, INPUT_PADDING);
	memset(setup + setup_size, 0, INPUT_PADDING);

	init_get_bits(gb, id, id_size);
	if (get_bits(gb, 8) != 1 || vorbis_parse_id_hdr(vc)) return 0;

	init_get_bits(gb, setup, setup_size);
	if (get_bits(gb, 8) != 5 || vorbis_parse_setup_hdr(vc)) return 0;

	return 1;
}

EMSCRIPTEN_KEEPALIVE int vorbis_sample_rate(vorbis_context* vc) { return vc->audio_samplerate; }
EMSCRIPTEN_KEEPALIVE int vorbis_channels(vorbis_context* vc) { return vc->audio_channels; }
EMSCRIPTEN_KEEPALIVE int vorbis_blocksize(vorbis_context* vc, int blockflag) { return vc->blocksize[blockflag]; }
EMSCRIPTEN_KEEPALIVE int vorbis_mode_count(vorbis_context* vc) { return vc->mode_count; }
EMSCRIPTEN_KEEPALIVE int vorbis_mode_blockflag(vorbis_context* vc, int mode) { return vc->modes[mode].blockflag; }

/** Seek: vorbis_decode_flush. */
EMSCRIPTEN_KEEPALIVE void vorbis_reset(vorbis_context* vc)
{
	memset(vc->saved, 0, (vc->blocksize[1] / 4) * vc->audio_channels * sizeof(float));
	vc->previous_window = -1;
}

/** Decodes one whole packet, with INPUT_PADDING bytes free past it: the samples it gives, 0 for none or a corrupt packet. */
EMSCRIPTEN_KEEPALIVE int vorbis_decode_packet(vorbis_context* vc, uint8_t* buf, int size)
{
	float* channel_ptrs[255];

	memset(buf + size, 0, INPUT_PADDING);
	if (size < 1 || (buf[0] & 1)) return 0;

	if (vc->audio_channels > 8) {
		for (int i = 0; i < vc->audio_channels; i++) channel_ptrs[i] = output_channel(vc, i);
	} else {
		for (int i = 0; i < vc->audio_channels; i++) {
			channel_ptrs[ff_vorbis_channel_layout_offsets[vc->audio_channels - 1][i]] = output_channel(vc, i);
		}
	}

	init_get_bits(&vc->gb, buf, size);
	const int len = vorbis_parse_audio_packet(vc, channel_ptrs);
	return len > 0 ? len : 0;
}

EMSCRIPTEN_KEEPALIVE float* vorbis_channel_ptr(vorbis_context* vc, int channel) { return output_channel(vc, channel); }

}	// extern "C"
