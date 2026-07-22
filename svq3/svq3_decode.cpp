/*
	Sorenson Video 3 (SVQ3) video decoder — WASM C++.

	Port of FFmpeg's SVQ3 decoder (libavcodec, release/8.1, LGPL): svq3.c plus the
	reachable 8-bit / 4:2:0 subset of the H.264 kernels it calls (intra prediction,
	the chroma-DC IDCT, half-pel / third-pel motion comp, edge emulation) and the
	interleaved exp-Golomb reader. SVQ3 is a *draft* of H.264 with its OWN inverse
	transform ((13,13,7,17) butterfly), third-pel motion comp, and no in-loop
	deblock; it is decoupled from H264Context in 8.1 (its SVQ3Context just embeds the
	DSP tables), which is what makes this a bounded, VP6-style port.

	The libavcodec framework (AVFrame / AVCodecContext / the DSP function-pointer
	dispatch / threading / AVERROR) is replaced by a self-contained handle (SVQ3Ctx)
	owning a 3-slot Y/U/V frame pool + rotating cur/last/next references, mirroring
	the VP6/Sorenson ports (see plugins/vp6/vp6_decode.cpp).

	-fwrapv is REQUIRED: the (13,13,7,17) transform + dequant rely on two's-complement
	wrap to be byte-identical to FFmpeg.

	Output (planar-YCbCr GPU path): Y stride = codedWidth, U/V stride = codedWidth/2,
	full-range (yuvj420p). Boundary is extern "C" (see svq3_decode_core.js).

	==== STATUS: PORT IN PROGRESS (foundation) ====
	Implemented: bit-reader + interleaved golomb, const tables, the two SVQ3
	transform kernels, svq3_decode_block. TODO: intra-pred / hpel / tpel / videodsp
	kernels, the SVQ3Context spine (decode_mb / slice header / extradata / MC glue),
	frame pool + lifecycle, and the extern "C" surface.
*/

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#ifdef __wasm_simd128__
#include <wasm_simd128.h>
#endif

#include "svq3_gen_tables.h"   // interleaved golomb tables + ff_zigzag_scan (generated from ffmpeg)

// Decode-abort flag (mirrors FFmpeg's AVERROR returns). Set on error; the
// macroblock loop bails and svq3_decode_frame returns 0. Reset each decode.
static int g_err = 0;

// SIMD kernels on by default; svq3_set_simd(0|1) toggles scalar/SIMD for the A/B
// conformance test (Svq3Wasm.test.js). The scalar kernels are the ffmpeg-bit-exact
// oracle; every SIMD kernel is proven byte-identical to its scalar twin.
static int g_simd = 1;

// ─── small helpers ────────────────────────────────────────────────────────────
static inline int av_clip(int a, int amin, int amax) { return a < amin ? amin : (a > amax ? amax : a); }
static inline int av_clip_uint8(int a) { return (a & (~255)) ? ((~a) >> 31) & 0xFF : a; }
// Macros (not inlines) so operand types promote exactly as in FFmpeg (unsigned mb_type etc.)
#define FFMIN(a, b) ((a) > (b) ? (b) : (a))
#define FFMAX(a, b) ((a) > (b) ? (a) : (b))
#define FFABS(a)    ((a) >= 0 ? (a) : (-(a)))
#define FFSWAP(type, a, b) do { type SWAP_tmp = (b); (b) = (a); (a) = SWAP_tmp; } while (0)

// median of three (== FFmpeg mid_pred, bit-exact)
static inline int mid_pred(int a, int b, int c) {
	if (a > b) { if (c > b) return c > a ? a : c; else return b; }
	else       { if (c > a) return c > b ? b : c; else return a; }
}

static inline int av_log2(unsigned v) { return v ? (31 - __builtin_clz(v)) : 0; }

// endian access (WASM is little-endian; AV_RB* byte-swaps, AV_RL*/AV_WL*/AV_RN32A native)
static inline unsigned AV_RB16(const uint8_t* p) { return ((unsigned)p[0] << 8) | p[1]; }
static inline uint32_t AV_RB32(const uint8_t* p) { return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) | ((uint32_t)p[2] << 8) | p[3]; }
static inline uint32_t AV_RL32(const uint8_t* p) { return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24); }
static inline void     AV_WL32(uint8_t* p, uint32_t v) { p[0] = v; p[1] = v >> 8; p[2] = v >> 16; p[3] = v >> 24; }
static inline uint32_t AV_RN32A(const void* p) { return *(const uint32_t*)p; }
static inline void     AV_WN32A(void* p, uint32_t v) { *(uint32_t*)p = v; }
static inline void     AV_COPY32(void* d, const void* s) { AV_WN32A(d, AV_RN32A(s)); }
static inline void     AV_ZERO32(void* d) { AV_WN32A(d, 0); }
static inline void     AV_ZERO128(void* d) { memset(d, 0, 16); }
static inline int32_t  pack16to32(int a, int b) { return (a & 0xFFFF) | (b << 16); }

#define INVALID_VLC 0x80000000

// ─── bit reader (FFmpeg get_bits.h, non-cached MSB-first semantics) ────────────
// Inputs are zero-padded by the caller, so reading past size_in_bits yields 0 —
// equivalent to FFmpeg's padded reader. size_in_bits is always byte-aligned for
// SVQ3 (slice/extradata/packet), so bit-level end handling is exact.
struct GetBitContext {
	const uint8_t* buffer;
	int index;
	int size_in_bits;
	int size_in_bits_plus8;
};

// Next 32 bits MSB-aligned from s->index (== FFmpeg GET_CACHE). Fast path: one
// 40-bit big-endian load (5 bytes) + a shift, instead of a 32-iteration bit loop —
// the entropy hot path. Near the end (last <5 bytes) a zero-padded assembly keeps
// the read-past-end == 0 semantics (size_in_bits is byte-aligned for SVQ3, so
// "beyond size" == "beyond the byte count"). This mirrors the register-refill idea
// in plugins/sorenson/sorenson_decode.cpp's BitReader, kept on the index interface.
static inline uint32_t peek_bits32(const GetBitContext* s) {
	const int idx = s->index, byte = idx >> 3, bit = idx & 7;
	const int nbytes = s->size_in_bits >> 3;
	const uint8_t* b = s->buffer;
	uint64_t v;
	if (byte + 5 <= nbytes) {
		v = ((uint64_t)b[byte] << 32) | ((uint64_t)b[byte + 1] << 24) |
		    ((uint64_t)b[byte + 2] << 16) | ((uint64_t)b[byte + 3] << 8) | (uint64_t)b[byte + 4];
	} else {
		v = 0;
		for (int k = 0; k < 5; k++) v = (v << 8) | (uint64_t)(byte + k < nbytes ? b[byte + k] : 0u);
	}
	return (uint32_t)(v >> (8 - bit));
}
static inline unsigned peek_bits(const GetBitContext* s, int n) {
	return n <= 0 ? 0u : (unsigned)(peek_bits32(s) >> (32 - n));
}
static inline unsigned get_bits(GetBitContext* s, int n) { unsigned v = peek_bits(s, n); s->index += n; return v; }
static inline unsigned get_bits1(GetBitContext* s) { return get_bits(s, 1); }
static inline unsigned show_bits(GetBitContext* s, int n) { return peek_bits(s, n); }
static inline void skip_bits(GetBitContext* s, int n) { s->index += n; }
static inline void skip_bits1(GetBitContext* s) { s->index += 1; }
static inline void skip_bits_long(GetBitContext* s, int n) { s->index += n; }
static inline int get_bits_count(const GetBitContext* s) { return s->index; }
static inline int get_bits_left(const GetBitContext* s) { return s->size_in_bits - s->index; }

static inline int init_get_bits(GetBitContext* s, const uint8_t* buffer, int bit_size) {
	int ret = 0;
	if (bit_size >= INT_MAX - 64 || bit_size < 0 || !buffer) { bit_size = 0; buffer = NULL; ret = -1; }
	s->buffer = buffer; s->size_in_bits = bit_size; s->size_in_bits_plus8 = bit_size + 8; s->index = 0;
	return ret;
}
static inline int init_get_bits8(GetBitContext* s, const uint8_t* buffer, int byte_size) {
	if (byte_size > INT_MAX / 8 || byte_size < 0) byte_size = -1;
	return init_get_bits(s, buffer, byte_size * 8);
}
static inline int skip_1stop_8data_bits(GetBitContext* gb) {
	if (get_bits_left(gb) <= 0) return -1;
	while (get_bits1(gb)) { skip_bits(gb, 8); if (get_bits_left(gb) <= 0) return -1; }
	return 0;
}

// interleaved exp-Golomb (FFmpeg golomb.h, non-cached branch)
static inline unsigned get_interleaved_ue_golomb(GetBitContext* gb) {
	uint32_t buf = peek_bits32(gb);
	if (buf & 0xAA800000) {
		buf >>= 32 - 8;
		skip_bits(gb, ff_interleaved_golomb_vlc_len[buf]);
		return ff_interleaved_ue_golomb_vlc_code[buf];
	} else {
		unsigned ret = 1;
		do {
			buf >>= 32 - 8;
			skip_bits(gb, FFMIN(ff_interleaved_golomb_vlc_len[buf], 8));
			if (ff_interleaved_golomb_vlc_len[buf] != 9) {
				ret <<= (ff_interleaved_golomb_vlc_len[buf] - 1) >> 1;
				ret  |= ff_interleaved_dirac_golomb_vlc_code[buf];
				break;
			}
			ret = (ret << 4) | ff_interleaved_dirac_golomb_vlc_code[buf];
			buf = peek_bits32(gb);
		} while (ret < 0x8000000U && get_bits_left(gb) > 0);
		return ret - 1;
	}
}
static inline int get_interleaved_se_golomb(GetBitContext* gb) {
	uint32_t buf = peek_bits32(gb);
	if (buf & 0xAA800000) {
		buf >>= 32 - 8;
		skip_bits(gb, ff_interleaved_golomb_vlc_len[buf]);
		return ff_interleaved_se_golomb_vlc_code[buf];
	} else {
		int log;
		skip_bits(gb, 8);
		buf |= 1 | (show_bits(gb, 24));
		if ((buf & 0xAAAAAAAA) == 0)
			return INVALID_VLC;
		for (log = 31; (buf & 0x80000000) == 0; log--)
			buf = (buf << 2) - ((buf << log) >> (log - 1)) + (buf >> 30);
		skip_bits(gb, 63 - 2 * log - 8);
		return (signed)(((((buf << log) >> log) - 1) ^ -(int)(buf & 0x1)) + 1) >> 1;
	}
}

// ─── scan / prediction tables ──────────────────────────────────────────────────
// This table must match scan8[constant] usage; from h264_parse.h.
static const uint8_t scan8[16 * 3 + 3] = {
	4 +  1 * 8, 5 +  1 * 8, 4 +  2 * 8, 5 +  2 * 8,
	6 +  1 * 8, 7 +  1 * 8, 6 +  2 * 8, 7 +  2 * 8,
	4 +  3 * 8, 5 +  3 * 8, 4 +  4 * 8, 5 +  4 * 8,
	6 +  3 * 8, 7 +  3 * 8, 6 +  4 * 8, 7 +  4 * 8,
	4 +  6 * 8, 5 +  6 * 8, 4 +  7 * 8, 5 +  7 * 8,
	6 +  6 * 8, 7 +  6 * 8, 6 +  7 * 8, 7 +  7 * 8,
	4 +  8 * 8, 5 +  8 * 8, 4 +  9 * 8, 5 +  9 * 8,
	6 +  8 * 8, 7 +  8 * 8, 6 +  9 * 8, 7 +  9 * 8,
	4 + 11 * 8, 5 + 11 * 8, 4 + 12 * 8, 5 + 12 * 8,
	6 + 11 * 8, 7 + 11 * 8, 6 + 12 * 8, 7 + 12 * 8,
	4 + 13 * 8, 5 + 13 * 8, 4 + 14 * 8, 5 + 14 * 8,
	6 + 13 * 8, 7 + 13 * 8, 6 + 14 * 8, 7 + 14 * 8,
	0 + 0 * 8, 0 + 5 * 8, 0 + 10 * 8
};

// SVQ3-private tables (svq3.c) — lifted verbatim.
static const uint8_t svq3_scan[16] = {
	0 + 0 * 4, 1 + 0 * 4, 2 + 0 * 4, 2 + 1 * 4,
	2 + 2 * 4, 3 + 0 * 4, 3 + 1 * 4, 3 + 2 * 4,
	0 + 1 * 4, 0 + 2 * 4, 1 + 1 * 4, 1 + 2 * 4,
	0 + 3 * 4, 1 + 3 * 4, 2 + 3 * 4, 3 + 3 * 4,
};
static const uint8_t luma_dc_zigzag_scan[16] = {
	0 * 16 + 0 * 64, 1 * 16 + 0 * 64, 2 * 16 + 0 * 64, 0 * 16 + 2 * 64,
	3 * 16 + 0 * 64, 0 * 16 + 1 * 64, 1 * 16 + 1 * 64, 2 * 16 + 1 * 64,
	1 * 16 + 2 * 64, 2 * 16 + 2 * 64, 3 * 16 + 2 * 64, 0 * 16 + 3 * 64,
	3 * 16 + 1 * 64, 1 * 16 + 3 * 64, 2 * 16 + 3 * 64, 3 * 16 + 3 * 64,
};
static const uint8_t svq3_pred_0[25][2] = {
	{ 0, 0 },
	{ 1, 0 }, { 0, 1 },
	{ 0, 2 }, { 1, 1 }, { 2, 0 },
	{ 3, 0 }, { 2, 1 }, { 1, 2 }, { 0, 3 },
	{ 0, 4 }, { 1, 3 }, { 2, 2 }, { 3, 1 }, { 4, 0 },
	{ 4, 1 }, { 3, 2 }, { 2, 3 }, { 1, 4 },
	{ 2, 4 }, { 3, 3 }, { 4, 2 },
	{ 4, 3 }, { 3, 4 },
	{ 4, 4 }
};
static const int8_t svq3_pred_1[6][6][5] = {
	{ { 2, -1, -1, -1, -1 }, { 2, 1, -1, -1, -1 }, { 1, 2, -1, -1, -1 },
	  { 2,  1, -1, -1, -1 }, { 1, 2, -1, -1, -1 }, { 1, 2, -1, -1, -1 } },
	{ { 0,  2, -1, -1, -1 }, { 0, 2,  1,  4,  3 }, { 0, 1,  2,  4,  3 },
	  { 0,  2,  1,  4,  3 }, { 2, 0,  1,  3,  4 }, { 0, 4,  2,  1,  3 } },
	{ { 2,  0, -1, -1, -1 }, { 2, 1,  0,  4,  3 }, { 1, 2,  4,  0,  3 },
	  { 2,  1,  0,  4,  3 }, { 2, 1,  4,  3,  0 }, { 1, 2,  4,  0,  3 } },
	{ { 2,  0, -1, -1, -1 }, { 2, 0,  1,  4,  3 }, { 1, 2,  0,  4,  3 },
	  { 2,  1,  0,  4,  3 }, { 2, 1,  3,  4,  0 }, { 2, 4,  1,  0,  3 } },
	{ { 0,  2, -1, -1, -1 }, { 0, 2,  1,  3,  4 }, { 1, 2,  3,  0,  4 },
	  { 2,  0,  1,  3,  4 }, { 2, 1,  3,  0,  4 }, { 2, 0,  4,  3,  1 } },
	{ { 0,  2, -1, -1, -1 }, { 0, 2,  4,  1,  3 }, { 1, 4,  2,  0,  3 },
	  { 4,  2,  0,  1,  3 }, { 2, 0,  1,  4,  3 }, { 4, 2,  1,  0,  3 } },
};
static const struct { uint8_t run; uint8_t level; } svq3_dct_tables[2][16] = {
	{ { 0, 0 }, { 0, 1 }, { 1, 1 }, { 2, 1 }, { 0, 2 }, { 3, 1 }, { 4, 1 }, { 5, 1 },
	  { 0, 3 }, { 1, 2 }, { 2, 2 }, { 6, 1 }, { 7, 1 }, { 8, 1 }, { 9, 1 }, { 0, 4 } },
	{ { 0, 0 }, { 0, 1 }, { 1, 1 }, { 0, 2 }, { 2, 1 }, { 0, 3 }, { 0, 4 }, { 0, 5 },
	  { 3, 1 }, { 4, 1 }, { 1, 2 }, { 1, 3 }, { 0, 6 }, { 0, 7 }, { 0, 8 }, { 0, 9 } }
};
static const uint32_t svq3_dequant_coeff[32] = {
	 3881,  4351,  4890,  5481,   6154,   6914,   7761,   8718,
	 9781, 10987, 12339, 13828,  15523,  17435,  19561,  21873,
	24552, 27656, 30847, 34870,  38807,  43747,  49103,  54683,
	61694, 68745, 77615, 89113, 100253, 109366, 126635, 141533
};

// H.264-shared tables that SVQ3 references (h264data.c) — reachable subset.
static const uint8_t ff_h264_golomb_to_intra4x4_cbp[48] = {
	47, 31, 15, 0,  23, 27, 29, 30, 7,  11, 13, 14, 39, 43, 45, 46,
	16, 3,  5,  10, 12, 19, 21, 26, 28, 35, 37, 42, 44, 1,  2,  4,
	8,  17, 18, 20, 24, 6,  9,  22, 25, 32, 33, 34, 36, 40, 38, 41
};
static const uint8_t ff_h264_golomb_to_inter_cbp[48] = {
	0,  16, 1,  2,  4,  8,  32, 3,  5,  10, 12, 15, 47, 7,  11, 13,
	14, 6,  9,  31, 35, 37, 42, 44, 33, 34, 36, 40, 39, 43, 45, 46,
	17, 18, 20, 24, 19, 21, 26, 28, 23, 27, 29, 30, 22, 25, 38, 41
};
static const uint8_t ff_h264_chroma_dc_scan[4] = {
	(0 + 0 * 2) * 16, (1 + 0 * 2) * 16,
	(0 + 1 * 2) * 16, (1 + 1 * 2) * 16,
};
static const uint8_t ff_h264_dequant4_coeff_init[6][3] = {
	{ 10, 13, 16 }, { 11, 14, 18 }, { 13, 16, 20 },
	{ 14, 18, 23 }, { 16, 20, 25 }, { 18, 23, 29 },
};
#define QP_MAX_NUM (51 + 6 * 6)
static const uint8_t ff_h264_quant_rem6[QP_MAX_NUM + 1] = {
	0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2,
	3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5,
	0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2,
	3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5,
	0, 1, 2, 3,
};
static const uint8_t ff_h264_quant_div6[QP_MAX_NUM + 1] = {
	0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3,  3,  3,
	3, 3, 3, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6,  6,  6,
	7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10,
	10,10,10,11,11,11,11,11,11,12,12,12,12,12,12,13,13,13, 13, 13, 13,
	14,14,14,14,
};
// ff_h264_chroma_qp row 0 (bit depth 8) — the only row SVQ3 indexes: [0][qscale+12].
static const uint8_t ff_h264_chroma_qp_row0[52] = {
	 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
	16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 29, 30,
	31, 32, 32, 33, 34, 34, 35, 35, 36, 36, 37, 37, 37, 38, 38, 38,
	39, 39, 39, 39
};

// pict types (subset of AVPictureType)
enum { PICT_NONE = 0, PICT_I = 1, PICT_P = 2, PICT_B = 3, PICT_SP = 6, PICT_SI = 5 };
static const uint8_t ff_h264_golomb_to_pict_type[5] = { PICT_P, PICT_B, PICT_I, PICT_SP, PICT_SI };

// ff_h264_i_mb_type_info: SVQ3 reads only .pred_mode and .cbp (index mb_type-8).
struct IMbInfo { int8_t pred_mode; int8_t cbp; };
static const IMbInfo ff_h264_i_mb_type_info[26] = {
	{ -1, -1 }, { 2, 0 }, { 1, 0 }, { 0, 0 }, { 3, 0 },
	{ 2, 16 }, { 1, 16 }, { 0, 16 }, { 3, 16 },
	{ 2, 32 }, { 1, 32 }, { 0, 32 }, { 3, 32 },
	{ 2, 15 }, { 1, 15 }, { 0, 15 }, { 3, 15 },
	{ 2, 31 }, { 1, 31 }, { 0, 31 }, { 3, 31 },
	{ 2, 47 }, { 1, 47 }, { 0, 47 }, { 3, 47 },
	{ -1, -1 },
};

// ─── SVQ3 inverse transform kernels (svq3.c) — lifted verbatim ─────────────────
static void svq3_luma_dc_dequant_idct_scalar(int16_t* output, int16_t* input, int qp) {
	const unsigned qmul = svq3_dequant_coeff[qp];
#define stride 16
	int i;
	int temp[16];
	static const uint8_t x_offset[4] = { 0, 1 * stride, 4 * stride, 5 * stride };

	for (i = 0; i < 4; i++) {
		const int z0 = 13 * (input[4 * i + 0] +      input[4 * i + 2]);
		const int z1 = 13 * (input[4 * i + 0] -      input[4 * i + 2]);
		const int z2 =  7 *  input[4 * i + 1] - 17 * input[4 * i + 3];
		const int z3 = 17 *  input[4 * i + 1] +  7 * input[4 * i + 3];
		temp[4 * i + 0] = z0 + z3;
		temp[4 * i + 1] = z1 + z2;
		temp[4 * i + 2] = z1 - z2;
		temp[4 * i + 3] = z0 - z3;
	}
	for (i = 0; i < 4; i++) {
		const int offset = x_offset[i];
		const int z0     = 13 * (temp[4 * 0 + i] +      temp[4 * 2 + i]);
		const int z1     = 13 * (temp[4 * 0 + i] -      temp[4 * 2 + i]);
		const int z2     =  7 *  temp[4 * 1 + i] - 17 * temp[4 * 3 + i];
		const int z3     = 17 *  temp[4 * 1 + i] +  7 * temp[4 * 3 + i];
		output[stride *  0 + offset] = (int)((z0 + z3) * qmul + 0x80000) >> 20;
		output[stride *  2 + offset] = (int)((z1 + z2) * qmul + 0x80000) >> 20;
		output[stride *  8 + offset] = (int)((z1 - z2) * qmul + 0x80000) >> 20;
		output[stride * 10 + offset] = (int)((z0 - z3) * qmul + 0x80000) >> 20;
	}
#undef stride
}

static void svq3_add_idct_scalar(uint8_t* dst, int16_t* block, int stride, int qp, int dc) {
	const int qmul = svq3_dequant_coeff[qp];
	int i;

	if (dc) {
		dc       = 13 * 13 * (dc == 1 ? 1538U * block[0]
		                              : qmul * (block[0] >> 3) / 2);
		block[0] = 0;
	}

	for (i = 0; i < 4; i++) {
		const int z0 = 13 * (block[0 + 4 * i] +      block[2 + 4 * i]);
		const int z1 = 13 * (block[0 + 4 * i] -      block[2 + 4 * i]);
		const int z2 =  7 *  block[1 + 4 * i] - 17 * block[3 + 4 * i];
		const int z3 = 17 *  block[1 + 4 * i] +  7 * block[3 + 4 * i];
		block[0 + 4 * i] = z0 + z3;
		block[1 + 4 * i] = z1 + z2;
		block[2 + 4 * i] = z1 - z2;
		block[3 + 4 * i] = z0 - z3;
	}

	for (i = 0; i < 4; i++) {
		const unsigned z0 = 13 * (block[i + 4 * 0] +      block[i + 4 * 2]);
		const unsigned z1 = 13 * (block[i + 4 * 0] -      block[i + 4 * 2]);
		const unsigned z2 =  7 *  block[i + 4 * 1] - 17 * block[i + 4 * 3];
		const unsigned z3 = 17 *  block[i + 4 * 1] +  7 * block[i + 4 * 3];
		const int rr = (dc + 0x80000u);
		dst[i + stride * 0] = av_clip_uint8(dst[i + stride * 0] + ((int)((z0 + z3) * qmul + rr) >> 20));
		dst[i + stride * 1] = av_clip_uint8(dst[i + stride * 1] + ((int)((z1 + z2) * qmul + rr) >> 20));
		dst[i + stride * 2] = av_clip_uint8(dst[i + stride * 2] + ((int)((z1 - z2) * qmul + rr) >> 20));
		dst[i + stride * 3] = av_clip_uint8(dst[i + stride * 3] + ((int)((z0 - z3) * qmul + rr) >> 20));
	}

	memset(block, 0, 16 * sizeof(int16_t));
}

#ifdef __wasm_simd128__
// 4x4 in-place transpose of four i32x4 vectors (rows <-> columns).
static inline void transpose4_i32(v128_t& a, v128_t& b, v128_t& c, v128_t& d) {
	v128_t t0 = wasm_i32x4_shuffle(a, b, 0, 4, 1, 5);
	v128_t t1 = wasm_i32x4_shuffle(a, b, 2, 6, 3, 7);
	v128_t t2 = wasm_i32x4_shuffle(c, d, 0, 4, 1, 5);
	v128_t t3 = wasm_i32x4_shuffle(c, d, 2, 6, 3, 7);
	a = wasm_i32x4_shuffle(t0, t2, 0, 1, 4, 5);
	b = wasm_i32x4_shuffle(t0, t2, 2, 3, 6, 7);
	c = wasm_i32x4_shuffle(t1, t3, 0, 1, 4, 5);
	d = wasm_i32x4_shuffle(t1, t3, 2, 3, 6, 7);
}

// Add 4 dst bytes at p to s (i32x4), av_clip_uint8, store back. (saturating
// i32->i16->u8 narrows == the [0,255] clamp; values are within i16 range.)
static inline void store_add_clip4(uint8_t* p, v128_t s) {
	v128_t d = wasm_u32x4_extend_low_u16x8(wasm_u16x8_extend_low_u8x16(wasm_v128_load32_zero(p)));
	v128_t sum = wasm_i32x4_add(s, d);
	v128_t p16 = wasm_i16x8_narrow_i32x4(sum, sum);
	v128_t p8  = wasm_u8x16_narrow_i16x8(p16, p16);
	wasm_v128_store32_lane(p, p8, 0);
}

// SIMD twin of svq3_add_idct_scalar — byte-identical (verified via svq3_test_idct).
// Pass 1 is the exact (13,13,7,17) row butterfly, truncated to int16 (matching the
// scalar int16 block store); pass 2 is the column butterfly in unsigned/wrapping
// i32 (i32x4_mul yields the low 32 bits == scalar `unsigned` mult), *qmul + rr,
// signed >>20, + dst, clip. Two transposes (one per pass axis).
static void svq3_add_idct_simd(uint8_t* dst, int16_t* block, int stride, int qp, int dc) {
	const int qmul = svq3_dequant_coeff[qp];   // int (matches scalar): dc==2 path needs signed /2
	if (dc) {
		dc       = 13 * 13 * (dc == 1 ? 1538U * block[0] : qmul * (block[0] >> 3) / 2);
		block[0] = 0;
	}
	const v128_t k13 = wasm_i32x4_splat(13), k7 = wasm_i32x4_splat(7), k17 = wasm_i32x4_splat(17);

	v128_t r0 = wasm_i32x4_load16x4(block + 0);
	v128_t r1 = wasm_i32x4_load16x4(block + 4);
	v128_t r2 = wasm_i32x4_load16x4(block + 8);
	v128_t r3 = wasm_i32x4_load16x4(block + 12);
	transpose4_i32(r0, r1, r2, r3);          // r0..r3 now = columns (lane = row)

	// pass 1: row transform, lane-wise across rows, truncate result to int16
	v128_t z0 = wasm_i32x4_mul(k13, wasm_i32x4_add(r0, r2));
	v128_t z1 = wasm_i32x4_mul(k13, wasm_i32x4_sub(r0, r2));
	v128_t z2 = wasm_i32x4_sub(wasm_i32x4_mul(k7, r1), wasm_i32x4_mul(k17, r3));
	v128_t z3 = wasm_i32x4_add(wasm_i32x4_mul(k17, r1), wasm_i32x4_mul(k7, r3));
	v128_t o0 = wasm_i32x4_add(z0, z3), o1 = wasm_i32x4_add(z1, z2);
	v128_t o2 = wasm_i32x4_sub(z1, z2), o3 = wasm_i32x4_sub(z0, z3);
	o0 = wasm_i32x4_shr(wasm_i32x4_shl(o0, 16), 16);   // (int16) truncation, sign-extended
	o1 = wasm_i32x4_shr(wasm_i32x4_shl(o1, 16), 16);
	o2 = wasm_i32x4_shr(wasm_i32x4_shl(o2, 16), 16);
	o3 = wasm_i32x4_shr(wasm_i32x4_shl(o3, 16), 16);
	transpose4_i32(o0, o1, o2, o3);          // o0..o3 now = rows of B' (lane = column)

	// pass 2: column transform, unsigned/wrapping i32, *qmul + rr, >>20, + dst, clip
	v128_t w0 = wasm_i32x4_mul(k13, wasm_i32x4_add(o0, o2));
	v128_t w1 = wasm_i32x4_mul(k13, wasm_i32x4_sub(o0, o2));
	v128_t w2 = wasm_i32x4_sub(wasm_i32x4_mul(k7, o1), wasm_i32x4_mul(k17, o3));
	v128_t w3 = wasm_i32x4_add(wasm_i32x4_mul(k17, o1), wasm_i32x4_mul(k7, o3));
	const v128_t qv  = wasm_i32x4_splat((int)qmul);
	const v128_t rrv = wasm_i32x4_splat((int)((uint32_t)dc + 0x80000u));
	v128_t s0 = wasm_i32x4_shr(wasm_i32x4_add(wasm_i32x4_mul(wasm_i32x4_add(w0, w3), qv), rrv), 20);
	v128_t s1 = wasm_i32x4_shr(wasm_i32x4_add(wasm_i32x4_mul(wasm_i32x4_add(w1, w2), qv), rrv), 20);
	v128_t s2 = wasm_i32x4_shr(wasm_i32x4_add(wasm_i32x4_mul(wasm_i32x4_sub(w1, w2), qv), rrv), 20);
	v128_t s3 = wasm_i32x4_shr(wasm_i32x4_add(wasm_i32x4_mul(wasm_i32x4_sub(w0, w3), qv), rrv), 20);
	store_add_clip4(dst + 0 * stride, s0);
	store_add_clip4(dst + 1 * stride, s1);
	store_add_clip4(dst + 2 * stride, s2);
	store_add_clip4(dst + 3 * stride, s3);

	memset(block, 0, 16 * sizeof(int16_t));
}
#endif

static inline void svq3_add_idct_c(uint8_t* dst, int16_t* block, int stride, int qp, int dc) {
#ifdef __wasm_simd128__
	if (g_simd) { svq3_add_idct_simd(dst, block, stride, qp, dc); return; }
#endif
	svq3_add_idct_scalar(dst, block, stride, qp, dc);
}

#ifdef __wasm_simd128__
// SIMD twin of svq3_luma_dc_dequant_idct_scalar. Same (13,13,7,17) butterfly, but
// pass 1 keeps full int32 (scalar temp[] is int, NOT int16-truncated), pass 2 is
// (z*qmul + 0x80000) >> 20 (no dc, no dst-add, no clip), scatter-stored int16 into
// the 16 luma-DC slots at output[{0,32,128,160}[row] + {0,16,64,80}[col]].
static void svq3_luma_dc_dequant_idct_simd(int16_t* output, int16_t* input, int qp) {
	const unsigned qmul = svq3_dequant_coeff[qp];
	const v128_t k13 = wasm_i32x4_splat(13), k7 = wasm_i32x4_splat(7), k17 = wasm_i32x4_splat(17);

	v128_t r0 = wasm_i32x4_load16x4(input + 0);
	v128_t r1 = wasm_i32x4_load16x4(input + 4);
	v128_t r2 = wasm_i32x4_load16x4(input + 8);
	v128_t r3 = wasm_i32x4_load16x4(input + 12);
	transpose4_i32(r0, r1, r2, r3);          // columns (lane = row)

	// pass 1: row transform, full int32 (no truncation)
	v128_t z0 = wasm_i32x4_mul(k13, wasm_i32x4_add(r0, r2));
	v128_t z1 = wasm_i32x4_mul(k13, wasm_i32x4_sub(r0, r2));
	v128_t z2 = wasm_i32x4_sub(wasm_i32x4_mul(k7, r1), wasm_i32x4_mul(k17, r3));
	v128_t z3 = wasm_i32x4_add(wasm_i32x4_mul(k17, r1), wasm_i32x4_mul(k7, r3));
	v128_t o0 = wasm_i32x4_add(z0, z3), o1 = wasm_i32x4_add(z1, z2);
	v128_t o2 = wasm_i32x4_sub(z1, z2), o3 = wasm_i32x4_sub(z0, z3);
	transpose4_i32(o0, o1, o2, o3);          // rows of temp (lane = column)

	// pass 2: (z*qmul + 0x80000) >> 20  (unsigned/wrapping mul, signed shift)
	v128_t w0 = wasm_i32x4_mul(k13, wasm_i32x4_add(o0, o2));
	v128_t w1 = wasm_i32x4_mul(k13, wasm_i32x4_sub(o0, o2));
	v128_t w2 = wasm_i32x4_sub(wasm_i32x4_mul(k7, o1), wasm_i32x4_mul(k17, o3));
	v128_t w3 = wasm_i32x4_add(wasm_i32x4_mul(k17, o1), wasm_i32x4_mul(k7, o3));
	const v128_t qv  = wasm_i32x4_splat((int)qmul);
	const v128_t rrv = wasm_i32x4_splat(0x80000);
	v128_t s0 = wasm_i32x4_shr(wasm_i32x4_add(wasm_i32x4_mul(wasm_i32x4_add(w0, w3), qv), rrv), 20);
	v128_t s1 = wasm_i32x4_shr(wasm_i32x4_add(wasm_i32x4_mul(wasm_i32x4_add(w1, w2), qv), rrv), 20);
	v128_t s2 = wasm_i32x4_shr(wasm_i32x4_add(wasm_i32x4_mul(wasm_i32x4_sub(w1, w2), qv), rrv), 20);
	v128_t s3 = wasm_i32x4_shr(wasm_i32x4_add(wasm_i32x4_mul(wasm_i32x4_sub(w0, w3), qv), rrv), 20);

	// scatter store (int16) to output[base + xoff]; xoff = {0,16,64,80} over lanes
#define SVQ3_DCSTORE(s, base)                                                          \
	output[(base) +  0] = (int16_t)wasm_i32x4_extract_lane((s), 0);                     \
	output[(base) + 16] = (int16_t)wasm_i32x4_extract_lane((s), 1);                     \
	output[(base) + 64] = (int16_t)wasm_i32x4_extract_lane((s), 2);                     \
	output[(base) + 80] = (int16_t)wasm_i32x4_extract_lane((s), 3)
	SVQ3_DCSTORE(s0, 16 * 0);
	SVQ3_DCSTORE(s1, 16 * 2);
	SVQ3_DCSTORE(s2, 16 * 8);
	SVQ3_DCSTORE(s3, 16 * 10);
#undef SVQ3_DCSTORE
}
#endif

static inline void svq3_luma_dc_dequant_idct_c(int16_t* output, int16_t* input, int qp) {
#ifdef __wasm_simd128__
	if (g_simd) { svq3_luma_dc_dequant_idct_simd(output, input, qp); return; }
#endif
	svq3_luma_dc_dequant_idct_scalar(output, input, qp);
}

// ─── residual block decode (svq3.c) — lifted verbatim ──────────────────────────
static inline int svq3_decode_block(GetBitContext* gb, int16_t* block, int index, const int type) {
	static const uint8_t* const scan_patterns[4] = {
		luma_dc_zigzag_scan, ff_zigzag_scan, svq3_scan, ff_h264_chroma_dc_scan
	};

	int run, level, sign, limit;
	unsigned vlc;
	const int intra           = 3 * type >> 2;
	const uint8_t* const scan = scan_patterns[type];

	for (limit = (16 >> intra); index < 16; index = limit, limit += 8) {
		for (; (vlc = get_interleaved_ue_golomb(gb)) != 0; index++) {
			if ((int32_t)vlc < 0)
				return -1;

			sign = (vlc & 1) ? 0 : -1;
			vlc  = vlc + 1 >> 1;

			if (type == 3) {
				if (vlc < 3) {
					run   = 0;
					level = vlc;
				} else if (vlc < 4) {
					run   = 1;
					level = 1;
				} else {
					run   = vlc & 0x3;
					level = (vlc + 9 >> 2) - run;
				}
			} else {
				if (vlc < 16U) {
					run   = svq3_dct_tables[intra][vlc].run;
					level = svq3_dct_tables[intra][vlc].level;
				} else if (intra) {
					run   = vlc & 0x7;
					level = (vlc >> 3) + ((run == 0) ? 8 : ((run < 2) ? 2 : ((run < 5) ? 0 : -1)));
				} else {
					run   = vlc & 0xF;
					level = (vlc >> 4) + ((run == 0) ? 4 : ((run < 3) ? 2 : ((run < 10) ? 1 : 0)));
				}
			}

			if ((index += run) >= limit)
				return -1;

			block[scan[index]] = (level ^ sign) - sign;
		}

		if (type != 2)
			break;
	}

	return 0;
}

// ─── leaf DSP kernels (ported from ffmpeg by the kernel workflow) ───────────────
// Order matters: hpel before tpel (tpel mc00 delegates to hpel's pel-copy funcs);
// intra defines the pred-mode enums (DC_PRED, DC_PRED8x8, …) the spine uses.
#include "svq3_chromaidct.inc"   // svq3_chroma_dc_dequant_idct
#include "svq3_videodsp.inc"     // svq3_emulated_edge_mc
#include "svq3_intra.inc"        // svq3_pred4x4/16x16/8x8[], svq3_pred_init, ff_h264_check_intra*_pred_mode + mode enums
#include "svq3_hpel.inc"         // svq3_put/avg_pixels_tab, svq3_hpel_init + pel-copy funcs
#include "svq3_tpel.inc"         // svq3_put/avg_tpel_pixels_tab, svq3_tpel_init

// ─── MB-type / prediction constants (mpegutils.h, h264pred.h) ──────────────────
#define MB_TYPE_INTRA4x4   (1 << 0)
#define MB_TYPE_INTRA16x16 (1 << 1)
#define MB_TYPE_INTRA_PCM  (1 << 2)
#define MB_TYPE_16x16      (1 << 3)
#define MB_TYPE_16x8       (1 << 4)
#define MB_TYPE_8x16       (1 << 5)
#define MB_TYPE_8x8        (1 << 6)
#define MB_TYPE_SKIP       (1 << 17)
#define IS_INTRA4x4(a)     ((a) & MB_TYPE_INTRA4x4)
#define IS_INTRA16x16(a)   ((a) & MB_TYPE_INTRA16x16)
#define IS_INTRA(a)        ((a) & 7)
#define IS_INTER(a)        ((a) & (MB_TYPE_16x16 | MB_TYPE_16x8 | MB_TYPE_8x16 | MB_TYPE_8x8))
#define IS_SKIP(a)         ((a) & MB_TYPE_SKIP)
#define PART_NOT_AVAILABLE -2

#define NUM_PICS      3
#define FULLPEL_MODE  1
#define HALFPEL_MODE  2
#define THIRDPEL_MODE 3
#define PREDICT_MODE  4

// SVQ3 only calls fill_rectangle with size==4 (packed motion vectors), w,h ∈ {1,2,4}.
static inline void fill_rectangle(void* vp, int w, int h, int stride, uint32_t val, int /*size*/) {
	uint8_t* p = (uint8_t*)vp;
	const int bstride = stride * 4;                 // element size == 4 bytes
	for (int y = 0; y < h; y++) {
		uint32_t* row = (uint32_t*)(p + y * bstride);
		for (int x = 0; x < w; x++) row[x] = val;
	}
}

// ─── decoder handle (de-frameworked SVQ3Context) ───────────────────────────────
struct SVQ3Frame {
	uint8_t*   data[3];
	ptrdiff_t  linesize[3];
	int        has_data;                            // replaces ffmpeg's f->data[0] != NULL validity flag
	uint32_t*  mb_type;                             // decoder-lifetime slice
	int16_t  (*motion_val[2])[2];                   // decoder-lifetime slice
};

struct SVQ3Context {
	SVQ3Frame* cur_pic;
	SVQ3Frame* next_pic;
	SVQ3Frame* last_pic;
	SVQ3Frame* output_pic;                          // frame exposed via svq3_*_ptr (may be NULL)

	GetBitContext gb;
	GetBitContext gb_slice;
	uint8_t*  slice_buf;
	unsigned  slice_buf_size;

	int halfpel_flag;
	int thirdpel_flag;
	int has_watermark;
	uint32_t watermark_key;
	int adaptive_quant;
	int h_edge_pos;
	int v_edge_pos;
	int slice_num;
	int qscale;
	int cbp;
	int frame_num;
	int frame_num_offset;
	int prev_frame_num_offset;
	int prev_frame_num;

	int pict_type;
	int slice_type;
	int low_delay;

	int mb_x, mb_y, mb_xy;
	int mb_width, mb_height;
	int mb_stride, mb_num;
	int b_stride;

	uint32_t* mb2br_xy;

	int chroma_pred_mode;
	int intra16x16_pred_mode;

	int8_t  intra4x4_pred_mode_cache[5 * 8];
	int8_t* intra4x4_pred_mode;

	unsigned int top_samples_available;
	unsigned int left_samples_available;

	uint8_t* edge_emu_buffer;

	alignas(16) int16_t mv_cache[2][5 * 8][2];
	alignas(8)  int8_t  ref_cache[2][5 * 8];
	alignas(16) int16_t mb[16 * 48 * 2];
	alignas(16) int16_t mb_luma_dc[3][16 * 2];
	alignas(8)  uint8_t non_zero_count_cache[15 * 8];
	uint32_t dequant4_coeff[QP_MAX_NUM + 1][16];
	int block_offset[2 * (16 * 3)];
	SVQ3Frame frames[NUM_PICS];

	uint32_t* mb_type_buf;
	int16_t (*motion_val_buf)[2];

	int width, height;                              // display dimensions (from SEQH)
	int buffers_ready;
};

// av_fast_padded_malloc equivalent: grow slice_buf to >= size + 64 zero-padded bytes.
static void ensure_slice_buf(SVQ3Context* s, unsigned size) {
	const unsigned need = size + 64;
	if (need > s->slice_buf_size) {
		uint8_t* nb = (uint8_t*)realloc(s->slice_buf, need);
		if (!nb) { g_err = 1; return; }
		s->slice_buf = nb;
		s->slice_buf_size = need;
	}
}

// ─── motion vector prediction (svq3.c) — verbatim ──────────────────────────────
static inline int svq3_fetch_diagonal_mv(const SVQ3Context* s, const int16_t** C,
                                         int i, int list, int part_width) {
	const int topright_ref = s->ref_cache[list][i - 8 + part_width];
	if (topright_ref != PART_NOT_AVAILABLE) {
		*C = s->mv_cache[list][i - 8 + part_width];
		return topright_ref;
	} else {
		*C = s->mv_cache[list][i - 8 - 1];
		return s->ref_cache[list][i - 8 - 1];
	}
}

static inline void svq3_pred_motion(const SVQ3Context* s, int n, int part_width,
                                    int list, int ref, int* const mx, int* const my) {
	const int index8       = scan8[n];
	const int top_ref      = s->ref_cache[list][index8 - 8];
	const int left_ref     = s->ref_cache[list][index8 - 1];
	const int16_t* const A = s->mv_cache[list][index8 - 1];
	const int16_t* const B = s->mv_cache[list][index8 - 8];
	const int16_t* C;
	int diagonal_ref, match_count;

	diagonal_ref = svq3_fetch_diagonal_mv(s, &C, index8, list, part_width);
	match_count  = (diagonal_ref == ref) + (top_ref == ref) + (left_ref == ref);
	if (match_count > 1) {
		*mx = mid_pred(A[0], B[0], C[0]);
		*my = mid_pred(A[1], B[1], C[1]);
	} else if (match_count == 1) {
		if (left_ref == ref)      { *mx = A[0]; *my = A[1]; }
		else if (top_ref == ref)  { *mx = B[0]; *my = B[1]; }
		else                      { *mx = C[0]; *my = C[1]; }
	} else {
		if (top_ref == PART_NOT_AVAILABLE && diagonal_ref == PART_NOT_AVAILABLE &&
		    left_ref != PART_NOT_AVAILABLE) {
			*mx = A[0]; *my = A[1];
		} else {
			*mx = mid_pred(A[0], B[0], C[0]);
			*my = mid_pred(A[1], B[1], C[1]);
		}
	}
}

// ─── motion compensation (svq3.c) — de-frameworked (planes + kernel tables) ────
static inline void svq3_mc_dir_part(SVQ3Context* s, int x, int y, int width, int height,
                                    int mx, int my, int dxy, int thirdpel, int dir, int avg) {
	const SVQ3Frame* pic = (dir == 0) ? s->last_pic : s->next_pic;
	uint8_t *src, *dest;
	int i, emu = 0;
	int blocksize = 2 - (width >> 3);               // 16->0, 8->1, 4->2
	int linesize   = s->cur_pic->linesize[0];
	int uvlinesize = s->cur_pic->linesize[1];

	mx += x;
	my += y;

	if (mx < 0 || mx >= s->h_edge_pos - width  - 1 ||
	    my < 0 || my >= s->v_edge_pos - height - 1) {
		emu = 1;
		mx = av_clip(mx, -16, s->h_edge_pos - width  + 15);
		my = av_clip(my, -16, s->v_edge_pos - height + 15);
	}

	dest = s->cur_pic->data[0] + x + y * linesize;
	src  = pic->data[0] + mx + my * linesize;

	if (emu) {
		svq3_emulated_edge_mc(s->edge_emu_buffer, src, linesize, linesize,
		                      width + 1, height + 1, mx, my, s->h_edge_pos, s->v_edge_pos);
		src = s->edge_emu_buffer;
	}
	if (thirdpel)
		(avg ? svq3_avg_tpel_pixels_tab : svq3_put_tpel_pixels_tab)[dxy](dest, src, linesize, width, height);
	else
		(avg ? svq3_avg_pixels_tab : svq3_put_pixels_tab)[blocksize][dxy](dest, src, linesize, height);

	// gray is never set for SVQ3 → always reconstruct chroma
	{
		mx     = mx + (mx < (int)x) >> 1;
		my     = my + (my < (int)y) >> 1;
		width  = width  >> 1;
		height = height >> 1;
		blocksize++;

		for (i = 1; i < 3; i++) {
			dest = s->cur_pic->data[i] + (x >> 1) + (y >> 1) * uvlinesize;
			src  = pic->data[i] + mx + my * uvlinesize;

			if (emu) {
				svq3_emulated_edge_mc(s->edge_emu_buffer, src, uvlinesize, uvlinesize,
				                      width + 1, height + 1, mx, my,
				                      (s->h_edge_pos >> 1), s->v_edge_pos >> 1);
				src = s->edge_emu_buffer;
			}
			if (thirdpel)
				(avg ? svq3_avg_tpel_pixels_tab : svq3_put_tpel_pixels_tab)[dxy](dest, src, uvlinesize, width, height);
			else
				(avg ? svq3_avg_pixels_tab : svq3_put_pixels_tab)[blocksize][dxy](dest, src, uvlinesize, height);
		}
	}
}

static inline int svq3_mc_dir(SVQ3Context* s, int size, int mode, int dir, int avg) {
	int i, j, k, mx, my, dx, dy, x, y;
	const int part_width    = ((size & 5) == 4) ? 4 : 16 >> (size & 1);
	const int part_height   = 16 >> ((unsigned)(size + 1) / 3);
	const int extra_width   = (mode == PREDICT_MODE) ? -16 * 6 : 0;
	const int h_edge_pos    = 6 * (s->h_edge_pos - part_width)  - extra_width;
	const int v_edge_pos    = 6 * (s->v_edge_pos - part_height) - extra_width;

	for (i = 0; i < 16; i += part_height)
		for (j = 0; j < 16; j += part_width) {
			const int b_xy = (4 * s->mb_x + (j >> 2)) + (4 * s->mb_y + (i >> 2)) * s->b_stride;
			int dxy;
			x = 16 * s->mb_x + j;
			y = 16 * s->mb_y + i;
			k = (j >> 2 & 1) + (i >> 1 & 2) + (j >> 1 & 4) + (i & 8);

			if (mode != PREDICT_MODE) {
				svq3_pred_motion(s, k, part_width >> 2, dir, 1, &mx, &my);
			} else {
				mx = s->next_pic->motion_val[0][b_xy][0] * 2;
				my = s->next_pic->motion_val[0][b_xy][1] * 2;

				if (dir == 0) {
					mx = mx * s->frame_num_offset / s->prev_frame_num_offset + 1 >> 1;
					my = my * s->frame_num_offset / s->prev_frame_num_offset + 1 >> 1;
				} else {
					mx = mx * (s->frame_num_offset - s->prev_frame_num_offset) / s->prev_frame_num_offset + 1 >> 1;
					my = my * (s->frame_num_offset - s->prev_frame_num_offset) / s->prev_frame_num_offset + 1 >> 1;
				}
			}

			mx = av_clip(mx, extra_width - 6 * x, h_edge_pos - 6 * x);
			my = av_clip(my, extra_width - 6 * y, v_edge_pos - 6 * y);

			if (mode == PREDICT_MODE) {
				dx = dy = 0;
			} else {
				dy = get_interleaved_se_golomb(&s->gb_slice);
				dx = get_interleaved_se_golomb(&s->gb_slice);
				if (dx != (int16_t)dx || dy != (int16_t)dy)
					return -1;                      // invalid MV vlc
			}

			if (mode == THIRDPEL_MODE) {
				int fx, fy;
				mx  = (mx + 1 >> 1) + dx;
				my  = (my + 1 >> 1) + dy;
				fx  = (unsigned)(mx + 0x30000) / 3 - 0x10000;
				fy  = (unsigned)(my + 0x30000) / 3 - 0x10000;
				dxy = (mx - 3 * fx) + 4 * (my - 3 * fy);
				svq3_mc_dir_part(s, x, y, part_width, part_height, fx, fy, dxy, 1, dir, avg);
				mx += mx;
				my += my;
			} else if (mode == HALFPEL_MODE || mode == PREDICT_MODE) {
				mx  = (unsigned)(mx + 1 + 0x30000) / 3 + dx - 0x10000;
				my  = (unsigned)(my + 1 + 0x30000) / 3 + dy - 0x10000;
				dxy = (mx & 1) + 2 * (my & 1);
				svq3_mc_dir_part(s, x, y, part_width, part_height, mx >> 1, my >> 1, dxy, 0, dir, avg);
				mx *= 3;
				my *= 3;
			} else {
				mx = (unsigned)(mx + 3 + 0x60000) / 6 + dx - 0x10000;
				my = (unsigned)(my + 3 + 0x60000) / 6 + dy - 0x10000;
				svq3_mc_dir_part(s, x, y, part_width, part_height, mx, my, 0, 0, dir, avg);
				mx *= 6;
				my *= 6;
			}

			if (mode != PREDICT_MODE) {
				int32_t mv = pack16to32(mx, my);
				if (part_height == 8 && i < 8) {
					AV_WN32A(s->mv_cache[dir][scan8[k] + 1 * 8], mv);
					if (part_width == 8 && j < 8)
						AV_WN32A(s->mv_cache[dir][scan8[k] + 1 + 1 * 8], mv);
				}
				if (part_width == 8 && j < 8)
					AV_WN32A(s->mv_cache[dir][scan8[k] + 1], mv);
				if (part_width == 4 || part_height == 4)
					AV_WN32A(s->mv_cache[dir][scan8[k]], mv);
			}

			fill_rectangle(s->cur_pic->motion_val[dir][b_xy], part_width >> 2, part_height >> 2,
			               s->b_stride, pack16to32(mx, my), 4);
		}

	return 0;
}

// ─── reconstruction glue (svq3.c) — de-frameworked ─────────────────────────────
static inline void hl_decode_mb_idct_luma(SVQ3Context* s, int mb_type, const int* block_offset,
                                          int linesize, uint8_t* dest_y) {
	int i;
	if (!IS_INTRA4x4(mb_type)) {
		for (i = 0; i < 16; i++)
			if (s->non_zero_count_cache[scan8[i]] || s->mb[i * 16]) {
				uint8_t* const ptr = dest_y + block_offset[i];
				svq3_add_idct_c(ptr, s->mb + i * 16, linesize, s->qscale, IS_INTRA(mb_type) ? 1 : 0);
			}
	}
}

static inline void hl_decode_mb_predict_luma(SVQ3Context* s, int mb_type, const int* block_offset,
                                             int linesize, uint8_t* dest_y) {
	int i;
	int qscale = s->qscale;

	if (IS_INTRA4x4(mb_type)) {
		for (i = 0; i < 16; i++) {
			uint8_t* const ptr = dest_y + block_offset[i];
			const int dir      = s->intra4x4_pred_mode_cache[scan8[i]];
			uint8_t* topright;
			int nnz;
			if (dir == DIAG_DOWN_LEFT_PRED || dir == VERT_LEFT_PRED)
				topright = ptr + 4 - linesize;
			else
				topright = NULL;
			svq3_pred4x4[dir](ptr, topright, linesize);
			nnz = s->non_zero_count_cache[scan8[i]];
			if (nnz)
				svq3_add_idct_c(ptr, s->mb + i * 16, linesize, qscale, 0);
		}
	} else {
		svq3_pred16x16[s->intra16x16_pred_mode](dest_y, linesize);
		svq3_luma_dc_dequant_idct_c(s->mb, s->mb_luma_dc[0], qscale);
	}
}

static void hl_decode_mb(SVQ3Context* s) {
	const int mb_x    = s->mb_x;
	const int mb_y    = s->mb_y;
	const int mb_xy   = s->mb_xy;
	const int mb_type = s->cur_pic->mb_type[mb_xy];
	uint8_t *dest_y, *dest_cb, *dest_cr;
	int linesize, uvlinesize;
	int i, j;
	const int* block_offset = &s->block_offset[0];
	const int block_h = 16 >> 1;

	linesize   = s->cur_pic->linesize[0];
	uvlinesize = s->cur_pic->linesize[1];

	dest_y  = s->cur_pic->data[0] + (mb_x     + mb_y * linesize)  * 16;
	dest_cb = s->cur_pic->data[1] +  mb_x * 8 + mb_y * uvlinesize * block_h;
	dest_cr = s->cur_pic->data[2] +  mb_x * 8 + mb_y * uvlinesize * block_h;

	if (IS_INTRA(mb_type)) {
		svq3_pred8x8[s->chroma_pred_mode](dest_cb, uvlinesize);
		svq3_pred8x8[s->chroma_pred_mode](dest_cr, uvlinesize);
		hl_decode_mb_predict_luma(s, mb_type, block_offset, linesize, dest_y);
	}

	hl_decode_mb_idct_luma(s, mb_type, block_offset, linesize, dest_y);

	if (s->cbp & 0x30) {
		uint8_t* dest[2] = { dest_cb, dest_cr };
		svq3_chroma_dc_dequant_idct(s->mb + 16 * 16 * 1, s->dequant4_coeff[4][0]);
		svq3_chroma_dc_dequant_idct(s->mb + 16 * 16 * 2, s->dequant4_coeff[4][0]);
		for (j = 1; j < 3; j++) {
			for (i = j * 16; i < j * 16 + 4; i++)
				if (s->non_zero_count_cache[scan8[i]] || s->mb[i * 16]) {
					uint8_t* const ptr = dest[j - 1] + block_offset[i];
					svq3_add_idct_c(ptr, s->mb + i * 16, uvlinesize,
					                ff_h264_chroma_qp_row0[s->qscale + 12] - 12, 2);
				}
		}
	}
}

// ─── macroblock decode (svq3.c) — de-frameworked ───────────────────────────────
static int svq3_decode_mb(SVQ3Context* s, unsigned int mb_type) {
	int i, j, k, m, dir, mode;
	int cbp = 0;
	uint32_t vlc;
	int8_t *top, *left;
	const int mb_xy = s->mb_xy;
	const int b_xy  = 4 * s->mb_x + 4 * s->mb_y * s->b_stride;

	s->top_samples_available  = (s->mb_y == 0) ? 0x33FF : 0xFFFF;
	s->left_samples_available = (s->mb_x == 0) ? 0x5F5F : 0xFFFF;

	if (mb_type == 0) {                             /* SKIP */
		if (s->pict_type == PICT_P || s->next_pic->mb_type[mb_xy] == (uint32_t)-1) {
			svq3_mc_dir_part(s, 16 * s->mb_x, 16 * s->mb_y, 16, 16, 0, 0, 0, 0, 0, 0);
			if (s->pict_type == PICT_B)
				svq3_mc_dir_part(s, 16 * s->mb_x, 16 * s->mb_y, 16, 16, 0, 0, 0, 0, 1, 1);
			mb_type = MB_TYPE_SKIP;
		} else {
			mb_type = FFMIN(s->next_pic->mb_type[mb_xy], 6);
			if (svq3_mc_dir(s, mb_type, PREDICT_MODE, 0, 0) < 0) return -1;
			if (svq3_mc_dir(s, mb_type, PREDICT_MODE, 1, 1) < 0) return -1;
			mb_type = MB_TYPE_16x16;
		}
	} else if (mb_type < 8) {                       /* INTER */
		if (s->thirdpel_flag && s->halfpel_flag == !get_bits1(&s->gb_slice))
			mode = THIRDPEL_MODE;
		else if (s->halfpel_flag && s->thirdpel_flag == !get_bits1(&s->gb_slice))
			mode = HALFPEL_MODE;
		else
			mode = FULLPEL_MODE;

		for (m = 0; m < 2; m++) {
			if (s->mb_x > 0 && s->intra4x4_pred_mode[s->mb2br_xy[mb_xy - 1] + 6] != -1) {
				for (i = 0; i < 4; i++)
					AV_COPY32(s->mv_cache[m][scan8[0] - 1 + i * 8],
					          s->cur_pic->motion_val[m][b_xy - 1 + i * s->b_stride]);
			} else {
				for (i = 0; i < 4; i++)
					AV_ZERO32(s->mv_cache[m][scan8[0] - 1 + i * 8]);
			}
			if (s->mb_y > 0) {
				memcpy(s->mv_cache[m][scan8[0] - 1 * 8],
				       s->cur_pic->motion_val[m][b_xy - s->b_stride], 4 * 2 * sizeof(int16_t));
				memset(&s->ref_cache[m][scan8[0] - 1 * 8],
				       (s->intra4x4_pred_mode[s->mb2br_xy[mb_xy - s->mb_stride]] == -1) ? PART_NOT_AVAILABLE : 1, 4);

				if (s->mb_x < s->mb_width - 1) {
					AV_COPY32(s->mv_cache[m][scan8[0] + 4 - 1 * 8],
					          s->cur_pic->motion_val[m][b_xy - s->b_stride + 4]);
					s->ref_cache[m][scan8[0] + 4 - 1 * 8] =
						(s->intra4x4_pred_mode[s->mb2br_xy[mb_xy - s->mb_stride + 1] + 6] == -1 ||
						 s->intra4x4_pred_mode[s->mb2br_xy[mb_xy - s->mb_stride]] == -1) ? PART_NOT_AVAILABLE : 1;
				} else
					s->ref_cache[m][scan8[0] + 4 - 1 * 8] = PART_NOT_AVAILABLE;
				if (s->mb_x > 0) {
					AV_COPY32(s->mv_cache[m][scan8[0] - 1 - 1 * 8],
					          s->cur_pic->motion_val[m][b_xy - s->b_stride - 1]);
					s->ref_cache[m][scan8[0] - 1 - 1 * 8] =
						(s->intra4x4_pred_mode[s->mb2br_xy[mb_xy - s->mb_stride - 1] + 3] == -1) ? PART_NOT_AVAILABLE : 1;
				} else
					s->ref_cache[m][scan8[0] - 1 - 1 * 8] = PART_NOT_AVAILABLE;
			} else
				memset(&s->ref_cache[m][scan8[0] - 1 * 8 - 1], PART_NOT_AVAILABLE, 8);

			if (s->pict_type != PICT_B)
				break;
		}

		if (s->pict_type == PICT_P) {
			if (svq3_mc_dir(s, mb_type - 1, mode, 0, 0) < 0) return -1;
		} else {                                    /* PICT_B */
			if (mb_type != 2) {
				if (svq3_mc_dir(s, 0, mode, 0, 0) < 0) return -1;
			} else {
				for (i = 0; i < 4; i++)
					memset(s->cur_pic->motion_val[0][b_xy + i * s->b_stride], 0, 4 * 2 * sizeof(int16_t));
			}
			if (mb_type != 1) {
				if (svq3_mc_dir(s, 0, mode, 1, mb_type == 3) < 0) return -1;
			} else {
				for (i = 0; i < 4; i++)
					memset(s->cur_pic->motion_val[1][b_xy + i * s->b_stride], 0, 4 * 2 * sizeof(int16_t));
			}
		}

		mb_type = MB_TYPE_16x16;
	} else if (mb_type == 8 || mb_type == 33) {     /* INTRA4x4 */
		int8_t* i4x4       = s->intra4x4_pred_mode + s->mb2br_xy[s->mb_xy];
		int8_t* i4x4_cache = s->intra4x4_pred_mode_cache;

		memset(s->intra4x4_pred_mode_cache, -1, 8 * 5 * sizeof(int8_t));

		if (mb_type == 8) {
			if (s->mb_x > 0) {
				for (i = 0; i < 4; i++)
					s->intra4x4_pred_mode_cache[scan8[0] - 1 + i * 8] = s->intra4x4_pred_mode[s->mb2br_xy[mb_xy - 1] + 6 - i];
				if (s->intra4x4_pred_mode_cache[scan8[0] - 1] == -1)
					s->left_samples_available = 0x5F5F;
			}
			if (s->mb_y > 0) {
				s->intra4x4_pred_mode_cache[4 + 8 * 0] = s->intra4x4_pred_mode[s->mb2br_xy[mb_xy - s->mb_stride] + 0];
				s->intra4x4_pred_mode_cache[5 + 8 * 0] = s->intra4x4_pred_mode[s->mb2br_xy[mb_xy - s->mb_stride] + 1];
				s->intra4x4_pred_mode_cache[6 + 8 * 0] = s->intra4x4_pred_mode[s->mb2br_xy[mb_xy - s->mb_stride] + 2];
				s->intra4x4_pred_mode_cache[7 + 8 * 0] = s->intra4x4_pred_mode[s->mb2br_xy[mb_xy - s->mb_stride] + 3];
				if (s->intra4x4_pred_mode_cache[4 + 8 * 0] == -1)
					s->top_samples_available = 0x33FF;
			}

			for (i = 0; i < 16; i += 2) {
				vlc = get_interleaved_ue_golomb(&s->gb_slice);
				if (vlc >= 25U)
					return -1;                      // luma prediction OOR

				left = &s->intra4x4_pred_mode_cache[scan8[i] - 1];
				top  = &s->intra4x4_pred_mode_cache[scan8[i] - 8];

				left[1] = svq3_pred_1[top[0] + 1][left[0] + 1][svq3_pred_0[vlc][0]];
				left[2] = svq3_pred_1[top[1] + 1][left[1] + 1][svq3_pred_0[vlc][1]];

				if (left[1] == -1 || left[2] == -1)
					return -1;                      // weird prediction
			}
		} else {                                    /* mb_type == 33, DC_128_PRED block type */
			for (i = 0; i < 4; i++)
				memset(&s->intra4x4_pred_mode_cache[scan8[0] + 8 * i], DC_PRED, 4);
		}

		AV_COPY32(i4x4, i4x4_cache + 4 + 8 * 4);
		i4x4[4] = i4x4_cache[7 + 8 * 3];
		i4x4[5] = i4x4_cache[7 + 8 * 2];
		i4x4[6] = i4x4_cache[7 + 8 * 1];

		if (mb_type == 8) {
			ff_h264_check_intra4x4_pred_mode(s->intra4x4_pred_mode_cache,
			                                 s->top_samples_available, s->left_samples_available);
			s->top_samples_available  = (s->mb_y == 0) ? 0x33FF : 0xFFFF;
			s->left_samples_available = (s->mb_x == 0) ? 0x5F5F : 0xFFFF;
		} else {
			for (i = 0; i < 4; i++)
				memset(&s->intra4x4_pred_mode_cache[scan8[0] + 8 * i], DC_128_PRED, 4);
			s->top_samples_available  = 0x33FF;
			s->left_samples_available = 0x5F5F;
		}

		mb_type = MB_TYPE_INTRA4x4;
	} else {                                        /* INTRA16x16 */
		dir = ff_h264_i_mb_type_info[mb_type - 8].pred_mode;
		dir = (dir >> 1) ^ 3 * (dir & 1) ^ 1;

		if ((s->intra16x16_pred_mode =
		         ff_h264_check_intra_pred_mode(s->top_samples_available,
		                                       s->left_samples_available, dir, 0)) < 0)
			return s->intra16x16_pred_mode;

		cbp     = ff_h264_i_mb_type_info[mb_type - 8].cbp;
		mb_type = MB_TYPE_INTRA16x16;
	}

	if (!IS_INTER(mb_type) && s->pict_type != PICT_I) {
		for (i = 0; i < 4; i++)
			memset(s->cur_pic->motion_val[0][b_xy + i * s->b_stride], 0, 4 * 2 * sizeof(int16_t));
		if (s->pict_type == PICT_B) {
			for (i = 0; i < 4; i++)
				memset(s->cur_pic->motion_val[1][b_xy + i * s->b_stride], 0, 4 * 2 * sizeof(int16_t));
		}
	}
	if (!IS_INTRA4x4(mb_type))
		memset(s->intra4x4_pred_mode + s->mb2br_xy[mb_xy], DC_PRED, 8);
	if (!IS_SKIP(mb_type) || s->pict_type == PICT_B)
		memset(s->non_zero_count_cache + 8, 0, 14 * 8 * sizeof(uint8_t));

	if (!IS_INTRA16x16(mb_type) && (!IS_SKIP(mb_type) || s->pict_type == PICT_B)) {
		if ((vlc = get_interleaved_ue_golomb(&s->gb_slice)) >= 48U)
			return -1;                              // cbp_vlc OOR
		cbp = IS_INTRA(mb_type) ? ff_h264_golomb_to_intra4x4_cbp[vlc]
		                        : ff_h264_golomb_to_inter_cbp[vlc];
	}
	if (IS_INTRA16x16(mb_type) ||
	    (s->pict_type != PICT_I && s->adaptive_quant && cbp)) {
		s->qscale += get_interleaved_se_golomb(&s->gb_slice);
		if (s->qscale > 31u)
			return -1;
	}
	if (IS_INTRA16x16(mb_type)) {
		AV_ZERO128(s->mb_luma_dc[0] + 0);
		AV_ZERO128(s->mb_luma_dc[0] + 8);
		if (svq3_decode_block(&s->gb_slice, s->mb_luma_dc[0], 0, 1))
			return -1;                              // intra luma dc
	}

	if (cbp) {
		const int index = IS_INTRA16x16(mb_type) ? 1 : 0;
		const int type  = ((s->qscale < 24 && IS_INTRA4x4(mb_type)) ? 2 : 1);

		for (i = 0; i < 4; i++)
			if ((cbp & (1 << i))) {
				for (j = 0; j < 4; j++) {
					k = index ? (1 * (j & 1) + 2 * (i & 1) + 2 * (j & 2) + 4 * (i & 2))
					          : (4 * i + j);
					s->non_zero_count_cache[scan8[k]] = 1;
					if (svq3_decode_block(&s->gb_slice, &s->mb[16 * k], index, type))
						return -1;                  // block
				}
			}

		if ((cbp & 0x30)) {
			for (i = 1; i < 3; ++i)
				if (svq3_decode_block(&s->gb_slice, &s->mb[16 * 16 * i], 0, 3))
					return -1;                      // chroma dc

			if ((cbp & 0x20)) {
				for (i = 1; i < 3; i++) {
					for (j = 0; j < 4; j++) {
						k = 16 * i + j;
						s->non_zero_count_cache[scan8[k]] = 1;
						if (svq3_decode_block(&s->gb_slice, &s->mb[16 * k], 1, 1))
							return -1;              // chroma ac
					}
				}
			}
		}
	}

	s->cbp                     = cbp;
	s->cur_pic->mb_type[mb_xy] = mb_type;

	if (IS_INTRA(mb_type)) {
		int cpm = ff_h264_check_intra_pred_mode(s->top_samples_available,
		                                        s->left_samples_available, DC_PRED8x8, 1);
		s->chroma_pred_mode = (cpm < 0) ? 0 : (cpm > 6 ? 6 : cpm);   // clamp: svq3_pred8x8 has 7 entries
	}

	return 0;
}

// ─── slice header (svq3.c) — de-frameworked ────────────────────────────────────
static int svq3_decode_slice_header(SVQ3Context* s) {
	const int mb_xy = s->mb_xy;
	int i, header;
	unsigned slice_id;

	header = get_bits(&s->gb, 8);

	if (((header & 0x9F) != 1 && (header & 0x9F) != 2) || (header & 0x60) == 0) {
		return -1;                                  // unsupported slice header
	} else {
		int slice_bits, slice_bytes, slice_length;
		int length = header >> 5 & 3;

		slice_length = show_bits(&s->gb, 8 * length);
		slice_bits   = slice_length * 8;
		slice_bytes  = slice_length + length - 1;

		skip_bits(&s->gb, 8);

		ensure_slice_buf(s, slice_bytes);
		if (!s->slice_buf) return -1;

		if (slice_bytes * 8LL > get_bits_left(&s->gb))
			return -1;                              // slice after bitstream end
		memcpy(s->slice_buf, s->gb.buffer + s->gb.index / 8, slice_bytes);

		if (length > 0)
			memmove(s->slice_buf, &s->slice_buf[slice_length], length - 1);

		if (s->watermark_key) {
			uint32_t hdr = AV_RL32(&s->slice_buf[1]);
			AV_WL32(&s->slice_buf[1], hdr ^ s->watermark_key);
		}
		init_get_bits(&s->gb_slice, s->slice_buf, slice_bits);

		skip_bits_long(&s->gb, slice_bytes * 8);
	}

	if ((slice_id = get_interleaved_ue_golomb(&s->gb_slice)) >= 3)
		return -1;                                  // illegal slice type

	s->slice_type = ff_h264_golomb_to_pict_type[slice_id];

	if ((header & 0x9F) == 2) {
		i = (s->mb_num < 64) ? 6 : (1 + av_log2(s->mb_num - 1));
		get_bits(&s->gb_slice, i);
	} else if (get_bits1(&s->gb_slice)) {
		return -1;                                  // media key encryption (unsupported)
	}

	s->slice_num      = get_bits(&s->gb_slice, 8);
	s->qscale         = get_bits(&s->gb_slice, 5);
	s->adaptive_quant = get_bits1(&s->gb_slice);

	skip_bits1(&s->gb_slice);
	if (s->has_watermark)
		skip_bits1(&s->gb_slice);
	skip_bits1(&s->gb_slice);
	skip_bits(&s->gb_slice, 2);

	if (skip_1stop_8data_bits(&s->gb_slice) < 0)
		return -1;

	/* reset intra predictors and invalidate motion vector references */
	if (s->mb_x > 0) {
		memset(s->intra4x4_pred_mode + s->mb2br_xy[mb_xy - 1] + 3, -1, 4 * sizeof(int8_t));
		memset(s->intra4x4_pred_mode + s->mb2br_xy[mb_xy - s->mb_x], -1, 8 * sizeof(int8_t) * s->mb_x);
	}
	if (s->mb_y > 0) {
		memset(s->intra4x4_pred_mode + s->mb2br_xy[mb_xy - s->mb_stride],
		       -1, 8 * sizeof(int8_t) * (s->mb_width - s->mb_x));
		if (s->mb_x > 0)
			s->intra4x4_pred_mode[s->mb2br_xy[mb_xy - s->mb_stride - 1] + 3] = -1;
	}

	return 0;
}

static void init_dequant4_coeff_table(SVQ3Context* s) {
	int q, x;
	const int max_qp = 51;
	for (q = 0; q < max_qp + 1; q++) {
		int shift = ff_h264_quant_div6[q] + 2;
		int idx   = ff_h264_quant_rem6[q];
		for (x = 0; x < 16; x++)
			s->dequant4_coeff[q][(x >> 2) | ((x << 2) & 0xF)] =
				((uint32_t)ff_h264_dequant4_coeff_init[idx][(x & 1) + ((x >> 2) & 1)] * 16) << shift;
	}
}

// ─── SEQH extradata parse (svq3.c) — de-frameworked; watermark hard-fails ──────
static int svq3_decode_seqh(SVQ3Context* s, const uint8_t* extradata, int extradata_size, int seqh_offset) {
	const uint8_t* p = extradata + seqh_offset;
	unsigned int size = AV_RB32(p + 4);
	GetBitContext gb;

	if (size > (unsigned)(extradata_size - seqh_offset - 8))
		return -1;
	p += 8;
	init_get_bits(&gb, p, size * 8);

	int frame_size_code = get_bits(&gb, 3);
	int w = 0, h = 0;
	switch (frame_size_code) {
	case 0: w = 160; h = 120; break;
	case 1: w = 128; h =  96; break;
	case 2: w = 176; h = 144; break;
	case 3: w = 352; h = 288; break;
	case 4: w = 704; h = 576; break;
	case 5: w = 240; h = 180; break;
	case 6: w = 320; h = 240; break;
	case 7:
		w = get_bits(&gb, 12);
		h = get_bits(&gb, 12);
		break;
	}
	if (w <= 0 || h <= 0 || w > 4096 || h > 4096)   // clamp case-7 12-bit dims (DoS guard)
		return -1;
	s->width  = w;
	s->height = h;

	s->halfpel_flag  = get_bits1(&gb);
	s->thirdpel_flag = get_bits1(&gb);

	get_bits1(&gb); get_bits1(&gb); get_bits1(&gb); get_bits1(&gb);   // unknown fields

	s->low_delay = get_bits1(&gb);

	get_bits1(&gb);                                 // unknown field

	if (skip_1stop_8data_bits(&gb) < 0)
		return -1;

	s->has_watermark = get_bits1(&gb);

	if (!s->has_watermark)
		return 0;

	return -1;                                       // watermarked SVQ3 not supported (see plan §4)
}

// ─── frame pool + lifecycle (de-frameworked from svq3_decode_{init,frame,end}) ─
static int svq3_alloc_buffers(SVQ3Context* s) {
	int x, y;
	s->mb_width   = (s->width + 15) / 16;
	s->mb_height  = (s->height + 15) / 16;
	s->mb_stride  = s->mb_width + 1;
	s->mb_num     = s->mb_width * s->mb_height;
	s->b_stride   = 4 * s->mb_width;
	s->h_edge_pos = s->mb_width * 16;
	s->v_edge_pos = s->mb_height * 16;

	const unsigned big_mb_num = s->mb_stride * (s->mb_height + 2) + 1;
	s->mb_type_buf = (uint32_t*)calloc(big_mb_num * NUM_PICS, sizeof(*s->mb_type_buf));
	if (!s->mb_type_buf) return -1;
	uint32_t* mb_type_buf = s->mb_type_buf + 2 * s->mb_stride + 1;

	const unsigned b4_stride = s->mb_width * 4 + 1;
	const unsigned b4_array_size = b4_stride * s->mb_height * 4;
	const unsigned motion_val_buf_size = b4_array_size + 4;
	s->motion_val_buf = (int16_t(*)[2])calloc(motion_val_buf_size * NUM_PICS * 2, sizeof(*s->motion_val_buf));
	if (!s->motion_val_buf) return -1;
	int16_t (*motion_val_buf)[2] = s->motion_val_buf + 4;

	const int lstride = s->mb_width * 16;
	const int cstride = s->mb_width * 8;
	const int lsize   = lstride * (s->mb_height * 16);
	const int csize   = cstride * (s->mb_height * 8);

	for (int i = 0; i < NUM_PICS; ++i) {
		SVQ3Frame* const pic = &s->frames[i];
		pic->mb_type = mb_type_buf;
		mb_type_buf += big_mb_num;
		for (int jj = 0; jj < 2; ++jj) {
			pic->motion_val[jj] = motion_val_buf;
			motion_val_buf     += motion_val_buf_size;
		}
		pic->data[0] = (uint8_t*)malloc(lsize);
		pic->data[1] = (uint8_t*)malloc(csize);
		pic->data[2] = (uint8_t*)malloc(csize);
		if (!pic->data[0] || !pic->data[1] || !pic->data[2]) return -1;
		pic->linesize[0] = lstride;
		pic->linesize[1] = cstride;
		pic->linesize[2] = cstride;
		pic->has_data = 0;
	}

	s->intra4x4_pred_mode = (int8_t*)calloc(s->mb_stride * 2 * 8, 1);
	if (!s->intra4x4_pred_mode) return -1;

	s->mb2br_xy = (uint32_t*)calloc(s->mb_stride * (s->mb_height + 1), sizeof(*s->mb2br_xy));
	if (!s->mb2br_xy) return -1;
	for (y = 0; y < s->mb_height; y++)
		for (x = 0; x < s->mb_width; x++) {
			const int mb_xy = x + y * s->mb_stride;
			s->mb2br_xy[mb_xy] = 8 * (mb_xy % (2 * s->mb_stride));
		}

	s->edge_emu_buffer = (uint8_t*)calloc(lstride, 17);
	if (!s->edge_emu_buffer) return -1;

	init_dequant4_coeff_table(s);
	s->buffers_ready = 1;
	return 0;
}

static void svq3_unref(SVQ3Frame* pic) { pic->has_data = 0; }

static void svq3_get_buffer(SVQ3Frame* pic) { pic->has_data = 1; }

static void svq3_alloc_dummy(SVQ3Context* s, SVQ3Frame* pic) {
	svq3_unref(pic);
	svq3_get_buffer(pic);
	memset(pic->data[0], 0,    s->mb_height * 16 * pic->linesize[0]);
	memset(pic->data[1], 0x80, s->mb_height * 8  * pic->linesize[1]);
	memset(pic->data[2], 0x80, s->mb_height * 8  * pic->linesize[2]);
}

// One de-framed packet -> one decode step (mirrors svq3_decode_frame).
static int svq3_decode_one(SVQ3Context* s, const uint8_t* buf, int buf_size, int* got_frame) {
	int ret, m, i;
	*got_frame = 0;
	s->output_pic = NULL;

	/* special case for last picture (drain) */
	if (buf_size == 0) {
		if (s->next_pic->has_data && !s->low_delay) {
			s->output_pic = s->next_pic;
			*got_frame = 1;
		}
		return 0;
	}

	s->mb_x = s->mb_y = s->mb_xy = 0;

	if (init_get_bits8(&s->gb, buf, buf_size) < 0) return -1;
	if (svq3_decode_slice_header(s) < 0) return -1;
	if (buf_size < s->mb_width * s->mb_height / 8) return -1;

	s->pict_type = s->slice_type;

	if (s->pict_type != PICT_B)
		FFSWAP(SVQ3Frame*, s->next_pic, s->last_pic);

	svq3_unref(s->cur_pic);
	svq3_get_buffer(s->cur_pic);

	for (i = 0; i < 16; i++) {
		s->block_offset[i]      = (4 * ((scan8[i] - scan8[0]) & 7)) + 4 * s->cur_pic->linesize[0] * ((scan8[i] - scan8[0]) >> 3);
		s->block_offset[48 + i] = (4 * ((scan8[i] - scan8[0]) & 7)) + 8 * s->cur_pic->linesize[0] * ((scan8[i] - scan8[0]) >> 3);
	}
	for (i = 0; i < 16; i++) {
		s->block_offset[16 + i] =
		s->block_offset[32 + i] = (4 * ((scan8[i] - scan8[0]) & 7)) + 4 * s->cur_pic->linesize[1] * ((scan8[i] - scan8[0]) >> 3);
		s->block_offset[48 + 16 + i] =
		s->block_offset[48 + 32 + i] = (4 * ((scan8[i] - scan8[0]) & 7)) + 8 * s->cur_pic->linesize[1] * ((scan8[i] - scan8[0]) >> 3);
	}

	if (s->pict_type != PICT_I) {
		if (!s->last_pic->has_data)
			svq3_alloc_dummy(s, s->last_pic);
		if (s->pict_type == PICT_B && !s->next_pic->has_data)
			svq3_alloc_dummy(s, s->next_pic);
	}

	if (s->pict_type == PICT_B) {
		s->frame_num_offset = s->slice_num - s->prev_frame_num;
		if (s->frame_num_offset < 0) s->frame_num_offset += 256;
		if (s->frame_num_offset == 0 || s->frame_num_offset >= s->prev_frame_num_offset)
			return -1;                              // B-frame picture id
	} else {
		s->prev_frame_num        = s->frame_num;
		s->frame_num             = s->slice_num;
		s->prev_frame_num_offset = s->frame_num - s->prev_frame_num;
		if (s->prev_frame_num_offset < 0) s->prev_frame_num_offset += 256;
	}

	for (m = 0; m < 2; m++) {
		int ii;
		for (ii = 0; ii < 4; ii++) {
			int jj;
			for (jj = -1; jj < 4; jj++)
				s->ref_cache[m][scan8[0] + 8 * ii + jj] = 1;
			if (ii < 3)
				s->ref_cache[m][scan8[0] + 8 * ii + jj] = PART_NOT_AVAILABLE;
		}
	}

	for (s->mb_y = 0; s->mb_y < s->mb_height; s->mb_y++) {
		for (s->mb_x = 0; s->mb_x < s->mb_width; s->mb_x++) {
			unsigned mb_type;
			s->mb_xy = s->mb_x + s->mb_y * s->mb_stride;

			if ((get_bits_left(&s->gb_slice)) <= 7) {
				if (((get_bits_count(&s->gb_slice) & 7) == 0 ||
				     show_bits(&s->gb_slice, get_bits_left(&s->gb_slice) & 7) == 0)) {
					if (svq3_decode_slice_header(s) < 0)
						return -1;
				}
				/* TODO: support non-constant slice type / mb_skip_run */
			}

			mb_type = get_interleaved_ue_golomb(&s->gb_slice);

			if (s->pict_type == PICT_I)
				mb_type += 8;
			else if (s->pict_type == PICT_B && mb_type >= 4)
				mb_type += 4;
			if (mb_type > 33 || svq3_decode_mb(s, mb_type))
				return -1;

			if (mb_type != 0 || s->cbp)
				hl_decode_mb(s);

			if (s->pict_type != PICT_B && !s->low_delay)
				s->cur_pic->mb_type[s->mb_x + s->mb_y * s->mb_stride] =
					(s->pict_type == PICT_P && mb_type < 8) ? (mb_type - 1) : (uint32_t)-1;
		}
	}

	int left = buf_size * 8 - get_bits_count(&s->gb_slice);
	if (left < 0)
		return -1;

	if (s->pict_type == PICT_B || s->low_delay)
		s->output_pic = s->cur_pic;
	else if (s->last_pic->has_data)
		s->output_pic = s->last_pic;

	if (s->last_pic->has_data || s->low_delay)
		*got_frame = 1;

	if (s->pict_type != PICT_B)
		FFSWAP(SVQ3Frame*, s->cur_pic, s->next_pic);
	else
		svq3_unref(s->cur_pic);

	return buf_size;
}

// ─── extern "C" boundary (see svq3_decode_core.js) ─────────────────────────────
extern "C" {

SVQ3Context* svq3_open(void) {
	SVQ3Context* s = (SVQ3Context*)calloc(1, sizeof(SVQ3Context));
	if (!s) return NULL;
	s->cur_pic  = &s->frames[0];
	s->last_pic = &s->frames[1];
	s->next_pic = &s->frames[2];
	s->halfpel_flag  = 1;
	s->thirdpel_flag = 1;
	s->has_watermark = 0;
	svq3_pred_init();
	svq3_hpel_init();
	svq3_tpel_init();
	return s;
}

// extra = the SVQ3 ImageDescription payload (from the QuickTime sample description);
// the decoder prowls it for the "SEQH" marker exactly like ffmpeg. Sets dims + flags
// and allocates the frame pool. Returns 0 on success, <0 on error.
int svq3_parse_extradata(SVQ3Context* s, const uint8_t* extra, int len) {
	if (extra && len > 8) {
		const uint8_t* p = extra;
		for (int m = 0; m + 8 < len; m++) {
			if (!memcmp(p, "SEQH", 4)) {
				if (svq3_decode_seqh(s, extra, len, m) < 0)
					return -1;
				break;
			}
			p++;
		}
	}
	if (s->width <= 0 || s->height <= 0)
		return -1;                                  // no usable dimensions (Phase 1: SEQH required)
	return svq3_alloc_buffers(s);
}

int svq3_low_delay(SVQ3Context* s) { return s->low_delay; }

void svq3_reset(SVQ3Context* s) {
	for (int i = 0; i < NUM_PICS; i++) s->frames[i].has_data = 0;
	s->frame_num = 0;
	s->frame_num_offset = 0;
	s->prev_frame_num = 0;
	s->prev_frame_num_offset = 0;
	s->output_pic = NULL;
}

int svq3_decode_frame(SVQ3Context* s, const uint8_t* data, int off, int len, int32_t* info) {
	g_err = 0;
	int got_frame = 0;
	int ret = s->buffers_ready ? svq3_decode_one(s, data + off, len, &got_frame) : -1;
	const int ok = (ret >= 0 && !g_err);
	info[0] = s->mb_width * 16;                      // codedWidth (= luma stride)
	info[1] = s->mb_height * 16;                     // codedHeight
	info[2] = (s->pict_type == PICT_I);              // isKeyframe
	info[3] = ok;
	info[4] = (ok && got_frame) ? 1 : 0;             // gotFrame
	return ok;
}

uint8_t* svq3_y_ptr(SVQ3Context* s) { return s->output_pic ? s->output_pic->data[0] : NULL; }
uint8_t* svq3_u_ptr(SVQ3Context* s) { return s->output_pic ? s->output_pic->data[1] : NULL; }
uint8_t* svq3_v_ptr(SVQ3Context* s) { return s->output_pic ? s->output_pic->data[2] : NULL; }

// Phase-3 SIMD A/B hooks. svq3_set_simd toggles scalar/SIMD kernels; svq3_test_idct
// runs add_idct (per the current mode) on a copy of `in` into `out` (which holds the
// initial dst pixels), for the byte-identical scalar-vs-SIMD lock in Svq3Wasm.test.js.
void svq3_set_simd(int on) { g_simd = on ? 1 : 0; }
int svq3_test_idct(const int16_t* in, uint8_t* out, int stride, int qp, int dc) {
	int16_t blk[16];
	memcpy(blk, in, 16 * sizeof(int16_t));
	svq3_add_idct_c(out, blk, stride, qp, dc);
	return 0;
}

void svq3_close(SVQ3Context* s) {
	if (!s) return;
	for (int i = 0; i < NUM_PICS; i++) {
		free(s->frames[i].data[0]);
		free(s->frames[i].data[1]);
		free(s->frames[i].data[2]);
	}
	free(s->mb_type_buf);
	free(s->motion_val_buf);
	free(s->slice_buf);
	free(s->intra4x4_pred_mode);
	free(s->edge_emu_buffer);
	free(s->mb2br_xy);
	free(s);
}

} // extern "C"
