/*
	On2 VP6 video decoder (VP6F / VP6A, the Flash/QuickTime variants) — WASM C++.

	This is a direct port of FFmpeg's VP6 decoder (libavcodec, release/8.1, LGPL):
	  vp56.c / vp6.c / vpx_rac.{c,h} / vp3dsp.c / vp6dsp.c / h264chroma_template.c /
	  videodsp_template.c / huffman.c + vp6data.h / vp56data.{c,h}
	The algorithm (range coder, header/coeff/MV decode, the sub-pel filters, the VP3
	IDCT, the loop filter, the Huffman tree build) is lifted essentially verbatim
	from those files; the surrounding libavcodec framework (AVFrame / AVCodecContext /
	the DSP function-pointer dispatch / threading / AVERROR / VLC infrastructure) has
	been replaced by a self-contained handle (VP6Ctx) that owns its own Y/U/V plane
	pool + prev/golden/current references and a flat Huffman table, mirroring the way
	the project's existing JS port (swf/vdec_VP6.js) de-frameworked the same code.

	Differences from FFmpeg, all behaviour-preserving (and proven bit-exact by the JS
	port the conformance test cross-checks against):
	  - We always decode top-down with a positive stride (FFmpeg's flip / frbi / srbi
	    are dropped); the bottom-up QuickTime "vp6" variant is mirrored at OUTPUT by
	    the JS shim, not here.
	  - VP6-alpha is decoded as an INDEPENDENT second VP6 stream (its luma plane is the
	    alpha channel), via a nested VP6Ctx — simpler than, and bit-exact to, FFmpeg's
	    shared-frame plane-3 scheme.
	  - Huffman VLCs use a flat peek-table built from the (length, symbol) pairs that
	    ff_huff_build_tree/get_tree_codes emit, instead of FFmpeg's multi-level VLC.
	  - Interlaced VP6 is not supported (errors out, as the JS port does).

	Output (matching the project's planar-YCbCr GPU path): separate Y/U/V planes are
	exposed via pointers; the JS shim crops/mirrors and interleaves Cb,Cr. Plane
	layout: Y stride = codedWidth, U/V stride = codedWidth/2.

	Boundary (extern "C", prefixed _ on the JS side; see vp6_decode_core.js):
	  VP6Ctx* vp6_open(int has_alpha)
	  void    vp6_close(VP6Ctx*)
	  void    vp6_reset(VP6Ctx*)                      drop prev/golden refs (seek)
	  int     vp6_decode_frame(VP6Ctx*, data, pOff, pLen, aOff, aLen, int info[4])
	            decode ONE de-framed frame (primary [pOff,pOff+pLen), alpha
	            [aOff,aOff+aLen) when has_alpha); info = {codedWidth, codedHeight,
	            useHuffman, ok}; returns 1/0
	  uint8_t* vp6_y_ptr/u_ptr/v_ptr(VP6Ctx*)         primary planes of the last frame
	  uint8_t* vp6_alpha_y_ptr(VP6Ctx*)               alpha plane (or NULL)
	  int      vp6_test_idct(...)                     SIMD-verification hook
*/

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#ifdef __wasm_simd128__
#include <wasm_simd128.h>
#endif

// Decode-abort flag (mirrors FFmpeg's AVERROR returns / the JS `throw` paths:
// invalid size, invalid sub-version, end-of-stream, bad VLC). Set on error; the
// macroblock loop bails and vp6_decode_frame returns 0. Reset each decode.
static int g_err = 0;

// ─── small helpers ──────────────────────────────────────────────────────────
static inline int av_clip_uint8(int a) { return a < 0 ? 0 : (a > 255 ? 255 : a); }
static inline int FFMIN(int a, int b) { return a < b ? a : b; }
static inline int FFABS(int a) { return a < 0 ? -a : a; }
static inline int AV_RB16(const uint8_t* p) { return (p[0] << 8) | p[1]; }

// libavutil RSHIFT: round to nearest, ties away from zero.
static inline int RSHIFT(int a, int b) {
	return a > 0 ? (a + (1 << (b - 1))) >> b : (a + (1 << (b - 1)) - 1) >> b;
}

// vp3dsp M(a,b) = ((int)((SUINT)a * b)) >> 16  (32-bit wrapping multiply, then
// arithmetic right shift). -fwrapv keeps the signed shift well-defined.
static inline int M(int a, int b) { return (int)((uint32_t)a * (uint32_t)b) >> 16; }

// ─── VP56 range coder (vpx_rac) ───────────────────────────────────────────────
static const uint8_t ff_vpx_norm_shift[256] = {
	8,7,6,6,5,5,5,5,4,4,4,4,4,4,4,4,
	3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,
	2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
	2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
	1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
	1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
	1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
	1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
	0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
	0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
	0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
	0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
};

// Out-of-range reads yield 0 (FFmpeg relies on input padding for the same effect).
struct RangeCoder {
	int high;
	int bits;
	const uint8_t* arr;
	int buffer;
	int end;
	int end_reached;
	uint32_t code_word;
};

static inline int rcb(const RangeCoder* c, int i) { return i < c->end ? c->arr[i] : 0; }

static void rangecoder_init(RangeCoder* c, const uint8_t* arr, int offset, int size) {
	c->arr = arr;
	c->high = 255;
	c->bits = -16;
	c->buffer = offset;
	c->end = offset + size;
	c->end_reached = 0;
	c->code_word = ((uint32_t)rcb(c, offset) << 16) | ((uint32_t)rcb(c, offset + 1) << 8) | (uint32_t)rcb(c, offset + 2);
	c->buffer = offset + 3;
}

static inline uint32_t rac_renorm(RangeCoder* c) {
	int shift = ff_vpx_norm_shift[c->high];
	int bits = c->bits;
	uint32_t code_word = c->code_word;

	c->high <<= shift;
	code_word <<= shift;
	bits += shift;
	if (bits >= 0 && c->buffer < c->end) {
		int be16 = ((int)rcb(c, c->buffer) << 8) | (int)rcb(c, c->buffer + 1);
		code_word |= (uint32_t)be16 << bits;
		c->buffer += 2;
		bits -= 16;
	}
	c->bits = bits;
	return code_word;
}

static inline int rac_get_prob(RangeCoder* c, uint8_t prob) {
	uint32_t code_word = rac_renorm(c);
	unsigned low = 1 + (((c->high - 1) * prob) >> 8);
	uint32_t low_shift = (uint32_t)low << 16;
	int bit = code_word >= low_shift;
	c->high = bit ? c->high - low : low;
	c->code_word = bit ? code_word - low_shift : code_word;
	return bit;
}
// branchy variant is functionally identical
static inline int rac_get_prob_branchy(RangeCoder* c, int prob) {
	uint32_t code_word = rac_renorm(c);
	unsigned low = 1 + (((c->high - 1) * prob) >> 8);
	uint32_t low_shift = (uint32_t)low << 16;
	if (code_word >= low_shift) {
		c->high -= low;
		c->code_word = code_word - low_shift;
		return 1;
	}
	c->high = low;
	c->code_word = code_word;
	return 0;
}

static inline int rac_get(RangeCoder* c) {
	uint32_t code_word = rac_renorm(c);
	int low = (c->high + 1) >> 1;
	uint32_t low_shift = (uint32_t)low << 16;
	int bit = code_word >= low_shift;
	if (bit) {
		c->high -= low;
		code_word -= low_shift;
	} else {
		c->high = low;
	}
	c->code_word = code_word;
	return bit;
}

static int rac_gets(RangeCoder* c, int bits) {
	int value = 0;
	while (bits--)
		value = (value << 1) | rac_get(c);
	return value;
}

static int rac_gets_nn(RangeCoder* c, int bits) {
	int v = rac_gets(c, 7) << 1;
	return v + !v;
}

static int rac_is_end(RangeCoder* c) {
	if (c->end <= c->buffer && c->bits >= 0)
		c->end_reached++;
	return c->end_reached > 10;
}

// VP56Tree: [val, prob_idx]; positive val jumps forward, a leaf's negated val is
// the decoded symbol.
struct VP56Tree { int8_t val; int8_t prob_idx; };

static int rac_get_tree(RangeCoder* c, const VP56Tree* tree, const uint8_t* probs) {
	while (tree->val > 0) {
		if (rac_get_prob_branchy(c, probs[tree->prob_idx]))
			tree += tree->val;
		else
			tree++;
	}
	return -tree->val;
}

// ─── MSB-first bit reader (get_bits) for the Huffman coeff path ───────────────
struct GetBits {
	const uint8_t* a;
	int byteStart;
	int nbits;
	int pos;
};

static void gb_init(GetBits* g, const uint8_t* a, int offset, int length) {
	g->a = a;
	g->byteStart = offset;
	g->nbits = length * 8;
	g->pos = 0;
}
static int gb_left(const GetBits* g) { return g->nbits - g->pos; }

static uint32_t gb_peek(const GetBits* g, int n) {
	uint32_t v = 0;
	for (int i = 0; i < n; i++) {
		int p = g->pos + i;
		int bit = (p < g->nbits) ? ((g->a[g->byteStart + (p >> 3)] >> (7 - (p & 7))) & 1) : 0;
		v = (v << 1) | (uint32_t)bit;
	}
	return v;
}
static uint32_t gb_get(GetBits* g, int n) {
	uint32_t v = gb_peek(g, n);
	g->pos += n;
	return v;
}
static int gb_get1(GetBits* g) {
	int bit;
	if (g->pos >= g->nbits) { g->pos++; return 0; }
	bit = (g->a[g->byteStart + (g->pos >> 3)] >> (7 - (g->pos & 7))) & 1;
	g->pos++;
	return bit;
}

// ─── tables (vp56data.c / vp6data.h, verbatim) ────────────────────────────────
static const uint8_t ff_zigzag_direct[64] = {
	 0,  1,  8, 16,  9,  2,  3, 10, 17, 24, 32, 25, 18, 11,  4,  5,
	12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13,  6,  7, 14, 21, 28,
	35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51,
	58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
};

static const uint8_t ff_vp56_b2p[]   = { 0, 0, 0, 0, 1, 2, 3, 3, 3, 3 };
static const uint8_t ff_vp56_b6to4[] = { 0, 0, 1, 1, 2, 3 };

static const uint8_t ff_vp56_coeff_parse_table[6][11] = {
	{ 159,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0 },
	{ 145, 165,   0,   0,   0,   0,   0,   0,   0,   0,   0 },
	{ 140, 148, 173,   0,   0,   0,   0,   0,   0,   0,   0 },
	{ 135, 140, 155, 176,   0,   0,   0,   0,   0,   0,   0 },
	{ 130, 134, 141, 157, 180,   0,   0,   0,   0,   0,   0 },
	{ 129, 130, 133, 140, 153, 177, 196, 230, 243, 254, 254 },
};
static const uint8_t ff_vp56_coeff_bias[] = { 0, 1, 2, 3, 4, 5, 7, 11, 19, 35, 67 };
static const uint8_t ff_vp56_coeff_bit_length[] = { 0, 1, 2, 3, 4, 10 };

static const VP56Tree ff_vp56_pc_tree[] = {
	{ 4, 6}, { 2, 7}, {-0}, {-1}, { 4, 8}, { 2, 9}, {-2}, {-3}, { 2,10}, {-4}, {-5},
};
static const VP56Tree ff_vp56_pva_tree[] = {
	{ 8, 0}, { 4, 1}, { 2, 2}, {-0}, {-1}, { 2, 3}, {-2}, {-3},
	{ 4, 4}, { 2, 5}, {-4}, {-5}, { 2, 6}, {-6}, {-7},
};
static const VP56Tree ff_vp56_pmbtm_tree[] = {
	{ 4, 0}, { 2, 1}, {-8}, {-4}, { 8, 2}, { 6, 3}, { 4, 4}, { 2, 5},
	{-24}, {-20}, {-16}, {-12}, {-0},
};
#define MB_NOVEC_PF 0
#define MB_INTRA    1
#define MB_DELTA_PF 2
#define MB_V1_PF    3
#define MB_V2_PF    4
#define MB_NOVEC_GF 5
#define MB_DELTA_GF 6
#define MB_4V       7
#define MB_V1_GF    8
#define MB_V2_GF    9
static const VP56Tree ff_vp56_pmbt_tree[] = {
	{ 8, 1}, { 4, 2},
	{ 2, 4}, {-MB_NOVEC_PF}, {-MB_DELTA_PF},
	{ 2, 5}, {-MB_V1_PF},    {-MB_V2_PF},
	{ 4, 3}, { 2, 6}, {-MB_INTRA}, {-MB_4V},
	{ 4, 7}, { 2, 8}, {-MB_NOVEC_GF}, {-MB_DELTA_GF},
	{ 2, 9}, {-MB_V1_GF}, {-MB_V2_GF},
};

// VP56Frame
#define FRAME_NONE     (-1)
#define FRAME_CURRENT  0
#define FRAME_PREVIOUS 1
#define FRAME_GOLDEN   2
static const int8_t ff_vp56_reference_frame[] = {
	FRAME_PREVIOUS, FRAME_CURRENT, FRAME_PREVIOUS, FRAME_PREVIOUS, FRAME_PREVIOUS,
	FRAME_GOLDEN,   FRAME_GOLDEN,  FRAME_PREVIOUS, FRAME_GOLDEN,   FRAME_GOLDEN,
};

static const uint8_t ff_vp56_ac_dequant[64] = {
	94, 92, 90, 88, 86, 82, 78, 74, 70, 66, 62, 58, 54, 53, 52, 51,
	50, 49, 48, 47, 46, 45, 44, 43, 42, 40, 39, 37, 36, 35, 34, 33,
	32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17,
	16, 15, 14, 13, 12, 11, 10,  9,  8,  7,  6,  5,  4,  3,  2,  1,
};
static const uint8_t ff_vp56_dc_dequant[64] = {
	47, 47, 47, 47, 45, 43, 43, 43, 43, 43, 42, 41, 41, 40, 40, 40,
	40, 35, 35, 35, 35, 33, 33, 33, 33, 32, 32, 32, 27, 27, 26, 26,
	25, 25, 24, 24, 23, 23, 19, 19, 19, 19, 18, 18, 17, 16, 16, 16,
	16, 16, 15, 11, 11, 11, 10, 10,  9,  8,  7,  5,  3,  3,  2,  2,
};
static const uint8_t ff_vp56_filter_threshold[] = {
	14, 14, 13, 13, 12, 12, 10, 10, 10, 10,  8,  8,  8,  8,  8,  8,
	 8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,
	 8,  8,  8,  8,  7,  7,  7,  7,  7,  7,  6,  6,  6,  6,  6,  6,
	 5,  5,  5,  5,  4,  4,  4,  4,  4,  4,  4,  3,  3,  3,  3,  2,
};
static const uint8_t ff_vp56_mb_type_model_model[] = { 171, 83, 199, 140, 125, 104 };
static const int8_t ff_vp56_candidate_predictor_pos[12][2] = {
	{ 0,-1},{-1, 0},{-1,-1},{ 1,-1},{ 0,-2},{-2, 0},
	{-2,-1},{-1,-2},{ 1,-2},{ 2,-1},{-2,-2},{ 2,-2},
};

static const uint8_t ff_vp56_def_mb_types_stats[3][10][2] = {
	{ { 69,42},{  1, 2},{ 1,  7},{ 44,42},{ 6,22},{  1, 3},{  0, 2},{ 1,  5},{ 0, 1},{ 0, 0} },
	{ {229, 8},{  1, 1},{ 0,  8},{  0, 0},{ 0, 0},{  1, 2},{  0, 1},{ 0,  0},{ 1, 1},{ 0, 0} },
	{ {122,35},{  1, 1},{ 1,  6},{ 46,34},{ 0, 0},{  1, 2},{  0, 1},{ 0,  1},{ 1, 1},{ 0, 0} },
};

static const uint8_t ff_vp56_pre_def_mb_type_stats[16][3][10][2] = {
  { { {  9,15},{ 32,25},{ 7, 19},{  9,21},{ 1,12},{ 14,12},{  3,18},{14, 23},{ 3,10},{ 0, 4} },
    { { 41,22},{  1, 0},{ 1, 31},{  0, 0},{ 0, 0},{  0, 1},{  1, 7},{ 0,  1},{98,25},{ 4,10} },
    { {  2, 3},{  2, 3},{ 0,  2},{  0, 2},{ 0, 0},{ 11, 4},{  1, 4},{ 0,  2},{ 3, 2},{ 0, 4} } },
  { { { 48,39},{  1, 2},{11, 27},{ 29,44},{ 7,27},{  1, 4},{  0, 3},{ 1,  6},{ 1, 2},{ 0, 0} },
    { {123,37},{  6, 4},{ 1, 27},{  0, 0},{ 0, 0},{  5, 8},{  1, 7},{ 0,  1},{12,10},{ 0, 2} },
    { { 49,46},{  3, 4},{ 7, 31},{ 42,41},{ 0, 0},{  2, 6},{  1, 7},{ 1,  4},{ 2, 4},{ 0, 1} } },
  { { { 21,32},{  1, 2},{ 4, 10},{ 32,43},{ 6,23},{  2, 3},{  1,19},{ 1,  6},{12,21},{ 0, 7} },
    { { 26,14},{ 14,12},{ 0, 24},{  0, 0},{ 0, 0},{ 55,17},{  1, 9},{ 0, 36},{ 5, 7},{ 1, 3} },
    { { 26,25},{  1, 1},{ 2, 10},{ 67,39},{ 0, 0},{  1, 1},{  0,14},{ 0,  2},{31,26},{ 1, 6} } },
  { { { 69,83},{  0, 0},{ 0,  2},{ 10,29},{ 3,12},{  0, 1},{  0, 3},{ 0,  3},{ 2, 2},{ 0, 0} },
    { {209, 5},{  0, 0},{ 0, 27},{  0, 0},{ 0, 0},{  0, 1},{  0, 1},{ 0,  1},{ 0, 0},{ 0, 0} },
    { {103,46},{  1, 2},{ 2, 10},{ 33,42},{ 0, 0},{  1, 4},{  0, 3},{ 0,  1},{ 1, 3},{ 0, 0} } },
  { { { 11,20},{  1, 4},{18, 36},{ 43,48},{13,35},{  0, 2},{  0, 5},{ 3, 12},{ 1, 2},{ 0, 0} },
    { {  2, 5},{  4, 5},{ 0,121},{  0, 0},{ 0, 0},{  0, 3},{  2, 4},{ 1,  4},{ 2, 2},{ 0, 1} },
    { { 14,31},{  9,13},{14, 54},{ 22,29},{ 0, 0},{  2, 6},{  4,18},{ 6, 13},{ 1, 5},{ 0, 1} } },
  { { { 70,44},{  0, 1},{ 2, 10},{ 37,46},{ 8,26},{  0, 2},{  0, 2},{ 0,  2},{ 0, 1},{ 0, 0} },
    { {175, 5},{  0, 1},{ 0, 48},{  0, 0},{ 0, 0},{  0, 2},{  0, 1},{ 0,  2},{ 0, 1},{ 0, 0} },
    { { 85,39},{  0, 0},{ 1,  9},{ 69,40},{ 0, 0},{  0, 1},{  0, 3},{ 0,  1},{ 2, 3},{ 0, 0} } },
  { { {  8,15},{  0, 1},{ 8, 21},{ 74,53},{22,42},{  0, 1},{  0, 2},{ 0,  3},{ 1, 2},{ 0, 0} },
    { { 83, 5},{  2, 3},{ 0,102},{  0, 0},{ 0, 0},{  1, 3},{  0, 2},{ 0,  1},{ 0, 0},{ 0, 0} },
    { { 31,28},{  0, 0},{ 3, 14},{130,34},{ 0, 0},{  0, 1},{  0, 3},{ 0,  1},{ 3, 3},{ 0, 1} } },
  { { {141,42},{  0, 0},{ 1,  4},{ 11,24},{ 1,11},{  0, 1},{  0, 1},{ 0,  2},{ 0, 0},{ 0, 0} },
    { {233, 6},{  0, 0},{ 0,  8},{  0, 0},{ 0, 0},{  0, 1},{  0, 1},{ 0,  0},{ 0, 1},{ 0, 0} },
    { {171,25},{  0, 0},{ 1,  5},{ 25,21},{ 0, 0},{  0, 1},{  0, 1},{ 0,  0},{ 0, 0},{ 0, 0} } },
  { { {  8,19},{  4,10},{24, 45},{ 21,37},{ 9,29},{  0, 3},{  1, 7},{11, 25},{ 0, 2},{ 0, 1} },
    { { 34,16},{112,21},{ 1, 28},{  0, 0},{ 0, 0},{  6, 8},{  1, 7},{ 0,  3},{ 2, 5},{ 0, 2} },
    { { 17,21},{ 68,29},{ 6, 15},{ 13,22},{ 0, 0},{  6,12},{  3,14},{ 4, 10},{ 1, 7},{ 0, 3} } },
  { { { 46,42},{  0, 1},{ 2, 10},{ 54,51},{10,30},{  0, 2},{  0, 2},{ 0,  1},{ 0, 1},{ 0, 0} },
    { {159,35},{  2, 2},{ 0, 25},{  0, 0},{ 0, 0},{  3, 6},{  0, 5},{ 0,  1},{ 4, 4},{ 0, 1} },
    { { 51,39},{  0, 1},{ 2, 12},{ 91,44},{ 0, 0},{  0, 2},{  0, 3},{ 0,  1},{ 2, 3},{ 0, 1} } },
  { { { 28,32},{  0, 0},{ 3, 10},{ 75,51},{14,33},{  0, 1},{  0, 2},{ 0,  1},{ 1, 2},{ 0, 0} },
    { { 75,39},{  5, 7},{ 2, 48},{  0, 0},{ 0, 0},{  3,11},{  2,16},{ 1,  4},{ 7,10},{ 0, 2} },
    { { 81,25},{  0, 0},{ 2,  9},{106,26},{ 0, 0},{  0, 1},{  0, 1},{ 0,  1},{ 1, 1},{ 0, 0} } },
  { { {100,46},{  0, 1},{ 3,  9},{ 21,37},{ 5,20},{  0, 1},{  0, 2},{ 1,  2},{ 0, 1},{ 0, 0} },
    { {212,21},{  0, 1},{ 0,  9},{  0, 0},{ 0, 0},{  1, 2},{  0, 2},{ 0,  0},{ 2, 2},{ 0, 0} },
    { {140,37},{  0, 1},{ 1,  8},{ 24,33},{ 0, 0},{  1, 2},{  0, 2},{ 0,  1},{ 1, 2},{ 0, 0} } },
  { { { 27,29},{  0, 1},{ 9, 25},{ 53,51},{12,34},{  0, 1},{  0, 3},{ 1,  5},{ 0, 2},{ 0, 0} },
    { {  4, 2},{  0, 0},{ 0,172},{  0, 0},{ 0, 0},{  0, 1},{  0, 2},{ 0,  0},{ 2, 0},{ 0, 0} },
    { { 14,23},{  1, 3},{11, 53},{ 90,31},{ 0, 0},{  0, 3},{  1, 5},{ 2,  6},{ 1, 2},{ 0, 0} } },
  { { { 80,38},{  0, 0},{ 1,  4},{ 69,33},{ 5,16},{  0, 1},{  0, 1},{ 0,  0},{ 0, 1},{ 0, 0} },
    { {187,22},{  1, 1},{ 0, 17},{  0, 0},{ 0, 0},{  3, 6},{  0, 4},{ 0,  1},{ 4, 4},{ 0, 1} },
    { {123,29},{  0, 0},{ 1,  7},{ 57,30},{ 0, 0},{  0, 1},{  0, 1},{ 0,  1},{ 0, 1},{ 0, 0} } },
  { { { 16,20},{  0, 0},{ 2,  8},{104,49},{15,33},{  0, 1},{  0, 1},{ 0,  1},{ 1, 1},{ 0, 0} },
    { {133, 6},{  1, 2},{ 1, 70},{  0, 0},{ 0, 0},{  0, 2},{  0, 4},{ 0,  3},{ 1, 1},{ 0, 0} },
    { { 13,14},{  0, 0},{ 4, 20},{175,20},{ 0, 0},{  0, 1},{  0, 1},{ 0,  1},{ 1, 1},{ 0, 0} } },
  { { {194,16},{  0, 0},{ 1,  1},{  1, 9},{ 1, 3},{  0, 0},{  0, 1},{ 0,  1},{ 0, 0},{ 0, 0} },
    { {251, 1},{  0, 0},{ 0,  2},{  0, 0},{ 0, 0},{  0, 0},{  0, 0},{ 0,  0},{ 0, 0},{ 0, 0} },
    { {202,23},{  0, 0},{ 1,  3},{  2, 9},{ 0, 0},{  0, 1},{  0, 1},{ 0,  1},{ 0, 0},{ 0, 0} } },
};

// vp6data.h
static const uint8_t vp6_def_fdv_vector_model[2][8] = {
	{ 247, 210, 135, 68, 138, 220, 239, 246 },
	{ 244, 184, 201, 44, 173, 221, 239, 253 },
};
static const uint8_t vp6_def_pdv_vector_model[2][7] = {
	{ 225, 146, 172, 147, 214,  39, 156 },
	{ 204, 170, 119, 235, 140, 230, 228 },
};
static const uint8_t vp6_def_coeff_reorder[64] = {
	 0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 4, 4, 4,
	 5, 5, 5, 5, 6, 6, 7, 7, 7, 7, 7, 8, 8, 9, 9, 9,
	 9, 9, 9,10,10,11,11,11,11,11,11,12,12,12,12,12,
	12,13,13,13,13,13,14,14,14,14,15,15,15,15,15,15,
};
static const uint8_t vp6_def_runv_coeff_model[2][14] = {
	{ 198, 197, 196, 146, 198, 204, 169, 142, 130, 136, 149, 149, 191, 249 },
	{ 135, 201, 181, 154,  98, 117, 132, 126, 146, 169, 184, 240, 246, 254 },
};
static const uint8_t vp6_sig_dct_pct[2][2] = { { 237, 246 }, { 231, 243 } };
static const uint8_t vp6_pdv_pct[2][7] = {
	{ 253, 253, 254, 254, 254, 254, 254 },
	{ 245, 253, 254, 254, 254, 254, 254 },
};
static const uint8_t vp6_fdv_pct[2][8] = {
	{ 254, 254, 254, 254, 254, 250, 250, 252 },
	{ 254, 254, 254, 254, 254, 251, 251, 254 },
};
static const uint8_t vp6_dccv_pct[2][11] = {
	{ 146, 255, 181, 207, 232, 243, 238, 251, 244, 250, 249 },
	{ 179, 255, 214, 240, 250, 255, 244, 255, 255, 255, 255 },
};
static const uint8_t vp6_coeff_reorder_pct[64] = {
	255, 132, 132, 159, 153, 151, 161, 170, 164, 162, 136, 110, 103, 114, 129, 118,
	124, 125, 132, 136, 114, 110, 142, 135, 134, 123, 143, 126, 153, 183, 166, 161,
	171, 180, 179, 164, 203, 218, 225, 217, 215, 206, 203, 217, 229, 241, 248, 243,
	253, 255, 253, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
};
static const uint8_t vp6_runv_pct[2][14] = {
	{ 219, 246, 238, 249, 232, 239, 249, 255, 248, 253, 239, 244, 241, 248 },
	{ 198, 232, 251, 253, 219, 241, 253, 255, 248, 249, 244, 238, 251, 255 },
};
static const uint8_t vp6_ract_pct[3][2][6][11] = {
  { { { 227,246,230,247,244,255,255,255,255,255,255 },{ 255,255,209,231,231,249,249,253,255,255,255 },
      { 255,255,225,242,241,251,253,255,255,255,255 },{ 255,255,241,253,252,255,255,255,255,255,255 },
      { 255,255,248,255,255,255,255,255,255,255,255 },{ 255,255,255,255,255,255,255,255,255,255,255 } },
    { { 240,255,248,255,255,255,255,255,255,255,255 },{ 255,255,240,253,255,255,255,255,255,255,255 },
      { 255,255,255,255,255,255,255,255,255,255,255 },{ 255,255,255,255,255,255,255,255,255,255,255 },
      { 255,255,255,255,255,255,255,255,255,255,255 },{ 255,255,255,255,255,255,255,255,255,255,255 } } },
  { { { 206,203,227,239,247,255,253,255,255,255,255 },{ 207,199,220,236,243,252,252,255,255,255,255 },
      { 212,219,230,243,244,253,252,255,255,255,255 },{ 236,237,247,252,253,255,255,255,255,255,255 },
      { 240,240,248,255,255,255,255,255,255,255,255 },{ 255,255,255,255,255,255,255,255,255,255,255 } },
    { { 230,233,249,255,255,255,255,255,255,255,255 },{ 238,238,250,255,255,255,255,255,255,255,255 },
      { 248,251,255,255,255,255,255,255,255,255,255 },{ 255,255,255,255,255,255,255,255,255,255,255 },
      { 255,255,255,255,255,255,255,255,255,255,255 },{ 255,255,255,255,255,255,255,255,255,255,255 } } },
  { { { 225,239,227,231,244,253,243,255,255,253,255 },{ 232,234,224,228,242,249,242,252,251,251,255 },
      { 235,249,238,240,251,255,249,255,253,253,255 },{ 249,253,251,250,255,255,255,255,255,255,255 },
      { 251,250,249,255,255,255,255,255,255,255,255 },{ 255,255,255,255,255,255,255,255,255,255,255 } },
    { { 243,244,250,250,255,255,255,255,255,255,255 },{ 249,248,250,253,255,255,255,255,255,255,255 },
      { 253,255,255,255,255,255,255,255,255,255,255 },{ 255,255,255,255,255,255,255,255,255,255,255 },
      { 255,255,255,255,255,255,255,255,255,255,255 },{ 255,255,255,255,255,255,255,255,255,255,255 } } },
};
static const int vp6_dccv_lc[3][5][2] = {
	{ { 122,133 },{ 0,1 },{ 78, 171 },{ 139,117 },{ 168, 79 } },
	{ { 133, 51 },{ 0,1 },{ 169,  71 },{ 214, 44 },{ 210, 38 } },
	{ { 142,-16 },{ 0,1 },{ 221, -30 },{ 246, -3 },{ 203, 17 } },
};
static const uint8_t vp6_coeff_groups[64] = {
	0,0,1,1,1,2,2,2,2,2,2,3,3,3,3,3,
	3,3,3,3,3,3,4,4,4,4,4,4,4,4,4,4,
	4,4,4,4,4,5,5,5,5,5,5,5,5,5,5,5,
	5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,
};
static const int16_t vp6_block_copy_filter[17][8][4] = {
	{ {0,128,0,0},{-3,122,9,0},{-4,109,24,-1},{-5,91,45,-3},{-4,68,68,-4},{-3,45,91,-5},{-1,24,109,-4},{0,9,122,-3} },
	{ {0,128,0,0},{-4,124,9,-1},{-5,110,25,-2},{-6,91,46,-3},{-5,69,69,-5},{-3,46,91,-6},{-2,25,110,-5},{-1,9,124,-4} },
	{ {0,128,0,0},{-4,123,10,-1},{-6,110,26,-2},{-7,92,47,-4},{-6,70,70,-6},{-4,47,92,-7},{-2,26,110,-6},{-1,10,123,-4} },
	{ {0,128,0,0},{-5,124,10,-1},{-7,110,27,-2},{-7,91,48,-4},{-6,70,70,-6},{-4,48,92,-8},{-2,27,110,-7},{-1,10,124,-5} },
	{ {0,128,0,0},{-6,124,11,-1},{-8,111,28,-3},{-8,92,49,-5},{-7,71,71,-7},{-5,49,92,-8},{-3,28,111,-8},{-1,11,124,-6} },
	{ {0,128,0,0},{-6,123,12,-1},{-9,111,29,-3},{-9,93,50,-6},{-8,72,72,-8},{-6,50,93,-9},{-3,29,111,-9},{-1,12,123,-6} },
	{ {0,128,0,0},{-7,124,12,-1},{-10,111,30,-3},{-10,93,51,-6},{-9,73,73,-9},{-6,51,93,-10},{-3,30,111,-10},{-1,12,124,-7} },
	{ {0,128,0,0},{-7,123,13,-1},{-11,112,31,-4},{-11,94,52,-7},{-10,74,74,-10},{-7,52,94,-11},{-4,31,112,-11},{-1,13,123,-7} },
	{ {0,128,0,0},{-8,124,13,-1},{-12,112,32,-4},{-12,94,53,-7},{-10,74,74,-10},{-7,53,94,-12},{-4,32,112,-12},{-1,13,124,-8} },
	{ {0,128,0,0},{-9,124,14,-1},{-13,112,33,-4},{-13,95,54,-8},{-11,75,75,-11},{-8,54,95,-13},{-4,33,112,-13},{-1,14,124,-9} },
	{ {0,128,0,0},{-9,123,15,-1},{-14,113,34,-5},{-14,95,55,-8},{-12,76,76,-12},{-8,55,95,-14},{-5,34,112,-13},{-1,15,123,-9} },
	{ {0,128,0,0},{-10,124,15,-1},{-14,113,34,-5},{-15,96,56,-9},{-13,77,77,-13},{-9,56,96,-15},{-5,34,113,-14},{-1,15,124,-10} },
	{ {0,128,0,0},{-10,123,16,-1},{-15,113,35,-5},{-16,98,56,-10},{-14,78,78,-14},{-10,56,98,-16},{-5,35,113,-15},{-1,16,123,-10} },
	{ {0,128,0,0},{-11,124,17,-2},{-16,113,36,-5},{-17,98,57,-10},{-14,78,78,-14},{-10,57,98,-17},{-5,36,113,-16},{-2,17,124,-11} },
	{ {0,128,0,0},{-12,125,17,-2},{-17,114,37,-6},{-18,99,58,-11},{-15,79,79,-15},{-11,58,99,-18},{-6,37,114,-17},{-2,17,125,-12} },
	{ {0,128,0,0},{-12,124,18,-2},{-18,114,38,-6},{-19,99,59,-11},{-16,80,80,-16},{-11,59,99,-19},{-6,38,114,-18},{-2,18,124,-12} },
	{ {0,128,0,0},{-4,118,16,-2},{-7,106,34,-5},{-8,90,53,-7},{-8,72,72,-8},{-7,53,90,-8},{-5,34,106,-7},{-2,16,118,-4} },
};
static const VP56Tree vp6_pcr_tree[] = {
	{ 8, 0}, { 4, 1}, { 2, 2}, {-1}, {-2}, { 2, 3}, {-3}, {-4},
	{ 8, 4}, { 4, 5}, { 2, 6}, {-5}, {-6}, { 2, 7}, {-7}, {-8}, {-0},
};
static const uint8_t vp6_coord_div[] = { 4, 4, 4, 4, 8, 8 };
static const uint8_t vp6_huff_coeff_map[] = {
	13, 14, 11, 0, 1, 15, 16, 18, 2, 17, 3, 4, 19, 20, 5, 6, 21, 22, 7, 8, 9, 10
};
static const uint8_t vp6_huff_run_map[] = {
	10, 13, 11, 12, 0, 1, 2, 3, 14, 8, 15, 16, 4, 5, 6, 7
};

// ─── model ────────────────────────────────────────────────────────────────────
struct VP56Model {
	uint8_t coeff_reorder[64];
	uint8_t coeff_index_to_pos[64];
	uint8_t coeff_index_to_idct_selector[64];
	uint8_t vector_sig[2];
	uint8_t vector_dct[2];
	uint8_t vector_pdv[2][7];
	uint8_t vector_fdv[2][8];
	uint8_t coeff_dccv[2][11];
	uint8_t coeff_ract[2][3][6][11];
	uint8_t coeff_dcct[2][3][5];
	uint8_t coeff_runv[2][14];
	uint8_t mb_type[3][10][10];
	uint8_t mb_types_stats[3][10][2];
};

// ─── Huffman (huffman.c) ──────────────────────────────────────────────────────
#define HNODE (-1)
#define FF_HUFFMAN_FLAG_HNODE_FIRST 1
struct Node { int16_t sym; int16_t n0; uint32_t count; };

static int vp6_huff_cmp(const void* va, const void* vb) {
	const Node* a = (const Node*)va;
	const Node* b = (const Node*)vb;
	return (a->count - b->count) * 16 + (b->sym - a->sym);
}

// Flat VLC: peek `maxLen` bits -> (len, sym). len==0 means no code (error).
struct FlatVLC { int maxLen; uint8_t len[4096]; uint8_t sym[4096]; };

static void get_tree_codes(int8_t* lens, uint8_t* xlat, Node* nodes, int node, int pl,
                           int* pos, int no_zero_count) {
	int s = nodes[node].sym;
	if (s != HNODE || (no_zero_count && !nodes[node].count)) {
		lens[*pos] = (int8_t)pl;
		xlat[*pos] = (uint8_t)s;
		(*pos)++;
	} else {
		get_tree_codes(lens, xlat, nodes, nodes[node].n0, pl + 1, pos, no_zero_count);
		get_tree_codes(lens, xlat, nodes, nodes[node].n0 + 1, pl + 1, pos, no_zero_count);
	}
}

// Build a flat decode table from (length, symbol) pairs in DFS order, assigning
// canonical MSB-justified codes (the ff_vlc_init_from_lengths accumulator).
static void build_flat_vlc(FlatVLC* v, const int8_t* lens, const uint8_t* xlat, int pos) {
	int maxLen = 0;
	for (int k = 0; k < pos; k++) if (lens[k] > maxLen) maxLen = lens[k];
	if (maxLen > 12) { g_err = 1; maxLen = 12; }   // shouldn't happen for <=12 symbols
	v->maxLen = maxLen;
	memset(v->len, 0, (size_t)(1 << maxLen) * sizeof(v->len[0]));
	uint32_t code = 0;
	for (int k = 0; k < pos; k++) {
		int len = lens[k];
		if (len <= 0 || len > maxLen) { code += len > 0 ? (1u << (32 - len)) : 0; continue; }
		uint32_t top = code >> (32 - len);
		uint32_t base = top << (maxLen - len);
		int fill = 1 << (maxLen - len);
		for (int j = 0; j < fill; j++) {
			v->len[base + j] = (uint8_t)len;
			v->sym[base + j] = xlat[k];
		}
		code += 1u << (32 - len);
	}
}

static int get_vlc(GetBits* g, const FlatVLC* v) {
	uint32_t idx = gb_peek(g, v->maxLen);
	int len = v->len[idx];
	if (len == 0) { g_err = 1; return 0; }
	g->pos += len;
	return v->sym[idx];
}

// ff_huff_build_tree -> build_flat_vlc. nodes[] sized 2*size with counts preset.
static void ff_huff_build_tree(FlatVLC* v, int nb_codes, Node* nodes, int flags) {
	int i, j, cur_node;
	for (i = 0; i < nb_codes; i++) { nodes[i].sym = (int16_t)i; nodes[i].n0 = -2; }

	// AV_QSORT(nodes, nb_codes, Node, vp6_huff_cmp) — a stable sort isn't required
	// (qsort isn't stable either); the comparator fully orders by (count, -sym).
	qsort(nodes, nb_codes, sizeof(Node), vp6_huff_cmp);

	cur_node = nb_codes;
	nodes[nb_codes * 2 - 1].count = 0;
	for (i = 0; i < nb_codes * 2 - 1; i += 2) {
		uint32_t cur_count = nodes[i].count + nodes[i + 1].count;
		for (j = cur_node; j > i + 2; j--) {
			if (cur_count > nodes[j - 1].count ||
			    (cur_count == nodes[j - 1].count && !(flags & FF_HUFFMAN_FLAG_HNODE_FIRST)))
				break;
			nodes[j] = nodes[j - 1];
		}
		nodes[j].sym = HNODE;
		nodes[j].count = cur_count;
		nodes[j].n0 = (int16_t)i;
		cur_node++;
	}

	int8_t lens[256];
	uint8_t xlat[256];
	int pos = 0;
	get_tree_codes(lens, xlat, nodes, nb_codes * 2 - 2, 0, &pos, 1);
	build_flat_vlc(v, lens, xlat, pos);
}

static void vp6_build_huff_tree(FlatVLC* v, const uint8_t* coeff_model, const uint8_t* map, int size) {
	Node nodes[2 * 12];
	Node* tmp = &nodes[size];
	tmp[0].count = 256;
	for (int i = 0; i < size - 1; i++) {
		int a = tmp[i].count * coeff_model[i] >> 8;
		int b = tmp[i].count * (255 - coeff_model[i]) >> 8;
		nodes[map[2 * i]].count = a + !a;
		nodes[map[2 * i + 1]].count = b + !b;
	}
	ff_huff_build_tree(v, size, nodes, FF_HUFFMAN_FLAG_HNODE_FIRST);
}

// ─── VP3 IDCT (vp3dsp.c) ──────────────────────────────────────────────────────
#define IdctAdjustBeforeShift 8
#define xC1S7 64277
#define xC2S6 60547
#define xC3S5 54491
#define xC4S4 46341
#define xC5S3 36410
#define xC6S2 25080
#define xC7S1 12785

#ifndef VP6_FORCE_SCALAR_IDCT
#define IDCT_HAVE_SIMD __wasm_simd128__
#endif

static void idct_scalar(uint8_t* dst, ptrdiff_t stride, int16_t* input, int type) {
	int16_t* ip = input;
	int A, B, C, D, Ad, Bd, Cd, Dd, E, F, G, H, Ed, Gd, Add, Bdd, Fd, Hd, i;

	for (i = 0; i < 8; i++) {
		if (ip[0 * 8] | ip[1 * 8] | ip[2 * 8] | ip[3 * 8] | ip[4 * 8] | ip[5 * 8] | ip[6 * 8] | ip[7 * 8]) {
			A = M(xC1S7, ip[1 * 8]) + M(xC7S1, ip[7 * 8]);
			B = M(xC7S1, ip[1 * 8]) - M(xC1S7, ip[7 * 8]);
			C = M(xC3S5, ip[3 * 8]) + M(xC5S3, ip[5 * 8]);
			D = M(xC3S5, ip[5 * 8]) - M(xC5S3, ip[3 * 8]);
			Ad = M(xC4S4, (A - C));
			Bd = M(xC4S4, (B - D));
			Cd = A + C;
			Dd = B + D;
			E = M(xC4S4, (ip[0 * 8] + ip[4 * 8]));
			F = M(xC4S4, (ip[0 * 8] - ip[4 * 8]));
			G = M(xC2S6, ip[2 * 8]) + M(xC6S2, ip[6 * 8]);
			H = M(xC6S2, ip[2 * 8]) - M(xC2S6, ip[6 * 8]);
			Ed = E - G; Gd = E + G;
			Add = F + Ad; Bdd = Bd - H;
			Fd = F - Ad; Hd = Bd + H;
			ip[0 * 8] = (int16_t)(Gd + Cd);
			ip[7 * 8] = (int16_t)(Gd - Cd);
			ip[1 * 8] = (int16_t)(Add + Hd);
			ip[2 * 8] = (int16_t)(Add - Hd);
			ip[3 * 8] = (int16_t)(Ed + Dd);
			ip[4 * 8] = (int16_t)(Ed - Dd);
			ip[5 * 8] = (int16_t)(Fd + Bdd);
			ip[6 * 8] = (int16_t)(Fd - Bdd);
		}
		ip += 1;
	}
	ip = input;
	for (i = 0; i < 8; i++) {
		if (ip[1] | ip[2] | ip[3] | ip[4] | ip[5] | ip[6] | ip[7]) {
			A = M(xC1S7, ip[1]) + M(xC7S1, ip[7]);
			B = M(xC7S1, ip[1]) - M(xC1S7, ip[7]);
			C = M(xC3S5, ip[3]) + M(xC5S3, ip[5]);
			D = M(xC3S5, ip[5]) - M(xC5S3, ip[3]);
			Ad = M(xC4S4, (A - C));
			Bd = M(xC4S4, (B - D));
			Cd = A + C;
			Dd = B + D;
			E = M(xC4S4, (ip[0] + ip[4])) + 8;
			F = M(xC4S4, (ip[0] - ip[4])) + 8;
			if (type == 1) { E += 16 * 128; F += 16 * 128; }
			G = M(xC2S6, ip[2]) + M(xC6S2, ip[6]);
			H = M(xC6S2, ip[2]) - M(xC2S6, ip[6]);
			Ed = E - G; Gd = E + G;
			Add = F + Ad; Bdd = Bd - H;
			Fd = F - Ad; Hd = Bd + H;
			if (type == 1) {
				dst[0 * stride] = av_clip_uint8((Gd + Cd) >> 4);
				dst[7 * stride] = av_clip_uint8((Gd - Cd) >> 4);
				dst[1 * stride] = av_clip_uint8((Add + Hd) >> 4);
				dst[2 * stride] = av_clip_uint8((Add - Hd) >> 4);
				dst[3 * stride] = av_clip_uint8((Ed + Dd) >> 4);
				dst[4 * stride] = av_clip_uint8((Ed - Dd) >> 4);
				dst[5 * stride] = av_clip_uint8((Fd + Bdd) >> 4);
				dst[6 * stride] = av_clip_uint8((Fd - Bdd) >> 4);
			} else {
				dst[0 * stride] = av_clip_uint8(dst[0 * stride] + ((Gd + Cd) >> 4));
				dst[7 * stride] = av_clip_uint8(dst[7 * stride] + ((Gd - Cd) >> 4));
				dst[1 * stride] = av_clip_uint8(dst[1 * stride] + ((Add + Hd) >> 4));
				dst[2 * stride] = av_clip_uint8(dst[2 * stride] + ((Add - Hd) >> 4));
				dst[3 * stride] = av_clip_uint8(dst[3 * stride] + ((Ed + Dd) >> 4));
				dst[4 * stride] = av_clip_uint8(dst[4 * stride] + ((Ed - Dd) >> 4));
				dst[5 * stride] = av_clip_uint8(dst[5 * stride] + ((Fd + Bdd) >> 4));
				dst[6 * stride] = av_clip_uint8(dst[6 * stride] + ((Fd - Bdd) >> 4));
			}
		} else {
			if (type == 1) {
				uint8_t v = av_clip_uint8(128 + ((xC4S4 * ip[0] + (IdctAdjustBeforeShift << 16)) >> 20));
				dst[0 * stride] = dst[1 * stride] = dst[2 * stride] = dst[3 * stride] =
				dst[4 * stride] = dst[5 * stride] = dst[6 * stride] = dst[7 * stride] = v;
			} else if (ip[0]) {
				int v = (xC4S4 * ip[0] + (IdctAdjustBeforeShift << 16)) >> 20;
				dst[0 * stride] = av_clip_uint8(dst[0 * stride] + v);
				dst[1 * stride] = av_clip_uint8(dst[1 * stride] + v);
				dst[2 * stride] = av_clip_uint8(dst[2 * stride] + v);
				dst[3 * stride] = av_clip_uint8(dst[3 * stride] + v);
				dst[4 * stride] = av_clip_uint8(dst[4 * stride] + v);
				dst[5 * stride] = av_clip_uint8(dst[5 * stride] + v);
				dst[6 * stride] = av_clip_uint8(dst[6 * stride] + v);
				dst[7 * stride] = av_clip_uint8(dst[7 * stride] + v);
			}
		}
		ip += 8;
		dst++;
	}
}

// idct10: fast path for blocks whose nonzero coeffs fit the top-left 4x4 region.
static void idct10(uint8_t* dst, ptrdiff_t stride, int16_t* input, int type) {
	int16_t* ip = input;
	int A, B, C, D, Ad, Bd, Cd, Dd, E, F, G, H, Ed, Gd, Add, Bdd, Fd, Hd, i;

	for (i = 0; i < 4; i++) {
		if (ip[0 * 8] | ip[1 * 8] | ip[2 * 8] | ip[3 * 8]) {
			A =  M(xC1S7, ip[1 * 8]);
			B =  M(xC7S1, ip[1 * 8]);
			C =  M(xC3S5, ip[3 * 8]);
			D = -M(xC5S3, ip[3 * 8]);
			Ad = M(xC4S4, (A - C));
			Bd = M(xC4S4, (B - D));
			Cd = A + C; Dd = B + D;
			E = M(xC4S4, ip[0 * 8]); F = E;
			G = M(xC2S6, ip[2 * 8]); H = M(xC6S2, ip[2 * 8]);
			Ed = E - G; Gd = E + G;
			Add = F + Ad; Bdd = Bd - H;
			Fd = F - Ad; Hd = Bd + H;
			ip[0 * 8] = (int16_t)(Gd + Cd);
			ip[7 * 8] = (int16_t)(Gd - Cd);
			ip[1 * 8] = (int16_t)(Add + Hd);
			ip[2 * 8] = (int16_t)(Add - Hd);
			ip[3 * 8] = (int16_t)(Ed + Dd);
			ip[4 * 8] = (int16_t)(Ed - Dd);
			ip[5 * 8] = (int16_t)(Fd + Bdd);
			ip[6 * 8] = (int16_t)(Fd - Bdd);
		}
		ip += 1;
	}
	ip = input;
	for (i = 0; i < 8; i++) {
		if (ip[0] | ip[1] | ip[2] | ip[3]) {
			A =  M(xC1S7, ip[1]);
			B =  M(xC7S1, ip[1]);
			C =  M(xC3S5, ip[3]);
			D = -M(xC5S3, ip[3]);
			Ad = M(xC4S4, (A - C));
			Bd = M(xC4S4, (B - D));
			Cd = A + C; Dd = B + D;
			E = M(xC4S4, ip[0]);
			if (type == 1) E += 16 * 128;
			F = E;
			G = M(xC2S6, ip[2]); H = M(xC6S2, ip[2]);
			Ed = E - G; Gd = E + G;
			Add = F + Ad; Bdd = Bd - H;
			Fd = F - Ad; Hd = Bd + H;
			Gd += 8; Add += 8; Ed += 8; Fd += 8;
			if (type == 1) {
				dst[0 * stride] = av_clip_uint8((Gd + Cd) >> 4);
				dst[7 * stride] = av_clip_uint8((Gd - Cd) >> 4);
				dst[1 * stride] = av_clip_uint8((Add + Hd) >> 4);
				dst[2 * stride] = av_clip_uint8((Add - Hd) >> 4);
				dst[3 * stride] = av_clip_uint8((Ed + Dd) >> 4);
				dst[4 * stride] = av_clip_uint8((Ed - Dd) >> 4);
				dst[5 * stride] = av_clip_uint8((Fd + Bdd) >> 4);
				dst[6 * stride] = av_clip_uint8((Fd - Bdd) >> 4);
			} else {
				dst[0 * stride] = av_clip_uint8(dst[0 * stride] + ((Gd + Cd) >> 4));
				dst[7 * stride] = av_clip_uint8(dst[7 * stride] + ((Gd - Cd) >> 4));
				dst[1 * stride] = av_clip_uint8(dst[1 * stride] + ((Add + Hd) >> 4));
				dst[2 * stride] = av_clip_uint8(dst[2 * stride] + ((Add - Hd) >> 4));
				dst[3 * stride] = av_clip_uint8(dst[3 * stride] + ((Ed + Dd) >> 4));
				dst[4 * stride] = av_clip_uint8(dst[4 * stride] + ((Ed - Dd) >> 4));
				dst[5 * stride] = av_clip_uint8(dst[5 * stride] + ((Fd + Bdd) >> 4));
				dst[6 * stride] = av_clip_uint8(dst[6 * stride] + ((Fd - Bdd) >> 4));
			}
		} else if (type == 1) {
			dst[0 * stride] = dst[1 * stride] = dst[2 * stride] = dst[3 * stride] =
			dst[4 * stride] = dst[5 * stride] = dst[6 * stride] = dst[7 * stride] = 128;
		}
		ip += 8;
		dst++;
	}
}

#ifdef __wasm_simd128__
// SIMD (i32x4) VP3 IDCT — "our own wasm simd idct". Bit-exact to idct_scalar:
//   * M(a,b) = (a*b)>>16 maps to i32x4_mul (modular low-32, matching the scalar
//     uint32 wrap) + arithmetic i32x4_shr(16); -fwrapv keeps it well-defined.
//   * Pass 1 (vertical) processes 4 columns/lanes at once — no transpose, since a
//     column's 8 inputs are 8 row-segments; results are TRUNCATED to int16 (low
//     halfword), matching the scalar `ip[..] = (int16_t)..` (which overflows i16).
//   * Pass 2 (horizontal) processes 4 rows/lanes at once via one 4x8 transpose; the
//     output is already transposed into dst (dst[row + k*stride]) so the 4 lanes are
//     4 contiguous dst bytes — no output transpose. clip8 == the saturating u8 narrow.
//   The scalar per-row/col all-zero shortcuts are provably identical to the full
//   butterfly on that data (verified: the DC-only fast paths equal (E)>>4), so they
//   are omitted here. type 1 = put (+128 bias), type 2 = add (read-modify-write dst).
static inline v128_t Msimd(int c, v128_t x) {
	return wasm_i32x4_shr(wasm_i32x4_mul(wasm_i32x4_splat(c), x), 16);
}
static inline void transpose4_i32(v128_t a, v128_t b, v128_t c, v128_t d,
                                  v128_t* o0, v128_t* o1, v128_t* o2, v128_t* o3) {
	v128_t i0 = wasm_i32x4_shuffle(a, b, 0, 4, 1, 5);
	v128_t i1 = wasm_i32x4_shuffle(a, b, 2, 6, 3, 7);
	v128_t i2 = wasm_i32x4_shuffle(c, d, 0, 4, 1, 5);
	v128_t i3 = wasm_i32x4_shuffle(c, d, 2, 6, 3, 7);
	*o0 = wasm_i32x4_shuffle(i0, i2, 0, 1, 4, 5);
	*o1 = wasm_i32x4_shuffle(i0, i2, 2, 3, 6, 7);
	*o2 = wasm_i32x4_shuffle(i1, i3, 0, 1, 4, 5);
	*o3 = wasm_i32x4_shuffle(i1, i3, 2, 3, 6, 7);
}
// truncating i32x4 -> 4xint16 store (low halfword of each lane), == (int16_t) cast
static inline void store_trunc16(int16_t* p, v128_t x) {
	wasm_v128_store64_lane(p, wasm_i16x8_shuffle(x, x, 0, 2, 4, 6, 0, 2, 4, 6), 0);
}

static void idct_simd(uint8_t* dst, ptrdiff_t stride, int16_t* ip, int type) {
	const v128_t c8 = wasm_i32x4_splat(8);
	const v128_t cbias = wasm_i32x4_splat(8 + 16 * 128);

	// Pass 1: vertical transform, 4 columns at a time (lane = column).
	for (int c = 0; c < 8; c += 4) {
		v128_t v0 = wasm_i32x4_load16x4(&ip[0 * 8 + c]);
		v128_t v1 = wasm_i32x4_load16x4(&ip[1 * 8 + c]);
		v128_t v2 = wasm_i32x4_load16x4(&ip[2 * 8 + c]);
		v128_t v3 = wasm_i32x4_load16x4(&ip[3 * 8 + c]);
		v128_t v4 = wasm_i32x4_load16x4(&ip[4 * 8 + c]);
		v128_t v5 = wasm_i32x4_load16x4(&ip[5 * 8 + c]);
		v128_t v6 = wasm_i32x4_load16x4(&ip[6 * 8 + c]);
		v128_t v7 = wasm_i32x4_load16x4(&ip[7 * 8 + c]);

		v128_t A = wasm_i32x4_add(Msimd(xC1S7, v1), Msimd(xC7S1, v7));
		v128_t B = wasm_i32x4_sub(Msimd(xC7S1, v1), Msimd(xC1S7, v7));
		v128_t C = wasm_i32x4_add(Msimd(xC3S5, v3), Msimd(xC5S3, v5));
		v128_t D = wasm_i32x4_sub(Msimd(xC3S5, v5), Msimd(xC5S3, v3));
		v128_t Ad = Msimd(xC4S4, wasm_i32x4_sub(A, C));
		v128_t Bd = Msimd(xC4S4, wasm_i32x4_sub(B, D));
		v128_t Cd = wasm_i32x4_add(A, C);
		v128_t Dd = wasm_i32x4_add(B, D);
		v128_t E = Msimd(xC4S4, wasm_i32x4_add(v0, v4));
		v128_t F = Msimd(xC4S4, wasm_i32x4_sub(v0, v4));
		v128_t G = wasm_i32x4_add(Msimd(xC2S6, v2), Msimd(xC6S2, v6));
		v128_t H = wasm_i32x4_sub(Msimd(xC6S2, v2), Msimd(xC2S6, v6));
		v128_t Ed = wasm_i32x4_sub(E, G), Gd = wasm_i32x4_add(E, G);
		v128_t Add = wasm_i32x4_add(F, Ad), Bdd = wasm_i32x4_sub(Bd, H);
		v128_t Fd = wasm_i32x4_sub(F, Ad), Hd = wasm_i32x4_add(Bd, H);
		store_trunc16(&ip[0 * 8 + c], wasm_i32x4_add(Gd, Cd));
		store_trunc16(&ip[7 * 8 + c], wasm_i32x4_sub(Gd, Cd));
		store_trunc16(&ip[1 * 8 + c], wasm_i32x4_add(Add, Hd));
		store_trunc16(&ip[2 * 8 + c], wasm_i32x4_sub(Add, Hd));
		store_trunc16(&ip[3 * 8 + c], wasm_i32x4_add(Ed, Dd));
		store_trunc16(&ip[4 * 8 + c], wasm_i32x4_sub(Ed, Dd));
		store_trunc16(&ip[5 * 8 + c], wasm_i32x4_add(Fd, Bdd));
		store_trunc16(&ip[6 * 8 + c], wasm_i32x4_sub(Fd, Bdd));
	}

	// Pass 2: horizontal transform, 4 rows at a time (lane = row), output transposed.
	for (int r = 0; r < 8; r += 4) {
		v128_t R0 = wasm_v128_load(&ip[(r + 0) * 8]);
		v128_t R1 = wasm_v128_load(&ip[(r + 1) * 8]);
		v128_t R2 = wasm_v128_load(&ip[(r + 2) * 8]);
		v128_t R3 = wasm_v128_load(&ip[(r + 3) * 8]);
		v128_t X0, X1, X2, X3, X4, X5, X6, X7;
		transpose4_i32(wasm_i32x4_extend_low_i16x8(R0), wasm_i32x4_extend_low_i16x8(R1),
		               wasm_i32x4_extend_low_i16x8(R2), wasm_i32x4_extend_low_i16x8(R3), &X0, &X1, &X2, &X3);
		transpose4_i32(wasm_i32x4_extend_high_i16x8(R0), wasm_i32x4_extend_high_i16x8(R1),
		               wasm_i32x4_extend_high_i16x8(R2), wasm_i32x4_extend_high_i16x8(R3), &X4, &X5, &X6, &X7);

		v128_t A = wasm_i32x4_add(Msimd(xC1S7, X1), Msimd(xC7S1, X7));
		v128_t B = wasm_i32x4_sub(Msimd(xC7S1, X1), Msimd(xC1S7, X7));
		v128_t C = wasm_i32x4_add(Msimd(xC3S5, X3), Msimd(xC5S3, X5));
		v128_t D = wasm_i32x4_sub(Msimd(xC3S5, X5), Msimd(xC5S3, X3));
		v128_t Ad = Msimd(xC4S4, wasm_i32x4_sub(A, C));
		v128_t Bd = Msimd(xC4S4, wasm_i32x4_sub(B, D));
		v128_t Cd = wasm_i32x4_add(A, C);
		v128_t Dd = wasm_i32x4_add(B, D);
		v128_t bias = (type == 1) ? cbias : c8;
		v128_t E = wasm_i32x4_add(Msimd(xC4S4, wasm_i32x4_add(X0, X4)), bias);
		v128_t F = wasm_i32x4_add(Msimd(xC4S4, wasm_i32x4_sub(X0, X4)), bias);
		v128_t G = wasm_i32x4_add(Msimd(xC2S6, X2), Msimd(xC6S2, X6));
		v128_t H = wasm_i32x4_sub(Msimd(xC6S2, X2), Msimd(xC2S6, X6));
		v128_t Ed = wasm_i32x4_sub(E, G), Gd = wasm_i32x4_add(E, G);
		v128_t Add = wasm_i32x4_add(F, Ad), Bdd = wasm_i32x4_sub(Bd, H);
		v128_t Fd = wasm_i32x4_sub(F, Ad), Hd = wasm_i32x4_add(Bd, H);

		v128_t Y[8];
		Y[0] = wasm_i32x4_add(Gd, Cd);
		Y[1] = wasm_i32x4_add(Add, Hd);
		Y[2] = wasm_i32x4_sub(Add, Hd);
		Y[3] = wasm_i32x4_add(Ed, Dd);
		Y[4] = wasm_i32x4_sub(Ed, Dd);
		Y[5] = wasm_i32x4_add(Fd, Bdd);
		Y[6] = wasm_i32x4_sub(Fd, Bdd);
		Y[7] = wasm_i32x4_sub(Gd, Cd);

		for (int k = 0; k < 8; k++) {
			v128_t sh = wasm_i32x4_shr(Y[k], 4);
			uint8_t* p = &dst[r + (ptrdiff_t)k * stride];
			if (type == 2) {
				v128_t d = wasm_u32x4_extend_low_u16x8(wasm_u16x8_extend_low_u8x16(wasm_v128_load32_zero(p)));
				sh = wasm_i32x4_add(d, sh);
			}
			// clip8 via saturating narrows (i32 fits i16; i16->u8 saturate == clamp)
			v128_t packed = wasm_u8x16_narrow_i16x8(wasm_i16x8_narrow_i32x4(sh, sh), wasm_i16x8_splat(0));
			wasm_v128_store32_lane(p, packed, 0);
		}
	}
}
#endif

// A/B toggle so the conformance test can verify SIMD == scalar in one build.
static int g_use_simd =
#ifdef __wasm_simd128__
	1;
#else
	0;
#endif

static void idct_full(uint8_t* d, ptrdiff_t s, int16_t* b, int type) {
#ifdef __wasm_simd128__
	if (g_use_simd) { idct_simd(d, s, b, type); return; }
#endif
	idct_scalar(d, s, b, type);
}

static void idct_put_full(uint8_t* d, ptrdiff_t s, int16_t* b) { idct_full(d, s, b, 1); memset(b, 0, 128); }
static void idct_add_full(uint8_t* d, ptrdiff_t s, int16_t* b) { idct_full(d, s, b, 2); memset(b, 0, 128); }
static void idct10_put(uint8_t* d, ptrdiff_t s, int16_t* b) { idct10(d, s, b, 1); memset(b, 0, 128); }
static void idct10_add(uint8_t* d, ptrdiff_t s, int16_t* b) { idct10(d, s, b, 2); memset(b, 0, 128); }
static void idct_dc_add(uint8_t* dest, ptrdiff_t stride, int16_t* block) {
	int dc = (block[0] + 15) >> 5;
	for (int i = 0; i < 8; i++) {
		dest[0] = av_clip_uint8(dest[0] + dc);
		dest[1] = av_clip_uint8(dest[1] + dc);
		dest[2] = av_clip_uint8(dest[2] + dc);
		dest[3] = av_clip_uint8(dest[3] + dc);
		dest[4] = av_clip_uint8(dest[4] + dc);
		dest[5] = av_clip_uint8(dest[5] + dc);
		dest[6] = av_clip_uint8(dest[6] + dc);
		dest[7] = av_clip_uint8(dest[7] + dc);
		dest += stride;
	}
	block[0] = 0;
}

static void set_bounding_values(int* bounding_values_array, int filter_limit) {
	int* bounding_values = bounding_values_array + 127;
	memset(bounding_values_array, 0, 256 * sizeof(int));
	for (int x = 0; x < filter_limit; x++) {
		bounding_values[-x] = -x;
		bounding_values[x] = x;
	}
	int value;
	for (int x = value = filter_limit; x < 128 && value; x++, value--) {
		bounding_values[x] = value;
		bounding_values[-x] = -value;
	}
	if (value)
		bounding_values[128] = value;
}

static void vp3_v_loop_filter_12(uint8_t* first_pixel, ptrdiff_t stride, int* bounding_values) {
	const ptrdiff_t nstride = -stride;
	for (uint8_t* end = first_pixel + 12; first_pixel < end; first_pixel++) {
		int fv = (first_pixel[2 * nstride] - first_pixel[stride]) + (first_pixel[0] - first_pixel[nstride]) * 3;
		fv = bounding_values[(fv + 4) >> 3];
		first_pixel[nstride] = av_clip_uint8(first_pixel[nstride] + fv);
		first_pixel[0] = av_clip_uint8(first_pixel[0] - fv);
	}
}
static void vp3_h_loop_filter_12(uint8_t* first_pixel, ptrdiff_t stride, int* bounding_values) {
	for (uint8_t* end = first_pixel + 12 * stride; first_pixel != end; first_pixel += stride) {
		int fv = (first_pixel[-2] - first_pixel[1]) + (first_pixel[0] - first_pixel[-1]) * 3;
		fv = bounding_values[(fv + 4) >> 3];
		first_pixel[-1] = av_clip_uint8(first_pixel[-1] + fv);
		first_pixel[0] = av_clip_uint8(first_pixel[0] - fv);
	}
}

// ─── sub-pel filters ──────────────────────────────────────────────────────────
// h264_chroma_mc8 / op_put (h264chroma_template.c)
static void h264_chroma_mc8(uint8_t* dst, const uint8_t* src, ptrdiff_t stride, int h, int x, int y) {
	const int A = (8 - x) * (8 - y);
	const int B = x * (8 - y);
	const int C = (8 - x) * y;
	const int D = x * y;
	if (D) {
		for (int i = 0; i < h; i++) {
			for (int j = 0; j < 8; j++)
				dst[j] = (A * src[j] + B * src[j + 1] + C * src[stride + j] + D * src[stride + j + 1] + 32) >> 6;
			dst += stride; src += stride;
		}
	} else if (B + C) {
		const int E = B + C;
		const ptrdiff_t step = C ? stride : 1;
		for (int i = 0; i < h; i++) {
			for (int j = 0; j < 8; j++)
				dst[j] = (A * src[j] + E * src[step + j] + 32) >> 6;
			dst += stride; src += stride;
		}
	} else {
		for (int i = 0; i < h; i++) {
			for (int j = 0; j < 8; j++)
				dst[j] = (A * src[j] + 32) >> 6;
			dst += stride; src += stride;
		}
	}
}

// vp6_filter_diag4_c (vp6dsp.c)
static void vp6_filter_diag4(uint8_t* dst, const uint8_t* src, ptrdiff_t stride,
                             const int16_t* h_weights, const int16_t* v_weights) {
	int tmp[8 * 11];
	int* t = tmp;
	src -= stride;
	for (int y = 0; y < 11; y++) {
		for (int x = 0; x < 8; x++)
			t[x] = av_clip_uint8((src[x - 1] * h_weights[0] + src[x] * h_weights[1]
			                    + src[x + 1] * h_weights[2] + src[x + 2] * h_weights[3] + 64) >> 7);
		src += stride; t += 8;
	}
	t = tmp + 8;
	for (int y = 0; y < 8; y++) {
		for (int x = 0; x < 8; x++)
			dst[x] = av_clip_uint8((t[x - 8] * v_weights[0] + t[x] * v_weights[1]
			                      + t[x + 8] * v_weights[2] + t[x + 16] * v_weights[3] + 64) >> 7);
		dst += stride; t += 8;
	}
}

// emulated_edge_mc — clamp-from-base form (== FFmpeg's edge replication).
static void emulated_edge_mc(uint8_t* buf, int bufLinesize, const uint8_t* plane, int planeStride,
                             int block_w, int block_h, int src_x, int src_y, int w, int h) {
	for (int oy = 0; oy < block_h; oy++) {
		int sy = src_y + oy; if (sy < 0) sy = 0; else if (sy > h - 1) sy = h - 1;
		uint8_t* bp = buf + oy * bufLinesize;
		const uint8_t* sp = plane + sy * planeStride;
		for (int ox = 0; ox < block_w; ox++) {
			int sx = src_x + ox; if (sx < 0) sx = 0; else if (sx > w - 1) sx = w - 1;
			bp[ox] = sp[sx];
		}
	}
}

// plain copy (hpel put_pixels_tab[*][0])
static void copy_block(uint8_t* dst, int dstStride, const uint8_t* src, int srcStride, int w, int h) {
	for (int r = 0; r < h; r++) {
		memcpy(dst, src, (size_t)w);
		dst += dstStride; src += srcStride;
	}
}

static int vp6_block_variance(const uint8_t* src, ptrdiff_t stride) {
	int sum = 0, square_sum = 0;
	for (int y = 0; y < 8; y += 2) {
		for (int x = 0; x < 8; x += 2) { sum += src[x]; square_sum += src[x] * src[x]; }
		src += 2 * stride;
	}
	return (16 * square_sum - sum * sum) >> 8;
}

static void vp6_filter_hv4(uint8_t* dst, const uint8_t* src, ptrdiff_t stride, int delta, const int16_t* w) {
	for (int y = 0; y < 8; y++) {
		for (int x = 0; x < 8; x++)
			dst[x] = av_clip_uint8((src[x - delta] * w[0] + src[x] * w[1]
			                      + src[x + delta] * w[2] + src[x + 2 * delta] * w[3] + 64) >> 7);
		src += stride; dst += stride;
	}
}

// ─── decoder handle ───────────────────────────────────────────────────────────
struct VP56RefDc { uint8_t not_null_dc; int8_t ref_frame; int16_t dc_coeff; };
struct VP56mv { int16_t x, y; };
struct VP56Macroblock { uint8_t type; VP56mv mv; };
struct PlaneBuf { uint8_t* y; uint8_t* u; uint8_t* v; };

struct VP6Ctx {
	int has_alpha;
	VP6Ctx* alpha;          // nested independent decoder (alpha sub-stream)

	int sub_version;
	int filter_header;
	int interlaced;
	int deblock_filtering;
	int filter_mode;
	int filter_selection;
	int max_vector_length;
	int sample_variance_threshold;
	int golden_frame;
	int use_huffman;
	int cur_key_frame;

	int quantizer;
	int dequant_dc;
	int dequant_ac;

	int codedWidth, codedHeight;
	int mb_width, mb_height;
	int plane_width[3], plane_height[3];
	ptrdiff_t stride[3];

	int bounding_values_array[256];
	uint8_t idct_scantable[64];
	const uint8_t* def_coeff_reorder;

	VP56Model model;

	RangeCoder c, cc;
	RangeCoder* ccp;
	GetBits gb;
	FlatVLC dccv_vlc[2];
	FlatVLC runv_vlc[2];
	FlatVLC ract_vlc[2][3][4];
	unsigned nb_null[2][2];

	VP56RefDc* above_blocks;
	int above_blocks_len;
	VP56RefDc left_block[4];
	int above_block_idx[6];
	int16_t prev_dc[3][3];

	uint8_t mb_type;
	VP56Macroblock* macroblocks;
	int16_t block_coeff[6][64];
	int idct_selector[6];
	int block_offset[6];

	VP56mv mv[6];
	VP56mv vector_candidate[2];
	int vector_candidate_pos;

	int allocated;
	PlaneBuf pool[3];
	PlaneBuf* bufCur;
	PlaneBuf* bufPrev;
	PlaneBuf* bufGolden;
	PlaneBuf* lastDecoded;
	uint8_t* edge_emu;
	int ySize, cSize;
};

static uint8_t* plane_of(PlaneBuf* b, int plane) { return plane == 0 ? b->y : (plane == 1 ? b->u : b->v); }

static void init_dequant(VP6Ctx* s, int quantizer) {
	if (s->quantizer != quantizer)
		set_bounding_values(s->bounding_values_array, ff_vp56_filter_threshold[quantizer]);
	s->quantizer = quantizer;
	s->dequant_dc = ff_vp56_dc_dequant[quantizer] << 2;
	s->dequant_ac = ff_vp56_ac_dequant[quantizer] << 2;
}

static void free_planes(VP6Ctx* s) {
	for (int i = 0; i < 3; i++) {
		free(s->pool[i].y); free(s->pool[i].u); free(s->pool[i].v);
		s->pool[i].y = s->pool[i].u = s->pool[i].v = NULL;
	}
	free(s->above_blocks); s->above_blocks = NULL;
	free(s->macroblocks); s->macroblocks = NULL;
	free(s->edge_emu); s->edge_emu = NULL;
}

static void alloc_planes(VP6Ctx* s) {
	const int cw = s->codedWidth, ch = s->codedHeight;
	s->stride[0] = cw;
	s->stride[1] = s->stride[2] = cw >> 1;
	s->plane_width[0] = cw; s->plane_width[1] = s->plane_width[2] = cw >> 1;
	s->plane_height[0] = ch; s->plane_height[1] = s->plane_height[2] = ch >> 1;

	// trailing padding so the 16-wide deblock copy can over-read a few cols/rows
	const int ySize = cw * ch + cw * 16 + 64;
	const int cSize = (cw >> 1) * (ch >> 1) + (cw >> 1) * 16 + 64;
	s->ySize = ySize; s->cSize = cSize;

	free_planes(s);
	for (int i = 0; i < 3; i++) {
		s->pool[i].y = (uint8_t*)calloc(1, ySize);
		s->pool[i].u = (uint8_t*)calloc(1, cSize);
		s->pool[i].v = (uint8_t*)calloc(1, cSize);
	}
	s->above_blocks_len = 4 * s->mb_width + 6;
	s->above_blocks = (VP56RefDc*)calloc(s->above_blocks_len, sizeof(VP56RefDc));
	s->macroblocks = (VP56Macroblock*)calloc(s->mb_width * s->mb_height, sizeof(VP56Macroblock));
	s->edge_emu = (uint8_t*)calloc(1, 16 * cw + 64);

	s->bufCur = s->bufPrev = s->bufGolden = s->lastDecoded = NULL;
	s->allocated = 1;
}

static PlaneBuf* pick_free_buffer(VP6Ctx* s) {
	for (int i = 0; i < 3; i++)
		if (&s->pool[i] != s->bufPrev && &s->pool[i] != s->bufGolden)
			return &s->pool[i];
	return &s->pool[0];
}

// ─── model parsing ────────────────────────────────────────────────────────────
static void coeff_order_table_init(VP6Ctx* s) {
	VP56Model* m = &s->model;
	m->coeff_index_to_pos[0] = 0;
	int idx = 1;
	for (int i = 0; i < 16; i++)
		for (int pos = 1; pos < 64; pos++)
			if (m->coeff_reorder[pos] == i)
				m->coeff_index_to_pos[idx++] = pos;
	for (int k = 0; k < 64; k++) {
		int max = 0;
		for (int i = 0; i <= k; i++) { int v = m->coeff_index_to_pos[i]; if (v > max) max = v; }
		if (s->sub_version > 6) max++;
		m->coeff_index_to_idct_selector[k] = (uint8_t)max;
	}
}

static void default_models_init(VP6Ctx* s) {
	VP56Model* m = &s->model;
	m->vector_dct[0] = 0xA2;
	m->vector_dct[1] = 0xA4;
	m->vector_sig[0] = 0x80;
	m->vector_sig[1] = 0x80;
	memcpy(m->mb_types_stats, ff_vp56_def_mb_types_stats, sizeof(m->mb_types_stats));
	memcpy(m->vector_fdv, vp6_def_fdv_vector_model, sizeof(m->vector_fdv));
	memcpy(m->vector_pdv, vp6_def_pdv_vector_model, sizeof(m->vector_pdv));
	memcpy(m->coeff_runv, vp6_def_runv_coeff_model, sizeof(m->coeff_runv));
	memcpy(m->coeff_reorder, s->def_coeff_reorder, sizeof(m->coeff_reorder));
	coeff_order_table_init(s);
}

static void parse_vector_models(VP6Ctx* s) {
	RangeCoder* c = &s->c;
	VP56Model* m = &s->model;
	for (int comp = 0; comp < 2; comp++) {
		if (rac_get_prob_branchy(c, vp6_sig_dct_pct[comp][0])) m->vector_dct[comp] = rac_gets_nn(c, 7);
		if (rac_get_prob_branchy(c, vp6_sig_dct_pct[comp][1])) m->vector_sig[comp] = rac_gets_nn(c, 7);
	}
	for (int comp = 0; comp < 2; comp++)
		for (int node = 0; node < 7; node++)
			if (rac_get_prob_branchy(c, vp6_pdv_pct[comp][node])) m->vector_pdv[comp][node] = rac_gets_nn(c, 7);
	for (int comp = 0; comp < 2; comp++)
		for (int node = 0; node < 8; node++)
			if (rac_get_prob_branchy(c, vp6_fdv_pct[comp][node])) m->vector_fdv[comp][node] = rac_gets_nn(c, 7);
}

static int parse_coeff_models(VP6Ctx* s) {
	RangeCoder* c = &s->c;
	VP56Model* m = &s->model;
	int def_prob[11];
	memset(def_prob, 0x80, sizeof(def_prob));
	const int keyFrame = s->cur_key_frame;

	for (int pt = 0; pt < 2; pt++)
		for (int node = 0; node < 11; node++) {
			if (rac_get_prob_branchy(c, vp6_dccv_pct[pt][node])) {
				def_prob[node] = rac_gets_nn(c, 7);
				m->coeff_dccv[pt][node] = def_prob[node];
			} else if (keyFrame) {
				m->coeff_dccv[pt][node] = def_prob[node];
			}
		}

	if (rac_get(c)) {
		for (int pos = 1; pos < 64; pos++)
			if (rac_get_prob_branchy(c, vp6_coeff_reorder_pct[pos]))
				m->coeff_reorder[pos] = rac_gets(c, 4);
		coeff_order_table_init(s);
	}

	for (int cg = 0; cg < 2; cg++)
		for (int node = 0; node < 14; node++)
			if (rac_get_prob_branchy(c, vp6_runv_pct[cg][node]))
				m->coeff_runv[cg][node] = rac_gets_nn(c, 7);

	for (int ct = 0; ct < 3; ct++)
		for (int pt = 0; pt < 2; pt++)
			for (int cg = 0; cg < 6; cg++)
				for (int node = 0; node < 11; node++) {
					if (rac_get_prob_branchy(c, vp6_ract_pct[ct][pt][cg][node])) {
						def_prob[node] = rac_gets_nn(c, 7);
						m->coeff_ract[pt][ct][cg][node] = def_prob[node];
					} else if (keyFrame) {
						m->coeff_ract[pt][ct][cg][node] = def_prob[node];
					}
				}

	if (s->use_huffman) {
		for (int pt = 0; pt < 2; pt++) {
			vp6_build_huff_tree(&s->dccv_vlc[pt], m->coeff_dccv[pt], vp6_huff_coeff_map, 12);
			vp6_build_huff_tree(&s->runv_vlc[pt], m->coeff_runv[pt], vp6_huff_run_map, 9);
			for (int ct = 0; ct < 3; ct++)
				for (int cg = 0; cg < 4; cg++)
					vp6_build_huff_tree(&s->ract_vlc[pt][ct][cg], m->coeff_ract[pt][ct][cg], vp6_huff_coeff_map, 12);
		}
		memset(s->nb_null, 0, sizeof(s->nb_null));
		return g_err;
	}

	// coeff_dcct is a linear combination of coeff_dccv (arithmetic path only)
	for (int pt = 0; pt < 2; pt++)
		for (int ctx = 0; ctx < 3; ctx++)
			for (int node = 0; node < 5; node++) {
				int v = ((m->coeff_dccv[pt][node] * vp6_dccv_lc[ctx][node][0] + 128) >> 8) + vp6_dccv_lc[ctx][node][1];
				if (v < 1) v = 1; else if (v > 255) v = 255;
				m->coeff_dcct[pt][ctx][node] = (uint8_t)v;
			}
	return 0;
}

static void parse_mb_type_models(VP6Ctx* s) {
	RangeCoder* c = &s->c;
	VP56Model* model = &s->model;
	for (int ctx = 0; ctx < 3; ctx++) {
		if (rac_get_prob_branchy(c, 174)) {
			int idx = rac_gets(c, 4);
			memcpy(model->mb_types_stats[ctx], ff_vp56_pre_def_mb_type_stats[idx][ctx],
			       sizeof(model->mb_types_stats[ctx]));
		}
		if (rac_get_prob_branchy(c, 254)) {
			for (int type = 0; type < 10; type++)
				for (int i = 0; i < 2; i++)
					if (rac_get_prob_branchy(c, 205)) {
						int sign = rac_get(c);
						int delta = rac_get_tree(c, ff_vp56_pmbtm_tree, ff_vp56_mb_type_model_model);
						if (!delta) delta = 4 * rac_gets(c, 7);
						model->mb_types_stats[ctx][type][i] += (delta ^ -sign) + sign;
					}
		}
	}
	for (int ctx = 0; ctx < 3; ctx++) {
		int p[10];
		for (int type = 0; type < 10; type++) p[type] = 100 * model->mb_types_stats[ctx][type][1];
		for (int type = 0; type < 10; type++) {
			model->mb_type[ctx][type][0] = 255 - (255 * model->mb_types_stats[ctx][type][0])
				/ (1 + model->mb_types_stats[ctx][type][0] + model->mb_types_stats[ctx][type][1]);
			p[type] = 0;
			int p02 = p[0] + p[2], p34 = p[3] + p[4], p0234 = p02 + p34;
			int p17 = p[1] + p[7], p56 = p[5] + p[6], p89 = p[8] + p[9];
			int p5689 = p56 + p89, p156789 = p17 + p5689;
			model->mb_type[ctx][type][1] = 1 + 255 * p0234 / (1 + p0234 + p156789);
			model->mb_type[ctx][type][2] = 1 + 255 * p02  / (1 + p0234);
			model->mb_type[ctx][type][3] = 1 + 255 * p17  / (1 + p156789);
			model->mb_type[ctx][type][4] = 1 + 255 * p[0] / (1 + p02);
			model->mb_type[ctx][type][5] = 1 + 255 * p[3] / (1 + p34);
			model->mb_type[ctx][type][6] = 1 + 255 * p[1] / (1 + p17);
			model->mb_type[ctx][type][7] = 1 + 255 * p56  / (1 + p5689);
			model->mb_type[ctx][type][8] = 1 + 255 * p[5] / (1 + p56);
			model->mb_type[ctx][type][9] = 1 + 255 * p[8] / (1 + p89);
			p[type] = 100 * model->mb_types_stats[ctx][type][1];
		}
	}
}

static int parse_mb_type(VP6Ctx* s, int prev_type, int ctx) {
	uint8_t* mb_type_model = s->model.mb_type[ctx][prev_type];
	RangeCoder* c = &s->c;
	if (rac_get_prob_branchy(c, mb_type_model[0]))
		return prev_type;
	return rac_get_tree(c, ff_vp56_pmbt_tree, mb_type_model);
}

static void parse_vector_adjustment(VP6Ctx* s, VP56mv* vect) {
	RangeCoder* c = &s->c;
	VP56Model* model = &s->model;
	static const uint8_t prob_order[7] = { 0, 1, 2, 7, 6, 5, 4 };
	vect->x = 0; vect->y = 0;
	if (s->vector_candidate_pos < 2)
		*vect = s->vector_candidate[0];
	for (int comp = 0; comp < 2; comp++) {
		int delta = 0;
		if (rac_get_prob_branchy(c, model->vector_dct[comp])) {
			for (int i = 0; i < 7; i++) {
				int j = prob_order[i];
				delta |= rac_get_prob(c, model->vector_fdv[comp][j]) << j;
			}
			if (delta & 0xF0) delta |= rac_get_prob(c, model->vector_fdv[comp][3]) << 3;
			else delta |= 8;
		} else {
			delta = rac_get_tree(c, ff_vp56_pva_tree, model->vector_pdv[comp]);
		}
		if (delta && rac_get_prob_branchy(c, model->vector_sig[comp])) delta = -delta;
		if (!comp) vect->x += delta; else vect->y += delta;
	}
}

static int get_vectors_predictors(VP6Ctx* s, int row, int col, int ref_frame) {
	int nb_pred = 0;
	VP56mv vect[2] = { {0,0}, {0,0} };
	for (int pos = 0; pos < 12; pos++) {
		int mx = col + ff_vp56_candidate_predictor_pos[pos][0];
		int my = row + ff_vp56_candidate_predictor_pos[pos][1];
		if (mx < 0 || mx >= s->mb_width || my < 0 || my >= s->mb_height) continue;
		int offset = mx + s->mb_width * my;
		if (ff_vp56_reference_frame[s->macroblocks[offset].type] != ref_frame) continue;
		VP56mv mbmv = s->macroblocks[offset].mv;
		if ((mbmv.x == vect[0].x && mbmv.y == vect[0].y) || (mbmv.x == 0 && mbmv.y == 0)) continue;
		vect[nb_pred++] = mbmv;
		if (nb_pred > 1) { nb_pred = -1; break; }
		s->vector_candidate_pos = pos;
	}
	s->vector_candidate[0] = vect[0];
	s->vector_candidate[1] = vect[1];
	return nb_pred + 1;
}

static void decode_4mv(VP6Ctx* s, int row, int col) {
	VP56mv mv = { 0, 0 };
	int type[4];
	for (int b = 0; b < 4; b++) {
		type[b] = rac_gets(&s->c, 2);
		if (type[b]) type[b]++;
	}
	for (int b = 0; b < 4; b++) {
		switch (type[b]) {
			case MB_NOVEC_PF: s->mv[b].x = 0; s->mv[b].y = 0; break;
			case MB_DELTA_PF: parse_vector_adjustment(s, &s->mv[b]); break;
			case MB_V1_PF: s->mv[b] = s->vector_candidate[0]; break;
			case MB_V2_PF: s->mv[b] = s->vector_candidate[1]; break;
		}
		mv.x += s->mv[b].x;
		mv.y += s->mv[b].y;
	}
	s->macroblocks[row * s->mb_width + col].mv = s->mv[3];
	s->mv[4].x = s->mv[5].x = RSHIFT(mv.x, 2);
	s->mv[4].y = s->mv[5].y = RSHIFT(mv.y, 2);
}

static int decode_mv(VP6Ctx* s, int row, int col) {
	VP56mv* mv;
	VP56mv vect = { 0, 0 };
	int ctx = get_vectors_predictors(s, row, col, FRAME_PREVIOUS);
	s->mb_type = parse_mb_type(s, s->mb_type, ctx);
	s->macroblocks[row * s->mb_width + col].type = s->mb_type;
	switch (s->mb_type) {
		case MB_V1_PF: mv = &s->vector_candidate[0]; break;
		case MB_V2_PF: mv = &s->vector_candidate[1]; break;
		case MB_V1_GF: get_vectors_predictors(s, row, col, FRAME_GOLDEN); mv = &s->vector_candidate[0]; break;
		case MB_V2_GF: get_vectors_predictors(s, row, col, FRAME_GOLDEN); mv = &s->vector_candidate[1]; break;
		case MB_DELTA_PF: parse_vector_adjustment(s, &vect); mv = &vect; break;
		case MB_DELTA_GF: get_vectors_predictors(s, row, col, FRAME_GOLDEN); parse_vector_adjustment(s, &vect); mv = &vect; break;
		case MB_4V: decode_4mv(s, row, col); return s->mb_type;
		default: mv = &vect; break;
	}
	s->macroblocks[row * s->mb_width + col].mv = *mv;
	for (int b = 0; b < 6; b++) s->mv[b] = *mv;
	return s->mb_type;
}

static void add_predictors_dc(VP6Ctx* s, int ref_frame) {
	int idx = s->idct_scantable[0];
	for (int b = 0; b < 6; b++) {
		VP56RefDc* ab = &s->above_blocks[s->above_block_idx[b]];
		VP56RefDc* lb = &s->left_block[ff_vp56_b6to4[b]];
		int count = 0, dc = 0;
		if (ref_frame == lb->ref_frame) { dc += lb->dc_coeff; count++; }
		if (ref_frame == ab->ref_frame) { dc += ab->dc_coeff; count++; }
		if (count == 0) dc = s->prev_dc[ff_vp56_b2p[b]][ref_frame];
		else if (count == 2) dc /= 2;
		s->block_coeff[b][idx] += dc;
		s->prev_dc[ff_vp56_b2p[b]][ref_frame] = s->block_coeff[b][idx];
		ab->dc_coeff = s->block_coeff[b][idx];
		ab->ref_frame = ref_frame;
		lb->dc_coeff = s->block_coeff[b][idx];
		lb->ref_frame = ref_frame;
		s->block_coeff[b][idx] *= s->dequant_dc;
	}
}

// ─── coefficient parsing ──────────────────────────────────────────────────────
static unsigned vp6_get_nb_null(VP6Ctx* s) {
	unsigned val = gb_get(&s->gb, 2);
	if (val == 2) val += gb_get(&s->gb, 2);
	else if (val == 3) {
		val = gb_get1(&s->gb) << 2;
		val = 6 + val + gb_get(&s->gb, 2 + val);
	}
	return val;
}

static int parse_coeff_huffman(VP6Ctx* s) {
	VP56Model* model = &s->model;
	const uint8_t* permute = s->idct_scantable;
	int pt = 0;
	for (int b = 0; b < 6; b++) {
		int ct = 0;
		if (b > 3) pt = 1;
		FlatVLC* vlc_coeff = &s->dccv_vlc[pt];
		int coeff_idx = 0;
		for (;;) {
			int run = 1;
			if (coeff_idx < 2 && s->nb_null[coeff_idx][pt]) {
				s->nb_null[coeff_idx][pt]--;
				if (coeff_idx) break;
			} else {
				if (gb_left(&s->gb) <= 0) { g_err = 1; return -1; }
				int coeff = get_vlc(&s->gb, vlc_coeff);
				if (g_err) return -1;
				if (coeff == 0) {
					if (coeff_idx) {
						int rpt = (coeff_idx >= 6);
						run += get_vlc(&s->gb, &s->runv_vlc[rpt]);
						if (g_err) return -1;
						if (run >= 9) run += gb_get(&s->gb, 6);
					} else {
						s->nb_null[0][pt] = vp6_get_nb_null(s);
					}
					ct = 0;
				} else if (coeff == 11) {
					if (coeff_idx == 1) s->nb_null[1][pt] = vp6_get_nb_null(s);
					break;
				} else {
					int coeff2 = ff_vp56_coeff_bias[coeff];
					if (coeff > 4) coeff2 += gb_get(&s->gb, coeff <= 9 ? coeff - 4 : 11);
					ct = 1 + (coeff2 > 1);
					int sign = gb_get1(&s->gb);
					coeff2 = (coeff2 ^ -sign) + sign;
					if (coeff_idx) coeff2 *= s->dequant_ac;
					int idx = model->coeff_index_to_pos[coeff_idx];
					s->block_coeff[b][permute[idx]] = (int16_t)coeff2;
				}
			}
			coeff_idx += run;
			if (coeff_idx >= 64) break;
			int cg = FFMIN(vp6_coeff_groups[coeff_idx], 3);
			vlc_coeff = &s->ract_vlc[pt][ct][cg];
		}
		s->idct_selector[b] = model->coeff_index_to_idct_selector[FFMIN(coeff_idx, 63)];
	}
	return 0;
}

static int parse_coeff(VP6Ctx* s) {
	if (s->use_huffman) return parse_coeff_huffman(s);
	RangeCoder* c = s->ccp;
	VP56Model* model = &s->model;
	const uint8_t* permute = s->idct_scantable;
	if (rac_is_end(c)) { g_err = 1; return -1; }

	int pt = 0;
	for (int b = 0; b < 6; b++) {
		int ct = 1, run = 1;
		if (b > 3) pt = 1;
		int ctx = s->left_block[ff_vp56_b6to4[b]].not_null_dc
		        + s->above_blocks[s->above_block_idx[b]].not_null_dc;
		uint8_t* model1 = model->coeff_dccv[pt];
		uint8_t* model2 = model->coeff_dcct[pt][ctx];

		int coeff_idx = 0;
		for (;;) {
			int coeff;
			if ((coeff_idx > 1 && ct == 0) || rac_get_prob_branchy(c, model2[0])) {
				if (rac_get_prob_branchy(c, model2[2])) {
					if (rac_get_prob_branchy(c, model2[3])) {
						int t = rac_get_tree(c, ff_vp56_pc_tree, model1);
						coeff = ff_vp56_coeff_bias[t + 5];
						for (int i = ff_vp56_coeff_bit_length[t]; i >= 0; i--)
							coeff += rac_get_prob(c, ff_vp56_coeff_parse_table[t][i]) << i;
					} else {
						if (rac_get_prob_branchy(c, model2[4])) coeff = 3 + rac_get_prob(c, model1[5]);
						else coeff = 2;
					}
					ct = 2;
				} else {
					ct = 1; coeff = 1;
				}
				int sign = rac_get(c);
				coeff = (coeff ^ -sign) + sign;
				if (coeff_idx) coeff *= s->dequant_ac;
				int idx = model->coeff_index_to_pos[coeff_idx];
				s->block_coeff[b][permute[idx]] = (int16_t)coeff;
				run = 1;
			} else {
				ct = 0;
				if (coeff_idx > 0) {
					if (!rac_get_prob_branchy(c, model2[1])) break;
					uint8_t* model3 = model->coeff_runv[coeff_idx >= 6];
					run = rac_get_tree(c, vp6_pcr_tree, model3);
					if (!run) {
						run = 9;
						for (int i = 0; i < 6; i++)
							run += rac_get_prob(c, model3[i + 8]) << i;
					}
				}
			}
			coeff_idx += run;
			if (coeff_idx >= 64) break;
			int cg = vp6_coeff_groups[coeff_idx];
			model1 = model2 = model->coeff_ract[pt][ct][cg];
		}

		s->left_block[ff_vp56_b6to4[b]].not_null_dc =
		s->above_blocks[s->above_block_idx[b]].not_null_dc = !!s->block_coeff[b][0];
		s->idct_selector[b] = model->coeff_index_to_idct_selector[FFMIN(coeff_idx, 63)];
	}
	return 0;
}

// ─── motion compensation / render ─────────────────────────────────────────────
static void deblock_filter(VP6Ctx* s, uint8_t* yuv, ptrdiff_t stride, int dx, int dy) {
	int* bounding_values = s->bounding_values_array + 127;
	if (dx) vp3_h_loop_filter_12(yuv + 10 - dx, stride, bounding_values);
	if (dy) vp3_v_loop_filter_12(yuv + stride * (10 - dy), stride, bounding_values);
}

static void vp6_filter(VP6Ctx* s, uint8_t* dst, uint8_t* src, int offset1, int offset2,
                       ptrdiff_t stride, VP56mv mv, int mask, int select, int luma) {
	int filter4 = 0;
	int x8 = mv.x & mask;
	int y8 = mv.y & mask;
	if (luma) {
		x8 *= 2; y8 *= 2;
		filter4 = s->filter_mode;
		if (filter4 == 2) {
			if (s->max_vector_length && (FFABS(mv.x) > s->max_vector_length || FFABS(mv.y) > s->max_vector_length)) {
				filter4 = 0;
			} else if (s->sample_variance_threshold
			           && vp6_block_variance(src + offset1, stride) < s->sample_variance_threshold) {
				filter4 = 0;
			}
		}
	}
	// (offset2-offset1)*flip<0 ; flip is always +1 here
	if ((y8 && (offset2 - offset1) < 0) || (!y8 && offset1 > offset2)) offset1 = offset2;

	if (filter4) {
		if (!y8) vp6_filter_hv4(dst, src + offset1, stride, 1, vp6_block_copy_filter[select][x8]);
		else if (!x8) vp6_filter_hv4(dst, src + offset1, stride, stride, vp6_block_copy_filter[select][y8]);
		else vp6_filter_diag4(dst, src + offset1 + ((mv.x ^ mv.y) >> 31), stride,
		                      vp6_block_copy_filter[select][x8], vp6_block_copy_filter[select][y8]);
	} else {
		if (!x8 || !y8) {
			h264_chroma_mc8(dst, src + offset1, stride, 8, x8, y8);
		} else {
			uint8_t* tmp = s->edge_emu + 16;
			h264_chroma_mc8(tmp, src + offset1 + ((mv.x ^ mv.y) >> 31), stride, 9, x8, 0);
			h264_chroma_mc8(dst, tmp, stride, 8, 0, y8);
		}
	}
}

static void vp56_mc(VP6Ctx* s, int b, int plane, uint8_t* src, ptrdiff_t stride, int x, int y) {
	uint8_t* dst = plane_of(s->bufCur, plane) + s->block_offset[b];
	int mask = vp6_coord_div[b] - 1;
	int deblock_filtering = s->deblock_filtering;
	int dx = s->mv[b].x / vp6_coord_div[b];
	int dy = s->mv[b].y / vp6_coord_div[b];

	if (b >= 4) { x /= 2; y /= 2; }
	x += dx - 2;
	y += dy - 2;

	uint8_t* src_block;
	int src_offset;
	const int pw = s->plane_width[plane], ph = s->plane_height[plane];
	const int blockOff = s->block_offset[b];

	if (x < 0 || x + 12 >= pw || y < 0 || y + 12 >= ph) {
		emulated_edge_mc(s->edge_emu, stride, src, stride, 12, 12, x, y, pw, ph);
		src_block = s->edge_emu;
		src_offset = 2 + 2 * stride;
	} else if (deblock_filtering) {
		copy_block(s->edge_emu, stride, src + blockOff + (dy - 2) * stride + (dx - 2), stride, 16, 12);
		src_block = s->edge_emu;
		src_offset = 2 + 2 * stride;
	} else {
		src_block = src;
		src_offset = blockOff + dy * stride + dx;
	}

	if (deblock_filtering)
		deblock_filter(s, src_block, stride, dx & 7, dy & 7);

	int overlap_offset = 0;
	if (s->mv[b].x & mask) overlap_offset += (s->mv[b].x > 0) ? 1 : -1;
	if (s->mv[b].y & mask) overlap_offset += (s->mv[b].y > 0) ? stride : -stride;

	if (overlap_offset) {
		vp6_filter(s, dst, src_block, src_offset, src_offset + overlap_offset, stride,
		           s->mv[b], mask, s->filter_selection, b < 4);
	} else {
		copy_block(dst, stride, src_block + src_offset, stride, 8, 8);
	}
}

static void idct_put_sel(uint8_t* dest, ptrdiff_t stride, int16_t* block, int selector) {
	if (selector > 10 || selector == 1) idct_put_full(dest, stride, block);
	else idct10_put(dest, stride, block);
}
static void idct_add_sel(uint8_t* dest, ptrdiff_t stride, int16_t* block, int selector) {
	if (selector > 10) idct_add_full(dest, stride, block);
	else if (selector > 1) idct10_add(dest, stride, block);
	else idct_dc_add(dest, stride, block);
}

static void render_mb(VP6Ctx* s, int row, int col, int mb_type) {
	int ref_frame = ff_vp56_reference_frame[mb_type];
	add_predictors_dc(s, ref_frame);

	PlaneBuf* cur = s->bufCur;
	PlaneBuf* ref = ref_frame == FRAME_PREVIOUS ? s->bufPrev : (ref_frame == FRAME_GOLDEN ? s->bufGolden : cur);
	if (mb_type != MB_INTRA && !ref) return;

	if (mb_type == MB_INTRA) {
		for (int b = 0; b < 6; b++) {
			int plane = ff_vp56_b2p[b];
			idct_put_sel(plane_of(cur, plane) + s->block_offset[b], s->stride[plane], s->block_coeff[b], s->idct_selector[b]);
		}
		return;
	}
	if (mb_type == MB_NOVEC_PF || mb_type == MB_NOVEC_GF) {
		for (int b = 0; b < 6; b++) {
			int plane = ff_vp56_b2p[b];
			int off = s->block_offset[b];
			copy_block(plane_of(cur, plane) + off, s->stride[plane], plane_of(ref, plane) + off, s->stride[plane], 8, 8);
			idct_add_sel(plane_of(cur, plane) + off, s->stride[plane], s->block_coeff[b], s->idct_selector[b]);
		}
		return;
	}
	for (int b = 0; b < 6; b++) {
		int plane = ff_vp56_b2p[b];
		int x_off = (b == 1 || b == 3) ? 8 : 0;
		int y_off = (b == 2 || b == 3) ? 8 : 0;
		vp56_mc(s, b, plane, plane_of(ref, plane), s->stride[plane], 16 * col + x_off, 16 * row + y_off);
		idct_add_sel(plane_of(cur, plane) + s->block_offset[b], s->stride[plane], s->block_coeff[b], s->idct_selector[b]);
	}
}

// ─── header parse ─────────────────────────────────────────────────────────────
static void parse_header(VP6Ctx* s, const uint8_t* data, int p, int n) {
	int separated_coeff = data[p] & 1;
	int keyFrame = !(data[p] & 0x80);
	init_dequant(s, (data[p] >> 1) & 0x3F);

	int coeff_offset = 0, parse_filter_info = 0, vrt_shift = 0;
	int cols = s->mb_width, rows = s->mb_height;

	if (keyFrame) {
		int sub_version = data[p + 1] >> 3;
		if (sub_version > 8) { g_err = 1; return; }
		s->filter_header = data[p + 1] & 0x06;
		s->interlaced = data[p + 1] & 1;
		if (s->interlaced) { g_err = 1; return; }   // interlaced VP6 unsupported
		s->def_coeff_reorder = vp6_def_coeff_reorder;
		if (separated_coeff || !s->filter_header) {
			coeff_offset = AV_RB16(data + p + 2) - 2;
			p += 2; n -= 2;
		}
		rows = data[p + 2];
		cols = data[p + 3];
		if (!rows || !cols) { g_err = 1; return; }

		if (!s->allocated || 16 * cols != s->codedWidth || 16 * rows != s->codedHeight) {
			s->mb_width = cols; s->mb_height = rows;
			s->codedWidth = 16 * cols; s->codedHeight = 16 * rows;
			alloc_planes(s);
		}
		rangecoder_init(&s->c, data, p + 6, n - 6);
		rac_gets(&s->c, 2);
		parse_filter_info = s->filter_header;
		if (sub_version < 8) vrt_shift = 5;
		s->sub_version = sub_version;
		s->golden_frame = 0;
	} else {
		if (!s->sub_version || !s->codedWidth || !s->codedHeight) { g_err = 1; return; }
		if (separated_coeff || !s->filter_header) {
			coeff_offset = AV_RB16(data + p + 1) - 2;
			p += 2; n -= 2;
		}
		rangecoder_init(&s->c, data, p + 1, n - 1);
		s->golden_frame = rac_get(&s->c);
		if (s->filter_header) {
			s->deblock_filtering = rac_get(&s->c);
			if (s->deblock_filtering) rac_get(&s->c);
			if (s->sub_version > 7) parse_filter_info = rac_get(&s->c);
		}
	}

	if (parse_filter_info) {
		if (rac_get(&s->c)) {
			s->filter_mode = 2;
			s->sample_variance_threshold = rac_gets(&s->c, 5) << vrt_shift;
			s->max_vector_length = 2 << rac_gets(&s->c, 3);
		} else if (rac_get(&s->c)) {
			s->filter_mode = 1;
		} else {
			s->filter_mode = 0;
		}
		s->filter_selection = s->sub_version > 7 ? rac_gets(&s->c, 4) : 16;
	}

	s->use_huffman = rac_get(&s->c);

	if (coeff_offset) {
		p += coeff_offset; n -= coeff_offset;
		if (n < 0) { g_err = 1; return; }
		if (s->use_huffman) gb_init(&s->gb, data, p, n);
		else { rangecoder_init(&s->cc, data, p, n); s->ccp = &s->cc; }
	} else {
		s->ccp = &s->c;
	}
	s->cur_key_frame = keyFrame;
}

// ─── frame decode ─────────────────────────────────────────────────────────────
static void decode_stream(VP6Ctx* s, const uint8_t* data, int off, int len) {
	parse_header(s, data, off, len);
	if (g_err) return;

	const int keyFrame = s->cur_key_frame;
	PlaneBuf* cur = pick_free_buffer(s);
	s->bufCur = cur;

	if (keyFrame) {
		default_models_init(s);
		for (int i = 0; i < s->mb_width * s->mb_height; i++) s->macroblocks[i].type = MB_INTRA;
		if (parse_coeff_models(s)) { /* huff build error */ }
	} else {
		parse_mb_type_models(s);
		parse_vector_models(s);
		s->mb_type = MB_NOVEC_PF;
		parse_coeff_models(s);
	}
	if (g_err) return;

	memset(s->prev_dc, 0, sizeof(s->prev_dc));
	s->prev_dc[1][FRAME_CURRENT] = 128;
	s->prev_dc[2][FRAME_CURRENT] = 128;

	for (int i = 0; i < s->above_blocks_len; i++) {
		s->above_blocks[i].ref_frame = FRAME_NONE;
		s->above_blocks[i].dc_coeff = 0;
		s->above_blocks[i].not_null_dc = 0;
	}
	s->above_blocks[2 * s->mb_width + 2].ref_frame = FRAME_CURRENT;
	s->above_blocks[3 * s->mb_width + 4].ref_frame = FRAME_CURRENT;

	const ptrdiff_t stride_y = s->stride[0], stride_uv = s->stride[1];

	for (int mb_row = 0; mb_row < s->mb_height; mb_row++) {
		for (int block = 0; block < 4; block++) {
			s->left_block[block].ref_frame = FRAME_NONE;
			s->left_block[block].dc_coeff = 0;
			s->left_block[block].not_null_dc = 0;
		}
		s->above_block_idx[0] = 1;
		s->above_block_idx[1] = 2;
		s->above_block_idx[2] = 1;
		s->above_block_idx[3] = 2;
		s->above_block_idx[4] = 2 * s->mb_width + 2 + 1;
		s->above_block_idx[5] = 3 * s->mb_width + 4 + 1;

		s->block_offset[0] = mb_row * 16 * stride_y;
		s->block_offset[2] = s->block_offset[0] + 8 * stride_y;
		s->block_offset[1] = s->block_offset[0] + 8;
		s->block_offset[3] = s->block_offset[2] + 8;
		s->block_offset[4] = mb_row * 8 * stride_uv;
		s->block_offset[5] = s->block_offset[4];

		for (int mb_col = 0; mb_col < s->mb_width; mb_col++) {
			int mb_type = keyFrame ? MB_INTRA : decode_mv(s, mb_row, mb_col);
			parse_coeff(s);
			if (g_err) return;
			render_mb(s, mb_row, mb_col, mb_type);

			for (int y = 0; y < 4; y++) { s->above_block_idx[y] += 2; s->block_offset[y] += 16; }
			for (int uv = 4; uv < 6; uv++) { s->above_block_idx[uv] += 1; s->block_offset[uv] += 8; }
		}
	}

	if (keyFrame || s->golden_frame) s->bufGolden = cur;
	s->bufPrev = cur;
	s->lastDecoded = cur;
}

static bool g_inited = false;
static void ensure_init() {
	if (g_inited) return;
	g_inited = true;
}

static void ctx_init(VP6Ctx* s, int has_alpha) {
	memset(s, 0, sizeof(*s));
	s->has_alpha = has_alpha;
	s->quantizer = -1;
	s->deblock_filtering = 0;
	s->def_coeff_reorder = vp6_def_coeff_reorder;
	for (int i = 0; i < 64; i++) {
		int x = ff_zigzag_direct[i];
		s->idct_scantable[i] = (x >> 3) | ((x & 7) << 3);
	}
	s->ccp = &s->c;
}

// ─── extern "C" boundary ──────────────────────────────────────────────────────
extern "C" {

VP6Ctx* vp6_open(int has_alpha) {
	ensure_init();
	VP6Ctx* s = (VP6Ctx*)malloc(sizeof(VP6Ctx));
	if (!s) return NULL;
	ctx_init(s, has_alpha);
	if (has_alpha) {
		s->alpha = (VP6Ctx*)malloc(sizeof(VP6Ctx));
		ctx_init(s->alpha, 0);
	}
	return s;
}

void vp6_close(VP6Ctx* s) {
	if (!s) return;
	if (s->alpha) { free_planes(s->alpha); free(s->alpha); }
	free_planes(s);
	free(s);
}

void vp6_reset(VP6Ctx* s) {
	if (!s) return;
	s->bufPrev = s->bufGolden = NULL;
	if (s->alpha) { s->alpha->bufPrev = s->alpha->bufGolden = NULL; }
}

// Force the scalar (0) or SIMD (1) IDCT — A/B verification hook. No-op effect on
// non-SIMD builds (always scalar).
void vp6_set_simd(int on) { g_use_simd = on; }

// info[4] = { codedWidth, codedHeight, useHuffman, ok }
int vp6_decode_frame(VP6Ctx* s, const uint8_t* data, int pOff, int pLen,
                     int aOff, int aLen, int32_t* info) {
	g_err = 0;
	decode_stream(s, data, pOff, pLen);
	int ok = !g_err;
	if (ok && s->has_alpha && s->alpha) {
		decode_stream(s->alpha, data, aOff, aLen);
		ok = !g_err;
	}
	info[0] = s->codedWidth;
	info[1] = s->codedHeight;
	info[2] = s->use_huffman;
	info[3] = ok;
	return ok;
}

// SIMD-verification hook: run one IDCT variant on `block` into `dst`.
// mode: 0 = put (selector-routed), 1 = add (selector-routed), 2 = dc_add.
void vp6_test_idct(int16_t* block, uint8_t* dst, int stride, int mode, int selector) {
	if (mode == 0) idct_put_sel(dst, stride, block, selector);
	else if (mode == 1) idct_add_sel(dst, stride, block, selector);
	else idct_dc_add(dst, stride, block);
}

uint8_t* vp6_y_ptr(VP6Ctx* s) { return s->lastDecoded ? s->lastDecoded->y : NULL; }
uint8_t* vp6_u_ptr(VP6Ctx* s) { return s->lastDecoded ? s->lastDecoded->u : NULL; }
uint8_t* vp6_v_ptr(VP6Ctx* s) { return s->lastDecoded ? s->lastDecoded->v : NULL; }
uint8_t* vp6_alpha_y_ptr(VP6Ctx* s) {
	return (s->alpha && s->alpha->lastDecoded) ? s->alpha->lastDecoded->y : NULL;
}

} // extern "C"
