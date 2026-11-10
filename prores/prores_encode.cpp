/*
 * Apple ProRes (iCodec Pro) encoder -- 'apco' Proxy, 'apcs' LT, 'apcn' 422, 'apch' HQ,
 * 'ap4h' 4444, 'ap4x' 4444 XQ (with an alpha plane for the 4444 profiles).
 *
 * A port of FFmpeg's prores_ks encoder (libavcodec, release 8.1): proresenc_kostya.c,
 * proresenc_kostya_common.c, the proresdata.c tables, the 10-bit instance of
 * jfdctint_template.c (ff_jpeg_fdct_islow_10, which fdctdsp.c selects for 10-bit input)
 * and the big-endian writer of put_bits.h. Those files are identical in release 9.0, and the
 * output is byte-identical to ffmpeg 9.0.2's prores_ks (fileformats/tests/ProResEncode.test.js):
 *
 *   Copyright (c) 2011 Anatoliy Wasserman
 *   Copyright (c) 2012 Konstantin Shishkov
 *   Copyright (c) 1991-1996 Thomas G. Lane, Independent JPEG Group (jfdctint_template.c)
 *   Copyright (c) 2003-2011 Michael Niedermayer (jfdctint_template.c modifications)
 *
 * This file is part of a port of FFmpeg and is licensed under the GNU Lesser General Public
 * License, version 2.1 or later (LGPL-2.1+). Its corresponding source is published at
 * https://github.com/sigurdle/ffmpeg-codecs.
 *
 * FFmpeg is free software; you can redistribute it and/or modify it under the terms of the
 * GNU Lesser General Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version. FFmpeg is distributed
 * in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
 * warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser
 * General Public License for more details.
 *
 * The libavcodec framework (AVCodecContext / AVFrame / AVPacket / slice threads) is replaced
 * by a self-contained handle (ProresEncCtx) owning the input planes and the output frame.
 * Progressive frames only: the interlaced (two-field) path of prores_ks is not ported.
 * FFmpeg's defaults are kept: 8 macroblocks per slice, the profile's quant matrices
 * (quant_mat auto, written into the frame header), the profile's bits per macroblock by
 * frame size, the trellis quantizer search, 16-bit alpha. The vendor fourcc is a parameter
 * (FFmpeg's default is "Lavc"), and so are the header's color tags.
 *
 * Input planes are yuv422p10 / yuv444p10 / yuva444p10 exactly as prores_ks takes them, so
 * the output is byte-identical to `ffmpeg -c:v prores_ks` for the same planes. One
 * departure, opt-in: prores_enc_set_alpha16(h, 1) takes the alpha plane as full 16-bit
 * values instead of 10-bit ones expanded by (a << 6) | (a >> 4); prores_enc_from_rgba16
 * writes it that way, so RGBA16 alpha keeps its depth.
 *
 * prores_enc_from_rgba16 converts straight-alpha RGBA16 to the planes: limited range
 * (Y 64-940, C 64-960), the matrix the caller names, 4:2:2 chroma as the mean of each
 * horizontal pair. The inverse of prores_decode.cpp's prores_to_rgba16.
 */

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

// ---------------------------------------------------------------- tables (proresdata.c)

static const uint8_t prores_progressive_scan[64] = {
	 0,  1,  8,  9,  2,  3, 10, 11,
	16, 17, 24, 25, 18, 19, 26, 27,
	 4,  5, 12, 20, 13,  6,  7, 14,
	21, 28, 29, 22, 15, 23, 30, 31,
	32, 33, 40, 48, 41, 34, 35, 42,
	49, 56, 57, 50, 43, 36, 37, 44,
	51, 58, 59, 52, 45, 38, 39, 46,
	53, 60, 61, 54, 47, 55, 62, 63
};

#define FIRST_DC_CB 0xB8	// rice_order = 5, exp_golomb_order = 6, switch_bits = 0

static const uint8_t prores_dc_codebook[7] = { 0x04, 0x28, 0x28, 0x4D, 0x4D, 0x70, 0x70 };
static const uint8_t prores_run_to_cb[16] = { 0x06, 0x06, 0x05, 0x05, 0x04, 0x29, 0x29, 0x29, 0x29, 0x28, 0x28, 0x28, 0x28, 0x28, 0x28, 0x4C };
static const uint8_t prores_level_to_cb[10] = { 0x04, 0x0A, 0x05, 0x06, 0x04, 0x28, 0x28, 0x28, 0x28, 0x4C };

// ---------------------------------------------------------------- tables (proresenc_kostya_common.c)

#define CFACTOR_Y422 2
#define CFACTOR_Y444 3
#define MAX_MBS_PER_SLICE 8
#define MAX_PLANES 4
#define NUM_MB_LIMITS 4
#define MAX_STORED_Q 16
#define TRELLIS_WIDTH 16
#define SCORE_LIMIT (INT32_MAX / 2)

enum {
	PRORES_PROFILE_PROXY = 0,
	PRORES_PROFILE_LT,
	PRORES_PROFILE_STANDARD,
	PRORES_PROFILE_HQ,
	PRORES_PROFILE_4444,
	PRORES_PROFILE_4444XQ,
};

enum {
	QUANT_MAT_PROXY = 0,
	QUANT_MAT_PROXY_CHROMA,
	QUANT_MAT_LT,
	QUANT_MAT_STANDARD,
	QUANT_MAT_HQ,
	QUANT_MAT_XQ_LUMA,
	QUANT_MAT_DEFAULT,
};

static const uint8_t prores_quant_matrices[][64] = {
	{ // proxy
		 4,  7,  9, 11, 13, 14, 15, 63,
		 7,  7, 11, 12, 14, 15, 63, 63,
		 9, 11, 13, 14, 15, 63, 63, 63,
		11, 11, 13, 14, 63, 63, 63, 63,
		11, 13, 14, 63, 63, 63, 63, 63,
		13, 14, 63, 63, 63, 63, 63, 63,
		13, 63, 63, 63, 63, 63, 63, 63,
		63, 63, 63, 63, 63, 63, 63, 63,
	},
	{ // proxy chromas
		 4,  7,  9, 11, 13, 14, 63, 63,
		 7,  7, 11, 12, 14, 63, 63, 63,
		 9, 11, 13, 14, 63, 63, 63, 63,
		11, 11, 13, 14, 63, 63, 63, 63,
		11, 13, 14, 63, 63, 63, 63, 63,
		13, 14, 63, 63, 63, 63, 63, 63,
		13, 63, 63, 63, 63, 63, 63, 63,
		63, 63, 63, 63, 63, 63, 63, 63
	},
	{ // LT
		 4,  5,  6,  7,  9, 11, 13, 15,
		 5,  5,  7,  8, 11, 13, 15, 17,
		 6,  7,  9, 11, 13, 15, 15, 17,
		 7,  7,  9, 11, 13, 15, 17, 19,
		 7,  9, 11, 13, 14, 16, 19, 23,
		 9, 11, 13, 14, 16, 19, 23, 29,
		 9, 11, 13, 15, 17, 21, 28, 35,
		11, 13, 16, 17, 21, 28, 35, 41,
	},
	{ // standard
		 4,  4,  5,  5,  6,  7,  7,  9,
		 4,  4,  5,  6,  7,  7,  9,  9,
		 5,  5,  6,  7,  7,  9,  9, 10,
		 5,  5,  6,  7,  7,  9,  9, 10,
		 5,  6,  7,  7,  8,  9, 10, 12,
		 6,  7,  7,  8,  9, 10, 12, 15,
		 6,  7,  7,  9, 10, 11, 14, 17,
		 7,  7,  9, 10, 11, 14, 17, 21,
	},
	{ // high quality
		 4,  4,  4,  4,  4,  4,  4,  4,
		 4,  4,  4,  4,  4,  4,  4,  4,
		 4,  4,  4,  4,  4,  4,  4,  4,
		 4,  4,  4,  4,  4,  4,  4,  5,
		 4,  4,  4,  4,  4,  4,  5,  5,
		 4,  4,  4,  4,  4,  5,  5,  6,
		 4,  4,  4,  4,  5,  5,  6,  7,
		 4,  4,  4,  4,  5,  6,  7,  7,
	},
	{ // XQ luma
		 2,  2,  2,  2,  2,  2,  2,  2,
		 2,  2,  2,  2,  2,  2,  2,  2,
		 2,  2,  2,  2,  2,  2,  2,  2,
		 2,  2,  2,  2,  2,  2,  2,  3,
		 2,  2,  2,  2,  2,  2,  3,  3,
		 2,  2,  2,  2,  2,  3,  3,  3,
		 2,  2,  2,  2,  3,  3,  3,  4,
		 2,  2,  2,  2,  3,  3,  4,  4,
	},
	{ // codec default
		 4,  4,  4,  4,  4,  4,  4,  4,
		 4,  4,  4,  4,  4,  4,  4,  4,
		 4,  4,  4,  4,  4,  4,  4,  4,
		 4,  4,  4,  4,  4,  4,  4,  4,
		 4,  4,  4,  4,  4,  4,  4,  4,
		 4,  4,  4,  4,  4,  4,  4,  4,
		 4,  4,  4,  4,  4,  4,  4,  4,
		 4,  4,  4,  4,  4,  4,  4,  4,
	},
};

static const int prores_mb_limits[NUM_MB_LIMITS] = {
	1620,	// up to 720x576
	2700,	// up to 960x720
	6075,	// up to 1440x1080
	9216,	// up to 2048x1152
};

struct ProresProfile {
	uint32_t tag;
	int min_quant;
	int max_quant;
	int br_tab[NUM_MB_LIMITS];
	int quant;
	int quant_chroma;
};

#define MKBETAG(a, b, c, d) ((uint32_t)(d) | (uint32_t)(c) << 8 | (uint32_t)(b) << 16 | (uint32_t)(a) << 24)

static const ProresProfile prores_profile_info[6] = {
	{ MKBETAG('a', 'p', 'c', 'o'), 4, 8, { 300, 242, 220, 194 }, QUANT_MAT_PROXY, QUANT_MAT_PROXY_CHROMA },
	{ MKBETAG('a', 'p', 'c', 's'), 1, 9, { 720, 560, 490, 440 }, QUANT_MAT_LT, QUANT_MAT_LT },
	{ MKBETAG('a', 'p', 'c', 'n'), 1, 6, { 1050, 808, 710, 632 }, QUANT_MAT_STANDARD, QUANT_MAT_STANDARD },
	{ MKBETAG('a', 'p', 'c', 'h'), 1, 6, { 1566, 1216, 1070, 950 }, QUANT_MAT_HQ, QUANT_MAT_HQ },
	{ MKBETAG('a', 'p', '4', 'h'), 1, 6, { 2350, 1828, 1600, 1425 }, QUANT_MAT_HQ, QUANT_MAT_HQ },
	// FFmpeg: "Fix me : use QUANT_MAT_XQ_LUMA"; kept as FFmpeg has it.
	{ MKBETAG('a', 'p', '4', 'x'), 1, 6, { 3525, 2742, 2400, 2137 }, QUANT_MAT_HQ, QUANT_MAT_HQ },
};

#define FRAME_ID MKBETAG('i', 'c', 'p', 'f')

// ---------------------------------------------------------------- helpers

static inline int av_log2(uint32_t v) { return 31 - __builtin_clz(v | 1); }
static inline int ffmin(int a, int b) { return a < b ? a : b; }
static inline int ffmax(int a, int b) { return a > b ? a : b; }
static inline int ffabs(int a) { return a < 0 ? -a : a; }
static inline unsigned zero_extend(unsigned v, int bits) { return v & ((1u << bits) - 1); }
static inline int popcount(uint32_t v) { return __builtin_popcount(v); }

static inline void put_be16(uint8_t **p, unsigned v) { (*p)[0] = (uint8_t)(v >> 8); (*p)[1] = (uint8_t)v; *p += 2; }
static inline void put_be32(uint8_t **p, uint32_t v) { (*p)[0] = (uint8_t)(v >> 24); (*p)[1] = (uint8_t)(v >> 16); (*p)[2] = (uint8_t)(v >> 8); (*p)[3] = (uint8_t)v; *p += 4; }
static inline void put_byte(uint8_t **p, unsigned v) { (*p)[0] = (uint8_t)v; *p += 1; }

// ---------------------------------------------------------------- bit writer (put_bits.h)

// MSB-first, as FFmpeg's big-endian writer; the byte stream does not depend on its cache
// width. A write past the end sets overflow instead of writing (FFmpeg logs and drops it).
struct PutBits {
	uint8_t *buf, *ptr, *end;
	uint64_t bit_buf;
	int bit_count;	// bits held in bit_buf, 0..7 between calls
	int overflow;
};

static void init_put_bits(PutBits *pb, uint8_t *buf, int size)
{
	pb->buf = pb->ptr = buf;
	pb->end = buf + (size > 0 ? size : 0);
	pb->bit_buf = 0;
	pb->bit_count = 0;
	pb->overflow = 0;
}

static inline void put_bits(PutBits *pb, int n, uint32_t value)
{
	pb->bit_buf = (pb->bit_buf << n) | (value & (n >= 32 ? 0xFFFFFFFFu : ((1u << n) - 1)));
	pb->bit_count += n;
	while (pb->bit_count >= 8) {
		pb->bit_count -= 8;
		if (pb->ptr < pb->end) *pb->ptr++ = (uint8_t)(pb->bit_buf >> pb->bit_count);
		else pb->overflow = 1;
	}
}

static inline void put_sbits(PutBits *pb, int n, int32_t value) { put_bits(pb, n, zero_extend((unsigned)value, n)); }

static void flush_put_bits(PutBits *pb)
{
	if (pb->bit_count) put_bits(pb, 8 - pb->bit_count, 0);
}

static inline int put_bytes_output(const PutBits *pb) { return (int)(pb->ptr - pb->buf); }

// ---------------------------------------------------------------- forward DCT (jfdctint_template.c, 10 bit)

#define CONST_BITS  13
#define PASS1_BITS  1
#define OUT_SHIFT   (PASS1_BITS + 1)
#define FIX_0_298631336  ((int32_t)  2446)
#define FIX_0_390180644  ((int32_t)  3196)
#define FIX_0_541196100  ((int32_t)  4433)
#define FIX_0_765366865  ((int32_t)  6270)
#define FIX_0_899976223  ((int32_t)  7373)
#define FIX_1_175875602  ((int32_t)  9633)
#define FIX_1_501321110  ((int32_t)  12299)
#define FIX_1_847759065  ((int32_t)  15137)
#define FIX_1_961570560  ((int32_t)  16069)
#define FIX_2_053119869  ((int32_t)  16819)
#define FIX_2_562915447  ((int32_t)  20995)
#define FIX_3_072711026  ((int32_t)  25172)
#define MULTIPLY(var, c) ((int)((var) * (unsigned)(c)))
#define DESCALE(x, n) (((int)(x) + (1 << ((n) - 1))) >> (n))

static void row_fdct(int16_t *data)
{
	int tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;
	int tmp10, tmp11, tmp12, tmp13;
	unsigned z1, z2, z3, z4, z5;
	int16_t *dataptr = data;

	for (int ctr = 7; ctr >= 0; ctr--) {
		tmp0 = dataptr[0] + dataptr[7];
		tmp7 = dataptr[0] - dataptr[7];
		tmp1 = dataptr[1] + dataptr[6];
		tmp6 = dataptr[1] - dataptr[6];
		tmp2 = dataptr[2] + dataptr[5];
		tmp5 = dataptr[2] - dataptr[5];
		tmp3 = dataptr[3] + dataptr[4];
		tmp4 = dataptr[3] - dataptr[4];

		tmp10 = tmp0 + tmp3;
		tmp13 = tmp0 - tmp3;
		tmp11 = tmp1 + tmp2;
		tmp12 = tmp1 - tmp2;

		dataptr[0] = (int16_t)((tmp10 + tmp11) * (1 << PASS1_BITS));
		dataptr[4] = (int16_t)((tmp10 - tmp11) * (1 << PASS1_BITS));

		z1 = MULTIPLY(tmp12 + tmp13, FIX_0_541196100);
		dataptr[2] = (int16_t)DESCALE(z1 + MULTIPLY(tmp13, FIX_0_765366865), CONST_BITS - PASS1_BITS);
		dataptr[6] = (int16_t)DESCALE(z1 + MULTIPLY(tmp12, -FIX_1_847759065), CONST_BITS - PASS1_BITS);

		z1 = tmp4 + tmp7;
		z2 = tmp5 + tmp6;
		z3 = tmp4 + tmp6;
		z4 = tmp5 + tmp7;
		z5 = MULTIPLY(z3 + z4, FIX_1_175875602);

		tmp4 = MULTIPLY(tmp4, FIX_0_298631336);
		tmp5 = MULTIPLY(tmp5, FIX_2_053119869);
		tmp6 = MULTIPLY(tmp6, FIX_3_072711026);
		tmp7 = MULTIPLY(tmp7, FIX_1_501321110);
		z1 = MULTIPLY(z1, -FIX_0_899976223);
		z2 = MULTIPLY(z2, -FIX_2_562915447);
		z3 = MULTIPLY(z3, -FIX_1_961570560);
		z4 = MULTIPLY(z4, -FIX_0_390180644);

		z3 += z5;
		z4 += z5;

		dataptr[7] = (int16_t)DESCALE(tmp4 + z1 + z3, CONST_BITS - PASS1_BITS);
		dataptr[5] = (int16_t)DESCALE(tmp5 + z2 + z4, CONST_BITS - PASS1_BITS);
		dataptr[3] = (int16_t)DESCALE(tmp6 + z2 + z3, CONST_BITS - PASS1_BITS);
		dataptr[1] = (int16_t)DESCALE(tmp7 + z1 + z4, CONST_BITS - PASS1_BITS);

		dataptr += 8;
	}
}

static void jpeg_fdct_islow_10(int16_t *data)
{
	int tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;
	int tmp10, tmp11, tmp12, tmp13;
	unsigned z1, z2, z3, z4, z5;
	int16_t *dataptr;

	row_fdct(data);

	dataptr = data;
	for (int ctr = 7; ctr >= 0; ctr--) {
		tmp0 = dataptr[8 * 0] + dataptr[8 * 7];
		tmp7 = dataptr[8 * 0] - dataptr[8 * 7];
		tmp1 = dataptr[8 * 1] + dataptr[8 * 6];
		tmp6 = dataptr[8 * 1] - dataptr[8 * 6];
		tmp2 = dataptr[8 * 2] + dataptr[8 * 5];
		tmp5 = dataptr[8 * 2] - dataptr[8 * 5];
		tmp3 = dataptr[8 * 3] + dataptr[8 * 4];
		tmp4 = dataptr[8 * 3] - dataptr[8 * 4];

		tmp10 = tmp0 + tmp3;
		tmp13 = tmp0 - tmp3;
		tmp11 = tmp1 + tmp2;
		tmp12 = tmp1 - tmp2;

		dataptr[8 * 0] = DESCALE(tmp10 + tmp11, OUT_SHIFT);
		dataptr[8 * 4] = DESCALE(tmp10 - tmp11, OUT_SHIFT);

		z1 = MULTIPLY(tmp12 + tmp13, FIX_0_541196100);
		dataptr[8 * 2] = DESCALE(z1 + MULTIPLY(tmp13, FIX_0_765366865), CONST_BITS + OUT_SHIFT);
		dataptr[8 * 6] = DESCALE(z1 + MULTIPLY(tmp12, -FIX_1_847759065), CONST_BITS + OUT_SHIFT);

		z1 = tmp4 + tmp7;
		z2 = tmp5 + tmp6;
		z3 = tmp4 + tmp6;
		z4 = tmp5 + tmp7;
		z5 = MULTIPLY(z3 + z4, FIX_1_175875602);

		tmp4 = MULTIPLY(tmp4, FIX_0_298631336);
		tmp5 = MULTIPLY(tmp5, FIX_2_053119869);
		tmp6 = MULTIPLY(tmp6, FIX_3_072711026);
		tmp7 = MULTIPLY(tmp7, FIX_1_501321110);
		z1 = MULTIPLY(z1, -FIX_0_899976223);
		z2 = MULTIPLY(z2, -FIX_2_562915447);
		z3 = MULTIPLY(z3, -FIX_1_961570560);
		z4 = MULTIPLY(z4, -FIX_0_390180644);

		z3 += z5;
		z4 += z5;

		dataptr[8 * 7] = DESCALE(tmp4 + z1 + z3, CONST_BITS + OUT_SHIFT);
		dataptr[8 * 5] = DESCALE(tmp5 + z2 + z4, CONST_BITS + OUT_SHIFT);
		dataptr[8 * 3] = DESCALE(tmp6 + z2 + z3, CONST_BITS + OUT_SHIFT);
		dataptr[8 * 1] = DESCALE(tmp7 + z1 + z4, CONST_BITS + OUT_SHIFT);

		dataptr++;
	}
}

// prores_fdct: an 8x8 block of the plane (linesize in bytes, as FFmpeg's) into block.
static void prores_fdct(const uint16_t *src, ptrdiff_t linesize, int16_t *block)
{
	const uint16_t *tsrc = src;
	for (int y = 0; y < 8; y++) {
		for (int x = 0; x < 8; x++)
			block[y * 8 + x] = (int16_t)tsrc[x];
		tsrc += linesize >> 1;
	}
	jpeg_fdct_islow_10(block);
}

// ---------------------------------------------------------------- context

struct TrellisNode {
	int prev_node;
	int quant;
	int bits;
	int score;
};

struct ProresEncCtx {
	int16_t blocks[MAX_PLANES][64 * 4 * MAX_MBS_PER_SLICE];	// ProresThreadData.blocks
	uint16_t emu_buf[16 * 16];
	int16_t quants[MAX_STORED_Q][64];
	int16_t quants_chroma[MAX_STORED_Q][64];
	int16_t custom_q[64];		// encode_slice's (ProresContext)
	int16_t custom_chroma_q[64];
	int16_t td_custom_q[64];	// find_slice_quant's (ProresThreadData)
	int16_t td_custom_chroma_q[64];
	const uint8_t *quant_mat;
	const uint8_t *quant_chroma_mat;
	const uint8_t *scantable;

	int width, height;
	int mb_width, mb_height;
	int mbs_per_slice;
	int chroma_factor;
	int slices_width;
	int slices_per_picture;
	int num_planes;
	int bits_per_mb;
	int alpha_bits;
	int alpha16;		// alpha plane holds 16-bit values (not FFmpeg's 10-bit)
	uint8_t vendor[4];
	uint8_t color_primaries, color_trc, colorspace;
	int frame_size_upper_bound;
	const ProresProfile *profile_info;

	int *slice_q;
	TrellisNode *nodes;

	// The input frame: tight planes, linesize in bytes as FFmpeg's.
	uint16_t *plane[4];
	int linesize[4];

	uint16_t *rgba;		// prores_enc_rgba_input's buffer, width * height * 4
	uint8_t *out;		// the encoded frame
	int out_cap;
};

// ---------------------------------------------------------------- proresenc_kostya.c

static void get_slice_data(ProresEncCtx *ctx, const uint16_t *src, ptrdiff_t linesize,
                           int x, int y, int w, int h, int16_t *blocks, uint16_t *emu_buf,
                           int mbs_per_slice, int blocks_per_mb, int is_chroma)
{
	const uint16_t *esrc;
	const int mb_width = 4 * blocks_per_mb;
	ptrdiff_t elinesize;

	for (int i = 0; i < mbs_per_slice; i++, src += mb_width) {
		if (x >= w) {
			memset(blocks, 0, 64 * (mbs_per_slice - i) * blocks_per_mb * sizeof(*blocks));
			return;
		}
		if (x + mb_width <= w && y + 16 <= h) {
			esrc = src;
			elinesize = linesize;
		} else {
			int bw, bh, pix, j;

			esrc = emu_buf;
			elinesize = 16 * sizeof(*emu_buf);

			bw = ffmin(w - x, mb_width);
			bh = ffmin(h - y, 16);

			for (j = 0; j < bh; j++) {
				memcpy(emu_buf + j * 16, (const uint8_t *)src + j * linesize, bw * sizeof(*src));
				pix = emu_buf[j * 16 + bw - 1];
				for (int k = bw; k < mb_width; k++)
					emu_buf[j * 16 + k] = (uint16_t)pix;
			}
			for (; j < 16; j++)
				memcpy(emu_buf + j * 16, emu_buf + (bh - 1) * 16, mb_width * sizeof(*emu_buf));
		}
		const ptrdiff_t eline = elinesize >> 1;	// in samples
		if (!is_chroma) {
			prores_fdct(esrc, elinesize, blocks);
			blocks += 64;
			if (blocks_per_mb > 2) {
				prores_fdct(esrc + 8, elinesize, blocks);
				blocks += 64;
			}
			prores_fdct(esrc + eline * 8, elinesize, blocks);
			blocks += 64;
			if (blocks_per_mb > 2) {
				prores_fdct(esrc + eline * 8 + 8, elinesize, blocks);
				blocks += 64;
			}
		} else {
			prores_fdct(esrc, elinesize, blocks);
			blocks += 64;
			prores_fdct(esrc + eline * 8, elinesize, blocks);
			blocks += 64;
			if (blocks_per_mb > 2) {
				prores_fdct(esrc + 8, elinesize, blocks);
				blocks += 64;
				prores_fdct(esrc + eline * 8 + 8, elinesize, blocks);
				blocks += 64;
			}
		}

		x += mb_width;
	}
}

// FFmpeg reads 10-bit alpha and widens it (8-bit: a >> 2, 16-bit: (a << 6) | (a >> 4)).
// With alpha16 the plane is already 16 bits: 8-bit alpha takes a >> 8, 16-bit takes a.
static void get_alpha_data(ProresEncCtx *ctx, const uint16_t *src, ptrdiff_t linesize,
                           int x, int y, int w, int h, uint16_t *blocks,
                           int mbs_per_slice, int abits)
{
	const int slice_width = 16 * mbs_per_slice;
	int i, j, copy_w, copy_h;

	copy_w = ffmin(w - x, slice_width);
	copy_h = ffmin(h - y, 16);
	for (i = 0; i < copy_h; i++) {
		memcpy(blocks, src, copy_w * sizeof(*src));
		if (ctx->alpha16) {
			if (abits == 8)
				for (j = 0; j < copy_w; j++)
					blocks[j] >>= 8;
		} else if (abits == 8) {
			for (j = 0; j < copy_w; j++)
				blocks[j] >>= 2;
		} else {
			for (j = 0; j < copy_w; j++)
				blocks[j] = (uint16_t)((blocks[j] << 6) | (blocks[j] >> 4));
		}
		for (j = copy_w; j < slice_width; j++)
			blocks[j] = blocks[copy_w - 1];
		blocks += slice_width;
		src += linesize >> 1;
	}
	for (; i < 16; i++) {
		memcpy(blocks, blocks - slice_width, slice_width * sizeof(*blocks));
		blocks += slice_width;
	}
}

// Write an unsigned rice/exp golomb codeword.
static inline void encode_vlc_codeword(PutBits *pb, unsigned codebook, int val)
{
	unsigned int rice_order, exp_order, switch_bits, switch_val;
	int exponent;

	switch_bits = (codebook & 3) + 1;
	rice_order = codebook >> 5;
	exp_order = (codebook >> 2) & 7;

	switch_val = switch_bits << rice_order;

	if ((unsigned)val >= switch_val) {
		val -= switch_val - (1 << exp_order);
		exponent = av_log2(val);

		put_bits(pb, exponent - exp_order + switch_bits, 0);
		put_bits(pb, exponent + 1, val);
	} else {
		exponent = val >> rice_order;

		if (exponent)
			put_bits(pb, exponent, 0);
		put_bits(pb, 1, 1);
		if (rice_order)
			put_sbits(pb, rice_order, val);
	}
}

#define GET_SIGN(x)  ((x) >> 31)
#define MAKE_CODE(x) (((x) * 2) ^ GET_SIGN(x))

static void encode_dcs(PutBits *pb, int16_t *blocks, int blocks_per_slice, int scale)
{
	int codebook = 5, code, dc, prev_dc, delta, sign, new_sign;

	prev_dc = (blocks[0] - 0x4000) / scale;
	encode_vlc_codeword(pb, FIRST_DC_CB, MAKE_CODE(prev_dc));
	sign = 0;
	blocks += 64;

	for (int i = 1; i < blocks_per_slice; i++, blocks += 64) {
		dc = (blocks[0] - 0x4000) / scale;
		delta = dc - prev_dc;
		new_sign = GET_SIGN(delta);
		delta = (delta ^ sign) - sign;
		code = MAKE_CODE(delta);
		encode_vlc_codeword(pb, prores_dc_codebook[codebook], code);
		codebook = ffmin(code, 6);
		sign = new_sign;
		prev_dc = dc;
	}
}

static void encode_acs(PutBits *pb, int16_t *blocks, int blocks_per_slice,
                       const uint8_t *scan, const int16_t *qmat)
{
	int prev_run = 4;
	int prev_level = 2;
	int run = 0, level, abs_level;
	const int max_coeffs = blocks_per_slice << 6;

	for (int i = 1; i < 64; i++) {
		for (int idx = scan[i]; idx < max_coeffs; idx += 64) {
			level = blocks[idx] / qmat[scan[i]];
			if (level) {
				abs_level = ffabs(level);
				encode_vlc_codeword(pb, prores_run_to_cb[prev_run], run);
				encode_vlc_codeword(pb, prores_level_to_cb[prev_level], abs_level - 1);
				put_sbits(pb, 1, GET_SIGN(level));

				prev_run = ffmin(run, 15);
				prev_level = ffmin(abs_level, 9);
				run = 0;
			} else {
				run++;
			}
		}
	}
}

static void encode_slice_plane(ProresEncCtx *ctx, PutBits *pb, int mbs_per_slice,
                               int16_t *blocks, int blocks_per_mb, const int16_t *qmat)
{
	const int blocks_per_slice = mbs_per_slice * blocks_per_mb;
	encode_dcs(pb, blocks, blocks_per_slice, qmat[0]);
	encode_acs(pb, blocks, blocks_per_slice, ctx->scantable, qmat);
}

static void put_alpha_diff(PutBits *pb, int cur, int prev, int abits)
{
	const int dbits = (abits == 8) ? 4 : 7;
	const int dsize = 1 << (dbits - 1);
	int diff = cur - prev;

	diff = (int)zero_extend((unsigned)diff, abits);
	if (diff >= (1 << abits) - dsize)
		diff -= 1 << abits;
	if (diff < -dsize || diff > dsize || !diff) {
		put_bits(pb, 1, 1);
		put_bits(pb, abits, diff);
	} else {
		put_bits(pb, 1, 0);
		put_bits(pb, dbits - 1, ffabs(diff) - 1);
		put_bits(pb, 1, diff < 0);
	}
}

static void put_alpha_run(PutBits *pb, int run)
{
	if (run) {
		put_bits(pb, 1, 0);
		if (run < 0x10)
			put_bits(pb, 4, run);
		else
			put_bits(pb, 15, run);
	} else {
		put_bits(pb, 1, 1);
	}
}

static void encode_alpha_plane(ProresEncCtx *ctx, PutBits *pb, int mbs_per_slice, uint16_t *blocks)
{
	const int abits = ctx->alpha_bits;
	const int mask = (1 << abits) - 1;
	const int num_coeffs = mbs_per_slice * 256;
	int prev = mask, cur;
	int idx = 0;
	int run = 0;

	cur = blocks[idx++];
	put_alpha_diff(pb, cur, prev, abits);
	prev = cur;
	do {
		cur = blocks[idx++];
		if (cur != prev) {
			put_alpha_run(pb, run);
			put_alpha_diff(pb, cur, prev, abits);
			prev = cur;
			run = 0;
		} else {
			run++;
		}
	} while (idx < num_coeffs);
	put_alpha_run(pb, run);
}

// Plane i's position and geometry for slice (x, y): xp, yp in samples, the blocks per MB
// and the plane width the encoder clips to (FFmpeg's width >> 1 for 4:2:2 chroma).
static void plane_geometry(const ProresEncCtx *ctx, int i, int x, int y, int *xp, int *yp, int *num_cblocks, int *pwidth)
{
	const int is_chroma = (i == 1 || i == 2);
	if (!is_chroma || ctx->chroma_factor == CFACTOR_Y444) {
		*xp = x << 4;
		*num_cblocks = 4;
		*pwidth = ctx->width;
	} else {
		*xp = x << 3;
		*num_cblocks = 2;
		*pwidth = ctx->width >> 1;
	}
	*yp = y << 4;
}

static int encode_slice(ProresEncCtx *ctx, PutBits *pb, int sizes[4], int x, int y, int quant, int mbs_per_slice)
{
	int total_size = 0;
	int16_t *qmat;
	int16_t *qmat_chroma;

	if (quant < MAX_STORED_Q) {
		qmat = ctx->quants[quant];
		qmat_chroma = ctx->quants_chroma[quant];
	} else {
		qmat = ctx->custom_q;
		qmat_chroma = ctx->custom_chroma_q;
		for (int i = 0; i < 64; i++) {
			qmat[i] = (int16_t)(ctx->quant_mat[i] * quant);
			qmat_chroma[i] = (int16_t)(ctx->quant_chroma_mat[i] * quant);
		}
	}

	for (int i = 0; i < ctx->num_planes; i++) {
		const int is_chroma = (i == 1 || i == 2);
		int xp, yp, num_cblocks, pwidth;
		plane_geometry(ctx, i, x, y, &xp, &yp, &num_cblocks, &pwidth);

		const ptrdiff_t linesize = ctx->linesize[i];
		const uint16_t *src = (const uint16_t *)((const uint8_t *)ctx->plane[i] + yp * linesize) + xp;

		if (i < 3) {
			get_slice_data(ctx, src, linesize, xp, yp, pwidth, ctx->height,
			               ctx->blocks[0], ctx->emu_buf, mbs_per_slice, num_cblocks, is_chroma);
			encode_slice_plane(ctx, pb, mbs_per_slice, ctx->blocks[0], num_cblocks, is_chroma ? qmat_chroma : qmat);
		} else {
			get_alpha_data(ctx, src, linesize, xp, yp, pwidth, ctx->height,
			               (uint16_t *)ctx->blocks[0], mbs_per_slice, ctx->alpha_bits);
			encode_alpha_plane(ctx, pb, mbs_per_slice, (uint16_t *)ctx->blocks[0]);
		}
		flush_put_bits(pb);
		sizes[i] = put_bytes_output(pb) - total_size;
		total_size = put_bytes_output(pb);
	}
	return pb->overflow ? -1 : total_size;
}

static inline int estimate_vlc(unsigned codebook, int val)
{
	unsigned int rice_order, exp_order, switch_bits, switch_val;
	int exponent;

	switch_bits = (codebook & 3) + 1;
	rice_order = codebook >> 5;
	exp_order = (codebook >> 2) & 7;

	switch_val = switch_bits << rice_order;

	if ((unsigned)val >= switch_val) {
		val -= switch_val - (1 << exp_order);
		exponent = av_log2(val);

		return exponent * 2 - exp_order + switch_bits + 1;
	} else {
		return (val >> rice_order) + rice_order + 1;
	}
}

static int estimate_dcs(int *error, int16_t *blocks, int blocks_per_slice, int scale)
{
	int codebook = 5, code, dc, prev_dc, delta, sign, new_sign;
	int bits;

	prev_dc = (blocks[0] - 0x4000) / scale;
	bits = estimate_vlc(FIRST_DC_CB, MAKE_CODE(prev_dc));
	sign = 0;
	blocks += 64;
	*error += ffabs(blocks[0] - 0x4000) % scale;	// FFmpeg counts block 1 here, block 0 never

	for (int i = 1; i < blocks_per_slice; i++, blocks += 64) {
		dc = (blocks[0] - 0x4000) / scale;
		*error += ffabs(blocks[0] - 0x4000) % scale;
		delta = dc - prev_dc;
		new_sign = GET_SIGN(delta);
		delta = (delta ^ sign) - sign;
		code = MAKE_CODE(delta);
		bits += estimate_vlc(prores_dc_codebook[codebook], code);
		codebook = ffmin(code, 6);
		sign = new_sign;
		prev_dc = dc;
	}

	return bits;
}

static int estimate_acs(int *error, int16_t *blocks, int blocks_per_slice,
                        const uint8_t *scan, const int16_t *qmat)
{
	int prev_run = 4;
	int prev_level = 2;
	int run = 0, level, abs_level;
	int bits = 0;
	const int max_coeffs = blocks_per_slice << 6;

	for (int i = 1; i < 64; i++) {
		for (int idx = scan[i]; idx < max_coeffs; idx += 64) {
			level = blocks[idx] / qmat[scan[i]];
			*error += ffabs(blocks[idx]) % qmat[scan[i]];
			if (level) {
				abs_level = ffabs(level);
				bits += estimate_vlc(prores_run_to_cb[prev_run], run);
				bits += estimate_vlc(prores_level_to_cb[prev_level], abs_level - 1) + 1;
				prev_run = ffmin(run, 15);
				prev_level = ffmin(abs_level, 9);
				run = 0;
			} else {
				run++;
			}
		}
	}

	return bits;
}

static int estimate_slice_plane(ProresEncCtx *ctx, int *error, int plane, int mbs_per_slice,
                                int blocks_per_mb, const int16_t *qmat)
{
	const int blocks_per_slice = mbs_per_slice * blocks_per_mb;
	int bits = estimate_dcs(error, ctx->blocks[plane], blocks_per_slice, qmat[0]);
	bits += estimate_acs(error, ctx->blocks[plane], blocks_per_slice, ctx->scantable, qmat);
	return (bits + 7) & ~7;
}

static int est_alpha_diff(int cur, int prev, int abits)
{
	const int dbits = (abits == 8) ? 4 : 7;
	const int dsize = 1 << (dbits - 1);
	int diff = cur - prev;

	diff = (int)zero_extend((unsigned)diff, abits);
	if (diff >= (1 << abits) - dsize)
		diff -= 1 << abits;
	if (diff < -dsize || diff > dsize || !diff)
		return abits + 1;
	else
		return dbits + 1;
}

// blocks as int16_t, as FFmpeg passes td->blocks[3] here: 16-bit alpha reads back signed,
// which the modular diff makes harmless.
static int estimate_alpha_plane(ProresEncCtx *ctx, int mbs_per_slice, const int16_t *blocks)
{
	const int abits = ctx->alpha_bits;
	const int mask = (1 << abits) - 1;
	const int num_coeffs = mbs_per_slice * 256;
	int prev = mask, cur;
	int idx = 0;
	int run = 0;
	int bits;

	cur = blocks[idx++];
	bits = est_alpha_diff(cur, prev, abits);
	prev = cur;
	do {
		cur = blocks[idx++];
		if (cur != prev) {
			if (!run)
				bits++;
			else if (run < 0x10)
				bits += 4;
			else
				bits += 15;
			bits += est_alpha_diff(cur, prev, abits);
			prev = cur;
			run = 0;
		} else {
			run++;
		}
	} while (idx < num_coeffs);

	if (run) {
		if (run < 0x10)
			bits += 4;
		else
			bits += 15;
	}

	return bits;
}

static int find_slice_quant(ProresEncCtx *ctx, int trellis_node, int x, int y, int mbs_per_slice)
{
	TrellisNode *nodes = ctx->nodes;
	int q, pq;
	int num_cblocks[MAX_PLANES];
	const int min_quant = ctx->profile_info->min_quant;
	const int max_quant = ctx->profile_info->max_quant;
	int error, bits, bits_limit;
	int mbs, prev, cur, new_score;
	int slice_bits[TRELLIS_WIDTH], slice_score[TRELLIS_WIDTH];
	int overquant;
	int16_t *qmat;
	int16_t *qmat_chroma;
	int alpha_bits = 0;

	mbs = x + mbs_per_slice;

	for (int i = 0; i < ctx->num_planes; i++) {
		const int is_chroma = (i == 1 || i == 2);
		int xp, yp, pwidth;
		plane_geometry(ctx, i, x, y, &xp, &yp, &num_cblocks[i], &pwidth);

		const ptrdiff_t linesize = ctx->linesize[i];
		const uint16_t *src = (const uint16_t *)((const uint8_t *)ctx->plane[i] + yp * linesize) + xp;

		if (i < 3) {
			get_slice_data(ctx, src, linesize, xp, yp, pwidth, ctx->height,
			               ctx->blocks[i], ctx->emu_buf, mbs_per_slice, num_cblocks[i], is_chroma);
		} else {
			get_alpha_data(ctx, src, linesize, xp, yp, pwidth, ctx->height,
			               (uint16_t *)ctx->blocks[i], mbs_per_slice, ctx->alpha_bits);
		}
	}

	for (q = min_quant; q < max_quant + 2; q++) {
		nodes[trellis_node + q].prev_node = -1;
		nodes[trellis_node + q].quant = q;
	}

	if (ctx->alpha_bits)
		alpha_bits = estimate_alpha_plane(ctx, mbs_per_slice, ctx->blocks[3]);

	for (q = min_quant; q <= max_quant; q++) {
		bits = alpha_bits;
		error = 0;
		bits += estimate_slice_plane(ctx, &error, 0, mbs_per_slice, num_cblocks[0], ctx->quants[q]);
		for (int i = 1; i < ctx->num_planes - !!ctx->alpha_bits; i++)
			bits += estimate_slice_plane(ctx, &error, i, mbs_per_slice, num_cblocks[i], ctx->quants_chroma[q]);
		if (bits > 65000 * 8)
			error = SCORE_LIMIT;

		slice_bits[q] = bits;
		slice_score[q] = error;
	}
	if (slice_bits[max_quant] <= ctx->bits_per_mb * mbs_per_slice) {
		slice_bits[max_quant + 1] = slice_bits[max_quant];
		slice_score[max_quant + 1] = slice_score[max_quant] + 1;
		overquant = max_quant;
	} else {
		for (q = max_quant + 1; q < 128; q++) {
			bits = alpha_bits;
			error = 0;
			if (q < MAX_STORED_Q) {
				qmat = ctx->quants[q];
				qmat_chroma = ctx->quants_chroma[q];
			} else {
				qmat = ctx->td_custom_q;
				qmat_chroma = ctx->td_custom_chroma_q;
				for (int i = 0; i < 64; i++) {
					qmat[i] = (int16_t)(ctx->quant_mat[i] * q);
					qmat_chroma[i] = (int16_t)(ctx->quant_chroma_mat[i] * q);
				}
			}
			bits += estimate_slice_plane(ctx, &error, 0, mbs_per_slice, num_cblocks[0], qmat);
			for (int i = 1; i < ctx->num_planes - !!ctx->alpha_bits; i++)
				bits += estimate_slice_plane(ctx, &error, i, mbs_per_slice, num_cblocks[i], qmat_chroma);
			if (bits <= ctx->bits_per_mb * mbs_per_slice)
				break;
		}

		slice_bits[max_quant + 1] = bits;
		slice_score[max_quant + 1] = error;
		overquant = q;
	}
	nodes[trellis_node + max_quant + 1].quant = overquant;

	bits_limit = mbs * ctx->bits_per_mb;
	for (pq = min_quant; pq < max_quant + 2; pq++) {
		prev = trellis_node - TRELLIS_WIDTH + pq;

		for (q = min_quant; q < max_quant + 2; q++) {
			cur = trellis_node + q;
			bits = nodes[prev].bits + slice_bits[q];
			error = slice_score[q];
			if (bits > bits_limit)
				error = SCORE_LIMIT;

			if (nodes[prev].score < SCORE_LIMIT && error < SCORE_LIMIT)
				new_score = nodes[prev].score + error;
			else
				new_score = SCORE_LIMIT;
			if (nodes[cur].prev_node == -1 || nodes[cur].score >= new_score) {
				nodes[cur].bits = bits;
				nodes[cur].score = new_score;
				nodes[cur].prev_node = prev;
			}
		}
	}

	error = nodes[trellis_node + min_quant].score;
	pq = trellis_node + min_quant;
	for (q = min_quant + 1; q < max_quant + 2; q++) {
		if (nodes[trellis_node + q].score <= error) {
			error = nodes[trellis_node + q].score;
			pq = trellis_node + q;
		}
	}

	return pq;
}

static void find_quant_row(ProresEncCtx *ctx, int y)
{
	int mbs_per_slice = ctx->mbs_per_slice;
	int x, mb, q = 0;

	for (x = mb = 0; x < ctx->mb_width; x += mbs_per_slice, mb++) {
		while (ctx->mb_width - x < mbs_per_slice)
			mbs_per_slice >>= 1;
		q = find_slice_quant(ctx, (mb + 1) * TRELLIS_WIDTH, x, y, mbs_per_slice);
	}

	for (x = ctx->slices_width - 1; x >= 0; x--) {
		ctx->slice_q[x + y * ctx->slices_width] = ctx->nodes[q].quant;
		q = ctx->nodes[q].prev_node;
	}
}

static int ensure_out(ProresEncCtx *ctx, int size)
{
	if (size <= ctx->out_cap) return 1;
	uint8_t *p = (uint8_t *)realloc(ctx->out, (size_t)size);
	if (!p) return 0;
	ctx->out = p;
	ctx->out_cap = size;
	return 1;
}

// encode_frame for one progressive picture. Returns the frame size, or -1.
static int encode_frame(ProresEncCtx *ctx)
{
	uint8_t *buf, *tmp;
	int picture_size_pos, slice_sizes_pos, slice_hdr_pos;
	PutBits pb;
	int sizes[4] = { 0 };
	const int slice_hdr_size = 2 * ctx->num_planes;
	int max_slice_size = (ctx->frame_size_upper_bound - 200) / (ctx->slices_per_picture + 1);
	int pkt_size = ctx->frame_size_upper_bound;

	if (!ensure_out(ctx, pkt_size)) return -1;

	// Offsets from the frame start (FFmpeg's pkt->data), since the buffer may grow.
	const int orig = 8;	// frame size + 'icpf'
	buf = ctx->out + 4;
	put_be32(&buf, FRAME_ID);

	// frame header
	tmp = buf;
	buf += 2;	// frame header size will be stored here
	put_be16(&buf, ctx->chroma_factor != CFACTOR_Y422 || ctx->alpha_bits ? 1 : 0);
	memcpy(buf, ctx->vendor, 4);
	buf += 4;
	put_be16(&buf, ctx->width);
	put_be16(&buf, ctx->height);
	put_byte(&buf, ctx->chroma_factor << 6);	// progressive: no field order bits
	put_byte(&buf, 0);	// reserved
	put_byte(&buf, ctx->color_primaries);
	put_byte(&buf, ctx->color_trc);
	put_byte(&buf, ctx->colorspace);
	put_byte(&buf, ctx->alpha_bits >> 3);
	put_byte(&buf, 0);	// reserved
	put_byte(&buf, 0x03);	// matrix flags - both matrices are present (quant_mat auto)
	memcpy(buf, ctx->quant_mat, 64);
	buf += 64;
	memcpy(buf, ctx->quant_chroma_mat, 64);
	buf += 64;
	put_be16(&tmp, (unsigned)(buf - (ctx->out + orig)));

	// picture header
	picture_size_pos = (int)(buf - ctx->out) + 1;
	put_byte(&buf, 0x40);	// picture header size (in bits)
	buf += 4;	// picture data size will be stored here
	put_be16(&buf, ctx->slices_per_picture);
	put_byte(&buf, av_log2(ctx->mbs_per_slice) << 4);	// slice width and height in MBs

	// seek table - filled during slice encoding
	slice_sizes_pos = (int)(buf - ctx->out);
	buf += ctx->slices_per_picture * 2;

	for (int y = 0; y < ctx->mb_height; y++)
		find_quant_row(ctx, y);

	int pos = (int)(buf - ctx->out);
	for (int y = 0; y < ctx->mb_height; y++) {
		int mbs_per_slice = ctx->mbs_per_slice;
		for (int x = 0, mb = 0; x < ctx->mb_width; x += mbs_per_slice, mb++) {
			const int q = ctx->slice_q[mb + y * ctx->slices_width];

			while (ctx->mb_width - x < mbs_per_slice)
				mbs_per_slice >>= 1;

			ctx->out[pos++] = (uint8_t)(slice_hdr_size << 3);
			slice_hdr_pos = pos;
			pos += slice_hdr_size - 1;
			if (pkt_size <= pos - orig + 2 * max_slice_size) {
				int delta = 200 + (ctx->slices_per_picture + 1) * max_slice_size - pkt_size;
				delta = ffmax(delta, 2 * max_slice_size);
				ctx->frame_size_upper_bound += delta;
				pkt_size += delta;
				if (!ensure_out(ctx, pkt_size)) return -1;
			}
			init_put_bits(&pb, ctx->out + pos, pkt_size - (pos - orig));
			if (encode_slice(ctx, &pb, sizes, x, y, q, mbs_per_slice) < 0)
				return -1;

			uint8_t *hdr = ctx->out + slice_hdr_pos;
			put_byte(&hdr, q);
			int slice_size = slice_hdr_size + sizes[ctx->num_planes - 1];
			for (int i = 0; i < ctx->num_planes - 1; i++) {
				put_be16(&hdr, sizes[i]);
				slice_size += sizes[i];
			}
			uint8_t *ss = ctx->out + slice_sizes_pos;
			put_be16(&ss, slice_size);
			slice_sizes_pos += 2;
			pos += slice_size - slice_hdr_size;
			if (max_slice_size < slice_size)
				max_slice_size = slice_size;
		}
	}

	uint8_t *pp = ctx->out + picture_size_pos;
	put_be32(&pp, (uint32_t)(pos - (picture_size_pos - 1)));

	uint8_t *fs = ctx->out;
	put_be32(&fs, (uint32_t)pos);
	return pos;
}

// ---------------------------------------------------------------- API

extern "C" {

// Opens an encoder. profile: 0 Proxy, 1 LT, 2 422 (standard), 3 HQ, 4 4444, 5 4444 XQ.
// chroma444: the input planes are 4:4:4 (FFmpeg follows the pixel format, not the
// profile). alpha: an alpha plane is given; it is coded only for 4444 / 4444 XQ, as in
// FFmpeg. alpha_bits: 8 or 16 (FFmpeg's default 16). vendor: big-endian fourcc. The color
// tags go into the frame header (ISO/IEC 23091-2 code points; 2 = unspecified).
// Returns the handle, or null.
ProresEncCtx *prores_enc_open(int profile, int width, int height, int chroma444, int alpha, int alpha_bits,
                              uint32_t vendor, int color_primaries, int color_trc, int colorspace)
{
	if (profile < 0 || profile > 5 || width <= 0 || height <= 0 || width > 65535 || height > 65535) return nullptr;
	if (alpha && alpha_bits != 8 && alpha_bits != 16) return nullptr;

	ProresEncCtx *ctx = (ProresEncCtx *)calloc(1, sizeof(ProresEncCtx));
	if (!ctx) return nullptr;

	const int mps = MAX_MBS_PER_SLICE;	// FFmpeg's mbs_per_slice default
	ctx->width = width;
	ctx->height = height;
	ctx->mbs_per_slice = mps;
	ctx->scantable = prores_progressive_scan;
	ctx->alpha_bits = (alpha && (profile == PRORES_PROFILE_4444 || profile == PRORES_PROFILE_4444XQ)) ? alpha_bits : 0;
	ctx->chroma_factor = chroma444 ? CFACTOR_Y444 : CFACTOR_Y422;
	ctx->profile_info = prores_profile_info + profile;
	ctx->num_planes = 3 + !!ctx->alpha_bits;
	ctx->vendor[0] = (uint8_t)(vendor >> 24);
	ctx->vendor[1] = (uint8_t)(vendor >> 16);
	ctx->vendor[2] = (uint8_t)(vendor >> 8);
	ctx->vendor[3] = (uint8_t)vendor;
	ctx->color_primaries = (uint8_t)color_primaries;
	ctx->color_trc = (uint8_t)color_trc;
	ctx->colorspace = (uint8_t)colorspace;

	ctx->mb_width = ((width + 15) & ~15) >> 4;
	ctx->mb_height = ((height + 15) & ~15) >> 4;
	ctx->slices_width = ctx->mb_width / mps;
	ctx->slices_width += popcount(ctx->mb_width - ctx->slices_width * mps);
	ctx->slices_per_picture = ctx->mb_height * ctx->slices_width;

	ctx->quant_mat = prores_quant_matrices[ctx->profile_info->quant];
	ctx->quant_chroma_mat = prores_quant_matrices[ctx->profile_info->quant_chroma];

	int i;
	for (i = 0; i < NUM_MB_LIMITS - 1; i++)
		if (prores_mb_limits[i] >= ctx->mb_width * ctx->mb_height)
			break;
	ctx->bits_per_mb = ctx->profile_info->br_tab[i];
	if (ctx->alpha_bits)
		ctx->bits_per_mb *= 20;

	for (i = ctx->profile_info->min_quant; i < MAX_STORED_Q; i++) {
		for (int j = 0; j < 64; j++) {
			ctx->quants[i][j] = (int16_t)(ctx->quant_mat[j] * i);
			ctx->quants_chroma[i][j] = (int16_t)(ctx->quant_chroma_mat[j] * i);
		}
	}

	ctx->frame_size_upper_bound = (ctx->slices_per_picture + 1) * (2 + 2 * ctx->num_planes + (mps * ctx->bits_per_mb) / 8) + 200;
	if (ctx->alpha_bits)
		ctx->frame_size_upper_bound += (ctx->slices_per_picture + 1) * ((ctx->mbs_per_slice * 256 * (1 + ctx->alpha_bits + 1) + 7) >> 3);

	ctx->slice_q = (int *)calloc((size_t)ctx->slices_per_picture, sizeof(int));
	ctx->nodes = (TrellisNode *)calloc((size_t)(ctx->slices_width + 1) * TRELLIS_WIDTH, sizeof(TrellisNode));
	for (i = 0; i < 4; i++) {
		const int pw = (i == 1 || i == 2) && !chroma444 ? (width + 1) >> 1 : width;
		ctx->linesize[i] = pw * 2;
		if (i < 3 || alpha)
			ctx->plane[i] = (uint16_t *)calloc((size_t)pw * height, sizeof(uint16_t));
	}
	if (!ctx->slice_q || !ctx->nodes || !ctx->plane[0] || !ctx->plane[1] || !ctx->plane[2] || (alpha && !ctx->plane[3])) {
		free(ctx->slice_q); free(ctx->nodes);
		for (i = 0; i < 4; i++) free(ctx->plane[i]);
		free(ctx);
		return nullptr;
	}
	for (i = ctx->profile_info->min_quant; i < ctx->profile_info->max_quant + 2; i++) {
		ctx->nodes[i].prev_node = -1;
		ctx->nodes[i].bits = 0;
		ctx->nodes[i].score = 0;
	}
	return ctx;
}

void prores_enc_close(ProresEncCtx *ctx)
{
	if (!ctx) return;
	free(ctx->slice_q);
	free(ctx->nodes);
	for (int i = 0; i < 4; i++) free(ctx->plane[i]);
	free(ctx->rgba);
	free(ctx->out);
	free(ctx);
}

// Input plane i (0 Y, 1 Cb, 2 Cr, 3 A): tight uint16 samples, stride prores_enc_stride.
uint16_t *prores_enc_plane(ProresEncCtx *ctx, int i) { return (i >= 0 && i < 4) ? ctx->plane[i] : nullptr; }
int prores_enc_stride(const ProresEncCtx *ctx, int i) { return (i >= 0 && i < 4) ? ctx->linesize[i] >> 1 : 0; }

// 1: plane 3 holds 16-bit alpha; 0 (default): 10-bit alpha as FFmpeg's yuva444p10.
void prores_enc_set_alpha16(ProresEncCtx *ctx, int on) { ctx->alpha16 = on ? 1 : 0; }

// Encodes the planes into one complete frame ('icpf' atom). Returns its size, or -1.
int prores_enc_frame(ProresEncCtx *ctx) { return encode_frame(ctx); }
uint8_t *prores_enc_output(ProresEncCtx *ctx) { return ctx->out; }

// A width x height x 4 uint16 buffer for prores_enc_from_rgba16, or null.
uint16_t *prores_enc_rgba_input(ProresEncCtx *ctx)
{
	if (!ctx->rgba) ctx->rgba = (uint16_t *)malloc((size_t)ctx->width * ctx->height * 4 * sizeof(uint16_t));
	return ctx->rgba;
}

// Converts the RGBA16 input (straight alpha, 65535 = opaque) to the planes, limited range
// 10 bit through the matrix with luma coefficients kr, kb (BT.709 0.2126, 0.0722). 4:2:2
// chroma is the mean of each horizontal pair (a last odd column stands alone). Alpha goes
// in at 16 bits (and alpha16 is set). The inverse of prores_decode.cpp's prores_to_rgba16.
void prores_enc_from_rgba16(ProresEncCtx *ctx, float kr, float kb)
{
	const int w = ctx->width, h = ctx->height;
	const float kg = 1.0f - kr - kb;
	const float yScale = 876.0f, yOff = 64.0f;	// 219 << 2, 16 << 2
	const float cScale = 896.0f, cOff = 512.0f;	// 224 << 2, 128 << 2
	const float cbDiv = 1.0f / (2.0f * (1.0f - kb)), crDiv = 1.0f / (2.0f * (1.0f - kr));
	const float inv = 1.0f / 65535.0f;
	const int c444 = ctx->chroma_factor == CFACTOR_Y444;
	const int cstride = ctx->linesize[1] >> 1;

	auto quant = [](float v) -> uint16_t {
		const int q = (int)(v + 0.5f);
		return (uint16_t)(q < 0 ? 0 : q > 1023 ? 1023 : q);
	};

	for (int y = 0; y < h; y++) {
		const uint16_t *in = ctx->rgba + (size_t)y * w * 4;
		uint16_t *Y = ctx->plane[0] + (size_t)y * w;
		uint16_t *U = ctx->plane[1] + (size_t)y * cstride;
		uint16_t *V = ctx->plane[2] + (size_t)y * cstride;
		uint16_t *A = ctx->plane[3] ? ctx->plane[3] + (size_t)y * w : nullptr;
		float cbSum = 0.0f, crSum = 0.0f;
		for (int x = 0; x < w; x++, in += 4) {
			const float r = in[0] * inv, g = in[1] * inv, b = in[2] * inv;
			const float l = kr * r + kg * g + kb * b;
			const float cb = (b - l) * cbDiv, cr = (r - l) * crDiv;
			Y[x] = quant(yOff + yScale * l);
			if (A) A[x] = in[3];
			if (c444) {
				U[x] = quant(cOff + cScale * cb);
				V[x] = quant(cOff + cScale * cr);
			} else if (!(x & 1)) {
				cbSum = cb;
				crSum = cr;
				if (x == w - 1) {
					U[x >> 1] = quant(cOff + cScale * cb);
					V[x >> 1] = quant(cOff + cScale * cr);
				}
			} else {
				U[x >> 1] = quant(cOff + cScale * 0.5f * (cbSum + cb));
				V[x >> 1] = quant(cOff + cScale * 0.5f * (crSum + cr));
			}
		}
	}
	ctx->alpha16 = 1;
}

}	// extern "C"
