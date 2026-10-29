/*
 * Apple ProRes (iCodec Pro) decoder — 'apco' Proxy, 'apcs' LT, 'apcn' 422, 'apch' HQ,
 * 'ap4h' 4444, 'ap4x' 4444 XQ.
 *
 * A port of FFmpeg's libavcodec ProRes decoder (proresdec.c, proresdsp.c, proresdata.c and
 * the 10-bit extra-shift / 12-bit instances of simple_idct_template.c, release 8.1):
 *
 *   Copyright (c) 2010-2011 Maxim Poliakovski
 *   Copyright (c) 2010-2011 Elvis Presley
 *   Copyright (c) 2000, 2001 Michael Niedermayer (simple_idct_template.c)
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
 * The libavcodec framework (AVCodecContext / AVFrame / slice threading / hwaccel) is
 * replaced by a self-contained handle (ProresCtx) owning the planes of one frame. The
 * bit reader reproduces FFmpeg's non-cached reader exactly — a 32-bit cache refilled by
 * UPDATE_CACHE_32, zeros shifted in by SKIP_BITS, the index clamped to size + 8 (the safe
 * reader) — so bit positions past a slice's end read what FFmpeg reads. Planes are
 * yuv422p10 / yuv444p10 / yuv(a)444p12 as FFmpeg outputs them, and bit-exact to
 * `ffmpeg -cpuflags 0 -c:v prores`.
 *
 * prores_to_rgba16 converts the planes to straight-alpha RGBA16 (limited range, the
 * matrix the caller names), since the planar YCbCr upload path takes 8-bit planes only.
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifdef __wasm_simd128__
#include <wasm_simd128.h>
#include "../common/wasm_simd_idct.h"	// wsi_load16x4 / wsi_transpose4_i32 / wsi_store_trunc16
#endif

#define PADDING 64	// AV_INPUT_BUFFER_PADDING_SIZE: zeroed bytes behind the packet

// SIMD kernels on by default; prores_set_simd(0|1) switches to the scalar ones for the
// A/B conformance check. Both are byte-identical.
static int g_simd = 1;

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

static const uint8_t prores_interlaced_scan[64] = {
	 0,  8,  1,  9, 16, 24, 17, 25,
	 2, 10,  3, 11, 18, 26, 19, 27,
	32, 40, 33, 34, 41, 48, 56, 49,
	42, 35, 43, 50, 57, 58, 51, 59,
	 4, 12,  5,  6, 13, 20, 28, 21,
	14,  7, 15, 22, 29, 36, 44, 37,
	30, 23, 31, 38, 45, 52, 60, 53,
	46, 39, 47, 54, 61, 62, 55, 63
};

#define FIRST_DC_CB 0xB8	// rice_order = 5, exp_golomb_order = 6, switch_bits = 0

static const uint8_t dc_codebook[7] = { 0x04, 0x28, 0x28, 0x4D, 0x4D, 0x70, 0x70 };

// adaptive codebook switching lut according to previous run/level values
static const uint8_t run_to_cb[16] = { 0x06, 0x06, 0x05, 0x05, 0x04, 0x29, 0x29, 0x29, 0x29, 0x28, 0x28, 0x28, 0x28, 0x28, 0x28, 0x4C };
static const uint8_t lev_to_cb[10] = { 0x04, 0x0A, 0x05, 0x06, 0x04, 0x28, 0x28, 0x28, 0x28, 0x4C };

// ---------------------------------------------------------------- helpers

static inline uint32_t rb16(const uint8_t *p) { return (uint32_t)p[0] << 8 | p[1]; }
static inline uint32_t rb32(const uint8_t *p) { return (uint32_t)p[0] << 24 | (uint32_t)p[1] << 16 | (uint32_t)p[2] << 8 | p[3]; }
static inline uint64_t rb64(const uint8_t *p) { return (uint64_t)rb32(p) << 32 | rb32(p + 4); }
static inline int clip(int v, int lo, int hi) { return v < lo ? lo : v > hi ? hi : v; }
static inline int av_log2(uint32_t v) { return 31 - __builtin_clz(v | 1); }
static inline int min_i(int a, int b) { return a < b ? a : b; }

// ---------------------------------------------------------------- bit reader (get_bits.h)

struct GetBits {
	const uint8_t *buffer;
	int index;
	int size_in_bits;
	int size_in_bits_plus8;
};

static void init_get_bits(GetBits *gb, const uint8_t *buf, int bit_size)
{
	gb->buffer = buf;
	gb->index = 0;
	gb->size_in_bits = bit_size;
	gb->size_in_bits_plus8 = bit_size + 8;
}

static inline int get_bits_left(const GetBits *gb) { return gb->size_in_bits - gb->index; }

static inline unsigned get_bits(GetBits *gb, int n)
{
	const uint32_t cache = rb32(gb->buffer + (gb->index >> 3)) << (gb->index & 7);
	const unsigned v = cache >> (32 - n);
	gb->index = min_i(gb->size_in_bits_plus8, gb->index + n);
	return v;
}

static inline unsigned get_bits1(GetBits *gb)
{
	const unsigned v = (uint8_t)(gb->buffer[gb->index >> 3] << (gb->index & 7)) >> 7;
	if (gb->index < gb->size_in_bits_plus8) gb->index++;
	return v;
}

// The OPEN_READER / UPDATE_CACHE_32 / SKIP_BITS reader the coefficient decoders use.
struct Reader {
	int index;
	uint32_t cache;
};

#define OPEN_READER(re, gb)        Reader re = { (gb)->index, 0 }
#define CLOSE_READER(re, gb)       ((gb)->index = re.index)
#define UPDATE_CACHE_32(re, gb)    (re.cache = (uint32_t)(rb64((gb)->buffer + (re.index >> 3)) << (re.index & 7) >> 32))
#define GET_CACHE(re, gb)          (re.cache)
#define SHOW_UBITS(re, gb, n)      (re.cache >> (32 - (n)))
#define SHOW_SBITS(re, gb, n)      ((int32_t)re.cache >> (32 - (n)))
#define SKIP_COUNTER(re, gb, n)    (re.index = min_i((gb)->size_in_bits_plus8, re.index + (n)))
#define SKIP_BITS(re, gb, n)       do { re.cache <<= (n); SKIP_COUNTER(re, gb, n); } while (0)
#define LAST_SKIP_BITS(re, gb, n)  SKIP_COUNTER(re, gb, n)

// ---------------------------------------------------------------- IDCT (simple_idct_template.c)

// One instance per bit depth: 10-bit with EXTRA_SHIFT (ROW 13, COL 18) and 12-bit (ROW 16,
// COL 17), IN_IDCT_DEPTH 16. The DC-only row shortcut is the 32-bit form; the 64-bit one
// gives the same values.
template <int W1, int W2, int W3, int W4, int W5, int W6, int W7, int ROW_SHIFT, int COL_SHIFT>
struct SimpleIdct {
	static inline void row(int16_t *row, int extra_shift)
	{
		unsigned a0, a1, a2, a3, b0, b1, b2, b3;

		if (!(row[1] | row[2] | row[3] | row[4] | row[5] | row[6] | row[7])) {
			// DC_SHIFT - extra_shift < 0 for both instances (1 - 2, -1 - 0)
			const int16_t temp = (int16_t)((row[0] + 1) >> 1);
			for (int i = 0; i < 8; i++) row[i] = temp;
			return;
		}

		a0 = ((unsigned)W4 * row[0]) + (1 << (ROW_SHIFT + extra_shift - 1));
		a1 = a0;
		a2 = a0;
		a3 = a0;

		a0 += (unsigned)W2 * row[2];
		a1 += (unsigned)W6 * row[2];
		a2 -= (unsigned)W6 * row[2];
		a3 -= (unsigned)W2 * row[2];

		b0 = (unsigned)W1 * row[1];
		b0 += (unsigned)W3 * row[3];
		b1 = (unsigned)W3 * row[1];
		b1 += (unsigned)-W7 * row[3];
		b2 = (unsigned)W5 * row[1];
		b2 += (unsigned)-W1 * row[3];
		b3 = (unsigned)W7 * row[1];
		b3 += (unsigned)-W5 * row[3];

		if (row[4] | row[5] | row[6] | row[7]) {
			a0 += (unsigned)  W4*row[4] + (unsigned)W6*row[6];
			a1 += (unsigned)- W4*row[4] - (unsigned)W2*row[6];
			a2 += (unsigned)- W4*row[4] + (unsigned)W2*row[6];
			a3 += (unsigned)  W4*row[4] - (unsigned)W6*row[6];

			b0 += (unsigned) W5 * row[5];
			b0 += (unsigned) W7 * row[7];

			b1 += (unsigned)-W1 * row[5];
			b1 += (unsigned)-W5 * row[7];

			b2 += (unsigned) W7 * row[5];
			b2 += (unsigned) W3 * row[7];

			b3 += (unsigned) W3 * row[5];
			b3 += (unsigned)-W1 * row[7];
		}

		const int s = ROW_SHIFT + extra_shift;
		row[0] = (int)(a0 + b0) >> s;
		row[7] = (int)(a0 - b0) >> s;
		row[1] = (int)(a1 + b1) >> s;
		row[6] = (int)(a1 - b1) >> s;
		row[2] = (int)(a2 + b2) >> s;
		row[5] = (int)(a2 - b2) >> s;
		row[3] = (int)(a3 + b3) >> s;
		row[4] = (int)(a3 - b3) >> s;
	}

	static inline void col(int16_t *col)
	{
		unsigned a0, a1, a2, a3, b0, b1, b2, b3;

		a0 = (unsigned)W4 * (col[8*0] + ((1<<(COL_SHIFT-1))/W4));
		a1 = a0;
		a2 = a0;
		a3 = a0;

		a0 += (unsigned) W2*col[8*2];
		a1 += (unsigned) W6*col[8*2];
		a2 += (unsigned)-W6*col[8*2];
		a3 += (unsigned)-W2*col[8*2];

		b0 = (unsigned)W1 * col[8*1];
		b1 = (unsigned)W3 * col[8*1];
		b2 = (unsigned)W5 * col[8*1];
		b3 = (unsigned)W7 * col[8*1];

		b0 += (unsigned) W3 * col[8*3];
		b1 += (unsigned)-W7 * col[8*3];
		b2 += (unsigned)-W1 * col[8*3];
		b3 += (unsigned)-W5 * col[8*3];

		if (col[8*4]) {
			a0 += (unsigned) W4*col[8*4];
			a1 += (unsigned)-W4*col[8*4];
			a2 += (unsigned)-W4*col[8*4];
			a3 += (unsigned) W4*col[8*4];
		}

		if (col[8*5]) {
			b0 += (unsigned) W5 * col[8*5];
			b1 += (unsigned)-W1 * col[8*5];
			b2 += (unsigned) W7 * col[8*5];
			b3 += (unsigned) W3 * col[8*5];
		}

		if (col[8*6]) {
			a0 += (unsigned) W6*col[8*6];
			a1 += (unsigned)-W2*col[8*6];
			a2 += (unsigned) W2*col[8*6];
			a3 += (unsigned)-W6*col[8*6];
		}

		if (col[8*7]) {
			b0 += (unsigned) W7 * col[8*7];
			b1 += (unsigned)-W5 * col[8*7];
			b2 += (unsigned) W3 * col[8*7];
			b3 += (unsigned)-W1 * col[8*7];
		}

		col[0 ] = ((int)(a0 + b0) >> COL_SHIFT);
		col[8 ] = ((int)(a1 + b1) >> COL_SHIFT);
		col[16] = ((int)(a2 + b2) >> COL_SHIFT);
		col[24] = ((int)(a3 + b3) >> COL_SHIFT);
		col[32] = ((int)(a3 - b3) >> COL_SHIFT);
		col[40] = ((int)(a2 - b2) >> COL_SHIFT);
		col[48] = ((int)(a1 - b1) >> COL_SHIFT);
		col[56] = ((int)(a0 - b0) >> COL_SHIFT);
	}
};

typedef SimpleIdct<22725, 21407, 19265, 16384, 12873, 8867, 4520, 13, 18> Idct10;
typedef SimpleIdct<45451, 42813, 38531, 32767, 25746, 17734, 9041, 16, 17> Idct12;

// ---------------------------------------------------------------- proresdsp.c

#define CLIP_MIN (1 << 2)                     // minimum value for clipping resulting pixels
#define CLIP_MAX_10 ((1 << 10) - CLIP_MIN - 1)  // maximum value for clipping resulting pixels
#define CLIP_MAX_12 ((1 << 12) - CLIP_MIN - 1)

// `linesize` is in pixels (FFmpeg's is in bytes and halved here).
static void prores_idct_put_10(uint16_t *out, int linesize, int16_t *block, const int16_t *qmat)
{
	for (int i = 0; i < 64; i++)
		block[i] *= qmat[i];

	for (int i = 0; i < 8; i++)
		Idct10::row(block + i*8, 2);

	for (int i = 0; i < 8; i++) {
		block[i] += 8192;
		Idct10::col(block + i);
	}

	for (int y = 0; y < 8; y++, out += linesize)
		for (int x = 0; x < 8; x++)
			out[x] = clip(block[(y << 3) + x], CLIP_MIN, CLIP_MAX_10);
}

static void prores_idct_put_12(uint16_t *out, int linesize, int16_t *block, const int16_t *qmat)
{
	for (int i = 0; i < 64; i++)
		block[i] *= qmat[i];

	for (int i = 0; i < 8; i++)
		Idct12::row(block + i*8, 0);

	for (int i = 0; i < 8; i++) {
		block[i] += 8192;
		Idct12::col(block + i);
	}

	for (int y = 0; y < 8; y++, out += linesize)
		for (int x = 0; x < 8; x++)
			out[x] = clip(block[(y << 3) + x], CLIP_MIN, CLIP_MAX_12);
}

#ifdef __wasm_simd128__
// ---------------------------------------------------------------- IDCT, WebAssembly i32x4
//
// Byte-identical to prores_idct_put_10/12 above, on the skeleton of the mpeg4 simple IDCT
// (plugins/mpeg4/simple_idct8.h): a row pass on 4 rows a vector, a 4x4 transpose, a column
// pass on 4 columns a vector, all i32x4 wrapping mod 2^32 like the scalar unsigned
// accumulators (-fwrapv). The scalar int16 steps are kept: the dequant multiply wraps in
// i16x8, the row results and the +8192 are truncated to int16, and so are the column
// results before the clip. Skipping a zero coefficient (the scalar `if (col[8*k])`) adds
// nothing, so the vectors always take every term. The row DC shortcut, (row[0] + 1) >> 1,
// equals the butterfly for 10 bits (W4 = 2^14) but not for 12 (W4 = 32767): DC-only rows
// are blended in per lane, as the scalar path takes it.
template <int W1, int W2, int W3, int W4, int W5, int W6, int W7, int ROW_SHIFT, int COL_SHIFT, int EXTRA_SHIFT, int CLIP_MAX>
struct SimdIdct {
	static inline v128_t imc(v128_t x, int c) { return wasm_i32x4_mul(x, wasm_i32x4_splat(c)); }
	static inline v128_t i16trunc(v128_t x) { return wasm_i32x4_shr(wasm_i32x4_shl(x, 16), 16); }

	// e[k]: coefficient k of 4 rows (one row a lane). o[k]: the int16-truncated outputs.
	static inline void row(const v128_t e[8], v128_t o[8])
	{
		const int S = ROW_SHIFT + EXTRA_SHIFT;
		v128_t a0 = wasm_i32x4_add(imc(e[0], W4), wasm_i32x4_splat(1 << (S - 1)));
		v128_t a1 = a0, a2 = a0, a3 = a0;
		a0 = wasm_i32x4_add(a0, imc(e[2], W2)); a1 = wasm_i32x4_add(a1, imc(e[2], W6));
		a2 = wasm_i32x4_sub(a2, imc(e[2], W6)); a3 = wasm_i32x4_sub(a3, imc(e[2], W2));
		a0 = wasm_i32x4_add(a0, wasm_i32x4_add(imc(e[4], W4), imc(e[6], W6)));
		a1 = wasm_i32x4_sub(a1, wasm_i32x4_add(imc(e[4], W4), imc(e[6], W2)));
		a2 = wasm_i32x4_add(wasm_i32x4_sub(a2, imc(e[4], W4)), imc(e[6], W2));
		a3 = wasm_i32x4_sub(wasm_i32x4_add(a3, imc(e[4], W4)), imc(e[6], W6));
		v128_t b0 = wasm_i32x4_add(imc(e[1], W1), imc(e[3], W3));
		v128_t b1 = wasm_i32x4_sub(imc(e[1], W3), imc(e[3], W7));
		v128_t b2 = wasm_i32x4_sub(imc(e[1], W5), imc(e[3], W1));
		v128_t b3 = wasm_i32x4_sub(imc(e[1], W7), imc(e[3], W5));
		b0 = wasm_i32x4_add(b0, wasm_i32x4_add(imc(e[5], W5), imc(e[7], W7)));
		b1 = wasm_i32x4_sub(b1, wasm_i32x4_add(imc(e[5], W1), imc(e[7], W5)));
		b2 = wasm_i32x4_add(b2, wasm_i32x4_add(imc(e[5], W7), imc(e[7], W3)));
		b3 = wasm_i32x4_add(b3, wasm_i32x4_sub(imc(e[5], W3), imc(e[7], W1)));
		v128_t f[8];
		f[0] = i16trunc(wasm_i32x4_shr(wasm_i32x4_add(a0, b0), S));
		f[7] = i16trunc(wasm_i32x4_shr(wasm_i32x4_sub(a0, b0), S));
		f[1] = i16trunc(wasm_i32x4_shr(wasm_i32x4_add(a1, b1), S));
		f[6] = i16trunc(wasm_i32x4_shr(wasm_i32x4_sub(a1, b1), S));
		f[2] = i16trunc(wasm_i32x4_shr(wasm_i32x4_add(a2, b2), S));
		f[5] = i16trunc(wasm_i32x4_shr(wasm_i32x4_sub(a2, b2), S));
		f[3] = i16trunc(wasm_i32x4_shr(wasm_i32x4_add(a3, b3), S));
		f[4] = i16trunc(wasm_i32x4_shr(wasm_i32x4_sub(a3, b3), S));
		const v128_t dc = i16trunc(wasm_i32x4_shr(wasm_i32x4_add(e[0], wasm_i32x4_splat(1)), 1));
		const v128_t ac = wasm_v128_or(wasm_v128_or(wasm_v128_or(e[1], e[2]), wasm_v128_or(e[3], e[4])),
		                               wasm_v128_or(wasm_v128_or(e[5], e[6]), e[7]));
		const v128_t dcOnly = wasm_i32x4_eq(ac, wasm_i32x4_splat(0));
		for (int k = 0; k < 8; k++) o[k] = wasm_v128_bitselect(dc, f[k], dcOnly);
	}

	// c[r]: row r of 4 columns (one column a lane). out[r]: the int16-truncated outputs.
	static inline void col(const v128_t c[8], v128_t out[8])
	{
		v128_t a0 = imc(wasm_i32x4_add(c[0], wasm_i32x4_splat((1 << (COL_SHIFT - 1)) / W4)), W4);
		v128_t a1 = a0, a2 = a0, a3 = a0;
		a0 = wasm_i32x4_add(a0, imc(c[2], W2)); a1 = wasm_i32x4_add(a1, imc(c[2], W6));
		a2 = wasm_i32x4_sub(a2, imc(c[2], W6)); a3 = wasm_i32x4_sub(a3, imc(c[2], W2));
		v128_t b0 = imc(c[1], W1), b1 = imc(c[1], W3), b2 = imc(c[1], W5), b3 = imc(c[1], W7);
		b0 = wasm_i32x4_add(b0, imc(c[3], W3)); b1 = wasm_i32x4_sub(b1, imc(c[3], W7));
		b2 = wasm_i32x4_sub(b2, imc(c[3], W1)); b3 = wasm_i32x4_sub(b3, imc(c[3], W5));
		a0 = wasm_i32x4_add(a0, imc(c[4], W4)); a1 = wasm_i32x4_sub(a1, imc(c[4], W4));
		a2 = wasm_i32x4_sub(a2, imc(c[4], W4)); a3 = wasm_i32x4_add(a3, imc(c[4], W4));
		b0 = wasm_i32x4_add(b0, imc(c[5], W5)); b1 = wasm_i32x4_sub(b1, imc(c[5], W1));
		b2 = wasm_i32x4_add(b2, imc(c[5], W7)); b3 = wasm_i32x4_add(b3, imc(c[5], W3));
		a0 = wasm_i32x4_add(a0, imc(c[6], W6)); a1 = wasm_i32x4_sub(a1, imc(c[6], W2));
		a2 = wasm_i32x4_add(a2, imc(c[6], W2)); a3 = wasm_i32x4_sub(a3, imc(c[6], W6));
		b0 = wasm_i32x4_add(b0, imc(c[7], W7)); b1 = wasm_i32x4_sub(b1, imc(c[7], W5));
		b2 = wasm_i32x4_add(b2, imc(c[7], W3)); b3 = wasm_i32x4_sub(b3, imc(c[7], W1));
		const int S = COL_SHIFT;
		out[0] = i16trunc(wasm_i32x4_shr(wasm_i32x4_add(a0, b0), S));
		out[7] = i16trunc(wasm_i32x4_shr(wasm_i32x4_sub(a0, b0), S));
		out[1] = i16trunc(wasm_i32x4_shr(wasm_i32x4_add(a1, b1), S));
		out[6] = i16trunc(wasm_i32x4_shr(wasm_i32x4_sub(a1, b1), S));
		out[2] = i16trunc(wasm_i32x4_shr(wasm_i32x4_add(a2, b2), S));
		out[5] = i16trunc(wasm_i32x4_shr(wasm_i32x4_sub(a2, b2), S));
		out[3] = i16trunc(wasm_i32x4_shr(wasm_i32x4_add(a3, b3), S));
		out[4] = i16trunc(wasm_i32x4_shr(wasm_i32x4_sub(a3, b3), S));
	}

	static inline void put(uint16_t *out, int linesize, int16_t *block, const int16_t *qmat)
	{
		for (int i = 0; i < 64; i += 8)
			wasm_v128_store(block + i, wasm_i16x8_mul(wasm_v128_load(block + i), wasm_v128_load(qmat + i)));

		int16_t tmp[64];
		for (int rb = 0; rb < 8; rb += 4) {
			v128_t e[8];
			wsi_transpose4_i32(wsi_load16x4(block + (rb + 0) * 8), wsi_load16x4(block + (rb + 1) * 8),
			                   wsi_load16x4(block + (rb + 2) * 8), wsi_load16x4(block + (rb + 3) * 8),
			                   &e[0], &e[1], &e[2], &e[3]);
			wsi_transpose4_i32(wsi_load16x4(block + (rb + 0) * 8 + 4), wsi_load16x4(block + (rb + 1) * 8 + 4),
			                   wsi_load16x4(block + (rb + 2) * 8 + 4), wsi_load16x4(block + (rb + 3) * 8 + 4),
			                   &e[4], &e[5], &e[6], &e[7]);
			v128_t o[8];
			row(e, o);
			v128_t r[8];
			wsi_transpose4_i32(o[0], o[1], o[2], o[3], &r[0], &r[1], &r[2], &r[3]);	// rows rb..rb+3, columns 0-3
			wsi_transpose4_i32(o[4], o[5], o[6], o[7], &r[4], &r[5], &r[6], &r[7]);	// rows rb..rb+3, columns 4-7
			for (int k = 0; k < 4; k++) {
				wsi_store_trunc16(tmp + (rb + k) * 8, r[k]);
				wsi_store_trunc16(tmp + (rb + k) * 8 + 4, r[k + 4]);
			}
		}

		const v128_t lo = wasm_i32x4_splat(CLIP_MIN), hi = wasm_i32x4_splat(CLIP_MAX);
		for (int cb = 0; cb < 8; cb += 4) {
			v128_t c[8];
			c[0] = i16trunc(wasm_i32x4_add(wsi_load16x4(tmp + cb), wasm_i32x4_splat(8192)));
			for (int r = 1; r < 8; r++) c[r] = wsi_load16x4(tmp + r * 8 + cb);
			v128_t o[8];
			col(c, o);
			for (int r = 0; r < 8; r++) {
				const v128_t v = wasm_i32x4_min(wasm_i32x4_max(o[r], lo), hi);
				wasm_v128_store64_lane(out + r * linesize + cb, wasm_u16x8_narrow_i32x4(v, v), 0);
			}
		}
	}
};

typedef SimdIdct<22725, 21407, 19265, 16384, 12873, 8867, 4520, 13, 18, 2, CLIP_MAX_10> SimdIdct10;
typedef SimdIdct<45451, 42813, 38531, 32767, 25746, 17734, 9041, 16, 17, 0, CLIP_MAX_12> SimdIdct12;
#endif

// ---------------------------------------------------------------- context

struct SliceContext {
	const uint8_t *data;
	unsigned mb_x;
	unsigned mb_y;
	unsigned mb_count;
	unsigned data_size;
	int ret;
};

struct ProresCtx {
	int bits;                    // 10 ('apc*') or 12 ('ap4*'), from the sample-entry fourcc
	int width, height;
	int frame_type;              // 0 = progressive, 1 = tff, 2 = bff
	int top_field_first;
	int chroma444;
	int alpha_info;              // 0 none, 1 8-bit, 2 16-bit
	int color_primaries, color_trc, colorspace;
	uint8_t qmat_luma[64];
	uint8_t qmat_chroma[64];
	SliceContext *slices;
	int slice_count;
	unsigned mb_width, mb_height;
	unsigned slice_mb_width, slice_mb_height;
	const uint8_t *scan;
	int first_field;

	// One frame: Y, Cb, Cr, A, each `stride[i]` pixels a row (FFmpeg's linesize / 2).
	uint16_t *plane[4];
	int stride[4];
	int alloc_w, alloc_h, alloc_444;

	uint8_t *in;                 // packet buffer with PADDING zeroed bytes behind it
	int in_cap;
	uint16_t *rgba;
	int rgba_cap;
};

static void free_planes(ProresCtx *ctx)
{
	for (int i = 0; i < 4; i++) { free(ctx->plane[i]); ctx->plane[i] = nullptr; ctx->stride[i] = 0; }
	ctx->alloc_w = ctx->alloc_h = 0;
}

// Planes cover whole macroblocks: 16-pixel columns, and 32 rows so an interlaced frame's
// two fields (each mb_height = (h + 31) >> 5 macroblock rows at twice the stride) fit.
static int alloc_planes(ProresCtx *ctx)
{
	const int w = (ctx->width + 15) & ~15;
	const int h = (ctx->height + 31) & ~31;
	if (w == 0) return -1;
	if (ctx->plane[0] && w == ctx->alloc_w && h == ctx->alloc_h && ctx->chroma444 == ctx->alloc_444)
		return 0;
	free_planes(ctx);
	const int cw = ctx->chroma444 ? w : w >> 1;
	const int widths[4] = { w, cw, cw, w };
	for (int i = 0; i < 4; i++) {
		ctx->plane[i] = (uint16_t *)calloc((size_t)widths[i] * h, sizeof(uint16_t));
		if (!ctx->plane[i]) { free_planes(ctx); return -1; }
		ctx->stride[i] = widths[i];
	}
	ctx->alloc_w = w;
	ctx->alloc_h = h;
	ctx->alloc_444 = ctx->chroma444;
	return 0;
}

// ---------------------------------------------------------------- alpha (proresdec.c)

#define ALPHA_SHIFT_16_TO_10(alpha_val) (alpha_val >> 6)
#define ALPHA_SHIFT_8_TO_10(alpha_val)  ((alpha_val << 2) | (alpha_val >> 6))
#define ALPHA_SHIFT_16_TO_12(alpha_val) (alpha_val >> 4)
#define ALPHA_SHIFT_8_TO_12(alpha_val)  ((alpha_val << 4) | (alpha_val >> 4))

static inline void unpack_alpha(GetBits *gb, uint16_t *dst, int num_coeffs,
                                const int num_bits, const int decode_precision)
{
	const int mask = (1 << num_bits) - 1;
	int i, idx, val, alpha_val;

	idx       = 0;
	alpha_val = mask;
	do {
		do {
			if (get_bits1(gb)) {
				val = get_bits(gb, num_bits);
			} else {
				int sign;
				val  = get_bits(gb, num_bits == 16 ? 7 : 4);
				sign = val & 1;
				val  = (val + 2) >> 1;
				if (sign)
					val = -val;
			}
			alpha_val = (alpha_val + val) & mask;
			if (num_bits == 16) {
				if (decode_precision == 10) {
					dst[idx++] = ALPHA_SHIFT_16_TO_10(alpha_val);
				} else { /* 12b */
					dst[idx++] = ALPHA_SHIFT_16_TO_12(alpha_val);
				}
			} else {
				if (decode_precision == 10) {
					dst[idx++] = ALPHA_SHIFT_8_TO_10(alpha_val);
				} else { /* 12b */
					dst[idx++] = ALPHA_SHIFT_8_TO_12(alpha_val);
				}
			}
			if (idx >= num_coeffs)
				break;
		} while (get_bits_left(gb)>0 && get_bits1(gb));
		val = get_bits(gb, 4);
		if (!val)
			val = get_bits(gb, 11);
		if (idx + val > num_coeffs)
			val = num_coeffs - idx;
		if (num_bits == 16) {
			for (i = 0; i < val; i++) {
				if (decode_precision == 10) {
					dst[idx++] = ALPHA_SHIFT_16_TO_10(alpha_val);
				} else { /* 12b */
					dst[idx++] = ALPHA_SHIFT_16_TO_12(alpha_val);
				}
			}
		} else {
			for (i = 0; i < val; i++) {
				if (decode_precision == 10) {
					dst[idx++] = ALPHA_SHIFT_8_TO_10(alpha_val);
				} else { /* 12b */
					dst[idx++] = ALPHA_SHIFT_8_TO_12(alpha_val);
				}
			}
		}
	} while (idx < num_coeffs);
}

// ---------------------------------------------------------------- headers

static int decode_frame_header(ProresCtx *ctx, const uint8_t *buf, const int data_size)
{
	int hdr_size, width, height, flags, version;
	const uint8_t *ptr;

	hdr_size = rb16(buf);
	if (hdr_size > data_size)
		return -1;	// wrong header size

	version = rb16(buf + 2);
	if (version > 1)
		return -2;	// unsupported version

	width  = rb16(buf + 8);
	height = rb16(buf + 10);
	if (!width || !height)
		return -1;
	ctx->width = width;
	ctx->height = height;

	ctx->frame_type = (buf[12] >> 2) & 3;
	ctx->alpha_info = buf[17] & 0xf;

	if (ctx->alpha_info > 2)
		return -1;	// invalid alpha mode

	if (ctx->frame_type == 0) {
		ctx->scan = prores_progressive_scan;
		ctx->top_field_first = 0;
	} else {
		ctx->scan = prores_interlaced_scan;
		ctx->top_field_first = ctx->frame_type == 1;
	}

	ctx->chroma444 = (buf[12] & 0xC0) == 0xC0;

	ctx->color_primaries = buf[14];
	ctx->color_trc       = buf[15];
	ctx->colorspace      = buf[16];

	ptr   = buf + 20;
	flags = buf[19];

	// FF_IDCT_PERM_NONE: the scantable permutations are identities.
	if (flags & 2) {
		if (buf + data_size - ptr < 64)
			return -1;	// header truncated
		memcpy(ctx->qmat_luma, ptr, 64);
		ptr += 64;
	} else {
		memset(ctx->qmat_luma, 4, 64);
	}

	if (flags & 1) {
		if (buf + data_size - ptr < 64)
			return -1;	// header truncated
		memcpy(ctx->qmat_chroma, ptr, 64);
	} else {
		memcpy(ctx->qmat_chroma, ctx->qmat_luma, 64);
	}

	return hdr_size;
}

static int decode_picture_header(ProresCtx *ctx, const uint8_t *buf, const int buf_size)
{
	int i, hdr_size, slice_count;
	unsigned pic_data_size;
	int log2_slice_mb_width, log2_slice_mb_height;
	int slice_mb_count, mb_x, mb_y;
	const uint8_t *data_ptr, *index_ptr;

	hdr_size = buf[0] >> 3;
	if (hdr_size < 8 || hdr_size > buf_size)
		return -1;	// wrong picture header size

	pic_data_size = rb32(buf + 1);
	if (pic_data_size > (unsigned)buf_size)
		return -1;	// wrong picture data size

	log2_slice_mb_width  = buf[7] >> 4;
	log2_slice_mb_height = buf[7] & 0xF;
	if (log2_slice_mb_width > 3 || log2_slice_mb_height)
		return -1;	// unsupported slice resolution

	ctx->slice_mb_width  = 1 << log2_slice_mb_width;
	ctx->slice_mb_height = 1 << log2_slice_mb_height;

	ctx->mb_width  = (ctx->width  + 15) >> 4;
	if (ctx->frame_type)
		ctx->mb_height = (ctx->height + 31) >> 5;
	else
		ctx->mb_height = (ctx->height + 15) >> 4;

	// QT ignores the written value
	// slice_count = AV_RB16(buf + 5);
	slice_count = ctx->mb_height * ((ctx->mb_width >> log2_slice_mb_width) +
	                                __builtin_popcount(ctx->mb_width & (ctx->slice_mb_width - 1)));

	if (ctx->slice_count != slice_count || !ctx->slices) {
		free(ctx->slices);
		ctx->slice_count = 0;
		ctx->slices = (SliceContext *)calloc(slice_count ? slice_count : 1, sizeof(*ctx->slices));
		if (!ctx->slices)
			return -3;
		ctx->slice_count = slice_count;
	}

	if (!slice_count)
		return -1;

	if (hdr_size + slice_count*2 > buf_size)
		return -1;	// wrong slice count

	// parse slice information
	index_ptr = buf + hdr_size;
	data_ptr  = index_ptr + slice_count*2;

	slice_mb_count = ctx->slice_mb_width;
	mb_x = 0;
	mb_y = 0;

	for (i = 0; i < slice_count; i++) {
		SliceContext *slice = &ctx->slices[i];

		slice->data = data_ptr;
		data_ptr += rb16(index_ptr + i*2);

		while ((int)ctx->mb_width - mb_x < slice_mb_count)
			slice_mb_count >>= 1;

		slice->mb_x = mb_x;
		slice->mb_y = mb_y;
		slice->mb_count = slice_mb_count;
		slice->data_size = data_ptr - slice->data;

		if (slice->data_size < 6)
			return -1;	// wrong slice data size

		mb_x += slice_mb_count;
		if (mb_x == (int)ctx->mb_width) {
			slice_mb_count = ctx->slice_mb_width;
			mb_x = 0;
			mb_y++;
		}
		if (data_ptr > buf + buf_size)
			return -1;	// slice out of bounds
	}

	if (mb_x || mb_y != (int)ctx->mb_height)
		return -1;	// wrong mb count

	return pic_data_size;
}

// ---------------------------------------------------------------- coefficients

#define DECODE_CODEWORD(val, codebook, SKIP)                            \
	do {                                                                \
		unsigned int rice_order, exp_order, switch_bits;                \
		unsigned int q, buf, bits;                                      \
		                                                                \
		UPDATE_CACHE_32(re, gb); /* We really need 32 bits */           \
		buf = GET_CACHE(re, gb);                                        \
		                                                                \
		/* number of bits to switch between rice and exp golomb */      \
		switch_bits =  codebook & 3;                                    \
		rice_order  =  codebook >> 5;                                   \
		exp_order   = (codebook >> 2) & 7;                              \
		                                                                \
		q = 31 - av_log2(buf);                                          \
		                                                                \
		if (q > switch_bits) { /* exp golomb */                         \
			bits = exp_order - switch_bits + (q<<1);                    \
			if (bits > 31)                                              \
				return -1;                                              \
			val = SHOW_UBITS(re, gb, bits) - (1 << exp_order) +         \
				((switch_bits + 1) << rice_order);                      \
			SKIP(re, gb, bits);                                         \
		} else if (rice_order) {                                        \
			SKIP_BITS(re, gb, q+1);                                     \
			val = (q << rice_order) + SHOW_UBITS(re, gb, rice_order);   \
			SKIP(re, gb, rice_order);                                   \
		} else {                                                        \
			val = q;                                                    \
			SKIP(re, gb, q+1);                                          \
		}                                                               \
	} while (0)

#define TOSIGNED(x) (((x) >> 1) ^ (-((x) & 1)))

static inline int decode_dc_coeffs(GetBits *gb, int16_t *out, int blocks_per_slice)
{
	int16_t prev_dc;
	int code, i, sign;

	OPEN_READER(re, gb);

	DECODE_CODEWORD(code, FIRST_DC_CB, LAST_SKIP_BITS);
	prev_dc = TOSIGNED(code);
	out[0] = prev_dc;

	out += 64; // dc coeff for the next block

	code = 5;
	sign = 0;
	for (i = 1; i < blocks_per_slice; i++, out += 64) {
		DECODE_CODEWORD(code, dc_codebook[(unsigned)code < 6U ? code : 6], LAST_SKIP_BITS);
		if(code) sign ^= -(code & 1);
		else     sign  = 0;
		prev_dc += (((code + 1) >> 1) ^ sign) - sign;
		out[0] = prev_dc;
	}
	CLOSE_READER(re, gb);
	return 0;
}

static inline int decode_ac_coeffs(const ProresCtx *ctx, GetBits *gb, int16_t *out, int blocks_per_slice)
{
	int block_mask, sign;
	unsigned pos, run, level;
	int max_coeffs, i, bits_left;
	int log2_block_count = av_log2(blocks_per_slice);

	OPEN_READER(re, gb);
	UPDATE_CACHE_32(re, gb);
	run   = 4;
	level = 2;

	max_coeffs = 64 << log2_block_count;
	block_mask = blocks_per_slice - 1;

	for (pos = block_mask;;) {
		bits_left = gb->size_in_bits - re.index;
		if (bits_left <= 0 || (bits_left < 32 && !SHOW_UBITS(re, gb, bits_left)))
			break;

		DECODE_CODEWORD(run, run_to_cb[run < 15 ? run : 15], LAST_SKIP_BITS);
		pos += run + 1;
		if (pos >= (unsigned)max_coeffs)
			return -1;	// ac tex damaged

		DECODE_CODEWORD(level, lev_to_cb[level < 9 ? level : 9], SKIP_BITS);
		level += 1;

		i = pos >> log2_block_count;

		sign = SHOW_SBITS(re, gb, 1);
		SKIP_BITS(re, gb, 1);
		out[((pos & block_mask) << 6) + ctx->scan[i]] = ((level ^ sign) - sign);
	}

	CLOSE_READER(re, gb);
	return 0;
}

// ---------------------------------------------------------------- slices

static inline void idct_put(const ProresCtx *ctx, uint16_t *out, int linesize, int16_t *block, const int16_t *qmat)
{
#ifdef __wasm_simd128__
	if (g_simd) {
		if (ctx->bits == 10) SimdIdct10::put(out, linesize, block, qmat);
		else                 SimdIdct12::put(out, linesize, block, qmat);
		return;
	}
#endif
	if (ctx->bits == 10) prores_idct_put_10(out, linesize, block, qmat);
	else                 prores_idct_put_12(out, linesize, block, qmat);
}

// Strides below are in pixels; FFmpeg's `dst + 4*dst_stride` with a byte stride on a
// uint16_t pointer is 8 rows down, written here as `dst + 8*dst_stride`.
static int decode_slice_luma(const ProresCtx *ctx, SliceContext *slice,
                             uint16_t *dst, int dst_stride,
                             const uint8_t *buf, unsigned buf_size,
                             const int16_t *qmat)
{
	alignas(32) int16_t blocks[8*4*64];
	int16_t *block;
	GetBits gb;
	int i, blocks_per_slice = slice->mb_count<<2;
	int ret;

	memset(blocks, 0, blocks_per_slice * 64 * sizeof(int16_t));

	init_get_bits(&gb, buf, buf_size << 3);

	if ((ret = decode_dc_coeffs(&gb, blocks, blocks_per_slice)) < 0)
		return ret;
	if ((ret = decode_ac_coeffs(ctx, &gb, blocks, blocks_per_slice)) < 0)
		return ret;

	block = blocks;
	for (i = 0; i < (int)slice->mb_count; i++) {
		idct_put(ctx, dst,                  dst_stride, block+(0<<6), qmat);
		idct_put(ctx, dst               +8, dst_stride, block+(1<<6), qmat);
		idct_put(ctx, dst+8*dst_stride    , dst_stride, block+(2<<6), qmat);
		idct_put(ctx, dst+8*dst_stride  +8, dst_stride, block+(3<<6), qmat);
		block += 4*64;
		dst += 16;
	}
	return 0;
}

static int decode_slice_chroma(const ProresCtx *ctx, SliceContext *slice,
                               uint16_t *dst, int dst_stride,
                               const uint8_t *buf, unsigned buf_size,
                               const int16_t *qmat, int log2_blocks_per_mb)
{
	alignas(32) int16_t blocks[8*4*64];
	int16_t *block;
	GetBits gb;
	int i, j, blocks_per_slice = slice->mb_count << log2_blocks_per_mb;
	int ret;

	memset(blocks, 0, blocks_per_slice * 64 * sizeof(int16_t));

	/* Some encodes have empty chroma scans to simulate grayscale */
	if (buf_size) {
		init_get_bits(&gb, buf, buf_size << 3);

		if ((ret = decode_dc_coeffs(&gb, blocks, blocks_per_slice)) < 0)
			return ret;
		if ((ret = decode_ac_coeffs(ctx, &gb, blocks, blocks_per_slice)) < 0)
			return ret;
	}

	block = blocks;
	for (i = 0; i < (int)slice->mb_count; i++) {
		for (j = 0; j < log2_blocks_per_mb; j++) {
			idct_put(ctx, dst,                dst_stride, block+(0<<6), qmat);
			idct_put(ctx, dst+8*dst_stride,   dst_stride, block+(1<<6), qmat);
			block += 2*64;
			dst += 8;
		}
	}
	return 0;
}

static void decode_slice_alpha(const ProresCtx *ctx,
                               uint16_t *dst, int dst_stride,
                               const uint8_t *buf, int buf_size,
                               int blocks_per_slice)
{
	GetBits gb;
	int i;
	alignas(32) uint16_t blocks[8*4*64];
	uint16_t *block;

	memset(blocks, 0, (blocks_per_slice << 2) * 64 * sizeof(uint16_t));

	init_get_bits(&gb, buf, buf_size << 3);

	unpack_alpha(&gb, blocks, blocks_per_slice * 4 * 64, ctx->alpha_info == 2 ? 16 : 8, ctx->bits);

	block = blocks;

	for (i = 0; i < 16; i++) {
		memcpy(dst, block, 16 * blocks_per_slice * sizeof(*dst));
		dst   += dst_stride;
		block += 16 * blocks_per_slice;
	}
}

static int decode_slice(ProresCtx *ctx, SliceContext *slice)
{
	const uint8_t *buf = slice->data;
	int i, hdr_size, qscale, log2_chroma_blocks_per_mb;
	int luma_stride, chroma_stride;
	int y_data_size, u_data_size, v_data_size, a_data_size, offset;
	uint16_t *dest_y, *dest_u, *dest_v;
	alignas(16) int16_t qmat_luma_scaled[64];
	alignas(16) int16_t qmat_chroma_scaled[64];
	int mb_x_shift;
	int ret;

	slice->ret = -1;

	// slice header
	hdr_size = buf[0] >> 3;
	qscale = clip(buf[1], 1, 224);
	qscale = qscale > 128 ? (qscale - 96) << 2 : qscale;
	y_data_size = rb16(buf + 2);
	u_data_size = rb16(buf + 4);
	v_data_size = slice->data_size - y_data_size - u_data_size - hdr_size;
	if (hdr_size > 7) v_data_size = rb16(buf + 6);
	a_data_size = slice->data_size - y_data_size - u_data_size -
	              v_data_size - hdr_size;

	if (y_data_size < 0 || u_data_size < 0 || v_data_size < 0
	    || hdr_size+y_data_size+u_data_size+v_data_size > (int)slice->data_size)
		return -1;	// invalid plane data size

	buf += hdr_size;

	for (i = 0; i < 64; i++) {
		qmat_luma_scaled  [i] = ctx->qmat_luma  [i] * qscale;
		qmat_chroma_scaled[i] = ctx->qmat_chroma[i] * qscale;
	}

	if (ctx->frame_type == 0) {
		luma_stride   = ctx->stride[0];
		chroma_stride = ctx->stride[1];
	} else {
		luma_stride   = ctx->stride[0] << 1;
		chroma_stride = ctx->stride[1] << 1;
	}

	if (ctx->chroma444) {
		mb_x_shift = 4;	// FFmpeg's 5, in bytes
		log2_chroma_blocks_per_mb = 2;
	} else {
		mb_x_shift = 3;
		log2_chroma_blocks_per_mb = 1;
	}

	offset = (slice->mb_y << 4) * luma_stride + (slice->mb_x << 4);
	dest_y = ctx->plane[0] + offset;
	dest_u = ctx->plane[1] + (slice->mb_y << 4) * chroma_stride + (slice->mb_x << mb_x_shift);
	dest_v = ctx->plane[2] + (slice->mb_y << 4) * chroma_stride + (slice->mb_x << mb_x_shift);

	if (ctx->frame_type && ctx->first_field ^ ctx->top_field_first) {
		dest_y += ctx->stride[0];
		dest_u += ctx->stride[1];
		dest_v += ctx->stride[2];
		offset += ctx->stride[3];
	}

	ret = decode_slice_luma(ctx, slice, dest_y, luma_stride,
	                        buf, y_data_size, qmat_luma_scaled);
	if (ret < 0)
		return ret;

	ret = decode_slice_chroma(ctx, slice, dest_u, chroma_stride,
	                          buf + y_data_size, u_data_size,
	                          qmat_chroma_scaled, log2_chroma_blocks_per_mb);
	if (ret < 0)
		return ret;

	ret = decode_slice_chroma(ctx, slice, dest_v, chroma_stride,
	                          buf + y_data_size + u_data_size, v_data_size,
	                          qmat_chroma_scaled, log2_chroma_blocks_per_mb);
	if (ret < 0)
		return ret;

	/* decode alpha plane if available */
	if (ctx->alpha_info && a_data_size) {
		uint16_t *dest_a = ctx->plane[3] + offset;
		decode_slice_alpha(ctx, dest_a, luma_stride,
		                   buf + y_data_size + u_data_size + v_data_size,
		                   a_data_size, slice->mb_count);
	}

	slice->ret = 0;
	return 0;
}

// A damaged slice leaves its area as it was; the picture fails only when every slice did.
static int decode_picture(ProresCtx *ctx)
{
	int error = 0;

	for (int i = 0; i < ctx->slice_count; i++)
		error += decode_slice(ctx, &ctx->slices[i]) < 0;

	if (error < ctx->slice_count)
		return 0;

	return ctx->slices[0].ret;
}

// ---------------------------------------------------------------- API

extern "C" {

// bits: 10 for 'apco'/'apcs'/'apcn'/'apch', 12 for 'ap4h'/'ap4x' (FFmpeg's decode_init).
ProresCtx *prores_open(int bits)
{
	ProresCtx *ctx = (ProresCtx *)calloc(1, sizeof(ProresCtx));
	if (ctx) ctx->bits = bits == 12 ? 12 : 10;
	return ctx;
}

void prores_close(ProresCtx *ctx)
{
	if (!ctx) return;
	free_planes(ctx);
	free(ctx->slices);
	free(ctx->in);
	free(ctx->rgba);
	free(ctx);
}

// The input buffer a packet of `len` bytes is copied into before prores_decode_frame /
// prores_parse_header; it keeps PADDING zeroed bytes behind the packet, as FFmpeg's do.
uint8_t *prores_input(ProresCtx *ctx, int len)
{
	if (len < 0) return nullptr;
	if (len + PADDING > ctx->in_cap) {
		free(ctx->in);
		ctx->in_cap = len + PADDING;
		ctx->in = (uint8_t *)malloc(ctx->in_cap);
		if (!ctx->in) { ctx->in_cap = 0; return nullptr; }
	}
	memset(ctx->in + len, 0, PADDING);
	return ctx->in;
}

// Parses the frame header only (dimensions, chroma format, alpha, color tags) of the
// packet in the input buffer. Returns the frame header size, or < 0.
int prores_parse_header(ProresCtx *ctx, int len)
{
	const uint8_t *buf = ctx->in;
	if (len < 28 || buf[4] != 'i' || buf[5] != 'c' || buf[6] != 'p' || buf[7] != 'f')
		return -1;	// invalid frame header
	return decode_frame_header(ctx, buf + 8, len - 8);
}

// Decodes the packet in the input buffer into the planes. Returns 0, or < 0.
int prores_decode_frame(ProresCtx *ctx, int len)
{
	int frame_hdr_size = prores_parse_header(ctx, len);
	if (frame_hdr_size < 0)
		return frame_hdr_size;

	const uint8_t *buf = ctx->in + 8 + frame_hdr_size;
	int buf_size = len - 8 - frame_hdr_size;

	if (alloc_planes(ctx) < 0)
		return -3;

	ctx->first_field = 1;

	for (;;) {
		int pic_size = decode_picture_header(ctx, buf, buf_size);
		if (pic_size < 0)
			return pic_size;

		int ret = decode_picture(ctx);
		if (ret < 0)
			return ret;

		buf += pic_size;
		buf_size -= pic_size;

		if (!(ctx->frame_type && buf_size > 0 && ctx->first_field))
			break;
		ctx->first_field = 0;
	}
	return 0;
}

// int32[9]: width, height, chroma444, alpha_info, frame_type, color_primaries, color_trc,
// colorspace, bits.
void prores_info(const ProresCtx *ctx, int32_t *out)
{
	out[0] = ctx->width;
	out[1] = ctx->height;
	out[2] = ctx->chroma444;
	out[3] = ctx->alpha_info;
	out[4] = ctx->frame_type;
	out[5] = ctx->color_primaries;
	out[6] = ctx->color_trc;
	out[7] = ctx->colorspace;
	out[8] = ctx->bits;
}

void prores_set_simd(int on) { g_simd = on ? 1 : 0; }

uint16_t *prores_plane(const ProresCtx *ctx, int i) { return (i >= 0 && i < 4) ? ctx->plane[i] : nullptr; }
int prores_stride(const ProresCtx *ctx, int i) { return (i >= 0 && i < 4) ? ctx->stride[i] : 0; }

// Converts the decoded planes to RGBA16 (straight alpha, 65535 = opaque), width x height
// tight. kr, kb: the YCbCr matrix's luma coefficients (BT.709 0.2126, 0.0722). Limited
// range at the decode depth (Y 64-940 for 10 bits); 4:2:2 chroma is co-sited with the
// even luma columns and interpolated linearly between them. Returns the buffer, or null.
uint16_t *prores_to_rgba16(ProresCtx *ctx, float kr, float kb)
{
	const int w = ctx->width, h = ctx->height;
	if (!ctx->plane[0] || !w || !h) return nullptr;
	const int need = w * h * 4;
	if (need > ctx->rgba_cap) {
		free(ctx->rgba);
		ctx->rgba = (uint16_t *)malloc((size_t)need * sizeof(uint16_t));
		ctx->rgba_cap = ctx->rgba ? need : 0;
		if (!ctx->rgba) return nullptr;
	}

	const float kg = 1.0f - kr - kb;
	const float unit = (float)(1 << (ctx->bits - 8));
	const float yOff = 16.0f * unit, yScale = 1.0f / (219.0f * unit);
	const float cOff = 128.0f * unit, cScale = 1.0f / (224.0f * unit);
	const float crToR = 2.0f * (1.0f - kr), cbToB = 2.0f * (1.0f - kb);
	const float crToG = -crToR * kr / kg, cbToG = -cbToB * kb / kg;
	const float alphaScale = 65535.0f / (float)((1 << ctx->bits) - 1);
	const int lastC = ctx->chroma444 ? w - 1 : (w - 1) >> 1;

#ifdef __wasm_simd128__
	const v128_t vYOff = wasm_f32x4_splat(yOff), vYScale = wasm_f32x4_splat(yScale);
	const v128_t vCOff = wasm_f32x4_splat(cOff), vCScale = wasm_f32x4_splat(cScale);
	const v128_t vCrToR = wasm_f32x4_splat(crToR), vCbToB = wasm_f32x4_splat(cbToB);
	const v128_t vCrToG = wasm_f32x4_splat(crToG), vCbToG = wasm_f32x4_splat(cbToG);
	const v128_t vZero = wasm_f32x4_splat(0.0f), vOne = wasm_f32x4_splat(1.0f);
	const v128_t vFull = wasm_f32x4_splat(65535.0f), vHalf = wasm_f32x4_splat(0.5f);
	const v128_t vAlphaScale = wasm_f32x4_splat(alphaScale);
	const v128_t vOpaque = wasm_i32x4_splat(65535);
	// 0..65535.5 -> uint16, as the scalar (uint16_t)(v + 0.5f) truncates.
	auto quantize = [&](v128_t v) { return wasm_i32x4_trunc_sat_f32x4(wasm_f32x4_add(wasm_f32x4_mul(v, vFull), vHalf)); };
	auto unit01 = [&](v128_t v) { return wasm_f32x4_pmin(wasm_f32x4_pmax(v, vZero), vOne); };
#endif

	for (int y = 0; y < h; y++) {
		const uint16_t *Y = ctx->plane[0] + y * ctx->stride[0];
		const uint16_t *U = ctx->plane[1] + y * ctx->stride[1];
		const uint16_t *V = ctx->plane[2] + y * ctx->stride[2];
		const uint16_t *A = ctx->plane[3] + y * ctx->stride[3];
		uint16_t *out = ctx->rgba + (size_t)y * w * 4;
		int x = 0;

#ifdef __wasm_simd128__
		// Four pixels a step, the scalar loop's operations in its order (no fused multiply-add
		// in wasm SIMD), so both give the same values. 4:2:2 steps need chroma c .. c + 3 within
		// the row's last chroma column, as the scalar clamps at the edge; the tail runs scalar.
		if (g_simd) {
			for (; x + 4 <= w && (ctx->chroma444 || (x >> 1) + 3 <= lastC); x += 4) {
				v128_t cb, cr;
				if (ctx->chroma444) {
					cb = wasm_f32x4_convert_i32x4(wasm_u32x4_load16x4(U + x));
					cr = wasm_f32x4_convert_i32x4(wasm_u32x4_load16x4(V + x));
				} else {
					// [c0, c1, c2, c3] -> [2c0, c0 + c1, 2c1, c1 + c2] * 0.5: even pixels take
					// their sample, odd ones the mean with the next.
					const v128_t u = wasm_u32x4_load16x4(U + (x >> 1)), v = wasm_u32x4_load16x4(V + (x >> 1));
					const v128_t us = wasm_i32x4_add(wasm_i32x4_shuffle(u, u, 0, 0, 1, 1), wasm_i32x4_shuffle(u, u, 0, 1, 1, 2));
					const v128_t vs = wasm_i32x4_add(wasm_i32x4_shuffle(v, v, 0, 0, 1, 1), wasm_i32x4_shuffle(v, v, 0, 1, 1, 2));
					cb = wasm_f32x4_mul(vHalf, wasm_f32x4_convert_i32x4(us));
					cr = wasm_f32x4_mul(vHalf, wasm_f32x4_convert_i32x4(vs));
				}
				const v128_t l = wasm_f32x4_mul(wasm_f32x4_sub(wasm_f32x4_convert_i32x4(wasm_u32x4_load16x4(Y + x)), vYOff), vYScale);
				cb = wasm_f32x4_mul(wasm_f32x4_sub(cb, vCOff), vCScale);
				cr = wasm_f32x4_mul(wasm_f32x4_sub(cr, vCOff), vCScale);
				const v128_t r = unit01(wasm_f32x4_add(l, wasm_f32x4_mul(vCrToR, cr)));
				const v128_t g = unit01(wasm_f32x4_add(wasm_f32x4_add(l, wasm_f32x4_mul(vCrToG, cr)), wasm_f32x4_mul(vCbToG, cb)));
				const v128_t b = unit01(wasm_f32x4_add(l, wasm_f32x4_mul(vCbToB, cb)));
				const v128_t a = ctx->alpha_info
					? wasm_i32x4_trunc_sat_f32x4(wasm_f32x4_add(wasm_f32x4_mul(wasm_f32x4_convert_i32x4(wasm_u32x4_load16x4(A + x)), vAlphaScale), vHalf))
					: vOpaque;
				const v128_t rg = wasm_u16x8_narrow_i32x4(quantize(r), quantize(g));	// r0..r3 g0..g3
				const v128_t ba = wasm_u16x8_narrow_i32x4(quantize(b), a);		// b0..b3 a0..a3
				wasm_v128_store(out,     wasm_i16x8_shuffle(rg, ba, 0, 4, 8, 12, 1, 5, 9, 13));
				wasm_v128_store(out + 8, wasm_i16x8_shuffle(rg, ba, 2, 6, 10, 14, 3, 7, 11, 15));
				out += 16;
			}
		}
#endif

		for (; x < w; x++) {
			float cb, cr;
			if (ctx->chroma444) {
				cb = U[x];
				cr = V[x];
			} else {
				const int c = x >> 1;
				if (x & 1) {
					const int c1 = c + 1 <= lastC ? c + 1 : lastC;
					cb = 0.5f * (U[c] + U[c1]);
					cr = 0.5f * (V[c] + V[c1]);
				} else {
					cb = U[c];
					cr = V[c];
				}
			}
			const float l = (Y[x] - yOff) * yScale;
			cb = (cb - cOff) * cScale;
			cr = (cr - cOff) * cScale;
			float r = l + crToR * cr;
			float g = l + crToG * cr + cbToG * cb;
			float b = l + cbToB * cb;
			r = r < 0.0f ? 0.0f : r > 1.0f ? 1.0f : r;
			g = g < 0.0f ? 0.0f : g > 1.0f ? 1.0f : g;
			b = b < 0.0f ? 0.0f : b > 1.0f ? 1.0f : b;
			out[0] = (uint16_t)(r * 65535.0f + 0.5f);
			out[1] = (uint16_t)(g * 65535.0f + 0.5f);
			out[2] = (uint16_t)(b * 65535.0f + 0.5f);
			out[3] = ctx->alpha_info ? (uint16_t)(A[x] * alphaScale + 0.5f) : 65535;
			out += 4;
		}
	}
	return ctx->rgba;
}

}	// extern "C"
