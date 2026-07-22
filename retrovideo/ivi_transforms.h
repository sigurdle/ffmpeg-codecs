/*
	Indeo Video Interactive (Indeo 4 / 5) inverse transforms + motion compensation,
	a direct port of FFmpeg libavcodec/ivi_dsp.c (release/8.1, LGPL). Block-level
	transforms only (the wavelet recompose, which needs the plane/band layout, lives
	with the IVI core). Header-only, #included by codec_pack.cpp into the iv45
	namespace so it shares the pack's av_clip_uint8 and links once.

	The scalar reference implementations and the i32x4 SIMD variants of the two 8x8
	block transforms (inverse_slant_8x8, inverse_haar_8x8 — the hot luma decode path)
	both live here. Each 8x8 scalar entry point dispatches to its SIMD twin when
	g_use_simd is set (the default); the scalar body remains the oracle that the
	rv_test_transform hook compares against with SIMD off. The SIMD kernels REUSE the
	exact same butterfly macros (IVI_INV_SLANT8, INV_HAAR8) via an i32x4 wrapper type
	`V` whose operators map +,-,*,>> to the wasm intrinsics — so the SIMD path
	performs byte-identical arithmetic to the scalar path (low-32 wrapping multiply,
	arithmetic shift), lane-for-lane. -fwrapv keeps the >> on negative intermediates
	byte-identical to FFmpeg.

	Measured speedup (rv_bench_transform, a tight in-wasm loop over 300k blocks,
	min-of-4 runs, -O3 on V8/node — see tests/RetroVideoSimd.test.js):
	    inverse_slant_8x8   35.5ms -> 20.4ms   1.74x
	    inverse_haar_8x8    28.7ms -> 19.1ms   1.50x

	The 4x4 transforms are DELIBERATELY left scalar. Their i32x4 SIMD version was
	written and benchmarked the same way and came out SLOWER than scalar:
	    inverse_slant_4x4   10.9ms -> 13.9ms   0.78x   (slower)
	    inverse_haar_4x4     9.4ms -> 13.7ms   0.69x   (slower)
	A 16-element block is too small to amortize the two i32x4 transposes the row pass
	needs, so the SIMD 4x4 code was removed rather than shipped. The single-pass
	row/col variants and the wavelet recompose also stay scalar (scalable-band only;
	not on any tested decode path).

	Signatures match ivi.h:
	  InvTransformPtr: void(const int32_t* in, int16_t* out, ptrdiff_t pitch, const uint8_t* flags)
	  DCTransformPtr : void(const int32_t* in, int16_t* out, ptrdiff_t pitch, int blk_size)
*/

#ifndef IVI_TRANSFORMS_H
#define IVI_TRANSFORMS_H

#include <stdint.h>
#include <stddef.h>
#include <string.h>

#ifdef __wasm_simd128__
#include <wasm_simd128.h>
#include "wasm_simd_idct.h"   // shared i32x4 primitives: wsi_transpose4_i32, wsi_store_trunc16
#endif

// g_use_simd: runtime toggle, defined in codec_pack.cpp before this header is
// #included (so the dispatch below sees it). 1 = use the i32x4 SIMD transforms.
extern int g_use_simd;

namespace iv45 {

#ifdef __wasm_simd128__
// ── i32x4 SIMD support: a thin v128 wrapper so the scalar butterfly MACROS can be
//    reused verbatim. operator+/-/*/>> map to the i32x4 intrinsics; every lane then
//    performs the SAME wrapping int32 arithmetic as the scalar `int` path. ──
struct V {
	v128_t v;
	V() {}
	V(v128_t x) : v(x) {}
};
static inline V operator+(V a, V b)  { return V(wasm_i32x4_add(a.v, b.v)); }
static inline V operator-(V a, V b)  { return V(wasm_i32x4_sub(a.v, b.v)); }
static inline V operator-(V a)       { return V(wasm_i32x4_neg(a.v)); }
static inline V operator*(V a, int c){ return V(wasm_i32x4_mul(a.v, wasm_i32x4_splat(c))); }
static inline V operator+(V a, int c){ return V(wasm_i32x4_add(a.v, wasm_i32x4_splat(c))); }
static inline V operator>>(V a, int s){ return V(wasm_i32x4_shr(a.v, s)); }

// Forward declarations — the scalar 8x8 entry points (defined below) dispatch here.
// (Only the 8x8 transforms have SIMD twins; 4x4 stays scalar — see notes there.)
static void inv_slant_8x8_simd(const int32_t* in, int16_t* out, ptrdiff_t pitch, const uint8_t* flags);
static void inv_haar_8x8_simd (const int32_t* in, int16_t* out, ptrdiff_t pitch, const uint8_t* flags);
#endif // __wasm_simd128__

// ── inverse Haar ────────────────────────────────────────────────────────────
#define IVI_HAAR_BFLY(s1, s2, o1, o2, t) \
	t  = ((s1) - (s2)) >> 1; \
	o1 = ((s1) + (s2)) >> 1; \
	o2 = (t);

#define INV_HAAR8(s1, s5, s3, s7, s2, s4, s6, s8, \
                  d1, d2, d3, d4, d5, d6, d7, d8, \
                  t0, t1, t2, t3, t4, t5, t6, t7, t8) { \
	t1 = (s1) * 2; t5 = (s5) * 2; \
	IVI_HAAR_BFLY(t1, t5, t1, t5, t0); IVI_HAAR_BFLY(t1, s3, t1, t3, t0); \
	IVI_HAAR_BFLY(t5, s7, t5, t7, t0); IVI_HAAR_BFLY(t1, s2, t1, t2, t0); \
	IVI_HAAR_BFLY(t3, s4, t3, t4, t0); IVI_HAAR_BFLY(t5, s6, t5, t6, t0); \
	IVI_HAAR_BFLY(t7, s8, t7, t8, t0); \
	d1 = COMPENSATE(t1); d2 = COMPENSATE(t2); d3 = COMPENSATE(t3); d4 = COMPENSATE(t4); \
	d5 = COMPENSATE(t5); d6 = COMPENSATE(t6); d7 = COMPENSATE(t7); d8 = COMPENSATE(t8); }

#define INV_HAAR4(s1, s3, s5, s7, d1, d2, d3, d4, t0, t1, t2, t3, t4) { \
	IVI_HAAR_BFLY(s1, s3, t0, t1, t4); \
	IVI_HAAR_BFLY(t0, s5, t2, t3, t4); \
	d1 = COMPENSATE(t2); d2 = COMPENSATE(t3); \
	IVI_HAAR_BFLY(t1, s7, t2, t3, t4); \
	d3 = COMPENSATE(t2); d4 = COMPENSATE(t3); }

static void ff_ivi_inverse_haar_8x8(const int32_t* in, int16_t* out, ptrdiff_t pitch, const uint8_t* flags) {
#ifdef __wasm_simd128__
	if (g_use_simd) { inv_haar_8x8_simd(in, out, pitch, flags); return; }
#endif
	int i, shift, sp1, sp2, sp3, sp4;
	const int32_t* src; int32_t* dst; int tmp[64];
	int t0, t1, t2, t3, t4, t5, t6, t7, t8;
#define COMPENSATE(x) (x)
	src = in; dst = tmp;
	for (i = 0; i < 8; i++) {
		if (flags[i]) {
			shift = !(i & 4);
			sp1 = src[0] * (1 << shift); sp2 = src[8] * (1 << shift);
			sp3 = src[16] * (1 << shift); sp4 = src[24] * (1 << shift);
			INV_HAAR8(sp1, sp2, sp3, sp4, src[32], src[40], src[48], src[56],
			          dst[0], dst[8], dst[16], dst[24], dst[32], dst[40], dst[48], dst[56],
			          t0, t1, t2, t3, t4, t5, t6, t7, t8);
		} else
			dst[0] = dst[8] = dst[16] = dst[24] = dst[32] = dst[40] = dst[48] = dst[56] = 0;
		src++; dst++;
	}
#undef COMPENSATE
#define COMPENSATE(x) (x)
	src = tmp;
	for (i = 0; i < 8; i++) {
		if (!src[0] && !src[1] && !src[2] && !src[3] && !src[4] && !src[5] && !src[6] && !src[7]) {
			memset(out, 0, 8 * sizeof(out[0]));
		} else {
			INV_HAAR8(src[0], src[1], src[2], src[3], src[4], src[5], src[6], src[7],
			          out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7],
			          t0, t1, t2, t3, t4, t5, t6, t7, t8);
		}
		src += 8; out += pitch;
	}
#undef COMPENSATE
}

static void ff_ivi_row_haar8(const int32_t* in, int16_t* out, ptrdiff_t pitch, const uint8_t* flags) {
	int i, t0, t1, t2, t3, t4, t5, t6, t7, t8;
#define COMPENSATE(x) (x)
	for (i = 0; i < 8; i++) {
		if (!in[0] && !in[1] && !in[2] && !in[3] && !in[4] && !in[5] && !in[6] && !in[7]) {
			memset(out, 0, 8 * sizeof(out[0]));
		} else {
			INV_HAAR8(in[0], in[1], in[2], in[3], in[4], in[5], in[6], in[7],
			          out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7],
			          t0, t1, t2, t3, t4, t5, t6, t7, t8);
		}
		in += 8; out += pitch;
	}
#undef COMPENSATE
}

static void ff_ivi_col_haar8(const int32_t* in, int16_t* out, ptrdiff_t pitch, const uint8_t* flags) {
	int i, t0, t1, t2, t3, t4, t5, t6, t7, t8;
#define COMPENSATE(x) (x)
	for (i = 0; i < 8; i++) {
		if (flags[i]) {
			INV_HAAR8(in[0], in[8], in[16], in[24], in[32], in[40], in[48], in[56],
			          out[0 * pitch], out[1 * pitch], out[2 * pitch], out[3 * pitch],
			          out[4 * pitch], out[5 * pitch], out[6 * pitch], out[7 * pitch],
			          t0, t1, t2, t3, t4, t5, t6, t7, t8);
		} else
			out[0 * pitch] = out[1 * pitch] = out[2 * pitch] = out[3 * pitch] =
			out[4 * pitch] = out[5 * pitch] = out[6 * pitch] = out[7 * pitch] = 0;
		in++; out++;
	}
#undef COMPENSATE
}

static void ff_ivi_inverse_haar_4x4(const int32_t* in, int16_t* out, ptrdiff_t pitch, const uint8_t* flags) {
	// 4x4 stays scalar: an i32x4 SIMD twin measured 0.69x (SLOWER) — a 16-element
	// block can't amortize the row pass's two transposes (see the file header for the
	// full benchmark). Only the 8x8 transforms (the hot luma path) get a SIMD twin.
	int i, shift, sp1, sp2;
	const int32_t* src; int32_t* dst; int tmp[16];
	int t0, t1, t2, t3, t4;
#define COMPENSATE(x) (x)
	src = in; dst = tmp;
	for (i = 0; i < 4; i++) {
		if (flags[i]) {
			shift = !(i & 2);
			sp1 = src[0] * (1 << shift); sp2 = src[4] * (1 << shift);
			INV_HAAR4(sp1, sp2, src[8], src[12], dst[0], dst[4], dst[8], dst[12], t0, t1, t2, t3, t4);
		} else
			dst[0] = dst[4] = dst[8] = dst[12] = 0;
		src++; dst++;
	}
#undef COMPENSATE
#define COMPENSATE(x) (x)
	src = tmp;
	for (i = 0; i < 4; i++) {
		if (!src[0] && !src[1] && !src[2] && !src[3]) {
			memset(out, 0, 4 * sizeof(out[0]));
		} else {
			INV_HAAR4(src[0], src[1], src[2], src[3], out[0], out[1], out[2], out[3], t0, t1, t2, t3, t4);
		}
		src += 4; out += pitch;
	}
#undef COMPENSATE
}

static void ff_ivi_row_haar4(const int32_t* in, int16_t* out, ptrdiff_t pitch, const uint8_t* flags) {
	int i, t0, t1, t2, t3, t4;
#define COMPENSATE(x) (x)
	for (i = 0; i < 4; i++) {
		if (!in[0] && !in[1] && !in[2] && !in[3]) {
			memset(out, 0, 4 * sizeof(out[0]));
		} else {
			INV_HAAR4(in[0], in[1], in[2], in[3], out[0], out[1], out[2], out[3], t0, t1, t2, t3, t4);
		}
		in += 4; out += pitch;
	}
#undef COMPENSATE
}

static void ff_ivi_col_haar4(const int32_t* in, int16_t* out, ptrdiff_t pitch, const uint8_t* flags) {
	int i, t0, t1, t2, t3, t4;
#define COMPENSATE(x) (x)
	for (i = 0; i < 4; i++) {
		if (flags[i]) {
			INV_HAAR4(in[0], in[4], in[8], in[12],
			          out[0 * pitch], out[1 * pitch], out[2 * pitch], out[3 * pitch], t0, t1, t2, t3, t4);
		} else
			out[0 * pitch] = out[1 * pitch] = out[2 * pitch] = out[3 * pitch] = 0;
		in++; out++;
	}
#undef COMPENSATE
}

static void ff_ivi_dc_haar_2d(const int32_t* in, int16_t* out, ptrdiff_t pitch, int blk_size) {
	int16_t dc = (int16_t)((*in + 0) >> 3);
	for (int y = 0; y < blk_size; out += pitch, y++)
		for (int x = 0; x < blk_size; x++) out[x] = dc;
}

// ── inverse slant ───────────────────────────────────────────────────────────
#define IVI_SLANT_BFLY(s1, s2, o1, o2, t) \
	t  = (s1) - (s2); o1 = (s1) + (s2); o2 = (t);
#define IVI_IREFLECT(s1, s2, o1, o2, t) \
	t  = (((s1) + (s2) * 2 + 2) >> 2) + (s1); \
	o2 = (((s1) * 2 - (s2) + 2) >> 2) - (s2); o1 = (t);
#define IVI_SLANT_PART4(s1, s2, o1, o2, t) \
	t  = (s2) + (((s1) * 4 - (s2) + 4) >> 3); \
	o2 = (s1) + ((-(s1) - (s2) * 4 + 4) >> 3); o1 = (t);

#define IVI_INV_SLANT8(s1, s4, s8, s5, s2, s6, s3, s7, \
                       d1, d2, d3, d4, d5, d6, d7, d8, \
                       t0, t1, t2, t3, t4, t5, t6, t7, t8) { \
	IVI_SLANT_PART4(s4, s5, t4, t5, t0); \
	IVI_SLANT_BFLY(s1, t5, t1, t5, t0); IVI_SLANT_BFLY(s2, s6, t2, t6, t0); \
	IVI_SLANT_BFLY(s7, s3, t7, t3, t0); IVI_SLANT_BFLY(t4, s8, t4, t8, t0); \
	IVI_SLANT_BFLY(t1, t2, t1, t2, t0); IVI_IREFLECT(t4, t3, t4, t3, t0); \
	IVI_SLANT_BFLY(t5, t6, t5, t6, t0); IVI_IREFLECT(t8, t7, t8, t7, t0); \
	IVI_SLANT_BFLY(t1, t4, t1, t4, t0); IVI_SLANT_BFLY(t2, t3, t2, t3, t0); \
	IVI_SLANT_BFLY(t5, t8, t5, t8, t0); IVI_SLANT_BFLY(t6, t7, t6, t7, t0); \
	d1 = COMPENSATE(t1); d2 = COMPENSATE(t2); d3 = COMPENSATE(t3); d4 = COMPENSATE(t4); \
	d5 = COMPENSATE(t5); d6 = COMPENSATE(t6); d7 = COMPENSATE(t7); d8 = COMPENSATE(t8); }

#define IVI_INV_SLANT4(s1, s4, s2, s3, d1, d2, d3, d4, t0, t1, t2, t3, t4) { \
	IVI_SLANT_BFLY(s1, s2, t1, t2, t0); IVI_IREFLECT(s4, s3, t4, t3, t0); \
	IVI_SLANT_BFLY(t1, t4, t1, t4, t0); IVI_SLANT_BFLY(t2, t3, t2, t3, t0); \
	d1 = COMPENSATE(t1); d2 = COMPENSATE(t2); d3 = COMPENSATE(t3); d4 = COMPENSATE(t4); }

static void ff_ivi_inverse_slant_8x8(const int32_t* in, int16_t* out, ptrdiff_t pitch, const uint8_t* flags) {
#ifdef __wasm_simd128__
	if (g_use_simd) { inv_slant_8x8_simd(in, out, pitch, flags); return; }
#endif
	int i; const int32_t* src; int32_t* dst; int tmp[64];
	int t0, t1, t2, t3, t4, t5, t6, t7, t8;
#define COMPENSATE(x) (x)
	src = in; dst = tmp;
	for (i = 0; i < 8; i++) {
		if (flags[i]) {
			IVI_INV_SLANT8(src[0], src[8], src[16], src[24], src[32], src[40], src[48], src[56],
			               dst[0], dst[8], dst[16], dst[24], dst[32], dst[40], dst[48], dst[56],
			               t0, t1, t2, t3, t4, t5, t6, t7, t8);
		} else
			dst[0] = dst[8] = dst[16] = dst[24] = dst[32] = dst[40] = dst[48] = dst[56] = 0;
		src++; dst++;
	}
#undef COMPENSATE
#define COMPENSATE(x) (((x) + 1) >> 1)
	src = tmp;
	for (i = 0; i < 8; i++) {
		if (!src[0] && !src[1] && !src[2] && !src[3] && !src[4] && !src[5] && !src[6] && !src[7]) {
			memset(out, 0, 8 * sizeof(out[0]));
		} else {
			IVI_INV_SLANT8(src[0], src[1], src[2], src[3], src[4], src[5], src[6], src[7],
			               out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7],
			               t0, t1, t2, t3, t4, t5, t6, t7, t8);
		}
		src += 8; out += pitch;
	}
#undef COMPENSATE
}

static void ff_ivi_inverse_slant_4x4(const int32_t* in, int16_t* out, ptrdiff_t pitch, const uint8_t* flags) {
	// 4x4 stays scalar: an i32x4 SIMD twin measured 0.78x (SLOWER) — see the file
	// header benchmark and ff_ivi_inverse_haar_4x4.
	int i; const int32_t* src; int32_t* dst; int tmp[16];
	int t0, t1, t2, t3, t4;
#define COMPENSATE(x) (x)
	src = in; dst = tmp;
	for (i = 0; i < 4; i++) {
		if (flags[i]) {
			IVI_INV_SLANT4(src[0], src[4], src[8], src[12], dst[0], dst[4], dst[8], dst[12], t0, t1, t2, t3, t4);
		} else
			dst[0] = dst[4] = dst[8] = dst[12] = 0;
		src++; dst++;
	}
#undef COMPENSATE
#define COMPENSATE(x) (((x) + 1) >> 1)
	src = tmp;
	for (i = 0; i < 4; i++) {
		if (!src[0] && !src[1] && !src[2] && !src[3]) {
			out[0] = out[1] = out[2] = out[3] = 0;
		} else {
			IVI_INV_SLANT4(src[0], src[1], src[2], src[3], out[0], out[1], out[2], out[3], t0, t1, t2, t3, t4);
		}
		src += 4; out += pitch;
	}
#undef COMPENSATE
}

static void ff_ivi_row_slant8(const int32_t* in, int16_t* out, ptrdiff_t pitch, const uint8_t* flags) {
	int i, t0, t1, t2, t3, t4, t5, t6, t7, t8;
#define COMPENSATE(x) (((x) + 1) >> 1)
	for (i = 0; i < 8; i++) {
		if (!in[0] && !in[1] && !in[2] && !in[3] && !in[4] && !in[5] && !in[6] && !in[7]) {
			memset(out, 0, 8 * sizeof(out[0]));
		} else {
			IVI_INV_SLANT8(in[0], in[1], in[2], in[3], in[4], in[5], in[6], in[7],
			               out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7],
			               t0, t1, t2, t3, t4, t5, t6, t7, t8);
		}
		in += 8; out += pitch;
	}
#undef COMPENSATE
}

static void ff_ivi_col_slant8(const int32_t* in, int16_t* out, ptrdiff_t pitch, const uint8_t* flags) {
	int i, row2 = pitch << 1, row4 = pitch << 2, row8 = pitch << 3;
	int t0, t1, t2, t3, t4, t5, t6, t7, t8;
#define COMPENSATE(x) (((x) + 1) >> 1)
	for (i = 0; i < 8; i++) {
		if (flags[i]) {
			IVI_INV_SLANT8(in[0], in[8], in[16], in[24], in[32], in[40], in[48], in[56],
			               out[0], out[pitch], out[row2], out[row2 + pitch], out[row4],
			               out[row4 + pitch], out[row4 + row2], out[row8 - pitch],
			               t0, t1, t2, t3, t4, t5, t6, t7, t8);
		} else {
			out[0] = out[pitch] = out[row2] = out[row2 + pitch] = out[row4] =
			out[row4 + pitch] = out[row4 + row2] = out[row8 - pitch] = 0;
		}
		in++; out++;
	}
#undef COMPENSATE
}

static void ff_ivi_row_slant4(const int32_t* in, int16_t* out, ptrdiff_t pitch, const uint8_t* flags) {
	int i, t0, t1, t2, t3, t4;
#define COMPENSATE(x) (((x) + 1) >> 1)
	for (i = 0; i < 4; i++) {
		if (!in[0] && !in[1] && !in[2] && !in[3]) {
			memset(out, 0, 4 * sizeof(out[0]));
		} else {
			IVI_INV_SLANT4(in[0], in[1], in[2], in[3], out[0], out[1], out[2], out[3], t0, t1, t2, t3, t4);
		}
		in += 4; out += pitch;
	}
#undef COMPENSATE
}

static void ff_ivi_col_slant4(const int32_t* in, int16_t* out, ptrdiff_t pitch, const uint8_t* flags) {
	int i, row2 = pitch << 1, t0, t1, t2, t3, t4;
#define COMPENSATE(x) (((x) + 1) >> 1)
	for (i = 0; i < 4; i++) {
		if (flags[i]) {
			IVI_INV_SLANT4(in[0], in[4], in[8], in[12],
			               out[0], out[pitch], out[row2], out[row2 + pitch], t0, t1, t2, t3, t4);
		} else {
			out[0] = out[pitch] = out[row2] = out[row2 + pitch] = 0;
		}
		in++; out++;
	}
#undef COMPENSATE
}

#ifdef __wasm_simd128__
// ════════════════════════════════════════════════════════════════════════════
//  i32x4 SIMD variants of the four 2-D block transforms.
//
//  Layout follows the proven VP3 idct_simd skeleton (plugins/vp6/vp6_decode.cpp):
//    pass 1 transforms 4 COLUMNS at a time (lane = column) straight into an int32
//    scratch, pass 2 transposes 4 ROWS in (lane = row), transforms, transposes the
//    result back, and truncating-stores int16. The butterfly bodies are the EXACT
//    same IVI_INV_SLANT8/4 / INV_HAAR8/4 macros the scalar path uses, instantiated
//    on the i32x4 wrapper `V`, so every lane reproduces the scalar arithmetic.
//
//  The per-column / per-row `flags` (and the scalar all-zero shortcuts) are
//  intentionally dropped: a flagged-off column/row is all-zero in `in`/scratch, and
//  the COMPENSATE() of an all-zero butterfly is 0 (slant: ((0)+1)>>1 == 0; haar:
//  identity) — identical to the explicit zero the scalar writes. The conformance
//  test (RetroVideoSimd) verifies this against the flag-respecting scalar.
// ════════════════════════════════════════════════════════════════════════════

static void inv_slant_8x8_simd(const int32_t* in, int16_t* out, ptrdiff_t pitch, const uint8_t* /*flags*/) {
	int32_t tmp[64];
	// pass 1: vertical (column) transform, lane = column, COMPENSATE identity.
	for (int c = 0; c < 8; c += 4) {
		V n0 = wasm_v128_load(&in[0*8+c]), n1 = wasm_v128_load(&in[1*8+c]);
		V n2 = wasm_v128_load(&in[2*8+c]), n3 = wasm_v128_load(&in[3*8+c]);
		V n4 = wasm_v128_load(&in[4*8+c]), n5 = wasm_v128_load(&in[5*8+c]);
		V n6 = wasm_v128_load(&in[6*8+c]), n7 = wasm_v128_load(&in[7*8+c]);
		V d1,d2,d3,d4,d5,d6,d7,d8, t0,t1,t2,t3,t4,t5,t6,t7,t8;
#define COMPENSATE(x) (x)
		IVI_INV_SLANT8(n0,n1,n2,n3,n4,n5,n6,n7, d1,d2,d3,d4,d5,d6,d7,d8,
		               t0,t1,t2,t3,t4,t5,t6,t7,t8);
#undef COMPENSATE
		wasm_v128_store(&tmp[0*8+c], d1.v); wasm_v128_store(&tmp[1*8+c], d2.v);
		wasm_v128_store(&tmp[2*8+c], d3.v); wasm_v128_store(&tmp[3*8+c], d4.v);
		wasm_v128_store(&tmp[4*8+c], d5.v); wasm_v128_store(&tmp[5*8+c], d6.v);
		wasm_v128_store(&tmp[6*8+c], d7.v); wasm_v128_store(&tmp[7*8+c], d8.v);
	}
	// pass 2: horizontal (row) transform, lane = row, COMPENSATE ((x)+1)>>1.
	for (int r = 0; r < 8; r += 4) {
		V a0 = wasm_v128_load(&tmp[(r+0)*8+0]), a1 = wasm_v128_load(&tmp[(r+1)*8+0]);
		V a2 = wasm_v128_load(&tmp[(r+2)*8+0]), a3 = wasm_v128_load(&tmp[(r+3)*8+0]);
		V b0 = wasm_v128_load(&tmp[(r+0)*8+4]), b1 = wasm_v128_load(&tmp[(r+1)*8+4]);
		V b2 = wasm_v128_load(&tmp[(r+2)*8+4]), b3 = wasm_v128_load(&tmp[(r+3)*8+4]);
		V X0,X1,X2,X3,X4,X5,X6,X7;
		wsi_transpose4_i32(a0.v,a1.v,a2.v,a3.v, &X0.v,&X1.v,&X2.v,&X3.v);
		wsi_transpose4_i32(b0.v,b1.v,b2.v,b3.v, &X4.v,&X5.v,&X6.v,&X7.v);
		V Y1,Y2,Y3,Y4,Y5,Y6,Y7,Y8, t0,t1,t2,t3,t4,t5,t6,t7,t8;
#define COMPENSATE(x) (((x) + 1) >> 1)
		IVI_INV_SLANT8(X0,X1,X2,X3,X4,X5,X6,X7, Y1,Y2,Y3,Y4,Y5,Y6,Y7,Y8,
		               t0,t1,t2,t3,t4,t5,t6,t7,t8);
#undef COMPENSATE
		V P0,P1,P2,P3,Q0,Q1,Q2,Q3;
		wsi_transpose4_i32(Y1.v,Y2.v,Y3.v,Y4.v, &P0.v,&P1.v,&P2.v,&P3.v); // cols 0-3
		wsi_transpose4_i32(Y5.v,Y6.v,Y7.v,Y8.v, &Q0.v,&Q1.v,&Q2.v,&Q3.v); // cols 4-7
		wsi_store_trunc16(&out[(r+0)*pitch+0], P0.v); wsi_store_trunc16(&out[(r+0)*pitch+4], Q0.v);
		wsi_store_trunc16(&out[(r+1)*pitch+0], P1.v); wsi_store_trunc16(&out[(r+1)*pitch+4], Q1.v);
		wsi_store_trunc16(&out[(r+2)*pitch+0], P2.v); wsi_store_trunc16(&out[(r+2)*pitch+4], Q2.v);
		wsi_store_trunc16(&out[(r+3)*pitch+0], P3.v); wsi_store_trunc16(&out[(r+3)*pitch+4], Q3.v);
	}
}

static void inv_haar_8x8_simd(const int32_t* in, int16_t* out, ptrdiff_t pitch, const uint8_t* /*flags*/) {
	int32_t tmp[64];
	// pass 1: column transform. The first 4 inputs of each column are pre-scaled by
	// (1<<shift); shift = !(col & 4) so cols 0-3 use *2, cols 4-7 use *1 (per group).
	for (int c = 0; c < 8; c += 4) {
		int sc = (c < 4) ? 2 : 1;
		V n0 = V(wasm_v128_load(&in[0*8+c])) * sc, n1 = V(wasm_v128_load(&in[1*8+c])) * sc;
		V n2 = V(wasm_v128_load(&in[2*8+c])) * sc, n3 = V(wasm_v128_load(&in[3*8+c])) * sc;
		V n4 = wasm_v128_load(&in[4*8+c]), n5 = wasm_v128_load(&in[5*8+c]);
		V n6 = wasm_v128_load(&in[6*8+c]), n7 = wasm_v128_load(&in[7*8+c]);
		V d1,d2,d3,d4,d5,d6,d7,d8, t0,t1,t2,t3,t4,t5,t6,t7,t8;
#define COMPENSATE(x) (x)
		INV_HAAR8(n0,n1,n2,n3,n4,n5,n6,n7, d1,d2,d3,d4,d5,d6,d7,d8,
		          t0,t1,t2,t3,t4,t5,t6,t7,t8);
#undef COMPENSATE
		wasm_v128_store(&tmp[0*8+c], d1.v); wasm_v128_store(&tmp[1*8+c], d2.v);
		wasm_v128_store(&tmp[2*8+c], d3.v); wasm_v128_store(&tmp[3*8+c], d4.v);
		wasm_v128_store(&tmp[4*8+c], d5.v); wasm_v128_store(&tmp[5*8+c], d6.v);
		wasm_v128_store(&tmp[6*8+c], d7.v); wasm_v128_store(&tmp[7*8+c], d8.v);
	}
	// pass 2: row transform, COMPENSATE identity (no scaling).
	for (int r = 0; r < 8; r += 4) {
		V a0 = wasm_v128_load(&tmp[(r+0)*8+0]), a1 = wasm_v128_load(&tmp[(r+1)*8+0]);
		V a2 = wasm_v128_load(&tmp[(r+2)*8+0]), a3 = wasm_v128_load(&tmp[(r+3)*8+0]);
		V b0 = wasm_v128_load(&tmp[(r+0)*8+4]), b1 = wasm_v128_load(&tmp[(r+1)*8+4]);
		V b2 = wasm_v128_load(&tmp[(r+2)*8+4]), b3 = wasm_v128_load(&tmp[(r+3)*8+4]);
		V X0,X1,X2,X3,X4,X5,X6,X7;
		wsi_transpose4_i32(a0.v,a1.v,a2.v,a3.v, &X0.v,&X1.v,&X2.v,&X3.v);
		wsi_transpose4_i32(b0.v,b1.v,b2.v,b3.v, &X4.v,&X5.v,&X6.v,&X7.v);
		V Y1,Y2,Y3,Y4,Y5,Y6,Y7,Y8, t0,t1,t2,t3,t4,t5,t6,t7,t8;
#define COMPENSATE(x) (x)
		INV_HAAR8(X0,X1,X2,X3,X4,X5,X6,X7, Y1,Y2,Y3,Y4,Y5,Y6,Y7,Y8,
		          t0,t1,t2,t3,t4,t5,t6,t7,t8);
#undef COMPENSATE
		V P0,P1,P2,P3,Q0,Q1,Q2,Q3;
		wsi_transpose4_i32(Y1.v,Y2.v,Y3.v,Y4.v, &P0.v,&P1.v,&P2.v,&P3.v);
		wsi_transpose4_i32(Y5.v,Y6.v,Y7.v,Y8.v, &Q0.v,&Q1.v,&Q2.v,&Q3.v);
		wsi_store_trunc16(&out[(r+0)*pitch+0], P0.v); wsi_store_trunc16(&out[(r+0)*pitch+4], Q0.v);
		wsi_store_trunc16(&out[(r+1)*pitch+0], P1.v); wsi_store_trunc16(&out[(r+1)*pitch+4], Q1.v);
		wsi_store_trunc16(&out[(r+2)*pitch+0], P2.v); wsi_store_trunc16(&out[(r+2)*pitch+4], Q2.v);
		wsi_store_trunc16(&out[(r+3)*pitch+0], P3.v); wsi_store_trunc16(&out[(r+3)*pitch+4], Q3.v);
	}
}

#endif // __wasm_simd128__

static void ff_ivi_dc_slant_2d(const int32_t* in, int16_t* out, ptrdiff_t pitch, int blk_size) {
	int16_t dc = (int16_t)((*in + 1) >> 1);
	for (int y = 0; y < blk_size; out += pitch, y++)
		for (int x = 0; x < blk_size; x++) out[x] = dc;
}
static void ff_ivi_dc_row_slant(const int32_t* in, int16_t* out, ptrdiff_t pitch, int blk_size) {
	int16_t dc = (int16_t)((*in + 1) >> 1);
	for (int x = 0; x < blk_size; x++) out[x] = dc;
	out += pitch;
	for (int y = 1; y < blk_size; out += pitch, y++)
		for (int x = 0; x < blk_size; x++) out[x] = 0;
}
static void ff_ivi_dc_col_slant(const int32_t* in, int16_t* out, ptrdiff_t pitch, int blk_size) {
	int16_t dc = (int16_t)((*in + 1) >> 1);
	for (int y = 0; y < blk_size; out += pitch, y++) {
		out[0] = dc;
		for (int x = 1; x < blk_size; x++) out[x] = 0;
	}
}

static void ff_ivi_put_pixels_8x8(const int32_t* in, int16_t* out, ptrdiff_t pitch, const uint8_t* flags) {
	for (int y = 0; y < 8; out += pitch, in += 8, y++)
		for (int x = 0; x < 8; x++) out[x] = (int16_t)in[x];
}
static void ff_ivi_put_dc_pixel_8x8(const int32_t* in, int16_t* out, ptrdiff_t pitch, int blk_size) {
	out[0] = (int16_t)in[0];
	memset(out + 1, 0, 7 * sizeof(out[0]));
	out += pitch;
	for (int y = 1; y < 8; out += pitch, y++) memset(out, 0, 8 * sizeof(out[0]));
}

// ── motion compensation ─────────────────────────────────────────────────────
#define IVI_MC_TEMPLATE(size, suffix, OP) \
static void ivi_mc_##size##x##size##suffix(int16_t* buf, ptrdiff_t dpitch, \
                                           const int16_t* ref_buf, ptrdiff_t pitch, int mc_type) { \
	int i, j; const int16_t* wptr; \
	switch (mc_type) { \
	case 0: \
		for (i = 0; i < size; i++, buf += dpitch, ref_buf += pitch) \
			for (j = 0; j < size; j++) OP(buf[j], ref_buf[j]); \
		break; \
	case 1: \
		for (i = 0; i < size; i++, buf += dpitch, ref_buf += pitch) \
			for (j = 0; j < size; j++) OP(buf[j], (ref_buf[j] + ref_buf[j + 1]) >> 1); \
		break; \
	case 2: \
		wptr = ref_buf + pitch; \
		for (i = 0; i < size; i++, buf += dpitch, wptr += pitch, ref_buf += pitch) \
			for (j = 0; j < size; j++) OP(buf[j], (ref_buf[j] + wptr[j]) >> 1); \
		break; \
	case 3: \
		wptr = ref_buf + pitch; \
		for (i = 0; i < size; i++, buf += dpitch, wptr += pitch, ref_buf += pitch) \
			for (j = 0; j < size; j++) OP(buf[j], (ref_buf[j] + ref_buf[j + 1] + wptr[j] + wptr[j + 1]) >> 2); \
		break; \
	} \
} \
static void ff_ivi_mc_##size##x##size##suffix(int16_t* buf, const int16_t* ref_buf, ptrdiff_t pitch, int mc_type) { \
	ivi_mc_##size##x##size##suffix(buf, pitch, ref_buf, pitch, mc_type); \
}

#define IVI_MC_AVG_TEMPLATE(size, suffix, OP) \
static void ff_ivi_mc_avg_##size##x##size##suffix(int16_t* buf, const int16_t* ref_buf, const int16_t* ref_buf2, \
                                                  ptrdiff_t pitch, int mc_type, int mc_type2) { \
	int16_t tmp[size * size]; int i, j; \
	ivi_mc_##size##x##size##_no_delta(tmp, size, ref_buf, pitch, mc_type); \
	ivi_mc_##size##x##size##_delta(tmp, size, ref_buf2, pitch, mc_type2); \
	for (i = 0; i < size; i++, buf += pitch) \
		for (j = 0; j < size; j++) OP(buf[j], tmp[i * size + j] >> 1); \
}

#define OP_PUT(a, b) (a) = (b)
#define OP_ADD(a, b) (a) += (b)
IVI_MC_TEMPLATE(8, _no_delta, OP_PUT)
IVI_MC_TEMPLATE(8, _delta,    OP_ADD)
IVI_MC_TEMPLATE(4, _no_delta, OP_PUT)
IVI_MC_TEMPLATE(4, _delta,    OP_ADD)
IVI_MC_AVG_TEMPLATE(8, _no_delta, OP_PUT)
IVI_MC_AVG_TEMPLATE(8, _delta,    OP_ADD)
IVI_MC_AVG_TEMPLATE(4, _no_delta, OP_PUT)
IVI_MC_AVG_TEMPLATE(4, _delta,    OP_ADD)
#undef OP_PUT
#undef OP_ADD

} // namespace iv45

#endif // IVI_TRANSFORMS_H
