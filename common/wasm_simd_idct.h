/*
	Shared WebAssembly-SIMD (i32x4) building blocks for the project's codec
	WASM ports. Header-only, `static inline` — each codec compiles ONE translation
	unit at -O3 so these inline fully and -Wl,--gc-sections strips anything a given
	module doesn't reference. No link-time dependency is introduced.

	Two layers live here:

	  (1) PRIMITIVES — transpose4_i32, the int16<->i32x4 widening load and the
	      truncating/saturating stores, and imc() (multiply-by-constant). These are
	      transform-agnostic: the VP3 IDCT (vp6), the islow IDCT (mpeg/jpeg) and the
	      Indeo slant/Haar transforms all share the same
	      load -> column-butterfly -> transpose4_i32 -> row-butterfly -> round/shift
	      -> truncate/narrow-store skeleton. Only the butterfly body differs.

	  (2) THE islow 1-D STAGE — wsi_idct_islow_1d(). This kernel is currently
	      duplicated VERBATIM in plugins/mpeg/mpeg_decode.cpp (idct_1d_i32x4) and
	      plugins/jpeg/jsimd_wasm.c (wj_idct_1d); only the surrounding 2-D drivers
	      differ (mpeg writes an int32 residual; jpeg folds dequant on input and adds
	      +128 then u8-saturates). It is provided here so those two can de-duplicate
	      onto it in a later, conformance-gated pass. The Indeo pack does NOT use this
	      kernel (the slant transform is not a DCT) — it uses only the layer-(1)
	      primitives.

	-fwrapv is REQUIRED in every consumer: the i32x4 lanes wrap mod 2^32 exactly like
	the scalar int32 paths (and FFmpeg / libjpeg), which is what makes the SIMD output
	byte-identical to the scalar oracle.
*/

#ifndef WASM_SIMD_IDCT_H
#define WASM_SIMD_IDCT_H

#ifdef __wasm_simd128__
#include <wasm_simd128.h>
#include <stdint.h>

// ─── Layer (1): transform-agnostic primitives ───────────────────────────────

// multiply each i32 lane by the scalar constant c (low-32 wrapping, like scalar).
static inline v128_t wsi_imc(v128_t x, int c) {
	return wasm_i32x4_mul(x, wasm_i32x4_splat(c));
}

// 4x4 i32 transpose: rows a,b,c,d -> columns o0,o1,o2,o3.
static inline void wsi_transpose4_i32(v128_t a, v128_t b, v128_t c, v128_t d,
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

// widen 4 consecutive int16 at p into an i32x4 (sign-extended).
static inline v128_t wsi_load16x4(const int16_t* p) {
	return wasm_i32x4_load16x4(p);
}

// truncating i32x4 -> 4x int16 store (low halfword of each lane), == (int16_t) cast.
static inline void wsi_store_trunc16(int16_t* p, v128_t x) {
	wasm_v128_store64_lane(p, wasm_i16x8_shuffle(x, x, 0, 2, 4, 6, 0, 2, 4, 6), 0);
}

// ─── Layer (2): the islow 1-D DCT stage (mpeg + jpeg only; NOT used by Indeo) ─
//
// One 1-D islow (LL&M / Loeffler) IDCT over 8 i32x4 lane-vectors in[0..7] ->
// out[0..7], descale by `sh`. The 13-bit fixed-point constants are passed by the
// caller (mpeg/jpeg both use the same values) so this header carries no table.
// Provided for the future mpeg/jpeg de-duplication; left unreferenced until then.

#endif // __wasm_simd128__
#endif // WASM_SIMD_IDCT_H
