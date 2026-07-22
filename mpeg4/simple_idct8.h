/*
    ff_simple_idct8 (8-bit) — transcription of FFmpeg release/8.1
    libavcodec/simple_idct_template.c with BIT_DEPTH==8, IN_IDCT_DEPTH==16
    (the kernel ff_idctdsp_init selects for 8-bit mpeg4/h263 decode).

    Copyright (c) 2001 Michael Niedermayer <michaelni@gmx.at>, LGPL-2.1+.
    FFmpeg is LGPL; see plugins/mpeg4/mpeg4_decode.cpp for the module-wide
    provenance note. This is its OWN translation unit (header-only, static
    inline) — deliberately NOT fused with the project's islow JPEG IDCT
    (plugins/common/wasm_simd_idct.h), which is a DIFFERENT IEEE-1180 transform.

    WHY this exact kernel: MPEG-4 Part 2 ASP is closed-loop predictive; a 1-LSB
    IDCT discrepancy is fed back into the prediction loop and compounds over a
    GOP. The ffmpeg CLI oracle decodes through this transform (its x86 SSE2/AVX
    build is bit-identical to this C reference, only the coefficient input order
    differs — folded into the scantable, FF_IDCT_PERM_NONE here), so to reach an
    exact max=0 compare we must reproduce it byte-for-byte: the W1..W7 constants,
    ROW_SHIFT=11 / COL_SHIFT=20, the row DC fast-path (which is NOT bit-identical
    to the full butterfly for large DC — e.g. row[0]=1025 -> DC-path 8200 vs
    full 8199 — and ffmpeg uses the DC-path), the W4-folded column rounding bias,
    and the int16 truncation of the intermediate between the row and column pass.

    -fwrapv is REQUIRED: the accumulators wrap mod 2^32 exactly like ffmpeg's
    SUINT (unsigned) arithmetic; the final `(int)(sum) >> SHIFT` is a signed
    arithmetic shift of that wrapped value.
*/

#ifndef MPEG4_SIMPLE_IDCT8_H
#define MPEG4_SIMPLE_IDCT8_H

#include <stdint.h>

namespace mp4 {

enum {
    IDCT_W1 = 22725, IDCT_W2 = 21407, IDCT_W3 = 19266, IDCT_W4 = 16383,
    IDCT_W5 = 12873, IDCT_W6 = 8867,  IDCT_W7 = 4520,
    IDCT_ROW_SHIFT = 11, IDCT_COL_SHIFT = 20, IDCT_DC_SHIFT = 3,
};

static inline uint8_t idct_clip_u8(int v) {
    return (uint8_t)(v < 0 ? 0 : (v > 255 ? 255 : v));
}

// One 8-point row IDCT, in place. `row` is int16[8]. extra_shift fixed at 0.
// Mirrors idctRowCondDC_int16_8bit (simple_idct_template.c:114-207).
static inline void idct_row(int16_t* row) {
    // DC-only fast path: condition row[1..7] all zero (ffmpeg AV_RN32A(row+2) |
    // AV_RN32A(row+4) | AV_RN32A(row+6) | row[1]).
    if (!(row[1] | row[2] | row[3] | row[4] | row[5] | row[6] | row[7])) {
        // temp = (row[0] << DC_SHIFT) & 0xffff, stored as int16 (sign-extended).
        int16_t t = (int16_t)((row[0] * (1 << IDCT_DC_SHIFT)) & 0xffff);
        row[0]=row[1]=row[2]=row[3]=row[4]=row[5]=row[6]=row[7] = t;
        return;
    }

    unsigned a0, a1, a2, a3, b0, b1, b2, b3;

    a0 = (unsigned)IDCT_W4 * row[0] + (1u << (IDCT_ROW_SHIFT - 1));
    a1 = a0; a2 = a0; a3 = a0;

    a0 += (unsigned)IDCT_W2 * row[2];
    a1 += (unsigned)IDCT_W6 * row[2];
    a2 -= (unsigned)IDCT_W6 * row[2];
    a3 -= (unsigned)IDCT_W2 * row[2];

    b0 = (unsigned)IDCT_W1 * row[1]; b0 += (unsigned)IDCT_W3 * row[3];
    b1 = (unsigned)IDCT_W3 * row[1]; b1 -= (unsigned)IDCT_W7 * row[3];
    b2 = (unsigned)IDCT_W5 * row[1]; b2 -= (unsigned)IDCT_W1 * row[3];
    b3 = (unsigned)IDCT_W7 * row[1]; b3 -= (unsigned)IDCT_W5 * row[3];

    if (row[4] | row[5] | row[6] | row[7]) {
        a0 += (unsigned)IDCT_W4 * row[4] + (unsigned)IDCT_W6 * row[6];
        a1 -= (unsigned)IDCT_W4 * row[4] + (unsigned)IDCT_W2 * row[6];
        a2 -= (unsigned)IDCT_W4 * row[4]; a2 += (unsigned)IDCT_W2 * row[6];
        a3 += (unsigned)IDCT_W4 * row[4]; a3 -= (unsigned)IDCT_W6 * row[6];

        b0 += (unsigned)IDCT_W5 * row[5]; b0 += (unsigned)IDCT_W7 * row[7];
        b1 -= (unsigned)IDCT_W1 * row[5]; b1 -= (unsigned)IDCT_W5 * row[7];
        b2 += (unsigned)IDCT_W7 * row[5]; b2 += (unsigned)IDCT_W3 * row[7];
        b3 += (unsigned)IDCT_W3 * row[5]; b3 -= (unsigned)IDCT_W1 * row[7];
    }

    row[0] = (int16_t)((int)(a0 + b0) >> IDCT_ROW_SHIFT);
    row[7] = (int16_t)((int)(a0 - b0) >> IDCT_ROW_SHIFT);
    row[1] = (int16_t)((int)(a1 + b1) >> IDCT_ROW_SHIFT);
    row[6] = (int16_t)((int)(a1 - b1) >> IDCT_ROW_SHIFT);
    row[2] = (int16_t)((int)(a2 + b2) >> IDCT_ROW_SHIFT);
    row[5] = (int16_t)((int)(a2 - b2) >> IDCT_ROW_SHIFT);
    row[3] = (int16_t)((int)(a3 + b3) >> IDCT_ROW_SHIFT);
    row[4] = (int16_t)((int)(a3 - b3) >> IDCT_ROW_SHIFT);
}

// Compute the 8 column butterflies (a0..a3,b0..b3) for the column whose top
// element is col[0], stride 8 (int16). Mirrors IDCT_COLS (template :209-257):
// the rounding bias is folded INSIDE the W4 multiply, and W*0 terms are skipped.
#define MP4_IDCT_COLS(col)                                                       \
    a0 = (unsigned)IDCT_W4 * ((col)[8*0] + ((1 << (IDCT_COL_SHIFT-1)) / IDCT_W4)); \
    a1 = a0; a2 = a0; a3 = a0;                                                    \
    a0 += (unsigned)IDCT_W2 * (col)[8*2];                                         \
    a1 += (unsigned)IDCT_W6 * (col)[8*2];                                         \
    a2 -= (unsigned)IDCT_W6 * (col)[8*2];                                         \
    a3 -= (unsigned)IDCT_W2 * (col)[8*2];                                         \
    b0 = (unsigned)IDCT_W1 * (col)[8*1];                                          \
    b1 = (unsigned)IDCT_W3 * (col)[8*1];                                          \
    b2 = (unsigned)IDCT_W5 * (col)[8*1];                                          \
    b3 = (unsigned)IDCT_W7 * (col)[8*1];                                          \
    b0 += (unsigned)IDCT_W3 * (col)[8*3];                                         \
    b1 -= (unsigned)IDCT_W7 * (col)[8*3];                                         \
    b2 -= (unsigned)IDCT_W1 * (col)[8*3];                                         \
    b3 -= (unsigned)IDCT_W5 * (col)[8*3];                                         \
    if ((col)[8*4]) {                                                            \
        a0 += (unsigned)IDCT_W4 * (col)[8*4];                                    \
        a1 -= (unsigned)IDCT_W4 * (col)[8*4];                                    \
        a2 -= (unsigned)IDCT_W4 * (col)[8*4];                                    \
        a3 += (unsigned)IDCT_W4 * (col)[8*4];                                    \
    }                                                                            \
    if ((col)[8*5]) {                                                            \
        b0 += (unsigned)IDCT_W5 * (col)[8*5];                                    \
        b1 -= (unsigned)IDCT_W1 * (col)[8*5];                                    \
        b2 += (unsigned)IDCT_W7 * (col)[8*5];                                    \
        b3 += (unsigned)IDCT_W3 * (col)[8*5];                                    \
    }                                                                            \
    if ((col)[8*6]) {                                                            \
        a0 += (unsigned)IDCT_W6 * (col)[8*6];                                    \
        a1 -= (unsigned)IDCT_W2 * (col)[8*6];                                    \
        a2 += (unsigned)IDCT_W2 * (col)[8*6];                                    \
        a3 -= (unsigned)IDCT_W6 * (col)[8*6];                                    \
    }                                                                            \
    if ((col)[8*7]) {                                                            \
        b0 += (unsigned)IDCT_W7 * (col)[8*7];                                    \
        b1 -= (unsigned)IDCT_W5 * (col)[8*7];                                    \
        b2 += (unsigned)IDCT_W3 * (col)[8*7];                                    \
        b3 -= (unsigned)IDCT_W1 * (col)[8*7];                                    \
    }

// Bare column IDCT: write int16 results back into the block (no clamp).
static inline void idct_col(int16_t* col) {
    unsigned a0, a1, a2, a3, b0, b1, b2, b3;
    MP4_IDCT_COLS(col);
    col[8*0] = (int16_t)((int)(a0 + b0) >> IDCT_COL_SHIFT);
    col[8*1] = (int16_t)((int)(a1 + b1) >> IDCT_COL_SHIFT);
    col[8*2] = (int16_t)((int)(a2 + b2) >> IDCT_COL_SHIFT);
    col[8*3] = (int16_t)((int)(a3 + b3) >> IDCT_COL_SHIFT);
    col[8*4] = (int16_t)((int)(a3 - b3) >> IDCT_COL_SHIFT);
    col[8*5] = (int16_t)((int)(a2 - b2) >> IDCT_COL_SHIFT);
    col[8*6] = (int16_t)((int)(a1 - b1) >> IDCT_COL_SHIFT);
    col[8*7] = (int16_t)((int)(a0 - b0) >> IDCT_COL_SHIFT);
}

// Column IDCT + clamp to [0,255] written to dest (intra, "put").
static inline void idct_col_put(uint8_t* dest, int stride, const int16_t* col) {
    unsigned a0, a1, a2, a3, b0, b1, b2, b3;
    MP4_IDCT_COLS(col);
    dest[0*stride] = idct_clip_u8((int)(a0 + b0) >> IDCT_COL_SHIFT);
    dest[1*stride] = idct_clip_u8((int)(a1 + b1) >> IDCT_COL_SHIFT);
    dest[2*stride] = idct_clip_u8((int)(a2 + b2) >> IDCT_COL_SHIFT);
    dest[3*stride] = idct_clip_u8((int)(a3 + b3) >> IDCT_COL_SHIFT);
    dest[4*stride] = idct_clip_u8((int)(a3 - b3) >> IDCT_COL_SHIFT);
    dest[5*stride] = idct_clip_u8((int)(a2 - b2) >> IDCT_COL_SHIFT);
    dest[6*stride] = idct_clip_u8((int)(a1 - b1) >> IDCT_COL_SHIFT);
    dest[7*stride] = idct_clip_u8((int)(a0 - b0) >> IDCT_COL_SHIFT);
}

// Column IDCT + add to dest + clamp (inter, "add").
static inline void idct_col_add(uint8_t* dest, int stride, const int16_t* col) {
    unsigned a0, a1, a2, a3, b0, b1, b2, b3;
    MP4_IDCT_COLS(col);
    dest[0*stride] = idct_clip_u8(dest[0*stride] + ((int)(a0 + b0) >> IDCT_COL_SHIFT));
    dest[1*stride] = idct_clip_u8(dest[1*stride] + ((int)(a1 + b1) >> IDCT_COL_SHIFT));
    dest[2*stride] = idct_clip_u8(dest[2*stride] + ((int)(a2 + b2) >> IDCT_COL_SHIFT));
    dest[3*stride] = idct_clip_u8(dest[3*stride] + ((int)(a3 + b3) >> IDCT_COL_SHIFT));
    dest[4*stride] = idct_clip_u8(dest[4*stride] + ((int)(a3 - b3) >> IDCT_COL_SHIFT));
    dest[5*stride] = idct_clip_u8(dest[5*stride] + ((int)(a2 - b2) >> IDCT_COL_SHIFT));
    dest[6*stride] = idct_clip_u8(dest[6*stride] + ((int)(a1 - b1) >> IDCT_COL_SHIFT));
    dest[7*stride] = idct_clip_u8(dest[7*stride] + ((int)(a0 - b0) >> IDCT_COL_SHIFT));
}

// ── 2-D drivers (8 row passes then 8 column passes) ─────────────────────────────

// Bare: int16 transform in place (ff_simple_idct, no put/add). For self-test.
static inline void idct8(int16_t* block) {
    for (int i = 0; i < 8; i++) idct_row(block + i*8);
    for (int i = 0; i < 8; i++) idct_col(block + i);
}

// Intra: clamp to [0,255] into dest[row*stride + col].
static inline void idct8_put(uint8_t* dest, int stride, int16_t* block) {
    for (int i = 0; i < 8; i++) idct_row(block + i*8);
    for (int i = 0; i < 8; i++) idct_col_put(dest + i, stride, block + i);
}

// Inter: residual add to dest then clamp.
static inline void idct8_add(uint8_t* dest, int stride, int16_t* block) {
    for (int i = 0; i < 8; i++) idct_row(block + i*8);
    for (int i = 0; i < 8; i++) idct_col_add(dest + i, stride, block + i);
}

#undef MP4_IDCT_COLS

} // namespace mp4

// ── SIMD (WebAssembly i32x4) ────────────────────────────────────────────────────
// Byte-identical to the scalar idct8 above. Mirrors the proven islow SIMD skeleton
// (plugins/mpeg/mpeg_decode.cpp): two 1-D passes with a 4x8 transpose, all ops i32x4
// wrapping mod 2^32 exactly like the scalar SUINT accumulators (-fwrapv). The mpeg4
// butterfly (W1..W7, ROW_SHIFT=11/COL_SHIFT=20, W4-folded column bias) is its own
// body. Reuses the transform-agnostic layer-1 primitives from common/wasm_simd_idct.h.
//
// Two non-obvious correctness points reproduced here:
//   * the ROW DC fast-path is NOT bit-identical to the full butterfly (row[0]=1025 ->
//     DC 8200 vs full 8199) and ffmpeg uses it, so DC-only rows are detected per-lane
//     (AC == 0) and blended in; the COLUMN pass has no DC path (matches scalar).
//   * the int16 truncation of the inter-pass intermediate (scalar stores row results
//     as int16) — applied via (x<<16)>>16 per lane, then round-tripped through int16 tmp.
#ifdef __wasm_simd128__
#include <wasm_simd128.h>
#include "../common/wasm_simd_idct.h"   // wsi_load16x4 / wsi_transpose4_i32 / wsi_store_trunc16

namespace mp4 {

enum { MP4_COL_BIAS = (1 << (IDCT_COL_SHIFT - 1)) / IDCT_W4 };   // = 32 (folded in W4 mul)

static inline v128_t mp4_imc(v128_t x, int c) { return wasm_i32x4_mul(x, wasm_i32x4_splat(c)); }
static inline v128_t mp4_i16trunc(v128_t x) { return wasm_i32x4_shr(wasm_i32x4_shl(x, 16), 16); }

static inline void mp4_store_u8x4(uint8_t* p, v128_t x) {            // clip [0,255], store 4 bytes
    x = wasm_i32x4_min(wasm_i32x4_max(x, wasm_i32x4_splat(0)), wasm_i32x4_splat(255));
    wasm_v128_store32_lane(p, wasm_u8x16_narrow_i16x8(wasm_i16x8_narrow_i32x4(x, x), x), 0);
}
static inline v128_t mp4_load_u8x4(const uint8_t* p) {              // 4 u8 -> i32x4
    return wasm_u32x4_extend_low_u16x8(wasm_u16x8_extend_low_u8x16(wasm_v128_load32_zero(p)));
}

// One 8-point ROW transform over 8 i32x4 element-vectors e[k] (lane = one of 4 rows);
// o[k] = (row IDCT >> ROW_SHIFT) int16-truncated, DC-only rows blended in.
static inline void mp4_idct_row_simd(const v128_t e[8], v128_t o[8]) {
    v128_t a0 = wasm_i32x4_add(mp4_imc(e[0], IDCT_W4), wasm_i32x4_splat(1 << (IDCT_ROW_SHIFT - 1)));
    v128_t a1 = a0, a2 = a0, a3 = a0;
    a0 = wasm_i32x4_add(a0, mp4_imc(e[2], IDCT_W2)); a1 = wasm_i32x4_add(a1, mp4_imc(e[2], IDCT_W6));
    a2 = wasm_i32x4_sub(a2, mp4_imc(e[2], IDCT_W6)); a3 = wasm_i32x4_sub(a3, mp4_imc(e[2], IDCT_W2));
    a0 = wasm_i32x4_add(a0, wasm_i32x4_add(mp4_imc(e[4], IDCT_W4), mp4_imc(e[6], IDCT_W6)));
    a1 = wasm_i32x4_sub(a1, wasm_i32x4_add(mp4_imc(e[4], IDCT_W4), mp4_imc(e[6], IDCT_W2)));
    a2 = wasm_i32x4_sub(a2, mp4_imc(e[4], IDCT_W4)); a2 = wasm_i32x4_add(a2, mp4_imc(e[6], IDCT_W2));
    a3 = wasm_i32x4_add(a3, mp4_imc(e[4], IDCT_W4)); a3 = wasm_i32x4_sub(a3, mp4_imc(e[6], IDCT_W6));
    v128_t b0 = wasm_i32x4_add(mp4_imc(e[1], IDCT_W1), mp4_imc(e[3], IDCT_W3));
    v128_t b1 = wasm_i32x4_sub(mp4_imc(e[1], IDCT_W3), mp4_imc(e[3], IDCT_W7));
    v128_t b2 = wasm_i32x4_sub(mp4_imc(e[1], IDCT_W5), mp4_imc(e[3], IDCT_W1));
    v128_t b3 = wasm_i32x4_sub(mp4_imc(e[1], IDCT_W7), mp4_imc(e[3], IDCT_W5));
    b0 = wasm_i32x4_add(b0, wasm_i32x4_add(mp4_imc(e[5], IDCT_W5), mp4_imc(e[7], IDCT_W7)));
    b1 = wasm_i32x4_sub(b1, wasm_i32x4_add(mp4_imc(e[5], IDCT_W1), mp4_imc(e[7], IDCT_W5)));
    b2 = wasm_i32x4_add(b2, wasm_i32x4_add(mp4_imc(e[5], IDCT_W7), mp4_imc(e[7], IDCT_W3)));
    b3 = wasm_i32x4_add(b3, wasm_i32x4_sub(mp4_imc(e[5], IDCT_W3), mp4_imc(e[7], IDCT_W1)));
    const int S = IDCT_ROW_SHIFT;
    v128_t f0 = mp4_i16trunc(wasm_i32x4_shr(wasm_i32x4_add(a0, b0), S));
    v128_t f7 = mp4_i16trunc(wasm_i32x4_shr(wasm_i32x4_sub(a0, b0), S));
    v128_t f1 = mp4_i16trunc(wasm_i32x4_shr(wasm_i32x4_add(a1, b1), S));
    v128_t f6 = mp4_i16trunc(wasm_i32x4_shr(wasm_i32x4_sub(a1, b1), S));
    v128_t f2 = mp4_i16trunc(wasm_i32x4_shr(wasm_i32x4_add(a2, b2), S));
    v128_t f5 = mp4_i16trunc(wasm_i32x4_shr(wasm_i32x4_sub(a2, b2), S));
    v128_t f3 = mp4_i16trunc(wasm_i32x4_shr(wasm_i32x4_add(a3, b3), S));
    v128_t f4 = mp4_i16trunc(wasm_i32x4_shr(wasm_i32x4_sub(a3, b3), S));
    // DC-only row blend: t = (int16_t)((row0 << DC_SHIFT) & 0xffff), all 8 outputs = t.
    v128_t dc = mp4_i16trunc(wasm_i32x4_shl(e[0], IDCT_DC_SHIFT));
    v128_t acOr = wasm_v128_or(wasm_v128_or(wasm_v128_or(e[1], e[2]), wasm_v128_or(e[3], e[4])),
                               wasm_v128_or(wasm_v128_or(e[5], e[6]), e[7]));
    v128_t m = wasm_i32x4_eq(acOr, wasm_i32x4_splat(0));     // all-ones where DC-only
    o[0] = wasm_v128_bitselect(dc, f0, m); o[1] = wasm_v128_bitselect(dc, f1, m);
    o[2] = wasm_v128_bitselect(dc, f2, m); o[3] = wasm_v128_bitselect(dc, f3, m);
    o[4] = wasm_v128_bitselect(dc, f4, m); o[5] = wasm_v128_bitselect(dc, f5, m);
    o[6] = wasm_v128_bitselect(dc, f6, m); o[7] = wasm_v128_bitselect(dc, f7, m);
}

// One 8-point COLUMN transform over 8 i32x4 vectors c[k] (lane = one of 4 columns);
// out[r] = column IDCT result >> COL_SHIFT (signed; caller clips/adds). No DC path.
static inline void mp4_idct_col_simd(const v128_t c[8], v128_t out[8]) {
    v128_t a0 = mp4_imc(wasm_i32x4_add(c[0], wasm_i32x4_splat(MP4_COL_BIAS)), IDCT_W4);
    v128_t a1 = a0, a2 = a0, a3 = a0;
    a0 = wasm_i32x4_add(a0, mp4_imc(c[2], IDCT_W2)); a1 = wasm_i32x4_add(a1, mp4_imc(c[2], IDCT_W6));
    a2 = wasm_i32x4_sub(a2, mp4_imc(c[2], IDCT_W6)); a3 = wasm_i32x4_sub(a3, mp4_imc(c[2], IDCT_W2));
    v128_t b0 = mp4_imc(c[1], IDCT_W1), b1 = mp4_imc(c[1], IDCT_W3), b2 = mp4_imc(c[1], IDCT_W5), b3 = mp4_imc(c[1], IDCT_W7);
    b0 = wasm_i32x4_add(b0, mp4_imc(c[3], IDCT_W3)); b1 = wasm_i32x4_sub(b1, mp4_imc(c[3], IDCT_W7));
    b2 = wasm_i32x4_sub(b2, mp4_imc(c[3], IDCT_W1)); b3 = wasm_i32x4_sub(b3, mp4_imc(c[3], IDCT_W5));
    a0 = wasm_i32x4_add(a0, mp4_imc(c[4], IDCT_W4)); a1 = wasm_i32x4_sub(a1, mp4_imc(c[4], IDCT_W4));
    a2 = wasm_i32x4_sub(a2, mp4_imc(c[4], IDCT_W4)); a3 = wasm_i32x4_add(a3, mp4_imc(c[4], IDCT_W4));
    b0 = wasm_i32x4_add(b0, mp4_imc(c[5], IDCT_W5)); b1 = wasm_i32x4_sub(b1, mp4_imc(c[5], IDCT_W1));
    b2 = wasm_i32x4_add(b2, mp4_imc(c[5], IDCT_W7)); b3 = wasm_i32x4_add(b3, mp4_imc(c[5], IDCT_W3));
    a0 = wasm_i32x4_add(a0, mp4_imc(c[6], IDCT_W6)); a1 = wasm_i32x4_sub(a1, mp4_imc(c[6], IDCT_W2));
    a2 = wasm_i32x4_add(a2, mp4_imc(c[6], IDCT_W2)); a3 = wasm_i32x4_sub(a3, mp4_imc(c[6], IDCT_W6));
    b0 = wasm_i32x4_add(b0, mp4_imc(c[7], IDCT_W7)); b1 = wasm_i32x4_sub(b1, mp4_imc(c[7], IDCT_W5));
    b2 = wasm_i32x4_add(b2, mp4_imc(c[7], IDCT_W3)); b3 = wasm_i32x4_sub(b3, mp4_imc(c[7], IDCT_W1));
    const int S = IDCT_COL_SHIFT;
    out[0] = wasm_i32x4_shr(wasm_i32x4_add(a0, b0), S); out[7] = wasm_i32x4_shr(wasm_i32x4_sub(a0, b0), S);
    out[1] = wasm_i32x4_shr(wasm_i32x4_add(a1, b1), S); out[6] = wasm_i32x4_shr(wasm_i32x4_sub(a1, b1), S);
    out[2] = wasm_i32x4_shr(wasm_i32x4_add(a2, b2), S); out[5] = wasm_i32x4_shr(wasm_i32x4_sub(a2, b2), S);
    out[3] = wasm_i32x4_shr(wasm_i32x4_add(a3, b3), S); out[4] = wasm_i32x4_shr(wasm_i32x4_sub(a3, b3), S);
}

// Row pass over the whole block -> int16 tmp[64] (row-major, int16-truncated).
static inline void mp4_idct_rows_simd(const int16_t* block, int16_t* tmp) {
    for (int rb = 0; rb < 8; rb += 4) {
        v128_t e[8];
        wsi_transpose4_i32(wsi_load16x4(block + (rb + 0) * 8), wsi_load16x4(block + (rb + 1) * 8),
                           wsi_load16x4(block + (rb + 2) * 8), wsi_load16x4(block + (rb + 3) * 8),
                           &e[0], &e[1], &e[2], &e[3]);
        wsi_transpose4_i32(wsi_load16x4(block + (rb + 0) * 8 + 4), wsi_load16x4(block + (rb + 1) * 8 + 4),
                           wsi_load16x4(block + (rb + 2) * 8 + 4), wsi_load16x4(block + (rb + 3) * 8 + 4),
                           &e[4], &e[5], &e[6], &e[7]);
        v128_t o[8]; mp4_idct_row_simd(e, o);
        v128_t r0, r1, r2, r3, r4, r5, r6, r7;
        wsi_transpose4_i32(o[0], o[1], o[2], o[3], &r0, &r1, &r2, &r3);   // -> rows rb+0..3, cols 0-3
        wsi_transpose4_i32(o[4], o[5], o[6], o[7], &r4, &r5, &r6, &r7);   // -> rows rb+0..3, cols 4-7
        wsi_store_trunc16(tmp + (rb + 0) * 8, r0); wsi_store_trunc16(tmp + (rb + 0) * 8 + 4, r4);
        wsi_store_trunc16(tmp + (rb + 1) * 8, r1); wsi_store_trunc16(tmp + (rb + 1) * 8 + 4, r5);
        wsi_store_trunc16(tmp + (rb + 2) * 8, r2); wsi_store_trunc16(tmp + (rb + 2) * 8 + 4, r6);
        wsi_store_trunc16(tmp + (rb + 3) * 8, r3); wsi_store_trunc16(tmp + (rb + 3) * 8 + 4, r7);
    }
}

static inline void idct8_simd(int16_t* block) {
    int16_t tmp[64]; mp4_idct_rows_simd(block, tmp);
    for (int cb = 0; cb < 8; cb += 4) {
        v128_t c[8]; for (int r = 0; r < 8; r++) c[r] = wsi_load16x4(tmp + r * 8 + cb);
        v128_t o[8]; mp4_idct_col_simd(c, o);
        for (int r = 0; r < 8; r++) wsi_store_trunc16(block + r * 8 + cb, o[r]);
    }
}
static inline void idct8_put_simd(uint8_t* dest, int stride, int16_t* block) {
    int16_t tmp[64]; mp4_idct_rows_simd(block, tmp);
    for (int cb = 0; cb < 8; cb += 4) {
        v128_t c[8]; for (int r = 0; r < 8; r++) c[r] = wsi_load16x4(tmp + r * 8 + cb);
        v128_t o[8]; mp4_idct_col_simd(c, o);
        for (int r = 0; r < 8; r++) mp4_store_u8x4(dest + r * stride + cb, o[r]);
    }
}
static inline void idct8_add_simd(uint8_t* dest, int stride, int16_t* block) {
    int16_t tmp[64]; mp4_idct_rows_simd(block, tmp);
    for (int cb = 0; cb < 8; cb += 4) {
        v128_t c[8]; for (int r = 0; r < 8; r++) c[r] = wsi_load16x4(tmp + r * 8 + cb);
        v128_t o[8]; mp4_idct_col_simd(c, o);
        for (int r = 0; r < 8; r++)
            mp4_store_u8x4(dest + r * stride + cb, wasm_i32x4_add(mp4_load_u8x4(dest + r * stride + cb), o[r]));
    }
}

} // namespace mp4
#else  // no SIMD: alias the *_simd entry points to the scalar transform
namespace mp4 {
static inline void idct8_simd(int16_t* b) { idct8(b); }
static inline void idct8_put_simd(uint8_t* d, int s, int16_t* b) { idct8_put(d, s, b); }
static inline void idct8_add_simd(uint8_t* d, int s, int16_t* b) { idct8_add(d, s, b); }
} // namespace mp4
#endif // __wasm_simd128__

#endif // MPEG4_SIMPLE_IDCT8_H
