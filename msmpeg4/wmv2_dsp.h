/*
    WMV2 (Windows Media Video 8) pixel kernels — transcription of FFmpeg release/8.1:
      wmv2dsp.c          wmv2_idct_row / wmv2_idct_col / wmv2_idct_{put,add}_c (the 8x8
                         IDCT WMV2 installs over the default one, for intra put AND
                         inter add, and the one IntraX8 uses)
      simple_idct.c      ff_simple_idct84_add / ff_simple_idct48_add (the ABT 8x4 / 4x8
                         transforms: idct4col_add / idct4row + the 8-point simple_idct
                         row / sparse-column-add, reused from ../mpeg4/simple_idct8.h)
      wmv2dec.c          wmv2_mspel8_{h,v}_lowpass + put_mspel8_mcXX_c (the 8 mspel
                         luma interpolators)
      h263dsp.c          h263_{h,v}_loop_filter_c + ff_h263_loop_filter_strength
    FFmpeg is LGPL-2.1+; see msmpeg4_decode.h for the module-wide provenance note.

    The mspel functions take separate dst / src strides (FFmpeg passes one stride for
    both, because its edge-emulation buffer shares the frame linesize); the arithmetic
    is unchanged. -fwrapv is REQUIRED (the 181U products wrap like FFmpeg's).
*/

#ifndef WMV2_DSP_H
#define WMV2_DSP_H

#include <stdint.h>
#include <string.h>

#include "../mpeg4/simple_idct8.h"   // mp4::idct_row (idctRowCondDC), mp4::idct_col_add (idctSparseColAdd)

namespace msm {

static inline uint8_t wmv2_clip_u8(int v) { return (uint8_t)(v < 0 ? 0 : (v > 255 ? 255 : v)); }

// ── wmv2dsp.c: the WMV2 8x8 IDCT ────────────────────────────────────────────────
enum { WMV2_W0 = 2048, WMV2_W1 = 2841, WMV2_W2 = 2676, WMV2_W3 = 2408,
       WMV2_W4 = 2048, WMV2_W5 = 1609, WMV2_W6 = 1108, WMV2_W7 = 565 };

static inline void wmv2_idct_row(int16_t* b) {
    int s1, s2;
    int a0, a1, a2, a3, a4, a5, a6, a7;

    /* step 1 */
    a1 = WMV2_W1 * b[1] + WMV2_W7 * b[7];
    a7 = WMV2_W7 * b[1] - WMV2_W1 * b[7];
    a5 = WMV2_W5 * b[5] + WMV2_W3 * b[3];
    a3 = WMV2_W3 * b[5] - WMV2_W5 * b[3];
    a2 = WMV2_W2 * b[2] + WMV2_W6 * b[6];
    a6 = WMV2_W6 * b[2] - WMV2_W2 * b[6];
    a0 = WMV2_W0 * b[0] + WMV2_W0 * b[4];
    a4 = WMV2_W0 * b[0] - WMV2_W0 * b[4];

    /* step 2 */
    s1 = (int)(181U * (unsigned)(a1 - a5 + a7 - a3) + 128) >> 8; // 1, 3, 5, 7
    s2 = (int)(181U * (unsigned)(a1 - a5 - a7 + a3) + 128) >> 8;

    /* step 3 */
    b[0] = (int16_t)((a0 + a2 + a1 + a5 + (1 << 7)) >> 8);
    b[1] = (int16_t)((a4 + a6 + s1      + (1 << 7)) >> 8);
    b[2] = (int16_t)((a4 - a6 + s2      + (1 << 7)) >> 8);
    b[3] = (int16_t)((a0 - a2 + a7 + a3 + (1 << 7)) >> 8);
    b[4] = (int16_t)((a0 - a2 - a7 - a3 + (1 << 7)) >> 8);
    b[5] = (int16_t)((a4 - a6 - s2      + (1 << 7)) >> 8);
    b[6] = (int16_t)((a4 + a6 - s1      + (1 << 7)) >> 8);
    b[7] = (int16_t)((a0 + a2 - a1 - a5 + (1 << 7)) >> 8);
}

static inline void wmv2_idct_col(int16_t* b) {
    int s1, s2;
    int a0, a1, a2, a3, a4, a5, a6, a7;

    /* step 1, with extended precision */
    a1 = (WMV2_W1 * b[8 * 1] + WMV2_W7 * b[8 * 7] + 4) >> 3;
    a7 = (WMV2_W7 * b[8 * 1] - WMV2_W1 * b[8 * 7] + 4) >> 3;
    a5 = (WMV2_W5 * b[8 * 5] + WMV2_W3 * b[8 * 3] + 4) >> 3;
    a3 = (WMV2_W3 * b[8 * 5] - WMV2_W5 * b[8 * 3] + 4) >> 3;
    a2 = (WMV2_W2 * b[8 * 2] + WMV2_W6 * b[8 * 6] + 4) >> 3;
    a6 = (WMV2_W6 * b[8 * 2] - WMV2_W2 * b[8 * 6] + 4) >> 3;
    a0 = (WMV2_W0 * b[8 * 0] + WMV2_W0 * b[8 * 4]    ) >> 3;
    a4 = (WMV2_W0 * b[8 * 0] - WMV2_W0 * b[8 * 4]    ) >> 3;

    /* step 2 */
    s1 = (int)(181U * (unsigned)(a1 - a5 + a7 - a3) + 128) >> 8;
    s2 = (int)(181U * (unsigned)(a1 - a5 - a7 + a3) + 128) >> 8;

    /* step 3 */
    b[8 * 0] = (int16_t)((a0 + a2 + a1 + a5 + (1 << 13)) >> 14);
    b[8 * 1] = (int16_t)((a4 + a6 + s1      + (1 << 13)) >> 14);
    b[8 * 2] = (int16_t)((a4 - a6 + s2      + (1 << 13)) >> 14);
    b[8 * 3] = (int16_t)((a0 - a2 + a7 + a3 + (1 << 13)) >> 14);

    b[8 * 4] = (int16_t)((a0 - a2 - a7 - a3 + (1 << 13)) >> 14);
    b[8 * 5] = (int16_t)((a4 - a6 - s2      + (1 << 13)) >> 14);
    b[8 * 6] = (int16_t)((a4 + a6 - s1      + (1 << 13)) >> 14);
    b[8 * 7] = (int16_t)((a0 + a2 - a1 - a5 + (1 << 13)) >> 14);
}

static inline void wmv2_idct_add(uint8_t* dest, int line_size, int16_t* block) {
    for (int i = 0; i < 64; i += 8) wmv2_idct_row(block + i);
    for (int i = 0; i < 8; i++)     wmv2_idct_col(block + i);
    for (int i = 0; i < 8; i++) {
        for (int k = 0; k < 8; k++) dest[k] = wmv2_clip_u8(dest[k] + block[k]);
        dest  += line_size;
        block += 8;
    }
}

static inline void wmv2_idct_put(uint8_t* dest, int line_size, int16_t* block) {
    for (int i = 0; i < 64; i += 8) wmv2_idct_row(block + i);
    for (int i = 0; i < 8; i++)     wmv2_idct_col(block + i);
    for (int i = 0; i < 8; i++) {
        for (int k = 0; k < 8; k++) dest[k] = wmv2_clip_u8(block[k]);
        dest  += line_size;
        block += 8;
    }
}

// ── simple_idct.c: the ABT 8x4 / 4x8 transforms ─────────────────────────────────
// C_FIX / R_FIX evaluated exactly as FFmpeg does (double, +0.5, truncate).
#define WMV2_SQRT2 1.41421356237309504880
enum {
    ABT_CN_SHIFT = 12, ABT_C_SHIFT = 4 + 1 + 12,
    ABT_C1 = (int)(0.6532814824 * WMV2_SQRT2 * (1 << ABT_CN_SHIFT) + 0.5),
    ABT_C2 = (int)(0.2705980501 * WMV2_SQRT2 * (1 << ABT_CN_SHIFT) + 0.5),
    ABT_C3 = (int)(0.5          * WMV2_SQRT2 * (1 << ABT_CN_SHIFT) + 0.5),
    ABT_RN_SHIFT = 15, ABT_R_SHIFT = 11,
    ABT_R1 = (int)(0.6532814824 * WMV2_SQRT2 * (1 << ABT_RN_SHIFT) + 0.5),
    ABT_R2 = (int)(0.2705980501 * WMV2_SQRT2 * (1 << ABT_RN_SHIFT) + 0.5),
    ABT_R3 = (int)(0.5          * WMV2_SQRT2 * (1 << ABT_RN_SHIFT) + 0.5),
};
#undef WMV2_SQRT2

static inline void abt_idct4col_add(uint8_t* dest, int line_size, const int16_t* col) {
    int c0, c1, c2, c3, a0, a1, a2, a3;

    a0 = col[8*0];
    a1 = col[8*1];
    a2 = col[8*2];
    a3 = col[8*3];
    c0 = (a0 + a2)*ABT_C3 + (1 << (ABT_C_SHIFT - 1));
    c2 = (a0 - a2)*ABT_C3 + (1 << (ABT_C_SHIFT - 1));
    c1 = a1 * ABT_C1 + a3 * ABT_C2;
    c3 = a1 * ABT_C2 - a3 * ABT_C1;
    dest[0] = wmv2_clip_u8(dest[0] + ((c0 + c1) >> ABT_C_SHIFT));
    dest += line_size;
    dest[0] = wmv2_clip_u8(dest[0] + ((c2 + c3) >> ABT_C_SHIFT));
    dest += line_size;
    dest[0] = wmv2_clip_u8(dest[0] + ((c2 - c3) >> ABT_C_SHIFT));
    dest += line_size;
    dest[0] = wmv2_clip_u8(dest[0] + ((c0 - c1) >> ABT_C_SHIFT));
}

static inline void abt_idct4row(int16_t* row) {
    unsigned c0, c1, c2, c3;
    int a0, a1, a2, a3;

    a0 = row[0];
    a1 = row[1];
    a2 = row[2];
    a3 = row[3];
    c0 = (unsigned)((a0 + a2)*ABT_R3 + (1 << (ABT_R_SHIFT - 1)));
    c2 = (unsigned)((a0 - a2)*ABT_R3 + (1 << (ABT_R_SHIFT - 1)));
    c1 = (unsigned)(a1 * ABT_R1 + a3 * ABT_R2);
    c3 = (unsigned)(a1 * ABT_R2 - a3 * ABT_R1);
    row[0] = (int16_t)((c0 + c1) >> ABT_R_SHIFT);
    row[1] = (int16_t)((c2 + c3) >> ABT_R_SHIFT);
    row[2] = (int16_t)((c2 - c3) >> ABT_R_SHIFT);
    row[3] = (int16_t)((c0 - c1) >> ABT_R_SHIFT);
}

// ff_simple_idct84_add: IDCT8 on each of 4 rows, then IDCT4 down each column.
static inline void simple_idct84_add(uint8_t* dest, int line_size, int16_t* block) {
    for (int i = 0; i < 4; i++) mp4::idct_row(block + i*8);
    for (int i = 0; i < 8; i++) abt_idct4col_add(dest + i, line_size, block + i);
}

// ff_simple_idct48_add: IDCT4 on each of 8 rows, then IDCT8 down each of 4 columns.
static inline void simple_idct48_add(uint8_t* dest, int line_size, int16_t* block) {
    for (int i = 0; i < 8; i++) abt_idct4row(block + i*8);
    for (int i = 0; i < 4; i++) mp4::idct_col_add(dest + i, line_size, block + i);
}

// ── wmv2dec.c: mspel luma interpolation ─────────────────────────────────────────
static inline void wmv2_mspel8_h_lowpass(uint8_t* dst, const uint8_t* src,
                                         int dstStride, int srcStride, int h) {
    for (int i = 0; i < h; i++) {
        dst[0] = wmv2_clip_u8((9 * (src[0] + src[1]) - (src[-1] + src[2]) + 8) >> 4);
        dst[1] = wmv2_clip_u8((9 * (src[1] + src[2]) - (src[0]  + src[3]) + 8) >> 4);
        dst[2] = wmv2_clip_u8((9 * (src[2] + src[3]) - (src[1]  + src[4]) + 8) >> 4);
        dst[3] = wmv2_clip_u8((9 * (src[3] + src[4]) - (src[2]  + src[5]) + 8) >> 4);
        dst[4] = wmv2_clip_u8((9 * (src[4] + src[5]) - (src[3]  + src[6]) + 8) >> 4);
        dst[5] = wmv2_clip_u8((9 * (src[5] + src[6]) - (src[4]  + src[7]) + 8) >> 4);
        dst[6] = wmv2_clip_u8((9 * (src[6] + src[7]) - (src[5]  + src[8]) + 8) >> 4);
        dst[7] = wmv2_clip_u8((9 * (src[7] + src[8]) - (src[6]  + src[9]) + 8) >> 4);
        dst += dstStride;
        src += srcStride;
    }
}

static inline void wmv2_mspel8_v_lowpass(uint8_t* dst, const uint8_t* src,
                                         int dstStride, int srcStride, int w) {
    for (int i = 0; i < w; i++) {
        const int src_1 = src[-srcStride];
        const int src0  = src[0];
        const int src1  = src[srcStride];
        const int src2  = src[2 * srcStride];
        const int src3  = src[3 * srcStride];
        const int src4  = src[4 * srcStride];
        const int src5  = src[5 * srcStride];
        const int src6  = src[6 * srcStride];
        const int src7  = src[7 * srcStride];
        const int src8  = src[8 * srcStride];
        const int src9  = src[9 * srcStride];
        dst[0 * dstStride] = wmv2_clip_u8((9 * (src0 + src1) - (src_1 + src2) + 8) >> 4);
        dst[1 * dstStride] = wmv2_clip_u8((9 * (src1 + src2) - (src0  + src3) + 8) >> 4);
        dst[2 * dstStride] = wmv2_clip_u8((9 * (src2 + src3) - (src1  + src4) + 8) >> 4);
        dst[3 * dstStride] = wmv2_clip_u8((9 * (src3 + src4) - (src2  + src5) + 8) >> 4);
        dst[4 * dstStride] = wmv2_clip_u8((9 * (src4 + src5) - (src3  + src6) + 8) >> 4);
        dst[5 * dstStride] = wmv2_clip_u8((9 * (src5 + src6) - (src4  + src7) + 8) >> 4);
        dst[6 * dstStride] = wmv2_clip_u8((9 * (src6 + src7) - (src5  + src8) + 8) >> 4);
        dst[7 * dstStride] = wmv2_clip_u8((9 * (src7 + src8) - (src6  + src9) + 8) >> 4);
        src++;
        dst++;
    }
}

// ff_put_pixels8_l2_8: rounded average of two 8x8 sources (rnd_avg32 per byte).
static inline void wmv2_put_pixels8_l2(uint8_t* dst, const uint8_t* src1, const uint8_t* src2,
                                       int dst_stride, int src_stride1, int src_stride2, int h) {
    for (int i = 0; i < h; i++) {
        for (int k = 0; k < 8; k++) dst[k] = (uint8_t)((src1[k] + src2[k] + 1) >> 1);
        dst += dst_stride; src1 += src_stride1; src2 += src_stride2;
    }
}

// put_mspel_pixels_tab[dxy], dxy = 2*(((my&1)<<1)|(mx&1)) + hshift.
static inline void wmv2_put_mspel8(int dxy, uint8_t* dst, int ds, const uint8_t* src, int ss) {
    uint8_t half[64], halfH[88], halfV[64], halfHV[64];
    switch (dxy) {
    case 0:   // ff_put_pixels8x8_c
        for (int i = 0; i < 8; i++) memcpy(dst + i * ds, src + i * ss, 8);
        break;
    case 1:   // put_mspel8_mc10_c
        wmv2_mspel8_h_lowpass(half, src, 8, ss, 8);
        wmv2_put_pixels8_l2(dst, src, half, ds, ss, 8, 8);
        break;
    case 2:   // put_mspel8_mc20_c
        wmv2_mspel8_h_lowpass(dst, src, ds, ss, 8);
        break;
    case 3:   // put_mspel8_mc30_c
        wmv2_mspel8_h_lowpass(half, src, 8, ss, 8);
        wmv2_put_pixels8_l2(dst, src + 1, half, ds, ss, 8, 8);
        break;
    case 4:   // put_mspel8_mc02_c
        wmv2_mspel8_v_lowpass(dst, src, ds, ss, 8);
        break;
    case 5:   // put_mspel8_mc12_c
        wmv2_mspel8_h_lowpass(halfH, src - ss, 8, ss, 11);
        wmv2_mspel8_v_lowpass(halfV, src, 8, ss, 8);
        wmv2_mspel8_v_lowpass(halfHV, halfH + 8, 8, 8, 8);
        wmv2_put_pixels8_l2(dst, halfV, halfHV, ds, 8, 8, 8);
        break;
    case 6:   // put_mspel8_mc22_c
        wmv2_mspel8_h_lowpass(halfH, src - ss, 8, ss, 11);
        wmv2_mspel8_v_lowpass(dst, halfH + 8, ds, 8, 8);
        break;
    default:  // 7: put_mspel8_mc32_c
        wmv2_mspel8_h_lowpass(halfH, src - ss, 8, ss, 11);
        wmv2_mspel8_v_lowpass(halfV, src + 1, 8, ss, 8);
        wmv2_mspel8_v_lowpass(halfHV, halfH + 8, 8, 8, 8);
        wmv2_put_pixels8_l2(dst, halfV, halfHV, ds, 8, 8, 8);
        break;
    }
}

// ── h263dsp.c: the H.263 deblocking loop filter ─────────────────────────────────
static const uint8_t ff_h263_loop_filter_strength[32] = {
    0, 1, 1, 2, 2, 3, 3,  4,  4,  4,  5,  5,  6,  6,  7, 7,
    7, 8, 8, 8, 9, 9, 9, 10, 10, 10, 11, 11, 11, 12, 12, 12
};

static inline int h263_clip(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

// One 8-sample filter across an edge. `a` steps across the edge (p0..p3), `b` steps
// along it. h_loop_filter: a=1 (vertical edge), b=stride; v_loop_filter: a=stride, b=1.
static inline void h263_loop_filter8(uint8_t* src, int a, int b, int qscale) {
    const int strength = ff_h263_loop_filter_strength[qscale];
    for (int k = 0; k < 8; k++, src += b) {
        int d1, d2, ad1;
        int p0 = src[-2 * a];
        int p1 = src[-1 * a];
        int p2 = src[ 0];
        int p3 = src[ 1 * a];
        int d  = (p0 - p3 + 4 * (p2 - p1)) / 8;

        if (d < -2 * strength)
            d1 = 0;
        else if (d < -strength)
            d1 = -2 * strength - d;
        else if (d < strength)
            d1 = d;
        else if (d < 2 * strength)
            d1 = 2 * strength - d;
        else
            d1 = 0;

        p1 += d1;
        p2 -= d1;
        if (p1 & 256)
            p1 = ~(p1 >> 31);
        if (p2 & 256)
            p2 = ~(p2 >> 31);

        src[-1 * a] = (uint8_t)p1;
        src[ 0]     = (uint8_t)p2;

        ad1 = (d1 < 0 ? -d1 : d1) >> 1;

        d2 = h263_clip((p0 - p3) / 4, -ad1, ad1);

        src[-2 * a] = (uint8_t)(p0 - d2);
        src[ 1 * a] = (uint8_t)(p3 + d2);
    }
}

static inline void h263_h_loop_filter(uint8_t* src, int stride, int qscale) { h263_loop_filter8(src, 1, stride, qscale); }
static inline void h263_v_loop_filter(uint8_t* src, int stride, int qscale) { h263_loop_filter8(src, stride, 1, qscale); }

} // namespace msm

#endif // WMV2_DSP_H
