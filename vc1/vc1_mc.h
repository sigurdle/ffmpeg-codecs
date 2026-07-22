/*
    VC-1 motion-compensation + residual DSP leaf kernels — verbatim transcription of
    FFmpeg release/8.1 libavcodec/vc1dsp.c (LGPL-2.1+). Pure functions, no decoder
    context: the variable-size inverse transforms (which ADD their residual to the
    motion-compensated destination), the mspel quarter-pel luma interpolation, the
    VC-1/H.264 chroma bilinear MC, and an emulated-edge (clamped) block fetch.

    The full 8x8 inverse transform (vc1_inv_trans_8x8, block->block) lives in
    vc1_decode.h and is shared with the I-frame path; everything here is P-frame MC.

    -fwrapv REQUIRED (transform / filter sign arithmetic byte-identical to ffmpeg).
*/

#ifndef VC1_MC_H
#define VC1_MC_H

#include <stdint.h>
#include <string.h>

namespace vc1 {

static inline uint8_t vc1_clip8(int x) { return (uint8_t)(x < 0 ? 0 : (x > 255 ? 255 : x)); }

// ff_add_pixels_clamped (idctdsp): dst[i] += block[i], clamped. Used after the full
// 8x8 inter inverse transform (block->block) to add the residual to the MC prediction.
static inline void vc1_add_pixels_clamped(const int16_t* block, uint8_t* dst, int stride) {
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) dst[j] = vc1_clip8(dst[j] + block[j]);
        block += 8; dst += stride;
    }
}

// ── DC-only inverse transforms (add to dest) ────────────────────────────────────
static inline void vc1_inv_trans_8x8_dc(uint8_t* dest, int stride, int16_t* block) {
    int dc = block[0];
    dc = (3 * dc +  1) >> 1;
    dc = (3 * dc + 16) >> 5;
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) dest[j] = vc1_clip8(dest[j] + dc);
        dest += stride;
    }
}
static inline void vc1_inv_trans_8x4_dc(uint8_t* dest, int stride, int16_t* block) {
    int dc = block[0];
    dc =  (3 * dc +  1) >> 1;
    dc = (17 * dc + 64) >> 7;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 8; j++) dest[j] = vc1_clip8(dest[j] + dc);
        dest += stride;
    }
}
static inline void vc1_inv_trans_4x8_dc(uint8_t* dest, int stride, int16_t* block) {
    int dc = block[0];
    dc = (17 * dc +  4) >> 3;
    dc = (12 * dc + 64) >> 7;
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 4; j++) dest[j] = vc1_clip8(dest[j] + dc);
        dest += stride;
    }
}
static inline void vc1_inv_trans_4x4_dc(uint8_t* dest, int stride, int16_t* block) {
    int dc = block[0];
    dc = (17 * dc +  4) >> 3;
    dc = (17 * dc + 64) >> 7;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) dest[j] = vc1_clip8(dest[j] + dc);
        dest += stride;
    }
}

// ── Full variable-size inverse transforms (add residual to dest) ────────────────
static inline void vc1_inv_trans_8x4(uint8_t* dest, int stride, int16_t* block) {
    int t1, t2, t3, t4, t5, t6, t7, t8;
    int16_t* src = block; int16_t* dst = block;
    for (int i = 0; i < 4; i++) {
        t1 = 12 * (src[0] + src[4]) + 4;  t2 = 12 * (src[0] - src[4]) + 4;
        t3 = 16 * src[2] +  6 * src[6];   t4 =  6 * src[2] - 16 * src[6];
        t5 = t1 + t3; t6 = t2 + t4; t7 = t2 - t4; t8 = t1 - t3;
        t1 = 16 * src[1] + 15 * src[3] +  9 * src[5] +  4 * src[7];
        t2 = 15 * src[1] -  4 * src[3] - 16 * src[5] -  9 * src[7];
        t3 =  9 * src[1] - 16 * src[3] +  4 * src[5] + 15 * src[7];
        t4 =  4 * src[1] -  9 * src[3] + 15 * src[5] - 16 * src[7];
        dst[0] = (int16_t)((t5 + t1) >> 3); dst[1] = (int16_t)((t6 + t2) >> 3);
        dst[2] = (int16_t)((t7 + t3) >> 3); dst[3] = (int16_t)((t8 + t4) >> 3);
        dst[4] = (int16_t)((t8 - t4) >> 3); dst[5] = (int16_t)((t7 - t3) >> 3);
        dst[6] = (int16_t)((t6 - t2) >> 3); dst[7] = (int16_t)((t5 - t1) >> 3);
        src += 8; dst += 8;
    }
    src = block;
    for (int i = 0; i < 8; i++) {
        t1 = 17 * (src[0] + src[16]) + 64; t2 = 17 * (src[0] - src[16]) + 64;
        t3 = 22 * src[8] + 10 * src[24];   t4 = 22 * src[24] - 10 * src[8];
        dest[0 * stride] = vc1_clip8(dest[0 * stride] + ((t1 + t3) >> 7));
        dest[1 * stride] = vc1_clip8(dest[1 * stride] + ((t2 - t4) >> 7));
        dest[2 * stride] = vc1_clip8(dest[2 * stride] + ((t2 + t4) >> 7));
        dest[3 * stride] = vc1_clip8(dest[3 * stride] + ((t1 - t3) >> 7));
        src++; dest++;
    }
}
static inline void vc1_inv_trans_4x8(uint8_t* dest, int stride, int16_t* block) {
    int t1, t2, t3, t4, t5, t6, t7, t8;
    int16_t* src = block; int16_t* dst = block;
    for (int i = 0; i < 8; i++) {
        t1 = 17 * (src[0] + src[2]) + 4; t2 = 17 * (src[0] - src[2]) + 4;
        t3 = 22 * src[1] + 10 * src[3];  t4 = 22 * src[3] - 10 * src[1];
        dst[0] = (int16_t)((t1 + t3) >> 3); dst[1] = (int16_t)((t2 - t4) >> 3);
        dst[2] = (int16_t)((t2 + t4) >> 3); dst[3] = (int16_t)((t1 - t3) >> 3);
        src += 8; dst += 8;
    }
    src = block;
    for (int i = 0; i < 4; i++) {
        t1 = 12 * (src[0] + src[32]) + 64; t2 = 12 * (src[0] - src[32]) + 64;
        t3 = 16 * src[16] +  6 * src[48];  t4 =  6 * src[16] - 16 * src[48];
        t5 = t1 + t3; t6 = t2 + t4; t7 = t2 - t4; t8 = t1 - t3;
        t1 = 16 * src[8] + 15 * src[24] +  9 * src[40] +  4 * src[56];
        t2 = 15 * src[8] -  4 * src[24] - 16 * src[40] -  9 * src[56];
        t3 =  9 * src[8] - 16 * src[24] +  4 * src[40] + 15 * src[56];
        t4 =  4 * src[8] -  9 * src[24] + 15 * src[40] - 16 * src[56];
        dest[0 * stride] = vc1_clip8(dest[0 * stride] + ((t5 + t1)     >> 7));
        dest[1 * stride] = vc1_clip8(dest[1 * stride] + ((t6 + t2)     >> 7));
        dest[2 * stride] = vc1_clip8(dest[2 * stride] + ((t7 + t3)     >> 7));
        dest[3 * stride] = vc1_clip8(dest[3 * stride] + ((t8 + t4)     >> 7));
        dest[4 * stride] = vc1_clip8(dest[4 * stride] + ((t8 - t4 + 1) >> 7));
        dest[5 * stride] = vc1_clip8(dest[5 * stride] + ((t7 - t3 + 1) >> 7));
        dest[6 * stride] = vc1_clip8(dest[6 * stride] + ((t6 - t2 + 1) >> 7));
        dest[7 * stride] = vc1_clip8(dest[7 * stride] + ((t5 - t1 + 1) >> 7));
        src++; dest++;
    }
}
static inline void vc1_inv_trans_4x4(uint8_t* dest, int stride, int16_t* block) {
    int t1, t2, t3, t4;
    int16_t* src = block; int16_t* dst = block;
    for (int i = 0; i < 4; i++) {
        t1 = 17 * (src[0] + src[2]) + 4; t2 = 17 * (src[0] - src[2]) + 4;
        t3 = 22 * src[1] + 10 * src[3];  t4 = 22 * src[3] - 10 * src[1];
        dst[0] = (int16_t)((t1 + t3) >> 3); dst[1] = (int16_t)((t2 - t4) >> 3);
        dst[2] = (int16_t)((t2 + t4) >> 3); dst[3] = (int16_t)((t1 - t3) >> 3);
        src += 8; dst += 8;
    }
    src = block;
    for (int i = 0; i < 4; i++) {
        t1 = 17 * (src[0] + src[16]) + 64; t2 = 17 * (src[0] - src[16]) + 64;
        t3 = 22 * src[8] + 10 * src[24];   t4 = 22 * src[24] - 10 * src[8];
        dest[0 * stride] = vc1_clip8(dest[0 * stride] + ((t1 + t3) >> 7));
        dest[1 * stride] = vc1_clip8(dest[1 * stride] + ((t2 - t4) >> 7));
        dest[2 * stride] = vc1_clip8(dest[2 * stride] + ((t2 + t4) >> 7));
        dest[3 * stride] = vc1_clip8(dest[3 * stride] + ((t1 - t3) >> 7));
        src++; dest++;
    }
}

// ── mspel quarter-pel luma interpolation (bicubic 4-tap) ────────────────────────
// mode: 0 none, 1 = 1/4, 2 = 1/2, 3 = 3/4 shift.
static inline int vc1_mspel_ver_filter_16bits(const uint8_t* src, int stride, int mode) {
    switch (mode) {
    case 1: return -4 * src[-stride] + 53 * src[0] + 18 * src[stride] -  3 * src[stride * 2];
    case 2: return -1 * src[-stride] +  9 * src[0] +  9 * src[stride] -  1 * src[stride * 2];
    case 3: return -3 * src[-stride] + 18 * src[0] + 53 * src[stride] -  4 * src[stride * 2];
    }
    return 0;
}
static inline int vc1_mspel_hor_filter_16bits(const int16_t* src, int stride, int mode) {
    switch (mode) {
    case 1: return -4 * src[-stride] + 53 * src[0] + 18 * src[stride] -  3 * src[stride * 2];
    case 2: return -1 * src[-stride] +  9 * src[0] +  9 * src[stride] -  1 * src[stride * 2];
    case 3: return -3 * src[-stride] + 18 * src[0] + 53 * src[stride] -  4 * src[stride * 2];
    }
    return 0;
}
static inline int vc1_mspel_filter(const uint8_t* src, int stride, int mode, int r) {
    switch (mode) {
    case 0: return src[0];
    case 1: return (-4 * src[-stride] + 53 * src[0] + 18 * src[stride] - 3 * src[stride * 2] + 32 - r) >> 6;
    case 2: return (-1 * src[-stride] +  9 * src[0] +  9 * src[stride] - 1 * src[stride * 2] +  8 - r) >> 4;
    case 3: return (-3 * src[-stride] + 18 * src[0] + 53 * src[stride] - 4 * src[stride * 2] + 32 - r) >> 6;
    }
    return 0;
}

// put_vc1_mspel_mc for an NxN block (N = 8 or 16). Separate src/dst strides (ffmpeg
// shares one because its emu-buffer matches linesize; we emulate into a tight buffer).
// hmode = mx&3, vmode = my&3, rnd = v->rnd.
static inline void vc1_put_mspel_mc(uint8_t* dst, int dstride, const uint8_t* src, int sstride,
                                    int hmode, int vmode, int rnd, int N) {
    static const int shift_value[] = { 0, 5, 1, 5 };
    int i, j;
    if (vmode) {
        int r;
        if (hmode) {                           // 2D: vertical to tmp, then horizontal
            int shift = (shift_value[hmode] + shift_value[vmode]) >> 1;
            int16_t tmp[19 * 16], *tptr = tmp;
            int W = N + 3;                      // 11 for N=8, 19 for N=16
            r = (1 << (shift - 1)) + rnd - 1;
            src -= 1;
            for (j = 0; j < N; j++) {
                for (i = 0; i < W; i++)
                    tptr[i] = (int16_t)((vc1_mspel_ver_filter_16bits(src + i, sstride, vmode) + r) >> shift);
                src += sstride; tptr += W;
            }
            r = 64 - rnd; tptr = tmp + 1;
            for (j = 0; j < N; j++) {
                for (i = 0; i < N; i++)
                    dst[i] = vc1_clip8((vc1_mspel_hor_filter_16bits(tptr + i, 1, hmode) + r) >> 7);
                dst += dstride; tptr += W;
            }
            return;
        } else {                               // vertical only
            r = 1 - rnd;
            for (j = 0; j < N; j++) {
                for (i = 0; i < N; i++) dst[i] = vc1_clip8(vc1_mspel_filter(src + i, sstride, vmode, r));
                src += sstride; dst += dstride;
            }
            return;
        }
    }
    // horizontal only (or straight copy when hmode==0)
    for (j = 0; j < N; j++) {
        for (i = 0; i < N; i++) dst[i] = vc1_clip8(vc1_mspel_filter(src + i, 1, hmode, rnd));
        dst += dstride; src += sstride;
    }
}

// Half-pel bilinear luma MC (mspel==0, i.e. the 1MV_HPEL_BILIN mode). dxy =
// (my&2)|((mx&2)>>1); rb = 1 - v->rnd (rounding bias). Src is a clamped (N+1)²
// footprint. Matches ffmpeg hpeldsp put_pixels / put_no_rnd_pixels.
static inline void vc1_put_hpel(uint8_t* dst, int dstride, const uint8_t* src, int sstride,
                                int N, int dxy, int rb) {
    for (int y = 0; y < N; y++) {
        const uint8_t* r0 = src + y * sstride;
        const uint8_t* r1 = src + (y + 1) * sstride;
        for (int x = 0; x < N; x++) {
            int a = r0[x], v;
            if      (dxy == 0) v = a;
            else if (dxy == 1) v = (a + r0[x + 1] + rb) >> 1;
            else if (dxy == 2) v = (a + r1[x] + rb) >> 1;
            else               v = (a + r0[x + 1] + r1[x] + r1[x + 1] + rb + 1) >> 2;
            dst[y * dstride + x] = (uint8_t)v;
        }
    }
}

// ── chroma bilinear MC (8xH). rnd = v->rnd: bias 32 (h264) vs 32-4 (vc1 no_rnd) ──
static inline void vc1_put_chroma_mc8(uint8_t* dst, int dstride, const uint8_t* src, int sstride,
                                      int h, int x, int y, int rnd) {
    const int A = (8 - x) * (8 - y), B = x * (8 - y), C = (8 - x) * y, D = x * y;
    const int bias = 32 - 4 * rnd;
    for (int i = 0; i < h; i++) {
        for (int a = 0; a < 8; a++)
            dst[a] = (uint8_t)((A * src[a] + B * src[a + 1] +
                                C * src[sstride + a] + D * src[sstride + a + 1] + bias) >> 6);
        dst += dstride; src += sstride;
    }
}

// ── emulated_edge_mc (videodsp): copy a bw x bh block from src at (sx,sy) into buf,
// replicating the [0,w)x[0,h) edge for out-of-range coordinates. We ALWAYS route MC
// through this (identical pixels to ffmpeg's draw_edges + direct read, and simpler).
static inline void vc1_emulated_edge(uint8_t* buf, int buf_stride,
                                     const uint8_t* src, int src_stride,
                                     int bw, int bh, int sx, int sy, int w, int h) {
    for (int y = 0; y < bh; y++) {
        int cy = sy + y; cy = cy < 0 ? 0 : (cy >= h ? h - 1 : cy);
        const uint8_t* s = src + cy * src_stride;
        uint8_t* d = buf + y * buf_stride;
        for (int x = 0; x < bw; x++) {
            int cx = sx + x; cx = cx < 0 ? 0 : (cx >= w ? w - 1 : cx);
            d[x] = s[cx];
        }
    }
}

} // namespace vc1

#endif // VC1_MC_H
