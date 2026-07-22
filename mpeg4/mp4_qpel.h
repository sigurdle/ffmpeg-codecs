/*
    MPEG-4 Part 2 ASP quarter-pel (qpel) luma motion compensation.

    PROVENANCE / LICENSE
    --------------------
    From-source TRANSCRIPTION of FFmpeg release/8.1's quarter-pel DSP
    (libavcodec/qpeldsp.c — put_/put_no_rnd_ mpeg4_qpel{8,16}_{h,v}_lowpass +
    the 16 mc<H><V>_c position functions — together with the pixels_l2 averaging
    from pel_template.c / hpel_template.c and rnd_avg.h). FFmpeg is LGPL-2.1+;
    this is an LGPL-derived work in the project's OWN mpeg4 wasm binary. See
    mpeg4_decode.cpp header + memory: divx-mjpeg-port-plan, codec-authorship-
    provenance.

    EQUIVALENCE NOTE (verified algebraically against ffmpeg's explicit unrolled
    formulas at interior and both edge columns, all 16 positions):
    ffmpeg's qpel{8,16}_mc<H><V>_c functions decompose EXACTLY into a separable
    two-stage filter, where H = hf = (mx & 3), V = vf = (my & 3):

      horizontal stage -> u8 buffer Hbuf (W cols x (W+1) rows):
        hf==0:  Hbuf[r][c] = src[r][c]                                  (integer cols)
        hf==2:  Hbuf[r][c] = clip((6tap_h(src[r],c) + RB) >> 5)         (half)
        hf==1:  Hbuf[r][c] = avg(clip(...), src[r][c])                  (quarter, low)
        hf==3:  Hbuf[r][c] = avg(clip(...), src[r][c+1])               (quarter, high)
      vertical stage -> dst (W x W):
        vf==0:  dst[r][c] = Hbuf[r][c]
        vf==2:  dst[r][c] = clip((6tap_v(Hbuf[:][c],r) + RB) >> 5)
        vf==1:  dst[r][c] = avg(Hbuf[r][c],   clip(...))
        vf==3:  dst[r][c] = avg(Hbuf[r+1][c], clip(...))

    The 6-tap lowpass (taps 20,-6,3,-1 / sum 32) is the half-position filter
        20*(s[i]+s[i+1]) - 6*(s[i-1]+s[i+2]) + 3*(s[i-2]+s[i+3]) - (s[i-3]+s[i+4])
    with the input edge handled by MIRRORING about the valid range [0,W]:
        s[-1]=s[0], s[-2]=s[1], s[-3]=s[2]; s[W+1]=s[W], s[W+2]=s[W-1], ...
    (this is exactly how ffmpeg's per-column unrolled formulas reuse src[0]/src[W]).

    RB = 16 (put / rounding) or 15 (put_no_rnd) — selected by RND. The pixels_l2
    average is round-up (put) or round-down (no_rnd) to match rnd_avg/no_rnd_avg.

    The reference block is pre-fetched clamped-to-frame-edge (== ffmpeg
    emulated_edge_mc) into a (W+1)x(W+1) buffer by the caller; the filter's
    mirroring above is the SEPARATE in-block edge handling. STANDARD qpel only
    (the FF_BUG_STD_QPEL "_old" variants apply solely to lavc_build<4653 streams).
    SCALAR; SIMD is a later perf pass.
*/
#ifndef MPEG4_MP4_QPEL_H
#define MPEG4_MP4_QPEL_H

#include <stdint.h>

namespace mp4 {

static inline uint8_t qpel_clip_u8(int v) {
    return v < 0 ? 0 : (v > 255 ? 255 : (uint8_t)v);
}

// Per-byte rounded average. RND: true = round up (a+b+1)>>1, false = (a+b)>>1.
template<bool RND>
static inline uint8_t qpel_avg2(int a, int b) {
    return (uint8_t)(RND ? ((a + b + 1) >> 1) : ((a + b) >> 1));
}

// Mirror an index into the valid input range [0,W] (filter edge handling).
static inline int qpel_mirror(int i, int W) {
    if (i < 0) return -i - 1;        // -1->0, -2->1, -3->2
    if (i > W) return 2 * W - i + 1; // W+1->W, W+2->W-1
    return i;
}

// 6-tap half-pel lowpass at the half-position between samples i and i+1, reading
// `base[k*stride]` for k mirrored into [0,W]. Returns the raw (un-rounded) sum.
static inline int qpel_tap(const uint8_t* base, int i, int stride, int W) {
    return ((int)base[qpel_mirror(i,     W) * stride] + (int)base[qpel_mirror(i + 1, W) * stride]) * 20
         - ((int)base[qpel_mirror(i - 1, W) * stride] + (int)base[qpel_mirror(i + 2, W) * stride]) * 6
         + ((int)base[qpel_mirror(i - 2, W) * stride] + (int)base[qpel_mirror(i + 3, W) * stride]) * 3
         - ((int)base[qpel_mirror(i - 3, W) * stride] + (int)base[qpel_mirror(i + 4, W) * stride]);
}

// Quarter-pel MC of a W x W luma block (W = 8 or 16) into dst.
// `full` = clamped (W+1) x (W+1) reference block (row stride fullStride).
// dxy = ((my & 3) << 2) | (mx & 3), the 16 subpel positions (0 = integer copy).
// avg != 0 -> rnd-average the result into dst (ffmpeg avg_qpel, for B bidir): the
// per-position PUT value is computed exactly as before, only the final store is
// dst = (dst + put + 1) >> 1 (intermediate stages stay PUT — matches ffmpeg).
template<bool RND>
static inline void qpel_mc_scalar(uint8_t* dst, int dstStride,
                           const uint8_t* full, int fullStride, int W, int dxy, int avg) {
    const int hf = dxy & 3, vf = (dxy >> 2) & 3;
    const int RB = RND ? 16 : 15;

    // Horizontal stage: Hbuf has W cols x (W+1) rows (rows 0..W), u8.
    uint8_t Hbuf[17 * 16];           // max W = 16
    const int HS = W;                // Hbuf row stride
    for (int r = 0; r <= W; r++) {
        const uint8_t* srow = full + (size_t)r * fullStride;
        uint8_t* hrow = Hbuf + (size_t)r * HS;
        if (hf == 0) {
            for (int c = 0; c < W; c++) hrow[c] = srow[c];
        } else if (hf == 2) {
            for (int c = 0; c < W; c++) hrow[c] = qpel_clip_u8((qpel_tap(srow, c, 1, W) + RB) >> 5);
        } else if (hf == 1) {
            for (int c = 0; c < W; c++) {
                int h = qpel_clip_u8((qpel_tap(srow, c, 1, W) + RB) >> 5);
                hrow[c] = qpel_avg2<RND>(h, srow[c]);
            }
        } else { // hf == 3
            for (int c = 0; c < W; c++) {
                int h = qpel_clip_u8((qpel_tap(srow, c, 1, W) + RB) >> 5);
                hrow[c] = qpel_avg2<RND>(h, srow[c + 1]);
            }
        }
    }

    // Vertical stage: dst W x W.
    for (int c = 0; c < W; c++) {
        const uint8_t* hcol = Hbuf + c;     // column c, stride HS
        for (int r = 0; r < W; r++) {
            uint8_t out;
            if (vf == 0) {
                out = Hbuf[(size_t)r * HS + c];
            } else {
                int v = qpel_clip_u8((qpel_tap(hcol, r, HS, W) + RB) >> 5);
                if (vf == 2)      out = (uint8_t)v;
                else if (vf == 1) out = qpel_avg2<RND>(Hbuf[(size_t)r * HS + c], v);
                else /* vf==3 */  out = qpel_avg2<RND>(Hbuf[(size_t)(r + 1) * HS + c], v);
            }
            uint8_t* o = dst + (size_t)r * dstStride + c;
            *o = avg ? (uint8_t)((*o + out + 1) >> 1) : out;   // B bidir: rnd-avg into dst
        }
    }
}

#ifdef __wasm_simd128__
#include <wasm_simd128.h>

// Per-lane rounded average over the low 8 u8 lanes. RND: avgr (ceil) / floor.
template<bool RND>
static inline v128_t qpel_v_avg2(v128_t a, v128_t b) {
    if (RND) return wasm_u8x16_avgr(a, b);
    v128_t carry = wasm_v128_and(wasm_v128_xor(a, b), wasm_i8x16_splat(1));
    return wasm_i8x16_sub(wasm_u8x16_avgr(a, b), carry);   // (a+b)>>1
}
// 6-tap over 8 u8 tap vectors t0..t7 (low-8 lanes) -> i16x8 raw sum:
//   20*(t3+t4) - 6*(t2+t5) + 3*(t1+t6) - (t0+t7).
static inline v128_t qpel_tap8(const uint8_t* p, int s0, int s1, int s2, int s3,
                               int s4, int s5, int s6, int s7) {
    v128_t t0 = wasm_u16x8_extend_low_u8x16(wasm_v128_load64_zero(p + s0));
    v128_t t1 = wasm_u16x8_extend_low_u8x16(wasm_v128_load64_zero(p + s1));
    v128_t t2 = wasm_u16x8_extend_low_u8x16(wasm_v128_load64_zero(p + s2));
    v128_t t3 = wasm_u16x8_extend_low_u8x16(wasm_v128_load64_zero(p + s3));
    v128_t t4 = wasm_u16x8_extend_low_u8x16(wasm_v128_load64_zero(p + s4));
    v128_t t5 = wasm_u16x8_extend_low_u8x16(wasm_v128_load64_zero(p + s5));
    v128_t t6 = wasm_u16x8_extend_low_u8x16(wasm_v128_load64_zero(p + s6));
    v128_t t7 = wasm_u16x8_extend_low_u8x16(wasm_v128_load64_zero(p + s7));
    v128_t acc = wasm_i16x8_mul(wasm_i16x8_add(t3, t4), wasm_i16x8_splat(20));
    acc = wasm_i16x8_sub(acc, wasm_i16x8_mul(wasm_i16x8_add(t2, t5), wasm_i16x8_splat(6)));
    acc = wasm_i16x8_add(acc, wasm_i16x8_mul(wasm_i16x8_add(t1, t6), wasm_i16x8_splat(3)));
    acc = wasm_i16x8_sub(acc, wasm_i16x8_add(t0, t7));
    return acc;
}
// (raw + RB) >> 5, saturating-clip to u8 (low 8 lanes hold the 8 results).
static inline v128_t qpel_clip8(v128_t raw, int RB) {
    raw = wasm_i16x8_shr(wasm_i16x8_add(raw, wasm_i16x8_splat((short)RB)), 5);
    return wasm_u8x16_narrow_i16x8(raw, raw);
}

// 16-wide variants for the common 1MV (W=16) path: one u8x16 load per tap (vs two
// load64s), 6-tap accumulated in both i16x8 halves. outLo/outHi = raw sums for
// cols 0..7 / 8..15. qpel_clip16 turns the two raw halves into one clipped u8x16.
static inline void qpel6_16(v128_t t0, v128_t t1, v128_t t2, v128_t t3,
                            v128_t t4, v128_t t5, v128_t t6, v128_t t7,
                            v128_t* outLo, v128_t* outHi) {
    const v128_t s20 = wasm_i16x8_splat(20), s6 = wasm_i16x8_splat(6), s3 = wasm_i16x8_splat(3);
    v128_t lo = wasm_i16x8_mul(wasm_i16x8_add(wasm_u16x8_extend_low_u8x16(t3), wasm_u16x8_extend_low_u8x16(t4)), s20);
    lo = wasm_i16x8_sub(lo, wasm_i16x8_mul(wasm_i16x8_add(wasm_u16x8_extend_low_u8x16(t2), wasm_u16x8_extend_low_u8x16(t5)), s6));
    lo = wasm_i16x8_add(lo, wasm_i16x8_mul(wasm_i16x8_add(wasm_u16x8_extend_low_u8x16(t1), wasm_u16x8_extend_low_u8x16(t6)), s3));
    lo = wasm_i16x8_sub(lo, wasm_i16x8_add(wasm_u16x8_extend_low_u8x16(t0), wasm_u16x8_extend_low_u8x16(t7)));
    v128_t hi = wasm_i16x8_mul(wasm_i16x8_add(wasm_u16x8_extend_high_u8x16(t3), wasm_u16x8_extend_high_u8x16(t4)), s20);
    hi = wasm_i16x8_sub(hi, wasm_i16x8_mul(wasm_i16x8_add(wasm_u16x8_extend_high_u8x16(t2), wasm_u16x8_extend_high_u8x16(t5)), s6));
    hi = wasm_i16x8_add(hi, wasm_i16x8_mul(wasm_i16x8_add(wasm_u16x8_extend_high_u8x16(t1), wasm_u16x8_extend_high_u8x16(t6)), s3));
    hi = wasm_i16x8_sub(hi, wasm_i16x8_add(wasm_u16x8_extend_high_u8x16(t0), wasm_u16x8_extend_high_u8x16(t7)));
    *outLo = lo; *outHi = hi;
}
static inline v128_t qpel_clip16(v128_t lo, v128_t hi, int RB) {
    const v128_t rb = wasm_i16x8_splat((short)RB);
    lo = wasm_i16x8_shr(wasm_i16x8_add(lo, rb), 5);
    hi = wasm_i16x8_shr(wasm_i16x8_add(hi, rb), 5);
    return wasm_u8x16_narrow_i16x8(lo, hi);
}

// Full-width (W=16) SIMD qpel — byte-identical to the 8-col-group path, one u8x16
// load/store per tap-row. Handles the common 1MV luma block.
template<bool RND>
static inline void qpel_mc_simd16(uint8_t* dst, int dstStride,
                                  const uint8_t* full, int fullStride, int dxy, int avg) {
    const int hf = dxy & 3, vf = (dxy >> 2) & 3, RB = RND ? 16 : 15, W = 16, HS = 16;
    uint8_t Hbuf[17 * 16];
    uint8_t ext[24];
    for (int r = 0; r <= W; r++) {
        const uint8_t* srow = full + (size_t)r * fullStride;
        uint8_t* hrow = Hbuf + (size_t)r * HS;
        if (hf == 0) { wasm_v128_store(hrow, wasm_v128_load(srow)); continue; }
        ext[0] = srow[2]; ext[1] = srow[1]; ext[2] = srow[0];
        for (int c = 0; c <= W; c++) ext[3 + c] = srow[c];
        ext[W + 4] = srow[W]; ext[W + 5] = srow[W - 1]; ext[W + 6] = srow[W - 2];
        v128_t lo, hi;
        qpel6_16(wasm_v128_load(ext + 0), wasm_v128_load(ext + 1), wasm_v128_load(ext + 2), wasm_v128_load(ext + 3),
                 wasm_v128_load(ext + 4), wasm_v128_load(ext + 5), wasm_v128_load(ext + 6), wasm_v128_load(ext + 7), &lo, &hi);
        v128_t h = qpel_clip16(lo, hi, RB);
        if (hf == 1)      h = qpel_v_avg2<RND>(h, wasm_v128_load(srow));
        else if (hf == 3) h = qpel_v_avg2<RND>(h, wasm_v128_load(srow + 1));
        wasm_v128_store(hrow, h);
    }
    for (int r = 0; r < W; r++) {
        uint8_t* orow = dst + (size_t)r * dstStride;
        if (vf == 0) {
            v128_t out = wasm_v128_load(Hbuf + (size_t)r * HS);
            if (avg) out = wasm_u8x16_avgr(wasm_v128_load(orow), out);
            wasm_v128_store(orow, out);
            continue;
        }
        const uint8_t* p[8];
        for (int k = 0; k < 8; k++) p[k] = Hbuf + (size_t)qpel_mirror(r - 3 + k, W) * HS;
        v128_t lo, hi;
        qpel6_16(wasm_v128_load(p[0]), wasm_v128_load(p[1]), wasm_v128_load(p[2]), wasm_v128_load(p[3]),
                 wasm_v128_load(p[4]), wasm_v128_load(p[5]), wasm_v128_load(p[6]), wasm_v128_load(p[7]), &lo, &hi);
        v128_t v = qpel_clip16(lo, hi, RB);
        v128_t out;
        if (vf == 2)      out = v;
        else if (vf == 1) out = qpel_v_avg2<RND>(wasm_v128_load(Hbuf + (size_t)r * HS), v);
        else              out = qpel_v_avg2<RND>(wasm_v128_load(Hbuf + (size_t)(r + 1) * HS), v);
        if (avg) out = wasm_u8x16_avgr(wasm_v128_load(orow), out);
        wasm_v128_store(orow, out);
    }
}

// SIMD quarter-pel MC — byte-identical to qpel_mc_scalar<RND>. Separable 6-tap:
// horizontal stage builds a u8 Hbuf (W cols x (W+1) rows) with block-edge mirroring
// pre-materialized into a small ext row, then the vertical stage convolves columns
// (each tap row is contiguous so loads are plain). W in {8,16}; processed in 8-col groups.
template<bool RND>
static inline void qpel_mc_simd(uint8_t* dst, int dstStride,
                                const uint8_t* full, int fullStride, int W, int dxy, int avg) {
    if (W == 16) { qpel_mc_simd16<RND>(dst, dstStride, full, fullStride, dxy, avg); return; }
    const int hf = dxy & 3, vf = (dxy >> 2) & 3, RB = RND ? 16 : 15;
    const int HS = W, NG = W >> 3;                 // groups of 8 columns
    uint8_t Hbuf[17 * 16];
    uint8_t ext[24];                               // mirrored source row (len W+7)

    // ── horizontal stage: rows 0..W ──
    for (int r = 0; r <= W; r++) {
        const uint8_t* srow = full + (size_t)r * fullStride;
        uint8_t* hrow = Hbuf + (size_t)r * HS;
        if (hf == 0) { for (int c = 0; c < W; c++) hrow[c] = srow[c]; continue; }
        ext[0] = srow[2]; ext[1] = srow[1]; ext[2] = srow[0];   // left mirror
        for (int c = 0; c <= W; c++) ext[3 + c] = srow[c];      // body [0,W]
        ext[W + 4] = srow[W]; ext[W + 5] = srow[W - 1]; ext[W + 6] = srow[W - 2];  // right mirror
        for (int g = 0; g < NG; g++) {
            const uint8_t* e = ext + g * 8;        // output cols g*8 read ext[col..col+7]
            v128_t raw = qpel_tap8(e, 0, 1, 2, 3, 4, 5, 6, 7);
            v128_t h = qpel_clip8(raw, RB);
            if (hf == 1)      h = qpel_v_avg2<RND>(h, wasm_v128_load64_zero(srow + g * 8));
            else if (hf == 3) h = qpel_v_avg2<RND>(h, wasm_v128_load64_zero(srow + g * 8 + 1));
            wasm_v128_store64_lane(hrow + g * 8, h, 0);
        }
    }

    // ── vertical stage: output rows 0..W-1 ──
    for (int r = 0; r < W; r++) {
        uint8_t* orow = dst + (size_t)r * dstStride;
        if (vf == 0) {
            for (int g = 0; g < NG; g++) {
                v128_t out = wasm_v128_load64_zero(Hbuf + (size_t)r * HS + g * 8);
                if (avg) out = wasm_u8x16_avgr(wasm_v128_load64_zero(orow + g * 8), out);
                wasm_v128_store64_lane(orow + g * 8, out, 0);
            }
            continue;
        }
        // mirrored row pointers for the 6 vertical taps (taps r-3..r+4 about [0,W]).
        const uint8_t* p[8];
        for (int k = 0; k < 8; k++) p[k] = Hbuf + (size_t)qpel_mirror(r - 3 + k, W) * HS;
        for (int g = 0; g < NG; g++) {
            int off = g * 8;
            v128_t t0 = wasm_u16x8_extend_low_u8x16(wasm_v128_load64_zero(p[0] + off));
            v128_t t1 = wasm_u16x8_extend_low_u8x16(wasm_v128_load64_zero(p[1] + off));
            v128_t t2 = wasm_u16x8_extend_low_u8x16(wasm_v128_load64_zero(p[2] + off));
            v128_t t3 = wasm_u16x8_extend_low_u8x16(wasm_v128_load64_zero(p[3] + off));
            v128_t t4 = wasm_u16x8_extend_low_u8x16(wasm_v128_load64_zero(p[4] + off));
            v128_t t5 = wasm_u16x8_extend_low_u8x16(wasm_v128_load64_zero(p[5] + off));
            v128_t t6 = wasm_u16x8_extend_low_u8x16(wasm_v128_load64_zero(p[6] + off));
            v128_t t7 = wasm_u16x8_extend_low_u8x16(wasm_v128_load64_zero(p[7] + off));
            v128_t acc = wasm_i16x8_mul(wasm_i16x8_add(t3, t4), wasm_i16x8_splat(20));
            acc = wasm_i16x8_sub(acc, wasm_i16x8_mul(wasm_i16x8_add(t2, t5), wasm_i16x8_splat(6)));
            acc = wasm_i16x8_add(acc, wasm_i16x8_mul(wasm_i16x8_add(t1, t6), wasm_i16x8_splat(3)));
            acc = wasm_i16x8_sub(acc, wasm_i16x8_add(t0, t7));
            v128_t v = qpel_clip8(acc, RB);
            v128_t out;
            if (vf == 2)      out = v;
            else if (vf == 1) out = qpel_v_avg2<RND>(wasm_v128_load64_zero(Hbuf + (size_t)r * HS + off), v);
            else              out = qpel_v_avg2<RND>(wasm_v128_load64_zero(Hbuf + (size_t)(r + 1) * HS + off), v);
            if (avg) out = wasm_u8x16_avgr(wasm_v128_load64_zero(orow + off), out);
            wasm_v128_store64_lane(orow + off, out, 0);
        }
    }
}
#endif // __wasm_simd128__

// Runtime-rounding dispatch wrapper (avg != 0 -> rnd-average into dst, for B bidir).
// Uses the SIMD kernel where available (byte-identical to the scalar reference).
static inline void qpel_mc_rt(uint8_t* dst, int dstStride, const uint8_t* full,
                              int fullStride, int W, int dxy, bool rnd, int avg) {
#ifdef __wasm_simd128__
    if (rnd) qpel_mc_simd<true>(dst, dstStride, full, fullStride, W, dxy, avg);
    else     qpel_mc_simd<false>(dst, dstStride, full, fullStride, W, dxy, avg);
#else
    if (rnd) qpel_mc_scalar<true>(dst, dstStride, full, fullStride, W, dxy, avg);
    else     qpel_mc_scalar<false>(dst, dstStride, full, fullStride, W, dxy, avg);
#endif
}

// Self-test entry: run scalar or SIMD qpel for cross-checking (M3 SIMD validation).
static inline void qpel_mc_test(uint8_t* dst, int dstStride, const uint8_t* full,
                                int fullStride, int W, int dxy, bool rnd, int avg, int useSimd) {
#ifdef __wasm_simd128__
    if (useSimd) { if (rnd) qpel_mc_simd<true>(dst, dstStride, full, fullStride, W, dxy, avg);
                   else     qpel_mc_simd<false>(dst, dstStride, full, fullStride, W, dxy, avg); return; }
#else
    (void)useSimd;
#endif
    if (rnd) qpel_mc_scalar<true>(dst, dstStride, full, fullStride, W, dxy, avg);
    else     qpel_mc_scalar<false>(dst, dstStride, full, fullStride, W, dxy, avg);
}

// Fetch a bw x bh block from `ref` (w x h, row stride refStride) at (sx,sy),
// clamping every coordinate to the frame edge (== ffmpeg emulated_edge_mc).
static inline void qpel_fetch_clamped(uint8_t* dst, int dstStride,
                                      const uint8_t* ref, int refStride, int w, int h,
                                      int sx, int sy, int bw, int bh) {
    for (int r = 0; r < bh; r++) {
        int yy = sy + r; yy = yy < 0 ? 0 : (yy >= h ? h - 1 : yy);
        const uint8_t* rrow = ref + (size_t)yy * refStride;
        uint8_t* drow = dst + (size_t)r * dstStride;
        for (int c = 0; c < bw; c++) {
            int xx = sx + c; xx = xx < 0 ? 0 : (xx >= w ? w - 1 : xx);
            drow[c] = rrow[xx];
        }
    }
}

// Qpel luma MC with on-demand edge prefetch: when the (W+1)x(W+1) source region is
// fully inside the frame (the common case), filter directly from `ref` and skip the
// per-pixel-clamped copy into `scratch` (qpel_fetch_clamped is the identity there, so
// this is byte-identical). Edge blocks still prefetch into scratch then filter.
static inline void mp4_qpel_luma(uint8_t* dst, int dstStride, const uint8_t* ref, int refStride,
                                 int w, int h, int sx, int sy, int W, int dxy, int rnd, int avg,
                                 uint8_t* scratch, int scratchStride) {
    if (sx >= 0 && sy >= 0 && sx + W < w && sy + W < h) {          // cols/rows sx..sx+W in range
        qpel_mc_rt(dst, dstStride, ref + (size_t)sy * refStride + sx, refStride, W, dxy, rnd, avg);
    } else {
        qpel_fetch_clamped(scratch, scratchStride, ref, refStride, w, h, sx, sy, W + 1, W + 1);
        qpel_mc_rt(dst, dstStride, scratch, scratchStride, W, dxy, rnd, avg);
    }
}

} // namespace mp4

#endif // MPEG4_MP4_QPEL_H
