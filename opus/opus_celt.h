/*
	The CELT layer of the Opus decoder: FFmpeg n8.1 libavcodec/opus/celt.h, celt.c, dec_celt.c, pvq.h, pvq.c, dsp.c (LGPL-2.1+) and
	three vector functions of libavutil/float_dsp.c (LGPL-2.1+), the same code in the same order. opus_mdct.h holds the inverse
	transform. x86 assembly is not ported: the C paths of postfilter, deemphasis and float_dsp are the only ones.

	Copyright (c) 2007-2008 CSIRO
	Copyright (c) 2007-2009 Xiph.Org Foundation
	Copyright (c) 2008-2009 Gregory Maxwell
	Copyright (c) 2012 Andrew D'Addesio
	Copyright (c) 2013-2014 Mozilla Corporation
	Copyright (c) 2016-2017 Rostislav Pehlivanov

	ff_celt_decode_frame keeps FFmpeg's shape: start and end band, coded and output channels. The hybrid path calls it with
	start_band 17, as dec.c does.

	Departures from FFmpeg:
	  - The encoder is gone: its fields, its calls, the pulse search, and the encode and quant switches of ff_celt_bitalloc and
	    quant_band_template (now quant_band). CeltPVQ's function pointers are direct calls.
	  - CeltFrame is a member of the stream, not allocated; ff_celt_init does not fail, ff_celt_free and the logging are gone.
	  - CeltBlockSize and CeltSpread are held as int, because C++ does not assign an int to an enum.
*/

#pragma once

#define CELT_SHORT_BLOCKSIZE         120
#define CELT_OVERLAP                 CELT_SHORT_BLOCKSIZE
#define CELT_MAX_LOG_BLOCKS          3
#define CELT_MAX_FRAME_SIZE          (CELT_SHORT_BLOCKSIZE * (1 << CELT_MAX_LOG_BLOCKS))
#define CELT_MAX_BANDS               21

#define CELT_VECTORS                 11
#define CELT_ALLOC_STEPS             6
#define CELT_FINE_OFFSET             21
#define CELT_MAX_FINE_BITS           8
#define CELT_NORM_SCALE              16384
#define CELT_QTHETA_OFFSET           4
#define CELT_QTHETA_OFFSET_TWOPHASE  16
#define CELT_POSTFILTER_MINPERIOD    15
#define CELT_ENERGY_SILENCE          (-28.0f)

enum CeltSpread {
    CELT_SPREAD_NONE,
    CELT_SPREAD_LIGHT,
    CELT_SPREAD_NORMAL,
    CELT_SPREAD_AGGRESSIVE
};

enum CeltBlockSize {
    CELT_BLOCK_120,
    CELT_BLOCK_240,
    CELT_BLOCK_480,
    CELT_BLOCK_960,

    CELT_BLOCK_NB
};

typedef struct CeltBlock {
    float energy[CELT_MAX_BANDS];
    float prev_energy[2][CELT_MAX_BANDS];

    uint8_t collapse_masks[CELT_MAX_BANDS];

    /* buffer for mdct output + postfilter */
    float buf[2048];
    float coeffs[CELT_MAX_FRAME_SIZE];

    /* postfilter parameters */
    int   pf_period_new;
    float pf_gains_new[3];
    int   pf_period;
    float pf_gains[3];
    int   pf_period_old;
    float pf_gains_old[3];

    float emph_coeff;
} CeltBlock;

typedef struct CeltPVQ {
    int   qcoeff[256];
    float hadamard_tmp[256];
} CeltPVQ;

typedef struct CeltFrame {
    // constant values that do not change during context lifetime
    CeltImdct           tx[4];
    CeltBlock           block[2];
    CeltPVQ             pvq;
    int channels;
    int output_channels;
    int apply_phase_inv;

    int size;
    int start_band;
    int end_band;
    int coded_bands;
    int transient;
    int alloc_trim;
    int blocks;        /* number of iMDCT blocks in the frame, depends on transient */
    int blocksize;     /* size of each block */
    int silence;       /* Frame is filled with silence */
    int anticollapse_needed; /* Whether to expect an anticollapse bit */
    int anticollapse;  /* Encoded anticollapse bit */
    int intensity_stereo;
    int dual_stereo;
    int flushed;
    uint32_t seed;
    int spread;

    /* Bit allocation */
    int framebits;
    int remaining;
    int remaining2;
    int caps         [CELT_MAX_BANDS];
    int fine_bits    [CELT_MAX_BANDS];
    int fine_priority[CELT_MAX_BANDS];
    int pulses       [CELT_MAX_BANDS];
    int tf_change    [CELT_MAX_BANDS];
} CeltFrame;

/* LCG for noise generation */
static inline uint32_t celt_rng(CeltFrame *f)
{
    f->seed = 1664525 * f->seed + 1013904223;
    return f->seed;
}

static inline void celt_renormalize_vector(float *X, int N, float gain)
{
    int i;
    float g = 1e-15f;
    for (i = 0; i < N; i++)
        g += X[i] * X[i];
    g = gain / sqrtf(g);

    for (i = 0; i < N; i++)
        X[i] *= g;
}

#define MUL16(ra, rb) ((ra) * (rb))
#define FFMIN3(a, b, c) FFMIN(FFMIN(a, b), c)
#define FFALIGN(x, a) (((x) + (a) - 1) & ~((a) - 1))
#define FF_ARRAY_ELEMS(a) (sizeof(a) / sizeof((a)[0]))

static inline unsigned av_zero_extend(unsigned a, unsigned p) { return a & ((1U << p) - 1); }

#define ROUND_MUL16(a,b)  ((MUL16(a, b) + 16384) >> 15)

#define CELT_PVQ_U(n, k) (ff_celt_pvq_u_row[FFMIN(n, k)][FFMAX(n, k)])
#define CELT_PVQ_V(n, k) (CELT_PVQ_U(n, k) + CELT_PVQ_U(n, (k) + 1))

static inline int16_t celt_cos(int16_t x)
{
    x = (MUL16(x, x) + 4096) >> 13;
    x = (32767-x) + ROUND_MUL16(x, (-7651 + ROUND_MUL16(x, (8277 + ROUND_MUL16(-626, x)))));
    return x + 1;
}

static inline int celt_log2tan(int isin, int icos)
{
    int lc, ls;
    lc = opus_ilog(icos);
    ls = opus_ilog(isin);
    icos <<= 15 - lc;
    isin <<= 15 - ls;
    return (ls << 11) - (lc << 11) +
           ROUND_MUL16(isin, ROUND_MUL16(isin, -2597) + 7932) -
           ROUND_MUL16(icos, ROUND_MUL16(icos, -2597) + 7932);
}

static inline int celt_bits2pulses(const uint8_t *cache, int bits)
{
    // TODO: Find the size of cache and make it into an array in the parameters list
    int i, low = 0, high;

    high = cache[0];
    bits--;

    for (i = 0; i < 6; i++) {
        int center = (low + high + 1) >> 1;
        if (cache[center] >= bits)
            high = center;
        else
            low = center;
    }

    return (bits - (low == 0 ? -1 : cache[low]) <= cache[high] - bits) ? low : high;
}

static inline int celt_pulses2bits(const uint8_t *cache, int pulses)
{
    // TODO: Find the size of cache and make it into an array in the parameters list
   return (pulses == 0) ? 0 : cache[pulses] + 1;
}

static inline void celt_normalize_residual(const int * __restrict iy, float * __restrict X,
                                           int N, float g)
{
    int i;
    for (i = 0; i < N; i++)
        X[i] = g * iy[i];
}

static void celt_exp_rotation_impl(float *X, uint32_t len, uint32_t stride,
                                   float c, float s)
{
    float *Xptr;
    int i;

    Xptr = X;
    for (i = 0; i < len - stride; i++) {
        float x1     = Xptr[0];
        float x2     = Xptr[stride];
        Xptr[stride] = c * x2 + s * x1;
        *Xptr++      = c * x1 - s * x2;
    }

    Xptr = &X[len - 2 * stride - 1];
    for (i = len - 2 * stride - 1; i >= 0; i--) {
        float x1     = Xptr[0];
        float x2     = Xptr[stride];
        Xptr[stride] = c * x2 + s * x1;
        *Xptr--      = c * x1 - s * x2;
    }
}

static inline void celt_exp_rotation(float *X, uint32_t len,
                                     uint32_t stride, uint32_t K,
                                     int spread)
{
    uint32_t stride2 = 0;
    float c, s;
    float gain, theta;
    int i;

    if (2*K >= len || spread == CELT_SPREAD_NONE)
        return;

    gain = (float)len / (len + (20 - 5*spread) * K);
    theta = M_PI * gain * gain / 4;

    c = cosf(theta);
    s = sinf(theta);

    if (len >= stride << 3) {
        stride2 = 1;
        /* This is just a simple (equivalent) way of computing sqrt(len/stride) with rounding.
        It's basically incrementing long as (stride2+0.5)^2 < len/stride. */
        while ((stride2 * stride2 + stride2) * stride + (stride >> 2) < len)
            stride2++;
    }

    len /= stride;
    for (i = 0; i < stride; i++) {
        if (stride2)
            celt_exp_rotation_impl(X + i * len, len, stride2, s, c);
        celt_exp_rotation_impl(X + i * len, len, 1, c, s);
    }
}

static inline uint32_t celt_extract_collapse_mask(const int *iy, uint32_t N, uint32_t B)
{
    int i, j, N0 = N / B;
    uint32_t collapse_mask = 0;

    if (B <= 1)
        return 1;

    for (i = 0; i < B; i++)
        for (j = 0; j < N0; j++)
            collapse_mask |= (!!iy[i*N0+j]) << i;
    return collapse_mask;
}

static inline void celt_stereo_merge(float *X, float *Y, float mid, int N)
{
    int i;
    float xp = 0, side = 0;
    float E[2];
    float mid2;
    float gain[2];

    /* Compute the norm of X+Y and X-Y as |X|^2 + |Y|^2 +/- sum(xy) */
    for (i = 0; i < N; i++) {
        xp   += X[i] * Y[i];
        side += Y[i] * Y[i];
    }

    /* Compensating for the mid normalization */
    xp *= mid;
    mid2 = mid;
    E[0] = mid2 * mid2 + side - 2 * xp;
    E[1] = mid2 * mid2 + side + 2 * xp;
    if (E[0] < 6e-4f || E[1] < 6e-4f) {
        for (i = 0; i < N; i++)
            Y[i] = X[i];
        return;
    }

    gain[0] = 1.0f / sqrtf(E[0]);
    gain[1] = 1.0f / sqrtf(E[1]);

    for (i = 0; i < N; i++) {
        float value[2];
        /* Apply mid scaling (side is already scaled) */
        value[0] = mid * X[i];
        value[1] = Y[i];
        X[i] = gain[0] * (value[0] - value[1]);
        Y[i] = gain[1] * (value[0] + value[1]);
    }
}

static void celt_interleave_hadamard(float *tmp, float *X, int N0,
                                     int stride, int hadamard)
{
    int i, j, N = N0*stride;
    const uint8_t *order = &ff_celt_hadamard_order[hadamard ? stride - 2 : 30];

    for (i = 0; i < stride; i++)
        for (j = 0; j < N0; j++)
            tmp[j*stride+i] = X[order[i]*N0+j];

    memcpy(X, tmp, N*sizeof(float));
}

static void celt_deinterleave_hadamard(float *tmp, float *X, int N0,
                                       int stride, int hadamard)
{
    int i, j, N = N0*stride;
    const uint8_t *order = &ff_celt_hadamard_order[hadamard ? stride - 2 : 30];

    for (i = 0; i < stride; i++)
        for (j = 0; j < N0; j++)
            tmp[order[i]*N0+j] = X[j*stride+i];

    memcpy(X, tmp, N*sizeof(float));
}

static void celt_haar1(float *X, int N0, int stride)
{
    int i, j;
    N0 >>= 1;
    for (i = 0; i < stride; i++) {
        for (j = 0; j < N0; j++) {
            float x0 = X[stride * (2 * j + 0) + i];
            float x1 = X[stride * (2 * j + 1) + i];
            X[stride * (2 * j + 0) + i] = (x0 + x1) * M_SQRT1_2;
            X[stride * (2 * j + 1) + i] = (x0 - x1) * M_SQRT1_2;
        }
    }
}

static inline int celt_compute_qn(int N, int b, int offset, int pulse_cap,
                                  int stereo)
{
    int qn, qb;
    int N2 = 2 * N - 1;
    if (stereo && N == 2)
        N2--;

    /* The upper limit ensures that in a stereo split with itheta===16384, we'll
     * always have enough bits left over to code at least one pulse in the
     * side; otherwise it would collapse, since it doesn't get folded. */
    qb = FFMIN3(b - pulse_cap - (4 << 3), (b + N2 * offset) / N2, 8 << 3);
    qn = (qb < (1 << 3 >> 1)) ? 1 : ((ff_celt_qn_exp2[qb & 0x7] >> (14 - (qb >> 3))) + 1) >> 1 << 1;
    return qn;
}

// this code was adapted from libopus
static inline uint64_t celt_cwrsi(uint32_t N, uint32_t K, uint32_t i, int *y)
{
    uint64_t norm = 0;
    uint32_t q, p;
    int s, val;
    int k0;

    while (N > 2) {
        /*Lots of pulses case:*/
        if (K >= N) {
            const uint32_t *row = ff_celt_pvq_u_row[N];

            /* Are the pulses in this dimension negative? */
            p  = row[K + 1];
            s  = -(i >= p);
            i -= p & s;

            /*Count how many pulses were placed in this dimension.*/
            k0 = K;
            q = row[N];
            if (q > i) {
                K = N;
                do {
                    p = ff_celt_pvq_u_row[--K][N];
                } while (p > i);
            } else
                for (p = row[K]; p > i; p = row[K])
                    K--;

            i    -= p;
            val   = (k0 - K + s) ^ s;
            norm += val * val;
            *y++  = val;
        } else { /*Lots of dimensions case:*/
            /*Are there any pulses in this dimension at all?*/
            p = ff_celt_pvq_u_row[K    ][N];
            q = ff_celt_pvq_u_row[K + 1][N];

            if (p <= i && i < q) {
                i -= p;
                *y++ = 0;
            } else {
                /*Are the pulses in this dimension negative?*/
                s  = -(i >= q);
                i -= q & s;

                /*Count how many pulses were placed in this dimension.*/
                k0 = K;
                do p = ff_celt_pvq_u_row[--K][N];
                while (p > i);

                i    -= p;
                val   = (k0 - K + s) ^ s;
                norm += val * val;
                *y++  = val;
            }
        }
        N--;
    }

    /* N === 2 */
    p  = 2 * K + 1;
    s  = -(i >= p);
    i -= p & s;
    k0 = K;
    K  = (i + 1) / 2;

    if (K)
        i -= 2 * K - 1;

    val   = (k0 - K + s) ^ s;
    norm += val * val;
    *y++  = val;

    /* N===1 */
    s     = -i;
    val   = (K + s) ^ s;
    norm += val * val;
    *y    = val;

    return norm;
}

static inline float celt_decode_pulses(OpusRangeCoder *rc, int *y, uint32_t N, uint32_t K)
{
    const uint32_t idx = ff_opus_rc_dec_uint(rc, CELT_PVQ_V(N, K));
    return celt_cwrsi(N, K, idx, y);
}

/** Decode pulse vector and combine the result with the pitch vector to produce
    the final normalized signal in the current band. */
static uint32_t celt_alg_unquant(OpusRangeCoder *rc, float *X, uint32_t N, uint32_t K,
                                 int spread, uint32_t blocks, float gain,
                                 CeltPVQ *pvq)
{
    int *y = pvq->qcoeff;

    gain /= sqrtf(celt_decode_pulses(rc, y, N, K));
    celt_normalize_residual(y, X, N, gain);
    celt_exp_rotation(X, N, blocks, K, spread);
    return celt_extract_collapse_mask(y, N, blocks);
}

static uint32_t quant_band(CeltPVQ *pvq, CeltFrame *f, OpusRangeCoder *rc, const int band, float *X,
                           float *Y, int N, int b, uint32_t blocks, float *lowband, int duration,
                           float *lowband_out, int level, float gain, float *lowband_scratch, int fill)
{
    int i;
    const uint8_t *cache;
    int stereo = !!Y, split = stereo;
    int imid = 0, iside = 0;
    uint32_t N0 = N;
    int N_B = N / blocks;
    int N_B0 = N_B;
    int B0 = blocks;
    int time_divide = 0;
    int recombine = 0;
    int inv = 0;
    float mid = 0, side = 0;
    int longblocks = (B0 == 1);
    uint32_t cm = 0;

    if (N == 1) {
        float *x = X;
        for (i = 0; i <= stereo; i++) {
            int sign = 0;
            if (f->remaining2 >= 1 << 3) {
                sign = ff_opus_rc_get_raw(rc, 1);
                f->remaining2 -= 1 << 3;
            }
            x[0] = 1.0f - 2.0f*sign;
            x = Y;
        }
        if (lowband_out)
            lowband_out[0] = X[0];
        return 1;
    }

    if (!stereo && level == 0) {
        int tf_change = f->tf_change[band];
        int k;
        if (tf_change > 0)
            recombine = tf_change;
        /* Band recombining to increase frequency resolution */

        if (lowband &&
            (recombine || ((N_B & 1) == 0 && tf_change < 0) || B0 > 1)) {
            for (i = 0; i < N; i++)
                lowband_scratch[i] = lowband[i];
            lowband = lowband_scratch;
        }

        for (k = 0; k < recombine; k++) {
            if (lowband)
                celt_haar1(lowband, N >> k, 1 << k);
            fill = ff_celt_bit_interleave[fill & 0xF] | ff_celt_bit_interleave[fill >> 4] << 2;
        }
        blocks >>= recombine;
        N_B <<= recombine;

        /* Increasing the time resolution */
        while ((N_B & 1) == 0 && tf_change < 0) {
            if (lowband)
                celt_haar1(lowband, N_B, blocks);
            fill |= fill << blocks;
            blocks <<= 1;
            N_B >>= 1;
            time_divide++;
            tf_change++;
        }
        B0 = blocks;
        N_B0 = N_B;

        /* Reorganize the samples in time order instead of frequency order */
        if (B0 > 1 && lowband)
            celt_deinterleave_hadamard(pvq->hadamard_tmp, lowband,
                                       N_B >> recombine, B0 << recombine,
                                       longblocks);
    }

    /* If we need 1.5 more bit than we can produce, split the band in two. */
    cache = ff_celt_cache_bits +
            ff_celt_cache_index[(duration + 1) * CELT_MAX_BANDS + band];
    if (!stereo && duration >= 0 && b > cache[cache[0]] + 12 && N > 2) {
        N >>= 1;
        Y = X + N;
        split = 1;
        duration -= 1;
        if (blocks == 1)
            fill = (fill & 1) | (fill << 1);
        blocks = (blocks + 1) >> 1;
    }

    if (split) {
        int qn;
        int itheta = 0;
        int mbits, sbits, delta;
        int qalloc;
        int pulse_cap;
        int offset;
        int orig_fill;
        int tell;

        /* Decide on the resolution to give to the split parameter theta */
        pulse_cap = ff_celt_log_freq_range[band] + duration * 8;
        offset = (pulse_cap >> 1) - (stereo && N == 2 ? CELT_QTHETA_OFFSET_TWOPHASE :
                                                          CELT_QTHETA_OFFSET);
        qn = (stereo && band >= f->intensity_stereo) ? 1 :
             celt_compute_qn(N, b, offset, pulse_cap, stereo);
        tell = opus_rc_tell_frac(rc);
        if (qn != 1) {
            /* Entropy coding of the angle. We use a uniform pdf for the
             * time split, a step for stereo, and a triangular one for the rest. */
            if (stereo && N > 2)
                itheta = ff_opus_rc_dec_uint_step(rc, qn / 2);
            else if (stereo || B0 > 1)
                itheta = ff_opus_rc_dec_uint(rc, qn+1);
            else
                itheta = ff_opus_rc_dec_uint_tri(rc, qn);
            itheta = itheta * 16384 / qn;
        } else if (stereo) {
            inv = (b > 2 << 3 && f->remaining2 > 2 << 3) ? ff_opus_rc_dec_log(rc, 2) : 0;
            inv = f->apply_phase_inv ? inv : 0;
            itheta = 0;
        }
        qalloc = opus_rc_tell_frac(rc) - tell;
        b -= qalloc;

        orig_fill = fill;
        if (itheta == 0) {
            imid = 32767;
            iside = 0;
            fill = av_zero_extend(fill, blocks);
            delta = -16384;
        } else if (itheta == 16384) {
            imid = 0;
            iside = 32767;
            fill &= ((1 << blocks) - 1) << blocks;
            delta = 16384;
        } else {
            imid = celt_cos(itheta);
            iside = celt_cos(16384-itheta);
            /* This is the mid vs side allocation that minimizes squared error
            in that band. */
            delta = ROUND_MUL16((N - 1) << 7, celt_log2tan(iside, imid));
        }

        mid  = imid  / 32768.0f;
        side = iside / 32768.0f;

        /* This is a special case for N=2 that only works for stereo and takes
        advantage of the fact that mid and side are orthogonal to encode
        the side with just one bit. */
        if (N == 2 && stereo) {
            int c;
            int sign = 0;
            float tmp;
            float *x2, *y2;
            mbits = b;
            /* Only need one bit for the side */
            sbits = (itheta != 0 && itheta != 16384) ? 1 << 3 : 0;
            mbits -= sbits;
            c = (itheta > 8192);
            f->remaining2 -= qalloc+sbits;

            x2 = c ? Y : X;
            y2 = c ? X : Y;
            if (sbits) {
                sign = ff_opus_rc_get_raw(rc, 1);
            }
            sign = 1 - 2 * sign;
            /* We use orig_fill here because we want to fold the side, but if
            itheta===16384, we'll have cleared the low bits of fill. */
            cm = quant_band(pvq, f, rc, band, x2, NULL, N, mbits, blocks, lowband, duration,
                                 lowband_out, level, gain, lowband_scratch, orig_fill);
            /* We don't split N=2 bands, so cm is either 1 or 0 (for a fold-collapse),
            and there's no need to worry about mixing with the other channel. */
            y2[0] = -sign * x2[1];
            y2[1] =  sign * x2[0];
            X[0] *= mid;
            X[1] *= mid;
            Y[0] *= side;
            Y[1] *= side;
            tmp = X[0];
            X[0] = tmp - Y[0];
            Y[0] = tmp + Y[0];
            tmp = X[1];
            X[1] = tmp - Y[1];
            Y[1] = tmp + Y[1];
        } else {
            /* "Normal" split code */
            float *next_lowband2     = NULL;
            float *next_lowband_out1 = NULL;
            int next_level = 0;
            int rebalance;
            uint32_t cmt;

            /* Give more bits to low-energy MDCTs than they would
             * otherwise deserve */
            if (B0 > 1 && !stereo && (itheta & 0x3fff)) {
                if (itheta > 8192)
                    /* Rough approximation for pre-echo masking */
                    delta -= delta >> (4 - duration);
                else
                    /* Corresponds to a forward-masking slope of
                     * 1.5 dB per 10 ms */
                    delta = FFMIN(0, delta + (N << 3 >> (5 - duration)));
            }
            mbits = av_clip((b - delta) / 2, 0, b);
            sbits = b - mbits;
            f->remaining2 -= qalloc;

            if (lowband && !stereo)
                next_lowband2 = lowband + N; /* >32-bit split case */

            /* Only stereo needs to pass on lowband_out.
             * Otherwise, it's handled at the end */
            if (stereo)
                next_lowband_out1 = lowband_out;
            else
                next_level = level + 1;

            rebalance = f->remaining2;
            if (mbits >= sbits) {
                /* In stereo mode, we do not apply a scaling to the mid
                 * because we need the normalized mid for folding later */
                cm = quant_band(pvq, f, rc, band, X, NULL, N, mbits, blocks,
                                     lowband, duration, next_lowband_out1, next_level,
                                     stereo ? 1.0f : (gain * mid), lowband_scratch, fill);
                rebalance = mbits - (rebalance - f->remaining2);
                if (rebalance > 3 << 3 && itheta != 0)
                    sbits += rebalance - (3 << 3);

                /* For a stereo split, the high bits of fill are always zero,
                 * so no folding will be done to the side. */
                cmt = quant_band(pvq, f, rc, band, Y, NULL, N, sbits, blocks,
                                      next_lowband2, duration, NULL, next_level,
                                      gain * side, NULL, fill >> blocks);
                cm |= cmt << ((B0 >> 1) & (stereo - 1));
            } else {
                /* For a stereo split, the high bits of fill are always zero,
                 * so no folding will be done to the side. */
                cm = quant_band(pvq, f, rc, band, Y, NULL, N, sbits, blocks,
                                     next_lowband2, duration, NULL, next_level,
                                     gain * side, NULL, fill >> blocks);
                cm <<= ((B0 >> 1) & (stereo - 1));
                rebalance = sbits - (rebalance - f->remaining2);
                if (rebalance > 3 << 3 && itheta != 16384)
                    mbits += rebalance - (3 << 3);

                /* In stereo mode, we do not apply a scaling to the mid because
                 * we need the normalized mid for folding later */
                cm |= quant_band(pvq, f, rc, band, X, NULL, N, mbits, blocks,
                                      lowband, duration, next_lowband_out1, next_level,
                                      stereo ? 1.0f : (gain * mid), lowband_scratch, fill);
            }
        }
    } else {
        /* This is the basic no-split case */
        uint32_t q         = celt_bits2pulses(cache, b);
        uint32_t curr_bits = celt_pulses2bits(cache, q);
        f->remaining2 -= curr_bits;

        /* Ensures we can never bust the budget */
        while (f->remaining2 < 0 && q > 0) {
            f->remaining2 += curr_bits;
            curr_bits      = celt_pulses2bits(cache, --q);
            f->remaining2 -= curr_bits;
        }

        if (q != 0) {
            /* Finally do the actual (de)quantization */
            cm = celt_alg_unquant(rc, X, N, (q < 8) ? q : (8 + (q & 7)) << ((q >> 3) - 1),
                                  f->spread, blocks, gain, pvq);
        } else {
            /* If there's no pulse, fill the band anyway */
            uint32_t cm_mask = (1 << blocks) - 1;
            fill &= cm_mask;
            if (fill) {
                if (!lowband) {
                    /* Noise */
                    for (i = 0; i < N; i++)
                        X[i] = (((int32_t)celt_rng(f)) >> 20);
                    cm = cm_mask;
                } else {
                    /* Folded spectrum */
                    for (i = 0; i < N; i++) {
                        /* About 48 dB below the "normal" folding level */
                        X[i] = lowband[i] + (((celt_rng(f)) & 0x8000) ? 1.0f / 256 : -1.0f / 256);
                    }
                    cm = fill;
                }
                celt_renormalize_vector(X, N, gain);
            } else {
                memset(X, 0, N*sizeof(float));
            }
        }
    }

    if (stereo) {
        if (N > 2)
            celt_stereo_merge(X, Y, mid, N);
        if (inv) {
            for (i = 0; i < N; i++)
                Y[i] *= -1;
        }
    } else if (level == 0) {
        int k;

        /* Undo the sample reorganization going from time order to frequency order */
        if (B0 > 1)
            celt_interleave_hadamard(pvq->hadamard_tmp, X, N_B >> recombine,
                                     B0 << recombine, longblocks);

        /* Undo time-freq changes that we did earlier */
        N_B = N_B0;
        blocks = B0;
        for (k = 0; k < time_divide; k++) {
            blocks >>= 1;
            N_B <<= 1;
            cm |= cm >> blocks;
            celt_haar1(X, N_B, blocks);
        }

        for (k = 0; k < recombine; k++) {
            cm = ff_celt_bit_deinterleave[cm];
            celt_haar1(X, N0>>k, 1<<k);
        }
        blocks <<= recombine;

        /* Scale output for later folding */
        if (lowband_out) {
            float n = sqrtf(N0);
            for (i = 0; i < N0; i++)
                lowband_out[i] = n * X[i];
        }
        cm = av_zero_extend(cm, blocks);
    }

    return cm;
}

static void ff_celt_quant_bands(CeltFrame *f, OpusRangeCoder *rc)
{
    float lowband_scratch[8 * 22];
    float norm1[2 * 8 * 100];
    float *norm2 = norm1 + 8 * 100;

    int totalbits = (f->framebits << 3) - f->anticollapse_needed;

    int update_lowband = 1;
    int lowband_offset = 0;

    int i, j;

    for (i = f->start_band; i < f->end_band; i++) {
        uint32_t cm[2] = { (1u << f->blocks) - 1, (1u << f->blocks) - 1 };
        int band_offset = ff_celt_freq_bands[i] << f->size;
        int band_size   = ff_celt_freq_range[i] << f->size;
        float *X = f->block[0].coeffs + band_offset;
        float *Y = (f->channels == 2) ? f->block[1].coeffs + band_offset : NULL;
        float *norm_loc1, *norm_loc2;

        int consumed = opus_rc_tell_frac(rc);
        int effective_lowband = -1;
        int b = 0;

        /* Compute how many bits we want to allocate to this band */
        if (i != f->start_band)
            f->remaining -= consumed;
        f->remaining2 = totalbits - consumed - 1;
        if (i <= f->coded_bands - 1) {
            int curr_balance = f->remaining / FFMIN(3, f->coded_bands-i);
            b = av_clip_uintp2(FFMIN(f->remaining2 + 1, f->pulses[i] + curr_balance), 14);
        }

        if ((ff_celt_freq_bands[i] - ff_celt_freq_range[i] >= ff_celt_freq_bands[f->start_band] ||
            i == f->start_band + 1) && (update_lowband || lowband_offset == 0))
            lowband_offset = i;

        if (i == f->start_band + 1) {
            /* Special Hybrid Folding (RFC 8251 section 9). Copy the first band into
            the second to ensure the second band never has to use the LCG. */
            int count = (ff_celt_freq_range[i] - ff_celt_freq_range[i-1]) << f->size;

            memcpy(&norm1[band_offset], &norm1[band_offset - count], count * sizeof(float));

            if (f->channels == 2)
                memcpy(&norm2[band_offset], &norm2[band_offset - count], count * sizeof(float));
        }

        /* Get a conservative estimate of the collapse_mask's for the bands we're
           going to be folding from. */
        if (lowband_offset != 0 && (f->spread != CELT_SPREAD_AGGRESSIVE ||
                                    f->blocks > 1 || f->tf_change[i] < 0)) {
            int foldstart, foldend;

            /* This ensures we never repeat spectral content within one band */
            effective_lowband = FFMAX(ff_celt_freq_bands[f->start_band],
                                      ff_celt_freq_bands[lowband_offset] - ff_celt_freq_range[i]);
            foldstart = lowband_offset;
            while (ff_celt_freq_bands[--foldstart] > effective_lowband);
            foldend = lowband_offset - 1;
            while (++foldend < i && ff_celt_freq_bands[foldend] < effective_lowband + ff_celt_freq_range[i]);

            cm[0] = cm[1] = 0;
            for (j = foldstart; j < foldend; j++) {
                cm[0] |= f->block[0].collapse_masks[j];
                cm[1] |= f->block[f->channels - 1].collapse_masks[j];
            }
        }

        if (f->dual_stereo && i == f->intensity_stereo) {
            /* Switch off dual stereo to do intensity */
            f->dual_stereo = 0;
            for (j = ff_celt_freq_bands[f->start_band] << f->size; j < band_offset; j++)
                norm1[j] = (norm1[j] + norm2[j]) / 2;
        }

        norm_loc1 = effective_lowband != -1 ? norm1 + (effective_lowband << f->size) : NULL;
        norm_loc2 = effective_lowband != -1 ? norm2 + (effective_lowband << f->size) : NULL;

        if (f->dual_stereo) {
            cm[0] = quant_band(&f->pvq, f, rc, i, X, NULL, band_size, b >> 1,
                                       f->blocks, norm_loc1, f->size,
                                       norm1 + band_offset, 0, 1.0f,
                                       lowband_scratch, cm[0]);

            cm[1] = quant_band(&f->pvq, f, rc, i, Y, NULL, band_size, b >> 1,
                                       f->blocks, norm_loc2, f->size,
                                       norm2 + band_offset, 0, 1.0f,
                                       lowband_scratch, cm[1]);
        } else {
            cm[0] = quant_band(&f->pvq, f, rc, i, X,    Y, band_size, b >> 0,
                                       f->blocks, norm_loc1, f->size,
                                       norm1 + band_offset, 0, 1.0f,
                                       lowband_scratch, cm[0] | cm[1]);
            cm[1] = cm[0];
        }

        f->block[0].collapse_masks[i]               = (uint8_t)cm[0];
        f->block[f->channels - 1].collapse_masks[i] = (uint8_t)cm[1];
        f->remaining += f->pulses[i] + consumed;

        /* Update the folding position only as long as we have 1 bit/sample depth */
        update_lowband = (b > band_size << 3);
    }
}

#define NORMC(bits) ((bits) << (f->channels - 1) << f->size >> 2)

static void ff_celt_bitalloc(CeltFrame *f, OpusRangeCoder *rc)
{
    int i, j, low, high, total, done, bandbits, remaining, tbits_8ths;
    int skip_startband      = f->start_band;
    int skip_bit            = 0;
    int intensitystereo_bit = 0;
    int dualstereo_bit      = 0;
    int dynalloc            = 6;
    int extrabits           = 0;

    int boost[CELT_MAX_BANDS] = { 0 };
    int trim_offset[CELT_MAX_BANDS];
    int threshold[CELT_MAX_BANDS];
    int bits1[CELT_MAX_BANDS];
    int bits2[CELT_MAX_BANDS];

    /* Spread */
    if (opus_rc_tell(rc) + 4 <= f->framebits) {
        f->spread = ff_opus_rc_dec_cdf(rc, ff_celt_model_spread);
    } else {
        f->spread = CELT_SPREAD_NORMAL;
    }

    /* Initialize static allocation caps */
    for (i = 0; i < CELT_MAX_BANDS; i++)
        f->caps[i] = NORMC((ff_celt_static_caps[f->size][f->channels - 1][i] + 64) * ff_celt_freq_range[i]);

    /* Band boosts */
    tbits_8ths = f->framebits << 3;
    for (i = f->start_band; i < f->end_band; i++) {
        int quanta = ff_celt_freq_range[i] << (f->channels - 1) << f->size;
        int b_dynalloc = dynalloc;
        quanta = FFMIN(quanta << 3, FFMAX(6 << 3, quanta));

        while (opus_rc_tell_frac(rc) + (b_dynalloc << 3) < tbits_8ths && boost[i] < f->caps[i]) {
            int is_boost = ff_opus_rc_dec_log(rc, b_dynalloc);

            if (!is_boost)
                break;

            boost[i]   += quanta;
            tbits_8ths -= quanta;

            b_dynalloc = 1;
        }

        if (boost[i])
            dynalloc = FFMAX(dynalloc - 1, 2);
    }

    /* Allocation trim */
    f->alloc_trim = 5;
    if (opus_rc_tell_frac(rc) + (6 << 3) <= tbits_8ths)
        f->alloc_trim = ff_opus_rc_dec_cdf(rc, ff_celt_model_alloc_trim);

    /* Anti-collapse bit reservation */
    tbits_8ths = (f->framebits << 3) - opus_rc_tell_frac(rc) - 1;
    f->anticollapse_needed = 0;
    if (f->transient && f->size >= 2 && tbits_8ths >= ((f->size + 2) << 3))
        f->anticollapse_needed = 1 << 3;
    tbits_8ths -= f->anticollapse_needed;

    /* Band skip bit reservation */
    if (tbits_8ths >= 1 << 3)
        skip_bit = 1 << 3;
    tbits_8ths -= skip_bit;

    /* Intensity/dual stereo bit reservation */
    if (f->channels == 2) {
        intensitystereo_bit = ff_celt_log2_frac[f->end_band - f->start_band];
        if (intensitystereo_bit <= tbits_8ths) {
            tbits_8ths -= intensitystereo_bit;
            if (tbits_8ths >= 1 << 3) {
                dualstereo_bit = 1 << 3;
                tbits_8ths -= 1 << 3;
            }
        } else {
            intensitystereo_bit = 0;
        }
    }

    /* Trim offsets */
    for (i = f->start_band; i < f->end_band; i++) {
        int trim     = f->alloc_trim - 5 - f->size;
        int band     = ff_celt_freq_range[i] * (f->end_band - i - 1);
        int duration = f->size + 3;
        int scale    = duration + f->channels - 1;

        /* PVQ minimum allocation threshold, below this value the band is
         * skipped */
        threshold[i] = FFMAX(3 * ff_celt_freq_range[i] << duration >> 4,
                             f->channels << 3);

        trim_offset[i] = trim * (band << scale) >> 6;

        if (ff_celt_freq_range[i] << f->size == 1)
            trim_offset[i] -= f->channels << 3;
    }

    /* Bisection */
    low  = 1;
    high = CELT_VECTORS - 1;
    while (low <= high) {
        int center = (low + high) >> 1;
        done = total = 0;

        for (i = f->end_band - 1; i >= f->start_band; i--) {
            bandbits = NORMC(ff_celt_freq_range[i] * ff_celt_static_alloc[center][i]);

            if (bandbits)
                bandbits = FFMAX(bandbits + trim_offset[i], 0);
            bandbits += boost[i];

            if (bandbits >= threshold[i] || done) {
                done = 1;
                total += FFMIN(bandbits, f->caps[i]);
            } else if (bandbits >= f->channels << 3) {
                total += f->channels << 3;
            }
        }

        if (total > tbits_8ths)
            high = center - 1;
        else
            low = center + 1;
    }
    high = low--;

    /* Bisection */
    for (i = f->start_band; i < f->end_band; i++) {
        bits1[i] = NORMC(ff_celt_freq_range[i] * ff_celt_static_alloc[low][i]);
        bits2[i] = high >= CELT_VECTORS ? f->caps[i] :
                   NORMC(ff_celt_freq_range[i] * ff_celt_static_alloc[high][i]);

        if (bits1[i])
            bits1[i] = FFMAX(bits1[i] + trim_offset[i], 0);
        if (bits2[i])
            bits2[i] = FFMAX(bits2[i] + trim_offset[i], 0);

        if (low)
            bits1[i] += boost[i];
        bits2[i] += boost[i];

        if (boost[i])
            skip_startband = i;
        bits2[i] = FFMAX(bits2[i] - bits1[i], 0);
    }

    /* Bisection */
    low  = 0;
    high = 1 << CELT_ALLOC_STEPS;
    for (i = 0; i < CELT_ALLOC_STEPS; i++) {
        int center = (low + high) >> 1;
        done = total = 0;

        for (j = f->end_band - 1; j >= f->start_band; j--) {
            bandbits = bits1[j] + (center * bits2[j] >> CELT_ALLOC_STEPS);

            if (bandbits >= threshold[j] || done) {
                done = 1;
                total += FFMIN(bandbits, f->caps[j]);
            } else if (bandbits >= f->channels << 3)
                total += f->channels << 3;
        }
        if (total > tbits_8ths)
            high = center;
        else
            low = center;
    }

    /* Bisection */
    done = total = 0;
    for (i = f->end_band - 1; i >= f->start_band; i--) {
        bandbits = bits1[i] + (low * bits2[i] >> CELT_ALLOC_STEPS);

        if (bandbits >= threshold[i] || done)
            done = 1;
        else
            bandbits = (bandbits >= f->channels << 3) ?
            f->channels << 3 : 0;

        bandbits     = FFMIN(bandbits, f->caps[i]);
        f->pulses[i] = bandbits;
        total      += bandbits;
    }

    /* Band skipping */
    for (f->coded_bands = f->end_band; ; f->coded_bands--) {
        int allocation;
        j = f->coded_bands - 1;

        if (j == skip_startband) {
            /* all remaining bands are not skipped */
            tbits_8ths += skip_bit;
            break;
        }

        /* determine the number of bits available for coding "do not skip" markers */
        remaining   = tbits_8ths - total;
        bandbits    = remaining / (ff_celt_freq_bands[j+1] - ff_celt_freq_bands[f->start_band]);
        remaining  -= bandbits  * (ff_celt_freq_bands[j+1] - ff_celt_freq_bands[f->start_band]);
        allocation  = f->pulses[j] + bandbits * ff_celt_freq_range[j];
        allocation += FFMAX(remaining - (ff_celt_freq_bands[j] - ff_celt_freq_bands[f->start_band]), 0);

        /* a "do not skip" marker is only coded if the allocation is
         * above the chosen threshold */
        if (allocation >= FFMAX(threshold[j], (f->channels + 1) << 3)) {
            int do_not_skip = ff_opus_rc_dec_log(rc, 1);

            if (do_not_skip)
                break;

            total      += 1 << 3;
            allocation -= 1 << 3;
        }

        /* the band is skipped, so reclaim its bits */
        total -= f->pulses[j];
        if (intensitystereo_bit) {
            total -= intensitystereo_bit;
            intensitystereo_bit = ff_celt_log2_frac[j - f->start_band];
            total += intensitystereo_bit;
        }

        total += f->pulses[j] = (allocation >= f->channels << 3) ? f->channels << 3 : 0;
    }

    /* IS start band */
    f->intensity_stereo = f->dual_stereo = 0;
    if (intensitystereo_bit)
        f->intensity_stereo = f->start_band + ff_opus_rc_dec_uint(rc, f->coded_bands + 1 - f->start_band);

    /* DS flag */
    if (f->intensity_stereo <= f->start_band)
        tbits_8ths += dualstereo_bit; /* no intensity stereo means no dual stereo */
    else if (dualstereo_bit)
        f->dual_stereo = ff_opus_rc_dec_log(rc, 1);

    /* Supply the remaining bits in this frame to lower bands */
    remaining = tbits_8ths - total;
    bandbits  = remaining / (ff_celt_freq_bands[f->coded_bands] - ff_celt_freq_bands[f->start_band]);
    remaining -= bandbits * (ff_celt_freq_bands[f->coded_bands] - ff_celt_freq_bands[f->start_band]);
    for (i = f->start_band; i < f->coded_bands; i++) {
        const int bits = FFMIN(remaining, ff_celt_freq_range[i]);
        f->pulses[i] += bits + bandbits * ff_celt_freq_range[i];
        remaining    -= bits;
    }

    /* Finally determine the allocation */
    for (i = f->start_band; i < f->coded_bands; i++) {
        int N = ff_celt_freq_range[i] << f->size;
        int prev_extra = extrabits;
        f->pulses[i] += extrabits;

        if (N > 1) {
            int dof;        /* degrees of freedom */
            int temp;       /* dof * channels * log(dof) */
            int fine_bits;
            int max_bits;
            int offset;     /* fine energy quantization offset, i.e.
                             * extra bits assigned over the standard
                             * totalbits/dof */

            extrabits = FFMAX(f->pulses[i] - f->caps[i], 0);
            f->pulses[i] -= extrabits;

            /* intensity stereo makes use of an extra degree of freedom */
            dof = N * f->channels + (f->channels == 2 && N > 2 && !f->dual_stereo && i < f->intensity_stereo);
            temp = dof * (ff_celt_log_freq_range[i] + (f->size << 3));
            offset = (temp >> 1) - dof * CELT_FINE_OFFSET;
            if (N == 2) /* dof=2 is the only case that doesn't fit the model */
                offset += dof << 1;

            /* grant an additional bias for the first and second pulses */
            if (f->pulses[i] + offset < 2 * (dof << 3))
                offset += temp >> 2;
            else if (f->pulses[i] + offset < 3 * (dof << 3))
                offset += temp >> 3;

            fine_bits = (f->pulses[i] + offset + (dof << 2)) / (dof << 3);
            max_bits  = FFMIN((f->pulses[i] >> 3) >> (f->channels - 1), CELT_MAX_FINE_BITS);
            max_bits  = FFMAX(max_bits, 0);
            f->fine_bits[i] = av_clip(fine_bits, 0, max_bits);

            /* If fine_bits was rounded down or capped,
             * give priority for the final fine energy pass */
            f->fine_priority[i] = (f->fine_bits[i] * (dof << 3) >= f->pulses[i] + offset);

            /* the remaining bits are assigned to PVQ */
            f->pulses[i] -= f->fine_bits[i] << (f->channels - 1) << 3;
        } else {
            /* all bits go to fine energy except for the sign bit */
            extrabits = FFMAX(f->pulses[i] - (f->channels << 3), 0);
            f->pulses[i] -= extrabits;
            f->fine_bits[i] = 0;
            f->fine_priority[i] = 1;
        }

        /* hand back a limited number of extra fine energy bits to this band */
        if (extrabits > 0) {
            int fineextra = FFMIN(extrabits >> (f->channels + 2),
                                  CELT_MAX_FINE_BITS - f->fine_bits[i]);
            f->fine_bits[i] += fineextra;

            fineextra <<= f->channels + 2;
            f->fine_priority[i] = (fineextra >= extrabits - prev_extra);
            extrabits -= fineextra;
        }
    }
    f->remaining = extrabits;

    /* skipped bands dedicate all of their bits for fine energy */
    for (; i < f->end_band; i++) {
        f->fine_bits[i]     = f->pulses[i] >> (f->channels - 1) >> 3;
        f->pulses[i]        = 0;
        f->fine_priority[i] = f->fine_bits[i] < 1;
    }
}

static void postfilter_c(float *data, int period, float *gains, int len)
{
    const float g0 = gains[0];
    const float g1 = gains[1];
    const float g2 = gains[2];

    float x4 = data[-period - 2];
    float x3 = data[-period - 1];
    float x2 = data[-period + 0];
    float x1 = data[-period + 1];

    for (int i = 0; i < len; i++) {
        float x0 = data[i - period + 2];
        data[i] += g0 * x2        +
                   g1 * (x1 + x3) +
                   g2 * (x0 + x4);
        x4 = x3;
        x3 = x2;
        x2 = x1;
        x1 = x0;
    }
}

static float deemphasis_c(float *y, float *x, float coeff, const float *weights, int len)
{
    const float c = weights[0];
    for (int i = 0; i < len; i++)
        coeff = y[i] = x[i] + coeff*c;

    return coeff;
}

static void vector_fmac_scalar(float *dst, const float *src, float mul, int len)
{
    for (int i = 0; i < len; i++)
        dst[i] += src[i] * mul;
}

static void vector_fmul_scalar(float *dst, const float *src, float mul, int len)
{
    for (int i = 0; i < len; i++)
        dst[i] = src[i] * mul;
}

static void vector_fmul_window(float *dst, const float *src0,
                               const float *src1, const float *win, int len)
{
    int i, j;

    dst  += len;
    win  += len;
    src0 += len;

    for (i = -len, j = len - 1; i < 0; i++, j--) {
        float s0 = src0[i];
        float s1 = src1[j];
        float wi = win[i];
        float wj = win[j];
        dst[i] = s0 * wj - s1 * wi;
        dst[j] = s0 * wi + s1 * wj;
    }
}

/* Use the 2D z-transform to apply prediction in both the time domain (alpha)
 * and the frequency domain (beta) */
static void celt_decode_coarse_energy(CeltFrame *f, OpusRangeCoder *rc)
{
    int i, j;
    float prev[2] = { 0 };
    float alpha = ff_celt_alpha_coef[f->size];
    float beta  = ff_celt_beta_coef[f->size];
    const uint8_t *model = ff_celt_coarse_energy_dist[f->size][0];

    /* intra frame */
    if (opus_rc_tell(rc) + 3 <= f->framebits && ff_opus_rc_dec_log(rc, 3)) {
        alpha = 0.0f;
        beta  = 1.0f - (4915.0f/32768.0f);
        model = ff_celt_coarse_energy_dist[f->size][1];
    }

    for (i = 0; i < CELT_MAX_BANDS; i++) {
        for (j = 0; j < f->channels; j++) {
            CeltBlock *block = &f->block[j];
            float value;
            int available;

            if (i < f->start_band || i >= f->end_band) {
                block->energy[i] = 0.0;
                continue;
            }

            available = f->framebits - opus_rc_tell(rc);
            if (available >= 15) {
                /* decode using a Laplace distribution */
                int k = FFMIN(i, 20) << 1;
                value = ff_opus_rc_dec_laplace(rc, model[k] << 7, model[k+1] << 6);
            } else if (available >= 2) {
                int x = ff_opus_rc_dec_cdf(rc, ff_celt_model_energy_small);
                value = (x>>1) ^ -(x&1);
            } else if (available >= 1) {
                value = -(float)ff_opus_rc_dec_log(rc, 1);
            } else value = -1;

            block->energy[i] = FFMAX(-9.0f, block->energy[i]) * alpha + prev[j] + value;
            prev[j] += beta * value;
        }
    }
}

static void celt_decode_fine_energy(CeltFrame *f, OpusRangeCoder *rc)
{
    int i;
    for (i = f->start_band; i < f->end_band; i++) {
        int j;
        if (!f->fine_bits[i])
            continue;

        for (j = 0; j < f->channels; j++) {
            CeltBlock *block = &f->block[j];
            int q2;
            float offset;
            q2 = ff_opus_rc_get_raw(rc, f->fine_bits[i]);
            offset = (q2 + 0.5f) * (1 << (14 - f->fine_bits[i])) / 16384.0f - 0.5f;
            block->energy[i] += offset;
        }
    }
}

static void celt_decode_final_energy(CeltFrame *f, OpusRangeCoder *rc)
{
    int priority, i, j;
    int bits_left = f->framebits - opus_rc_tell(rc);

    for (priority = 0; priority < 2; priority++) {
        for (i = f->start_band; i < f->end_band && bits_left >= f->channels; i++) {
            if (f->fine_priority[i] != priority || f->fine_bits[i] >= CELT_MAX_FINE_BITS)
                continue;

            for (j = 0; j < f->channels; j++) {
                int q2;
                float offset;
                q2 = ff_opus_rc_get_raw(rc, 1);
                offset = (q2 - 0.5f) * (1 << (14 - f->fine_bits[i] - 1)) / 16384.0f;
                f->block[j].energy[i] += offset;
                bits_left--;
            }
        }
    }
}

static void celt_decode_tf_changes(CeltFrame *f, OpusRangeCoder *rc)
{
    int i, diff = 0, tf_select = 0, tf_changed = 0, tf_select_bit;
    int consumed, bits = f->transient ? 2 : 4;

    consumed = opus_rc_tell(rc);
    tf_select_bit = (f->size != 0 && consumed+bits+1 <= f->framebits);

    for (i = f->start_band; i < f->end_band; i++) {
        if (consumed+bits+tf_select_bit <= f->framebits) {
            diff ^= ff_opus_rc_dec_log(rc, bits);
            consumed = opus_rc_tell(rc);
            tf_changed |= diff;
        }
        f->tf_change[i] = diff;
        bits = f->transient ? 4 : 5;
    }

    if (tf_select_bit && ff_celt_tf_select[f->size][f->transient][0][tf_changed] !=
                         ff_celt_tf_select[f->size][f->transient][1][tf_changed])
        tf_select = ff_opus_rc_dec_log(rc, 1);

    for (i = f->start_band; i < f->end_band; i++) {
        f->tf_change[i] = ff_celt_tf_select[f->size][f->transient][tf_select][f->tf_change[i]];
    }
}

static void celt_denormalize(CeltFrame *f, CeltBlock *block, float *data)
{
    int i, j;

    for (i = f->start_band; i < f->end_band; i++) {
        float *dst = data + (ff_celt_freq_bands[i] << f->size);
        float log_norm = block->energy[i] + ff_celt_mean_energy[i];
        float norm = exp2f(FFMIN(log_norm, 32.0f));

        for (j = 0; j < ff_celt_freq_range[i] << f->size; j++)
            dst[j] *= norm;
    }
}

static void celt_postfilter_apply_transition(CeltBlock *block, float *data)
{
    const int T0 = block->pf_period_old;
    const int T1 = block->pf_period;

    float g00, g01, g02;
    float g10, g11, g12;

    float x0, x1, x2, x3, x4;

    int i;

    if (block->pf_gains[0]     == 0.0 &&
        block->pf_gains_old[0] == 0.0)
        return;

    g00 = block->pf_gains_old[0];
    g01 = block->pf_gains_old[1];
    g02 = block->pf_gains_old[2];
    g10 = block->pf_gains[0];
    g11 = block->pf_gains[1];
    g12 = block->pf_gains[2];

    x1 = data[-T1 + 1];
    x2 = data[-T1];
    x3 = data[-T1 - 1];
    x4 = data[-T1 - 2];

    for (i = 0; i < CELT_OVERLAP; i++) {
        float w = ff_celt_window2[i];
        x0 = data[i - T1 + 2];

        data[i] +=  (1.0 - w) * g00 * data[i - T0]                          +
                    (1.0 - w) * g01 * (data[i - T0 - 1] + data[i - T0 + 1]) +
                    (1.0 - w) * g02 * (data[i - T0 - 2] + data[i - T0 + 2]) +
                    w         * g10 * x2                                    +
                    w         * g11 * (x1 + x3)                             +
                    w         * g12 * (x0 + x4);
        x4 = x3;
        x3 = x2;
        x2 = x1;
        x1 = x0;
    }
}

static void celt_postfilter(CeltFrame *f, CeltBlock *block)
{
    int len = f->blocksize * f->blocks;
    const int filter_len = len - 2 * CELT_OVERLAP;

    celt_postfilter_apply_transition(block, block->buf + 1024);

    block->pf_period_old = block->pf_period;
    memcpy(block->pf_gains_old, block->pf_gains, sizeof(block->pf_gains));

    block->pf_period = block->pf_period_new;
    memcpy(block->pf_gains, block->pf_gains_new, sizeof(block->pf_gains));

    if (len > CELT_OVERLAP) {
        celt_postfilter_apply_transition(block, block->buf + 1024 + CELT_OVERLAP);

        if (block->pf_gains[0] > FLT_EPSILON && filter_len > 0)
            postfilter_c(block->buf + 1024 + 2 * CELT_OVERLAP,
                         block->pf_period, block->pf_gains,
                         filter_len);

        block->pf_period_old = block->pf_period;
        memcpy(block->pf_gains_old, block->pf_gains, sizeof(block->pf_gains));
    }

    memmove(block->buf, block->buf + len, (1024 + CELT_OVERLAP / 2) * sizeof(float));
}

static int parse_postfilter(CeltFrame *f, OpusRangeCoder *rc, int consumed)
{
    int i;

    memset(f->block[0].pf_gains_new, 0, sizeof(f->block[0].pf_gains_new));
    memset(f->block[1].pf_gains_new, 0, sizeof(f->block[1].pf_gains_new));

    if (f->start_band == 0 && consumed + 16 <= f->framebits) {
        int has_postfilter = ff_opus_rc_dec_log(rc, 1);
        if (has_postfilter) {
            float gain;
            int tapset, octave, period;

            octave = ff_opus_rc_dec_uint(rc, 6);
            period = (16 << octave) + ff_opus_rc_get_raw(rc, 4 + octave) - 1;
            gain   = 0.09375f * (ff_opus_rc_get_raw(rc, 3) + 1);
            tapset = (opus_rc_tell(rc) + 2 <= f->framebits) ?
                     ff_opus_rc_dec_cdf(rc, ff_celt_model_tapset) : 0;

            for (i = 0; i < 2; i++) {
                CeltBlock *block = &f->block[i];

                block->pf_period_new = FFMAX(period, CELT_POSTFILTER_MINPERIOD);
                block->pf_gains_new[0] = gain * ff_celt_postfilter_taps[tapset][0];
                block->pf_gains_new[1] = gain * ff_celt_postfilter_taps[tapset][1];
                block->pf_gains_new[2] = gain * ff_celt_postfilter_taps[tapset][2];
            }
        }

        consumed = opus_rc_tell(rc);
    }

    return consumed;
}

static void process_anticollapse(CeltFrame *f, CeltBlock *block, float *X)
{
    int i, j, k;

    for (i = f->start_band; i < f->end_band; i++) {
        int renormalize = 0;
        float *xptr;
        float prev[2];
        float Ediff, r;
        float thresh, sqrt_1;
        int depth;

        /* depth in 1/8 bits */
        depth = (1 + f->pulses[i]) / (ff_celt_freq_range[i] << f->size);
        thresh = exp2f(-1.0 - 0.125f * depth);
        sqrt_1 = 1.0f / sqrtf(ff_celt_freq_range[i] << f->size);

        xptr = X + (ff_celt_freq_bands[i] << f->size);

        prev[0] = block->prev_energy[0][i];
        prev[1] = block->prev_energy[1][i];
        if (f->channels == 1) {
            CeltBlock *block1 = &f->block[1];

            prev[0] = FFMAX(prev[0], block1->prev_energy[0][i]);
            prev[1] = FFMAX(prev[1], block1->prev_energy[1][i]);
        }
        Ediff = block->energy[i] - FFMIN(prev[0], prev[1]);
        Ediff = FFMAX(0, Ediff);

        /* r needs to be multiplied by 2 or 2*sqrt(2) depending on LM because
        short blocks don't have the same energy as long */
        r = exp2f(1 - Ediff);
        if (f->size == 3)
            r *= M_SQRT2;
        r = FFMIN(thresh, r) * sqrt_1;
        for (k = 0; k < 1 << f->size; k++) {
            /* Detect collapse */
            if (!(block->collapse_masks[i] & 1 << k)) {
                /* Fill with noise */
                for (j = 0; j < ff_celt_freq_range[i]; j++)
                    xptr[(j << f->size) + k] = (celt_rng(f) & 0x8000) ? r : -r;
                renormalize = 1;
            }
        }

        /* We just added some energy, so we need to renormalize */
        if (renormalize)
            celt_renormalize_vector(xptr, ff_celt_freq_range[i] << f->size, 1.0f);
    }
}

static int ff_celt_decode_frame(CeltFrame *f, OpusRangeCoder *rc,
                         float **output, int channels, int frame_size,
                         int start_band,  int end_band)
{
    int i, j, downmix = 0;
    int consumed;           // bits of entropy consumed thus far for this frame
    CeltImdct *imdct;

    if (channels != 1 && channels != 2) {
        return ERROR_INVALID;
    }
    if (start_band < 0 || start_band > end_band || end_band > CELT_MAX_BANDS) {
        return ERROR_INVALID;
    }

    f->silence        = 0;
    f->transient      = 0;
    f->anticollapse   = 0;
    f->flushed        = 0;
    f->channels       = channels;
    f->start_band     = start_band;
    f->end_band       = end_band;
    f->framebits      = rc->rb.bytes * 8;

    f->size = av_log2(frame_size / CELT_SHORT_BLOCKSIZE);
    if (f->size > CELT_MAX_LOG_BLOCKS ||
        frame_size != CELT_SHORT_BLOCKSIZE * (1 << f->size)) {
        return ERROR_INVALID;
    }

    if (!f->output_channels)
        f->output_channels = channels;

    for (i = 0; i < f->channels; i++) {
        memset(f->block[i].coeffs,         0, sizeof(f->block[i].coeffs));
        memset(f->block[i].collapse_masks, 0, sizeof(f->block[i].collapse_masks));
    }

    consumed = opus_rc_tell(rc);

    /* obtain silence flag */
    if (consumed >= f->framebits)
        f->silence = 1;
    else if (consumed == 1)
        f->silence = ff_opus_rc_dec_log(rc, 15);


    if (f->silence) {
        consumed = f->framebits;
        rc->total_bits += f->framebits - opus_rc_tell(rc);
    }

    /* obtain post-filter options */
    consumed = parse_postfilter(f, rc, consumed);

    /* obtain transient flag */
    if (f->size != 0 && consumed+3 <= f->framebits)
        f->transient = ff_opus_rc_dec_log(rc, 3);

    f->blocks    = f->transient ? 1 << f->size : 1;
    f->blocksize = frame_size / f->blocks;

    imdct = &f->tx[f->transient ? 0 : f->size];

    if (channels == 1) {
        for (i = 0; i < CELT_MAX_BANDS; i++)
            f->block[0].energy[i] = FFMAX(f->block[0].energy[i], f->block[1].energy[i]);
    }

    celt_decode_coarse_energy(f, rc);
    celt_decode_tf_changes   (f, rc);
    ff_celt_bitalloc         (f, rc);
    celt_decode_fine_energy  (f, rc);
    ff_celt_quant_bands      (f, rc);

    if (f->anticollapse_needed)
        f->anticollapse = ff_opus_rc_get_raw(rc, 1);

    celt_decode_final_energy(f, rc);

    /* apply anti-collapse processing and denormalization to
     * each coded channel */
    for (i = 0; i < f->channels; i++) {
        CeltBlock *block = &f->block[i];

        if (f->anticollapse)
            process_anticollapse(f, block, f->block[i].coeffs);

        celt_denormalize(f, block, f->block[i].coeffs);
    }

    /* stereo -> mono downmix */
    if (f->output_channels < f->channels) {
        vector_fmac_scalar(f->block[0].coeffs, f->block[1].coeffs, 1.0, FFALIGN(frame_size, 16));
        downmix = 1;
    } else if (f->output_channels > f->channels)
        memcpy(f->block[1].coeffs, f->block[0].coeffs, frame_size * sizeof(float));

    if (f->silence) {
        for (i = 0; i < 2; i++) {
            CeltBlock *block = &f->block[i];

            for (j = 0; j < FF_ARRAY_ELEMS(block->energy); j++)
                block->energy[j] = CELT_ENERGY_SILENCE;
        }
        memset(f->block[0].coeffs, 0, sizeof(f->block[0].coeffs));
        memset(f->block[1].coeffs, 0, sizeof(f->block[1].coeffs));
    }

    /* transform and output for each output channel */
    for (i = 0; i < f->output_channels; i++) {
        CeltBlock *block = &f->block[i];

        /* iMDCT and overlap-add */
        for (j = 0; j < f->blocks; j++) {
            float *dst  = block->buf + 1024 + j * f->blocksize;

            ff_celt_imdct(imdct, dst + CELT_OVERLAP / 2, f->block[i].coeffs + j, f->blocks);
            vector_fmul_window(dst, dst, dst + CELT_OVERLAP / 2,
                               ff_celt_window, CELT_OVERLAP / 2);
        }

        if (downmix)
            vector_fmul_scalar(&block->buf[1024], &block->buf[1024], 0.5f, frame_size);

        /* postfilter */
        celt_postfilter(f, block);

        /* deemphasis */
        block->emph_coeff = deemphasis_c(output[i],
                                         &block->buf[1024 - frame_size],
                                         block->emph_coeff,
                                         ff_opus_deemph_weights,
                                         frame_size);
        if (!std::isnormal(block->emph_coeff))
            block->emph_coeff = 0.0;
    }

    if (channels == 1)
        memcpy(f->block[1].energy, f->block[0].energy, sizeof(f->block[0].energy));

    for (i = 0; i < 2; i++ ) {
        CeltBlock *block = &f->block[i];

        if (!f->transient) {
            memcpy(block->prev_energy[1], block->prev_energy[0], sizeof(block->prev_energy[0]));
            memcpy(block->prev_energy[0], block->energy,         sizeof(block->prev_energy[0]));
        } else {
            for (j = 0; j < CELT_MAX_BANDS; j++)
                block->prev_energy[0][j] = FFMIN(block->prev_energy[0][j], block->energy[j]);
        }

        for (j = 0; j < f->start_band; j++) {
            block->prev_energy[0][j] = CELT_ENERGY_SILENCE;
            block->energy[j]         = 0.0;
        }
        for (j = f->end_band; j < CELT_MAX_BANDS; j++) {
            block->prev_energy[0][j] = CELT_ENERGY_SILENCE;
            block->energy[j]         = 0.0;
        }
    }

    f->seed = rc->range;

    return 0;
}

static void ff_celt_flush(CeltFrame *f)
{
    int i, j;

    if (f->flushed)
        return;

    for (i = 0; i < 2; i++) {
        CeltBlock *block = &f->block[i];

        for (j = 0; j < CELT_MAX_BANDS; j++)
            block->prev_energy[0][j] = block->prev_energy[1][j] = CELT_ENERGY_SILENCE;

        memset(block->energy, 0, sizeof(block->energy));
        memset(block->buf,    0, sizeof(block->buf));

        memset(block->pf_gains,     0, sizeof(block->pf_gains));
        memset(block->pf_gains_old, 0, sizeof(block->pf_gains_old));
        memset(block->pf_gains_new, 0, sizeof(block->pf_gains_new));

        /* libopus uses CELT_EMPH_COEFF on init, but 0 is better since there's
         * a lesser discontinuity when seeking.
         * The deemphasis functions differ from libopus in that they require
         * an initial state divided by the coefficient. */
        block->emph_coeff = 0.0f / ff_opus_deemph_weights[0];
    }
    f->seed = 0;

    f->flushed = 1;
}

static void ff_celt_init(CeltFrame *f, int output_channels, int apply_phase_inv)
{
    f->output_channels = output_channels;
    f->apply_phase_inv = apply_phase_inv;

    for (int i = 0; i < 4; i++)
        ff_celt_imdct_init(&f->tx[i], 15 << (i + 3), -1.0f/32768);

    ff_celt_flush(f);
}
