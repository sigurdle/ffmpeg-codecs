/*
	The inverse MDCT CELT calls: FFmpeg n8.1 libavutil/tx_template.c's float path for AV_TX_FLOAT_MDCT (LGPL-2.1+) at lengths 120,
	240, 480 and 960, with the half-length output of ff_tx_mdct_inv and the scale of ff_tx_mdct_gen_exp.

	Copyright (c) Lynne

	Departures from FFmpeg:
	  - A plain transform in place of the tx framework: no codelet list, no permutation tables. The complex transform of 15 * 2^k
	    points is the same prime-factor split as ff_tx_mdct_pfa_15xM_inv, written by index: input n = M*n1 + 15*n2 and output k with
	    n1's transform at k % 15 and n2's at k % M, and 15 likewise into 3 and 5. fft3 and the fft5 butterfly are FFmpeg's.
	  - The 2^k points are a radix-2 loop, not the split-radix codelets. Its sum order differs, as FFmpeg's x86 assembly does, so
	    the float output differs by rounding only.
	  - Reading the input backward makes it the inverse, as FFmpeg's inverse maps do.
*/

#pragma once

#define CELT_IMDCT_MAX_POINTS 480

typedef struct TXComplex {
    float re;
    float im;
} TXComplex;

typedef struct CeltImdct {
    int coefficients;
    int points;      /* of the complex transform: coefficients / 2 */
    int power2;      /* points / 15 */
    TXComplex exp[CELT_IMDCT_MAX_POINTS];
    TXComplex twiddle[16];
    TXComplex tmp[CELT_IMDCT_MAX_POINTS];
} CeltImdct;

#define BF(x, y, a, b)  \
    do {                \
        x = (a) - (b);  \
        y = (a) + (b);  \
    } while (0)

#define CMUL(dre, dim, are, aim, bre, bim)      \
    do {                                        \
        (dre) = (are) * (bre) - (aim) * (bim);  \
        (dim) = (are) * (bim) + (aim) * (bre);  \
    } while (0)

#define SMUL(dre, dim, are, aim, bre, bim)      \
    do {                                        \
        (dre) = (are) * (bre) - (aim) * (bim);  \
        (dim) = (are) * (bim) - (aim) * (bre);  \
    } while (0)

#define CMUL3(c, a, b) CMUL((c).re, (c).im, (a).re, (a).im, (b).re, (b).im)

static const float tx_cos_2pi_5 = (float)cos(2 * M_PI / 5);
static const float tx_cos_2pi_10 = (float)cos(2 * M_PI / 10);
static const float tx_sin_2pi_5 = (float)sin(2 * M_PI / 5);
static const float tx_sin_2pi_10 = (float)sin(2 * M_PI / 10);
static const float tx_cos_2pi_12 = (float)cos(2 * M_PI / 12);
static const float tx_cos_2pi_6 = (float)cos(2 * M_PI / 6);

static inline void fft3(TXComplex *out, const TXComplex *in)
{
    TXComplex tmp[3];

    tmp[0] = in[0];
    BF(tmp[1].re, tmp[2].im, in[1].im, in[2].im);
    BF(tmp[1].im, tmp[2].re, in[1].re, in[2].re);

    out[0].re = tmp[0].re + tmp[2].re;
    out[0].im = tmp[0].im + tmp[2].im;
    tmp[1].re = tx_cos_2pi_12 * tmp[1].re;
    tmp[1].im = tx_cos_2pi_12 * tmp[1].im;
    tmp[2].re = tx_cos_2pi_6 * tmp[2].re;
    tmp[2].im = tx_cos_2pi_6 * tmp[2].im;
    out[1].re = tmp[0].re - tmp[2].re + tmp[1].re;
    out[1].im = tmp[0].im - tmp[2].im - tmp[1].im;
    out[2].re = tmp[0].re - tmp[2].re - tmp[1].re;
    out[2].im = tmp[0].im - tmp[2].im + tmp[1].im;
}

static inline void fft5(TXComplex *out, const TXComplex *in)
{
    TXComplex dc, z0[4], t[6];

    dc = in[0];
    BF(t[1].im, t[0].re, in[1].re, in[4].re);
    BF(t[1].re, t[0].im, in[1].im, in[4].im);
    BF(t[3].im, t[2].re, in[2].re, in[3].re);
    BF(t[3].re, t[2].im, in[2].im, in[3].im);

    out[0].re = dc.re + t[0].re + t[2].re;
    out[0].im = dc.im + t[0].im + t[2].im;

    SMUL(t[4].re, t[0].re, tx_cos_2pi_5, tx_cos_2pi_10, t[2].re, t[0].re);
    SMUL(t[4].im, t[0].im, tx_cos_2pi_5, tx_cos_2pi_10, t[2].im, t[0].im);
    CMUL(t[5].re, t[1].re, tx_sin_2pi_5, tx_sin_2pi_10, t[3].re, t[1].re);
    CMUL(t[5].im, t[1].im, tx_sin_2pi_5, tx_sin_2pi_10, t[3].im, t[1].im);

    BF(z0[0].re, z0[3].re, t[0].re, t[1].re);
    BF(z0[0].im, z0[3].im, t[0].im, t[1].im);
    BF(z0[2].re, z0[1].re, t[4].re, t[5].re);
    BF(z0[2].im, z0[1].im, t[4].im, t[5].im);

    out[1].re = dc.re + z0[3].re;
    out[1].im = dc.im + z0[0].im;
    out[2].re = dc.re + z0[2].re;
    out[2].im = dc.im + z0[1].im;
    out[3].re = dc.re + z0[1].re;
    out[3].im = dc.im + z0[2].im;
    out[4].re = dc.re + z0[0].re;
    out[4].im = dc.im + z0[3].im;
}

/** Forward: the sign of the exponent is negative. `in` holds n = 5*a + 3*b, `out` holds k. */
static inline void fft15(TXComplex *out, const TXComplex *in)
{
    TXComplex row[3], column[5][3], line[5], result[5];

    for (int b = 0; b < 5; b++) {
        for (int a = 0; a < 3; a++)
            row[a] = in[(5 * a + 3 * b) % 15];
        fft3(column[b], row);
    }
    for (int a = 0; a < 3; a++) {
        for (int b = 0; b < 5; b++)
            line[b] = column[b][a];
        fft5(result, line);
        for (int k = 0; k < 5; k++)
            out[(10 * a + 6 * k) % 15] = result[k];
    }
}

/** Forward and in place, `points` a power of 2 up to 32. */
static void fft_power2(TXComplex *z, int points, const TXComplex *twiddle)
{
    for (int i = 1, j = 0; i < points; i++) {
        int bit = points >> 1;
        for (; j & bit; bit >>= 1)
            j ^= bit;
        j ^= bit;
        if (i < j) {
            const TXComplex swap = z[i];
            z[i] = z[j];
            z[j] = swap;
        }
    }
    for (int half = 1; half < points; half <<= 1) {
        const int step = points / (half * 2);
        for (int start = 0; start < points; start += half * 2) {
            for (int i = 0; i < half; i++) {
                const TXComplex w = twiddle[i * step];
                TXComplex *a = z + start + i;
                TXComplex *b = a + half;
                TXComplex t;
                CMUL3(t, *b, w);
                BF(b->re, a->re, a->re, t.re);
                BF(b->im, a->im, a->im, t.im);
            }
        }
    }
}

/** The inverse complex transform of s->points points, in place. */
static void imdct_fft(CeltImdct *s, TXComplex *z)
{
    const int points = s->points;
    const int m = s->power2;
    TXComplex *tmp = s->tmp;
    TXComplex in[15], out[15];

    for (int n2 = 0; n2 < m; n2++) {
        for (int n1 = 0; n1 < 15; n1++)
            in[n1] = z[(points - (m * n1 + 15 * n2) % points) % points];
        fft15(out, in);
        for (int k1 = 0; k1 < 15; k1++)
            tmp[k1 * m + n2] = out[k1];
    }
    for (int k1 = 0; k1 < 15; k1++)
        fft_power2(tmp + k1 * m, m, s->twiddle);
    for (int k = 0; k < points; k++)
        z[k] = tmp[(k % 15) * m + (k % m)];
}

/** ff_tx_mdct_gen_exp for a negative scale. */
static void ff_celt_imdct_init(CeltImdct *s, int coefficients, float scale)
{
    const int points = coefficients >> 1;
    const double theta = points + 1.0 / 8.0;
    const double root = sqrt(fabs((double)scale));

    s->coefficients = coefficients;
    s->points = points;
    s->power2 = points / 15;

    for (int i = 0; i < points; i++) {
        const double alpha = M_PI_2 * (i + theta) / points;
        s->exp[i].re = (float)(cos(alpha) * root);
        s->exp[i].im = (float)(sin(alpha) * root);
    }
    for (int i = 0; i < s->power2; i++) {
        const double alpha = 2 * M_PI * i / s->power2;
        s->twiddle[i].re = (float)cos(alpha);
        s->twiddle[i].im = (float)-sin(alpha);
    }
}

/** ff_tx_mdct_inv: s->coefficients samples at `dst` from every `stride`-th of as many coefficients at `src`. */
static void ff_celt_imdct(CeltImdct *s, float *dst, const float *src, int stride)
{
    TXComplex *z = (TXComplex *)dst;
    const TXComplex *exp = s->exp;
    const int len2 = s->points;
    const int len4 = len2 >> 1;
    const float *in1 = src;
    const float *in2 = src + (s->coefficients - 1) * stride;

    for (int i = 0; i < len2; i++) {
        const int k = 2 * i;
        const TXComplex tmp = { in2[-k * stride], in1[k * stride] };
        CMUL3(z[i], tmp, exp[i]);
    }

    imdct_fft(s, z);

    for (int i = 0; i < len4; i++) {
        const int i0 = len4 + i, i1 = len4 - i - 1;
        const TXComplex src1 = { z[i1].im, z[i1].re };
        const TXComplex src0 = { z[i0].im, z[i0].re };

        CMUL(z[i1].re, z[i0].im, src1.re, src1.im, exp[i1].im, exp[i1].re);
        CMUL(z[i0].re, z[i1].im, src0.re, src0.im, exp[i0].im, exp[i0].re);
    }
}
