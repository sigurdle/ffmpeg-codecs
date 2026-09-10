/*
	MPEG audio DSP: the polyphase synthesis filterbank and the Layer III hybrid
	(IMDCT) stage.

	Ported from FFmpeg's mpegaudiodsp_template.c / mpegaudiodsp.c / dct32_template.c
	(libavcodec, release/8.1, LGPL), specialized to the float build. FFmpeg compiles
	those files twice through a macro layer (USE_FLOATS on/off) to get a fixed-point
	and a floating-point decoder from one source; we only need the float one, so the
	macros are resolved here:

	    MULH3(x, y, s) -> (s)*(y)*(x)      MULLx(x, y, s) -> (y)*(x)
	    FIXR(x), FIXHR(x) -> (float)(x)    SHR(a, b) -> (a) * (1.0f / (1 << b))

	The 32-point DCT itself lives in the generated dct32_float.h — see the note there
	for why it is lifted verbatim instead of rewritten.
*/

#ifndef MPEGAUDIO_DSP_H
#define MPEGAUDIO_DSP_H

#include <stdint.h>
#include <string.h>
#include <math.h>

#include "mpegaudio_tables.h"
#include "dct32_float.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define SBLIMIT        32	/* subbands */
#define MDCT_BUF_SIZE  40	/* FFALIGN(36, 8) */
#define IMDCT_SCALAR   1.759

/* Window for the Layer III MDCT. Only [0,17] and [MDCT_BUF_SIZE/2, +17] are used;
   the rest exists to keep FFmpeg's SIMD alignment, and is kept so the indexing
   below can stay identical to FFmpeg's. */
static float ff_mdct_win[8][MDCT_BUF_SIZE];

/* Polyphase synthesis window, 512 taps plus FFmpeg's 256-entry shuffled tail. */
static float ff_mpa_synth_window[512 + 256];

static void mpa_dsp_init_tabs(void)
{
	/* MDCT windows (mpegaudiodsp.c mpadsp_init_tabs) */
	for (int i = 0; i < 36; i++) {
		for (int j = 0; j < 4; j++) {
			double d;
			if (j == 2 && i % 3 != 1) continue;

			d = sin(M_PI * (i + 0.5) / 36.0);
			if (j == 1) {
				if      (i >= 30) d = 0;
				else if (i >= 24) d = sin(M_PI * (i - 18 + 0.5) / 12.0);
				else if (i >= 18) d = 1;
			} else if (j == 3) {
				if      (i <   6) d = 0;
				else if (i <  12) d = sin(M_PI * (i -  6 + 0.5) / 12.0);
				else if (i <  18) d = 1;
			}
			// the last stage of the IMDCT is merged into the window coefficients
			d *= 0.5 * IMDCT_SCALAR / cos(M_PI * (2 * i + 19) / 72);

			if (j == 2) {
				ff_mdct_win[j][i / 3] = (float)(d / (1 << 5));
			} else {
				int idx = i < 18 ? i : i + (MDCT_BUF_SIZE / 2 - 18);
				ff_mdct_win[j][idx] = (float)(d / (1 << 5));
			}
		}
	}
	/* Frequency inversion after the MDCT, done by flipping the sign of the odd
	   window coefficients into a second set of four windows. */
	for (int j = 0; j < 4; j++) {
		for (int i = 0; i < MDCT_BUF_SIZE; i += 2) {
			ff_mdct_win[j + 4][i    ] =  ff_mdct_win[j][i    ];
			ff_mdct_win[j + 4][i + 1] = -ff_mdct_win[j][i + 1];
		}
	}

	/* Synthesis window (mpegaudiodsp_template.c mpa_synth_init). The stored table is
	   a quarter period; the rest is mirrored, with every 64th sign flipped. */
	float* window = ff_mpa_synth_window;
	for (int i = 0; i < 257; i++) {
		float v = (float)ff_mpa_enwindow[i] * (1.0f / (1LL << (16 + 23)));	// 16 + FRAC_BITS
		window[i] = v;
		if ((i & 63) != 0) v = -v;
		if (i != 0) window[512 - i] = v;
	}
	// FFmpeg's pre-shuffled tail (kept so apply_window indexes identically).
	for (int i = 0; i < 8; i++)
		for (int j = 0; j < 16; j++)
			window[512 + 16 * i + j] = window[64 * i + 32 - j];
	for (int i = 0; i < 8; i++)
		for (int j = 0; j < 16; j++)
			window[512 + 128 + 16 * i + j] = window[64 * i + 48 - j];
}

// ─── windowing (ff_mpadsp_apply_window_float) ────────────────────────────────

#define SUM8(op, sum, w, p)            \
{                                      \
	op(sum, (w)[0 * 64], (p)[0 * 64]); \
	op(sum, (w)[1 * 64], (p)[1 * 64]); \
	op(sum, (w)[2 * 64], (p)[2 * 64]); \
	op(sum, (w)[3 * 64], (p)[3 * 64]); \
	op(sum, (w)[4 * 64], (p)[4 * 64]); \
	op(sum, (w)[5 * 64], (p)[5 * 64]); \
	op(sum, (w)[6 * 64], (p)[6 * 64]); \
	op(sum, (w)[7 * 64], (p)[7 * 64]); \
}

#define SUM8P2(sum1, op1, sum2, op2, w1, w2, p) \
{                                               \
	float tmp;                                  \
	tmp = p[0 * 64]; op1(sum1, (w1)[0 * 64], tmp); op2(sum2, (w2)[0 * 64], tmp); \
	tmp = p[1 * 64]; op1(sum1, (w1)[1 * 64], tmp); op2(sum2, (w2)[1 * 64], tmp); \
	tmp = p[2 * 64]; op1(sum1, (w1)[2 * 64], tmp); op2(sum2, (w2)[2 * 64], tmp); \
	tmp = p[3 * 64]; op1(sum1, (w1)[3 * 64], tmp); op2(sum2, (w2)[3 * 64], tmp); \
	tmp = p[4 * 64]; op1(sum1, (w1)[4 * 64], tmp); op2(sum2, (w2)[4 * 64], tmp); \
	tmp = p[5 * 64]; op1(sum1, (w1)[5 * 64], tmp); op2(sum2, (w2)[5 * 64], tmp); \
	tmp = p[6 * 64]; op1(sum1, (w1)[6 * 64], tmp); op2(sum2, (w2)[6 * 64], tmp); \
	tmp = p[7 * 64]; op1(sum1, (w1)[7 * 64], tmp); op2(sum2, (w2)[7 * 64], tmp); \
}

#define MACS(rt, ra, rb) rt += (ra) * (rb)
#define MLSS(rt, ra, rb) rt -= (ra) * (rb)

static inline float round_sample(float* sum)
{
	float sum1 = *sum;
	*sum = 0;
	return sum1;
}

static void mpa_apply_window(float* synth_buf, float* window, int* dither_state,
                             float* samples, ptrdiff_t incr)
{
	const float *w, *w2, *p;
	float *samples2;
	float sum, sum2;

	// copy to avoid a wrap check in the inner loops
	memcpy(synth_buf + 512, synth_buf, 32 * sizeof(*synth_buf));

	samples2 = samples + 31 * incr;
	w  = window;
	w2 = window + 31;

	sum = (float)*dither_state;
	p = synth_buf + 16;
	SUM8(MACS, sum, w, p);
	p = synth_buf + 48;
	SUM8(MLSS, sum, w + 32, p);
	*samples = round_sample(&sum);
	samples += incr;
	w++;

	// two samples at a time, to halve the memory traffic
	for (int j = 1; j < 16; j++) {
		sum2 = 0;
		p = synth_buf + 16 + j;
		SUM8P2(sum, MACS, sum2, MLSS, w, w2, p);
		p = synth_buf + 48 - j;
		SUM8P2(sum, MLSS, sum2, MLSS, w + 32, w2 + 32, p);

		*samples = round_sample(&sum);
		samples += incr;
		sum += sum2;
		*samples2 = round_sample(&sum);
		samples2 -= incr;
		w++;
		w2--;
	}

	p = synth_buf + 32;
	SUM8(MLSS, sum, w + 32, p);
	*samples = round_sample(&sum);
	*dither_state = (int)sum;
}

/* 32 subband samples in, 32 PCM samples out (ff_mpa_synth_filter_float). */
static void mpa_synth_filter(float* synth_buf_ptr, int* synth_buf_offset,
                             float* window, int* dither_state,
                             float* samples, ptrdiff_t incr, float* sb_samples)
{
	int offset = *synth_buf_offset;
	float* synth_buf = synth_buf_ptr + offset;

	ff_dct32_float(synth_buf, sb_samples);
	mpa_apply_window(synth_buf, window, dither_state, samples, incr);

	*synth_buf_offset = (offset - 32) & 511;
}

// ─── Layer III hybrid synthesis: IMDCT (mpegaudiodsp_template.c) ─────────────

/*
	CAREFUL: FFmpeg defines two DIFFERENT sets of C3..C8 — one in
	mpegaudiodsp_template.c for imdct36 (cos(i*pi/18)) and one in
	mpegaudiodec_template.c for imdct12 (0.5/cos(pi*k/36)) — and they collide on the
	names C4, C5 and C6. Because each lives in its own translation unit upstream, the
	clash is invisible there; folding both into this one header is not, so they are
	prefixed here. C3 happens to agree; the rest do not.
*/
#define IM36_C1 ((float)(0.98480775301220805936 / 2))	/* cos(i*pi/18) */
#define IM36_C2 ((float)(0.93969262078590838405 / 2))
#define IM36_C3 ((float)(0.86602540378443864676 / 2))
#define IM36_C4 ((float)(0.76604444311897803520 / 2))
#define IM36_C5 ((float)(0.64278760968653932632 / 2))
#define IM36_C6 ((float)(0.5 / 2))
#define IM36_C7 ((float)(0.34202014332566873304 / 2))
#define IM36_C8 ((float)(0.17364817766693034885 / 2))

#define IM12_C3 ((float)(0.86602540378443864676 / 2))	/* 0.5 / cos(pi*k/36) */
#define IM12_C4 ((float)(0.70710678118654752439 / 2))
#define IM12_C5 ((float)(0.51763809020504152469 / 2))
#define IM12_C6 ((float)(1.93185165257813657349 / 4))

static const float icos36[9] = {
	0.50190991877167369479f, 0.51763809020504152469f, 0.55168895948124587824f,
	0.61038729438072803416f, 0.70710678118654752439f, 0.87172339781054900991f,
	1.18310079157624925896f, 1.93185165257813657349f, 5.73685662283492756461f,
};

/* 0.5 / cos(pi*(2*i+1)/36) */
static const float icos36h[9] = {
	(float)(0.50190991877167369479 / 2), (float)(0.51763809020504152469 / 2),
	(float)(0.55168895948124587824 / 2), (float)(0.61038729438072803416 / 2),
	(float)(0.70710678118654752439 / 2), (float)(0.87172339781054900991 / 2),
	(float)(1.18310079157624925896 / 4), (float)(1.93185165257813657349 / 4),
};

#define MULH3F(x, y, s) ((s) * (y) * (x))
#define SHRF(a, b)      ((a) * (1.0f / (1 << (b))))

/* 36-point IMDCT: a Lee-style decomposition followed by a hand-coded 9-point DCT.
   Windows, overlap-adds with `buf`, and scatters into `out` with SBLIMIT stride. */
static void imdct36(float* out, float* buf, float* in, float* win)
{
	float t0, t1, t2, t3, s0, s1, s2, s3;
	float tmp[18], *tmp1, *in1;

	for (int i = 17; i >= 1; i--) in[i] += in[i - 1];
	for (int i = 17; i >= 3; i -= 2) in[i] += in[i - 2];

	for (int j = 0; j < 2; j++) {
		tmp1 = tmp + j;
		in1 = in + j;

		t2 = in1[2 * 4] + in1[2 * 8] - in1[2 * 2];

		t3 = in1[2 * 0] + SHRF(in1[2 * 6], 1);
		t1 = in1[2 * 0] - in1[2 * 6];
		tmp1[ 6] = t1 - SHRF(t2, 1);
		tmp1[16] = t1 + t2;

		t0 = MULH3F(in1[2 * 2] + in1[2 * 4],    IM36_C2, 2);
		t1 = MULH3F(in1[2 * 4] - in1[2 * 8], -2 * IM36_C8, 1);
		t2 = MULH3F(in1[2 * 2] + in1[2 * 8],   -IM36_C4, 2);

		tmp1[10] = t3 - t0 - t2;
		tmp1[ 2] = t3 + t0 + t1;
		tmp1[14] = t3 + t2 - t1;

		tmp1[ 4] = MULH3F(in1[2 * 5] + in1[2 * 7] - in1[2 * 1], -IM36_C3, 2);
		t2 = MULH3F(in1[2 * 1] + in1[2 * 5],    IM36_C1, 2);
		t3 = MULH3F(in1[2 * 5] - in1[2 * 7], -2 * IM36_C7, 1);
		t0 = MULH3F(in1[2 * 3], IM36_C3, 2);

		t1 = MULH3F(in1[2 * 1] + in1[2 * 7],   -IM36_C5, 2);

		tmp1[ 0] = t2 + t3 + t0;
		tmp1[12] = t2 + t1 - t0;
		tmp1[ 8] = t3 - t1 - t0;
	}

	int i = 0;
	for (int j = 0; j < 4; j++) {
		t0 = tmp[i];
		t1 = tmp[i + 2];
		s0 = t1 + t0;
		s2 = t1 - t0;

		t2 = tmp[i + 1];
		t3 = tmp[i + 3];
		s1 = MULH3F(t3 + t2, icos36h[j], 2);
		s3 = (t3 - t2) * icos36[8 - j];		/* MULLx */

		t0 = s0 + s1;
		t1 = s0 - s1;
		out[(9 + j) * SBLIMIT] = MULH3F(t1, win[     9 + j], 1) + buf[4 * (9 + j)];
		out[(8 - j) * SBLIMIT] = MULH3F(t1, win[     8 - j], 1) + buf[4 * (8 - j)];
		buf[4 * ( 9 + j     )] = MULH3F(t0, win[MDCT_BUF_SIZE / 2 + 9 + j], 1);
		buf[4 * ( 8 - j     )] = MULH3F(t0, win[MDCT_BUF_SIZE / 2 + 8 - j], 1);

		t0 = s2 + s3;
		t1 = s2 - s3;
		out[(9 + 8 - j) * SBLIMIT] = MULH3F(t1, win[     9 + 8 - j], 1) + buf[4 * (9 + 8 - j)];
		out[         j  * SBLIMIT] = MULH3F(t1, win[             j], 1) + buf[4 * (        j)];
		buf[4 * ( 9 + 8 - j     )] = MULH3F(t0, win[MDCT_BUF_SIZE / 2 + 9 + 8 - j], 1);
		buf[4 * (         j     )] = MULH3F(t0, win[MDCT_BUF_SIZE / 2         + j], 1);
		i += 4;
	}

	s0 = tmp[16];
	s1 = MULH3F(tmp[17], icos36h[4], 2);
	t0 = s0 + s1;
	t1 = s0 - s1;
	out[(9 + 4) * SBLIMIT] = MULH3F(t1, win[     9 + 4], 1) + buf[4 * (9 + 4)];
	out[(8 - 4) * SBLIMIT] = MULH3F(t1, win[     8 - 4], 1) + buf[4 * (8 - 4)];
	buf[4 * ( 9 + 4     )] = MULH3F(t0, win[MDCT_BUF_SIZE / 2 + 9 + 4], 1);
	buf[4 * ( 8 - 4     )] = MULH3F(t0, win[MDCT_BUF_SIZE / 2 + 8 - 4], 1);
}

static void imdct36_blocks(float* out, float* buf, float* in,
                           int count, int switch_point, int block_type)
{
	for (int j = 0; j < count; j++) {
		/* window selection: the first two subbands of a switched granule use the
		   long window, and odd subbands use the frequency-inverted set */
		int win_idx = (switch_point && j < 2) ? 0 : block_type;
		float* win = ff_mdct_win[win_idx + (4 & -(j & 1))];

		imdct36(out, buf, in, win);

		in  += 18;
		buf += ((j & 3) != 3 ? 1 : (72 - 3));
		out++;
	}
}

/* 12-point IMDCT for short blocks, factorized by hand. */
static void imdct12(float* out, float* in)
{
	float in0, in1, in2, in3, in4, in5, t1, t2;

	in0 = in[0 * 3];
	in1 = in[1 * 3] + in[0 * 3];
	in2 = in[2 * 3] + in[1 * 3];
	in3 = in[3 * 3] + in[2 * 3];
	in4 = in[4 * 3] + in[3 * 3];
	in5 = in[5 * 3] + in[4 * 3];
	in5 += in3;
	in3 += in1;

	in2 = MULH3F(in2, IM12_C3, 2);
	in3 = MULH3F(in3, IM12_C3, 4);

	t1 = in0 - in4;
	t2 = MULH3F(in1 - in5, IM12_C4, 2);

	out[ 7] =
	out[10] = t1 + t2;
	out[ 1] =
	out[ 4] = t1 - t2;

	in0 += SHRF(in4, 1);
	in4 = in0 + in2;
	in5 += 2 * in1;
	in1 = MULH3F(in5 + in3, IM12_C5, 1);
	out[ 8] =
	out[ 9] = in4 + in1;
	out[ 2] =
	out[ 3] = in4 - in1;

	in0 -= in2;
	in5 = MULH3F(in5 - in3, IM12_C6, 2);
	out[ 0] =
	out[ 5] = in0 - in5;
	out[ 6] =
	out[11] = in0 + in5;
}

#endif /* MPEGAUDIO_DSP_H */
