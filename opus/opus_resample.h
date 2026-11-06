/*
	The resampler from SILK's 8, 12 or 16 kHz to 48 kHz: the path FFmpeg n8.1's libavcodec/opus/dec.c takes through libswresample
	(LGPL-2.1+): resample.c, resample_template.c, the buffer handling of swresample.c.

	dec.c sets filter_size 16, float planar in and out, and leaves the rest at options.c's defaults: phase_shift 10, linear_interp 1,
	exact_rational 1, cutoff 0 (0.97 for this engine), Kaiser with beta 9, no dither, no rematrix. With those, for 48000 / rate
	(6, 4 or 3):
	  - exact_rational makes phase_count that ratio, so src_incr equals dst_incr and frac and dst_incr_mod stay 0;
	  - that makes resample_linear never run: every output is resample_common_float at the next phase;
	  - factor is min(48000 * 0.97 / rate, 1) = 1, so the filter is 16 taps with no scaling.

	Departures from FFmpeg:
	  - Only that one path: one engine, one format, no compensation; the increments are gone since the phase just counts up.
	  - One linear buffer in place of the in_buffer and input juggling of resample(): which samples are read and when the output
	    stops are the same, x86's padless included, since padless only moves the last 7 samples from the input to in_buffer.
	  - av_bessel_i0 is the power series, as ac3_decode.cpp's; the taps are stored in float, so it does not show.
	  - The sum runs in the order of FFmpeg's C code. The x86 builds of FFmpeg sum in vector order, so `ffmpeg -f f32le` differs by
	    float round-off.
*/

#pragma once

#define SWR_FILTER_LENGTH 16
#define SWR_KAISER_BETA 9.0
#define SWR_MAX_PHASES 6
// The longest input (60 ms at 16 kHz) past what a call leaves over, and the 16 samples invert_initial_buffer writes before the start.
#define SWR_BUFFER_SAMPLES (960 + 3 * SWR_FILTER_LENGTH)

struct SwrContext {
	int in_sample_rate;		// 0 while closed, what swr_is_initialized says
	int channels;
	int phase_count;
	const float* filter_bank;
	int index;				// the phase, negative until invert_initial_buffer has run
	int in_buffer_index;
	int in_buffer_count;
	int flushed;
	float in_buffer[2][SWR_BUFFER_SAMPLES];
};

static double swr_bessel_i0(double x)
{
	double sum = 1.0, term = 1.0;
	const double x2 = x * x / 4.0;
	for (int k = 1; k < 200; k++) {
		term *= x2 / ((double)k * (double)k);
		sum += term;
		if (term < sum * 1e-20) break;
	}
	return sum;
}

/** build_filter of resample.c for factor 1, the Kaiser window and scale 1. */
static void swr_build_filter(float* filter, int phase_count)
{
	const int tap_count = SWR_FILTER_LENGTH;
	const int ph_nb = phase_count % 2 ? phase_count : phase_count / 2 + 1;
	const int center = (tap_count - 1) / 2;
	double tab[SWR_FILTER_LENGTH];
	double norm = 0;

	for (int ph = 0; ph < ph_nb; ph++) {
		double s = sin(M_PI * ph / phase_count) * (center & 1 ? 1 : -1);
		for (int i = 0; i < tap_count; i++) {
			const double x = M_PI * ((double)(i - center) - (double)ph / phase_count);
			double y = x == 0 ? 1.0 : s / x;
			const double w = 2.0 * x / (tap_count * M_PI);
			y *= swr_bessel_i0(SWR_KAISER_BETA * sqrt(fmax(1 - w * w, 0)));

			tab[i] = y;
			s = -s;
			if (!ph)
				norm += y;
		}

		for (int i = 0; i < tap_count; i++)
			filter[ph * tap_count + i] = tab[i] / norm;
		if (phase_count % 2 || !ph)
			continue;
		for (int i = 0; i < tap_count; i++)
			filter[(phase_count - ph) * tap_count + tap_count - 1 - i] = filter[ph * tap_count + i];
	}
}

/** swr_init with the rate named: the bank for each rate is built once. */
static void swr_init(SwrContext* s, int in_sample_rate)
{
	static float banks[3][SWR_MAX_PHASES * SWR_FILTER_LENGTH];
	static bool built[3];
	const int phase_count = 48000 / in_sample_rate;
	const int which = phase_count == 6 ? 0 : phase_count == 4 ? 1 : 2;
	if (!built[which]) {
		swr_build_filter(banks[which], phase_count);
		built[which] = true;
	}

	s->in_sample_rate = in_sample_rate;
	s->phase_count = phase_count;
	s->filter_bank = banks[which];
	s->index = -phase_count * ((SWR_FILTER_LENGTH - 1) / 2);
	s->in_buffer_index = SWR_FILTER_LENGTH;
	s->in_buffer_count = 0;
	s->flushed = 0;
}

static inline bool swr_is_initialized(const SwrContext* s) { return s->in_sample_rate != 0; }

static inline void swr_close(SwrContext* s) { s->in_sample_rate = 0; }

/** invert_initial_buffer: the samples before the first are its mirror image, once the filter's length and one are in. */
static void swr_invert_initial_buffer(SwrContext* s)
{
	if (s->index >= 0 || s->in_buffer_count < SWR_FILTER_LENGTH + 1) {
		return;
	}
	for (int ch = 0; ch < s->channels; ch++) {
		for (int n = 1; n <= SWR_FILTER_LENGTH; n++) {
			s->in_buffer[ch][SWR_FILTER_LENGTH - n] = s->in_buffer[ch][SWR_FILTER_LENGTH + n];
		}
	}
	while (s->index < 0) {
		--s->in_buffer_index;
		++s->in_buffer_count;
		s->index += s->phase_count;
	}
}

/** multiple_resample and resample_common_float: as many outputs as the buffer holds, at most `out_count`. */
static int swr_resample(SwrContext* s, float* const* out, int out_count)
{
	if (s->index < 0) {
		return 0;
	}
	const int phase_count = s->phase_count;
	const int end_index = (1 + s->in_buffer_count - SWR_FILTER_LENGTH) * phase_count;
	const int n = FFMAX(FFMIN(out_count, end_index - s->index), 0);
	if (!n) {
		return 0;
	}

	int index = 0, sample_index = 0;
	for (int ch = 0; ch < s->channels; ch++) {
		const float* src = s->in_buffer[ch] + s->in_buffer_index;
		float* dst = out[ch];
		index = s->index;
		sample_index = 0;
		for (int dst_index = 0; dst_index < n; dst_index++) {
			const float* filter = s->filter_bank + SWR_FILTER_LENGTH * index;
			float val = 0, val2 = 0;
			for (int i = 0; i + 1 < SWR_FILTER_LENGTH; i += 2) {
				val += src[sample_index + i] * filter[i];
				val2 += src[sample_index + i + 1] * filter[i + 1];
			}
			dst[dst_index] = val + val2;

			if (++index >= phase_count) {
				sample_index++;
				index -= phase_count;
			}
		}
	}
	s->index = index;
	s->in_buffer_index += sample_index;
	s->in_buffer_count -= sample_index;
	return n;
}

/** swr_convert with input: takes all of it, gives what it can up to `out_count`. */
static int swr_convert(SwrContext* s, float* const* out, int out_count, const float* const* in, int in_count)
{
	// Move what is left to the front. Before the inversion the buffer starts at 16 already.
	if (s->index >= 0 && s->in_buffer_index) {
		for (int ch = 0; ch < s->channels; ch++) {
			memmove(s->in_buffer[ch], s->in_buffer[ch] + s->in_buffer_index, s->in_buffer_count * sizeof(float));
		}
		s->in_buffer_index = 0;
	}
	for (int ch = 0; ch < s->channels; ch++) {
		memcpy(s->in_buffer[ch] + s->in_buffer_index + s->in_buffer_count, in[ch], in_count * sizeof(float));
	}
	s->in_buffer_count += in_count;

	swr_invert_initial_buffer(s);
	return swr_resample(s, out, out_count);
}

/** swr_convert with no input: resample_flush's mirror image at the end, then what is left. */
static int swr_flush(SwrContext* s, float* const* out, int out_count)
{
	if (!s->flushed) {
		const int reflection = (FFMIN(s->in_buffer_count, SWR_FILTER_LENGTH) + 1) / 2;
		const int end = s->in_buffer_index + s->in_buffer_count;
		for (int ch = 0; ch < s->channels; ch++) {
			for (int j = 0; j < reflection; j++) {
				s->in_buffer[ch][end + j] = s->in_buffer[ch][end - j - 1];
			}
		}
		s->in_buffer_count += reflection;
		s->flushed = 1;
	}
	return swr_resample(s, out, out_count);
}
