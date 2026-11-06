/*
	Opus audio decoder -- WASM C++.

	A port of FFmpeg n8.1's Opus decoder (LGPL-2.1+): libavcodec/opus/dec.c over rc.c (opus_rc.h), parse.c (opus_parse.h), silk.c
	(opus_silk.h), tab.c (opus_tables.h) and libswresample (opus_resample.h). One handle is one Opus stream, 1 or 2 channels, with
	planar float output at 48 kHz.

	A packet is one whole Opus packet: opus_decode_packet gives the samples its frames complete, as dec.c's opus_decode_packet does.
	It drops nothing: the pre-skip is the container's. A call with no packet, opus_flush, gives the samples the resampler still
	holds, as the NULL packet does at the end of a stream.

	So far: SILK-only packets. Copyright as in the headers named above; dec.c is
	Copyright (c) 2012 Andrew D'Addesio, Copyright (c) 2013-2014 Mozilla Corporation.

	Departures from FFmpeg:
	  - A hybrid or CELT packet, and a SILK frame with a redundancy frame in its tail, refuse with OPUS_ERROR_UNSUPPORTED before
	    or at that frame.
	  - The libavcodec framework (AVCodecContext, AVFrame, AVAudioFifo, the options) is gone. A frame that fails gives silence, as
	    without AV_EF_EXPLODE.

	-fwrapv is REQUIRED, as in FFmpeg's build.
*/

#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include <emscripten/emscripten.h>

#define FFMIN(a, b) ((a) > (b) ? (b) : (a))
#define FFMAX(a, b) ((a) > (b) ? (a) : (b))
#define FFABS(a) ((a) >= 0 ? (a) : (-(a)))
#define MUL64(a, b) ((int64_t)(a) * (int64_t)(b))
#define MULL(a, b, s) (MUL64(a, b) >> (s))

static inline int MULH(int a, int b) { return MUL64(a, b) >> 32; }
static inline int av_log2(unsigned v) { return v ? 31 - __builtin_clz(v) : 0; }
static inline float av_clipf(float a, float amin, float amax) { return a < amin ? amin : a > amax ? amax : a; }
static inline int av_clip(int a, int amin, int amax) { return a < amin ? amin : a > amax ? amax : a; }
static inline int16_t av_clip_int16(int a) { return a < -32768 ? -32768 : a > 32767 ? 32767 : a; }
static inline unsigned av_clip_uintp2(int a, int p)
{
	if (a & ~((1 << p) - 1)) return (~a) >> 31 & ((1 << p) - 1);
	return a;
}
static inline int av_sat_sub32(int a, int b)
{
	const int64_t d = (int64_t)a - b;
	return d < INT32_MIN ? INT32_MIN : d > INT32_MAX ? INT32_MAX : (int)d;
}

#include "opus_tables.h"
#include "opus_parse.h"
#include "opus_rc.h"
#include "opus_silk.h"
#include "opus_resample.h"

#define ERROR_INVALID (-1)
#define OPUS_ERROR_UNSUPPORTED (-2)
// Bytes this file zeroes past a packet's end: a bit read loads 4 past where it stops, itself 1 past the end.
#define INPUT_PADDING 16
// 120 ms and what a flush at a bandwidth change adds in front of it.
#define MAX_OUTPUT_SAMPLES (OPUS_MAX_PACKET_DUR + 120)

static const uint16_t silk_frame_duration_ms[16] = {
	10, 20, 40, 60,
	10, 20, 40, 60,
	10, 20, 40, 60,
	10, 20,
	10, 20,
};

/* number of samples of silence to feed to the resampler at the beginning */
static const int silk_resample_delay[] = {
	4, 8, 11, 11, 11
};

struct OpusStreamContext {
	int output_channels;
	/* current output buffers for this stream */
	float* out[2];

	OpusRangeCoder rc;
	SilkContext silk;

	float silk_buf[2][960];
	float* silk_output[2];

	/* buffers for the next samples to be decoded */
	float* cur_out[2];

	SwrContext swr;
	int silk_samplerate;
	/* number of samples we still want to get from the resampler */
	int delayed_samples;

	OpusPacket packet;
};

static int get_silk_samplerate(int config)
{
	if (config < 4)
		return 8000;
	else if (config < 8)
		return 12000;
	return 16000;
}

static int opus_flush_resample(OpusStreamContext* s, int nb_samples)
{
	const int ret = swr_flush(&s->swr, s->cur_out, nb_samples);
	if (ret != nb_samples) {
		return ERROR_INVALID;
	}

	s->cur_out[0] += nb_samples;
	s->cur_out[1] += nb_samples;

	return 0;
}

static void opus_init_resample(OpusStreamContext* s)
{
	static const float delay[16] = { 0.0 };
	const float* delayptr[2] = { delay, delay };

	swr_init(&s->swr, s->silk_samplerate);
	swr_convert(&s->swr, nullptr, 0, delayptr, silk_resample_delay[s->packet.bandwidth]);
}

static int opus_decode_frame(OpusStreamContext* s, const uint8_t* data, int size)
{
	int samples = s->packet.frame_duration;
	int consumed;

	ff_opus_rc_dec_init(&s->rc, data, size);

	/* decode the silk frame */
	if (!swr_is_initialized(&s->swr)) {
		opus_init_resample(s);
	}

	samples = ff_silk_decode_superframe(&s->silk, &s->rc, s->silk_output,
	                                    (OpusBandwidth)FFMIN((int)s->packet.bandwidth, (int)OPUS_BANDWIDTH_WIDEBAND),
	                                    s->packet.stereo + 1,
	                                    silk_frame_duration_ms[s->packet.config]);
	if (samples < 0) {
		return samples;
	}
	samples = swr_convert(&s->swr, s->cur_out, s->packet.frame_duration, s->silk_output, samples);
	s->delayed_samples += s->packet.frame_duration - samples;

	// decode redundancy information
	consumed = opus_rc_tell(&s->rc);
	if (consumed + 17 <= size * 8) {
		return OPUS_ERROR_UNSUPPORTED;
	}

	return samples;
}

/** dec.c's opus_decode_subpacket: `buf` is null at the end of the stream. */
static int opus_decode_subpacket(OpusStreamContext* s, const uint8_t* buf)
{
	int output_samples = 0;
	int flush_needed = 0;

	s->cur_out[0] = s->out[0];
	s->cur_out[1] = s->out[1];

	/* check if we need to flush the resampler */
	if (swr_is_initialized(&s->swr)) {
		if (buf) {
			flush_needed = (s->packet.mode == OPUS_MODE_CELT) || (s->swr.in_sample_rate != s->silk_samplerate);
		} else {
			flush_needed = !!s->delayed_samples;
		}
	}

	if (!buf && !flush_needed)
		return 0;

	/* flush the resampler if necessary */
	if (flush_needed) {
		const int ret = opus_flush_resample(s, s->delayed_samples);
		if (ret < 0) {
			return ret;
		}
		swr_close(&s->swr);
		output_samples += s->delayed_samples;
		s->delayed_samples = 0;

		if (!buf)
			return output_samples;
	}

	/* decode all the frames in the packet */
	for (int i = 0; i < s->packet.frame_count; i++) {
		const int size = s->packet.frame_size[i];
		int samples = opus_decode_frame(s, buf + s->packet.frame_offset[i], size);

		if (samples == OPUS_ERROR_UNSUPPORTED) {
			return samples;
		}
		if (samples < 0) {
			for (int j = 0; j < s->output_channels; j++)
				memset(s->cur_out[j], 0, s->packet.frame_duration * sizeof(float));
			samples = s->packet.frame_duration;
		}
		output_samples += samples;

		for (int j = 0; j < s->output_channels; j++)
			s->cur_out[j] += samples;
	}

	return output_samples;
}

struct OpusHandle {
	OpusStreamContext stream;
	float output[2][MAX_OUTPUT_SAMPLES];
};

extern "C" {

EMSCRIPTEN_KEEPALIVE OpusHandle* opus_open(int channels)
{
	OpusHandle* handle = (OpusHandle*)calloc(1, sizeof(OpusHandle));
	if (!handle) {
		return nullptr;
	}
	OpusStreamContext* s = &handle->stream;
	s->output_channels = channels;
	s->swr.channels = channels;
	for (int j = 0; j < 2; j++) {
		s->silk_output[j] = s->silk_buf[j];
		s->out[j] = handle->output[j];
	}
	ff_silk_init(&s->silk, channels);
	return handle;
}

EMSCRIPTEN_KEEPALIVE void opus_close(OpusHandle* handle)
{
	free(handle);
}

/** dec.c's opus_decode_flush: for a seek. */
EMSCRIPTEN_KEEPALIVE void opus_reset(OpusHandle* handle)
{
	OpusStreamContext* s = &handle->stream;
	memset(&s->packet, 0, sizeof(s->packet));
	s->delayed_samples = 0;
	swr_close(&s->swr);
	ff_silk_flush(&s->silk);
}

/**
 * Decode ONE whole packet at `data`, which has INPUT_PADDING writable bytes past `size`.
 * @returns the samples per channel at opus_channel_ptr, or a negative error.
 */
EMSCRIPTEN_KEEPALIVE int opus_decode_packet(OpusHandle* handle, uint8_t* data, int size)
{
	OpusStreamContext* s = &handle->stream;
	memset(data + size, 0, INPUT_PADDING);

	if (ff_opus_parse_packet(&s->packet, data, size, 0) < 0) {
		return ERROR_INVALID;
	}
	if (s->packet.mode != OPUS_MODE_SILK) {
		return OPUS_ERROR_UNSUPPORTED;
	}
	s->silk_samplerate = get_silk_samplerate(s->packet.config);
	return opus_decode_subpacket(s, data);
}

/** What the resampler still holds at the end of the stream: the samples per channel at opus_channel_ptr. */
EMSCRIPTEN_KEEPALIVE int opus_flush(OpusHandle* handle)
{
	return opus_decode_subpacket(&handle->stream, nullptr);
}

EMSCRIPTEN_KEEPALIVE float* opus_channel_ptr(OpusHandle* handle, int channel)
{
	return handle->output[channel];
}

}
