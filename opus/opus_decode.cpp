/*
	Opus audio decoder -- WASM C++.

	A port of FFmpeg n8.1's Opus decoder (LGPL-2.1+): libavcodec/opus/dec.c over rc.c (opus_rc.h), parse.c (opus_parse.h), silk.c
	(opus_silk.h), the CELT layer (opus_celt.h, opus_mdct.h), tab.c (opus_tables.h) and libswresample (opus_resample.h). One handle
	is one Opus stream as OpusHead describes it: one or more substreams, with planar float output at 48 kHz, each output channel
	named by opus_map_channel.

	A packet is one whole Opus packet: opus_decode_packet gives the samples its frames complete, as dec.c's opus_decode_packet does.
	It drops nothing: the pre-skip is the container's. A call with no packet, opus_flush, gives the samples the resampler still
	holds, as the NULL packet does at the end of a stream.

	Copyright as in the headers named above; dec.c is
	Copyright (c) 2012 Andrew D'Addesio, Copyright (c) 2013-2014 Mozilla Corporation.

	Departures from FFmpeg:
	  - The libavcodec framework (AVCodecContext, AVFrame, AVAudioFifo, the options) is gone. A frame that fails gives silence, as
	    without AV_EF_EXPLODE. Each substream decodes into buffers of its own and the output channels copy from them, so a channel
	    that nothing maps and a copied channel take no path of their own.
	  - A channel mapped to 255 is silent and nothing more. In dec.c its ChannelMap is zero, so it also takes stream 0's first output
	    pointer, and that stream's samples land in the silent channel: OBSERVED, a 6-channel file with one channel patched to 255
	    gives silence in FFmpeg's channel 0 too, and here what libopus gives.
	  - The sync buffer: a substream's samples are what its fifo held plus what it decoded, and what the shortest of them leaves
	    over goes back in the fifo. dec.c leaves the fifo out of that sum, so a substream with samples in the fifo loses them at the
	    end of the next packet and stays early from there on. OBSERVED with narrowband SILK (24 samples behind) beside CELT in one
	    file: FFmpeg's CELT channel is 24 samples early and the stream ends 24 short; here that channel equals libopus's at -137 dB.
	  - opus_reset clears redundancy_idx: dec.c's flush leaves a fade pending across a seek.

	-fwrapv is REQUIRED, as in FFmpeg's build.
*/

#include <cfloat>
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
#define ERROR_INVALID (-1)

#include "opus_silk.h"
#include "opus_resample.h"
#include "opus_mdct.h"
#include "opus_celt.h"

// Bytes this file zeroes past a packet's end: a bit read loads 4 past where it stops, itself 1 past the end.
#define INPUT_PADDING 16
// 120 ms, what a flush at a bandwidth change adds in front of it, and a sync buffer.
#define MAX_OUTPUT_SAMPLES (OPUS_MAX_PACKET_DUR + 240)
// The resampler holds a few dozen samples, so these are far past what a stream that decodes leaves.
#define MAX_SYNC_SAMPLES 120
#define MAX_CELT_DELAY_SAMPLES 960
// A CELT redundancy frame, 5 ms.
#define REDUNDANCY_FRAME_SAMPLES 240
// Mapping family 1 ends here.
#define MAX_OUTPUT_CHANNELS 8
// What a channel maps to for silence.
#define CHANNEL_SILENT 255

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
	/* number of decoded samples for this stream, its sync buffer's included */
	int decoded_samples;
	/* current output buffers for this stream */
	float* out[2];
	/* samples from this stream for synchronizing the streams when they have different resampling delays */
	float sync_buffer[2][MAX_SYNC_SAMPLES];
	int sync_size;

	OpusRangeCoder rc;
	OpusRangeCoder redundancy_rc;
	SilkContext silk;
	CeltFrame celt;

	float silk_buf[2][960];
	float* silk_output[2];
	float celt_buf[2][960];
	float* celt_output[2];

	float redundancy_buf[2][960];
	float* redundancy_output[2];

	/* buffers for the next samples to be decoded */
	float* cur_out[2];

	SwrContext swr;
	float celt_delay[2][MAX_CELT_DELAY_SAMPLES];
	int celt_delay_size;
	int silk_samplerate;
	/* number of samples we still want to get from the resampler */
	int delayed_samples;

	OpusPacket packet;

	int redundancy_idx;

	float output[2][MAX_OUTPUT_SAMPLES];
};

static int get_silk_samplerate(int config)
{
	if (config < 4)
		return 8000;
	else if (config < 8)
		return 12000;
	return 16000;
}

static void opus_fade(float* out, const float* in1, const float* in2, const float* window, int len)
{
	for (int i = 0; i < len; i++)
		out[i] = in2[i] * window[i] + in1[i] * (1.0 - window[i]);
}

static void vector_add(float* dst, const float* src, int len)
{
	for (int i = 0; i < len; i++)
		dst[i] += src[i];
}

static int opus_flush_resample(OpusStreamContext* s, int nb_samples)
{
	const int celt_size = s->celt_delay_size;
	const int ret = swr_flush(&s->swr, s->cur_out, nb_samples);
	if (ret != nb_samples) {
		return ERROR_INVALID;
	}

	if (celt_size) {
		if (celt_size != nb_samples) {
			return ERROR_INVALID;
		}
		s->celt_delay_size = 0;
		for (int i = 0; i < s->output_channels; i++) {
			vector_add(s->cur_out[i], s->celt_delay[i], nb_samples);
		}
	}

	if (s->redundancy_idx) {
		for (int i = 0; i < s->output_channels; i++)
			opus_fade(s->cur_out[i], s->cur_out[i],
			          s->redundancy_output[i] + 120 + s->redundancy_idx,
			          ff_celt_window2 + s->redundancy_idx, 120 - s->redundancy_idx);
		s->redundancy_idx = 0;
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

static int opus_decode_redundancy(OpusStreamContext* s, const uint8_t* data, int size)
{
	int ret = ff_opus_rc_dec_init(&s->redundancy_rc, data, size);
	if (ret < 0)
		return ret;
	ff_opus_rc_dec_raw_init(&s->redundancy_rc, data + size, size);

	return ff_celt_decode_frame(&s->celt, &s->redundancy_rc, s->redundancy_output,
	                            s->packet.stereo + 1, REDUNDANCY_FRAME_SAMPLES,
	                            0, ff_celt_band_end[s->packet.bandwidth]);
}

static int opus_decode_frame(OpusStreamContext* s, const uint8_t* data, int size)
{
	int samples = s->packet.frame_duration;
	int redundancy = 0;
	int redundancy_size = 0, redundancy_pos = 0;
	int ret, consumed;
	const int delayed_samples = s->delayed_samples;

	ret = ff_opus_rc_dec_init(&s->rc, data, size);
	if (ret < 0)
		return ret;

	/* decode the silk frame */
	if (s->packet.mode == OPUS_MODE_SILK || s->packet.mode == OPUS_MODE_HYBRID) {
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
	} else {
		ff_silk_flush(&s->silk);
	}

	// decode redundancy information
	consumed = opus_rc_tell(&s->rc);
	if (s->packet.mode == OPUS_MODE_HYBRID && consumed + 37 <= size * 8)
		redundancy = ff_opus_rc_dec_log(&s->rc, 12);
	else if (s->packet.mode == OPUS_MODE_SILK && consumed + 17 <= size * 8)
		redundancy = 1;

	if (redundancy) {
		redundancy_pos = ff_opus_rc_dec_log(&s->rc, 1);

		if (s->packet.mode == OPUS_MODE_HYBRID)
			redundancy_size = ff_opus_rc_dec_uint(&s->rc, 256) + 2;
		else
			redundancy_size = size - (consumed + 7) / 8;
		size -= redundancy_size;
		if (size < 0) {
			return ERROR_INVALID;
		}

		if (redundancy_pos) {
			ret = opus_decode_redundancy(s, data + size, redundancy_size);
			if (ret < 0)
				return ret;
			ff_celt_flush(&s->celt);
		}
	}

	/* decode the CELT frame */
	if (s->packet.mode == OPUS_MODE_CELT || s->packet.mode == OPUS_MODE_HYBRID) {
		float* out_tmp[2] = { s->cur_out[0], s->cur_out[1] };
		float** dst = (s->packet.mode == OPUS_MODE_CELT) ? out_tmp : s->celt_output;
		int celt_output_samples = samples;
		const int delay_samples = s->celt_delay_size;

		if (delay_samples) {
			s->celt_delay_size = 0;
			if (s->packet.mode == OPUS_MODE_HYBRID) {
				for (int i = 0; i < s->output_channels; i++) {
					vector_add(out_tmp[i], s->celt_delay[i], delay_samples);
					out_tmp[i] += delay_samples;
				}
				celt_output_samples -= delay_samples;
			}
		}

		ff_opus_rc_dec_raw_init(&s->rc, data + size, size);

		ret = ff_celt_decode_frame(&s->celt, &s->rc, dst,
		                           s->packet.stereo + 1,
		                           s->packet.frame_duration,
		                           (s->packet.mode == OPUS_MODE_HYBRID) ? 17 : 0,
		                           ff_celt_band_end[s->packet.bandwidth]);
		if (ret < 0)
			return ret;

		if (s->packet.mode == OPUS_MODE_HYBRID) {
			const int celt_delay = s->packet.frame_duration - celt_output_samples;
			if (celt_output_samples < 0 || celt_delay > MAX_CELT_DELAY_SAMPLES)
				return ERROR_INVALID;

			for (int i = 0; i < s->output_channels; i++) {
				vector_add(out_tmp[i], s->celt_output[i], celt_output_samples);
				memcpy(s->celt_delay[i], s->celt_output[i] + celt_output_samples, celt_delay * sizeof(float));
			}
			s->celt_delay_size = celt_delay;
		}
	} else {
		ff_celt_flush(&s->celt);
	}

	if (s->redundancy_idx) {
		for (int i = 0; i < s->output_channels; i++)
			opus_fade(s->cur_out[i], s->cur_out[i],
			          s->redundancy_output[i] + 120 + s->redundancy_idx,
			          ff_celt_window2 + s->redundancy_idx, 120 - s->redundancy_idx);
		s->redundancy_idx = 0;
	}
	if (redundancy) {
		if (!redundancy_pos) {
			ff_celt_flush(&s->celt);
			ret = opus_decode_redundancy(s, data + size, redundancy_size);
			if (ret < 0)
				return ret;

			for (int i = 0; i < s->output_channels; i++) {
				opus_fade(s->cur_out[i] + samples - 120 + delayed_samples,
				          s->cur_out[i] + samples - 120 + delayed_samples,
				          s->redundancy_output[i] + 120,
				          ff_celt_window2, 120 - delayed_samples);
				if (delayed_samples)
					s->redundancy_idx = 120 - delayed_samples;
			}
		} else {
			for (int i = 0; i < s->output_channels; i++) {
				memcpy(s->cur_out[i] + delayed_samples, s->redundancy_output[i], 120 * sizeof(float));
				opus_fade(s->cur_out[i] + 120 + delayed_samples,
				          s->redundancy_output[i] + 120,
				          s->cur_out[i] + 120 + delayed_samples,
				          ff_celt_window2, 120);
			}
		}
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

struct ChannelMap {
	int stream_idx;
	int channel_idx;
	int silence;
};

struct OpusHandle {
	int channels;
	int nb_streams;
	int nb_stereo_streams;
	float gain;		// 0 for none
	ChannelMap channel_maps[MAX_OUTPUT_CHANNELS];
	OpusStreamContext* streams;
	float output[MAX_OUTPUT_CHANNELS][MAX_OUTPUT_SAMPLES];
};

/** dec.c's opus_decode_packet: `buf` is null at the end of the stream. */
static int opus_decode_multistream(OpusHandle* c, const uint8_t* buf, int buf_size)
{
	int coded_samples = 0;
	int decoded_samples = INT_MAX;
	int delayed_samples = 0;

	/* calculate the number of delayed samples */
	for (int i = 0; i < c->nb_streams; i++) {
		const OpusStreamContext* s = &c->streams[i];
		delayed_samples = FFMAX(delayed_samples, s->delayed_samples + s->sync_size);
	}

	/* decode the header of the first sub-packet to find out the sample count */
	if (buf) {
		OpusPacket* pkt = &c->streams[0].packet;
		const int ret = ff_opus_parse_packet(pkt, buf, buf_size, c->nb_streams > 1);
		if (ret < 0) {
			return ret;
		}
		coded_samples += pkt->frame_count * pkt->frame_duration;
		c->streams[0].silk_samplerate = get_silk_samplerate(pkt->config);
	}

	/* no input or buffered data => nothing to do */
	if (!(coded_samples + delayed_samples)) {
		return 0;
	}
	if (coded_samples + delayed_samples > MAX_OUTPUT_SAMPLES) {
		return ERROR_INVALID;
	}

	/* read the data from the sync buffers */
	for (int i = 0; i < c->nb_streams; i++) {
		OpusStreamContext* s = &c->streams[i];
		for (int j = 0; j < s->output_channels; j++) {
			memcpy(s->output[j], s->sync_buffer[j], s->sync_size * sizeof(float));
			s->out[j] = s->output[j] + s->sync_size;
		}
	}

	/* decode each sub-packet */
	for (int i = 0; i < c->nb_streams; i++) {
		OpusStreamContext* s = &c->streams[i];

		if (i && buf) {
			const int ret = ff_opus_parse_packet(&s->packet, buf, buf_size, i != c->nb_streams - 1);
			if (ret < 0) {
				return ret;
			}
			if (coded_samples != s->packet.frame_count * s->packet.frame_duration) {
				return ERROR_INVALID;
			}

			s->silk_samplerate = get_silk_samplerate(s->packet.config);
		}

		const int ret = opus_decode_subpacket(s, buf);
		if (ret < 0) {
			return ret;
		}
		s->decoded_samples = s->sync_size + ret;
		decoded_samples = FFMIN(decoded_samples, s->decoded_samples);

		if (!buf)
			continue;

		buf += s->packet.packet_size;
		buf_size -= s->packet.packet_size;
	}

	/* buffer the extra samples */
	for (int i = 0; i < c->nb_streams; i++) {
		OpusStreamContext* s = &c->streams[i];
		const int buffer_samples = s->decoded_samples - decoded_samples;
		if (buffer_samples > MAX_SYNC_SAMPLES) {
			return ERROR_INVALID;
		}
		for (int j = 0; j < s->output_channels; j++) {
			memcpy(s->sync_buffer[j], s->output[j] + decoded_samples, buffer_samples * sizeof(float));
		}
		s->sync_size = buffer_samples;
	}

	for (int i = 0; i < c->channels; i++) {
		const ChannelMap* map = &c->channel_maps[i];
		if (map->silence) {
			memset(c->output[i], 0, decoded_samples * sizeof(float));
		} else {
			memcpy(c->output[i], c->streams[map->stream_idx].output[map->channel_idx], decoded_samples * sizeof(float));
		}
		if (c->gain) {
			for (int j = 0; j < decoded_samples; j++) {
				c->output[i][j] *= c->gain;
			}
		}
	}

	return decoded_samples;
}

extern "C" {

/**
 * @param channels what opus_channel_ptr gives, up to MAX_OUTPUT_CHANNELS; each maps as in family 0 until opus_map_channel
 * @param streams the substreams of a packet, at least 1
 * @param stereo_streams how many of them, the first, are stereo
 * @param gain_q8 the output gain in dB, Q7.8, signed
 */
EMSCRIPTEN_KEEPALIVE OpusHandle* opus_open(int channels, int streams, int stereo_streams, int gain_q8)
{
	if (channels < 1 || channels > MAX_OUTPUT_CHANNELS || streams < 1 || stereo_streams > streams) {
		return nullptr;
	}
	OpusHandle* c = (OpusHandle*)calloc(1, sizeof(OpusHandle));
	if (!c) {
		return nullptr;
	}
	c->streams = (OpusStreamContext*)calloc(streams, sizeof(OpusStreamContext));
	if (!c->streams) {
		free(c);
		return nullptr;
	}
	c->channels = channels;
	c->nb_streams = streams;
	c->nb_stereo_streams = stereo_streams;
	if (gain_q8) {
		c->gain = pow(10.0, gain_q8 / (20.0 * 256));
	}

	for (int i = 0; i < channels; i++) {
		c->channel_maps[i].channel_idx = i % 2;
	}
	for (int i = 0; i < streams; i++) {
		OpusStreamContext* s = &c->streams[i];
		s->output_channels = i < stereo_streams ? 2 : 1;
		s->swr.channels = s->output_channels;
		for (int j = 0; j < 2; j++) {
			s->silk_output[j] = s->silk_buf[j];
			s->celt_output[j] = s->celt_buf[j];
			s->redundancy_output[j] = s->redundancy_buf[j];
			s->out[j] = s->output[j];
		}
		ff_silk_init(&s->silk, s->output_channels);
		ff_celt_init(&s->celt, s->output_channels, 1);
	}
	return c;
}

/** parse.c's channel map: output `channel` takes `index` of RFC 7845's mapping table. False for an index past the streams. */
EMSCRIPTEN_KEEPALIVE int opus_map_channel(OpusHandle* c, int channel, int index)
{
	ChannelMap* map = &c->channel_maps[channel];
	map->silence = index == CHANNEL_SILENT;
	if (map->silence) {
		return 1;
	}
	if (index >= c->nb_streams + c->nb_stereo_streams) {
		return 0;
	}
	if (index < 2 * c->nb_stereo_streams) {
		map->stream_idx = index / 2;
		map->channel_idx = index & 1;
	} else {
		map->stream_idx = index - c->nb_stereo_streams;
		map->channel_idx = 0;
	}
	return 1;
}

EMSCRIPTEN_KEEPALIVE void opus_close(OpusHandle* c)
{
	if (c) {
		free(c->streams);
	}
	free(c);
}

/** dec.c's opus_decode_flush: for a seek. */
EMSCRIPTEN_KEEPALIVE void opus_reset(OpusHandle* c)
{
	for (int i = 0; i < c->nb_streams; i++) {
		OpusStreamContext* s = &c->streams[i];
		memset(&s->packet, 0, sizeof(s->packet));
		s->delayed_samples = 0;
		s->celt_delay_size = 0;
		s->sync_size = 0;
		s->redundancy_idx = 0;
		swr_close(&s->swr);
		ff_silk_flush(&s->silk);
		ff_celt_flush(&s->celt);
	}
}

/** The samples of the whole packet at `data`, 0 for one that does not parse, which gives none. */
EMSCRIPTEN_KEEPALIVE int opus_packet_samples(OpusHandle* c, const uint8_t* data, int size)
{
	OpusPacket packet;
	if (ff_opus_parse_packet(&packet, data, size, c->nb_streams > 1) < 0) {
		return 0;
	}
	return packet.frame_count * packet.frame_duration;
}

/**
 * Decode ONE whole packet at `data`, which has INPUT_PADDING writable bytes past `size`.
 * @returns the samples per channel at opus_channel_ptr, or a negative error.
 */
EMSCRIPTEN_KEEPALIVE int opus_decode_packet(OpusHandle* c, uint8_t* data, int size)
{
	memset(data + size, 0, INPUT_PADDING);
	return opus_decode_multistream(c, data, size);
}

/** What the resampler still holds at the end of the stream: the samples per channel at opus_channel_ptr. */
EMSCRIPTEN_KEEPALIVE int opus_flush(OpusHandle* c)
{
	return opus_decode_multistream(c, nullptr, 0);
}

EMSCRIPTEN_KEEPALIVE float* opus_channel_ptr(OpusHandle* c, int channel)
{
	return c->output[channel];
}

}
