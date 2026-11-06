/*
	WAVE audio decoders -- WASM C++. One module for the block- or byte-framed WAVE format tags:
	  0x0002 MS ADPCM, 0x0011 IMA ADPCM (the WAV form), 0x0006 A-law, 0x0007 mu-law,
	and QuickTime's 'ima4' (WA_IMA_QT), which has no WAVE tag.

	A direct port of FFmpeg n8.1's libavcodec/adpcm.c, adpcm_data.c, adpcm_data.h, adpcm.h,
	pcm.c and pcm_tablegen.h: get_nb_samples, the ADPCM_MS, ADPCM_IMA_WAV and ADPCM_IMA_QT cases of
	adpcm_decode_frame, their expand_nibble functions, alaw2linear and ulaw2linear.

	Copyright (c) 2001-2003 The FFmpeg project, and the authors those files name. This file is
	free software; you can redistribute it and/or modify it under the terms of the GNU Lesser
	General Public License as published by the Free Software Foundation; either version 2.1 of
	the License, or (at your option) any later version. It is distributed WITHOUT ANY WARRANTY;
	without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
	See the GNU Lesser General Public License for more details.

	The libavcodec framework is replaced by a handle, WaveAudioCtx. FFmpeg decodes one block of a
	packet per call and returns what it consumed; here one call is one block, its framing
	waveaudio_decode_core.js's. As with AV_INPUT_BUFFER_PADDING_SIZE, the input is zero-padded: a
	short last block's header fields read as FFmpeg's unchecked readers read them.

	Boundary (extern "C", prefixed _ on the JS side; see waveaudio_decode_core.js):
	  WaveAudioCtx* wa_open(int codec, int channels, int bitsPerSample, int blockBytes)
	             null for what FFmpeg's init refuses
	  void     wa_close(WaveAudioCtx*)
	  void     wa_reset(WaveAudioCtx*)
	  uint8_t* wa_input_ptr(WaveAudioCtx*)        room for blockBytes
	  int      wa_decode(WaveAudioCtx*, int size) decode the block at wa_input_ptr: the samples per
	             channel, 0 for no samples or a corrupt block
	  float*   wa_channel_ptr(WaveAudioCtx*, int ch)
*/

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <emscripten/emscripten.h>

#include "waveaudio_tables.h"

enum {
	WA_MS = 0x0002,
	WA_ALAW = 0x0006,
	WA_MULAW = 0x0007,
	WA_IMA_WAV = 0x0011,
	WA_IMA_QT = 0x10000,	// past any WAVE tag
};

static const int MAX_ADPCM_CHANNELS = 6;
static const int INPUT_PADDING = 64;

struct ChannelStatus
{
	int predictor;
	int16_t step_index;
	int sample1;
	int sample2;
	int coeff1;
	int coeff2;
	int idelta;
};

struct WaveAudioCtx
{
	int codec;
	int channels;
	int bits;
	int blockBytes;
	int capacity;		// samples per channel
	uint8_t* input;
	int16_t* pcm;		// planar, capacity per channel
	float* output;		// the same
	int16_t lut[256];
	ChannelStatus status[MAX_ADPCM_CHANNELS];
};

static inline int clip(int value, int low, int high) { return value < low ? low : value > high ? high : value; }
static inline int clip_int16(int value) { return clip(value, -32768, 32767); }
static inline int get_le16s(const uint8_t* p) { return (int16_t)(p[0] | (p[1] << 8)); }
static inline int get_be16s(const uint8_t* p) { return (int16_t)((p[0] << 8) | p[1]); }

// pcm_tablegen.h, from g711.c by Sun Microsystems (unrestricted use)
static int alaw2linear(unsigned char a_val)
{
	a_val ^= 0x55;
	int t = a_val & 0xF;
	const int seg = ((unsigned)a_val & 0x70) >> 4;
	if (seg) t = (t + t + 1 + 32) << (seg + 2);
	else t = (t + t + 1) << 3;
	return (a_val & 0x80) ? t : -t;
}

static int ulaw2linear(unsigned char u_val)
{
	u_val = ~u_val;
	int t = ((u_val & 0xF) << 3) + 0x84;
	t <<= ((unsigned)u_val & 0x70) >> 4;
	return (u_val & 0x80) ? (0x84 - t) : (t - 0x84);
}

static inline int16_t adpcm_ima_qt_expand_nibble(ChannelStatus* c, int nibble)
{
	const int step = adpcm_step_table[c->step_index];
	const int step_index = clip(c->step_index + adpcm_index_table[nibble], 0, 88);

	int diff = step >> 3;
	if (nibble & 4) diff += step;
	if (nibble & 2) diff += step >> 1;
	if (nibble & 1) diff += step >> 2;

	const int predictor = (nibble & 8) ? c->predictor - diff : c->predictor + diff;
	c->predictor = clip_int16(predictor);
	c->step_index = step_index;
	return c->predictor;
}

/** get_bits_le over the run in `temp`, at bit `*bit`, for bits below 8. */
static inline int get_bits_le(const uint8_t* temp, int* bit, int bits)
{
	const int at = *bit;
	*bit = at + bits;
	const int word = temp[at >> 3] | (temp[(at >> 3) + 1] << 8);
	return (word >> (at & 7)) & ((1 << bits) - 1);
}

static inline int16_t adpcm_ima_wav_expand_nibble(ChannelStatus* c, const uint8_t* temp, int* bit, int bps)
{
	const int shift = bps - 1;
	const int nibble = get_bits_le(temp, bit, bps);
	const int step = adpcm_step_table[c->step_index];
	const int step_index = clip(c->step_index + adpcm_index_tables[bps - 2][nibble], 0, 88);

	const int sign = nibble & (1 << shift);
	const int delta = nibble & ((1 << shift) - 1);
	int diff = step >> shift;
	for (int i = 0; i < shift; i++) {
		diff += (step >> (shift - 1 - i)) * !!(delta & (1 << i));
	}
	int predictor = c->predictor;
	if (sign) predictor -= diff;
	else predictor += diff;

	c->predictor = clip_int16(predictor);
	c->step_index = step_index;
	return c->predictor;
}

static inline int16_t adpcm_ms_expand_nibble(ChannelStatus* c, int nibble)
{
	int predictor = ((c->sample1 * c->coeff1) + (c->sample2 * c->coeff2)) / 64;
	predictor += ((nibble & 0x08) ? (nibble - 0x10) : nibble) * c->idelta;

	c->sample2 = c->sample1;
	c->sample1 = clip_int16(predictor);
	c->idelta = (adpcm_AdaptationTable[nibble] * c->idelta) >> 8;
	if (c->idelta < 16) c->idelta = 16;
	if (c->idelta > INT_MAX / 768) c->idelta = INT_MAX / 768;
	return c->sample1;
}

/** get_nb_samples: the samples per channel of a block of `size` bytes, 0 or less for none. */
static int get_nb_samples(const WaveAudioCtx* c, int size)
{
	const int ch = c->channels;
	switch (c->codec) {
	case WA_IMA_QT:
		return size < 34 * ch ? 0 : 64;
	case WA_IMA_WAV: {
		if (size > c->blockBytes) size = c->blockBytes;
		if (size < 4 * ch) return 0;
		const int bsize = adpcm_ima_block_sizes[c->bits - 2];
		const int bsamples = adpcm_ima_block_samples[c->bits - 2];
		return 1 + (size - 4 * ch) / (bsize * ch) * bsamples;
	}
	case WA_MS:
		if (size > c->blockBytes) size = c->blockBytes;
		return (size - 6 * ch) * 2 / ch;
	default:
		return size / ch;
	}
}

static int decode_ima_qt(WaveAudioCtx* c)
{
	const uint8_t* in = c->input;
	for (int channel = 0; channel < c->channels; channel++) {
		ChannelStatus* cs = &c->status[channel];
		int predictor = get_be16s(in);
		in += 2;
		const int step_index = predictor & 0x7F;
		predictor &= ~0x7F;

		bool update = true;
		if (cs->step_index == step_index) {
			int diff = predictor - cs->predictor;
			if (diff < 0) diff = -diff;
			update = diff > 0x7f;
		}
		if (update) {
			cs->step_index = step_index;
			cs->predictor = predictor;
		}
		if ((unsigned)cs->step_index > 88u) return 0;

		int16_t* samples = c->pcm + channel * c->capacity;
		for (int m = 0; m < 64; m += 2) {
			const int byte = *in++;
			samples[m] = adpcm_ima_qt_expand_nibble(cs, byte & 0x0F);
			samples[m + 1] = adpcm_ima_qt_expand_nibble(cs, byte >> 4);
		}
	}
	return 64;
}

static int decode_ima_wav(WaveAudioCtx* c, int nb_samples)
{
	const int channels = c->channels;
	const uint8_t* buf = c->input;
	const uint8_t* in = buf;
	for (int i = 0; i < channels; i++) {
		ChannelStatus* cs = &c->status[i];
		cs->predictor = c->pcm[i * c->capacity] = get_le16s(in);
		cs->step_index = in[2];
		in += 4;
		if ((unsigned)cs->step_index > 88u) return 0;
	}

	if (c->bits != 4) {
		const int samples_per_block = adpcm_ima_block_samples[c->bits - 2];
		const int block_size = adpcm_ima_block_sizes[c->bits - 2];
		uint8_t temp[20 + 2] = { 0 };

		for (int n = 0; n < (nb_samples - 1) / samples_per_block; n++) {
			for (int i = 0; i < channels; i++) {
				ChannelStatus* cs = &c->status[i];
				int16_t* samples = c->pcm + i * c->capacity + 1 + n * samples_per_block;
				for (int j = 0; j < block_size; j++) {
					temp[j] = buf[4 * channels + block_size * n * channels +
						(j % 4) + (j / 4) * (channels * 4) + i * 4];
				}
				int bit = 0;
				for (int m = 0; m < samples_per_block; m++) {
					samples[m] = adpcm_ima_wav_expand_nibble(cs, temp, &bit, c->bits);
				}
			}
		}
	} else {
		for (int n = 0; n < (nb_samples - 1) / 8; n++) {
			for (int i = 0; i < channels; i++) {
				ChannelStatus* cs = &c->status[i];
				int16_t* samples = c->pcm + i * c->capacity + 1 + n * 8;
				for (int m = 0; m < 8; m += 2) {
					const int v = *in++;
					samples[m] = adpcm_ima_qt_expand_nibble(cs, v & 0x0F);
					samples[m + 1] = adpcm_ima_qt_expand_nibble(cs, v >> 4);
				}
			}
		}
	}
	return nb_samples;
}

static int decode_ms(WaveAudioCtx* c, int nb_samples)
{
	const uint8_t* in = c->input;
	const int channels = c->channels;

	if (channels > 2) {
		for (int channel = 0; channel < channels; channel++) {
			ChannelStatus* cs = &c->status[channel];
			int16_t* samples = c->pcm + channel * c->capacity;
			const int block_predictor = *in++;
			if (block_predictor > 6) return 0;
			cs->coeff1 = adpcm_AdaptCoeff1[block_predictor];
			cs->coeff2 = adpcm_AdaptCoeff2[block_predictor];
			cs->idelta = get_le16s(in);
			cs->sample1 = get_le16s(in + 2);
			cs->sample2 = get_le16s(in + 4);
			in += 6;
			*samples++ = cs->sample2;
			*samples++ = cs->sample1;
			for (int n = (nb_samples - 2) >> 1; n > 0; n--) {
				const int byte = *in++;
				*samples++ = adpcm_ms_expand_nibble(cs, byte >> 4);
				*samples++ = adpcm_ms_expand_nibble(cs, byte & 0x0F);
			}
		}
		return nb_samples;
	}

	const int st = channels == 2 ? 1 : 0;
	ChannelStatus* c0 = &c->status[0];
	ChannelStatus* c1 = &c->status[st];

	for (int channel = 0; channel <= st; channel++) {
		const int block_predictor = *in++;
		if (block_predictor > 6) return 0;
		c->status[channel].coeff1 = adpcm_AdaptCoeff1[block_predictor];
		c->status[channel].coeff2 = adpcm_AdaptCoeff2[block_predictor];
	}
	for (int channel = 0; channel <= st; channel++, in += 2) c->status[channel].idelta = get_le16s(in);
	for (int channel = 0; channel <= st; channel++, in += 2) c->status[channel].sample1 = get_le16s(in);
	for (int channel = 0; channel <= st; channel++, in += 2) c->status[channel].sample2 = get_le16s(in);

	int16_t* s0 = c->pcm;
	int16_t* s1 = c->pcm + st * c->capacity;
	*s0++ = c0->sample2;
	if (st) *s1++ = c1->sample2;
	*s0++ = c0->sample1;
	if (st) *s1++ = c1->sample1;
	// Mono: both nibbles of a byte follow each other in one channel. Stereo: one to each.
	for (int n = (nb_samples - 2) >> (1 - st); n > 0; n--) {
		const int byte = *in++;
		*s0++ = adpcm_ms_expand_nibble(c0, byte >> 4);
		if (st) *s1++ = adpcm_ms_expand_nibble(c1, byte & 0x0F);
		else *s0++ = adpcm_ms_expand_nibble(c0, byte & 0x0F);
	}
	return nb_samples;
}

static int decode_law(WaveAudioCtx* c, int nb_samples)
{
	const uint8_t* in = c->input;
	for (int i = 0; i < nb_samples; i++) {
		for (int channel = 0; channel < c->channels; channel++) {
			c->pcm[channel * c->capacity + i] = c->lut[*in++];
		}
	}
	return nb_samples;
}

extern "C" {

EMSCRIPTEN_KEEPALIVE
WaveAudioCtx* wa_open(int codec, int channels, int bits, int blockBytes)
{
	switch (codec) {
	case WA_IMA_WAV:
		if (channels < 1 || channels > 2 || bits < 2 || bits > 5) return nullptr;
		break;
	case WA_IMA_QT:
		if (channels < 1 || channels > 2) return nullptr;
		break;
	case WA_MS:
		if (channels < 1 || channels > MAX_ADPCM_CHANNELS) return nullptr;
		break;
	case WA_ALAW:
	case WA_MULAW:
		if (channels < 1) return nullptr;
		break;
	default:
		return nullptr;
	}
	if (blockBytes <= 0) return nullptr;

	WaveAudioCtx* c = (WaveAudioCtx*)calloc(1, sizeof(WaveAudioCtx));
	c->codec = codec;
	c->channels = channels;
	c->bits = bits;
	c->blockBytes = blockBytes;

	// A stereo MS block one byte short of its header still writes both of the header's samples.
	const int blockSamples = get_nb_samples(c, blockBytes);
	c->capacity = blockSamples < 2 ? 2 : blockSamples;
	c->input = (uint8_t*)malloc(blockBytes + INPUT_PADDING);
	c->pcm = (int16_t*)calloc(c->capacity * channels, sizeof(int16_t));
	c->output = (float*)malloc(c->capacity * channels * sizeof(float));

	if (codec == WA_ALAW || codec == WA_MULAW) {
		for (int i = 0; i < 256; i++) {
			c->lut[i] = codec == WA_ALAW ? alaw2linear(i) : ulaw2linear(i);
		}
	}
	return c;
}

EMSCRIPTEN_KEEPALIVE
void wa_close(WaveAudioCtx* c)
{
	free(c->input);
	free(c->pcm);
	free(c->output);
	free(c);
}

EMSCRIPTEN_KEEPALIVE
void wa_reset(WaveAudioCtx* c)
{
	memset(c->status, 0, sizeof(c->status));
}

EMSCRIPTEN_KEEPALIVE
uint8_t* wa_input_ptr(WaveAudioCtx* c) { return c->input; }

EMSCRIPTEN_KEEPALIVE
int wa_decode(WaveAudioCtx* c, int size)
{
	memset(c->input + size, 0, INPUT_PADDING);

	const int nb_samples = get_nb_samples(c, size);
	if (nb_samples <= 0) return 0;

	int decoded;
	switch (c->codec) {
	case WA_MS: decoded = decode_ms(c, nb_samples); break;
	case WA_IMA_WAV: decoded = decode_ima_wav(c, nb_samples); break;
	case WA_IMA_QT: decoded = decode_ima_qt(c); break;
	default: decoded = decode_law(c, nb_samples); break;
	}

	for (int channel = 0; channel < c->channels; channel++) {
		const int16_t* pcm = c->pcm + channel * c->capacity;
		float* output = c->output + channel * c->capacity;
		for (int i = 0; i < decoded; i++) {
			output[i] = pcm[i] * (1.0f / 32768);
		}
	}
	return decoded;
}

EMSCRIPTEN_KEEPALIVE
float* wa_channel_ptr(WaveAudioCtx* c, int channel) { return c->output + channel * c->capacity; }

} // extern "C"
