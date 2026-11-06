/*
	WAVE audio decoders -- WASM C++. One module for the block- or byte-framed audio codecs of WAV,
	AVI, AIFF-C, VOC and MOV:
	  0x0002 MS ADPCM, 0x0006 A-law, 0x0007 mu-law, 0x0010 IMA OKI, 0x0011 IMA ADPCM (the WAV form),
	  0x0020 Yamaha ADPCM, 0x0031 GSM 6.10 (the Microsoft packing), 0x0045 G.726, 0x0200 Creative ADPCM,
	and the codecs with no WAVE tag of their own, numbered past any: QuickTime's 'ima4', MACE 3:1
	and 6:1, plain GSM 6.10, G.726 in AIFF's bit order, the SDX2 and CBD2 DPCMs and the Sound
	Blaster Pro ADPCMs of a VOC.

	A direct port of FFmpeg n8.1's libavcodec/adpcm.c, adpcm_data.c, adpcm_data.h, adpcm.h, pcm.c,
	pcm_tablegen.h, mace.c, gsmdec.c, gsmdec_template.c, gsmdec_data.c, msgsmdec.c, g726.c and the
	SDX2 and CBD2 cases of dpcm.c: get_nb_samples, the cases of adpcm_decode_frame, their
	expand_nibble functions, alaw2linear and ulaw2linear.

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
	WA_IMA_OKI = 0x0010,
	WA_IMA_WAV = 0x0011,
	WA_YAMAHA = 0x0020,
	WA_GSM_MS = 0x0031,
	WA_G726 = 0x0045,
	WA_CT = 0x0200,
	WA_IMA_QT = 0x10000,	// past any WAVE tag
	WA_MACE3,
	WA_MACE6,
	WA_GSM,
	WA_G726LE,
	WA_SDX2,
	WA_CBD2,
	WA_SBPRO_4,
	WA_SBPRO_3,
	WA_SBPRO_2,
};

static const int MAX_ADPCM_CHANNELS = 6;
static const int INPUT_PADDING = 64;
static const int GSM_BLOCK_SIZE = 33;
static const int GSM_MS_BLOCK_SIZE = 65;
static const int MSN_MIN_BLOCK_SIZE = 41;
static const int GSM_FRAME_SIZE = 160;

struct ChannelStatus
{
	int predictor;
	int16_t step_index;
	int sample1;
	int sample2;
	int coeff1;
	int coeff2;
	int idelta;
	int step;
};

struct MaceChannel
{
	int16_t index, factor, prev2, previous, level;
};

struct GsmState
{
	// The first 120 of the frame before, for the long-term filter's lag, then the 160 of this one.
	int16_t ref_buf[280];
	int v[9];
	int lar[2][8];
	int lar_idx;
	int msr;
};

/** G.726's 11-bit float. */
struct Float11
{
	uint8_t sign, exp, mant;
};

struct G726State
{
	Float11 sr[2];		// the samples before, reconstructed
	Float11 dq[6];		// the differences before
	int a[2];			// second order predictor
	int b[6];			// sixth order predictor
	int pk[2];			// the signs of the 2 before of sez + dq
	int ap, yu, yl, dms, dml, td, se, sez, y;
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
	int16_t lut[256];	// a law's samples, a DPCM's deltas
	ChannelStatus status[MAX_ADPCM_CHANNELS];
	MaceChannel mace[2];
	GsmState gsm;
	G726State g726;
	int dpcm[2];
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

/** A block's samples in the order FFmpeg interleaves them, into planes. */
struct Interleaved
{
	int16_t* pcm;
	int capacity, channels, at;

	void put(int value)
	{
		pcm[(at % channels) * capacity + at / channels] = (int16_t)value;
		at++;
	}
};

static inline int adpcm_ct_expand_nibble(ChannelStatus* c, int nibble)
{
	const int delta = nibble & 7;
	const int diff = ((2 * delta + 1) * c->step) >> 3;
	// The predictor decays by 254/256 before each update.
	c->predictor = clip_int16(((c->predictor * 254) >> 8) + ((nibble & 8) ? -diff : diff));
	c->step = clip((adpcm_AdaptationTable[delta] * c->step) >> 8, 511, 32767);
	return c->predictor;
}

static inline int adpcm_sbpro_expand_nibble(ChannelStatus* c, int nibble, int size, int shift)
{
	const int sign = nibble & (1 << (size - 1));
	const int delta = nibble & ((1 << (size - 1)) - 1);
	const int diff = delta << (7 + c->step + shift);

	c->predictor = clip(c->predictor + (sign ? -diff : diff), -16384, 16256);
	if (delta >= 2 * size - 3 && c->step < 3) c->step++;
	else if (delta == 0 && c->step > 0) c->step--;
	return c->predictor;
}

static inline int adpcm_yamaha_expand_nibble(ChannelStatus* c, int nibble)
{
	if (!c->step) {
		c->predictor = 0;
		c->step = 127;
	}
	c->predictor = clip_int16(c->predictor + (c->step * adpcm_yamaha_difflookup[nibble]) / 8);
	c->step = clip((c->step * adpcm_yamaha_indexscale[nibble]) >> 8, 127, 24576);
	return c->predictor;
}

static inline int adpcm_ima_oki_expand_nibble(ChannelStatus* c, int nibble)
{
	const int step = oki_step_table[c->step_index];
	const int step_index = clip(c->step_index + adpcm_index_table[nibble], 0, 48);
	const int delta = nibble & 7;
	const int diff = ((2 * delta + 1) * step) >> 3;
	const int predictor = (nibble & 8) ? c->predictor - diff : c->predictor + diff;

	c->predictor = clip(predictor, -2048, 2047);
	c->step_index = step_index;
	return c->predictor * 16;
}

/** The samples per byte of a Sound Blaster Pro ADPCM. */
static inline int sbpro_samples_per_byte(int codec)
{
	return codec == WA_SBPRO_2 ? 4 : codec == WA_SBPRO_3 ? 3 : 2;
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
	case WA_CT:
	case WA_YAMAHA:
	case WA_IMA_OKI:
		return size * 2 / ch;
	case WA_SBPRO_4:
	case WA_SBPRO_3:
	case WA_SBPRO_2: {
		int samples = 0;
		if (!c->status[0].step_index) {	// the first byte of a channel is a raw sample
			if (size < ch) return 0;
			samples = 1;
			size -= ch;
		}
		return samples + size * sbpro_samples_per_byte(c->codec) / ch;
	}
	case WA_MACE3:
		return 3 * (size - size % (ch << 1)) / ch;
	case WA_MACE6:
		return 3 * ((size - size % ch) << 1) / ch;
	case WA_GSM:
		return size < GSM_BLOCK_SIZE ? 0 : GSM_FRAME_SIZE;
	case WA_GSM_MS:
		return size < c->blockBytes ? 0 : 2 * GSM_FRAME_SIZE;
	case WA_G726:
	case WA_G726LE:
		return size * 8 / c->bits;
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

static int decode_ct_family(WaveAudioCtx* c, int nb_samples)
{
	const int st = c->channels == 2 ? 1 : 0;
	const uint8_t* in = c->input;
	Interleaved out = { c->pcm, c->capacity, c->channels, 0 };
	ChannelStatus* c0 = &c->status[0];
	ChannelStatus* c1 = &c->status[st];
	// A whole number of samples per channel: mono takes two from a byte, stereo one each.
	for (int n = nb_samples >> (1 - st); n > 0; n--) {
		const int v = *in++;
		switch (c->codec) {
		case WA_CT:
			out.put(adpcm_ct_expand_nibble(c0, v >> 4));
			out.put(adpcm_ct_expand_nibble(c1, v & 0x0F));
			break;
		case WA_IMA_OKI:
			out.put(adpcm_ima_oki_expand_nibble(c0, v >> 4));
			out.put(adpcm_ima_oki_expand_nibble(c1, v & 0x0F));
			break;
		default:
			out.put(adpcm_yamaha_expand_nibble(c0, v & 0x0F));
			out.put(adpcm_yamaha_expand_nibble(c1, v >> 4));
			break;
		}
	}
	return nb_samples;
}

static int decode_sbpro(WaveAudioCtx* c, int nb_samples)
{
	const int st = c->channels == 2 ? 1 : 0;
	const uint8_t* in = c->input;
	Interleaved out = { c->pcm, c->capacity, c->channels, 0 };
	ChannelStatus* c0 = &c->status[0];
	ChannelStatus* c1 = &c->status[st];
	int left = nb_samples;

	if (!c0->step_index) {
		out.put(128 * (*in++ - 0x80));
		if (st) out.put(128 * (*in++ - 0x80));
		c0->step_index = 1;
		left--;
	}
	if (c->codec == WA_SBPRO_4) {
		for (int n = left >> (1 - st); n > 0; n--) {
			const int byte = *in++;
			out.put(adpcm_sbpro_expand_nibble(c0, byte >> 4, 4, 0));
			out.put(adpcm_sbpro_expand_nibble(c1, byte & 0x0F, 4, 0));
		}
	} else if (c->codec == WA_SBPRO_3) {
		for (int n = (left << st) / 3; n > 0; n--) {
			const int byte = *in++;
			out.put(adpcm_sbpro_expand_nibble(c0, byte >> 5, 3, 0));
			out.put(adpcm_sbpro_expand_nibble(c0, (byte >> 2) & 0x07, 3, 0));
			out.put(adpcm_sbpro_expand_nibble(c0, byte & 0x03, 2, 0));
		}
	} else {
		for (int n = left >> (2 - st); n > 0; n--) {
			const int byte = *in++;
			out.put(adpcm_sbpro_expand_nibble(c0, byte >> 6, 2, 2));
			out.put(adpcm_sbpro_expand_nibble(c1, (byte >> 4) & 0x03, 2, 2));
			out.put(adpcm_sbpro_expand_nibble(c0, (byte >> 2) & 0x03, 2, 2));
			out.put(adpcm_sbpro_expand_nibble(c1, byte & 0x03, 2, 2));
		}
	}
	return nb_samples;
}

static int decode_dpcm(WaveAudioCtx* c, int nb_samples)
{
	const uint8_t* in = c->input;
	Interleaved out = { c->pcm, c->capacity, c->channels, 0 };
	const int stereo = c->channels - 1;
	int ch = 0;
	for (int i = nb_samples * c->channels; i > 0; i--) {
		const int8_t n = (int8_t)*in++;
		if (!(n & 1)) c->dpcm[ch] = 0;
		c->dpcm[ch] = clip_int16(c->dpcm[ch] + c->lut[n + 128]);
		out.put(c->dpcm[ch]);
		ch ^= stereo;
	}
	return nb_samples;
}

/** MACE's own clip, for output identical to the binary decoder's. */
static inline int mace_broken_clip_int16(int n)
{
	return n > 32767 ? 32767 : n < -32768 ? -32767 : n;
}

static inline int mace_read_table(MaceChannel* chd, int val, int tab_idx)
{
	// Tables 0 and 2 are the same pair, 1 the other: a sample's three parts take them in that order.
	const int stride = tab_idx == 1 ? 2 : 4;
	const int16_t* tab1 = tab_idx == 1 ? mace_tab3 : mace_tab1;
	const int16_t* tab2 = tab_idx == 1 ? &mace_tab4[0][0] : &mace_tab2[0][0];
	const int row = ((chd->index & 0x7f0) >> 4) * stride;

	int16_t current;
	if (val < stride) current = tab2[row + val];
	else current = (int16_t)(-1 - tab2[row + 2 * stride - val - 1]);

	chd->index = (int16_t)(chd->index + tab1[val] - (chd->index >> 5));
	if (chd->index < 0) chd->index = 0;
	return current;
}

static inline int qt_8s_2_16s(int x) { return (int16_t)((x & 0xFF00) | ((x >> 8) & 0xFF)); }

static inline void mace_chomp3(MaceChannel* chd, int16_t* output, int val, int tab_idx)
{
	int16_t current = (int16_t)mace_read_table(chd, val, tab_idx);
	current = (int16_t)mace_broken_clip_int16(current + chd->level);
	chd->level = (int16_t)(current - (current >> 3));
	*output = (int16_t)qt_8s_2_16s(current);
}

static inline void mace_chomp6(MaceChannel* chd, int16_t* output, int val, int tab_idx)
{
	int16_t current = (int16_t)mace_read_table(chd, val, tab_idx);

	if ((chd->previous ^ current) >= 0) {
		chd->factor = (int16_t)(chd->factor + 506 < 32767 ? chd->factor + 506 : 32767);
	} else if (chd->factor - 314 < -32768) {
		chd->factor = -32767;
	} else {
		chd->factor -= 314;
	}

	current = (int16_t)mace_broken_clip_int16(current + chd->level);
	chd->level = (int16_t)((current * chd->factor) >> 15);
	current >>= 1;

	output[0] = (int16_t)qt_8s_2_16s(chd->previous + chd->prev2 - ((chd->prev2 - current) >> 2));
	output[1] = (int16_t)qt_8s_2_16s(chd->previous + current + ((chd->prev2 - current) >> 2));
	chd->prev2 = chd->previous;
	chd->previous = current;
}

static int decode_mace(WaveAudioCtx* c, int size, int nb_samples)
{
	const int channels = c->channels;
	const int is_mace3 = c->codec == WA_MACE3;
	const int unit = channels << is_mace3;
	const uint8_t* buf = c->input;

	for (int i = 0; i < channels; i++) {
		int16_t* output = c->pcm + i * c->capacity;
		for (int j = 0; j < size / unit; j++) {
			for (int k = 0; k < (1 << is_mace3); k++) {
				const int pkt = buf[(i << is_mace3) + (j * channels << is_mace3) + k];
				const int val[2][3] = { { pkt >> 5, (pkt >> 3) & 3, pkt & 7 }, { pkt & 7, (pkt >> 3) & 3, pkt >> 5 } };
				for (int l = 0; l < 3; l++) {
					if (is_mace3) mace_chomp3(&c->mace[i], output, val[1][l], l);
					else mace_chomp6(&c->mace[i], output, val[0][l], l);
					output += 1 << (1 - is_mace3);
				}
			}
		}
	}
	return nb_samples;
}

struct GetBits
{
	const uint8_t* buf;
	int bit;
	bool little;
};

static inline int get_bits(GetBits* gb, int bits)
{
	int value = 0;
	for (int i = 0; i < bits; i++, gb->bit++) {
		const int byte = gb->buf[gb->bit >> 3];
		if (gb->little) value |= ((byte >> (gb->bit & 7)) & 1) << i;
		else value = (value << 1) | ((byte >> (7 - (gb->bit & 7))) & 1);
	}
	return value;
}

static inline int gsm_mult(int a, int b) { return (a * b + (1 << 14)) >> 15; }

static void gsm_apcm_dequant_add(GetBits* gb, int16_t* dst, const uint8_t* frame_bits)
{
	const int16_t* tab = gsm_dequant_tab[get_bits(gb, 6)];
	for (int i = 0; i < 13; i++) {
		const int val = get_bits(gb, frame_bits[i]);
		dst[3 * i] += tab[gsm_requant_tab[frame_bits[i]][val]];
	}
}

static void gsm_long_term_synth(int16_t* dst, int lag, int gain_idx)
{
	const int16_t* src = dst - lag;
	const int gain = gsm_long_term_gain_tab[gain_idx];
	for (int i = 0; i < 40; i++) dst[i] = (int16_t)gsm_mult(gain, src[i]);
}

static inline int gsm_decode_log_area(int coded, int factor, int offset)
{
	coded <<= 10;
	coded -= offset;
	return gsm_mult(coded, factor) * 2;
}

static int gsm_get_rrp(int filtered)
{
	int value = filtered < 0 ? -filtered : filtered;
	if (value < 11059) value <<= 1;
	else if (value < 20070) value += 11059;
	else value = (value >> 2) + 26112;
	return filtered < 0 ? -value : value;
}

static inline int gsm_filter_value(int in, const int rrp[8], int v[9])
{
	for (int i = 7; i >= 0; i--) {
		in -= gsm_mult(rrp[i], v[i]);
		v[i + 1] = v[i] + gsm_mult(rrp[i], in);
	}
	v[0] = in;
	return in;
}

static void gsm_short_term_synth(GsmState* ctx, int16_t* dst, const int16_t* src)
{
	int rrp[8];
	const int* lar = ctx->lar[ctx->lar_idx];
	const int* lar_prev = ctx->lar[ctx->lar_idx ^ 1];

	for (int i = 0; i < 8; i++) rrp[i] = gsm_get_rrp((lar_prev[i] >> 2) + (lar_prev[i] >> 1) + (lar[i] >> 2));
	for (int i = 0; i < 13; i++) dst[i] = (int16_t)gsm_filter_value(src[i], rrp, ctx->v);

	for (int i = 0; i < 8; i++) rrp[i] = gsm_get_rrp((lar_prev[i] >> 1) + (lar[i] >> 1));
	for (int i = 13; i < 27; i++) dst[i] = (int16_t)gsm_filter_value(src[i], rrp, ctx->v);

	for (int i = 0; i < 8; i++) rrp[i] = gsm_get_rrp((lar_prev[i] >> 2) + (lar[i] >> 1) + (lar[i] >> 2));
	for (int i = 27; i < 40; i++) dst[i] = (int16_t)gsm_filter_value(src[i], rrp, ctx->v);

	for (int i = 0; i < 8; i++) rrp[i] = gsm_get_rrp(lar[i]);
	for (int i = 40; i < 160; i++) dst[i] = (int16_t)gsm_filter_value(src[i], rrp, ctx->v);

	ctx->lar_idx ^= 1;
}

static int gsm_postprocess(int16_t* data, int msr)
{
	for (int i = 0; i < 160; i++) {
		msr = clip_int16(data[i] + gsm_mult(msr, 28180));
		data[i] = (int16_t)(clip_int16(msr * 2) & ~7);
	}
	return msr;
}

static void gsm_decode_block(GsmState* ctx, int16_t* samples, GetBits* gb, int mode)
{
	int16_t* ref_dst = ctx->ref_buf + 120;
	int* lar = ctx->lar[ctx->lar_idx];
	lar[0] = gsm_decode_log_area(get_bits(gb, 6), 13107, 1 << 15);
	lar[1] = gsm_decode_log_area(get_bits(gb, 6), 13107, 1 << 15);
	lar[2] = gsm_decode_log_area(get_bits(gb, 5), 13107, (1 << 14) + 2048 * 2);
	lar[3] = gsm_decode_log_area(get_bits(gb, 5), 13107, (1 << 14) - 2560 * 2);
	lar[4] = gsm_decode_log_area(get_bits(gb, 4), 19223, (1 << 13) + 94 * 2);
	lar[5] = gsm_decode_log_area(get_bits(gb, 4), 17476, (1 << 13) - 1792 * 2);
	lar[6] = gsm_decode_log_area(get_bits(gb, 3), 31454, (1 << 12) - 341 * 2);
	lar[7] = gsm_decode_log_area(get_bits(gb, 3), 29708, (1 << 12) - 1144 * 2);

	for (int i = 0; i < 4; i++) {
		const int lag = clip(get_bits(gb, 7), 40, 120);
		const int gain_idx = get_bits(gb, 2);
		const int offset = get_bits(gb, 2);
		gsm_long_term_synth(ref_dst, lag, gain_idx);
		gsm_apcm_dequant_add(gb, ref_dst + offset, gsm_apcm_bits[gsm_mode_bits[mode][i]]);
		ref_dst += 40;
	}
	memcpy(ctx->ref_buf, ctx->ref_buf + 160, 120 * sizeof(*ctx->ref_buf));
	gsm_short_term_synth(ctx, samples, ctx->ref_buf + 120);
	ctx->msr = gsm_postprocess(samples, ctx->msr);
}

static int decode_gsm(WaveAudioCtx* c)
{
	GetBits gb = { c->input, 0, false };
	get_bits(&gb, 4);	// the magic: FFmpeg only warns
	gsm_decode_block(&c->gsm, c->pcm, &gb, 0);
	return GSM_FRAME_SIZE;
}

static int decode_gsm_ms(WaveAudioCtx* c)
{
	const int mode = (GSM_MS_BLOCK_SIZE - c->blockBytes) / 3;
	GetBits gb = { c->input, 0, true };
	gsm_decode_block(&c->gsm, c->pcm, &gb, mode);
	gsm_decode_block(&c->gsm, c->pcm + GSM_FRAME_SIZE, &gb, mode);
	return 2 * GSM_FRAME_SIZE;
}

static inline int log2_16bit(unsigned value)
{
	int n = 0;
	if (value & 0xff00) {
		value >>= 8;
		n = 8;
	}
	return value ? n + 31 - __builtin_clz(value) : n;
}

static inline Float11* i2f(int i, Float11* f)
{
	f->sign = i < 0;
	if (f->sign) i = -i;
	f->exp = (uint8_t)(log2_16bit(i) + !!i);
	f->mant = i ? (uint8_t)((i << 6) >> f->exp) : 1 << 5;
	return f;
}

static inline int16_t g726_mult(const Float11* f1, const Float11* f2)
{
	const int exp = f1->exp + f2->exp;
	int res = ((f1->mant * f2->mant) + 0x30) >> 4;
	res = exp > 19 ? res << (exp - 19) : res >> (19 - exp);
	return (int16_t)((f1->sign ^ f2->sign) ? -res : res);
}

static inline int sgn(int value) { return value < 0 ? -1 : 1; }

static inline int g726_inverse_quant(const G726State* c, int code_size, int i)
{
	const int dql = g726_iquant[code_size - 2][i] + (c->y >> 2);
	const int dex = (dql >> 7) & 0xf;			// a 4-bit exponent
	const int dqt = (1 << 7) + (dql & 0x7f);	// log2 to linear
	return (int16_t)(dql < 0 ? 0 : (dqt << dex) >> 7);
}

static int16_t g726_decode(G726State* c, int code_size, int I)
{
	const int16_t* W = g726_W[code_size - 2];
	const uint8_t* F = g726_F[code_size - 2];
	Float11 f;
	const int I_sig = I >> (code_size - 1);
	int dq = g726_inverse_quant(c, code_size, I);

	// Transition detect
	const int ylint = c->yl >> 15;
	const int ylfrac = (c->yl >> 10) & 0x1f;
	const int thr2 = ylint > 9 ? 0x1f << 10 : (0x20 + ylfrac) << ylint;
	const int tr = c->td == 1 && dq > ((3 * thr2) >> 2);

	if (I_sig) dq = -dq;
	const int re_signal = (int16_t)(c->se + dq);

	// The second order predictor's A2 and A1
	const int pk0 = (c->sez + dq) ? sgn(c->sez + dq) : 0;
	const int dq0 = dq ? sgn(dq) : 0;
	if (tr) {
		c->a[0] = 0;
		c->a[1] = 0;
		for (int i = 0; i < 6; i++) c->b[i] = 0;
	} else {
		const int fa1 = clip((-c->a[0] * c->pk[0] * pk0) >> 5, -256, 255);	// +255 and not +256: it really is

		c->a[1] += 128 * pk0 * c->pk[1] + fa1 - (c->a[1] >> 7);
		c->a[1] = clip(c->a[1], -12288, 12288);
		c->a[0] += 64 * 3 * pk0 * c->pk[0] - (c->a[0] >> 8);
		c->a[0] = clip(c->a[0], -(15360 - c->a[1]), 15360 - c->a[1]);

		for (int i = 0; i < 6; i++) c->b[i] += 128 * dq0 * sgn(-c->dq[i].sign) - (c->b[i] >> 8);
	}

	c->pk[1] = c->pk[0];
	c->pk[0] = pk0 ? pk0 : 1;
	c->sr[1] = c->sr[0];
	i2f(re_signal, &c->sr[0]);
	for (int i = 5; i > 0; i--) c->dq[i] = c->dq[i - 1];
	i2f(dq, &c->dq[0]);
	c->dq[0].sign = (uint8_t)I_sig;

	c->td = c->a[1] < -11776;

	// Ap
	c->dms += (F[I] << 4) + ((-c->dms) >> 5);
	c->dml += (F[I] << 4) + ((-c->dml) >> 7);
	if (tr) {
		c->ap = 256;
	} else {
		c->ap += (-c->ap) >> 4;
		const int spread = (c->dms << 2) - c->dml;
		if (c->y <= 1535 || c->td || (spread < 0 ? -spread : spread) >= (c->dml >> 3)) c->ap += 0x20;
	}

	// Yu and Yl
	c->yu = clip(c->y + W[I] + ((-c->y) >> 5), 544, 5120);
	c->yl += c->yu + ((-c->yl) >> 6);

	// The next iteration's Y, SE and SEZ
	const int al = c->ap >= 256 ? 1 << 6 : c->ap >> 2;
	c->y = (c->yl + (c->yu - (c->yl >> 6)) * al) >> 6;

	c->se = 0;
	for (int i = 0; i < 6; i++) c->se += g726_mult(i2f(c->b[i] >> 2, &f), &c->dq[i]);
	c->sez = c->se >> 1;
	for (int i = 0; i < 2; i++) c->se += g726_mult(i2f(c->a[i] >> 2, &f), &c->sr[i]);
	c->se >>= 1;

	return (int16_t)clip(re_signal * 4, -0xffff, 0xffff);
}

static int decode_g726(WaveAudioCtx* c, int nb_samples)
{
	GetBits gb = { c->input, 0, c->codec == WA_G726LE };
	for (int i = 0; i < nb_samples; i++) c->pcm[i] = g726_decode(&c->g726, c->bits, get_bits(&gb, c->bits));
	return nb_samples;
}

/** Every codec's history as it stands before a stream's first byte. */
static void reset_state(WaveAudioCtx* c)
{
	memset(c->status, 0, sizeof(c->status));
	memset(c->mace, 0, sizeof(c->mace));
	memset(&c->gsm, 0, sizeof(c->gsm));
	memset(&c->g726, 0, sizeof(c->g726));
	c->dpcm[0] = c->dpcm[1] = 0;

	if (c->codec == WA_CT) {
		c->status[0].step = c->status[1].step = 511;
	} else if (c->codec == WA_G726 || c->codec == WA_G726LE) {
		G726State* g = &c->g726;
		for (int i = 0; i < 2; i++) {
			g->sr[i].mant = 1 << 5;
			g->pk[i] = 1;
		}
		for (int i = 0; i < 6; i++) g->dq[i].mant = 1 << 5;
		g->yu = 544;
		g->yl = 34816;
		g->y = 544;
	}
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
	case WA_CT:
	case WA_YAMAHA:
	case WA_IMA_OKI:
	case WA_MACE3:
	case WA_MACE6:
	case WA_SDX2:
	case WA_CBD2:
	case WA_SBPRO_4:
	case WA_SBPRO_3:
	case WA_SBPRO_2:
		if (channels < 1 || channels > 2) return nullptr;
		break;
	case WA_GSM:
		if (channels != 1) return nullptr;
		break;
	case WA_GSM_MS:
		if (channels != 1 || blockBytes < MSN_MIN_BLOCK_SIZE || blockBytes > GSM_MS_BLOCK_SIZE ||
			(blockBytes - MSN_MIN_BLOCK_SIZE) % 3) return nullptr;
		break;
	case WA_G726:
	case WA_G726LE:
		if (channels != 1 || bits < 2 || bits > 5) return nullptr;
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
	// A Sound Blaster Pro stream's first block, sized above, holds up to 3 samples fewer than the rest.
	c->capacity = (blockSamples < 2 ? 2 : blockSamples) + 3;
	c->input = (uint8_t*)malloc(blockBytes + INPUT_PADDING);
	c->pcm = (int16_t*)calloc(c->capacity * channels, sizeof(int16_t));
	c->output = (float*)malloc(c->capacity * channels * sizeof(float));

	if (codec == WA_ALAW || codec == WA_MULAW) {
		for (int i = 0; i < 256; i++) {
			c->lut[i] = codec == WA_ALAW ? alaw2linear(i) : ulaw2linear(i);
		}
	} else if (codec == WA_SDX2) {
		for (int i = -128; i < 128; i++) {
			const int16_t square = (int16_t)(i * i * 2);
			c->lut[i + 128] = (int16_t)(i < 0 ? -square : square);
		}
	} else if (codec == WA_CBD2) {
		for (int i = -128; i < 128; i++) {
			c->lut[i + 128] = (int16_t)((i * i * i) / 64);
		}
	}
	reset_state(c);
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
	reset_state(c);
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
	case WA_CT:
	case WA_YAMAHA:
	case WA_IMA_OKI: decoded = decode_ct_family(c, nb_samples); break;
	case WA_SBPRO_4:
	case WA_SBPRO_3:
	case WA_SBPRO_2: decoded = decode_sbpro(c, nb_samples); break;
	case WA_MACE3:
	case WA_MACE6: decoded = decode_mace(c, size, nb_samples); break;
	case WA_GSM: decoded = decode_gsm(c); break;
	case WA_GSM_MS: decoded = decode_gsm_ms(c); break;
	case WA_G726:
	case WA_G726LE: decoded = decode_g726(c, nb_samples); break;
	case WA_SDX2:
	case WA_CBD2: decoded = decode_dpcm(c, nb_samples); break;
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
