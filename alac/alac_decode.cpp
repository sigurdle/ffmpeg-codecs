/*
	ALAC (Apple Lossless) audio decoder -- WASM C++.

	A port of FFmpeg n8.1's ALAC decoder (LGPL-2.1+): libavcodec/alac.c, alac_data.c, alacdsp.c and
	the get_unary_0_9 of unary.h. Copyright (c) 2005 David Hammerton. The element parse, the Rice
	residuals, the adaptive predictor, the stereo decorrelation and the extra bits are FFmpeg's; the
	libavcodec framework around them (AVCodecContext, AVFrame, the DSP pointers, threads) is replaced
	by one handle that owns planar float output.

	A packet is one whole MP4 sample: alac_decode_frame takes its bytes, as FFmpeg's decoder takes an
	AVPacket. Nothing carries from one packet to the next, so a seek needs no reset.

	Departures from FFmpeg:
	  - Output is planar float at every depth: FFmpeg gives s16 at 16 bits and s32 above. 16, 20 and
	    24 bits convert exactly; 32-bit samples round to float's 24 bits.
	  - The magic cookie is parsed in JS (alacConfigInfo in alac_decode_core.js), which is
	    alac_set_info and the checks of alac_decode_init, because a reader asks before it opens a
	    decoder. alac_configure takes the fields. A cookie of channel count 0 refuses: FFmpeg would
	    take the container's, but mov.c has overwritten that with the cookie's 0 by then.
	  - The extra_bits_bug option, for streams from one old broken encoder, is gone, and with it the
	    extra bits appended before the decorrelation. So are the assembly paths of alacdsp.
	  - alac_packet_samples is new: the first element's header, what a span index needs without
	    decoding. FFmpeg has no such walk.
	  - Log messages are gone; each error returns as FFmpeg's does.

	-fwrapv is REQUIRED: the predictor relies on two's-complement wrap.
*/

#include <cstdint>
#include <cstdlib>
#include <cstring>

#include <emscripten/emscripten.h>

#define ALAC_MAX_CHANNELS 8
// Bytes the input must hold free past its end: every bit read loads 8.
#define INPUT_PADDING 16

enum AlacRawDataBlockType { TYPE_SCE, TYPE_CPE, TYPE_CCE, TYPE_LFE, TYPE_DSE, TYPE_PCE, TYPE_FIL, TYPE_END };

// Output plane of each channel in bitstream order, for each channel count (ff_alac_channel_layout_offsets).
static const uint8_t alac_channel_layout_offsets[ALAC_MAX_CHANNELS][ALAC_MAX_CHANNELS] = {
	{ 0 },
	{ 0, 1 },
	{ 2, 0, 1 },
	{ 2, 0, 1, 3 },
	{ 2, 0, 1, 3, 4 },
	{ 2, 0, 1, 4, 5, 3 },
	{ 2, 0, 1, 4, 5, 6, 3 },
	{ 2, 6, 7, 0, 1, 4, 5, 3 },
};

// --- the bit reader, FFmpeg's checked get_bits: an index clamped a byte past the end ---------

struct GetBits {
	const uint8_t *buffer;
	int index;
	int size_in_bits;
	int size_in_bits_plus8;
};

static inline void init_get_bits8(GetBits *gb, const uint8_t *buffer, int size)
{
	gb->buffer = buffer;
	gb->index = 0;
	gb->size_in_bits = size * 8;
	gb->size_in_bits_plus8 = size * 8 + 8;
}

/** The bits at the index, first at the top, 57 of them good. */
static inline uint64_t show_cache(const GetBits *gb)
{
	uint64_t word;
	memcpy(&word, gb->buffer + (gb->index >> 3), 8);
	return __builtin_bswap64(word) << (gb->index & 7);
}

static inline void skip_bits(GetBits *gb, int n)
{
	const int index = gb->index + n;
	gb->index = index < gb->size_in_bits_plus8 ? index : gb->size_in_bits_plus8;
}

/** 0 to 32 bits. */
static inline unsigned show_bits(const GetBits *gb, int n)
{
	return n ? (unsigned)(show_cache(gb) >> (64 - n)) : 0;
}

/** 0 to 32 bits. */
static inline unsigned get_bits(GetBits *gb, int n)
{
	const unsigned value = show_bits(gb, n);
	skip_bits(gb, n);
	return value;
}

static inline unsigned get_bits1(GetBits *gb)
{
	return get_bits(gb, 1);
}

/** 1 to 32 bits, sign extended. */
static inline int get_sbits(GetBits *gb, int n)
{
	return (int)(get_bits(gb, n) << (32 - n)) >> (32 - n);
}

static inline int get_bits_left(const GetBits *gb)
{
	return gb->size_in_bits - gb->index;
}

static inline int get_unary_0_9(GetBits *gb)
{
	int i;
	for (i = 0; i < 9 && get_bits1(gb) != 0; i++);
	return i;
}

static inline int av_log2(unsigned v)
{
	return 31 - __builtin_clz(v | 1);
}

static inline int sign_extend(int val, int bits)
{
	const int shift = 32 - bits;
	return (int)((unsigned)val << shift) >> shift;
}

static inline int sign_only(int v)
{
	return (v > 0) - (v < 0);
}

static inline int imin(int a, int b)
{
	return a < b ? a : b;
}

// --- the handle --------------------------------------------------------------------------

struct ALACContext {
	GetBits gb;
	int channels;

	int32_t *predict_error_buffer[2];
	int32_t *output_samples_buffer[2];
	int32_t *extra_bits_buffer[2];

	uint32_t max_samples_per_frame;
	int sample_size;
	int rice_history_mult;
	int rice_initial_history;
	int rice_limit;
	int sample_rate;

	int extra_bits;		// number of extra bits beyond 16-bit
	int nb_samples;		// number of samples in the current frame

	float *output[ALAC_MAX_CHANNELS];
	int output_capacity;
};

static inline unsigned decode_scalar(GetBits *gb, int k, int bps)
{
	unsigned x = get_unary_0_9(gb);

	if (x > 8) {	// Rice threshold: the alternative encoding
		x = get_bits(gb, bps);
	} else if (k != 1) {
		const int extrabits = show_bits(gb, k);

		// multiply x by 2^k - 1
		x = (x << k) - x;

		if (extrabits > 1) {
			x += extrabits - 1;
			skip_bits(gb, k);
		} else {
			skip_bits(gb, k - 1);
		}
	}
	return x;
}

static int rice_decompress(ALACContext *alac, int32_t *output_buffer, int nb_samples, int bps, int rice_history_mult)
{
	GetBits *gb = &alac->gb;
	unsigned history = alac->rice_initial_history;
	int sign_modifier = 0;

	for (int i = 0; i < nb_samples; i++) {
		if (get_bits_left(gb) <= 0) {
			return -1;
		}

		// the Rice parameter, then the next value
		int k = av_log2((history >> 9) + 3);
		k = imin(k, alac->rice_limit);
		unsigned x = decode_scalar(gb, k, bps);
		x += sign_modifier;
		sign_modifier = 0;
		output_buffer[i] = (x >> 1) ^ -(x & 1);

		// update the history
		if (x > 0xffff) {
			history = 0xffff;
		} else {
			history += x * rice_history_mult - ((history * rice_history_mult) >> 9);
		}

		// there may be compressed blocks of 0
		if (history < 128 && i + 1 < nb_samples) {
			k = 7 - av_log2(history) + ((history + 16) >> 6);
			k = imin(k, alac->rice_limit);
			int block_size = decode_scalar(gb, k, 16);

			if (block_size > 0) {
				if (block_size >= nb_samples - i) {
					block_size = nb_samples - i - 1;
				}
				memset(&output_buffer[i + 1], 0, block_size * sizeof(*output_buffer));
				i += block_size;
			}
			if (block_size <= 0xffff) {
				sign_modifier = 1;
			}
			history = 0;
		}
	}
	return 0;
}

static void lpc_prediction(int32_t *error_buffer, uint32_t *buffer_out, int nb_samples, int bps,
	int16_t *lpc_coefs, int lpc_order, int lpc_quant)
{
	int i;
	uint32_t *pred = buffer_out;

	// first sample always copies
	*buffer_out = *error_buffer;

	if (nb_samples <= 1) {
		return;
	}

	if (!lpc_order) {
		memcpy(&buffer_out[1], &error_buffer[1], (nb_samples - 1) * sizeof(*buffer_out));
		return;
	}

	if (lpc_order == 31) {
		// simple 1st-order prediction
		for (i = 1; i < nb_samples; i++) {
			buffer_out[i] = sign_extend(buffer_out[i - 1] + error_buffer[i], bps);
		}
		return;
	}

	// warm-up samples
	for (i = 1; i <= lpc_order && i < nb_samples; i++) {
		buffer_out[i] = sign_extend(buffer_out[i - 1] + error_buffer[i], bps);
	}

	for (; i < nb_samples; i++) {
		int val = 0;
		unsigned error_val = error_buffer[i];
		const int d = *pred++;

		// LPC prediction
		for (int j = 0; j < lpc_order; j++) {
			val += (pred[j] - d) * lpc_coefs[j];
		}
		val = (val + (1LL << (lpc_quant - 1))) >> lpc_quant;
		val += d + error_val;
		buffer_out[i] = sign_extend(val, bps);

		// adapt the LPC coefficients
		const int error_sign = sign_only(error_val);
		if (error_sign) {
			for (int j = 0; j < lpc_order && (int)(error_val * error_sign) > 0; j++) {
				val = d - pred[j];
				const int sign = sign_only(val) * error_sign;
				lpc_coefs[j] -= sign;
				val *= (unsigned)sign;
				error_val -= (val >> lpc_quant) * (j + 1U);
			}
		}
	}
}

static void decorrelate_stereo(int32_t *buffer[2], int nb_samples, int decorr_shift, int decorr_left_weight)
{
	for (int i = 0; i < nb_samples; i++) {
		uint32_t a = buffer[0][i];
		uint32_t b = buffer[1][i];

		a -= (int)(b * decorr_left_weight) >> decorr_shift;
		b += a;

		buffer[0][i] = b;
		buffer[1][i] = a;
	}
}

static void append_extra_bits(int32_t *buffer[2], int32_t *extra_bits_buffer[2], int extra_bits, int channels, int nb_samples)
{
	for (int ch = 0; ch < channels; ch++) {
		for (int i = 0; i < nb_samples; i++) {
			buffer[ch][i] = ((unsigned)buffer[ch][i] << extra_bits) | extra_bits_buffer[ch][i];
		}
	}
}

/** The output planes hold `nb_samples`: what ff_thread_get_buffer did. */
static int reserve_output(ALACContext *alac, int nb_samples)
{
	if (nb_samples <= alac->output_capacity) {
		return 0;
	}
	alac->output_capacity = 0;
	for (int ch = 0; ch < alac->channels; ch++) {
		free(alac->output[ch]);
		alac->output[ch] = (float *)malloc(nb_samples * sizeof(float));
		if (!alac->output[ch]) {
			return -1;
		}
	}
	alac->output_capacity = nb_samples;
	return 0;
}

/** The planes at output, from the 32 bits a depth sits at the top of, or the 16 of the s16 FFmpeg gives. */
static void write_output(ALACContext *alac, int ch_index, int channels)
{
	const int nb_samples = alac->nb_samples;
	for (int ch = 0; ch < channels; ch++) {
		const int32_t *in = alac->output_samples_buffer[ch];
		float *out = alac->output[ch_index + ch];
		switch (alac->sample_size) {
		case 16:
			for (int i = 0; i < nb_samples; i++) {
				out[i] = (int16_t)in[i] / 32768.0f;
			}
			break;
		case 20:
			for (int i = 0; i < nb_samples; i++) {
				out[i] = (int32_t)((uint32_t)in[i] << 12) / 2147483648.0f;
			}
			break;
		case 24:
			for (int i = 0; i < nb_samples; i++) {
				out[i] = (int32_t)((uint32_t)in[i] << 8) / 2147483648.0f;
			}
			break;
		default:
			for (int i = 0; i < nb_samples; i++) {
				out[i] = in[i] / 2147483648.0f;
			}
			break;
		}
	}
}

static int decode_element(ALACContext *alac, int ch_index, int channels)
{
	GetBits *gb = &alac->gb;
	int decorr_shift, decorr_left_weight;

	skip_bits(gb, 4);	// element instance tag
	skip_bits(gb, 12);	// unused header bits

	// the number of output samples is stored in the frame
	const int has_size = get_bits1(gb);

	alac->extra_bits = get_bits(gb, 2) << 3;
	const int bps = alac->sample_size - alac->extra_bits + channels - 1;
	if (bps > 32 || bps < 1) {
		return -1;
	}

	const int is_compressed = !get_bits1(gb);

	const uint32_t output_samples = has_size ? get_bits(gb, 32) : alac->max_samples_per_frame;
	if (!output_samples || output_samples > alac->max_samples_per_frame) {
		return -1;
	}
	if (!alac->nb_samples) {
		if (reserve_output(alac, output_samples) < 0) {
			return -1;
		}
	} else if ((int)output_samples != alac->nb_samples) {
		return -1;
	}
	alac->nb_samples = output_samples;

	if (is_compressed) {
		int16_t lpc_coefs[2][32];
		int lpc_order[2];
		int prediction_type[2];
		int lpc_quant[2];
		int rice_history_mult[2];

		if (!alac->rice_limit) {
			return -1;
		}

		decorr_shift = get_bits(gb, 8);
		decorr_left_weight = get_bits(gb, 8);

		if (channels == 2 && decorr_left_weight && decorr_shift > 31) {
			return -1;
		}

		for (int ch = 0; ch < channels; ch++) {
			prediction_type[ch] = get_bits(gb, 4);
			lpc_quant[ch] = get_bits(gb, 4);
			rice_history_mult[ch] = get_bits(gb, 3);
			lpc_order[ch] = get_bits(gb, 5);

			if ((uint32_t)lpc_order[ch] >= alac->max_samples_per_frame || !lpc_quant[ch]) {
				return -1;
			}

			// the predictor table
			for (int i = lpc_order[ch] - 1; i >= 0; i--) {
				lpc_coefs[ch][i] = get_sbits(gb, 16);
			}
		}

		if (alac->extra_bits) {
			const int extra_bits = alac->extra_bits;
			if (get_bits_left(gb) < (int64_t)alac->nb_samples * channels * extra_bits) {
				return -1;
			}
			for (int i = 0; i < alac->nb_samples; i++) {
				for (int ch = 0; ch < channels; ch++) {
					alac->extra_bits_buffer[ch][i] = get_bits(gb, extra_bits);
				}
			}
		}
		for (int ch = 0; ch < channels; ch++) {
			if (rice_decompress(alac, alac->predict_error_buffer[ch], alac->nb_samples, bps,
					rice_history_mult[ch] * alac->rice_history_mult / 4) < 0) {
				return -1;
			}

			// Prediction type 15 runs the adaptive FIR twice, the first pass the special case of order 31. Any other
			// nonzero type is unknown, and FFmpeg warns and goes on.
			if (prediction_type[ch] == 15) {
				lpc_prediction(alac->predict_error_buffer[ch], (uint32_t *)alac->predict_error_buffer[ch],
					alac->nb_samples, bps, nullptr, 31, 0);
			}
			lpc_prediction(alac->predict_error_buffer[ch], (uint32_t *)alac->output_samples_buffer[ch], alac->nb_samples,
				bps, lpc_coefs[ch], lpc_order[ch], lpc_quant[ch]);
		}
	} else {
		// not compressed, the easy case
		const int sample_size = alac->sample_size;
		if (get_bits_left(gb) < (int64_t)alac->nb_samples * channels * sample_size) {
			return -1;
		}
		for (int i = 0; i < alac->nb_samples; i++) {
			for (int ch = 0; ch < channels; ch++) {
				alac->output_samples_buffer[ch][i] = get_sbits(gb, sample_size);
			}
		}
		alac->extra_bits = 0;
		decorr_shift = 0;
		decorr_left_weight = 0;
	}

	if (channels == 2 && decorr_left_weight) {
		decorrelate_stereo(alac->output_samples_buffer, alac->nb_samples, decorr_shift, decorr_left_weight);
	}
	if (alac->extra_bits) {
		append_extra_bits(alac->output_samples_buffer, alac->extra_bits_buffer, alac->extra_bits, channels, alac->nb_samples);
	}

	write_output(alac, ch_index, channels);
	return 0;
}

// --- exports -----------------------------------------------------------------------------

extern "C" {

EMSCRIPTEN_KEEPALIVE
void *alac_open()
{
	return calloc(1, sizeof(ALACContext));
}

EMSCRIPTEN_KEEPALIVE
void alac_close(ALACContext *alac)
{
	for (int ch = 0; ch < 2; ch++) {
		free(alac->predict_error_buffer[ch]);
		free(alac->output_samples_buffer[ch]);
		free(alac->extra_bits_buffer[ch]);
	}
	for (int ch = 0; ch < ALAC_MAX_CHANNELS; ch++) {
		free(alac->output[ch]);
	}
	free(alac);
}

/** The cookie's fields, which alacConfigInfo has checked. 0 when the buffers do not fit in memory. */
EMSCRIPTEN_KEEPALIVE
int alac_configure(ALACContext *alac, uint32_t max_samples_per_frame, int sample_size, int rice_history_mult,
	int rice_initial_history, int rice_limit, int channels, int sample_rate)
{
	alac->max_samples_per_frame = max_samples_per_frame;
	alac->sample_size = sample_size;
	alac->rice_history_mult = rice_history_mult;
	alac->rice_initial_history = rice_initial_history;
	alac->rice_limit = rice_limit;
	alac->channels = channels;
	alac->sample_rate = sample_rate;

	const size_t buf_size = max_samples_per_frame * sizeof(int32_t);
	for (int ch = 0; ch < imin(channels, 2); ch++) {
		alac->predict_error_buffer[ch] = (int32_t *)malloc(buf_size);
		alac->output_samples_buffer[ch] = (int32_t *)malloc(buf_size);
		alac->extra_bits_buffer[ch] = (int32_t *)malloc(buf_size);
		if (!alac->predict_error_buffer[ch] || !alac->output_samples_buffer[ch] || !alac->extra_bits_buffer[ch]) {
			return 0;
		}
	}
	return 1;
}

/** The samples a packet gives, from its first element's header: 0 for none. `data` holds INPUT_PADDING free bytes. */
EMSCRIPTEN_KEEPALIVE
int alac_packet_samples(ALACContext *alac, uint8_t *data, int size)
{
	GetBits gb;
	memset(data + size, 0, INPUT_PADDING);
	init_get_bits8(&gb, data, size);

	const int element = get_bits(&gb, 3);
	if (element > TYPE_CPE && element != TYPE_LFE) {
		return 0;
	}
	skip_bits(&gb, 16);
	const int has_size = get_bits1(&gb);
	skip_bits(&gb, 3);
	const uint32_t samples = has_size ? get_bits(&gb, 32) : alac->max_samples_per_frame;
	return samples <= alac->max_samples_per_frame ? (int)samples : 0;
}

/**
 * Decode one whole packet. `data` holds INPUT_PADDING free bytes. info gets the sample rate, the channels and the sample
 * count. 0 when the packet gives no frame.
 */
EMSCRIPTEN_KEEPALIVE
int alac_decode_frame(ALACContext *alac, uint8_t *data, int size, int32_t *info)
{
	memset(data + size, 0, INPUT_PADDING);
	init_get_bits8(&alac->gb, data, size);

	int got_end = 0;
	int ch = 0;
	alac->nb_samples = 0;
	while (get_bits_left(&alac->gb) >= 3) {
		const int element = get_bits(&alac->gb, 3);
		if (element == TYPE_END) {
			got_end = 1;
			break;
		}
		if (element > TYPE_CPE && element != TYPE_LFE) {
			return 0;
		}

		const int channels = element == TYPE_CPE ? 2 : 1;
		if (ch + channels > alac->channels
			|| alac_channel_layout_offsets[alac->channels - 1][ch] + channels > alac->channels) {
			return 0;
		}

		const int ret = decode_element(alac, alac_channel_layout_offsets[alac->channels - 1][ch], channels);
		if (ret < 0 && get_bits_left(&alac->gb)) {
			return 0;
		}

		ch += channels;
	}
	if (!got_end || alac->channels != ch || !alac->nb_samples) {
		return 0;
	}

	info[0] = alac->sample_rate;
	info[1] = alac->channels;
	info[2] = alac->nb_samples;
	return 1;
}

EMSCRIPTEN_KEEPALIVE
float *alac_channel_ptr(ALACContext *alac, int channel)
{
	return alac->output[channel];
}

}	// extern "C"
