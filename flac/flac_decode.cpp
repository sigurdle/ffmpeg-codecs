/*
	FLAC audio decoder -- WASM C++.

	A port of FFmpeg n8.1's FLAC decoder (LGPL-2.1+): libavcodec/flacdec.c, flac.c, flacdata.c,
	flacdsp.c, flacdsp_template.c and the Rice code of golomb.h. The header parse, the subframes,
	the residuals, the predictors, the 33-bit side channel and the decorrelation are FFmpeg's; the
	libavcodec framework around them (AVCodecContext, AVFrame, the DSP pointers, threads) is replaced
	by one handle that owns planar float output.

	A frame is one whole packet: flac_decode_frame takes the bytes of one frame, as FFmpeg's decoder
	takes one from its parser or the Ogg demuxer, and skips the header packets of FLAC-in-Ogg as it
	does. Nothing carries from one frame to the next but the stream's parameters, so a seek needs no
	reset.

	Departures from FFmpeg:
	  - A FLAC file has no packets, and FFmpeg cuts it with libavcodec/flac_parser.c, which scores
	    chains of candidate headers. flac_frame_size answers the same question from one frame: the
	    frame ends at the first valid header before which the CRC-16 of the bytes comes to 0. A
	    corrupt frame, whose CRC never does, ends past the largest size its header allows at the
	    first valid header carrying the next number, or at any valid header. A last frame ends where
	    its decode does, and a last frame that does not decode holds no samples, as FFmpeg's overread
	    error drops it.
	  - Output is planar float at every depth: FFmpeg gives s16 up to 16 bits and s32 above. Depths
	    to 24 bits convert exactly; 32-bit samples round to float's 24 bits.
	  - The sample_fmt request and the use_buggy_lpc option, for streams from old lavc encoders, are
	    gone. So are the assembly paths; the C paths give the same samples.
	  - A STREAMINFO that FFmpeg's decoder refuses at init leaves the handle as one without any, and
	    the frame headers decide.

	The CRC-16 of a frame is never checked by the decode, as in FFmpeg without -err_detect.

	-fwrapv is REQUIRED: the predictors rely on two's-complement wrap (FFmpeg's SUINT).
*/

#include <cstdint>
#include <cstdlib>
#include <cstring>

#include <emscripten/emscripten.h>

#define FLAC_STREAMINFO_SIZE     34
#define FLAC_MAX_CHANNELS         8
#define FLAC_MIN_BLOCKSIZE       16
#define FLAC_MAX_BLOCKSIZE    65535
#define FLAC_MIN_FRAME_SIZE      10
// As flac_parser.c's MAX_FRAME_HEADER_SIZE.
#define MAX_FRAME_HEADER_SIZE    16
// Bytes the input must hold free past its end: every bit read loads 8.
#define INPUT_PADDING            16

enum {
	FLAC_CHMODE_INDEPENDENT = 0,
	FLAC_CHMODE_LEFT_SIDE   = 1,
	FLAC_CHMODE_RIGHT_SIDE  = 2,
	FLAC_CHMODE_MID_SIDE    = 3,
};

static const int8_t sample_size_table[] = { 0, 8, 12, 0, 16, 20, 24, 32 };

static const int ff_flac_sample_rate_table[16] = {
	0, 88200, 176400, 192000, 8000, 16000, 22050, 24000, 32000, 44100, 48000, 96000, 0, 0, 0, 0,
};

static const int32_t ff_flac_blocksize_table[16] = {
	0, 192, 576 << 0, 576 << 1, 576 << 2, 576 << 3, 0, 0,
	256 << 0, 256 << 1, 256 << 2, 256 << 3, 256 << 4, 256 << 5, 256 << 6, 256 << 7,
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

/** 1 to 57 bits. */
static inline uint64_t get_bits64(GetBits *gb, int n)
{
	const uint64_t value = show_cache(gb) >> (64 - n);
	skip_bits(gb, n);
	return value;
}

static inline unsigned get_bits(GetBits *gb, int n)
{
	return n ? (unsigned)get_bits64(gb, n) : 0;
}

static inline unsigned get_bits1(GetBits *gb)
{
	return (unsigned)get_bits64(gb, 1);
}

static inline unsigned show_bits_long(const GetBits *gb, int n)
{
	return n ? (unsigned)(show_cache(gb) >> (64 - n)) : 0;
}

static inline int get_sbits_long(GetBits *gb, int n)
{
	return n ? (int)(get_bits(gb, n) << (32 - n)) >> (32 - n) : 0;
}

static inline int64_t get_sbits64(GetBits *gb, int n)
{
	return (int64_t)(get_bits64(gb, n) << (64 - n)) >> (64 - n);
}

static inline int get_bits_left(const GetBits *gb)
{
	return gb->size_in_bits - gb->index;
}

static inline void align_get_bits(GetBits *gb)
{
	skip_bits(gb, -gb->index & 7);
}

/** The zeros before the next 1, which is read too, up to `len`: unary.h's get_unary(gb, 1, len). */
static inline int get_unary(GetBits *gb, int len)
{
	int i;
	for (i = 0; i < len && get_bits1(gb) != 1; i++);
	return i;
}

/** get_ur_golomb_jpegls with esc_len 1, as get_sr_golomb_flac calls it. */
static inline unsigned get_ur_golomb_flac(GetBits *gb, int k, int limit)
{
	const uint64_t cache = show_cache(gb);
	const int top = (int)(cache >> 32) ? __builtin_clz((unsigned)(cache >> 32)) : 32;
	if (top < 32 && top + 1 + k <= 57 && top < limit - 1) {
		const unsigned low = k ? (unsigned)((cache << (top + 1)) >> (64 - k)) : 0;
		skip_bits(gb, top + 1 + k);
		return ((unsigned)top << k) + low;
	}

	int i;
	for (i = 0; i < limit && !(show_cache(gb) >> 63); i++) {
		if (gb->size_in_bits <= gb->index) {
			return (unsigned)-1;
		}
		skip_bits(gb, 1);
	}
	skip_bits(gb, 1);

	if (i < limit - 1) {
		return get_bits(gb, k) + ((unsigned)i << k);
	}
	if (i == limit - 1) {
		return get_bits(gb, 1) + 1;
	}
	return (unsigned)-1;
}

static inline int get_sr_golomb_flac(GetBits *gb, int k, int limit)
{
	const unsigned v = get_ur_golomb_flac(gb, k, limit);
	return (int)((v >> 1) ^ -(v & 1));
}

static inline int av_log2(unsigned v)
{
	return 31 - __builtin_clz(v | 1);
}

// --- CRC: libavutil's AV_CRC_8_ATM and AV_CRC_16_ANSI, MSB first, from 0 ---------------------

static uint8_t crc8_table[256];
static uint16_t crc16_table[256];

static void init_crc_tables()
{
	if (crc16_table[1]) {
		return;
	}
	for (unsigned i = 0; i < 256; i++) {
		unsigned c8 = i;
		unsigned c16 = i << 8;
		for (int bit = 0; bit < 8; bit++) {
			c8 = (c8 << 1) ^ (c8 & 0x80 ? 0x07 : 0);
			c16 = (c16 << 1) ^ (c16 & 0x8000 ? 0x8005 : 0);
		}
		crc8_table[i] = (uint8_t)c8;
		crc16_table[i] = (uint16_t)c16;
	}
}

// --- the handle --------------------------------------------------------------------------------

struct FLACStreaminfo {
	int samplerate;
	int channels;
	int bps;
	int max_blocksize;
};

struct FLACFrameInfo {
	int samplerate;
	int channels;
	int bps;
	int blocksize;
	int ch_mode;
	int64_t frame_or_sample_num;
	int is_var_size;
};

struct FLACContext {
	FLACStreaminfo stream_info;
	GetBits gb;
	int blocksize;
	int ch_mode;
	int got_streaminfo;
	int32_t *decoded[FLAC_MAX_CHANNELS];
	int32_t *decoded_buffer;
	int64_t *decoded_33bps;
	float *output[FLAC_MAX_CHANNELS];
	float *output_buffer;
};

static int allocate_buffers(FLACContext *s)
{
	const int channels = s->stream_info.channels;
	const int blocksize = s->stream_info.max_blocksize;
	free(s->decoded_buffer);
	free(s->output_buffer);
	free(s->decoded_33bps);
	s->decoded_buffer = (int32_t *)malloc((size_t)channels * blocksize * sizeof(int32_t));
	s->output_buffer = (float *)malloc((size_t)channels * blocksize * sizeof(float));
	s->decoded_33bps = nullptr;
	if (!s->decoded_buffer || !s->output_buffer) {
		return 0;
	}
	for (int c = 0; c < channels; c++) {
		s->decoded[c] = s->decoded_buffer + c * blocksize;
		s->output[c] = s->output_buffer + c * blocksize;
	}
	if (s->stream_info.bps == 32 && channels == 2) {
		s->decoded_33bps = (int64_t *)malloc((size_t)blocksize * sizeof(int64_t));
		if (!s->decoded_33bps) {
			return 0;
		}
	}
	return 1;
}

// --- flac.c ------------------------------------------------------------------------------------

static int64_t get_utf8(GetBits *gb)
{
	int64_t val = get_bits(gb, 8);
	uint32_t top = (val & 128) >> 1;
	if ((val & 0xc0) == 0x80 || val >= 0xFE) {
		return -1;
	}
	while (val & top) {
		const unsigned tmp = get_bits(gb, 8) - 128;
		if (tmp >> 6) {
			return -1;
		}
		val = (val << 6) + tmp;
		top <<= 5;
	}
	val &= (top << 1) - 1;
	return val;
}

static int decode_frame_header(GetBits *gb, FLACFrameInfo *fi)
{
	if ((get_bits(gb, 15) & 0x7FFF) != 0x7FFC) {
		return -1;
	}

	fi->is_var_size = get_bits1(gb);

	const int bs_code = get_bits(gb, 4);
	const int sr_code = get_bits(gb, 4);

	fi->ch_mode = get_bits(gb, 4);
	if (fi->ch_mode < FLAC_MAX_CHANNELS) {
		fi->channels = fi->ch_mode + 1;
		fi->ch_mode = FLAC_CHMODE_INDEPENDENT;
	} else if (fi->ch_mode < FLAC_MAX_CHANNELS + FLAC_CHMODE_MID_SIDE) {
		fi->channels = 2;
		fi->ch_mode -= FLAC_MAX_CHANNELS - 1;
	} else {
		return -1;
	}

	const int bps_code = get_bits(gb, 3);
	if (bps_code == 3) {
		return -1;
	}
	fi->bps = sample_size_table[bps_code];

	if (get_bits1(gb)) {
		return -1;
	}

	fi->frame_or_sample_num = get_utf8(gb);
	if (fi->frame_or_sample_num < 0) {
		return -1;
	}

	if (bs_code == 0) {
		return -1;
	} else if (bs_code == 6) {
		fi->blocksize = get_bits(gb, 8) + 1;
	} else if (bs_code == 7) {
		fi->blocksize = get_bits(gb, 16) + 1;
	} else {
		fi->blocksize = ff_flac_blocksize_table[bs_code];
	}

	if (sr_code < 12) {
		fi->samplerate = ff_flac_sample_rate_table[sr_code];
	} else if (sr_code == 12) {
		fi->samplerate = get_bits(gb, 8) * 1000;
	} else if (sr_code == 13) {
		fi->samplerate = get_bits(gb, 16);
	} else if (sr_code == 14) {
		fi->samplerate = get_bits(gb, 16) * 10;
	} else {
		return -1;
	}

	skip_bits(gb, 8);
	unsigned crc = 0;
	for (int i = 0, n = gb->index / 8; i < n; i++) {
		crc = crc8_table[crc ^ gb->buffer[i]];
	}
	return crc ? -1 : 0;
}

/** The 34 bytes of a STREAMINFO: 0 for one FFmpeg refuses. */
static int parse_streaminfo(FLACStreaminfo *s, const uint8_t *buffer)
{
	GetBits gb;
	init_get_bits8(&gb, buffer, FLAC_STREAMINFO_SIZE);

	skip_bits(&gb, 16);
	const int max_blocksize = get_bits(&gb, 16);
	skip_bits(&gb, 24 + 24);
	const int samplerate = get_bits(&gb, 20);
	const int channels = get_bits(&gb, 3) + 1;
	const int bps = get_bits(&gb, 5) + 1;
	if (max_blocksize < FLAC_MIN_BLOCKSIZE || bps < 4) {
		return 0;
	}
	s->max_blocksize = max_blocksize;
	s->samplerate = samplerate;
	s->channels = channels;
	s->bps = bps;
	return 1;
}

// --- flacdsp.c ---------------------------------------------------------------------------------

static void flac_lpc_16_c(int32_t *decoded, const int coeffs[32], int pred_order, int qlevel, int len)
{
	int i, j;

	for (i = pred_order; i < len - 1; i += 2, decoded += 2) {
		int c = coeffs[0];
		int d = decoded[0];
		int s0 = 0, s1 = 0;
		for (j = 1; j < pred_order; j++) {
			s0 += c * d;
			d = decoded[j];
			s1 += c * d;
			c = coeffs[j];
		}
		s0 += c * d;
		d = decoded[j] += s0 >> qlevel;
		s1 += c * d;
		decoded[j + 1] += s1 >> qlevel;
	}
	if (i < len) {
		int sum = 0;
		for (j = 0; j < pred_order; j++) {
			sum += coeffs[j] * decoded[j];
		}
		decoded[j] = decoded[j] + (sum >> qlevel);
	}
}

static void flac_lpc_32_c(int32_t *decoded, const int coeffs[32], int pred_order, int qlevel, int len)
{
	int i, j;

	for (i = pred_order; i < len; i++, decoded++) {
		int64_t sum = 0;
		for (j = 0; j < pred_order; j++) {
			sum += (int64_t)coeffs[j] * decoded[j];
		}
		decoded[j] += sum >> qlevel;
	}
}

static void flac_lpc_33_c(int64_t *decoded, const int32_t *residual, const int coeffs[32], int pred_order, int qlevel, int len)
{
	int i, j;

	for (i = pred_order; i < len; i++, decoded++) {
		int64_t sum = 0;
		for (j = 0; j < pred_order; j++) {
			sum += (int64_t)coeffs[j] * (uint64_t)decoded[j];
		}
		decoded[j] = (uint64_t)residual[i] + (uint64_t)(sum >> qlevel);
	}
}

// --- flacdec.c ---------------------------------------------------------------------------------

static int decode_residuals(FLACContext *s, int32_t *decoded, int pred_order)
{
	GetBits gb = s->gb;

	const int method_type = get_bits(&gb, 2);
	const int rice_order = get_bits(&gb, 4);

	const int samples = s->blocksize >> rice_order;
	const int rice_bits = 4 + method_type;
	const int rice_esc = (1 << rice_bits) - 1;

	decoded += pred_order;
	int i = pred_order;

	if (method_type > 1) {
		return -1;
	}
	if (samples << rice_order != s->blocksize) {
		return -1;
	}
	if (pred_order > samples) {
		return -1;
	}

	for (int partition = 0; partition < (1 << rice_order); partition++) {
		int tmp = get_bits(&gb, rice_bits);
		if (tmp == rice_esc) {
			tmp = get_bits(&gb, 5);
			for (; i < samples; i++) {
				*decoded++ = get_sbits_long(&gb, tmp);
			}
		} else {
			const int real_limit = (tmp > 1) ? (INT32_MAX >> (tmp - 1)) + 2 : INT32_MAX;
			for (; i < samples; i++) {
				const int v = get_sr_golomb_flac(&gb, tmp, real_limit);
				if (v == INT32_MIN) {
					return -1;
				}
				*decoded++ = v;
			}
		}
		i = 0;
	}

	s->gb = gb;
	return 0;
}

static int decode_subframe_fixed(FLACContext *s, int32_t *decoded, int pred_order, int bps)
{
	const int blocksize = s->blocksize;
	unsigned a = 0, b = 0, c = 0, d = 0;
	int i;

	for (i = 0; i < pred_order; i++) {
		decoded[i] = get_sbits_long(&s->gb, bps);
	}

	if (decode_residuals(s, decoded, pred_order) < 0) {
		return -1;
	}

	if (pred_order > 0) {
		a = decoded[pred_order - 1];
	}
	if (pred_order > 1) {
		b = a - decoded[pred_order - 2];
	}
	if (pred_order > 2) {
		c = b - decoded[pred_order - 2] + decoded[pred_order - 3];
	}
	if (pred_order > 3) {
		d = c - decoded[pred_order - 2] + 2U * decoded[pred_order - 3] - decoded[pred_order - 4];
	}

	switch (pred_order) {
	case 0:
		break;
	case 1:
		for (i = pred_order; i < blocksize; i++) {
			decoded[i] = a += decoded[i];
		}
		break;
	case 2:
		for (i = pred_order; i < blocksize; i++) {
			decoded[i] = a += b += decoded[i];
		}
		break;
	case 3:
		for (i = pred_order; i < blocksize; i++) {
			decoded[i] = a += b += c += decoded[i];
		}
		break;
	case 4:
		for (i = pred_order; i < blocksize; i++) {
			decoded[i] = a += b += c += d += decoded[i];
		}
		break;
	default:
		return -1;
	}
	return 0;
}

/** DECODER_SUBFRAME_FIXED_WIDE: an order whose running sums pass 32 bits, or 33-bit samples. */
template <typename Sample>
static int fixed_wide(FLACContext *s, Sample *decoded, int32_t *residual, int pred_order)
{
	const int blocksize = s->blocksize;

	if (decode_residuals(s, residual, pred_order) < 0) {
		return -1;
	}

	switch (pred_order) {
	case 0:
		for (int i = pred_order; i < blocksize; i++) {
			decoded[i] = residual[i];
		}
		break;
	case 1:
		for (int i = pred_order; i < blocksize; i++) {
			decoded[i] = (uint64_t)residual[i] + (uint64_t)decoded[i - 1];
		}
		break;
	case 2:
		for (int i = pred_order; i < blocksize; i++) {
			decoded[i] = (uint64_t)residual[i] + 2 * (uint64_t)decoded[i - 1] - (uint64_t)decoded[i - 2];
		}
		break;
	case 3:
		for (int i = pred_order; i < blocksize; i++) {
			decoded[i] = (uint64_t)residual[i] + 3 * (uint64_t)decoded[i - 1] - 3 * (uint64_t)decoded[i - 2] + (uint64_t)decoded[i - 3];
		}
		break;
	case 4:
		for (int i = pred_order; i < blocksize; i++) {
			decoded[i] = (uint64_t)residual[i] + 4 * (uint64_t)decoded[i - 1] - 6 * (uint64_t)decoded[i - 2] + 4 * (uint64_t)decoded[i - 3] - (uint64_t)decoded[i - 4];
		}
		break;
	default:
		return -1;
	}
	return 0;
}

static int decode_subframe_fixed_wide(FLACContext *s, int32_t *decoded, int pred_order, int bps)
{
	for (int i = 0; i < pred_order; i++) {
		decoded[i] = get_sbits_long(&s->gb, bps);
	}
	return fixed_wide(s, decoded, decoded, pred_order);
}

static int decode_subframe_fixed_33bps(FLACContext *s, int64_t *decoded, int32_t *residual, int pred_order)
{
	for (int i = 0; i < pred_order; i++) {
		decoded[i] = get_sbits64(&s->gb, 33);
	}
	return fixed_wide(s, decoded, residual, pred_order);
}

static void lpc_analyze_remodulate(int32_t *decoded, const int coeffs[32], int order, int qlevel, int len, int bps)
{
	int i, j;
	const int ebps = 1 << (bps - 1);
	unsigned sigma = 0;

	for (i = order; i < len; i++) {
		sigma |= decoded[i] + ebps;
	}

	if (sigma < 2U * ebps) {
		return;
	}

	for (i = len - 1; i >= order; i--) {
		int64_t p = 0;
		for (j = 0; j < order; j++) {
			p += coeffs[j] * (int64_t)decoded[i - order + j];
		}
		decoded[i] -= p >> qlevel;
	}
	for (i = order; i < len; i++, decoded++) {
		int32_t p = 0;
		for (j = 0; j < order; j++) {
			p += coeffs[j] * (uint32_t)decoded[j];
		}
		decoded[j] += p >> qlevel;
	}
}

/** What decode_subframe_lpc and its 33-bit form read before the residuals. */
static int read_lpc_header(FLACContext *s, int coeffs[32], int pred_order, int *qlevel)
{
	const int coeff_prec = get_bits(&s->gb, 4) + 1;
	if (coeff_prec == 16) {
		return -1;
	}
	*qlevel = get_sbits_long(&s->gb, 5);
	if (*qlevel < 0) {
		return -1;
	}
	for (int i = 0; i < pred_order; i++) {
		coeffs[pred_order - i - 1] = get_sbits_long(&s->gb, coeff_prec);
	}
	return coeff_prec;
}

static int decode_subframe_lpc(FLACContext *s, int32_t *decoded, int pred_order, int bps)
{
	int qlevel;
	int coeffs[32];

	for (int i = 0; i < pred_order; i++) {
		decoded[i] = get_sbits_long(&s->gb, bps);
	}

	const int coeff_prec = read_lpc_header(s, coeffs, pred_order, &qlevel);
	if (coeff_prec < 0 || decode_residuals(s, decoded, pred_order) < 0) {
		return -1;
	}

	if (bps <= 16 && bps + coeff_prec + av_log2(pred_order) <= 32) {
		flac_lpc_16_c(decoded, coeffs, pred_order, qlevel, s->blocksize);
	} else {
		flac_lpc_32_c(decoded, coeffs, pred_order, qlevel, s->blocksize);
		if (s->stream_info.bps <= 16) {
			lpc_analyze_remodulate(decoded, coeffs, pred_order, qlevel, s->blocksize, bps);
		}
	}
	return 0;
}

static int decode_subframe_lpc_33bps(FLACContext *s, int64_t *decoded, int32_t *residual, int pred_order)
{
	int qlevel;
	int coeffs[32];

	for (int i = 0; i < pred_order; i++) {
		decoded[i] = get_sbits64(&s->gb, 33);
	}

	if (read_lpc_header(s, coeffs, pred_order, &qlevel) < 0 || decode_residuals(s, residual, pred_order) < 0) {
		return -1;
	}

	flac_lpc_33_c(decoded, residual, coeffs, pred_order, qlevel, s->blocksize);
	return 0;
}

static inline int decode_subframe(FLACContext *s, int channel)
{
	int32_t *decoded = s->decoded[channel];
	int wasted = 0;
	int bps = s->stream_info.bps;
	int i;

	if (channel == 0) {
		if (s->ch_mode == FLAC_CHMODE_RIGHT_SIDE) {
			bps++;
		}
	} else {
		if (s->ch_mode == FLAC_CHMODE_LEFT_SIDE || s->ch_mode == FLAC_CHMODE_MID_SIDE) {
			bps++;
		}
	}

	if (get_bits1(&s->gb)) {
		return -1;
	}
	const int type = get_bits(&s->gb, 6);

	if (get_bits1(&s->gb)) {
		const int left = get_bits_left(&s->gb);
		if (left <= 0 || (left < bps && !show_bits_long(&s->gb, left)) || !show_bits_long(&s->gb, bps - 1)) {
			return -1;
		}
		wasted = 1 + get_unary(&s->gb, get_bits_left(&s->gb));
		bps -= wasted;
	}

	if (type == 0) {
		if (bps < 33) {
			const int32_t tmp = get_sbits_long(&s->gb, bps);
			for (i = 0; i < s->blocksize; i++) {
				decoded[i] = tmp;
			}
		} else {
			const int64_t tmp = get_sbits64(&s->gb, 33);
			for (i = 0; i < s->blocksize; i++) {
				s->decoded_33bps[i] = tmp;
			}
		}
	} else if (type == 1) {
		if (bps < 33) {
			for (i = 0; i < s->blocksize; i++) {
				decoded[i] = get_sbits_long(&s->gb, bps);
			}
		} else {
			for (i = 0; i < s->blocksize; i++) {
				s->decoded_33bps[i] = get_sbits64(&s->gb, 33);
			}
		}
	} else if (type >= 8 && type <= 12) {
		const int order = type & ~0x8;
		if (bps < 33) {
			if (bps + order <= 32) {
				if (decode_subframe_fixed(s, decoded, order, bps) < 0) {
					return -1;
				}
			} else if (decode_subframe_fixed_wide(s, decoded, order, bps) < 0) {
				return -1;
			}
		} else if (decode_subframe_fixed_33bps(s, s->decoded_33bps, decoded, order) < 0) {
			return -1;
		}
	} else if (type >= 32) {
		if (bps < 33) {
			if (decode_subframe_lpc(s, decoded, (type & ~0x20) + 1, bps) < 0) {
				return -1;
			}
		} else if (decode_subframe_lpc_33bps(s, s->decoded_33bps, decoded, (type & ~0x20) + 1) < 0) {
			return -1;
		}
	} else {
		return -1;
	}

	if (wasted) {
		if (wasted + bps == 33) {
			// flac_wasted_33_c
			for (i = 0; i < s->blocksize; i++) {
				s->decoded_33bps[i] = (uint64_t)decoded[i] << wasted;
			}
		} else if (wasted < 32) {
			// flac_wasted_32_c
			for (i = 0; i < s->blocksize; i++) {
				decoded[i] = (unsigned)decoded[i] << wasted;
			}
		}
	}

	return 0;
}

static int decode_frame(FLACContext *s)
{
	GetBits *gb = &s->gb;
	FLACFrameInfo fi;

	if (decode_frame_header(gb, &fi) < 0) {
		return -1;
	}

	if (s->stream_info.channels && fi.channels != s->stream_info.channels && s->got_streaminfo) {
		s->stream_info.channels = fi.channels;
		if (!allocate_buffers(s)) {
			return -1;
		}
	}
	s->stream_info.channels = fi.channels;
	s->ch_mode = fi.ch_mode;

	if (!s->stream_info.bps && !fi.bps) {
		return -1;
	}
	if (!fi.bps) {
		fi.bps = s->stream_info.bps;
	} else if (s->stream_info.bps && fi.bps != s->stream_info.bps) {
		return -1;
	}

	if (!s->stream_info.bps) {
		s->stream_info.bps = fi.bps;
	}

	if (!s->stream_info.max_blocksize) {
		s->stream_info.max_blocksize = FLAC_MAX_BLOCKSIZE;
	}
	if (fi.blocksize > s->stream_info.max_blocksize) {
		return -1;
	}
	s->blocksize = fi.blocksize;

	if (!s->stream_info.samplerate && !fi.samplerate) {
		return -1;
	}
	if (fi.samplerate == 0) {
		fi.samplerate = s->stream_info.samplerate;
	}
	s->stream_info.samplerate = fi.samplerate;

	// A handle given a STREAMINFO allocates here too, at its first frame: a scan never comes this far.
	if (!s->got_streaminfo || !s->decoded_buffer) {
		if (!allocate_buffers(s)) {
			return -1;
		}
		s->got_streaminfo = 1;
	}

	for (int i = 0; i < s->stream_info.channels; i++) {
		if (decode_subframe(s, i) < 0) {
			return -1;
		}
	}

	align_get_bits(gb);
	skip_bits(gb, 16);	// the data CRC

	return 0;
}

static void decorrelate_33bps(int ch_mode, int32_t **decoded, const int64_t *decoded_33bps, int len)
{
	if (ch_mode == FLAC_CHMODE_LEFT_SIDE) {
		for (int i = 0; i < len; i++) {
			decoded[1][i] = decoded[0][i] - (uint64_t)decoded_33bps[i];
		}
	} else if (ch_mode == FLAC_CHMODE_RIGHT_SIDE) {
		for (int i = 0; i < len; i++) {
			decoded[0][i] = decoded[1][i] + (uint64_t)decoded_33bps[i];
		}
	} else if (ch_mode == FLAC_CHMODE_MID_SIDE) {
		for (int i = 0; i < len; i++) {
			uint64_t a = decoded[0][i];
			const int64_t b = decoded_33bps[i];
			a -= b >> 1;
			decoded[0][i] = (a + b);
			decoded[1][i] = a;
		}
	}
}

/** One sample, left-aligned in 32 bits as FFmpeg's s32, as float. */
static inline float sample_to_float(unsigned left_aligned)
{
	return (float)(int32_t)left_aligned * (1.0f / 2147483648.0f);
}

/** flacdsp_template.c's decorrelators, into the float output. */
static void decorrelate(FLACContext *s, int ch_mode)
{
	const int shift = 32 - s->stream_info.bps;
	const int len = s->blocksize;
	int32_t **in = s->decoded;
	float **out = s->output;

	switch (ch_mode) {
	case FLAC_CHMODE_INDEPENDENT:
		for (int c = 0; c < s->stream_info.channels; c++) {
			for (int i = 0; i < len; i++) {
				out[c][i] = sample_to_float((unsigned)in[c][i] << shift);
			}
		}
		break;
	case FLAC_CHMODE_LEFT_SIDE:
		for (int i = 0; i < len; i++) {
			const unsigned a = in[0][i];
			const unsigned b = in[1][i];
			out[0][i] = sample_to_float(a << shift);
			out[1][i] = sample_to_float((a - b) << shift);
		}
		break;
	case FLAC_CHMODE_RIGHT_SIDE:
		for (int i = 0; i < len; i++) {
			const unsigned a = in[0][i];
			const unsigned b = in[1][i];
			out[0][i] = sample_to_float((a + b) << shift);
			out[1][i] = sample_to_float(b << shift);
		}
		break;
	case FLAC_CHMODE_MID_SIDE:
		for (int i = 0; i < len; i++) {
			unsigned a = in[0][i];
			const int b = in[1][i];
			a -= b >> 1;
			out[0][i] = sample_to_float((a + b) << shift);
			out[1][i] = sample_to_float(a << shift);
		}
		break;
	}
}

/** flac_decode_frame: the samples of the frame `buf` holds, 0 for none, with `bytes_read` set when it decoded. */
static int decode_packet(FLACContext *s, const uint8_t *buf, int buf_size, int *bytes_read)
{
	if (buf_size > 5 && !memcmp(buf, "\177FLAC", 5)) {
		return 0;
	}
	if (buf_size > 0 && (*buf & 0x7F) == 4) {	// FLAC_METADATA_TYPE_VORBIS_COMMENT
		return 0;
	}
	if (buf_size < FLAC_MIN_FRAME_SIZE) {
		return 0;
	}
	if (!memcmp(buf, "fLaC", 4)) {
		if (!s->got_streaminfo && buf_size >= FLAC_STREAMINFO_SIZE + 8 && (buf[4] & 0x7F) == 0
			&& ((buf[5] << 16) | (buf[6] << 8) | buf[7]) == FLAC_STREAMINFO_SIZE && parse_streaminfo(&s->stream_info, buf + 8)) {
			s->got_streaminfo = 1;
		}
		return 0;
	}

	init_get_bits8(&s->gb, buf, buf_size);
	if (decode_frame(s) < 0) {
		return 0;
	}
	*bytes_read = s->gb.index / 8;

	if (s->stream_info.bps == 32 && s->ch_mode > 0) {
		decorrelate_33bps(s->ch_mode, s->decoded, s->decoded_33bps, s->blocksize);
		decorrelate(s, FLAC_CHMODE_INDEPENDENT);
	} else {
		decorrelate(s, s->ch_mode);
	}

	if (*bytes_read > buf_size) {
		return 0;
	}
	return s->blocksize;
}

// --- the frame's end, in place of flac_parser.c --------------------------------------------------

/** No frame of this header is longer: every sample verbatim at its widest, and as much again for a poor encoder. */
static int largest_frame_bytes(const FLACContext *s, const FLACFrameInfo *fi)
{
	const int bps = (fi->bps ? fi->bps : s->stream_info.bps ? s->stream_info.bps : 32) + 1;
	return MAX_FRAME_HEADER_SIZE + 2 + 2 * fi->channels * (2 + (int)(((int64_t)fi->blocksize * bps + 7) / 8));
}

/** The samples a frame of this header gives, 0 for one decode_frame refuses. */
static int playable_samples(const FLACContext *s, const FLACFrameInfo *fi)
{
	const FLACStreaminfo *info = &s->stream_info;
	if ((info->bps && fi->bps && fi->bps != info->bps) || (!info->bps && !fi->bps)
		|| (info->max_blocksize && fi->blocksize > info->max_blocksize) || (!info->samplerate && !fi->samplerate)) {
		return 0;
	}
	return fi->blocksize;
}

/**
 * The bytes of the frame `buf` opens with: >0 its size, 0 when more input is needed, -1 when no frame opens here. `*samples`
 * is what the frame gives. `flushed` says no input follows `size`.
 */
static int frame_size(FLACContext *s, const uint8_t *buf, int size, int flushed, int *samples)
{
	if (size < MAX_FRAME_HEADER_SIZE && !flushed) {
		return 0;
	}

	GetBits gb;
	FLACFrameInfo fi, candidate;
	init_get_bits8(&gb, buf, size);
	if (decode_frame_header(&gb, &fi) < 0) {
		return -1;
	}
	*samples = playable_samples(s, &fi);

	const int64_t next_number = fi.frame_or_sample_num + (fi.is_var_size ? fi.blocksize : 1);
	const int largest = largest_frame_bytes(s, &fi);
	int same_number = 0, any_header = 0;
	unsigned crc = 0;

	for (int pos = 0; pos < size; pos++) {
		const uint8_t byte = buf[pos];
		if (byte == 0xFF && pos >= FLAC_MIN_FRAME_SIZE && pos + 1 < size && (buf[pos + 1] & 0xFE) == 0xF8) {
			if (pos + MAX_FRAME_HEADER_SIZE > size && !flushed) {
				return 0;
			}
			init_get_bits8(&gb, buf + pos, size - pos);
			if (decode_frame_header(&gb, &candidate) == 0) {
				if (crc == 0) {
					return pos;
				}
				if (!same_number && candidate.frame_or_sample_num == next_number) {
					same_number = pos;
				}
				if (!any_header) {
					any_header = pos;
				}
			}
		}
		if (pos > largest && any_header) {
			return same_number ? same_number : any_header;
		}
		crc = ((crc << 8) & 0xFFFF) ^ crc16_table[(crc >> 8) ^ byte];
	}

	if (!flushed) {
		return 0;
	}
	if (any_header) {
		return same_number ? same_number : any_header;
	}
	int bytes_read = 0;
	*samples = decode_packet(s, buf, size, &bytes_read);
	return size;
}

// --- exports ---------------------------------------------------------------------------------

extern "C" {

EMSCRIPTEN_KEEPALIVE FLACContext *flac_open()
{
	init_crc_tables();
	return (FLACContext *)calloc(1, sizeof(FLACContext));
}

EMSCRIPTEN_KEEPALIVE void flac_close(FLACContext *s)
{
	free(s->decoded_buffer);
	free(s->output_buffer);
	free(s->decoded_33bps);
	free(s);
}

/** The 34 bytes of a STREAMINFO. */
EMSCRIPTEN_KEEPALIVE int flac_set_streaminfo(FLACContext *s, const uint8_t *streaminfo)
{
	if (!parse_streaminfo(&s->stream_info, streaminfo)) {
		return 0;
	}
	s->got_streaminfo = 1;
	return 1;
}

/** `buf` holds INPUT_PADDING free bytes past `size`. info: [samples] */
EMSCRIPTEN_KEEPALIVE int flac_frame_size(FLACContext *s, uint8_t *buf, int size, int flushed, int *info)
{
	memset(buf + size, 0, INPUT_PADDING);
	info[0] = 0;
	return frame_size(s, buf, size, flushed, &info[0]);
}

/** The samples of the whole packet `buf`, from its header alone: 0 for a header packet. `buf` as flac_frame_size's. */
EMSCRIPTEN_KEEPALIVE int flac_packet_samples(FLACContext *s, uint8_t *buf, int size)
{
	memset(buf + size, 0, INPUT_PADDING);
	GetBits gb;
	FLACFrameInfo fi;
	init_get_bits8(&gb, buf, size);
	return size >= FLAC_MIN_FRAME_SIZE && decode_frame_header(&gb, &fi) == 0 ? playable_samples(s, &fi) : 0;
}

/** One whole frame; `buf` holds INPUT_PADDING free bytes past `size`. info: [sampleRate, channels, samples] */
EMSCRIPTEN_KEEPALIVE int flac_decode_frame(FLACContext *s, uint8_t *buf, int size, int *info)
{
	memset(buf + size, 0, INPUT_PADDING);
	int bytes_read = 0;
	const int samples = decode_packet(s, buf, size, &bytes_read);
	info[0] = s->stream_info.samplerate;
	info[1] = s->stream_info.channels;
	info[2] = samples;
	return samples > 0;
}

EMSCRIPTEN_KEEPALIVE float *flac_channel_ptr(FLACContext *s, int channel)
{
	return s->output[channel];
}

}	// extern "C"
