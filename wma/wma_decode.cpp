/*
	WMA v1 and v2 audio decoder -- WASM C++.

	A direct port of FFmpeg's WMA decoder (libavcodec, tag n9.0.2):
	  wmadec.c / wma.c / wma.h / wma_common.c / wma_freqs.c / wmadata.h / sinewin_tablegen.h
	  + the inverse MDCT of libavutil/tx_template.c and the three loops of libavutil/float_dsp.c it uses

	This file is part of a derivative of FFmpeg. FFmpeg is free software; you can
	redistribute it and/or modify it under the terms of the GNU Lesser General Public
	License as published by the Free Software Foundation; either version 2.1 of the
	License, or (at your option) any later version. It is distributed in the hope that it
	will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
	License for more details. Copyright (c) 2002 The FFmpeg Project.

	Scope -- WAVEFORMATEX tags 0x0160 (v1) and 0x0161 (v2), one or two channels, up to 50000 Hz,
	float output: superframes and the bit reservoir, variable block lengths, LSP and VLC
	exponents, noise coding, mid/side stereo. WMA Pro, Lossless and Voice are other codecs.

	Differences from FFmpeg, all behavior-preserving for that scope:
	  - The surrounding framework (AVCodecContext / AVFrame / the DSP function pointers) is
	    replaced by a self-contained handle; a packet's frames go to the handle's own planar
	    buffer.
	  - The inverse MDCT runs its own complex FFT, in double, in place of libavutil/tx; the
	    twiddles and the pre- and post-rotation follow ff_tx_mdct_gen_exp and ff_tx_mdct_inv, and
	    wma_test_imdct checks the result against a direct O(N^2) evaluation of the definition.
	    FFmpeg computes in float, so the PCM differs from `ffmpeg -f f32le` by float round-off
	    (tests/WMA.test.js).
	  - The VLCs are built here as two-level tables rather than by vlc_init; an invalid code
	    reads as -1, as get_vlc2's, and is taken where FFmpeg takes it.
	  - The bit reader clamps at the end of the buffer, as FFmpeg's checked reader does.

	Seek: wma_reset clears what ties a packet to the packets before it (the bit reservoir, the
	overlap, the block lengths, the exponents a short block may reuse) and places the noise
	generator. FFmpeg's flush clears only the reservoir.

	Boundary (extern "C", prefixed _ on the JS side; see wma_decode_core.js):
	  WMAContext* wma_open(void)
	  void    wma_close(WMAContext*)
	  int     wma_configure(WMAContext*, int version, int channels, int sampleRate, int bitRate,
	            int blockAlign, uint8_t* extradata, int extradataSize, int info[6])
	            info = {frameLen, useBitReservoir, useNoiseCoding, useExpVlc, useVariableBlockLen,
	            nbBlockSizes}; 1, 0 when FFmpeg's init refuses the parameters
	  void    wma_reset(WMAContext*, int noiseIndex)
	  int     wma_decode_packet(WMAContext*, uint8_t* data)   one block_align packet; the samples
	            per channel it gave (0 for a packet that only fills the reservoir), -1 on error.
	            `data` has room for WMA_INPUT_PADDING bytes past block_align.
	  int     wma_drain(WMAContext*)          the tail FFmpeg gives at the end of the stream
	  float*  wma_channel_ptr(WMAContext*, int ch)
	  int     wma_test_imdct(int n)           max error x 1e9 against the definition
*/

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define FFMIN(a, b) ((a) > (b) ? (b) : (a))

/* size of blocks */
#define BLOCK_MIN_BITS 7
#define BLOCK_MAX_BITS 11
#define BLOCK_MAX_SIZE (1 << BLOCK_MAX_BITS)
#define BLOCK_NB_SIZES (BLOCK_MAX_BITS - BLOCK_MIN_BITS + 1)
#define HIGH_BAND_MAX_SIZE 16
#define NB_LSP_COEFS 10
#define MAX_CODED_SUPERFRAME_SIZE 32768
#define MAX_CHANNELS 2
#define NOISE_TAB_SIZE 8192
#define LSP_POW_BITS 7
#define VLCBITS 9
#define EXPVLCBITS 8
#define HGAINVLCBITS 9
#define MIN_CACHE_BITS 25
#define AV_INPUT_BUFFER_PADDING_SIZE 64
#define WMA_INPUT_PADDING 16
/** The most frames one superframe header can count. */
#define MAX_SUPERFRAME_FRAMES 15

#include "wma_tables.h"

///////////////////////////////////////////////////////////////////////////////
// Bit reader. Top-aligned, MSB first; every buffer read holds padding zero bytes past the end, and
// the index clamps there, so an overread answers zeros, as FFmpeg's checked reader.

struct GetBitContext {
	const uint8_t* buffer;
	int index;
	int size_in_bits;
	int size_in_bits_plus8;
};

static inline void init_get_bits(GetBitContext* s, const uint8_t* buffer, int bit_size)
{
	s->buffer = buffer;
	s->index = 0;
	s->size_in_bits = bit_size;
	s->size_in_bits_plus8 = bit_size + 8;
}

static inline int get_bits_count(const GetBitContext* s) { return s->index; }
static inline int get_bits_left(const GetBitContext* s) { return s->size_in_bits - s->index; }

static inline uint32_t show_cache(const GetBitContext* s)
{
	const uint8_t* p = s->buffer + (s->index >> 3);
	const uint64_t v = ((uint64_t)p[0] << 32) | ((uint64_t)p[1] << 24) | ((uint64_t)p[2] << 16) | ((uint64_t)p[3] << 8) | p[4];
	return (uint32_t)(v >> (8 - (s->index & 7)));
}

static inline void skip_bits(GetBitContext* s, int n)
{
	s->index = FFMIN(s->index + n, s->size_in_bits_plus8);
}
static inline uint32_t show_bits(const GetBitContext* s, int n) { return n ? show_cache(s) >> (32 - n) : 0; }

static inline uint32_t get_bits(GetBitContext* s, int n)
{
	const uint32_t v = show_bits(s, n);
	skip_bits(s, n);
	return v;
}
static inline uint32_t get_bits1(GetBitContext* s) { return get_bits(s, 1); }
static inline void align_get_bits(GetBitContext* s)
{
	const int n = -get_bits_count(s) & 7;
	if (n) skip_bits(s, n);
}

///////////////////////////////////////////////////////////////////////////////
// VLC: two levels, the first indexed by `bits` bits. A first-level entry with a negative length links
// to a subtable of -length bits at pool offset sym. Length 0 is an invalid code. Tables live in one
// growing pool, so a VLC holds its offset.

struct VLCElem { int32_t sym; int16_t len; };
struct VLC { int base; int bits; };

static VLCElem* g_vlc_pool;
static int g_vlc_capacity;
static int g_vlc_used;

static int vlc_reserve(int count)
{
	if (g_vlc_used + count > g_vlc_capacity) {
		int capacity = g_vlc_capacity ? g_vlc_capacity : 1 << 16;
		while (g_vlc_used + count > capacity) capacity <<= 1;
		VLCElem* pool = (VLCElem*)realloc(g_vlc_pool, capacity * sizeof(VLCElem));
		if (!pool) return -1;
		g_vlc_pool = pool;
		g_vlc_capacity = capacity;
	}
	const int base = g_vlc_used;
	memset(g_vlc_pool + base, 0, count * sizeof(VLCElem));
	g_vlc_used += count;
	return base;
}

/** `code(i)`, `len(i)`, `sym(i)` for n codes. Returns 0 when out of memory. */
template <class CodeAt, class LenAt, class SymAt>
static int vlc_build(VLC* vlc, int bits, int n, CodeAt code, LenAt len, SymAt sym)
{
	const int size = 1 << bits;
	const int base = vlc_reserve(size);
	if (base < 0) return 0;

	for (int i = 0; i < n; i++) {
		const int l = len(i);
		if (!l || l > bits) continue;
		const uint32_t first = (uint32_t)code(i) << (bits - l);
		for (uint32_t k = 0; k < (1u << (bits - l)); k++) {
			g_vlc_pool[base + first + k].sym = sym(i);
			g_vlc_pool[base + first + k].len = (int16_t)l;
		}
	}

	// The width of each prefix's subtable is the longest code below it.
	for (int i = 0; i < n; i++) {
		const int l = len(i);
		if (l <= bits) continue;
		const uint32_t prefix = (uint32_t)code(i) >> (l - bits);
		if (g_vlc_pool[base + prefix].len > -(l - bits)) g_vlc_pool[base + prefix].len = (int16_t)-(l - bits);
	}
	for (int prefix = 0; prefix < size; prefix++) {
		if (g_vlc_pool[base + prefix].len >= 0) continue;
		const int sub = vlc_reserve(1 << -g_vlc_pool[base + prefix].len);
		if (sub < 0) return 0;
		g_vlc_pool[base + prefix].sym = sub;
	}
	for (int i = 0; i < n; i++) {
		const int l = len(i);
		if (l <= bits) continue;
		const uint32_t c = (uint32_t)code(i);
		const VLCElem link = g_vlc_pool[base + (c >> (l - bits))];
		const int width = -link.len;
		const int rest = l - bits;
		const uint32_t first = (c & ((1u << rest) - 1)) << (width - rest);
		for (uint32_t k = 0; k < (1u << (width - rest)); k++) {
			g_vlc_pool[link.sym + first + k].sym = sym(i);
			g_vlc_pool[link.sym + first + k].len = (int16_t)rest;
		}
	}

	vlc->base = base;
	vlc->bits = bits;
	return 1;
}

static inline int get_vlc(GetBitContext* gb, const VLC* vlc)
{
	const uint32_t cache = show_cache(gb);
	VLCElem e = g_vlc_pool[vlc->base + (cache >> (32 - vlc->bits))];
	int consumed = vlc->bits;
	if (e.len < 0) {
		const int width = -e.len;
		e = g_vlc_pool[e.sym + ((cache << vlc->bits) >> (32 - width))];
	} else {
		consumed = 0;
	}
	if (e.len <= 0) return -1;
	skip_bits(gb, consumed + e.len);
	return e.sym;
}

///////////////////////////////////////////////////////////////////////////////
// Tables computed at start-up: the VLCs, the run/level tables, the sine windows, the MDCT twiddles.

struct CoefVLCTable {
	int n;                      ///< total number of codes
	int max_level;
	const uint32_t* huffcodes;  ///< VLC bit values
	const uint8_t* huffbits;    ///< VLC bit size
	const uint16_t* levels;     ///< table to build run/level tables
};

#define COUNT(a) ((int)(sizeof(a) / sizeof((a)[0])))

static const CoefVLCTable coef_vlcs[6] = {
	{ COUNT(coef0_huffbits), COUNT(levels0), coef0_huffcodes, coef0_huffbits, levels0 },
	{ COUNT(coef1_huffbits), COUNT(levels1), coef1_huffcodes, coef1_huffbits, levels1 },
	{ COUNT(coef2_huffbits), COUNT(levels2), coef2_huffcodes, coef2_huffbits, levels2 },
	{ COUNT(coef3_huffbits), COUNT(levels3), coef3_huffcodes, coef3_huffbits, levels3 },
	{ COUNT(coef4_huffbits), COUNT(levels4), coef4_huffcodes, coef4_huffbits, levels4 },
	{ COUNT(coef5_huffbits), COUNT(levels5), coef5_huffcodes, coef5_huffbits, levels5 },
};

/** A coefficient table's VLC with the run and level of each code (init_coef_vlc; int_table is the encoder's). */
struct CoefCodebook {
	VLC vlc;
	uint16_t* run_table;
	float* level_table;
};

static CoefCodebook g_coef_books[6];
static VLC g_exp_vlc;
static VLC g_hgain_vlc;
static float g_sine_windows[BLOCK_MAX_BITS + 1][BLOCK_MAX_SIZE];

struct Cpx { double re, im; };

/** Per block size (log2 7..11), ff_tx_mdct_gen_exp's rotation for a scale of 1/32768. */
static Cpx g_mdct_exp[BLOCK_MAX_BITS + 1][BLOCK_MAX_SIZE / 2];
static Cpx g_fft_tw[BLOCK_MAX_SIZE / 4];

static int init_coef_vlc(CoefCodebook* book, const CoefVLCTable* vlc_table)
{
	const int n = vlc_table->n;
	const uint8_t* table_bits = vlc_table->huffbits;
	const uint32_t* table_codes = vlc_table->huffcodes;
	const uint16_t* levels_table = vlc_table->levels;

	if (!vlc_build(&book->vlc, VLCBITS, n, [=](int i) { return table_codes[i]; }, [=](int i) { return (int)table_bits[i]; },
			[](int i) { return (int32_t)i; }))
		return 0;

	uint16_t* run_table = (uint16_t*)calloc(n, sizeof(uint16_t));
	float* flevel_table = (float*)calloc(n, sizeof(float));
	if (!run_table || !flevel_table) return 0;
	int i = 2, level = 1, k = 0;
	while (i < n) {
		const int l = levels_table[k++];
		for (int j = 0; j < l; j++) {
			run_table[i] = (uint16_t)j;
			flevel_table[i] = (float)level;
			i++;
		}
		level++;
	}
	book->run_table = run_table;
	book->level_table = flevel_table;
	return 1;
}

static void mdct_exp_init(Cpx* exp, int len, double scale)
{
	// ff_tx_mdct_gen_exp: a scale whose root goes on each of the pre- and post-rotation.
	const int len2 = len >> 1;
	const double theta = (scale < 0 ? len2 : 0) + 1.0 / 8.0;
	const double root = sqrt(fabs(scale));
	for (int i = 0; i < len2; i++) {
		const double alpha = M_PI_2 * (i + theta) / len2;
		exp[i].re = cos(alpha) * root;
		exp[i].im = sin(alpha) * root;
	}
}

static int g_tables_ready;

static int tables_init(void)
{
	if (g_tables_ready) return 1;

	for (int t = 0; t < 6; t++) {
		if (!init_coef_vlc(&g_coef_books[t], &coef_vlcs[t])) return 0;
	}
	if (!vlc_build(&g_exp_vlc, EXPVLCBITS, 121, [](int i) { return ff_aac_scalefactor_code[i]; },
			[](int i) { return (int)ff_aac_scalefactor_bits[i]; }, [](int i) { return (int32_t)i; }))
		return 0;

	// ff_vlc_init_from_lengths: the codes are canonical, in table order; the symbols offset by -18.
	{
		static uint32_t codes[37];
		uint32_t code = 0;
		for (int i = 0; i < 37; i++) {
			codes[i] = code >> (32 - ff_wma_hgain_hufftab[i][1]);
			code += 1u << (32 - ff_wma_hgain_hufftab[i][1]);
		}
		if (!vlc_build(&g_hgain_vlc, HGAINVLCBITS, 37, [](int i) { return codes[i]; },
				[](int i) { return (int)ff_wma_hgain_hufftab[i][1]; }, [](int i) { return (int32_t)ff_wma_hgain_hufftab[i][0] - 18; }))
			return 0;
	}

	for (int bits = BLOCK_MIN_BITS; bits <= BLOCK_MAX_BITS; bits++) {
		// ff_sine_window_init
		const int n = 1 << bits;
		for (int i = 0; i < n; i++) g_sine_windows[bits][i] = sinf((float)((i + 0.5) * (M_PI / (2.0 * n))));
		mdct_exp_init(g_mdct_exp[bits], n, 1.0 / 32768.0);
	}
	for (int i = 0; i < BLOCK_MAX_SIZE / 4; i++) {
		const double ang = 2.0 * M_PI * i / (BLOCK_MAX_SIZE / 2);
		g_fft_tw[i].re = cos(ang);
		g_fft_tw[i].im = sin(ang);
	}

	g_tables_ready = 1;
	return 1;
}

///////////////////////////////////////////////////////////////////////////////
// IMDCT: ff_tx_mdct_inv over a radix-2 inverse FFT, and ff_tx_mdct_inv_full around it. `len`
// coefficients go in and 2*len samples come out.

static void fft_inverse(Cpx* z, int n)
{
	for (int i = 1, j = 0; i < n; i++) {
		int bit = n >> 1;
		for (; j & bit; bit >>= 1) j ^= bit;
		j ^= bit;
		if (i < j) { const Cpx t = z[i]; z[i] = z[j]; z[j] = t; }
	}
	for (int size = 2; size <= n; size <<= 1) {
		const int half = size >> 1;
		const int stride = (BLOCK_MAX_SIZE / 2) / size;
		for (int i = 0; i < n; i += size) {
			for (int j = 0; j < half; j++) {
				const Cpx w = g_fft_tw[j * stride];
				Cpx* a = &z[i + j];
				Cpx* b = &z[i + j + half];
				const double vr = b->re * w.re - b->im * w.im;
				const double vi = b->re * w.im + b->im * w.re;
				b->re = a->re - vr;
				b->im = a->im - vi;
				a->re += vr;
				a->im += vi;
			}
		}
	}
}

static void imdct_half(float* dst, const float* src, int len, const Cpx* exp)
{
	static Cpx z[BLOCK_MAX_SIZE / 2];
	const int len2 = len >> 1;
	const int len4 = len >> 2;

	for (int i = 0; i < len2; i++) {
		const double re = src[len - 1 - 2 * i];
		const double im = src[2 * i];
		z[i].re = re * exp[i].re - im * exp[i].im;
		z[i].im = re * exp[i].im + im * exp[i].re;
	}

	fft_inverse(z, len2);

	for (int i = 0; i < len4; i++) {
		const int i0 = len4 + i, i1 = len4 - i - 1;
		const Cpx src1 = { z[i1].im, z[i1].re };
		const Cpx src0 = { z[i0].im, z[i0].re };

		z[i1].re = src1.re * exp[i1].im - src1.im * exp[i1].re;
		z[i0].im = src1.re * exp[i1].re + src1.im * exp[i1].im;
		z[i0].re = src0.re * exp[i0].im - src0.im * exp[i0].re;
		z[i1].im = src0.re * exp[i0].re + src0.im * exp[i0].im;
	}

	for (int i = 0; i < len2; i++) {
		dst[2 * i] = (float)z[i].re;
		dst[2 * i + 1] = (float)z[i].im;
	}
}

/** ff_tx_mdct_inv_full: the half transform in the middle, the outer quarters mirrored from it. */
static void imdct_full(float* dst, const float* src, int len_bits)
{
	const int n = 1 << len_bits;
	imdct_half(dst + n / 2, src, n, g_mdct_exp[len_bits]);
	const int len = n << 1, len2 = n, len4 = n >> 1;
	for (int i = 0; i < len4; i++) {
		dst[i] = -dst[len2 - i - 1];
		dst[len - i - 1] = dst[len2 + i];
	}
}

///////////////////////////////////////////////////////////////////////////////
// Context (wma.h's WMACodecContext, the avctx fields it reads folded in)

struct WMAContext {
	GetBitContext gb;
	int channels;
	int sample_rate;
	int bit_rate;
	int block_align;
	int version;                            ///< 1 = 0x160 (WMAV1), 2 = 0x161 (WMAV2)
	int use_bit_reservoir;
	int use_variable_block_len;
	int use_exp_vlc;                        ///< exponent coding: 0 = lsp, 1 = vlc + delta
	int use_noise_coding;                   ///< true if perceptual noise is added
	int byte_offset_bits;
	int exponent_sizes[BLOCK_NB_SIZES];
	uint16_t exponent_bands[BLOCK_NB_SIZES][25];
	int high_band_start[BLOCK_NB_SIZES];    ///< index of first coef in high band
	int coefs_start;                        ///< first coded coef
	int coefs_end[BLOCK_NB_SIZES];          ///< max number of coded coefficients
	int exponent_high_sizes[BLOCK_NB_SIZES];
	int exponent_high_bands[BLOCK_NB_SIZES][HIGH_BAND_MAX_SIZE];

	/* coded values in high bands */
	int high_band_coded[MAX_CHANNELS][HIGH_BAND_MAX_SIZE];
	int high_band_values[MAX_CHANNELS][HIGH_BAND_MAX_SIZE];

	/* there are two possible tables for spectral coefficients */
	const CoefCodebook* coef_books[2];
	/* frame info */
	int frame_len;                          ///< frame length in samples
	int frame_len_bits;                     ///< frame_len = 1 << frame_len_bits
	int nb_block_sizes;                     ///< number of block sizes
	/* block info */
	int reset_block_lengths;
	int block_len_bits;                     ///< log2 of current block length
	int next_block_len_bits;                ///< log2 of next block length
	int prev_block_len_bits;                ///< log2 of prev block length
	int block_len;                          ///< block length in samples
	int block_num;                          ///< block number in current frame
	int block_pos;                          ///< current position in frame
	uint8_t ms_stereo;                      ///< true if mid/side stereo mode
	uint8_t channel_coded[MAX_CHANNELS];    ///< true if channel is coded
	int exponents_bsize[MAX_CHANNELS];      ///< log2 ratio frame/exp. length
	float exponents[MAX_CHANNELS][BLOCK_MAX_SIZE];
	float max_exponent[MAX_CHANNELS];
	float coefs1[MAX_CHANNELS][BLOCK_MAX_SIZE];
	float coefs[MAX_CHANNELS][BLOCK_MAX_SIZE];
	float output[BLOCK_MAX_SIZE * 2];
	const float* windows[BLOCK_NB_SIZES];
	/* output buffer for one frame and the last for IMDCT windowing */
	float frame_out[MAX_CHANNELS][BLOCK_MAX_SIZE * 2];
	/* last frame info */
	uint8_t last_superframe[MAX_CODED_SUPERFRAME_SIZE + AV_INPUT_BUFFER_PADDING_SIZE]; /* padding added */
	int last_bitoffset;
	int last_superframe_len;
	int exponents_initialized[MAX_CHANNELS];
	float noise_table[NOISE_TAB_SIZE];
	int noise_index;
	float noise_mult;
	/* lsp_to_curve tables */
	float lsp_cos_table[BLOCK_MAX_SIZE];
	float lsp_pow_e_table[256];
	float lsp_pow_m_table1[(1 << LSP_POW_BITS)];
	float lsp_pow_m_table2[(1 << LSP_POW_BITS)];

	int eof_done; /* decode flag to output remaining samples after EOF */
	int decoded_any; /* a packet gave samples since the open or the last reset */

	/** The planar output of the last packet (or drain), frame after frame. */
	float samples[MAX_CHANNELS][MAX_SUPERFRAME_FRAMES * BLOCK_MAX_SIZE];
};

///////////////////////////////////////////////////////////////////////////////
// wma_common.c / wma.c

static int ff_wma_get_frame_len_bits(int sample_rate, int version)
{
	int frame_len_bits;

	if (sample_rate <= 16000)
		frame_len_bits = 9;
	else if (sample_rate <= 22050 || (sample_rate <= 32000 && version == 1))
		frame_len_bits = 10;
	else
		frame_len_bits = 11;	// sample_rate <= 48000 || version < 3: v1 and v2 stop here

	return frame_len_bits;
}

/** av_log2 of a positive value. */
static int av_log2(unsigned v)
{
	int n = 0;
	while (v >>= 1) n++;
	return n;
}

static int ff_wma_init(WMAContext* s, int flags2)
{
	const int channels = s->channels;
	float bps1, high_freq;
	float bps;
	int sample_rate1;
	int coef_vlc_table;

	if (s->sample_rate > 50000 || channels > 2 || s->bit_rate <= 0) return 0;

	/* compute MDCT block size */
	s->frame_len_bits = ff_wma_get_frame_len_bits(s->sample_rate, s->version);
	s->next_block_len_bits = s->frame_len_bits;
	s->prev_block_len_bits = s->frame_len_bits;
	s->block_len_bits = s->frame_len_bits;

	s->frame_len = 1 << s->frame_len_bits;
	if (s->use_variable_block_len) {
		int nb_max, nb;
		nb = ((flags2 >> 3) & 3) + 1;
		if ((s->bit_rate / channels) >= 32000) nb += 2;
		nb_max = s->frame_len_bits - BLOCK_MIN_BITS;
		if (nb > nb_max) nb = nb_max;
		s->nb_block_sizes = nb + 1;
	} else {
		s->nb_block_sizes = 1;
	}

	/* init rate dependent parameters */
	s->use_noise_coding = 1;
	high_freq = s->sample_rate * 0.5;

	/* if version 2, then the rates are normalized */
	sample_rate1 = s->sample_rate;
	if (s->version == 2) {
		if (sample_rate1 >= 44100)
			sample_rate1 = 44100;
		else if (sample_rate1 >= 22050)
			sample_rate1 = 22050;
		else if (sample_rate1 >= 16000)
			sample_rate1 = 16000;
		else if (sample_rate1 >= 11025)
			sample_rate1 = 11025;
		else if (sample_rate1 >= 8000)
			sample_rate1 = 8000;
	}

	bps = (float)s->bit_rate / (float)(channels * s->sample_rate);
	s->byte_offset_bits = av_log2((int)(bps * s->frame_len / 8.0 + 0.5)) + 2;
	if (s->byte_offset_bits + 3 > MIN_CACHE_BITS) return 0;

	/* compute high frequency value and choose if noise coding should
	 * be activated */
	bps1 = bps;
	if (channels == 2) bps1 = bps * 1.6;
	if (sample_rate1 == 44100) {
		if (bps1 >= 0.61)
			s->use_noise_coding = 0;
		else
			high_freq = high_freq * 0.4;
	} else if (sample_rate1 == 22050) {
		if (bps1 >= 1.16)
			s->use_noise_coding = 0;
		else if (bps1 >= 0.72)
			high_freq = high_freq * 0.7;
		else
			high_freq = high_freq * 0.6;
	} else if (sample_rate1 == 16000) {
		if (bps > 0.5)
			high_freq = high_freq * 0.5;
		else
			high_freq = high_freq * 0.3;
	} else if (sample_rate1 == 11025) {
		high_freq = high_freq * 0.7;
	} else if (sample_rate1 == 8000) {
		if (bps <= 0.625)
			high_freq = high_freq * 0.5;
		else if (bps > 0.75)
			s->use_noise_coding = 0;
		else
			high_freq = high_freq * 0.65;
	} else {
		if (bps >= 0.8)
			high_freq = high_freq * 0.75;
		else if (bps >= 0.6)
			high_freq = high_freq * 0.6;
		else
			high_freq = high_freq * 0.5;
	}

	/* compute the scale factor band sizes for each MDCT block size */
	{
		int a, b, pos, lpos, k, block_len, i, j, n;
		const uint8_t* table;

		if (s->version == 1)
			s->coefs_start = 3;
		else
			s->coefs_start = 0;
		for (k = 0; k < s->nb_block_sizes; k++) {
			block_len = s->frame_len >> k;

			if (s->version == 1) {
				lpos = 0;
				for (i = 0; i < 25; i++) {
					a = ff_wma_critical_freqs[i];
					b = s->sample_rate;
					pos = ((block_len * 2 * a) + (b >> 1)) / b;
					if (pos > block_len) pos = block_len;
					s->exponent_bands[0][i] = (uint16_t)(pos - lpos);
					if (pos >= block_len) {
						i++;
						break;
					}
					lpos = pos;
				}
				s->exponent_sizes[0] = i;
			} else {
				/* hardcoded tables */
				table = NULL;
				a = s->frame_len_bits - BLOCK_MIN_BITS - k;
				if (a < 3) {
					if (s->sample_rate >= 44100)
						table = exponent_band_44100[a];
					else if (s->sample_rate >= 32000)
						table = exponent_band_32000[a];
					else if (s->sample_rate >= 22050)
						table = exponent_band_22050[a];
				}
				if (table) {
					n = *table++;
					for (i = 0; i < n; i++) s->exponent_bands[k][i] = table[i];
					s->exponent_sizes[k] = n;
				} else {
					j = 0;
					lpos = 0;
					for (i = 0; i < 25; i++) {
						a = ff_wma_critical_freqs[i];
						b = s->sample_rate;
						pos = ((block_len * 2 * a) + (b << 1)) / (4 * b);
						pos <<= 2;
						if (pos > block_len) pos = block_len;
						if (pos > lpos) s->exponent_bands[k][j++] = (uint16_t)(pos - lpos);
						if (pos >= block_len) break;
						lpos = pos;
					}
					s->exponent_sizes[k] = j;
				}
			}

			/* max number of coefs */
			s->coefs_end[k] = (s->frame_len - ((s->frame_len * 9) / 100)) >> k;
			/* high freq computation */
			s->high_band_start[k] = (int)((block_len * 2 * high_freq) / s->sample_rate + 0.5);
			n = s->exponent_sizes[k];
			j = 0;
			pos = 0;
			for (i = 0; i < n; i++) {
				int start, end;
				start = pos;
				pos += s->exponent_bands[k][i];
				end = pos;
				if (start < s->high_band_start[k]) start = s->high_band_start[k];
				if (end > s->coefs_end[k]) end = s->coefs_end[k];
				if (end > start) s->exponent_high_bands[k][j++] = end - start;
			}
			s->exponent_high_sizes[k] = j;
		}
	}

	/* init MDCT windows : simple sine window */
	for (int i = 0; i < s->nb_block_sizes; i++) s->windows[i] = g_sine_windows[s->frame_len_bits - i];

	s->reset_block_lengths = 1;

	if (s->use_noise_coding) {
		/* init the noise generator */
		if (s->use_exp_vlc)
			s->noise_mult = 0.02;
		else
			s->noise_mult = 0.04;

		{
			unsigned int seed;
			float norm;
			seed = 1;
			norm = (1.0 / (float)(1LL << 31)) * sqrt(3) * s->noise_mult;
			for (int i = 0; i < NOISE_TAB_SIZE; i++) {
				seed = seed * 314159 + 1;
				s->noise_table[i] = (float)((int)seed) * norm;
			}
		}
	}

	/* choose the VLC tables for the coefficients */
	coef_vlc_table = 2;
	if (s->sample_rate >= 32000) {
		if (bps1 < 0.72)
			coef_vlc_table = 0;
		else if (bps1 < 1.16)
			coef_vlc_table = 1;
	}
	s->coef_books[0] = &g_coef_books[coef_vlc_table * 2];
	s->coef_books[1] = &g_coef_books[coef_vlc_table * 2 + 1];
	return 1;
}

static int ff_wma_total_gain_to_bits(int total_gain)
{
	if (total_gain < 15)
		return 13;
	else if (total_gain < 32)
		return 12;
	else if (total_gain < 40)
		return 11;
	else if (total_gain < 45)
		return 10;
	else
		return 9;
}

/** ff_wma_run_level_decode, version 0 (WMA v1/v2): 0 on success, -1 when the run overflows. */
static int ff_wma_run_level_decode(GetBitContext* gb, const CoefCodebook* book, float* ptr, int offset, int num_coefs,
	int block_len, int frame_len_bits, int coef_nb_bits)
{
	int code, level, sign;
	const uint32_t* ilvl = (const uint32_t*)book->level_table;
	uint32_t* iptr = (uint32_t*)ptr;
	const unsigned int coef_mask = block_len - 1;
	for (; offset < num_coefs; offset++) {
		code = get_vlc(gb, &book->vlc);
		if (code > 1) {
			/** normal code */
			offset += book->run_table[code];
			sign = get_bits1(gb) - 1;
			iptr[offset & coef_mask] = ilvl[code] ^ (sign & 0x80000000);
		} else if (code == 1) {
			/** EOB */
			break;
		} else {
			/** escape */
			level = get_bits(gb, coef_nb_bits);
			/** NOTE: this is rather suboptimal. reading
			 *  block_len_bits would be better */
			offset += get_bits(gb, frame_len_bits);
			sign = get_bits1(gb) - 1;
			ptr[offset & coef_mask] = (float)((level ^ sign) - sign);
		}
	}
	/** NOTE: EOB can be omitted */
	if (offset > num_coefs) return -1;
	return 0;
}

///////////////////////////////////////////////////////////////////////////////
// wmadec.c

static void wma_lsp_to_curve_init(WMAContext* s, int frame_len)
{
	float wdel, a, b;
	int i, e, m;

	wdel = M_PI / frame_len;
	for (i = 0; i < frame_len; i++) s->lsp_cos_table[i] = 2.0f * cos(wdel * i);

	/* tables for x^-0.25 computation */
	for (i = 0; i < 256; i++) {
		e = i - 126;
		s->lsp_pow_e_table[i] = exp2f(e * -0.25);
	}

	/* NOTE: these two tables are needed to avoid two operations in
	 * pow_m1_4 */
	b = 1.0;
	for (i = (1 << LSP_POW_BITS) - 1; i >= 0; i--) {
		m = (1 << LSP_POW_BITS) + i;
		a = (float)m * (0.5 / (1 << LSP_POW_BITS));
		a = 1 / sqrt(sqrt(a));
		s->lsp_pow_m_table1[i] = 2 * a - b;
		s->lsp_pow_m_table2[i] = b - a;
		b = a;
	}
}

/**
 * compute x^-0.25 with an exponent and mantissa table. We use linear
 * interpolation to reduce the mantissa table size at a small speed
 * expense (linear interpolation approximately doubles the number of
 * bits of precision).
 */
static inline float pow_m1_4(WMAContext* s, float x)
{
	union {
		float f;
		unsigned int v;
	} u, t;
	unsigned int e, m;
	float a, b;

	u.f = x;
	e = u.v >> 23;
	m = (u.v >> (23 - LSP_POW_BITS)) & ((1 << LSP_POW_BITS) - 1);
	/* build interpolation scale: 1 <= t < 2. */
	t.v = ((u.v << LSP_POW_BITS) & ((1 << 23) - 1)) | (127 << 23);
	a = s->lsp_pow_m_table1[m];
	b = s->lsp_pow_m_table2[m];
	return s->lsp_pow_e_table[e] * (a + b * t.f);
}

static void wma_lsp_to_curve(WMAContext* s, float* out, float* val_max_ptr, int n, float* lsp)
{
	int i, j;
	float p, q, w, v, val_max;

	val_max = 0;
	for (i = 0; i < n; i++) {
		p = 0.5f;
		q = 0.5f;
		w = s->lsp_cos_table[i];
		for (j = 1; j < NB_LSP_COEFS; j += 2) {
			q *= w - lsp[j - 1];
			p *= w - lsp[j];
		}
		p *= p * (2.0f - w);
		q *= q * (2.0f + w);
		v = p + q;
		v = pow_m1_4(s, v);
		if (v > val_max) val_max = v;
		out[i] = v;
	}
	*val_max_ptr = val_max;
}

/**
 * decode exponents coded with LSP coefficients (same idea as Vorbis)
 */
static void decode_exp_lsp(WMAContext* s, int ch)
{
	float lsp_coefs[NB_LSP_COEFS];
	int val, i;

	for (i = 0; i < NB_LSP_COEFS; i++) {
		if (i == 0 || i >= 8)
			val = get_bits(&s->gb, 3);
		else
			val = get_bits(&s->gb, 4);
		lsp_coefs[i] = ff_wma_lsp_codebook[i][val];
	}

	wma_lsp_to_curve(s, s->exponents[ch], &s->max_exponent[ch], s->block_len, lsp_coefs);
}

/** pow(10, i / 16.0) for i in -60..95 */
static const float pow_tab[] = {
	1.7782794100389e-04, 2.0535250264571e-04,
	2.3713737056617e-04, 2.7384196342644e-04,
	3.1622776601684e-04, 3.6517412725484e-04,
	4.2169650342858e-04, 4.8696752516586e-04,
	5.6234132519035e-04, 6.4938163157621e-04,
	7.4989420933246e-04, 8.6596432336006e-04,
	1.0000000000000e-03, 1.1547819846895e-03,
	1.3335214321633e-03, 1.5399265260595e-03,
	1.7782794100389e-03, 2.0535250264571e-03,
	2.3713737056617e-03, 2.7384196342644e-03,
	3.1622776601684e-03, 3.6517412725484e-03,
	4.2169650342858e-03, 4.8696752516586e-03,
	5.6234132519035e-03, 6.4938163157621e-03,
	7.4989420933246e-03, 8.6596432336006e-03,
	1.0000000000000e-02, 1.1547819846895e-02,
	1.3335214321633e-02, 1.5399265260595e-02,
	1.7782794100389e-02, 2.0535250264571e-02,
	2.3713737056617e-02, 2.7384196342644e-02,
	3.1622776601684e-02, 3.6517412725484e-02,
	4.2169650342858e-02, 4.8696752516586e-02,
	5.6234132519035e-02, 6.4938163157621e-02,
	7.4989420933246e-02, 8.6596432336007e-02,
	1.0000000000000e-01, 1.1547819846895e-01,
	1.3335214321633e-01, 1.5399265260595e-01,
	1.7782794100389e-01, 2.0535250264571e-01,
	2.3713737056617e-01, 2.7384196342644e-01,
	3.1622776601684e-01, 3.6517412725484e-01,
	4.2169650342858e-01, 4.8696752516586e-01,
	5.6234132519035e-01, 6.4938163157621e-01,
	7.4989420933246e-01, 8.6596432336007e-01,
	1.0000000000000e+00, 1.1547819846895e+00,
	1.3335214321633e+00, 1.5399265260595e+00,
	1.7782794100389e+00, 2.0535250264571e+00,
	2.3713737056617e+00, 2.7384196342644e+00,
	3.1622776601684e+00, 3.6517412725484e+00,
	4.2169650342858e+00, 4.8696752516586e+00,
	5.6234132519035e+00, 6.4938163157621e+00,
	7.4989420933246e+00, 8.6596432336007e+00,
	1.0000000000000e+01, 1.1547819846895e+01,
	1.3335214321633e+01, 1.5399265260595e+01,
	1.7782794100389e+01, 2.0535250264571e+01,
	2.3713737056617e+01, 2.7384196342644e+01,
	3.1622776601684e+01, 3.6517412725484e+01,
	4.2169650342858e+01, 4.8696752516586e+01,
	5.6234132519035e+01, 6.4938163157621e+01,
	7.4989420933246e+01, 8.6596432336007e+01,
	1.0000000000000e+02, 1.1547819846895e+02,
	1.3335214321633e+02, 1.5399265260595e+02,
	1.7782794100389e+02, 2.0535250264571e+02,
	2.3713737056617e+02, 2.7384196342644e+02,
	3.1622776601684e+02, 3.6517412725484e+02,
	4.2169650342858e+02, 4.8696752516586e+02,
	5.6234132519035e+02, 6.4938163157621e+02,
	7.4989420933246e+02, 8.6596432336007e+02,
	1.0000000000000e+03, 1.1547819846895e+03,
	1.3335214321633e+03, 1.5399265260595e+03,
	1.7782794100389e+03, 2.0535250264571e+03,
	2.3713737056617e+03, 2.7384196342644e+03,
	3.1622776601684e+03, 3.6517412725484e+03,
	4.2169650342858e+03, 4.8696752516586e+03,
	5.6234132519035e+03, 6.4938163157621e+03,
	7.4989420933246e+03, 8.6596432336007e+03,
	1.0000000000000e+04, 1.1547819846895e+04,
	1.3335214321633e+04, 1.5399265260595e+04,
	1.7782794100389e+04, 2.0535250264571e+04,
	2.3713737056617e+04, 2.7384196342644e+04,
	3.1622776601684e+04, 3.6517412725484e+04,
	4.2169650342858e+04, 4.8696752516586e+04,
	5.6234132519035e+04, 6.4938163157621e+04,
	7.4989420933246e+04, 8.6596432336007e+04,
	1.0000000000000e+05, 1.1547819846895e+05,
	1.3335214321633e+05, 1.5399265260595e+05,
	1.7782794100389e+05, 2.0535250264571e+05,
	2.3713737056617e+05, 2.7384196342644e+05,
	3.1622776601684e+05, 3.6517412725484e+05,
	4.2169650342858e+05, 4.8696752516586e+05,
	5.6234132519035e+05, 6.4938163157621e+05,
	7.4989420933246e+05, 8.6596432336007e+05,
};

/** Writes `iv` into the next n entries of q, as FFmpeg's unrolled switch does (n >= 1). */
static inline uint32_t* fill_exponent(uint32_t* q, int n, uint32_t iv)
{
	switch (n & 3) do {
	case 0: *q++ = iv; [[fallthrough]];
	case 3: *q++ = iv; [[fallthrough]];
	case 2: *q++ = iv; [[fallthrough]];
	case 1: *q++ = iv;
	} while ((n -= 4) > 0);
	return q;
}

/**
 * decode exponents coded with VLC codes
 */
static int decode_exp_vlc(WMAContext* s, int ch)
{
	int last_exp, n, code;
	const uint16_t* ptr;
	float v, max_scale;
	uint32_t *q, *q_end, iv;
	const float* ptab = pow_tab + 60;
	const uint32_t* iptab = (const uint32_t*)ptab;

	ptr = s->exponent_bands[s->frame_len_bits - s->block_len_bits];
	q = (uint32_t*)s->exponents[ch];
	q_end = q + s->block_len;
	max_scale = 0;
	if (s->version == 1) {
		last_exp = get_bits(&s->gb, 5) + 10;
		v = ptab[last_exp];
		iv = iptab[last_exp];
		max_scale = v;
		n = *ptr++;
		q = fill_exponent(q, n, iv);
	} else {
		last_exp = 36;
	}

	while (q < q_end) {
		code = get_vlc(&s->gb, &g_exp_vlc);
		/* NOTE: this offset is the same as MPEG-4 AAC! */
		last_exp += code - 60;
		if ((unsigned)last_exp + 60 >= sizeof(pow_tab) / sizeof(pow_tab[0])) return -1;
		v = ptab[last_exp];
		iv = iptab[last_exp];
		if (v > max_scale) max_scale = v;
		n = *ptr++;
		q = fill_exponent(q, n, iv);
	}
	s->max_exponent[ch] = max_scale;
	return 0;
}

/** vector_fmul_add: dst = src0 * src1 + src2 */
static void vector_fmul_add(float* dst, const float* src0, const float* src1, const float* src2, int len)
{
	for (int i = 0; i < len; i++) dst[i] = src0[i] * src1[i] + src2[i];
}

/** vector_fmul_reverse: dst = src0 * src1 reversed */
static void vector_fmul_reverse(float* dst, const float* src0, const float* src1, int len)
{
	src1 += len - 1;
	for (int i = 0; i < len; i++) dst[i] = src0[i] * src1[-i];
}

static void butterflies_float(float* v1, float* v2, int len)
{
	for (int i = 0; i < len; i++) {
		const float t = v1[i] - v2[i];
		v1[i] += v2[i];
		v2[i] = t;
	}
}

/**
 * Apply MDCT window and add into output.
 *
 * We ensure that when the windows overlap their squared sum
 * is always 1 (MDCT reconstruction rule).
 */
static void wma_window(WMAContext* s, float* out)
{
	float* in = s->output;
	int block_len, bsize, n;

	/* left part */
	if (s->block_len_bits <= s->prev_block_len_bits) {
		block_len = s->block_len;
		bsize = s->frame_len_bits - s->block_len_bits;

		vector_fmul_add(out, in, s->windows[bsize], out, block_len);
	} else {
		block_len = 1 << s->prev_block_len_bits;
		n = (s->block_len - block_len) / 2;
		bsize = s->frame_len_bits - s->prev_block_len_bits;

		vector_fmul_add(out + n, in + n, s->windows[bsize], out + n, block_len);

		memcpy(out + n + block_len, in + n + block_len, n * sizeof(float));
	}

	out += s->block_len;
	in += s->block_len;

	/* right part */
	if (s->block_len_bits <= s->next_block_len_bits) {
		block_len = s->block_len;
		bsize = s->frame_len_bits - s->block_len_bits;

		vector_fmul_reverse(out, in, s->windows[bsize], block_len);
	} else {
		block_len = 1 << s->next_block_len_bits;
		n = (s->block_len - block_len) / 2;
		bsize = s->frame_len_bits - s->next_block_len_bits;

		memcpy(out, in, n * sizeof(float));

		vector_fmul_reverse(out + n, in + n, s->windows[bsize], block_len);

		memset(out + n + block_len, 0, n * sizeof(float));
	}
}

/**
 * @return
 * 0 if OK.
 * 1 if last block of frame.
 * -1 if unrecoverable error.
 */
static int wma_decode_block(WMAContext* s)
{
	const int channels = s->channels;
	int n, v, a, bsize;
	int coef_nb_bits, total_gain;
	int nb_coefs[MAX_CHANNELS];
	float mdct_norm;

	/* compute current block length */
	if (s->use_variable_block_len) {
		n = av_log2(s->nb_block_sizes - 1) + 1;

		if (s->reset_block_lengths) {
			s->reset_block_lengths = 0;
			v = get_bits(&s->gb, n);
			if (v >= s->nb_block_sizes) return -1;
			s->prev_block_len_bits = s->frame_len_bits - v;
			v = get_bits(&s->gb, n);
			if (v >= s->nb_block_sizes) return -1;
			s->block_len_bits = s->frame_len_bits - v;
		} else {
			/* update block lengths */
			s->prev_block_len_bits = s->block_len_bits;
			s->block_len_bits = s->next_block_len_bits;
		}
		v = get_bits(&s->gb, n);
		if (v >= s->nb_block_sizes) return -1;
		s->next_block_len_bits = s->frame_len_bits - v;
	} else {
		/* fixed block len */
		s->next_block_len_bits = s->frame_len_bits;
		s->prev_block_len_bits = s->frame_len_bits;
		s->block_len_bits = s->frame_len_bits;
	}

	if (s->frame_len_bits - s->block_len_bits >= s->nb_block_sizes) return -1;

	/* now check if the block length is coherent with the frame length */
	s->block_len = 1 << s->block_len_bits;
	if ((s->block_pos + s->block_len) > s->frame_len) return -1;

	if (channels == 2) s->ms_stereo = (uint8_t)get_bits1(&s->gb);
	v = 0;
	for (int ch = 0; ch < channels; ch++) {
		a = get_bits1(&s->gb);
		s->channel_coded[ch] = (uint8_t)a;
		v |= a;
	}

	bsize = s->frame_len_bits - s->block_len_bits;

	/* if no channel coded, no need to go further */
	/* XXX: fix potential framing problems */
	if (!v) goto next;

	/* read total gain and extract corresponding number of bits for
	 * coef escape coding */
	total_gain = 1;
	for (;;) {
		if (get_bits_left(&s->gb) < 7) return -1;
		a = get_bits(&s->gb, 7);
		total_gain += a;
		if (a != 127) break;
	}

	coef_nb_bits = ff_wma_total_gain_to_bits(total_gain);

	/* compute number of coefficients */
	n = s->coefs_end[bsize] - s->coefs_start;
	for (int ch = 0; ch < channels; ch++) nb_coefs[ch] = n;

	/* complex coding */
	if (s->use_noise_coding) {
		for (int ch = 0; ch < channels; ch++) {
			if (s->channel_coded[ch]) {
				const int n = s->exponent_high_sizes[bsize];
				for (int i = 0; i < n; i++) {
					const unsigned a = get_bits1(&s->gb);
					s->high_band_coded[ch][i] = a;
					/* if noise coding, the coefficients are not transmitted */
					if (a) nb_coefs[ch] -= s->exponent_high_bands[bsize][i];
				}
			}
		}
		for (int ch = 0; ch < channels; ch++) {
			if (s->channel_coded[ch]) {
				const int n = s->exponent_high_sizes[bsize];
				int val = (int)0x80000000;
				for (int i = 0; i < n; i++) {
					if (s->high_band_coded[ch][i]) {
						if (val == (int)0x80000000) {
							val = get_bits(&s->gb, 7) - 19;
						} else {
							val += get_vlc(&s->gb, &g_hgain_vlc);
						}
						s->high_band_values[ch][i] = val;
					}
				}
			}
		}
	}

	/* exponents can be reused in short blocks. */
	if ((s->block_len_bits == s->frame_len_bits) || get_bits1(&s->gb)) {
		for (int ch = 0; ch < channels; ch++) {
			if (s->channel_coded[ch]) {
				if (s->use_exp_vlc) {
					if (decode_exp_vlc(s, ch) < 0) return -1;
				} else {
					decode_exp_lsp(s, ch);
				}
				s->exponents_bsize[ch] = bsize;
				s->exponents_initialized[ch] = 1;
			}
		}
	}

	for (int ch = 0; ch < channels; ch++) {
		if (s->channel_coded[ch] && !s->exponents_initialized[ch]) return -1;
	}

	/* parse spectral coefficients : just RLE encoding */
	for (int ch = 0; ch < channels; ch++) {
		if (s->channel_coded[ch]) {
			float* ptr = &s->coefs1[ch][0];

			/* special VLC tables are used for ms stereo because
			 * there is potentially less energy there */
			const int tindex = (ch == 1 && s->ms_stereo);
			memset(ptr, 0, s->block_len * sizeof(float));
			if (ff_wma_run_level_decode(&s->gb, s->coef_books[tindex], ptr, 0, nb_coefs[ch], s->block_len,
					s->frame_len_bits, coef_nb_bits) < 0)
				return -1;
		}
		if (s->version == 1 && channels >= 2) align_get_bits(&s->gb);
	}

	/* normalize */
	{
		const int n4 = s->block_len / 2;
		mdct_norm = 1.0 / (float)n4;
		if (s->version == 1) mdct_norm *= sqrt(n4);
	}

	/* finally compute the MDCT coefficients */
	for (int ch = 0; ch < channels; ch++) {
		if (s->channel_coded[ch]) {
			float* coefs1;
			float *coefs, *exponents, mult, mult1, noise;
			int n, n1, last_high_band, esize;
			float exp_power[HIGH_BAND_MAX_SIZE];

			coefs1 = s->coefs1[ch];
			exponents = s->exponents[ch];
			esize = s->exponents_bsize[ch];
			mult = pow(10, total_gain * 0.05) / s->max_exponent[ch];	// ff_exp10
			mult *= mdct_norm;
			coefs = s->coefs[ch];
			if (s->use_noise_coding) {
				mult1 = mult;
				/* very low freqs : noise */
				for (int i = 0; i < s->coefs_start; i++) {
					*coefs++ = s->noise_table[s->noise_index] * exponents[i << bsize >> esize] * mult1;
					s->noise_index = (s->noise_index + 1) & (NOISE_TAB_SIZE - 1);
				}

				n1 = s->exponent_high_sizes[bsize];

				/* compute power of high bands */
				exponents = s->exponents[ch] + (s->high_band_start[bsize] << bsize >> esize);
				last_high_band = 0; /* avoid warning */
				for (int j = 0; j < n1; j++) {
					n = s->exponent_high_bands[s->frame_len_bits - s->block_len_bits][j];
					if (s->high_band_coded[ch][j]) {
						float e2, v;
						e2 = 0;
						for (int i = 0; i < n; i++) {
							v = exponents[i << bsize >> esize];
							e2 += v * v;
						}
						exp_power[j] = e2 / n;
						last_high_band = j;
					}
					exponents += n << bsize >> esize;
				}

				/* main freqs and high freqs */
				exponents = s->exponents[ch] + (s->coefs_start << bsize >> esize);
				for (int j = -1; j < n1; j++) {
					if (j < 0)
						n = s->high_band_start[bsize] - s->coefs_start;
					else
						n = s->exponent_high_bands[s->frame_len_bits - s->block_len_bits][j];
					if (j >= 0 && s->high_band_coded[ch][j]) {
						/* use noise with specified power */
						mult1 = sqrt(exp_power[j] / exp_power[last_high_band]);
						/* XXX: use a table */
						mult1 = mult1 * pow(10, s->high_band_values[ch][j] * 0.05);	// ff_exp10
						mult1 = mult1 / (s->max_exponent[ch] * s->noise_mult);
						mult1 *= mdct_norm;
						for (int i = 0; i < n; i++) {
							noise = s->noise_table[s->noise_index];
							s->noise_index = (s->noise_index + 1) & (NOISE_TAB_SIZE - 1);
							*coefs++ = noise * exponents[i << bsize >> esize] * mult1;
						}
						exponents += n << bsize >> esize;
					} else {
						/* coded values + small noise */
						for (int i = 0; i < n; i++) {
							noise = s->noise_table[s->noise_index];
							s->noise_index = (s->noise_index + 1) & (NOISE_TAB_SIZE - 1);
							*coefs++ = ((*coefs1++) + noise) * exponents[i << bsize >> esize] * mult;
						}
						exponents += n << bsize >> esize;
					}
				}

				/* very high freqs : noise */
				n = s->block_len - s->coefs_end[bsize];
				mult1 = mult * exponents[(-(1 << bsize)) >> esize];
				for (int i = 0; i < n; i++) {
					*coefs++ = s->noise_table[s->noise_index] * mult1;
					s->noise_index = (s->noise_index + 1) & (NOISE_TAB_SIZE - 1);
				}
			} else {
				/* XXX: optimize more */
				for (int i = 0; i < s->coefs_start; i++) *coefs++ = 0.0;
				n = nb_coefs[ch];
				for (int i = 0; i < n; i++) *coefs++ = coefs1[i] * exponents[i << bsize >> esize] * mult;
				n = s->block_len - s->coefs_end[bsize];
				for (int i = 0; i < n; i++) *coefs++ = 0.0;
			}
		}
	}

	if (s->ms_stereo && s->channel_coded[1]) {
		/* nominal case for ms stereo: we do it before mdct */
		/* no need to optimize this case because it should almost
		 * never happen */
		if (!s->channel_coded[0]) {
			memset(s->coefs[0], 0, sizeof(float) * s->block_len);
			s->channel_coded[0] = 1;
		}

		butterflies_float(s->coefs[0], s->coefs[1], s->block_len);
	}

next:
	for (int ch = 0; ch < channels; ch++) {
		int n4, index;

		n4 = s->block_len / 2;
		if (s->channel_coded[ch])
			imdct_full(s->output, s->coefs[ch], s->block_len_bits);
		else if (!(s->ms_stereo && ch == 1))
			memset(s->output, 0, sizeof(s->output));

		/* multiply by the window and add in the frame */
		index = (s->frame_len / 2) + s->block_pos - n4;
		wma_window(s, &s->frame_out[ch][index]);
	}

	/* update block number */
	s->block_num++;
	s->block_pos += s->block_len;
	if (s->block_pos >= s->frame_len)
		return 1;
	else
		return 0;
}

/* decode a frame of frame_len samples */
static int wma_decode_frame(WMAContext* s, int samples_offset)
{
	/* read each block */
	s->block_num = 0;
	s->block_pos = 0;
	for (;;) {
		const int ret = wma_decode_block(s);
		if (ret < 0) return ret;
		if (ret) break;
	}

	for (int ch = 0; ch < s->channels; ch++) {
		/* copy current block to output */
		memcpy(s->samples[ch] + samples_offset, s->frame_out[ch], s->frame_len * sizeof(*s->frame_out[ch]));
		/* prepare for next block */
		memmove(&s->frame_out[ch][0], &s->frame_out[ch][s->frame_len], s->frame_len * sizeof(*s->frame_out[ch]));
	}

	return 0;
}

/** wma_decode_superframe on one block_align packet: the samples per channel it gave, -1 on error. */
static int wma_decode_superframe(WMAContext* s, const uint8_t* buf)
{
	const int buf_size = s->block_align;
	int nb_frames, bit_offset, pos, len;
	uint8_t* q;
	int samples_offset;

	init_get_bits(&s->gb, buf, buf_size * 8);

	if (s->use_bit_reservoir) {
		/* read super frame header */
		skip_bits(&s->gb, 4); /* super frame index */
		nb_frames = get_bits(&s->gb, 4) - (s->last_superframe_len <= 0);
		if (nb_frames <= 0) {
			const int is_error = nb_frames < 0 || get_bits_left(&s->gb) <= 8;
			if (is_error) return -1;

			if ((s->last_superframe_len + buf_size - 1) > MAX_CODED_SUPERFRAME_SIZE) goto fail;

			q = s->last_superframe + s->last_superframe_len;
			len = buf_size - 1;
			while (len > 0) {
				*q++ = (uint8_t)get_bits(&s->gb, 8);
				len--;
			}
			memset(q, 0, AV_INPUT_BUFFER_PADDING_SIZE);

			s->last_superframe_len += 8 * buf_size - 8;
			return 0;
		}
	} else {
		nb_frames = 1;
	}

	samples_offset = 0;

	if (s->use_bit_reservoir) {
		bit_offset = get_bits(&s->gb, s->byte_offset_bits + 3);
		if (bit_offset > get_bits_left(&s->gb)) goto fail;

		if (s->last_superframe_len > 0) {
			/* add bit_offset bits to last frame */
			if ((s->last_superframe_len + ((bit_offset + 7) >> 3)) > MAX_CODED_SUPERFRAME_SIZE) goto fail;
			q = s->last_superframe + s->last_superframe_len;
			len = bit_offset;
			while (len > 7) {
				*q++ = (uint8_t)get_bits(&s->gb, 8);
				len -= 8;
			}
			if (len > 0) *q++ = (uint8_t)(get_bits(&s->gb, len) << (8 - len));
			memset(q, 0, AV_INPUT_BUFFER_PADDING_SIZE);

			/* XXX: bit_offset bits into last frame */
			init_get_bits(&s->gb, s->last_superframe, s->last_superframe_len * 8 + bit_offset);
			/* skip unused bits */
			if (s->last_bitoffset > 0) skip_bits(&s->gb, s->last_bitoffset);
			/* this frame is stored in the last superframe and in the
			 * current one */
			if (wma_decode_frame(s, samples_offset) < 0) goto fail;
			samples_offset += s->frame_len;
			nb_frames--;
		}

		/* read each frame starting from bit_offset */
		pos = bit_offset + 4 + 4 + s->byte_offset_bits + 3;
		if (pos >= MAX_CODED_SUPERFRAME_SIZE * 8 || pos > buf_size * 8) return -1;
		init_get_bits(&s->gb, buf + (pos >> 3), (buf_size - (pos >> 3)) * 8);
		len = pos & 7;
		if (len > 0) skip_bits(&s->gb, len);

		s->reset_block_lengths = 1;
		for (int i = 0; i < nb_frames; i++) {
			if (wma_decode_frame(s, samples_offset) < 0) goto fail;
			samples_offset += s->frame_len;
		}

		/* we copy the end of the frame in the last frame buffer */
		pos = get_bits_count(&s->gb) + ((bit_offset + 4 + 4 + s->byte_offset_bits + 3) & ~7);
		s->last_bitoffset = pos & 7;
		pos >>= 3;
		len = buf_size - pos;
		if (len > MAX_CODED_SUPERFRAME_SIZE || len < 0) goto fail;
		s->last_superframe_len = len;
		memcpy(s->last_superframe, buf + pos, len);
	} else {
		/* single frame decode */
		if (wma_decode_frame(s, samples_offset) < 0) goto fail;
		samples_offset += s->frame_len;
	}

	return samples_offset;

fail:
	/* when error, we reset the bit reservoir */
	s->last_superframe_len = 0;
	return -1;
}

///////////////////////////////////////////////////////////////////////////////
// Boundary

extern "C" {

WMAContext* wma_open(void)
{
	if (!tables_init()) return NULL;
	return (WMAContext*)calloc(1, sizeof(WMAContext));
}

void wma_close(WMAContext* s) { free(s); }

/** wma_decode_init. */
int wma_configure(WMAContext* s, int version, int channels, int sample_rate, int bit_rate, int block_align,
	const uint8_t* extradata, int extradata_size, int* info)
{
	memset(info, 0, 6 * sizeof(int));
	if (!block_align || block_align > MAX_CODED_SUPERFRAME_SIZE || channels < 1 || (version != 1 && version != 2)) return 0;
	s->version = version;
	s->channels = channels;
	s->sample_rate = sample_rate;
	s->bit_rate = bit_rate;
	s->block_align = block_align;

	/* extract flag info */
	int flags2 = 0;
	if (version == 1 && extradata_size >= 4)
		flags2 = extradata[2] | (extradata[3] << 8);
	else if (version == 2 && extradata_size >= 6)
		flags2 = extradata[4] | (extradata[5] << 8);

	s->use_exp_vlc = flags2 & 0x0001;
	s->use_bit_reservoir = flags2 & 0x0002;
	s->use_variable_block_len = flags2 & 0x0004;

	if (version == 2 && extradata_size >= 8) {
		if ((extradata[4] | (extradata[5] << 8)) == 0xd && s->use_variable_block_len) {
			s->use_variable_block_len = 0; // this fixes issue1503
		}
	}

	for (int i = 0; i < MAX_CHANNELS; i++) s->max_exponent[i] = 1.0;

	if (!ff_wma_init(s, flags2)) return 0;

	if (!s->use_exp_vlc) wma_lsp_to_curve_init(s, s->frame_len);

	info[0] = s->frame_len;
	info[1] = s->use_bit_reservoir;
	info[2] = s->use_noise_coding;
	info[3] = s->use_exp_vlc;
	info[4] = s->use_variable_block_len;
	info[5] = s->nb_block_sizes;
	return 1;
}

/**
 * Seek. FFmpeg's flush plus the state a decode from the stream's start has: no overlap, block lengths read anew, no
 * exponents a short block could reuse, and the noise generator where `noise_index` says.
 */
void wma_reset(WMAContext* s, int noise_index)
{
	s->last_bitoffset = s->last_superframe_len = 0;
	s->eof_done = 0;
	s->decoded_any = 0;
	memset(s->frame_out, 0, sizeof(s->frame_out));
	s->reset_block_lengths = 1;
	s->next_block_len_bits = s->prev_block_len_bits = s->block_len_bits = s->frame_len_bits;
	s->ms_stereo = 0;
	for (int i = 0; i < MAX_CHANNELS; i++) {
		s->exponents_initialized[i] = 0;
		s->max_exponent[i] = 1.0;
	}
	s->noise_index = noise_index & (NOISE_TAB_SIZE - 1);
}

int wma_decode_packet(WMAContext* s, const uint8_t* data)
{
	const int samples = wma_decode_superframe(s, data);
	if (samples > 0) s->decoded_any = 1;
	return samples;
}

/** The empty packet at the end of the stream: the last frame's overlap, once. */
int wma_drain(WMAContext* s)
{
	if (s->eof_done || !s->decoded_any) return 0;
	for (int ch = 0; ch < s->channels; ch++) memcpy(s->samples[ch], s->frame_out[ch], s->frame_len * sizeof(float));
	s->last_superframe_len = 0;
	s->eof_done = 1;
	return s->frame_len;
}

float* wma_channel_ptr(WMAContext* s, int ch) { return s->samples[ch]; }

/** Max abs error x 1e9 of imdct_full against the definition, for a block of 1 << bits; random input. */
int wma_test_imdct(int bits)
{
	if (!tables_init() || bits < BLOCK_MIN_BITS || bits > BLOCK_MAX_BITS) return -1;
	static float in[BLOCK_MAX_SIZE], out[BLOCK_MAX_SIZE * 2];
	const int n = 1 << bits;
	unsigned seed = 12345;
	for (int k = 0; k < n; k++) {
		seed = seed * 1664525 + 1013904223;
		in[k] = (float)((int)(seed >> 8) - (1 << 23)) / (1 << 23) * 32768.0f;
	}
	imdct_full(out, in, bits);
	// y[i] = -scale * sum_k x[k] cos(pi / n * (i + 0.5 + n / 2) * (k + 0.5)), lavu's inverse MDCT
	double max_err = 0;
	for (int i = 0; i < 2 * n; i++) {
		double sum = 0;
		for (int k = 0; k < n; k++) sum += in[k] * cos(M_PI / n * (i + 0.5 + n / 2.0) * (k + 0.5));
		const double err = fabs(sum / 32768.0 + out[i]);
		if (err > max_err) max_err = err;
	}
	return (int)(max_err * 1e9);
}

}
