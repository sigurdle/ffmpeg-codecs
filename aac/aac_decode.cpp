/*
	AAC-LC and HE-AAC v1 audio decoder -- WASM C++.

	A direct port of FFmpeg's native AAC decoder (libavcodec, tag n8.1):
	  aac/aacdec.c / aac/aacdec_float.c / aac/aacdec_float_coupling.h /
	  aac/aacdec_dsp_template.c / aac/aacdec_proc_template.c / aac/aacdec_tab.c /
	  aactab.c / kbdwin.c / mpeg4audio.c / adts_header.c / lpc_functions.h /
	  sinewin_tablegen.h + libavutil/float_dsp.c

	This file is part of a derivative of FFmpeg. FFmpeg is free software; you can
	redistribute it and/or modify it under the terms of the GNU Lesser General Public
	License as published by the Free Software Foundation; either version 2.1 of the
	License, or (at your option) any later version. It is distributed in the hope that it
	will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
	License for more details. Copyright is held by the FFmpeg authors: Oded Shimon, Maxim
	Gavrilov, Alex Converse, Paul Kendall, Janne Grunau and the others named in the files above.

	Scope -- AAC-LC at 1024 samples per frame, and SBR over it at 2048 (aac_sbr.h, pinned to
	FFmpeg n9.0.2), float output, as the containers we read carry it: AudioSpecificConfig (or ADTS headers), raw_data_block with SCE, CPE, CCE, LFE,
	DSE, PCE and FIL; ICS, section data, scale factors, spectral Huffman, pulses, M/S,
	intensity, PNS, TNS, coupling; long, start, eight-short and stop windows, sine and KBD.
	Everything else is refused, never approximated: Main, SSR, LTP, ER, LD, ELD, USAC, and
	the 960-sample frame length.

	SBR turns on where FFmpeg turns it on: AOT 5 or 29, an explicit extension in the
	AudioSpecificConfig, or an EXT_SBR_DATA fill element in the first frame. The last is why a
	stream's rate and frame size may only be known once a frame has decoded. Parametric Stereo
	is not decoded: a stream carrying it plays its mono core, and aac_decode_frame says so.

	Differences from FFmpeg, all behavior-preserving for that scope:
	  - The surrounding framework (AVCodecContext / AVFrame / the DSP function pointers /
	    threading) is replaced by a self-contained handle. Each element keeps its own output
	    buffer, so there is no frame_configure_elements; a channel reads through
	    output_element.
	  - Fixed point, the USAC paths and the dual-mono side data are gone.
	  - The IMDCT runs its own complex FFT, in double, in place of libavutil/tx; the twiddles
	    and the pre- and post-rotation follow ff_tx_mdct_inv, and aac_test_imdct checks the
	    result against a direct O(N^2) evaluation of the definition. FFmpeg computes in float,
	    so the PCM differs from `ffmpeg -f f32le` by float round-off (tests/AAC.test.js).
	  - The VLCs are built here as two-level tables rather than by ff_vlc_init.
	  - The pow2sf and cbrt tables are computed with pow and cbrt rather than FFmpeg's
	    incremental forms; they differ at most in the last bit of a float.
	  - The bit reader clamps at the end of the packet, as FFmpeg's checked reader does.

	Priming: this file drops nothing. The first frame comes out overlapped against silence, as
	FFmpeg's does, and every frame gives 1024 samples, which is what `ffmpeg -i x.aac -f f32le`
	outputs. Dropping a container's encoder delay (an mp4 edit list) is the container's. The one
	drop FFmpeg's decoder makes, the 1024 samples of a frame whose fill element names libfaac
	(decode_fill), is made in AACDecoder.js, because a scan has to see it without decoding.

	Seek: aac_reset clears what ties a frame to the frames before it (the overlap, the window
	history of each ICS, the PNS generator). FFmpeg's flush clears only the overlap, which
	would make a read of frame 0 depend on where the decoder had been.

	Boundary (extern "C", prefixed _ on the JS side; see aac_decode_core.js):
	  AACContext* aac_open(void)
	  void    aac_close(AACContext*)
	  void    aac_reset(AACContext*, int frame)    the next frame's number in the stream
	  int     aac_configure(AACContext*, uint8_t* asc, int size, int info[4])
	            info = {sampleRate, channels, objectType, sbrPresent}, the output's rate; 1, 0
	            when invalid or not LC. `asc` has room for AAC_INPUT_PADDING bytes past `size`.
	  int     aac_decode_frame(AACContext*, uint8_t* data, int size, int info[12])
	            decode one raw_data_block, or one ADTS frame; info as AAC_INFO_* below; 1, 0 on
	            error. `data` has room for AAC_INPUT_PADDING bytes past `size`.
	  float*  aac_channel_ptr(AACContext*, int ch)    planar output, 1024 or 2048 samples
	  int     aac_test_imdct(int n)                   max error x 1e9 against the definition
*/

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MAX_CHANNELS 64
#define MAX_ELEM_ID 64
#define TNS_MAX_ORDER 20
#define AAC_INPUT_PADDING 16

#define SCALE_DIFF_ZERO 60
#define POW_SF2_ZERO 200
#define NOISE_PRE 256
#define NOISE_PRE_BITS 9
#define NOISE_OFFSET 90

#define AVERROR_INVALIDDATA (-1)
#define AVERROR_UNSUPPORTED (-2)

#define FFMIN(a, b) ((a) < (b) ? (a) : (b))
#define FFMAX(a, b) ((a) > (b) ? (a) : (b))

enum RawDataBlockType { TYPE_SCE, TYPE_CPE, TYPE_CCE, TYPE_LFE, TYPE_DSE, TYPE_PCE, TYPE_FIL, TYPE_END };
enum ExtensionPayloadID {
	EXT_FILL, EXT_FILL_DATA, EXT_DATA_ELEMENT, EXT_DYNAMIC_RANGE = 0xb, EXT_SBR_DATA = 0xd, EXT_SBR_DATA_CRC = 0xe,
};
enum WindowSequence { ONLY_LONG_SEQUENCE, LONG_START_SEQUENCE, EIGHT_SHORT_SEQUENCE, LONG_STOP_SEQUENCE };
enum BandType {
	ZERO_BT = 0, FIRST_PAIR_BT = 5, ESC_BT = 11, RESERVED_BT = 12, NOISE_BT = 13, INTENSITY_BT2 = 14, INTENSITY_BT = 15,
};
enum ChannelPosition {
	AAC_CHANNEL_OFF = 0, AAC_CHANNEL_FRONT = 1, AAC_CHANNEL_SIDE = 2, AAC_CHANNEL_BACK = 3, AAC_CHANNEL_LFE = 4, AAC_CHANNEL_CC = 5,
};
enum OCStatus { OC_NONE, OC_TRIAL_PCE, OC_TRIAL_FRAME, OC_GLOBAL_HDR, OC_LOCKED };
enum CouplingPoint { BEFORE_TNS, BETWEEN_TNS_AND_IMDCT, AFTER_IMDCT = 3 };
enum { AOT_AAC_LC = 2, AOT_SBR = 5, AOT_ER_BSAC = 22, AOT_ESCAPE = 31, AOT_PS = 29, AOT_NULL = 0 };

#include "aac_tables.h"
#include "aac_sbr_tables.h"

static const int ff_mpeg4audio_sample_rates[16] = {
	96000, 88200, 64000, 48000, 44100, 32000, 24000, 22050, 16000, 12000, 11025, 8000, 7350,
};
static const uint8_t ff_mpeg4audio_channels[15] = { 0, 1, 2, 3, 4, 5, 6, 8, 0, 0, 0, 7, 8, 24, 8 };

// What aac_decode_frame answers. The counts cover one frame; they let a conformance run prove it
// exercised a tool, which is otherwise easy to leave silently untested.
enum {
	AAC_INFO_SAMPLE_RATE, AAC_INFO_CHANNELS, AAC_INFO_SAMPLES, AAC_INFO_PS_PRESENT,
	AAC_INFO_SHORT, AAC_INFO_KBD, AAC_INFO_PNS, AAC_INFO_INTENSITY, AAC_INFO_MS, AAC_INFO_TNS,
	AAC_INFO_PULSE, AAC_INFO_CCE, AAC_INFO_COUNT,
};

///////////////////////////////////////////////////////////////////////////////
// Bit reader. Top-aligned, MSB first; the caller's buffer holds AAC_INPUT_PADDING zero bytes past
// the end, and the index clamps there, so an overread answers zeros, as FFmpeg's checked reader.

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
static inline void skip_bits_long(GetBitContext* s, int n) { skip_bits(s, n); }
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
// to a subtable of -length bits at pool index sym. Length 0 is an invalid code.

struct VLCElem { uint16_t sym; int16_t len; };
struct VLC { const VLCElem* table; int bits; };

static VLCElem g_vlc_pool[32768];
static int g_vlc_used;
static VLC g_vlc_spectral[11];
static VLC g_vlc_scalefactors;
static VLC g_vlc_sbr[10];

/** `code(i)`, `len(i)`, `sym(i)` for n codes. Returns 0 if the pool is too small. */
template <class CodeAt, class LenAt, class SymAt>
static int vlc_build(VLC* vlc, int bits, int n, CodeAt code, LenAt len, SymAt sym)
{
	const int base = g_vlc_used;
	const int size = 1 << bits;
	if (base + size > (int)(sizeof(g_vlc_pool) / sizeof(g_vlc_pool[0]))) return 0;
	g_vlc_used += size;
	VLCElem* table = g_vlc_pool + base;
	memset(table, 0, size * sizeof(*table));

	for (int i = 0; i < n; i++) {
		const int l = len(i);
		if (!l || l > bits) continue;
		const uint32_t first = (uint32_t)code(i) << (bits - l);
		for (uint32_t k = 0; k < (1u << (bits - l)); k++) {
			table[first + k].sym = sym(i);
			table[first + k].len = (int16_t)l;
		}
	}

	// The width of each prefix's subtable is the longest code below it.
	for (int i = 0; i < n; i++) {
		const int l = len(i);
		if (l <= bits) continue;
		const uint32_t prefix = (uint32_t)code(i) >> (l - bits);
		if (table[prefix].len > -(l - bits)) table[prefix].len = (int16_t)-(l - bits);
	}
	for (int prefix = 0; prefix < size; prefix++) {
		if (table[prefix].len >= 0) continue;
		const int width = -table[prefix].len;
		if (g_vlc_used + (1 << width) > (int)(sizeof(g_vlc_pool) / sizeof(g_vlc_pool[0]))) return 0;
		table[prefix].sym = (uint16_t)g_vlc_used;
		memset(g_vlc_pool + g_vlc_used, 0, (1u << width) * sizeof(VLCElem));
		g_vlc_used += 1 << width;
	}
	for (int i = 0; i < n; i++) {
		const int l = len(i);
		if (l <= bits) continue;
		const uint32_t c = (uint32_t)code(i);
		const VLCElem link = table[c >> (l - bits)];
		const int width = -link.len;
		const int rest = l - bits;
		const uint32_t first = (c & ((1u << rest) - 1)) << (width - rest);
		for (uint32_t k = 0; k < (1u << (width - rest)); k++) {
			g_vlc_pool[link.sym + first + k].sym = sym(i);
			g_vlc_pool[link.sym + first + k].len = (int16_t)rest;
		}
	}

	vlc->table = table;
	vlc->bits = bits;
	return 1;
}

static inline int get_vlc(GetBitContext* gb, const VLC* vlc)
{
	const uint32_t cache = show_cache(gb);
	VLCElem e = vlc->table[cache >> (32 - vlc->bits)];
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
// Tables computed at start-up

static float g_pow2sf_tab[428];
static float g_cbrt_tab[1 << 13];
static float g_sine_1024[1024];
static float g_sine_128[128];
static float g_kbd_long_1024[1024];
static float g_kbd_short_128[128];

struct Cpx { double re, im; };

// MDCT twiddles and the FFT's, for the 512- and 64-point complex transforms inside.
static Cpx g_mdct_exp_long[512];
static Cpx g_mdct_exp_short[64];
static Cpx g_sbr_exp_analysis[32];
static Cpx g_sbr_exp_synthesis[32];
static Cpx g_fft_tw[256];

static double bessel_i0(double x)
{
	double sum = 1.0, term = 1.0;
	const double x2 = x * x / 4.0;
	for (int k = 1; k < 500; k++) {
		term *= x2 / ((double)k * (double)k);
		sum += term;
		if (term < sum * 1e-20) break;
	}
	return sum;
}

static void kbd_window_init(float* window, double alpha, int n)
{
	double temp[1024 / 2 + 1];
	double sum = 0.0, scale = 0.0;
	const double alpha2 = 4 * (alpha * M_PI / n) * (alpha * M_PI / n);

	for (int i = 0; i <= n / 2; i++) {
		const double tmp = (double)i * (n - i) * alpha2;
		temp[i] = bessel_i0(sqrt(tmp));
		scale += temp[i] * (1 + (i && i < n / 2));
	}
	scale = 1.0 / (scale + 1);

	int i = 0;
	for (; i <= n / 2; i++) {
		sum += temp[i];
		window[i] = (float)sqrt(sum * scale);
	}
	for (; i < n; i++) {
		sum += temp[n - i];
		window[i] = (float)sqrt(sum * scale);
	}
}

static void sine_window_init(float* window, int n)
{
	for (int i = 0; i < n; i++) window[i] = sinf((float)((i + 0.5) * (M_PI / (2.0 * n))));
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

	for (int book = 0; book < 11; book++) {
		const uint16_t* codes = ff_aac_spectral_codes[book];
		const uint8_t* bits = ff_aac_spectral_bits[book];
		const uint16_t* idx = ff_aac_codebook_vector_idx[book];
		if (!vlc_build(&g_vlc_spectral[book], 8, ff_aac_spectral_sizes[book],
				[=](int i) { return (uint32_t)codes[i]; }, [=](int i) { return (int)bits[i]; }, [=](int i) { return idx[i]; }))
			return 0;
	}
	if (!vlc_build(&g_vlc_scalefactors, 8, 121,
			[](int i) { return ff_aac_scalefactor_code[i]; }, [](int i) { return (int)ff_aac_scalefactor_bits[i]; },
			[](int i) { return (uint16_t)i; }))
		return 0;

	// ff_vlc_init_tables_from_lengths: the codes are canonical, in table order.
	const uint8_t (*tab)[2] = sbr_huffman_tab;
	for (int table = 0; table < 10; table++) {
		const int n = sbr_huffman_nb_codes[table];
		static uint32_t codes[121];
		uint32_t code = 0;
		for (int i = 0; i < n; i++) {
			codes[i] = code >> (32 - tab[i][1]);
			code += 1u << (32 - tab[i][1]);
		}
		if (!vlc_build(&g_vlc_sbr[table], 9, n, [](int i) { return codes[i]; }, [=](int i) { return (int)tab[i][1]; },
				[=](int i) { return (uint16_t)tab[i][0]; }))
			return 0;
		tab += n;
	}

	for (int i = 0; i < 428; i++) g_pow2sf_tab[i] = (float)pow(2.0, (i - POW_SF2_ZERO) / 4.0);
	for (int i = 0; i < (1 << 13); i++) g_cbrt_tab[i] = (float)(i * cbrt((double)i));

	sine_window_init(g_sine_1024, 1024);
	sine_window_init(g_sine_128, 128);
	kbd_window_init(g_kbd_long_1024, 4.0, 1024);
	kbd_window_init(g_kbd_short_128, 6.0, 128);

	mdct_exp_init(g_mdct_exp_long, 1024, 1.0 / 1024 / 32768.0);
	mdct_exp_init(g_mdct_exp_short, 128, 1.0 / 128 / 32768.0);
	// SBR works on samples scaled to +/-32768, so analysis scales up and synthesis back down.
	mdct_exp_init(g_sbr_exp_analysis, 64, -2.0 * 32768);
	mdct_exp_init(g_sbr_exp_synthesis, 64, 1.0 / (64 * 32768));
	for (int i = 0; i < 256; i++) {
		const double ang = 2.0 * M_PI * i / 512;
		g_fft_tw[i].re = cos(ang);
		g_fft_tw[i].im = sin(ang);
	}

	g_tables_ready = 1;
	return 1;
}

///////////////////////////////////////////////////////////////////////////////
// IMDCT: ff_tx_mdct_inv over a radix-2 inverse FFT. `len` coefficients go in and `len` samples come
// out: the half of the 2*len outputs the rest can be recovered from, as lavu's inverse MDCT returns.

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
		const int stride = 512 / size;
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

static void imdct_half_exp(float* dst, const float* src, int len, const Cpx* exp)
{
	static Cpx z[512];
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

static void imdct_half(float* dst, const float* src, int len)
{
	imdct_half_exp(dst, src, len, len == 1024 ? g_mdct_exp_long : g_mdct_exp_short);
}

///////////////////////////////////////////////////////////////////////////////
// Context

struct MPEG4AudioConfig {
	int object_type;
	int sampling_index;
	int sample_rate;
	int chan_config;
	int sbr;				// -1 implicit, 1 present
	int ext_object_type;
	int ext_sampling_index;
	int ext_sample_rate;
	int ext_chan_config;
	int channels;
	int ps;
};

struct OutputConfiguration {
	MPEG4AudioConfig m4ac;
	uint8_t layout_map[MAX_ELEM_ID * 4][3];
	int layout_map_tags;
	int channels;
	OCStatus status;
};

struct TemporalNoiseShaping {
	int present;
	int n_filt[8];
	int length[8][4];
	int direction[8][4];
	int order[8][4];
	float coef[8][4][TNS_MAX_ORDER];
};

struct IndividualChannelStream {
	uint8_t max_sfb;
	uint8_t window_sequence[2];
	uint8_t use_kb_window[2];
	int num_window_groups;
	uint8_t group_len[8];
	const uint16_t* swb_offset;
	int num_swb;
	int num_windows;
	int tns_max_bands;
};

struct Pulse {
	int num_pulse;
	int pos[4];
	int amp[4];
};

struct ChannelCoupling {
	CouplingPoint coupling_point;
	int num_coupled;
	int type[8];
	int id_select[8];
	int ch_select[8];
	float gain[16][120];
};

struct SingleChannelElement {
	IndividualChannelStream ics;
	TemporalNoiseShaping tns;
	uint8_t band_type[128];
	int sfo[128];
	float sf[128];
	float coeffs[1024];
	float saved[1024];
	float ret_buf[2048];	// 1024 samples, or 2048 once SBR has run over them
};

struct ChannelElement {
	int present;
	int max_sfb_ste;
	uint8_t ms_mask[128];
	SingleChannelElement ch[2];
	ChannelCoupling coup;
};

struct AACContext {
	OutputConfiguration oc[2];
	ChannelElement* che[4][MAX_ELEM_ID];
	ChannelElement* tag_che_map[4][MAX_ELEM_ID];
	SingleChannelElement* output_element[MAX_CHANNELS];
	int tags_mapped;
	int channels;
	int random_state;
	int ps_present;			// Parametric Stereo data was seen and stepped over
	int seek_frame;			// what aac_reset was told, for an element that first shows up after it
	int stats[AAC_INFO_COUNT];
	float buf_mdct[1024];
	float temp[128];
	float silence[2048];
};

#include "aac_sbr.h"

///////////////////////////////////////////////////////////////////////////////
// Channel configuration (aacdec.c)

static int che_configure(AACContext* ac, int che_pos, int type, int id, int* channels)
{
	if (*channels >= MAX_CHANNELS) return AVERROR_INVALIDDATA;
	if (che_pos) {
		if (!ac->che[type][id]) {
			ac->che[type][id] = (ChannelElement*)malloc(sizeof(ExtChannelElement));
			if (!ac->che[type][id]) return AVERROR_INVALIDDATA;
			memset(ac->che[type][id], 0, sizeof(ChannelElement));
			sbr_ctx_init(get_sbr(ac->che[type][id]), type, ac->seek_frame);
		}
		if (type != TYPE_CCE) {
			if (*channels >= MAX_CHANNELS - (type == TYPE_CPE)) return AVERROR_INVALIDDATA;
			ac->output_element[(*channels)++] = &ac->che[type][id]->ch[0];
			if (type == TYPE_CPE) ac->output_element[(*channels)++] = &ac->che[type][id]->ch[1];
		}
	} else {
		if (ac->che[type][id]) {
			for (int i = 0; i < 4; i++)
				for (int j = 0; j < MAX_ELEM_ID; j++)
					if (ac->tag_che_map[i][j] == ac->che[type][id]) ac->tag_che_map[i][j] = NULL;
		}
		free(ac->che[type][id]);
		ac->che[type][id] = NULL;
		memset(ac->output_element, 0, sizeof(ac->output_element));
	}
	return 0;
}

struct elem_to_channel {
	uint64_t av_position;
	uint8_t syn_ele;
	uint8_t elem_id;
	uint8_t aac_position;
};

static int assign_pair(elem_to_channel e2c_vec[MAX_ELEM_ID], uint8_t (*layout_map)[3], int offset, uint64_t left,
	uint64_t right, int pos, uint64_t* layout)
{
	if (layout_map[offset][0] == TYPE_CPE) {
		e2c_vec[offset] = elem_to_channel{ left | right, TYPE_CPE, layout_map[offset][1], (uint8_t)pos };
		if (e2c_vec[offset].av_position != UINT64_MAX) *layout |= e2c_vec[offset].av_position;
		return 1;
	}
	e2c_vec[offset] = elem_to_channel{ left, TYPE_SCE, layout_map[offset][1], (uint8_t)pos };
	e2c_vec[offset + 1] = elem_to_channel{ right, TYPE_SCE, layout_map[offset + 1][1], (uint8_t)pos };
	if (left != UINT64_MAX) *layout |= left;
	if (right != UINT64_MAX) *layout |= right;
	return 2;
}

static int count_paired_channels(uint8_t (*layout_map)[3], int tags, int pos, int current)
{
	int num_pos_channels = 0;
	int first_cpe = 0;
	int sce_parity = 0;
	for (int i = current; i < tags; i++) {
		if (layout_map[i][2] != pos) break;
		if (layout_map[i][0] == TYPE_CPE) {
			if (sce_parity) {
				if (pos == AAC_CHANNEL_FRONT && !first_cpe) sce_parity = 0;
				else return -1;
			}
			num_pos_channels += 2;
			first_cpe = 1;
		} else {
			num_pos_channels++;
			sce_parity ^= (pos != AAC_CHANNEL_LFE);
		}
	}
	if (sce_parity && (pos == AAC_CHANNEL_FRONT && first_cpe)) return -1;
	return num_pos_channels;
}

static int assign_channels(elem_to_channel e2c_vec[MAX_ELEM_ID], uint8_t (*layout_map)[3], uint64_t* layout, int tags,
	int layer, int pos, int* current)
{
	int i = *current, j = 0;
	int nb_channels = count_paired_channels(layout_map, tags, pos, i);

	if (nb_channels < 0 || nb_channels > 5) return 0;

	if (pos == AAC_CHANNEL_LFE) {
		while (nb_channels) {
			if (ff_aac_channel_map[layer][pos - 1][j] == AV_CHAN_NONE) return -1;
			e2c_vec[i] = elem_to_channel{ 1ULL << ff_aac_channel_map[layer][pos - 1][j], layout_map[i][0], layout_map[i][1], (uint8_t)pos };
			*layout |= e2c_vec[i].av_position;
			i++;
			j++;
			nb_channels--;
		}
		*current = i;
		return 0;
	}

	while (nb_channels & 1) {
		if (ff_aac_channel_map[layer][pos - 1][0] == AV_CHAN_NONE) return -1;
		if (ff_aac_channel_map[layer][pos - 1][0] == AV_CHAN_UNUSED) break;
		e2c_vec[i] = elem_to_channel{ 1ULL << ff_aac_channel_map[layer][pos - 1][0], layout_map[i][0], layout_map[i][1], (uint8_t)pos };
		*layout |= e2c_vec[i].av_position;
		i++;
		nb_channels--;
	}

	j = (pos != AAC_CHANNEL_SIDE) && nb_channels <= 3 ? 3 : 1;
	while (nb_channels >= 2) {
		if (ff_aac_channel_map[layer][pos - 1][j] == AV_CHAN_NONE || ff_aac_channel_map[layer][pos - 1][j + 1] == AV_CHAN_NONE)
			return -1;
		i += assign_pair(e2c_vec, layout_map, i, 1ULL << ff_aac_channel_map[layer][pos - 1][j],
			1ULL << ff_aac_channel_map[layer][pos - 1][j + 1], pos, layout);
		j += 2;
		nb_channels -= 2;
	}
	while (nb_channels & 1) {
		if (ff_aac_channel_map[layer][pos - 1][5] == AV_CHAN_NONE) return -1;
		e2c_vec[i] = elem_to_channel{ 1ULL << ff_aac_channel_map[layer][pos - 1][5], layout_map[i][0], layout_map[i][1], (uint8_t)pos };
		*layout |= e2c_vec[i].av_position;
		i++;
		nb_channels--;
	}
	if (nb_channels) return -1;

	*current = i;
	return 0;
}

static void swap_e2c(elem_to_channel* a, elem_to_channel* b)
{
	const elem_to_channel t = *a;
	*a = *b;
	*b = t;
}

/** Sorts layout_map into the order FFmpeg outputs the channels; leaves it alone when no order fits. */
static void sniff_channel_order(uint8_t (*layout_map)[3], int tags)
{
	int i, n, total_non_cc_elements;
	elem_to_channel e2c_vec[4 * MAX_ELEM_ID];
	uint64_t layout = 0;

	memset(e2c_vec, 0, sizeof(e2c_vec));
	if (4 * MAX_ELEM_ID < tags) return;

	for (n = 0, i = 0; n < 3 && i < tags; n++) {
		if (assign_channels(e2c_vec, layout_map, &layout, tags, n, AAC_CHANNEL_FRONT, &i) < 0) return;
		if (assign_channels(e2c_vec, layout_map, &layout, tags, n, AAC_CHANNEL_SIDE, &i) < 0) return;
		if (assign_channels(e2c_vec, layout_map, &layout, tags, n, AAC_CHANNEL_BACK, &i) < 0) return;
		if (assign_channels(e2c_vec, layout_map, &layout, tags, n, AAC_CHANNEL_LFE, &i) < 0) return;
	}

	total_non_cc_elements = n = i;

	if (layout == AV_CH_LAYOUT_22POINT2) {
		swap_e2c(&e2c_vec[2], &e2c_vec[0]);
		swap_e2c(&e2c_vec[2], &e2c_vec[1]);
		swap_e2c(&e2c_vec[6], &e2c_vec[2]);
		swap_e2c(&e2c_vec[4], &e2c_vec[3]);
		swap_e2c(&e2c_vec[6], &e2c_vec[4]);
		swap_e2c(&e2c_vec[7], &e2c_vec[6]);
		swap_e2c(&e2c_vec[9], &e2c_vec[8]);
		swap_e2c(&e2c_vec[11], &e2c_vec[10]);
		swap_e2c(&e2c_vec[12], &e2c_vec[11]);
	} else {
		do {
			int next_n = 0;
			for (i = 1; i < n; i++)
				if (e2c_vec[i - 1].av_position > e2c_vec[i].av_position) {
					swap_e2c(&e2c_vec[i - 1], &e2c_vec[i]);
					next_n = i;
				}
			n = next_n;
		} while (n > 0);
	}

	for (i = 0; i < total_non_cc_elements; i++) {
		layout_map[i][0] = e2c_vec[i].syn_ele;
		layout_map[i][1] = e2c_vec[i].elem_id;
		layout_map[i][2] = e2c_vec[i].aac_position;
	}
}

static int push_output_configuration(AACContext* ac)
{
	int pushed = 0;
	if (ac->oc[1].status == OC_LOCKED || ac->oc[0].status == OC_NONE) {
		ac->oc[0] = ac->oc[1];
		pushed = 1;
	}
	ac->oc[1].status = OC_NONE;
	return pushed;
}

static int output_configure(AACContext* ac, uint8_t layout_map[MAX_ELEM_ID * 4][3], int tags, OCStatus oc_type);

static void pop_output_configuration(AACContext* ac)
{
	if (ac->oc[1].status != OC_LOCKED && ac->oc[0].status != OC_NONE) {
		ac->oc[1] = ac->oc[0];
		ac->channels = ac->oc[1].channels;
		output_configure(ac, ac->oc[1].layout_map, ac->oc[1].layout_map_tags, ac->oc[1].status);
	}
}

static int output_configure(AACContext* ac, uint8_t layout_map[MAX_ELEM_ID * 4][3], int tags, OCStatus oc_type)
{
	int channels = 0;
	uint8_t id_map[TYPE_END][MAX_ELEM_ID];
	uint8_t type_counts[TYPE_END];

	memset(id_map, 0, sizeof(id_map));
	memset(type_counts, 0, sizeof(type_counts));

	if (ac->oc[1].layout_map != layout_map) {
		memcpy(ac->oc[1].layout_map, layout_map, tags * sizeof(layout_map[0]));
		ac->oc[1].layout_map_tags = tags;
	}
	for (int i = 0; i < tags; i++) {
		const int type = layout_map[i][0];
		const int id = layout_map[i][1];
		id_map[type][id] = type_counts[type]++;
		if (id_map[type][id] >= MAX_ELEM_ID) return AVERROR_UNSUPPORTED;
	}
	sniff_channel_order(layout_map, tags);
	for (int i = 0; i < tags; i++) {
		const int type = layout_map[i][0];
		const int id = layout_map[i][1];
		const int iid = id_map[type][id];
		const int position = layout_map[i][2];
		const int ret = che_configure(ac, position, type, iid, &channels);
		if (ret < 0) return ret;
		ac->tag_che_map[type][id] = ac->che[type][iid];
	}

	ac->oc[1].channels = channels;
	ac->channels = channels;
	ac->oc[1].status = oc_type;
	return 0;
}

static int set_default_channel_config(uint8_t (*layout_map)[3], int* tags, int channel_config)
{
	if (channel_config < 1 || (channel_config > 7 && channel_config < 11) || channel_config > 14) return AVERROR_INVALIDDATA;
	*tags = ff_tags_per_config[channel_config];
	memcpy(layout_map, ff_aac_channel_layout_map[channel_config - 1], *tags * sizeof(*layout_map));

	// As FFmpeg without -strict: 7.1(wide) is rare, and Nero and others put the sides there.
	if (channel_config == 7) layout_map[2][2] = AAC_CHANNEL_BACK;
	return 0;
}

static ChannelElement* get_che(AACContext* ac, int type, int elem_id)
{
	// For PCE based channel configurations map the channels solely based on tags.
	if (!ac->oc[1].m4ac.chan_config) return ac->tag_che_map[type][elem_id];

	// Allow single CPE stereo files to be signaled with mono configuration.
	if (!ac->tags_mapped && type == TYPE_CPE && ac->oc[1].m4ac.chan_config == 1) {
		uint8_t layout_map[MAX_ELEM_ID * 4][3];
		int layout_map_tags;
		push_output_configuration(ac);
		if (set_default_channel_config(layout_map, &layout_map_tags, 2) < 0) return NULL;
		if (output_configure(ac, layout_map, layout_map_tags, OC_TRIAL_FRAME) < 0) return NULL;
		ac->oc[1].m4ac.chan_config = 2;
	}
	// And vice-versa
	if (!ac->tags_mapped && type == TYPE_SCE && ac->oc[1].m4ac.chan_config == 2) {
		uint8_t layout_map[MAX_ELEM_ID * 4][3];
		push_output_configuration(ac);
		layout_map[0][0] = layout_map[1][0] = TYPE_SCE;
		layout_map[0][2] = layout_map[1][2] = AAC_CHANNEL_FRONT;
		layout_map[0][1] = 0;
		layout_map[1][1] = 1;
		if (output_configure(ac, layout_map, 2, OC_TRIAL_FRAME) < 0) return NULL;
	}

	// For indexed channel configurations map the channels solely based on position.
	switch (ac->oc[1].m4ac.chan_config) {
	case 14:
		if (ac->tags_mapped > 2 && ((type == TYPE_CPE && elem_id < 3) || (type == TYPE_LFE && elem_id < 1))) {
			ac->tags_mapped++;
			return ac->tag_che_map[type][elem_id] = ac->che[type][elem_id];
		}
		// fallthrough
	case 13:
		if (ac->tags_mapped > 3 && ((type == TYPE_CPE && elem_id < 8) || (type == TYPE_SCE && elem_id < 6) ||
				(type == TYPE_LFE && elem_id < 2))) {
			ac->tags_mapped++;
			return ac->tag_che_map[type][elem_id] = ac->che[type][elem_id];
		}
		// fallthrough
	case 12:
	case 7:
		if (ac->tags_mapped == 3 && type == TYPE_CPE) {
			ac->tags_mapped++;
			return ac->tag_che_map[TYPE_CPE][elem_id] = ac->che[TYPE_CPE][2];
		}
		// fallthrough
	case 11:
		if (ac->tags_mapped == 3 && type == TYPE_SCE) {
			ac->tags_mapped++;
			return ac->tag_che_map[TYPE_SCE][elem_id] = ac->che[TYPE_SCE][1];
		}
		// fallthrough
	case 6:
		// Some streams incorrectly code 5.1 audio as SCE[0] CPE[0] CPE[1] SCE[1] instead of
		// SCE[0] CPE[0] CPE[1] LFE[0]: transfer the LFE[0] element to the SCE[1]'s mapping.
		if (ac->tags_mapped == ff_tags_per_config[ac->oc[1].m4ac.chan_config] - 1 && (type == TYPE_LFE || type == TYPE_SCE)) {
			ac->tags_mapped++;
			return ac->tag_che_map[type][elem_id] = ac->che[TYPE_LFE][0];
		}
		// fallthrough
	case 5:
		if (ac->tags_mapped == 2 && type == TYPE_CPE) {
			ac->tags_mapped++;
			return ac->tag_che_map[TYPE_CPE][elem_id] = ac->che[TYPE_CPE][1];
		}
		// fallthrough
	case 4:
		// Some streams incorrectly code 4.0 audio as SCE[0] CPE[0] LFE[0] instead of
		// SCE[0] CPE[0] SCE[1]: transfer the SCE[1] element to the LFE[0]'s mapping.
		if (ac->tags_mapped == ff_tags_per_config[ac->oc[1].m4ac.chan_config] - 1 && (type == TYPE_LFE || type == TYPE_SCE)) {
			ac->tags_mapped++;
			return ac->tag_che_map[type][elem_id] = ac->che[TYPE_SCE][1];
		}
		if (ac->tags_mapped == 2 && ac->oc[1].m4ac.chan_config == 4 && type == TYPE_SCE) {
			ac->tags_mapped++;
			return ac->tag_che_map[TYPE_SCE][elem_id] = ac->che[TYPE_SCE][1];
		}
		// fallthrough
	case 3:
	case 2:
		if (ac->tags_mapped == (ac->oc[1].m4ac.chan_config != 2) && type == TYPE_CPE) {
			ac->tags_mapped++;
			return ac->tag_che_map[TYPE_CPE][elem_id] = ac->che[TYPE_CPE][0];
		} else if (ac->tags_mapped == 1 && ac->oc[1].m4ac.chan_config == 2 && type == TYPE_SCE) {
			ac->tags_mapped++;
			return ac->tag_che_map[TYPE_SCE][elem_id] = ac->che[TYPE_SCE][1];
		}
		// fallthrough
	case 1:
		if (!ac->tags_mapped && type == TYPE_SCE) {
			ac->tags_mapped++;
			return ac->tag_che_map[TYPE_SCE][elem_id] = ac->che[TYPE_SCE][0];
		}
		// fallthrough
	default:
		return NULL;
	}
}

static void decode_channel_map(uint8_t layout_map[][3], int type, GetBitContext* gb, int n)
{
	while (n--) {
		int syn_ele;
		switch (type) {
		case AAC_CHANNEL_FRONT:
		case AAC_CHANNEL_BACK:
		case AAC_CHANNEL_SIDE:
			syn_ele = get_bits1(gb);
			break;
		case AAC_CHANNEL_CC:
			skip_bits(gb, 1);
			syn_ele = TYPE_CCE;
			break;
		default:	// AAC_CHANNEL_LFE
			syn_ele = TYPE_LFE;
			break;
		}
		layout_map[0][0] = (uint8_t)syn_ele;
		layout_map[0][1] = (uint8_t)get_bits(gb, 4);
		layout_map[0][2] = (uint8_t)type;
		layout_map++;
	}
}

static int decode_pce(uint8_t (*layout_map)[3], GetBitContext* gb, int byte_align_ref)
{
	skip_bits(gb, 2);	// object_type
	skip_bits(gb, 4);	// sampling_index

	const int num_front = get_bits(gb, 4);
	const int num_side = get_bits(gb, 4);
	const int num_back = get_bits(gb, 4);
	const int num_lfe = get_bits(gb, 2);
	const int num_assoc_data = get_bits(gb, 3);
	const int num_cc = get_bits(gb, 4);

	if (get_bits1(gb)) skip_bits(gb, 4);	// mono_mixdown_tag
	if (get_bits1(gb)) skip_bits(gb, 4);	// stereo_mixdown_tag
	if (get_bits1(gb)) skip_bits(gb, 3);	// mixdown_coeff_index and pseudo_surround

	if (get_bits_left(gb) < 5 * (num_front + num_side + num_back + num_cc) + 4 * (num_lfe + num_assoc_data + num_cc))
		return -1;

	decode_channel_map(layout_map, AAC_CHANNEL_FRONT, gb, num_front);
	int tags = num_front;
	decode_channel_map(layout_map + tags, AAC_CHANNEL_SIDE, gb, num_side);
	tags += num_side;
	decode_channel_map(layout_map + tags, AAC_CHANNEL_BACK, gb, num_back);
	tags += num_back;
	decode_channel_map(layout_map + tags, AAC_CHANNEL_LFE, gb, num_lfe);
	tags += num_lfe;

	skip_bits_long(gb, 4 * num_assoc_data);

	decode_channel_map(layout_map + tags, AAC_CHANNEL_CC, gb, num_cc);
	tags += num_cc;

	const int n = (byte_align_ref - get_bits_count(gb)) & 7;
	if (n) skip_bits(gb, n);

	const int comment_len = get_bits(gb, 8) * 8;
	if (get_bits_left(gb) < comment_len) return AVERROR_INVALIDDATA;
	skip_bits_long(gb, comment_len);
	return tags;
}

static int count_channels(uint8_t (*layout)[3], int tags)
{
	int sum = 0;
	for (int i = 0; i < tags; i++) {
		const int syn_ele = layout[i][0];
		const int pos = layout[i][2];
		sum += (1 + (syn_ele == TYPE_CPE)) * (pos != AAC_CHANNEL_OFF && pos != AAC_CHANNEL_CC);
	}
	return sum;
}

static int decode_ga_specific_config(AACContext* ac, GetBitContext* gb, int channel_config)
{
	uint8_t layout_map[MAX_ELEM_ID * 4][3];
	int tags = 0;

	if (get_bits1(gb)) return AVERROR_UNSUPPORTED;	// frame_length_short: 960 samples
	if (get_bits1(gb)) skip_bits(gb, 14);			// dependsOnCoreCoder: coreCoderDelay
	const int extension_flag = get_bits1(gb);

	if (channel_config == 0) {
		skip_bits(gb, 4);	// element_instance_tag
		tags = decode_pce(layout_map, gb, 0);
		if (tags < 0) return tags;
	} else {
		const int ret = set_default_channel_config(layout_map, &tags, channel_config);
		if (ret) return ret;
	}

	if (count_channels(layout_map, tags) > 1) {
		ac->oc[1].m4ac.ps = 0;
	} else if (ac->oc[1].m4ac.sbr == 1 && ac->oc[1].m4ac.ps == -1) {
		ac->oc[1].m4ac.ps = 1;
	}

	const int ret = output_configure(ac, layout_map, tags, OC_GLOBAL_HDR);
	if (ret) return ret;

	if (extension_flag) skip_bits(gb, 1);	// extensionFlag3
	return 0;
}

static int get_object_type(GetBitContext* gb)
{
	int object_type = get_bits(gb, 5);
	if (object_type == AOT_ESCAPE) object_type = 32 + get_bits(gb, 6);
	return object_type;
}

static int get_sample_rate(GetBitContext* gb, int* index)
{
	*index = get_bits(gb, 4);
	return *index == 0x0f ? get_bits(gb, 24) : ff_mpeg4audio_sample_rates[*index];
}

/** ff_mpeg4audio_get_config_gb with sync_extension set. */
static int mpeg4audio_get_config(MPEG4AudioConfig* c, GetBitContext* gb)
{
	c->object_type = get_object_type(gb);
	c->sample_rate = get_sample_rate(gb, &c->sampling_index);
	c->chan_config = get_bits(gb, 4);
	if (c->chan_config >= 15) return AVERROR_INVALIDDATA;
	c->channels = ff_mpeg4audio_channels[c->chan_config];
	c->sbr = -1;
	c->ps = -1;
	if (c->object_type == AOT_SBR || (c->object_type == AOT_PS &&
			// check for W6132 Annex YYYY draft MP3onMP4
			!(show_bits(gb, 3) & 0x03 && !(show_bits(gb, 9) & 0x3F)))) {
		if (c->object_type == AOT_PS) c->ps = 1;
		c->ext_object_type = AOT_SBR;
		c->sbr = 1;
		c->ext_sample_rate = get_sample_rate(gb, &c->ext_sampling_index);
		c->object_type = get_object_type(gb);
		if (c->object_type == AOT_ER_BSAC) c->ext_chan_config = get_bits(gb, 4);
	} else {
		c->ext_object_type = AOT_NULL;
		c->ext_sample_rate = 0;
	}
	const int specific_config_bitindex = get_bits_count(gb);

	if (c->ext_object_type != AOT_SBR) {
		while (get_bits_left(gb) > 15) {
			if (show_bits(gb, 11) == 0x2b7) {	// sync extension
				get_bits(gb, 11);
				c->ext_object_type = get_object_type(gb);
				if (c->ext_object_type == AOT_SBR && (c->sbr = get_bits1(gb)) == 1) {
					c->ext_sample_rate = get_sample_rate(gb, &c->ext_sampling_index);
					if (c->ext_sample_rate == c->sample_rate) c->sbr = -1;
				}
				if (get_bits_left(gb) > 11 && get_bits(gb, 11) == 0x548) c->ps = get_bits1(gb);
				break;
			} else {
				get_bits1(gb);	// skip 1 bit
			}
		}
	}
	return specific_config_bitindex;
}

static int decode_audio_specific_config(AACContext* ac, const uint8_t* data, int bit_size)
{
	GetBitContext gb, gbc;
	MPEG4AudioConfig* m4ac = &ac->oc[1].m4ac;
	const MPEG4AudioConfig m4ac_bak = *m4ac;

	init_get_bits(&gb, data, bit_size);
	gbc = gb;

	const int i = mpeg4audio_get_config(m4ac, &gbc);
	if (i < 0 || m4ac->sampling_index > 12) {
		*m4ac = m4ac_bak;
		return AVERROR_INVALIDDATA;
	}
	skip_bits_long(&gb, i);

	if (m4ac->object_type != AOT_AAC_LC) return AVERROR_UNSUPPORTED;
	return decode_ga_specific_config(ac, &gb, m4ac->chan_config);
}

///////////////////////////////////////////////////////////////////////////////
// Syntax (aacdec.c)

static int skip_data_stream_element(GetBitContext* gb)
{
	const int byte_align = get_bits1(gb);
	int count = get_bits(gb, 8);
	if (count == 255) count += get_bits(gb, 8);
	if (byte_align) align_get_bits(gb);
	if (get_bits_left(gb) < 8 * count) return AVERROR_INVALIDDATA;
	skip_bits_long(gb, 8 * count);
	return 0;
}

static int decode_ics_info(AACContext* ac, IndividualChannelStream* ics, GetBitContext* gb)
{
	const int sampling_index = ac->oc[1].m4ac.sampling_index;

	skip_bits(gb, 1);	// reserved; FFmpeg only logs it
	ics->window_sequence[1] = ics->window_sequence[0];
	ics->window_sequence[0] = (uint8_t)get_bits(gb, 2);
	ics->use_kb_window[1] = ics->use_kb_window[0];
	ics->use_kb_window[0] = (uint8_t)get_bits1(gb);
	ics->num_window_groups = 1;
	ics->group_len[0] = 1;
	if (ics->window_sequence[0] == EIGHT_SHORT_SEQUENCE) {
		ics->max_sfb = (uint8_t)get_bits(gb, 4);
		for (int i = 0; i < 7; i++) {
			if (get_bits1(gb)) {
				ics->group_len[ics->num_window_groups - 1]++;
			} else {
				ics->num_window_groups++;
				ics->group_len[ics->num_window_groups - 1] = 1;
			}
		}
		ics->num_windows = 8;
		ics->swb_offset = ff_swb_offset_128[sampling_index];
		ics->num_swb = ff_aac_num_swb_128[sampling_index];
		ics->tns_max_bands = ff_tns_max_bands_128[sampling_index];
	} else {
		ics->max_sfb = (uint8_t)get_bits(gb, 6);
		ics->num_windows = 1;
		ics->num_swb = ff_aac_num_swb_1024[sampling_index];
		ics->swb_offset = ff_swb_offset_1024[sampling_index];
		ics->tns_max_bands = ff_tns_max_bands_1024[sampling_index];
		if (get_bits1(gb)) {	// predictor_data_present: not allowed in AAC-LC
			ics->max_sfb = 0;
			return AVERROR_INVALIDDATA;
		}
	}

	if (ics->max_sfb > ics->num_swb) {
		ics->max_sfb = 0;
		return AVERROR_INVALIDDATA;
	}
	ac->stats[AAC_INFO_SHORT] += ics->window_sequence[0] == EIGHT_SHORT_SEQUENCE;
	ac->stats[AAC_INFO_KBD] += ics->use_kb_window[0];
	return 0;
}

static int decode_band_types(SingleChannelElement* sce, GetBitContext* gb)
{
	IndividualChannelStream* ics = &sce->ics;
	const int bits = (ics->window_sequence[0] == EIGHT_SHORT_SEQUENCE) ? 3 : 5;

	for (int g = 0; g < ics->num_window_groups; g++) {
		int k = 0;
		while (k < ics->max_sfb) {
			uint8_t sect_end = (uint8_t)k;
			int sect_len_incr;
			const int sect_band_type = get_bits(gb, 4);
			if (sect_band_type == 12) return AVERROR_INVALIDDATA;
			do {
				sect_len_incr = get_bits(gb, bits);
				sect_end = (uint8_t)(sect_end + sect_len_incr);
				if (get_bits_left(gb) < 0) return AVERROR_INVALIDDATA;
				if (sect_end > ics->max_sfb) return AVERROR_INVALIDDATA;
			} while (sect_len_incr == (1 << bits) - 1);
			for (; k < sect_end; k++) sce->band_type[g * ics->max_sfb + k] = (uint8_t)sect_band_type;
		}
	}
	return 0;
}

static inline int av_clip(int a, int amin, int amax)
{
	return a < amin ? amin : a > amax ? amax : a;
}

static int decode_scalefactors(AACContext* ac, SingleChannelElement* sce, GetBitContext* gb, unsigned int global_gain)
{
	IndividualChannelStream* ics = &sce->ics;
	int offset[3] = { (int)global_gain, (int)global_gain - NOISE_OFFSET, 0 };
	int noise_flag = 1;

	for (int g = 0; g < ics->num_window_groups; g++) {
		for (int sfb = 0; sfb < ics->max_sfb; sfb++) {
			const int idx = g * ics->max_sfb + sfb;
			switch (sce->band_type[idx]) {
			case ZERO_BT:
				sce->sfo[idx] = 0;
				break;
			case INTENSITY_BT:	// fallthrough
			case INTENSITY_BT2:
				offset[2] += get_vlc(gb, &g_vlc_scalefactors) - SCALE_DIFF_ZERO;
				sce->sfo[idx] = av_clip(offset[2], -155, 100) - 100;
				ac->stats[AAC_INFO_INTENSITY]++;
				break;
			case NOISE_BT:
				if (noise_flag-- > 0) offset[1] += get_bits(gb, NOISE_PRE_BITS) - NOISE_PRE;
				else offset[1] += get_vlc(gb, &g_vlc_scalefactors) - SCALE_DIFF_ZERO;
				sce->sfo[idx] = av_clip(offset[1], -100, 155);
				ac->stats[AAC_INFO_PNS]++;
				break;
			default:
				offset[0] += get_vlc(gb, &g_vlc_scalefactors) - SCALE_DIFF_ZERO;
				if ((unsigned)offset[0] > 255U) return AVERROR_INVALIDDATA;
				sce->sfo[idx] = offset[0] - 100;
				break;
			}
		}
	}
	return 0;
}

static void dequant_scalefactors(SingleChannelElement* sce)
{
	IndividualChannelStream* ics = &sce->ics;
	const int* sfo = sce->sfo;
	float* sf = sce->sf;

	int idx = 0;
	for (int g = 0; g < ics->num_window_groups; g++) {
		for (int sfb = 0; sfb < ics->max_sfb; sfb++, idx++) {
			switch (sce->band_type[g * ics->max_sfb + sfb]) {
			case ZERO_BT:
				sf[idx] = 0.f;
				break;
			case INTENSITY_BT:	// fallthrough
			case INTENSITY_BT2:
				sf[idx] = g_pow2sf_tab[-sfo[idx] - 100 + POW_SF2_ZERO];
				break;
			case NOISE_BT:
				sf[idx] = -g_pow2sf_tab[sfo[idx] + POW_SF2_ZERO];
				break;
			default:
				sf[idx] = -g_pow2sf_tab[sfo[idx] + POW_SF2_ZERO];
				break;
			}
		}
	}
}

static int decode_pulses(Pulse* pulse, GetBitContext* gb, const uint16_t* swb_offset, int num_swb)
{
	pulse->num_pulse = get_bits(gb, 2) + 1;
	const int pulse_swb = get_bits(gb, 6);
	if (pulse_swb >= num_swb) return -1;
	pulse->pos[0] = swb_offset[pulse_swb];
	pulse->pos[0] += get_bits(gb, 5);
	if (pulse->pos[0] >= swb_offset[num_swb]) return -1;
	pulse->amp[0] = get_bits(gb, 4);
	for (int i = 1; i < pulse->num_pulse; i++) {
		pulse->pos[i] = get_bits(gb, 5) + pulse->pos[i - 1];
		if (pulse->pos[i] >= swb_offset[num_swb]) return -1;
		pulse->amp[i] = get_bits(gb, 4);
	}
	return 0;
}

static int decode_tns(TemporalNoiseShaping* tns, GetBitContext* gb, const IndividualChannelStream* ics)
{
	const int is8 = ics->window_sequence[0] == EIGHT_SHORT_SEQUENCE;
	const int tns_max_order = is8 ? 7 : 12;

	for (int w = 0; w < ics->num_windows; w++) {
		if ((tns->n_filt[w] = get_bits(gb, 2 - is8))) {
			const int coef_res = get_bits1(gb);

			for (int filt = 0; filt < tns->n_filt[w]; filt++) {
				tns->length[w][filt] = get_bits(gb, 6 - 2 * is8);
				tns->order[w][filt] = get_bits(gb, 5 - (2 * is8));

				if (tns->order[w][filt] > tns_max_order) {
					tns->order[w][filt] = 0;
					return AVERROR_INVALIDDATA;
				}
				if (tns->order[w][filt]) {
					tns->direction[w][filt] = get_bits1(gb);
					const int coef_compress = get_bits1(gb);
					const int coef_len = coef_res + 3 - coef_compress;
					const int tmp2_idx = 2 * coef_compress + coef_res;

					for (int i = 0; i < tns->order[w][filt]; i++)
						tns->coef[w][filt][i] = ff_tns_tmp2_map[tmp2_idx][get_bits(gb, coef_len)];
				}
			}
		}
	}
	return 0;
}

static void decode_gain_control(SingleChannelElement* sce, GetBitContext* gb)
{
	// wd_num, wd_test, aloc_size
	static const uint8_t gain_mode[4][3] = {
		{ 1, 0, 5 },	// ONLY_LONG_SEQUENCE = 0,
		{ 2, 1, 2 },	// LONG_START_SEQUENCE,
		{ 8, 0, 2 },	// EIGHT_SHORT_SEQUENCE,
		{ 2, 1, 5 },	// LONG_STOP_SEQUENCE
	};

	const int mode = sce->ics.window_sequence[0];
	const int max_band = get_bits(gb, 2);
	for (int bd = 0; bd < max_band; bd++) {
		for (int wd = 0; wd < gain_mode[mode][0]; wd++) {
			const int adjust_num = get_bits(gb, 3);
			for (int ad = 0; ad < adjust_num; ad++)
				skip_bits(gb, 4 + ((wd == 0 && gain_mode[mode][1]) ? 4 : gain_mode[mode][2]));
		}
	}
}

///////////////////////////////////////////////////////////////////////////////
// Spectrum (aacdec_proc_template.c, aacdec_float.c)

static inline int lcg_random(unsigned previous_val)
{
	union { unsigned u; int s; } v = { previous_val * 1664525u + 1013904223 };
	return v.s;
}

union intfloat32 { uint32_t i; float f; };

static inline float* VMUL2(float* dst, const float* v, unsigned idx, const float* scale)
{
	const float s = *scale;
	*dst++ = v[idx & 15] * s;
	*dst++ = v[idx >> 4 & 15] * s;
	return dst;
}

static inline float* VMUL4(float* dst, const float* v, unsigned idx, const float* scale)
{
	const float s = *scale;
	*dst++ = v[idx & 3] * s;
	*dst++ = v[idx >> 2 & 3] * s;
	*dst++ = v[idx >> 4 & 3] * s;
	*dst++ = v[idx >> 6 & 3] * s;
	return dst;
}

static inline float* VMUL2S(float* dst, const float* v, unsigned idx, unsigned sign, const float* scale)
{
	intfloat32 s0, s1;

	s0.f = s1.f = *scale;
	s0.i ^= sign >> 1 << 31;
	s1.i ^= sign << 31;

	*dst++ = v[idx & 15] * s0.f;
	*dst++ = v[idx >> 4 & 15] * s1.f;
	return dst;
}

static inline float* VMUL4S(float* dst, const float* v, unsigned idx, unsigned sign, const float* scale)
{
	unsigned nz = idx >> 12;
	intfloat32 s, t;
	s.f = *scale;

	t.i = s.i ^ (sign & 1U << 31);
	*dst++ = v[idx & 3] * t.f;

	sign <<= nz & 1;
	nz >>= 1;
	t.i = s.i ^ (sign & 1U << 31);
	*dst++ = v[idx >> 2 & 3] * t.f;

	sign <<= nz & 1;
	nz >>= 1;
	t.i = s.i ^ (sign & 1U << 31);
	*dst++ = v[idx >> 4 & 3] * t.f;

	sign <<= nz & 1;
	t.i = s.i ^ (sign & 1U << 31);
	*dst++ = v[idx >> 6 & 3] * t.f;

	return dst;
}

static int decode_spectrum_and_dequant(AACContext* ac, GetBitContext* gb, const Pulse* pulse, SingleChannelElement* sce)
{
	int i, k, g, idx = 0;
	float* coef = sce->coeffs;
	IndividualChannelStream* ics = &sce->ics;
	const int c = 1024 / ics->num_windows;
	const uint16_t* offsets = ics->swb_offset;
	const float* sf = sce->sf;
	const uint8_t* band_type = sce->band_type;
	float* coef_base = coef;

	for (g = 0; g < ics->num_windows; g++)
		memset(coef + g * 128 + offsets[ics->max_sfb], 0, sizeof(float) * (c - offsets[ics->max_sfb]));

	for (g = 0; g < ics->num_window_groups; g++) {
		const unsigned g_len = ics->group_len[g];

		for (i = 0; i < ics->max_sfb; i++, idx++) {
			const unsigned cbt_m1 = band_type[idx] - 1;
			float* cfo = coef + offsets[i];
			const int off_len = offsets[i + 1] - offsets[i];
			int group;

			if (cbt_m1 >= INTENSITY_BT2 - 1) {
				for (group = 0; group < (int)g_len; group++, cfo += 128) memset(cfo, 0, off_len * sizeof(*cfo));
			} else if (cbt_m1 == NOISE_BT - 1) {
				for (group = 0; group < (int)g_len; group++, cfo += 128) {
					for (k = 0; k < off_len; k++) {
						ac->random_state = lcg_random(ac->random_state);
						cfo[k] = (float)ac->random_state;
					}

					float band_energy = 0;
					for (k = 0; k < off_len; k++) band_energy += cfo[k] * cfo[k];
					const float scale = sf[idx] / sqrtf(band_energy);
					for (k = 0; k < off_len; k++) cfo[k] *= scale;
				}
			} else {
				const float* vq = ff_aac_codebook_vector_vals[cbt_m1];
				const VLC* vlc_tab = &g_vlc_spectral[cbt_m1];

				switch (cbt_m1 >> 1) {
				case 0:
					for (group = 0; group < (int)g_len; group++, cfo += 128) {
						float* cf = cfo;
						int len = off_len;
						do {
							const int code = get_vlc(gb, vlc_tab);
							if (code < 0) return AVERROR_INVALIDDATA;
							cf = VMUL4(cf, vq, code, sf + idx);
						} while (len -= 4);
					}
					break;

				case 1:
					for (group = 0; group < (int)g_len; group++, cfo += 128) {
						float* cf = cfo;
						int len = off_len;
						do {
							const int code = get_vlc(gb, vlc_tab);
							if (code < 0) return AVERROR_INVALIDDATA;
							const unsigned cb_idx = code;
							const unsigned nnz = cb_idx >> 8 & 15;
							const uint32_t bits = nnz ? show_cache(gb) : 0;
							skip_bits(gb, nnz);
							cf = VMUL4S(cf, vq, cb_idx, bits, sf + idx);
						} while (len -= 4);
					}
					break;

				case 2:
					for (group = 0; group < (int)g_len; group++, cfo += 128) {
						float* cf = cfo;
						int len = off_len;
						do {
							const int code = get_vlc(gb, vlc_tab);
							if (code < 0) return AVERROR_INVALIDDATA;
							cf = VMUL2(cf, vq, code, sf + idx);
						} while (len -= 2);
					}
					break;

				case 3:
				case 4:
					for (group = 0; group < (int)g_len; group++, cfo += 128) {
						float* cf = cfo;
						int len = off_len;
						do {
							const int code = get_vlc(gb, vlc_tab);
							if (code < 0) return AVERROR_INVALIDDATA;
							const unsigned cb_idx = code;
							const unsigned nnz = cb_idx >> 8 & 15;
							const unsigned sign = nnz ? show_bits(gb, nnz) << (cb_idx >> 12) : 0;
							skip_bits(gb, nnz);
							cf = VMUL2S(cf, vq, cb_idx, sign, sf + idx);
						} while (len -= 2);
					}
					break;

				default:
					for (group = 0; group < (int)g_len; group++, cfo += 128) {
						uint32_t* icf = (uint32_t*)cfo;
						int len = off_len;
						do {
							const int code = get_vlc(gb, vlc_tab);
							if (code < 0) return AVERROR_INVALIDDATA;
							unsigned cb_idx = code;

							if (cb_idx == 0x0000) {
								*icf++ = 0;
								*icf++ = 0;
								continue;
							}

							const unsigned nnz = cb_idx >> 12;
							const unsigned nzt = cb_idx >> 8;
							uint32_t bits = nnz ? show_bits(gb, nnz) << (32 - nnz) : 0;
							skip_bits(gb, nnz);

							for (int j = 0; j < 2; j++) {
								if (nzt & 1 << j) {
									// The total length of escape_sequence must be < 22 bits according to the
									// specification (i.e. max is 111111110xxxxxxxxxxxx).
									const uint32_t cache = ~show_cache(gb);
									uint32_t b = cache ? __builtin_clz(cache) : 32;

									if (b > 8) return AVERROR_INVALIDDATA;

									skip_bits(gb, b + 1);
									b += 4;
									const int n = (1 << b) + show_bits(gb, b);
									skip_bits(gb, b);
									intfloat32 v;
									v.f = g_cbrt_tab[n];
									*icf++ = v.i | (bits & 1U << 31);
									bits <<= 1;
								} else {
									intfloat32 v;
									v.f = vq[cb_idx & 15];
									*icf++ = (bits & 1U << 31) | v.i;
									bits <<= !!v.i;
								}
								cb_idx >>= 4;
							}
						} while (len -= 2);
						for (k = 0; k < off_len; k++) cfo[k] *= sf[idx];
					}
				}
			}
		}
		coef += g_len << 7;
	}

	if (pulse) {
		idx = 0;
		for (i = 0; i < pulse->num_pulse; i++) {
			float co = coef_base[pulse->pos[i]];
			while (offsets[idx + 1] <= pulse->pos[i]) idx++;
			if (band_type[idx] != NOISE_BT && sf[idx]) {
				float ico = (float)-pulse->amp[i];
				if (co) {
					co /= sf[idx];
					ico = co / sqrtf(sqrtf(fabsf(co))) + (co > 0 ? -ico : ico);
				}
				coef_base[pulse->pos[i]] = cbrtf(fabsf(ico)) * ico * sf[idx];
			}
		}
	}
	return 0;
}

static int decode_ics(AACContext* ac, SingleChannelElement* sce, GetBitContext* gb, int common_window)
{
	Pulse pulse;
	TemporalNoiseShaping* tns = &sce->tns;
	IndividualChannelStream* ics = &sce->ics;
	int ret, pulse_present = 0;

	pulse.num_pulse = 0;

	const int global_gain = get_bits(gb, 8);

	if (!common_window) {
		ret = decode_ics_info(ac, ics, gb);
		if (ret < 0) goto fail;
	}

	if ((ret = decode_band_types(sce, gb)) < 0) goto fail;
	if ((ret = decode_scalefactors(ac, sce, gb, global_gain)) < 0) goto fail;

	dequant_scalefactors(sce);

	if ((pulse_present = get_bits1(gb))) {
		if (ics->window_sequence[0] == EIGHT_SHORT_SEQUENCE) {
			ret = AVERROR_INVALIDDATA;
			goto fail;
		}
		if (decode_pulses(&pulse, gb, ics->swb_offset, ics->num_swb)) {
			ret = AVERROR_INVALIDDATA;
			goto fail;
		}
		ac->stats[AAC_INFO_PULSE]++;
	}
	tns->present = get_bits1(gb);
	if (tns->present) {
		ret = decode_tns(tns, gb, ics);
		if (ret < 0) goto fail;
		ac->stats[AAC_INFO_TNS]++;
	}
	if (get_bits1(gb)) decode_gain_control(sce, gb);

	ret = decode_spectrum_and_dequant(ac, gb, pulse_present ? &pulse : NULL, sce);
	if (ret < 0) goto fail;
	return 0;

fail:
	memset(sce->sfo, 0, sizeof(sce->sfo));
	tns->present = 0;
	return ret;
}

static void apply_mid_side_stereo(ChannelElement* cpe)
{
	const IndividualChannelStream* ics = &cpe->ch[0].ics;
	float* ch0 = cpe->ch[0].coeffs;
	float* ch1 = cpe->ch[1].coeffs;
	const uint16_t* offsets = ics->swb_offset;
	for (int g = 0; g < ics->num_window_groups; g++) {
		for (int sfb = 0; sfb < cpe->max_sfb_ste; sfb++) {
			const int idx = g * cpe->max_sfb_ste + sfb;
			if (cpe->ms_mask[idx] && cpe->ch[0].band_type[idx] < NOISE_BT && cpe->ch[1].band_type[idx] < NOISE_BT) {
				for (int group = 0; group < ics->group_len[g]; group++) {
					float* v1 = ch0 + group * 128 + offsets[sfb];
					float* v2 = ch1 + group * 128 + offsets[sfb];
					const int len = offsets[sfb + 1] - offsets[sfb];
					for (int i = 0; i < len; i++) {
						const float t = v1[i] - v2[i];
						v1[i] += v2[i];
						v2[i] = t;
					}
				}
			}
		}
		ch0 += ics->group_len[g] * 128;
		ch1 += ics->group_len[g] * 128;
	}
}

static void apply_intensity_stereo(ChannelElement* cpe, int ms_present)
{
	const IndividualChannelStream* ics = &cpe->ch[1].ics;
	SingleChannelElement* sce1 = &cpe->ch[1];
	float* coef0 = cpe->ch[0].coeffs;
	float* coef1 = cpe->ch[1].coeffs;
	const uint16_t* offsets = ics->swb_offset;
	for (int g = 0; g < ics->num_window_groups; g++) {
		for (int sfb = 0; sfb < ics->max_sfb; sfb++) {
			const int idx = g * ics->max_sfb + sfb;
			if (sce1->band_type[idx] == INTENSITY_BT || sce1->band_type[idx] == INTENSITY_BT2) {
				int c = -1 + 2 * (sce1->band_type[idx] - 14);
				if (ms_present) c *= 1 - 2 * cpe->ms_mask[idx];
				const float scale = c * sce1->sf[idx];
				for (int group = 0; group < ics->group_len[g]; group++) {
					const float* src = coef0 + group * 128 + offsets[sfb];
					float* dst = coef1 + group * 128 + offsets[sfb];
					const int len = offsets[sfb + 1] - offsets[sfb];
					for (int i = 0; i < len; i++) dst[i] = src[i] * scale;
				}
			}
		}
		coef0 += ics->group_len[g] * 128;
		coef1 += ics->group_len[g] * 128;
	}
}

static int decode_cpe(AACContext* ac, GetBitContext* gb, ChannelElement* cpe)
{
	int ret, ms_present = 0;

	const int common_window = get_bits1(gb);
	if (common_window) {
		if (decode_ics_info(ac, &cpe->ch[0].ics, gb)) return AVERROR_INVALIDDATA;
		const int i = cpe->ch[1].ics.use_kb_window[0];
		cpe->ch[1].ics = cpe->ch[0].ics;
		cpe->ch[1].ics.use_kb_window[1] = (uint8_t)i;
		ms_present = get_bits(gb, 2);
		if (ms_present == 3) return AVERROR_INVALIDDATA;
		if (ms_present) {
			const int max_idx = cpe->ch[0].ics.num_window_groups * cpe->ch[0].ics.max_sfb;
			cpe->max_sfb_ste = cpe->ch[0].ics.max_sfb;
			if (ms_present == 1) {
				for (int idx = 0; idx < max_idx; idx++) cpe->ms_mask[idx] = (uint8_t)get_bits1(gb);
			} else {
				memset(cpe->ms_mask, 1, max_idx * sizeof(cpe->ms_mask[0]));
			}
			ac->stats[AAC_INFO_MS]++;
		}
	}
	if ((ret = decode_ics(ac, &cpe->ch[0], gb, common_window))) return ret;
	if ((ret = decode_ics(ac, &cpe->ch[1], gb, common_window))) return ret;

	if (common_window && ms_present) apply_mid_side_stereo(cpe);

	apply_intensity_stereo(cpe, ms_present);
	return 0;
}

static const float cce_scale[] = {
	1.09050773266525765921,	// 2^(1/8)
	1.18920711500272106672,	// 2^(1/4)
	(float)M_SQRT2,
	2,
};

static int decode_cce(AACContext* ac, GetBitContext* gb, ChannelElement* che)
{
	int num_gain = 0;
	int c, g, sfb, ret;
	SingleChannelElement* sce = &che->ch[0];
	ChannelCoupling* coup = &che->coup;

	int coupling_point = 2 * get_bits1(gb);
	coup->num_coupled = get_bits(gb, 3);
	for (c = 0; c <= coup->num_coupled; c++) {
		num_gain++;
		coup->type[c] = get_bits1(gb) ? TYPE_CPE : TYPE_SCE;
		coup->id_select[c] = get_bits(gb, 4);
		if (coup->type[c] == TYPE_CPE) {
			coup->ch_select[c] = get_bits(gb, 2);
			if (coup->ch_select[c] == 3) num_gain++;
		} else {
			coup->ch_select[c] = 2;
		}
	}
	coupling_point += get_bits1(gb) || (coupling_point >> 1);
	coup->coupling_point = (CouplingPoint)coupling_point;

	const int sign = get_bits(gb, 1);
	const float scale = cce_scale[get_bits(gb, 2)];

	if ((ret = decode_ics(ac, sce, gb, 0))) return ret;

	for (c = 0; c < num_gain; c++) {
		int idx = 0;
		int cge = 1;
		int gain = 0;
		float gain_cache = 1.f;
		if (c) {
			cge = coup->coupling_point == AFTER_IMDCT ? 1 : get_bits1(gb);
			gain = cge ? get_vlc(gb, &g_vlc_scalefactors) - 60 : 0;
			gain_cache = powf(scale, (float)-gain);
		}
		if (coup->coupling_point == AFTER_IMDCT) {
			coup->gain[c][0] = gain_cache;
		} else {
			for (g = 0; g < sce->ics.num_window_groups; g++) {
				for (sfb = 0; sfb < sce->ics.max_sfb; sfb++, idx++) {
					if (sce->band_type[idx] != ZERO_BT) {
						if (!cge) {
							int t = get_vlc(gb, &g_vlc_scalefactors) - 60;
							if (t) {
								int s = 1;
								t = gain += t;
								if (sign) {
									s -= 2 * (t & 0x1);
									t >>= 1;
								}
								gain_cache = powf(scale, (float)-t) * s;
							}
						}
						coup->gain[c][idx] = gain_cache;
					}
				}
			}
		}
	}
	ac->stats[AAC_INFO_CCE]++;
	return 0;
}

static int decode_dynamic_range(GetBitContext* gb)
{
	int n = 1;
	int drc_num_bands = 1;

	if (get_bits1(gb)) {	// pce_tag_present
		skip_bits(gb, 8);
		n++;
	}
	if (get_bits1(gb)) {	// excluded_chns_present
		int num_excl_chan = 0;
		do {
			skip_bits(gb, 7);
			num_excl_chan += 7;
		} while (num_excl_chan < MAX_CHANNELS - 7 && get_bits1(gb));
		n += num_excl_chan / 7;
	}
	if (get_bits1(gb)) {	// drc_bands_present
		const int band_incr = get_bits(gb, 4);
		skip_bits(gb, 4);	// interpolation_scheme
		n++;
		drc_num_bands += band_incr;
		for (int i = 0; i < drc_num_bands; i++) {
			skip_bits(gb, 8);
			n++;
		}
	}
	if (get_bits1(gb)) {	// prog_ref_level_present
		skip_bits(gb, 8);
		n++;
	}
	for (int i = 0; i < drc_num_bands; i++) {
		skip_bits(gb, 8);
		n++;
	}
	return n;
}

/** Returns bytes consumed. */
static int decode_extension_payload(AACContext* ac, GetBitContext* gb, int cnt, ChannelElement* che, int elem_type)
{
	int crc_flag = 0;
	int res = cnt;
	const int type = get_bits(gb, 4);

	switch (type) {
	case EXT_SBR_DATA_CRC:
		crc_flag++;
		// fall through
	case EXT_SBR_DATA:
		if (!che) {
			return res;
		} else if (!ac->oc[1].m4ac.sbr) {
			skip_bits_long(gb, 8 * cnt - 4);
			return res;
		} else if (ac->oc[1].m4ac.sbr == -1 && ac->oc[1].status == OC_LOCKED) {
			skip_bits_long(gb, 8 * cnt - 4);
			return res;
		} else if (ac->oc[1].m4ac.ps == -1 && ac->oc[1].status < OC_LOCKED && ac->channels == 1) {
			ac->oc[1].m4ac.sbr = 1;
			ac->oc[1].m4ac.ps = 1;
		} else {
			ac->oc[1].m4ac.sbr = 1;
		}
		sbr_decode_extension(ac, che, gb, crc_flag, cnt, elem_type, 0);
		break;
	case EXT_DYNAMIC_RANGE:
		res = decode_dynamic_range(gb);
		break;
	default:
		skip_bits_long(gb, 8 * cnt - 4);
		break;
	}
	return res;
}

///////////////////////////////////////////////////////////////////////////////
// Filterbank and tools after the parse (aacdec_dsp_template.c, aacdec_float_coupling.h, float_dsp.c)

static void vector_fmul_window(float* dst, const float* src0, const float* src1, const float* win, int len)
{
	dst += len;
	win += len;
	src0 += len;

	for (int i = -len, j = len - 1; i < 0; i++, j--) {
		const float s0 = src0[i];
		const float s1 = src1[j];
		const float wi = win[i];
		const float wj = win[j];
		dst[i] = s0 * wj - s1 * wi;
		dst[j] = s0 * wi + s1 * wj;
	}
}

static void apply_tns(float* coef, TemporalNoiseShaping* tns, IndividualChannelStream* ics)
{
	const int mmm = FFMIN(ics->tns_max_bands, ics->max_sfb);
	float lpc[TNS_MAX_ORDER];

	if (!mmm) return;

	for (int w = 0; w < ics->num_windows; w++) {
		int bottom = ics->num_swb;
		for (int filt = 0; filt < tns->n_filt[w]; filt++) {
			const int top = bottom;
			bottom = FFMAX(0, top - tns->length[w][filt]);
			const int order = tns->order[w][filt];
			if (order == 0) continue;

			// tns_decode_coef: compute_lpc_coefs without normalization
			const float* autoc = tns->coef[w][filt];
			for (int i = 0; i < order; i++) {
				const float r = -autoc[i];
				lpc[i] = r;
				for (int j = 0; j < (i + 1) >> 1; j++) {
					const float f = lpc[j];
					const float b = lpc[i - 1 - j];
					lpc[j] = f + r * b;
					lpc[i - 1 - j] = b + r * f;
				}
			}

			int start = ics->swb_offset[FFMIN(bottom, mmm)];
			const int end = ics->swb_offset[FFMIN(top, mmm)];
			const int size = end - start;
			if (size <= 0) continue;
			int inc;
			if (tns->direction[w][filt]) {
				inc = -1;
				start = end - 1;
			} else {
				inc = 1;
			}
			start += w * 128;

			for (int m = 0; m < size; m++, start += inc)
				for (int i = 1; i <= FFMIN(m, order); i++) coef[start] -= coef[start - i * inc] * lpc[i - 1];
		}
	}
}

static void imdct_and_windowing(AACContext* ac, SingleChannelElement* sce)
{
	IndividualChannelStream* ics = &sce->ics;
	float* in = sce->coeffs;
	float* out = sce->ret_buf;
	float* saved = sce->saved;
	const float* swindow = ics->use_kb_window[0] ? g_kbd_short_128 : g_sine_128;
	const float* lwindow_prev = ics->use_kb_window[1] ? g_kbd_long_1024 : g_sine_1024;
	const float* swindow_prev = ics->use_kb_window[1] ? g_kbd_short_128 : g_sine_128;
	float* buf = ac->buf_mdct;
	float* temp = ac->temp;

	if (ics->window_sequence[0] == EIGHT_SHORT_SEQUENCE) {
		for (int i = 0; i < 1024; i += 128) imdct_half(buf + i, in + i, 128);
	} else {
		imdct_half(buf, in, 1024);
	}

	// All 'meaningless' short to long and long to short transitions are considered to be short to
	// short transitions. This leaves just two cases (long to long and short to short) with a little
	// special sauce for EIGHT_SHORT_SEQUENCE.
	if ((ics->window_sequence[1] == ONLY_LONG_SEQUENCE || ics->window_sequence[1] == LONG_STOP_SEQUENCE) &&
			(ics->window_sequence[0] == ONLY_LONG_SEQUENCE || ics->window_sequence[0] == LONG_START_SEQUENCE)) {
		vector_fmul_window(out, saved, buf, lwindow_prev, 512);
	} else {
		memcpy(out, saved, 448 * sizeof(*out));

		if (ics->window_sequence[0] == EIGHT_SHORT_SEQUENCE) {
			vector_fmul_window(out + 448 + 0 * 128, saved + 448, buf + 0 * 128, swindow_prev, 64);
			vector_fmul_window(out + 448 + 1 * 128, buf + 0 * 128 + 64, buf + 1 * 128, swindow, 64);
			vector_fmul_window(out + 448 + 2 * 128, buf + 1 * 128 + 64, buf + 2 * 128, swindow, 64);
			vector_fmul_window(out + 448 + 3 * 128, buf + 2 * 128 + 64, buf + 3 * 128, swindow, 64);
			vector_fmul_window(temp, buf + 3 * 128 + 64, buf + 4 * 128, swindow, 64);
			memcpy(out + 448 + 4 * 128, temp, 64 * sizeof(*out));
		} else {
			vector_fmul_window(out + 448, saved + 448, buf, swindow_prev, 64);
			memcpy(out + 576, buf + 64, 448 * sizeof(*out));
		}
	}

	// buffer update
	if (ics->window_sequence[0] == EIGHT_SHORT_SEQUENCE) {
		memcpy(saved, temp + 64, 64 * sizeof(*saved));
		vector_fmul_window(saved + 64, buf + 4 * 128 + 64, buf + 5 * 128, swindow, 64);
		vector_fmul_window(saved + 192, buf + 5 * 128 + 64, buf + 6 * 128, swindow, 64);
		vector_fmul_window(saved + 320, buf + 6 * 128 + 64, buf + 7 * 128, swindow, 64);
		memcpy(saved + 448, buf + 7 * 128 + 64, 64 * sizeof(*saved));
	} else if (ics->window_sequence[0] == LONG_START_SEQUENCE) {
		memcpy(saved, buf + 512, 448 * sizeof(*saved));
		memcpy(saved + 448, buf + 7 * 128 + 64, 64 * sizeof(*saved));
	} else {	// LONG_STOP or ONLY_LONG
		memcpy(saved, buf + 512, 512 * sizeof(*saved));
	}
}

static void apply_dependent_coupling(SingleChannelElement* target, ChannelElement* cce, int index)
{
	IndividualChannelStream* ics = &cce->ch[0].ics;
	const uint16_t* offsets = ics->swb_offset;
	float* dest = target->coeffs;
	const float* src = cce->ch[0].coeffs;
	int idx = 0;
	for (int g = 0; g < ics->num_window_groups; g++) {
		for (int i = 0; i < ics->max_sfb; i++, idx++) {
			if (cce->ch[0].band_type[idx] != ZERO_BT) {
				const float gain = cce->coup.gain[index][idx];
				for (int group = 0; group < ics->group_len[g]; group++)
					for (int k = offsets[i]; k < offsets[i + 1]; k++) dest[group * 128 + k] += gain * src[group * 128 + k];
			}
		}
		dest += ics->group_len[g] * 128;
		src += ics->group_len[g] * 128;
	}
}

static void apply_independent_coupling(SingleChannelElement* target, ChannelElement* cce, int index)
{
	const float gain = cce->coup.gain[index][0];
	const float* src = cce->ch[0].ret_buf;
	float* dest = target->ret_buf;
	for (int i = 0; i < 1024; i++) dest[i] += src[i] * gain;
}

static void apply_channel_coupling(AACContext* ac, ChannelElement* cc, int type, int elem_id, CouplingPoint coupling_point,
	void (*apply_coupling_method)(SingleChannelElement* target, ChannelElement* cce, int index))
{
	for (int i = 0; i < MAX_ELEM_ID; i++) {
		ChannelElement* cce = ac->che[TYPE_CCE][i];
		int index = 0;

		if (cce && cce->coup.coupling_point == coupling_point) {
			ChannelCoupling* coup = &cce->coup;

			for (int c = 0; c <= coup->num_coupled; c++) {
				if (coup->type[c] == type && coup->id_select[c] == elem_id) {
					if (coup->ch_select[c] != 1) {
						apply_coupling_method(&cc->ch[0], cce, index);
						if (coup->ch_select[c] != 0) index++;
					}
					if (coup->ch_select[c] != 2) apply_coupling_method(&cc->ch[1], cce, index++);
				} else {
					index += 1 + (coup->ch_select[c] == 3);
				}
			}
		}
	}
}

static void spectral_to_sample(AACContext* ac)
{
	for (int type = 3; type >= 0; type--) {
		for (int i = 0; i < MAX_ELEM_ID; i++) {
			ChannelElement* che = ac->che[type][i];
			if (che && che->present) {
				if (type <= TYPE_CPE) apply_channel_coupling(ac, che, type, i, BEFORE_TNS, apply_dependent_coupling);
				if (che->ch[0].tns.present) apply_tns(che->ch[0].coeffs, &che->ch[0].tns, &che->ch[0].ics);
				if (che->ch[1].tns.present) apply_tns(che->ch[1].coeffs, &che->ch[1].tns, &che->ch[1].ics);
				if (type <= TYPE_CPE) apply_channel_coupling(ac, che, type, i, BETWEEN_TNS_AND_IMDCT, apply_dependent_coupling);
				if (type != TYPE_CCE || che->coup.coupling_point == AFTER_IMDCT) {
					imdct_and_windowing(ac, &che->ch[0]);
					if (type == TYPE_CPE) imdct_and_windowing(ac, &che->ch[1]);
					if (ac->oc[1].m4ac.sbr > 0) sbr_apply(ac, che, type, 0, che->ch[0].ret_buf, che->ch[1].ret_buf);
				}
				if (type <= TYPE_CCE) apply_channel_coupling(ac, che, type, i, AFTER_IMDCT, apply_independent_coupling);
				che->present = 0;
			}
		}
	}
}

static int parse_adts_frame_header(AACContext* ac, GetBitContext* gb)
{
	uint8_t layout_map[MAX_ELEM_ID * 4][3];
	int layout_map_tags, ret;

	// ff_adts_header_parse
	skip_bits(gb, 12);	// syncword
	skip_bits(gb, 1);	// id
	skip_bits(gb, 2);	// layer
	const int crc_absent = get_bits1(gb);
	const int aot = get_bits(gb, 2);
	const int sr = get_bits(gb, 4);
	if (!ff_mpeg4audio_sample_rates[sr]) return AVERROR_INVALIDDATA;
	skip_bits(gb, 1);	// private_bit
	const int chan_config = get_bits(gb, 3);
	skip_bits(gb, 1);	// original/copy
	skip_bits(gb, 1);	// home
	skip_bits(gb, 1);	// copyright_identification_bit
	skip_bits(gb, 1);	// copyright_identification_start
	const int size = get_bits(gb, 13);
	if (size < 7) return AVERROR_INVALIDDATA;
	skip_bits(gb, 11);	// adts_buffer_fullness
	skip_bits(gb, 2);	// number_of_raw_data_blocks_in_frame
	if (aot + 1 != AOT_AAC_LC) return AVERROR_UNSUPPORTED;

	push_output_configuration(ac);
	if (chan_config) {
		ac->oc[1].m4ac.chan_config = chan_config;
		if ((ret = set_default_channel_config(layout_map, &layout_map_tags, chan_config)) < 0) return ret;
		if ((ret = output_configure(ac, layout_map, layout_map_tags, FFMAX(ac->oc[1].status, OC_TRIAL_FRAME))) < 0) return ret;
	} else {
		ac->oc[1].m4ac.chan_config = 0;
	}
	ac->oc[1].m4ac.sample_rate = ff_mpeg4audio_sample_rates[sr];
	ac->oc[1].m4ac.sampling_index = sr;
	ac->oc[1].m4ac.object_type = aot + 1;
	if (ac->oc[0].status != OC_LOCKED || ac->oc[0].m4ac.chan_config != chan_config ||
			ac->oc[0].m4ac.sample_rate != ff_mpeg4audio_sample_rates[sr]) {
		ac->oc[1].m4ac.sbr = -1;
		ac->oc[1].m4ac.ps = -1;
	}
	if (!crc_absent) skip_bits(gb, 16);
	return size;
}

/** Returns 0, or an error; *samples is what the frame gave. */
static int output_rate_shift(const AACContext* ac)
{
	const MPEG4AudioConfig* m4ac = &ac->oc[1].m4ac;
	return m4ac->sbr == 1 && m4ac->ext_sample_rate > m4ac->sample_rate;
}

static int decode_frame_ga(AACContext* ac, GetBitContext* gb, int* samples)
{
	int err;
	int elem_id;
	int elem_type;
	uint8_t che_presence[4][MAX_ELEM_ID];
	ChannelElement* che = NULL;
	ChannelElement* che_prev = NULL;
	int che_prev_type = TYPE_END;
	int audio_found = 0, pce_found = 0;

	memset(che_presence, 0, sizeof(che_presence));
	*samples = 0;

	const int payload_alignment = get_bits_count(gb);
	while ((elem_type = get_bits(gb, 3)) != TYPE_END) {
		elem_id = get_bits(gb, 4);

		if (!ac->channels && elem_type != TYPE_PCE) return AVERROR_INVALIDDATA;

		if (elem_type < TYPE_DSE) {
			if (che_presence[elem_type][elem_id] > 1) return AVERROR_INVALIDDATA;
			che_presence[elem_type][elem_id]++;

			if (!(che = get_che(ac, elem_type, elem_id))) return AVERROR_INVALIDDATA;
			*samples = 1024;
			che->present = 1;
		}

		switch (elem_type) {
		case TYPE_SCE:
			err = decode_ics(ac, &che->ch[0], gb, 0);
			audio_found = 1;
			break;
		case TYPE_CPE:
			err = decode_cpe(ac, gb, che);
			audio_found = 1;
			break;
		case TYPE_CCE:
			err = decode_cce(ac, gb, che);
			break;
		case TYPE_LFE:
			err = decode_ics(ac, &che->ch[0], gb, 0);
			audio_found = 1;
			break;
		case TYPE_DSE:
			err = skip_data_stream_element(gb);
			break;
		case TYPE_PCE: {
			uint8_t layout_map[MAX_ELEM_ID * 4][3];
			memset(layout_map, 0, sizeof(layout_map));

			const int pushed = push_output_configuration(ac);
			if (pce_found && !pushed) return AVERROR_INVALIDDATA;

			const int tags = decode_pce(layout_map, gb, payload_alignment);
			if (tags < 0) {
				err = tags;
				break;
			}
			err = 0;
			if (pce_found) {
				pop_output_configuration(ac);
			} else {
				err = output_configure(ac, layout_map, tags, OC_TRIAL_PCE);
				if (!err) ac->oc[1].m4ac.chan_config = 0;
				pce_found = 1;
			}
			break;
		}
		case TYPE_FIL:
			if (elem_id == 15) elem_id += get_bits(gb, 8) - 1;
			if (get_bits_left(gb) < 8 * elem_id) return AVERROR_INVALIDDATA;
			err = 0;
			while (elem_id > 0) {
				const int ret = decode_extension_payload(ac, gb, elem_id, che_prev, che_prev_type);
				if (ret < 0) {
					err = ret;
					break;
				}
				elem_id -= ret;
			}
			break;
		default:
			err = AVERROR_INVALIDDATA;
			break;
		}

		if (elem_type < TYPE_DSE) {
			che_prev = che;
			che_prev_type = elem_type;
		}

		if (err) return err;

		if (get_bits_left(gb) < 3) return AVERROR_INVALIDDATA;
	}

	if (!ac->channels) return 0;

	if (*samples && ac->oc[1].m4ac.sample_rate <= 0) return AVERROR_INVALIDDATA;

	*samples <<= output_rate_shift(ac);

	spectral_to_sample(ac);

	if (ac->oc[1].status && audio_found) ac->oc[1].status = OC_LOCKED;
	return 0;
}

///////////////////////////////////////////////////////////////////////////////
// Boundary

extern "C" {

AACContext* aac_open(void)
{
	if (!tables_init()) return NULL;
	AACContext* ac = (AACContext*)calloc(1, sizeof(AACContext));
	if (!ac) return NULL;
	ac->oc[1].m4ac.sbr = -1;
	ac->oc[1].m4ac.ps = -1;
	ac->random_state = 0x1f2e3d4c;
	return ac;
}

void aac_close(AACContext* ac)
{
	if (!ac) return;
	for (int type = 0; type < 4; type++)
		for (int i = 0; i < MAX_ELEM_ID; i++) free(ac->che[type][i]);
	free(ac);
}

void aac_reset(AACContext* ac, int frame)
{
	ac->seek_frame = frame;
	for (int type = 3; type >= 0; type--) {
		for (int i = 0; i < MAX_ELEM_ID; i++) {
			ChannelElement* che = ac->che[type][i];
			if (che) {
				for (int j = 0; j <= 1; j++) {
					memset(che->ch[j].saved, 0, sizeof(che->ch[j].saved));
					memset(&che->ch[j].ics, 0, sizeof(che->ch[j].ics));
				}
				sbr_ctx_init(get_sbr(che), type, frame);
			}
		}
	}
	ac->random_state = 0x1f2e3d4c;
}

int aac_configure(AACContext* ac, uint8_t* asc, int size, int* info)
{
	memset(asc + size, 0, AAC_INPUT_PADDING);
	const int err = decode_audio_specific_config(ac, asc, size * 8);
	info[0] = ac->oc[1].m4ac.sample_rate << output_rate_shift(ac);
	info[1] = ac->channels;
	info[2] = ac->oc[1].m4ac.object_type;
	info[3] = ac->oc[1].m4ac.sbr == 1;
	return err >= 0;
}

int aac_decode_frame(AACContext* ac, uint8_t* data, int size, int* info)
{
	GetBitContext gb;
	int err, samples = 0;

	memset(data + size, 0, AAC_INPUT_PADDING);
	memset(ac->stats, 0, sizeof(ac->stats));
	for (int c = 0; c < ac->channels; c++)
		if (ac->output_element[c]) memset(ac->output_element[c]->ret_buf, 0, sizeof(ac->output_element[c]->ret_buf));

	init_get_bits(&gb, data, size * 8);
	ac->tags_mapped = 0;

	if (show_bits(&gb, 12) == 0xfff) {
		err = parse_adts_frame_header(ac, &gb);
		if (err < 0) goto fail;
		if (ac->oc[1].m4ac.sampling_index > 12) {
			err = AVERROR_INVALIDDATA;
			goto fail;
		}
	}

	if ((err = decode_frame_ga(ac, &gb, &samples)) < 0) goto fail;

	memcpy(info, ac->stats, sizeof(ac->stats));
	info[AAC_INFO_SAMPLE_RATE] = ac->oc[1].m4ac.sample_rate << output_rate_shift(ac);
	info[AAC_INFO_CHANNELS] = ac->channels;
	info[AAC_INFO_SAMPLES] = samples;
	info[AAC_INFO_PS_PRESENT] = ac->ps_present;
	return 1;

fail:
	pop_output_configuration(ac);
	return 0;
}

float* aac_channel_ptr(AACContext* ac, int ch)
{
	return ac->output_element[ch] ? ac->output_element[ch]->ret_buf : ac->silence;
}

/**
 * The largest error of the half IMDCT of length n (1024 or 128) against the definition it computes,
 * x[p] = -scale * sum over k of X[k] cos(pi/n (p + n/2 + 1/2 + n/2)(k + 1/2)), scaled by 1e9. The
 * sign is lavu's, and why the scalefactors are negative.
 */
int aac_test_imdct(int n)
{
	static float in[1024], out[1024];
	if (!tables_init()) return -1;

	uint32_t seed = 12345;
	for (int k = 0; k < n; k++) {
		seed = seed * 1664525u + 1013904223u;
		in[k] = (float)((int32_t)seed >> 12);
	}
	imdct_half(out, in, n);

	const double scale = 1.0 / n / 32768.0;
	double worst = 0;
	for (int p = 0; p < n; p++) {
		double sum = 0;
		for (int k = 0; k < n; k++) sum += in[k] * cos(M_PI / n * (p + n / 2 + 0.5 + n / 2) * (k + 0.5));
		const double d = fabs(sum * scale + out[p]);
		if (d > worst) worst = d;
	}
	return (int)(worst * 1e9);
}

}	// extern "C"
