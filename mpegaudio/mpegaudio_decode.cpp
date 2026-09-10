/*
	MPEG-1/2/2.5 audio decoder, Layers I, II and III (MP1 / MP2 / MP3) — WASM C++.

	A direct port of FFmpeg's MPEG audio decoder (libavcodec, release/8.1, LGPL):
	  mpegaudiodec_template.c / mpegaudiodec_float.c / mpegaudiodec_common.c /
	  mpegaudiodecheader.c / mpegaudiodata.c / mpegaudiodsp*.c / dct32_template.c
	The algorithm (header parse, Layer I/II bit allocation and scalefactors, Layer III
	side info, scalefactors, Huffman residue, the bit reservoir, stereo processing,
	alias reduction, the hybrid IMDCT and the polyphase synthesis filterbank) is lifted
	from those files; libavcodec's framework (AVCodecContext / AVFrame / AVPacket, the
	DSP function-pointer dispatch, VLC tables, threading, AVERROR) is replaced by a
	self-contained handle (MPACtx) that owns its own planar float output.

	Why this matters more than any single container: one decoder covers AVI
	(wFormatTag 0x0050/0x0055), FLV (soundFormat 2 and 14), SWF (DefineSound and
	SoundStreamHead codec 2), MPEG-1/2 program and transport streams (Layer II is the
	standard there, and on DVD), MP4/MOV '.mp3' sample entries, and bare .mp1/.mp2/.mp3
	files.

	Scope:
	  - MPEG-1, MPEG-2 LSF and MPEG-2.5, Layers I, II and III; mono, stereo, dual
	    mono and joint stereo (both MS and intensity).
	  - Float output only (FFmpeg builds this source twice, fixed and float; the
	    fixed-point variant is not needed here).
	  - NOT the MP3ADU or MP3on4 wrappers, and no free-format streams (a frame with
	    bitrate_index 0 has no derivable size).
	  - CRC words are parsed and skipped but not verified, matching FFmpeg's default
	    (it only checks with AV_EF_CRCCHECK).

	Fixed-point-valued floats: Layers I and II compute sample values with FFmpeg's
	integer unscaling helpers (l1_unscale / l2_unscale_group) and store the resulting
	FRAC_BITS-scaled INTEGER into the float subband array. That is not a porting slip
	— the synthesis window carries the matching 2^-(16+FRAC_BITS) factor, so the
	product comes out correctly scaled. Layer III mixes a float table (expval_table)
	with the same integer helper (l3_unscale) for the linbits path; the two agree
	because both evaluate value^(4/3) * 2^(exponent/4 - 72) / IMDCT_SCALAR.

	Boundary (extern "C", prefixed _ on the JS side; see mpegaudio_decode_core.js):
	  MPACtx*  mpa_open(void)
	  void     mpa_close(MPACtx*)
	  void     mpa_reset(MPACtx*)                    drop the bit reservoir + filter
	                                                 history (seek)
	  int      mpa_probe_frame(const uint8_t*, int, int info[6])
	             parse a 4-byte header; info = {frameSize, sampleRate, channels,
	             bitRate, layer, samplesPerFrame}; returns 1/0
	  int      mpa_decode_frame(MPACtx*, const uint8_t*, int, int info[7])
	             decode ONE frame; info as above plus shortBlocks; returns 1/0
	  float*   mpa_channel_ptr(MPACtx*, int ch)      planar output
*/

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "mpegaudio_bits.h"
#include "mpegaudio_dsp.h"		// also pulls in mpegaudio_tables.h + dct32_float.h

// ─── limits and modes (mpegaudio.h / mpegaudiodec_template.c) ────────────────
#define MPA_MAX_CHANNELS 2
#define MPA_FRAME_SIZE   1152		/* max PCM samples per channel per frame */
#define FRAC_BITS        23
#define FRAC_ONE         (1 << FRAC_BITS)

#define MPA_STEREO   0
#define MPA_JSTEREO  1
#define MPA_DUAL     2
#define MPA_MONO     3

#define MODE_EXT_MS_STEREO 2
#define MODE_EXT_I_STEREO  1

#define HEADER_SIZE   4
#define BACKSTEP_SIZE 512
#define EXTRABYTES    24
#define LAST_BUF_SIZE (2 * BACKSTEP_SIZE + EXTRABYTES)

/* Largest coded frame: MPA_MAX_CODED_FRAME_SIZE, plus slack for the reader. */
#define MPA_MAX_FRAME_BYTES 1792
#define PADDING 8			/* so peek32 never needs a bounds branch in practice */

static inline int FFMIN(int a, int b) { return a < b ? a : b; }
static inline int FFMAX(int a, int b) { return a > b ? a : b; }

///////////////////////////////////////////////////////////////////////////////
// Runtime tables (mpegaudiodec_common.c / mpegaudio_tablegen.h / decode_init_static)

#define TABLE_4_3_SIZE ((8191 + 16) * 4)

static int8_t   ff_table_4_3_exp[TABLE_4_3_SIZE];
static uint32_t ff_table_4_3_value[TABLE_4_3_SIZE];
static float    expval_table_float[512][16];
static float    exp_table_float[512];
static uint16_t ff_scale_factor_modshift[64];
static int32_t  scale_factor_mult[15][3];
static float    is_table_lsf[2][2][16];
static uint16_t ff_band_index_long[9][23];
static int16_t  division_tab3[1 << 6];
static int16_t  division_tab5[1 << 8];
static int16_t  division_tab9[1 << 11];
static int16_t* const ff_division_tabs[4] = { division_tab3, division_tab5, 0, division_tab9 };

/* Intensity stereo table (mpegaudiodec_float.c). */
static const float is_table[2][16] = {
	{ 0.000000000000000000e+00f, 2.113248705863952637e-01f, 3.660253882408142090e-01f,
	  5.000000000000000000e-01f, 6.339746117591857910e-01f, 7.886751294136047363e-01f,
	  1.000000000000000000e+00f },
	{ 1.000000000000000000e+00f, 7.886751294136047363e-01f, 6.339746117591857910e-01f,
	  5.000000000000000000e-01f, 3.660253882408142090e-01f, 2.113248705863952637e-01f,
	  0.000000000000000000e+00f }
};

/* Alias reduction table (mpegaudiodec_float.c). */
static const float csa_table[8][4] = {
	{ 8.574929237365722656e-01f, -5.144957900047302246e-01f, 3.429971337318420410e-01f, -1.371988654136657715e+00f },
	{ 8.817420005798339844e-01f, -4.717319905757904053e-01f, 4.100100100040435791e-01f, -1.353474020957946777e+00f },
	{ 9.496286511421203613e-01f, -3.133774697780609131e-01f, 6.362511515617370605e-01f, -1.263006091117858887e+00f },
	{ 9.833145737648010254e-01f, -1.819131970405578613e-01f, 8.014013767242431641e-01f, -1.165227770805358887e+00f },
	{ 9.955177903175354004e-01f, -9.457419067621231079e-02f, 9.009436368942260742e-01f, -1.090092062950134277e+00f },
	{ 9.991605877876281738e-01f, -4.096558317542076111e-02f, 9.581949710845947266e-01f, -1.040126085281372070e+00f },
	{ 9.998992085456848145e-01f, -1.419856864959001541e-02f, 9.857006072998046875e-01f, -1.014097809791564941e+00f },
	{ 9.999931454658508301e-01f, -3.699974622577428818e-03f, 9.962931871414184570e-01f, -1.003693103790283203e+00f }
};

// Huffman: 15 "big value" tables (indexed 1..15) plus the two count1 quad tables.
static Vlc g_huff_vlc[16];
static Vlc g_huff_quad_vlc[2];
static uint32_t g_vlc_sub_pool[16384];
static int g_tables_ready = 0;

static int mpegaudio_init_static(void)
{
	if (g_tables_ready) return 0;

	/* scale factor multipliers for Layer I/II */
	for (int i = 0; i < 64; i++) {
		const int shift = i / 3, mod = i % 3;
		ff_scale_factor_modshift[i] = (uint16_t)(mod | (shift << 2));
	}
	for (int i = 0; i < 15; i++) {
		const int n = i + 2;
		const int norm = (int)(((int64_t)(1 << n) * FRAC_ONE) / ((1 << n) - 1));
		// MULLx(norm, FIXR(x), FRAC_BITS) is (float)x * norm in the float build,
		// truncated on assignment to int32 — reproduce that exactly.
		scale_factor_mult[i][0] = (int32_t)(1.0f * 2.0f * norm);
		scale_factor_mult[i][1] = (int32_t)(0.7937005259f * 2.0f * norm);
		scale_factor_mult[i][2] = (int32_t)(0.6299605249f * 2.0f * norm);
	}

	/* value^(4/3) * 2^(exponent/4), mantissa/exponent form (common_tableinit) */
	{
		static const double exp2_lut[4] = {
			1.00000000000000000000, 1.18920711500272106672,
			1.41421356237309504880, 1.68179283050742908606,
		};
		double pow43_val = 0;
		for (int i = 1; i < TABLE_4_3_SIZE; i++) {
			const double value = (double)(i / 4);	// integer division, as upstream
			if ((i & 3) == 0) pow43_val = value / IMDCT_SCALAR * cbrt(value);
			const double f = pow43_val * exp2_lut[i & 3];
			int e;
			const double fm = frexp(f, &e);
			const int64_t m = (int64_t)llrint(fm * (double)(1LL << 31));
			e += FRAC_BITS - 31 + 5 - 100;
			ff_table_4_3_value[i] = (uint32_t)m;
			ff_table_4_3_exp[i]   = (int8_t)(-e);
		}
	}

	/* the same quantity as a float table, for the common (x < 15) path */
	{
		static const double exp2_lut[4] = {
			1.00000000000000000000, 1.18920711500272106672,
			1.41421356237309504880, 1.68179283050742908606,
		};
		double pow43_lut[16];
		double exp2_base = 2.11758236813575084767080625169910490512847900390625e-22;	// 2^-72
		for (int i = 0; i < 16; ++i) pow43_lut[i] = i * cbrt((double)i);

		for (int exponent = 0; exponent < 512; exponent++) {
			if (exponent && (exponent & 3) == 0) exp2_base *= 2;
			const double exp2_val = exp2_base * exp2_lut[exponent & 3] / IMDCT_SCALAR;
			for (int value = 0; value < 16; value++)
				expval_table_float[exponent][value] = (float)(pow43_lut[value] * exp2_val);
			exp_table_float[exponent] = expval_table_float[exponent][1];
		}
	}

	/* intensity stereo table for LSF */
	for (int i = 0; i < 16; i++) {
		for (int j = 0; j < 2; j++) {
			const int e = -(j + 1) * ((i + 1) >> 1);
			const float f = (float)exp2((double)e / 4.0);
			const int k = i & 1;
			is_table_lsf[j][k ^ 1][i] = f;
			is_table_lsf[j][k    ][i] = 1.0f;
		}
	}

	/* Layer III scalefactor band start indexes */
	for (int i = 0; i < 9; i++) {
		int k = 0;
		for (int j = 0; j < 22; j++) {
			ff_band_index_long[i][j] = (uint16_t)k;
			k += ff_band_size_long[i][j] >> 1;
		}
		ff_band_index_long[i][22] = (uint16_t)k;
	}

	/* Layer II grouped-sample division tables */
	for (int i = 0; i < 4; i++) {
		if (ff_mpa_quant_bits[i] < 0) {
			for (int j = 0; j < (1 << (-ff_mpa_quant_bits[i] + 1)); j++) {
				int val = j;
				const int steps = ff_mpa_quant_steps[i];
				const int val1 = val % steps; val /= steps;
				const int val2 = val % steps;
				const int val3 = val / steps;
				ff_division_tabs[i][j] = (int16_t)(val1 + (val2 << 4) + (val3 << 8));
			}
		}
	}

	/* Layer III Huffman tables. FFmpeg packs the (x, y) pair of each symbol as
	   high<<1 | (both-nonzero)<<4 | low; huffman_decode() below reads it back with
	   the same shifts, so build the identical symbol values here. */
	{
		const uint8_t* huff_lens = mpa_hufflens;
		const uint8_t* huff_sym  = mpa_huffsymbols;
		int pool = 0;
		for (int i = 0; i < 15;) {
			const int nb = mpa_huff_sizes_minus_one[i] + 1;
			uint16_t tmp_symbols[256];
			for (int j = 0; j < nb; j++) {
				const uint8_t high = huff_sym[j] & 0xF0, low = huff_sym[j] & 0x0F;
				tmp_symbols[j] = (uint16_t)(high << 1 | ((high && low) << 4) | low);
			}
			if (vlc_build(&g_huff_vlc[++i], huff_lens, tmp_symbols, nb,
			              g_vlc_sub_pool + pool,
			              (int)(sizeof(g_vlc_sub_pool) / sizeof(g_vlc_sub_pool[0])) - pool))
				return -1;
			pool += g_huff_vlc[i].sub_size;
			huff_lens += nb;
			huff_sym  += nb;
		}
		for (int i = 0; i < 2; i++)
			if (vlc_build_codes(&g_huff_quad_vlc[i], mpa_quad_bits[i], mpa_quad_codes[i], 16, 0, 0))
				return -1;
	}

	mpa_dsp_init_tabs();
	g_tables_ready = 1;
	return 0;
}

///////////////////////////////////////////////////////////////////////////////
// Unscaling helpers (mpegaudiodec_template.c)

/* Layer I: n = mantissa bits minus 1. */
static inline int l1_unscale(int n, int mant, int scale_factor)
{
	int shift = ff_scale_factor_modshift[scale_factor];
	const int mod = shift & 3;
	shift >>= 2;
	int64_t val = (int64_t)((int)(mant + (-1U << n) + 1)) * (int64_t)scale_factor_mult[n - 1][mod];
	shift += n;
	return (int)((val + (1LL << (shift - 1))) >> shift);
}

/* Layer II grouped triples: 2^(-j/3) * FRAC_ONE * v, for the 3-, 5- and 9-step
   quantizers (FFmpeg's SCALE_GEN with FIXR_OLD(a) = (int)(a * FRAC_ONE + 0.5)). */
#define FIXR_OLD(a) ((int32_t)((a) * (double)FRAC_ONE + 0.5))
#define SCALE_GEN(v) \
	{ FIXR_OLD(1.0 * (v)), FIXR_OLD(0.7937005259 * (v)), FIXR_OLD(0.6299605249 * (v)) }

static const int32_t scale_factor_mult2[3][3] = {
	SCALE_GEN(4.0 / 3.0),	/* 3 steps */
	SCALE_GEN(4.0 / 5.0),	/* 5 steps */
	SCALE_GEN(4.0 / 9.0),	/* 9 steps */
};

static inline int l2_unscale_group(int steps, int mant, int scale_factor)
{
	int shift = ff_scale_factor_modshift[scale_factor];
	const int mod = shift & 3;
	shift >>= 2;
	int val = (mant - (steps >> 1)) * scale_factor_mult2[steps >> 2][mod];
	if (shift > 0) val = (val + (1 << (shift - 1))) >> shift;
	return val;
}

/* Layer III: value^(4/3) * 2^(exponent/4), normalized to FRAC_BITS. */
static inline int l3_unscale(int value, int exponent)
{
	int e = ff_table_4_3_exp[4 * value + (exponent & 3)];
	unsigned m = ff_table_4_3_value[4 * value + (exponent & 3)];
	e -= exponent >> 2;
	if ((unsigned)e > 31u) return 0;
	m = (m + ((1U << e) >> 1)) >> e;
	return (int)m;
}

///////////////////////////////////////////////////////////////////////////////
// Decoder context

struct GranuleDef {
	uint8_t scfsi;
	int part2_3_length;
	int big_values;
	int global_gain;
	int scalefac_compress;
	uint8_t block_type;
	uint8_t switch_point;
	int table_select[3];
	int subblock_gain[3];
	uint8_t scalefac_scale;
	uint8_t count1table_select;
	int region_size[3];		/* number of Huffman codes in each region */
	int preflag;
	int short_start, long_end;
	uint8_t scale_factors[40];
	float sb_hybrid[SBLIMIT * 18];	/* 576 samples */
};

struct MPACtx {
	/* header */
	int layer, lsf, mpeg25, sample_rate, sample_rate_index, bit_rate;
	int frame_size, nb_channels, mode, mode_ext, error_protection;
	int frames_per_frame;		/* PCM samples per channel in this frame */
	/* Count of short (block_type 2) granule-channels in the frame just decoded.
	   Reported through info[6] so a conformance run can prove it exercised the
	   12-point IMDCT path rather than only the 36-point one. */
	int short_blocks;

	/* bit reservoir (Layer III): bytes carried over from previous frames */
	uint8_t last_buf[LAST_BUF_SIZE + PADDING];
	int last_buf_size;
	int extrasize;

	GetBits gb;
	GetBits in_gb;
	int in_gb_valid;

	/* synthesis */
	float synth_buf[MPA_MAX_CHANNELS][512 * 2];
	int synth_buf_offset[MPA_MAX_CHANNELS];
	float sb_samples[MPA_MAX_CHANNELS][36][SBLIMIT];
	float mdct_buf[MPA_MAX_CHANNELS][SBLIMIT * 18];
	GranuleDef granules[2][2];
	int dither_state;

	/* frame input, padded so the reader never walks off the end */
	uint8_t frame_buf[MPA_MAX_FRAME_BYTES + PADDING];

	/* planar float output */
	float out[MPA_MAX_CHANNELS][MPA_FRAME_SIZE];
};

///////////////////////////////////////////////////////////////////////////////
// Frame header (mpegaudiodecheader.c avpriv_mpegaudio_decode_header)

struct MPAHeader {
	int layer, lsf, mpeg25, sample_rate, sample_rate_index, bit_rate;
	int frame_size, nb_channels, mode, mode_ext, error_protection;
	int samples_per_frame;
};

/* ff_mpa_check_header: sync, layer and rate fields must all be valid. */
static int mpa_check_header(uint32_t header)
{
	if ((header & 0xffe00000) != 0xffe00000) return -1;	/* syncword */
	if ((header & (3 << 17)) == 0)           return -1;	/* layer 0 is reserved */
	if ((header & (0xf << 12)) == 0xf << 12) return -1;	/* bitrate index 15 */
	if ((header & (3 << 10)) == 3 << 10)     return -1;	/* sample rate 3 */
	return 0;
}

static int mpa_decode_header(MPAHeader* h, uint32_t header)
{
	if (mpa_check_header(header) < 0) return -1;

	if (header & (1 << 20)) {
		h->lsf = (header & (1 << 19)) ? 0 : 1;
		h->mpeg25 = 0;
	} else {
		h->lsf = 1;
		h->mpeg25 = 1;
	}

	h->layer = 4 - ((header >> 17) & 3);

	int sample_rate_index = (header >> 10) & 3;
	if (sample_rate_index >= 3) sample_rate_index = 0;
	const int sample_rate = ff_mpa_freq_tab[sample_rate_index] >> (h->lsf + h->mpeg25);
	h->sample_rate_index = sample_rate_index + 3 * (h->lsf + h->mpeg25);
	h->sample_rate = sample_rate;
	h->error_protection = ((header >> 16) & 1) ^ 1;

	const int bitrate_index = (header >> 12) & 0xf;
	const int padding = (header >> 9) & 1;
	h->mode = (header >> 6) & 3;
	h->mode_ext = (header >> 4) & 3;
	h->nb_channels = (h->mode == MPA_MONO) ? 1 : 2;

	if (bitrate_index == 0) return -1;	/* free format: size is not derivable */

	int frame_size = ff_mpa_bitrate_tab[h->lsf][h->layer - 1][bitrate_index];
	h->bit_rate = frame_size * 1000;
	switch (h->layer) {
	case 1:
		frame_size = (frame_size * 12000) / sample_rate;
		frame_size = (frame_size + padding) * 4;
		h->samples_per_frame = 384;
		break;
	case 2:
		frame_size = (frame_size * 144000) / sample_rate;
		frame_size += padding;
		h->samples_per_frame = 1152;
		break;
	default:
	case 3:
		frame_size = (frame_size * 144000) / (sample_rate << h->lsf);
		frame_size += padding;
		h->samples_per_frame = h->lsf ? 576 : 1152;
		break;
	}
	h->frame_size = frame_size;
	return 0;
}

///////////////////////////////////////////////////////////////////////////////
// Layer I (mp_decode_layer1)

static int mp_decode_layer1(MPACtx* s)
{
	uint8_t allocation[MPA_MAX_CHANNELS][SBLIMIT];
	uint8_t scale_factors[MPA_MAX_CHANNELS][SBLIMIT];
	GetBits* gb = &s->gb;

	const int bound = (s->mode == MPA_JSTEREO) ? (s->mode_ext + 1) * 4 : SBLIMIT;

	for (int i = 0; i < bound; i++)
		for (int ch = 0; ch < s->nb_channels; ch++)
			allocation[ch][i] = (uint8_t)get_bits(gb, 4);
	for (int i = bound; i < SBLIMIT; i++)
		allocation[0][i] = (uint8_t)get_bits(gb, 4);

	for (int i = 0; i < bound; i++)
		for (int ch = 0; ch < s->nb_channels; ch++)
			if (allocation[ch][i]) scale_factors[ch][i] = (uint8_t)get_bits(gb, 6);
	for (int i = bound; i < SBLIMIT; i++) {
		if (allocation[0][i]) {
			scale_factors[0][i] = (uint8_t)get_bits(gb, 6);
			scale_factors[1][i] = (uint8_t)get_bits(gb, 6);
		}
	}

	for (int j = 0; j < 12; j++) {
		for (int i = 0; i < bound; i++) {
			for (int ch = 0; ch < s->nb_channels; ch++) {
				const int n = allocation[ch][i];
				int v = 0;
				if (n) {
					const int mant = get_bits(gb, n + 1);
					v = l1_unscale(n, mant, scale_factors[ch][i]);
				}
				s->sb_samples[ch][j][i] = (float)v;
			}
		}
		for (int i = bound; i < SBLIMIT; i++) {
			const int n = allocation[0][i];
			if (n) {
				const int mant = get_bits(gb, n + 1);
				s->sb_samples[0][j][i] = (float)l1_unscale(n, mant, scale_factors[0][i]);
				s->sb_samples[1][j][i] = (float)l1_unscale(n, mant, scale_factors[1][i]);
			} else {
				s->sb_samples[0][j][i] = 0;
				s->sb_samples[1][j][i] = 0;
			}
		}
	}
	return 12;
}

///////////////////////////////////////////////////////////////////////////////
// Layer II (mp_decode_layer2)

/* Which of the five bit-allocation tables applies (mpegaudio.c). */
static int mpa_l2_select_table(int bitrate, int nb_channels, int freq, int lsf)
{
	const int ch_bitrate = bitrate / nb_channels;
	if (lsf) return 4;
	if ((freq == 48000 && ch_bitrate >= 56) || (ch_bitrate >= 56 && ch_bitrate <= 80)) return 0;
	if (freq != 48000 && ch_bitrate >= 96) return 1;
	if (freq != 32000 && ch_bitrate <= 48) return 2;
	return 3;
}

static int mp_decode_layer2(MPACtx* s)
{
	unsigned char bit_alloc[MPA_MAX_CHANNELS][SBLIMIT];
	unsigned char scale_code[MPA_MAX_CHANNELS][SBLIMIT];
	unsigned char scale_factors[MPA_MAX_CHANNELS][SBLIMIT][3];
	GetBits* gb = &s->gb;
	int j, v;

	const int table = mpa_l2_select_table(s->bit_rate / 1000, s->nb_channels,
	                                      s->sample_rate, s->lsf);
	const int sblimit = ff_mpa_sblimit_table[table];
	const unsigned char* alloc_table = ff_mpa_alloc_tables[table];

	int bound = (s->mode == MPA_JSTEREO) ? (s->mode_ext + 1) * 4 : sblimit;
	if (bound > sblimit) bound = sblimit;

	/* bit allocation */
	j = 0;
	for (int i = 0; i < bound; i++) {
		const int bit_alloc_bits = alloc_table[j];
		for (int ch = 0; ch < s->nb_channels; ch++)
			bit_alloc[ch][i] = (unsigned char)get_bits(gb, bit_alloc_bits);
		j += 1 << bit_alloc_bits;
	}
	for (int i = bound; i < sblimit; i++) {
		const int bit_alloc_bits = alloc_table[j];
		v = get_bits(gb, bit_alloc_bits);
		bit_alloc[0][i] = (unsigned char)v;
		bit_alloc[1][i] = (unsigned char)v;
		j += 1 << bit_alloc_bits;
	}

	/* scale codes */
	for (int i = 0; i < sblimit; i++)
		for (int ch = 0; ch < s->nb_channels; ch++)
			if (bit_alloc[ch][i]) scale_code[ch][i] = (unsigned char)get_bits(gb, 2);

	/* scale factors */
	for (int i = 0; i < sblimit; i++) {
		for (int ch = 0; ch < s->nb_channels; ch++) {
			if (!bit_alloc[ch][i]) continue;
			unsigned char* sf = scale_factors[ch][i];
			switch (scale_code[ch][i]) {
			default:
			case 0:
				sf[0] = (unsigned char)get_bits(gb, 6);
				sf[1] = (unsigned char)get_bits(gb, 6);
				sf[2] = (unsigned char)get_bits(gb, 6);
				break;
			case 2:
				sf[0] = (unsigned char)get_bits(gb, 6);
				sf[1] = sf[0];
				sf[2] = sf[0];
				break;
			case 1:
				sf[0] = (unsigned char)get_bits(gb, 6);
				sf[2] = (unsigned char)get_bits(gb, 6);
				sf[1] = sf[0];
				break;
			case 3:
				sf[0] = (unsigned char)get_bits(gb, 6);
				sf[2] = (unsigned char)get_bits(gb, 6);
				sf[1] = sf[2];
				break;
			}
		}
	}

	/* samples */
	for (int k = 0; k < 3; k++) {
		for (int l = 0; l < 12; l += 3) {
			j = 0;
			for (int i = 0; i < bound; i++) {
				const int bit_alloc_bits = alloc_table[j];
				for (int ch = 0; ch < s->nb_channels; ch++) {
					const int b = bit_alloc[ch][i];
					if (b) {
						const int scale = scale_factors[ch][i][k];
						const int qindex = alloc_table[j + b];
						const int bits = ff_mpa_quant_bits[qindex];
						if (bits < 0) {
							/* three values packed into one code */
							v = get_bits(gb, -bits);
							const int v2 = ff_division_tabs[qindex][v];
							const int steps = ff_mpa_quant_steps[qindex];
							s->sb_samples[ch][k * 12 + l + 0][i] = (float)l2_unscale_group(steps,  v2       & 15, scale);
							s->sb_samples[ch][k * 12 + l + 1][i] = (float)l2_unscale_group(steps, (v2 >> 4) & 15, scale);
							s->sb_samples[ch][k * 12 + l + 2][i] = (float)l2_unscale_group(steps,  v2 >> 8,       scale);
						} else {
							for (int m = 0; m < 3; m++) {
								v = get_bits(gb, bits);
								s->sb_samples[ch][k * 12 + l + m][i] = (float)l1_unscale(bits - 1, v, scale);
							}
						}
					} else {
						s->sb_samples[ch][k * 12 + l + 0][i] = 0;
						s->sb_samples[ch][k * 12 + l + 1][i] = 0;
						s->sb_samples[ch][k * 12 + l + 2][i] = 0;
					}
				}
				j += 1 << bit_alloc_bits;
			}
			for (int i = bound; i < sblimit; i++) {
				const int bit_alloc_bits = alloc_table[j];
				const int b = bit_alloc[0][i];
				if (b) {
					const int scale0 = scale_factors[0][i][k];
					const int scale1 = scale_factors[1][i][k];
					const int qindex = alloc_table[j + b];
					const int bits = ff_mpa_quant_bits[qindex];
					if (bits < 0) {
						v = get_bits(gb, -bits);
						const int steps = ff_mpa_quant_steps[qindex];
						int mant = v % steps; v = v / steps;
						s->sb_samples[0][k * 12 + l + 0][i] = (float)l2_unscale_group(steps, mant, scale0);
						s->sb_samples[1][k * 12 + l + 0][i] = (float)l2_unscale_group(steps, mant, scale1);
						mant = v % steps; v = v / steps;
						s->sb_samples[0][k * 12 + l + 1][i] = (float)l2_unscale_group(steps, mant, scale0);
						s->sb_samples[1][k * 12 + l + 1][i] = (float)l2_unscale_group(steps, mant, scale1);
						s->sb_samples[0][k * 12 + l + 2][i] = (float)l2_unscale_group(steps, v, scale0);
						s->sb_samples[1][k * 12 + l + 2][i] = (float)l2_unscale_group(steps, v, scale1);
					} else {
						for (int m = 0; m < 3; m++) {
							const int mant = get_bits(gb, bits);
							s->sb_samples[0][k * 12 + l + m][i] = (float)l1_unscale(bits - 1, mant, scale0);
							s->sb_samples[1][k * 12 + l + m][i] = (float)l1_unscale(bits - 1, mant, scale1);
						}
					}
				} else {
					for (int m = 0; m < 3; m++) {
						s->sb_samples[0][k * 12 + l + m][i] = 0;
						s->sb_samples[1][k * 12 + l + m][i] = 0;
					}
				}
				j += 1 << bit_alloc_bits;
			}
			/* unused subbands */
			for (int i = sblimit; i < SBLIMIT; i++)
				for (int ch = 0; ch < s->nb_channels; ch++)
					for (int m = 0; m < 3; m++)
						s->sb_samples[ch][k * 12 + l + m][i] = 0;
		}
	}
	return 3 * 12;
}

///////////////////////////////////////////////////////////////////////////////
// Layer III (mp_decode_layer3 and helpers)

static void region_offset2size(GranuleDef* g)
{
	int j = 0;
	g->region_size[2] = 576 / 2;
	for (int i = 0; i < 3; i++) {
		const int k = FFMIN(g->region_size[i], g->big_values);
		g->region_size[i] = k - j;
		j = k;
	}
}

static void init_short_region(MPACtx* s, GranuleDef* g)
{
	if (g->block_type == 2) {
		g->region_size[0] = (s->sample_rate_index != 8) ? (36 / 2) : (72 / 2);
	} else {
		if (s->sample_rate_index <= 2)      g->region_size[0] = (36 / 2);
		else if (s->sample_rate_index != 8) g->region_size[0] = (54 / 2);
		else                                g->region_size[0] = (108 / 2);
	}
	g->region_size[1] = (576 / 2);
}

static void init_long_region(MPACtx* s, GranuleDef* g, int ra1, int ra2)
{
	g->region_size[0] = ff_band_index_long[s->sample_rate_index][ra1 + 1];
	const int l = FFMIN(ra1 + ra2 + 2, 22);
	g->region_size[1] = ff_band_index_long[s->sample_rate_index][l];
}

static void compute_band_indexes(MPACtx* s, GranuleDef* g)
{
	if (g->block_type == 2) {
		if (g->switch_point) {
			/* the first 36 samples (72 at 8 kHz) are handled as long blocks */
			g->long_end = (s->sample_rate_index <= 2) ? 8 : 6;
			g->short_start = 3;
		} else {
			g->long_end = 0;
			g->short_start = 0;
		}
	} else {
		g->short_start = 13;
		g->long_end = 22;
	}
}

static void exponents_from_scale_factors(MPACtx* s, GranuleDef* g, int16_t* exponents)
{
	int16_t* exp_ptr = exponents;
	const int gain = g->global_gain - 210;
	const int shift = g->scalefac_scale + 1;

	const uint8_t* bstab = ff_band_size_long[s->sample_rate_index];
	const uint8_t* pretab = ff_mpa_pretab[g->preflag];
	for (int i = 0; i < g->long_end; i++) {
		const int v0 = gain - ((g->scale_factors[i] + pretab[i]) << shift) + 400;
		for (int j = bstab[i]; j > 0; j--) *exp_ptr++ = (int16_t)v0;
	}

	if (g->short_start < 13) {
		bstab = ff_band_size_short[s->sample_rate_index];
		const int gains[3] = {
			gain - (g->subblock_gain[0] << 3),
			gain - (g->subblock_gain[1] << 3),
			gain - (g->subblock_gain[2] << 3),
		};
		int k = g->long_end;
		for (int i = g->short_start; i < 13; i++) {
			const int len = bstab[i];
			for (int l = 0; l < 3; l++) {
				const int v0 = gains[l] - (g->scale_factors[k++] << shift) + 400;
				for (int j = len; j > 0; j--) *exp_ptr++ = (int16_t)v0;
			}
		}
	}
}

/* Cross from the reservoir buffer into the current frame's bits mid-granule. */
static void switch_buffer(MPACtx* s, int* pos, int* end_pos, int* end_pos2)
{
	if (s->in_gb_valid && *pos >= s->gb.size_in_bits - s->extrasize * 8) {
		s->gb = s->in_gb;
		s->in_gb_valid = 0;
		s->extrasize = 0;
		skip_bits_long(&s->gb, *pos - *end_pos);
		*end_pos2 = *end_pos = *end_pos2 + get_bits_count(&s->gb) - *pos;
		*pos = get_bits_count(&s->gb);
	}
}

/* FFmpeg's READ_FLIP_SIGN for the float build: read one bit and use it as the
   sign BIT of the float, rather than negating it. */
static inline void read_flip_sign(GetBits* gb, float* dst, const float* src)
{
	uint32_t bits;
	memcpy(&bits, src, 4);
	bits ^= get_bits1(gb) << 31;
	memcpy(dst, &bits, 4);
}

static int huffman_decode(MPACtx* s, GranuleDef* g, int16_t* exponents, int end_pos2)
{
	int s_index = 0;
	int end_pos = FFMIN(end_pos2, s->gb.size_in_bits - s->extrasize * 8);

	/* low frequencies: the "big values" region */
	for (int i = 0; i < 3; i++) {
		int j = g->region_size[i];
		if (j == 0) continue;
		const int k = g->table_select[i];
		const int l = ff_mpa_huff_data[k][0];
		const int linbits = ff_mpa_huff_data[k][1];

		if (!l) {
			memset(&g->sb_hybrid[s_index], 0, sizeof(*g->sb_hybrid) * 2 * j);
			s_index += 2 * j;
			continue;
		}
		const Vlc* vlctab = &g_huff_vlc[l];

		for (; j > 0; j--) {
			int pos = get_bits_count(&s->gb);
			if (pos >= end_pos) {
				switch_buffer(s, &pos, &end_pos, &end_pos2);
				if (pos >= end_pos) break;
			}
			int y = get_vlc(&s->gb, vlctab);

			if (!y) {
				g->sb_hybrid[s_index] = g->sb_hybrid[s_index + 1] = 0;
				s_index += 2;
				continue;
			}

			const int exponent = exponents[s_index];
			int x, v;
			if (y & 16) {
				x = y >> 5;
				y = y & 0x0f;
				if (x < 15) {
					read_flip_sign(&s->gb, &g->sb_hybrid[s_index], &expval_table_float[exponent][x]);
				} else {
					x += get_bitsz(&s->gb, linbits);
					v = l3_unscale(x, exponent);
					if (get_bits1(&s->gb)) v = -v;
					g->sb_hybrid[s_index] = (float)v;
				}
				if (y < 15) {
					read_flip_sign(&s->gb, &g->sb_hybrid[s_index + 1], &expval_table_float[exponent][y]);
				} else {
					y += get_bitsz(&s->gb, linbits);
					v = l3_unscale(y, exponent);
					if (get_bits1(&s->gb)) v = -v;
					g->sb_hybrid[s_index + 1] = (float)v;
				}
			} else {
				x = y >> 5;
				y = y & 0x0f;
				x += y;
				if (x < 15) {
					read_flip_sign(&s->gb, &g->sb_hybrid[s_index + !!y], &expval_table_float[exponent][x]);
				} else {
					x += get_bitsz(&s->gb, linbits);
					v = l3_unscale(x, exponent);
					if (get_bits1(&s->gb)) v = -v;
					g->sb_hybrid[s_index + !!y] = (float)v;
				}
				g->sb_hybrid[s_index + !y] = 0;
			}
			s_index += 2;
		}
	}

	/* high frequencies: the count1 region, four values at a time */
	const Vlc* vlc = &g_huff_quad_vlc[g->count1table_select];
	int last_pos = 0;
	while (s_index <= 572) {
		int pos = get_bits_count(&s->gb);
		if (pos >= end_pos) {
			if (pos > end_pos2 && last_pos) {
				/* some encoders write a wrong part2_3_length; step back into the data */
				s_index -= 4;
				skip_bits_long(&s->gb, last_pos - pos);
				break;
			}
			switch_buffer(s, &pos, &end_pos, &end_pos2);
			if (pos >= end_pos) break;
		}
		last_pos = pos;

		int code = get_vlc(&s->gb, vlc);
		g->sb_hybrid[s_index + 0] = g->sb_hybrid[s_index + 1] =
		g->sb_hybrid[s_index + 2] = g->sb_hybrid[s_index + 3] = 0;
		while (code) {
			static const int idxtab[16] = { 3,3,2,2,1,1,1,1,0,0,0,0,0,0,0,0 };
			const int pos2 = s_index + idxtab[code];
			code ^= 8 >> idxtab[code];
			read_flip_sign(&s->gb, &g->sb_hybrid[pos2], &exp_table_float[exponents[pos2]]);
		}
		s_index += 4;
	}

	/* skip the extension bits */
	const int bits_left = end_pos2 - get_bits_count(&s->gb);
	if (s_index < 0) s_index = 0;
	memset(&g->sb_hybrid[s_index], 0, sizeof(*g->sb_hybrid) * (576 - s_index));
	skip_bits_long(&s->gb, bits_left);

	int i = get_bits_count(&s->gb);
	switch_buffer(s, &i, &end_pos, &end_pos2);
	return 0;
}

/* Short blocks arrive grouped by window; interleave them back into frequency order. */
static void reorder_block(MPACtx* s, GranuleDef* g)
{
	float tmp[576];
	float* ptr;

	if (g->block_type != 2) return;

	if (g->switch_point)
		ptr = g->sb_hybrid + ((s->sample_rate_index != 8) ? 36 : 72);
	else
		ptr = g->sb_hybrid;

	for (int i = g->short_start; i < 13; i++) {
		const int len = ff_band_size_short[s->sample_rate_index][i];
		float* ptr1 = ptr;
		float* dst = tmp;
		for (int j = len; j > 0; j--) {
			*dst++ = ptr[0 * len];
			*dst++ = ptr[1 * len];
			*dst++ = ptr[2 * len];
			ptr++;
		}
		ptr += 2 * len;
		memcpy(ptr1, tmp, len * 3 * sizeof(*ptr1));
	}
}

#define ISQRT2 0.70710678118654752440f

static void compute_stereo(MPACtx* s, GranuleDef* g0, GranuleDef* g1)
{
	float *tab0, *tab1;
	const float (*is_tab)[16];
	int sf_max;

	if (s->mode_ext & MODE_EXT_I_STEREO) {
		if (!s->lsf) {
			is_tab = is_table;
			sf_max = 7;
		} else {
			is_tab = is_table_lsf[g1->scalefac_compress & 1];
			sf_max = 16;
		}

		tab0 = g0->sb_hybrid + 576;
		tab1 = g1->sb_hybrid + 576;

		int non_zero_found_short[3] = { 0, 0, 0 };
		int k = (13 - g1->short_start) * 3 + g1->long_end - 3;
		for (int i = 12; i >= g1->short_start; i--) {
			if (i != 11) k -= 3;	/* the last band reuses the previous scale factor */
			const int len = ff_band_size_short[s->sample_rate_index][i];
			for (int l = 2; l >= 0; l--) {
				tab0 -= len;
				tab1 -= len;
				int do_ms = non_zero_found_short[l];
				if (!do_ms) {
					/* a non-zero band ends intensity stereo from here down */
					for (int j = 0; j < len; j++) {
						if (tab1[j] != 0) { non_zero_found_short[l] = 1; do_ms = 1; break; }
					}
					if (!do_ms) {
						const int sf = g1->scale_factors[k + l];
						if (sf >= sf_max) {
							do_ms = 1;
						} else {
							const float v1 = is_tab[0][sf];
							const float v2 = is_tab[1][sf];
							for (int j = 0; j < len; j++) {
								const float tmp0 = tab0[j];
								tab0[j] = tmp0 * v1;
								tab1[j] = tmp0 * v2;
							}
						}
					}
				}
				if (do_ms && (s->mode_ext & MODE_EXT_MS_STEREO)) {
					for (int j = 0; j < len; j++) {
						const float tmp0 = tab0[j], tmp1 = tab1[j];
						tab0[j] = (tmp0 + tmp1) * ISQRT2;
						tab1[j] = (tmp0 - tmp1) * ISQRT2;
					}
				}
			}
		}

		int non_zero_found = non_zero_found_short[0] | non_zero_found_short[1] | non_zero_found_short[2];

		for (int i = g1->long_end - 1; i >= 0; i--) {
			const int len = ff_band_size_long[s->sample_rate_index][i];
			tab0 -= len;
			tab1 -= len;
			int do_ms = non_zero_found;
			if (!do_ms) {
				for (int j = 0; j < len; j++) {
					if (tab1[j] != 0) { non_zero_found = 1; do_ms = 1; break; }
				}
				if (!do_ms) {
					const int k2 = (i == 21) ? 20 : i;	/* last band reuses the previous sf */
					const int sf = g1->scale_factors[k2];
					if (sf >= sf_max) {
						do_ms = 1;
					} else {
						const float v1 = is_tab[0][sf];
						const float v2 = is_tab[1][sf];
						for (int j = 0; j < len; j++) {
							const float tmp0 = tab0[j];
							tab0[j] = tmp0 * v1;
							tab1[j] = tmp0 * v2;
						}
					}
				}
			}
			if (do_ms && (s->mode_ext & MODE_EXT_MS_STEREO)) {
				for (int j = 0; j < len; j++) {
					const float tmp0 = tab0[j], tmp1 = tab1[j];
					tab0[j] = (tmp0 + tmp1) * ISQRT2;
					tab1[j] = (tmp0 - tmp1) * ISQRT2;
				}
			}
		}
	} else if (s->mode_ext & MODE_EXT_MS_STEREO) {
		/* MS stereo only. The 1/sqrt(2) is already folded into the global gain,
		   so this is a plain butterfly (FFmpeg's butterflies_float). */
		tab0 = g0->sb_hybrid;
		tab1 = g1->sb_hybrid;
		for (int i = 0; i < 576; i++) {
			const float tmp0 = tab0[i], tmp1 = tab1[i];
			tab0[i] = tmp0 + tmp1;
			tab1[i] = tmp0 - tmp1;
		}
	}
}

/* Alias reduction across subband boundaries (long blocks only). */
static void compute_antialias(MPACtx* s, GranuleDef* g)
{
	int n;
	if (g->block_type == 2) {
		if (!g->switch_point) return;
		n = 1;
	} else {
		n = SBLIMIT - 1;
	}

	float* ptr = g->sb_hybrid + 18;
	for (int i = n; i > 0; i--) {
		for (int j = 0; j < 8; j++) {
			const float tmp0 = ptr[-1 - j];
			const float tmp1 = ptr[j];
			ptr[-1 - j] = tmp0 * csa_table[j][0] - tmp1 * csa_table[j][1];
			ptr[    j ] = tmp0 * csa_table[j][1] + tmp1 * csa_table[j][0];
		}
		ptr += 18;
	}
	(void)s;
}

static void compute_imdct(MPACtx* s, GranuleDef* g, float* sb_samples, float* mdct_buf)
{
	float out2[12];
	int mdct_long_end;

	/* find the last non-zero subband */
	float* ptr = g->sb_hybrid + 576;
	float* ptr1 = g->sb_hybrid + 2 * 18;
	while (ptr >= ptr1) {
		ptr -= 6;
		int32_t p[6];
		memcpy(p, ptr, sizeof(p));
		if (p[0] | p[1] | p[2] | p[3] | p[4] | p[5]) break;
	}
	const int sblimit = (int)((ptr - g->sb_hybrid) / 18) + 1;

	if (g->block_type == 2)
		mdct_long_end = g->switch_point ? 2 : 0;
	else
		mdct_long_end = sblimit;

	imdct36_blocks(sb_samples, mdct_buf, g->sb_hybrid,
	               mdct_long_end, g->switch_point, g->block_type);

	float* buf = mdct_buf + 4 * 18 * (mdct_long_end >> 2) + (mdct_long_end & 3);
	ptr = g->sb_hybrid + 18 * mdct_long_end;

	for (int j = mdct_long_end; j < sblimit; j++) {
		/* odd subbands use the frequency-inverted window set */
		float* win = ff_mdct_win[2 + (4 & -(j & 1))];
		float* out_ptr = sb_samples + j;

		for (int i = 0; i < 6; i++) {
			*out_ptr = buf[4 * i];
			out_ptr += SBLIMIT;
		}
		imdct12(out2, ptr + 0);
		for (int i = 0; i < 6; i++) {
			*out_ptr             = out2[i]     * win[i]     + buf[4 * (i + 6 * 1)];
			buf[4 * (i + 6 * 2)] = out2[i + 6] * win[i + 6];
			out_ptr += SBLIMIT;
		}
		imdct12(out2, ptr + 1);
		for (int i = 0; i < 6; i++) {
			*out_ptr             = out2[i]     * win[i]     + buf[4 * (i + 6 * 2)];
			buf[4 * (i + 6 * 0)] = out2[i + 6] * win[i + 6];
			out_ptr += SBLIMIT;
		}
		imdct12(out2, ptr + 2);
		for (int i = 0; i < 6; i++) {
			buf[4 * (i + 6 * 0)] = out2[i]     * win[i]     + buf[4 * (i + 6 * 0)];
			buf[4 * (i + 6 * 1)] = out2[i + 6] * win[i + 6];
			buf[4 * (i + 6 * 2)] = 0;
		}
		ptr += 18;
		buf += (j & 3) != 3 ? 1 : (4 * 18 - 3);
	}
	/* zero subbands: overlap only */
	for (int j = sblimit; j < SBLIMIT; j++) {
		float* out_ptr = sb_samples + j;
		for (int i = 0; i < 18; i++) {
			*out_ptr = buf[4 * i];
			buf[4 * i] = 0;
			out_ptr += SBLIMIT;
		}
		buf += (j & 3) != 3 ? 1 : (4 * 18 - 3);
	}
	(void)s;
}

/* SPLIT: unpack the LSF scalefactor-length code (mpegaudiodec_template.c). */
static inline void lsf_split(int* dst, int* sf, int n)
{
	if (n == 3)      { const int m = (*sf * 171) >> 9;  *dst = *sf - 3 * m; *sf = m; }
	else if (n == 4) { *dst = *sf & 3; *sf >>= 2; }
	else if (n == 5) { const int m = (*sf * 205) >> 10; *dst = *sf - 5 * m; *sf = m; }
	else if (n == 6) { const int m = (*sf * 171) >> 10; *dst = *sf - 6 * m; *sf = m; }
	else             { *dst = 0; }
}

static void lsf_sf_expand(int* slen, int sf, int n1, int n2, int n3)
{
	lsf_split(&slen[3], &sf, n3);
	lsf_split(&slen[2], &sf, n2);
	lsf_split(&slen[1], &sf, n1);
	slen[0] = sf;
}

static inline int av_clip_i(int a, int lo, int hi) { return a < lo ? lo : (a > hi ? hi : a); }

static int mp_decode_layer3(MPACtx* s)
{
	int nb_granules, main_data_begin;
	int gr, ch, i, j, k, n;
	int16_t exponents[576];
	GetBits* gb = &s->gb;

	/* ── side info ── */
	if (s->lsf) {
		main_data_begin = get_bits(gb, 8);
		skip_bits(gb, s->nb_channels);
		nb_granules = 1;
	} else {
		main_data_begin = get_bits(gb, 9);
		skip_bits(gb, s->nb_channels == 2 ? 3 : 5);
		nb_granules = 2;
		for (ch = 0; ch < s->nb_channels; ch++) {
			s->granules[ch][0].scfsi = 0;	/* granule 0 always sends all scalefactors */
			s->granules[ch][1].scfsi = (uint8_t)get_bits(gb, 4);
		}
	}

	for (gr = 0; gr < nb_granules; gr++) {
		for (ch = 0; ch < s->nb_channels; ch++) {
			GranuleDef* g = &s->granules[ch][gr];
			g->part2_3_length = get_bits(gb, 12);
			g->big_values = get_bits(gb, 9);
			if (g->big_values > 288) return -1;

			g->global_gain = get_bits(gb, 8);
			/* with MS stereo only, fold the 1/sqrt(2) renormalization into the gain */
			if ((s->mode_ext & (MODE_EXT_MS_STEREO | MODE_EXT_I_STEREO)) == MODE_EXT_MS_STEREO)
				g->global_gain -= 2;
			g->scalefac_compress = s->lsf ? get_bits(gb, 9) : get_bits(gb, 4);

			const int blocksplit_flag = get_bits1(gb);
			if (blocksplit_flag) {
				g->block_type = (uint8_t)get_bits(gb, 2);
				if (g->block_type == 0) return -1;
				g->switch_point = (uint8_t)get_bits1(gb);
				for (i = 0; i < 2; i++) g->table_select[i] = get_bits(gb, 5);
				for (i = 0; i < 3; i++) g->subblock_gain[i] = get_bits(gb, 3);
				init_short_region(s, g);
			} else {
				g->block_type = 0;
				g->switch_point = 0;
				for (i = 0; i < 3; i++) g->table_select[i] = get_bits(gb, 5);
				const int region_address1 = get_bits(gb, 4);
				const int region_address2 = get_bits(gb, 3);
				init_long_region(s, g, region_address1, region_address2);
			}
			region_offset2size(g);
			compute_band_indexes(s, g);

			g->preflag = 0;
			if (!s->lsf) g->preflag = get_bits1(gb);
			g->scalefac_scale = (uint8_t)get_bits1(gb);
			g->count1table_select = (uint8_t)get_bits1(gb);
		}
	}

	/* ── bit reservoir: main data starts main_data_begin bytes BEFORE this frame ── */
	{
		const uint8_t* ptr = s->gb.buf + (get_bits_count(&s->gb) >> 3);
		s->extrasize = av_clip_i((get_bits_left(&s->gb) >> 3) - s->extrasize, 0,
		                         FFMAX(0, LAST_BUF_SIZE - s->last_buf_size));
		memcpy(s->last_buf + s->last_buf_size, ptr, s->extrasize);
		s->in_gb = s->gb;
		s->in_gb_valid = 1;
		init_get_bits(&s->gb, s->last_buf, (s->last_buf_size + s->extrasize) * 8);
		s->last_buf_size <<= 3;

		/* not enough history for the first granule(s): emit silence through the
		   IMDCT so the overlap-add state stays consistent */
		for (gr = 0; gr < nb_granules && (s->last_buf_size >> 3) < main_data_begin; gr++) {
			for (ch = 0; ch < s->nb_channels; ch++) {
				GranuleDef* g = &s->granules[ch][gr];
				s->last_buf_size += g->part2_3_length;
				memset(g->sb_hybrid, 0, sizeof(g->sb_hybrid));
				compute_imdct(s, g, &s->sb_samples[ch][18 * gr][0], s->mdct_buf[ch]);
			}
		}
		const int skip = s->last_buf_size - 8 * main_data_begin;
		if (skip >= s->gb.size_in_bits - s->extrasize * 8 && s->in_gb_valid) {
			skip_bits_long(&s->in_gb, skip - s->gb.size_in_bits + s->extrasize * 8);
			s->gb = s->in_gb;
			s->in_gb_valid = 0;
			s->extrasize = 0;
		} else {
			skip_bits_long(&s->gb, skip);
		}
	}

	/* ── granules ── */
	for (; gr < nb_granules; gr++) {
		for (ch = 0; ch < s->nb_channels; ch++) {
			GranuleDef* g = &s->granules[ch][gr];
			const int bits_pos = get_bits_count(&s->gb);

			if (!s->lsf) {
				/* MPEG-1 scale factors */
				const int slen1 = ff_slen_table[0][g->scalefac_compress];
				const int slen2 = ff_slen_table[1][g->scalefac_compress];
				if (g->block_type == 2) {
					n = g->switch_point ? 17 : 18;
					j = 0;
					if (slen1) { for (i = 0; i < n; i++) g->scale_factors[j++] = (uint8_t)get_bits(&s->gb, slen1); }
					else       { for (i = 0; i < n; i++) g->scale_factors[j++] = 0; }
					if (slen2) {
						for (i = 0; i < 18; i++) g->scale_factors[j++] = (uint8_t)get_bits(&s->gb, slen2);
						for (i = 0; i < 3; i++)  g->scale_factors[j++] = 0;
					} else {
						for (i = 0; i < 21; i++) g->scale_factors[j++] = 0;
					}
				} else {
					const uint8_t* sc = s->granules[ch][0].scale_factors;
					j = 0;
					for (k = 0; k < 4; k++) {
						n = (k == 0) ? 6 : 5;
						if ((g->scfsi & (0x8 >> k)) == 0) {
							const int slen = (k < 2) ? slen1 : slen2;
							if (slen) { for (i = 0; i < n; i++) g->scale_factors[j++] = (uint8_t)get_bits(&s->gb, slen); }
							else      { for (i = 0; i < n; i++) g->scale_factors[j++] = 0; }
						} else {
							/* reuse granule 0's */
							for (i = 0; i < n; i++) { g->scale_factors[j] = sc[j]; j++; }
						}
					}
					g->scale_factors[j++] = 0;
				}
			} else {
				/* LSF scale factors */
				int tindex, tindex2, slen[4], sl, sf;

				if (g->block_type == 2) tindex = g->switch_point ? 2 : 1;
				else                    tindex = 0;

				sf = g->scalefac_compress;
				if ((s->mode_ext & MODE_EXT_I_STEREO) && ch == 1) {
					sf >>= 1;
					if (sf < 180)      { lsf_sf_expand(slen, sf, 6, 6, 0);       tindex2 = 3; }
					else if (sf < 244) { lsf_sf_expand(slen, sf - 180, 4, 4, 0); tindex2 = 4; }
					else               { lsf_sf_expand(slen, sf - 244, 3, 0, 0); tindex2 = 5; }
				} else {
					if (sf < 400)      { lsf_sf_expand(slen, sf, 5, 4, 4);       tindex2 = 0; }
					else if (sf < 500) { lsf_sf_expand(slen, sf - 400, 5, 4, 0); tindex2 = 1; }
					else               { lsf_sf_expand(slen, sf - 500, 3, 0, 0); tindex2 = 2; g->preflag = 1; }
				}

				j = 0;
				for (k = 0; k < 4; k++) {
					n  = ff_lsf_nsf_table[tindex2][tindex][k];
					sl = slen[k];
					if (sl) { for (i = 0; i < n; i++) g->scale_factors[j++] = (uint8_t)get_bits(&s->gb, sl); }
					else    { for (i = 0; i < n; i++) g->scale_factors[j++] = 0; }
				}
				for (; j < 40; j++) g->scale_factors[j] = 0;
			}

			exponents_from_scale_factors(s, g, exponents);
			huffman_decode(s, g, exponents, bits_pos + g->part2_3_length);
		}

		if (s->mode == MPA_JSTEREO)
			compute_stereo(s, &s->granules[0][gr], &s->granules[1][gr]);

		for (ch = 0; ch < s->nb_channels; ch++) {
			GranuleDef* g = &s->granules[ch][gr];
			if (g->block_type == 2) s->short_blocks++;
			reorder_block(s, g);
			compute_antialias(s, g);
			compute_imdct(s, g, &s->sb_samples[ch][18 * gr][0], s->mdct_buf[ch]);
		}
	}

	if (get_bits_count(&s->gb) < 0)
		skip_bits_long(&s->gb, -get_bits_count(&s->gb));
	return nb_granules * 18;
}

///////////////////////////////////////////////////////////////////////////////
// Frame decode and the extern "C" boundary

static int mp_decode_frame(MPACtx* s, const uint8_t* buf, int buf_size)
{
	int nb_frames;

	init_get_bits(&s->gb, buf + HEADER_SIZE, (buf_size - HEADER_SIZE) * 8);
	if (s->error_protection) skip_bits(&s->gb, 16);	/* CRC word; not verified */

	switch (s->layer) {
	case 1:
		nb_frames = mp_decode_layer1(s);
		break;
	case 2:
		nb_frames = mp_decode_layer2(s);
		break;
	default:
	case 3:
		nb_frames = mp_decode_layer3(s);

		/* Carry the tail of this frame into the reservoir for the next one. */
		s->last_buf_size = 0;
		if (s->in_gb_valid) {
			align_get_bits(&s->gb);
			int i = (get_bits_left(&s->gb) >> 3) - s->extrasize;
			if (i >= 0 && i <= BACKSTEP_SIZE) {
				memmove(s->last_buf, s->gb.buf + (get_bits_count(&s->gb) >> 3), i);
				s->last_buf_size = i;
			}
			s->gb = s->in_gb;
			s->in_gb_valid = 0;
			s->extrasize = 0;
		}
		align_get_bits(&s->gb);
		int i = (get_bits_left(&s->gb) >> 3) - s->extrasize;
		if (i < 0 || i > BACKSTEP_SIZE || nb_frames < 0)
			i = FFMIN(BACKSTEP_SIZE, buf_size - HEADER_SIZE);
		if (i > 0 && s->last_buf_size + i <= LAST_BUF_SIZE) {
			memcpy(s->last_buf + s->last_buf_size, s->gb.buf + buf_size - HEADER_SIZE - i, i);
			s->last_buf_size += i;
		}
		break;
	}

	if (nb_frames < 0) return nb_frames;

	/* polyphase synthesis: 32 subband samples -> 32 PCM samples, per block */
	for (int ch = 0; ch < s->nb_channels; ch++) {
		float* samples_ptr = s->out[ch];
		for (int i = 0; i < nb_frames; i++) {
			mpa_synth_filter(s->synth_buf[ch], &s->synth_buf_offset[ch],
			                 ff_mpa_synth_window, &s->dither_state,
			                 samples_ptr, 1, s->sb_samples[ch][i]);
			samples_ptr += 32;
		}
	}
	return nb_frames * 32;
}

extern "C" {

MPACtx* mpa_open(void)
{
	if (mpegaudio_init_static()) return 0;
	MPACtx* s = (MPACtx*)calloc(1, sizeof(MPACtx));
	if (!s) return 0;
	for (int ch = 0; ch < MPA_MAX_CHANNELS; ch++) s->synth_buf_offset[ch] = 0;
	return s;
}

void mpa_close(MPACtx* s) { free(s); }

/* Seek: drop the bit reservoir and the filterbank/IMDCT history so a new position
   does not inherit the previous one's tail (FFmpeg's mp_flush). */
void mpa_reset(MPACtx* s)
{
	if (!s) return;
	memset(s->synth_buf, 0, sizeof(s->synth_buf));
	memset(s->mdct_buf, 0, sizeof(s->mdct_buf));
	memset(s->sb_samples, 0, sizeof(s->sb_samples));
	for (int ch = 0; ch < MPA_MAX_CHANNELS; ch++) s->synth_buf_offset[ch] = 0;
	s->last_buf_size = 0;
	s->extrasize = 0;
	s->in_gb_valid = 0;
	s->dither_state = 0;
}

/*
	Parse the 4-byte header at `buf`, without decoding and without touching decoder
	state. info = { frameSize, sampleRate, channels, bitRate, layer, samplesPerFrame }.
*/
int mpa_probe_frame(const uint8_t* buf, int len, int* info)
{
	MPAHeader h;
	if (!buf || len < HEADER_SIZE) return 0;

	const uint32_t header = ((uint32_t)buf[0] << 24) | ((uint32_t)buf[1] << 16) |
	                        ((uint32_t)buf[2] << 8)  |  (uint32_t)buf[3];
	if (mpa_decode_header(&h, header)) return 0;
	if (h.frame_size < HEADER_SIZE || h.frame_size > MPA_MAX_FRAME_BYTES) return 0;

	info[0] = h.frame_size;
	info[1] = h.sample_rate;
	info[2] = h.nb_channels;
	info[3] = h.bit_rate;
	info[4] = h.layer;
	info[5] = h.samples_per_frame;
	return 1;
}

/*
	Decode ONE frame starting at the header in buf[0].
	info = { frameSize, sampleRate, channels, bitRate, layer, samplesDecoded,
	shortBlocks }.

	Layer III frames are NOT independent: the Huffman data for a frame generally
	lives partly in the preceding frames' spare bytes (the bit reservoir), so frames
	must be fed in order and mpa_reset() called after a seek. The first frame or two
	after a reset can legitimately decode to silence for that reason.
*/
int mpa_decode_frame(MPACtx* s, const uint8_t* buf, int len, int* info)
{
	MPAHeader h;
	if (!s || !buf || len < HEADER_SIZE) return 0;

	const uint32_t header = ((uint32_t)buf[0] << 24) | ((uint32_t)buf[1] << 16) |
	                        ((uint32_t)buf[2] << 8)  |  (uint32_t)buf[3];
	if (mpa_decode_header(&h, header)) return 0;
	if (h.frame_size < HEADER_SIZE || h.frame_size > MPA_MAX_FRAME_BYTES) return 0;
	if (h.frame_size > len) return 0;	/* incomplete frame */

	/* A layer or channel-count change invalidates the filterbank history. */
	if (s->layer != h.layer || s->nb_channels != h.nb_channels ||
	    s->sample_rate != h.sample_rate)
		mpa_reset(s);

	s->layer = h.layer;
	s->lsf = h.lsf;
	s->mpeg25 = h.mpeg25;
	s->sample_rate = h.sample_rate;
	s->sample_rate_index = h.sample_rate_index;
	s->bit_rate = h.bit_rate;
	s->frame_size = h.frame_size;
	s->nb_channels = h.nb_channels;
	s->mode = h.mode;
	s->mode_ext = h.mode_ext;
	s->error_protection = h.error_protection;

	/* Copy into the padded frame buffer so the bit reader can never walk off the
	   end of the caller's memory (FFmpeg relies on input buffer padding instead). */
	memcpy(s->frame_buf, buf, h.frame_size);
	memset(s->frame_buf + h.frame_size, 0, PADDING);

	s->short_blocks = 0;
	const int n = mp_decode_frame(s, s->frame_buf, h.frame_size);

	info[0] = h.frame_size;
	info[1] = h.sample_rate;
	info[2] = h.nb_channels;
	info[3] = h.bit_rate;
	info[4] = h.layer;
	info[5] = (n > 0) ? n : 0;
	info[6] = s->short_blocks;
	return n > 0;
}

float* mpa_channel_ptr(MPACtx* s, int ch)
{
	if (!s || ch < 0 || ch >= MPA_MAX_CHANNELS) return 0;
	return s->out[ch];
}

} // extern "C"
