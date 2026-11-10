/*
	DTS Coherent Acoustics core audio decoder -- WASM C++.

	A direct port of FFmpeg's DCA core decoder (libavcodec, tag n9.0.2):
	  dca_core.c / dca_core.h / dca.c / dca.h / dcadec.c / dcadsp.c / synth_filter.c / dcaadpcm.h /
	  dcamath.h / dca_sample_rate_tab.h / dca_syncwords.h, the tables of dcadata.c and dcahuff.c
	  (dts_tables.h), and the inverse MDCT definition of libavutil/tx_template.c

	This file is part of a derivative of FFmpeg. FFmpeg is free software; you can
	redistribute it and/or modify it under the terms of the GNU Lesser General Public
	License as published by the Free Software Foundation; either version 2.1 of the
	License, or (at your option) any later version. It is distributed in the hope that it
	will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
	License for more details. Copyright (C) 2016 foo86; Copyright (C) 2004 Gildas Bazin,
	Benjamin Zores; Copyright (C) 2006 Benjamin Larsson; Copyright (C) 2007 Konstantin Shishkov.

	Scope -- the core substream only, as FFmpeg's `-core_only 1`: every audio mode up to
	C L R Ls Rs with or without LFE, any core sample rate, float output (FFmpeg's default
	path, filter_frame_float). This is what a DVD carries (5.1 at 48 kHz).

	NOT decoded: the core extensions XCH, XXCH and X96, and the extension substream (XBR, XLL
	/ DTS-HD Master Audio, LBR / DTS Express). A stream that carries one plays its core: a
	DTS-ES 6.1 track gives 5.1 with the back center still mixed into the surrounds, a 96/24
	track gives 48 kHz. An extension substream frame between core frames is not a core frame
	and the caller steps over it.

	Differences from FFmpeg, all behavior-preserving for that scope:
	  - The surrounding framework (AVCodecContext / AVFrame / the DSP function pointers) is
	    replaced by a self-contained handle; a frame's samples go to the handle's own planar
	    buffer, in FFmpeg's native channel order (ff_dca_set_channel_layout: FL FR FC LFE BC
	    SL SR, the ones the frame has).
	  - The 32-point inverse MDCT of the synthesis filter is the definition
	    (ff_tx_mdct_naive_inv) evaluated in double, in place of libavutil/tx's float FFT, so the
	    PCM differs from `ffmpeg -f f32le` by float round-off (tests/DTS.test.js).
	  - The VLCs are built here as two-level tables rather than by ff_vlc_init_from_lengths;
	    dts_self_test checks that every code book is a complete prefix code.
	  - The bit reader reads zeros past the end of the frame, where FFmpeg's reads its padding.
	  - The auxiliary data (the stereo downmix coefficients) is not parsed: nothing here
	    downmixes, and FFmpeg does not fail a frame on it either.
	  - The subband history keeps its place when a stream changes its frame length, where
	    FFmpeg re-lays its buffer out.
	  - Input: the 16-bit big-endian stream (sync 7F FE 80 01), which is what a DVD and the
	    samples carry, and its byte-swapped little-endian form (FE 7F 01 80; conversion as
	    avpriv_dca_convert_bitstream, not seen in a sample). The 14-bit forms are not taken.

	Seek: dts_reset clears what ties a frame to the frames before it (the ADPCM history, the
	LFE history and the filter bank's), as ff_dca_core_flush.

	Boundary (extern "C", prefixed _ on the JS side; see dts_decode_core.js):
	  DTSContext* dts_open(void)
	  void    dts_close(DTSContext*)
	  void    dts_reset(DTSContext*)
	  int     dts_probe_frame(const uint8_t* data, int size, int info[5])   size >= 16
	            info = {frameSize, sampleRate, channels, bitRate, sampleCount}; 1, 0 for what is
	            not a core frame header
	  int     dts_decode_frame(DTSContext*, const uint8_t* data, int size, int info[5])
	            info = {frameSize, sampleRate, channels, sampleCount, extension}; frameSize 0 when
	            the header did not parse; 1 when the frame decoded. extension: 0 none, else the
	            core extension the frame says it carries + 1 (1 XCH, 3 X96, 7 XXCH)
	  float*  dts_channel_ptr(DTSContext*, int ch)   the last frame's channel, sampleCount floats
	  int     dts_self_test(void)             0, or the number of code books that are not complete
*/

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "dts_tables.h"

#define FFMAX(a, b) ((a) > (b) ? (a) : (b))
#define FF_ARRAY_ELEMS(a) (sizeof(a) / sizeof((a)[0]))

#define DCA_CODE_BOOKS          10
#define DCA_ADPCM_COEFFS        4

/* FFmpeg's 7 leaves room for the XCH and XXCH channels, which are not decoded here. */
#define DCA_CHANNELS            5
#define DCA_SUBBANDS            32
#define DCA_SUBFRAMES           16
#define DCA_SUBBAND_SAMPLES     8
#define DCA_PCMBLOCK_SAMPLES    32
#define DCA_LFE_HISTORY         8
#define DCA_ABITS_MAX           26

/* NBLKS is 7 bits. */
#define DCA_PCMBLOCKS_MAX       128
#define DCA_FRAME_SAMPLES_MAX   (DCA_PCMBLOCKS_MAX * DCA_PCMBLOCK_SAMPLES)

/* FSIZE is 14 bits. */
#define DCA_FRAME_BYTES_MAX     16384
#define DCA_INPUT_PADDING       8

#define DCA_SYNCWORD_CORE_BE    0x7FFE8001U
#define DCA_SYNCWORD_CORE_LE    0xFE7F0180U

#define DCA_TMODE_VLC_BITS      3
#define DCA_SCALES_VLC_BITS     9

enum DCASpeaker {
	DCA_SPEAKER_C, DCA_SPEAKER_L, DCA_SPEAKER_R, DCA_SPEAKER_Ls,
	DCA_SPEAKER_Rs, DCA_SPEAKER_LFE1, DCA_SPEAKER_Cs,
	DCA_SPEAKER_COUNT
};

#define DCA_SPEAKER_MASK_C      (1 << DCA_SPEAKER_C)
#define DCA_SPEAKER_MASK_L      (1 << DCA_SPEAKER_L)
#define DCA_SPEAKER_MASK_R      (1 << DCA_SPEAKER_R)
#define DCA_SPEAKER_MASK_Ls     (1 << DCA_SPEAKER_Ls)
#define DCA_SPEAKER_MASK_Rs     (1 << DCA_SPEAKER_Rs)
#define DCA_SPEAKER_MASK_LFE1   (1 << DCA_SPEAKER_LFE1)
#define DCA_SPEAKER_MASK_Cs     (1 << DCA_SPEAKER_Cs)

#define DCA_SPEAKER_LAYOUT_MONO     (DCA_SPEAKER_MASK_C)
#define DCA_SPEAKER_LAYOUT_STEREO   (DCA_SPEAKER_MASK_L | DCA_SPEAKER_MASK_R)
#define DCA_SPEAKER_LAYOUT_3_0      (DCA_SPEAKER_LAYOUT_STEREO | DCA_SPEAKER_MASK_C)
#define DCA_SPEAKER_LAYOUT_2_1      (DCA_SPEAKER_LAYOUT_STEREO | DCA_SPEAKER_MASK_Cs)
#define DCA_SPEAKER_LAYOUT_3_1      (DCA_SPEAKER_LAYOUT_3_0 | DCA_SPEAKER_MASK_Cs)
#define DCA_SPEAKER_LAYOUT_2_2      (DCA_SPEAKER_LAYOUT_STEREO | DCA_SPEAKER_MASK_Ls | DCA_SPEAKER_MASK_Rs)
#define DCA_SPEAKER_LAYOUT_5POINT0  (DCA_SPEAKER_LAYOUT_3_0 | DCA_SPEAKER_MASK_Ls | DCA_SPEAKER_MASK_Rs)

enum DCACoreAudioMode {
	DCA_AMODE_MONO,             // Mode 0: A (mono)
	DCA_AMODE_MONO_DUAL,        // Mode 1: A + B (dual mono)
	DCA_AMODE_STEREO,           // Mode 2: L + R (stereo)
	DCA_AMODE_STEREO_SUMDIFF,   // Mode 3: (L+R) + (L-R) (sum-diff)
	DCA_AMODE_STEREO_TOTAL,     // Mode 4: LT + RT (left and right total)
	DCA_AMODE_3F,               // Mode 5: C + L + R
	DCA_AMODE_2F1R,             // Mode 6: L + R + S
	DCA_AMODE_3F1R,             // Mode 7: C + L + R + S
	DCA_AMODE_2F2R,             // Mode 8: L + R + SL + SR
	DCA_AMODE_3F2R,             // Mode 9: C + L + R + SL + SR

	DCA_AMODE_COUNT
};

enum DCACoreLFEFlag {
	DCA_LFE_FLAG_NONE,
	DCA_LFE_FLAG_128,
	DCA_LFE_FLAG_64,
	DCA_LFE_FLAG_INVALID
};

static const uint32_t ff_dca_sample_rates[16] = {
	0, 8000, 16000, 32000, 0, 0, 11025, 22050, 44100, 0, 0,
	12000, 24000, 48000, 96000, 192000
};

static const uint8_t ff_dca_bits_per_sample[8] = {
	16, 16, 20, 20, 0, 24, 24, 0
};

static const int8_t prm_ch_to_spkr_map[DCA_AMODE_COUNT][5] = {
	{ DCA_SPEAKER_C,            -1,             -1,             -1,             -1 },
	{ DCA_SPEAKER_L, DCA_SPEAKER_R,             -1,             -1,             -1 },
	{ DCA_SPEAKER_L, DCA_SPEAKER_R,             -1,             -1,             -1 },
	{ DCA_SPEAKER_L, DCA_SPEAKER_R,             -1,             -1,             -1 },
	{ DCA_SPEAKER_L, DCA_SPEAKER_R,             -1,             -1,             -1 },
	{ DCA_SPEAKER_C, DCA_SPEAKER_L, DCA_SPEAKER_R ,             -1,             -1 },
	{ DCA_SPEAKER_L, DCA_SPEAKER_R, DCA_SPEAKER_Cs,             -1,             -1 },
	{ DCA_SPEAKER_C, DCA_SPEAKER_L, DCA_SPEAKER_R , DCA_SPEAKER_Cs,             -1 },
	{ DCA_SPEAKER_L, DCA_SPEAKER_R, DCA_SPEAKER_Ls, DCA_SPEAKER_Rs,             -1 },
	{ DCA_SPEAKER_C, DCA_SPEAKER_L, DCA_SPEAKER_R,  DCA_SPEAKER_Ls, DCA_SPEAKER_Rs }
};

static const uint8_t audio_mode_ch_mask[DCA_AMODE_COUNT] = {
	DCA_SPEAKER_LAYOUT_MONO,
	DCA_SPEAKER_LAYOUT_STEREO,
	DCA_SPEAKER_LAYOUT_STEREO,
	DCA_SPEAKER_LAYOUT_STEREO,
	DCA_SPEAKER_LAYOUT_STEREO,
	DCA_SPEAKER_LAYOUT_3_0,
	DCA_SPEAKER_LAYOUT_2_1,
	DCA_SPEAKER_LAYOUT_3_1,
	DCA_SPEAKER_LAYOUT_2_2,
	DCA_SPEAKER_LAYOUT_5POINT0
};

/*
 * The speakers in the order of the output planes: ff_dca_set_channel_layout's dca2wav_norm puts
 * L R C LFE1 Cs Ls Rs at WAVE positions 0 1 2 3 8 9 10.
 */
static const uint8_t output_speaker_order[DCA_SPEAKER_COUNT] = {
	DCA_SPEAKER_L, DCA_SPEAKER_R, DCA_SPEAKER_C, DCA_SPEAKER_LFE1,
	DCA_SPEAKER_Cs, DCA_SPEAKER_Ls, DCA_SPEAKER_Rs
};

static const uint8_t block_code_nbits[7] = {
	7, 10, 12, 13, 15, 17, 19
};

/* ---------------------------------------------------------------- dcamath.h */

static inline int32_t norm__(int64_t a, int bits)
{
	if (bits > 0)
		return (int32_t)((a + (INT64_C(1) << (bits - 1))) >> bits);
	else
		return (int32_t)a;
}

static inline int32_t mul__(int32_t a, int32_t b, int bits)
{
	return norm__((int64_t)a * b, bits);
}

static inline int32_t norm13(int64_t a) { return norm__(a, 13); }
static inline int32_t mul17(int32_t a, int32_t b) { return mul__(a, b, 17); }
static inline int32_t mul23(int32_t a, int32_t b) { return mul__(a, b, 23); }

/* av_clip_intp2(a, 23) */
static inline int32_t clip23(int32_t a)
{
	if (((uint32_t)a + (1U << 23)) & ~((2U << 23) - 1))
		return (a >> 31) ^ ((1 << 23) - 1);
	else
		return a;
}

static inline int av_log2(uint32_t v)
{
	return 31 - __builtin_clz(v | 1);
}

static inline int popcount(uint32_t v)
{
	return __builtin_popcount(v);
}

/* ---------------------------------------------------------------- bit reader */

struct GetBitContext {
	const uint8_t *buffer;
	int index;
	int size_in_bits;
};

static void init_get_bits8(GetBitContext *s, const uint8_t *buffer, int byte_size)
{
	s->buffer = buffer;
	s->index = 0;
	s->size_in_bits = byte_size * 8;
}

/* The next n bits (1..32), not consumed; zeros past the end. The buffer has DCA_INPUT_PADDING. */
static inline uint32_t show_bits(const GetBitContext *s, int n)
{
	if (s->index >= s->size_in_bits)
		return 0;
	const uint8_t *p = s->buffer + (s->index >> 3);
	uint64_t v = ((uint64_t)p[0] << 32) | ((uint64_t)p[1] << 24) | ((uint64_t)p[2] << 16) |
	             ((uint64_t)p[3] << 8) | (uint64_t)p[4];
	return (uint32_t)((v >> (40 - (s->index & 7) - n)) & ((UINT64_C(1) << n) - 1));
}

static inline uint32_t get_bits(GetBitContext *s, int n)
{
	if (!n)
		return 0;
	uint32_t v = show_bits(s, n);
	s->index += n;
	return v;
}

static inline uint32_t get_bits_long(GetBitContext *s, int n) { return get_bits(s, n); }
static inline uint32_t get_bits1(GetBitContext *s) { return get_bits(s, 1); }
static inline void skip_bits(GetBitContext *s, int n) { s->index += n; }
static inline int get_bits_left(const GetBitContext *s) { return s->size_in_bits - s->index; }

static inline int32_t get_sbits(GetBitContext *s, int n)
{
	uint32_t v = get_bits(s, n);
	return (int32_t)(v << (32 - n)) >> (32 - n);
}

/* ---------------------------------------------------------------- VLC (dcahuff.c) */

/* len > 0: a symbol of len bits. len < 0: -len more bits index the subtable at vlc_pool[sym]. */
struct VLCElem {
	int16_t sym;
	int16_t len;
};

struct VLC {
	const VLCElem *table;
	int bits;
};

/* FFmpeg's own table, which also holds DTS Express's code books, has 30218 entries. */
#define VLC_POOL_SIZE 32768
static VLCElem vlc_pool[VLC_POOL_SIZE];
static int vlc_pool_used;
static int vlc_incomplete;	// code books that are not a complete prefix code, or did not fit
static int vlc_ready;

static VLC ff_dca_vlc_bit_allocation[5];
static VLC ff_dca_vlc_transition_mode[4];
static VLC ff_dca_vlc_scale_factor[5];
static VLC ff_dca_vlc_quant_index[DCA_CODE_BOOKS][7];

static VLCElem *vlc_alloc(int bits)
{
	int n = 1 << bits;
	if (vlc_pool_used + n > VLC_POOL_SIZE)
		return NULL;
	VLCElem *t = vlc_pool + vlc_pool_used;
	vlc_pool_used += n;
	for (int i = 0; i < n; i++) {
		t[i].sym = -1;
		t[i].len = 0;
	}
	return t;
}

/*
 * ff_vlc_init_from_lengths over {symbol, length} rows: the codes are given out in row order,
 * each the one after the last at its own length; a symbol is the row's plus `offset`.
 */
static void vlc_init(VLC *vlc, int nb_bits, int nb_codes, const uint8_t (*src)[2], int offset)
{
	uint32_t codes[129];
	uint64_t code = 0;

	for (int i = 0; i < nb_codes; i++) {
		codes[i] = (uint32_t)code;
		code += UINT64_C(1) << (32 - src[i][1]);
	}
	if (code != (UINT64_C(1) << 32))
		vlc_incomplete++;

	VLCElem *table = vlc_alloc(nb_bits);
	vlc->table = table;
	vlc->bits = nb_bits;
	if (!table) {
		vlc_incomplete++;
		return;
	}

	for (int i = 0; i < nb_codes; i++) {
		int len = src[i][1];
		uint32_t prefix = codes[i] >> (32 - nb_bits);
		if (len <= nb_bits) {
			for (int k = 0; k < (1 << (nb_bits - len)); k++) {
				table[prefix + k].sym = (int16_t)(src[i][0] + offset);
				table[prefix + k].len = (int16_t)len;
			}
			continue;
		}
		if (!table[prefix].len) {
			int subbits = 0;
			for (int j = 0; j < nb_codes; j++)
				if (src[j][1] > nb_bits && (codes[j] >> (32 - nb_bits)) == prefix)
					subbits = FFMAX(subbits, src[j][1] - nb_bits);
			VLCElem *sub = vlc_alloc(subbits);
			if (!sub) {
				vlc_incomplete++;
				return;
			}
			table[prefix].sym = (int16_t)(sub - vlc_pool);
			table[prefix].len = (int16_t)-subbits;
		}
		int subbits = -table[prefix].len;
		VLCElem *sub = vlc_pool + table[prefix].sym;
		uint32_t at = (uint32_t)(codes[i] << nb_bits) >> (32 - subbits);
		for (int k = 0; k < (1 << (subbits - (len - nb_bits))); k++) {
			sub[at + k].sym = (int16_t)(src[i][0] + offset);
			sub[at + k].len = (int16_t)(len - nb_bits);
		}
	}
}

/* ff_dca_init_vlcs, the core's code books. */
static void dca_init_vlcs(void)
{
	const uint8_t (*src_table)[2] = ff_dca_vlc_src_tables;

	if (vlc_ready)
		return;
	vlc_ready = 1;

	for (unsigned i = 0; i < DCA_CODE_BOOKS; i++) {
		for (unsigned j = 0; j < ff_dca_quant_index_group_size[i]; j++) {
			vlc_init(&ff_dca_vlc_quant_index[i][j], bitalloc_maxbits[i][j],
			         ff_dca_bitalloc_sizes[i], src_table, ff_dca_bitalloc_offsets[i]);
			src_table += ff_dca_bitalloc_sizes[i];
		}
	}

	for (unsigned i = 0; i < FF_ARRAY_ELEMS(ff_dca_vlc_bit_allocation); i++) {
		vlc_init(&ff_dca_vlc_bit_allocation[i], bitalloc_12_vlc_bits[i], 12, src_table, 1);
		src_table += 12;
	}

	for (unsigned i = 0; i < FF_ARRAY_ELEMS(ff_dca_vlc_scale_factor); i++) {
		vlc_init(&ff_dca_vlc_scale_factor[i], DCA_SCALES_VLC_BITS, 129, src_table, -64);
		src_table += 129;
	}

	for (unsigned i = 0; i < FF_ARRAY_ELEMS(ff_dca_vlc_transition_mode); i++) {
		vlc_init(&ff_dca_vlc_transition_mode[i], DCA_TMODE_VLC_BITS, 4, src_table, 0);
		src_table += 4;
	}
}

static inline int dca_get_vlc(GetBitContext *s, const VLC *vlc)
{
	const VLCElem *e = &vlc->table[show_bits(s, vlc->bits)];
	if (e->len < 0) {
		s->index += vlc->bits;
		e = &vlc_pool[e->sym + show_bits(s, -e->len)];
	}
	s->index += e->len;
	return e->sym;
}

static void get_array(GetBitContext *s, int32_t *array, int size, int n)
{
	for (int i = 0; i < size; i++)
		array[i] = get_sbits(s, n);
}

/* ---------------------------------------------------------------- the decoder */

struct DCACoreFrameHeader {
	uint8_t     normal_frame;       ///< Frame type
	uint8_t     deficit_samples;    ///< Deficit sample count
	uint8_t     crc_present;        ///< CRC present flag
	uint8_t     npcmblocks;         ///< Number of PCM sample blocks
	uint16_t    frame_size;         ///< Primary frame byte size
	uint8_t     audio_mode;         ///< Audio channel arrangement
	uint8_t     sr_code;            ///< Core audio sampling frequency
	uint8_t     br_code;            ///< Transmission bit rate
	uint8_t     drc_present;        ///< Embedded dynamic range flag
	uint8_t     ts_present;         ///< Embedded time stamp flag
	uint8_t     aux_present;        ///< Auxiliary data flag
	uint8_t     hdcd_master;        ///< HDCD mastering flag
	uint8_t     ext_audio_type;     ///< Extension audio descriptor flag
	uint8_t     ext_audio_present;  ///< Extended coding flag
	uint8_t     sync_ssf;           ///< Audio sync word insertion flag
	uint8_t     lfe_present;        ///< Low frequency effects flag
	uint8_t     predictor_history;  ///< Predictor history flag switch
	uint8_t     filter_perfect;     ///< Multirate interpolator switch
	uint8_t     encoder_rev;        ///< Encoder software revision
	uint8_t     copy_hist;          ///< Copy history
	uint8_t     pcmr_code;          ///< Source PCM resolution
	uint8_t     sumdiff_front;      ///< Front sum/difference flag
	uint8_t     sumdiff_surround;   ///< Surround sum/difference flag
	uint8_t     dn_code;            ///< Dialog normalization / unspecified
};

struct DCADSPData {
	float hist1[512];
	float hist2[32];
	int offset;
};

struct DTSContext {
	GetBitContext   gb;

	// Bit stream header
	int     crc_present;        ///< CRC present flag
	int     npcmblocks;         ///< Number of PCM sample blocks
	int     frame_size;         ///< Primary frame byte size
	int     audio_mode;         ///< Audio channel arrangement
	int     sample_rate;        ///< Core audio sampling frequency
	int     bit_rate;           ///< Transmission bit rate
	int     drc_present;        ///< Embedded dynamic range flag
	int     ext_audio_type;     ///< Extension audio descriptor flag
	int     ext_audio_present;  ///< Extended coding flag
	int     sync_ssf;           ///< Audio sync word insertion flag
	int     lfe_present;        ///< Low frequency effects flag
	int     predictor_history;  ///< Predictor history flag switch
	int     filter_perfect;     ///< Multirate interpolator switch
	int     sumdiff_front;      ///< Front sum/difference flag
	int     sumdiff_surround;   ///< Surround sum/difference flag

	// Primary audio coding header
	int         nsubframes;     ///< Number of subframes
	int         nchannels;      ///< Number of primary audio channels
	int         ch_mask;        ///< Speaker layout mask (incl. LFE)
	int8_t      nsubbands[DCA_CHANNELS];                ///< Subband activity count
	int8_t      subband_vq_start[DCA_CHANNELS];         ///< High frequency VQ start subband
	int8_t      joint_intensity_index[DCA_CHANNELS];    ///< Joint intensity coding index
	int8_t      transition_mode_sel[DCA_CHANNELS];      ///< Transient mode code book
	int8_t      scale_factor_sel[DCA_CHANNELS];         ///< Scale factor code book
	int8_t      bit_allocation_sel[DCA_CHANNELS];       ///< Bit allocation quantizer select
	int8_t      quant_index_sel[DCA_CHANNELS][DCA_CODE_BOOKS];  ///< Quantization index codebook select
	int32_t     scale_factor_adj[DCA_CHANNELS][DCA_CODE_BOOKS]; ///< Scale factor adjustment

	// Primary audio coding side information
	int8_t      nsubsubframes[DCA_SUBFRAMES];   ///< Subsubframe count for each subframe
	int8_t      prediction_mode[DCA_CHANNELS][DCA_SUBBANDS];            ///< Prediction mode
	int16_t     prediction_vq_index[DCA_CHANNELS][DCA_SUBBANDS];        ///< Prediction coefficients VQ address
	int8_t      bit_allocation[DCA_CHANNELS][DCA_SUBBANDS];             ///< Bit allocation index
	int8_t      transition_mode[DCA_SUBFRAMES][DCA_CHANNELS][DCA_SUBBANDS]; ///< Transition mode
	int32_t     scale_factors[DCA_CHANNELS][DCA_SUBBANDS][2];           ///< Scale factors (2x for transients)
	int8_t      joint_scale_sel[DCA_CHANNELS];                          ///< Joint subband codebook select
	int32_t     joint_scale_factors[DCA_CHANNELS][DCA_SUBBANDS];        ///< Scale factors for joint subband coding

	// Sample buffers: a band's samples follow its DCA_ADPCM_COEFFS of history
	int32_t     subband_buffer[DCA_CHANNELS][DCA_SUBBANDS][DCA_ADPCM_COEFFS + DCA_PCMBLOCKS_MAX];
	int32_t     lfe_samples[DCA_LFE_HISTORY + DCA_PCMBLOCKS_MAX / 2];   ///< Decimated LFE samples

	DCADSPData  dcadsp_data[DCA_CHANNELS];      ///< FIR history buffers

	// PCM output, by speaker
	float       output_samples[DCA_SPEAKER_COUNT][DCA_FRAME_SAMPLES_MAX];
	int         npcmsamples;    ///< Number of PCM samples per channel

	// The frame as the 16-bit big-endian stream
	uint8_t     buffer[DCA_FRAME_BYTES_MAX + DCA_INPUT_PADDING];
};

static inline int32_t *subband_samples(DTSContext *s, int ch, int band)
{
	return s->subband_buffer[ch][band] + DCA_ADPCM_COEFFS;
}

/* dca.c ff_dca_parse_core_frame_header */
static int parse_core_frame_header(DCACoreFrameHeader *h, GetBitContext *gb)
{
	if (get_bits_long(gb, 32) != DCA_SYNCWORD_CORE_BE)
		return -1;

	h->normal_frame = get_bits1(gb);
	h->deficit_samples = get_bits(gb, 5) + 1;
	if (h->deficit_samples != DCA_PCMBLOCK_SAMPLES)
		return -2;

	h->crc_present = get_bits1(gb);
	h->npcmblocks = get_bits(gb, 7) + 1;
	if (h->npcmblocks & (DCA_SUBBAND_SAMPLES - 1))
		return -3;

	h->frame_size = get_bits(gb, 14) + 1;
	if (h->frame_size < 96)
		return -4;

	h->audio_mode = get_bits(gb, 6);
	if (h->audio_mode >= DCA_AMODE_COUNT)
		return -5;

	h->sr_code = get_bits(gb, 4);
	if (!ff_dca_sample_rates[h->sr_code])
		return -6;

	h->br_code = get_bits(gb, 5);
	if (get_bits1(gb))
		return -7;

	h->drc_present = get_bits1(gb);
	h->ts_present = get_bits1(gb);
	h->aux_present = get_bits1(gb);
	h->hdcd_master = get_bits1(gb);
	h->ext_audio_type = get_bits(gb, 3);
	h->ext_audio_present = get_bits1(gb);
	h->sync_ssf = get_bits1(gb);
	h->lfe_present = get_bits(gb, 2);
	if (h->lfe_present == DCA_LFE_FLAG_INVALID)
		return -8;

	h->predictor_history = get_bits1(gb);
	if (h->crc_present)
		skip_bits(gb, 16);
	h->filter_perfect = get_bits1(gb);
	h->encoder_rev = get_bits(gb, 4);
	h->copy_hist = get_bits(gb, 2);
	h->pcmr_code = get_bits(gb, 3);
	if (!ff_dca_bits_per_sample[h->pcmr_code])
		return -9;

	h->sumdiff_front = get_bits1(gb);
	h->sumdiff_surround = get_bits1(gb);
	h->dn_code = get_bits(gb, 4);
	return 0;
}

// 5.3.1 - Bit stream header
static int parse_frame_header(DTSContext *s)
{
	DCACoreFrameHeader h;
	memset(&h, 0, sizeof(h));
	int err = parse_core_frame_header(&h, &s->gb);

	if (err < 0)
		return err;

	s->crc_present          = h.crc_present;
	s->npcmblocks           = h.npcmblocks;
	s->frame_size           = h.frame_size;
	s->audio_mode           = h.audio_mode;
	s->sample_rate          = ff_dca_sample_rates[h.sr_code];
	s->bit_rate             = ff_dca_bit_rates[h.br_code];
	s->drc_present          = h.drc_present;
	s->ext_audio_type       = h.ext_audio_type;
	s->ext_audio_present    = h.ext_audio_present;
	s->sync_ssf             = h.sync_ssf;
	s->lfe_present          = h.lfe_present;
	s->predictor_history    = h.predictor_history;
	s->filter_perfect       = h.filter_perfect;
	s->sumdiff_front        = h.sumdiff_front;
	s->sumdiff_surround     = h.sumdiff_surround;

	return 0;
}

// 5.3.2 - Primary audio coding header
static int parse_coding_header(DTSContext *s)
{
	int n, ch;

	if (get_bits_left(&s->gb) < 0)
		return -1;

	// Number of subframes
	s->nsubframes = get_bits(&s->gb, 4) + 1;

	// Number of primary audio channels
	s->nchannels = get_bits(&s->gb, 3) + 1;
	if (s->nchannels != ff_dca_channels[s->audio_mode])
		return -1;

	s->ch_mask = audio_mode_ch_mask[s->audio_mode];

	// Add LFE channel if present
	if (s->lfe_present)
		s->ch_mask |= DCA_SPEAKER_MASK_LFE1;

	// Subband activity count
	for (ch = 0; ch < s->nchannels; ch++) {
		s->nsubbands[ch] = get_bits(&s->gb, 5) + 2;
		if (s->nsubbands[ch] > DCA_SUBBANDS)
			return -1;
	}

	// High frequency VQ start subband
	for (ch = 0; ch < s->nchannels; ch++)
		s->subband_vq_start[ch] = get_bits(&s->gb, 5) + 1;

	// Joint intensity coding index
	for (ch = 0; ch < s->nchannels; ch++) {
		n = get_bits(&s->gb, 3);
		if (n > s->nchannels)
			return -1;
		s->joint_intensity_index[ch] = n;
	}

	// Transient mode code book
	for (ch = 0; ch < s->nchannels; ch++)
		s->transition_mode_sel[ch] = get_bits(&s->gb, 2);

	// Scale factor code book
	for (ch = 0; ch < s->nchannels; ch++) {
		s->scale_factor_sel[ch] = get_bits(&s->gb, 3);
		if (s->scale_factor_sel[ch] == 7)
			return -1;
	}

	// Bit allocation quantizer select
	for (ch = 0; ch < s->nchannels; ch++) {
		s->bit_allocation_sel[ch] = get_bits(&s->gb, 3);
		if (s->bit_allocation_sel[ch] == 7)
			return -1;
	}

	// Quantization index codebook select
	for (n = 0; n < DCA_CODE_BOOKS; n++)
		for (ch = 0; ch < s->nchannels; ch++)
			s->quant_index_sel[ch][n] = get_bits(&s->gb, ff_dca_quant_index_sel_nbits[n]);

	// Scale factor adjustment index
	for (n = 0; n < DCA_CODE_BOOKS; n++)
		for (ch = 0; ch < s->nchannels; ch++)
			if (s->quant_index_sel[ch][n] < ff_dca_quant_index_group_size[n])
				s->scale_factor_adj[ch][n] = ff_dca_scale_factor_adj[get_bits(&s->gb, 2)];

	// Audio header CRC check word
	if (s->crc_present)
		skip_bits(&s->gb, 16);

	return 0;
}

static inline int parse_scale(DTSContext *s, int *scale_index, int sel)
{
	const uint32_t *scale_table;
	unsigned int scale_size;

	// Select the root square table
	if (sel > 5) {
		scale_table = ff_dca_scale_factor_quant7;
		scale_size = FF_ARRAY_ELEMS(ff_dca_scale_factor_quant7);
	} else {
		scale_table = ff_dca_scale_factor_quant6;
		scale_size = FF_ARRAY_ELEMS(ff_dca_scale_factor_quant6);
	}

	// If Huffman code was used, the difference of scales was encoded
	if (sel < 5)
		*scale_index += dca_get_vlc(&s->gb, &ff_dca_vlc_scale_factor[sel]);
	else
		*scale_index = get_bits(&s->gb, sel + 1);

	// Look up scale factor from the root square table
	if ((unsigned int)*scale_index >= scale_size)
		return -1;

	return scale_table[*scale_index];
}

static inline int parse_joint_scale(DTSContext *s, int sel)
{
	int scale_index;

	// Absolute value was encoded even when Huffman code was used
	if (sel < 5)
		scale_index = dca_get_vlc(&s->gb, &ff_dca_vlc_scale_factor[sel]);
	else
		scale_index = get_bits(&s->gb, sel + 1);

	// Bias by 64
	scale_index += 64;

	// Look up joint scale factor
	if ((unsigned int)scale_index >= FF_ARRAY_ELEMS(ff_dca_joint_scale_factors))
		return -1;

	return ff_dca_joint_scale_factors[scale_index];
}

// 5.4.1 - Primary audio coding side information
static int parse_subframe_header(DTSContext *s, int sf)
{
	int ch, band, ret;

	if (get_bits_left(&s->gb) < 0)
		return -1;

	// Subsubframe count
	s->nsubsubframes[sf] = get_bits(&s->gb, 2) + 1;

	// Partial subsubframe sample count
	skip_bits(&s->gb, 3);

	// Prediction mode
	for (ch = 0; ch < s->nchannels; ch++)
		for (band = 0; band < s->nsubbands[ch]; band++)
			s->prediction_mode[ch][band] = get_bits1(&s->gb);

	// Prediction coefficients VQ address
	for (ch = 0; ch < s->nchannels; ch++)
		for (band = 0; band < s->nsubbands[ch]; band++)
			if (s->prediction_mode[ch][band])
				s->prediction_vq_index[ch][band] = get_bits(&s->gb, 12);

	// Bit allocation index
	for (ch = 0; ch < s->nchannels; ch++) {
		int sel = s->bit_allocation_sel[ch];

		for (band = 0; band < s->subband_vq_start[ch]; band++) {
			int abits;

			if (sel < 5)
				abits = dca_get_vlc(&s->gb, &ff_dca_vlc_bit_allocation[sel]);
			else
				abits = get_bits(&s->gb, sel - 1);

			if (abits < 0 || abits > DCA_ABITS_MAX)
				return -1;

			s->bit_allocation[ch][band] = abits;
		}
	}

	// Transition mode
	for (ch = 0; ch < s->nchannels; ch++) {
		// Clear transition mode for all subbands
		memset(s->transition_mode[sf][ch], 0, sizeof(s->transition_mode[0][0]));

		// Transient possible only if more than one subsubframe
		if (s->nsubsubframes[sf] > 1) {
			int sel = s->transition_mode_sel[ch];
			for (band = 0; band < s->subband_vq_start[ch]; band++)
				if (s->bit_allocation[ch][band])
					s->transition_mode[sf][ch][band] = dca_get_vlc(&s->gb, &ff_dca_vlc_transition_mode[sel]);
		}
	}

	// Scale factors
	for (ch = 0; ch < s->nchannels; ch++) {
		int sel = s->scale_factor_sel[ch];
		int scale_index = 0;

		// Extract scales for subbands up to VQ
		for (band = 0; band < s->subband_vq_start[ch]; band++) {
			if (s->bit_allocation[ch][band]) {
				if ((ret = parse_scale(s, &scale_index, sel)) < 0)
					return ret;
				s->scale_factors[ch][band][0] = ret;
				if (s->transition_mode[sf][ch][band]) {
					if ((ret = parse_scale(s, &scale_index, sel)) < 0)
						return ret;
					s->scale_factors[ch][band][1] = ret;
				}
			} else {
				s->scale_factors[ch][band][0] = 0;
			}
		}

		// High frequency VQ subbands
		for (band = s->subband_vq_start[ch]; band < s->nsubbands[ch]; band++) {
			if ((ret = parse_scale(s, &scale_index, sel)) < 0)
				return ret;
			s->scale_factors[ch][band][0] = ret;
		}
	}

	// Joint subband codebook select
	for (ch = 0; ch < s->nchannels; ch++) {
		if (s->joint_intensity_index[ch]) {
			s->joint_scale_sel[ch] = get_bits(&s->gb, 3);
			if (s->joint_scale_sel[ch] == 7)
				return -1;
		}
	}

	// Scale factors for joint subband coding
	for (ch = 0; ch < s->nchannels; ch++) {
		int src_ch = s->joint_intensity_index[ch] - 1;
		if (src_ch >= 0) {
			int sel = s->joint_scale_sel[ch];
			for (band = s->nsubbands[ch]; band < s->nsubbands[src_ch]; band++) {
				if ((ret = parse_joint_scale(s, sel)) < 0)
					return ret;
				s->joint_scale_factors[ch][band] = ret;
			}
		}
	}

	// Dynamic range coefficient
	if (s->drc_present)
		skip_bits(&s->gb, 8);

	// Side information CRC check word
	if (s->crc_present)
		skip_bits(&s->gb, 16);

	return 0;
}

static inline int decode_blockcodes(int code1, int code2, int levels, int32_t *audio)
{
	int offset = (levels - 1) / 2;
	int n, div;

	for (n = 0; n < DCA_SUBBAND_SAMPLES / 2; n++) {
		div = code1 / levels;
		audio[n] = code1 - div * levels - offset;
		code1 = div;
	}
	for (; n < DCA_SUBBAND_SAMPLES; n++) {
		div = code2 / levels;
		audio[n] = code2 - div * levels - offset;
		code2 = div;
	}

	return code1 | code2;
}

static inline int parse_block_codes(DTSContext *s, int32_t *audio, int abits)
{
	// Extract block code indices from the bit stream
	int code1 = get_bits(&s->gb, block_code_nbits[abits - 1]);
	int code2 = get_bits(&s->gb, block_code_nbits[abits - 1]);
	int levels = ff_dca_quant_levels[abits];

	// Look up samples from the block code book
	if (decode_blockcodes(code1, code2, levels, audio))
		return -1;

	return 0;
}

static inline int parse_huffman_codes(DTSContext *s, int32_t *audio, int abits, int sel)
{
	// Extract Huffman codes from the bit stream
	for (int i = 0; i < DCA_SUBBAND_SAMPLES; i++)
		audio[i] = dca_get_vlc(&s->gb, &ff_dca_vlc_quant_index[abits - 1][sel]);

	return 1;
}

static inline int extract_audio(DTSContext *s, int32_t *audio, int abits, int ch)
{
	if (abits == 0) {
		// No bits allocated
		memset(audio, 0, DCA_SUBBAND_SAMPLES * sizeof(*audio));
		return 0;
	}

	if (abits <= DCA_CODE_BOOKS) {
		int sel = s->quant_index_sel[ch][abits - 1];
		if (sel < ff_dca_quant_index_group_size[abits - 1]) {
			// Huffman codes
			return parse_huffman_codes(s, audio, abits, sel);
		}
		if (abits <= 7) {
			// Block codes
			return parse_block_codes(s, audio, abits);
		}
	}

	// No further encoding
	get_array(&s->gb, audio, DCA_SUBBAND_SAMPLES, abits - 3);
	return 0;
}

/* dca_core.h ff_dca_core_dequantize, not residual */
static inline void core_dequantize(int32_t *output, const int32_t *input,
                                   int32_t step_size, int32_t scale, int len)
{
	// Account for quantizer step size
	int64_t step_scale = (int64_t)step_size * scale;
	int n, shift = 0;

	// Limit scale factor resolution to 22 bits
	if (step_scale > (1 << 23)) {
		shift = av_log2((uint32_t)(step_scale >> 23)) + 1;
		step_scale >>= shift;
	}

	// Scale the samples
	for (n = 0; n < len; n++)
		output[n] = clip23(norm__(input[n] * step_scale, 22 - shift));
}

/* dcaadpcm.h */
static inline int64_t dcaadpcm_predict(int pred_vq_index, const int32_t *input)
{
	const int16_t *coeff = ff_dca_adpcm_vb[pred_vq_index];
	int64_t pred = 0;
	for (int i = 0; i < DCA_ADPCM_COEFFS; i++)
		pred += (int64_t)input[DCA_ADPCM_COEFFS - 1 - i] * coeff[i];

	return clip23(norm13(pred));
}

static inline void inverse_adpcm(DTSContext *s, int ch, int sb_start, int sb_end, int ofs, int len)
{
	for (int i = sb_start; i < sb_end; i++) {
		if (s->prediction_mode[ch][i]) {
			const int pred_id = s->prediction_vq_index[ch][i];
			int32_t *ptr = subband_samples(s, ch, i) + ofs;
			for (int j = 0; j < len; j++) {
				int32_t x = (int32_t)dcaadpcm_predict(pred_id, ptr + j - DCA_ADPCM_COEFFS);
				ptr[j] = clip23(ptr[j] + x);
			}
		}
	}
}

// 5.5 - Primary audio data arrays
static int parse_subframe_audio(DTSContext *s, int sf, int *sub_pos, int *lfe_pos)
{
	int32_t audio[16], scale;
	int n, ssf, ofs, ch, band;

	// Check number of subband samples in this subframe
	int nsamples = s->nsubsubframes[sf] * DCA_SUBBAND_SAMPLES;
	if (*sub_pos + nsamples > s->npcmblocks)
		return -1;

	if (get_bits_left(&s->gb) < 0)
		return -1;

	// VQ encoded subbands
	for (ch = 0; ch < s->nchannels; ch++) {
		int32_t vq_index[DCA_SUBBANDS];

		for (band = s->subband_vq_start[ch]; band < s->nsubbands[ch]; band++)
			// Extract the VQ address from the bit stream
			vq_index[band] = get_bits(&s->gb, 10);

		// dcadsp.c decode_hf_c
		for (band = s->subband_vq_start[ch]; band < s->nsubbands[ch]; band++) {
			const int8_t *coeff = ff_dca_high_freq_vq[vq_index[band]];
			int32_t hf_scale = s->scale_factors[ch][band][0];
			int32_t *dst = subband_samples(s, ch, band) + *sub_pos;
			for (n = 0; n < nsamples; n++)
				dst[n] = clip23((coeff[n] * hf_scale + (1 << 3)) >> 4);
		}
	}

	// Low frequency effect data
	if (s->lfe_present) {
		unsigned int index;

		// Determine number of LFE samples in this subframe
		int nlfesamples = 2 * s->lfe_present * s->nsubsubframes[sf];

		// Extract LFE samples from the bit stream
		get_array(&s->gb, audio, nlfesamples, 8);

		// Extract scale factor index from the bit stream
		index = get_bits(&s->gb, 8);
		if (index >= FF_ARRAY_ELEMS(ff_dca_scale_factor_quant7))
			return -1;

		// Look up the 7-bit root square quantization table
		scale = ff_dca_scale_factor_quant7[index];

		// Account for quantizer step size which is 0.035
		scale = mul23(4697620 /* 0.035 * (1 << 27) */, scale);

		// Scale and take the LFE samples
		for (n = 0, ofs = *lfe_pos; n < nlfesamples; n++, ofs++)
			s->lfe_samples[ofs] = clip23(audio[n] * scale >> 4);

		// Advance LFE sample pointer for the next subframe
		*lfe_pos = ofs;
	}

	// Audio data
	for (ssf = 0, ofs = *sub_pos; ssf < s->nsubsubframes[sf]; ssf++) {
		for (ch = 0; ch < s->nchannels; ch++) {
			if (get_bits_left(&s->gb) < 0)
				return -1;

			// Not high frequency VQ subbands
			for (band = 0; band < s->subband_vq_start[ch]; band++) {
				int ret, trans_ssf, abits = s->bit_allocation[ch][band];
				int32_t step_size;

				// Extract bits from the bit stream
				if ((ret = extract_audio(s, audio, abits, ch)) < 0)
					return ret;

				// Select quantization step size table and look up
				// quantization step size
				if (s->bit_rate == 3)
					step_size = ff_dca_lossless_quant[abits];
				else
					step_size = ff_dca_lossy_quant[abits];

				// Identify transient location
				trans_ssf = s->transition_mode[sf][ch][band];

				// Determine proper scale factor
				if (trans_ssf == 0 || ssf < trans_ssf)
					scale = s->scale_factors[ch][band][0];
				else
					scale = s->scale_factors[ch][band][1];

				// Adjust scale factor when SEL indicates Huffman code
				if (ret > 0) {
					int64_t adj = s->scale_factor_adj[ch][abits - 1];
					scale = clip23((int32_t)(adj * scale >> 22));
				}

				core_dequantize(subband_samples(s, ch, band) + ofs,
				                audio, step_size, scale, DCA_SUBBAND_SAMPLES);
			}
		}

		// DSYNC
		if ((ssf == s->nsubsubframes[sf] - 1 || s->sync_ssf) && get_bits(&s->gb, 16) != 0xffff)
			return -1;

		ofs += DCA_SUBBAND_SAMPLES;
	}

	// Inverse ADPCM
	for (ch = 0; ch < s->nchannels; ch++)
		inverse_adpcm(s, ch, 0, s->nsubbands[ch], *sub_pos, nsamples);

	// Joint subband coding (dcadsp.c decode_joint_c)
	for (ch = 0; ch < s->nchannels; ch++) {
		int src_ch = s->joint_intensity_index[ch] - 1;
		if (src_ch >= 0) {
			for (band = s->nsubbands[ch]; band < s->nsubbands[src_ch]; band++) {
				int32_t joint_scale = s->joint_scale_factors[ch][band];
				const int32_t *src = subband_samples(s, src_ch, band) + *sub_pos;
				int32_t *dst = subband_samples(s, ch, band) + *sub_pos;
				for (n = 0; n < nsamples; n++)
					dst[n] = clip23(mul17(src[n], joint_scale));
			}
		}
	}

	// Advance subband sample pointer for the next subframe
	*sub_pos = ofs;
	return 0;
}

static void erase_adpcm_history(DTSContext *s)
{
	// Erase ADPCM history from previous frame if
	// predictor history switch was disabled
	for (int ch = 0; ch < DCA_CHANNELS; ch++)
		for (int band = 0; band < DCA_SUBBANDS; band++)
			memset(s->subband_buffer[ch][band], 0, DCA_ADPCM_COEFFS * sizeof(int32_t));
}

static int parse_frame_data(DTSContext *s)
{
	int sf, ch, ret, band, sub_pos, lfe_pos;

	if ((ret = parse_coding_header(s)) < 0)
		return ret;

	for (sf = 0, sub_pos = 0, lfe_pos = DCA_LFE_HISTORY; sf < s->nsubframes; sf++) {
		if ((ret = parse_subframe_header(s, sf)) < 0)
			return ret;
		if ((ret = parse_subframe_audio(s, sf, &sub_pos, &lfe_pos)) < 0)
			return ret;
	}

	for (ch = 0; ch < s->nchannels; ch++) {
		// Determine number of active subbands for this channel
		int nsubbands = s->nsubbands[ch];
		if (s->joint_intensity_index[ch])
			nsubbands = FFMAX(nsubbands, s->nsubbands[s->joint_intensity_index[ch] - 1]);

		// Update history for ADPCM
		for (band = 0; band < nsubbands; band++) {
			int32_t *samples = s->subband_buffer[ch][band];
			memcpy(samples, samples + s->npcmblocks, DCA_ADPCM_COEFFS * sizeof(int32_t));
		}

		// Clear inactive subbands
		for (; band < DCA_SUBBANDS; band++) {
			int32_t *samples = s->subband_buffer[ch][band];
			memset(samples, 0, (DCA_ADPCM_COEFFS + s->npcmblocks) * sizeof(int32_t));
		}
	}

	return 0;
}

/* ff_dca_core_parse, as with core_only: the optional information after the audio is not read. */
static int core_parse(DTSContext *s, const uint8_t *data, int size)
{
	int ret;

	init_get_bits8(&s->gb, data, size);

	if ((ret = parse_frame_header(s)) < 0)
		return ret;
	if (!s->predictor_history)
		erase_adpcm_history(s);
	if ((ret = parse_frame_data(s)) < 0)
		return ret;

	return 0;
}

/* ---------------------------------------------------------------- the filter bank */

/*
 * The 32-point inverse MDCT, 32 coefficients to 32 samples, by its definition
 * (tx_template.c ff_tx_mdct_naive_inv, scale 1).
 */
static double imdct32_cos[32][32];
static int imdct32_ready;

static void imdct32_init(void)
{
	const int len = 16, len2 = 32;
	const double phase = M_PI / (4.0 * len2);

	if (imdct32_ready)
		return;
	imdct32_ready = 1;

	for (int i = 0; i < len; i++) {
		double i_d = phase * (4 * len - 2 * i - 1);
		double i_u = phase * (3 * len2 + 2 * i + 1);
		for (int j = 0; j < len2; j++) {
			double a = (2 * j + 1);
			imdct32_cos[i][j] = cos(a * i_d);
			imdct32_cos[i + len][j] = -cos(a * i_u);
		}
	}
}

static void imdct32(float *dst, const float *src)
{
	for (int i = 0; i < 32; i++) {
		const double *c = imdct32_cos[i];
		double sum = 0.0;
		for (int j = 0; j < 32; j++)
			sum += c[j] * src[j];
		dst[i] = (float)sum;
	}
}

/* synth_filter.c synth_filter_float */
static void synth_filter_float(float *synth_buf_ptr, int *synth_buf_offset,
                               float synth_buf2[32], const float window[512],
                               float out[32], const float in[32], float scale)
{
	float *synth_buf = synth_buf_ptr + *synth_buf_offset;
	int i, j;

	imdct32(synth_buf, in);

	for (i = 0; i < 16; i++) {
		float a = synth_buf2[i     ];
		float b = synth_buf2[i + 16];
		float c = 0;
		float d = 0;
		for (j = 0; j < 512 - *synth_buf_offset; j += 64) {
			a += window[i + j     ] * (-synth_buf[15 - i + j      ]);
			b += window[i + j + 16] * ( synth_buf[     i + j      ]);
			c += window[i + j + 32] * ( synth_buf[16 + i + j      ]);
			d += window[i + j + 48] * ( synth_buf[31 - i + j      ]);
		}
		for (     ; j < 512; j += 64) {
			a += window[i + j     ] * (-synth_buf[15 - i + j - 512]);
			b += window[i + j + 16] * ( synth_buf[     i + j - 512]);
			c += window[i + j + 32] * ( synth_buf[16 + i + j - 512]);
			d += window[i + j + 48] * ( synth_buf[31 - i + j - 512]);
		}
		out[i     ] = a * scale;
		out[i + 16] = b * scale;
		synth_buf2[i     ] = c;
		synth_buf2[i + 16] = d;
	}

	*synth_buf_offset = (*synth_buf_offset - 32) & 511;
}

/* dcadsp.c sub_qmf32_float_c */
static void sub_qmf32_float(DTSContext *s, int ch, float *pcm_samples,
                            const float *filter_coeff, int npcmblocks, float scale)
{
	DCADSPData *dsp = &s->dcadsp_data[ch];
	float input[32];

	for (int j = 0; j < npcmblocks; j++) {
		// Load in one sample from each subband
		for (int i = 0; i < 32; i++) {
			if ((i - 1) & 2)
				input[i] = -subband_samples(s, ch, i)[j];
			else
				input[i] =  subband_samples(s, ch, i)[j];
		}

		// One subband sample generates 32 interpolated ones
		synth_filter_float(dsp->hist1, &dsp->offset, dsp->hist2, filter_coeff,
		                   pcm_samples, input, scale);
		pcm_samples += 32;
	}
}

/* dcadsp.c lfe_fir_float_c */
static void lfe_fir_float(float *pcm_samples, const int32_t *lfe_samples,
                          const float *filter_coeff, int npcmblocks, int dec_select)
{
	// Select decimation factor
	int factor = 64 << dec_select;
	int ncoeffs = 8 >> dec_select;
	int nlfesamples = npcmblocks >> (dec_select + 1);
	int i, j, k;

	for (i = 0; i < nlfesamples; i++) {
		// One decimated sample generates 64 or 128 interpolated ones
		for (j = 0; j < factor / 2; j++) {
			float a = 0;
			float b = 0;

			for (k = 0; k < ncoeffs; k++) {
				a += filter_coeff[      j * ncoeffs + k] * lfe_samples[-k];
				b += filter_coeff[255 - j * ncoeffs - k] * lfe_samples[-k];
			}

			pcm_samples[             j] = a;
			pcm_samples[factor / 2 + j] = b;
		}

		lfe_samples++;
		pcm_samples += factor;
	}
}

/* float_dsp.c butterflies_float_c */
static void butterflies_float(float *v1, float *v2, int len)
{
	for (int i = 0; i < len; i++) {
		float t = v1[i] - v2[i];
		v1[i] += v2[i];
		v2[i] = t;
	}
}

/* dca_core.c filter_frame_float, without X96 and the extension channels */
static void filter_frame_float(DTSContext *s)
{
	int n, ch, nsamples;
	const float *filter_coeff;

	s->npcmsamples = nsamples = s->npcmblocks * DCA_PCMBLOCK_SAMPLES;

	// Select filter
	if (s->filter_perfect)
		filter_coeff = ff_dca_fir_32bands_perfect;
	else
		filter_coeff = ff_dca_fir_32bands_nonperfect;

	// Filter primary channels
	for (ch = 0; ch < s->nchannels; ch++) {
		// Map this primary channel to speaker
		int spkr = prm_ch_to_spkr_map[s->audio_mode][ch];

		// Filter bank reconstruction
		sub_qmf32_float(s, ch, s->output_samples[spkr], filter_coeff, s->npcmblocks,
		                1.0f / (1 << 17));
	}

	// Filter LFE channel
	if (s->lfe_present) {
		int dec_select = (s->lfe_present == DCA_LFE_FLAG_128);
		float *samples = s->output_samples[DCA_SPEAKER_LFE1];
		int nlfesamples = s->npcmblocks >> (dec_select + 1);

		// Select filter
		if (dec_select)
			filter_coeff = ff_dca_lfe_fir_128;
		else
			filter_coeff = ff_dca_lfe_fir_64;

		// Interpolate LFE channel
		lfe_fir_float(samples, s->lfe_samples + DCA_LFE_HISTORY, filter_coeff, s->npcmblocks, dec_select);

		// Update LFE history
		for (n = DCA_LFE_HISTORY - 1; n >= 0; n--)
			s->lfe_samples[n] = s->lfe_samples[nlfesamples + n];
	}

	// Front sum/difference decoding
	if ((s->sumdiff_front && s->audio_mode > DCA_AMODE_MONO)
	    || s->audio_mode == DCA_AMODE_STEREO_SUMDIFF) {
		butterflies_float(s->output_samples[DCA_SPEAKER_L], s->output_samples[DCA_SPEAKER_R], nsamples);
	}

	// Surround sum/difference decoding
	if (s->sumdiff_surround && s->audio_mode >= DCA_AMODE_2F2R) {
		butterflies_float(s->output_samples[DCA_SPEAKER_Ls], s->output_samples[DCA_SPEAKER_Rs], nsamples);
	}
}

/* ---------------------------------------------------------------- boundary */

/*
 * dca.c avpriv_dca_convert_bitstream for the two 16-bit forms: `count` bytes at `src` as the
 * big-endian stream at `dst`. 0 for another sync word.
 */
static int convert_bitstream(const uint8_t *src, int count, uint8_t *dst)
{
	uint32_t mrk = ((uint32_t)src[0] << 24) | (src[1] << 16) | (src[2] << 8) | src[3];

	if (mrk == DCA_SYNCWORD_CORE_BE) {
		memcpy(dst, src, count);
		return 1;
	}
	if (mrk == DCA_SYNCWORD_CORE_LE) {
		for (int i = 0; i + 1 < count; i += 2) {
			dst[i] = src[i + 1];
			dst[i + 1] = src[i];
		}
		return 1;
	}
	return 0;
}

extern "C" {

DTSContext *dts_open(void)
{
	dca_init_vlcs();
	imdct32_init();
	return (DTSContext *)calloc(1, sizeof(DTSContext));
}

void dts_close(DTSContext *s)
{
	free(s);
}

/* ff_dca_core_flush */
void dts_reset(DTSContext *s)
{
	erase_adpcm_history(s);
	memset(s->lfe_samples, 0, DCA_LFE_HISTORY * sizeof(int32_t));
	memset(s->dcadsp_data, 0, sizeof(s->dcadsp_data));
}

int dts_probe_frame(const uint8_t *data, int size, int *info)
{
	uint8_t head[16 + DCA_INPUT_PADDING];
	DCACoreFrameHeader h;
	GetBitContext gb;

	if (size < 16)
		return 0;
	memset(head, 0, sizeof(head));
	if (!convert_bitstream(data, 16, head))
		return 0;
	init_get_bits8(&gb, head, 16);
	memset(&h, 0, sizeof(h));
	if (parse_core_frame_header(&h, &gb) < 0)
		return 0;

	info[0] = h.frame_size;
	info[1] = ff_dca_sample_rates[h.sr_code];
	info[2] = ff_dca_channels[h.audio_mode] + !!h.lfe_present;
	info[3] = ff_dca_bit_rates[h.br_code];
	info[4] = h.npcmblocks * DCA_PCMBLOCK_SAMPLES;
	return 1;
}

int dts_decode_frame(DTSContext *s, const uint8_t *data, int size, int *info)
{
	info[0] = info[1] = info[2] = info[3] = info[4] = 0;

	if (size < 16)
		return 0;
	if (size > DCA_FRAME_BYTES_MAX)
		size = DCA_FRAME_BYTES_MAX;
	if (!convert_bitstream(data, size, s->buffer))
		return 0;
	memset(s->buffer + size, 0, DCA_INPUT_PADDING);

	int ret = core_parse(s, s->buffer, size);
	if (ret < 0) {
		// The header parsed when the audio did not: the frame can still be stepped over.
		init_get_bits8(&s->gb, s->buffer, size);
		DCACoreFrameHeader h;
		memset(&h, 0, sizeof(h));
		if (parse_core_frame_header(&h, &s->gb) == 0)
			info[0] = h.frame_size;
		return 0;
	}

	filter_frame_float(s);

	info[0] = s->frame_size;
	info[1] = s->sample_rate;
	info[2] = popcount(s->ch_mask);
	info[3] = s->npcmsamples;
	info[4] = s->ext_audio_present ? s->ext_audio_type + 1 : 0;
	return 1;
}

float *dts_channel_ptr(DTSContext *s, int ch)
{
	for (int i = 0; i < DCA_SPEAKER_COUNT; i++) {
		int spkr = output_speaker_order[i];
		if ((s->ch_mask & (1 << spkr)) && ch-- == 0)
			return s->output_samples[spkr];
	}
	return NULL;
}

int dts_self_test(void)
{
	dca_init_vlcs();
	return vlc_incomplete;
}

}
