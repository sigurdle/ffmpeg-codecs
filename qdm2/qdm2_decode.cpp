/*
	QDM2 (QDesign Music 2) audio decoder -- WASM C++.

	A direct port of FFmpeg's decoder (libavcodec, tag n8.1): qdm2.c, qdm2data.h (as
	qdm2_tables.h), qdm2_tablegen.h, the parts of vlc.c and get_bits.h it uses (the little-endian
	safe reader, ff_vlc_init_from_lengths), and tx_template.c's inverse RDFT.

	This file is part of a derivative of FFmpeg. FFmpeg is free software; you can
	redistribute it and/or modify it under the terms of the GNU Lesser General Public
	License as published by the Free Software Foundation; either version 2.1 of the
	License, or (at your option) any later version. It is distributed in the hope that it
	will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
	License for more details. Copyright is held by the FFmpeg authors: Ewald Snel, Benjamin
	Larsson, Alex Beregszaszi, Roberto Togni, Reimar Doeffinger and the others named in the
	files above.

	A QDM2 packet ("super block") is 16 sub-frames: an FFT part of decaying tones plus, in the
	first sub-frame of a packet, the subband data of an MPEG audio style synthesis filterbank
	(the one in ../mpegaudio/mpegaudio_dsp.h, FFmpeg shares it the same way). The packet size
	and the rest of the stream's shape come from the QDCA atom, so the container frames the
	packets: there is no sync word.

	Kept as FFmpeg has it, because FFmpeg is the oracle:
	  - process_subpacket_12 opens its reader on 0 bytes, so subbands 8 and up never read their
	    data, and synthfilt_build_sb_samples takes its dithering noise branch there.
	  - The VLCs are FFmpeg's two-level tables, and the reader is its clamped one: a code that
	    is not assigned, or a read past the end, lands where FFmpeg's lands.

	Different from FFmpeg:
	  - The RDFT's complex FFT is a radix-2 in double in place of lavu's split radix (the same
	    choice as the AAC port's IMDCT). Output is truncated to 16 bits, so a difference in the
	    last bits of a float moves a sample by at most one step.
	  - qdm2_reset clears the running state for a seek; FFmpeg has no flush for this codec.
*/

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>

#include "../mpegaudio/mpegaudio_dsp.h"		// mpa_dsp_init_tabs, mpa_synth_filter, ff_mpa_synth_window
#include "qdm2_tables.h"

#define WASM_EXPORT __attribute__((used)) __attribute__((visibility("default")))

#define MPA_MAX_CHANNELS 2
#define MPA_FRAME_SIZE 1152

#define QDM2_MAX_FRAME_SIZE 512

// Result is 8, 16 or 30
#define QDM2_SB_USED(sub_sampling) (((sub_sampling) >= 2) ? 30 : 8 << (sub_sampling))

#define FIX_NOISE_IDX(noise_idx) \
	if ((noise_idx) >= 3840) \
		(noise_idx) -= 3840;

#define SB_DITHERING_NOISE(sb, noise_idx) (noise_table[(noise_idx)++] * sb_noise_attenuation[(sb)])

#define SOFTCLIP_THRESHOLD 27600
#define HARDCLIP_THRESHOLD 35716

#define ARRAY_ELEMS(a) (sizeof(a) / sizeof((a)[0]))

static inline int FFMIN(int a, int b) { return a < b ? a : b; }
static inline int FFMAX(int a, int b) { return a > b ? a : b; }

///////////////////////////////////////////////////////////////////////////////
// get_bits.h, BITSTREAM_READER_LE, the safe reader

struct GetBitContext {
	const uint8_t* buffer;
	int index;
	int size_in_bits;
	int size_in_bits_plus8;
};

static inline uint32_t rl32(const uint8_t* p)
{
	return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static int init_get_bits8(GetBitContext* s, const uint8_t* buffer, int byte_size)
{
	s->index = 0;
	if (byte_size < 0 || byte_size > INT_MAX / 8 || !buffer) {
		s->buffer = NULL;
		s->size_in_bits = s->size_in_bits_plus8 = 0;
		return -1;
	}
	s->buffer = buffer;
	s->size_in_bits = byte_size * 8;
	s->size_in_bits_plus8 = s->size_in_bits + 8;
	return 0;
}

static inline int get_bits_left(const GetBitContext* s) { return s->size_in_bits - s->index; }
static inline int get_bits_count(const GetBitContext* s) { return s->index; }

static inline uint32_t update_cache(const GetBitContext* s, int index)
{
	return rl32(s->buffer + (index >> 3)) >> (index & 7);
}

static inline int skip_counter(const GetBitContext* s, int index, int num)
{
	return FFMIN(s->size_in_bits_plus8, index + num);
}

static inline unsigned get_bits(GetBitContext* s, int n)
{
	const unsigned value = update_cache(s, s->index) & ((1u << n) - 1);
	s->index = skip_counter(s, s->index, n);
	return value;
}

static inline unsigned get_bits1(GetBitContext* s)
{
	const int index = s->index;
	const unsigned value = (s->buffer[index >> 3] >> (index & 7)) & 1;
	if (s->index < s->size_in_bits_plus8) {
		s->index = index + 1;
	}
	return value;
}

static inline void skip_bits(GetBitContext* s, int n) { s->index = skip_counter(s, s->index, n); }

///////////////////////////////////////////////////////////////////////////////
// vlc.c: ff_vlc_init_from_lengths with VLC_INIT_STATIC_OVERLONG | VLC_INIT_LE, and get_vlc2

struct VLCElem { int16_t sym; int16_t len; };

struct VLC {
	int bits;
	VLCElem* table;
	int table_size;
};

struct VLCcode {
	uint8_t bits;
	int16_t symbol;
	uint32_t code;
};

static inline uint32_t bitswap_32(uint32_t x)
{
	x = ((x >> 1) & 0x55555555u) | ((x & 0x55555555u) << 1);
	x = ((x >> 2) & 0x33333333u) | ((x & 0x33333333u) << 2);
	x = ((x >> 4) & 0x0F0F0F0Fu) | ((x & 0x0F0F0F0Fu) << 4);
	return (x >> 24) | ((x >> 8) & 0xFF00u) | ((x << 8) & 0xFF0000u) | (x << 24);
}

static int build_table(VLC* vlc, int table_nb_bits, int nb_codes, VLCcode* codes)
{
	const int table_size = 1 << table_nb_bits;
	const int table_index = vlc->table_size;
	vlc->table_size += table_size;
	VLCElem* table = &vlc->table[table_index];

	for (int i = 0; i < nb_codes; i++) {
		int n = codes[i].bits;
		uint32_t code = codes[i].code;
		const int symbol = codes[i].symbol;
		if (n <= table_nb_bits) {
			int j = bitswap_32(code);
			const int nb = 1 << (table_nb_bits - n);
			const int inc = 1 << n;
			for (int k = 0; k < nb; k++) {
				table[j].len = n;
				table[j].sym = symbol;
				j += inc;
			}
		} else {
			n -= table_nb_bits;
			const uint32_t code_prefix = code >> (32 - table_nb_bits);
			int subtable_bits = n;
			codes[i].bits = n;
			codes[i].code = code << table_nb_bits;
			int k;
			for (k = i + 1; k < nb_codes; k++) {
				n = codes[k].bits - table_nb_bits;
				if (n <= 0) {
					break;
				}
				code = codes[k].code;
				if (code >> (32 - table_nb_bits) != code_prefix) {
					break;
				}
				codes[k].bits = n;
				codes[k].code = code << table_nb_bits;
				subtable_bits = FFMAX(subtable_bits, n);
			}
			subtable_bits = FFMIN(subtable_bits, table_nb_bits);
			const int j = bitswap_32(code_prefix) >> (32 - table_nb_bits);
			table[j].len = -subtable_bits;
			const int index = build_table(vlc, subtable_bits, k - i, codes + i);
			table[j].sym = index;
			i = k - 1;
		}
	}

	for (int i = 0; i < table_size; i++) {
		if (table[i].len == 0) {
			table[i].sym = -1;
		}
	}
	return table_index;
}

static VLC vlc_tab_level;
static VLC vlc_tab_diff;
static VLC vlc_tab_run;
static VLC fft_level_exp_alt_vlc;
static VLC fft_level_exp_vlc;
static VLC fft_stereo_exp_vlc;
static VLC fft_stereo_phase_vlc;
static VLC vlc_tab_tone_level_idx_hi1;
static VLC vlc_tab_tone_level_idx_mid;
static VLC vlc_tab_tone_level_idx_hi2;
static VLC vlc_tab_type30;
static VLC vlc_tab_type34;
static VLC vlc_tab_fft_tone_offset[5];

static VLCElem qdm2_table[3838];

/** The symbols are offset by -1, FFmpeg's escape: a table's symbol 0 reads as -1. */
static void build_vlc(VLC* vlc, int nb_bits, int nb_codes, unsigned* offset, const uint8_t tab[][2])
{
	VLCcode codes[128];
	uint64_t code = 0;
	int j = 0;
	for (int i = 0; i < nb_codes; i++) {
		const int len = (int8_t)tab[i][1];
		if (len > 0) {
			codes[j].bits = len;
			codes[j].symbol = (int)tab[i][0] - 1;
			codes[j++].code = (uint32_t)code;
			code += 1ull << (32 - len);
		}
	}
	vlc->bits = nb_bits;
	vlc->table = &qdm2_table[*offset];
	vlc->table_size = 0;
	build_table(vlc, nb_bits, j, codes);
	*offset += vlc->table_size;
}

static void qdm2_init_vlc(void)
{
	const uint8_t (*tab)[2] = tab_fft_tone_offset;
	unsigned offset = 0;

	build_vlc(&vlc_tab_level, 8, 24, &offset, tab_level);
	build_vlc(&vlc_tab_diff,  8, 33, &offset, tab_diff);
	build_vlc(&vlc_tab_run,   5,  6, &offset, tab_run);

	build_vlc(&fft_level_exp_alt_vlc, 8, 28, &offset, fft_level_exp_alt);
	build_vlc(&fft_level_exp_vlc,     8, 20, &offset, fft_level_exp);

	build_vlc(&fft_stereo_exp_vlc,   6, 7, &offset, fft_stereo_exp);
	build_vlc(&fft_stereo_phase_vlc, 6, 9, &offset, fft_stereo_phase);

	build_vlc(&vlc_tab_tone_level_idx_hi1, 8, 20, &offset, tab_tone_level_idx_hi1);
	build_vlc(&vlc_tab_tone_level_idx_mid, 8, 13, &offset, tab_tone_level_idx_mid);
	build_vlc(&vlc_tab_tone_level_idx_hi2, 8, 18, &offset, tab_tone_level_idx_hi2);

	build_vlc(&vlc_tab_type30, 6,  9, &offset, tab_type30);
	build_vlc(&vlc_tab_type34, 5, 10, &offset, tab_type34);

	for (int i = 0; i < 5; i++) {
		build_vlc(&vlc_tab_fft_tone_offset[i], 8, tab_fft_tone_offset_sizes[i], &offset, tab);
		tab += tab_fft_tone_offset_sizes[i];
	}
}

static inline int get_vlc2(GetBitContext* s, const VLCElem* table, int bits, int max_depth)
{
	int index = s->index;
	uint32_t cache = update_cache(s, index);

	unsigned entry = cache & ((1u << bits) - 1);
	int code = table[entry].sym;
	int n = table[entry].len;

	if (max_depth > 1 && n < 0) {
		index = skip_counter(s, index, bits);
		cache = update_cache(s, index);

		const int nb_bits = -n;
		entry = (cache & ((1u << nb_bits) - 1)) + code;
		code = table[entry].sym;
		n = table[entry].len;
	}
	s->index = skip_counter(s, index, n);
	return code;
}

///////////////////////////////////////////////////////////////////////////////
// qdm2_tablegen.h

static uint16_t softclip_table[HARDCLIP_THRESHOLD - SOFTCLIP_THRESHOLD + 1];
static float noise_table[4096 + 20];
static uint8_t random_dequant_index[256][5];
static uint8_t random_dequant_type24[128][3];
static float noise_samples[128];

static void softclip_table_init(void)
{
	double dfl = SOFTCLIP_THRESHOLD - 32767;
	float delta = 1.0 / -dfl;
	for (int i = 0; i < HARDCLIP_THRESHOLD - SOFTCLIP_THRESHOLD + 1; i++) {
		softclip_table[i] = SOFTCLIP_THRESHOLD - ((int)(sin((float)i * delta) * dfl) & 0x0000FFFF);
	}
}

// random generated table
static void rnd_table_init(void)
{
	uint32_t ldw;
	uint64_t random_seed = 0;
	float delta = 1.0 / 16384.0;
	for (int i = 0; i < 4096; i++) {
		random_seed = random_seed * 214013 + 2531011;
		noise_table[i] = (delta * (float)(((int32_t)random_seed >> 16) & 0x00007FFF) - 1.0) * 1.3;
	}

	for (int i = 0; i < 256; i++) {
		random_seed = 81;
		ldw = i;
		for (int j = 0; j < 5; j++) {
			random_dequant_index[i][j] = ldw / random_seed;
			ldw %= random_seed;
			random_seed /= 3;
		}
	}
	for (int i = 0; i < 128; i++) {
		random_seed = 25;
		ldw = i;
		for (int j = 0; j < 3; j++) {
			random_dequant_type24[i][j] = ldw / random_seed;
			ldw %= random_seed;
			random_seed /= 5;
		}
	}
}

static void init_noise_samples(void)
{
	unsigned random_seed = 0;
	float delta = 1.0 / 16384.0;
	for (int i = 0; i < 128; i++) {
		random_seed = random_seed * 214013 + 2531011;
		noise_samples[i] = (delta * (float)((random_seed >> 16) & 0x00007fff) - 1.0);
	}
}

///////////////////////////////////////////////////////////////////////////////
// qdm2.c

typedef int8_t sb_int8_array[2][30][64];

struct AVComplexFloat { float re, im; };

struct QDM2SubPacket {
	int type;
	unsigned int size;
	const uint8_t* data;	// into the input packet
};

struct QDM2SubPNode {
	QDM2SubPacket* packet;
	QDM2SubPNode* next;
};

struct FFTTone {
	float level;
	AVComplexFloat* complex;
	const float* table;
	int phase;
	int phase_shift;
	int duration;
	short time_index;
	short cutoff;
};

struct FFTCoefficient {
	int16_t sub_packet;
	uint8_t channel;
	int16_t offset;
	int16_t exp;
	uint8_t phase;
};

struct QDM2FFT {
	AVComplexFloat complex[MPA_MAX_CHANNELS][256 + 1];
	AVComplexFloat temp[MPA_MAX_CHANNELS][256];
};

struct Cpx { double re, im; };

struct QDM2Context {
	/// Parameters from the codec header, do not change during playback
	int nb_channels;
	int channels;
	int sample_rate;
	int bit_rate;
	int group_size;		// size of frame group (16 frames per group)
	int fft_size;		// size of FFT, in complex numbers
	int checksum_size;	// size of data block, used also for checksum

	/// Parameters built from them
	int group_order;
	int fft_order;		// order of FFT (actually fftorder+1)
	int frame_size;
	int frequency_range;
	int sub_sampling;	// subsampling: 0=25%, 1=50%, 2=100%
	int coeff_per_sb_select;
	int cm_table_select;

	/// The RDFT (twiddles of the length configured, the FFT's and the half-complex step's)
	Cpx fft_tw[256];
	float rdft_cos[128];
	float rdft_sin[128];

	/// Everything below is running state: qdm2_reset clears it, from `sub_packets` on.
	QDM2SubPacket sub_packets[16];
	QDM2SubPNode sub_packet_list_A[16];
	QDM2SubPNode sub_packet_list_B[16];
	int sub_packets_B;
	QDM2SubPNode sub_packet_list_D[16];

	/// FFT and tones
	FFTTone fft_tones[1000];
	int fft_tone_start;
	int fft_tone_end;
	FFTCoefficient fft_coefs[1000];
	int fft_coefs_index;
	int fft_coefs_min_index[5];
	int fft_coefs_max_index[5];
	int fft_level_exp[6];
	QDM2FFT fft;

	/// I/O data
	const uint8_t* compressed_data;
	int compressed_size;
	float output_buffer[QDM2_MAX_FRAME_SIZE * MPA_MAX_CHANNELS * 2];

	/// Synthesis filter
	float synth_buf[MPA_MAX_CHANNELS][512 * 2];
	int synth_buf_offset[MPA_MAX_CHANNELS];
	float sb_samples[MPA_MAX_CHANNELS][128][SBLIMIT];
	float samples[MPA_MAX_CHANNELS * MPA_FRAME_SIZE];

	/// Mixed temporary data used in decoding
	float tone_level[MPA_MAX_CHANNELS][30][64];
	int8_t coding_method[MPA_MAX_CHANNELS][30][64];
	int8_t quantized_coeffs[MPA_MAX_CHANNELS][10][8];
	int8_t tone_level_idx_base[MPA_MAX_CHANNELS][30][8];
	int8_t tone_level_idx_hi1[MPA_MAX_CHANNELS][3][8][8];
	int8_t tone_level_idx_mid[MPA_MAX_CHANNELS][26][8];
	int8_t tone_level_idx_hi2[MPA_MAX_CHANNELS][26];
	int8_t tone_level_idx[MPA_MAX_CHANNELS][30][64];
	int8_t tone_level_idx_temp[MPA_MAX_CHANNELS][30][64];

	// Flags
	int has_errors;
	int superblocktype_2_3;
	int do_synth_filter;

	int sub_packet;
	int noise_idx;

	/// One packet's output, planar
	float out[MPA_MAX_CHANNELS][16 * QDM2_MAX_FRAME_SIZE];
};

static const int switchtable[23] = {
	0, 5, 1, 5, 5, 5, 5, 5, 2, 5, 5, 5, 5, 5, 5, 5, 3, 5, 5, 5, 5, 5, 4
};

static int qdm2_get_vlc(GetBitContext* gb, const VLC* vlc, int flag, int depth)
{
	int value = get_vlc2(gb, vlc->table, vlc->bits, depth);

	/* stage-2, 3 bits exponent escape sequence */
	if (value < 0) {
		const int count = get_bits(gb, 3) + 1;
		value = get_bits(gb, count);
	}

	/* stage-3, optional */
	if (flag) {
		if (value >= 60) {
			return 0;
		}
		int tmp = vlc_stage3_values[value];
		if ((value & ~3) > 0) {
			tmp += get_bits(gb, value >> 2);
		}
		value = tmp;
	}
	return value;
}

static int qdm2_get_se_vlc(const VLC* vlc, GetBitContext* gb, int depth)
{
	const int value = qdm2_get_vlc(gb, vlc, 0, depth);
	return (value & 1) ? ((value + 1) >> 1) : -(value >> 1);
}

static uint16_t qdm2_packet_checksum(const uint8_t* data, int length, int value)
{
	for (int i = 0; i < length; i++) {
		value -= data[i];
	}
	return (uint16_t)(value & 0xffff);
}

static void qdm2_decode_sub_packet_header(GetBitContext* gb, QDM2SubPacket* sub_packet)
{
	sub_packet->type = get_bits(gb, 8);

	if (sub_packet->type == 0) {
		sub_packet->size = 0;
		sub_packet->data = NULL;
	} else {
		sub_packet->size = get_bits(gb, 8);

		if (sub_packet->type & 0x80) {
			sub_packet->size <<= 8;
			sub_packet->size |= get_bits(gb, 8);
			sub_packet->type &= 0x7f;
		}

		if (sub_packet->type == 0x7f) {
			sub_packet->type |= (get_bits(gb, 8) << 8);
		}

		sub_packet->data = &gb->buffer[get_bits_count(gb) / 8];
	}
}

static QDM2SubPNode* qdm2_search_subpacket_type_in_list(QDM2SubPNode* list, int type)
{
	while (list && list->packet) {
		if (list->packet->type == type) {
			return list;
		}
		list = list->next;
	}
	return NULL;
}

/** Replace 8 elements with their average value, before a superblock's subblocks decode. */
static void average_quantized_coeffs(QDM2Context* q)
{
	const int n = coeff_per_sb_for_avg[q->coeff_per_sb_select][QDM2_SB_USED(q->sub_sampling) - 1] + 1;

	for (int ch = 0; ch < q->nb_channels; ch++) {
		for (int i = 0; i < n; i++) {
			int sum = 0;
			for (int j = 0; j < 8; j++) {
				sum += q->quantized_coeffs[ch][i][j];
			}
			sum /= 8;
			if (sum > 0) {
				sum--;
			}
			for (int j = 0; j < 8; j++) {
				q->quantized_coeffs[ch][i][j] = sum;
			}
		}
	}
}

/** Build subband samples with noise weighted by q->tone_level. */
static int build_sb_samples_from_noise(QDM2Context* q, int sb)
{
	FIX_NOISE_IDX(q->noise_idx);

	if (!q->nb_channels) {
		return -1;
	}

	for (int ch = 0; ch < q->nb_channels; ch++) {
		for (int j = 0; j < 64; j++) {
			q->sb_samples[ch][j * 2][sb] = SB_DITHERING_NOISE(sb, q->noise_idx) * q->tone_level[ch][sb][j];
			q->sb_samples[ch][j * 2 + 1][sb] = SB_DITHERING_NOISE(sb, q->noise_idx) * q->tone_level[ch][sb][j];
		}
	}
	return 0;
}

/** Used after making changes to the coding_method array. */
static int fix_coding_method_array(int sb, int channels, sb_int8_array coding_method)
{
	for (int ch = 0; ch < channels; ch++) {
		for (int j = 0; j < 64; ) {
			int run, case_val;
			if (coding_method[ch][sb][j] < 8) {
				return -1;
			}
			if ((coding_method[ch][sb][j] - 8) > 22) {
				run = 1;
				case_val = 8;
			} else {
				switch (switchtable[coding_method[ch][sb][j] - 8]) {
				case 0: run = 10; case_val = 10; break;
				case 1: run = 1;  case_val = 16; break;
				case 2: run = 5;  case_val = 24; break;
				case 3: run = 3;  case_val = 30; break;
				case 4: run = 1;  case_val = 30; break;
				case 5: run = 1;  case_val = 8;  break;
				default: run = 1; case_val = 8;  break;
				}
			}
			for (int k = 0; k < run; k++) {
				if (j + k < 128) {
					const int sbjk = sb + (j + k) / 64;
					if (sbjk > 29) {
						continue;
					}
					if (coding_method[ch][sbjk][(j + k) % 64] > coding_method[ch][sb][j]) {
						if (k > 0) {
							//not debugged, almost never used
							memset(&coding_method[ch][sb][j + k], case_val, k * sizeof(int8_t));
							memset(&coding_method[ch][sb][j + k], case_val, 3 * sizeof(int8_t));
						}
					}
				}
			}
			j += run;
		}
	}
	return 0;
}

/** @param flag 1 if called after getting data from subpacket 10, 0 if no subpacket 10 */
static void fill_tone_level_array(QDM2Context* q, int flag)
{
	int tmp, tab;

	for (int ch = 0; ch < q->nb_channels; ch++) {
		for (int sb = 0; sb < 30; sb++) {
			for (int i = 0; i < 8; i++) {
				if ((tab = coeff_per_sb_for_dequant[q->coeff_per_sb_select][sb]) < (last_coeff[q->coeff_per_sb_select] - 1)) {
					tmp = q->quantized_coeffs[ch][tab + 1][i] * dequant_table[q->coeff_per_sb_select][tab + 1][sb] +
					      q->quantized_coeffs[ch][tab][i] * dequant_table[q->coeff_per_sb_select][tab][sb];
				} else {
					tmp = q->quantized_coeffs[ch][tab][i] * dequant_table[q->coeff_per_sb_select][tab][sb];
				}
				if (tmp < 0) {
					tmp += 0xff;
				}
				q->tone_level_idx_base[ch][sb][i] = (tmp / 256) & 0xff;
			}
		}
	}

	const int sb_used = QDM2_SB_USED(q->sub_sampling);

	if ((q->superblocktype_2_3 != 0) && !flag) {
		for (int sb = 0; sb < sb_used; sb++) {
			for (int ch = 0; ch < q->nb_channels; ch++) {
				for (int i = 0; i < 64; i++) {
					q->tone_level_idx[ch][sb][i] = q->tone_level_idx_base[ch][sb][i / 8];
					if (q->tone_level_idx[ch][sb][i] < 0) {
						q->tone_level[ch][sb][i] = 0;
					} else {
						q->tone_level[ch][sb][i] = fft_tone_level_table[0][q->tone_level_idx[ch][sb][i] & 0x3f];
					}
				}
			}
		}
	} else {
		tab = q->superblocktype_2_3 ? 0 : 1;
		for (int sb = 0; sb < sb_used; sb++) {
			if ((sb >= 4) && (sb <= 23)) {
				for (int ch = 0; ch < q->nb_channels; ch++) {
					for (int i = 0; i < 64; i++) {
						tmp = q->tone_level_idx_base[ch][sb][i / 8] -
						      q->tone_level_idx_hi1[ch][sb / 8][i / 8][i % 8] -
						      q->tone_level_idx_mid[ch][sb - 4][i / 8] -
						      q->tone_level_idx_hi2[ch][sb - 4];
						q->tone_level_idx[ch][sb][i] = tmp & 0xff;
						if ((tmp < 0) || (!q->superblocktype_2_3 && !tmp)) {
							q->tone_level[ch][sb][i] = 0;
						} else {
							q->tone_level[ch][sb][i] = fft_tone_level_table[tab][tmp & 0x3f];
						}
					}
				}
			} else if (sb > 4) {
				for (int ch = 0; ch < q->nb_channels; ch++) {
					for (int i = 0; i < 64; i++) {
						tmp = q->tone_level_idx_base[ch][sb][i / 8] -
						      q->tone_level_idx_hi1[ch][2][i / 8][i % 8] -
						      q->tone_level_idx_hi2[ch][sb - 4];
						q->tone_level_idx[ch][sb][i] = tmp & 0xff;
						if ((tmp < 0) || (!q->superblocktype_2_3 && !tmp)) {
							q->tone_level[ch][sb][i] = 0;
						} else {
							q->tone_level[ch][sb][i] = fft_tone_level_table[tab][tmp & 0x3f];
						}
					}
				}
			} else {
				for (int ch = 0; ch < q->nb_channels; ch++) {
					for (int i = 0; i < 64; i++) {
						tmp = q->tone_level_idx[ch][sb][i] = q->tone_level_idx_base[ch][sb][i / 8];
						if ((tmp < 0) || (!q->superblocktype_2_3 && !tmp)) {
							q->tone_level[ch][sb][i] = 0;
						} else {
							q->tone_level[ch][sb][i] = fft_tone_level_table[tab][tmp & 0x3f];
						}
					}
				}
			}
		}
	}
}

/** FFmpeg has no code for a superblock type other than 2 or 3 here, no samples being known. */
static int fill_coding_method_array(sb_int8_array coding_method, int nb_channels,
                                    int superblocktype_2_3, int cm_table_select)
{
	if (!superblocktype_2_3) {
		return -1;
	}
	for (int ch = 0; ch < nb_channels; ch++) {
		for (int sb = 0; sb < 30; sb++) {
			for (int j = 0; j < 64; j++) {
				coding_method[ch][sb][j] = coding_method_table[cm_table_select][sb];
			}
		}
	}
	return 0;
}

/**
 * Subpacket 11's data for sb 0-8, subpacket 12's for sb 8-sb_used.
 *
 * @param length packet length in bits
 * @param sb_min lower subband processed (included)
 * @param sb_max higher subband processed (excluded)
 */
static int synthfilt_build_sb_samples(QDM2Context* q, GetBitContext* gb, int length, int sb_min, int sb_max)
{
	int k, n, run;
	int joined_stereo, zero_encoding;
	int type34_first;
	float type34_div = 0;
	float type34_predictor;
	float samples[10];
	int sign_bits[16] = {0};

	if (length == 0) {
		// If no data use noise
		for (int sb = sb_min; sb < sb_max; sb++) {
			const int ret = build_sb_samples_from_noise(q, sb);
			if (ret < 0) {
				return ret;
			}
		}
		return 0;
	}

	for (int sb = sb_min; sb < sb_max; sb++) {
		int channels = q->nb_channels;

		if (q->nb_channels <= 1 || sb < 12) {
			joined_stereo = 0;
		} else if (sb >= 24) {
			joined_stereo = 1;
		} else {
			joined_stereo = (get_bits_left(gb) >= 1) ? get_bits1(gb) : 0;
		}

		if (joined_stereo) {
			if (get_bits_left(gb) >= 16) {
				for (int j = 0; j < 16; j++) {
					sign_bits[j] = get_bits1(gb);
				}
			}

			for (int j = 0; j < 64; j++) {
				if (q->coding_method[1][sb][j] > q->coding_method[0][sb][j]) {
					q->coding_method[0][sb][j] = q->coding_method[1][sb][j];
				}
			}

			if (fix_coding_method_array(sb, q->nb_channels, q->coding_method)) {
				const int ret = build_sb_samples_from_noise(q, sb);
				if (ret < 0) {
					return ret;
				}
				continue;
			}
			channels = 1;
		}

		for (int ch = 0; ch < channels; ch++) {
			FIX_NOISE_IDX(q->noise_idx);
			zero_encoding = (get_bits_left(gb) >= 1) ? get_bits1(gb) : 0;
			type34_predictor = 0.0;
			type34_first = 1;

			for (int j = 0; j < 128; ) {
				switch (q->coding_method[ch][sb][j / 2]) {
				case 8:
					if (get_bits_left(gb) >= 10) {
						if (zero_encoding) {
							for (k = 0; k < 5; k++) {
								if ((j + 2 * k) >= 128) {
									break;
								}
								if (get_bits1(gb)) {
									samples[2 * k] = dequant_1bit[joined_stereo][2 * get_bits1(gb)];
								} else {
									samples[2 * k] = 0;
								}
							}
						} else {
							n = get_bits(gb, 8);
							if (n >= 243) {
								return -1;
							}
							for (k = 0; k < 5; k++) {
								samples[2 * k] = dequant_1bit[joined_stereo][random_dequant_index[n][k]];
							}
						}
						for (k = 0; k < 5; k++) {
							samples[2 * k + 1] = SB_DITHERING_NOISE(sb, q->noise_idx);
						}
					} else {
						for (k = 0; k < 10; k++) {
							samples[k] = SB_DITHERING_NOISE(sb, q->noise_idx);
						}
					}
					run = 10;
					break;

				case 10:
					if (get_bits_left(gb) >= 1) {
						float f = 0.81;
						if (get_bits1(gb)) {
							f = -f;
						}
						f -= noise_samples[((sb + 1) * (j + 5 * ch + 1)) & 127] * 9.0 / 40.0;
						samples[0] = f;
					} else {
						samples[0] = SB_DITHERING_NOISE(sb, q->noise_idx);
					}
					run = 1;
					break;

				case 16:
					if (get_bits_left(gb) >= 10) {
						if (zero_encoding) {
							for (k = 0; k < 5; k++) {
								if ((j + k) >= 128) {
									break;
								}
								if (get_bits1(gb) == 0) {
									samples[k] = 0;
								} else {
									samples[k] = dequant_1bit[joined_stereo][2 * get_bits1(gb)];
								}
							}
						} else {
							n = get_bits(gb, 8);
							if (n >= 243) {
								return -1;
							}
							for (k = 0; k < 5; k++) {
								samples[k] = dequant_1bit[joined_stereo][random_dequant_index[n][k]];
							}
						}
					} else {
						for (k = 0; k < 5; k++) {
							samples[k] = SB_DITHERING_NOISE(sb, q->noise_idx);
						}
					}
					run = 5;
					break;

				case 24:
					if (get_bits_left(gb) >= 7) {
						n = get_bits(gb, 7);
						if (n >= 125) {
							return -1;
						}
						for (k = 0; k < 3; k++) {
							samples[k] = (random_dequant_type24[n][k] - 2.0) * 0.5;
						}
					} else {
						for (k = 0; k < 3; k++) {
							samples[k] = SB_DITHERING_NOISE(sb, q->noise_idx);
						}
					}
					run = 3;
					break;

				case 30:
					if (get_bits_left(gb) >= 4) {
						const unsigned index = qdm2_get_vlc(gb, &vlc_tab_type30, 0, 1);
						if (index >= ARRAY_ELEMS(type30_dequant)) {
							return -1;
						}
						samples[0] = type30_dequant[index];
					} else {
						samples[0] = SB_DITHERING_NOISE(sb, q->noise_idx);
					}
					run = 1;
					break;

				case 34:
					if (get_bits_left(gb) >= 7) {
						if (type34_first) {
							type34_div = (float)(1 << get_bits(gb, 2));
							samples[0] = ((float)get_bits(gb, 5) - 16.0) / 15.0;
							type34_predictor = samples[0];
							type34_first = 0;
						} else {
							const unsigned index = qdm2_get_vlc(gb, &vlc_tab_type34, 0, 1);
							if (index >= ARRAY_ELEMS(type34_delta)) {
								return -1;
							}
							samples[0] = type34_delta[index] / type34_div + type34_predictor;
							type34_predictor = samples[0];
						}
					} else {
						samples[0] = SB_DITHERING_NOISE(sb, q->noise_idx);
					}
					run = 1;
					break;

				default:
					samples[0] = SB_DITHERING_NOISE(sb, q->noise_idx);
					run = 1;
					break;
				}

				if (joined_stereo) {
					for (k = 0; k < run && j + k < 128; k++) {
						q->sb_samples[0][j + k][sb] = q->tone_level[0][sb][(j + k) / 2] * samples[k];
						if (q->nb_channels == 2) {
							if (sign_bits[(j + k) / 8]) {
								q->sb_samples[1][j + k][sb] = q->tone_level[1][sb][(j + k) / 2] * -samples[k];
							} else {
								q->sb_samples[1][j + k][sb] = q->tone_level[1][sb][(j + k) / 2] * samples[k];
							}
						}
					}
				} else {
					for (k = 0; k < run; k++) {
						if ((j + k) < 128) {
							q->sb_samples[ch][j + k][sb] = q->tone_level[ch][sb][(j + k) / 2] * samples[k];
						}
					}
				}

				j += run;
			}
		}
	}
	return 0;
}

/** The first element of a channel in quantized_coeffs, from packet 10. */
static int init_quantized_coeffs_elem0(int8_t* quantized_coeffs, GetBitContext* gb)
{
	if (get_bits_left(gb) < 16) {
		return -1;
	}
	int level = qdm2_get_vlc(gb, &vlc_tab_level, 0, 2);

	quantized_coeffs[0] = level;

	for (int i = 0; i < 7; ) {
		if (get_bits_left(gb) < 16) {
			return -1;
		}
		const int run = qdm2_get_vlc(gb, &vlc_tab_run, 0, 1) + 1;

		if (i + run >= 8) {
			return -1;
		}

		if (get_bits_left(gb) < 16) {
			return -1;
		}
		const int diff = qdm2_get_se_vlc(&vlc_tab_diff, gb, 2);

		for (int k = 1; k <= run; k++) {
			quantized_coeffs[i + k] = (level + ((k * diff) / run));
		}

		level += diff;
		i += run;
	}
	return 0;
}

/** Packet 10: part of quantized_coeffs, and tone_level_idx_hi1, _hi2 and _mid. */
static int init_tone_level_dequantization(QDM2Context* q, GetBitContext* gb)
{
	for (int ch = 0; ch < q->nb_channels; ch++) {
		const int ret = init_quantized_coeffs_elem0(q->quantized_coeffs[ch][0], gb);
		if (ret < 0) {
			return ret;
		}
		if (get_bits_left(gb) < 16) {
			memset(q->quantized_coeffs[ch][0], 0, 8);
			break;
		}
	}

	int n = q->sub_sampling + 1;

	for (int sb = 0; sb < n; sb++) {
		for (int ch = 0; ch < q->nb_channels; ch++) {
			for (int j = 0; j < 8; j++) {
				if (get_bits_left(gb) < 1) {
					break;
				}
				if (get_bits1(gb)) {
					for (int k = 0; k < 8; k++) {
						if (get_bits_left(gb) < 16) {
							break;
						}
						q->tone_level_idx_hi1[ch][sb][j][k] = qdm2_get_vlc(gb, &vlc_tab_tone_level_idx_hi1, 0, 2);
					}
				} else {
					for (int k = 0; k < 8; k++) {
						q->tone_level_idx_hi1[ch][sb][j][k] = 0;
					}
				}
			}
		}
	}

	n = QDM2_SB_USED(q->sub_sampling) - 4;

	for (int sb = 0; sb < n; sb++) {
		for (int ch = 0; ch < q->nb_channels; ch++) {
			if (get_bits_left(gb) < 16) {
				break;
			}
			q->tone_level_idx_hi2[ch][sb] = qdm2_get_vlc(gb, &vlc_tab_tone_level_idx_hi2, 0, 2);
			if (sb > 19) {
				q->tone_level_idx_hi2[ch][sb] -= 16;
			} else {
				for (int j = 0; j < 8; j++) {
					q->tone_level_idx_mid[ch][sb][j] = -16;
				}
			}
		}
	}

	n = QDM2_SB_USED(q->sub_sampling) - 5;

	for (int sb = 0; sb < n; sb++) {
		for (int ch = 0; ch < q->nb_channels; ch++) {
			for (int j = 0; j < 8; j++) {
				if (get_bits_left(gb) < 16) {
					break;
				}
				q->tone_level_idx_mid[ch][sb][j] = qdm2_get_vlc(gb, &vlc_tab_tone_level_idx_mid, 0, 2) - 32;
			}
		}
	}
	return 0;
}

/** Subpacket 9: quantized_coeffs. */
static int process_subpacket_9(QDM2Context* q, QDM2SubPNode* node)
{
	GetBitContext gb;

	const int ret = init_get_bits8(&gb, node->packet->data, node->packet->size);
	if (ret < 0) {
		return ret;
	}

	const int n = coeff_per_sb_for_avg[q->coeff_per_sb_select][QDM2_SB_USED(q->sub_sampling) - 1] + 1;

	for (int i = 1; i < n; i++) {
		for (int ch = 0; ch < q->nb_channels; ch++) {
			int level = qdm2_get_vlc(&gb, &vlc_tab_level, 0, 2);
			q->quantized_coeffs[ch][i][0] = level;

			for (int j = 0; j < (8 - 1); ) {
				const int run = qdm2_get_vlc(&gb, &vlc_tab_run, 0, 1) + 1;
				const int diff = qdm2_get_se_vlc(&vlc_tab_diff, &gb, 2);

				if (j + run >= 8) {
					return -1;
				}

				for (int k = 1; k <= run; k++) {
					q->quantized_coeffs[ch][i][j + k] = (level + ((k * diff) / run));
				}

				level += diff;
				j += run;
			}
		}
	}

	for (int ch = 0; ch < q->nb_channels; ch++) {
		for (int i = 0; i < 8; i++) {
			q->quantized_coeffs[ch][0][i] = 0;
		}
	}
	return 0;
}

static int process_subpacket_10(QDM2Context* q, QDM2SubPNode* node)
{
	GetBitContext gb;

	if (node) {
		int ret = init_get_bits8(&gb, node->packet->data, node->packet->size);
		if (ret < 0) {
			return ret;
		}
		ret = init_tone_level_dequantization(q, &gb);
		if (ret < 0) {
			return ret;
		}
		fill_tone_level_array(q, 1);
	} else {
		fill_tone_level_array(q, 0);
	}
	return 0;
}

static int process_subpacket_11(QDM2Context* q, QDM2SubPNode* node)
{
	GetBitContext gb;
	int length = 0;

	if (node) {
		const int ret = init_get_bits8(&gb, node->packet->data, node->packet->size);
		if (ret < 0) {
			return ret;
		}
		length = node->packet->size * 8;
	}

	if (length >= 32) {
		const int c = get_bits(&gb, 13);

		if (c > 3) {
			const int ret = fill_coding_method_array(q->coding_method, q->nb_channels,
			                                         q->superblocktype_2_3, q->cm_table_select);
			if (ret < 0) {
				return ret;
			}
		}
	}

	return synthfilt_build_sb_samples(q, &gb, length, 0, 8);
}

/** The reader is opened on `length` while that is still 0, as in FFmpeg: see the head of this file. */
static int process_subpacket_12(QDM2Context* q, QDM2SubPNode* node)
{
	GetBitContext gb;
	int length = 0;

	if (node) {
		const int ret = init_get_bits8(&gb, node->packet->data, length);
		if (ret < 0) {
			return ret;
		}
		length = node->packet->size * 8;
	}

	return synthfilt_build_sb_samples(q, &gb, length, 8, QDM2_SB_USED(q->sub_sampling));
}

static int process_synthesis_subpackets(QDM2Context* q, QDM2SubPNode* list)
{
	QDM2SubPNode* nodes[4];
	int ret = 0;

	nodes[0] = qdm2_search_subpacket_type_in_list(list, 9);
	if (nodes[0]) {
		ret = process_subpacket_9(q, nodes[0]);
	}
	if (ret < 0) {
		return ret;
	}

	nodes[1] = qdm2_search_subpacket_type_in_list(list, 10);
	ret = process_subpacket_10(q, nodes[1] ? nodes[1] : NULL);
	if (ret < 0) {
		return ret;
	}

	nodes[2] = qdm2_search_subpacket_type_in_list(list, 11);
	ret = process_subpacket_11(q, (nodes[0] && nodes[1] && nodes[2]) ? nodes[2] : NULL);
	if (ret < 0) {
		return ret;
	}

	nodes[3] = qdm2_search_subpacket_type_in_list(list, 12);
	return process_subpacket_12(q, (nodes[0] && nodes[1] && nodes[3]) ? nodes[3] : NULL);
}

static void list_add(QDM2SubPNode* list, int* size, QDM2SubPacket* packet)
{
	if (*size > 0) {
		list[*size - 1].next = &list[*size];
	}
	list[*size].packet = packet;
	list[*size].next = NULL;
	++*size;
}

/** Decode a superblock, fill the packet lists. */
static int qdm2_decode_super_block(QDM2Context* q)
{
	GetBitContext gb;
	QDM2SubPacket header, *packet;
	int i, packet_bytes, sub_packet_size, sub_packets_D;
	int ret;
	unsigned int next_index = 0;

	memset(q->tone_level_idx_hi1, 0, sizeof(q->tone_level_idx_hi1));
	memset(q->tone_level_idx_mid, 0, sizeof(q->tone_level_idx_mid));
	memset(q->tone_level_idx_hi2, 0, sizeof(q->tone_level_idx_hi2));

	q->sub_packets_B = 0;
	sub_packets_D = 0;

	average_quantized_coeffs(q);

	ret = init_get_bits8(&gb, q->compressed_data, q->compressed_size);
	if (ret < 0) {
		return ret;
	}

	qdm2_decode_sub_packet_header(&gb, &header);

	if (header.type < 2 || header.type >= 8) {
		q->has_errors = 1;
		return -1;
	}

	q->superblocktype_2_3 = (header.type == 2 || header.type == 3);
	packet_bytes = (q->compressed_size - get_bits_count(&gb) / 8);

	ret = init_get_bits8(&gb, header.data, header.size);
	if (ret < 0) {
		return ret;
	}

	if (header.type == 2 || header.type == 4 || header.type == 5) {
		int csum = 257 * get_bits(&gb, 8);
		csum += 2 * get_bits(&gb, 8);

		csum = qdm2_packet_checksum(q->compressed_data, q->checksum_size, csum);

		if (csum != 0) {
			q->has_errors = 1;
			return -1;
		}
	}

	q->sub_packet_list_B[0].packet = NULL;
	q->sub_packet_list_D[0].packet = NULL;

	for (i = 0; i < 6; i++) {
		if (--q->fft_level_exp[i] < 0) {
			q->fft_level_exp[i] = 0;
		}
	}

	for (i = 0; packet_bytes > 0; i++) {
		if (i >= (int)ARRAY_ELEMS(q->sub_packet_list_A)) {
			return -1;
		}

		q->sub_packet_list_A[i].next = NULL;

		if (i > 0) {
			q->sub_packet_list_A[i - 1].next = &q->sub_packet_list_A[i];

			/* seek to next block */
			ret = init_get_bits8(&gb, header.data, header.size);
			if (ret < 0) {
				return ret;
			}

			skip_bits(&gb, next_index * 8);

			if (next_index >= header.size) {
				break;
			}
		}

		/* decode subpacket */
		packet = &q->sub_packets[i];
		qdm2_decode_sub_packet_header(&gb, packet);
		next_index = packet->size + get_bits_count(&gb) / 8;
		sub_packet_size = ((packet->size > 0xff) ? 1 : 0) + packet->size + 2;

		if (packet->type == 0) {
			break;
		}

		if (sub_packet_size > packet_bytes) {
			if (packet->type != 10 && packet->type != 11 && packet->type != 12) {
				break;
			}
			packet->size += packet_bytes - sub_packet_size;
		}

		packet_bytes -= sub_packet_size;

		/* add subpacket to 'all subpackets' list */
		q->sub_packet_list_A[i].packet = packet;

		/* add subpacket to related list */
		if (packet->type == 8) {
			return -1;
		} else if (packet->type >= 9 && packet->type <= 12) {
			/* packets for MPEG Audio like Synthesis Filter */
			list_add(q->sub_packet_list_D, &sub_packets_D, packet);
		} else if (packet->type == 13) {
			for (int j = 0; j < 6; j++) {
				q->fft_level_exp[j] = get_bits(&gb, 6);
			}
		} else if (packet->type == 14) {
			for (int j = 0; j < 6; j++) {
				q->fft_level_exp[j] = qdm2_get_vlc(&gb, &fft_level_exp_vlc, 0, 2);
			}
		} else if (packet->type == 15) {
			return -1;
		} else if (packet->type >= 16 && packet->type < 48 && !fft_subpackets[packet->type - 16]) {
			/* packets for FFT */
			list_add(q->sub_packet_list_B, &q->sub_packets_B, packet);
		}
	}

	if (q->sub_packet_list_D[0].packet) {
		ret = process_synthesis_subpackets(q, q->sub_packet_list_D);
		if (ret < 0) {
			return ret;
		}
		q->do_synth_filter = 1;
	} else if (q->do_synth_filter) {
		ret = process_subpacket_10(q, NULL);
		if (ret < 0) {
			return ret;
		}
		ret = process_subpacket_11(q, NULL);
		if (ret < 0) {
			return ret;
		}
		ret = process_subpacket_12(q, NULL);
		if (ret < 0) {
			return ret;
		}
	}
	return 0;
}

static void qdm2_fft_init_coefficient(QDM2Context* q, int sub_packet, int offset, int duration,
                                      int channel, int exp, int phase)
{
	if (q->fft_coefs_min_index[duration] < 0) {
		q->fft_coefs_min_index[duration] = q->fft_coefs_index;
	}

	q->fft_coefs[q->fft_coefs_index].sub_packet = ((sub_packet >= 16) ? (sub_packet - 16) : sub_packet);
	q->fft_coefs[q->fft_coefs_index].channel = channel;
	q->fft_coefs[q->fft_coefs_index].offset = offset;
	q->fft_coefs[q->fft_coefs_index].exp = exp;
	q->fft_coefs[q->fft_coefs_index].phase = phase;
	q->fft_coefs_index++;
}

static int qdm2_fft_decode_tones(QDM2Context* q, int duration, GetBitContext* gb, int b)
{
	int channel, stereo, phase, exp;
	int local_int_4, local_int_8, stereo_phase, local_int_10;
	int local_int_14, stereo_exp, local_int_20, local_int_28;
	int n, offset;

	local_int_4 = 0;
	local_int_28 = 0;
	local_int_20 = 2;
	local_int_8 = (4 - duration);
	local_int_10 = 1 << (q->group_order - duration - 1);
	offset = 1;

	while (get_bits_left(gb) > 0) {
		if (q->superblocktype_2_3) {
			while ((n = qdm2_get_vlc(gb, &vlc_tab_fft_tone_offset[local_int_8], 1, 2)) < 2) {
				if (get_bits_left(gb) < 0) {
					return -1;
				}
				offset = 1;
				if (n == 0) {
					local_int_4 += local_int_10;
					local_int_28 += (1 << local_int_8);
				} else {
					local_int_4 += 8 * local_int_10;
					local_int_28 += (8 << local_int_8);
				}
			}
			offset += (n - 2);
		} else {
			if (local_int_10 <= 2) {
				return -1;
			}
			offset += qdm2_get_vlc(gb, &vlc_tab_fft_tone_offset[local_int_8], 1, 2);
			while (offset >= (local_int_10 - 1)) {
				offset += (1 - (local_int_10 - 1));
				local_int_4 += local_int_10;
				local_int_28 += (1 << local_int_8);
			}
		}

		if (local_int_4 >= q->group_size) {
			return -1;
		}

		local_int_14 = (offset >> local_int_8);
		if (local_int_14 >= (int)ARRAY_ELEMS(fft_level_index_table)) {
			return -1;
		}

		if (q->nb_channels > 1) {
			channel = get_bits1(gb);
			stereo = get_bits1(gb);
		} else {
			channel = 0;
			stereo = 0;
		}

		exp = qdm2_get_vlc(gb, (b ? &fft_level_exp_vlc : &fft_level_exp_alt_vlc), 0, 2);
		exp += q->fft_level_exp[fft_level_index_table[local_int_14]];
		exp = (exp < 0) ? 0 : exp;

		phase = get_bits(gb, 3);
		stereo_exp = 0;
		stereo_phase = 0;

		if (stereo) {
			stereo_exp = (exp - qdm2_get_vlc(gb, &fft_stereo_exp_vlc, 0, 1));
			stereo_phase = (phase - qdm2_get_vlc(gb, &fft_stereo_phase_vlc, 0, 1));
			if (stereo_phase < 0) {
				stereo_phase += 8;
			}
		}

		if (q->frequency_range > (local_int_14 + 1)) {
			const int sub_packet = (local_int_20 + local_int_28);

			if (q->fft_coefs_index + stereo >= (int)ARRAY_ELEMS(q->fft_coefs)) {
				return -1;
			}

			qdm2_fft_init_coefficient(q, sub_packet, offset, duration, channel, exp, phase);
			if (stereo) {
				qdm2_fft_init_coefficient(q, sub_packet, offset, duration, 1 - channel, stereo_exp, stereo_phase);
			}
		}
		offset++;
	}
	return 0;
}

static int qdm2_decode_fft_packets(QDM2Context* q)
{
	int i, j, min, max, value, type, unknown_flag;
	GetBitContext gb;

	if (!q->sub_packet_list_B[0].packet) {
		return -1;
	}

	/* reset minimum indexes for FFT coefficients */
	q->fft_coefs_index = 0;
	for (i = 0; i < 5; i++) {
		q->fft_coefs_min_index[i] = -1;
	}

	/* process subpackets ordered by type, largest type first */
	for (i = 0, max = 256; i < q->sub_packets_B; i++) {
		QDM2SubPacket* packet = NULL;

		/* find subpacket with largest type less than max */
		for (j = 0, min = 0; j < q->sub_packets_B; j++) {
			value = q->sub_packet_list_B[j].packet->type;
			if (value > min && value < max) {
				min = value;
				packet = q->sub_packet_list_B[j].packet;
			}
		}

		max = min;

		/* check for errors (?) */
		if (!packet) {
			return -1;
		}

		if (i == 0 && (packet->type < 16 || packet->type >= 48 || fft_subpackets[packet->type - 16])) {
			return -1;
		}

		/* decode FFT tones */
		const int ret = init_get_bits8(&gb, packet->data, packet->size);
		if (ret < 0) {
			return ret;
		}

		if (packet->type >= 32 && packet->type < 48 && !fft_subpackets[packet->type - 16]) {
			unknown_flag = 1;
		} else {
			unknown_flag = 0;
		}

		type = packet->type;

		if ((type >= 17 && type < 24) || (type >= 33 && type < 40)) {
			const int duration = q->sub_sampling + 5 - (type & 15);

			if (duration >= 0 && duration < 4) {
				qdm2_fft_decode_tones(q, duration, &gb, unknown_flag);
			}
		} else if (type == 31) {
			for (j = 0; j < 4; j++) {
				qdm2_fft_decode_tones(q, j, &gb, unknown_flag);
			}
		} else if (type == 46) {
			for (j = 0; j < 6; j++) {
				q->fft_level_exp[j] = get_bits(&gb, 6);
			}
			for (j = 0; j < 4; j++) {
				qdm2_fft_decode_tones(q, j, &gb, unknown_flag);
			}
		}
	}

	/* calculate maximum indexes for FFT coefficients */
	for (i = 0, j = -1; i < 5; i++) {
		if (q->fft_coefs_min_index[i] >= 0) {
			if (j >= 0) {
				q->fft_coefs_max_index[j] = q->fft_coefs_min_index[i];
			}
			j = i;
		}
	}
	if (j >= 0) {
		q->fft_coefs_max_index[j] = q->fft_coefs_index;
	}
	return 0;
}

static void qdm2_fft_generate_tone(QDM2Context* q, FFTTone* tone)
{
	float level, f[6];
	AVComplexFloat c;
	const double iscale = 2.0 * M_PI / 512.0;

	tone->phase += tone->phase_shift;

	/* calculate current level (maximum amplitude) of tone */
	level = fft_tone_envelope_table[tone->duration][tone->time_index] * tone->level;
	c.im = level * sin(tone->phase * iscale);
	c.re = level * cos(tone->phase * iscale);

	/* generate FFT coefficients for tone */
	if (tone->duration >= 3 || tone->cutoff >= 3) {
		tone->complex[0].im += c.im;
		tone->complex[0].re += c.re;
		tone->complex[1].im -= c.im;
		tone->complex[1].re -= c.re;
	} else {
		f[1] = -tone->table[4];
		f[0] = tone->table[3] - tone->table[0];
		f[2] = 1.0 - tone->table[2] - tone->table[3];
		f[3] = tone->table[1] + tone->table[4] - 1.0;
		f[4] = tone->table[0] - tone->table[1];
		f[5] = tone->table[2];
		for (int i = 0; i < 2; i++) {
			tone->complex[fft_cutoff_index_table[tone->cutoff][i]].re += c.re * f[i];
			tone->complex[fft_cutoff_index_table[tone->cutoff][i]].im += c.im * ((tone->cutoff <= i) ? -f[i] : f[i]);
		}
		for (int i = 0; i < 4; i++) {
			tone->complex[i].re += c.re * f[i + 2];
			tone->complex[i].im += c.im * f[i + 2];
		}
	}

	/* copy the tone if it has not yet died out */
	if (++tone->time_index < ((1 << (5 - tone->duration)) - 1)) {
		memcpy(&q->fft_tones[q->fft_tone_end], tone, sizeof(FFTTone));
		q->fft_tone_end = (q->fft_tone_end + 1) % 1000;
	}
}

static void qdm2_fft_tone_synthesizer(QDM2Context* q, int sub_packet)
{
	int i, j, ch;
	const double iscale = 0.25 * M_PI;

	for (ch = 0; ch < q->channels; ch++) {
		memset(q->fft.complex[ch], 0, q->fft_size * sizeof(AVComplexFloat));
	}

	/* apply FFT tones with duration 4 (1 FFT period) */
	if (q->fft_coefs_min_index[4] >= 0) {
		for (i = q->fft_coefs_min_index[4]; i < q->fft_coefs_max_index[4]; i++) {
			float level;
			AVComplexFloat c;

			if (q->fft_coefs[i].sub_packet != sub_packet) {
				break;
			}

			ch = (q->channels == 1) ? 0 : q->fft_coefs[i].channel;
			level = (q->fft_coefs[i].exp < 0) ? 0.0 : fft_tone_level_table[q->superblocktype_2_3 ? 0 : 1][q->fft_coefs[i].exp & 63];

			c.re = level * cos(q->fft_coefs[i].phase * iscale);
			c.im = level * sin(q->fft_coefs[i].phase * iscale);
			q->fft.complex[ch][q->fft_coefs[i].offset + 0].re += c.re;
			q->fft.complex[ch][q->fft_coefs[i].offset + 0].im += c.im;
			q->fft.complex[ch][q->fft_coefs[i].offset + 1].re -= c.re;
			q->fft.complex[ch][q->fft_coefs[i].offset + 1].im -= c.im;
		}
	}

	/* generate existing FFT tones */
	for (i = q->fft_tone_end; i != q->fft_tone_start; ) {
		qdm2_fft_generate_tone(q, &q->fft_tones[q->fft_tone_start]);
		q->fft_tone_start = (q->fft_tone_start + 1) % 1000;
	}

	/* create and generate new FFT tones with duration 0 (long) to 3 (short) */
	for (i = 0; i < 4; i++) {
		if (q->fft_coefs_min_index[i] >= 0) {
			for (j = q->fft_coefs_min_index[i]; j < q->fft_coefs_max_index[i]; j++) {
				int offset, four_i;
				FFTTone tone;

				if (q->fft_coefs[j].sub_packet != sub_packet) {
					break;
				}

				four_i = (4 - i);
				offset = q->fft_coefs[j].offset >> four_i;
				ch = (q->channels == 1) ? 0 : q->fft_coefs[j].channel;

				if (offset < q->frequency_range) {
					if (offset < 2) {
						tone.cutoff = offset;
					} else {
						tone.cutoff = (offset >= 60) ? 3 : 2;
					}

					tone.level = (q->fft_coefs[j].exp < 0) ? 0.0 : fft_tone_level_table[q->superblocktype_2_3 ? 0 : 1][q->fft_coefs[j].exp & 63];
					tone.complex = &q->fft.complex[ch][offset];
					tone.table = fft_tone_sample_table[i][q->fft_coefs[j].offset - (offset << four_i)];
					tone.phase = 64 * q->fft_coefs[j].phase - (offset << 8) - 128;
					tone.phase_shift = (2 * q->fft_coefs[j].offset + 1) << (7 - four_i);
					tone.duration = i;
					tone.time_index = 0;

					qdm2_fft_generate_tone(q, &tone);
				}
			}
			q->fft_coefs_min_index[i] = j;
		}
	}
}

/** The inverse complex FFT inside the RDFT: e^(+2 pi i nk / n), unscaled, n a power of two up to 256. */
static void fft_inverse(const QDM2Context* q, Cpx* z, int n)
{
	for (int i = 1, j = 0; i < n; i++) {
		int bit = n >> 1;
		for (; j & bit; bit >>= 1) {
			j ^= bit;
		}
		j ^= bit;
		if (i < j) {
			const Cpx t = z[i];
			z[i] = z[j];
			z[j] = t;
		}
	}
	for (int size = 2; size <= n; size <<= 1) {
		const int half = size >> 1;
		const int stride = n / size;
		for (int i = 0; i < n; i += size) {
			for (int j = 0; j < half; j++) {
				const Cpx w = q->fft_tw[j * stride];
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

/** tx_template.c's rdft_c2r with scale 0.5: fft_size + 1 complex bins in, 2 * fft_size reals out as complex pairs. */
static void rdft_inverse(QDM2Context* q, AVComplexFloat* dst, AVComplexFloat* data)
{
	const int len2 = q->fft_size;
	const int len4 = len2 >> 1;
	const float* tcos = q->rdft_cos;
	const float* tsin = q->rdft_sin;
	AVComplexFloat t[3];

	data[0].im = data[len2].re;

	t[0].re = data[0].re;
	data[0].re = t[0].re + data[0].im;
	data[0].im = t[0].re - data[0].im;
	data[0].re = 0.5f * data[0].re;
	data[0].im = 0.5f * data[0].im;
	data[len4].im = -data[len4].im;

	for (int i = 1; i < len4; i++) {
		/* Separate even and odd FFTs */
		t[0].re = 0.5f * (data[i].re + data[len2 - i].re);
		t[0].im = -0.5f * (data[i].im - data[len2 - i].im);
		t[1].re = -0.5f * (data[i].im + data[len2 - i].im);
		t[1].im = 0.5f * (data[i].re - data[len2 - i].re);

		/* Apply twiddle factors to the odd FFT and add to the even FFT */
		t[2].re = t[1].re * tcos[i] - t[1].im * tsin[i];
		t[2].im = t[1].re * tsin[i] + t[1].im * tcos[i];

		data[i].re = t[0].re + t[2].re;
		data[i].im = t[2].im - t[0].im;
		data[len2 - i].re = t[0].re - t[2].re;
		data[len2 - i].im = t[2].im + t[0].im;
	}

	Cpx z[256];
	for (int i = 0; i < len2; i++) {
		z[i].re = data[i].re;
		z[i].im = data[i].im;
	}
	fft_inverse(q, z, len2);
	for (int i = 0; i < len2; i++) {
		dst[i].re = (float)z[i].re;
		dst[i].im = (float)z[i].im;
	}
}

static void qdm2_calculate_fft(QDM2Context* q, int channel)
{
	const float gain = (q->channels == 1 && q->nb_channels == 2) ? 0.5f : 1.0f;
	float* out = q->output_buffer + channel;

	q->fft.complex[channel][0].re *= 2.0f;
	q->fft.complex[channel][0].im = 0.0f;
	q->fft.complex[channel][q->fft_size].re = 0.0f;
	q->fft.complex[channel][q->fft_size].im = 0.0f;

	rdft_inverse(q, q->fft.temp[channel], q->fft.complex[channel]);

	/* add samples to output buffer */
	for (int i = 0; i < q->fft_size; i++) {
		out[0] += q->fft.temp[channel][i].re * gain;
		out[q->channels] += q->fft.temp[channel][i].im * gain;
		out += 2 * q->channels;
	}
}

static void qdm2_synthesis_filter(QDM2Context* q, int index)
{
	int dither_state = 0;

	/* copy sb_samples */
	const int sb_used = QDM2_SB_USED(q->sub_sampling);

	for (int ch = 0; ch < q->channels; ch++) {
		for (int i = 0; i < 8; i++) {
			for (int k = sb_used; k < SBLIMIT; k++) {
				q->sb_samples[ch][(8 * index) + i][k] = 0;
			}
		}
	}

	for (int ch = 0; ch < q->nb_channels; ch++) {
		float* samples_ptr = q->samples + ch;

		for (int i = 0; i < 8; i++) {
			mpa_synth_filter(q->synth_buf[ch], &q->synth_buf_offset[ch], ff_mpa_synth_window, &dither_state,
			                 samples_ptr, q->nb_channels, q->sb_samples[ch][(8 * index) + i]);
			samples_ptr += 32 * q->nb_channels;
		}
	}

	/* add samples to output buffer */
	const int sub_sampling = (4 >> q->sub_sampling);

	for (int ch = 0; ch < q->channels; ch++) {
		for (int i = 0; i < q->frame_size; i++) {
			q->output_buffer[q->channels * i + ch] += (1 << 23) * q->samples[q->nb_channels * sub_sampling * i + ch];
		}
	}
}

/** One sub-frame, into q->out at `at`. */
static int qdm2_decode(QDM2Context* q, const uint8_t* in, int at)
{
	const int frame_size = q->frame_size * q->channels;

	if ((unsigned)frame_size > ARRAY_ELEMS(q->output_buffer) / 2) {
		return -1;
	}

	/* select input buffer */
	q->compressed_data = in;
	q->compressed_size = q->checksum_size;

	/* copy old block, clear new block of output samples */
	memmove(q->output_buffer, &q->output_buffer[frame_size], frame_size * sizeof(float));
	memset(&q->output_buffer[frame_size], 0, frame_size * sizeof(float));

	/* decode block of QDM2 compressed data */
	if (q->sub_packet == 0) {
		q->has_errors = 0;	// zero it for a new super block
		const int ret = qdm2_decode_super_block(q);
		if (ret < 0) {
			return ret;
		}
	}

	/* parse subpackets */
	if (!q->has_errors) {
		int ret = 0;
		if (q->sub_packet == 2) {
			ret = qdm2_decode_fft_packets(q);
		}
		if (ret < 0) {
			return ret;
		}
		qdm2_fft_tone_synthesizer(q, q->sub_packet);
	}

	/* sound synthesis stage 1 (FFT) */
	for (int ch = 0; ch < q->channels; ch++) {
		qdm2_calculate_fft(q, ch);
	}

	/* sound synthesis stage 2 (MPEG audio like synthesis filter) */
	if (!q->has_errors && q->do_synth_filter) {
		qdm2_synthesis_filter(q, q->sub_packet);
	}

	q->sub_packet = (q->sub_packet + 1) % 16;

	/* clip to FFmpeg's 16-bit samples, and give them as float */
	for (int i = 0; i < frame_size; i++) {
		int value = (int)q->output_buffer[i];

		if (value > SOFTCLIP_THRESHOLD) {
			value = (value > HARDCLIP_THRESHOLD) ? 32767 : softclip_table[value - SOFTCLIP_THRESHOLD];
		} else if (value < -SOFTCLIP_THRESHOLD) {
			value = (value < -HARDCLIP_THRESHOLD) ? -32767 : -softclip_table[-value - SOFTCLIP_THRESHOLD];
		}

		q->out[i % q->channels][at + i / q->channels] = (int16_t)value / 32768.0f;
	}
	return 0;
}

static int read_be32(const uint8_t* p)
{
	return (int)(((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) | ((uint32_t)p[2] << 8) | p[3]);
}

static int ilog2(unsigned value)
{
	return 31 - __builtin_clz(value | 1);
}

/** qdm2_decode_init, with the extradata being what a 'wave' atom holds: frma, QDCA, QDCP. */
static int qdm2_configure_context(QDM2Context* s, const uint8_t* extradata, int size)
{
	if (size < 48) {
		return 0;
	}

	const uint8_t* p = extradata;
	const uint8_t* end = extradata + size;
	static const uint8_t frma[8] = {'f', 'r', 'm', 'a', 'Q', 'D', 'M', '2'};
	while (end - p > 8) {
		if (memcmp(p, frma, 8) == 0) {
			break;
		}
		++p;
	}

	if (end - p < 44) {
		return 0;
	}

	p += 8;
	const int qdca_size = read_be32(p);
	p += 4;
	if (qdca_size > end - p) {
		return 0;
	}

	if (memcmp(p, "QDCA", 4) != 0) {
		return 0;
	}
	p += 8;	// the tag, and a field of 1

	s->nb_channels = s->channels = read_be32(p);
	if (s->channels <= 0 || s->channels > MPA_MAX_CHANNELS) {
		return 0;
	}

	s->sample_rate = read_be32(p + 4);
	s->bit_rate = read_be32(p + 8);
	s->group_size = read_be32(p + 12);
	s->fft_size = read_be32(p + 16);
	s->checksum_size = read_be32(p + 20);
	if ((unsigned)s->checksum_size >= 1u << 28 || (unsigned)s->checksum_size <= 1) {
		return 0;
	}

	s->fft_order = ilog2(s->fft_size) + 1;

	// Fail on unknown fft order
	if ((s->fft_order < 7) || (s->fft_order > 9)) {
		return 0;
	}

	// something like max decodable tones
	s->group_order = ilog2(s->group_size) + 1;
	s->frame_size = s->group_size / 16;	// 16 iterations per super block

	if (s->frame_size > QDM2_MAX_FRAME_SIZE || s->frame_size <= 0) {
		return 0;
	}

	s->sub_sampling = s->fft_order - 7;
	s->frequency_range = 255 / (1 << (2 - s->sub_sampling));

	if (s->frame_size * 4 >> s->sub_sampling > MPA_FRAME_SIZE) {
		return 0;
	}

	int tmp;
	switch (s->sub_sampling * 2 + s->channels - 1) {
	case 0: tmp = 40; break;
	case 1: tmp = 48; break;
	case 2: tmp = 56; break;
	case 3: tmp = 72; break;
	case 4: tmp = 80; break;
	case 5: tmp = 100; break;
	default: tmp = s->sub_sampling; break;
	}
	int tmp_val = 0;
	if ((tmp * 1000) < s->bit_rate) tmp_val = 1;
	if ((tmp * 1440) < s->bit_rate) tmp_val = 2;
	if ((tmp * 1760) < s->bit_rate) tmp_val = 3;
	if ((tmp * 2240) < s->bit_rate) tmp_val = 4;
	s->cm_table_select = tmp_val;

	if (s->bit_rate <= 8000) {
		s->coeff_per_sb_select = 0;
	} else if (s->bit_rate < 16000) {
		s->coeff_per_sb_select = 1;
	} else {
		s->coeff_per_sb_select = 2;
	}

	if (s->fft_size != (1 << (s->fft_order - 1))) {
		return 0;
	}

	const int len = 2 * s->fft_size;
	const double f = 2 * M_PI / len;
	for (int i = 0; i < s->fft_size / 2; i++) {
		s->rdft_cos[i] = (float)cos(i * f);
		s->rdft_sin[i] = (float)cos(((len - i * 4) / 4.0) * f);
	}
	for (int i = 0; i < s->fft_size; i++) {
		s->fft_tw[i].re = cos(i * 2 * f);
		s->fft_tw[i].im = sin(i * 2 * f);
	}
	return 1;
}

static void qdm2_clear_running_state(QDM2Context* s)
{
	memset(s->sub_packets, 0, sizeof(*s) - offsetof(QDM2Context, sub_packets));
}

extern "C" {

WASM_EXPORT QDM2Context* qdm2_open(void)
{
	static int s_ready = 0;
	if (!s_ready) {
		s_ready = 1;
		qdm2_init_vlc();
		softclip_table_init();
		rnd_table_init();
		init_noise_samples();
		mpa_dsp_init_tabs();
	}
	return (QDM2Context*)calloc(1, sizeof(QDM2Context));
}

WASM_EXPORT void qdm2_close(QDM2Context* s) { free(s); }

/** Seek: everything a decode has built up goes. */
WASM_EXPORT void qdm2_reset(QDM2Context* s) { qdm2_clear_running_state(s); }

/**
 * info: int32 sampleRate, channels, samples per packet, bytes per packet.
 * @returns 1 for a stream FFmpeg's decoder takes, else 0
 */
WASM_EXPORT int qdm2_configure(QDM2Context* s, const uint8_t* extradata, int size, int* info)
{
	const int ok = qdm2_configure_context(s, extradata, size);
	info[0] = s->sample_rate;
	info[1] = s->channels;
	info[2] = 16 * s->frame_size;
	info[3] = s->checksum_size;
	return ok;
}

/**
 * One packet: 16 sub-frames of frame_size samples a channel, planar in qdm2_channel_ptr. As
 * in FFmpeg a packet that fails partway has no output, and what it built up stays.
 * @returns 1, or 0 for a packet shorter than the stream's or one that does not decode
 */
WASM_EXPORT int qdm2_decode_packet(QDM2Context* s, const uint8_t* in, int size)
{
	if (size < s->checksum_size) {
		return 0;
	}
	s->sub_packet = 0;
	for (int i = 0; i < 16; i++) {
		if (qdm2_decode(s, in, i * s->frame_size) < 0) {
			return 0;
		}
	}
	return 1;
}

WASM_EXPORT float* qdm2_channel_ptr(QDM2Context* s, int ch) { return s->out[ch]; }

}
