/*
	AC-3 constant tables (ATSC A/52).

	Values are taken from FFmpeg's ac3tab.c / ac3.c / ac3dec_data.c (LGPL) — they are
	the numeric tables of the A/52 standard itself. Split out of ac3_decode.cpp only
	to keep that file readable; there is no separate translation unit.

	The runtime-generated tables at the bottom mirror ac3dec_data.c's ac3_init_static()
	and ac3dec_float.c's ac3_float_tables_init(): AC-3 derives them from short closed
	formulas rather than storing them, and so do we.
*/

#ifndef AC3_TABLES_H
#define AC3_TABLES_H

#include <stdint.h>
#include <math.h>

// A/52 Table 5.18 Frame Size Code Table, in 16-bit words, [frmsizecod][fscod].
static const uint16_t ff_ac3_frame_size_tab[38][3] = {
	{ 64,69,96 },{ 64,70,96 },{ 80,87,120 },{ 80,88,120 },{ 96,104,144 },{ 96,105,144 },
	{ 112,121,168 },{ 112,122,168 },{ 128,139,192 },{ 128,140,192 },{ 160,174,240 },{ 160,175,240 },
	{ 192,208,288 },{ 192,209,288 },{ 224,243,336 },{ 224,244,336 },{ 256,278,384 },{ 256,279,384 },
	{ 320,348,480 },{ 320,349,480 },{ 384,417,576 },{ 384,418,576 },{ 448,487,672 },{ 448,488,672 },
	{ 512,557,768 },{ 512,558,768 },{ 640,696,960 },{ 640,697,960 },{ 768,835,1152 },{ 768,836,1152 },
	{ 896,975,1344 },{ 896,976,1344 },{ 1024,1114,1536 },{ 1024,1115,1536 },{ 1152,1253,1728 },{ 1152,1254,1728 },
	{ 1280,1393,1920 },{ 1280,1394,1920 },
};

// acmod -> number of full-bandwidth channels (A/52 Table 5.8).
static const uint8_t ff_ac3_channels_tab[8] = { 2, 1, 2, 3, 3, 4, 4, 5 };

// Remap channels from AC-3 coded order to FFmpeg/SMPTE output order:
// [channel_mode][lfe_on][output slot] -> decoder channel index.
// 3F2R + LFE: coded L,C,R,Ls,Rs,LFE -> FL,FR,FC,LFE,SL,SR.
static const uint8_t ff_ac3_dec_channel_map[8][2][6] = {
	{ { 0, 1,          }, { 0, 1, 2,         } },
	{ { 0,             }, { 0, 1,            } },
	{ { 0, 1,          }, { 0, 1, 2,         } },
	{ { 0, 2, 1,       }, { 0, 2, 1, 3,      } },
	{ { 0, 1, 2,       }, { 0, 1, 3, 2,      } },
	{ { 0, 2, 1, 3,    }, { 0, 2, 1, 4, 3,   } },
	{ { 0, 1, 2, 3,    }, { 0, 1, 4, 2, 3,   } },
	{ { 0, 2, 1, 3, 4, }, { 0, 2, 1, 5, 3, 4 } },
};

static const int ff_ac3_sample_rate_tab[4] = { 48000, 44100, 32000, 0 };

static const uint16_t ff_ac3_bitrate_tab[19] = {
	32, 40, 48, 56, 64, 80, 96, 112, 128,
	160, 192, 224, 256, 320, 384, 448, 512, 576, 640
};

// Bin locations for rematrixing bands (A/52 7.5.2).
static const uint8_t ff_ac3_rematrix_band_tab[5] = { 13, 25, 37, 61, 253 };

static const uint8_t ff_ac3_bap_tab[64] = {
	0, 1, 1, 1, 1, 1, 2, 2, 3, 3,
	3, 4, 4, 5, 5, 6, 6, 6, 6, 7,
	7, 7, 7, 8, 8, 8, 8, 9, 9, 9,
	9, 10, 10, 10, 10, 11, 11, 11, 11, 12,
	12, 12, 12, 13, 13, 13, 13, 14, 14, 14,
	14, 14, 14, 14, 14, 15, 15, 15, 15, 15,
	15, 15, 15, 15,
};

static const uint8_t  ff_ac3_slow_decay_tab[4] = { 0x0f, 0x11, 0x13, 0x15 };
static const uint8_t  ff_ac3_fast_decay_tab[4] = { 0x3f, 0x53, 0x67, 0x7b };
static const uint16_t ff_ac3_slow_gain_tab[4]  = { 0x540, 0x4d8, 0x478, 0x410 };
static const uint16_t ff_ac3_db_per_bit_tab[4] = { 0x000, 0x700, 0x900, 0xb00 };
static const int16_t  ff_ac3_floor_tab[8]      = { 0x2f0, 0x2b0, 0x270, 0x230, 0x1f0, 0x170, 0x0f0, (int16_t)0xf800 };
static const uint16_t ff_ac3_fast_gain_tab[8]  = { 0x080, 0x100, 0x180, 0x200, 0x280, 0x300, 0x380, 0x400 };

// Mantissa bit widths by bap (A/52 Table 7.24); 0 for grouped/dithered baps.
static const uint8_t ff_ac3_quantization_tab[16] = {
	0, 3, 5, 7, 11, 15,
	5, 6, 7, 8, 9, 10, 11, 12, 14, 16
};

// Starting frequency coefficient bin for each critical band.
static const uint8_t ff_ac3_band_start_tab[AC3_CRITICAL_BANDS + 1] = {
	  0,  1,   2,   3,   4,   5,   6,   7,   8,   9,
	 10,  11, 12,  13,  14,  15,  16,  17,  18,  19,
	 20,  21, 22,  23,  24,  25,  26,  27,  28,  31,
	 34,  37, 40,  43,  46,  49,  55,  61,  67,  73,
	 79,  85, 97, 109, 121, 133, 157, 181, 205, 229, 253
};

// Map each frequency coefficient bin to the critical band containing it.
static const uint8_t ff_ac3_bin_to_band_tab[253] = {
	 0,
	 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12,
	13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
	25, 26, 27, 28, 28, 28, 29, 29, 29, 30, 30, 30,
	31, 31, 31, 32, 32, 32, 33, 33, 33, 34, 34, 34,
	35, 35, 35, 35, 35, 35, 36, 36, 36, 36, 36, 36,
	37, 37, 37, 37, 37, 37, 38, 38, 38, 38, 38, 38,
	39, 39, 39, 39, 39, 39, 40, 40, 40, 40, 40, 40,
	41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41,
	42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
	43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43,
	44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44,
	45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45,
	45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45,
	46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46,
	46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46,
	47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47,
	47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47,
	48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48,
	48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48,
	49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49,
	49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49
};

static const uint8_t ac3_log_add_tab[260] = {
0x40,0x3f,0x3e,0x3d,0x3c,0x3b,0x3a,0x39,0x38,0x37,
0x36,0x35,0x34,0x34,0x33,0x32,0x31,0x30,0x2f,0x2f,
0x2e,0x2d,0x2c,0x2c,0x2b,0x2a,0x29,0x29,0x28,0x27,
0x26,0x26,0x25,0x24,0x24,0x23,0x23,0x22,0x21,0x21,
0x20,0x20,0x1f,0x1e,0x1e,0x1d,0x1d,0x1c,0x1c,0x1b,
0x1b,0x1a,0x1a,0x19,0x19,0x18,0x18,0x17,0x17,0x16,
0x16,0x15,0x15,0x15,0x14,0x14,0x13,0x13,0x13,0x12,
0x12,0x12,0x11,0x11,0x11,0x10,0x10,0x10,0x0f,0x0f,
0x0f,0x0e,0x0e,0x0e,0x0d,0x0d,0x0d,0x0d,0x0c,0x0c,
0x0c,0x0c,0x0b,0x0b,0x0b,0x0b,0x0a,0x0a,0x0a,0x0a,
0x0a,0x09,0x09,0x09,0x09,0x09,0x08,0x08,0x08,0x08,
0x08,0x08,0x07,0x07,0x07,0x07,0x07,0x07,0x06,0x06,
0x06,0x06,0x06,0x06,0x06,0x06,0x05,0x05,0x05,0x05,
0x05,0x05,0x05,0x05,0x04,0x04,0x04,0x04,0x04,0x04,
0x04,0x04,0x04,0x04,0x04,0x03,0x03,0x03,0x03,0x03,
0x03,0x03,0x03,0x03,0x03,0x03,0x03,0x03,0x03,0x02,
0x02,0x02,0x02,0x02,0x02,0x02,0x02,0x02,0x02,0x02,
0x02,0x02,0x02,0x02,0x02,0x02,0x02,0x02,0x01,0x01,
0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,
0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,
0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,
0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
};

static const uint16_t ac3_hearing_threshold_tab[AC3_CRITICAL_BANDS][3] = {
{ 0x04d0,0x04f0,0x0580 },{ 0x04d0,0x04f0,0x0580 },{ 0x0440,0x0460,0x04b0 },
{ 0x0400,0x0410,0x0450 },{ 0x03e0,0x03e0,0x0420 },{ 0x03c0,0x03d0,0x03f0 },
{ 0x03b0,0x03c0,0x03e0 },{ 0x03b0,0x03b0,0x03d0 },{ 0x03a0,0x03b0,0x03c0 },
{ 0x03a0,0x03a0,0x03b0 },{ 0x03a0,0x03a0,0x03b0 },{ 0x03a0,0x03a0,0x03b0 },
{ 0x03a0,0x03a0,0x03a0 },{ 0x0390,0x03a0,0x03a0 },{ 0x0390,0x0390,0x03a0 },
{ 0x0390,0x0390,0x03a0 },{ 0x0380,0x0390,0x03a0 },{ 0x0380,0x0380,0x03a0 },
{ 0x0370,0x0380,0x03a0 },{ 0x0370,0x0380,0x03a0 },{ 0x0360,0x0370,0x0390 },
{ 0x0360,0x0370,0x0390 },{ 0x0350,0x0360,0x0390 },{ 0x0350,0x0360,0x0390 },
{ 0x0340,0x0350,0x0380 },{ 0x0340,0x0350,0x0380 },{ 0x0330,0x0340,0x0380 },
{ 0x0320,0x0340,0x0370 },{ 0x0310,0x0320,0x0360 },{ 0x0300,0x0310,0x0350 },
{ 0x02f0,0x0300,0x0340 },{ 0x02f0,0x02f0,0x0330 },{ 0x02f0,0x02f0,0x0320 },
{ 0x02f0,0x02f0,0x0310 },{ 0x0300,0x02f0,0x0300 },{ 0x0310,0x0300,0x02f0 },
{ 0x0340,0x0320,0x02f0 },{ 0x0390,0x0350,0x02f0 },{ 0x03e0,0x0390,0x0300 },
{ 0x0420,0x03e0,0x0310 },{ 0x0460,0x0420,0x0330 },{ 0x0490,0x0450,0x0350 },
{ 0x04a0,0x04a0,0x03c0 },{ 0x0460,0x0490,0x0410 },{ 0x0440,0x0460,0x0470 },
{ 0x0440,0x0440,0x04a0 },{ 0x0520,0x0480,0x0460 },{ 0x0800,0x0630,0x0440 },
{ 0x0840,0x0840,0x0450 },{ 0x0840,0x0840,0x04e0 },
};

// Ungroup 3 values stored in 5 bits (bap=1 mantissas): { i/9, (i%9)/3, (i%9)%3 }.
static const uint8_t ff_ac3_ungroup_3_in_5_bits_tab[32][3] = {
	{ 0,0,0 },{ 0,0,1 },{ 0,0,2 },{ 0,1,0 },{ 0,1,1 },{ 0,1,2 },{ 0,2,0 },{ 0,2,1 },
	{ 0,2,2 },{ 1,0,0 },{ 1,0,1 },{ 1,0,2 },{ 1,1,0 },{ 1,1,1 },{ 1,1,2 },{ 1,2,0 },
	{ 1,2,1 },{ 1,2,2 },{ 2,0,0 },{ 2,0,1 },{ 2,0,2 },{ 2,1,0 },{ 2,1,1 },{ 2,1,2 },
	{ 2,2,0 },{ 2,2,1 },{ 2,2,2 },{ 3,0,0 },{ 3,0,1 },{ 3,0,2 },{ 3,1,0 },{ 3,1,1 }
};

/*
	Dither RNG seed state.

	FFmpeg calls av_lfg_init(&dith_state, 0), which fills state[8..63] from a chain of
	MD5 rounds and leaves state[0..7] at zero (the decoder context is zeroed). Pulling
	MD5 in just to recompute a fixed constant would be silly, so the resulting state is
	embedded. It must match bit-for-bit, or dithered (bap=0) bins diverge from the
	reference decoder — including the zero prefix, which is an FFmpeg quirk, not a
	property of the A/52 dither.

	The chaining is the subtle part. av_lfg_init hashes a 16-byte scratch IN PLACE:

	    for (i = 8; i < 64; i += 4) {
	        AV_WL32(tmp, seed); tmp[4] = i;      // only the first 5 bytes are reset
	        av_md5_sum(tmp, tmp, 16);            // tmp[5..15] carry the PREVIOUS digest
	        state[i..i+3] = AV_RL32(tmp + 0,4,8,12);
	    }

	so every round but the first hashes a buffer whose tail is the previous round's
	output. The superseded JS decoder's copy of this table did not chain, so it agreed
	only on state[8..11] (the first round, where the scratch really is zero-filled) and
	was wrong from state[12] on. Its conformance test decoded with dither disabled, so
	nothing ever exercised the table. Regenerated here against av_lfg_init(0), and the
	AC-3 test now runs dither ON precisely so this stays covered.
*/
static const uint32_t LFG_INIT_STATE[64] = {
	0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
	993487710u, 4117489507u, 3416434684u, 2538713799u, 1528817306u, 2248869295u, 723931338u, 2460581808u,
	1930815723u, 2338490275u, 769382579u, 487165090u, 1785050232u, 3383897524u, 2520251343u, 3561126196u,
	3012616885u, 2460503346u, 554741894u, 631838767u, 596809201u, 3338159863u, 1549482577u, 3869751860u,
	3596186405u, 2771903455u, 3414222132u, 3941397212u, 2612751313u, 782810385u, 2349256288u, 1750707997u,
	4049204814u, 3256073478u, 710768517u, 3860978772u, 3100351118u, 3869544952u, 2245517093u, 2440927982u,
	2975253779u, 1540958685u, 4138326705u, 3287047978u, 1377116941u, 3803609047u, 1881028327u, 1409671879u,
	3721083618u, 2511890154u, 514931866u, 879118018u, 2342815014u, 3005606335u, 703626569u, 3381055288u,
};

// ─── runtime-generated tables (ac3dec_data.c / ac3dec_float.c) ───────────────
static uint8_t ff_ac3_ungroup_3_in_7_bits_tab[128][3];
static int ff_ac3_bap1_mantissas[32][3];
static int ff_ac3_bap2_mantissas[128][3];
static int ff_ac3_bap4_mantissas[128][2];
static int ff_ac3_bap3_mantissas[8];
static int ff_ac3_bap5_mantissas[16];
static float dynamic_range_tab[256];
static float heavy_dynamic_range_tab[256];
static int ac3_tables_ready = 0;

// A/52 7.3.3 symmetrical dequantization (Tables 7.19-7.23).
static inline int symmetric_dequant(int code, int levels)
{
	return ((code - (levels >> 1)) * (1 << 24)) / levels;
}

static void ac3_init_static(void)
{
	if (ac3_tables_ready) return;

	// ungrouping 3 values in 7 bits (exponents and bap=2 mantissas), A/52 7.1.3
	for (int i = 0; i < 128; ++i) {
		ff_ac3_ungroup_3_in_7_bits_tab[i][0] =  i / 25;
		ff_ac3_ungroup_3_in_7_bits_tab[i][1] = (i % 25) / 5;
		ff_ac3_ungroup_3_in_7_bits_tab[i][2] = (i % 25) % 5;
	}
	// grouped mantissa tables, A/52 7.3.5
	for (int i = 0; i < 32; ++i) {
		ff_ac3_bap1_mantissas[i][0] = symmetric_dequant(ff_ac3_ungroup_3_in_5_bits_tab[i][0], 3);
		ff_ac3_bap1_mantissas[i][1] = symmetric_dequant(ff_ac3_ungroup_3_in_5_bits_tab[i][1], 3);
		ff_ac3_bap1_mantissas[i][2] = symmetric_dequant(ff_ac3_ungroup_3_in_5_bits_tab[i][2], 3);
	}
	for (int i = 0; i < 128; ++i) {
		ff_ac3_bap2_mantissas[i][0] = symmetric_dequant(ff_ac3_ungroup_3_in_7_bits_tab[i][0], 5);
		ff_ac3_bap2_mantissas[i][1] = symmetric_dequant(ff_ac3_ungroup_3_in_7_bits_tab[i][1], 5);
		ff_ac3_bap2_mantissas[i][2] = symmetric_dequant(ff_ac3_ungroup_3_in_7_bits_tab[i][2], 5);
		ff_ac3_bap4_mantissas[i][0] = symmetric_dequant(i / 11, 11);
		ff_ac3_bap4_mantissas[i][1] = symmetric_dequant(i % 11, 11);
	}
	for (int i = 0; i < 7; ++i)  ff_ac3_bap3_mantissas[i] = symmetric_dequant(i, 7);
	for (int i = 0; i < 15; ++i) ff_ac3_bap5_mantissas[i] = symmetric_dequant(i, 15);

	// dynamic range tables, A/52 7.7.1 / 7.7.2
	for (int i = 0; i < 256; i++) {
		int v = (i >> 5) - ((i >> 7) << 3) - 5;
		dynamic_range_tab[i] = powf(2.0f, (float)v) * (float)((i & 0x1F) | 0x20);
	}
	for (int i = 0; i < 256; i++) {
		int v = (i >> 4) - ((i >> 7) << 4) - 4;
		heavy_dynamic_range_tab[i] = powf(2.0f, (float)v) * (float)((i & 0xF) | 0x10);
	}
	ac3_tables_ready = 1;
}

#endif /* AC3_TABLES_H */
