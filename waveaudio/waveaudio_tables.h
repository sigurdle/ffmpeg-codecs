/*
	Tables of FFmpeg's ADPCM decoders, n8.1: libavcodec/adpcm_data.c, adpcm_data.h and the
	index tables of adpcm.c. LGPL-2.1+, as waveaudio_decode.cpp's header says.
*/

#pragma once

#include <stdint.h>

static const int8_t adpcm_index_table[16] = {
	-1, -1, -1, -1, 2, 4, 6, 8,
	-1, -1, -1, -1, 2, 4, 6, 8,
};

static const int8_t adpcm_index_table2[4] = {
	-1, 2,
	-1, 2,
};

static const int8_t adpcm_index_table3[8] = {
	-1, -1, 1, 2,
	-1, -1, 1, 2,
};

static const int8_t adpcm_index_table5[32] = {
	-1, -1, -1, -1, -1, -1, -1, -1, 1, 2, 4, 6, 8, 10, 13, 16,
	-1, -1, -1, -1, -1, -1, -1, -1, 1, 2, 4, 6, 8, 10, 13, 16,
};

// By bits per sample, from 2.
static const int8_t* const adpcm_index_tables[4] = {
	adpcm_index_table2,
	adpcm_index_table3,
	adpcm_index_table,
	adpcm_index_table5,
};

static const int16_t adpcm_step_table[89] = {
	    7,     8,     9,    10,    11,    12,    13,    14,    16,    17,
	   19,    21,    23,    25,    28,    31,    34,    37,    41,    45,
	   50,    55,    60,    66,    73,    80,    88,    97,   107,   118,
	  130,   143,   157,   173,   190,   209,   230,   253,   279,   307,
	  337,   371,   408,   449,   494,   544,   598,   658,   724,   796,
	  876,   963,  1060,  1166,  1282,  1411,  1552,  1707,  1878,  2066,
	 2272,  2499,  2749,  3024,  3327,  3660,  4026,  4428,  4871,  5358,
	 5894,  6484,  7132,  7845,  8630,  9493, 10442, 11487, 12635, 13899,
	15289, 16818, 18500, 20350, 22385, 24623, 27086, 29794, 32767,
};

static const int16_t adpcm_AdaptationTable[16] = {
	230, 230, 230, 230, 307, 409, 512, 614,
	768, 614, 512, 409, 307, 230, 230, 230,
};

// Divided by 4 to fit in 8 bits.
static const uint8_t adpcm_AdaptCoeff1[7] = { 64, 128, 0, 48, 60, 115, 98 };
static const int8_t adpcm_AdaptCoeff2[7] = { 0, -64, 0, 16, 0, -52, -58 };

// By bits per sample, from 2: the bytes of a channel's run and the samples they hold.
static const uint8_t adpcm_ima_block_sizes[4] = { 4, 12, 4, 20 };
static const uint8_t adpcm_ima_block_samples[4] = { 16, 32, 8, 32 };
