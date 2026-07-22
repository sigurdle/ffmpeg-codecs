/*
    MPEG-4 Part 2 ASP + H.263 coefficient/VLC tables — transcribed VERBATIM from
    FFmpeg release/8.1 (libavcodec/h263data.c + mpeg4data.h, LGPL). See
    mpeg4_decode.cpp for provenance. Only the INTRA milestone-1 set is here
    (intra MCBPC, CBPY, MPEG-4 intra DC-size VLC, MPEG-4 intra DCT3D RL, the DC
    scale/threshold tables, the default quant matrices); inter (P/B) tables are
    added at M2. The flat VLC LUTs are built once by mp4_init_tables().
*/

#ifndef MPEG4_MP4_TABLES_H
#define MPEG4_MP4_TABLES_H

#include <stdint.h>
#include "mp4_vlc.h"

namespace mp4 {

// ── intra MCBPC (h263data.c:32) — mb_type=(intra),(intraq); idx 8 = stuffing ──
static const uint8_t ff_h263_intra_MCBPC_code[9] = { 1, 1, 2, 3, 1, 1, 2, 3, 1 };
static const uint8_t ff_h263_intra_MCBPC_bits[9] = { 1, 3, 3, 3, 4, 6, 6, 6, 9 };

// ── inter MCBPC (h263data.c:38) — &3=cbpc,&4=intra,&8=dquant,&16=4MV; 20=stuffing ──
static const uint8_t ff_h263_inter_MCBPC_code[28] = {
    1,  3,  2,  5,  3,  4,  3,  3,  3,  7,  6,  5,  4,  4,  3,  2,
    2,  5,  4,  5,  1,  0,  0,  0,  2, 12, 14, 15,
};
static const uint8_t ff_h263_inter_MCBPC_bits[28] = {
    1,  4,  4,  6,  5,  8,  8,  7,  3,  7,  7,  9,  6,  9,  9,  9,
    3,  7,  7,  8,  9,  0,  0,  0, 11, 13, 13, 13,
};

// ── MV residual VLC (h263data.c:88) — symbol = abs delta magnitude 0..32 ──
static const uint8_t ff_mvtab[33][2] = {
    {  1,  1 }, {  1,  2 }, {  1,  3 }, {  1,  4 }, {  3,  6 }, {  5,  7 },
    {  4,  7 }, {  3,  7 }, { 11,  9 }, { 10,  9 }, {  9,  9 }, { 17, 10 },
    { 16, 10 }, { 15, 10 }, { 14, 10 }, { 13, 10 }, { 12, 10 }, { 11, 10 },
    { 10, 10 }, {  9, 10 }, {  8, 10 }, {  7, 10 }, {  6, 10 }, {  5, 10 },
    {  4, 10 }, {  7, 11 }, {  6, 11 }, {  5, 11 }, {  4, 11 }, {  3, 11 },
    {  2, 11 }, {  3, 12 }, {  2, 12 }
};

// ── inter DCT3D RL (h263data.c:98/127/143) — n=102, last=58 ──
static const uint16_t ff_inter_vlc[103][2] = {
    {  0x2,  2 }, {  0xf,  4 }, { 0x15,  6 }, { 0x17,  7 }, { 0x1f,  8 }, { 0x25,  9 }, { 0x24,  9 }, { 0x21, 10 },
    { 0x20, 10 }, {  0x7, 11 }, {  0x6, 11 }, { 0x20, 11 }, {  0x6,  3 }, { 0x14,  6 }, { 0x1e,  8 }, {  0xf, 10 },
    { 0x21, 11 }, { 0x50, 12 }, {  0xe,  4 }, { 0x1d,  8 }, {  0xe, 10 }, { 0x51, 12 }, {  0xd,  5 }, { 0x23,  9 },
    {  0xd, 10 }, {  0xc,  5 }, { 0x22,  9 }, { 0x52, 12 }, {  0xb,  5 }, {  0xc, 10 }, { 0x53, 12 }, { 0x13,  6 },
    {  0xb, 10 }, { 0x54, 12 }, { 0x12,  6 }, {  0xa, 10 }, { 0x11,  6 }, {  0x9, 10 }, { 0x10,  6 }, {  0x8, 10 },
    { 0x16,  7 }, { 0x55, 12 }, { 0x15,  7 }, { 0x14,  7 }, { 0x1c,  8 }, { 0x1b,  8 }, { 0x21,  9 }, { 0x20,  9 },
    { 0x1f,  9 }, { 0x1e,  9 }, { 0x1d,  9 }, { 0x1c,  9 }, { 0x1b,  9 }, { 0x1a,  9 }, { 0x22, 11 }, { 0x23, 11 },
    { 0x56, 12 }, { 0x57, 12 }, {  0x7,  4 }, { 0x19,  9 }, {  0x5, 11 }, {  0xf,  6 }, {  0x4, 11 }, {  0xe,  6 },
    {  0xd,  6 }, {  0xc,  6 }, { 0x13,  7 }, { 0x12,  7 }, { 0x11,  7 }, { 0x10,  7 }, { 0x1a,  8 }, { 0x19,  8 },
    { 0x18,  8 }, { 0x17,  8 }, { 0x16,  8 }, { 0x15,  8 }, { 0x14,  8 }, { 0x13,  8 }, { 0x18,  9 }, { 0x17,  9 },
    { 0x16,  9 }, { 0x15,  9 }, { 0x14,  9 }, { 0x13,  9 }, { 0x12,  9 }, { 0x11,  9 }, {  0x7, 10 }, {  0x6, 10 },
    {  0x5, 10 }, {  0x4, 10 }, { 0x24, 11 }, { 0x25, 11 }, { 0x26, 11 }, { 0x27, 11 }, { 0x58, 12 }, { 0x59, 12 },
    { 0x5a, 12 }, { 0x5b, 12 }, { 0x5c, 12 }, { 0x5d, 12 }, { 0x5e, 12 }, { 0x5f, 12 }, {  0x3,  7 },
};
static const int8_t ff_inter_level[102] = {
    1,  2,  3,  4, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4,
    5,  6,  1,  2, 3, 4, 1, 2, 3,  1,  2,  3, 1, 2, 3, 1,
    2,  3,  1,  2, 1, 2, 1, 2, 1,  2,  1,  1, 1, 1, 1, 1,
    1,  1,  1,  1, 1, 1, 1, 1, 1,  1,  1,  2, 3, 1, 2, 1,
    1,  1,  1,  1, 1, 1, 1, 1, 1,  1,  1,  1, 1, 1, 1, 1,
    1,  1,  1,  1, 1, 1, 1, 1, 1,  1,  1,  1, 1, 1, 1, 1,
    1,  1,  1,  1, 1, 1,
};
static const int8_t ff_inter_run[102] = {
    0,   0,  0,  0,  0,  0,  0,  0,  0,   0,  0,  0,  1,  1,  1,  1,
    1,   1,  2,  2,  2,  2,  3,  3,  3,   4,  4,  4,  5,  5,  5,  6,
    6,   6,  7,  7,  8,  8,  9,  9, 10,  10, 11, 12, 13, 14, 15, 16,
    17, 18, 19, 20, 21, 22, 23, 24, 25,  26,  0,  0,  0,  1,  1,  2,
    3,   4,  5,  6,  7,  8,  9, 10, 11,  12, 13, 14, 15, 16, 17, 18,
    19, 20, 21, 22, 23, 24, 25, 26, 27,  28, 29, 30, 31, 32, 33, 34,
    35, 36, 37, 38, 39, 40,
};
#define H263_RL_INTER_N    102
#define H263_RL_INTER_LAST  58

// ── CBPY (h263data.c:82) — {code, len} ──
static const uint8_t ff_h263_cbpy_tab[16][2] = {
    {  3, 4 }, { 5, 5 }, { 4, 5 }, { 9, 4 }, {  3, 5 }, { 7, 4 }, { 2, 6 },
    { 11, 4 }, { 2, 5 }, { 3, 6 }, { 5, 4 }, { 10, 4 }, { 4, 4 }, { 8, 4 },
    {  6, 4 }, { 3, 2 }
};

// ── MPEG-4 intra DC-size VLC (mpeg4data.h:34/40) — {code, len}; nb_codes=10 ──
static const uint8_t ff_mpeg4_DCtab_lum[13][2] = {
    {3,3}, {3,2}, {2,2}, {2,3}, {1,3}, {1,4}, {1,5}, {1,6}, {1,7},
    {1,8}, {1,9}, {1,10}, {1,11},
};
static const uint8_t ff_mpeg4_DCtab_chrom[13][2] = {
    {3,2}, {2,2}, {1,2}, {1,3}, {1,4}, {1,5}, {1,6}, {1,7}, {1,8},
    {1,9}, {1,10}, {1,11}, {1,12},
};

// ── MPEG-4 intra DCT3D RL (mpeg4data.h:46) — n=102, last=67 ──
static const uint16_t ff_mpeg4_intra_vlc[103][2] = {
{ 0x2, 2 },
{ 0x6, 3 },{ 0xf, 4 },{ 0xd, 5 },{ 0xc, 5 },
{ 0x15, 6 },{ 0x13, 6 },{ 0x12, 6 },{ 0x17, 7 },
{ 0x1f, 8 },{ 0x1e, 8 },{ 0x1d, 8 },{ 0x25, 9 },
{ 0x24, 9 },{ 0x23, 9 },{ 0x21, 9 },{ 0x21, 10 },
{ 0x20, 10 },{ 0xf, 10 },{ 0xe, 10 },{ 0x7, 11 },
{ 0x6, 11 },{ 0x20, 11 },{ 0x21, 11 },{ 0x50, 12 },
{ 0x51, 12 },{ 0x52, 12 },{ 0xe, 4 },{ 0x14, 6 },
{ 0x16, 7 },{ 0x1c, 8 },{ 0x20, 9 },{ 0x1f, 9 },
{ 0xd, 10 },{ 0x22, 11 },{ 0x53, 12 },{ 0x55, 12 },
{ 0xb, 5 },{ 0x15, 7 },{ 0x1e, 9 },{ 0xc, 10 },
{ 0x56, 12 },{ 0x11, 6 },{ 0x1b, 8 },{ 0x1d, 9 },
{ 0xb, 10 },{ 0x10, 6 },{ 0x22, 9 },{ 0xa, 10 },
{ 0xd, 6 },{ 0x1c, 9 },{ 0x8, 10 },{ 0x12, 7 },
{ 0x1b, 9 },{ 0x54, 12 },{ 0x14, 7 },{ 0x1a, 9 },
{ 0x57, 12 },{ 0x19, 8 },{ 0x9, 10 },{ 0x18, 8 },
{ 0x23, 11 },{ 0x17, 8 },{ 0x19, 9 },{ 0x18, 9 },
{ 0x7, 10 },{ 0x58, 12 },{ 0x7, 4 },{ 0xc, 6 },
{ 0x16, 8 },{ 0x17, 9 },{ 0x6, 10 },{ 0x5, 11 },
{ 0x4, 11 },{ 0x59, 12 },{ 0xf, 6 },{ 0x16, 9 },
{ 0x5, 10 },{ 0xe, 6 },{ 0x4, 10 },{ 0x11, 7 },
{ 0x24, 11 },{ 0x10, 7 },{ 0x25, 11 },{ 0x13, 7 },
{ 0x5a, 12 },{ 0x15, 8 },{ 0x5b, 12 },{ 0x14, 8 },
{ 0x13, 8 },{ 0x1a, 8 },{ 0x15, 9 },{ 0x14, 9 },
{ 0x13, 9 },{ 0x12, 9 },{ 0x11, 9 },{ 0x26, 11 },
{ 0x27, 11 },{ 0x5c, 12 },{ 0x5d, 12 },{ 0x5e, 12 },
{ 0x5f, 12 },{ 0x3, 7 },
};

static const int8_t ff_mpeg4_intra_level[102] = {
  1,  2,  3,  4,  5,  6,  7,  8,
  9, 10, 11, 12, 13, 14, 15, 16,
 17, 18, 19, 20, 21, 22, 23, 24,
 25, 26, 27,  1,  2,  3,  4,  5,
  6,  7,  8,  9, 10,  1,  2,  3,
  4,  5,  1,  2,  3,  4,  1,  2,
  3,  1,  2,  3,  1,  2,  3,  1,
  2,  3,  1,  2,  1,  2,  1,  1,
  1,  1,  1,  1,  2,  3,  4,  5,
  6,  7,  8,  1,  2,  3,  1,  2,
  1,  2,  1,  2,  1,  2,  1,  2,
  1,  1,  1,  1,  1,  1,  1,  1,
  1,  1,  1,  1,  1,  1,
};

static const int8_t ff_mpeg4_intra_run[102] = {
  0,  0,  0,  0,  0,  0,  0,  0,
  0,  0,  0,  0,  0,  0,  0,  0,
  0,  0,  0,  0,  0,  0,  0,  0,
  0,  0,  0,  1,  1,  1,  1,  1,
  1,  1,  1,  1,  1,  2,  2,  2,
  2,  2,  3,  3,  3,  3,  4,  4,
  4,  5,  5,  5,  6,  6,  6,  7,
  7,  7,  8,  8,  9,  9, 10, 11,
 12, 13, 14,  0,  0,  0,  0,  0,
  0,  0,  0,  1,  1,  1,  2,  2,
  3,  3,  4,  4,  5,  5,  6,  6,
  7,  8,  9, 10, 11, 12, 13, 14,
 15, 16, 17, 18, 19, 20,
};

#define MPEG4_RL_INTRA_N    102
#define MPEG4_RL_INTRA_LAST  67

// ── DC scale / threshold (mpeg4data.h:356/360/365) ──
static const uint8_t ff_mpeg4_y_dc_scale_table[32] = {
    0, 8, 8, 8, 8,10,12,14,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,34,36,38,40,42,44,46
};
static const uint8_t ff_mpeg4_c_dc_scale_table[32] = {
    0, 8, 8, 8, 8, 9, 9,10,10,11,11,12,12,13,13,14,14,15,15,16,16,17,17,18,18,19,20,21,22,23,24,25
};
static const uint8_t ff_mpeg4_dc_threshold[8] = { 99, 13, 15, 17, 19, 21, 23, 0 };

// ── Default quant matrices (mpeg4data.h:334/345), raster order ──
static const int16_t ff_mpeg4_default_intra_matrix[64] = {
  8, 17, 18, 19, 21, 23, 25, 27,
 17, 18, 19, 21, 23, 25, 27, 28,
 20, 21, 22, 23, 24, 26, 28, 30,
 21, 22, 23, 24, 26, 28, 30, 32,
 22, 23, 24, 26, 28, 30, 32, 35,
 23, 24, 26, 28, 30, 32, 35, 38,
 25, 26, 28, 30, 32, 35, 38, 41,
 27, 28, 30, 32, 35, 38, 41, 45,
};
static const int16_t ff_mpeg4_default_non_intra_matrix[64] = {
 16, 17, 18, 19, 20, 21, 22, 23,
 17, 18, 19, 20, 21, 22, 23, 24,
 18, 19, 20, 21, 22, 23, 24, 25,
 19, 20, 21, 22, 23, 24, 26, 27,
 20, 21, 22, 23, 25, 26, 27, 28,
 21, 22, 23, 24, 26, 27, 28, 30,
 22, 23, 24, 26, 27, 28, 30, 31,
 23, 24, 25, 27, 28, 30, 31, 33,
};

// ── Coefficient scan tables (mathtables.c:137, mpegvideodata.c:52/63) ──
static const uint8_t ff_zigzag_direct[64] = {
    0,   1,  8, 16,  9,  2,  3, 10,
   17, 24, 32, 25, 18, 11,  4,  5,
   12, 19, 26, 33, 40, 48, 41, 34,
   27, 20, 13,  6,  7, 14, 21, 28,
   35, 42, 49, 56, 57, 50, 43, 36,
   29, 22, 15, 23, 30, 37, 44, 51,
   58, 59, 52, 45, 38, 31, 39, 46,
   53, 60, 61, 54, 47, 55, 62, 63
};
static const uint8_t ff_alternate_horizontal_scan[64] = {
    0,  1,  2,  3,  8,  9, 16, 17,
   10, 11,  4,  5,  6,  7, 15, 14,
   13, 12, 19, 18, 24, 25, 32, 33,
   26, 27, 20, 21, 22, 23, 28, 29,
   30, 31, 34, 35, 40, 41, 48, 49,
   42, 43, 36, 37, 38, 39, 44, 45,
   46, 47, 50, 51, 56, 57, 58, 59,
   52, 53, 54, 55, 60, 61, 62, 63,
};
static const uint8_t ff_alternate_vertical_scan[64] = {
    0,  8, 16, 24,  1,  9,  2, 10,
   17, 25, 32, 40, 48, 56, 57, 49,
   41, 33, 26, 18,  3, 11,  4, 12,
   19, 27, 34, 42, 50, 58, 35, 43,
   51, 59, 20, 28,  5, 13,  6, 14,
   21, 29, 36, 44, 52, 60, 37, 45,
   53, 61, 22, 30,  7, 15, 23, 31,
   38, 46, 54, 62, 39, 47, 55, 63,
};

// ── Built flat VLC LUTs (sized to each table's max code length) ──
static VLCElem      s_intra_mcbpc[1 << 9];
static VLCElem      s_inter_mcbpc[1 << 13];
static VLCElem      s_cbpy[1 << 6];
static VLCElem      s_mv[1 << 12];
static VLCElem      s_dc_lum[1 << 8];
static VLCElem      s_dc_chrom[1 << 9];
static RL_VLC_ELEM  s_rl_intra[1 << 12];
static RL_VLC_ELEM  s_rl_inter[32][1 << 12];   // one per quant (MPEG-4 indexes by qscale)

static VlcTab g_intra_mcbpc;
static VlcTab g_inter_mcbpc;
static VlcTab g_cbpy;
static VlcTab g_mv;
static VlcTab g_dc_lum;
static VlcTab g_dc_chrom;
static RlTab  g_rl_intra;
static RlTab  g_rl_inter[32];

static bool g_tables_inited = false;

static void mp4_init_tables() {
    if (g_tables_inited) return;

    int b = mp4_max_len(ff_h263_intra_MCBPC_bits, 9, 1, 1);
    build_flat_vlc(s_intra_mcbpc, b, 9,
                   ff_h263_intra_MCBPC_code, 1, 1,
                   ff_h263_intra_MCBPC_bits, 1, 1, NULL);
    g_intra_mcbpc.table = s_intra_mcbpc; g_intra_mcbpc.bits = b;

    b = mp4_max_len(&ff_h263_cbpy_tab[0][1], 16, 2, 1);
    build_flat_vlc(s_cbpy, b, 16,
                   &ff_h263_cbpy_tab[0][0], 2, 1,
                   &ff_h263_cbpy_tab[0][1], 2, 1, NULL);
    g_cbpy.table = s_cbpy; g_cbpy.bits = b;

    b = mp4_max_len(&ff_mpeg4_DCtab_lum[0][1], 10, 2, 1);
    build_flat_vlc(s_dc_lum, b, 10,
                   &ff_mpeg4_DCtab_lum[0][0], 2, 1,
                   &ff_mpeg4_DCtab_lum[0][1], 2, 1, NULL);
    g_dc_lum.table = s_dc_lum; g_dc_lum.bits = b;

    b = mp4_max_len(&ff_mpeg4_DCtab_chrom[0][1], 10, 2, 1);
    build_flat_vlc(s_dc_chrom, b, 10,
                   &ff_mpeg4_DCtab_chrom[0][0], 2, 1,
                   &ff_mpeg4_DCtab_chrom[0][1], 2, 1, NULL);
    g_dc_chrom.table = s_dc_chrom; g_dc_chrom.bits = b;

    build_rl(&g_rl_intra, s_rl_intra, MPEG4_RL_INTRA_N, MPEG4_RL_INTRA_LAST,
             ff_mpeg4_intra_vlc, ff_mpeg4_intra_run, ff_mpeg4_intra_level,
             /*qmul*/ 1, /*qadd*/ 0);

    // ── inter (P-VOP) tables ──
    b = mp4_max_len(ff_h263_inter_MCBPC_bits, 28, 1, 1);
    build_flat_vlc(s_inter_mcbpc, b, 28,
                   ff_h263_inter_MCBPC_code, 1, 1,
                   ff_h263_inter_MCBPC_bits, 1, 1, NULL);
    g_inter_mcbpc.table = s_inter_mcbpc; g_inter_mcbpc.bits = b;

    b = mp4_max_len(&ff_mvtab[0][1], 33, 2, 1);
    build_flat_vlc(s_mv, b, 33, &ff_mvtab[0][0], 2, 1, &ff_mvtab[0][1], 2, 1, NULL);
    g_mv.table = s_mv; g_mv.bits = b;

    // ff_h263_rl_inter: all 32 quant copies (rl_vlc[q]; q==0 -> qmul=1/qadd=0).
    for (int q = 0; q < 32; q++) {
        int qmul = q == 0 ? 1 : q * 2;
        int qadd = q == 0 ? 0 : (q - 1) | 1;
        build_rl(&g_rl_inter[q], s_rl_inter[q], H263_RL_INTER_N, H263_RL_INTER_LAST,
                 ff_inter_vlc, ff_inter_run, ff_inter_level, qmul, qadd);
    }

    g_tables_inited = true;
}

} // namespace mp4

#endif // MPEG4_MP4_TABLES_H
