/*
    Microsoft MPEG-4 / Windows Media Video 7-8 (WMV1/WMV2, a.k.a. msmpeg4 v1-v3 +
    WMV1/WMV2) video decoder — C++/WebAssembly. From-source TRANSCRIPTION of FFmpeg
    release/8.1's msmpeg4 decoder (libavcodec/msmpeg4dec.c, msmpeg4.c,
    msmpeg4data.c, msmpeg4_vc1_data.c) and, for WMV2, wmv2dec.c / wmv2dsp.c /
    intrax8.c / intrax8dsp.c / h263.c (loop filter) — see wmv2_decode.h. FFmpeg is
    LGPL-2.1+; this is an LGPL-derived work in its OWN wasm binary. Conformance oracle
    is the ffmpeg CLI (`ffmpeg -cpuflags 0 -i sample -f rawvideo -pix_fmt yuv420p`),
    NOT any ffmpeg-derived JS.

    msmpeg4 is "MPEG-4 with a twist": it reuses the H.263/mpegvideo block machinery
    (8x8 DCT blocks, H.263 dequant, DC/AC intra prediction, the same macroblock grid
    geometry) but with Microsoft's own picture header, VLC tables, DC prediction and
    coefficient escapes. This module REUSES the kernels the sibling mpeg4 port ships —
    get_bits (mp4_bits.h), the VLC/RL runtime (mp4_vlc.h) and simple_idct8.h — plus
    that port's grid layout, and adds the msmpeg4 layer.

    This file holds the WMV1 picture header and the I/P macroblock layer that WMV2
    shares (block decode, DC/AC prediction, MV decode, half-pel MC). WMV2 replaces
    the picture header and MB layer, the IDCT, and adds mspel MC, ABT, the H.263 loop
    filter and IntraX8 J-frames (wmv2_decode.h, wmv2_dsp.h, intrax8.h).

    -fwrapv is REQUIRED (dequant / sign arithmetic), same as the mpeg4 port.
*/

#ifndef MSMPEG4_DECODE_H
#define MSMPEG4_DECODE_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "../mpeg4/mp4_bits.h"       // get_bits reader (namespace mp4)
#include "../mpeg4/simple_idct8.h"   // mp4::idct8_put
#include "msmpeg4_tables.h"          // msm:: tables + VLC/RL builders
#include "wmv2_dsp.h"                // WMV2 IDCT / ABT / mspel / H.263 loop filter
#include "intrax8.h"                 // IntraX8 (WMV2 J-frames)

namespace msm {

// FFmpeg msmpeg4_version (mpegvideo.h): 0 unused, then v1..v3, WMV1, WMV2.
enum { MSMP4_UNUSED = 0, MSMP4_V1 = 1, MSMP4_V2 = 2, MSMP4_V3 = 3, MSMP4_WMV1 = 4, MSMP4_WMV2 = 5 };
enum { PICT_I = 1, PICT_P = 2 };
enum { MBAC_BITRATE = 50 * 1024, II_BITRATE = 128 * 1024, DC_MAX = 119 };
enum { MSM_OK = 0, MSM_ERR = -1 };

struct MsmDecoder {
    int version;
    int width, height;
    int mb_width, mb_height, mb_stride, b8_stride;

    // Per-picture header state.
    int pict_type, qscale, slice_height, no_rounding;
    int rl_table_index, rl_chroma_table_index, dc_table_index, mv_table_index;
    int use_skip_mb_code, per_mb_rl_table, inter_intra_pred;
    int bit_rate, flipflop_rounding;
    int esc3_level_length, esc3_run_length;
    int y_dc_scale, c_dc_scale;

    // Per-MB state.
    int mb_x, mb_y, first_slice_line, ac_pred;
    int mb_intra, mb_skipped, h263_aic_dir, cur_mvx, cur_mvy;
    int block_wrap[6], block_index[6], block_last_index[6];
    int16_t block[6][64];

    // Frame-persistent prediction grids (mpeg4-port geometry: dc_val int16 grid,
    // ac_val 16 int16 per block, coded_block uint8 grid, motion_val 2 int16 per
    // block — all offset by lead+dc_off so edge OOB neighbor reads land in a guard).
    int dc_off, grid_elems, grid_lead;
    int16_t *dc_base, *dc_val;
    int16_t *ac_base, *ac_val;
    int16_t *mot_base, *mot_val;
    uint8_t *coded_base, *coded_block;

    // Scan tables (PERM_NONE: used verbatim) + intra raster-end for dequant bound.
    uint8_t intra_scan[64], inter_scan[64], h_scan[64], v_scan[64], intra_raster_end[64];

    // Output planes (mb-aligned). yStride = codedW, cStride = codedW/2. ref* hold the
    // previous decoded frame (P-frame motion-compensation source); swapped per frame.
    int codedW, codedH, yStride, cStride;
    uint8_t *y, *u, *v, *refY, *refU, *refV;
    size_t yCap, cCap;

    // Built-once decode tables.
    int tables_ready;
    mp4::RlTab rl[6];                 // intra rl_vlc[0] (qmul=1) for the 6 ff_rl_table entries
    mp4::RlTab rl_inter[3];           // ff_rl_table[3..5] with per-P-frame qscale dequant baked
    mp4::VlcTab mb_i;                 // I-frame MB-type VLC
    mp4::VlcTab mv_tab[2];            // P-frame motion-vector VLCs (mv_table_index)
    mp4::VlcTab inter_intra;         // h263_aic_dir VLC (inter_intra_pred)
    int inter_qscale;                 // qscale rl_inter was last built for

    // WMV2 (version == MSMP4_WMV2). Extradata flags (decode_ext_header), then the
    // per-picture / per-MB state of wmv2dec.c's WMV2DecContext.
    int mspel_bit, loop_filter, abt_flag, j_type_bit, top_left_mv_flag, per_mb_rl_bit;
    int j_type, mspel, cbp_table_index, hshift;
    int abt_type, per_mb_abt, per_block_abt, abt_type_table[6];
    int16_t abt_block2[6][64];
    uint8_t* mb_skip;                 // per-MB skip flag (MB_TYPE_SKIP), mb_stride * mb_height
    IntraX8Context x8;
};

// ff_h263-style ternary field: 0 -> 1 bit '0'; else 1 bit '1' + 1 bit (+1).
static inline int decode012(mp4::GetBits* gb) {
    int n = mp4::get_bits1(gb);
    if (n == 0) return 0;
    return mp4::get_bits1(gb) + 1;
}

// ── table build (once) ──────────────────────────────────────────────────────────
static inline void msm_init_tables(MsmDecoder* d) {
    if (d->tables_ready) return;
    static mp4::RL_VLC_ELEM rl_store[6][1 << 15];   // sized to the 15-bit max code
    static mp4::VLCElem mb_i_store[1 << 13];

    // ff_rl_table order: [0]=table0 [1]=table2 [2]=mpeg4_intra [3]=table1 [4]=table4 [5]=inter.
    mp4::build_rl(&d->rl[0], rl_store[0], 132,  85, table0_vlc, table0_run, table0_level, 1, 0);
    mp4::build_rl(&d->rl[1], rl_store[1], 185, 119, table2_vlc, table2_run, table2_level, 1, 0);
    mp4::build_rl(&d->rl[2], rl_store[2], 102,  67, ff_mpeg4_intra_vlc, ff_mpeg4_intra_run, ff_mpeg4_intra_level, 1, 0);
    mp4::build_rl(&d->rl[3], rl_store[3], 148,  81, table1_vlc, table1_run, table1_level, 1, 0);
    mp4::build_rl(&d->rl[4], rl_store[4], 168,  99, table4_vlc, table4_run, table4_level, 1, 0);
    mp4::build_rl(&d->rl[5], rl_store[5], 102,  58, ff_inter_vlc, ff_inter_run, ff_inter_level, 1, 0);

    int mbbits = mp4::mp4_max_len(&ff_msmp4_mb_i_table[0][1], 64, 4, 2);
    mp4::build_flat_vlc(mb_i_store, mbbits, 64,
                        &ff_msmp4_mb_i_table[0][0], 4, 2,
                        &ff_msmp4_mb_i_table[0][1], 4, 2, NULL);
    d->mb_i.table = mb_i_store; d->mb_i.bits = mbbits;

    // P-frame motion-vector VLCs (built canonically from code lengths).
    static mp4::VLCElem mv0_store[1 << 17], mv1_store[1 << 17];
    msm_build_vlc_from_lengths(mv0_store, 17, MSMPEG4_MV_TABLES_NB_ELEMS, ff_msmp4_mv_table0_lens, ff_msmp4_mv_table0);
    msm_build_vlc_from_lengths(mv1_store, 17, MSMPEG4_MV_TABLES_NB_ELEMS, ff_msmp4_mv_table1_lens, ff_msmp4_mv_table1);
    d->mv_tab[0].table = mv0_store; d->mv_tab[0].bits = 17;
    d->mv_tab[1].table = mv1_store; d->mv_tab[1].bits = 17;

    static mp4::VLCElem ii_store[8];
    mp4::build_flat_vlc(ii_store, 3, 4, &ff_table_inter_intra[0][0], 2, 1, &ff_table_inter_intra[0][1], 2, 1, NULL);
    d->inter_intra.table = ii_store; d->inter_intra.bits = 3;

    d->tables_ready = 1;
}

// Rebuild the inter RL tables (ff_rl_table[3..5]) with the frame's qscale baked into
// the dequant (qmul=2q, qadd=(q-1)|1) — ffmpeg unquantizes inter coeffs during parse.
static inline void msm_build_inter_rl(MsmDecoder* d, int qscale) {
    if (d->inter_qscale == qscale) return;
    static mp4::RL_VLC_ELEM rli_store[3][1 << 15];
    int qmul = qscale << 1, qadd = (qscale - 1) | 1;
    mp4::build_rl(&d->rl_inter[0], rli_store[0], 148,  81, table1_vlc, table1_run, table1_level, qmul, qadd);
    mp4::build_rl(&d->rl_inter[1], rli_store[1], 168,  99, table4_vlc, table4_run, table4_level, qmul, qadd);
    mp4::build_rl(&d->rl_inter[2], rli_store[2], 102,  58, ff_inter_vlc, ff_inter_run, ff_inter_level, qmul, qadd);
    d->inter_qscale = qscale;
}

// ── geometry (mirror of the mpeg4 port's mp4_alloc / block_index) ────────────────
static inline void msm_make_raster_end(const uint8_t* scan, uint8_t* raster_end) {
    int end = -1;
    for (int i = 0; i < 64; i++) { if (scan[i] > end) end = scan[i]; raster_end[i] = (uint8_t)end; }
}

static inline int msm_alloc(MsmDecoder* d) {
    d->mb_width  = (d->width + 15) >> 4;
    d->mb_height = (d->height + 15) >> 4;
    d->mb_stride = d->mb_width + 1;
    d->b8_stride = d->mb_width * 2 + 1;
    d->codedW = d->mb_width * 16;  d->codedH = d->mb_height * 16;
    d->yStride = d->codedW;        d->cStride = d->codedW >> 1;

    size_t yNeed = (size_t)d->yStride * d->codedH;
    size_t cNeed = (size_t)d->cStride * (d->codedH >> 1);
    if (yNeed > d->yCap) { free(d->y); free(d->refY); d->y = (uint8_t*)malloc(yNeed); d->refY = (uint8_t*)malloc(yNeed);
                           if (!d->y || !d->refY) return MSM_ERR; d->yCap = yNeed; }
    if (cNeed > d->cCap) { free(d->u); free(d->v); free(d->refU); free(d->refV);
                           d->u = (uint8_t*)malloc(cNeed); d->v = (uint8_t*)malloc(cNeed);
                           d->refU = (uint8_t*)malloc(cNeed); d->refV = (uint8_t*)malloc(cNeed);
                           if (!d->u || !d->v || !d->refU || !d->refV) return MSM_ERR; d->cCap = cNeed; }

    int y_size  = d->b8_stride * (2 * d->mb_height + 1);
    int c_size  = d->mb_stride * (d->mb_height + 1);
    int yc_size = y_size + 2 * c_size;
    d->dc_off   = d->b8_stride + 1;
    int lead    = 2 * d->b8_stride + 16;
    int total   = lead + yc_size + (2 * d->b8_stride + 16);
    if (total != d->grid_elems) {
        free(d->dc_base); free(d->ac_base); free(d->coded_base); free(d->mot_base);
        d->dc_base    = (int16_t*)malloc((size_t)total * sizeof(int16_t));
        d->ac_base    = (int16_t*)malloc((size_t)total * 16 * sizeof(int16_t));
        d->mot_base   = (int16_t*)malloc((size_t)total * 2 * sizeof(int16_t));
        d->coded_base = (uint8_t*)malloc((size_t)total);
        if (!d->dc_base || !d->ac_base || !d->mot_base || !d->coded_base) return MSM_ERR;
        d->grid_elems = total; d->grid_lead = lead;
    }
    d->dc_val      = d->dc_base    + lead + d->dc_off;
    d->ac_val      = d->ac_base    + (size_t)(lead + d->dc_off) * 16;
    d->mot_val     = d->mot_base   + (size_t)(lead + d->dc_off) * 2;
    d->coded_block = d->coded_base + lead + d->dc_off;

    if (d->version == MSMP4_WMV2) {
        free(d->mb_skip);
        d->mb_skip = (uint8_t*)calloc((size_t)d->mb_stride * d->mb_height, 1);
        if (!d->mb_skip || x8_common_init(&d->x8, d->mb_width, d->mb_height) < 0) return MSM_ERR;
    }

    d->block_wrap[0] = d->block_wrap[1] = d->block_wrap[2] = d->block_wrap[3] = d->b8_stride;
    d->block_wrap[4] = d->block_wrap[5] = d->mb_stride;

    msm_init_tables(d);
    // WMV1 scans: [0]=inter [1]=intra [2]=intra-h [3]=intra-v (msmpeg4_common_init).
    memcpy(d->inter_scan, ff_wmv1_scantable[0], 64);
    memcpy(d->intra_scan, ff_wmv1_scantable[1], 64);
    memcpy(d->h_scan,     ff_wmv1_scantable[2], 64);
    memcpy(d->v_scan,     ff_wmv1_scantable[3], 64);
    msm_make_raster_end(d->intra_scan, d->intra_raster_end);
    return MSM_OK;
}

static inline void msm_init_block_index(MsmDecoder* d) {
    int x = d->mb_x, y = d->mb_y, b8 = d->b8_stride, ms = d->mb_stride, mh = d->mb_height;
    d->block_index[0] = b8 * (y * 2)     - 2 + x * 2;
    d->block_index[1] = b8 * (y * 2)     - 1 + x * 2;
    d->block_index[2] = b8 * (y * 2 + 1) - 2 + x * 2;
    d->block_index[3] = b8 * (y * 2 + 1) - 1 + x * 2;
    d->block_index[4] = ms * (y + 1)          + b8 * mh * 2 + x - 1;
    d->block_index[5] = ms * (y + mh + 2)     + b8 * mh * 2 + x - 1;
}

// ff_wmv1 DC scale (WMV1 uses ff_wmv1_*_dc_scale_table for both luma+chroma).
static inline void msm_set_qscale(MsmDecoder* d, int q) {
    if (q < 1) q = 1; else if (q > 31) q = 31;
    d->qscale = q;
    d->y_dc_scale = ff_wmv1_y_dc_scale_table[q];
    d->c_dc_scale = ff_wmv1_c_dc_scale_table[q];
}

// ff_mpeg4_clean_buffers equivalent: neutral DC (1024), zero AC/MV, no coded blocks.
static inline void msm_clean_buffers(MsmDecoder* d) {
    for (int i = 0; i < d->grid_elems; i++) d->dc_base[i] = 1024;
    memset(d->ac_base, 0, (size_t)d->grid_elems * 16 * sizeof(int16_t));
    memset(d->mot_base, 0, (size_t)d->grid_elems * 2 * sizeof(int16_t));
    memset(d->coded_base, 0, (size_t)d->grid_elems);
}

// ── prediction ──────────────────────────────────────────────────────────────────
// ff_msmpeg4_coded_block_pred: B==C ? A : C over the b8 coded-block grid.
static inline int msm_coded_block_pred(MsmDecoder* d, int n, uint8_t** ptr) {
    int xy = d->block_index[n], wrap = d->b8_stride;
    uint8_t* cb = d->coded_block + xy;
    int a = cb[-1], b = cb[-1 - wrap], c = cb[-wrap];
    *ptr = cb;
    return (b == c) ? a : c;
}

// get_dc (msmpeg4.c): rounded mean of a reconstructed block, in DC-level units.
static inline int msm_get_dc(const uint8_t* src, int stride, int scale, int block_size) {
    int sum = 0;
    for (int y = 0; y < block_size; y++)
        for (int x = 0; x < block_size; x++)
            sum += src[x + y * stride];
    return (sum + (scale >> 1)) / scale;   // FASTDIV
}

// ff_msmpeg4_pred_dc, WMV1/WMV2 branch (version > V3).
static inline int msm_pred_dc(MsmDecoder* d, int n, int16_t** dc_val_ptr, int* dir_ptr) {
    int scale = (n < 4) ? d->y_dc_scale : d->c_dc_scale;
    int wrap = d->block_wrap[n];
    int16_t* dc_val = d->dc_val + d->block_index[n];
    int a = dc_val[-1], b = dc_val[-1 - wrap], c = dc_val[-wrap];
    a = (a + (scale >> 1)) / scale;   // FASTDIV (DC values are non-negative)
    b = (b + (scale >> 1)) / scale;
    c = (c + (scale >> 1)) / scale;
    int pred;
    if (d->inter_intra_pred) {
        // Intra MB in a WMV1 P-frame (small, low-bitrate streams): blocks 1-3 use a
        // fixed neighbor; blocks 0/4/5 predict from the pixel DC of the reconstructed
        // left / top blocks, picked by h263_aic_dir.
        if (n == 1) {
            pred = a; *dir_ptr = 0;
        } else if (n == 2) {
            pred = c; *dir_ptr = 1;
        } else if (n == 3) {
            if (abs(a - b) < abs(b - c)) { pred = c; *dir_ptr = 1; }
            else                          { pred = a; *dir_ptr = 0; }
        } else {
            const int bs = 8;
            const uint8_t* dest;
            int pw;
            if (n < 4) {
                pw = d->yStride;
                dest = d->y + (size_t)((n >> 1) + 2 * d->mb_y) * bs * pw + ((n & 1) + 2 * d->mb_x) * bs;
            } else {
                pw = d->cStride;
                dest = (n == 4 ? d->u : d->v) + (size_t)(d->mb_y * bs) * pw + d->mb_x * bs;
            }
            if (d->mb_x == 0) a = (1024 + (scale >> 1)) / scale;
            else              a = msm_get_dc(dest - bs, pw, scale * 8, bs);
            if (d->mb_y == 0) c = (1024 + (scale >> 1)) / scale;
            else              c = msm_get_dc(dest - bs * pw, pw, scale * 8, bs);

            if (d->h263_aic_dir == 0) {
                pred = a; *dir_ptr = 0;
            } else if (d->h263_aic_dir == 1) {
                if (n == 0) { pred = c; *dir_ptr = 1; }
                else        { pred = a; *dir_ptr = 0; }
            } else if (d->h263_aic_dir == 2) {
                if (n == 0) { pred = a; *dir_ptr = 0; }
                else        { pred = c; *dir_ptr = 1; }
            } else {
                pred = c; *dir_ptr = 1;
            }
        }
    } else {
        if (abs(a - b) < abs(b - c)) { pred = c; *dir_ptr = 1; }
        else                          { pred = a; *dir_ptr = 0; }
    }
    *dc_val_ptr = dc_val;
    return pred;
}

// msmpeg4_decode_dc (WMV1/v3 path). Returns the predicted DC level; updates the
// predictor grid with level*scale. <0 on illegal VLC.
static inline int msm_decode_dc(MsmDecoder* d, mp4::GetBits* gb, int n, int* dir_ptr) {
    int level = msm_dc_vlc(gb, ff_msmp4_dc_tables[d->dc_table_index][n >= 4 ? 1 : 0]);
    if (level < 0) return -1;
    if (level == DC_MAX) {
        level = (int)mp4::get_bits(gb, 8);
        if (mp4::get_bits1(gb)) level = -level;
    } else if (level != 0) {
        if (mp4::get_bits1(gb)) level = -level;
    }
    int16_t* dc_val;
    int pred = msm_pred_dc(d, n, &dc_val, dir_ptr);
    level += pred;
    *dc_val = (int16_t)(level * ((n < 4) ? d->y_dc_scale : d->c_dc_scale));
    return level;
}

// ff_mpeg4_pred_ac (shared with mpeg4; reads/writes the ac_val grid).
static inline void msm_pred_ac(MsmDecoder* d, int16_t* block, int n, int dir) {
    int16_t* ac_val  = d->ac_val + (size_t)d->block_index[n] * 16;
    int16_t* ac_val1 = ac_val;
    if (d->ac_pred) {
        if (dir == 0) { ac_val -= 16;                    for (int i = 1; i < 8; i++) block[i << 3] += ac_val[i]; }
        else          { ac_val -= 16 * d->block_wrap[n]; for (int i = 1; i < 8; i++) block[i]      += ac_val[i + 8]; }
    }
    for (int i = 1; i < 8; i++) ac_val1[i]     = block[i << 3];
    for (int i = 1; i < 8; i++) ac_val1[8 + i] = block[i];
}

// ── block decode — ff_msmpeg4_decode_block (intra + inter) ──────────────────────
// `inter_scan` overrides the inter scan (WMV2 ABT passes its 8x4 / 4x8 scans);
// NULL selects the regular inter scan. Ignored for intra blocks.
static inline int msm_decode_block(MsmDecoder* d, mp4::GetBits* gb, int16_t* block, int n, int coded, int intra,
                                   const uint8_t* inter_scan) {
    const mp4::RlTab* rl;
    int qmul, qadd, i, dc_pred_dir = -1;
    if (intra) {
        qmul = 1; qadd = 0;                     // raw levels; dequant applies qscale later
        int level = msm_decode_dc(d, gb, n, &dc_pred_dir);
        if (level < 0) return MSM_ERR;
        block[0] = (int16_t)level;
        rl = (n < 4) ? &d->rl[d->rl_table_index] : &d->rl[3 + d->rl_chroma_table_index];
        i = 0;
    } else {
        qmul = d->qscale << 1; qadd = (d->qscale - 1) | 1;   // dequant baked into rl_inter
        rl = &d->rl_inter[d->rl_table_index];
        i = -1;
        if (!coded) { d->block_last_index[n] = -1; return MSM_OK; }
    }
    const int run_diff = 1;                     // WMV1
    if (intra && !coded) goto not_coded;
    {
        const uint8_t* scan_table = intra ? (d->ac_pred ? (dc_pred_dir == 0 ? d->v_scan : d->h_scan) : d->intra_scan)
                                          : (inter_scan ? inter_scan : d->inter_scan);
        const mp4::RL_VLC_ELEM* rl_vlc = rl->table;
        const int bits = rl->bits;
        for (;;) {
            int run, lvl;
            mp4::get_rl_vlc(gb, rl_vlc, bits, &lvl, &run);
            if (lvl == 0) {                     // escape (msmpeg4 prefix convention)
                uint32_t cache = mp4::get_cache(gb);
                if ((cache & 0x80000000u) == 0) {
                    if ((cache & 0x40000000u) == 0) {
                        // third escape (WMV1 ESC-3: variable run/level bit lengths)
                        mp4::skip_bits(gb, 2);
                        int last = (int)mp4::show_bits(gb, 1); mp4::skip_bits(gb, 1);
                        if (!d->esc3_level_length) {
                            int ll;
                            if (d->qscale < 8) {
                                ll = (int)mp4::show_bits(gb, 3); mp4::skip_bits(gb, 3);
                                if (ll == 0) { ll = 8 + (int)mp4::show_bits(gb, 1); mp4::skip_bits(gb, 1); }
                            } else {
                                ll = 2;
                                while (ll < 8 && mp4::show_bits(gb, 1) == 0) { ll++; mp4::skip_bits(gb, 1); }
                                if (ll < 8) mp4::skip_bits(gb, 1);
                            }
                            d->esc3_level_length = ll;
                            d->esc3_run_length = (int)mp4::show_bits(gb, 2) + 3; mp4::skip_bits(gb, 2);
                        }
                        run = (int)mp4::show_bits(gb, d->esc3_run_length);   mp4::skip_bits(gb, d->esc3_run_length);
                        int sign = (int)mp4::show_bits(gb, 1);               mp4::skip_bits(gb, 1);
                        lvl = (int)mp4::show_bits(gb, d->esc3_level_length); mp4::skip_bits(gb, d->esc3_level_length);
                        if (sign) lvl = -lvl;
                        lvl = (lvl > 0) ? lvl * qmul + qadd : lvl * qmul - qadd;
                        i += run + 1;
                        if (last) i += 192;
                    } else {
                        // second escape
                        mp4::skip_bits(gb, 2);
                        mp4::get_rl_vlc(gb, rl_vlc, bits, &lvl, &run);
                        i += run + rl->max_run[run >> 7][lvl / qmul] + run_diff;
                        int s = mp4::show_sign1(gb); lvl = (lvl ^ s) - s; mp4::skip_bits(gb, 1);
                    }
                } else {
                    // first escape
                    mp4::skip_bits(gb, 1);
                    mp4::get_rl_vlc(gb, rl_vlc, bits, &lvl, &run);
                    i += run;
                    lvl = lvl + rl->max_level[run >> 7][(run - 1) & 63] * qmul;
                    int s = mp4::show_sign1(gb); lvl = (lvl ^ s) - s; mp4::skip_bits(gb, 1);
                }
            } else {                            // normal coefficient
                i += run;
                int s = mp4::show_sign1(gb); lvl = (lvl ^ s) - s; mp4::skip_bits(gb, 1);
            }
            if (i > 62) {
                i -= 192;
                if (i & ~63) { i = 63; block[scan_table[i]] = (int16_t)lvl; break; }
                block[scan_table[i]] = (int16_t)lvl;
                break;
            }
            block[scan_table[i]] = (int16_t)lvl;
        }
    }
not_coded:
    if (intra) msm_pred_ac(d, block, n, dc_pred_dir);
    d->block_last_index[n] = i;
    return MSM_OK;
}

// ── dequant + IDCT + reconstruct ────────────────────────────────────────────────
static inline void msm_dequant_intra(MsmDecoder* d, int16_t* block, int n) {
    int qmul = d->qscale << 1;
    int qadd = (d->qscale - 1) | 1;
    block[0] = (int16_t)(block[0] * ((n < 4) ? d->y_dc_scale : d->c_dc_scale));
    int nCoeffs = d->ac_pred ? 63 : d->intra_raster_end[d->block_last_index[n]];
    for (int i = 1; i <= nCoeffs; i++) {
        int level = block[i];
        if (level) block[i] = (int16_t)(level < 0 ? level * qmul - qadd : level * qmul + qadd);
    }
}

static inline void msm_reconstruct_mb(MsmDecoder* d) {
    for (int n = 0; n < 6; n++) {
        msm_dequant_intra(d, d->block[n], n);
        uint8_t* dest; int stride;
        if (n < 4) {
            stride = d->yStride;
            dest = d->y + (size_t)(d->mb_y * 16 + (n >> 1) * 8) * stride + d->mb_x * 16 + (n & 1) * 8;
        } else {
            stride = d->cStride;
            uint8_t* plane = (n == 4) ? d->u : d->v;
            dest = plane + (size_t)(d->mb_y * 8) * stride + d->mb_x * 8;
        }
        if (d->version == MSMP4_WMV2) wmv2_idct_put(dest, stride, d->block[n]);   // ff_wmv2dsp_init
        else                          mp4::idct8_put(dest, stride, d->block[n]);
    }
}

// ── macroblock + frame decode (I-frame) ─────────────────────────────────────────
static inline int msm_decode_mb_intra(MsmDecoder* d, mp4::GetBits* gb) {
    int code = mp4::get_vlc2(gb, d->mb_i.table, d->mb_i.bits);
    if (code < 0) return MSM_ERR;
    int cbp = 0;
    for (int i = 0; i < 6; i++) {
        int val = (code >> (5 - i)) & 1;
        if (i < 4) {
            uint8_t* coded_val;
            val ^= msm_coded_block_pred(d, i, &coded_val);
            *coded_val = (uint8_t)val;
        }
        cbp |= val << (5 - i);
    }
    d->ac_pred = (int)mp4::get_bits1(gb);
    if (d->per_mb_rl_table && cbp) {
        d->rl_table_index = decode012(gb);
        d->rl_chroma_table_index = d->rl_table_index;
    }
    memset(d->block, 0, sizeof(d->block));
    for (int i = 0; i < 6; i++)
        if (msm_decode_block(d, gb, d->block[i], i, (cbp >> (5 - i)) & 1, 1, NULL) < 0) return MSM_ERR;
    return MSM_OK;
}

static inline int msm_decode_i_frame(MsmDecoder* d, mp4::GetBits* gb) {
    msm_set_qscale(d, d->qscale);
    msm_clean_buffers(d);
    for (d->mb_y = 0; d->mb_y < d->mb_height; d->mb_y++) {
        d->first_slice_line = (d->slice_height <= 0) ? (d->mb_y == 0) : (d->mb_y % d->slice_height == 0);
        for (d->mb_x = 0; d->mb_x < d->mb_width; d->mb_x++) {
            msm_init_block_index(d);
            if (msm_decode_mb_intra(d, gb) < 0) return MSM_ERR;
            msm_reconstruct_mb(d);
        }
    }
    return MSM_OK;
}

// ── P-frame: motion vectors + half-pel MC + inter reconstruct ───────────────────
static inline int msm_mid_pred(int a, int b, int c) {
    if (a > b) { if (c > b) return c > a ? a : c; else return b; }
    else       { if (c > b) return b; else return c > a ? c : a; }
}

// ff_h263_pred_motion, 16x16 (block 0). resync_mb_x/y = 0 for msmpeg4 (no packets).
static inline void msm_pred_motion(MsmDecoder* d, int* px, int* py) {
    int wrap = d->b8_stride, xy = d->block_index[0];
    int16_t* m = d->mot_val;
    int16_t* A = m + (xy - 1) * 2;
    if (d->first_slice_line) {
        if (d->mb_x == 0) { *px = *py = 0; }
        else { *px = A[0]; *py = A[1]; }
    } else {
        int16_t* B = m + (xy - wrap) * 2;
        int16_t* C = m + (xy + 2 - wrap) * 2;   // off[0]=2 (top-right)
        *px = msm_mid_pred(A[0], B[0], C[0]);
        *py = msm_mid_pred(A[1], B[1], C[1]);
    }
}

// ff_msmpeg4_decode_motion.
static inline void msm_decode_motion(MsmDecoder* d, mp4::GetBits* gb, int* mx_ptr, int* my_ptr) {
    const mp4::VlcTab* mv = &d->mv_tab[d->mv_table_index];
    int sym = mp4::get_vlc2(gb, mv->table, mv->bits);
    int mx, my;
    if (sym) { mx = sym >> 8; my = sym & 0xFF; }
    else { mx = (int)mp4::get_bits(gb, 6); my = (int)mp4::get_bits(gb, 6); }
    mx += *mx_ptr - 32;
    my += *my_ptr - 32;
    if (mx <= -64) mx += 64; else if (mx >= 64) mx -= 64;
    if (my <= -64) my += 64; else if (my >= 64) my -= 64;
    *mx_ptr = mx; *my_ptr = my;
}

// ff_h263_update_motion_val, 16x16: write the MB's MV into its 4 b8 grid cells.
static inline void msm_update_motion_val(MsmDecoder* d) {
    int wrap = d->b8_stride, xy = d->block_index[0];
    int mx = d->mb_intra ? 0 : d->cur_mvx, my = d->mb_intra ? 0 : d->cur_mvy;
    int16_t* m = d->mot_val;
    m[xy*2]=mx;         m[xy*2+1]=my;
    m[(xy+1)*2]=mx;     m[(xy+1)*2+1]=my;
    m[(xy+wrap)*2]=mx;  m[(xy+wrap)*2+1]=my;
    m[(xy+1+wrap)*2]=mx;m[(xy+1+wrap)*2+1]=my;
}

// Scalar half-pel MC of a bw x bh block with per-tap edge clamp (== ffmpeg
// emulated_edge + no_rnd/rnd average). rb = 1-no_rounding (put bias).
static inline int msm_clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }
static inline void msm_mc_block(uint8_t* dst, int dst_stride, const uint8_t* ref, int ref_stride,
                                int w, int h, int bw, int bh, int sx, int sy, int dxy, int rb) {
    for (int by = 0; by < bh; by++) {
        const uint8_t* r0 = ref + msm_clampi(sy+by,   0, h-1) * ref_stride;
        const uint8_t* r1 = ref + msm_clampi(sy+by+1, 0, h-1) * ref_stride;
        uint8_t* o = dst + by * dst_stride;
        for (int bx = 0; bx < bw; bx++) {
            int x0 = msm_clampi(sx+bx, 0, w-1), x1 = msm_clampi(sx+bx+1, 0, w-1);
            int a = r0[x0], v;
            if      (dxy == 0) v = a;
            else if (dxy == 1) v = (a + r0[x1] + rb) >> 1;
            else if (dxy == 2) v = (a + r1[x0] + rb) >> 1;
            else               v = (a + r0[x1] + r1[x0] + r1[x1] + rb + 1) >> 2;
            o[bx] = (uint8_t)v;
        }
    }
}

// Predict the current MB (16x16 Y + 8x8 U/V) from the reference frame (1 MV).
static inline void msm_mc_pred(MsmDecoder* d, int mx, int my) {
    int rb = 1 - d->no_rounding, cw = d->codedW >> 1, ch = d->codedH >> 1;
    int yoff = (d->mb_y * 16) * d->yStride + d->mb_x * 16;
    int coff = (d->mb_y * 8)  * d->cStride + d->mb_x * 8;
    int dxy = ((my & 1) << 1) | (mx & 1);
    int sx = d->mb_x * 16 + (mx >> 1), sy = d->mb_y * 16 + (my >> 1);
    msm_mc_block(d->y + yoff, d->yStride, d->refY, d->yStride, d->codedW, d->codedH, 16, 16, sx, sy, dxy, rb);
    int uvdxy = dxy | (my & 2) | ((mx & 2) >> 1), usx = sx >> 1, usy = sy >> 1;
    msm_mc_block(d->u + coff, d->cStride, d->refU, d->cStride, cw, ch, 8, 8, usx, usy, uvdxy, rb);
    msm_mc_block(d->v + coff, d->cStride, d->refV, d->cStride, cw, ch, 8, 8, usx, usy, uvdxy, rb);
}

// Reconstruct a P-frame MB: intra -> as I-frame; inter -> MC predict + add residual.
static inline void msm_reconstruct_mb_p(MsmDecoder* d) {
    if (d->mb_intra) { msm_reconstruct_mb(d); return; }
    msm_mc_pred(d, d->cur_mvx, d->cur_mvy);
    if (d->mb_skipped) return;
    for (int n = 0; n < 6; n++) {
        if (d->block_last_index[n] < 0) continue;   // uncoded inter block: MC only
        uint8_t* dest; int stride;
        if (n < 4) { stride = d->yStride; dest = d->y + (size_t)(d->mb_y * 16 + (n >> 1) * 8) * stride + d->mb_x * 16 + (n & 1) * 8; }
        else       { stride = d->cStride; uint8_t* pl = (n == 4) ? d->u : d->v; dest = pl + (size_t)(d->mb_y * 8) * stride + d->mb_x * 8; }
        mp4::idct8_add(dest, stride, d->block[n]);   // inter coeffs are pre-dequantized
    }
}

// msmpeg4v34_decode_mb (P-frame path).
static inline int msm_decode_mb_p(MsmDecoder* d, mp4::GetBits* gb) {
    if (mp4::get_bits_left(gb) <= 0) return MSM_ERR;
    d->mb_skipped = 0;
    if (d->use_skip_mb_code && mp4::get_bits1(gb)) {
        d->mb_intra = 0; d->mb_skipped = 1; d->cur_mvx = d->cur_mvy = 0;
        for (int i = 0; i < 6; i++) d->block_last_index[i] = -1;
        msm_update_motion_val(d);
        return MSM_OK;
    }
    int code = msm_vlc_slow(gb, ff_table_mb_non_intra, 128, 21);
    if (code < 0) return MSM_ERR;
    d->mb_intra = (~code & 0x40) >> 6;
    int cbp = code & 0x3f;
    if (!d->mb_intra) {
        if (d->per_mb_rl_table && cbp) { d->rl_table_index = decode012(gb); d->rl_chroma_table_index = d->rl_table_index; }
        int mx, my;
        msm_pred_motion(d, &mx, &my);
        msm_decode_motion(d, gb, &mx, &my);
        d->cur_mvx = mx; d->cur_mvy = my;
    } else {
        d->ac_pred = (int)mp4::get_bits1(gb);
        if (d->inter_intra_pred) d->h263_aic_dir = mp4::get_vlc2(gb, d->inter_intra.table, d->inter_intra.bits);
        if (d->per_mb_rl_table && cbp) { d->rl_table_index = decode012(gb); d->rl_chroma_table_index = d->rl_table_index; }
    }
    msm_update_motion_val(d);
    memset(d->block, 0, sizeof(d->block));
    for (int i = 0; i < 6; i++)
        if (msm_decode_block(d, gb, d->block[i], i, (cbp >> (5 - i)) & 1, d->mb_intra, NULL) < 0) return MSM_ERR;
    return MSM_OK;
}

static inline int msm_decode_p_frame(MsmDecoder* d, mp4::GetBits* gb) {
    msm_set_qscale(d, d->qscale);
    msm_build_inter_rl(d, d->qscale);
    msm_clean_buffers(d);
    for (d->mb_y = 0; d->mb_y < d->mb_height; d->mb_y++) {
        d->first_slice_line = (d->slice_height <= 0) ? (d->mb_y == 0) : (d->mb_y % d->slice_height == 0);
        for (d->mb_x = 0; d->mb_x < d->mb_width; d->mb_x++) {
            msm_init_block_index(d);
            if (msm_decode_mb_p(d, gb) < 0) return MSM_ERR;
            msm_reconstruct_mb_p(d);
        }
    }
    return MSM_OK;
}

// ── header parse ────────────────────────────────────────────────────────────────
static inline void msm_decode_ext_header(MsmDecoder* d, mp4::GetBits* gb, int buf_size) {
    int left = buf_size * 8 - mp4::get_bits_count(gb);
    int length = d->version >= MSMP4_V3 ? 17 : 16;
    if (left >= length && left < length + 8) {
        mp4::skip_bits(gb, 5);
        d->bit_rate = (int)mp4::get_bits(gb, 11) * 1024;
        d->flipflop_rounding = d->version >= MSMP4_V3 ? (int)mp4::get_bits1(gb) : 0;
    } else if (left < length + 8) {
        d->flipflop_rounding = 0;
    }
}

static inline int msm_decode_picture_header(MsmDecoder* d, mp4::GetBits* gb) {
    if ((int64_t)mp4::get_bits_left(gb) * 8 < (int64_t)d->mb_width * d->mb_height) return MSM_ERR;
    if (d->version == MSMP4_V1) {
        if (mp4::get_bits_long(gb, 32) != 0x00000100) return MSM_ERR;
        mp4::skip_bits(gb, 5);
    }
    d->pict_type = (int)mp4::get_bits(gb, 2) + 1;
    if (d->pict_type != PICT_I && d->pict_type != PICT_P) return MSM_ERR;
    d->qscale = (int)mp4::get_bits(gb, 5);
    if (d->qscale == 0) return MSM_ERR;

    if (d->pict_type == PICT_I) {
        int code = (int)mp4::get_bits(gb, 5);
        if (code < 0x17) return MSM_ERR;
        d->slice_height = d->mb_height / (code - 0x16);
        msm_decode_ext_header(d, gb, (2 + 5 + 5 + 17 + 7) / 8);
        d->per_mb_rl_table = d->bit_rate > MBAC_BITRATE ? (int)mp4::get_bits1(gb) : 0;
        if (!d->per_mb_rl_table) {
            d->rl_chroma_table_index = decode012(gb);
            d->rl_table_index        = decode012(gb);
        }
        d->dc_table_index   = (int)mp4::get_bits1(gb);
        d->inter_intra_pred = 0;
        d->no_rounding      = 1;
    } else {
        d->use_skip_mb_code = (int)mp4::get_bits1(gb);
        d->per_mb_rl_table  = d->bit_rate > MBAC_BITRATE ? (int)mp4::get_bits1(gb) : 0;
        if (!d->per_mb_rl_table) {
            d->rl_table_index        = decode012(gb);
            d->rl_chroma_table_index = d->rl_table_index;
        }
        d->dc_table_index   = (int)mp4::get_bits1(gb);
        d->mv_table_index   = (int)mp4::get_bits1(gb);
        d->inter_intra_pred = (d->width * d->height < 320 * 240) && (d->bit_rate <= II_BITRATE);
        if (d->flipflop_rounding) d->no_rounding ^= 1; else d->no_rounding = 0;
    }
    d->esc3_level_length = 0;
    d->esc3_run_length   = 0;
    return MSM_OK;
}

// Decode one WMV1 frame (I keyframe or P inter). Must be called in stream order —
// each P-frame motion-compensates from the previous frame. Returns MSM_OK / MSM_ERR.
static inline int msm_decode_frame(MsmDecoder* d, const uint8_t* bytes, int len) {
    if (!d->y) { if (msm_alloc(d) != MSM_OK) return MSM_ERR; }
    // The just-decoded frame becomes the reference for the next P-frame.
    uint8_t* t;
    t = d->y; d->y = d->refY; d->refY = t;
    t = d->u; d->u = d->refU; d->refU = t;
    t = d->v; d->v = d->refV; d->refV = t;
    mp4::GetBits gb;
    mp4::init_get_bits(&gb, bytes, len);
    if (msm_decode_picture_header(d, &gb) != MSM_OK) return MSM_ERR;
    return (d->pict_type == PICT_I) ? msm_decode_i_frame(d, &gb) : msm_decode_p_frame(d, &gb);
}

} // namespace msm

#endif // MSMPEG4_DECODE_H
