/*
    MPEG-4 Part 2 ASP I-VOP (intra) decoder — transcription of FFmpeg release/8.1
    libavcodec/mpeg4videodec.c + mpegvideo*.c (LGPL). See mpeg4_decode.cpp header.

    Scope (M1–M3): rectangular, progressive, non-scalable DivX/XviD/mp4v —
    start-code dispatch + VOL/VOP header parse with the gate list; the intra
    MB -> block -> DC/AC-prediction -> DCT3D-RL -> H.263 dequant -> idct_put
    pipeline (M1); P-VOP inter with median MV prediction + half-pel MC, 1MV/4MV,
    skip, and the f_code-dependent resync/video-packet machinery (M2); quarter-pel
    (qpel) luma MC via mp4_qpel.h with qpel-derived half-pel chroma (M3). Everything
    else (B-VOP, interlaced, sprite/GMC, data-partition, mpeg-quant, studio,
    H.263 short-header) is gated to a clean UNSUPPORTED return. Oracle: ffmpeg CLI
    max=0 (scalar C reference, `-cpuflags 0`).
*/

#ifndef MPEG4_MP4_DECODE_H
#define MPEG4_MP4_DECODE_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "mp4_bits.h"
#include "mp4_tables.h"
#include "simple_idct8.h"
#include "mp4_qpel.h"

namespace mp4 {

// Decode-call return codes. MP4_DECODE is an internal "header parsed, decode the
// picture" signal from mp4_decode_vop_header — it MUST differ from MP4_SKIP so the
// caller can tell "decode this VOP" from "skip it" (an NVOP). They used to both be 1,
// which made NVOPs fall through into mp4_decode_picture instead of being skipped.
enum {
    MP4_OK          =  0,
    MP4_SKIP        =  1,   // NVOP (vop_coded==0) — keep previous picture
    MP4_DECODE      =  2,   // mp4_decode_vop_header: header parsed, decode the picture
    MP4_ERR_DATA    = -1,
    MP4_ERR_UNSUP   = -2,   // a gated (not-yet-supported) feature
    MP4_ERR_NOVOP   = -3,   // no VOP found in the buffer
};

// MPEG-4 start codes (low byte of 0x000001xx).
enum {
    SC_VO_LOW   = 0x00,  // 0x00..0x1F video_object
    SC_VOL_LOW  = 0x20,  // 0x20..0x2F video_object_layer
    SC_VOS      = 0xB0, SC_USERDATA = 0xB2, SC_GOP = 0xB3,
    SC_VISOBJ   = 0xB5, SC_VOP = 0xB6,
};
enum { SHAPE_RECT = 0, SHAPE_BIN = 1, SHAPE_BIN_ONLY = 2, SHAPE_GRAY = 3 };

static const uint8_t mp4_quant_tab[4] = { 0xFF, 0xFE, 1, 2 }; // -1,-2,1,2 as int8

struct Mp4Decoder {
    GetBits gb;

    // VOL config (persists across VOPs).
    int configured;
    int is_h263;                  // 1 = ITU-T H.263 short header (M5); 0 = MPEG-4 Part 2
    int gob_index;                // H.263 GOB row stride = H263_GOB_HEIGHT(height)
    int width, height;            // display
    int mb_width, mb_height;
    int mb_stride, b8_stride;
    int vo_ver_id;
    int quant_precision;
    int time_increment_bits;
    int shape;
    int progressive_sequence;
    int mpeg_quant;               // 0 = H.263 quant (gated to 0 for M1)
    int quarter_sample;           // VOL quarter_sample: 1 = qpel luma MC (M3)
    int low_delay;
    int time_increment_resolution;// VOL vop_time_increment_resolution (== ffmpeg framerate.num)

    // Per-VOP state.
    int pict_type;                // 1=I, 2=P, 3=B (AV_PICTURE_TYPE_*)
    int qscale;
    int chroma_qscale;
    int y_dc_scale, c_dc_scale;
    int intra_dc_threshold;       // ff_mpeg4_dc_threshold[idx]
    int ac_pred;
    int interlaced_dct;
    int no_rounding;              // vop_rounding_type (P)
    int f_code;                   // forward MV fcode (P/B)
    int b_code;                   // backward MV fcode (B)

    // B-VOP timing (ticks); pp_time = TRD (fwd ref -> bwd ref), pb_time = TRB
    // (fwd ref -> this B). Differences cancel any absolute time_base offset, so
    // these are robust to re-seeking. Used by direct-mode MV scaling.
    int time_base, last_time_base, last_non_b_time, pp_time, pb_time;

    // Per-MB state.
    int mb_x, mb_y;
    int resync_mb_x, resync_mb_y;   // current video-packet start
    int first_slice_line;
    int mb_intra;                 // current MB intra?
    int mb_skipped;               // current MB was skip-coded (P-skip), for B's bwd skip table
    int mv_type;                  // 0 = 16x16 (1MV), 1 = 8x8 (4MV)
    int mv_dir;                   // B only: bit0 = forward, bit1 = backward
    int mv[2][4][2];              // [dir][block][x/y]; P uses mv[0], B uses mv[0]=fwd, mv[1]=bwd
    int b_last_mv[2][2];          // B running MV predictor [dir][x/y] (reset at mb_x==0)

    // DC/AC prediction grids (frame-persistent; +192-style leading guard so the
    // edge predictor OOB reads — which ffmpeg overwrites — never fault).
    int16_t* dc_base; int16_t* dc_val;   // dc_val = dc_base + lead; [block_index[n]]
    int16_t* ac_base; int16_t* ac_val;   // 16 ints per block: [block_index[n]*16]
    int16_t* mot_base; int16_t* mot_val; // motion_val: 2 ints per block (P MV prediction)
    int dc_off;                          // ffmpeg's b8_stride+1 offset (folded into dc_val)
    int grid_elems;                      // total int16 per grid (incl. guards)
    int block_wrap[6];
    int block_index[6];
    int block_last_index[6];

    // Scan tables (PERM_NONE -> permutated == scan). raster_end = running max.
    uint8_t intra_scan[64];   uint8_t intra_raster_end[64];
    uint8_t h_scan[64];       uint8_t v_scan[64];

    int16_t block[6][64];

    // Output planes (coded = mb-aligned). yStride=codedW, cStride=codedW/2.
    // I/P: y/u/v = the just-decoded frame (== backward ref / next_pic for B's);
    //      refY/U/V = the previous I/P (== forward ref / last_pic). Swapped at the
    //      start of each I/P decode. B: forward = refY/U/V, backward = y/u/v, output
    //      written to bY/U/V (no swap; B is never a reference).
    int codedW, codedH, yStride, cStride;
    uint8_t* y; uint8_t* u; uint8_t* v;
    uint8_t* refY; uint8_t* refU; uint8_t* refV;
    uint8_t* bY; uint8_t* bU; uint8_t* bV;   // B-VOP output planes
    size_t yCap, cCap;

    // Backward-ref (next_pic) metadata captured at the end of each I/P decode, read
    // by the following B-VOPs: motion_val copy (direct-mode co-located MVs), per-MB
    // skip flag (B mb_skipped), per-MB 8x8 flag (direct mode 4MV vs 16x16).
    int16_t* bwd_mot; int16_t* bwd_mot_val;  // copy of last I/P motion_val (direct co-located)
    uint8_t* bwd_skip;            // [mb_y*mb_stride + mb_x]
    uint8_t* bwd_is8x8;           // [mb_y*mb_stride + mb_x]
    int last_output_is_b;         // 1 if the most recent decode produced a B (output = bY/U/V)
};

// ── scan / grid setup ──────────────────────────────────────────────────────────

static inline void mp4_make_scan(const uint8_t* src, uint8_t* perm, uint8_t* raster_end) {
    int end = -1;
    for (int i = 0; i < 64; i++) {
        int j = src[i];          // FF_IDCT_PERM_NONE -> permutation[j] = j
        perm[i] = (uint8_t)j;
        if (j > end) end = j;
        if (raster_end) raster_end[i] = (uint8_t)end;
    }
}

// (Re)allocate the planes + DC/AC grids for the configured dimensions.
static inline int mp4_alloc(Mp4Decoder* d) {
    d->mb_width  = (d->width + 15) >> 4;
    d->mb_height = (d->height + 15) >> 4;
    d->mb_stride = d->mb_width + 1;
    d->b8_stride = d->mb_width * 2 + 1;
    d->codedW = d->mb_width * 16;
    d->codedH = d->mb_height * 16;
    d->yStride = d->codedW;
    d->cStride = d->codedW >> 1;

    size_t yNeed = (size_t)d->yStride * d->codedH;
    size_t cNeed = (size_t)d->cStride * (d->codedH >> 1);
    if (yNeed > d->yCap) {
        free(d->y);    d->y    = (uint8_t*)malloc(yNeed);
        free(d->refY); d->refY = (uint8_t*)malloc(yNeed);
        free(d->bY);   d->bY   = (uint8_t*)malloc(yNeed);
        if (!d->y || !d->refY || !d->bY) return MP4_ERR_DATA;
        d->yCap = yNeed;
    }
    if (cNeed > d->cCap) {
        free(d->u); free(d->v); free(d->refU); free(d->refV); free(d->bU); free(d->bV);
        d->u    = (uint8_t*)malloc(cNeed); d->v    = (uint8_t*)malloc(cNeed);
        d->refU = (uint8_t*)malloc(cNeed); d->refV = (uint8_t*)malloc(cNeed);
        d->bU   = (uint8_t*)malloc(cNeed); d->bV   = (uint8_t*)malloc(cNeed);
        if (!d->u || !d->v || !d->refU || !d->refV || !d->bU || !d->bV) return MP4_ERR_DATA;
        d->cCap = cNeed;
    }

    // Grid: yc_size = b8_stride*(2*mb_height+1) + 2*mb_stride*(mb_height+1).
    int y_size  = d->b8_stride * (2 * d->mb_height + 1);
    int c_size  = d->mb_stride * (d->mb_height + 1);
    int yc_size = y_size + 2 * c_size;
    d->dc_off   = d->b8_stride + 1;
    int lead    = 2 * d->b8_stride + 16;      // leading guard for edge OOB reads
    int trail   = 2 * d->b8_stride + 16;
    int total   = lead + yc_size + trail;
    if (total != d->grid_elems) {
        free(d->dc_base); free(d->ac_base); free(d->mot_base); free(d->bwd_mot);
        free(d->bwd_skip); free(d->bwd_is8x8);
        d->dc_base  = (int16_t*)malloc((size_t)total * sizeof(int16_t));
        d->ac_base  = (int16_t*)malloc((size_t)total * 16 * sizeof(int16_t));
        d->mot_base = (int16_t*)malloc((size_t)total * 2 * sizeof(int16_t));
        d->bwd_mot  = (int16_t*)malloc((size_t)total * 2 * sizeof(int16_t));
        size_t mbtab = (size_t)d->mb_stride * (d->mb_height + 1);
        d->bwd_skip  = (uint8_t*)calloc(mbtab, 1);
        d->bwd_is8x8 = (uint8_t*)calloc(mbtab, 1);
        if (!d->dc_base || !d->ac_base || !d->mot_base || !d->bwd_mot ||
            !d->bwd_skip || !d->bwd_is8x8) return MP4_ERR_DATA;
        d->grid_elems = total;
    }
    d->dc_val  = d->dc_base + lead + d->dc_off;   // == ffmpeg dc_val_base + (b8_stride+1)
    d->ac_val  = d->ac_base + (size_t)(lead + d->dc_off) * 16;
    d->mot_val = d->mot_base + (size_t)(lead + d->dc_off) * 2;
    d->bwd_mot_val = d->bwd_mot + (size_t)(lead + d->dc_off) * 2;

    d->block_wrap[0] = d->block_wrap[1] = d->block_wrap[2] = d->block_wrap[3] = d->b8_stride;
    d->block_wrap[4] = d->block_wrap[5] = d->mb_stride;

    mp4_make_scan(ff_zigzag_direct, d->intra_scan, d->intra_raster_end);
    mp4_make_scan(ff_alternate_horizontal_scan, d->h_scan, NULL);
    mp4_make_scan(ff_alternate_vertical_scan, d->v_scan, NULL);
    return MP4_OK;
}

static inline void mp4_init_block_index(Mp4Decoder* d) {
    int x = d->mb_x, y = d->mb_y, b8 = d->b8_stride, ms = d->mb_stride, mh = d->mb_height;
    d->block_index[0] = b8 * (y * 2)     - 2 + x * 2;
    d->block_index[1] = b8 * (y * 2)     - 1 + x * 2;
    d->block_index[2] = b8 * (y * 2 + 1) - 2 + x * 2;
    d->block_index[3] = b8 * (y * 2 + 1) - 1 + x * 2;
    d->block_index[4] = ms * (y + 1)          + b8 * mh * 2 + x - 1;
    d->block_index[5] = ms * (y + mh + 2)     + b8 * mh * 2 + x - 1;
}

static inline void mp4_set_qscale(Mp4Decoder* d, int q) {
    if (q < 1) q = 1; else if (q > 31) q = 31;
    d->qscale = q;
    d->chroma_qscale = q;                       // MPEG-4 uses identity chroma_qscale_table
    if (d->is_h263) {
        // H.263 (non-AIC) uses ff_mpeg1_dc_scale_table (a constant 8) for the intra DC
        // scale; AC/inter dequant qmul/qadd are the same as MPEG-4's H.263 quant.
        d->y_dc_scale = d->c_dc_scale = 8;
    } else {
        d->y_dc_scale = ff_mpeg4_y_dc_scale_table[q];
        d->c_dc_scale = ff_mpeg4_c_dc_scale_table[d->chroma_qscale];
    }
}

// ── DC / AC prediction (mpeg4videodec.c:902/940/328) ───────────────────────────

static inline int mp4_pred_dc(Mp4Decoder* d, int n, int* dir_ptr) {
    const int16_t* dc_val = d->dc_val + d->block_index[n];
    const int wrap = d->block_wrap[n];
    int a = dc_val[-1];
    int b = dc_val[-1 - wrap];
    int c = dc_val[-wrap];
    if (d->first_slice_line && n != 3) {
        if (n != 2) b = c = 1024;
        if (n != 1 && d->mb_x == d->resync_mb_x) b = a = 1024;
    }
    if (d->mb_x == d->resync_mb_x && d->mb_y == d->resync_mb_y + 1) {
        if (n == 0 || n == 4 || n == 5) b = 1024;
    }
    int pred;
    if (abs(a - b) < abs(b - c)) { pred = c; *dir_ptr = 1; }
    else                          { pred = a; *dir_ptr = 0; }
    return pred;
}

static inline int mp4_get_level_dc(Mp4Decoder* d, int n, int pred, int level) {
    int scale = n < 4 ? d->y_dc_scale : d->c_dc_scale;
    pred = (pred + (scale >> 1)) / scale;   // FASTDIV, pred assumed >= 0
    level += pred;
    int ret = level;
    level *= scale;
    if (level & (~2047)) { if (level < 0) level = 0; else level = 2047; }
    d->dc_val[d->block_index[n]] = (int16_t)level;
    return ret;
}

static inline int mp4_decode_dc(Mp4Decoder* d, int n, int* dir_ptr) {
    int code = (n < 4) ? get_vlc2(&d->gb, g_dc_lum.table, g_dc_lum.bits)
                       : get_vlc2(&d->gb, g_dc_chrom.table, g_dc_chrom.bits);
    if (code < 0) return MP4_ERR_DATA;
    int level;
    if (code == 0) {
        level = 0;
    } else {
        level = get_xbits(&d->gb, code);
        if (code > 8) get_bits1(&d->gb);   // marker bit (ffmpeg only warns on 0)
    }
    int pred = mp4_pred_dc(d, n, dir_ptr);
    return mp4_get_level_dc(d, n, pred, level);
}

static inline void mp4_pred_ac(Mp4Decoder* d, int16_t* block, int n, int dir) {
    int16_t* ac_val  = d->ac_val + (size_t)d->block_index[n] * 16;
    int16_t* ac_val1 = ac_val;
    if (d->ac_pred) {
        if (dir == 0) {                 // left
            ac_val -= 16;
            for (int i = 1; i < 8; i++) block[i << 3] += ac_val[i];     // idct_perm none
        } else {                        // top
            ac_val -= 16 * d->block_wrap[n];
            for (int i = 1; i < 8; i++) block[i] += ac_val[i + 8];
        }
    }
    for (int i = 1; i < 8; i++) ac_val1[i]     = block[i << 3];   // left col
    for (int i = 1; i < 8; i++) ac_val1[8 + i] = block[i];        // top row
}
// NOTE: M1/M2 streams use a single qscale per frame, so the cross-qscale AC
// rescale branch (ROUNDED_DIV by neighbor qscale) is omitted (add if a corpus
// stream uses per-MB dquant with intra+ac_pred).

// ── motion vectors (ituh263dec.c:279 / h263.c:182) ─────────────────────────────

static inline int mp4_sign_extend(int val, int bits) {
    int shift = 32 - bits;
    return (int)((int32_t)((uint32_t)val << shift) >> shift);
}
static inline int mp4_mid_pred(int a, int b, int c) {
    if (a > b) { if (c > b) return c > a ? a : c; else return b; }
    else       { if (c > b) return b; else return c > a ? c : a; }
}
static inline int mp4_round_chroma(int x) {
    static const uint8_t tab[16] = { 0,0,0,1,1,1,1,1,0,0,0,0,0,0,1,1 };
    return tab[x & 0xf] + (x >> 3);
}

// MV residual decode (ff_h263_decode_motion). 0xffff on VLC error.
static inline int mp4_decode_motion(Mp4Decoder* d, int pred, int f_code) {
    int code = get_vlc2(&d->gb, g_mv.table, g_mv.bits);
    if (code == 0) return pred;
    if (code < 0)  return 0xffff;
    int sign = (int)get_bits1(&d->gb);
    int shift = f_code - 1;
    int val = code;
    if (shift) { val = (val - 1) << shift; val |= (int)get_bits(&d->gb, shift); val++; }
    if (sign) val = -val;
    val += pred;
    return mp4_sign_extend(val, 5 + f_code);   // h263_long_vectors = 0
}

// Median MV predictor (ff_h263_pred_motion). Writes *px/*py; returns the block's
// motion_val grid offset (xy) so the 4MV caller can write the MV back.
static inline int mp4_pred_motion(Mp4Decoder* d, int block, int* px, int* py) {
    static const int off[4] = { 2, 1, 1, -1 };
    int wrap = d->b8_stride;
    int xy = d->block_index[block];
    int16_t* A = d->mot_val + (xy - 1) * 2;
    int16_t *B, *C;
    if (d->first_slice_line && block < 3) {
        if (block == 0) {
            if (d->mb_x == d->resync_mb_x) { *px = *py = 0; }
            else if (d->mb_x + 1 == d->resync_mb_x) {
                C = d->mot_val + (xy + off[0] - wrap) * 2;
                if (d->mb_x == 0) { *px = C[0]; *py = C[1]; }
                else { *px = mp4_mid_pred(A[0], 0, C[0]); *py = mp4_mid_pred(A[1], 0, C[1]); }
            } else { *px = A[0]; *py = A[1]; }
        } else if (block == 1) {
            if (d->mb_x + 1 == d->resync_mb_x) {
                C = d->mot_val + (xy + off[1] - wrap) * 2;
                *px = mp4_mid_pred(A[0], 0, C[0]); *py = mp4_mid_pred(A[1], 0, C[1]);
            } else { *px = A[0]; *py = A[1]; }
        } else { // block == 2
            B = d->mot_val + (xy - wrap) * 2;
            C = d->mot_val + (xy + off[2] - wrap) * 2;
            if (d->mb_x == d->resync_mb_x) { A[0] = A[1] = 0; }
            *px = mp4_mid_pred(A[0], B[0], C[0]); *py = mp4_mid_pred(A[1], B[1], C[1]);
        }
    } else {
        B = d->mot_val + (xy - wrap) * 2;
        C = d->mot_val + (xy + off[block] - wrap) * 2;
        *px = mp4_mid_pred(A[0], B[0], C[0]); *py = mp4_mid_pred(A[1], B[1], C[1]);
    }
    return xy;
}

// Propagate the MB's MV(s) into the motion_val grid (ff_h263_update_motion_val).
static inline void mp4_update_motion_val(Mp4Decoder* d) {
    if (d->mv_type == 1) return;   // 8x8: already written during parse
    int wrap = d->b8_stride, xy = d->block_index[0];
    int mx = d->mb_intra ? 0 : d->mv[0][0][0];
    int my = d->mb_intra ? 0 : d->mv[0][0][1];
    int16_t* m = d->mot_val;
    m[xy*2]=mx; m[xy*2+1]=my;
    m[(xy+1)*2]=mx; m[(xy+1)*2+1]=my;
    m[(xy+wrap)*2]=mx; m[(xy+wrap)*2+1]=my;
    m[(xy+1+wrap)*2]=mx; m[(xy+1+wrap)*2+1]=my;
}

// ── block decode — intra + inter (mpeg4videodec.c:1381) ────────────────────────

static inline int mp4_decode_block(Mp4Decoder* d, int16_t* block, int n,
                                   int coded, int intra, int use_intra_dc_vlc) {
    GetBits* gb = &d->gb;
    const RlTab* rl;
    const RL_VLC_ELEM* rl_vlc;
    int bits, qmul, qadd;
    int level, i, last, run, pred = 0, dc_pred_dir = 0;
    const uint8_t* scan_table;

    if (intra) {
        rl = &g_rl_intra; rl_vlc = rl->table; qmul = 1; qadd = 0;
        if (use_intra_dc_vlc) {
            level = mp4_decode_dc(d, n, &dc_pred_dir);
            if (level < 0) return MP4_ERR_DATA;
            block[0] = (int16_t)level;
            i = 0;
        } else {
            i = -1;
            pred = mp4_pred_dc(d, n, &dc_pred_dir);
        }
        if (!coded) goto not_coded;
        if (d->ac_pred) scan_table = (dc_pred_dir == 0) ? d->v_scan : d->h_scan;
        else            scan_table = d->intra_scan;
    } else {
        i = -1;
        if (!coded) { d->block_last_index[n] = -1; return MP4_OK; }
        rl = &g_rl_inter[d->qscale]; rl_vlc = rl->table;
        qmul = d->qscale << 1; qadd = (d->qscale - 1) | 1;
        scan_table = d->intra_scan;
    }
    bits = rl->bits;

    for (;;) {
        get_rl_vlc(gb, rl_vlc, bits, &level, &run);
        if (level == 0) {                                  // escape
            uint32_t cache = get_cache(gb);
            if (cache & 0x80000000u) {
                if (cache & 0x40000000u) {                 // third escape
                    skip_bits(gb, 2);
                    last = (int)get_bits1(gb);
                    run  = (int)get_bits(gb, 6);
                    get_bits1(gb);                         // marker
                    level = get_sbits(gb, 12);
                    get_bits1(gb);                         // marker
                    if (level > 0) level = level * qmul + qadd;
                    else           level = level * qmul - qadd;
                    if ((unsigned)(level + 2048) > 4095) level = level < 0 ? -2048 : 2047;
                    i += run + 1;
                    if (last) i += 192;
                } else {                                   // second escape
                    skip_bits(gb, 2);
                    get_rl_vlc(gb, rl_vlc, bits, &level, &run);
                    i += run + rl->max_run[run >> 7][level / qmul] + 1;
                    int s = show_sign1(gb); level = (level ^ s) - s; skip_bits(gb, 1);
                }
            } else {                                       // first escape
                skip_bits(gb, 1);
                get_rl_vlc(gb, rl_vlc, bits, &level, &run);
                i += run;
                level = level + rl->max_level[run >> 7][(run - 1) & 63] * qmul;
                int s = show_sign1(gb); level = (level ^ s) - s; skip_bits(gb, 1);
            }
        } else {                                           // normal coefficient
            i += run;
            int s = show_sign1(gb); level = (level ^ s) - s; skip_bits(gb, 1);
        }
        if (i > 62) {
            i -= 192;
            if (i & ~63) return MP4_ERR_DATA;
            block[scan_table[i]] = (int16_t)level;
            break;
        }
        block[scan_table[i]] = (int16_t)level;
    }

not_coded:
    if (intra) {
        if (!use_intra_dc_vlc) {
            block[0] = (int16_t)mp4_get_level_dc(d, n, pred, block[0]);
            i -= i >> 31;   // i==-1 -> 0
        }
        mp4_pred_ac(d, block, n, dc_pred_dir);
        if (d->ac_pred) i = 63;
    }
    d->block_last_index[n] = i;
    return MP4_OK;
}

// ── dequant + IDCT + write (dct_unquantize_h263_intra + idct_put) ──────────────

static inline void mp4_dequant_h263_intra(Mp4Decoder* d, int16_t* block, int n) {
    int qmul = d->qscale << 1;
    int qadd = (d->qscale - 1) | 1;
    block[0] *= (n < 4) ? d->y_dc_scale : d->c_dc_scale;
    int nCoeffs = d->ac_pred ? 63 : d->intra_raster_end[d->block_last_index[n]];
    for (int i = 1; i <= nCoeffs; i++) {
        int level = block[i];
        if (level) block[i] = (int16_t)(level < 0 ? level * qmul - qadd : level * qmul + qadd);
    }
}

static inline int mp4_clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

// Scalar half-pel MC with per-tap edge clamp to [0,w-1]x[0,h-1] (== ffmpeg
// emulated_edge_mc + av_clip + dxy-fixup). The slow edge fallback; the fast inside
// path below (used when the whole block + its half-pel taps lie within the frame)
// produces byte-identical results because every clamp there is the identity.
static inline void mp4_mc_block_edge(uint8_t* dst, int dst_stride,
                                     const uint8_t* ref, int ref_stride, int w, int h,
                                     int block_w, int block_h, int src_x, int src_y,
                                     int dxy, int rb, int avg) {
    for (int by = 0; by < block_h; by++) {
        int y0 = mp4_clampi(src_y + by,     0, h - 1);
        int y1 = mp4_clampi(src_y + by + 1, 0, h - 1);
        const uint8_t* r0 = ref + y0 * ref_stride;
        const uint8_t* r1 = ref + y1 * ref_stride;
        uint8_t* o = dst + by * dst_stride;
        for (int bx = 0; bx < block_w; bx++) {
            int x0 = mp4_clampi(src_x + bx,     0, w - 1);
            int x1 = mp4_clampi(src_x + bx + 1, 0, w - 1);
            int a = r0[x0], v;
            if      (dxy == 0) v = a;
            else if (dxy == 1) v = (a + r0[x1] + rb) >> 1;
            else if (dxy == 2) v = (a + r1[x0] + rb) >> 1;
            else               v = (a + r0[x1] + r1[x0] + r1[x1] + rb + 1) >> 2;
            o[bx] = avg ? (uint8_t)((o[bx] + v + 1) >> 1) : (uint8_t)v;
        }
    }
}

#ifdef __wasm_simd128__
#include <wasm_simd128.h>
// Load/store block_w (8 or 16) contiguous bytes as the low lanes of a v128.
static inline v128_t mp4_v_load(const uint8_t* p, int n) {
    return n == 16 ? wasm_v128_load(p) : wasm_v128_load64_zero(p);
}
static inline void mp4_v_store(uint8_t* p, v128_t v, int n) {
    if (n == 16) wasm_v128_store(p, v); else wasm_v128_store64_lane(p, v, 0);
}
// Per-lane floor average (a+b)>>1 (avgr gives ceil; subtract the odd-sum carry).
static inline v128_t mp4_v_avgfloor(v128_t a, v128_t b) {
    v128_t carry = wasm_v128_and(wasm_v128_xor(a, b), wasm_i8x16_splat(1));
    return wasm_i8x16_sub(wasm_u8x16_avgr(a, b), carry);
}
// Per-lane (a+b+c+d+bias)>>2 via u16 widen (sum <= 1020 fits in u16; result <= 255).
static inline v128_t mp4_v_bilin4(v128_t a, v128_t b, v128_t c, v128_t d, int bias) {
    v128_t bv = wasm_i16x8_splat((short)bias);
    v128_t lo = wasm_i16x8_add(wasm_i16x8_add(wasm_u16x8_extend_low_u8x16(a), wasm_u16x8_extend_low_u8x16(b)),
                               wasm_i16x8_add(wasm_u16x8_extend_low_u8x16(c), wasm_u16x8_extend_low_u8x16(d)));
    v128_t hi = wasm_i16x8_add(wasm_i16x8_add(wasm_u16x8_extend_high_u8x16(a), wasm_u16x8_extend_high_u8x16(b)),
                               wasm_i16x8_add(wasm_u16x8_extend_high_u8x16(c), wasm_u16x8_extend_high_u8x16(d)));
    lo = wasm_u16x8_shr(wasm_i16x8_add(lo, bv), 2);
    hi = wasm_u16x8_shr(wasm_i16x8_add(hi, bv), 2);
    return wasm_u8x16_narrow_i16x8(lo, hi);
}
#endif

// Fast half-pel MC for a block whose taps all lie inside the frame (no clamp). The
// dxy branch + avg flag are hoisted out of the pixel loop; the row op is SIMD where
// available. Byte-identical to mp4_mc_block_edge under the inside guard.
static inline void mp4_mc_block_inside(uint8_t* dst, int dst_stride,
                                       const uint8_t* base, int ref_stride,
                                       int block_w, int block_h, int dxy, int rb, int avg) {
#ifdef __wasm_simd128__
    const int n = block_w;
    for (int by = 0; by < block_h; by++) {
        const uint8_t* r0 = base + by * ref_stride;
        uint8_t* o = dst + by * dst_stride;
        v128_t a = mp4_v_load(r0, n), v;
        if (dxy == 0)      v = a;
        else if (dxy == 1) { v128_t b = mp4_v_load(r0 + 1, n);            v = rb ? wasm_u8x16_avgr(a, b) : mp4_v_avgfloor(a, b); }
        else if (dxy == 2) { v128_t b = mp4_v_load(r0 + ref_stride, n);   v = rb ? wasm_u8x16_avgr(a, b) : mp4_v_avgfloor(a, b); }
        else {
            const uint8_t* r1 = r0 + ref_stride;
            v = mp4_v_bilin4(a, mp4_v_load(r0 + 1, n), mp4_v_load(r1, n), mp4_v_load(r1 + 1, n), rb + 1);
        }
        if (avg) v = wasm_u8x16_avgr(mp4_v_load(o, n), v);
        mp4_v_store(o, v, n);
    }
#else
    for (int by = 0; by < block_h; by++) {
        const uint8_t* r0 = base + by * ref_stride;
        const uint8_t* r1 = r0 + ref_stride;
        uint8_t* o = dst + by * dst_stride;
        for (int bx = 0; bx < block_w; bx++) {
            int a = r0[bx], v;
            if      (dxy == 0) v = a;
            else if (dxy == 1) v = (a + r0[bx + 1] + rb) >> 1;
            else if (dxy == 2) v = (a + r1[bx] + rb) >> 1;
            else               v = (a + r0[bx + 1] + r1[bx] + r1[bx + 1] + rb + 1) >> 2;
            o[bx] = avg ? (uint8_t)((o[bx] + v + 1) >> 1) : (uint8_t)v;
        }
    }
#endif
}

// Half-pel MC of a block_w x block_h region from `ref` (w x h, stride) at
// (src_x,src_y) + half-pel `dxy` into dst. rb = put bias (1 = rounding, 0 = no-
// rounding). avg != 0 -> rnd-average into dst (ffmpeg avg_pixels, B bidir).
// Dispatches to the SIMD inside path when the block + half-pel taps are within the
// frame (the common case), else the per-tap-clamped edge fallback.
static inline void mp4_mc_block(uint8_t* dst, int dst_stride,
                                const uint8_t* ref, int ref_stride, int w, int h,
                                int block_w, int block_h, int src_x, int src_y,
                                int dxy, int rb, int avg) {
    int hx = src_x + block_w - 1 + (dxy & 1);            // max x tap index read
    int hy = src_y + block_h - 1 + ((dxy >> 1) & 1);     // max y tap index read
    if (src_x >= 0 && src_y >= 0 && hx <= w - 1 && hy <= h - 1) {
        mp4_mc_block_inside(dst, dst_stride, ref + src_y * ref_stride + src_x, ref_stride,
                            block_w, block_h, dxy, rb, avg);
    } else {
        mp4_mc_block_edge(dst, dst_stride, ref, ref_stride, w, h, block_w, block_h, src_x, src_y, dxy, rb, avg);
    }
}

// ── EDGECLIP toggle (magic word: "EDGECLIP") ─────────────────────────────────────
// ffmpeg's 4MV paths (apply_8x8 / hpel_motion / chroma_4mv_motion) clip each 8x8
// sub-block's src to the DISPLAY dims and clear the subpel bit at that edge BEFORE
// the (coded-edge) emulated-edge fetch. We do NOT, by default — it is a verified
// no-op on every clip reachable here (ffmpeg's own encoder is self-consistent;
// libxvid/DivX edge-pad so the band [displayW, codedW) is ~constant, making the
// clip's read of col displayW vs col displayW+k identical). Confirmed empirically:
// ffmpeg-mp4v/DivX = max=0 both ways; a libxvid 178x100 / 170x106 qpel+4MV clip is
// ALSO max=0 WITHOUT this clip once the IDCT is isolated (ffmpeg `-idct simpleauto`,
// because xvid_build>=0 makes ffmpeg switch to the XviD IDCT — a separate, out-of-
// scope ~1-2 LSB algorithm difference, NOT this clip). Left OFF for a lean hot path.
//
// TO REVISIT — say "EDGECLIP" to an AI: flip MP4_EDGECLIP to 1 to compile in the
// exact ffmpeg-faithful clip (zero risk; matches ffmpeg's decoder for ANY input,
// incl. adversarial/hand-crafted non-aligned streams whose [displayW,codedW) band
// is non-constant). To EARN a regression test it must first be made to TRIGGER:
// sweep MANY non-16-aligned clips, ALL frames, from varied/third-party encoders,
// comparing vs ffmpeg `-cpuflags 0 -idct simpleauto` (isolates MC from the XviD
// IDCT). If a triggering clip is found, set MP4_EDGECLIP=1 and commit that clip.
#ifndef MP4_EDGECLIP
#define MP4_EDGECLIP 0
#endif

// Motion-compensate the current MB into dest{Y,U,V} from ref{Y,U,V} using one MV
// set (mv[block][xy], up to 4MV). Luma half-pel (M2) or qpel (M3); chroma always
// half-pel w/ the qpel-derived MV. rnd/rb select rounding; avg!=0 rnd-averages
// into dest (B bidir — ffmpeg's second-direction op_avg). Shared by P (cur<-ref)
// and B (bbuf<-forward/backward). See EDGECLIP above for the gated 4MV edge clip.
static inline void mp4_mc_pred(Mp4Decoder* d, uint8_t* destY, uint8_t* destU, uint8_t* destV,
                               const uint8_t* refY, const uint8_t* refU, const uint8_t* refV,
                               const int mv[4][2], int mv_type, int rnd, int rb, int avg) {
    int cw = d->codedW >> 1, ch = d->codedH >> 1;
    int yoff = (d->mb_y * 16) * d->yStride + d->mb_x * 16;
    int coff = (d->mb_y * 8)  * d->cStride + d->mb_x * 8;
#ifdef MP4_PROF_NO_MC
    // Profiling stub: skip interpolation. Touch one ref byte per plane so the ref
    // pointers / MV decode aren't dead-code-eliminated (timing only — not correct).
    destY[yoff] = refY[yoff]; destU[coff] = refU[coff]; destV[coff] = refV[coff];
    (void)mv; (void)mv_type; (void)rnd; (void)rb; (void)avg; (void)cw; (void)ch; return;
#endif
    uint8_t full[17 * 17];                       // clamped luma ref block (qpel)

    if (mv_type == 0) {                         // 16x16 (1MV)
        int mx = mv[0][0], my = mv[0][1];
        if (d->quarter_sample) {                // qpel luma (ffmpeg qpel_motion)
            int dxy = ((my & 3) << 2) | (mx & 3);
            int sx = d->mb_x * 16 + (mx >> 2), sy = d->mb_y * 16 + (my >> 2);
            mp4_qpel_luma(destY + yoff, d->yStride, refY, d->yStride, d->codedW, d->codedH,
                          sx, sy, 16, dxy, rnd, avg, full, 17);
            int cmx = mx / 2, cmy = my / 2;     // trunc toward zero (NOT >>1)
            cmx = (cmx >> 1) | (cmx & 1);
            cmy = (cmy >> 1) | (cmy & 1);
            int uvdxy = (cmx & 1) | ((cmy & 1) << 1);
            cmx >>= 1; cmy >>= 1;
            int usx = d->mb_x * 8 + cmx, usy = d->mb_y * 8 + cmy;
            mp4_mc_block(destU + coff, d->cStride, refU, d->cStride, cw, ch, 8, 8, usx, usy, uvdxy, rb, avg);
            mp4_mc_block(destV + coff, d->cStride, refV, d->cStride, cw, ch, 8, 8, usx, usy, uvdxy, rb, avg);
        } else {                                // half-pel luma (M2)
            int dxy = ((my & 1) << 1) | (mx & 1);
            int sx = d->mb_x * 16 + (mx >> 1), sy = d->mb_y * 16 + (my >> 1);
            mp4_mc_block(destY + yoff, d->yStride, refY, d->yStride, d->codedW, d->codedH, 16, 16, sx, sy, dxy, rb, avg);
            int uvdxy = dxy | (my & 2) | ((mx & 2) >> 1);
            int usx = sx >> 1, usy = sy >> 1;
            mp4_mc_block(destU + coff, d->cStride, refU, d->cStride, cw, ch, 8, 8, usx, usy, uvdxy, rb, avg);
            mp4_mc_block(destV + coff, d->cStride, refV, d->cStride, cw, ch, 8, 8, usx, usy, uvdxy, rb, avg);
        }
    } else {                                    // 8x8 (4MV)
        int sumx = 0, sumy = 0;
        for (int i = 0; i < 4; i++) {
            int mx = mv[i][0], my = mv[i][1];
            int bx = (i & 1) * 8, byy = (i >> 1) * 8;
            if (d->quarter_sample) {            // qpel luma (ffmpeg apply_8x8)
                int dxy = ((my & 3) << 2) | (mx & 3);
                int sx = d->mb_x * 16 + bx + (mx >> 2), sy = d->mb_y * 16 + byy + (my >> 2);
#if MP4_EDGECLIP
                sx = mp4_clampi(sx, -16, d->width);  if (sx == d->width)  dxy &= ~3;
                sy = mp4_clampi(sy, -16, d->height); if (sy == d->height) dxy &= ~12;
#endif
                mp4_qpel_luma(destY + yoff + byy * d->yStride + bx, d->yStride, refY, d->yStride,
                              d->codedW, d->codedH, sx, sy, 8, dxy, rnd, avg, full, 9);
                sumx += mx / 2; sumy += my / 2; // trunc toward zero
            } else {                            // half-pel luma (M2)
                int dxy = ((my & 1) << 1) | (mx & 1);
                int sx = d->mb_x * 16 + bx + (mx >> 1), sy = d->mb_y * 16 + byy + (my >> 1);
#if MP4_EDGECLIP
                sx = mp4_clampi(sx, -16, d->width);  if (sx == d->width)  dxy &= ~1;
                sy = mp4_clampi(sy, -16, d->height); if (sy == d->height) dxy &= ~2;
#endif
                mp4_mc_block(destY + yoff + byy * d->yStride + bx, d->yStride, refY, d->yStride,
                             d->codedW, d->codedH, 8, 8, sx, sy, dxy, rb, avg);
                sumx += mx; sumy += my;
            }
        }
        // single chroma MV with H.263 special rounding (chroma_4mv_motion)
        int mx = mp4_round_chroma(sumx), my = mp4_round_chroma(sumy);
        int dxy = ((my & 1) << 1) | (mx & 1);
        mx >>= 1; my >>= 1;
        int sx = d->mb_x * 8 + mx, sy = d->mb_y * 8 + my;
#if MP4_EDGECLIP
        sx = mp4_clampi(sx, -8, d->width >> 1);  if (sx == (d->width >> 1))  dxy &= ~1;
        sy = mp4_clampi(sy, -8, d->height >> 1); if (sy == (d->height >> 1)) dxy &= ~2;
#endif
        mp4_mc_block(destU + coff, d->cStride, refU, d->cStride, cw, ch, 8, 8, sx, sy, dxy, rb, avg);
        mp4_mc_block(destV + coff, d->cStride, refV, d->cStride, cw, ch, 8, 8, sx, sy, dxy, rb, avg);
    }
}

// Block destination pointer within the given base planes (cur=y/u/v or B=bY/bU/bV).
static inline void mp4_block_dest_in(Mp4Decoder* d, uint8_t* baseY, uint8_t* baseU, uint8_t* baseV,
                                     int n, uint8_t** dest, int* stride) {
    if (n < 4)       { *stride = d->yStride; *dest = baseY + (size_t)(d->mb_y * 16 + (n >> 1) * 8) * d->yStride + d->mb_x * 16 + (n & 1) * 8; }
    else if (n == 4) { *stride = d->cStride; *dest = baseU + (size_t)(d->mb_y * 8) * d->cStride + d->mb_x * 8; }
    else             { *stride = d->cStride; *dest = baseV + (size_t)(d->mb_y * 8) * d->cStride + d->mb_x * 8; }
}

// ── profiling stubs (compile-time; no effect on the shipped build) ─────────────
// MP4_PROF_NO_IDCT replaces the transform with an O(1) consumer of block[0] so the
// coefficient parse is preserved (not DCE'd) while the butterfly cost is removed.
#ifdef MP4_PROF_NO_IDCT
static inline void mp4_idct_put_stub(uint8_t* dest, int stride, int16_t* b) { (void)stride; dest[0] = (uint8_t)mp4_clampi(b[0] >> 3, 0, 255); }
static inline void mp4_idct_add_stub(uint8_t* dest, int stride, int16_t* b) { (void)stride; dest[0] = (uint8_t)mp4_clampi(dest[0] + (b[0] >> 3), 0, 255); }
#define MP4_IDCT8_PUT(dst, s, blk) mp4_idct_put_stub(dst, s, blk)
#define MP4_IDCT8_ADD(dst, s, blk) mp4_idct_add_stub(dst, s, blk)
#else
#define MP4_IDCT8_PUT(dst, s, blk) idct8_put_simd(dst, s, blk)
#define MP4_IDCT8_ADD(dst, s, blk) idct8_add_simd(dst, s, blk)
#endif

// I/P reconstruct into the current planes (d->y/u/v); P uses the forward MV set.
static inline void mp4_reconstruct_mb(Mp4Decoder* d) {
    if (d->mb_intra) {
        for (int n = 0; n < 6; n++) {
            mp4_dequant_h263_intra(d, d->block[n], n);
            uint8_t* dest; int stride; mp4_block_dest_in(d, d->y, d->u, d->v, n, &dest, &stride);
            MP4_IDCT8_PUT(dest, stride, d->block[n]);
        }
    } else {
        int rb = d->no_rounding ? 0 : 1;
        mp4_mc_pred(d, d->y, d->u, d->v, d->refY, d->refU, d->refV, d->mv[0], d->mv_type, !d->no_rounding, rb, 0);
        for (int n = 0; n < 6; n++) {
            if (d->block_last_index[n] < 0) continue;   // not coded (incl. skip)
            uint8_t* dest; int stride; mp4_block_dest_in(d, d->y, d->u, d->v, n, &dest, &stride);
            MP4_IDCT8_ADD(dest, stride, d->block[n]);    // inter blocks arrive pre-dequantized
        }
    }
}

// B reconstruct into the B-output planes (d->bY/U/V). Forward MC from the forward
// ref (refY/U/V = last I/P) with put; backward MC from the backward ref (y/u/v =
// next I/P) with avg when bidir (ffmpeg mpv_reconstruct_mb B path). B always uses
// rounding (no_rounding is 0 for B). Residual (pre-dequantized inter) is idct_add'd.
static inline void mp4_reconstruct_mb_b(Mp4Decoder* d) {
    int did_fwd = 0;
    if (d->mv_dir & 1) {                                  // MV_DIR_FORWARD
        mp4_mc_pred(d, d->bY, d->bU, d->bV, d->refY, d->refU, d->refV, d->mv[0], d->mv_type, 1, 1, 0);
        did_fwd = 1;
    }
    if (d->mv_dir & 2) {                                  // MV_DIR_BACKWARD
        mp4_mc_pred(d, d->bY, d->bU, d->bV, d->y, d->u, d->v, d->mv[1], d->mv_type, 1, 1, did_fwd);
    }
    for (int n = 0; n < 6; n++) {
        if (d->block_last_index[n] < 0) continue;
        uint8_t* dest; int stride; mp4_block_dest_in(d, d->bY, d->bU, d->bV, n, &dest, &stride);
        MP4_IDCT8_ADD(dest, stride, d->block[n]);
    }
}

// ── I-VOP macroblock (mpeg4videodec.c:2038) ────────────────────────────────────

// Shared intra-MB body (ffmpeg's `intra:` label — reached from I-VOP and P-intra).
static inline int mp4_decode_intra_blocks(Mp4Decoder* d, int cbpc, int dquant) {
    d->mb_intra = 1;
    d->ac_pred = (int)get_bits1(&d->gb);
    int cbpy = get_vlc2(&d->gb, g_cbpy.table, g_cbpy.bits);
    if (cbpy < 0) return MP4_ERR_DATA;
    int cbp = (cbpc & 3) | (cbpy << 2);        // intra: cbpy used directly
    int use_intra_dc_vlc = d->qscale < d->intra_dc_threshold;
    if (dquant)
        mp4_set_qscale(d, d->qscale + (int8_t)mp4_quant_tab[get_bits(&d->gb, 2)]);
    if (!d->progressive_sequence)
        d->interlaced_dct = (int)get_bits1(&d->gb);   // gated (progressive) but read
    memset(d->block, 0, sizeof(d->block));
    for (int n = 0; n < 6; n++) {
        if (mp4_decode_block(d, d->block[n], n, cbp & 32, 1, use_intra_dc_vlc) < 0)
            return MP4_ERR_DATA;
        cbp += cbp;
    }
    return MP4_OK;
}

static inline int mp4_decode_mb_i(Mp4Decoder* d) {
    d->mb_skipped = 0;
    int cbpc;
    do {
        cbpc = get_vlc2(&d->gb, g_intra_mcbpc.table, g_intra_mcbpc.bits);
        if (cbpc < 0) return MP4_ERR_DATA;
    } while (cbpc == 8);                        // stuffing
    return mp4_decode_intra_blocks(d, cbpc, cbpc & 4);
}

static inline int mp4_decode_mb_p(Mp4Decoder* d) {
    if (get_bits1(&d->gb)) {                    // not_coded -> skip MB (mv 0,0, no residual)
        d->mb_intra = 0; d->mv_type = 0; d->mb_skipped = 1;
        d->mv[0][0][0] = d->mv[0][0][1] = 0;
        for (int n = 0; n < 6; n++) d->block_last_index[n] = -1;
        return MP4_OK;
    }
    d->mb_skipped = 0;
    int cbpc;
    do {
        cbpc = get_vlc2(&d->gb, g_inter_mcbpc.table, g_inter_mcbpc.bits);
        if (cbpc < 0) return MP4_ERR_DATA;
    } while (cbpc == 20);                       // stuffing
    int dquant = cbpc & 8;
    if (cbpc & 4) return mp4_decode_intra_blocks(d, cbpc, dquant);  // intra MB inside a P-VOP

    d->mb_intra = 0;
    memset(d->block, 0, sizeof(d->block));
    int cbpy = get_vlc2(&d->gb, g_cbpy.table, g_cbpy.bits) ^ 0x0F;  // INVERTED for inter
    if (cbpy < 0) return MP4_ERR_DATA;
    int cbp = (cbpc & 3) | (cbpy << 2);
    if (dquant)
        mp4_set_qscale(d, d->qscale + (int8_t)mp4_quant_tab[get_bits(&d->gb, 2)]);
    // progressive: no interlaced_dct

    if ((cbpc & 16) == 0) {                     // 1MV (16x16)
        d->mv_type = 0;
        int px, py; mp4_pred_motion(d, 0, &px, &py);
        int mx = mp4_decode_motion(d, px, d->f_code); if (mx >= 0xffff) return MP4_ERR_DATA;
        int my = mp4_decode_motion(d, py, d->f_code); if (my >= 0xffff) return MP4_ERR_DATA;
        d->mv[0][0][0] = mx; d->mv[0][0][1] = my;
    } else {                                    // 4MV (8x8)
        d->mv_type = 1;
        for (int i = 0; i < 4; i++) {
            int px, py; int xy = mp4_pred_motion(d, i, &px, &py);
            int mx = mp4_decode_motion(d, px, d->f_code); if (mx >= 0xffff) return MP4_ERR_DATA;
            int my = mp4_decode_motion(d, py, d->f_code); if (my >= 0xffff) return MP4_ERR_DATA;
            d->mv[0][i][0] = mx; d->mv[0][i][1] = my;
            d->mot_val[xy*2] = (int16_t)mx; d->mot_val[xy*2+1] = (int16_t)my;
        }
    }

    for (int n = 0; n < 6; n++) {
        if (mp4_decode_block(d, d->block[n], n, cbp & 32, 0, 0) < 0)
            return MP4_ERR_DATA;
        cbp += cbp;
    }
    return MP4_OK;
}

// ── B-VOP macroblock (mpeg4videodec.c:1885) ────────────────────────────────────

// Direct-mode MV for one block from the co-located backward-ref (next P) MV.
// Division form == ffmpeg's direct_scale_mv table (bit-identical): forward =
// p_mv*pb/pp + delta; backward = delta ? forward - p_mv : p_mv*(pb-pp)/pp.
static inline void mp4_set_one_direct_mv(Mp4Decoder* d, int mx, int my, int i) {
    int xy = d->block_index[i];
    int pp = d->pp_time, pb = d->pb_time;
    // pp_time<=0 is degenerate: ffmpeg's sequential decode keeps pp_time strictly
    // positive (consecutive I/P anchors have increasing time), so it only arises when a
    // reader re-decodes a GOP out of its natural monotonic time order — a random-access
    // scrub — which scrambles the ABSOLUTE B timing. ffmpeg FRAME_SKIPs such B's after a
    // seek (and would itself divide by zero otherwise — it has no guard); our readers
    // instead decode them from a keyframe walk (the references ARE valid), so we decode
    // with a safe non-trapping scale rather than trap the 0/0. The scaled direct MV may
    // be wrong on this path, but it only occurs on streams ffmpeg would skip; in the
    // common case the backward ref is the GOP-boundary I (intra, co-located MV 0) so the
    // result is 0 regardless and matches ffmpeg's sequential reference.
    if (pp <= 0) pp = 1;
    int p_mx = d->bwd_mot_val[xy * 2];
    d->mv[0][i][0] = p_mx * pb / pp + mx;
    d->mv[1][i][0] = mx ? d->mv[0][i][0] - p_mx : p_mx * (pb - pp) / pp;
    int p_my = d->bwd_mot_val[xy * 2 + 1];
    d->mv[0][i][1] = p_my * pb / pp + my;
    d->mv[1][i][1] = my ? d->mv[0][i][1] - p_my : p_my * (pb - pp) / pp;
}

// Direct-mode (ff_mpeg4_set_direct_mv). Sets mv_type + mv[0]/mv[1] from the
// co-located next-P MB type (8x8 -> 4 per-block scaled MVs; else 16x16 replicated;
// for qpel the 16x16 case still uses mv_type 8x8 with 4 identical MVs, per ffmpeg).
static inline void mp4_set_direct_mv(Mp4Decoder* d, int mx, int my) {
    int mb_index = d->mb_x + d->mb_y * d->mb_stride;
    if (d->bwd_is8x8[mb_index]) {
        d->mv_type = 1;
        for (int i = 0; i < 4; i++) mp4_set_one_direct_mv(d, mx, my, i);
    } else {
        mp4_set_one_direct_mv(d, mx, my, 0);
        for (int dir = 0; dir < 2; dir++)
            for (int i = 1; i < 4; i++) { d->mv[dir][i][0] = d->mv[dir][0][0]; d->mv[dir][i][1] = d->mv[dir][0][1]; }
        d->mv_type = d->quarter_sample ? 1 : 0;   // qpel uses 8x8 (4 identical) to match ffmpeg
    }
}

// B mb_type VLC: lengths {1,2,3,4} all code 1 -> count leading zeros. Returns the
// symbol 0=direct, 1=interpolate(bidir), 2=backward, 3=forward; -1 on 4 zeros.
static inline int mp4_decode_mb_type_b(GetBits* gb) {
    for (int n = 0; n < 4; n++) if (get_bits1(gb)) return n;
    return -1;
}

static inline int mp4_decode_mb_b(Mp4Decoder* d) {
    int mb_index = d->mb_x + d->mb_y * d->mb_stride;
    d->mb_intra = 0;
    if (d->mb_x == 0)
        d->b_last_mv[0][0] = d->b_last_mv[0][1] = d->b_last_mv[1][0] = d->b_last_mv[1][1] = 0;

    // B-skip: co-located P-MB was skipped -> forward MV(0,0), no residual.
    if (d->bwd_skip[mb_index]) {
        for (int n = 0; n < 6; n++) d->block_last_index[n] = -1;
        d->mv_dir = 1; d->mv_type = 0;
        d->mv[0][0][0] = d->mv[0][0][1] = 0;
        return MP4_OK;
    }

    int cbp = 0, is_direct, direct_skip = 0, sym = -1;
    if (get_bits1(&d->gb)) {                    // modb1: direct, no coded vectors
        is_direct = 1; direct_skip = 1;
    } else {
        int modb2 = (int)get_bits1(&d->gb);
        sym = mp4_decode_mb_type_b(&d->gb);
        if (sym < 0) return MP4_ERR_DATA;
        is_direct = (sym == 0);
        if (modb2) cbp = 0;
        else { memset(d->block, 0, sizeof(d->block)); cbp = (int)get_bits(&d->gb, 6); }
        if (!is_direct && cbp) {               // dquant
            if (get_bits1(&d->gb)) mp4_set_qscale(d, d->qscale + (get_bits1(&d->gb) ? 2 : -2));
        }
        d->mv_dir = 0;
        if (!is_direct) {                      // explicit 16x16 fwd/bwd/bidir
            d->mv_type = 0;
            if (sym == 1 || sym == 3) {        // forward
                d->mv_dir |= 1;
                int mx = mp4_decode_motion(d, d->b_last_mv[0][0], d->f_code); if (mx >= 0xffff) return MP4_ERR_DATA;
                int my = mp4_decode_motion(d, d->b_last_mv[0][1], d->f_code); if (my >= 0xffff) return MP4_ERR_DATA;
                d->b_last_mv[0][0] = d->mv[0][0][0] = mx;
                d->b_last_mv[0][1] = d->mv[0][0][1] = my;
            }
            if (sym == 1 || sym == 2) {        // backward
                d->mv_dir |= 2;
                int mx = mp4_decode_motion(d, d->b_last_mv[1][0], d->b_code); if (mx >= 0xffff) return MP4_ERR_DATA;
                int my = mp4_decode_motion(d, d->b_last_mv[1][1], d->b_code); if (my >= 0xffff) return MP4_ERR_DATA;
                d->b_last_mv[1][0] = d->mv[1][0][0] = mx;
                d->b_last_mv[1][1] = d->mv[1][0][1] = my;
            }
        }
    }

    if (is_direct) {
        int mx, my;
        if (direct_skip) { mx = my = 0; }
        else {
            mx = mp4_decode_motion(d, 0, 1); if (mx >= 0xffff) return MP4_ERR_DATA;
            my = mp4_decode_motion(d, 0, 1); if (my >= 0xffff) return MP4_ERR_DATA;
        }
        d->mv_dir = 1 | 2;
        mp4_set_direct_mv(d, mx, my);
    }

    for (int n = 0; n < 6; n++) {
        if (mp4_decode_block(d, d->block[n], n, cbp & 32, 0, 0) < 0) return MP4_ERR_DATA;
        cbp += cbp;
    }
    return MP4_OK;
}

static inline int mp4_decode_mb(Mp4Decoder* d) {
    if (d->pict_type == 1) return mp4_decode_mb_i(d);
    if (d->pict_type == 2) return mp4_decode_mb_p(d);
    return mp4_decode_mb_b(d);
}

// ── video packet (resync marker) handling — needed because the encoder splits
// each MB row into an independently-decodable video packet ───────────────────

static inline int mp4_av_log2(unsigned v) { int n = 0; while (v >>= 1) n++; return n; }

// Clean the AC-prediction buffers above the new packet + reset the B last_mv
// predictor (ff_mpeg4_clean_buffers).
static inline void mp4_clean_buffers(Mp4Decoder* d) {
    int l_wrap = d->b8_stride, c_wrap = d->mb_stride;
    int l_xy = (2 * d->mb_y - 1) * l_wrap + d->mb_x * 2 - 1;
    int u_xy = 2 * d->mb_height * l_wrap + d->mb_y * c_wrap + d->mb_x - 1;
    int v_xy = u_xy + c_wrap * (d->mb_height + 1);
    memset(d->ac_val + (size_t)l_xy * 16, 0, (size_t)(l_wrap * 2 + 1) * 16 * sizeof(int16_t));
    memset(d->ac_val + (size_t)u_xy * 16, 0, (size_t)(c_wrap + 1) * 16 * sizeof(int16_t));
    memset(d->ac_val + (size_t)v_xy * 16, 0, (size_t)(c_wrap + 1) * 16 * sizeof(int16_t));
    d->b_last_mv[0][0] = d->b_last_mv[0][1] = d->b_last_mv[1][0] = d->b_last_mv[1][1] = 0;
}

// Resync-marker prefix length (ff_mpeg4_get_video_packet_prefix_length): I=16,
// P/S = f_code+15, B = max(f_code,b_code,2)+15. NOTE f_code==1 P gives 16 (== I),
// which is why M1/M2's hardcoded 16 happened to work — every M2 clip was f_code==1.
static inline int mp4_resync_prefix_len(Mp4Decoder* d) {
    if (d->pict_type == 1) return 16;
    if (d->pict_type == 3) {                          // B
        int m = d->f_code > d->b_code ? d->f_code : d->b_code;
        if (m < 2) m = 2;
        return m + 15;
    }
    return d->f_code + 15;                             // P / S
}

// Peek for a resync marker at the current MB boundary (mpeg4_is_resync, RECT path).
// Returns the packet's mb_num (>0) if found, else 0. The stacked-marker while loop
// permanently advances gb (matches ffmpeg); the len-check is save/restored.
static inline int mp4_is_resync(Mp4Decoder* d) {
    static const uint16_t prefix[8] = { 0x7F00, 0x7E00, 0x7C00, 0x7800, 0x7000, 0x6000, 0x4000, 0x0000 };
    int mb_num = d->mb_width * d->mb_height;
    int bits_count = get_bits_count(&d->gb);
    int v = (int)show_bits(&d->gb, 16);

    while (v <= 0xFF) {
        if (d->pict_type == 3 /*B*/ || (v >> (8 - d->pict_type)) != 1) break;  // (no data-partition)
        skip_bits(&d->gb, 8 + d->pict_type);
        bits_count += 8 + d->pict_type;
        v = (int)show_bits(&d->gb, 16);
    }

    if (bits_count + 8 >= d->gb.size_in_bits) {
        v >>= 8;
        v |= 0x7F >> (7 - (bits_count & 7));
        if (v == 0x7F) return mb_num;
    } else if (v == prefix[bits_count & 7]) {
        GetBits save = d->gb;
        int mb_num_bits = mp4_av_log2(mb_num - 1) + 1;
        skip_bits1(&d->gb);
        align_get_bits(&d->gb);
        int len = 0;
        for (; len < 32; len++) if (get_bits1(&d->gb)) break;
        int mbn = (int)get_bits(&d->gb, mb_num_bits);
        if (!mbn || mbn > mb_num || get_bits_count(&d->gb) + 6 > d->gb.size_in_bits) mbn = -1;
        d->gb = save;
        if (len >= mp4_resync_prefix_len(d)) return mbn;
    }
    return 0;
}

// Consume the video packet header (ff_h263_resync + ff_mpeg4_decode_video_packet_header,
// RECT). Sets mb_x/mb_y from mb_num and the packet qscale. Returns 0/neg.
static inline int mp4_consume_video_packet(Mp4Decoder* d) {
    skip_bits1(&d->gb);
    align_get_bits(&d->gb);
    int mb_num = d->mb_width * d->mb_height;
    int mb_num_bits = mp4_av_log2(mb_num - 1) + 1;
    if (get_bits_count(&d->gb) > d->gb.size_in_bits - 20) return MP4_ERR_DATA;
    int len = 0;
    for (; len < 32; len++) if (get_bits1(&d->gb)) break;
    if (len != mp4_resync_prefix_len(d)) return MP4_ERR_DATA;
    int mbn = (int)get_bits(&d->gb, mb_num_bits);
    if (mbn >= mb_num || !mbn) return MP4_ERR_DATA;
    d->mb_x = mbn % d->mb_width;
    d->mb_y = mbn / d->mb_width;
    int q = (int)get_bits(&d->gb, d->quant_precision); // quant_scale
    if (q) mp4_set_qscale(d, q);
    int hec = (int)get_bits1(&d->gb);                  // header_extension_code
    if (hec) {
        while (get_bits1(&d->gb) != 0) {}              // modulo_time_base
        get_bits1(&d->gb);                             // marker
        skip_bits(&d->gb, d->time_increment_bits);     // vop_time_increment
        get_bits1(&d->gb);                             // marker
        skip_bits(&d->gb, 2);                          // vop_coding_type
        skip_bits(&d->gb, 3);                          // intra_dc_vlc_thr
        if (d->pict_type != 1) skip_bits(&d->gb, 3);   // f_code (P/S/B)
        if (d->pict_type == 3) skip_bits(&d->gb, 3);   // b_code (B)
    }
    return MP4_OK;
}

static inline int mp4_decode_picture(Mp4Decoder* d) {
    // Reset DC/AC/MV prediction grids (per-picture spatial prediction).
    for (int i = 0; i < d->grid_elems; i++) d->dc_base[i] = 1024;
    memset(d->ac_base, 0, (size_t)d->grid_elems * 16 * sizeof(int16_t));
    memset(d->mot_base, 0, (size_t)d->grid_elems * 2 * sizeof(int16_t));

    int mb_num = d->mb_width * d->mb_height;
    d->mb_x = 0; d->mb_y = 0;
    d->resync_mb_x = 0; d->resync_mb_y = 0;
    d->first_slice_line = 1;

    int is_b = (d->pict_type == 3);
    for (int decoded = 0; decoded < mb_num; decoded++) {
        if (d->mb_x == d->resync_mb_x && d->mb_y == d->resync_mb_y + 1)
            d->first_slice_line = 0;
        mp4_init_block_index(d);
        if (mp4_decode_mb(d) < 0) return MP4_ERR_DATA;
        if (is_b) {
            mp4_reconstruct_mb_b(d);
        } else {
            if (d->pict_type == 2) mp4_update_motion_val(d);
            mp4_reconstruct_mb(d);
            // Capture per-MB metadata for the B-VOPs that reference this I/P backward.
            int mbi = d->mb_y * d->mb_stride + d->mb_x;
            d->bwd_skip[mbi]  = (uint8_t)d->mb_skipped;
            d->bwd_is8x8[mbi] = (uint8_t)(!d->mb_intra && d->mv_type == 1);
        }

        if (decoded + 1 < mb_num) {
            int next = mp4_is_resync(d);
            // For B-VOPs, trailing implicit-skip MBs (no bits in the stream) sit
            // between the last coded MB and the resync marker. ffmpeg keeps decoding
            // them (SLICE_OK) while the NEXT MB is a co-located skip, only consuming
            // the marker once the resync target (mb_num) is reached. Without this the
            // trailing B-skips are left undecoded (stale) — gross edge garbage.
            int consume = next != 0;
            if (next && is_b) {
                int cur_abs = d->mb_x + d->mb_y * d->mb_width;
                if (cur_abs + 1 < next) {
                    int delta = (d->mb_x + 1 == d->mb_width) ? 2 : 1;
                    int xy = d->mb_y * d->mb_stride + d->mb_x;
                    if (d->bwd_skip[xy + delta]) consume = 0;   // next MB is a trailing skip
                }
            }
            if (consume) {
                if (mp4_consume_video_packet(d) < 0) return MP4_ERR_DATA;
                d->resync_mb_x = d->mb_x;
                d->resync_mb_y = d->mb_y;
                d->first_slice_line = 1;
                mp4_clean_buffers(d);
            } else {
                if (++d->mb_x >= d->mb_width) { d->mb_x = 0; d->mb_y++; }
            }
        }
    }
    // Snapshot the I/P motion field for the following B-VOPs' direct mode.
    if (!is_b) memcpy(d->bwd_mot, d->mot_base, (size_t)d->grid_elems * 2 * sizeof(int16_t));
    return MP4_OK;
}

// ════════════════════════════════════════════════════════════════════════════════
//  H.263 short-header (ITU-T) decode (M5) — transcription of FFmpeg release/8.1
//  libavcodec/ituh263dec.c + h263dec.c + h263data.c (LGPL). Reuses the shared
//  pieces above: MV prediction/decode (== ff_h263_pred_motion / ff_h263_decode_
//  motion), half-pel MC (mp4_mc_pred), H.263 intra dequant (mp4_dequant_h263_intra),
//  ff_simple_idct8, and the inter MCBPC/CBPY/MV/RL VLC tables. The codec-specific
//  deltas are: a 22-bit picture start code + short picture header (vs VOL/VOP), the
//  H.263 intra block (fixed 8-bit INTRADC, no DC/AC prediction), the H.263 TCOEF
//  escape (vs MPEG-4's 3-mode escape), and the GOB layer (vs MPEG-4 video packets).
//  Scope: baseline H.263 I/P, rectangular QCIF/CIF family. Advanced modes (UMV, SAC,
//  Advanced Prediction, PB-frames, CPM) and extended/custom PTYPE are gated UNSUP.
// ════════════════════════════════════════════════════════════════════════════════

// Source format -> picture dimensions (h263data.c ff_h263_format; idx 0/6/7 invalid
// here: 0 forbidden, 6 custom CPFMT, 7 extended PLUSPTYPE — all gated).
static const uint16_t ff_h263_format[6][2] = {
    {    0,    0 }, {  128,   96 }, {  176,  144 },
    {  352,  288 }, {  704,  576 }, { 1408, 1152 },
};

// H.263 inter dequant (dct_unquantize_h263_inter_c): raw levels read via rl_vlc[0]
// are scaled here (qmul = 2*qscale, qadd = (qscale-1)|1) over the whole block incl.
// DC (no special intra-DC term for inter). nCoeffs from the zigzag raster_end.
static inline void mp4_dequant_h263_inter(Mp4Decoder* d, int16_t* block, int n) {
    int qmul = d->qscale << 1;
    int qadd = (d->qscale - 1) | 1;
    int nCoeffs = d->intra_raster_end[d->block_last_index[n]];
    for (int i = 0; i <= nCoeffs; i++) {
        int level = block[i];
        if (level) block[i] = (int16_t)(level < 0 ? level * qmul - qadd : level * qmul + qadd);
    }
}

// H.263 block decode (h263_decode_block, non-AIC / non-RV10 / non-FLV). Reads RAW
// levels from rl_vlc[0] (qmul=1/qadd=0); intra DC is a fixed 8-bit field (255->128),
// the rest dequantizes later. Escape is the H.263 TCOEF form: 7-bit run (top bit =
// LAST, folded so i>=64 triggers last-coeff recovery), 8-bit signed level, and the
// 11-bit extended escape when level == -128. `intra` selects the fixed-DC prologue.
static inline int h263_decode_block(Mp4Decoder* d, int16_t* block, int n,
                                    int coded, int intra) {
    GetBits* gb = &d->gb;
    const RL_VLC_ELEM* rl_vlc = g_rl_inter[0].table;   // raw level (qmul=1, qadd=0)
    int bits = g_rl_inter[0].bits;
    const uint8_t* scan_table = d->intra_scan;          // permutated zigzag
    int level, i, run;

    if (intra) {
        level = (int)get_bits(gb, 8);
        if (level == 255) level = 128;                  // 128 is escaped as 255
        block[0] = (int16_t)level;
        i = 1;
    } else {
        i = 0;
    }
    if (!coded) { d->block_last_index[n] = i - 1; return MP4_OK; }

    i--;                                                // offset by -1 for direct scan index
    for (;;) {
        get_rl_vlc(gb, rl_vlc, bits, &level, &run);
        if (run == 66) {                                // ESC (level 0) or illegal (level 64)
            if (level) return MP4_ERR_DATA;
            run   = (int)get_bits(gb, 7) + 1;           // LAST(1)|RUN(6), +1
            level = (int8_t)get_bits(gb, 8);            // signed 8-bit
            if (level == -128) {                        // 11-bit extended escape
                int lo = (int)get_bits(gb, 5);
                level = lo | (get_sbits(gb, 6) << 5);
            }
        } else {
            if (get_bits1(gb)) level = -level;          // sign
        }
        i += run;
        if (i >= 64) {                                  // last coefficient
            i = i - run + ((run - 1) & 63) + 1;
            if (i < 64) { block[scan_table[i]] = (int16_t)level; break; }
            return MP4_ERR_DATA;                        // run overflow
        }
        block[scan_table[i]] = (int16_t)level;
    }
    d->block_last_index[n] = i;
    return MP4_OK;
}

// H.263 intra-MB block group (shared by I-frame MBs and intra MBs inside a P-frame).
// Non-AIC: there is NO ac_pred bit and NO DC/AC prediction. cbpy is used directly
// (not inverted). dquant uses the same {-1,-2,1,2} table as MPEG-4.
static inline int h263_decode_intra_mb(Mp4Decoder* d, int cbpc, int dquant) {
    d->mb_intra = 1;
    d->ac_pred = 0;
    int cbpy = get_vlc2(&d->gb, g_cbpy.table, g_cbpy.bits);
    if (cbpy < 0) return MP4_ERR_DATA;
    int cbp = (cbpc & 3) | (cbpy << 2);
    if (dquant)
        mp4_set_qscale(d, d->qscale + (int8_t)mp4_quant_tab[get_bits(&d->gb, 2)]);
    memset(d->block, 0, sizeof(d->block));
    for (int n = 0; n < 6; n++) {
        if (h263_decode_block(d, d->block[n], n, cbp & 32, 1) < 0) return MP4_ERR_DATA;
        cbp += cbp;
    }
    return MP4_OK;
}

static inline int h263_decode_mb_i(Mp4Decoder* d) {
    d->mb_skipped = 0;
    int cbpc;
    do {
        cbpc = get_vlc2(&d->gb, g_intra_mcbpc.table, g_intra_mcbpc.bits);
        if (cbpc < 0) return MP4_ERR_DATA;
    } while (cbpc == 8);                                 // stuffing
    return h263_decode_intra_mb(d, cbpc, cbpc & 4);      // dquant = cbpc & 4
}

static inline int h263_decode_mb_p(Mp4Decoder* d) {
    if (get_bits1(&d->gb)) {                             // COD: skipped MB -> mv(0,0), no residual
        d->mb_intra = 0; d->mv_type = 0; d->mb_skipped = 1;
        d->mv[0][0][0] = d->mv[0][0][1] = 0;
        for (int n = 0; n < 6; n++) d->block_last_index[n] = -1;
        return MP4_OK;
    }
    d->mb_skipped = 0;
    int cbpc;
    do {
        cbpc = get_vlc2(&d->gb, g_inter_mcbpc.table, g_inter_mcbpc.bits);
        if (cbpc < 0) return MP4_ERR_DATA;
    } while (cbpc == 20);                                // stuffing
    int dquant = cbpc & 8;
    if (cbpc & 4) return h263_decode_intra_mb(d, cbpc, dquant);   // intra MB inside a P-frame

    d->mb_intra = 0;
    memset(d->block, 0, sizeof(d->block));
    int cbpy = get_vlc2(&d->gb, g_cbpy.table, g_cbpy.bits) ^ 0x0F; // INVERTED for inter
    if (cbpy < 0) return MP4_ERR_DATA;
    int cbp = (cbpc & 3) | (cbpy << 2);
    if (dquant)
        mp4_set_qscale(d, d->qscale + (int8_t)mp4_quant_tab[get_bits(&d->gb, 2)]);

    if ((cbpc & 16) == 0) {                              // 1MV (16x16); H.263 f_code is always 1
        d->mv_type = 0;
        int px, py; mp4_pred_motion(d, 0, &px, &py);
        int mx = mp4_decode_motion(d, px, 1); if (mx >= 0xffff) return MP4_ERR_DATA;
        int my = mp4_decode_motion(d, py, 1); if (my >= 0xffff) return MP4_ERR_DATA;
        d->mv[0][0][0] = mx; d->mv[0][0][1] = my;
    } else {                                             // 4MV (8x8)
        d->mv_type = 1;
        for (int i = 0; i < 4; i++) {
            int px, py; int xy = mp4_pred_motion(d, i, &px, &py);
            int mx = mp4_decode_motion(d, px, 1); if (mx >= 0xffff) return MP4_ERR_DATA;
            int my = mp4_decode_motion(d, py, 1); if (my >= 0xffff) return MP4_ERR_DATA;
            d->mv[0][i][0] = mx; d->mv[0][i][1] = my;
            d->mot_val[xy*2] = (int16_t)mx; d->mot_val[xy*2+1] = (int16_t)my;
        }
    }

    for (int n = 0; n < 6; n++) {
        if (h263_decode_block(d, d->block[n], n, cbp & 32, 0) < 0) return MP4_ERR_DATA;
        cbp += cbp;
    }
    return MP4_OK;
}

// Reconstruct an H.263 MB into the current planes. Intra: H.263 dequant (DC*8) +
// idct_put. Inter: half-pel MC + per-coded-block H.263 inter dequant + idct_add.
static inline void mp4_reconstruct_mb_h263(Mp4Decoder* d) {
    if (d->mb_intra) {
        for (int n = 0; n < 6; n++) {
            mp4_dequant_h263_intra(d, d->block[n], n);
            uint8_t* dest; int stride; mp4_block_dest_in(d, d->y, d->u, d->v, n, &dest, &stride);
            MP4_IDCT8_PUT(dest, stride, d->block[n]);
        }
    } else {
        int rb = d->no_rounding ? 0 : 1;
        mp4_mc_pred(d, d->y, d->u, d->v, d->refY, d->refU, d->refV, d->mv[0], d->mv_type, !d->no_rounding, rb, 0);
        for (int n = 0; n < 6; n++) {
            if (d->block_last_index[n] < 0) continue;
            mp4_dequant_h263_inter(d, d->block[n], n);
            uint8_t* dest; int stride; mp4_block_dest_in(d, d->y, d->u, d->v, n, &dest, &stride);
            MP4_IDCT8_ADD(dest, stride, d->block[n]);
        }
    }
}

// Consume an H.263 GOB header at a slice boundary (h263_decode_gob_header, non-slice-
// structured): GBSC = 16 zeros + GSTUFF zeros + '1', then GN(5)+GFID(2)+GQUANT(5).
// Sets mb_x=0, mb_y=gob_index*GN, qscale. Returns 0/neg. (Clean-stream path only —
// ffmpeg's corrupt-stream resync byte-scan is omitted; the oracle is clean ffmpeg.)
static inline int h263_resync_gob(Mp4Decoder* d) {
    GetBits* gb = &d->gb;
    if (show_bits(gb, 16) != 0) return -1;              // not a GBSC
    skip_bits(gb, 16);                                  // drop the 16 zeros
    int left = get_bits_left(gb);
    if (left > 32) left = 32;
    for (; left > 13; left--) { if (get_bits1(gb)) break; }   // seek the GBSC '1'
    if (left <= 13) return -1;
    int gn = (int)get_bits(gb, 5);                      // GN
    d->mb_x = 0;
    d->mb_y = d->gob_index * gn;
    skip_bits(gb, 2);                                   // GFID
    int gquant = (int)get_bits(gb, 5);                  // GQUANT
    if (d->mb_y >= d->mb_height) return -1;
    if (gquant == 0) return -1;
    mp4_set_qscale(d, gquant);
    return 0;
}

// Parse the H.263 short picture header (h263_decode_picture_header, baseline path).
// Returns MP4_DECODE (decode the picture) or a negative gate/error code.
static inline int h263_decode_picture_header(Mp4Decoder* d) {
    GetBits* gb = &d->gb;
    align_get_bits(gb);

    // 22-bit picture start code == 0x20 (scan forward like ffmpeg in case of leading junk).
    uint32_t startcode = get_bits(gb, 22 - 8);          // 14 bits
    for (int i = get_bits_left(gb); i > 24; i -= 8) {
        startcode = ((startcode << 8) | get_bits(gb, 8)) & 0x003FFFFF;
        if (startcode == 0x20) break;
    }
    if (startcode != 0x20) return MP4_ERR_NOVOP;

    skip_bits(gb, 8);                                   // temporal reference (I/P: timing unused)

    // PTYPE
    if (check_marker(gb) != 1) return MP4_ERR_DATA;     // bit 1 must be 1
    if (get_bits1(gb) != 0)    return MP4_ERR_DATA;     // bit 2 (H.263 id) must be 0
    skip_bits(gb, 3);                                   // split-screen, doc-camera, freeze-release
    int format = (int)get_bits(gb, 3);
    if (format == 7 || format == 6) return MP4_ERR_UNSUP;   // extended/custom PTYPE (PLUSPTYPE/CPFMT)
    int w = ff_h263_format[format][0], h = ff_h263_format[format][1];
    if (!w) return MP4_ERR_DATA;                        // format 0 = forbidden

    d->pict_type = 1 + (int)get_bits1(gb);              // 0 -> I(1), 1 -> P(2)
    if (get_bits1(gb)) return MP4_ERR_UNSUP;            // UMV (h263_long_vectors)
    if (get_bits1(gb)) return MP4_ERR_UNSUP;            // SAC
    if (get_bits1(gb)) return MP4_ERR_UNSUP;            // Advanced Prediction (OBMC)
    if (get_bits1(gb)) return MP4_ERR_UNSUP;            // PB-frames
    int pquant = (int)get_bits(gb, 5);
    if (get_bits1(gb)) return MP4_ERR_UNSUP;            // CPM (continuous presence multipoint)

    d->is_h263 = 1;
    d->configured = 1;
    d->no_rounding = 0;                                 // baseline H.263: standard half-pel rounding
    d->quarter_sample = 0;
    d->progressive_sequence = 1;
    d->interlaced_dct = 0;
    if (w != d->width || h != d->height) { d->width = w; d->height = h; }
    if (mp4_alloc(d) != MP4_OK) return MP4_ERR_DATA;
    mp4_set_qscale(d, pquant);
    d->gob_index = (h <= 400) ? 1 : (h <= 800) ? 2 : 4; // H263_GOB_HEIGHT

    while (get_bits1(gb)) skip_bits(gb, 8);             // PEI (extra insertion info)
    return MP4_DECODE;
}

static inline int h263_decode_picture(Mp4Decoder* d) {
    int r = h263_decode_picture_header(d);
    if (r != MP4_DECODE) return r;

    // I/P reference swap (H.263 has no B in this scope): the previous frame becomes the
    // forward ref, the current planes are overwritten and become the next ref.
    uint8_t* t;
    t = d->y; d->y = d->refY; d->refY = t;
    t = d->u; d->u = d->refU; d->refU = t;
    t = d->v; d->v = d->refV; d->refV = t;
    d->last_output_is_b = 0;

    // Reset prediction grids (mot_val for MV prediction; dc_val unused by H.263 non-AIC
    // intra but reset for parity / shared edge helpers).
    for (int i = 0; i < d->grid_elems; i++) d->dc_base[i] = 1024;
    memset(d->mot_base, 0, (size_t)d->grid_elems * 2 * sizeof(int16_t));

    int mb_num = d->mb_width * d->mb_height;
    d->mb_x = 0; d->mb_y = 0;
    d->resync_mb_x = 0; d->resync_mb_y = 0;
    d->first_slice_line = 1;

    int decoded = 0;
    for (;;) {                                          // slice (GOB-group) loop
        for (;;) {                                      // MB loop within the slice
            if (d->resync_mb_x == d->mb_x && d->resync_mb_y + 1 == d->mb_y)
                d->first_slice_line = 0;
            mp4_init_block_index(d);
            int mr = (d->pict_type == 1) ? h263_decode_mb_i(d) : h263_decode_mb_p(d);
            if (mr < 0) return MP4_ERR_DATA;
            mp4_update_motion_val(d);                   // ff_h263_update_motion_val (non-B)
            mp4_reconstruct_mb_h263(d);
            if (++decoded >= mb_num) return MP4_OK;

            // per-MB end-of-slice check (ff_h263_decode_mb `end:`): a fully-consumed
            // buffer is a data error (no-op on conformant streams — the last MB exits via
            // decoded>=mb_num above); else show_bits(16)==0 -> GOB start code or frame end.
            if (get_bits_left(&d->gb) < 0) return MP4_ERR_DATA;
            int left = get_bits_left(&d->gb);
            unsigned v = show_bits(&d->gb, 16);
            if (left < 16) v >>= (16 - left);
            int slice_end = (v == 0);

            if (++d->mb_x >= d->mb_width) { d->mb_x = 0; d->mb_y++; }
            if (slice_end) break;
        }
        if (d->mb_y >= d->mb_height) return MP4_OK;     // frame end (no more GOBs)
        if (h263_resync_gob(d) < 0) return MP4_ERR_DATA;
        d->resync_mb_x = d->mb_x;
        d->resync_mb_y = d->mb_y;
        d->first_slice_line = 1;
    }
}

// ── header parse ───────────────────────────────────────────────────────────────

// Returns MP4_OK on success, MP4_ERR_UNSUP if a gate is hit.
static inline int mp4_decode_vol_header(Mp4Decoder* d) {
    GetBits* gb = &d->gb;
    skip_bits(gb, 1);                                  // random_accessible_vol
    int vo_type = get_bits(gb, 8);
    if (vo_type == 14 || vo_type == 15) return MP4_ERR_UNSUP;   // studio
    int is_ol_id = get_bits1(gb);
    if (is_ol_id) { d->vo_ver_id = get_bits(gb, 4); skip_bits(gb, 3); }
    else            d->vo_ver_id = 1;
    int aspect = get_bits(gb, 4);
    if (aspect == 15) skip_bits(gb, 16);               // par_width/height

    int vol_control = get_bits1(gb);
    if (vol_control) {
        skip_bits(gb, 2);                              // chroma_format
        d->low_delay = get_bits1(gb);
        int vbv = get_bits1(gb);
        if (vbv) {                                     // ffmpeg decode_vol_header: 79 bits
            skip_bits(gb, 15); skip_bits(gb, 1);       // first_half_bit_rate, marker
            skip_bits(gb, 15); skip_bits(gb, 1);       // latter_half_bit_rate, marker
            skip_bits(gb, 15); skip_bits(gb, 1);       // first_half_vbv_buffer_size, marker
            skip_bits(gb, 3);                          // latter_half_vbv_buffer_size
            skip_bits(gb, 11); skip_bits(gb, 1);       // first_half_vbv_occupancy, marker
            skip_bits(gb, 15); skip_bits(gb, 1);       // latter_half_vbv_occupancy, marker
        }
    }

    d->shape = get_bits(gb, 2);
    if (d->shape != SHAPE_RECT) return MP4_ERR_UNSUP;
    skip_bits(gb, 1);                                  // marker

    int vti_res = get_bits(gb, 16);                    // vop_time_increment_resolution
    if (vti_res == 0) return MP4_ERR_DATA;
    d->time_increment_resolution = vti_res;            // == ffmpeg framerate.num (B timing)
    d->time_increment_bits = 1;
    { int t = vti_res - 1; while (t > 1) { t >>= 1; d->time_increment_bits++; } } // av_log2(res-1)+1
    if (d->time_increment_bits < 1) d->time_increment_bits = 1;
    skip_bits(gb, 1);                                  // marker

    int fixed_rate = get_bits1(gb);
    if (fixed_rate) skip_bits(gb, d->time_increment_bits);

    // shape == RECT: marker,width(13),marker,height(13),marker
    skip_bits(gb, 1);
    int w = get_bits(gb, 13);
    skip_bits(gb, 1);
    int h = get_bits(gb, 13);
    skip_bits(gb, 1);
    if (w) d->width = w;
    if (h) d->height = h;

    d->progressive_sequence = get_bits1(gb) ^ 1;       // interlaced bit, inverted
    if (!d->progressive_sequence) return MP4_ERR_UNSUP;
    d->interlaced_dct = 0;

    skip_bits(gb, 1);                                  // obmc_disable

    int sprite = (d->vo_ver_id == 1) ? get_bits1(gb) : get_bits(gb, 2);
    if (sprite != 0) return MP4_ERR_UNSUP;             // STATIC/GMC

    int not_8bit = get_bits1(gb);
    if (not_8bit) {
        d->quant_precision = get_bits(gb, 4);
        int bpp = get_bits(gb, 4);
        if (bpp != 8) return MP4_ERR_UNSUP;
        if (d->quant_precision < 3 || d->quant_precision > 9) d->quant_precision = 5;
    } else {
        d->quant_precision = 5;
    }

    d->mpeg_quant = get_bits1(gb);
    if (d->mpeg_quant) return MP4_ERR_UNSUP;           // M1: H.263 quant only

    d->quarter_sample = (d->vo_ver_id != 1) ? (int)get_bits1(gb) : 0;     // qpel (M3)

    int cplx_dis = get_bits1(gb);
    if (!cplx_dis) return MP4_ERR_UNSUP;               // complexity estimation (would need trash-bit accounting)

    // resync_marker_disable: markers being *enabled* here doesn't mean any are
    // inserted in the bitstream. M1 decodes as a single slice (resync_mb_x/y = 0);
    // if a stream actually inserts a resync marker mid-VOP we'd desync (revisit M2).
    skip_bits(gb, 1);

    int data_part = get_bits1(gb);
    if (data_part) { get_bits1(gb); return MP4_ERR_UNSUP; }  // data_partitioning (+rvlc)

    if (d->vo_ver_id != 1) {
        if (get_bits1(gb)) return MP4_ERR_UNSUP;       // new_pred
        if (get_bits1(gb)) return MP4_ERR_UNSUP;       // reduced_resolution_vop
    }
    if (get_bits1(gb)) return MP4_ERR_UNSUP;           // scalability

    d->configured = 1;
    return mp4_alloc(d) == MP4_OK ? MP4_OK : MP4_ERR_DATA;
}

// Returns MP4_DECODE (header parsed, decode the picture), MP4_SKIP (NVOP — keep the
// previous picture), or a negative gate/error code.
static inline int mp4_decode_vop_header(Mp4Decoder* d) {
    GetBits* gb = &d->gb;
    if (!d->configured) return MP4_ERR_DATA;

    d->pict_type = get_bits(gb, 2) + 1;                // 1=I,2=P,3=B,4=S
    if (d->pict_type == 4) return MP4_ERR_UNSUP;       // S (GMC/sprite) gated

    int time_incr = 0;
    while (get_bits1(gb) != 0) time_incr++;            // modulo_time_base (unary)
    skip_bits(gb, 1);                                  // marker
    int time_increment = (int)get_bits(gb, d->time_increment_bits);  // vop_time_increment
    skip_bits(gb, 1);                                  // marker

    // VOP timing (ffmpeg decode_vop_header). framerate.num == time_increment_resolution.
    if (d->pict_type != 3) {                           // I / P (non-B)
        d->last_time_base = d->time_base;
        d->time_base     += time_incr;
        int time          = d->time_base * d->time_increment_resolution + time_increment;
        d->pp_time        = time - d->last_non_b_time;
        d->last_non_b_time = time;
    } else {                                           // B
        int time   = (d->last_time_base + time_incr) * d->time_increment_resolution + time_increment;
        d->pb_time = d->pp_time - (d->last_non_b_time - time);
        // ffmpeg returns FRAME_SKIPPED here on any of pp<=pb || pp<=pp-pb || pp<=0 (its
        // flush-on-seek model treats non-monotonic B timing as undecodable post-seek
        // garbage). Our readers instead decode B's from a keyframe walk, so the
        // references are valid even when a random-access re-decode scrambles the
        // *absolute* time_base. We do NOT skip — the frame is real and wanted. The only
        // timing-dependent step is direct-mode MV scaling: mp4_set_one_direct_mv guards
        // the pp<=0 divide; the pb/pp ratio is otherwise unbounded, so a hand-crafted
        // stream with degenerate (but ffmpeg-would-skip) B timing could yield a wrong
        // direct MV — never a trap, and never on a conformant monotonic-timestamp stream.
    }

    int vop_coded = get_bits1(gb);
    if (!vop_coded) return MP4_SKIP;                   // NVOP — keep previous picture

    // rounding: read for P (not I/B); shape RECT, not S/GMC.
    d->no_rounding = (d->pict_type == 2) ? (int)get_bits1(gb) : 0;
    // complexity-estimation trash skipped (cplx disabled, trash=0).
    d->intra_dc_threshold = ff_mpeg4_dc_threshold[get_bits(gb, 3)];
    if (!d->progressive_sequence) { skip_bits(gb, 1); skip_bits(gb, 1); }  // not reached (gated)

    int vop_quant = get_bits(gb, d->quant_precision);
    if (vop_quant == 0) return MP4_ERR_DATA;
    mp4_set_qscale(d, vop_quant);

    if (d->pict_type != 1) {                           // P / B: forward fcode
        d->f_code = (int)get_bits(gb, 3);
        if (d->f_code == 0) return MP4_ERR_DATA;
    }
    if (d->pict_type == 3) {                           // B: backward fcode
        d->b_code = (int)get_bits(gb, 3);
        if (d->b_code == 0) return MP4_ERR_DATA;
    }
    return MP4_DECODE;
}

// Top-level: scan start codes, parse headers, decode one I-VOP. Returns MP4_OK
// (decoded), MP4_SKIP (NVOP), or a negative gate/error code.
static inline int mp4_decode_one(Mp4Decoder* d, const uint8_t* buf, int len) {
    mp4_init_tables();
    init_get_bits(&d->gb, buf, len);
    GetBits* gb = &d->gb;
    align_get_bits(gb);

    // H.263 short header vs MPEG-4 Part 2: the H.263 picture starts with a 22-bit PSC
    // (value 0x20; byte-aligned 00 00 8x), whereas every MPEG-4 start code is 00 00 01 xx
    // (low 22 bits == 0). ffmpeg routes by codec_id, but this one module serves both
    // fourcc families, so sniff the leading 22 bits. (Once configured for H.263 the flag
    // is sticky — a stream never mixes the two — but per-frame detection is robust too.)
    if (show_bits(gb, 22) == 0x20 || d->is_h263)
        return h263_decode_picture(d);

    uint32_t startcode = 0xFF;

    for (;;) {
        if (get_bits_count(gb) >= gb->size_in_bits + 8) return MP4_ERR_NOVOP;
        startcode = ((startcode << 8) | get_bits(gb, 8)) & 0xFFFFFFFFu;
        if ((startcode & 0xFFFFFF00u) != 0x100u) continue;
        int low = startcode & 0xFF;

        if (low >= SC_VOL_LOW && low <= 0x2F) {
            int r = mp4_decode_vol_header(d);
            if (r < 0) return r;
            align_get_bits(gb); startcode = 0xFF; continue;
        } else if (low == SC_VOS) {
            int profile = get_bits(gb, 4), level = get_bits(gb, 4);
            if (profile == 0 && level == 8) level = 0;
            // SIMPLE_STUDIO profile gate handled at VOL (vo_type 14/15).
            align_get_bits(gb); startcode = 0xFF; continue;
        } else if (low == SC_USERDATA) {
            // consume up to 255 bytes until a start-code prefix appears
            for (int i = 0; i < 255 && get_bits_count(gb) < gb->size_in_bits; i++) {
                if (show_bits(gb, 23) == 0) break;
                get_bits(gb, 8);
            }
            align_get_bits(gb); startcode = 0xFF; continue;
        } else if (low == SC_GOP) {
            int hours = (int)get_bits(gb, 5), minutes = (int)get_bits(gb, 6);
            skip_bits(gb, 1);                                       // marker
            int seconds = (int)get_bits(gb, 6);
            skip_bits(gb, 1); skip_bits(gb, 1);                     // closed_gov, broken_link
            // GOP timecode sets time_base absolutely (ffmpeg mpeg4_decode_gop_header):
            // the following I/P VOP timing (and thus B direct-mode pp/pb) is relative
            // to this base, so it must be reset here for multi-GOP streams.
            d->time_base = seconds + 60 * minutes + 3600 * hours;
            align_get_bits(gb); startcode = 0xFF; continue;
        } else if (low == SC_VISOBJ) {
            int is_id = get_bits1(gb);
            if (is_id) skip_bits(gb, 7);                 // verid(4)+priority(3)
            int vtype = get_bits(gb, 4);
            if (vtype == 1 || vtype == 2) {
                if (get_bits1(gb)) {                     // video_signal_type
                    skip_bits(gb, 3); skip_bits(gb, 1);  // format, range
                    if (get_bits1(gb)) skip_bits(gb, 24);// color_description
                }
            }
            align_get_bits(gb); startcode = 0xFF; continue;
        } else if (low == SC_VOP) {
            break;
        } else {
            align_get_bits(gb); startcode = 0xFF; continue;
        }
    }

    int r = mp4_decode_vop_header(d);
    if (r != MP4_DECODE) return r;         // MP4_SKIP (NVOP), gate, or error — propagate

    if (d->pict_type == 3) {
        // B-VOP: forward ref = refY/U/V (last I/P), backward ref = y/u/v (next I/P).
        // No swap — B is never a reference; output goes to bY/U/V.
        d->last_output_is_b = 1;
    } else {
        // I/P: swap so the previous I/P becomes the forward ref (refY/U/V) and the
        // current planes (y/u/v) are overwritten — they become the next backward ref.
        uint8_t* t;
        t = d->y; d->y = d->refY; d->refY = t;
        t = d->u; d->u = d->refU; d->refU = t;
        t = d->v; d->v = d->refV; d->refV = t;
        d->last_output_is_b = 0;
    }

    return mp4_decode_picture(d);
}

// Current output planes: B -> bY/U/V; I/P -> y/u/v (the just-decoded frame).
static inline uint8_t* mp4_out_y(Mp4Decoder* d) { return d->last_output_is_b ? d->bY : d->y; }
static inline uint8_t* mp4_out_u(Mp4Decoder* d) { return d->last_output_is_b ? d->bU : d->u; }
static inline uint8_t* mp4_out_v(Mp4Decoder* d) { return d->last_output_is_b ? d->bV : d->v; }

static inline void mp4_free(Mp4Decoder* d) {
    free(d->y); free(d->u); free(d->v);
    free(d->refY); free(d->refU); free(d->refV);
    free(d->bY); free(d->bU); free(d->bV);
    free(d->dc_base); free(d->ac_base); free(d->mot_base);
    free(d->bwd_mot); free(d->bwd_skip); free(d->bwd_is8x8);
}

} // namespace mp4

#endif // MPEG4_MP4_DECODE_H
