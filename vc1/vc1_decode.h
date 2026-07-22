/*
    SMPTE 421M (VC-1) / Windows Media Video 9 (WMV3, WVC1) video decoder —
    C++/WebAssembly. From-source TRANSCRIPTION of FFmpeg release/8.1's VC-1 decoder
    (libavcodec/vc1.c, vc1dec.c, vc1_block.c, vc1_pred.c, vc1_mc.c,
    vc1_loopfilter.c, vc1dsp.c, vc1data.c). FFmpeg is LGPL-2.1+; this is an
    LGPL-derived work in its OWN wasm binary. Conformance oracle is the ffmpeg CLI
    (`ffmpeg -cpuflags 0 -i sample -f rawvideo -pix_fmt yuv420p`), NOT ffmpeg-JS.

    VC-1 is a large codec (its own 8x8/8x4/4x8/4x4 integer transforms — NOT the
    simple_idct8 the mpeg4/msmpeg4 ports use — plus bitplane coding, AC/DC intra
    prediction, overlap smoothing, an in-loop deblocking filter, and quarter-pel
    motion compensation). It reuses only the leaf get_bits reader from the mpeg4
    port; everything else is VC-1-specific.

    STATUS: milestone-staged multi-part port.
      1. Simple/Main SEQUENCE HEADER (STRUCT_C extradata)  — DONE, field-exact.
      2. Simple/Main PICTURE (frame) HEADER, I + P paths, plus BITPLANE decoding
         (imode/norm2/norm6 VLC + rowskip/colskip + the DIFF operator)  — THIS FILE.
         Validated by the per-P-frame (Imode, Invert) pair that ffmpeg's decoder
         prints for the MV-type and skip bitplanes (tests/Vc1Wasm.test.js).
    NEXT: I-frame block decode (transforms + AC/DC prediction + dequant + in-loop
    deblock), then P-frame inter (quarter-pel MC, intensity comp).

    -fwrapv is REQUIRED (transform / dequant sign arithmetic), as in the sibling ports.
*/

#ifndef VC1_DECODE_H
#define VC1_DECODE_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "../mpeg4/mp4_bits.h"   // reused get_bits reader (namespace mp4)
#include "../mpeg4/mp4_vlc.h"    // reused VLCElem / build_flat_vlc / get_vlc2
#include "../msmpeg4/msmpeg4_tables.h"  // reused: msm_dc_vlc, ff_msmp4_dc_tables, ff_msmp4_mb_i_table, ff_wmv1_scantable
#include "vc1_vlc.h"             // multi-level VLC for the AC coefficient tables
#include "vc1_tables.h"          // VC-1 data tables (ac_tables, dc_scale, dqscale, ...)
#include "vc1_mc.h"              // P-frame MC/residual DSP leaf kernels

namespace vc1 {

// Profiles (vc1_common.h): 0 Simple, 1 Main, 2 Complex, 3 Advanced.
enum { PROFILE_SIMPLE = 0, PROFILE_MAIN = 1, PROFILE_COMPLEX = 2, PROFILE_ADVANCED = 3 };
enum { VC1_OK = 0, VC1_ERR = -1 };

// AV_PICTURE_TYPE_* subset used by the header logic.
enum { PICT_I = 1, PICT_P = 2, PICT_B = 3, PICT_BI = 5 };

// enum Imode (vc1.h) — reported to the conformance test as `status >> 1`.
enum { IMODE_RAW = 0, IMODE_NORM2, IMODE_DIFF2, IMODE_NORM6, IMODE_DIFF6,
       IMODE_ROWSKIP, IMODE_COLSKIP };

// enum QuantizerMode (vc1.h).
enum { QUANT_FRAME_IMPLICIT = 0, QUANT_FRAME_EXPLICIT, QUANT_NON_UNIFORM, QUANT_UNIFORM };

// enum MVModes (vc1.h).
enum { MV_PMODE_1MV_HPEL_BILIN = 0, MV_PMODE_1MV, MV_PMODE_1MV_HPEL,
       MV_PMODE_MIXED_MV, MV_PMODE_INTENSITY_COMP };

// enum TransformTypes: defined in vc1_tables.h (TT_8X8=0 .. TT_4X4).

struct VC1Context {
    int width, height;
    int mb_width, mb_height, mb_stride;

    // Sequence header (Simple/Main profile, from STRUCT_C extradata).
    int profile;
    int res_y411, res_sprite;
    int frmrtq_postproc, bitrtq_postproc;
    int loop_filter;
    int res_x8, multires, res_fasttx;
    int fastuvmc, extended_mv;
    int dquant, vstransform, res_transtab, overlap;
    int resync_marker, rangered;
    int max_b_frames, quantizer_mode, finterpflag;
    int res_rtm_flag;

    // Picture (frame) header — Simple/Main. Reset/filled per frame.
    int pict_type;
    int rnd;
    int interpfrm, rangeredfrm;
    int pqindex, pq, halfpq, pquantizer;
    int mvrange, k_x, k_y, range_x, range_y;
    int x8_type, tt_index;
    int mv_mode, mv_mode2, lumscale, lumshift;
    int mv_table_index, cbptab;
    int ttmbf, ttfrm;
    int dquantfrm, dqprofile, dqsbedge, dqbilevel;
    int c_ac_table_index, y_ac_table_index, dc_table_index;

    // Bitplane state / results (last parsed frame).
    int mv_type_is_raw, skip_is_raw;
    int has_mvtype_plane;
    int mvtype_imode, mvtype_invert, skip_imode, skip_invert;
    uint8_t* mv_type_mb_plane;   // mb_stride * mb_height
    uint8_t* mbskip_table;       // mb_stride * mb_height

    // ── I-frame block decode (mpegvideo-shaped grid, mirrors the msmpeg4 port) ──
    int b8_stride, dc_off, grid_total, grid_lead;
    int block_wrap[6], block_index[6];
    int mb_x, mb_y, first_slice_line, ac_pred;
    int codingset, codingset2, y_dc_scale;
    int esc3_level_length, esc3_run_length;
    int left_blk_sh, top_blk_sh;
    uint8_t zz_8x8[4][64];          // transposed scan tables (ff_wmv1_scantable)

    int16_t *dc_base, *dc_val;      // unscaled DC predictor grid
    int16_t *ac_base, *ac_val;      // 16 int16 per block (AC predictor grid)
    uint8_t *coded_base, *coded_block;

    // Output planes (MB-aligned; cropped to width×height on read).
    int codedW, codedH, yStride, cStride;
    uint8_t *planeY, *planeU, *planeV;
    size_t yCap, cCap;

    // ── P-frame inter decode ────────────────────────────────────────────────────
    uint8_t *refY, *refU, *refV;        // previous decoded frame (MC source)
    int16_t *mv_base, *mv_val;          // motion_val[0]: 2 int16 per b8 block
    uint8_t *mbtype_base, *mbtype;      // per-b8-block is_intra (get_chroma_mv)
    int16_t cur_mv[6][2];               // s->mv[0][n] for the current MB
    int quarter_sample, mspel;          // from mv_mode
    const mp4::VLCElem* cbpcy_vlc;      // = g_cbpcy_p_vlc[cbptab]
    int a_avail, c_avail;
    // per-MB frame arrays for the P deblock (full frame, not ffmpeg's rolling buffer)
    uint32_t *fr_cbp;                   // block_cbp per MB (mb_width*mb_height)
    int      *fr_ttblk;                 // block_tt  per MB
    uint8_t  *fr_isintra;              // block_intra bitmask per MB
    int16_t (*fr_lumamv)[2];            // chroma luma_mv per MB (deblock chroma)
};

// ── Decode tables built once (AC coefficient VLCs + the reused MB-intra VLC) ─────
static mp4::VLCElem g_ac_pool[49152];        // shared multi-level pool (AC + P-frame VLCs)
static const mp4::VLCElem* g_ac_vlc[8];
static const mp4::VLCElem* g_mv_diff_vlc[4]; // P: MV differential
static const mp4::VLCElem* g_cbpcy_p_vlc[4]; // P: coded-block pattern
static const mp4::VLCElem* g_ttmb_vlc[3];    // P: transform type / MB
static const mp4::VLCElem* g_ttblk_vlc[3];   // P: transform type / block
static const mp4::VLCElem* g_subblkpat_vlc[3];
static mp4::VLCElem g_mb_i_store[1 << 13];
static mp4::VlcTab g_mb_i;
static int g_itables_ready = 0;

enum { AC_VLC_BITS = 9 };

static inline void vc1_init_iframe_tables() {
    if (g_itables_ready) return;
    MLBuild b = { g_ac_pool, 0, (int)(sizeof(g_ac_pool) / sizeof(g_ac_pool[0])), 1 };  // {pool,cursor,cap,ok}
    for (int i = 0; i < 8; i++)
        g_ac_vlc[i] = ml_init(&b, AC_VLC_BITS, ff_vc1_ac_sizes[i],
                              &vc1_ac_tables[i][0][1], 8, 4,
                              &vc1_ac_tables[i][0][0], 8, 4);
    // P-frame VLCs (multi-level; codes/bits arrays from vc1_tables.h).
    for (int i = 0; i < 4; i++) {
        g_mv_diff_vlc[i] = ml_init(&b, 9, 73, &vc1_mv_diff_bits[i][0], 1, 1, &vc1_mv_diff_codes[i][0], 2, 2);
        g_cbpcy_p_vlc[i] = ml_init(&b, 9, 64, &vc1_cbpcy_p_bits[i][0], 1, 1, &vc1_cbpcy_p_codes[i][0], 2, 2);
    }
    for (int i = 0; i < 3; i++) {
        g_ttmb_vlc[i]     = ml_init(&b, 9, 16, &vc1_ttmb_bits[i][0],     1, 1, &vc1_ttmb_codes[i][0],     2, 2);
        g_ttblk_vlc[i]    = ml_init(&b, 5,  8, &vc1_ttblk_bits[i][0],    1, 1, &vc1_ttblk_codes[i][0],    1, 1);
        g_subblkpat_vlc[i]= ml_init(&b, 6, 15, &vc1_subblkpat_bits[i][0],1, 1, &vc1_subblkpat_codes[i][0],1, 1);
    }
    int mbbits = mp4::mp4_max_len(&msm::ff_msmp4_mb_i_table[0][1], 64, 4, 2);
    mp4::build_flat_vlc(g_mb_i_store, mbbits, 64,
                        &msm::ff_msmp4_mb_i_table[0][0], 4, 2,
                        &msm::ff_msmp4_mb_i_table[0][1], 4, 2, NULL);
    g_mb_i.table = g_mb_i_store; g_mb_i.bits = mbbits;
    g_itables_ready = 1;
}

// decode210 (get_bits.h): '1'->0; '00'->2; '01'->1.
static inline int vc1_decode210(mp4::GetBits* gb) {
    if (mp4::get_bits1(gb)) return 0;
    return 2 - (int)mp4::get_bits1(gb);
}

// ── Tables (vc1data.c) ─────────────────────────────────────────────────────────
static const uint8_t vc1_pquant_table[3][32] = {
    {  0,  1,  2,  3,  4,  5,  6,  7,  8,  6,  7,  8,  9, 10, 11, 12,
      13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 29, 31 },
    {  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
      16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31 },
    {  0,  1,  1,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13,
      14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 29, 31 }
};
static const uint8_t vc1_mv_pmode_table[2][5] = {
    { MV_PMODE_1MV_HPEL_BILIN, MV_PMODE_1MV, MV_PMODE_1MV_HPEL, MV_PMODE_INTENSITY_COMP, MV_PMODE_MIXED_MV },
    { MV_PMODE_1MV, MV_PMODE_MIXED_MV, MV_PMODE_1MV_HPEL, MV_PMODE_INTENSITY_COMP, MV_PMODE_1MV_HPEL_BILIN }
};
static const uint8_t vc1_mv_pmode_table2[2][4] = {
    { MV_PMODE_1MV_HPEL_BILIN, MV_PMODE_1MV, MV_PMODE_1MV_HPEL, MV_PMODE_MIXED_MV },
    { MV_PMODE_1MV, MV_PMODE_MIXED_MV, MV_PMODE_1MV_HPEL, MV_PMODE_1MV_HPEL_BILIN }
};
static const int vc1_ttfrm_to_tt[4] = { TT_8X8, TT_8X4, TT_4X8, TT_4X4 };

// Bitplane VLC source arrays (vc1_vlc_data.h). Max code lengths: imode 4, norm2 3,
// norm6 13 — small enough for the mpeg4 port's flat single-level get_vlc2.
static const uint8_t  vc1_imode_codes[7] = { 0, 2, 1, 3, 1, 2, 3 };
static const uint8_t  vc1_imode_bits [7] = { 4, 2, 3, 2, 4, 3, 3 };
static const uint8_t  vc1_norm2_codes[4] = { 0, 4, 5, 3 };
static const uint8_t  vc1_norm2_bits [4] = { 1, 3, 3, 2 };
static const uint16_t vc1_norm6_codes[64] = {
    0x001, 0x002, 0x003, 0x000, 0x004, 0x001, 0x002, 0x047, 0x005, 0x003, 0x004, 0x04B, 0x005, 0x04D, 0x04E, 0x30E,
    0x006, 0x006, 0x007, 0x053, 0x008, 0x055, 0x056, 0x30D, 0x009, 0x059, 0x05A, 0x30C, 0x05C, 0x30B, 0x30A, 0x037,
    0x007, 0x00A, 0x00B, 0x043, 0x00C, 0x045, 0x046, 0x309, 0x00D, 0x049, 0x04A, 0x308, 0x04C, 0x307, 0x306, 0x036,
    0x00E, 0x051, 0x052, 0x305, 0x054, 0x304, 0x303, 0x035, 0x058, 0x302, 0x301, 0x034, 0x300, 0x033, 0x032, 0x007,
};
static const uint8_t vc1_norm6_bits[64] = {
    1,  4,  4,  8,  4,  8,  8, 10,  4,  8,  8, 10,  8, 10, 10, 13,
    4,  8,  8, 10,  8, 10, 10, 13,  8, 10, 10, 13, 10, 13, 13,  9,
    4,  8,  8, 10,  8, 10, 10, 13,  8, 10, 10, 13, 10, 13, 13,  9,
    8, 10, 10, 13, 10, 13, 13,  9, 10, 13, 13,  9, 13,  9,  9,  6,
};

enum { VC1_IMODE_VLC_BITS = 4, VC1_NORM2_VLC_BITS = 3, VC1_NORM6_VLC_BITS = 13 };

static mp4::VLCElem g_imode_vlc[1 << VC1_IMODE_VLC_BITS];
static mp4::VLCElem g_norm2_vlc[1 << VC1_NORM2_VLC_BITS];
static mp4::VLCElem g_norm6_vlc[1 << VC1_NORM6_VLC_BITS];
static int g_vlc_ready = 0;

static inline void vc1_init_vlcs() {
    if (g_vlc_ready) return;
    mp4::build_flat_vlc(g_imode_vlc, VC1_IMODE_VLC_BITS, 7,
                        vc1_imode_codes, 1, 1, vc1_imode_bits, 1, 1, NULL);
    mp4::build_flat_vlc(g_norm2_vlc, VC1_NORM2_VLC_BITS, 4,
                        vc1_norm2_codes, 1, 1, vc1_norm2_bits, 1, 1, NULL);
    mp4::build_flat_vlc(g_norm6_vlc, VC1_NORM6_VLC_BITS, 64,
                        vc1_norm6_codes, 2, 2, vc1_norm6_bits, 1, 1, NULL);
    g_vlc_ready = 1;
}

// ── Small bitstream helpers (get_bits.h / mathops.h) ────────────────────────────
// get_unary: read up to `len` bits, counting until the first `stop` bit.
static inline int vc1_get_unary(mp4::GetBits* gb, int stop, int len) {
    int i;
    for (i = 0; i < len && (int)mp4::get_bits1(gb) != stop; i++) {}
    return i;
}
// decode012: 0 => 0; 10 => 1; 11 => 2 (get_bits.h).
static inline int vc1_decode012(mp4::GetBits* gb) {
    if (!mp4::get_bits1(gb)) return 0;
    return (int)mp4::get_bits1(gb) + 1;
}

// ff_vc1_decode_sequence_header — Simple/Main profile (WMV3). The Advanced-profile
// (WVC1) path (get_bits(2)==PROFILE_ADVANCED) is a separate header and lands later.
static inline int vc1_decode_sequence_header(VC1Context* v, mp4::GetBits* gb) {
    v->profile = (int)mp4::get_bits(gb, 2);
    if (v->profile == PROFILE_ADVANCED) return VC1_ERR;   // WVC1 — not yet supported

    v->res_y411   = (int)mp4::get_bits1(gb);
    v->res_sprite = (int)mp4::get_bits1(gb);
    if (v->res_y411) return VC1_ERR;                       // old interlaced mode

    v->frmrtq_postproc = (int)mp4::get_bits(gb, 3);
    v->bitrtq_postproc = (int)mp4::get_bits(gb, 5);
    v->loop_filter     = (int)mp4::get_bits1(gb);
    v->res_x8          = (int)mp4::get_bits1(gb);
    v->multires        = (int)mp4::get_bits1(gb);
    v->res_fasttx      = (int)mp4::get_bits1(gb);
    v->fastuvmc        = (int)mp4::get_bits1(gb);
    if (!v->profile && !v->fastuvmc) return VC1_ERR;       // FASTUVMC required in Simple
    v->extended_mv     = (int)mp4::get_bits1(gb);
    if (!v->profile && v->extended_mv) return VC1_ERR;     // no extended MV in Simple
    v->dquant          = (int)mp4::get_bits(gb, 2);
    v->vstransform     = (int)mp4::get_bits1(gb);
    v->res_transtab    = (int)mp4::get_bits1(gb);
    if (v->res_transtab) return VC1_ERR;
    v->overlap         = (int)mp4::get_bits1(gb);
    v->resync_marker   = (int)mp4::get_bits1(gb);
    v->rangered        = (int)mp4::get_bits1(gb);
    v->max_b_frames    = (int)mp4::get_bits(gb, 3);
    v->quantizer_mode  = (int)mp4::get_bits(gb, 2);
    v->finterpflag     = (int)mp4::get_bits1(gb);

    if (v->res_sprite) return VC1_ERR;                     // sprite mode — not supported
    v->res_rtm_flag = (int)mp4::get_bits1(gb);
    if (!v->res_fasttx) mp4::skip_bits(gb, 16);            // reserved (always 0x402F)
    return VC1_OK;
}

// ── Bitplane decoding (vc1.c decode_rowskip / decode_colskip / bitplane_decoding) ─
static inline void vc1_decode_rowskip(uint8_t* plane, int width, int height, int stride,
                                      mp4::GetBits* gb) {
    for (int y = 0; y < height; y++) {
        if (!mp4::get_bits1(gb))
            memset(plane, 0, width);
        else
            for (int x = 0; x < width; x++)
                plane[x] = (uint8_t)mp4::get_bits1(gb);
        plane += stride;
    }
}

static inline void vc1_decode_colskip(uint8_t* plane, int width, int height, int stride,
                                      mp4::GetBits* gb) {
    for (int x = 0; x < width; x++) {
        if (!mp4::get_bits1(gb))
            for (int y = 0; y < height; y++)
                plane[y * stride] = 0;
        else
            for (int y = 0; y < height; y++)
                plane[y * stride] = (uint8_t)mp4::get_bits1(gb);
        plane++;
    }
}

// Returns (imode << 1) | invert, or `invert` for a raw plane (both match the value
// ffmpeg logs as "Imode: status>>1, Invert: status&1"). -1 on an invalid NORM-6 VLC.
static inline int vc1_bitplane_decoding(uint8_t* data, int* raw_flag, VC1Context* v,
                                        mp4::GetBits* gb) {
    int imode, x, y, code, offset;
    uint8_t invert, *planep = data;
    int width  = v->mb_width;
    int height = v->mb_height;   // field_mode == 0 (progressive) — no >> shift
    int stride = v->mb_stride;

    invert = (uint8_t)mp4::get_bits1(gb);
    imode  = mp4::get_vlc2(gb, g_imode_vlc, VC1_IMODE_VLC_BITS);

    *raw_flag = 0;
    switch (imode) {
    case IMODE_RAW:
        *raw_flag = 1;               // data read later in the MB layer; invert ignored
        return invert;
    case IMODE_DIFF2:
    case IMODE_NORM2:
        if ((height * width) & 1) {
            *planep++ = (uint8_t)mp4::get_bits1(gb);
            y = offset = 1;
            if (offset == width) { offset = 0; planep += stride - width; }
        } else
            y = offset = 0;
        for (; y < height * width; y += 2) {
            code = mp4::get_vlc2(gb, g_norm2_vlc, VC1_NORM2_VLC_BITS);
            *planep++ = code & 1;
            offset++;
            if (offset == width) { offset = 0; planep += stride - width; }
            *planep++ = (code >> 1) & 1;
            offset++;
            if (offset == width) { offset = 0; planep += stride - width; }
        }
        break;
    case IMODE_DIFF6:
    case IMODE_NORM6:
        if (!(height % 3) && (width % 3)) {     // 2x3 decoding
            for (y = 0; y < height; y += 3) {
                for (x = width & 1; x < width; x += 2) {
                    code = mp4::get_vlc2(gb, g_norm6_vlc, VC1_NORM6_VLC_BITS);
                    if (code < 0) return -1;
                    planep[x + 0]              = (code >> 0) & 1;
                    planep[x + 1]              = (code >> 1) & 1;
                    planep[x + 0 + stride]     = (code >> 2) & 1;
                    planep[x + 1 + stride]     = (code >> 3) & 1;
                    planep[x + 0 + stride * 2] = (code >> 4) & 1;
                    planep[x + 1 + stride * 2] = (code >> 5) & 1;
                }
                planep += stride * 3;
            }
            if (width & 1)
                vc1_decode_colskip(data, 1, height, stride, gb);
        } else {                                 // 3x2 decoding
            planep += (height & 1) * stride;
            for (y = height & 1; y < height; y += 2) {
                for (x = width % 3; x < width; x += 3) {
                    code = mp4::get_vlc2(gb, g_norm6_vlc, VC1_NORM6_VLC_BITS);
                    if (code < 0) return -1;
                    planep[x + 0]          = (code >> 0) & 1;
                    planep[x + 1]          = (code >> 1) & 1;
                    planep[x + 2]          = (code >> 2) & 1;
                    planep[x + 0 + stride] = (code >> 3) & 1;
                    planep[x + 1 + stride] = (code >> 4) & 1;
                    planep[x + 2 + stride] = (code >> 5) & 1;
                }
                planep += stride * 2;
            }
            x = width % 3;
            if (x)          vc1_decode_colskip(data,        x, height, stride, gb);
            if (height & 1) vc1_decode_rowskip(data + x, width - x, 1, stride, gb);
        }
        break;
    case IMODE_ROWSKIP:
        vc1_decode_rowskip(data, width, height, stride, gb);
        break;
    case IMODE_COLSKIP:
        vc1_decode_colskip(data, width, height, stride, gb);
        break;
    default:
        break;
    }

    // Apply the DIFF operator.
    if (imode == IMODE_DIFF2 || imode == IMODE_DIFF6) {
        planep = data;
        planep[0] ^= invert;
        for (x = 1; x < width; x++)
            planep[x] ^= planep[x - 1];
        for (y = 1; y < height; y++) {
            planep += stride;
            planep[0] ^= planep[-stride];
            for (x = 1; x < width; x++) {
                if (planep[x - 1] != planep[x - stride]) planep[x] ^= invert;
                else                                     planep[x] ^= planep[x - 1];
            }
        }
    } else if (invert) {
        planep = data;
        for (x = 0; x < stride * height; x++)
            planep[x] = !planep[x];
    }
    return (imode << 1) + invert;
}

// ── Picture (frame) header — ff_vc1_parse_frame_header, Simple/Main (I + P) ──────
// B frames are not selected here (max_b_frames == 0 in the WMV3 samples we target)
// and the interlace/advanced paths are elsewhere; a B pict_type returns VC1_ERR.
static inline int vc1_parse_frame_header(VC1Context* v, mp4::GetBits* gb) {
    int pqindex;
    vc1_init_vlcs();

    if (v->finterpflag)
        v->interpfrm = (int)mp4::get_bits1(gb);
    mp4::skip_bits(gb, 2);                       // framecnt (unused, non-MSS2)
    v->rangeredfrm = 0;
    if (v->rangered)
        v->rangeredfrm = (int)mp4::get_bits1(gb);

    if (mp4::get_bits1(gb)) {
        v->pict_type = PICT_P;
    } else {
        if (v->max_b_frames && !mp4::get_bits1(gb)) v->pict_type = PICT_B;
        else                                        v->pict_type = PICT_I;
    }
    if (v->pict_type == PICT_B) return VC1_ERR;  // B not yet supported

    if (v->pict_type == PICT_I)
        mp4::skip_bits(gb, 7);                   // buffer fullness

    // RND
    if (v->pict_type == PICT_I) v->rnd = 1;
    if (v->pict_type == PICT_P) v->rnd ^= 1;

    // Quantizer
    pqindex = (int)mp4::get_bits(gb, 5);
    if (!pqindex) return VC1_ERR;
    if (v->quantizer_mode == QUANT_FRAME_IMPLICIT)
        v->pq = vc1_pquant_table[0][pqindex];
    else
        v->pq = vc1_pquant_table[1][pqindex];
    v->pqindex = pqindex;
    v->halfpq  = (pqindex < 9) ? (int)mp4::get_bits1(gb) : 0;
    switch (v->quantizer_mode) {
    case QUANT_FRAME_IMPLICIT: v->pquantizer = pqindex < 9;              break;
    case QUANT_NON_UNIFORM:    v->pquantizer = 0;                        break;
    case QUANT_FRAME_EXPLICIT: v->pquantizer = (int)mp4::get_bits1(gb);  break;
    default:                   v->pquantizer = 1;                        break;
    }
    v->dquantfrm = 0;
    if (v->extended_mv == 1)
        v->mvrange = vc1_get_unary(gb, 0, 3);
    v->k_x     = v->mvrange + 9 + (v->mvrange >> 1);
    v->k_y     = v->mvrange + 8;
    v->range_x = 1 << (v->k_x - 1);
    v->range_y = 1 << (v->k_y - 1);
    // multires respic (multires == 0 for our stream), then x8_type.
    v->x8_type = (v->res_x8 && v->pict_type == PICT_I) ? (int)mp4::get_bits1(gb) : 0;

    v->has_mvtype_plane = 0;
    v->mvtype_imode = v->mvtype_invert = v->skip_imode = v->skip_invert = 0;

    if (v->pict_type == PICT_P) {
        int lowquant, status;
        v->tt_index = (v->pq > 4) + (v->pq > 12);

        lowquant = (v->pq > 12) ? 0 : 1;
        v->mv_mode = vc1_mv_pmode_table[lowquant][vc1_get_unary(gb, 1, 4)];
        if (v->mv_mode == MV_PMODE_INTENSITY_COMP) {
            v->mv_mode2 = vc1_mv_pmode_table2[lowquant][vc1_get_unary(gb, 1, 3)];
            v->lumscale = (int)mp4::get_bits(gb, 6);
            v->lumshift = (int)mp4::get_bits(gb, 6);   // INIT_LUT fills tables, no bits
        }
        {   // quarter_sample / mspel from the effective mode
            int m = (v->mv_mode == MV_PMODE_INTENSITY_COMP) ? v->mv_mode2 : v->mv_mode;
            v->quarter_sample = (m != MV_PMODE_1MV_HPEL && m != MV_PMODE_1MV_HPEL_BILIN);
            v->mspel          = (m != MV_PMODE_1MV_HPEL_BILIN);
        }

        if ((v->mv_mode  == MV_PMODE_INTENSITY_COMP && v->mv_mode2 == MV_PMODE_MIXED_MV) ||
            v->mv_mode   == MV_PMODE_MIXED_MV) {
            status = vc1_bitplane_decoding(v->mv_type_mb_plane, &v->mv_type_is_raw, v, gb);
            if (status < 0) return VC1_ERR;
            v->has_mvtype_plane = 1;
            v->mvtype_imode = status >> 1; v->mvtype_invert = status & 1;
        } else {
            v->mv_type_is_raw = 0;
            memset(v->mv_type_mb_plane, 0, v->mb_stride * v->mb_height);
        }
        status = vc1_bitplane_decoding(v->mbskip_table, &v->skip_is_raw, v, gb);
        if (status < 0) return VC1_ERR;
        v->skip_imode = status >> 1; v->skip_invert = status & 1;

        v->mv_table_index = (int)mp4::get_bits(gb, 2);
        v->cbptab         = (int)mp4::get_bits(gb, 2);

        // dquant == 0 in our stream — vop_dquant_decoding not invoked.

        if (v->vstransform) {
            v->ttmbf = (int)mp4::get_bits1(gb);
            v->ttfrm = v->ttmbf ? vc1_ttfrm_to_tt[mp4::get_bits(gb, 2)] : 0;
        } else {
            v->ttmbf = 1; v->ttfrm = TT_8X8;
        }
    }

    // AC / DC syntax (x8_type == 0).
    if (!v->x8_type) {
        v->c_ac_table_index = vc1_decode012(gb);
        if (v->pict_type == PICT_I || v->pict_type == PICT_BI)
            v->y_ac_table_index = vc1_decode012(gb);
        v->dc_table_index = (int)mp4::get_bits1(gb);
    }
    return VC1_OK;
}

// ── I-frame block decode (Simple/Main) ──────────────────────────────────────────
// Grid geometry mirrors the msmpeg4 port (ff_mpv_common_init): dc/ac/coded grids
// offset by lead+dc_off so edge OOB neighbour reads land in a zeroed guard.
static inline void vc1_init_block_index(VC1Context* v) {
    int x = v->mb_x, y = v->mb_y, b8 = v->b8_stride, ms = v->mb_stride, mh = v->mb_height;
    v->block_index[0] = b8 * (y * 2)     - 2 + x * 2;
    v->block_index[1] = b8 * (y * 2)     - 1 + x * 2;
    v->block_index[2] = b8 * (y * 2 + 1) - 2 + x * 2;
    v->block_index[3] = b8 * (y * 2 + 1) - 1 + x * 2;
    v->block_index[4] = ms * (y + 1)      + b8 * mh * 2 + x - 1;
    v->block_index[5] = ms * (y + mh + 2) + b8 * mh * 2 + x - 1;
}

// vc1_i_pred_dc (vc1_block.c) — VC-1 stores UNSCALED dcdiff in the grid; block[0]
// is scaled by y_dc_scale by the caller.
static inline int vc1_i_pred_dc(VC1Context* v, int overlap, int pq, int n,
                                int16_t** dc_val_ptr, int* dir_ptr) {
    static const uint16_t dcpred[32] = {
        (uint16_t)-1, 1024, 512, 341, 256, 205, 171, 146, 128,
                       114,  102,  93,  85,  79,  73,  68,  64,
                        60,   57,  54,  51,  49,  47,  45,  43,
                        41,   39,  38,  37,  35,  34,  33
    };
    int a, b, c, wrap, pred, scale = v->y_dc_scale;
    int16_t* dc_val;
    wrap   = v->block_wrap[n];
    dc_val = v->dc_val + v->block_index[n];
    c = dc_val[-1];
    b = dc_val[-1 - wrap];
    a = dc_val[-wrap];
    if (pq < 9 || !overlap) {
        if (v->first_slice_line && (n != 2 && n != 3)) b = a = dcpred[scale];
        if (v->mb_x == 0        && (n != 1 && n != 3)) b = c = dcpred[scale];
    } else {
        if (v->first_slice_line && (n != 2 && n != 3)) b = a = 0;
        if (v->mb_x == 0        && (n != 1 && n != 3)) b = c = 0;
    }
    if (abs(a - b) <= abs(b - c)) { pred = c; *dir_ptr = 1; }
    else                          { pred = a; *dir_ptr = 0; }
    *dc_val_ptr = dc_val;
    return pred;
}

static inline int vc1_coded_block_pred(VC1Context* v, int n, int diff) {
    int xy = v->block_index[n], wrap = v->b8_stride;
    int a = v->coded_block[xy - 1];
    int b = v->coded_block[xy - 1 - wrap];
    int c = v->coded_block[xy - wrap];
    int pred = (b == c) ? a : c;
    v->coded_block[xy] = (uint8_t)(pred ^ diff);
    return pred ^ diff;
}

// vc1_decode_ac_coeff (vc1_block.c) — RL VLC with the 3-way escape (8.1.3.4).
static inline int vc1_decode_ac_coeff(VC1Context* v, int* last, int* skip,
                                      int* value, int codingset, mp4::GetBits* gb) {
    int index, run, level, lst, sign;
    index = ml_get_vlc2(gb, g_ac_vlc[codingset], AC_VLC_BITS, 3);
    if (index < 0) return index;
    if (index != ff_vc1_ac_sizes[codingset] - 1) {
        run   = vc1_index_decode_table[codingset][index][0];
        level = vc1_index_decode_table[codingset][index][1];
        lst   = index >= vc1_last_decode_table[codingset] || mp4::get_bits_left(gb) < 0;
        sign  = (int)mp4::get_bits1(gb);
    } else {
        int escape = vc1_decode210(gb);
        if (escape != 2) {
            index = ml_get_vlc2(gb, g_ac_vlc[codingset], AC_VLC_BITS, 3);
            if ((unsigned)index >= (unsigned)(ff_vc1_ac_sizes[codingset] - 1)) return VC1_ERR;
            run   = vc1_index_decode_table[codingset][index][0];
            level = vc1_index_decode_table[codingset][index][1];
            lst   = index >= vc1_last_decode_table[codingset];
            if (escape == 0) {
                if (lst) level += vc1_last_delta_level_table[codingset][run];
                else     level += vc1_delta_level_table[codingset][run];
            } else {
                if (lst) run += vc1_last_delta_run_table[codingset][level] + 1;
                else     run += vc1_delta_run_table[codingset][level] + 1;
            }
            sign = (int)mp4::get_bits1(gb);
        } else {
            lst = (int)mp4::get_bits1(gb);
            if (v->esc3_level_length == 0) {
                if (v->pq < 8 || v->dquantfrm) {          // table 59
                    v->esc3_level_length = (int)mp4::get_bits(gb, 3);
                    if (!v->esc3_level_length)
                        v->esc3_level_length = (int)mp4::get_bits(gb, 2) + 8;
                } else {                                   // table 60
                    v->esc3_level_length = vc1_get_unary(gb, 1, 6) + 2;
                }
                v->esc3_run_length = 3 + (int)mp4::get_bits(gb, 2);
            }
            run   = (int)mp4::get_bits(gb, v->esc3_run_length);
            sign  = (int)mp4::get_bits1(gb);
            level = (int)mp4::get_bits(gb, v->esc3_level_length);
        }
    }
    *last  = lst;
    *skip  = run;
    *value = (level ^ -sign) + sign;
    return 0;
}

// vc1_decode_i_block (vc1_block.c) — one 8x8 intra block: DC diff + prediction, AC
// RL decode + AC prediction, dequant. Coefficients land in `block` (pre-transform).
static inline int vc1_decode_i_block(VC1Context* v, int16_t block[64], int n,
                                     int coded, int codingset, mp4::GetBits* gb) {
    int dc_pred_dir = 0;
    int16_t *dc_val, *ac_val, *ac_val2;
    int dcdiff, scale;

    dcdiff = msm::msm_dc_vlc(gb, msm::ff_msmp4_dc_tables[v->dc_table_index][n >= 4 ? 1 : 0]);
    if (dcdiff < 0) return VC1_ERR;
    if (dcdiff) {
        const int m = (v->pq == 1 || v->pq == 2) ? 3 - v->pq : 0;
        if (dcdiff == 119) dcdiff = (int)mp4::get_bits(gb, 8 + m);   // ESC
        else if (m)        dcdiff = (dcdiff << m) + (int)mp4::get_bits(gb, m) - ((1 << m) - 1);
        if (mp4::get_bits1(gb)) dcdiff = -dcdiff;
    }

    dcdiff += vc1_i_pred_dc(v, v->overlap, v->pq, n, &dc_val, &dc_pred_dir);
    *dc_val  = (int16_t)dcdiff;
    block[0] = (int16_t)(dcdiff * v->y_dc_scale);

    ac_val  = v->ac_val + (size_t)v->block_index[n] * 16;
    ac_val2 = ac_val;
    if (dc_pred_dir) ac_val -= 16;
    else             ac_val -= 16 * v->block_wrap[n];

    scale = v->pq * 2 + v->halfpq;

    if (coded) {
        int last = 0, skip, value, k;
        const uint8_t* zz_table;
        if (v->ac_pred) zz_table = (!dc_pred_dir) ? v->zz_8x8[2] : v->zz_8x8[3];
        else            zz_table = v->zz_8x8[1];

        for (int i = 1; !last; ++i) {
            int ret = vc1_decode_ac_coeff(v, &last, &skip, &value, codingset, gb);
            if (ret < 0) return ret;
            i += skip;
            if (i > 63) break;
            block[zz_table[i]] = (int16_t)value;
        }
        if (v->ac_pred) {
            int sh;
            if (dc_pred_dir) sh = v->left_blk_sh;
            else            { sh = v->top_blk_sh; ac_val += 8; }
            for (k = 1; k < 8; k++) block[k << sh] += ac_val[k];
        }
        for (k = 1; k < 8; k++) {
            ac_val2[k]     = block[k << v->left_blk_sh];
            ac_val2[k + 8] = block[k << v->top_blk_sh];
        }
        for (k = 1; k < 64; k++)
            if (block[k]) {
                block[k] *= scale;
                if (!v->pquantizer) block[k] += (block[k] < 0) ? -v->pq : v->pq;
            }
    } else {
        int k;
        memset(ac_val2, 0, 16 * 2);
        if (v->ac_pred) {
            int sh;
            if (dc_pred_dir) sh = v->left_blk_sh;
            else            { sh = v->top_blk_sh; ac_val += 8; ac_val2 += 8; }
            memcpy(ac_val2, ac_val, 8 * 2);
            for (k = 1; k < 8; k++) {
                block[k << sh] = ac_val[k] * scale;
                if (!v->pquantizer && block[k << sh])
                    block[k << sh] += (block[k << sh] < 0) ? -v->pq : v->pq;
            }
        }
    }
    return 0;
}

// vc1_inv_trans_8x8_c (vc1dsp.c) — VC-1's own separable 8x8 integer inverse transform.
static inline void vc1_inv_trans_8x8(int16_t block[64]) {
    int t1, t2, t3, t4, t5, t6, t7, t8;
    int16_t temp[64];
    const int16_t* src = block;
    int16_t* dst = temp;
    for (int i = 0; i < 8; i++) {
        t1 = 12 * (src[0] + src[32]) + 4;
        t2 = 12 * (src[0] - src[32]) + 4;
        t3 = 16 * src[16] +  6 * src[48];
        t4 =  6 * src[16] - 16 * src[48];
        t5 = t1 + t3; t6 = t2 + t4; t7 = t2 - t4; t8 = t1 - t3;
        t1 = 16 * src[8] + 15 * src[24] +  9 * src[40] +  4 * src[56];
        t2 = 15 * src[8] -  4 * src[24] - 16 * src[40] -  9 * src[56];
        t3 =  9 * src[8] - 16 * src[24] +  4 * src[40] + 15 * src[56];
        t4 =  4 * src[8] -  9 * src[24] + 15 * src[40] - 16 * src[56];
        dst[0] = (int16_t)((t5 + t1) >> 3); dst[1] = (int16_t)((t6 + t2) >> 3);
        dst[2] = (int16_t)((t7 + t3) >> 3); dst[3] = (int16_t)((t8 + t4) >> 3);
        dst[4] = (int16_t)((t8 - t4) >> 3); dst[5] = (int16_t)((t7 - t3) >> 3);
        dst[6] = (int16_t)((t6 - t2) >> 3); dst[7] = (int16_t)((t5 - t1) >> 3);
        src += 1; dst += 8;
    }
    src = temp; dst = block;
    for (int i = 0; i < 8; i++) {
        t1 = 12 * (src[0] + src[32]) + 64;
        t2 = 12 * (src[0] - src[32]) + 64;
        t3 = 16 * src[16] +  6 * src[48];
        t4 =  6 * src[16] - 16 * src[48];
        t5 = t1 + t3; t6 = t2 + t4; t7 = t2 - t4; t8 = t1 - t3;
        t1 = 16 * src[8] + 15 * src[24] +  9 * src[40] +  4 * src[56];
        t2 = 15 * src[8] -  4 * src[24] - 16 * src[40] -  9 * src[56];
        t3 =  9 * src[8] - 16 * src[24] +  4 * src[40] + 15 * src[56];
        t4 =  4 * src[8] -  9 * src[24] + 15 * src[40] - 16 * src[56];
        dst[ 0] = (int16_t)((t5 + t1)     >> 7); dst[ 8] = (int16_t)((t6 + t2)     >> 7);
        dst[16] = (int16_t)((t7 + t3)     >> 7); dst[24] = (int16_t)((t8 + t4)     >> 7);
        dst[32] = (int16_t)((t8 - t4 + 1) >> 7); dst[40] = (int16_t)((t7 - t3 + 1) >> 7);
        dst[48] = (int16_t)((t6 - t2 + 1) >> 7); dst[56] = (int16_t)((t5 - t1 + 1) >> 7);
        src++; dst++;
    }
}

// ff_put_pixels_clamped_c (idctdsp) — clip block to [0,255] into dst (no +128; I-frame
// intra with overlap disabled uses the unsigned put).
static inline void vc1_put_pixels_clamped(const int16_t* block, uint8_t* dst, int stride) {
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            int x = block[j];
            dst[j] = (uint8_t)(x < 0 ? 0 : (x > 255 ? 255 : x));
        }
        block += 8; dst += stride;
    }
}

static inline void vc1_put_mb(VC1Context* v, int16_t block[6][64]) {
    int ys = v->yStride, cs = v->cStride;
    uint8_t* yb = v->planeY + (size_t)(v->mb_y * 16) * ys + v->mb_x * 16;
    for (int i = 0; i < 4; i++)
        vc1_put_pixels_clamped(block[i], yb + ((i & 2) >> 1) * 8 * ys + (i & 1) * 8, ys);
    uint8_t* ub = v->planeU + (size_t)(v->mb_y * 8) * cs + v->mb_x * 8;
    uint8_t* vb = v->planeV + (size_t)(v->mb_y * 8) * cs + v->mb_x * 8;
    vc1_put_pixels_clamped(block[4], ub, cs);
    vc1_put_pixels_clamped(block[5], vb, cs);
}

// ── In-loop deblocking filter (vc1dsp.c / vc1_loopfilter.c), Simple/Main I-frame ──
// vc1_filter_line: filters one line straddling an edge (stride = distance between the
// two pixels across the edge). Returns 1 if the neighbouring 3 pairs should filter too.
static inline int vc1_filter_line(uint8_t* src, int stride, int pq) {
    int a0 = (2 * (src[-2 * stride] - src[1 * stride]) -
              5 * (src[-1 * stride] - src[0 * stride]) + 4) >> 3;
    int a0_sign = a0 >> 31;
    a0 = (a0 ^ a0_sign) - a0_sign;                 // FFABS(a0)
    if (a0 < pq) {
        int a1 = abs((2 * (src[-4 * stride] - src[-1 * stride]) -
                      5 * (src[-3 * stride] - src[-2 * stride]) + 4) >> 3);
        int a2 = abs((2 * (src[0 * stride] - src[3 * stride]) -
                      5 * (src[1 * stride] - src[2 * stride]) + 4) >> 3);
        if (a1 < a0 || a2 < a0) {
            int clip = src[-1 * stride] - src[0 * stride];
            int clip_sign = clip >> 31;
            clip = ((clip ^ clip_sign) - clip_sign) >> 1;
            if (clip) {
                int a3 = a1 < a2 ? a1 : a2;
                int d  = 5 * (a3 - a0);
                int d_sign = d >> 31;
                d = ((d ^ d_sign) - d_sign) >> 3;
                d_sign ^= a0_sign;
                if (d_sign ^ clip_sign) d = 0;
                else {
                    d = d < clip ? d : clip;
                    d = (d ^ d_sign) - d_sign;      // restore sign
                    int p = src[-1 * stride] - d; src[-1 * stride] = (uint8_t)(p < 0 ? 0 : (p > 255 ? 255 : p));
                    int q = src[ 0 * stride] + d; src[ 0 * stride] = (uint8_t)(q < 0 ? 0 : (q > 255 ? 255 : q));
                }
                return 1;
            }
        }
    }
    return 0;
}

// vc1_loop_filter: `len` pixels along the edge in groups of 4; the 3rd sample of each
// group decides whether the group filters (8.6).
static inline void vc1_loop_filter(uint8_t* src, int step, int stride, int len, int pq) {
    for (int i = 0; i < len; i += 4) {
        if (vc1_filter_line(src + 2 * step, stride, pq)) {
            vc1_filter_line(src + 0 * step, stride, pq);
            vc1_filter_line(src + 1 * step, stride, pq);
            vc1_filter_line(src + 3 * step, stride, pq);
        }
        src += step * 4;
    }
}

// ff_vc1_i_loop_filter reduced to two full-frame passes: ffmpeg's trailing schedule
// guarantees every horizontal-edge (V) filter precedes every vertical-edge (H) filter
// on the pixels they share; within a pass, 8x8 block edges are >= 8px apart so each
// filter (±4px reach) is independent of the others — order within a pass is irrelevant.
// Filters all block-boundary edges except the frame borders (pq constant on I-frames).
static inline void vc1_i_deblock_frame(VC1Context* v) {
    int pq = v->pq, ys = v->yStride, cs = v->cStride, mw = v->mb_width, mh = v->mb_height;

    // Pass 1 — horizontal edges (luma internal + top boundary, 16px; chroma top, 8px).
    for (int my = 0; my < mh; my++)
        for (int mx = 0; mx < mw; mx++) {
            uint8_t* y = v->planeY + (size_t)(my * 16) * ys + mx * 16;
            vc1_loop_filter(y + 8 * ys, 1, ys, 16, pq);           // internal (block 2)
            if (my > 0) {
                vc1_loop_filter(y, 1, ys, 16, pq);                // top boundary (block 0)
                uint8_t* u = v->planeU + (size_t)(my * 8) * cs + mx * 8;
                uint8_t* w = v->planeV + (size_t)(my * 8) * cs + mx * 8;
                vc1_loop_filter(u, 1, cs, 8, pq);                 // chroma top (blocks 4,5)
                vc1_loop_filter(w, 1, cs, 8, pq);
            }
        }

    // Pass 2 — vertical edges (luma internal + left boundary, 16px; chroma left, 8px).
    for (int my = 0; my < mh; my++)
        for (int mx = 0; mx < mw; mx++) {
            uint8_t* y = v->planeY + (size_t)(my * 16) * ys + mx * 16;
            vc1_loop_filter(y + 8, ys, 1, 16, pq);                // internal (block 1)
            if (mx > 0) {
                vc1_loop_filter(y, ys, 1, 16, pq);                // left boundary (block 0)
                uint8_t* u = v->planeU + (size_t)(my * 8) * cs + mx * 8;
                uint8_t* w = v->planeV + (size_t)(my * 8) * cs + mx * 8;
                vc1_loop_filter(u, cs, 1, 8, pq);                 // chroma left (blocks 4,5)
                vc1_loop_filter(w, cs, 1, 8, pq);
            }
        }
}

// vc1_decode_i_blocks (vc1_block.c), simplified: overlap is disabled in our target
// stream, so blocks are put immediately (no rotating-buffer delay). The in-loop
// deblock runs as a separate full-frame pass (added later). Returns VC1_OK.
static inline int vc1_decode_i_frame(VC1Context* v, mp4::GetBits* gb) {
    vc1_init_iframe_tables();

    switch (v->y_ac_table_index) {
    case 0: v->codingset = (v->pqindex <= 8) ? CS_HIGH_RATE_INTRA : CS_LOW_MOT_INTRA; break;
    case 1: v->codingset = CS_HIGH_MOT_INTRA; break;
    case 2: v->codingset = CS_MID_RATE_INTRA; break;
    }
    switch (v->c_ac_table_index) {
    case 0: v->codingset2 = (v->pqindex <= 8) ? CS_HIGH_RATE_INTER : CS_LOW_MOT_INTER; break;
    case 1: v->codingset2 = CS_HIGH_MOT_INTER; break;
    case 2: v->codingset2 = CS_MID_RATE_INTER; break;
    }
    v->y_dc_scale = ff_wmv3_dc_scale_table[v->pq];
    v->esc3_level_length = 0;

    memset(v->dc_base, 0, (size_t)v->grid_total * sizeof(int16_t));
    memset(v->ac_base, 0, (size_t)v->grid_total * 16 * sizeof(int16_t));
    memset(v->coded_base, 0, (size_t)v->grid_total);

    for (v->mb_y = 0; v->mb_y < v->mb_height; v->mb_y++) {
        v->first_slice_line = (v->mb_y == 0);
        for (v->mb_x = 0; v->mb_x < v->mb_width; v->mb_x++) {
            vc1_init_block_index(v);
            int16_t block[6][64];
            memset(block, 0, sizeof(block));

            int cbp = mp4::get_vlc2(gb, g_mb_i.table, g_mb_i.bits);
            if (cbp < 0) return VC1_ERR;
            v->ac_pred = (int)mp4::get_bits1(gb);

            for (int k = 0; k < 6; k++) {
                int val = (cbp >> (5 - k)) & 1;
                if (k < 4) val = vc1_coded_block_pred(v, k, val);
                int ret = vc1_decode_i_block(v, block[k], k, val,
                                             (k < 4) ? v->codingset : v->codingset2, gb);
                if (ret < 0) return ret;
                vc1_inv_trans_8x8(block[k]);
            }
            vc1_put_mb(v, block);
        }
    }
    if (v->loop_filter) vc1_i_deblock_frame(v);
    return VC1_OK;
}

// ═══════════════════════ P-frame inter decode (Simple/Main) ═══════════════════════
// NOTE: this path assumes dquant == 0 (true for the target stream) — every MB shares
// qscale == pq, so the per-MB dqscale predictor rescaling in ff_vc1_pred_dc /
// vc1_decode_intra_block collapses (q1 == q2) and is omitted.

static inline int vc1_clampi(int x, int lo, int hi) { return x < lo ? lo : (x > hi ? hi : x); }
static inline int vc1_mid_pred(int a, int b, int c) {
    int mn = a < b ? a : b; mn = mn < c ? mn : c;
    int mx = a > b ? a : b; mx = mx > c ? mx : c;
    return a + b + c - mn - mx;
}
static inline int vc1_median4(int a, int b, int c, int d) {
    if (a < b) {
        if (c < d) return ((b < d ? b : d) + (a > c ? a : c)) / 2;
        else       return ((b < c ? b : c) + (a > d ? a : d)) / 2;
    } else {
        if (c < d) return ((a < d ? a : d) + (b > c ? b : c)) / 2;
        else       return ((a < c ? a : c) + (b > d ? b : d)) / 2;
    }
}

// ff_vc1_pred_dc (dquant==0 subset — predictor rescaling omitted).
static inline int vc1_pred_dc(VC1Context* v, int n, int a_avail, int c_avail,
                              int16_t** dc_val_ptr, int* dir_ptr) {
    int wrap = v->block_wrap[n];
    int16_t* dc_val = v->dc_val + v->block_index[n];
    int c = dc_val[-1], b = dc_val[-1 - wrap], a = dc_val[-wrap], pred;
    if (c_avail && (!a_avail || abs(a - b) <= abs(b - c))) { pred = c; *dir_ptr = 1; }
    else if (a_avail)                                      { pred = a; *dir_ptr = 0; }
    else                                                  { pred = 0; *dir_ptr = 1; }
    *dc_val_ptr = dc_val;
    return pred;
}

// vc1_decode_intra_block (P-frame intra; dquant==0 subset). Coeffs land in `block`.
static inline int vc1_decode_intra_block_p(VC1Context* v, int16_t block[64], int n,
                                           int coded, int mquant, int codingset, mp4::GetBits* gb) {
    int dc_pred_dir = 0;
    int16_t *dc_val, *ac_val, *ac_val2;
    int dcdiff, scale;
    int a_avail = v->a_avail, c_avail = v->c_avail, use_pred = v->ac_pred;
    int quant = mquant < 0 ? -mquant : mquant;

    memset(block, 0, 64 * sizeof(int16_t));
    v->y_dc_scale = ff_wmv3_dc_scale_table[quant];

    dcdiff = msm::msm_dc_vlc(gb, msm::ff_msmp4_dc_tables[v->dc_table_index][n >= 4 ? 1 : 0]);
    if (dcdiff < 0) return VC1_ERR;
    if (dcdiff) {
        const int m = (quant == 1 || quant == 2) ? 3 - quant : 0;
        if (dcdiff == 119) dcdiff = (int)mp4::get_bits(gb, 8 + m);
        else if (m)        dcdiff = (dcdiff << m) + (int)mp4::get_bits(gb, m) - ((1 << m) - 1);
        if (mp4::get_bits1(gb)) dcdiff = -dcdiff;
    }
    dcdiff += vc1_pred_dc(v, n, a_avail, c_avail, &dc_val, &dc_pred_dir);
    *dc_val  = (int16_t)dcdiff;
    block[0] = (int16_t)(dcdiff * v->y_dc_scale);

    if (!a_avail) dc_pred_dir = 1;
    if (!c_avail) dc_pred_dir = 0;
    if (!a_avail && !c_avail) use_pred = 0;
    ac_val  = v->ac_val + (size_t)v->block_index[n] * 16;
    ac_val2 = ac_val;
    scale = quant * 2 + (mquant < 0 ? 0 : v->halfpq);
    if (dc_pred_dir) ac_val -= 16;
    else             ac_val -= 16 * v->block_wrap[n];

    if (coded) {
        int last = 0, skip, value, k;
        for (int i = 1; !last; ++i) {
            if (vc1_decode_ac_coeff(v, &last, &skip, &value, codingset, gb) < 0) return VC1_ERR;
            i += skip;
            if (i > 63) break;
            block[v->zz_8x8[0][i]] = (int16_t)value;   // PROGRESSIVE scan
        }
        if (use_pred) {                                 // dquant==0 ⇒ plain add
            if (dc_pred_dir) for (k = 1; k < 8; k++) block[k << v->left_blk_sh] += ac_val[k];
            else             for (k = 1; k < 8; k++) block[k << v->top_blk_sh]  += ac_val[k + 8];
        }
        for (k = 1; k < 8; k++) {
            ac_val2[k]     = block[k << v->left_blk_sh];
            ac_val2[k + 8] = block[k << v->top_blk_sh];
        }
        for (k = 1; k < 64; k++) if (block[k]) {
            block[k] *= scale;
            if (!v->pquantizer) block[k] += (block[k] < 0) ? -quant : quant;
        }
    } else {
        int k;
        memset(ac_val2, 0, 16 * 2);
        if (use_pred) {
            if (dc_pred_dir) {
                memcpy(ac_val2, ac_val, 8 * 2);
                for (k = 1; k < 8; k++) {
                    block[k << v->left_blk_sh] = ac_val2[k] * scale;
                    if (!v->pquantizer && block[k << v->left_blk_sh])
                        block[k << v->left_blk_sh] += (block[k << v->left_blk_sh] < 0) ? -quant : quant;
                }
            } else {
                memcpy(ac_val2 + 8, ac_val + 8, 8 * 2);
                for (k = 1; k < 8; k++) {
                    block[k << v->top_blk_sh] = ac_val2[k + 8] * scale;
                    if (!v->pquantizer && block[k << v->top_blk_sh])
                        block[k << v->top_blk_sh] += (block[k << v->top_blk_sh] < 0) ? -quant : quant;
                }
            }
        }
    }
    return 0;
}

// vc1_decode_p_block — inter residual, variable transform sizes; adds to `dst`.
static inline int vc1_decode_p_block(VC1Context* v, int16_t block[64], int n, int mquant,
                                     int ttmb, int first_block, uint8_t* dst, int linesize,
                                     int* ttmb_out, mp4::GetBits* gb) {
    int i, j, subblkpat = 0, off, idx, last, skip, value;
    int ttblk = ttmb & 7, pat = 0;
    int quant = mquant < 0 ? -mquant : mquant;
    int scale = quant * 2 + (mquant < 0 ? 0 : v->halfpq);
    memset(block, 0, 64 * sizeof(int16_t));

    if (ttmb == -1)
        ttblk = ff_vc1_ttblk_to_tt[v->tt_index][ml_get_vlc2(gb, g_ttblk_vlc[v->tt_index], 5, 1)];
    if (ttblk == TT_4X4)
        subblkpat = ~(ml_get_vlc2(gb, g_subblkpat_vlc[v->tt_index], 6, 1) + 1);
    if ((ttblk != TT_8X8 && ttblk != TT_4X4)
        && ((v->ttmbf || (ttmb != -1 && (ttmb & 8) && !first_block))
            || (!v->res_rtm_flag && !first_block))) {
        subblkpat = vc1_decode012(gb);
        if (subblkpat) subblkpat ^= 3;
        if (ttblk == TT_8X4_TOP || ttblk == TT_8X4_BOTTOM) ttblk = TT_8X4;
        if (ttblk == TT_4X8_RIGHT || ttblk == TT_4X8_LEFT) ttblk = TT_4X8;
    }
    if (ttblk == TT_8X4_TOP || ttblk == TT_8X4_BOTTOM) { subblkpat = 2 - (ttblk == TT_8X4_TOP);  ttblk = TT_8X4; }
    if (ttblk == TT_4X8_RIGHT || ttblk == TT_4X8_LEFT) { subblkpat = 2 - (ttblk == TT_4X8_LEFT); ttblk = TT_4X8; }

    switch (ttblk) {
    case TT_8X8:
        pat = 0xF; i = 0; last = 0;
        while (!last) {
            if (vc1_decode_ac_coeff(v, &last, &skip, &value, v->codingset2, gb) < 0) return VC1_ERR;
            i += skip; if (i > 63) break;
            idx = v->zz_8x8[0][i++];
            block[idx] = (int16_t)(value * scale);
            if (!v->pquantizer) block[idx] += (block[idx] < 0) ? -quant : quant;
        }
        if (i == 1) vc1_inv_trans_8x8_dc(dst, linesize, block);
        else { vc1_inv_trans_8x8(block); vc1_add_pixels_clamped(block, dst, linesize); }
        break;
    case TT_4X4:
        pat = ~subblkpat & 0xF;
        for (j = 0; j < 4; j++) {
            last = subblkpat & (1 << (3 - j)); i = 0; off = (j & 1) * 4 + (j & 2) * 16;
            while (!last) {
                if (vc1_decode_ac_coeff(v, &last, &skip, &value, v->codingset2, gb) < 0) return VC1_ERR;
                i += skip; if (i > 15) break;
                idx = ff_vc1_simple_progressive_4x4_zz[i++];
                block[idx + off] = (int16_t)(value * scale);
                if (!v->pquantizer) block[idx + off] += (block[idx + off] < 0) ? -quant : quant;
            }
            if (!(subblkpat & (1 << (3 - j)))) {
                uint8_t* d = dst + (j & 1) * 4 + (j & 2) * 2 * linesize;
                if (i == 1) vc1_inv_trans_4x4_dc(d, linesize, block + off);
                else        vc1_inv_trans_4x4(d, linesize, block + off);
            }
        }
        break;
    case TT_8X4:
        pat = ~((subblkpat & 2) * 6 + (subblkpat & 1) * 3) & 0xF;
        for (j = 0; j < 2; j++) {
            last = subblkpat & (1 << (1 - j)); i = 0; off = j * 32;
            while (!last) {
                if (vc1_decode_ac_coeff(v, &last, &skip, &value, v->codingset2, gb) < 0) return VC1_ERR;
                i += skip; if (i > 31) break;
                idx = ff_wmv2_scantableA[i++] + off;      // zz_8x4
                block[idx] = (int16_t)(value * scale);
                if (!v->pquantizer) block[idx] += (block[idx] < 0) ? -quant : quant;
            }
            if (!(subblkpat & (1 << (1 - j)))) {
                uint8_t* d = dst + j * 4 * linesize;
                if (i == 1) vc1_inv_trans_8x4_dc(d, linesize, block + off);
                else        vc1_inv_trans_8x4(d, linesize, block + off);
            }
        }
        break;
    case TT_4X8:
        pat = ~(subblkpat * 5) & 0xF;
        for (j = 0; j < 2; j++) {
            last = subblkpat & (1 << (1 - j)); i = 0; off = j * 4;
            while (!last) {
                if (vc1_decode_ac_coeff(v, &last, &skip, &value, v->codingset2, gb) < 0) return VC1_ERR;
                i += skip; if (i > 31) break;
                idx = ff_wmv2_scantableB[i++] + off;      // zz_4x8
                block[idx] = (int16_t)(value * scale);
                if (!v->pquantizer) block[idx] += (block[idx] < 0) ? -quant : quant;
            }
            if (!(subblkpat & (1 << (1 - j)))) {
                uint8_t* d = dst + j * 4;
                if (i == 1) vc1_inv_trans_4x8_dc(d, linesize, block + off);
                else        vc1_inv_trans_4x8(d, linesize, block + off);
            }
        }
        break;
    }
    if (ttmb_out) *ttmb_out |= ttblk << (n * 4);
    return pat;
}

// GET_MVDATA (progressive). Sets *dmv_x/y, *has_coeffs, *intra.
static inline void vc1_get_mvdata(VC1Context* v, mp4::GetBits* gb, int* pdx, int* pdy,
                                  int* phc, int* pintra) {
    int index = 1 + ml_get_vlc2(gb, g_mv_diff_vlc[v->mv_table_index], 9, 2);
    int qs = v->quarter_sample, val, sign, i1, dx, dy;
    if (index > 36) { *phc = 1; index -= 37; } else *phc = 0;
    *pintra = 0;
    if (!index) { dx = dy = 0; }
    else if (index == 35) {
        dx = (int)mp4::get_bits(gb, v->k_x - 1 + qs);
        dy = (int)mp4::get_bits(gb, v->k_y - 1 + qs);
    } else if (index == 36) { dx = dy = 0; *pintra = 1; }
    else {
        i1 = index % 6; dx = vc1_offset_table[1][i1];
        val = vc1_size_table[i1] - (!qs && i1 == 5);
        if (val > 0) { val = (int)mp4::get_bits(gb, val); sign = 0 - (val & 1); dx = (sign ^ ((val >> 1) + dx)) - sign; }
        i1 = index / 6; dy = vc1_offset_table[1][i1];
        val = vc1_size_table[i1] - (!qs && i1 == 5);
        if (val > 0) { val = (int)mp4::get_bits(gb, val); sign = 0 - (val & 1); dy = (sign ^ ((val >> 1) + dy)) - sign; }
    }
    *pdx = dx; *pdy = dy;
}

// ff_vc1_pred_mv (PROGRESSIVE subset: field_mode=0, numref=0, blocks_off=0, mv_f=0).
static inline void vc1_pred_mv(VC1Context* v, int n, int dmv_x, int dmv_y, int mv1,
                               int r_x, int r_y, int mb_intra, mp4::GetBits* gb) {
    int wrap = v->b8_stride, xy = v->block_index[n], off = 0, px, py, sum;
    int16_t* mvbase = v->mv_val;
    if (!v->quarter_sample) { dmv_x *= 2; dmv_y *= 2; }
    if (mb_intra) {
        v->cur_mv[n][0] = mvbase[xy * 2] = 0;
        v->cur_mv[n][1] = mvbase[xy * 2 + 1] = 0;
        if (mv1) {
            mvbase[(xy + 1) * 2] = mvbase[(xy + 1) * 2 + 1] = 0;
            mvbase[(xy + wrap) * 2] = mvbase[(xy + wrap) * 2 + 1] = 0;
            mvbase[(xy + wrap + 1) * 2] = mvbase[(xy + wrap + 1) * 2 + 1] = 0;
        }
        return;
    }
    int a_valid = !v->first_slice_line || (n == 2 || n == 3);
    int b_valid = a_valid;
    int c_valid = v->mb_x || (n == 1 || n == 3);
    if (mv1) {
        off = (v->mb_x == v->mb_width - 1) ? -1 : 2;
        b_valid = b_valid && v->mb_width > 1;
    } else {
        switch (n) {
        case 0: off = v->mb_x ? -1 : 1; break;              // res_rtm_flag == 1
        case 1: off = (v->mb_x == v->mb_width - 1) ? -1 : 1; break;
        case 2: off = 1; break;
        case 3: off = -1; break;
        }
    }
    int A0 = 0, A1 = 0, B0 = 0, B1 = 0, C0 = 0, C1 = 0, nvalid = 0;
    if (a_valid) { A0 = mvbase[(xy - wrap) * 2]; A1 = mvbase[(xy - wrap) * 2 + 1]; nvalid++; }
    if (b_valid) { B0 = mvbase[(xy - wrap + off) * 2]; B1 = mvbase[(xy - wrap + off) * 2 + 1]; nvalid++; }
    if (c_valid) { C0 = mvbase[(xy - 1) * 2]; C1 = mvbase[(xy - 1) * 2 + 1]; nvalid++; }

    if (a_valid)      { px = A0; py = A1; }
    else if (c_valid) { px = C0; py = C1; }
    else if (b_valid) { px = B0; py = B1; }
    else              { px = py = 0; }
    if (nvalid > 1) { px = vc1_mid_pred(A0, B0, C0); py = vc1_mid_pred(A1, B1, C1); }

    // Pullback (8.3.5.3.4)
    {
        int MV = mv1 ? -60 : -28;
        int qx = (v->mb_x << 6) + ((n == 1 || n == 3) ? 32 : 0);
        int qy = (v->mb_y << 6) + ((n == 2 || n == 3) ? 32 : 0);
        int X = (v->mb_width << 6) - 4, Y = (v->mb_height << 6) - 4;
        if (qx + px < MV) px = MV - qx;
        if (qy + py < MV) py = MV - qy;
        if (qx + px > X) px = X - qx;
        if (qy + py > Y) py = Y - qy;
    }
    // Hybrid prediction (8.3.5.3.5)
    if (a_valid && c_valid) {
        sum = abs(px - A0) + abs(py - A1);
        if (sum > 32) {
            if (mp4::get_bits1(gb)) { px = A0; py = A1; } else { px = C0; py = C1; }
        } else {
            sum = abs(px - C0) + abs(py - C1);
            if (sum > 32) {
                if (mp4::get_bits1(gb)) { px = A0; py = A1; } else { px = C0; py = C1; }
            }
        }
    }
    int mx = ((px + dmv_x + r_x) & ((r_x << 1) - 1)) - r_x;
    int my = ((py + dmv_y + r_y) & ((r_y << 1) - 1)) - r_y;
    v->cur_mv[n][0] = (int16_t)mx; v->cur_mv[n][1] = (int16_t)my;
    mvbase[xy * 2] = (int16_t)mx; mvbase[xy * 2 + 1] = (int16_t)my;
    if (mv1) {
        mvbase[(xy + 1) * 2] = (int16_t)mx; mvbase[(xy + 1) * 2 + 1] = (int16_t)my;
        mvbase[(xy + wrap) * 2] = (int16_t)mx; mvbase[(xy + wrap) * 2 + 1] = (int16_t)my;
        mvbase[(xy + wrap + 1) * 2] = (int16_t)mx; mvbase[(xy + wrap + 1) * 2 + 1] = (int16_t)my;
    }
}

// ── Motion compensation (always emulate the source footprint = ffmpeg pixel-exact) ─
static inline void vc1_mc_luma_block(VC1Context* v, uint8_t* dest, int dstride, const uint8_t* ref, int rstride,
                                     int src_x, int src_y, int mx, int my, int rnd, int cw, int ch, int N) {
    uint8_t buf[19 * 19];
    if (v->mspel) {
        const int k = N + 3;   // 19 (N=16) / 11 (N=8)
        vc1_emulated_edge(buf, k, ref, rstride, k, k, src_x - 1, src_y - 1, cw, ch);
        vc1_put_mspel_mc(dest, dstride, buf + k + 1, k, mx & 3, my & 3, rnd, N);
    } else {               // half-pel bilinear (1MV_HPEL_BILIN)
        const int k = N + 1;
        vc1_emulated_edge(buf, k, ref, rstride, k, k, src_x, src_y, cw, ch);
        vc1_put_hpel(dest, dstride, buf, k, N, (my & 2) | ((mx & 2) >> 1), 1 - rnd);
    }
}
static inline void vc1_mc_chroma_block(uint8_t* dest, int dstride, const uint8_t* ref, int rstride,
                                       int src_x, int src_y, int cx, int cy, int rnd, int cw, int ch) {
    uint8_t buf[9 * 9];
    vc1_emulated_edge(buf, 9, ref, rstride, 9, 9, src_x, src_y, cw, ch);
    vc1_put_chroma_mc8(dest, dstride, buf, 9, 8, cx, cy, rnd);
}

static inline void vc1_mc_1mv(VC1Context* v) {
    int mx = v->cur_mv[0][0], my = v->cur_mv[0][1];
    int uvmx = (mx + ((mx & 3) == 3)) >> 1, uvmy = (my + ((my & 3) == 3)) >> 1;
    v->fr_lumamv[v->mb_x + v->mb_y * v->mb_width][0] = (int16_t)uvmx;
    v->fr_lumamv[v->mb_x + v->mb_y * v->mb_width][1] = (int16_t)uvmy;
    int cw = v->codedW, ch = v->codedH;
    int src_x = vc1_clampi(v->mb_x * 16 + (mx >> 2), -16, cw);
    int src_y = vc1_clampi(v->mb_y * 16 + (my >> 2), -16, ch);
    uint8_t* dY = v->planeY + (size_t)(v->mb_y * 16) * v->yStride + v->mb_x * 16;
    vc1_mc_luma_block(v, dY, v->yStride, v->refY, v->yStride, src_x, src_y, mx, my, v->rnd, cw, ch, 16);
    int ucw = cw >> 1, uch = ch >> 1;
    int usx = vc1_clampi(v->mb_x * 8 + (uvmx >> 2), -8, ucw);
    int usy = vc1_clampi(v->mb_y * 8 + (uvmy >> 2), -8, uch);
    int cx = (uvmx & 3) << 1, cy = (uvmy & 3) << 1;
    uint8_t* dU = v->planeU + (size_t)(v->mb_y * 8) * v->cStride + v->mb_x * 8;
    uint8_t* dV = v->planeV + (size_t)(v->mb_y * 8) * v->cStride + v->mb_x * 8;
    vc1_mc_chroma_block(dU, v->cStride, v->refU, v->cStride, usx, usy, cx, cy, v->rnd, ucw, uch);
    vc1_mc_chroma_block(dV, v->cStride, v->refV, v->cStride, usx, usy, cx, cy, v->rnd, ucw, uch);
}

static inline void vc1_mc_4mv_luma(VC1Context* v, int n) {
    int mx = v->cur_mv[n][0], my = v->cur_mv[n][1];
    int cw = v->codedW, ch = v->codedH;
    int off = v->yStride * 4 * (n & 2) + (n & 1) * 8;
    int src_x = vc1_clampi(v->mb_x * 16 + (n & 1) * 8 + (mx >> 2), -16, cw);
    int src_y = vc1_clampi(v->mb_y * 16 + (n & 2) * 4 + (my >> 2), -16, ch);
    uint8_t* dY = v->planeY + (size_t)(v->mb_y * 16) * v->yStride + v->mb_x * 16 + off;
    vc1_mc_luma_block(v, dY, v->yStride, v->refY, v->yStride, src_x, src_y, mx, my, v->rnd, cw, ch, 8);
}

static inline void vc1_mc_4mv_chroma(VC1Context* v) {
    int idx = (!v->mbtype[v->block_index[0]]) | (!v->mbtype[v->block_index[1]] << 1)
            | (!v->mbtype[v->block_index[2]] << 2) | (!v->mbtype[v->block_index[3]] << 3);
    static const uint8_t popcount4[16] = { 0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4 };
    static const uint8_t index2[16] = { 0,0,0,0x01,0,0x02,0x12,0,0,0x03,0x13,0,0x23,0,0,0 };
    int valid = popcount4[idx], tx, ty;
    if (valid == 4) {
        tx = vc1_median4(v->cur_mv[0][0], v->cur_mv[1][0], v->cur_mv[2][0], v->cur_mv[3][0]);
        ty = vc1_median4(v->cur_mv[0][1], v->cur_mv[1][1], v->cur_mv[2][1], v->cur_mv[3][1]);
    } else if (valid == 3) {
        tx = vc1_mid_pred(v->cur_mv[idx > 0xd][0], v->cur_mv[1 + (idx > 0xb)][0], v->cur_mv[2 + (idx > 0x7)][0]);
        ty = vc1_mid_pred(v->cur_mv[idx > 0xd][1], v->cur_mv[1 + (idx > 0xb)][1], v->cur_mv[2 + (idx > 0x7)][1]);
    } else if (valid == 2) {
        tx = (v->cur_mv[index2[idx] >> 4][0] + v->cur_mv[index2[idx] & 0xf][0]) / 2;
        ty = (v->cur_mv[index2[idx] >> 4][1] + v->cur_mv[index2[idx] & 0xf][1]) / 2;
    } else return;
    int uvmx = (tx + ((tx & 3) == 3)) >> 1, uvmy = (ty + ((ty & 3) == 3)) >> 1;
    v->fr_lumamv[v->mb_x + v->mb_y * v->mb_width][0] = (int16_t)uvmx;
    v->fr_lumamv[v->mb_x + v->mb_y * v->mb_width][1] = (int16_t)uvmy;
    int ucw = v->codedW >> 1, uch = v->codedH >> 1;
    int usx = vc1_clampi(v->mb_x * 8 + (uvmx >> 2), -8, ucw);
    int usy = vc1_clampi(v->mb_y * 8 + (uvmy >> 2), -8, uch);
    int cx = (uvmx & 3) << 1, cy = (uvmy & 3) << 1;
    uint8_t* dU = v->planeU + (size_t)(v->mb_y * 8) * v->cStride + v->mb_x * 8;
    uint8_t* dV = v->planeV + (size_t)(v->mb_y * 8) * v->cStride + v->mb_x * 8;
    vc1_mc_chroma_block(dU, v->cStride, v->refU, v->cStride, usx, usy, cx, cy, v->rnd, ucw, uch);
    vc1_mc_chroma_block(dV, v->cStride, v->refV, v->cStride, usx, usy, cx, cy, v->rnd, ucw, uch);
}

// put_signed_pixels_clamped: dst = clip(block + 128) — for intra blocks in a P frame.
static inline void vc1_put_signed_block(const int16_t* block, uint8_t* dst, int stride) {
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) dst[j] = vc1_clip8(block[j] + 128);
        block += 8; dst += stride;
    }
}

static inline int vc1_decode_p_mb(VC1Context* v, mp4::GetBits* gb) {
    int mb_pos = v->mb_x + v->mb_y * v->mb_stride;
    int cbp, mquant = v->pq, ttmb = v->ttfrm;
    int dmv_x, dmv_y, mb_has_coeffs = 1, mb_intra;
    int fourmv, skipped, first_block = 1, dst_idx, off;
    int block_cbp = 0, block_tt = 0, block_intra = 0, i;
    int16_t block[6][64];

    uint8_t* dY = v->planeY + (size_t)(v->mb_y * 16) * v->yStride + v->mb_x * 16;
    uint8_t* dU = v->planeU + (size_t)(v->mb_y * 8) * v->cStride + v->mb_x * 8;
    uint8_t* dV = v->planeV + (size_t)(v->mb_y * 8) * v->cStride + v->mb_x * 8;
    uint8_t* dest[3] = { dY, dU, dV };
    int lss[3] = { v->yStride, v->cStride, v->cStride };

    fourmv  = v->mv_type_is_raw ? (int)mp4::get_bits1(gb) : v->mv_type_mb_plane[mb_pos];
    skipped = v->skip_is_raw    ? (int)mp4::get_bits1(gb) : v->mbskip_table[mb_pos];

    if (!fourmv) {
        if (!skipped) {
            vc1_get_mvdata(v, gb, &dmv_x, &dmv_y, &mb_has_coeffs, &mb_intra);
            vc1_pred_mv(v, 0, dmv_x, dmv_y, 1, v->range_x, v->range_y, mb_intra, gb);

            if (mb_intra && !mb_has_coeffs) { v->ac_pred = (int)mp4::get_bits1(gb); cbp = 0; }
            else if (mb_has_coeffs) {
                if (mb_intra) v->ac_pred = (int)mp4::get_bits1(gb);
                cbp = ml_get_vlc2(gb, v->cbpcy_vlc, 9, 2);
            } else cbp = 0;

            if (!v->ttmbf && !mb_intra && mb_has_coeffs)
                ttmb = ml_get_vlc2(gb, g_ttmb_vlc[v->tt_index], 9, 2);
            if (!mb_intra) vc1_mc_1mv(v);
            dst_idx = 0;
            for (i = 0; i < 6; i++) {
                v->dc_val[v->block_index[i]] = 0;
                dst_idx += i >> 2;
                int val = (cbp >> (5 - i)) & 1;
                off = (i & 4) ? 0 : ((i & 1) * 8 + (i & 2) * 4 * v->yStride);
                v->mbtype[v->block_index[i]] = (uint8_t)(mb_intra ? 1 : 0);
                if (mb_intra) {
                    v->a_avail = v->c_avail = 0;
                    if (i == 2 || i == 3 || !v->first_slice_line) v->a_avail = v->mbtype[v->block_index[i] - v->block_wrap[i]];
                    if (i == 1 || i == 3 || v->mb_x)              v->c_avail = v->mbtype[v->block_index[i] - 1];
                    if (vc1_decode_intra_block_p(v, block[i], i, val, mquant, (i & 4) ? v->codingset2 : v->codingset, gb) < 0) return VC1_ERR;
                    vc1_inv_trans_8x8(block[i]);
                    block_cbp |= 0xF << (i << 2); block_intra |= 1 << i;
                } else if (val) {
                    int pat = vc1_decode_p_block(v, block[i], i, mquant, ttmb, first_block,
                                                 dest[dst_idx] + off, lss[dst_idx], &block_tt, gb);
                    if (pat < 0) return VC1_ERR;
                    block_cbp |= pat << (i << 2);
                    if (!v->ttmbf && ttmb < 8) ttmb = -1;
                    first_block = 0;
                }
            }
        } else {  // skipped
            for (i = 0; i < 6; i++) { v->mbtype[v->block_index[i]] = 0; v->dc_val[v->block_index[i]] = 0; }
            vc1_pred_mv(v, 0, 0, 0, 1, v->range_x, v->range_y, 0, gb);
            vc1_mc_1mv(v);
        }
    } else {  // 4MV
        if (!skipped) {
            int intra_count = 0, coded_inter = 0, is_intra[6], is_coded[6];
            cbp = ml_get_vlc2(gb, v->cbpcy_vlc, 9, 2);
            for (i = 0; i < 6; i++) {
                int val = (cbp >> (5 - i)) & 1;
                v->dc_val[v->block_index[i]] = 0;
                if (i < 4) {
                    dmv_x = dmv_y = 0; mb_intra = 0; mb_has_coeffs = 0;
                    if (val) vc1_get_mvdata(v, gb, &dmv_x, &dmv_y, &mb_has_coeffs, &mb_intra);
                    vc1_pred_mv(v, i, dmv_x, dmv_y, 0, v->range_x, v->range_y, mb_intra, gb);
                    if (!mb_intra) vc1_mc_4mv_luma(v, i);
                    intra_count += mb_intra; is_intra[i] = mb_intra; is_coded[i] = mb_has_coeffs;
                }
                if (i & 4) { is_intra[i] = (intra_count >= 3); is_coded[i] = val; }
                if (i == 4) vc1_mc_4mv_chroma(v);
                v->mbtype[v->block_index[i]] = (uint8_t)is_intra[i];
                if (!coded_inter) coded_inter = !is_intra[i] & is_coded[i];
            }
            dst_idx = 0;
            if (!intra_count && !coded_inter) goto p_end;
            // GET_MQUANT (dquant==0 ⇒ mquant stays pq)
            {   // ac_pred
                int intrapred = 0;
                for (i = 0; i < 6; i++) if (is_intra[i]) {
                    if (((!v->first_slice_line || (i == 2 || i == 3)) && v->mbtype[v->block_index[i] - v->block_wrap[i]])
                        || ((v->mb_x || (i == 1 || i == 3)) && v->mbtype[v->block_index[i] - 1])) { intrapred = 1; break; }
                }
                v->ac_pred = intrapred ? (int)mp4::get_bits1(gb) : 0;
            }
            if (!v->ttmbf && coded_inter) ttmb = ml_get_vlc2(gb, g_ttmb_vlc[v->tt_index], 9, 2);
            for (i = 0; i < 6; i++) {
                dst_idx += i >> 2;
                off = (i & 4) ? 0 : ((i & 1) * 8 + (i & 2) * 4 * v->yStride);
                if (is_intra[i]) {
                    v->a_avail = v->c_avail = 0;
                    if (i == 2 || i == 3 || !v->first_slice_line) v->a_avail = v->mbtype[v->block_index[i] - v->block_wrap[i]];
                    if (i == 1 || i == 3 || v->mb_x)              v->c_avail = v->mbtype[v->block_index[i] - 1];
                    if (vc1_decode_intra_block_p(v, block[i], i, is_coded[i], mquant, (i & 4) ? v->codingset2 : v->codingset, gb) < 0) return VC1_ERR;
                    vc1_inv_trans_8x8(block[i]);
                    block_cbp |= 0xF << (i << 2); block_intra |= 1 << i;
                } else if (is_coded[i]) {
                    int pat = vc1_decode_p_block(v, block[i], i, mquant, ttmb, first_block,
                                                 dest[dst_idx] + off, lss[dst_idx], &block_tt, gb);
                    if (pat < 0) return VC1_ERR;
                    block_cbp |= pat << (i << 2);
                    if (!v->ttmbf && ttmb < 8) ttmb = -1;
                    first_block = 0;
                }
            }
        } else {  // skipped 4MV
            for (i = 0; i < 6; i++) { v->mbtype[v->block_index[i]] = 0; v->dc_val[v->block_index[i]] = 0; }
            for (i = 0; i < 4; i++) { vc1_pred_mv(v, i, 0, 0, 0, v->range_x, v->range_y, 0, gb); vc1_mc_4mv_luma(v, i); }
            vc1_mc_4mv_chroma(v);
        }
    }
p_end:
    // Put the intra blocks (put_signed +128); inter blocks were added to dest directly.
    for (i = 0; i < 6; i++) {
        if (!(block_intra & (1 << i))) continue;
        off = (i & 4) ? 0 : ((i & 1) * 8 + (i & 2) * 4 * v->yStride);
        int di = i >> 2; if (i == 5) di = 2;
        uint8_t* d = (i < 4) ? dY + off : (i == 4 ? dU : dV);
        vc1_put_signed_block(block[i], d, (i < 4) ? v->yStride : v->cStride);
        (void)di;
    }
    int mb_fr = v->mb_x + v->mb_y * v->mb_width;   // fr_* arrays use mb_width, not mb_stride
    v->fr_cbp[mb_fr]     = block_cbp;
    v->fr_ttblk[mb_fr]   = block_tt;
    v->fr_isintra[mb_fr] = (uint8_t)block_intra;
    return VC1_OK;
}

static inline void vc1_p_deblock_frame(VC1Context* v);  // defined after (isolated via skip_loop_filter first)

static inline int vc1_decode_p_frame(VC1Context* v, mp4::GetBits* gb) {
    vc1_init_iframe_tables();
    switch (v->c_ac_table_index) {
    case 0: v->codingset = (v->pqindex <= 8) ? CS_HIGH_RATE_INTRA : CS_LOW_MOT_INTRA;
            v->codingset2 = (v->pqindex <= 8) ? CS_HIGH_RATE_INTER : CS_LOW_MOT_INTER; break;
    case 1: v->codingset = CS_HIGH_MOT_INTRA; v->codingset2 = CS_HIGH_MOT_INTER; break;
    case 2: v->codingset = CS_MID_RATE_INTRA; v->codingset2 = CS_MID_RATE_INTER; break;
    }
    v->cbpcy_vlc = g_cbpcy_p_vlc[v->cbptab];
    v->esc3_level_length = 0;
    v->y_dc_scale = ff_wmv3_dc_scale_table[v->pq];

    memset(v->dc_base, 0, (size_t)v->grid_total * sizeof(int16_t));
    memset(v->ac_base, 0, (size_t)v->grid_total * 16 * sizeof(int16_t));
    memset(v->mbtype_base, 0, (size_t)v->grid_total);
    memset(v->mv_base, 0, (size_t)v->grid_total * 2 * sizeof(int16_t));
    memset(v->fr_lumamv, 0, (size_t)v->mb_width * v->mb_height * sizeof(int16_t) * 2);

    for (v->mb_y = 0; v->mb_y < v->mb_height; v->mb_y++) {
        v->first_slice_line = (v->mb_y == 0);
        for (v->mb_x = 0; v->mb_x < v->mb_width; v->mb_x++) {
            vc1_init_block_index(v);
            if (vc1_decode_p_mb(v, gb) < 0) return VC1_ERR;
        }
    }
    if (v->loop_filter) vc1_p_deblock_frame(v);
    return VC1_OK;
}

// ── P-frame in-loop deblock (ff_vc1_p_loop_filter) ──────────────────────────────
// Two full-frame passes (all horizontal-edge/"V" then all vertical-edge/"H"), MBs in
// raster order, blocks 0..5, and within a block the 8-edge before the 4-edge — which
// reproduces ffmpeg's trailing-schedule result (edges that interfere are ordered the
// same way). Edge extent per block is chosen from cbp / is_intra / MV-difference.
static inline void vc1_v8(uint8_t* d, int ls, int pq)  { vc1_loop_filter(d, 1, ls, 8, pq); }
static inline void vc1_v4(uint8_t* d, int ls, int pq)  { vc1_loop_filter(d, 1, ls, 4, pq); }
static inline void vc1_h8(uint8_t* d, int ls, int pq)  { vc1_loop_filter(d, ls, 1, 8, pq); }
static inline void vc1_h4(uint8_t* d, int ls, int pq)  { vc1_loop_filter(d, ls, 1, 4, pq); }

// Horizontal edges (V loop filter): filters the bottom edge of block `i`.
static inline void vc1_p_v_filter(VC1Context* v, int mbx, int mby, int i,
                                  uint8_t* dY, uint8_t* dU, uint8_t* dV) {
    int pq = v->pq, mw = v->mb_width, mp = mbx + mby * mw;
    int ls = (i > 3) ? v->cStride : v->yStride;
    uint8_t* dst = (i > 3) ? (i == 4 ? dU : dV) : dY + (i & 2) * 4 * v->yStride + (i & 1) * 8;
    int bottom_edge = (mby == v->mb_height - 1);
    uint32_t top_cbp = v->fr_cbp[mp] >> (i * 4), bottom_cbp;
    int top_is_intra = v->fr_isintra[mp] & (1 << i), bottom_is_intra;

    if (!bottom_edge || i < 2) {
        int mp_b = mp + mw;    // MB below
        if (i > 3)      { bottom_is_intra = v->fr_isintra[mp_b] & (1 << i);       bottom_cbp = v->fr_cbp[mp_b] >> (i * 4); }
        else if (i < 2) { bottom_is_intra = v->fr_isintra[mp]   & (1 << (i + 2)); bottom_cbp = v->fr_cbp[mp]   >> ((i + 2) * 4); }
        else            { bottom_is_intra = v->fr_isintra[mp_b] & (1 << (i - 2)); bottom_cbp = v->fr_cbp[mp_b] >> ((i - 2) * 4); }

        int diff;
        if (i > 3) {
            int16_t* a = v->fr_lumamv[mp]; int16_t* b = v->fr_lumamv[mp_b];
            diff = a[0] != b[0] || a[1] != b[1];
        } else {
            int xy = v->block_index[i];
            int16_t* a = v->mv_val + xy * 2; int16_t* b = v->mv_val + (xy + v->b8_stride) * 2;
            diff = a[0] != b[0] || a[1] != b[1];
        }
        if (top_is_intra || bottom_is_intra || diff)
            vc1_v8(dst + 8 * ls, ls, pq);
        else {
            int idx = (top_cbp | (bottom_cbp >> 2)) & 3;
            if (idx & 1) vc1_v4(dst + 8 * ls + 4, ls, pq);
            if (idx & 2) vc1_v4(dst + 8 * ls, ls, pq);
        }
    }
    int tt = v->fr_ttblk[mp] >> (i * 4) & 0xf;
    if (tt == TT_4X4 || tt == TT_8X4) {
        if (top_cbp & 5)  vc1_v4(dst + 4 * ls + 4, ls, pq);
        if (top_cbp & 10) vc1_v4(dst + 4 * ls, ls, pq);
    }
}

// Vertical edges (H loop filter): filters the right edge of block `i`.
static inline void vc1_p_h_filter(VC1Context* v, int mbx, int mby, int i,
                                  uint8_t* dY, uint8_t* dU, uint8_t* dV) {
    int pq = v->pq, mw = v->mb_width, mp = mbx + mby * mw;
    int ls = (i > 3) ? v->cStride : v->yStride;
    uint8_t* dst = (i > 3) ? (i == 4 ? dU : dV) : dY + (i & 2) * 4 * v->yStride + (i & 1) * 8;
    int right_edge = (mbx == v->mb_width - 1);
    uint32_t left_cbp = v->fr_cbp[mp] >> (i * 4), right_cbp;
    int left_is_intra = v->fr_isintra[mp] & (1 << i), right_is_intra;

    if (!right_edge || !(i & 5)) {
        int mp_r = mp + 1;     // MB to the right
        if (i > 3)          { right_is_intra = v->fr_isintra[mp_r] & (1 << i);       right_cbp = v->fr_cbp[mp_r] >> (i * 4); }
        else if (i & 1)     { right_is_intra = v->fr_isintra[mp_r] & (1 << (i - 1)); right_cbp = v->fr_cbp[mp_r] >> ((i - 1) * 4); }
        else                { right_is_intra = v->fr_isintra[mp]   & (1 << (i + 1)); right_cbp = v->fr_cbp[mp]   >> ((i + 1) * 4); }

        int diff;
        if (i > 3) {
            int16_t* a = v->fr_lumamv[mp]; int16_t* b = v->fr_lumamv[mp_r];
            diff = a[0] != b[0] || a[1] != b[1];
        } else {
            int xy = v->block_index[i];
            int16_t* a = v->mv_val + xy * 2; int16_t* b = v->mv_val + (xy + 1) * 2;
            diff = a[0] != b[0] || a[1] != b[1];
        }
        if (left_is_intra || right_is_intra || diff)
            vc1_h8(dst + 8, ls, pq);
        else {
            int idx = (left_cbp | (right_cbp >> 1)) & 5;
            if (idx & 1) vc1_h4(dst + 4 * ls + 8, ls, pq);
            if (idx & 4) vc1_h4(dst + 8, ls, pq);
        }
    }
    int tt = v->fr_ttblk[mp] >> (i * 4) & 0xf;
    if (tt == TT_4X4 || tt == TT_4X8) {
        if (left_cbp & 3)  vc1_h4(dst + 4 * ls + 4, ls, pq);
        if (left_cbp & 12) vc1_h4(dst + 4, ls, pq);
    }
}

static inline void vc1_p_deblock_frame(VC1Context* v) {
    for (int mby = 0; mby < v->mb_height; mby++)
        for (int mbx = 0; mbx < v->mb_width; mbx++) {
            v->mb_x = mbx; v->mb_y = mby; vc1_init_block_index(v);
            uint8_t* dY = v->planeY + (size_t)(mby * 16) * v->yStride + mbx * 16;
            uint8_t* dU = v->planeU + (size_t)(mby * 8) * v->cStride + mbx * 8;
            uint8_t* dV = v->planeV + (size_t)(mby * 8) * v->cStride + mbx * 8;
            for (int i = 0; i < 6; i++) vc1_p_v_filter(v, mbx, mby, i, dY, dU, dV);
        }
    for (int mby = 0; mby < v->mb_height; mby++)
        for (int mbx = 0; mbx < v->mb_width; mbx++) {
            v->mb_x = mbx; v->mb_y = mby; vc1_init_block_index(v);
            uint8_t* dY = v->planeY + (size_t)(mby * 16) * v->yStride + mbx * 16;
            uint8_t* dU = v->planeU + (size_t)(mby * 8) * v->cStride + mbx * 8;
            uint8_t* dV = v->planeV + (size_t)(mby * 8) * v->cStride + mbx * 8;
            for (int i = 0; i < 6; i++) vc1_p_h_filter(v, mbx, mby, i, dY, dU, dV);
        }
}

} // namespace vc1

#endif // VC1_DECODE_H
