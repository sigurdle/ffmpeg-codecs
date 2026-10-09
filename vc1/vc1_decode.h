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

    STATUS — Simple/Main (WMV3) progressive I, P, B and BI pictures:
      VERIFIED pixel-exact vs ffmpeg (every frame, Y/U/V max diff 0):
        - the committed Adobe Fresco timelapses (Main, loop filter, mixed 1MV/4MV P,
          1MV_HPEL_BILIN, VSTransform; tests/Vc1Decode.test.js, Vc1Avi.test.js);
        - FATE SMM0005 (B pictures: direct / interpolated / forward / backward, B intra,
          display reordering) and SMM0015 (MultiRes syntax with RESPIC 0, FASTUVMC with
          half-pel bilinear) — local clips, tests/Vc1Conformance.test.js.
      UNVERIFIED (transcribed from ffmpeg release/8.1, no sample exercises them):
        - overlap smoothing (OVERLAP with pq >= 9, I/BI and the intra blocks of P);
        - nonzero DQUANT: VOPDQUANT, per-MB MQUANT, dqscale DC/AC predictor rescaling;
        - intensity compensation (mv_mode INTENSITY_COMP, LUT applied to last_pic, also
          for the following B pictures' forward prediction);
        - range reduction (RANGEREDFRM: expanded intra output, halved reference);
        - FASTUVMC in B interpolated MC (ffmpeg rounds away from zero there);
        - res_rtm_flag 0 (the 4MV block-0 B predictor offset).
      UNSUPPORTED (decode returns an error code, never a silently wrong picture):
        - nonzero RESPIC (picture coded at reduced size; ffmpeg 8.1 parses RESPIC but
          never rescales) -> VC1_ERR_RESPIC;
        - IntraX8 I pictures (x8_type) -> VC1_ERR_X8.
      Advanced profile (WVC1): sequence / entry-point / progressive I headers and I
      decode only (vc1_decode_adv), not wired into the readers.

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
    int dquantfrm, dqprofile, dqsbedge, dqbilevel, altpq;
    int c_ac_table_index, y_ac_table_index, dc_table_index;

    // ── Advanced Profile (WVC1) — sequence + entry-point + frame header ──────────
    // Sequence header (decode_sequence_header_adv). res_rtm_flag reused from above.
    int level, chromaformat, postprocflag;
    int max_coded_width, max_coded_height;
    int broadcast, interlace, tfcntrflag, psf;
    int hrd_param_flag, hrd_num_leaky_buckets;
    // Entry-point header (ff_vc1_decode_entry_point). dquant/vstransform/overlap/
    // quantizer_mode/fastuvmc/extended_mv/loop_filter reused from the Simple/Main set.
    int broken_link, closed_entry, panscanflag, refdist_flag, extended_dmv;
    int range_mapy_flag, range_mapy, range_mapuv_flag, range_mapuv;
    // Frame header (ff_vc1_parse_frame_header_adv), progressive subset.
    int fcm, rptfrm, tff, rff, uvsamp, condover;
    int postproc, p_frame_skipped;
    int acpred_is_raw, acpred_imode, acpred_invert;
    int overflg_is_raw;
    uint8_t* acpred_plane;       // mb_stride * mb_height (per-MB AC-pred flag, AP I)
    uint8_t* over_flags_plane;   // mb_stride * mb_height (condover SELECT)
    int16_t* qscale_table;       // mb_stride * mb_height (per-MB mquant, signed; dqscale pred)

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

    // ── P/B-frame inter decode ──────────────────────────────────────────────────
    uint8_t *refY, *refU, *refV;        // last_pic: forward reference (MC source of P, B dir 0)
    uint8_t *nextY, *nextU, *nextV;     // next_pic: backward reference (B dir 1)
    int16_t *mv_base, *mv_val;          // cur_pic.motion_val[0]: 2 int16 per b8 block
    int16_t *mv1_base, *mv1_val;        // cur_pic.motion_val[1] of a B picture (backward MVs)
    int16_t *nmv_base, *nmv_val;        // motion_val[1] of the newest I/P = B direct-mode source
    int16_t *mvb;                       // cur_pic.motion_val[1] of the picture being decoded
    uint8_t *mbtype_base, *mbtype;      // per-b8-block is_intra (get_chroma_mv, overlap, put)
    int16_t cur_mv[2][6][2];            // s->mv[dir][n] for the current MB
    int quarter_sample, mspel;          // from mv_mode
    const mp4::VLCElem* cbpcy_vlc;      // = g_cbpcy_p_vlc[cbptab]
    int a_avail, c_avail;
    // per-MB frame arrays for the P deblock (full frame, not ffmpeg's rolling buffer)
    uint32_t *fr_cbp;                   // block_cbp per MB (mb_width*mb_height)
    int      *fr_ttblk;                 // block_tt  per MB
    uint8_t  *fr_isintra;              // block_intra bitmask per MB
    int16_t (*fr_lumamv)[2];            // chroma luma_mv per MB (deblock chroma)

    // ── B / BI pictures (Simple/Main) ───────────────────────────────────────────
    int bfraction, bfraction_lut_index, bi_type, respic;
    int dmb_is_raw, direct_imode, direct_invert;
    uint8_t* direct_mb_plane;            // mb_stride * mb_height

    // ── Intensity compensation (rotate_luts / INIT_LUT, progressive subset) ─────
    // Only the LUT applied to last_pic matters for progressive Simple/Main: next_pic's
    // LUT is reset to identity by every I/P header, and B/BI never set one.
    int last_use_ic;
    uint8_t last_luty[256], last_lutuv[256];

    // ── Picture store: three I/P pictures + one B/BI output picture ────────────
    uint8_t *picY[3], *picU[3], *picV[3];
    uint8_t *bpY, *bpU, *bpV;
    int idx_last, idx_next, nrefs;       // nrefs = I/P pictures decoded since open (capped at 2)
    int ref_broken;                      // an unsupported I/P was skipped: refs are stale until the next I

    // ── Rolling block buffer (v->block[n_allocated_blks][6][64]) for the delayed
    // overlap-smoothing + put of intra blocks, exactly as vc1_block.c schedules it.
    int16_t (*blk)[6][64];
    int n_blks, cur_blk_idx, left_blk_idx, topleft_blk_idx, top_blk_idx;
    uint8_t* dest[3];                    // s->dest[]: the current MB in planeY/U/V
};

// Return codes beyond VC1_OK / VC1_ERR (vc1_decode_frame).
enum {
    VC1_NO_PICTURE = 1,    // nothing new to show (B without references, payload too short)
    VC1_ERR_RESPIC = -2,   // multires frame coded at reduced size (nonzero RESPIC)
    VC1_ERR_X8     = -3,   // WMV3 IntraX8 I-frame (x8_type)
};

enum { BMV_TYPE_BACKWARD = 0, BMV_TYPE_FORWARD, BMV_TYPE_INTERPOLATED, BMV_TYPE_DIRECT };
enum { B_FRACTION_DEN = 256 };

// ff_vc1_bfraction_lut (vc1data.c, B_FRACTION_DEN 256).
static const int16_t vc1_bfraction_lut[23] = {
    128 /*1/2*/,  85 /*1/3*/, 170 /*2/3*/,  64 /*1/4*/,
    192 /*3/4*/,  51 /*1/5*/, 102 /*2/5*/,
    153 /*3/5*/, 204 /*4/5*/,  43 /*1/6*/, 215 /*5/6*/,
     37 /*1/7*/,  74 /*2/7*/, 111 /*3/7*/, 148 /*4/7*/,
    185 /*5/7*/, 222 /*6/7*/,  32 /*1/8*/,  96 /*3/8*/,
    160 /*5/8*/, 224 /*7/8*/,
     -1 /*inv.*/,  0 /*BI fm*/
};

// mapping table for internal block representation (vc1_block.c block_map)
static const int vc1_block_map[6] = { 0, 2, 1, 3, 4, 5 };

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

// FrameCodingMode (vc1.h) and CondOver possibilities.
enum { PROGRESSIVE = 0, ILACE_FRAME = 1, ILACE_FIELD = 2 };
enum { CONDOVER_NONE = 0, CONDOVER_ALL = 1, CONDOVER_SELECT = 2 };

// decode_sequence_header_adv (vc1.c:385) — Advanced Profile (WVC1) sequence header.
// Called after the profile==ADVANCED discriminator; consumes all display-info / HRD
// bits so the entry-point BDU that follows is bit-aligned. Progressive 4:2:0 only.
static inline int vc1_decode_sequence_header_adv(VC1Context* v, mp4::GetBits* gb) {
    v->res_rtm_flag  = 1;
    v->level         = (int)mp4::get_bits(gb, 3);
    v->chromaformat  = (int)mp4::get_bits(gb, 2);
    if (v->chromaformat != 1) return VC1_ERR;              // only 4:2:0 supported
    v->frmrtq_postproc = (int)mp4::get_bits(gb, 3);
    v->bitrtq_postproc = (int)mp4::get_bits(gb, 5);
    v->postprocflag    = (int)mp4::get_bits1(gb);
    v->max_coded_width  = ((int)mp4::get_bits(gb, 12) + 1) << 1;
    v->max_coded_height = ((int)mp4::get_bits(gb, 12) + 1) << 1;
    v->broadcast   = (int)mp4::get_bits1(gb);
    v->interlace   = (int)mp4::get_bits1(gb);
    v->tfcntrflag  = (int)mp4::get_bits1(gb);
    v->finterpflag = (int)mp4::get_bits1(gb);
    mp4::skip_bits(gb, 1);                                 // reserved
    v->psf = (int)mp4::get_bits1(gb);
    if (v->psf) return VC1_ERR;                            // PsF — not supported
    v->max_b_frames = 7;
    if (mp4::get_bits1(gb)) {                              // Display Info (decode-irrelevant)
        int ar = 0;
        mp4::skip_bits(gb, 14);                            // display width  - 1
        mp4::skip_bits(gb, 14);                            // display height - 1
        if (mp4::get_bits1(gb)) ar = (int)mp4::get_bits(gb, 4);
        if (ar == 15) mp4::skip_bits(gb, 16);             // custom SAR (8+8); table cases read nothing
        if (mp4::get_bits1(gb)) {                          // framerate
            if (mp4::get_bits1(gb)) mp4::skip_bits(gb, 16);           // fr num (32nds)
            else { mp4::skip_bits(gb, 8); mp4::skip_bits(gb, 4); }    // fr nr + dr
        }
        if (mp4::get_bits1(gb)) mp4::skip_bits(gb, 24);   // color_prim/transfer/matrix (8+8+8)
    }
    v->hrd_param_flag = (int)mp4::get_bits1(gb);
    if (v->hrd_param_flag) {
        v->hrd_num_leaky_buckets = (int)mp4::get_bits(gb, 5);
        mp4::skip_bits(gb, 8);                             // bitrate + buffer size exponents (4+4)
        for (int i = 0; i < v->hrd_num_leaky_buckets; i++)
            mp4::skip_bits(gb, 32);                        // hrd_rate[n] + hrd_buffer[n] (16+16)
    }
    return VC1_OK;
}

// ff_vc1_decode_entry_point (vc1.c:493) — Advanced Profile entry-point header.
// Sets the runtime frame dimensions (coded-size override, else max_coded_*).
static inline int vc1_decode_entry_point(VC1Context* v, mp4::GetBits* gb) {
    v->broken_link    = (int)mp4::get_bits1(gb);
    v->closed_entry   = (int)mp4::get_bits1(gb);
    v->panscanflag    = (int)mp4::get_bits1(gb);
    v->refdist_flag   = (int)mp4::get_bits1(gb);
    v->loop_filter    = (int)mp4::get_bits1(gb);
    v->fastuvmc       = (int)mp4::get_bits1(gb);
    v->extended_mv    = (int)mp4::get_bits1(gb);
    v->dquant         = (int)mp4::get_bits(gb, 2);
    v->vstransform    = (int)mp4::get_bits1(gb);
    v->overlap        = (int)mp4::get_bits1(gb);
    v->quantizer_mode = (int)mp4::get_bits(gb, 2);
    if (v->hrd_param_flag)
        for (int i = 0; i < v->hrd_num_leaky_buckets; i++)
            mp4::skip_bits(gb, 8);                         // hrd_full[n]
    int w, h;
    if (mp4::get_bits1(gb)) {                              // coded-size override
        w = ((int)mp4::get_bits(gb, 12) + 1) << 1;
        h = ((int)mp4::get_bits(gb, 12) + 1) << 1;
    } else {
        w = v->max_coded_width;
        h = v->max_coded_height;
    }
    v->width = w; v->height = h;
    if (v->extended_mv)
        v->extended_dmv = (int)mp4::get_bits1(gb);
    if ((v->range_mapy_flag = (int)mp4::get_bits1(gb)))
        v->range_mapy = (int)mp4::get_bits(gb, 3);         // parsed, NOT applied (matches ffmpeg)
    if ((v->range_mapuv_flag = (int)mp4::get_bits1(gb)))
        v->range_mapuv = (int)mp4::get_bits(gb, 3);
    return VC1_OK;
}

// ── Advanced Profile bitstream framing (BDUs, start codes, EBDU) ─────────────────
// VC-1 AP wraps sequence/entry-point/frame data in BDUs delimited by 4-byte start
// codes (00 00 01 xx). find_next_marker (vc1_common.h) returns the start of the next
// 00 00 01 prefix at or after src (the marker value is 0x000001xx, xx = BDU type).
static inline const uint8_t* vc1_find_next_marker(const uint8_t* src, const uint8_t* end) {
    for (; src + 4 <= end; src++)
        if (src[0] == 0 && src[1] == 0 && src[2] == 1) return src;
    return end;
}

// vc1_unescape_buffer (vc1_common.h) — strip emulation-prevention bytes: a 0x03 that
// follows 00 00 and precedes a byte < 4 was inserted to avoid a false start code.
static inline int vc1_unescape_buffer(const uint8_t* src, int size, uint8_t* dst) {
    int dsize = 0, i;
    if (size < 4) {
        for (dsize = 0; dsize < size; dsize++) *dst++ = *src++;
        return size;
    }
    for (i = 0; i < size; i++, src++) {
        if (src[0] == 3 && i >= 2 && !src[-1] && !src[-2] && i < size - 1 && src[1] < 4) {
            dst[dsize++] = src[1];
            src++; i++;
        } else {
            dst[dsize++] = *src;
        }
    }
    return dsize;
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

// INIT_LUT (vc1.c) — intensity-compensation lookup tables. `chain` applies the new
// mapping on top of the table's current contents.
static inline void vc1_init_lut(int lumscale, int lumshift, uint8_t* luty, uint8_t* lutuv, int chain) {
    int scale, shift, i;
    if (!lumscale) {
        scale = -64;
        shift = (255 - lumshift * 2) * 64;
        if (lumshift > 31)
            shift += 128 << 6;
    } else {
        scale = lumscale + 32;
        if (lumshift > 31)
            shift = (lumshift - 64) * 64;
        else
            shift = lumshift << 6;
    }
    for (i = 0; i < 256; i++) {
        int iy = chain ? luty[i]  : i;
        int iu = chain ? lutuv[i] : i;
        int y = (scale * iy + shift + 32) >> 6;
        int u = (scale * (iu - 128) + 128 * 64 + 32) >> 6;
        luty[i]  = (uint8_t)(y < 0 ? 0 : (y > 255 ? 255 : y));
        lutuv[i] = (uint8_t)(u < 0 ? 0 : (u > 255 ? 255 : u));
    }
}

// read_bfraction (vc1.c).
static inline int vc1_read_bfraction(VC1Context* v, mp4::GetBits* gb) {
    int bfraction_lut_index = (int)mp4::get_bits(gb, 3);
    if (bfraction_lut_index == 7)
        bfraction_lut_index = 7 + (int)mp4::get_bits(gb, 4);
    if (bfraction_lut_index == 21) return VC1_ERR;     // bfraction invalid
    v->bfraction_lut_index = bfraction_lut_index;
    v->bfraction           = vc1_bfraction_lut[bfraction_lut_index];
    return VC1_OK;
}

// enum DQProfile (vc1.h).
enum { DQPROFILE_FOUR_EDGES = 0, DQPROFILE_DOUBLE_EDGES, DQPROFILE_SINGLE_EDGE, DQPROFILE_ALL_MBS };

// vop_dquant_decoding (vc1.c:228) — per-frame differential-quant parameters.
static inline void vc1_vop_dquant_decoding(VC1Context* v, mp4::GetBits* gb) {
    int pqdiff;
    if (v->dquant != 2) {
        v->dquantfrm = (int)mp4::get_bits1(gb);
        if (!v->dquantfrm) return;
        v->dqprofile = (int)mp4::get_bits(gb, 2);
        switch (v->dqprofile) {
        case DQPROFILE_SINGLE_EDGE:
        case DQPROFILE_DOUBLE_EDGES: v->dqsbedge = (int)mp4::get_bits(gb, 2); break;
        case DQPROFILE_ALL_MBS:
            v->dqbilevel = (int)mp4::get_bits1(gb);
            if (!v->dqbilevel) { v->halfpq = 0; return; }
            break;
        default: break;
        }
    }
    pqdiff = (int)mp4::get_bits(gb, 3);
    v->altpq = (pqdiff == 7) ? (int)mp4::get_bits(gb, 5) : v->pq + pqdiff + 1;
}

// ── Picture (frame) header — ff_vc1_parse_frame_header, Simple/Main (I/P/B/BI) ──
// A BI picture is reported as pict_type B with bi_type = 1 (as ffmpeg does).
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

    v->bi_type = 0;
    if (v->pict_type == PICT_B) {
        if (vc1_read_bfraction(v, gb) != VC1_OK) return VC1_ERR;
        if (v->bfraction == 0) v->pict_type = PICT_BI;
    }
    if (v->pict_type == PICT_I || v->pict_type == PICT_BI)
        mp4::skip_bits(gb, 7);                   // buffer fullness

    // RND
    if (v->pict_type == PICT_I || v->pict_type == PICT_BI) v->rnd = 1;
    if (v->pict_type == PICT_P) v->rnd ^= 1;

    if (mp4::get_bits_left(gb) < 5) return VC1_NO_PICTURE;

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
    if (v->multires && v->pict_type != PICT_B)
        v->respic = (int)mp4::get_bits(gb, 2);
    v->x8_type = (v->res_x8 && (v->pict_type == PICT_I || v->pict_type == PICT_BI))
               ? (int)mp4::get_bits1(gb) : 0;

    // rotate_luts: an I/P header makes last_pic's LUT the (identity) old next_pic LUT;
    // a B/BI header uses its own aux LUT and leaves last/next alone.
    if (v->pict_type != PICT_B && v->pict_type != PICT_BI) {
        v->last_use_ic = 0;
        vc1_init_lut(32, 0, v->last_luty, v->last_lutuv, 0);
    }

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
            v->lumshift = (int)mp4::get_bits(gb, 6);
            v->last_use_ic = 1;                          // fill lookup tables (no bits)
            vc1_init_lut(v->lumscale, v->lumshift, v->last_luty, v->last_lutuv, 1);
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

        if (mp4::get_bits_left(gb) < 4) return VC1_ERR;

        v->mv_table_index = (int)mp4::get_bits(gb, 2);
        v->cbptab         = (int)mp4::get_bits(gb, 2);

        if (v->dquant) vc1_vop_dquant_decoding(v, gb);

        if (v->vstransform) {
            v->ttmbf = (int)mp4::get_bits1(gb);
            v->ttfrm = v->ttmbf ? vc1_ttfrm_to_tt[mp4::get_bits(gb, 2)] : 0;
        } else {
            v->ttmbf = 1; v->ttfrm = TT_8X8;
        }
    } else if (v->pict_type == PICT_B) {
        int status;
        v->tt_index = (v->pq > 4) + (v->pq > 12);

        v->mv_mode        = mp4::get_bits1(gb) ? MV_PMODE_1MV : MV_PMODE_1MV_HPEL_BILIN;
        v->quarter_sample = (v->mv_mode == MV_PMODE_1MV);
        v->mspel          = v->quarter_sample;

        status = vc1_bitplane_decoding(v->direct_mb_plane, &v->dmb_is_raw, v, gb);
        if (status < 0) return VC1_ERR;
        v->direct_imode = status >> 1; v->direct_invert = status & 1;
        status = vc1_bitplane_decoding(v->mbskip_table, &v->skip_is_raw, v, gb);
        if (status < 0) return VC1_ERR;
        v->skip_imode = status >> 1; v->skip_invert = status & 1;

        v->mv_table_index = (int)mp4::get_bits(gb, 2);
        v->cbptab         = (int)mp4::get_bits(gb, 2);

        if (v->dquant) vc1_vop_dquant_decoding(v, gb);

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

    if (v->pict_type == PICT_BI) {
        v->pict_type = PICT_B;
        v->bi_type   = 1;
    }
    return VC1_OK;
}

// GET_MQUANT (vc1_block.c:186) — per-MB quantizer for the current (mb_x,mb_y). Reads
// bits only when dquantfrm && dqprofile==ALL_MBS. Sign of the return is significant
// (mquant<0 drops the halfpq term in the block dequant). Starts from pq.
static inline int vc1_get_mquant(VC1Context* v, mp4::GetBits* gb) {
    int mquant = v->pq;
    if (v->dquantfrm) {
        int edges = 0;
        if (v->dqprofile == DQPROFILE_ALL_MBS) {
            if (v->dqbilevel) {
                mquant = mp4::get_bits1(gb) ? -v->altpq : v->pq;
            } else {
                int mqdiff = (int)mp4::get_bits(gb, 3);
                mquant = (mqdiff != 7) ? -v->pq - mqdiff : -(int)mp4::get_bits(gb, 5);
            }
        }
        if (v->dqprofile == DQPROFILE_SINGLE_EDGE)  edges = 1 << v->dqsbedge;
        else if (v->dqprofile == DQPROFILE_DOUBLE_EDGES) edges = (3 << v->dqsbedge) % 15;
        else if (v->dqprofile == DQPROFILE_FOUR_EDGES)   edges = 15;
        if ((edges & 1) && !v->mb_x)                     mquant = -v->altpq;
        if ((edges & 2) && !v->mb_y)                     mquant = -v->altpq;
        if ((edges & 4) && v->mb_x == v->mb_width - 1)   mquant = -v->altpq;
        if ((edges & 8) && v->mb_y == v->mb_height - 1)  mquant = -v->altpq;
        if (!mquant || mquant > 31 || mquant < -31)      mquant = 1;
    }
    return mquant;
}

// ff_vc1_parse_frame_header_adv (vc1.c:836) — Advanced Profile picture header,
// PROGRESSIVE I/BI subset (interlace frame/field paths and P/B are elsewhere).
static inline int vc1_parse_frame_header_adv(VC1Context* v, mp4::GetBits* gb) {
    int pqindex;
    vc1_init_vlcs();

    v->p_frame_skipped = 0;
    int fcm = PROGRESSIVE;
    if (v->interlace) {
        fcm = vc1_decode012(gb);
        if (fcm == ILACE_FIELD) return VC1_ERR;            // field pictures — not yet
    }
    v->fcm = fcm;

    switch (vc1_get_unary(gb, 0, 4)) {                     // picture type (progressive)
    case 0: v->pict_type = PICT_P; break;
    case 1: v->pict_type = PICT_B; break;
    case 2: v->pict_type = PICT_I; break;
    case 3: v->pict_type = PICT_BI; break;
    default: v->pict_type = PICT_P; v->p_frame_skipped = 1; break;   // 4 = skipped
    }
    if (v->pict_type == PICT_B || v->pict_type == PICT_BI)
        return VC1_ERR;                                    // B/BI — not yet supported

    if (v->tfcntrflag) mp4::skip_bits(gb, 8);
    if (v->broadcast) {
        if (!v->interlace || v->psf) v->rptfrm = (int)mp4::get_bits(gb, 2);
        else { v->tff = (int)mp4::get_bits1(gb); v->rff = (int)mp4::get_bits1(gb); }
    } else {
        v->tff = 1;
    }
    if (v->panscanflag) return VC1_ERR;                    // pan-scan — not supported
    if (v->p_frame_skipped) return VC1_OK;

    v->rnd = (int)mp4::get_bits1(gb);
    if (v->interlace) v->uvsamp = (int)mp4::get_bits1(gb);
    if (v->fcm == PROGRESSIVE && v->finterpflag)
        v->interpfrm = (int)mp4::get_bits1(gb);

    // parse_common_info
    pqindex = (int)mp4::get_bits(gb, 5);
    if (!pqindex) return VC1_ERR;
    v->pq = (v->quantizer_mode == QUANT_FRAME_IMPLICIT) ? vc1_pquant_table[0][pqindex]
                                                        : vc1_pquant_table[1][pqindex];
    v->pqindex = pqindex;
    v->halfpq  = (pqindex < 9) ? (int)mp4::get_bits1(gb) : 0;
    switch (v->quantizer_mode) {
    case QUANT_FRAME_IMPLICIT: v->pquantizer = pqindex < 9;             break;
    case QUANT_NON_UNIFORM:    v->pquantizer = 0;                       break;
    case QUANT_FRAME_EXPLICIT: v->pquantizer = (int)mp4::get_bits1(gb); break;
    default:                   v->pquantizer = 1;                       break;
    }
    v->dquantfrm = 0;
    if (v->postprocflag) v->postproc = (int)mp4::get_bits(gb, 2);

    // I / BI (progressive frame): ACPRED bitplane, then CONDOVER (overlap only).
    int status = vc1_bitplane_decoding(v->acpred_plane, &v->acpred_is_raw, v, gb);
    if (status < 0) return VC1_ERR;
    v->acpred_imode = status >> 1; v->acpred_invert = status & 1;
    v->condover = CONDOVER_NONE;
    if (v->overlap && v->pq <= 8) {
        v->condover = vc1_decode012(gb);
        if (v->condover == CONDOVER_SELECT) {
            status = vc1_bitplane_decoding(v->over_flags_plane, &v->overflg_is_raw, v, gb);
            if (status < 0) return VC1_ERR;
        }
    }

    // AC syntax (TRANSACFRM / TRANSACFRM2) — I/BI reads both Y and C indices.
    v->c_ac_table_index = vc1_decode012(gb);
    if (v->pict_type == PICT_I || v->pict_type == PICT_BI)
        v->y_ac_table_index = vc1_decode012(gb);
    // DC syntax (TRANSDCTAB) + VOPDQUANT (I/BI when entry-point dquant set).
    v->dc_table_index = (int)mp4::get_bits1(gb);
    if ((v->pict_type == PICT_I || v->pict_type == PICT_BI) && v->dquant)
        vc1_vop_dquant_decoding(v, gb);
    return VC1_OK;
}

// ── I-frame block decode (Simple/Main) ──────────────────────────────────────────
// Grid geometry mirrors the msmpeg4 port (ff_mpv_common_init): dc/ac/coded grids
// offset by lead+dc_off so edge OOB neighbor reads land in a zeroed guard.
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

// put_signed_pixels_clamped (idsp.c) — reconstruct SIGNED residual blocks (+128).
// Advanced Profile intra ALWAYS uses this (vc1_put_blocks_clamped(v,1)); Simple/Main
// I uses the unsigned put (v,0) because its DC predictor carries the +128*8 default.
static inline void vc1_put_signed_pixels_clamped(const int16_t* block, uint8_t* dst, int stride) {
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            int x = block[j] + 128;
            dst[j] = (uint8_t)(x < 0 ? 0 : (x > 255 ? 255 : x));
        }
        block += 8; dst += stride;
    }
}

// put_mb: `sgn` selects put_signed_pixels_clamped (+128, Advanced) vs plain (Simple/Main).
static inline void vc1_put_mb(VC1Context* v, int16_t block[6][64], int sgn = 0) {
    int ys = v->yStride, cs = v->cStride;
    uint8_t* yb = v->planeY + (size_t)(v->mb_y * 16) * ys + v->mb_x * 16;
    uint8_t* ub = v->planeU + (size_t)(v->mb_y * 8) * cs + v->mb_x * 8;
    uint8_t* vb = v->planeV + (size_t)(v->mb_y * 8) * cs + v->mb_x * 8;
    if (sgn) {
        for (int i = 0; i < 4; i++)
            vc1_put_signed_pixels_clamped(block[i], yb + ((i & 2) >> 1) * 8 * ys + (i & 1) * 8, ys);
        vc1_put_signed_pixels_clamped(block[4], ub, cs);
        vc1_put_signed_pixels_clamped(block[5], vb, cs);
    } else {
        for (int i = 0; i < 4; i++)
            vc1_put_pixels_clamped(block[i], yb + ((i & 2) >> 1) * 8 * ys + (i & 1) * 8, ys);
        vc1_put_pixels_clamped(block[4], ub, cs);
        vc1_put_pixels_clamped(block[5], vb, cs);
    }
}

// ── In-loop deblocking filter (vc1dsp.c / vc1_loopfilter.c), Simple/Main I-frame ──
// vc1_filter_line: filters one line straddling an edge (stride = distance between the
// two pixels across the edge). Returns 1 if the neighboring 3 pairs should filter too.
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

// ── Overlap smoothing (vc1dsp.c vc1_v_s_overlap_c / vc1_h_s_overlap_c) ────────────
// Operates on the reconstructed (pre-put) int16 blocks across 8x8 block edges.
static inline void vc1_v_s_overlap(int16_t* top, int16_t* bottom) {
    int i, a, b, c, d, d1, d2;
    int rnd1 = 4, rnd2 = 3;
    for (i = 0; i < 8; i++) {
        a  = top[48];
        b  = top[56];
        c  = bottom[0];
        d  = bottom[8];
        d1 = a - d;
        d2 = a - d + b - c;

        top[48]   = (int16_t)(((a * 8) - d1 + rnd1) >> 3);
        top[56]   = (int16_t)(((b * 8) - d2 + rnd2) >> 3);
        bottom[0] = (int16_t)(((c * 8) + d2 + rnd1) >> 3);
        bottom[8] = (int16_t)(((d * 8) + d1 + rnd2) >> 3);

        bottom++;
        top++;
        rnd2 = 7 - rnd2;
        rnd1 = 7 - rnd1;
    }
}

static inline void vc1_h_s_overlap(int16_t* left, int16_t* right, int left_stride, int right_stride, int flags) {
    int i, a, b, c, d, d1, d2;
    int rnd1 = flags & 2 ? 3 : 4;
    int rnd2 = 7 - rnd1;
    for (i = 0; i < 8; i++) {
        a  = left[6];
        b  = left[7];
        c  = right[0];
        d  = right[1];
        d1 = a - d;
        d2 = a - d + b - c;

        left[6]  = (int16_t)(((a * 8) - d1 + rnd1) >> 3);
        left[7]  = (int16_t)(((b * 8) - d2 + rnd2) >> 3);
        right[0] = (int16_t)(((c * 8) + d2 + rnd1) >> 3);
        right[1] = (int16_t)(((d * 8) + d1 + rnd2) >> 3);

        right += right_stride;
        left  += left_stride;
        if (flags & 1) {
            rnd2 = 7 - rnd2;
            rnd1 = 7 - rnd1;
        }
    }
}

// vc1_h_overlap_filter / vc1_v_overlap_filter (vc1_loopfilter.c), progressive pictures
// (no field transform: left_fieldtx = right_fieldtx = 0). Blocks are indexed in the
// block_map storage order {0 TL, 1 BL, 2 TR, 3 BR, 4 U, 5 V}.
static inline void vc1_h_overlap_filter(int16_t (*left_block)[64], int16_t (*right_block)[64], int block_num) {
    switch (block_num) {
    case 0: vc1_h_s_overlap(left_block[2], right_block[0], 8, 8, 1); break;
    case 1: vc1_h_s_overlap(right_block[0], right_block[2], 8, 8, 1); break;
    case 2: vc1_h_s_overlap(left_block[3], right_block[1], 8, 8, 1); break;
    case 3: vc1_h_s_overlap(right_block[1], right_block[3], 8, 8, 1); break;
    case 4:
    case 5: vc1_h_s_overlap(left_block[block_num], right_block[block_num], 8, 8, 1); break;
    }
}

static inline void vc1_v_overlap_filter(int16_t (*top_block)[64], int16_t (*bottom_block)[64], int block_num) {
    switch (block_num) {
    case 0: vc1_v_s_overlap(top_block[1], bottom_block[0]); break;
    case 1: vc1_v_s_overlap(top_block[3], bottom_block[2]); break;
    case 2: vc1_v_s_overlap(bottom_block[0], bottom_block[1]); break;
    case 3: vc1_v_s_overlap(bottom_block[2], bottom_block[3]); break;
    case 4:
    case 5: vc1_v_s_overlap(top_block[block_num], bottom_block[block_num]); break;
    }
}

// The rolling buffer slots (v->block[...]); left is -1 before the first MB of a row.
static inline int16_t (*vc1_blk(VC1Context* v, int idx))[64] {
    return idx >= 0 ? v->blk[idx] : (int16_t (*)[64])0;
}

// ff_vc1_i_overlap_filter (Simple/Main: called only when overlap && pq >= 9, which is
// the whole of its per-edge condition outside Advanced profile). Within a MB the H
// overlap runs before the V; the V trails by one MB column (it runs on the left MB).
static inline void vc1_i_overlap_filter(VC1Context* v) {
    int16_t (*topleft_blk)[64] = vc1_blk(v, v->topleft_blk_idx);
    int16_t (*top_blk)[64]     = vc1_blk(v, v->top_blk_idx);
    int16_t (*left_blk)[64]    = vc1_blk(v, v->left_blk_idx);
    int16_t (*cur_blk)[64]     = vc1_blk(v, v->cur_blk_idx);
    int i;

    for (i = 0; i < 6; i++) {
        if (v->mb_x == 0 && (i & 5) != 1)
            continue;
        vc1_h_overlap_filter(v->mb_x ? left_blk : cur_blk, cur_blk, i);
    }
    for (i = 0; i < 6; i++) {
        if (v->first_slice_line && !(i & 2))
            continue;
        if (v->mb_x)
            vc1_v_overlap_filter(v->first_slice_line ? left_blk : topleft_blk, left_blk, i);
        if (v->mb_x == v->mb_width - 1)
            vc1_v_overlap_filter(v->first_slice_line ? cur_blk : top_blk, cur_blk, i);
    }
}

// ff_vc1_p_overlap_filter — only across edges where both blocks are intra.
static inline void vc1_p_overlap_filter(VC1Context* v) {
    int16_t (*topleft_blk)[64] = vc1_blk(v, v->topleft_blk_idx);
    int16_t (*top_blk)[64]     = vc1_blk(v, v->top_blk_idx);
    int16_t (*left_blk)[64]    = vc1_blk(v, v->left_blk_idx);
    int16_t (*cur_blk)[64]     = vc1_blk(v, v->cur_blk_idx);
    int i;

    for (i = 0; i < 6; i++) {
        if (v->mb_x == 0 && (i & 5) != 1)
            continue;
        if (v->mbtype[v->block_index[i]] && v->mbtype[v->block_index[i] - 1])
            vc1_h_overlap_filter(v->mb_x ? left_blk : cur_blk, cur_blk, i);
    }
    for (i = 0; i < 6; i++) {
        if (v->first_slice_line && !(i & 2))
            continue;
        if (v->mb_x && v->mbtype[v->block_index[i] - 2 + (i > 3)] &&
            v->mbtype[v->block_index[i] - v->block_wrap[i] - 2 + (i > 3)])
            vc1_v_overlap_filter(v->first_slice_line ? left_blk : topleft_blk, left_blk, i);
        if (v->mb_x == v->mb_width - 1)
            if (v->mbtype[v->block_index[i]] &&
                v->mbtype[v->block_index[i] - v->block_wrap[i]])
                vc1_v_overlap_filter(v->first_slice_line ? cur_blk : top_blk, cur_blk, i);
    }
}

static inline void vc1_put_block(int put_signed, const int16_t* block, uint8_t* dst, int stride) {
    if (put_signed) vc1_put_signed_pixels_clamped(block, dst, stride);
    else            vc1_put_pixels_clamped(block, dst, stride);
}

// vc1_put_blocks_clamped (vc1_block.c), progressive. The put runs one MB row and one MB
// column behind the decode (overlap smoothing must finish first); only blocks whose
// mb_type is intra are put — inter residual was already added to the picture.
static inline void vc1_put_blocks_clamped(VC1Context* v, int put_signed) {
    const int ls = v->yStride, uvls = v->cStride;
    uint8_t* dest;
    int i;

    if (!v->first_slice_line) {
        if (v->mb_x) {
            for (i = 0; i < 6; i++) {
                if (i > 3 ? v->mbtype[v->block_index[i] - v->block_wrap[i] - 1] :
                            v->mbtype[v->block_index[i] - 2 * v->block_wrap[i] - 2]) {
                    dest = v->dest[0] + ((i & 2) - 4) * 4 * ls + ((i & 1) - 2) * 8;
                    vc1_put_block(put_signed, v->blk[v->topleft_blk_idx][vc1_block_map[i]],
                                  i > 3 ? v->dest[i - 3] - 8 * uvls - 8 : dest,
                                  i > 3 ? uvls : ls);
                }
            }
        }
        if (v->mb_x == v->mb_width - 1) {
            for (i = 0; i < 6; i++) {
                if (i > 3 ? v->mbtype[v->block_index[i] - v->block_wrap[i]] :
                            v->mbtype[v->block_index[i] - 2 * v->block_wrap[i]]) {
                    dest = v->dest[0] + ((i & 2) - 4) * 4 * ls + (i & 1) * 8;
                    vc1_put_block(put_signed, v->blk[v->top_blk_idx][vc1_block_map[i]],
                                  i > 3 ? v->dest[i - 3] - 8 * uvls : dest,
                                  i > 3 ? uvls : ls);
                }
            }
        }
    }
    if (v->mb_y == v->mb_height - 1) {
        if (v->mb_x) {
            for (i = 0; i < 6; i++) {
                if (i > 3 ? v->mbtype[v->block_index[i] - 1] :
                            v->mbtype[v->block_index[i] - 2]) {
                    dest = v->dest[0] + (i & 2) * 4 * ls + ((i & 1) - 2) * 8;
                    vc1_put_block(put_signed, v->blk[v->left_blk_idx][vc1_block_map[i]],
                                  i > 3 ? v->dest[i - 3] - 8 : dest,
                                  i > 3 ? uvls : ls);
                }
            }
        }
        if (v->mb_x == v->mb_width - 1) {
            for (i = 0; i < 6; i++) {
                if (v->mbtype[v->block_index[i]]) {
                    dest = v->dest[0] + (i & 2) * 4 * ls + (i & 1) * 8;
                    vc1_put_block(put_signed, v->blk[v->cur_blk_idx][vc1_block_map[i]],
                                  i > 3 ? v->dest[i - 3] : dest,
                                  i > 3 ? uvls : ls);
                }
            }
        }
    }
}

static inline void vc1_set_dest(VC1Context* v) {
    v->dest[0] = v->planeY + (size_t)(v->mb_y * 16) * v->yStride + v->mb_x * 16;
    v->dest[1] = v->planeU + (size_t)(v->mb_y * 8) * v->cStride + v->mb_x * 8;
    v->dest[2] = v->planeV + (size_t)(v->mb_y * 8) * v->cStride + v->mb_x * 8;
}

// ff_vc1_decode_blocks: cur = 0, left = -1, topleft = 1, top = 2.
static inline void vc1_reset_blk_idx(VC1Context* v) {
    v->cur_blk_idx = 0; v->left_blk_idx = -1; v->topleft_blk_idx = 1; v->top_blk_idx = 2;
}

// inc_blk_idx (n_allocated_blks = mb_width + 2; the I-frame loop's `% (end_mb_x + 2)`
// is the same wrap).
static inline void vc1_inc_blk_idx(VC1Context* v) {
    if (++v->topleft_blk_idx >= v->n_blks) v->topleft_blk_idx = 0;
    if (++v->top_blk_idx     >= v->n_blks) v->top_blk_idx     = 0;
    if (++v->left_blk_idx    >= v->n_blks) v->left_blk_idx    = 0;
    if (++v->cur_blk_idx     >= v->n_blks) v->cur_blk_idx     = 0;
}

// vc1_decode_i_blocks (vc1_block.c) — Simple/Main I and BI pictures. Overlap smoothing
// (overlap && pq >= 9) switches the reconstruction to the signed put (the DC predictor
// edge default is 0 then, not 1024/scale); RANGEREDFRM expands the reconstructed
// samples (UNVERIFIED: no range-reduced sample). The in-loop deblock runs as two
// full-frame passes after every block has been put (see vc1_i_deblock_frame).
static inline int vc1_decode_i_frame(VC1Context* v, mp4::GetBits* gb) {
    int k, j;
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
    memset(v->mbtype_base, 0, (size_t)v->grid_total);
    vc1_reset_blk_idx(v);

    for (v->mb_y = 0; v->mb_y < v->mb_height; v->mb_y++) {
        v->first_slice_line = (v->mb_y == 0);
        for (v->mb_x = 0; v->mb_x < v->mb_width; v->mb_x++) {
            vc1_init_block_index(v);
            vc1_set_dest(v);
            int16_t (*blk)[64] = v->blk[v->cur_blk_idx];
            memset(blk, 0, 6 * 64 * sizeof(int16_t));
            for (int i = 0; i < 4; i++) {
                v->mvb[v->block_index[i] * 2]     = 0;
                v->mvb[v->block_index[i] * 2 + 1] = 0;
            }

            int cbp = mp4::get_vlc2(gb, g_mb_i.table, g_mb_i.bits);
            if (cbp < 0) return VC1_ERR;
            v->ac_pred = (int)mp4::get_bits1(gb);

            for (k = 0; k < 6; k++) {
                v->mbtype[v->block_index[k]] = 1;
                int val = (cbp >> (5 - k)) & 1;
                if (k < 4) val = vc1_coded_block_pred(v, k, val);
                cbp |= val << (5 - k);
                if (vc1_decode_i_block(v, blk[vc1_block_map[k]], k, val,
                                       (k < 4) ? v->codingset : v->codingset2, gb) < 0)
                    return VC1_ERR;
                vc1_inv_trans_8x8(blk[vc1_block_map[k]]);
            }

            if (v->overlap && v->pq >= 9) {
                vc1_i_overlap_filter(v);
                if (v->rangeredfrm)
                    for (k = 0; k < 6; k++)
                        for (j = 0; j < 64; j++)
                            blk[vc1_block_map[k]][j] *= 2;
                vc1_put_blocks_clamped(v, 1);
            } else {
                if (v->rangeredfrm)
                    for (k = 0; k < 6; k++)
                        for (j = 0; j < 64; j++)
                            blk[vc1_block_map[k]][j] = (int16_t)((blk[vc1_block_map[k]][j] - 64) * 2);
                vc1_put_blocks_clamped(v, 0);
            }

            if (mp4::get_bits_left(gb) < 0) return VC1_ERR;   // bits overconsumption
            vc1_inc_blk_idx(v);
        }
    }
    if (v->loop_filter) vc1_i_deblock_frame(v);
    return VC1_OK;
}

// ═══════════════════════ P/B-frame inter decode (Simple/Main) ═══════════════════════

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

// ff_vc1_pred_dc (vc1_block.c:408) — full DC predictor with per-MB dqscale rescaling
// of the neighbors (needed when dquantfrm makes adjacent MBs use different mquant).
static inline int vc1_pred_dc_adv(VC1Context* v, int n, int a_avail, int c_avail,
                                  int16_t** dc_val_ptr, int* dir_ptr) {
    int mb_pos = v->mb_x + v->mb_y * v->mb_stride;
    int q1 = abs(v->qscale_table[mb_pos]);
    int dqscale_index = ff_wmv3_dc_scale_table[q1] - 1;
    int wrap = v->block_wrap[n];
    int16_t* dc_val = v->dc_val + v->block_index[n];
    *dc_val_ptr = dc_val;
    if (dqscale_index < 0) return 0;
    int c = dc_val[-1], b = dc_val[-1 - wrap], a = dc_val[-wrap], q2 = 0, pred;
    if (c_avail && (n != 1 && n != 3)) {
        q2 = abs(v->qscale_table[mb_pos - 1]);
        if (q2 && q2 != q1)
            c = (int)((unsigned)c * ff_wmv3_dc_scale_table[q2] * ff_vc1_dqscale[dqscale_index] + 0x20000) >> 18;
    }
    if (a_avail && (n != 2 && n != 3)) {
        q2 = abs(v->qscale_table[mb_pos - v->mb_stride]);
        if (q2 && q2 != q1)
            a = (int)((unsigned)a * ff_wmv3_dc_scale_table[q2] * ff_vc1_dqscale[dqscale_index] + 0x20000) >> 18;
    }
    if (a_avail && c_avail && (n != 3)) {
        int off = mb_pos;
        if (n != 1) off--;
        if (n != 2) off -= v->mb_stride;
        q2 = abs(v->qscale_table[off]);
        if (q2 && q2 != q1)
            b = (int)((unsigned)b * ff_wmv3_dc_scale_table[q2] * ff_vc1_dqscale[dqscale_index] + 0x20000) >> 18;
    }
    if (c_avail && (!a_avail || abs(a - b) <= abs(b - c))) { pred = c; *dir_ptr = 1; }
    else if (a_avail)                                      { pred = a; *dir_ptr = 0; }
    else                                                  { pred = 0; *dir_ptr = 1; }
    *dc_val_ptr = dc_val;
    return pred;
}

// vc1_decode_i_block_adv (vc1_block.c:708) — Advanced-Profile I-frame intra block.
// Full version: dqscale-rescaled DC/AC prediction (mquant varies per MB when
// dquantfrm), adv 8x8 scan by ac_pred/direction. Coeffs land in `block`.
static inline int vc1_decode_i_block_adv(VC1Context* v, int16_t block[64], int n,
                                         int coded, int codingset, int mquant, mp4::GetBits* gb) {
    int dc_pred_dir = 0;
    int16_t *dc_val, *ac_val, *ac_val2;
    int dcdiff, scale, k;
    int a_avail = v->a_avail, c_avail = v->c_avail, use_pred = v->ac_pred;
    int quant = mquant < 0 ? -mquant : mquant;
    int mb_pos = v->mb_x + v->mb_y * v->mb_stride;
    int q1, q2 = 0;

    dcdiff = msm::msm_dc_vlc(gb, msm::ff_msmp4_dc_tables[v->dc_table_index][n >= 4 ? 1 : 0]);
    if (dcdiff < 0) return VC1_ERR;
    if (dcdiff) {
        const int m = (quant == 1 || quant == 2) ? 3 - quant : 0;
        if (dcdiff == 119) dcdiff = (int)mp4::get_bits(gb, 8 + m);
        else if (m)        dcdiff = (dcdiff << m) + (int)mp4::get_bits(gb, m) - ((1 << m) - 1);
        if (mp4::get_bits1(gb)) dcdiff = -dcdiff;
    }
    dcdiff += vc1_pred_dc_adv(v, n, a_avail, c_avail, &dc_val, &dc_pred_dir);
    *dc_val  = (int16_t)dcdiff;
    block[0] = (int16_t)(dcdiff * v->y_dc_scale);

    if (!a_avail && !c_avail) use_pred = 0;
    scale = quant * 2 + (mquant < 0 ? 0 : v->halfpq);
    ac_val  = v->ac_val + (size_t)v->block_index[n] * 16;
    ac_val2 = ac_val;
    if (dc_pred_dir) ac_val -= 16;
    else             ac_val -= 16 * v->block_wrap[n];

    q1 = v->qscale_table[mb_pos];
    if (n == 3) q2 = q1;
    else if (dc_pred_dir) {
        if (n == 1) q2 = q1;
        else if (c_avail && mb_pos) q2 = v->qscale_table[mb_pos - 1];
    } else {
        if (n == 2) q2 = q1;
        else if (a_avail && mb_pos >= v->mb_stride) q2 = v->qscale_table[mb_pos - v->mb_stride];
    }

    if (coded) {
        int last = 0, skip, value;
        const uint8_t* zz = v->ac_pred ? (dc_pred_dir ? v->zz_8x8[3] : v->zz_8x8[2]) : v->zz_8x8[1];
        for (int i = 1; !last; ++i) {
            if (vc1_decode_ac_coeff(v, &last, &skip, &value, codingset, gb) < 0) return VC1_ERR;
            i += skip;
            if (i > 63) break;
            block[zz[i]] = (int16_t)value;
        }
        if (use_pred) {
            int sh;
            if (dc_pred_dir) sh = v->left_blk_sh;
            else            { sh = v->top_blk_sh; ac_val += 8; }
            q1 = (q1 < 0 ? -q1 : q1) * 2 + (q1 < 0 ? 0 : v->halfpq) - 1;
            if (q1 < 1) return VC1_ERR;
            if (q2) q2 = (q2 < 0 ? -q2 : q2) * 2 + (q2 < 0 ? 0 : v->halfpq) - 1;
            if (q2 && q1 != q2)
                for (k = 1; k < 8; k++)
                    block[k << sh] += (int)(ac_val[k] * (unsigned)q2 * ff_vc1_dqscale[q1 - 1] + 0x20000) >> 18;
            else
                for (k = 1; k < 8; k++) block[k << sh] += ac_val[k];
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
        memset(ac_val2, 0, 16 * 2);
        if (use_pred) {
            int sh;
            if (dc_pred_dir) sh = v->left_blk_sh;
            else            { sh = v->top_blk_sh; ac_val += 8; ac_val2 += 8; }
            memcpy(ac_val2, ac_val, 8 * 2);
            q1 = (q1 < 0 ? -q1 : q1) * 2 + (q1 < 0 ? 0 : v->halfpq) - 1;
            if (q1 < 1) return VC1_ERR;
            if (q2) q2 = (q2 < 0 ? -q2 : q2) * 2 + (q2 < 0 ? 0 : v->halfpq) - 1;
            if (q2 && q1 != q2)
                for (k = 1; k < 8; k++)
                    ac_val2[k] = (int)(ac_val2[k] * q2 * (unsigned)ff_vc1_dqscale[q1 - 1] + 0x20000) >> 18;
            for (k = 1; k < 8; k++) {
                block[k << sh] = ac_val2[k] * scale;
                if (!v->pquantizer && block[k << sh])
                    block[k << sh] += (block[k << sh] < 0) ? -quant : quant;
            }
        }
    }
    return VC1_OK;
}

// vc1_decode_i_blocks_adv (vc1_block.c:2619) — Advanced-Profile PROGRESSIVE I-frame.
// Mirrors the Simple/Main vc1_decode_i_frame but: ac_pred comes from the ACPRED
// bitplane (not a per-MB bit), a_avail/c_avail drive the DC/AC predictor per block,
// mquant == pq (dquantfrm==0 for I-frames), and blocks use the adv scan (iadv=1).
// overlap==0 in the target stream so the overlap smoothing / condover pass is skipped.
static inline int vc1_decode_i_frame_adv(VC1Context* v, mp4::GetBits* gb) {
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
            int mb_pos = v->mb_x + v->mb_y * v->mb_stride;

            int cbp = mp4::get_vlc2(gb, g_mb_i.table, g_mb_i.bits);
            if (cbp < 0) return VC1_ERR;
            v->ac_pred = v->acpred_is_raw ? (int)mp4::get_bits1(gb) : v->acpred_plane[mb_pos];
            int mquant = vc1_get_mquant(v, gb);
            v->qscale_table[mb_pos] = (int16_t)mquant;               // dqscale neighbor lookup
            v->y_dc_scale = ff_wmv3_dc_scale_table[mquant < 0 ? -mquant : mquant];

            for (int k = 0; k < 6; k++) {
                int val = (cbp >> (5 - k)) & 1;
                if (k < 4) val = vc1_coded_block_pred(v, k, val);
                v->a_avail = !v->first_slice_line || (k == 2 || k == 3);
                v->c_avail = (v->mb_x != 0)        || (k == 1 || k == 3);
                int ret = vc1_decode_i_block_adv(v, block[k], k, val,
                                                 (k < 4) ? v->codingset : v->codingset2, mquant, gb);
                if (ret < 0) return ret;
                vc1_inv_trans_8x8(block[k]);
            }
            vc1_put_mb(v, block, /*sgn=*/1);   // Advanced intra: put_signed (+128)
        }
    }
    if (v->loop_filter) vc1_i_deblock_frame(v);
    return VC1_OK;
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

// vc1_decode_intra_block (vc1_block.c:888) — intra block inside a P or B picture.
// DC/AC predictors are rescaled through ff_vc1_dqscale when the neighboring MB used a
// different MQUANT (dquant); with one quantizer per picture q1 equals q2 and it reduces to
// the plain add. PROGRESSIVE scan zz_8x8[0]. Coefficients land in `block`.
static inline int vc1_decode_intra_block(VC1Context* v, int16_t block[64], int n,
                                         int coded, int mquant, int codingset, mp4::GetBits* gb) {
    int dc_pred_dir = 0;
    int16_t *dc_val = NULL, *ac_val, *ac_val2;
    int dcdiff;
    int mb_pos = v->mb_x + v->mb_y * v->mb_stride;
    int a_avail = v->a_avail, c_avail = v->c_avail;
    int use_pred = v->ac_pred;
    int scale;
    int q1, q2 = 0;
    int quant = mquant < 0 ? -mquant : mquant;

    memset(block, 0, 64 * sizeof(int16_t));

    quant = quant > 31 ? 31 : quant;                  // av_clip_uintp2(quant, 5)
    v->y_dc_scale = ff_wmv3_dc_scale_table[quant];

    dcdiff = msm::msm_dc_vlc(gb, msm::ff_msmp4_dc_tables[v->dc_table_index][n >= 4 ? 1 : 0]);
    if (dcdiff < 0) return VC1_ERR;
    if (dcdiff) {
        const int m = (quant == 1 || quant == 2) ? 3 - quant : 0;
        if (dcdiff == 119) dcdiff = (int)mp4::get_bits(gb, 8 + m);
        else if (m)        dcdiff = (dcdiff << m) + (int)mp4::get_bits(gb, m) - ((1 << m) - 1);
        if (mp4::get_bits1(gb)) dcdiff = -dcdiff;
    }

    dcdiff += vc1_pred_dc_adv(v, n, a_avail, c_avail, &dc_val, &dc_pred_dir);
    *dc_val = (int16_t)dcdiff;
    block[0] = (int16_t)(dcdiff * v->y_dc_scale);

    if (!a_avail) dc_pred_dir = 1;
    if (!c_avail) dc_pred_dir = 0;
    if (!a_avail && !c_avail) use_pred = 0;
    ac_val  = v->ac_val + (size_t)v->block_index[n] * 16;
    ac_val2 = ac_val;

    scale = quant * 2 + ((mquant < 0) ? 0 : v->halfpq);

    if (dc_pred_dir) ac_val -= 16;
    else             ac_val -= 16 * v->block_wrap[n];

    q1 = v->qscale_table[mb_pos];
    if (dc_pred_dir && c_avail && mb_pos)
        q2 = v->qscale_table[mb_pos - 1];
    if (!dc_pred_dir && a_avail && mb_pos >= v->mb_stride)
        q2 = v->qscale_table[mb_pos - v->mb_stride];
    if (dc_pred_dir && n == 1) q2 = q1;
    if (!dc_pred_dir && n == 2) q2 = q1;
    if (n == 3) q2 = q1;

    if (coded) {
        int last = 0, skip, value, k;
        for (int i = 1; !last; ++i) {
            if (vc1_decode_ac_coeff(v, &last, &skip, &value, codingset, gb) < 0) return VC1_ERR;
            i += skip;
            if (i > 63) break;
            block[v->zz_8x8[0][i]] = (int16_t)value;           // PROGRESSIVE scan
        }
        if (use_pred) {
            q1 = (q1 < 0 ? -q1 : q1) * 2 + ((q1 < 0) ? 0 : v->halfpq) - 1;
            if (q1 < 1) return VC1_ERR;
            if (q2) q2 = (q2 < 0 ? -q2 : q2) * 2 + ((q2 < 0) ? 0 : v->halfpq) - 1;
            if (q2 && q1 != q2) {
                if (dc_pred_dir)
                    for (k = 1; k < 8; k++)
                        block[k << v->left_blk_sh] += (int)(ac_val[k] * q2 * (unsigned)ff_vc1_dqscale[q1 - 1] + 0x20000) >> 18;
                else
                    for (k = 1; k < 8; k++)
                        block[k << v->top_blk_sh] += (int)(ac_val[k + 8] * q2 * (unsigned)ff_vc1_dqscale[q1 - 1] + 0x20000) >> 18;
            } else {
                if (dc_pred_dir) for (k = 1; k < 8; k++) block[k << v->left_blk_sh] += ac_val[k];
                else             for (k = 1; k < 8; k++) block[k << v->top_blk_sh]  += ac_val[k + 8];
            }
        }
        for (k = 1; k < 8; k++) {
            ac_val2[k]     = block[k << v->left_blk_sh];
            ac_val2[k + 8] = block[k << v->top_blk_sh];
        }
        for (k = 1; k < 64; k++)
            if (block[k]) {
                block[k] *= scale;
                if (!v->pquantizer) block[k] += (block[k] < 0) ? -quant : quant;
            }
    } else {
        int k;
        memset(ac_val2, 0, 16 * 2);
        if (dc_pred_dir) {
            if (use_pred) {
                memcpy(ac_val2, ac_val, 8 * 2);
                q1 = (q1 < 0 ? -q1 : q1) * 2 + ((q1 < 0) ? 0 : v->halfpq) - 1;
                if (q1 < 1) return VC1_ERR;
                if (q2) q2 = (q2 < 0 ? -q2 : q2) * 2 + ((q2 < 0) ? 0 : v->halfpq) - 1;
                if (q2 && q1 != q2)
                    for (k = 1; k < 8; k++)
                        ac_val2[k] = (int16_t)((int)(ac_val2[k] * (unsigned)q2 * ff_vc1_dqscale[q1 - 1] + 0x20000) >> 18);
            }
        } else {
            if (use_pred) {
                memcpy(ac_val2 + 8, ac_val + 8, 8 * 2);
                q1 = (q1 < 0 ? -q1 : q1) * 2 + ((q1 < 0) ? 0 : v->halfpq) - 1;
                if (q1 < 1) return VC1_ERR;
                if (q2) q2 = (q2 < 0 ? -q2 : q2) * 2 + ((q2 < 0) ? 0 : v->halfpq) - 1;
                if (q2 && q1 != q2)
                    for (k = 1; k < 8; k++)
                        ac_val2[k + 8] = (int16_t)((int)(ac_val2[k + 8] * (unsigned)q2 * ff_vc1_dqscale[q1 - 1] + 0x20000) >> 18);
            }
        }
        if (use_pred) {
            if (dc_pred_dir) {
                for (k = 1; k < 8; k++) {
                    block[k << v->left_blk_sh] = (int16_t)(ac_val2[k] * scale);
                    if (!v->pquantizer && block[k << v->left_blk_sh])
                        block[k << v->left_blk_sh] += (block[k << v->left_blk_sh] < 0) ? -quant : quant;
                }
            } else {
                for (k = 1; k < 8; k++) {
                    block[k << v->top_blk_sh] = (int16_t)(ac_val2[k + 8] * scale);
                    if (!v->pquantizer && block[k << v->top_blk_sh])
                        block[k << v->top_blk_sh] += (block[k << v->top_blk_sh] < 0) ? -quant : quant;
                }
            }
        }
    }
    return 0;
}

// ff_vc1_pred_mv (vc1_pred.c:212) — PROGRESSIVE subset (field_mode = 0, numref = 0,
// blocks_off = 0, mv_f all 0, so the field-scaling branches drop out). `is_intra` is
// v->mbtype. Writes cur_mv[dir][n], motion_val[dir] (and zeroes motion_val[1] for intra).
static inline void vc1_pred_mv(VC1Context* v, int n, int dmv_x, int dmv_y, int mv1,
                               int r_x, int r_y, int mb_intra, mp4::GetBits* gb) {
    const uint8_t* is_intra = v->mbtype;
    int wrap = v->b8_stride, xy = v->block_index[n], off = 0, px, py, sum;
    int16_t* mvbase = v->mv_val;
    int16_t* mvb    = v->mvb;
    if (!v->quarter_sample) { dmv_x *= 2; dmv_y *= 2; }
    if (mb_intra) {
        v->cur_mv[0][n][0] = mvbase[xy * 2] = 0;
        v->cur_mv[0][n][1] = mvbase[xy * 2 + 1] = 0;
        mvb[xy * 2] = mvb[xy * 2 + 1] = 0;
        if (mv1) {                                       // duplicate motion data for 1-MV block
            const int d[3] = { 1, wrap, wrap + 1 };
            for (int k = 0; k < 3; k++) {
                mvbase[(xy + d[k]) * 2] = mvbase[(xy + d[k]) * 2 + 1] = 0;
                mvb[(xy + d[k]) * 2]    = mvb[(xy + d[k]) * 2 + 1]    = 0;
            }
            v->fr_lumamv[v->mb_x + v->mb_y * v->mb_width][0] = 0;
            v->fr_lumamv[v->mb_x + v->mb_y * v->mb_width][1] = 0;
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
        // in 4-MV mode different blocks have different B predictor position
        switch (n) {
        case 0:
            if (v->res_rtm_flag) off = v->mb_x ? -1 : 1;
            else                 off = v->mb_x ? -1 : 2 * v->mb_width - wrap - 1;
            break;
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

    {   // Pullback MV as specified in 8.3.5.3.4
        int MV = mv1 ? -60 : -28;
        int qx = (v->mb_x << 6) + ((n == 1 || n == 3) ? 32 : 0);
        int qy = (v->mb_y << 6) + ((n == 2 || n == 3) ? 32 : 0);
        int X = (v->mb_width << 6) - 4, Y = (v->mb_height << 6) - 4;
        if (qx + px < MV) px = MV - qx;
        if (qy + py < MV) py = MV - qy;
        if (qx + px > X) px = X - qx;
        if (qy + py > Y) py = Y - qy;
    }
    // Calculate hybrid prediction as specified in 8.3.5.3.5
    if (a_valid && c_valid) {
        if (is_intra[xy - wrap]) sum = abs(px) + abs(py);
        else                     sum = abs(px - A0) + abs(py - A1);
        if (sum > 32) {
            if (mp4::get_bits1(gb)) { px = A0; py = A1; } else { px = C0; py = C1; }
        } else {
            if (is_intra[xy - 1]) sum = abs(px) + abs(py);
            else                  sum = abs(px - C0) + abs(py - C1);
            if (sum > 32) {
                if (mp4::get_bits1(gb)) { px = A0; py = A1; } else { px = C0; py = C1; }
            }
        }
    }
    // store MV using signed modulus of MV range defined in 4.11
    int mx = ((px + dmv_x + r_x) & ((r_x << 1) - 1)) - r_x;
    int my = ((py + dmv_y + r_y) & ((r_y << 1) - 1)) - r_y;
    v->cur_mv[0][n][0] = (int16_t)mx; v->cur_mv[0][n][1] = (int16_t)my;
    mvbase[xy * 2] = (int16_t)mx; mvbase[xy * 2 + 1] = (int16_t)my;
    if (mv1) {
        mvbase[(xy + 1) * 2] = (int16_t)mx;        mvbase[(xy + 1) * 2 + 1] = (int16_t)my;
        mvbase[(xy + wrap) * 2] = (int16_t)mx;     mvbase[(xy + wrap) * 2 + 1] = (int16_t)my;
        mvbase[(xy + wrap + 1) * 2] = (int16_t)mx; mvbase[(xy + wrap + 1) * 2 + 1] = (int16_t)my;
    }
}

// scale_mv (vc1_pred.h), B_FRACTION_DEN 256.
static inline int vc1_scale_mv(int value, int bfrac, int inv, int qs) {
    int n = bfrac;
    if (inv) n -= 256;
    if (!qs) return 2 * ((value * n + 255) >> 9);
    return (value * n + 128) >> 8;
}

// ff_vc1_pred_b_mv (vc1_pred.c:690) — progressive B picture. Direct mode scales the
// co-located motion_val[1] of next_pic (the newest I/P); forward/backward predictors
// are the median of this picture's own motion_val[0]/[1] (hybrid prediction disabled).
static inline void vc1_pred_b_mv(VC1Context* v, int dmv_x[2], int dmv_y[2], int direct,
                                 int mvtype, int mb_intra) {
    int xy, wrap, px = 0, py = 0;
    int r_x = v->range_x, r_y = v->range_y;
    int16_t* mv0 = v->mv_val;
    int16_t* mv1 = v->mv1_val;
    const int16_t* nmv = v->nmv_val;

    if (!v->quarter_sample) {
        dmv_x[0] *= 2; dmv_y[0] *= 2;
        dmv_x[1] *= 2; dmv_y[1] *= 2;
    }
    wrap = v->b8_stride;
    xy   = v->block_index[0];

    if (mb_intra) {
        mv0[xy * 2] = mv0[xy * 2 + 1] = mv1[xy * 2] = mv1[xy * 2 + 1] = 0;
        return;
    }

    v->cur_mv[0][0][0] = (int16_t)vc1_scale_mv(nmv[xy * 2],     v->bfraction, 0, v->quarter_sample);
    v->cur_mv[0][0][1] = (int16_t)vc1_scale_mv(nmv[xy * 2 + 1], v->bfraction, 0, v->quarter_sample);
    v->cur_mv[1][0][0] = (int16_t)vc1_scale_mv(nmv[xy * 2],     v->bfraction, 1, v->quarter_sample);
    v->cur_mv[1][0][1] = (int16_t)vc1_scale_mv(nmv[xy * 2 + 1], v->bfraction, 1, v->quarter_sample);

    // Pullback predicted motion vectors as specified in 8.4.5.4
    for (int d = 0; d < 2; d++) {
        v->cur_mv[d][0][0] = (int16_t)vc1_clampi(v->cur_mv[d][0][0], -60 - (v->mb_x << 6),
                                                 (v->mb_width  << 6) - 4 - (v->mb_x << 6));
        v->cur_mv[d][0][1] = (int16_t)vc1_clampi(v->cur_mv[d][0][1], -60 - (v->mb_y << 6),
                                                 (v->mb_height << 6) - 4 - (v->mb_y << 6));
    }
    if (direct) {
        mv0[xy * 2] = v->cur_mv[0][0][0]; mv0[xy * 2 + 1] = v->cur_mv[0][0][1];
        mv1[xy * 2] = v->cur_mv[1][0][0]; mv1[xy * 2 + 1] = v->cur_mv[1][0][1];
        return;
    }

    for (int dir = 0; dir < 2; dir++) {
        if (dir == 0 && !(mvtype == BMV_TYPE_FORWARD  || mvtype == BMV_TYPE_INTERPOLATED)) continue;
        if (dir == 1 && !(mvtype == BMV_TYPE_BACKWARD || mvtype == BMV_TYPE_INTERPOLATED)) continue;
        int16_t* grid = dir ? mv1 : mv0;
        int16_t*       C = grid + (xy - 2) * 2;
        const int16_t* A = grid + (xy - wrap * 2) * 2;
        int off          = (v->mb_x == (v->mb_width - 1)) ? -2 : 2;
        const int16_t* B = grid + (xy - wrap * 2 + off) * 2;

        if (!v->mb_x) C[0] = C[1] = 0;
        if (!v->first_slice_line) {                  // predictor A is not out of bounds
            if (v->mb_width == 1) { px = A[0]; py = A[1]; }
            else { px = vc1_mid_pred(A[0], B[0], C[0]); py = vc1_mid_pred(A[1], B[1], C[1]); }
        } else if (v->mb_x) {                        // predictor C is not out of bounds
            px = C[0]; py = C[1];
        } else {
            px = py = 0;
        }
        {   // Pullback MV as specified in 8.3.5.3.4 (sh = 5 below Advanced profile)
            int sh = 5;
            int MV = 4 - (1 << sh);
            int qx = (v->mb_x << sh), qy = (v->mb_y << sh);
            int X = (v->mb_width << sh) - 4, Y = (v->mb_height << sh) - 4;
            if (qx + px < MV) px = MV - qx;
            if (qy + py < MV) py = MV - qy;
            if (qx + px > X) px = X - qx;
            if (qy + py > Y) py = Y - qy;
        }
        // (hybrid prediction is compiled out in ffmpeg: `if (0 && ...)`)
        v->cur_mv[dir][0][0] = (int16_t)(((px + dmv_x[dir] + r_x) & ((r_x << 1) - 1)) - r_x);
        v->cur_mv[dir][0][1] = (int16_t)(((py + dmv_y[dir] + r_y) & ((r_y << 1) - 1)) - r_y);
    }
    mv0[xy * 2] = v->cur_mv[0][0][0]; mv0[xy * 2 + 1] = v->cur_mv[0][0][1];
    mv1[xy * 2] = v->cur_mv[1][0][0]; mv1[xy * 2 + 1] = v->cur_mv[1][0][1];
}

// ── Motion compensation (always emulate the source footprint = ffmpeg pixel-exact) ─
// vc1_scale_luma / vc1_scale_chroma + vc1_lut_scale_* (vc1_mc.c): when the CURRENT
// picture is range-reduced the reference footprint is halved toward 128, then the
// intensity-compensation LUT is applied. Progressive frames use the same LUT for both
// row parities. (Both UNVERIFIED: no sample uses them.)
static inline void vc1_scale_ref(const VC1Context* v, uint8_t* buf, int stride, int k,
                                 int use_ic, const uint8_t* lut) {
    if (v->rangeredfrm)
        for (int y = 0; y < k; y++)
            for (int x = 0; x < k; x++)
                buf[y * stride + x] = (uint8_t)(((buf[y * stride + x] - 128) >> 1) + 128);
    if (use_ic)
        for (int y = 0; y < k; y++)
            for (int x = 0; x < k; x++)
                buf[y * stride + x] = lut[buf[y * stride + x]];
}

// avg_* variants: dst = (dst + prediction + 1) >> 1 (op_avg / rnd_avg32 / avg2).
static inline void vc1_avg_block(uint8_t* dst, int dstride, const uint8_t* src, int sstride, int N) {
    for (int y = 0; y < N; y++)
        for (int x = 0; x < N; x++)
            dst[y * dstride + x] = (uint8_t)((dst[y * dstride + x] + src[y * sstride + x] + 1) >> 1);
}

static inline void vc1_mc_luma_block(VC1Context* v, uint8_t* dest, int dstride, const uint8_t* ref, int rstride,
                                     int src_x, int src_y, int mx, int my, int cw, int ch, int N,
                                     int use_ic, int avg) {
    uint8_t buf[19 * 19], tmp[16 * 16];
    uint8_t* out = avg ? tmp : dest;
    const int ostride = avg ? 16 : dstride;
    if (v->mspel) {
        const int k = N + 3;   // 19 (N=16) / 11 (N=8): src_x-1 .. src_x+N+1
        vc1_emulated_edge(buf, k, ref, rstride, k, k, src_x - 1, src_y - 1, cw, ch);
        vc1_scale_ref(v, buf, k, k, use_ic, v->last_luty);
        vc1_put_mspel_mc(out, ostride, buf + k + 1, k, mx & 3, my & 3, v->rnd, N);
    } else {               // half-pel bilinear (1MV_HPEL_BILIN)
        const int k = N + 1;
        vc1_emulated_edge(buf, k, ref, rstride, k, k, src_x, src_y, cw, ch);
        vc1_scale_ref(v, buf, k, k, use_ic, v->last_luty);
        vc1_put_hpel(out, ostride, buf, k, N, (my & 2) | ((mx & 2) >> 1), 1 - v->rnd);
    }
    if (avg) vc1_avg_block(dest, dstride, tmp, 16, N);
}

static inline void vc1_mc_chroma_block(VC1Context* v, uint8_t* dest, int dstride, const uint8_t* ref, int rstride,
                                       int src_x, int src_y, int cx, int cy, int cw, int ch,
                                       int use_ic, int avg) {
    uint8_t buf[9 * 9], tmp[8 * 8];
    vc1_emulated_edge(buf, 9, ref, rstride, 9, 9, src_x, src_y, cw, ch);
    vc1_scale_ref(v, buf, 9, 9, use_ic, v->last_lutuv);
    vc1_put_chroma_mc8(avg ? tmp : dest, avg ? 8 : dstride, buf, 9, 8, cx, cy, v->rnd);
    if (avg) vc1_avg_block(dest, dstride, tmp, 8, 8);
}

// ff_vc1_mc_1mv (vc1_mc.c:172). dir 0 = last_pic (with its intensity compensation),
// dir 1 = next_pic. A P picture also records its MV in motion_val[1] for B direct mode.
static inline void vc1_mc_1mv(VC1Context* v, int dir) {
    int mx = v->cur_mv[dir][0][0], my = v->cur_mv[dir][0][1];
    if (v->pict_type == PICT_P)
        for (int i = 0; i < 4; i++) {
            v->mvb[v->block_index[i] * 2]     = (int16_t)mx;
            v->mvb[v->block_index[i] * 2 + 1] = (int16_t)my;
        }
    int uvmx = (mx + ((mx & 3) == 3)) >> 1, uvmy = (my + ((my & 3) == 3)) >> 1;
    v->fr_lumamv[v->mb_x + v->mb_y * v->mb_width][0] = (int16_t)uvmx;
    v->fr_lumamv[v->mb_x + v->mb_y * v->mb_width][1] = (int16_t)uvmy;
    if (v->fastuvmc) {
        uvmx = uvmx + ((uvmx < 0) ? (uvmx & 1) : -(uvmx & 1));
        uvmy = uvmy + ((uvmy < 0) ? (uvmy & 1) : -(uvmy & 1));
    }
    const uint8_t *sY, *sU, *sV;
    int use_ic;
    if (!dir) { sY = v->refY;  sU = v->refU;  sV = v->refV;  use_ic = v->last_use_ic; }
    else      { sY = v->nextY; sU = v->nextU; sV = v->nextV; use_ic = 0; }

    int cw = v->codedW, ch = v->codedH;
    int src_x = vc1_clampi(v->mb_x * 16 + (mx >> 2), -16, cw);
    int src_y = vc1_clampi(v->mb_y * 16 + (my >> 2), -16, ch);
    vc1_mc_luma_block(v, v->dest[0], v->yStride, sY, v->yStride, src_x, src_y, mx, my, cw, ch, 16, use_ic, 0);
    int ucw = cw >> 1, uch = ch >> 1;
    int usx = vc1_clampi(v->mb_x * 8 + (uvmx >> 2), -8, ucw);
    int usy = vc1_clampi(v->mb_y * 8 + (uvmy >> 2), -8, uch);
    int cx = (uvmx & 3) << 1, cy = (uvmy & 3) << 1;
    vc1_mc_chroma_block(v, v->dest[1], v->cStride, sU, v->cStride, usx, usy, cx, cy, ucw, uch, use_ic, 0);
    vc1_mc_chroma_block(v, v->dest[2], v->cStride, sV, v->cStride, usx, usy, cx, cy, ucw, uch, use_ic, 0);
}

// ff_vc1_interp_mc (vc1_mc.c:1004) — averages the backward (next_pic) prediction into
// the forward one already in dest. NB ffmpeg's FASTUVMC rounding here is away from
// zero, the opposite of mc_1mv's; transcribed as-is.
static inline void vc1_interp_mc(VC1Context* v) {
    int mx = v->cur_mv[1][0][0], my = v->cur_mv[1][0][1];
    int uvmx = (mx + ((mx & 3) == 3)) >> 1, uvmy = (my + ((my & 3) == 3)) >> 1;
    if (v->fastuvmc) {
        uvmx = uvmx + ((uvmx < 0) ? -(uvmx & 1) : (uvmx & 1));
        uvmy = uvmy + ((uvmy < 0) ? -(uvmy & 1) : (uvmy & 1));
    }
    int cw = v->codedW, ch = v->codedH;
    int src_x = vc1_clampi(v->mb_x * 16 + (mx >> 2), -16, cw);
    int src_y = vc1_clampi(v->mb_y * 16 + (my >> 2), -16, ch);
    vc1_mc_luma_block(v, v->dest[0], v->yStride, v->nextY, v->yStride, src_x, src_y, mx, my, cw, ch, 16, 0, 1);
    int ucw = cw >> 1, uch = ch >> 1;
    int usx = vc1_clampi(v->mb_x * 8 + (uvmx >> 2), -8, ucw);
    int usy = vc1_clampi(v->mb_y * 8 + (uvmy >> 2), -8, uch);
    int cx = (uvmx & 3) << 1, cy = (uvmy & 3) << 1;
    vc1_mc_chroma_block(v, v->dest[1], v->cStride, v->nextU, v->cStride, usx, usy, cx, cy, ucw, uch, 0, 1);
    vc1_mc_chroma_block(v, v->dest[2], v->cStride, v->nextV, v->cStride, usx, usy, cx, cy, ucw, uch, 0, 1);
}

// ff_vc1_mc_4mv_luma (vc1_mc.c:452), progressive P (dir 0, put).
static inline void vc1_mc_4mv_luma(VC1Context* v, int n) {
    int mx = v->cur_mv[0][n][0], my = v->cur_mv[0][n][1];
    int cw = v->codedW, ch = v->codedH;
    int off = v->yStride * 4 * (n & 2) + (n & 1) * 8;
    int src_x = vc1_clampi(v->mb_x * 16 + (n & 1) * 8 + (mx >> 2), -16, cw);
    int src_y = vc1_clampi(v->mb_y * 16 + (n & 2) * 4 + (my >> 2), -16, ch);
    vc1_mc_luma_block(v, v->dest[0] + off, v->yStride, v->refY, v->yStride, src_x, src_y, mx, my, cw, ch, 8,
                      v->last_use_ic, 0);
}

// ff_vc1_mc_4mv_chroma (vc1_mc.c:634), progressive P (dir 0). The chroma MV derived
// from the luma MVs is also motion_val[1] of block 0 (B direct mode source).
static inline void vc1_mc_4mv_chroma(VC1Context* v) {
    int idx = (!v->mbtype[v->block_index[0]]) | (!v->mbtype[v->block_index[1]] << 1)
            | (!v->mbtype[v->block_index[2]] << 2) | (!v->mbtype[v->block_index[3]] << 3);
    static const uint8_t popcount4[16] = { 0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4 };
    static const uint8_t index2[16] = { 0,0,0,0x01,0,0x02,0x12,0,0,0x03,0x13,0,0x23,0,0,0 };
    int valid = popcount4[idx], tx, ty;
    int mbfr = v->mb_x + v->mb_y * v->mb_width;
    int16_t (*mv)[2] = v->cur_mv[0];
    if (valid == 4) {
        tx = vc1_median4(mv[0][0], mv[1][0], mv[2][0], mv[3][0]);
        ty = vc1_median4(mv[0][1], mv[1][1], mv[2][1], mv[3][1]);
    } else if (valid == 3) {
        tx = vc1_mid_pred(mv[idx > 0xd][0], mv[1 + (idx > 0xb)][0], mv[2 + (idx > 0x7)][0]);
        ty = vc1_mid_pred(mv[idx > 0xd][1], mv[1 + (idx > 0xb)][1], mv[2 + (idx > 0x7)][1]);
    } else if (valid == 2) {
        tx = (mv[index2[idx] >> 4][0] + mv[index2[idx] & 0xf][0]) / 2;
        ty = (mv[index2[idx] >> 4][1] + mv[index2[idx] & 0xf][1]) / 2;
    } else {                                   // no need to do MC for intra blocks
        v->mvb[v->block_index[0] * 2] = v->mvb[v->block_index[0] * 2 + 1] = 0;
        v->fr_lumamv[mbfr][0] = v->fr_lumamv[mbfr][1] = 0;
        return;
    }
    v->mvb[v->block_index[0] * 2]     = (int16_t)tx;
    v->mvb[v->block_index[0] * 2 + 1] = (int16_t)ty;
    int uvmx = (tx + ((tx & 3) == 3)) >> 1, uvmy = (ty + ((ty & 3) == 3)) >> 1;
    v->fr_lumamv[mbfr][0] = (int16_t)uvmx;
    v->fr_lumamv[mbfr][1] = (int16_t)uvmy;
    if (v->fastuvmc) {
        uvmx = uvmx + ((uvmx < 0) ? (uvmx & 1) : -(uvmx & 1));
        uvmy = uvmy + ((uvmy < 0) ? (uvmy & 1) : -(uvmy & 1));
    }
    int ucw = v->codedW >> 1, uch = v->codedH >> 1;
    int usx = vc1_clampi(v->mb_x * 8 + (uvmx >> 2), -8, ucw);
    int usy = vc1_clampi(v->mb_y * 8 + (uvmy >> 2), -8, uch);
    int cx = (uvmx & 3) << 1, cy = (uvmy & 3) << 1;
    vc1_mc_chroma_block(v, v->dest[1], v->cStride, v->refU, v->cStride, usx, usy, cx, cy, ucw, uch, v->last_use_ic, 0);
    vc1_mc_chroma_block(v, v->dest[2], v->cStride, v->refV, v->cStride, usx, usy, cx, cy, ucw, uch, v->last_use_ic, 0);
}

// put_signed_pixels_clamped: dst = clip(block + 128) — intra blocks of a B picture.
static inline void vc1_put_signed_block(const int16_t* block, uint8_t* dst, int stride) {
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) dst[j] = vc1_clip8(block[j] + 128);
        block += 8; dst += stride;
    }
}

// vc1_decode_p_mb (vc1_block.c:1252) — progressive P macroblock. Intra blocks are
// reconstructed into the rolling block buffer and put (after optional overlap
// smoothing) by vc1_put_blocks_clamped; inter residual is added straight to the picture.
static inline int vc1_decode_p_mb(VC1Context* v, mp4::GetBits* gb) {
    int mb_pos = v->mb_x + v->mb_y * v->mb_stride;
    int cbp, mquant, ttmb = v->ttfrm;
    int dmv_x, dmv_y, mb_has_coeffs = 1, mb_intra = 0;
    int fourmv, skipped, first_block = 1, dst_idx, off, i, j;
    int block_cbp = 0, block_tt = 0, block_intra = 0;
    int16_t (*blk)[64] = v->blk[v->cur_blk_idx];
    const int lss[3] = { v->yStride, v->cStride, v->cStride };

    mquant = v->pq;                                         // lossy initialization

    fourmv  = v->mv_type_is_raw ? (int)mp4::get_bits1(gb) : v->mv_type_mb_plane[mb_pos];
    skipped = v->skip_is_raw    ? (int)mp4::get_bits1(gb) : v->mbskip_table[mb_pos];

    if (!fourmv) {                                          // 1MV mode
        if (!skipped) {
            vc1_get_mvdata(v, gb, &dmv_x, &dmv_y, &mb_has_coeffs, &mb_intra);
            if (mb_intra) {
                v->mvb[v->block_index[0] * 2] = 0;
                v->mvb[v->block_index[0] * 2 + 1] = 0;
            }
            vc1_pred_mv(v, 0, dmv_x, dmv_y, 1, v->range_x, v->range_y, mb_intra, gb);

            if (mb_intra && !mb_has_coeffs) {
                mquant = vc1_get_mquant(v, gb);
                v->ac_pred = (int)mp4::get_bits1(gb);
                cbp = 0;
            } else if (mb_has_coeffs) {
                if (mb_intra) v->ac_pred = (int)mp4::get_bits1(gb);
                cbp = ml_get_vlc2(gb, v->cbpcy_vlc, 9, 2);
                mquant = vc1_get_mquant(v, gb);
            } else {
                mquant = v->pq;
                cbp = 0;
            }
            v->qscale_table[mb_pos] = (int16_t)mquant;

            if (!v->ttmbf && !mb_intra && mb_has_coeffs)
                ttmb = ml_get_vlc2(gb, g_ttmb_vlc[v->tt_index], 9, 2);
            if (!mb_intra) vc1_mc_1mv(v, 0);
            dst_idx = 0;
            for (i = 0; i < 6; i++) {
                v->dc_val[v->block_index[i]] = 0;
                dst_idx += i >> 2;
                int val = (cbp >> (5 - i)) & 1;
                off = (i & 4) ? 0 : ((i & 1) * 8 + (i & 2) * 4 * v->yStride);
                v->mbtype[v->block_index[i]] = (uint8_t)mb_intra;
                if (mb_intra) {
                    // check if prediction blocks A and C are available
                    v->a_avail = v->c_avail = 0;
                    if (i == 2 || i == 3 || !v->first_slice_line)
                        v->a_avail = v->mbtype[v->block_index[i] - v->block_wrap[i]];
                    if (i == 1 || i == 3 || v->mb_x)
                        v->c_avail = v->mbtype[v->block_index[i] - 1];
                    if (vc1_decode_intra_block(v, blk[vc1_block_map[i]], i, val, mquant,
                                               (i & 4) ? v->codingset2 : v->codingset, gb) < 0)
                        return VC1_ERR;
                    vc1_inv_trans_8x8(blk[vc1_block_map[i]]);
                    if (v->rangeredfrm)
                        for (j = 0; j < 64; j++) blk[vc1_block_map[i]][j] *= 2;
                    block_cbp   |= 0xF << (i << 2);
                    block_intra |= 1 << i;
                } else if (val) {
                    int pat = vc1_decode_p_block(v, blk[vc1_block_map[i]], i, mquant, ttmb, first_block,
                                                 v->dest[dst_idx] + off, lss[dst_idx], &block_tt, gb);
                    if (pat < 0) return VC1_ERR;
                    block_cbp |= pat << (i << 2);
                    if (!v->ttmbf && ttmb < 8) ttmb = -1;
                    first_block = 0;
                }
            }
        } else {                                            // skipped
            for (i = 0; i < 6; i++) {
                v->mbtype[v->block_index[i]] = 0;
                v->dc_val[v->block_index[i]] = 0;
            }
            v->qscale_table[mb_pos] = 0;
            vc1_pred_mv(v, 0, 0, 0, 1, v->range_x, v->range_y, 0, gb);
            vc1_mc_1mv(v, 0);
        }
    } else {                                                // 4MV mode
        if (!skipped) {
            int intra_count = 0, coded_inter = 0, is_intra[6], is_coded[6];
            cbp = ml_get_vlc2(gb, v->cbpcy_vlc, 9, 2);
            for (i = 0; i < 6; i++) {
                int val = (cbp >> (5 - i)) & 1;
                v->dc_val[v->block_index[i]] = 0;
                mb_intra = 0;
                if (i < 4) {
                    dmv_x = dmv_y = 0;
                    mb_intra = 0;
                    mb_has_coeffs = 0;
                    if (val) vc1_get_mvdata(v, gb, &dmv_x, &dmv_y, &mb_has_coeffs, &mb_intra);
                    vc1_pred_mv(v, i, dmv_x, dmv_y, 0, v->range_x, v->range_y, mb_intra, gb);
                    if (!mb_intra) vc1_mc_4mv_luma(v, i);
                    intra_count += mb_intra;
                    is_intra[i]  = mb_intra;
                    is_coded[i]  = mb_has_coeffs;
                }
                if (i & 4) {
                    is_intra[i] = (intra_count >= 3);
                    is_coded[i] = val;
                }
                if (i == 4) vc1_mc_4mv_chroma(v);
                v->mbtype[v->block_index[i]] = (uint8_t)is_intra[i];
                if (!coded_inter) coded_inter = (!is_intra[i]) & is_coded[i];
            }
            // if there are no coded blocks then don't do anything more
            dst_idx = 0;
            if (!intra_count && !coded_inter) goto end;
            mquant = vc1_get_mquant(v, gb);
            v->qscale_table[mb_pos] = (int16_t)mquant;
            {   // test if block is intra and has pred
                int intrapred = 0;
                for (i = 0; i < 6; i++)
                    if (is_intra[i]) {
                        if (((!v->first_slice_line || (i == 2 || i == 3)) && v->mbtype[v->block_index[i] - v->block_wrap[i]])
                            || ((v->mb_x || (i == 1 || i == 3)) && v->mbtype[v->block_index[i] - 1])) {
                            intrapred = 1;
                            break;
                        }
                    }
                v->ac_pred = intrapred ? (int)mp4::get_bits1(gb) : 0;
            }
            if (!v->ttmbf && coded_inter)
                ttmb = ml_get_vlc2(gb, g_ttmb_vlc[v->tt_index], 9, 2);
            for (i = 0; i < 6; i++) {
                dst_idx += i >> 2;
                off = (i & 4) ? 0 : ((i & 1) * 8 + (i & 2) * 4 * v->yStride);
                if (is_intra[i]) {
                    v->a_avail = v->c_avail = 0;
                    if (i == 2 || i == 3 || !v->first_slice_line)
                        v->a_avail = v->mbtype[v->block_index[i] - v->block_wrap[i]];
                    if (i == 1 || i == 3 || v->mb_x)
                        v->c_avail = v->mbtype[v->block_index[i] - 1];
                    if (vc1_decode_intra_block(v, blk[vc1_block_map[i]], i, is_coded[i], mquant,
                                               (i & 4) ? v->codingset2 : v->codingset, gb) < 0)
                        return VC1_ERR;
                    vc1_inv_trans_8x8(blk[vc1_block_map[i]]);
                    if (v->rangeredfrm)
                        for (j = 0; j < 64; j++) blk[vc1_block_map[i]][j] *= 2;
                    block_cbp   |= 0xF << (i << 2);
                    block_intra |= 1 << i;
                } else if (is_coded[i]) {
                    int pat = vc1_decode_p_block(v, blk[vc1_block_map[i]], i, mquant, ttmb, first_block,
                                                 v->dest[dst_idx] + off, lss[dst_idx], &block_tt, gb);
                    if (pat < 0) return VC1_ERR;
                    block_cbp |= pat << (i << 2);
                    if (!v->ttmbf && ttmb < 8) ttmb = -1;
                    first_block = 0;
                }
            }
        } else {                                            // skipped MB
            v->qscale_table[mb_pos] = 0;
            for (i = 0; i < 6; i++) {
                v->mbtype[v->block_index[i]] = 0;
                v->dc_val[v->block_index[i]] = 0;
            }
            for (i = 0; i < 4; i++) {
                vc1_pred_mv(v, i, 0, 0, 0, v->range_x, v->range_y, 0, gb);
                vc1_mc_4mv_luma(v, i);
            }
            vc1_mc_4mv_chroma(v);
        }
    }
end:
    if (v->overlap && v->pq >= 9)
        vc1_p_overlap_filter(v);
    vc1_put_blocks_clamped(v, 1);

    int mb_fr = v->mb_x + v->mb_y * v->mb_width;   // fr_* arrays use mb_width, not mb_stride
    v->fr_cbp[mb_fr]     = block_cbp;
    v->fr_ttblk[mb_fr]   = block_tt;
    v->fr_isintra[mb_fr] = (uint8_t)block_intra;
    return VC1_OK;
}

static inline void vc1_p_deblock_frame(VC1Context* v);   // defined below

static inline void vc1_select_codingsets_inter(VC1Context* v) {
    switch (v->c_ac_table_index) {
    case 0: v->codingset = (v->pqindex <= 8) ? CS_HIGH_RATE_INTRA : CS_LOW_MOT_INTRA;
            v->codingset2 = (v->pqindex <= 8) ? CS_HIGH_RATE_INTER : CS_LOW_MOT_INTER; break;
    case 1: v->codingset = CS_HIGH_MOT_INTRA; v->codingset2 = CS_HIGH_MOT_INTER; break;
    case 2: v->codingset = CS_MID_RATE_INTRA; v->codingset2 = CS_MID_RATE_INTER; break;
    }
}

// vc1_decode_p_blocks (vc1_block.c:2750), progressive.
static inline int vc1_decode_p_frame(VC1Context* v, mp4::GetBits* gb) {
    vc1_init_iframe_tables();
    vc1_select_codingsets_inter(v);
    v->cbpcy_vlc = g_cbpcy_p_vlc[v->cbptab];
    v->esc3_level_length = 0;
    v->y_dc_scale = ff_wmv3_dc_scale_table[v->pq];

    memset(v->dc_base, 0, (size_t)v->grid_total * sizeof(int16_t));
    memset(v->ac_base, 0, (size_t)v->grid_total * 16 * sizeof(int16_t));
    memset(v->mbtype_base, 0, (size_t)v->grid_total);
    memset(v->mv_base, 0, (size_t)v->grid_total * 2 * sizeof(int16_t));
    memset(v->fr_lumamv, 0, (size_t)v->mb_width * v->mb_height * sizeof(int16_t) * 2);
    vc1_reset_blk_idx(v);

    for (v->mb_y = 0; v->mb_y < v->mb_height; v->mb_y++) {
        v->first_slice_line = (v->mb_y == 0);
        for (v->mb_x = 0; v->mb_x < v->mb_width; v->mb_x++) {
            vc1_init_block_index(v);
            vc1_set_dest(v);
            if (v->mv_type_is_raw || v->skip_is_raw)
                if (mp4::get_bits_left(gb) < 1) return VC1_ERR;
            if (vc1_decode_p_mb(v, gb) < 0 || mp4::get_bits_left(gb) < 0) return VC1_ERR;
            vc1_inc_blk_idx(v);
        }
    }
    if (v->loop_filter) vc1_p_deblock_frame(v);
    return VC1_OK;
}

// vc1_b_mc (vc1_block.c:318).
static inline void vc1_b_mc(VC1Context* v, int direct, int mode) {
    if (direct || mode == BMV_TYPE_INTERPOLATED) {
        vc1_mc_1mv(v, 0);
        vc1_interp_mc(v);
        return;
    }
    vc1_mc_1mv(v, mode == BMV_TYPE_BACKWARD);
}

// vc1_decode_b_mb (vc1_block.c:1822) — progressive B macroblock (always 1MV). Intra
// blocks are put immediately (B pictures have no overlap smoothing).
static inline int vc1_decode_b_mb(VC1Context* v, mp4::GetBits* gb) {
    int mb_pos = v->mb_x + v->mb_y * v->mb_stride;
    int cbp = 0, mquant, ttmb = v->ttfrm;
    int mb_has_coeffs = 0, mb_intra = 0;
    int first_block = 1, dst_idx, off, skipped, direct, i, j;
    int dmv_x[2], dmv_y[2];
    int bmvtype = BMV_TYPE_BACKWARD;
    int16_t block[64];
    const int lss[3] = { v->yStride, v->cStride, v->cStride };

    mquant = v->pq;                                         // lossy initialization

    direct  = v->dmb_is_raw  ? (int)mp4::get_bits1(gb) : v->direct_mb_plane[mb_pos];
    skipped = v->skip_is_raw ? (int)mp4::get_bits1(gb) : v->mbskip_table[mb_pos];

    dmv_x[0] = dmv_x[1] = dmv_y[0] = dmv_y[1] = 0;
    for (i = 0; i < 6; i++) {
        v->mbtype[v->block_index[i]] = 0;
        v->dc_val[v->block_index[i]] = 0;
    }
    v->qscale_table[mb_pos] = 0;

    if (!direct) {
        if (!skipped) {
            vc1_get_mvdata(v, gb, &dmv_x[0], &dmv_y[0], &mb_has_coeffs, &mb_intra);
            dmv_x[1] = dmv_x[0];
            dmv_y[1] = dmv_y[0];
        }
        if (skipped || !mb_intra) {
            bmvtype = vc1_decode012(gb);
            switch (bmvtype) {
            case 0:
                bmvtype = (v->bfraction >= (B_FRACTION_DEN / 2)) ? BMV_TYPE_BACKWARD : BMV_TYPE_FORWARD;
                break;
            case 1:
                bmvtype = (v->bfraction >= (B_FRACTION_DEN / 2)) ? BMV_TYPE_FORWARD : BMV_TYPE_BACKWARD;
                break;
            case 2:
                bmvtype  = BMV_TYPE_INTERPOLATED;
                dmv_x[0] = dmv_y[0] = 0;
            }
        }
    }
    for (i = 0; i < 6; i++)
        v->mbtype[v->block_index[i]] = (uint8_t)mb_intra;

    if (skipped) {
        if (direct) bmvtype = BMV_TYPE_INTERPOLATED;
        vc1_pred_b_mv(v, dmv_x, dmv_y, direct, bmvtype, mb_intra);
        vc1_b_mc(v, direct, bmvtype);
        return VC1_OK;
    }
    if (direct) {
        cbp = ml_get_vlc2(gb, v->cbpcy_vlc, 9, 2);
        mquant = vc1_get_mquant(v, gb);
        mb_intra = 0;
        v->qscale_table[mb_pos] = (int16_t)mquant;
        if (!v->ttmbf)
            ttmb = ml_get_vlc2(gb, g_ttmb_vlc[v->tt_index], 9, 2);
        dmv_x[0] = dmv_y[0] = dmv_x[1] = dmv_y[1] = 0;
        vc1_pred_b_mv(v, dmv_x, dmv_y, direct, bmvtype, mb_intra);
        vc1_b_mc(v, direct, bmvtype);
    } else {
        if (!mb_has_coeffs && !mb_intra) {
            // no coded blocks - effectively skipped
            vc1_pred_b_mv(v, dmv_x, dmv_y, direct, bmvtype, mb_intra);
            vc1_b_mc(v, direct, bmvtype);
            return VC1_OK;
        }
        if (mb_intra && !mb_has_coeffs) {
            mquant = vc1_get_mquant(v, gb);
            v->qscale_table[mb_pos] = (int16_t)mquant;
            v->ac_pred = (int)mp4::get_bits1(gb);
            cbp = 0;
            vc1_pred_b_mv(v, dmv_x, dmv_y, direct, bmvtype, mb_intra);
        } else {
            if (bmvtype == BMV_TYPE_INTERPOLATED) {
                vc1_get_mvdata(v, gb, &dmv_x[0], &dmv_y[0], &mb_has_coeffs, &mb_intra);
                if (!mb_has_coeffs) {
                    // interpolated skipped block
                    vc1_pred_b_mv(v, dmv_x, dmv_y, direct, bmvtype, mb_intra);
                    vc1_b_mc(v, direct, bmvtype);
                    return VC1_OK;
                }
            }
            vc1_pred_b_mv(v, dmv_x, dmv_y, direct, bmvtype, mb_intra);
            if (!mb_intra)
                vc1_b_mc(v, direct, bmvtype);
            if (mb_intra)
                v->ac_pred = (int)mp4::get_bits1(gb);
            cbp = ml_get_vlc2(gb, v->cbpcy_vlc, 9, 2);
            mquant = vc1_get_mquant(v, gb);
            v->qscale_table[mb_pos] = (int16_t)mquant;
            if (!v->ttmbf && !mb_intra && mb_has_coeffs)
                ttmb = ml_get_vlc2(gb, g_ttmb_vlc[v->tt_index], 9, 2);
        }
    }
    dst_idx = 0;
    for (i = 0; i < 6; i++) {
        v->dc_val[v->block_index[i]] = 0;
        dst_idx += i >> 2;
        int val = (cbp >> (5 - i)) & 1;
        off = (i & 4) ? 0 : ((i & 1) * 8 + (i & 2) * 4 * v->yStride);
        v->mbtype[v->block_index[i]] = (uint8_t)mb_intra;
        if (mb_intra) {
            // check if prediction blocks A and C are available
            v->a_avail = v->c_avail = 0;
            if (i == 2 || i == 3 || !v->first_slice_line)
                v->a_avail = v->mbtype[v->block_index[i] - v->block_wrap[i]];
            if (i == 1 || i == 3 || v->mb_x)
                v->c_avail = v->mbtype[v->block_index[i] - 1];
            if (vc1_decode_intra_block(v, block, i, val, mquant,
                                       (i & 4) ? v->codingset2 : v->codingset, gb) < 0)
                return VC1_ERR;
            vc1_inv_trans_8x8(block);
            if (v->rangeredfrm)
                for (j = 0; j < 64; j++) block[j] *= 2;
            vc1_put_signed_block(block, v->dest[dst_idx] + off, lss[dst_idx]);
        } else if (val) {
            int pat = vc1_decode_p_block(v, block, i, mquant, ttmb, first_block,
                                         v->dest[dst_idx] + off, lss[dst_idx], NULL, gb);
            if (pat < 0) return VC1_ERR;
            if (!v->ttmbf && ttmb < 8) ttmb = -1;
            first_block = 0;
        }
    }
    return VC1_OK;
}

// vc1_decode_b_blocks (vc1_block.c:2839), progressive. The B in-loop filter is the
// I-picture one (every block edge at pq, ff_vc1_i_loop_filter).
static inline int vc1_decode_b_frame(VC1Context* v, mp4::GetBits* gb) {
    vc1_init_iframe_tables();
    vc1_select_codingsets_inter(v);
    v->cbpcy_vlc = g_cbpcy_p_vlc[v->cbptab];
    v->esc3_level_length = 0;

    memset(v->dc_base, 0, (size_t)v->grid_total * sizeof(int16_t));
    memset(v->ac_base, 0, (size_t)v->grid_total * 16 * sizeof(int16_t));
    memset(v->mbtype_base, 0, (size_t)v->grid_total);
    memset(v->mv_base, 0, (size_t)v->grid_total * 2 * sizeof(int16_t));
    memset(v->mv1_base, 0, (size_t)v->grid_total * 2 * sizeof(int16_t));

    for (v->mb_y = 0; v->mb_y < v->mb_height; v->mb_y++) {
        v->first_slice_line = (v->mb_y == 0);
        for (v->mb_x = 0; v->mb_x < v->mb_width; v->mb_x++) {
            vc1_init_block_index(v);
            vc1_set_dest(v);
            if (v->skip_is_raw || v->dmb_is_raw)
                if (mp4::get_bits_left(gb) < 1) return VC1_ERR;
            if (vc1_decode_b_mb(v, gb) < 0 || mp4::get_bits_left(gb) < 0) return VC1_ERR;
        }
    }
    if (v->loop_filter) vc1_i_deblock_frame(v);
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
