/*
    WMV2 (Windows Media Video 8) decoder layer — transcription of FFmpeg release/8.1
    libavcodec/wmv2dec.c (extradata header, picture + secondary header, MB-skip map,
    the WMV2 MB layer with top-left MV prediction, ABT 8x4/4x8 blocks, mspel motion,
    ff_wmv2_add_mb), the slice / reconstruct / deblock order of h263dec.c's
    decode_slice + mpegvideo_dec.c's ff_mpv_reconstruct_mb, and ff_h263_loop_filter
    (h263.c). J-frames go to IntraX8 (intrax8.h). FFmpeg is LGPL-2.1+; see
    msmpeg4_decode.h for the module-wide provenance note.

    WMV2 shares the WMV1 block layer (msmpeg4_decode.h): coefficient VLCs + escapes,
    DC/AC prediction, the MV VLC, half-pel MC and H.263 intra dequant. It differs in
    the IDCT (wmv2dsp, also for intra put), the inter MB-type table set, the MV
    predictor, quarter-ish "mspel" luma MC, the ABT split transforms and the in-loop
    deblocking filter.
*/

#ifndef WMV2_DECODE_H
#define WMV2_DECODE_H

#include "msmpeg4_decode.h"
#include "wmv2_tables.h"

namespace msm {

enum { WMV2_FRAME_SKIPPED = 1 };

// decode_ext_header: the 4-byte extradata. FFmpeg ignores its result (a missing or
// short header leaves every flag 0 and slice_height = mb_height).
static inline int wmv2_decode_ext_header(MsmDecoder* d, const uint8_t* extradata, int size) {
    if (size < 4) return MSM_ERR;
    uint8_t buf[12] = { 0 };                    // get_bits reads 4 bytes past the position
    memcpy(buf, extradata, 4);
    mp4::GetBits gb;
    mp4::init_get_bits(&gb, buf, 4);

    mp4::get_bits(&gb, 5);                       // fps
    d->bit_rate         = (int)mp4::get_bits(&gb, 11) * 1024;
    d->mspel_bit        = (int)mp4::get_bits1(&gb);
    d->loop_filter      = (int)mp4::get_bits1(&gb);
    d->abt_flag         = (int)mp4::get_bits1(&gb);
    d->j_type_bit       = (int)mp4::get_bits1(&gb);
    d->top_left_mv_flag = (int)mp4::get_bits1(&gb);
    d->per_mb_rl_bit    = (int)mp4::get_bits1(&gb);
    int code            = (int)mp4::get_bits(&gb, 3);
    if (code == 0) return MSM_ERR;
    d->slice_height = d->mb_height / code;
    return MSM_OK;
}

// wmv2_decode_picture_header. Returns WMV2_FRAME_SKIPPED when a P picture's skip map
// marks every MB skipped (FFmpeg then outputs no frame; the picture is unchanged).
static inline int wmv2_decode_picture_header(MsmDecoder* d, mp4::GetBits* gb) {
    d->pict_type = (int)mp4::get_bits1(gb) + 1;
    if (d->pict_type == PICT_I)
        mp4::get_bits(gb, 7);
    d->qscale = (int)mp4::get_bits(gb, 5);
    if (d->qscale <= 0) return MSM_ERR;

    if (d->pict_type != PICT_I && mp4::show_bits(gb, 1)) {
        mp4::GetBits g = *gb;
        int skip_type = (int)mp4::get_bits(&g, 2);
        int run = skip_type == SKIP_TYPE_COL ? d->mb_width : d->mb_height;

        while (run > 0) {
            int block = run < 25 ? run : 25;
            if ((int)mp4::get_bits(&g, block) + 1 != 1 << block)
                break;
            run -= block;
        }
        if (!run)
            return WMV2_FRAME_SKIPPED;
    }
    return MSM_OK;
}

// parse_mb_skip: fills mb_skip[mb_y * mb_stride + mb_x].
static inline int wmv2_parse_mb_skip(MsmDecoder* d, mp4::GetBits* gb) {
    const int mw = d->mb_width, mh = d->mb_height, ms = d->mb_stride;
    uint8_t* skip = d->mb_skip;
    int coded_mb_count = 0;

    int skip_type = (int)mp4::get_bits(gb, 2);
    switch (skip_type) {
    case SKIP_TYPE_NONE:
        for (int mb_y = 0; mb_y < mh; mb_y++)
            for (int mb_x = 0; mb_x < mw; mb_x++)
                skip[mb_y * ms + mb_x] = 0;
        break;
    case SKIP_TYPE_MPEG:
        if (mp4::get_bits_left(gb) < mh * mw)
            return MSM_ERR;
        for (int mb_y = 0; mb_y < mh; mb_y++)
            for (int mb_x = 0; mb_x < mw; mb_x++)
                skip[mb_y * ms + mb_x] = (uint8_t)mp4::get_bits1(gb);
        break;
    case SKIP_TYPE_ROW:
        for (int mb_y = 0; mb_y < mh; mb_y++) {
            if (mp4::get_bits_left(gb) < 1)
                return MSM_ERR;
            if (mp4::get_bits1(gb)) {
                for (int mb_x = 0; mb_x < mw; mb_x++)
                    skip[mb_y * ms + mb_x] = 1;
            } else {
                if (mp4::get_bits_left(gb) < mw)
                    return MSM_ERR;
                for (int mb_x = 0; mb_x < mw; mb_x++)
                    skip[mb_y * ms + mb_x] = (uint8_t)mp4::get_bits1(gb);
            }
        }
        break;
    case SKIP_TYPE_COL:
        for (int mb_x = 0; mb_x < mw; mb_x++) {
            if (mp4::get_bits_left(gb) < 1)
                return MSM_ERR;
            if (mp4::get_bits1(gb)) {
                for (int mb_y = 0; mb_y < mh; mb_y++)
                    skip[mb_y * ms + mb_x] = 1;
            } else {
                if (mp4::get_bits_left(gb) < mh)
                    return MSM_ERR;
                for (int mb_y = 0; mb_y < mh; mb_y++)
                    skip[mb_y * ms + mb_x] = (uint8_t)mp4::get_bits1(gb);
            }
        }
        break;
    }

    for (int mb_y = 0; mb_y < mh; mb_y++)
        for (int mb_x = 0; mb_x < mw; mb_x++)
            coded_mb_count += !skip[mb_y * ms + mb_x];

    if (coded_mb_count > mp4::get_bits_left(gb))
        return MSM_ERR;
    return MSM_OK;
}

// ff_wmv2_decode_secondary_picture_header (+ the J-frame IntraX8 decode it runs).
// Returns MSM_OK, MSM_ERR, or 1 when the picture was a J-frame (already decoded).
static inline int wmv2_decode_secondary_picture_header(MsmDecoder* d, mp4::GetBits* gb) {
    if (d->pict_type == PICT_I) {
        memset(d->mb_skip, 0, (size_t)d->mb_stride * d->mb_height);
        d->j_type = d->j_type_bit ? (int)mp4::get_bits1(gb) : 0;

        if (!d->j_type) {
            d->per_mb_rl_table = d->per_mb_rl_bit ? (int)mp4::get_bits1(gb) : 0;

            if (!d->per_mb_rl_table) {
                d->rl_chroma_table_index = decode012(gb);
                d->rl_table_index        = decode012(gb);
            }

            d->dc_table_index = (int)mp4::get_bits1(gb);

            // at minimum one bit per macroblock is required in a valid frame
            if ((int64_t)mp4::get_bits_left(gb) * 8 < (int64_t)((d->width + 15) / 16) * ((d->height + 15) / 16))
                return MSM_ERR;
        }
        d->inter_intra_pred = 0;
        d->no_rounding      = 1;
    } else {
        d->j_type = 0;

        if (wmv2_parse_mb_skip(d, gb) < 0)
            return MSM_ERR;
        int cbp_index = decode012(gb);
        d->cbp_table_index = wmv2_get_cbp_table_index(d->qscale, cbp_index);

        d->mspel = d->mspel_bit ? (int)mp4::get_bits1(gb) : 0;

        if (d->abt_flag) {
            d->per_mb_abt = (int)mp4::get_bits1(gb) ^ 1;
            if (!d->per_mb_abt)
                d->abt_type = decode012(gb);
        }

        d->per_mb_rl_table = d->per_mb_rl_bit ? (int)mp4::get_bits1(gb) : 0;

        if (!d->per_mb_rl_table) {
            d->rl_table_index        = decode012(gb);
            d->rl_chroma_table_index = d->rl_table_index;
        }

        if (mp4::get_bits_left(gb) < 2)
            return MSM_ERR;

        d->dc_table_index = (int)mp4::get_bits1(gb);
        d->mv_table_index = (int)mp4::get_bits1(gb);

        d->inter_intra_pred = 0;
        d->no_rounding     ^= 1;
    }
    d->esc3_level_length = 0;
    d->esc3_run_length   = 0;

    if (d->j_type) {
        x8_decode_picture(&d->x8, gb, d->y, d->u, d->v, d->yStride, d->cStride,
                          2 * d->qscale, (d->qscale - 1) | 1, d->loop_filter);
        return 1;
    }
    return MSM_OK;
}

// wmv2_pred_motion: median of A/B/C, or (top_left_mv_flag) an explicit A-or-B pick.
static inline void wmv2_pred_motion(MsmDecoder* d, mp4::GetBits* gb, int* px, int* py) {
    int diff, type;
    int wrap = d->b8_stride, xy = d->block_index[0];
    const int16_t* A = d->mot_val + (xy - 1) * 2;
    const int16_t* B = d->mot_val + (xy - wrap) * 2;
    const int16_t* C = d->mot_val + (xy + 2 - wrap) * 2;

    if (d->mb_x && !d->first_slice_line && !d->mspel && d->top_left_mv_flag) {
        int dx = abs(A[0] - B[0]), dy = abs(A[1] - B[1]);
        diff = dx > dy ? dx : dy;
    } else
        diff = 0;

    if (diff >= 8)
        type = (int)mp4::get_bits1(gb);
    else
        type = 2;

    if (type == 0) {
        *px = A[0];
        *py = A[1];
    } else if (type == 1) {
        *px = B[0];
        *py = B[1];
    } else {
        /* special case for first (slice) line */
        if (d->first_slice_line) {
            *px = A[0];
            *py = A[1];
        } else {
            *px = msm_mid_pred(A[0], B[0], C[0]);
            *py = msm_mid_pred(A[1], B[1], C[1]);
        }
    }
}

// wmv2_decode_motion: the msmpeg4 MV, plus the mspel half-shift bit.
static inline void wmv2_decode_motion(MsmDecoder* d, mp4::GetBits* gb, int* mx, int* my) {
    msm_decode_motion(d, gb, mx, my);
    if (((*mx | *my) & 1) && d->mspel)
        d->hshift = (int)mp4::get_bits1(gb);
    else
        d->hshift = 0;
}

static inline int wmv2_decode_inter_block(MsmDecoder* d, mp4::GetBits* gb, int16_t* block, int n, int cbp) {
    static const int sub_cbp_table[3] = { 2, 3, 1 };

    if (!cbp) {
        d->block_last_index[n] = -1;
        return MSM_OK;
    }

    if (d->per_block_abt)
        d->abt_type = decode012(gb);
    d->abt_type_table[n] = d->abt_type;

    if (d->abt_type) {
        const uint8_t* scantable = d->abt_type == 1 ? ff_wmv2_scantableA : ff_wmv2_scantableB;

        int sub_cbp = sub_cbp_table[decode012(gb)];

        if (sub_cbp & 1) {
            if (msm_decode_block(d, gb, block, n, 1, 0, scantable) < 0)
                return MSM_ERR;
        }
        if (sub_cbp & 2) {
            if (msm_decode_block(d, gb, d->abt_block2[n], n, 1, 0, scantable) < 0)
                return MSM_ERR;
        }
        d->block_last_index[n] = 63;
        return MSM_OK;
    }
    return msm_decode_block(d, gb, block, n, 1, 0, NULL);
}

// wmv2_decode_mb (I and P pictures; J-frames never get here).
static inline int wmv2_decode_mb(MsmDecoder* d, mp4::GetBits* gb) {
    int cbp, code;

    if (d->pict_type == PICT_P) {
        if (d->mb_skip[d->mb_y * d->mb_stride + d->mb_x]) {
            d->mb_intra = 0;
            for (int i = 0; i < 6; i++) d->block_last_index[i] = -1;
            d->cur_mvx = d->cur_mvy = 0;
            d->mb_skipped = 1;
            d->hshift = 0;
            return MSM_OK;
        }
        if (mp4::get_bits_left(gb) <= 0)
            return MSM_ERR;

        code = msm_vlc_slow(gb, ff_wmv2_inter_table[d->cbp_table_index], 128, 32);
        if (code < 0) return MSM_ERR;
        d->mb_intra = (~code & 0x40) >> 6;
        cbp = code & 0x3f;
    } else {
        d->mb_intra = 1;
        if (mp4::get_bits_left(gb) <= 0)
            return MSM_ERR;
        code = mp4::get_vlc2(gb, d->mb_i.table, d->mb_i.bits);
        if (code < 0) return MSM_ERR;
        /* predict coded block pattern */
        cbp = 0;
        for (int i = 0; i < 6; i++) {
            int val = (code >> (5 - i)) & 1;
            if (i < 4) {
                uint8_t* coded_val;
                int pred   = msm_coded_block_pred(d, i, &coded_val);
                val        = val ^ pred;
                *coded_val = (uint8_t)val;
            }
            cbp |= val << (5 - i);
        }
    }

    memset(d->block, 0, sizeof(d->block));
    if (!d->mb_intra) {
        int mx, my;
        wmv2_pred_motion(d, gb, &mx, &my);

        if (cbp) {
            if (d->per_mb_rl_table) {
                d->rl_table_index        = decode012(gb);
                d->rl_chroma_table_index = d->rl_table_index;
            }

            if (d->abt_flag && d->per_mb_abt) {
                d->per_block_abt = (int)mp4::get_bits1(gb);
                if (!d->per_block_abt)
                    d->abt_type = decode012(gb);
            } else
                d->per_block_abt = 0;
        }

        wmv2_decode_motion(d, gb, &mx, &my);
        d->cur_mvx = mx;
        d->cur_mvy = my;

        for (int i = 0; i < 6; i++)
            if (wmv2_decode_inter_block(d, gb, d->block[i], i, (cbp >> (5 - i)) & 1) < 0)
                return MSM_ERR;
    } else {
        d->ac_pred = (int)mp4::get_bits1(gb);
        if (d->per_mb_rl_table && cbp) {
            d->rl_table_index        = decode012(gb);
            d->rl_chroma_table_index = d->rl_table_index;
        }
        for (int i = 0; i < 6; i++)
            if (msm_decode_block(d, gb, d->block[i], i, (cbp >> (5 - i)) & 1, 1, NULL) < 0)
                return MSM_ERR;
    }
    return MSM_OK;
}

// ff_mspel_motion (luma mspel 16x16 + chroma half-pel 8x8). Edge emulation is a
// clamped fetch into a local window — the pixels FFmpeg's emulated_edge_mc yields.
static inline void wmv2_mspel_motion(MsmDecoder* d, int motion_x, int motion_y) {
    const int ys = d->yStride, cs = d->cStride;
    uint8_t* dest_y = d->y + (size_t)(d->mb_y * 16) * ys + d->mb_x * 16;
    int dxy, mx, my, src_x, src_y;

    dxy   = ((motion_y & 1) << 1) | (motion_x & 1);
    dxy   = 2 * dxy + d->hshift;
    src_x = d->mb_x * 16 + (motion_x >> 1);
    src_y = d->mb_y * 16 + (motion_y >> 1);

    /* WARNING: do no forget half pels */
    src_x = msm_clampi(src_x, -16, d->width);
    src_y = msm_clampi(src_y, -16, d->height);

    if (src_x <= -16 || src_x >= d->width)
        dxy &= ~3;
    if (src_y <= -16 || src_y >= d->height)
        dxy &= ~4;

    const int h_edge_pos = d->codedW, v_edge_pos = d->codedH;
    const uint8_t* ptr = d->refY + (ptrdiff_t)src_y * ys + src_x;
    int ps = ys;
    uint8_t emu_buf[19 * 19];
    if (src_x < 1 || src_y < 1 || src_x + 17 >= h_edge_pos || src_y + 16 + 1 >= v_edge_pos) {
        for (int r = 0; r < 19; r++) {
            const uint8_t* row = d->refY + (size_t)msm_clampi(src_y - 1 + r, 0, v_edge_pos - 1) * ys;
            for (int c = 0; c < 19; c++)
                emu_buf[r * 19 + c] = row[msm_clampi(src_x - 1 + c, 0, h_edge_pos - 1)];
        }
        ptr = emu_buf + 1 + 19;
        ps  = 19;
    }

    wmv2_put_mspel8(dxy, dest_y,              ys, ptr,              ps);
    wmv2_put_mspel8(dxy, dest_y + 8,          ys, ptr + 8,          ps);
    wmv2_put_mspel8(dxy, dest_y + 8 * ys,     ys, ptr + 8 * ps,     ps);
    wmv2_put_mspel8(dxy, dest_y + 8 + 8 * ys, ys, ptr + 8 + 8 * ps, ps);

    dxy = 0;
    if ((motion_x & 3) != 0)
        dxy |= 1;
    if ((motion_y & 3) != 0)
        dxy |= 2;
    mx = motion_x >> 2;
    my = motion_y >> 2;

    src_x = d->mb_x * 8 + mx;
    src_y = d->mb_y * 8 + my;
    src_x = msm_clampi(src_x, -8, d->width >> 1);
    if (src_x == (d->width >> 1))
        dxy &= ~1;
    src_y = msm_clampi(src_y, -8, d->height >> 1);
    if (src_y == (d->height >> 1))
        dxy &= ~2;

    // pix_op[1][dxy]: 8x8 half-pel, rounding per no_rounding. FFmpeg emulates the
    // chroma edge only when it emulated luma; without luma emulation the 9x9 chroma
    // read lies inside the plane, so the clamped fetch equals its plain read.
    const int rb = 1 - d->no_rounding;
    const size_t coff = (size_t)(d->mb_y * 8) * cs + d->mb_x * 8;
    msm_mc_block(d->u + coff, cs, d->refU, cs, d->codedW >> 1, d->codedH >> 1, 8, 8, src_x, src_y, dxy, rb);
    msm_mc_block(d->v + coff, cs, d->refV, cs, d->codedW >> 1, d->codedH >> 1, 8, 8, src_x, src_y, dxy, rb);
}

// wmv2_add_block / ff_wmv2_add_mb: the inter residual by ABT type.
static inline void wmv2_add_block(MsmDecoder* d, uint8_t* dst, int stride, int n) {
    if (d->block_last_index[n] < 0) return;
    int16_t* block1 = d->block[n];
    switch (d->abt_type_table[n]) {
    case 0:
        wmv2_idct_add(dst, stride, block1);
        break;
    case 1:
        simple_idct84_add(dst, stride, block1);
        simple_idct84_add(dst + 4 * stride, stride, d->abt_block2[n]);
        memset(d->abt_block2[n], 0, sizeof(d->abt_block2[n]));
        break;
    default:   // 2
        simple_idct48_add(dst, stride, block1);
        simple_idct48_add(dst + 4, stride, d->abt_block2[n]);
        memset(d->abt_block2[n], 0, sizeof(d->abt_block2[n]));
        break;
    }
}

// ff_mpv_reconstruct_mb for WMV2.
static inline void wmv2_reconstruct_mb(MsmDecoder* d) {
    if (d->mb_intra) { msm_reconstruct_mb(d); return; }
    if (d->mspel) wmv2_mspel_motion(d, d->cur_mvx, d->cur_mvy);
    else          msm_mc_pred(d, d->cur_mvx, d->cur_mvy);
    const int ys = d->yStride, cs = d->cStride;
    uint8_t* dest_y = d->y + (size_t)(d->mb_y * 16) * ys + d->mb_x * 16;
    const size_t coff = (size_t)(d->mb_y * 8) * cs + d->mb_x * 8;
    wmv2_add_block(d, dest_y,               ys, 0);
    wmv2_add_block(d, dest_y + 8,           ys, 1);
    wmv2_add_block(d, dest_y + 8 * ys,      ys, 2);
    wmv2_add_block(d, dest_y + 8 + 8 * ys,  ys, 3);
    wmv2_add_block(d, d->u + coff, cs, 4);
    wmv2_add_block(d, d->v + coff, cs, 5);
}

// ff_h263_loop_filter for the current MB (run right after its reconstruction).
// The quantizer is per-picture in WMV2 and chroma_qscale_table is the identity, so
// a neighbor's qscale_table entry is the picture qscale; a skipped MB counts as 0.
static inline void wmv2_loop_filter_mb(MsmDecoder* d) {
    const int linesize = d->yStride, uvlinesize = d->cStride;
    const int xy = d->mb_y * d->mb_stride + d->mb_x;
    const uint8_t* skip = d->mb_skip;
    uint8_t* dest_y  = d->y + (size_t)(d->mb_y * 16) * linesize + d->mb_x * 16;
    uint8_t* dest_cb = d->u + (size_t)(d->mb_y * 8) * uvlinesize + d->mb_x * 8;
    uint8_t* dest_cr = d->v + (size_t)(d->mb_y * 8) * uvlinesize + d->mb_x * 8;
    int qp_c;

    if (!skip[xy]) {
        qp_c = d->qscale;
        h263_v_loop_filter(dest_y + 8 * linesize,     linesize, qp_c);
        h263_v_loop_filter(dest_y + 8 * linesize + 8, linesize, qp_c);
    } else
        qp_c = 0;

    if (d->mb_y) {
        int qp_dt, qp_tt, qp_tc;

        if (skip[xy - d->mb_stride])
            qp_tt = 0;
        else
            qp_tt = d->qscale;

        if (qp_c)
            qp_tc = qp_c;
        else
            qp_tc = qp_tt;

        if (qp_tc) {
            const int chroma_qp = qp_tc;
            h263_v_loop_filter(dest_y,     linesize, qp_tc);
            h263_v_loop_filter(dest_y + 8, linesize, qp_tc);

            h263_v_loop_filter(dest_cb, uvlinesize, chroma_qp);
            h263_v_loop_filter(dest_cr, uvlinesize, chroma_qp);
        }

        if (qp_tt)
            h263_h_loop_filter(dest_y - 8 * linesize + 8, linesize, qp_tt);

        if (d->mb_x) {
            if (qp_tt || skip[xy - 1 - d->mb_stride])
                qp_dt = qp_tt;
            else
                qp_dt = d->qscale;

            if (qp_dt) {
                const int chroma_qp = qp_dt;
                h263_h_loop_filter(dest_y  - 8 * linesize,   linesize,   qp_dt);
                h263_h_loop_filter(dest_cb - 8 * uvlinesize, uvlinesize, chroma_qp);
                h263_h_loop_filter(dest_cr - 8 * uvlinesize, uvlinesize, chroma_qp);
            }
        }
    }

    if (qp_c) {
        h263_h_loop_filter(dest_y + 8, linesize, qp_c);
        if (d->mb_y + 1 == d->mb_height)
            h263_h_loop_filter(dest_y + 8 * linesize + 8, linesize, qp_c);
    }

    if (d->mb_x) {
        int qp_lc;
        if (qp_c || skip[xy - 1])
            qp_lc = qp_c;
        else
            qp_lc = d->qscale;

        if (qp_lc) {
            h263_h_loop_filter(dest_y, linesize, qp_lc);
            if (d->mb_y + 1 == d->mb_height) {
                const int chroma_qp = qp_lc;
                h263_h_loop_filter(dest_y + 8 * linesize, linesize, qp_lc);
                h263_h_loop_filter(dest_cb, uvlinesize, chroma_qp);
                h263_h_loop_filter(dest_cr, uvlinesize, chroma_qp);
            }
        }
    }
}

// Decode one WMV2 picture (I, P or J) in stream order. A P picture whose skip map
// skips every MB leaves the planes untouched (the previous picture stays current).
static inline int wmv2_decode_frame(MsmDecoder* d, const uint8_t* bytes, int len) {
    if (!d->y) { if (msm_alloc(d) != MSM_OK) return MSM_ERR; }
    mp4::GetBits gb;
    mp4::init_get_bits(&gb, bytes, len);
    int r = wmv2_decode_picture_header(d, &gb);
    if (r == WMV2_FRAME_SKIPPED) return MSM_OK;
    if (r != MSM_OK) return MSM_ERR;

    // The previous picture becomes the motion-compensation reference.
    uint8_t* t;
    t = d->y; d->y = d->refY; d->refY = t;
    t = d->u; d->u = d->refU; d->refU = t;
    t = d->v; d->v = d->refV; d->refV = t;

    r = wmv2_decode_secondary_picture_header(d, &gb);
    if (r < 0) return MSM_ERR;
    if (r == 1) return MSM_OK;                  // J-frame, decoded by IntraX8

    msm_set_qscale(d, d->qscale);
    if (d->pict_type == PICT_P) msm_build_inter_rl(d, d->qscale);
    msm_clean_buffers(d);
    for (d->mb_y = 0; d->mb_y < d->mb_height; d->mb_y++) {
        d->first_slice_line = (d->slice_height <= 0) ? (d->mb_y == 0) : (d->mb_y % d->slice_height == 0);
        for (d->mb_x = 0; d->mb_x < d->mb_width; d->mb_x++) {
            msm_init_block_index(d);
            d->mb_skipped = 0;
            if (wmv2_decode_mb(d, &gb) < 0) return MSM_ERR;
            msm_update_motion_val(d);
            wmv2_reconstruct_mb(d);
            if (d->loop_filter) wmv2_loop_filter_mb(d);
        }
    }
    return MSM_OK;
}

} // namespace msm

#endif // WMV2_DECODE_H
