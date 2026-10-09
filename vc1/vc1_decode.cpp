/*
    VC-1 / WMV3 decoder — C ABI + self-test hooks for the WASM build. See
    vc1_decode.h for provenance/license and the milestone status. The decode surface
    will mirror the mpeg4/msmpeg4 ports (open/close/decode + plane pointers); this
    milestone exposes the sequence-header and picture-header parse (incl. bitplane
    decoding) for conformance against ffmpeg.
*/

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "vc1_decode.h"

#if defined(__EMSCRIPTEN__)
#include <emscripten.h>
#define KEEP EMSCRIPTEN_KEEPALIVE
#else
#define KEEP
#endif

using namespace vc1;

extern "C" {

KEEP void vc1_close(int handle);   // forward decl (used by vc1_open's error path)

KEEP int vc1_open(int width, int height) {
    VC1Context* v = (VC1Context*)calloc(1, sizeof(VC1Context));
    if (!v) return 0;
    v->width = width; v->height = height;
    v->mb_width  = (width + 15) >> 4;
    v->mb_height = (height + 15) >> 4;
    v->mb_stride = v->mb_width + 1;
    v->b8_stride = v->mb_width * 2 + 1;
    int plane = v->mb_stride * v->mb_height;
    v->mv_type_mb_plane = (uint8_t*)calloc(plane, 1);
    v->mbskip_table     = (uint8_t*)calloc(plane, 1);
    v->acpred_plane     = (uint8_t*)calloc(plane, 1);   // AP I: per-MB AC-pred flag
    v->over_flags_plane = (uint8_t*)calloc(plane, 1);   // AP I: condover SELECT flag
    v->qscale_table     = (int16_t*)calloc(plane, sizeof(int16_t));  // AP: per-MB mquant

    v->direct_mb_plane  = (uint8_t*)calloc(plane, 1);   // B: per-MB direct-mode flag

    // Pictures (MB-aligned), like the msmpeg4 port: three I/P pictures (last_pic,
    // next_pic and the one being decoded) plus the B/BI output picture.
    v->codedW  = v->mb_width * 16;  v->codedH = v->mb_height * 16;
    v->yStride = v->codedW;         v->cStride = v->codedW >> 1;
    size_t ysz = (size_t)v->yStride * v->codedH, csz = (size_t)v->cStride * (v->codedH >> 1);
    for (int k = 0; k < 3; k++) {
        v->picY[k] = (uint8_t*)calloc(ysz, 1);
        v->picU[k] = (uint8_t*)calloc(csz, 1);
        v->picV[k] = (uint8_t*)calloc(csz, 1);
    }
    v->bpY = (uint8_t*)calloc(ysz, 1);
    v->bpU = (uint8_t*)calloc(csz, 1);
    v->bpV = (uint8_t*)calloc(csz, 1);
    v->idx_last = 0; v->idx_next = 1; v->nrefs = 0;
    v->planeY = v->picY[0]; v->planeU = v->picU[0]; v->planeV = v->picV[0];

    // Rolling block buffer for the delayed overlap + put (n_allocated_blks).
    v->n_blks = v->mb_width + 2;
    v->blk = (int16_t (*)[6][64])calloc((size_t)v->n_blks, sizeof(int16_t) * 6 * 64);
    vc1_init_lut(32, 0, v->last_luty, v->last_lutuv, 0);

    // Prediction grids (mpegvideo geometry, mirrors the msmpeg4 port).
    int y_size  = v->b8_stride * (2 * v->mb_height + 1);
    int c_size  = v->mb_stride * (v->mb_height + 1);
    int yc_size = y_size + 2 * c_size;
    v->dc_off   = v->b8_stride + 1;
    int lead    = 2 * v->b8_stride + 16;
    int total   = lead + yc_size + (2 * v->b8_stride + 16);
    v->grid_total = total; v->grid_lead = lead;
    v->dc_base    = (int16_t*)calloc((size_t)total, sizeof(int16_t));
    v->ac_base    = (int16_t*)calloc((size_t)total * 16, sizeof(int16_t));
    v->coded_base = (uint8_t*)calloc((size_t)total, 1);

    // P-frame inter state: motion_val grid, per-b8 mbtype, reference planes, per-MB deblock arrays.
    v->mv_base     = (int16_t*)calloc((size_t)total * 2, sizeof(int16_t));
    v->mv1_base    = (int16_t*)calloc((size_t)total * 2, sizeof(int16_t));
    v->nmv_base    = (int16_t*)calloc((size_t)total * 2, sizeof(int16_t));
    v->mbtype_base = (uint8_t*)calloc((size_t)total, 1);
    int nmb = v->mb_width * v->mb_height;
    v->fr_cbp     = (uint32_t*)calloc((size_t)nmb, sizeof(uint32_t));
    v->fr_ttblk   = (int*)     calloc((size_t)nmb, sizeof(int));
    v->fr_isintra = (uint8_t*) calloc((size_t)nmb, 1);
    v->fr_lumamv  = (int16_t(*)[2])calloc((size_t)nmb, sizeof(int16_t) * 2);
    int pics_ok = v->bpY && v->bpU && v->bpV;
    for (int k = 0; k < 3; k++) pics_ok = pics_ok && v->picY[k] && v->picU[k] && v->picV[k];
    if (!v->mv_type_mb_plane || !v->mbskip_table || !v->acpred_plane || !v->over_flags_plane ||
        !v->qscale_table || !v->direct_mb_plane || !pics_ok || !v->blk ||
        !v->dc_base || !v->ac_base || !v->coded_base || !v->mv_base || !v->mv1_base || !v->nmv_base ||
        !v->mbtype_base || !v->fr_cbp || !v->fr_ttblk || !v->fr_isintra || !v->fr_lumamv) {
        vc1_close((int)(intptr_t)v);
        return 0;
    }
    v->mv_val  = v->mv_base     + (size_t)(lead + v->dc_off) * 2;
    v->mv1_val = v->mv1_base    + (size_t)(lead + v->dc_off) * 2;
    v->nmv_val = v->nmv_base    + (size_t)(lead + v->dc_off) * 2;
    v->mvb     = v->nmv_val;
    v->mbtype  = v->mbtype_base +  lead + v->dc_off;
    v->dc_val      = v->dc_base    + lead + v->dc_off;
    v->ac_val      = v->ac_base    + (size_t)(lead + v->dc_off) * 16;
    v->coded_block = v->coded_base + lead + v->dc_off;

    v->block_wrap[0] = v->block_wrap[1] = v->block_wrap[2] = v->block_wrap[3] = v->b8_stride;
    v->block_wrap[4] = v->block_wrap[5] = v->mb_stride;
    v->left_blk_sh = 0; v->top_blk_sh = 3;

    // Transposed scan tables (ff_vc1_init_transposed_scantables): zz_8x8[k] =
    // transpose(ff_wmv1_scantable[k]); [1]=intra [2]=intra-h [3]=intra-v.
    for (int i = 0; i < 64; i++)
        for (int k = 0; k < 4; k++) {
            int x = msm::ff_wmv1_scantable[k][i];
            v->zz_8x8[k][i] = (uint8_t)((x >> 3) | ((x & 7) << 3));
        }
    return (int)(intptr_t)v;
}

KEEP void vc1_close(int handle) {
    VC1Context* v = (VC1Context*)(intptr_t)handle;
    if (!v) return;
    free(v->mv_type_mb_plane);
    free(v->mbskip_table);
    free(v->acpred_plane);
    free(v->over_flags_plane);
    free(v->qscale_table);
    free(v->direct_mb_plane);
    for (int k = 0; k < 3; k++) { free(v->picY[k]); free(v->picU[k]); free(v->picV[k]); }
    free(v->bpY); free(v->bpU); free(v->bpV);
    free(v->blk);
    free(v->dc_base); free(v->ac_base); free(v->coded_base);
    free(v->mv_base); free(v->mv1_base); free(v->nmv_base); free(v->mbtype_base);
    free(v->fr_cbp); free(v->fr_ttblk); free(v->fr_isintra); free(v->fr_lumamv);
    free(v);
}

// Parse the Simple/Main sequence header (STRUCT_C extradata). Writes the decoded
// fields into out[16]; returns 0 on success. `bytes` padded with >= 8 zero bytes.
KEEP int vc1_test_seqhdr(int handle, const uint8_t* bytes, int len, int* out) {
    VC1Context* v = (VC1Context*)(intptr_t)handle;
    if (!v || !out) return VC1_ERR;
    mp4::GetBits gb;
    mp4::init_get_bits(&gb, bytes, len);
    int r = vc1_decode_sequence_header(v, &gb);
    out[0]  = v->profile;         out[1]  = v->frmrtq_postproc;
    out[2]  = v->bitrtq_postproc; out[3]  = v->loop_filter;
    out[4]  = v->multires;        out[5]  = v->fastuvmc;
    out[6]  = v->extended_mv;     out[7]  = v->rangered;
    out[8]  = v->vstransform;     out[9]  = v->overlap;
    out[10] = v->resync_marker;   out[11] = v->dquant;
    out[12] = v->quantizer_mode;  out[13] = v->max_b_frames;
    out[14] = v->finterpflag;     out[15] = (r == VC1_OK) ? 1 : 0;
    return r;
}

// Parse a Simple/Main PICTURE header (one demuxed frame payload; bit 0 = start of
// the picture header). The sequence header must already have been parsed into this
// handle. Writes decoded fields into out[24]; returns 0 on success. `bytes` padded
// with >= 8 trailing zero bytes.
KEEP int vc1_test_pichdr(int handle, const uint8_t* bytes, int len, int* out) {
    VC1Context* v = (VC1Context*)(intptr_t)handle;
    if (!v || !out) return VC1_ERR;
    mp4::GetBits gb;
    mp4::init_get_bits(&gb, bytes, len);
    int r = vc1_parse_frame_header(v, &gb);
    out[0]  = v->pict_type;            // 1=I, 2=P
    out[1]  = v->pqindex;
    out[2]  = v->pq;
    out[3]  = v->halfpq;
    out[4]  = v->mvrange;
    out[5]  = v->c_ac_table_index;
    out[6]  = v->y_ac_table_index;
    out[7]  = v->dc_table_index;
    out[8]  = v->mv_mode;
    out[9]  = v->mv_table_index;
    out[10] = v->cbptab;
    out[11] = v->has_mvtype_plane;
    out[12] = v->mvtype_imode;
    out[13] = v->mvtype_invert;
    out[14] = v->skip_imode;
    out[15] = v->skip_invert;
    out[16] = v->ttfrm;
    out[17] = v->rnd;
    out[18] = mp4::get_bits_count(&gb);   // bits consumed by the header
    out[23] = (r == VC1_OK) ? 1 : 0;
    return r;
}

// Decode one Simple/Main picture payload (I, P, B or BI — from its picture header).
// Reference management follows ffmpeg's mpegvideo: an I/P picture is decoded into the
// oldest of the three reference slots and becomes next_pic (the previous next_pic
// becomes last_pic); a B/BI picture is decoded into its own slot from last_pic
// (forward) and next_pic (backward) and is never a reference. The decoded picture is
// what vc1_plane_* return — DECODE order; the reader maps display order onto it.
// Returns VC1_OK, VC1_NO_PICTURE (output unchanged), VC1_ERR, or an unsupported-
// feature code (VC1_ERR_RESPIC / VC1_ERR_X8; the references are then stale until
// the next decodable I picture, and every P/B until then reports the same code).
static int vc1_decode_picture(VC1Context* v, const uint8_t* bytes, int len) {
    mp4::GetBits gb;
    mp4::init_get_bits(&gb, bytes, len);
    int r = vc1_parse_frame_header(v, &gb);
    if (r != VC1_OK) return r;

    if (v->pict_type == PICT_B) {
        if (v->ref_broken) return v->ref_broken;
        if (v->nrefs < 2) return VC1_NO_PICTURE;   // skip B-frames without reference frames
        if (v->x8_type) return VC1_ERR_X8;
        v->refY  = v->picY[v->idx_last]; v->refU  = v->picU[v->idx_last]; v->refV  = v->picV[v->idx_last];
        v->nextY = v->picY[v->idx_next]; v->nextU = v->picU[v->idx_next]; v->nextV = v->picV[v->idx_next];
        v->planeY = v->bpY; v->planeU = v->bpU; v->planeV = v->bpV;
        v->mvb = v->mv1_val;
        return v->bi_type ? vc1_decode_i_frame(v, &gb) : vc1_decode_b_frame(v, &gb);
    }

    if (v->pict_type == PICT_I) {
        if (v->x8_type) { v->ref_broken = VC1_ERR_X8;     return VC1_ERR_X8; }
        if (v->respic)  { v->ref_broken = VC1_ERR_RESPIC; return VC1_ERR_RESPIC; }
    } else {
        if (v->ref_broken) return v->ref_broken;
        if (v->respic)  { v->ref_broken = VC1_ERR_RESPIC; return VC1_ERR_RESPIC; }
        if (!v->nrefs) return VC1_ERR;               // P without a reference picture
    }
    int dst = v->idx_last;                           // oldest reference slot is free now
    v->refY = v->picY[v->idx_next]; v->refU = v->picU[v->idx_next]; v->refV = v->picV[v->idx_next];
    v->planeY = v->picY[dst]; v->planeU = v->picU[dst]; v->planeV = v->picV[dst];
    v->mvb = v->nmv_val;
    r = (v->pict_type == PICT_I) ? vc1_decode_i_frame(v, &gb) : vc1_decode_p_frame(v, &gb);
    v->idx_last = v->idx_next;
    v->idx_next = dst;
    if (v->nrefs < 2) v->nrefs++;
    if (v->pict_type == PICT_I && r == VC1_OK) v->ref_broken = 0;
    return r;
}

// Test hooks kept for the I+P conformance tests: both decode whatever picture the
// payload carries (vc1_decode_picture). `bytes` padded with >= 8 zero bytes.
KEEP int vc1_decode_iframe(int handle, const uint8_t* bytes, int len) {
    VC1Context* v = (VC1Context*)(intptr_t)handle;
    if (!v) return VC1_ERR;
    return vc1_decode_picture(v, bytes, len);
}
KEEP int vc1_decode_pframe(int handle, const uint8_t* bytes, int len) {
    VC1Context* v = (VC1Context*)(intptr_t)handle;
    if (!v) return VC1_ERR;
    return vc1_decode_picture(v, bytes, len);
}

// Parse the STRUCT_C sequence header (from ASF extradata) into the handle. 0 on ok.
KEEP int vc1_seqhdr(int handle, const uint8_t* bytes, int len) {
    VC1Context* v = (VC1Context*)(intptr_t)handle;
    if (!v) return VC1_ERR;
    mp4::GetBits gb;
    mp4::init_get_bits(&gb, bytes, len);
    return vc1_decode_sequence_header(v, &gb) == VC1_OK ? 0 : VC1_ERR;
}

// Decode one frame payload — the single entry the VC1Decoder.js factory drives.
// `bytes` padded >= 8 zeros. Return codes as vc1_decode_picture.
KEEP int vc1_decode_frame(int handle, const uint8_t* bytes, int len) {
    VC1Context* v = (VC1Context*)(intptr_t)handle;
    if (!v) return VC1_ERR;
    return vc1_decode_picture(v, bytes, len);
}

// Picture type of the last parsed picture header: 1 = I, 2 = P, 3 = B (BI included).
KEEP int vc1_pict_type(int handle) { return ((VC1Context*)(intptr_t)handle)->pict_type; }

// MAXBFRAMES from the sequence header (0 = no B pictures, output is never delayed).
KEEP int vc1_max_b_frames(int handle) { return ((VC1Context*)(intptr_t)handle)->max_b_frames; }

// Point vc1_plane_* at a held picture: 0 = the picture just decoded, 1 = last_pic (the
// older reference), 2 = next_pic (the newest reference). VC1Decoder.js uses it to give
// ffmpeg's display order: a B/BI picture shows at once, an I/P shows the previous I/P,
// and the end of the stream shows the newest reference.
KEEP void vc1_select_output(int handle, int which) {
    VC1Context* v = (VC1Context*)(intptr_t)handle;
    if (!v || which == 0) return;
    int k = which == 1 ? v->idx_last : v->idx_next;
    v->planeY = v->picY[k]; v->planeU = v->picU[k]; v->planeV = v->picV[k];
}

// Advanced Profile (WVC1) header conformance hook. Feeds a raw .vc1 elementary
// stream (or extradata+frame): splits BDUs by start code, unescapes each, parses the
// sequence + entry-point headers and the FIRST frame header, and dumps the decoded
// fields to out[40] for field-exact comparison against ffmpeg `-v debug`. Returns 0
// once a frame header has been parsed. The handle must be opened at the coded size.
KEEP int vc1_test_adv_headers(int handle, const uint8_t* bytes, int len, int* out) {
    VC1Context* v = (VC1Context*)(intptr_t)handle;
    if (!v || !out || len <= 0) return VC1_ERR;
    uint8_t* scratch = (uint8_t*)malloc((size_t)len + 8);
    if (!scratch) return VC1_ERR;

    const uint8_t* end = bytes + len;
    const uint8_t* start = vc1_find_next_marker(bytes, end);
    int got_seq = 0, got_entry = 0, got_frame = 0;

    while (start < end && !got_frame) {
        const uint8_t* next = vc1_find_next_marker(start + 4, end);
        int size = (int)(next - start - 4);
        if (size > 0) {
            int usize = vc1_unescape_buffer(start + 4, size, scratch);
            memset(scratch + usize, 0, 8);                 // trailing pad for the bit reader
            int type = start[3];                           // low byte of 0x000001xx
            mp4::GetBits gb;
            mp4::init_get_bits(&gb, scratch, usize);
            if (type == 0x0F) {                            // VC1_CODE_SEQHDR
                v->profile = (int)mp4::get_bits(&gb, 2);
                if (v->profile == PROFILE_ADVANCED &&
                    vc1_decode_sequence_header_adv(v, &gb) == VC1_OK) got_seq = 1;
            } else if (type == 0x0E) {                     // VC1_CODE_ENTRYPOINT
                if (vc1_decode_entry_point(v, &gb) == VC1_OK) got_entry = 1;
            } else if (type == 0x0D) {                     // VC1_CODE_FRAME
                if (vc1_parse_frame_header_adv(v, &gb) == VC1_OK) {
                    out[38] = mp4::get_bits_count(&gb);
                    got_frame = 1;
                }
            }
        }
        start = next;
    }
    free(scratch);

    out[0]  = v->level;            out[1]  = v->chromaformat;
    out[2]  = v->frmrtq_postproc;  out[3]  = v->bitrtq_postproc;
    out[4]  = v->postprocflag;     out[5]  = v->broadcast;
    out[6]  = v->interlace;        out[7]  = v->tfcntrflag;
    out[8]  = v->finterpflag;      out[9]  = v->max_coded_width;
    out[10] = v->max_coded_height; out[11] = v->hrd_param_flag;
    out[12] = v->broken_link;      out[13] = v->closed_entry;
    out[14] = v->panscanflag;      out[15] = v->refdist_flag;
    out[16] = v->loop_filter;      out[17] = v->fastuvmc;
    out[18] = v->extended_mv;      out[19] = v->dquant;
    out[20] = v->vstransform;      out[21] = v->overlap;
    out[22] = v->quantizer_mode;   out[23] = v->width;
    out[24] = v->height;           out[25] = v->range_mapy_flag;
    out[26] = v->range_mapuv_flag; out[27] = v->fcm;
    out[28] = v->pict_type;        out[29] = v->pqindex;
    out[30] = v->pq;               out[31] = v->halfpq;
    out[32] = v->pquantizer;       out[33] = v->rnd;
    out[34] = v->acpred_imode;     out[35] = v->acpred_invert;
    out[36] = v->condover;         out[37] = v->rptfrm;
    out[39] = (got_seq && got_entry && got_frame) ? 1 : 0;
    return out[39] ? 0 : VC1_ERR;
}

// Decode the n-th (0-based) FRAME BDU of an Advanced-Profile (WVC1) elementary
// stream into the handle's planes. Walks BDUs from the start, parsing the sequence
// and entry-point headers as encountered (so per-frame state is correct), unescapes
// each payload, and decodes the target I-frame. The handle must be opened at the
// coded size. Returns 0 on success. (Progressive I-frames only for now.)
KEEP int vc1_decode_adv(int handle, const uint8_t* bytes, int len, int frame_index) {
    VC1Context* v = (VC1Context*)(intptr_t)handle;
    if (!v || len <= 0) return VC1_ERR;
    uint8_t* scratch = (uint8_t*)malloc((size_t)len + 8);
    if (!scratch) return VC1_ERR;

    const uint8_t* endp = bytes + len;
    const uint8_t* start = vc1_find_next_marker(bytes, endp);
    int seen_frames = 0, rc = VC1_ERR;

    while (start < endp) {
        const uint8_t* next = vc1_find_next_marker(start + 4, endp);
        int size = (int)(next - start - 4);
        if (size > 0) {
            int usize = vc1_unescape_buffer(start + 4, size, scratch);
            memset(scratch + usize, 0, 8);
            int type = start[3];
            mp4::GetBits gb;
            mp4::init_get_bits(&gb, scratch, usize);
            if (type == 0x0F) {                            // SEQHDR
                v->profile = (int)mp4::get_bits(&gb, 2);
                if (v->profile != PROFILE_ADVANCED ||
                    vc1_decode_sequence_header_adv(v, &gb) != VC1_OK) { rc = VC1_ERR; break; }
            } else if (type == 0x0E) {                     // ENTRYPOINT
                if (vc1_decode_entry_point(v, &gb) != VC1_OK) { rc = VC1_ERR; break; }
            } else if (type == 0x0D) {                     // FRAME
                if (seen_frames == frame_index) {
                    if (vc1_parse_frame_header_adv(v, &gb) != VC1_OK ||
                        v->pict_type != PICT_I) { rc = VC1_ERR; break; }
                    rc = vc1_decode_i_frame_adv(v, &gb);
                    break;
                }
                seen_frames++;
            }
        }
        start = next;
    }
    free(scratch);
    return rc;
}

KEEP int vc1_coded_h(int handle) { return ((VC1Context*)(intptr_t)handle)->codedH; }
KEEP int vc1_plane_y(int handle) { return (int)(intptr_t)((VC1Context*)(intptr_t)handle)->planeY; }
KEEP int vc1_plane_u(int handle) { return (int)(intptr_t)((VC1Context*)(intptr_t)handle)->planeU; }
KEEP int vc1_plane_v(int handle) { return (int)(intptr_t)((VC1Context*)(intptr_t)handle)->planeV; }
KEEP int vc1_stride_y(int handle) { return ((VC1Context*)(intptr_t)handle)->yStride; }
KEEP int vc1_stride_c(int handle) { return ((VC1Context*)(intptr_t)handle)->cStride; }

} // extern "C"
