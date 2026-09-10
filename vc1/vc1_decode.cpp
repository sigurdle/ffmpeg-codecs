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

    // Output planes (MB-aligned), like the msmpeg4 port.
    v->codedW  = v->mb_width * 16;  v->codedH = v->mb_height * 16;
    v->yStride = v->codedW;         v->cStride = v->codedW >> 1;
    v->planeY = (uint8_t*)calloc((size_t)v->yStride * v->codedH, 1);
    v->planeU = (uint8_t*)calloc((size_t)v->cStride * (v->codedH >> 1), 1);
    v->planeV = (uint8_t*)calloc((size_t)v->cStride * (v->codedH >> 1), 1);

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
    v->mbtype_base = (uint8_t*)calloc((size_t)total, 1);
    v->refY = (uint8_t*)calloc((size_t)v->yStride * v->codedH, 1);
    v->refU = (uint8_t*)calloc((size_t)v->cStride * (v->codedH >> 1), 1);
    v->refV = (uint8_t*)calloc((size_t)v->cStride * (v->codedH >> 1), 1);
    int nmb = v->mb_width * v->mb_height;
    v->fr_cbp     = (uint32_t*)calloc((size_t)nmb, sizeof(uint32_t));
    v->fr_ttblk   = (int*)     calloc((size_t)nmb, sizeof(int));
    v->fr_isintra = (uint8_t*) calloc((size_t)nmb, 1);
    v->fr_lumamv  = (int16_t(*)[2])calloc((size_t)nmb, sizeof(int16_t) * 2);
    if (!v->mv_type_mb_plane || !v->mbskip_table || !v->acpred_plane || !v->over_flags_plane ||
        !v->qscale_table ||
        !v->planeY || !v->planeU || !v->planeV ||
        !v->dc_base || !v->ac_base || !v->coded_base || !v->mv_base || !v->mbtype_base ||
        !v->refY || !v->refU || !v->refV || !v->fr_cbp || !v->fr_ttblk || !v->fr_isintra || !v->fr_lumamv) {
        vc1_close((int)(intptr_t)v);
        return 0;
    }
    v->mv_val  = v->mv_base     + (size_t)(lead + v->dc_off) * 2;
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
    free(v->planeY); free(v->planeU); free(v->planeV);
    free(v->dc_base); free(v->ac_base); free(v->coded_base);
    free(v->mv_base); free(v->mbtype_base);
    free(v->refY); free(v->refU); free(v->refV);
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

// Decode one Simple/Main I-frame payload (picture header + block decode) into the
// handle's planes. Sequence header must already be parsed. `bytes` padded with >= 8
// zero bytes. Returns 0 on success. Plane data read via vc1_plane_* / vc1_stride_*.
KEEP int vc1_decode_iframe(int handle, const uint8_t* bytes, int len) {
    VC1Context* v = (VC1Context*)(intptr_t)handle;
    if (!v) return VC1_ERR;
    mp4::GetBits gb;
    mp4::init_get_bits(&gb, bytes, len);
    if (vc1_parse_frame_header(v, &gb) != VC1_OK) return VC1_ERR;
    if (v->pict_type != PICT_I) return VC1_ERR;
    return vc1_decode_i_frame(v, &gb);
}

// Decode one Simple/Main P-frame payload into the handle's planes. The previous
// decoded frame (in planeY/U/V) is copied to refY/U/V first (MC source). Sequence
// header must already be parsed and an I-frame decoded. `bytes` padded >= 8 zeros.
KEEP int vc1_decode_pframe(int handle, const uint8_t* bytes, int len) {
    VC1Context* v = (VC1Context*)(intptr_t)handle;
    if (!v) return VC1_ERR;
    memcpy(v->refY, v->planeY, (size_t)v->yStride * v->codedH);
    memcpy(v->refU, v->planeU, (size_t)v->cStride * (v->codedH >> 1));
    memcpy(v->refV, v->planeV, (size_t)v->cStride * (v->codedH >> 1));
    mp4::GetBits gb;
    mp4::init_get_bits(&gb, bytes, len);
    if (vc1_parse_frame_header(v, &gb) != VC1_OK) return VC1_ERR;
    if (v->pict_type != PICT_P) return VC1_ERR;
    return vc1_decode_p_frame(v, &gb);
}

// Parse the STRUCT_C sequence header (from ASF extradata) into the handle. 0 on ok.
KEEP int vc1_seqhdr(int handle, const uint8_t* bytes, int len) {
    VC1Context* v = (VC1Context*)(intptr_t)handle;
    if (!v) return VC1_ERR;
    mp4::GetBits gb;
    mp4::init_get_bits(&gb, bytes, len);
    return vc1_decode_sequence_header(v, &gb) == VC1_OK ? 0 : VC1_ERR;
}

// Decode one frame (I or P, auto-detected from the picture header) into the handle's
// planes. For P frames the previous output is copied to the reference first. This is
// the single entry ASFReader's decoder factory drives. `bytes` padded >= 8 zeros.
KEEP int vc1_decode_frame(int handle, const uint8_t* bytes, int len) {
    VC1Context* v = (VC1Context*)(intptr_t)handle;
    if (!v) return VC1_ERR;
    mp4::GetBits gb;
    mp4::init_get_bits(&gb, bytes, len);
    if (vc1_parse_frame_header(v, &gb) != VC1_OK) return VC1_ERR;
    if (v->pict_type == PICT_I) return vc1_decode_i_frame(v, &gb);
    if (v->pict_type == PICT_P) {
        memcpy(v->refY, v->planeY, (size_t)v->yStride * v->codedH);
        memcpy(v->refU, v->planeU, (size_t)v->cStride * (v->codedH >> 1));
        memcpy(v->refV, v->planeV, (size_t)v->cStride * (v->codedH >> 1));
        return vc1_decode_p_frame(v, &gb);
    }
    return VC1_ERR;
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
