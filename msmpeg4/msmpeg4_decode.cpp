/*
    Microsoft MPEG-4 / WMV1-2 decoder — C ABI + self-test hooks for the WASM build.
    See msmpeg4_decode.h for provenance/license and the milestone status.

    The decode surface mirrors the mpeg4 port's C ABI (open/close/decode_frame +
    plane pointers) so the JS wrapper (MsMpeg4Decoder.js) can drive it the same way.
    Block decode is staged in behind the header parse; the header self-test below is
    the current milestone's conformance hook (validated against ffmpeg frame types).
*/

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "msmpeg4_decode.h"

#if defined(__EMSCRIPTEN__)
#include <emscripten.h>
#define KEEP EMSCRIPTEN_KEEPALIVE
#else
#define KEEP
#endif

using namespace msm;

extern "C" {

KEEP int msm_open(int version, int width, int height) {
    MsmDecoder* d = (MsmDecoder*)calloc(1, sizeof(MsmDecoder));
    if (!d) return 0;
    d->version   = version;
    d->width     = width;
    d->height    = height;
    d->mb_width  = (width  + 15) >> 4;
    d->mb_height = (height + 15) >> 4;
    d->slice_height = d->mb_height;   // avoid 1/0 before the first keyframe
    return (int)(intptr_t)d;
}

KEEP void msm_close(int handle) {
    MsmDecoder* d = (MsmDecoder*)(intptr_t)handle;
    if (d) free(d);
}

// Header-parse self-test. Parses one frame's picture header and writes
// [pict_type, qscale, rl, rl_chroma, dc, mv, per_mb_rl, slice_height, no_rounding,
// ok] into out[10]. The caller pads `bytes` with >= 8 trailing zeros (the get_bits
// window reads ahead), same contract as the mpeg4 port.
KEEP int msm_test_header(int handle, const uint8_t* bytes, int len, int* out) {
    MsmDecoder* d = (MsmDecoder*)(intptr_t)handle;
    if (!d || !out) return MSM_ERR;
    if (!d->mb_width) { d->mb_width = (d->width + 15) >> 4; d->mb_height = (d->height + 15) >> 4; }
    mp4::GetBits gb;
    mp4::init_get_bits(&gb, bytes, len);
    int r = msm_decode_picture_header(d, &gb);
    out[0] = d->pict_type; out[1] = d->qscale;
    out[2] = d->rl_table_index; out[3] = d->rl_chroma_table_index;
    out[4] = d->dc_table_index; out[5] = d->mv_table_index;
    out[6] = d->per_mb_rl_table; out[7] = d->slice_height;
    out[8] = d->no_rounding; out[9] = (r == MSM_OK) ? 1 : 0;
    return r;
}

// Decode one frame (I-frame keyframe for now). `bytes` is padded with >= 8 trailing
// zeros. Writes [codedW, codedH, yStride, cStride, pict_type, ok] into info[6].
KEEP int msm_decode(int handle, const uint8_t* bytes, int len, int* info) {
    MsmDecoder* d = (MsmDecoder*)(intptr_t)handle;
    if (!d || !info) return MSM_ERR;
    for (int i = 0; i < 6; i++) info[i] = 0;
    int r = msm_decode_frame(d, bytes, len);
    info[0] = d->codedW; info[1] = d->codedH;
    info[2] = d->yStride; info[3] = d->cStride;
    info[4] = d->pict_type; info[5] = (r == MSM_OK) ? 1 : 0;
    return r;
}

KEEP int msm_y_ptr(int handle) { MsmDecoder* d = (MsmDecoder*)(intptr_t)handle; return d ? (int)(intptr_t)d->y : 0; }
KEEP int msm_u_ptr(int handle) { MsmDecoder* d = (MsmDecoder*)(intptr_t)handle; return d ? (int)(intptr_t)d->u : 0; }
KEEP int msm_v_ptr(int handle) { MsmDecoder* d = (MsmDecoder*)(intptr_t)handle; return d ? (int)(intptr_t)d->v : 0; }

} // extern "C"
