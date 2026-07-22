/*
    MPEG-4 Part 2 Advanced Simple Profile (DivX / XviD / mp4v) + ITU-T H.263
    video decoder — C++/WebAssembly. C ABI + the test self-test hooks.

    PROVENANCE / LICENSE
    --------------------
    From-source TRANSCRIPTION of FFmpeg release/8.1's MPEG-4 Part 2 + H.263 decoder
    (libavcodec/mpeg4videodec.c, ituh263dec.c, h263dec.c, simple_idct8, the VLC/RL
    machinery, h263data.c + mpeg4data.h). FFmpeg is LGPL-2.1+; this module is an
    LGPL-derived work in its OWN wasm binary, separate from the project's
    hand-authored MPEG-1/2 + Sorenson kernels. See memory: codec-authorship-
    provenance, divx-mjpeg-port-plan; full blueprint in tmp/mpeg4_port_blueprint.md.
    Conformance oracle is the ffmpeg CLI (NOT ffmpeg-derived JS): decoded planes are
    bit-compared against `ffmpeg -i sample -f rawvideo -pix_fmt yuv420p`.

    STATUS: M1–M3 — rectangular progressive non-scalable decode is live
    (plugins/mpeg4/mp4_decode.h): I-VOP intra, P-VOP inter (half-pel MC, 1MV/4MV),
    and quarter-pel (qpel) luma MC (mp4_qpel.h). B-VOP, interlaced, sprite/GMC,
    data-partition, mpeg-quant, studio and H.263 short-header are gated to a clean
    UNSUPPORTED return and land in later milestones (M4 = B-VOP, M5 = H.263).

    -fwrapv is REQUIRED (simple_idct8, dequant, get_xbits sign arithmetic).
*/

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "simple_idct8.h"   // ported ff_simple_idct8 (own TU; ffmpeg LGPL)
#include "mp4_tables.h"     // get_bits reader + VLC/RL machinery + intra tables
#include "mp4_decode.h"     // header parse + I-VOP intra decode
#include "idct_ref.h"       // verbatim ffmpeg simple_idct8 (test cross-check oracle)

#if defined(__EMSCRIPTEN__)
#include <emscripten.h>
#define KEEP EMSCRIPTEN_KEEPALIVE
#else
#define KEEP
#endif

// info[] int32 layout returned to JS (caller-owned, reused per frame):
// [0]=codedWidth [1]=codedHeight [2]=ybufRowbytes [3]=ccbufRowbytes
// [4]=codingType (1=I,2=P,3=B) [5]=ok [6]=warnCode [7]=reserved
enum { INFO_W=0, INFO_H=1, INFO_YSTRIDE=2, INFO_CSTRIDE=3,
       INFO_CODING=4, INFO_OK=5, INFO_WARN=6, INFO_RESV=7, INFO_N=8 };

extern "C" {

KEEP int mp4v_open() {
    mp4::Mp4Decoder* d = (mp4::Mp4Decoder*)calloc(1, sizeof(mp4::Mp4Decoder));
    return (int)(intptr_t)d;
}

KEEP void mp4v_close(int handle) {
    mp4::Mp4Decoder* d = (mp4::Mp4Decoder*)(intptr_t)handle;
    if (!d) return;
    mp4::mp4_free(d);
    free(d);
}

KEEP void mp4v_reset(int handle) {
    // Intra frames are self-contained; nothing to drop. VOL config is kept.
    (void)handle;
}

KEEP void mp4v_set_simd(int on) { (void)on; }

// Decode one elementary-stream frame [inPtr, inPtr+len) into the handle's planes.
// The caller pads the input with >= 8 trailing zero bytes (the 4-byte get_bits
// window reads ahead). Writes the INFO_* fields; returns 0 on a decoded/skipped
// frame, <0 on error/unsupported.
KEEP int mp4v_decode_frame(int handle, const uint8_t* inPtr, int len, int* infoPtr) {
    mp4::Mp4Decoder* d = (mp4::Mp4Decoder*)(intptr_t)handle;
    if (!d || !infoPtr) return -1;
    for (int i = 0; i < INFO_N; i++) infoPtr[i] = 0;

    int r = mp4::mp4_decode_one(d, inPtr, len);

    if (r == mp4::MP4_OK || r == mp4::MP4_SKIP) {
        infoPtr[INFO_W]       = d->codedW;
        infoPtr[INFO_H]       = d->codedH;
        infoPtr[INFO_YSTRIDE] = d->yStride;
        infoPtr[INFO_CSTRIDE] = d->cStride;
        infoPtr[INFO_CODING]  = d->pict_type;
        infoPtr[INFO_OK]      = (r == mp4::MP4_OK) ? 1 : 0;
        if (r == mp4::MP4_SKIP) infoPtr[INFO_WARN] = 1;
        return 0;
    }
    infoPtr[INFO_OK]   = 0;
    infoPtr[INFO_WARN] = -r;
    return r;
}

// Output planes: I/P -> the just-decoded frame (y/u/v); B -> the B buffer (bY/U/V).
KEEP int mp4v_y_ptr(int handle) { mp4::Mp4Decoder* d = (mp4::Mp4Decoder*)(intptr_t)handle; return d ? (int)(intptr_t)mp4::mp4_out_y(d) : 0; }
KEEP int mp4v_u_ptr(int handle) { mp4::Mp4Decoder* d = (mp4::Mp4Decoder*)(intptr_t)handle; return d ? (int)(intptr_t)mp4::mp4_out_u(d) : 0; }
KEEP int mp4v_v_ptr(int handle) { mp4::Mp4Decoder* d = (mp4::Mp4Decoder*)(intptr_t)handle; return d ? (int)(intptr_t)mp4::mp4_out_v(d) : 0; }

// ── self-test hooks ────────────────────────────────────────────────────────────

// IDCT (M0c): variant 0=bare (int16 in/out via block), 1=put, 2=add (dest u8[64]).
// _mp4v_ref_idct runs idct_ref.h — a byte-for-byte copy of ffmpeg's C simple_idct8
// — so the M0c suite can cross-check our optimized simple_idct8.h against ffmpeg's
// reference IDCT directly, independent of the ffmpeg CLI (whose default build uses a
// SIMD IDCT that may differ by 1 LSB from its own C reference on rare HF blocks).
KEEP void mp4v_ref_idct(int16_t* block) { mp4ref::idct8(block); }
KEEP void mp4v_test_idct(int16_t* block, uint8_t* dest, int variant) {
    if (variant == 0)      mp4::idct8(block);
    else if (variant == 1) mp4::idct8_put(dest, 8, block);
    else                   mp4::idct8_add(dest, 8, block);
}
// Same variants via the SIMD transform — lets the M0c suite pin idct8_simd byte-
// identical to the scalar idct8 (and thus to ffmpeg's idct_ref) before it ships.
KEEP void mp4v_test_idct_simd(int16_t* block, uint8_t* dest, int variant) {
    if (variant == 0)      mp4::idct8_simd(block);
    else if (variant == 1) mp4::idct8_put_simd(dest, 8, block);
    else                   mp4::idct8_add_simd(dest, 8, block);
}

// qpel (M3 SIMD): run scalar (useSimd=0) or SIMD (1) qpel_mc into dst from the
// caller's (W+1)x(W+1) `full` block. Lets the node suite prove SIMD==scalar bit-exact
// over all (W, dxy, rnd, avg) before trusting the SIMD luma MC path.
KEEP void mp4v_test_qpel(uint8_t* dst, int dstStride, const uint8_t* full, int fullStride,
                         int W, int dxy, int rnd, int avg, int useSimd) {
    mp4::qpel_mc_test(dst, dstStride, full, fullStride, W, dxy, rnd != 0, avg, useSimd);
}

// get_bits (M0b): read `count` MSB-first fields; return total bits consumed.
KEEP int mp4v_test_getbits(const uint8_t* bytes, int nbytes, const int* widths, int count, int* out) {
    mp4::GetBits gb; mp4::init_get_bits(&gb, bytes, nbytes);
    for (int i = 0; i < count; i++) out[i] = (int)mp4::get_bits(&gb, widths[i]);
    return gb.index;
}

// VLC (M0b): 0=intra_MCBPC, 1=cbpy, 2=dc_lum, 3=dc_chrom. Returns the symbol.
KEEP int mp4v_test_vlc(int tableId, const uint8_t* bytes, int nbytes) {
    mp4::mp4_init_tables();
    mp4::GetBits gb; mp4::init_get_bits(&gb, bytes, nbytes);
    mp4::VlcTab t;
    switch (tableId) {
        case 0: t = mp4::g_intra_mcbpc; break;
        case 1: t = mp4::g_cbpy;        break;
        case 2: t = mp4::g_dc_lum;      break;
        case 3: t = mp4::g_dc_chrom;    break;
        case 4: t = mp4::g_inter_mcbpc; break;
        case 5: t = mp4::g_mv;          break;
        default: return -2;
    }
    return mp4::get_vlc2(&gb, t.table, t.bits);
}

// RL: tableId 0 = ff_mpeg4_rl_intra; 1..31 = ff_h263_rl_inter[qscale=tableId].
// out3 = {level, run, bitsConsumed}.
KEEP void mp4v_test_rlvlc(int tableId, const uint8_t* bytes, int nbytes, int* out3) {
    mp4::mp4_init_tables();
    mp4::GetBits gb; mp4::init_get_bits(&gb, bytes, nbytes);
    const mp4::RlTab& rl = tableId == 0 ? mp4::g_rl_intra : mp4::g_rl_inter[tableId];
    int level, run;
    mp4::get_rl_vlc(&gb, rl.table, rl.bits, &level, &run);
    out3[0] = level; out3[1] = run; out3[2] = gb.index;
}

} // extern "C"
