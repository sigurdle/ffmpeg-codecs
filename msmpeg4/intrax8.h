/*
    IntraX8 — the WMV2 "J-frame" intra sub-decoder (also used by VC-1). Transcription
    of FFmpeg release/8.1 libavcodec/intrax8.c + intrax8dsp.c; the VLC tables are
    intrax8_tables.h (intrax8huf.h verbatim). FFmpeg is LGPL-2.1+; see
    msmpeg4_decode.h for the module-wide provenance note.

    A J-frame codes the picture as 8x8 blocks with spatial (pixel-domain) prediction
    from the already-decoded neighbors, its own DC/AC VLC families, the WMV2 IDCT and
    an optional per-block deblocking filter. It writes straight into the current
    picture planes.

    The VLCs are built with ff_vlc_init_from_lengths (codes handed out MSB-aligned in
    table order), so the left-aligned codes increase monotonically with the entry
    index; x8_get_vlc finds the entry by binary search on the 32-bit bit cache — the
    same symbol FFmpeg's multi-level lookup returns, without its 2^16-entry tables.
*/

#ifndef MSM_INTRAX8_H
#define MSM_INTRAX8_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "../mpeg4/mp4_bits.h"
#include "msmpeg4_tables.h"   // ff_wmv1_scantable
#include "intrax8_tables.h"
#include "wmv2_dsp.h"         // wmv2_idct_add (the IntraX8 IDCT is the WMV2 one)

namespace msm {

// ── VLC: {symbol, length} tables, canonical codes in table order ────────────────
enum { X8_MAX_CODES = 77 };

struct X8Vlc {
    int n;
    uint32_t code[X8_MAX_CODES];   // left-aligned (MSB-first) code
    uint8_t len[X8_MAX_CODES];
    uint8_t sym[X8_MAX_CODES];
};

// ff_vlc_init_from_lengths over table[i] = {symbol, length}.
static inline void x8_init_vlc(X8Vlc* v, int nb_codes, const uint8_t table[][2]) {
    uint32_t code = 0;
    int j = 0;
    for (int i = 0; i < nb_codes; i++) {
        int len = table[i][1];
        if (len <= 0) continue;
        v->code[j] = code; v->len[j] = (uint8_t)len; v->sym[j] = table[i][0]; j++;
        code += 1u << (32 - len);
    }
    v->n = j;
}

// get_vlc2 on an X8Vlc. Returns the symbol, or -1 for a code not in the table.
static inline int x8_get_vlc(mp4::GetBits* gb, const X8Vlc* v) {
    uint32_t cache = mp4::get_cache(gb);
    int lo = 0, hi = v->n - 1;
    while (lo < hi) {                       // last entry with code <= cache
        int mid = (lo + hi + 1) >> 1;
        if (v->code[mid] <= cache) lo = mid; else hi = mid - 1;
    }
    int len = v->len[lo];
    if ((cache - v->code[lo]) >> (32 - len)) return -1;
    mp4::skip_bits(gb, len);
    return v->sym[lo];
}

struct X8Tables {
    X8Vlc ac[2][2][8];      // [quant < 13][intra / inter][select]
    X8Vlc dc[2][8];         // [quant < 13][select]
    X8Vlc orient_hq[2];     // j_orient_vlc[0][..]
    X8Vlc orient_lq[4];     // j_orient_vlc[1][..]
};

static inline const X8Tables* x8_tables() {
    static X8Tables t;
    static int ready = 0;
    if (!ready) {
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
                for (int k = 0; k < 8; k++)
                    x8_init_vlc(&t.ac[i][j][k], 77, x8_ac_quant_table[i][j][k]);
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 8; j++)
                x8_init_vlc(&t.dc[i][j], 34, x8_dc_quant_table[i][j]);
        for (int i = 0; i < 2; i++) x8_init_vlc(&t.orient_hq[i], 12, x8_orient_highquant_table[i]);
        for (int i = 0; i < 4; i++) x8_init_vlc(&t.orient_lq[i], 12, x8_orient_lowquant_table[i]);
        ready = 1;
    }
    return &t;
}

// ── context ─────────────────────────────────────────────────────────────────────
struct IntraX8Context {
    const X8Vlc* j_ac_vlc_table[4];
    const X8Vlc* j_orient_vlc_table;
    const X8Vlc* j_dc_vlc_table[3];

    int use_quant_matrix;
    uint8_t* prediction_table;      // 2 * (mb_width * 2)
    int16_t block[64];

    int quant, dquant, qsum, loopfilter;
    mp4::GetBits* gb;
    uint8_t* planes[3];
    int linesize[3];

    int quant_dc_chroma, divide_quant_dc_luma, divide_quant_dc_chroma;
    uint8_t* dest[3];
    uint8_t scratchpad[42];

    int edges, flat_dc, predicted_dc, raw_orient, chroma_orient, orient, est_run;

    int mb_x, mb_y, mb_width, mb_height;
};

// ── intrax8dsp.c ────────────────────────────────────────────────────────────────
/*
 * area positions, #3 is 1 pixel only, other are 8 pixels
 *    |66666666|
 *   3|44444444|55555555|
 * - -+--------+--------+
 * 1 2|XXXXXXXX|
 * ...
 */
enum { X8_AREA1 = 0, X8_AREA2 = 8, X8_AREA3 = 8 + 8, X8_AREA4 = 8 + 8 + 1,
       X8_AREA5 = 8 + 8 + 1 + 8, X8_AREA6 = 8 + 8 + 1 + 16 };

static inline int x8_min(int a, int b) { return a < b ? a : b; }
static inline int x8_max(int a, int b) { return a > b ? a : b; }
static inline int x8_abs(int a) { return a < 0 ? -a : a; }

static inline void x8_setup_spatial_compensation(const uint8_t* src, uint8_t* dst, int stride,
                                                 int* range, int* psum, int edges) {
    const uint8_t* ptr;
    int sum, i, min_pix, max_pix;
    uint8_t c = 0;

    if ((edges & 3) == 3) {
        *psum  = 0x80 * (8 + 1 + 8 + 2);
        *range = 0;
        memset(dst, 0x80, 16 + 1 + 16 + 8);
        return;
    }

    min_pix = 256;
    max_pix = -1;
    sum = 0;

    if (!(edges & 1)) {          // there is a previous block on this row
        ptr = src - 1;           // left column, area 2
        for (i = 7; i >= 0; i--) {
            c                 = *(ptr - 1);   // area1, same mb as area2
            dst[X8_AREA1 + i] = c;
            c                 = *ptr;

            sum              += c;
            min_pix           = x8_min(min_pix, c);
            max_pix           = x8_max(max_pix, c);
            dst[X8_AREA2 + i] = c;

            ptr += stride;
        }
    }

    if (!(edges & 2)) {          // there is a row above
        ptr = src - stride;      // top line
        for (i = 0; i < 8; i++) {
            c       = *(ptr + i);
            sum    += c;
            min_pix = x8_min(min_pix, c);
            max_pix = x8_max(max_pix, c);
        }
        if (edges & 4) {         // last block on the row
            memset(dst + X8_AREA5, c, 8);
            memcpy(dst + X8_AREA4, ptr, 8);
        } else {
            memcpy(dst + X8_AREA4, ptr, 16);
        }
        memcpy(dst + X8_AREA6, ptr - stride, 8);   // area6 always present in the above block
    }
    if (edges & 3) {
        int avg = (sum + 4) >> 3;

        if (edges & 1)           // implies mb_y != 0
            memset(dst + X8_AREA1, avg, 8 + 8 + 1);
        else                     // implies y == 0 x != 0
            memset(dst + X8_AREA3, avg, 1 + 16 + 8);

        sum += avg * 9;
    } else {
        uint8_t e = *(src - 1 - stride);   // the corner pixel; not part of min/max
        dst[X8_AREA3] = e;
        sum          += e;
    }
    *range = max_pix - min_pix;
    sum   += *(dst + X8_AREA5) + *(dst + X8_AREA5 + 1);
    *psum  = sum;
}

static const uint16_t x8_zero_prediction_weights[64 * 2] = {
    640,  640, 669,  480, 708,  354, 748, 257,
    792,  198, 760,  143, 808,  101, 772,  72,
    480,  669, 537,  537, 598,  416, 661, 316,
    719,  250, 707,  185, 768,  134, 745,  97,
    354,  708, 416,  598, 488,  488, 564, 388,
    634,  317, 642,  241, 716,  179, 706, 132,
    257,  748, 316,  661, 388,  564, 469, 469,
    543,  395, 571,  311, 655,  238, 660, 180,
    198,  792, 250,  719, 317,  634, 395, 543,
    469,  469, 507,  380, 597,  299, 616, 231,
    161,  855, 206,  788, 266,  710, 340, 623,
    411,  548, 455,  455, 548,  366, 576, 288,
    122,  972, 159,  914, 211,  842, 276, 758,
    341,  682, 389,  584, 483,  483, 520, 390,
    110, 1172, 144, 1107, 193, 1028, 254, 932,
    317,  846, 366,  731, 458,  611, 499, 499,
};

static inline void x8_spatial_compensation(int orient, const uint8_t* src, uint8_t* dst, int stride) {
    int x, y;
    switch (orient) {
    case 0: {
        int i, j;
        unsigned p;
        int a;
        uint16_t left_sum[2][8] = { { 0 } };
        uint16_t top_sum[2][8]  = { { 0 } };

        for (i = 0; i < 8; i++) {
            a = src[X8_AREA2 + 7 - i] << 4;
            for (j = 0; j < 8; j++) {
                p                   = (unsigned)x8_abs(i - j);
                left_sum[p & 1][j] += a >> (p >> 1);
            }
        }
        for (i = 0; i < 8; i++) {
            a = src[X8_AREA4 + i] << 4;
            for (j = 0; j < 8; j++) {
                p                  = (unsigned)x8_abs(i - j);
                top_sum[p & 1][j] += a >> (p >> 1);
            }
        }
        for (; i < 10; i++) {
            a = src[X8_AREA4 + i] << 4;
            for (j = 5; j < 8; j++) {
                p                  = (unsigned)x8_abs(i - j);
                top_sum[p & 1][j] += a >> (p >> 1);
            }
        }
        for (; i < 12; i++) {
            a = src[X8_AREA4 + i] << 4;
            for (j = 7; j < 8; j++) {
                p                  = (unsigned)x8_abs(i - j);
                top_sum[p & 1][j] += a >> (p >> 1);
            }
        }
        for (i = 0; i < 8; i++) {
            top_sum[0][i]  +=  (top_sum[1][i] * 181 + 128) >> 8;   // 181 is sqrt(2)/2
            left_sum[0][i] += (left_sum[1][i] * 181 + 128) >> 8;
        }
        for (y = 0; y < 8; y++) {
            for (x = 0; x < 8; x++)
                dst[x] = (uint8_t)(((uint32_t)  top_sum[0][x] * x8_zero_prediction_weights[y * 16 + x * 2 + 0] +
                                    (uint32_t) left_sum[0][y] * x8_zero_prediction_weights[y * 16 + x * 2 + 1] +
                                    0x8000) >> 16);
            dst += stride;
        }
        break;
    }
    case 1:
        for (y = 0; y < 8; y++) {
            for (x = 0; x < 8; x++)
                dst[x] = src[X8_AREA4 + x8_min(2 * y + x + 2, 15)];
            dst += stride;
        }
        break;
    case 2:
        for (y = 0; y < 8; y++) {
            memcpy(dst, src + X8_AREA4 + 1 + y, 8);
            dst += stride;
        }
        break;
    case 3:
        for (y = 0; y < 8; y++) {
            memcpy(dst, src + X8_AREA4 + ((y + 1) >> 1), 8);
            dst += stride;
        }
        break;
    case 4:
        for (y = 0; y < 8; y++) {
            for (x = 0; x < 8; x++)
                dst[x] = (uint8_t)((src[X8_AREA4 + x] + src[X8_AREA6 + x] + 1) >> 1);
            dst += stride;
        }
        break;
    case 5:
        for (y = 0; y < 8; y++) {
            for (x = 0; x < 8; x++) {
                if (2 * x - y < 0)
                    dst[x] = src[X8_AREA2 + 9 + 2 * x - y];
                else
                    dst[x] = src[X8_AREA4 + x - ((y + 1) >> 1)];
            }
            dst += stride;
        }
        break;
    case 6:
        for (y = 0; y < 8; y++) {
            memcpy(dst, src + X8_AREA3 - y, 8);
            dst += stride;
        }
        break;
    case 7:
        for (y = 0; y < 8; y++) {
            for (x = 0; x < 8; x++) {
                if (x - 2 * y > 0)
                    dst[x] = (uint8_t)((src[X8_AREA3 - 1 + x - 2 * y] + src[X8_AREA3 + x - 2 * y] + 1) >> 1);
                else
                    dst[x] = src[X8_AREA2 + 8 - y + (x >> 1)];
            }
            dst += stride;
        }
        break;
    case 8:
        for (y = 0; y < 8; y++) {
            for (x = 0; x < 8; x++)
                dst[x] = (uint8_t)((src[X8_AREA1 + 7 - y] + src[X8_AREA2 + 7 - y] + 1) >> 1);
            dst += stride;
        }
        break;
    case 9:
        for (y = 0; y < 8; y++) {
            for (x = 0; x < 8; x++)
                dst[x] = src[X8_AREA2 + 6 - x8_min(x + y, 6)];
            dst += stride;
        }
        break;
    case 10:
        for (y = 0; y < 8; y++) {
            for (x = 0; x < 8; x++)
                dst[x] = (uint8_t)((src[X8_AREA2 + 7 - y] * (8 - x) + src[X8_AREA4 + x] * x + 4) >> 3);
            dst += stride;
        }
        break;
    default:   // 11
        for (y = 0; y < 8; y++) {
            for (x = 0; x < 8; x++)
                dst[x] = (uint8_t)((src[X8_AREA2 + 7 - y] * y + src[X8_AREA4 + x] * (8 - y) + 4) >> 3);
            dst += stride;
        }
        break;
    }
}

static inline void x8_loop_filter(uint8_t* ptr, const int a_stride, const int b_stride, int quant) {
    int i, t;
    int p0, p1, p2, p3, p4, p5, p6, p7, p8, p9;
    int ql = (quant + 10) >> 3;

    for (i = 0; i < 8; i++, ptr += b_stride) {
        p0 = ptr[-5 * a_stride];
        p1 = ptr[-4 * a_stride];
        p2 = ptr[-3 * a_stride];
        p3 = ptr[-2 * a_stride];
        p4 = ptr[-1 * a_stride];
        p5 = ptr[0];
        p6 = ptr[1 * a_stride];
        p7 = ptr[2 * a_stride];
        p8 = ptr[3 * a_stride];
        p9 = ptr[4 * a_stride];

        t = (x8_abs(p1 - p2) <= ql) +
            (x8_abs(p2 - p3) <= ql) +
            (x8_abs(p3 - p4) <= ql) +
            (x8_abs(p4 - p5) <= ql);

        // You need at least 1 to be able to reach a total score of 6.
        if (t > 0) {
            t += (x8_abs(p5 - p6) <= ql) +
                 (x8_abs(p6 - p7) <= ql) +
                 (x8_abs(p7 - p8) <= ql) +
                 (x8_abs(p8 - p9) <= ql) +
                 (x8_abs(p0 - p1) <= ql);
            if (t >= 6) {
                int min, max;

                min = max = p1;
                min = x8_min(min, p3);
                max = x8_max(max, p3);
                min = x8_min(min, p5);
                max = x8_max(max, p5);
                min = x8_min(min, p8);
                max = x8_max(max, p8);
                if (max - min < 2 * quant) {   // early stop
                    min = x8_min(min, p2);
                    max = x8_max(max, p2);
                    min = x8_min(min, p4);
                    max = x8_max(max, p4);
                    min = x8_min(min, p6);
                    max = x8_max(max, p6);
                    min = x8_min(min, p7);
                    max = x8_max(max, p7);
                    if (max - min < 2 * quant) {
                        ptr[-2 * a_stride] = (uint8_t)((4 * p2 + 3 * p3 + 1 * p7 + 4) >> 3);
                        ptr[-1 * a_stride] = (uint8_t)((3 * p2 + 3 * p4 + 2 * p7 + 4) >> 3);
                        ptr[0]             = (uint8_t)((2 * p2 + 3 * p5 + 3 * p7 + 4) >> 3);
                        ptr[1 * a_stride]  = (uint8_t)((1 * p2 + 3 * p6 + 4 * p7 + 4) >> 3);
                        continue;
                    }
                }
            }
        }
        {
            int x, x0, x1, x2;
            int m;

            x0 = (2 * p3 - 5 * p4 + 5 * p5 - 2 * p6 + 4) >> 3;
            if (x8_abs(x0) < quant) {
                x1 = (2 * p1 - 5 * p2 + 5 * p3 - 2 * p4 + 4) >> 3;
                x2 = (2 * p5 - 5 * p6 + 5 * p7 - 2 * p8 + 4) >> 3;

                x = x8_abs(x0) - x8_min(x8_abs(x1), x8_abs(x2));
                m = p4 - p5;

                if (x > 0 && (m ^ x0) < 0) {
                    int32_t sign;

                    sign = m >> 31;
                    m    = (m ^ sign) - sign;   // abs(m)
                    m  >>= 1;

                    x = 5 * x >> 3;

                    if (x > m)
                        x = m;

                    x = (x ^ sign) - sign;

                    ptr[-1 * a_stride] = (uint8_t)(ptr[-1 * a_stride] - x);
                    ptr[0]             = (uint8_t)(ptr[0] + x);
                }
            }
        }
    }
}

static inline void x8_h_loop_filter(uint8_t* src, int stride, int qscale) { x8_loop_filter(src, stride, 1, qscale); }
static inline void x8_v_loop_filter(uint8_t* src, int stride, int qscale) { x8_loop_filter(src, 1, stride, qscale); }

// ── intrax8.c ───────────────────────────────────────────────────────────────────
static inline void x8_reset_vlc_tables(IntraX8Context* w) {
    memset(w->j_dc_vlc_table, 0, sizeof(w->j_dc_vlc_table));
    memset(w->j_ac_vlc_table, 0, sizeof(w->j_ac_vlc_table));
    w->j_orient_vlc_table = NULL;
}

static inline void x8_select_ac_table(IntraX8Context* w, int mode) {
    if (w->j_ac_vlc_table[mode]) return;
    int table_index = (int)mp4::get_bits(w->gb, 3);
    w->j_ac_vlc_table[mode] = &x8_tables()->ac[w->quant < 13][mode >> 1][table_index];   // 2 modes use same tables
}

static inline int x8_get_orient_vlc(IntraX8Context* w) {
    if (!w->j_orient_vlc_table) {
        int table_index = (int)mp4::get_bits(w->gb, 1 + (w->quant < 13));
        w->j_orient_vlc_table = (w->quant < 13) ? &x8_tables()->orient_lq[table_index]
                                                : &x8_tables()->orient_hq[table_index];
    }
    return x8_get_vlc(w->gb, w->j_orient_vlc_table);
}

#define X8_EXTRA_BITS(eb)  (eb)          // 3 bits
#define X8_EXTRA_RUN       (0xFFu << 8)  // 1 bit
#define X8_EXTRA_LEVEL     (0x00u << 8)  // 1 bit
#define X8_RUN_OFFSET(r)   ((uint32_t)(r) << 16) // 6 bits
#define X8_LEVEL_OFFSET(l) ((uint32_t)(l) << 24) // 5 bits
static const uint32_t x8_ac_decode_table[] = {
    /* 46 */ X8_EXTRA_BITS(3) | X8_EXTRA_RUN   | X8_RUN_OFFSET(16) | X8_LEVEL_OFFSET(0),
    /* 47 */ X8_EXTRA_BITS(3) | X8_EXTRA_RUN   | X8_RUN_OFFSET(24) | X8_LEVEL_OFFSET(0),
    /* 48 */ X8_EXTRA_BITS(2) | X8_EXTRA_RUN   | X8_RUN_OFFSET(4)  | X8_LEVEL_OFFSET(1),
    /* 49 */ X8_EXTRA_BITS(3) | X8_EXTRA_RUN   | X8_RUN_OFFSET(8)  | X8_LEVEL_OFFSET(1),

    /* 50 */ X8_EXTRA_BITS(5) | X8_EXTRA_RUN   | X8_RUN_OFFSET(32) | X8_LEVEL_OFFSET(0),
    /* 51 */ X8_EXTRA_BITS(4) | X8_EXTRA_RUN   | X8_RUN_OFFSET(16) | X8_LEVEL_OFFSET(1),

    /* 52 */ X8_EXTRA_BITS(2) | X8_EXTRA_LEVEL | X8_RUN_OFFSET(0)  | X8_LEVEL_OFFSET(4),
    /* 53 */ X8_EXTRA_BITS(2) | X8_EXTRA_LEVEL | X8_RUN_OFFSET(0)  | X8_LEVEL_OFFSET(8),
    /* 54 */ X8_EXTRA_BITS(2) | X8_EXTRA_LEVEL | X8_RUN_OFFSET(0)  | X8_LEVEL_OFFSET(12),
    /* 55 */ X8_EXTRA_BITS(3) | X8_EXTRA_LEVEL | X8_RUN_OFFSET(0)  | X8_LEVEL_OFFSET(16),
    /* 56 */ X8_EXTRA_BITS(3) | X8_EXTRA_LEVEL | X8_RUN_OFFSET(0)  | X8_LEVEL_OFFSET(24),

    /* 57 */ X8_EXTRA_BITS(2) | X8_EXTRA_LEVEL | X8_RUN_OFFSET(1)  | X8_LEVEL_OFFSET(3),
    /* 58 */ X8_EXTRA_BITS(3) | X8_EXTRA_LEVEL | X8_RUN_OFFSET(1)  | X8_LEVEL_OFFSET(7),

    /* 59 */ X8_EXTRA_BITS(2) | X8_EXTRA_RUN   | X8_RUN_OFFSET(16) | X8_LEVEL_OFFSET(0),
    /* 60 */ X8_EXTRA_BITS(2) | X8_EXTRA_RUN   | X8_RUN_OFFSET(20) | X8_LEVEL_OFFSET(0),
    /* 61 */ X8_EXTRA_BITS(2) | X8_EXTRA_RUN   | X8_RUN_OFFSET(24) | X8_LEVEL_OFFSET(0),
    /* 62 */ X8_EXTRA_BITS(2) | X8_EXTRA_RUN   | X8_RUN_OFFSET(28) | X8_LEVEL_OFFSET(0),
    /* 63 */ X8_EXTRA_BITS(4) | X8_EXTRA_RUN   | X8_RUN_OFFSET(32) | X8_LEVEL_OFFSET(0),
    /* 64 */ X8_EXTRA_BITS(4) | X8_EXTRA_RUN   | X8_RUN_OFFSET(48) | X8_LEVEL_OFFSET(0),

    /* 65 */ X8_EXTRA_BITS(2) | X8_EXTRA_RUN   | X8_RUN_OFFSET(4)  | X8_LEVEL_OFFSET(1),
    /* 66 */ X8_EXTRA_BITS(3) | X8_EXTRA_RUN   | X8_RUN_OFFSET(8)  | X8_LEVEL_OFFSET(1),
    /* 67 */ X8_EXTRA_BITS(4) | X8_EXTRA_RUN   | X8_RUN_OFFSET(16) | X8_LEVEL_OFFSET(1),

    /* 68 */ X8_EXTRA_BITS(2) | X8_EXTRA_LEVEL | X8_RUN_OFFSET(0)  | X8_LEVEL_OFFSET(4),
    /* 69 */ X8_EXTRA_BITS(3) | X8_EXTRA_LEVEL | X8_RUN_OFFSET(0)  | X8_LEVEL_OFFSET(8),
    /* 70 */ X8_EXTRA_BITS(4) | X8_EXTRA_LEVEL | X8_RUN_OFFSET(0)  | X8_LEVEL_OFFSET(16),

    /* 71 */ X8_EXTRA_BITS(2) | X8_EXTRA_LEVEL | X8_RUN_OFFSET(1)  | X8_LEVEL_OFFSET(3),
    /* 72 */ X8_EXTRA_BITS(3) | X8_EXTRA_LEVEL | X8_RUN_OFFSET(1)  | X8_LEVEL_OFFSET(7),
};
#undef X8_EXTRA_BITS
#undef X8_EXTRA_RUN
#undef X8_EXTRA_LEVEL
#undef X8_RUN_OFFSET
#undef X8_LEVEL_OFFSET

static inline void x8_get_ac_rlf(IntraX8Context* w, const int mode, int* run, int* level, int* final) {
    int i, e;

    i = x8_get_vlc(w->gb, w->j_ac_vlc_table[mode]);

    if (i < 46) {   // [0-45]
        int t, l;
        if (i < 0) {
            *level = *final = *run = 64;   // causes the error exit in the ac loop
            return;
        }
        *final =
        t      = i > 22;
        i     -= 23 * t;

        /* l = lut_l[i / 2] = { 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 3 }[i >> 1]; */
        l = (0xE50000 >> (i & 0x1E)) & 3;

        /* t = lut_mask[l] = { 0x0f, 0x03, 0x01, 0x00 }[l]; */
        t = 0x01030F >> (l << 3);

        *run   = i & t;
        *level = l;
    } else if (i < 73) {   // [46-72]
        uint32_t sm;
        uint32_t mask;

        i -= 46;
        sm = x8_ac_decode_table[i];

        e    = (int)mp4::get_bits(w->gb, (int)(sm & 0xF));
        sm >>= 8;                               // 3 bits
        mask = sm & 0xff;
        sm >>= 8;                               // 1 bit

        *run   = (int)((sm &  0xff) + ((uint32_t)e &  mask));   // 6 bits
        *level = (int)((sm >>    8) + ((uint32_t)e & ~mask));   // 5 bits
        *final = i > (58 - 46);
    } else if (i < 75) {   // [73-74]
        static const uint8_t crazy_mix_runlevel[32] = {
            0x22, 0x32, 0x33, 0x53, 0x23, 0x42, 0x43, 0x63,
            0x24, 0x52, 0x34, 0x73, 0x25, 0x62, 0x44, 0x83,
            0x26, 0x72, 0x35, 0x54, 0x27, 0x82, 0x45, 0x64,
            0x28, 0x92, 0x36, 0x74, 0x29, 0xa2, 0x46, 0x84,
        };

        *final = !(i & 1);
        e      = (int)mp4::get_bits(w->gb, 5);
        *run   = crazy_mix_runlevel[e] >> 4;
        *level = crazy_mix_runlevel[e] & 0x0F;
    } else {
        *level = (int)mp4::get_bits(w->gb, 7 - 3 * (i & 1));
        *run   = (int)mp4::get_bits(w->gb, 6);
        *final = (int)mp4::get_bits1(w->gb);
    }
}

static const uint8_t x8_dc_index_offset[] = {
    0, 1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193,
};

static inline int x8_get_dc_rlf(IntraX8Context* w, const int mode, int* level, int* final) {
    int i, e, c;

    if (!w->j_dc_vlc_table[mode]) {
        int table_index = (int)mp4::get_bits(w->gb, 3);
        w->j_dc_vlc_table[mode] = &x8_tables()->dc[w->quant < 13][table_index];   // 4 modes, same table
    }

    i = x8_get_vlc(w->gb, w->j_dc_vlc_table[mode]);

    /* (i >= 17) { i -= 17; final =1; } */
    c      = i > 16;
    *final = c;
    i     -= 17 * c;

    if (i <= 0) {
        *level = 0;
        return -i;
    }
    c  = (i + 1) >> 1;   // hackish way to calculate dc_extra_sbits[]
    c -= c > 1;

    e = (int)mp4::get_bits(w->gb, c);
    i = x8_dc_index_offset[i] + (e >> 1);

    e      = -(e & 1);
    *level =  (i ^ e) - e;
    return 0;
}

static inline int x8_setup_spatial_predictor(IntraX8Context* w, const int chroma) {
    int range, sum, quant;

    x8_setup_spatial_compensation(w->dest[chroma], w->scratchpad, w->linesize[chroma > 0],
                                  &range, &sum, w->edges);
    if (chroma) {
        w->orient = w->chroma_orient;
        quant     = w->quant_dc_chroma;
    } else {
        quant = w->quant;
    }

    w->flat_dc = 0;
    if (range < quant || range < 3) {
        w->orient = 0;

        // yep you read right, a +-1 idct error may break decoding!
        if (range < 3) {
            w->flat_dc      = 1;
            sum            += 9;
            // ((1 << 17) + 9) / (8 + 8 + 1 + 2) = 6899
            w->predicted_dc = sum * 6899 >> 17;
        }
    }
    if (chroma)
        return 0;

    if (range < 2 * w->quant) {
        if ((w->edges & 3) == 0) {
            if (w->orient == 1)
                w->orient = 11;
            if (w->orient == 2)
                w->orient = 10;
        } else {
            w->orient = 0;
        }
        w->raw_orient = 0;
    } else {
        static const uint8_t prediction_table[3][12] = {
            { 0, 8, 4, 10, 11, 2, 6, 9, 1, 3, 5, 7 },
            { 4, 0, 8, 11, 10, 3, 5, 2, 6, 9, 1, 7 },
            { 8, 0, 4, 10, 11, 1, 7, 2, 6, 9, 3, 5 },
        };
        w->raw_orient = x8_get_orient_vlc(w);
        if (w->raw_orient < 0)
            return -1;
        w->orient = prediction_table[w->orient][w->raw_orient];
    }
    return 0;
}

static inline void x8_update_predictions(IntraX8Context* w, const int orient, const int est_run) {
    w->prediction_table[w->mb_x * 2 + (w->mb_y & 1)] =
        (uint8_t)((est_run << 2) + 1 * (orient == 4) + 2 * (orient == 8));
}

static inline void x8_get_prediction_chroma(IntraX8Context* w) {
    w->edges  = 1 * !(w->mb_x >> 1);
    w->edges |= 2 * !(w->mb_y >> 1);
    w->edges |= 4 * (w->mb_x >= (2 * w->mb_width - 1));   // mb_x for chroma would always be odd

    w->raw_orient = 0;
    // lut_co[8] = {inv,4,8,8, inv,4,8,8} <- => {1,1,0,0;1,1,0,0} => 0xCC
    if (w->edges & 3) {
        w->chroma_orient = 4 << ((0xCC >> w->edges) & 1);
        return;
    }
    // block[x - 1][y | 1 - 1)]
    w->chroma_orient = (w->prediction_table[2 * w->mb_x - 2] & 0x03) << 2;
}

static inline void x8_get_prediction(IntraX8Context* w) {
    int a, b, c, i;

    w->edges  = 1 * !w->mb_x;
    w->edges |= 2 * !w->mb_y;
    w->edges |= 4 * (w->mb_x >= (2 * w->mb_width - 1));

    switch (w->edges & 3) {
    case 0:
        break;
    case 1:
        // take the one from the above block[0][y - 1]
        w->est_run = w->prediction_table[!(w->mb_y & 1)] >> 2;
        w->orient  = 1;
        return;
    case 2:
        // take the one from the previous block[x - 1][0]
        w->est_run = w->prediction_table[2 * w->mb_x - 2] >> 2;
        w->orient  = 2;
        return;
    case 3:
        w->est_run = 16;
        w->orient  = 0;
        return;
    }
    // no edge cases
    b = w->prediction_table[2 * w->mb_x     + !(w->mb_y & 1)];   // block[x    ][y - 1]
    a = w->prediction_table[2 * w->mb_x - 2 +  (w->mb_y & 1)];   // block[x - 1][y    ]
    c = w->prediction_table[2 * w->mb_x - 2 + !(w->mb_y & 1)];   // block[x - 1][y - 1]

    w->est_run = x8_min(b, a);
    /* This condition has nothing to do with w->edges, even if it looks
     * similar it would trigger if e.g. x = 3; y = 2;
     * I guess somebody wrote something wrong and it became standard. */
    if ((w->mb_x & w->mb_y) != 0)
        w->est_run = x8_min(c, w->est_run);
    w->est_run >>= 2;

    a &= 3;
    b &= 3;
    c &= 3;

    i = (int)((0xFFEAF4C4u >> (2 * b + 8 * a)) & 3);
    if (i != 3)
        w->orient = i;
    else
        w->orient = (0xFFEAD8 >> (2 * c + 8 * (w->quant > 12))) & 3;
}

static inline void x8_ac_compensation(IntraX8Context* w, const int direction, const int dc_level) {
    int t;
#define B(x, y)  w->block[(x) + (y) * 8]
#define T(x)  ((x) * dc_level + 0x8000) >> 16;
    switch (direction) {
    case 0:
        t        = T(3811); // h
        B(1, 0) -= t;
        B(0, 1) -= t;

        t        = T(487); // e
        B(2, 0) -= t;
        B(0, 2) -= t;

        t        = T(506); // f
        B(3, 0) -= t;
        B(0, 3) -= t;

        t        = T(135); // c
        B(4, 0) -= t;
        B(0, 4) -= t;
        B(2, 1) += t;
        B(1, 2) += t;
        B(3, 1) += t;
        B(1, 3) += t;

        t        = T(173); // d
        B(5, 0) -= t;
        B(0, 5) -= t;

        t        = T(61); // b
        B(6, 0) -= t;
        B(0, 6) -= t;
        B(5, 1) += t;
        B(1, 5) += t;

        t        = T(42); // a
        B(7, 0) -= t;
        B(0, 7) -= t;
        B(4, 1) += t;
        B(1, 4) += t;
        B(4, 4) += t;

        t        = T(1084); // g
        B(1, 1) += t;
        break;
    case 1:
        B(0, 1) -= T(6269);
        B(0, 3) -= T(708);
        B(0, 5) -= T(172);
        B(0, 7) -= T(73);
        break;
    case 2:
        B(1, 0) -= T(6269);
        B(3, 0) -= T(708);
        B(5, 0) -= T(172);
        B(7, 0) -= T(73);
        break;
    }
#undef B
#undef T
}

static inline void x8_put_solidcolor(const uint8_t pix, uint8_t* dst, const int linesize) {
    for (int k = 0; k < 8; k++) {
        memset(dst, pix, 8);
        dst += linesize;
    }
}

static const int16_t x8_quant_table[64] = {
    256, 256, 256, 256, 256, 256, 259, 262,
    265, 269, 272, 275, 278, 282, 285, 288,
    292, 295, 299, 303, 306, 310, 314, 317,
    321, 325, 329, 333, 337, 341, 345, 349,
    353, 358, 362, 366, 371, 375, 379, 384,
    389, 393, 398, 403, 408, 413, 417, 422,
    428, 433, 438, 443, 448, 454, 459, 465,
    470, 476, 482, 488, 493, 499, 505, 511,
};

static inline int x8_decode_intra_mb(IntraX8Context* w, const int chroma) {
    const uint8_t* scantable;
    int final, run, level;
    int ac_mode, dc_mode, est_run, dc_level;
    int pos, n;
    int zeros_only;
    int use_quant_matrix;
    int sign;
    const int linesize = w->linesize[!!chroma];

    memset(w->block, 0, sizeof(w->block));

    if (chroma)
        dc_mode = 2;
    else
        dc_mode = !!w->est_run;   // 0, 1

    if (x8_get_dc_rlf(w, dc_mode, &dc_level, &final))
        return -1;
    n          = 0;
    zeros_only = 0;
    if (!final) {   // decode ac
        use_quant_matrix = w->use_quant_matrix;
        if (chroma) {
            ac_mode = 1;
            est_run = 64;   // not used
        } else {
            if (w->raw_orient < 3)
                use_quant_matrix = 0;

            if (w->raw_orient > 4) {
                ac_mode = 0;
                est_run = 64;
            } else {
                if (w->est_run > 1) {
                    ac_mode = 2;
                    est_run = w->est_run;
                } else {
                    ac_mode = 3;
                    est_run = 64;
                }
            }
        }
        x8_select_ac_table(w, ac_mode);
        /* scantable_selector[12] = { 0, 2, 0, 1, 1, 1, 0, 2, 2, 0, 1, 2 }; <-
         * -> 10'01' 00'10' 10'00' 01'01' 01'00' 10'00 => 0x928548 */
        static const int x8_scan_index[3] = { 0, 2, 3 };   // permutated_scantable[k] = ff_wmv1_scantable[{0,2,3}[k]]
        scantable = ff_wmv1_scantable[x8_scan_index[(0x928548 >> (2 * w->orient)) & 3]];
        pos       = 0;
        do {
            n++;
            if (n >= est_run) {
                ac_mode = 3;
                x8_select_ac_table(w, 3);
            }

            x8_get_ac_rlf(w, ac_mode, &run, &level, &final);

            pos += run + 1;
            if (pos > 63) {
                // this also handles vlc error in x8_get_ac_rlf
                return -1;
            }
            level  = (level + 1) * w->dquant;
            level += w->qsum;

            sign  = -(int)mp4::get_bits1(w->gb);
            level = (level ^ sign) - sign;

            if (use_quant_matrix)
                level = (level * x8_quant_table[pos]) >> 8;

            w->block[scantable[pos]] = (int16_t)level;
        } while (!final);
    } else {   // DC only
        if (w->flat_dc && ((unsigned)(dc_level + 1)) < 3) {   // [-1; 1]
            int32_t divide_quant = !chroma ? w->divide_quant_dc_luma : w->divide_quant_dc_chroma;
            int32_t dc_quant     = !chroma ? w->quant : w->quant_dc_chroma;

            // original intent dc_level += predicted_dc/quant;
            // but it got lost somewhere in the rounding
            dc_level += (w->predicted_dc * divide_quant + (1 << 12)) >> 13;

            x8_put_solidcolor(wmv2_clip_u8((dc_level * dc_quant + 4) >> 3), w->dest[chroma], linesize);

            goto block_placed;
        }
        zeros_only = dc_level == 0;
    }
    if (!chroma)
        w->block[0] = (int16_t)(dc_level * w->quant);
    else
        w->block[0] = (int16_t)(dc_level * w->quant_dc_chroma);

    // there is !zero_only check in the original, but dc_level check is enough
    if ((unsigned int)(dc_level + 1) >= 3 && (w->edges & 3) != 3) {
        int direction;
        /* ac_comp_direction[orient] = { 0, 3, 3, 1, 1, 0, 0, 0, 2, 2, 2, 1 }; <-
         * -> 01'10' 10'10' 00'00' 00'01' 01'11' 11'00 => 0x6A017C */
        direction = (0x6A017C >> (w->orient * 2)) & 3;
        if (direction != 3)
            x8_ac_compensation(w, direction, w->block[0]);
    }

    if (w->flat_dc)
        x8_put_solidcolor((uint8_t)w->predicted_dc, w->dest[chroma], linesize);
    else
        x8_spatial_compensation(w->orient, w->scratchpad, w->dest[chroma], linesize);
    if (!zeros_only)
        wmv2_idct_add(w->dest[chroma], linesize, w->block);

block_placed:
    if (!chroma)
        x8_update_predictions(w, w->orient, n);

    if (w->loopfilter) {
        uint8_t* ptr = w->dest[chroma];

        if (!((w->edges & 2) || (zeros_only && (w->orient | 4) == 4)))
            x8_h_loop_filter(ptr, linesize, w->quant);

        if (!((w->edges & 1) || (zeros_only && (w->orient | 8) == 8)))
            x8_v_loop_filter(ptr, linesize, w->quant);
    }
    return 0;
}

static inline void x8_init_block_index(IntraX8Context* w) {
    w->dest[0] = w->planes[0] + (size_t)(w->mb_y * w->linesize[0] << 3);
    // chroma blocks are on odd rows
    w->dest[1] = w->planes[1] + (size_t)((w->mb_y & ~1) * w->linesize[1] << 2);
    w->dest[2] = w->planes[2] + (size_t)((w->mb_y & ~1) * w->linesize[2] << 2);
}

// ff_intrax8_common_init: the two-row prediction table (zeroed, persists across frames).
static inline int x8_common_init(IntraX8Context* w, int mb_width, int mb_height) {
    w->mb_width  = mb_width;
    w->mb_height = mb_height;
    free(w->prediction_table);
    w->prediction_table = (uint8_t*)calloc((size_t)mb_width * 2 * 2, 1);
    return w->prediction_table ? 0 : -1;
}

// ff_intrax8_decode_picture. dquant = 2*qscale, quant_offset = (qscale-1)|1.
static inline int x8_decode_picture(IntraX8Context* w, mp4::GetBits* gb,
                                    uint8_t* y, uint8_t* u, uint8_t* v, int ylinesize, int clinesize,
                                    int dquant, int quant_offset, int loopfilter) {
    w->gb     = gb;
    w->dquant = dquant;
    w->quant  = dquant >> 1;
    w->qsum   = quant_offset;
    w->planes[0] = y; w->planes[1] = u; w->planes[2] = v;
    w->linesize[0] = ylinesize; w->linesize[1] = clinesize; w->linesize[2] = clinesize;
    w->loopfilter = loopfilter;
    w->use_quant_matrix = (int)mp4::get_bits1(w->gb);

    w->divide_quant_dc_luma = ((1 << 16) + (w->quant >> 1)) / w->quant;
    if (w->quant < 5) {
        w->quant_dc_chroma        = w->quant;
        w->divide_quant_dc_chroma = w->divide_quant_dc_luma;
    } else {
        w->quant_dc_chroma        = w->quant + ((w->quant + 3) >> 3);
        w->divide_quant_dc_chroma = ((1 << 16) + (w->quant_dc_chroma >> 1)) / w->quant_dc_chroma;
    }
    x8_reset_vlc_tables(w);

    for (w->mb_y = 0; w->mb_y < w->mb_height * 2; w->mb_y++) {
        x8_init_block_index(w);
        if (mp4::get_bits_left(gb) < 1)
            return 0;
        for (w->mb_x = 0; w->mb_x < w->mb_width * 2; w->mb_x++) {
            x8_get_prediction(w);
            if (x8_setup_spatial_predictor(w, 0))
                return 0;
            if (x8_decode_intra_mb(w, 0))
                return 0;

            if (w->mb_x & w->mb_y & 1) {
                x8_get_prediction_chroma(w);

                // when setting up chroma, no vlc is read, so no error can be reached
                x8_setup_spatial_predictor(w, 1);
                if (x8_decode_intra_mb(w, 1))
                    return 0;

                x8_setup_spatial_predictor(w, 2);
                if (x8_decode_intra_mb(w, 2))
                    return 0;

                w->dest[1] += 8;
                w->dest[2] += 8;
            }
            w->dest[0] += 8;
        }
    }
    return 0;
}

} // namespace msm

#endif // MSM_INTRAX8_H
