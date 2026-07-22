/*
    VLC / RL (run-length) table machinery — port of the relevant pieces of
    FFmpeg's libavcodec/vlc.c + rl.c + the get_vlc2 / GET_RL_VLC consumers in
    get_bits.h (LGPL). See mpeg4_decode.cpp header for provenance.

    DELIBERATE SIMPLIFICATION vs ffmpeg: ffmpeg builds 9-bit primary tables with
    recursively-allocated subtables for longer codes (build_table()). Here we
    build a FLAT single-level LUT sized to each table's MAX code length, so the
    lookup is one show_bits() with no subtable chase. This is functionally
    identical (the leaf fill is the same as build_table's direct-entry pass) and
    much harder to get wrong; the cost is a little more .rodata for the few intra
    tables (all small). The decode call-sites pass each table's stored `bits`.

    VLCElem matches ffmpeg's union exactly so the RL transform (ff_rl_init_vlc)
    reads {sym,len} and writes {level,len8,run} over the same memory.
*/

#ifndef MPEG4_MP4_VLC_H
#define MPEG4_MP4_VLC_H

#include <stdint.h>
#include "mp4_bits.h"

namespace mp4 {

// Mirrors FFmpeg VLCElem (vlc.h): ordinary VLC view {sym,len} aliased with the
// run-length view {level,len8,run}.
typedef union VLCElem {
    struct { int16_t sym; int16_t len; };
    struct { int16_t level; int8_t len8; uint8_t run; };
} VLCElem;
typedef VLCElem RL_VLC_ELEM;

#define MAX_RUN_MP4   64
#define MAX_LEVEL_MP4 64

struct VlcTab { const VLCElem* table; int bits; };
struct RlTab  {
    const RL_VLC_ELEM* table;
    int bits;
    int n, last;
    const int8_t* table_run;
    const int8_t* table_level;
    int8_t max_level[2][MAX_LEVEL_MP4 + 1];  // [last][run]  (ff_rl_init)
    int8_t max_run[2][MAX_RUN_MP4 + 1];      // [last][level]
};

// Read one strided table cell (uint8 or uint16), like ffmpeg GET_DATA.
static inline uint32_t mp4_cell(const void* base, int i, int stride, int size) {
    const uint8_t* p = (const uint8_t*)base + (size_t)i * stride;
    return size == 1 ? (uint32_t)*p : (uint32_t)*(const uint16_t*)p;
}

// Scan the lengths for the max code length (table size = 1<<max).
static inline int mp4_max_len(const void* lens, int nb, int stride, int size) {
    int m = 0;
    for (int i = 0; i < nb; i++) {
        int L = (int)mp4_cell(lens, i, stride, size);
        if (L > m) m = L;
    }
    return m;
}

// Build a flat single-level VLC LUT (size 1<<bits) from (code,len[,sym]) arrays.
// Unused entries -> {sym=-1, len=0}. syms==NULL => sym=i.
static inline void build_flat_vlc(VLCElem* table, int bits, int nb_codes,
                                  const void* codes, int code_stride, int code_size,
                                  const void* lens, int len_stride, int len_size,
                                  const int16_t* syms) {
    int size = 1 << bits;
    for (int i = 0; i < size; i++) { table[i].sym = -1; table[i].len = 0; }
    for (int i = 0; i < nb_codes; i++) {
        int L = (int)mp4_cell(lens, i, len_stride, len_size);
        if (L <= 0) continue;
        uint32_t code = mp4_cell(codes, i, code_stride, code_size);
        int sym  = syms ? syms[i] : i;
        int base = (int)(code << (bits - L));
        int nb   = 1 << (bits - L);
        for (int k = 0; k < nb; k++) {
            table[base + k].sym = (int16_t)sym;
            table[base + k].len = (int16_t)L;
        }
    }
}

// Build an RlTab: a flat RL LUT (level/len8/run) sized to the max code length,
// plus the max_level/max_run aux tables the 1st/2nd MPEG-4 escapes need.
// n = entries-1 (index n == ESC), `last` = boundary for the +192 last-flag.
// qmul/qadd bake the dequant (qmul=1,qadd=0 for intra rl_vlc[0]). The LUT fill
// mirrors ff_rl_init_vlc minus the subtable (len<0) branch (flat tables have
// none); the aux fill mirrors ff_rl_init. storage must be >= 1<<maxlen elements.
static inline void build_rl(RlTab* rl, RL_VLC_ELEM* storage, int n, int last,
                            const uint16_t (*table_vlc)[2],
                            const int8_t* table_run, const int8_t* table_level,
                            int qmul, int qadd) {
    int bits = mp4_max_len(&table_vlc[0][1], n + 1, 4, 2);
    build_flat_vlc(storage, bits, n + 1,
                   &table_vlc[0][0], 4, 2,
                   &table_vlc[0][1], 4, 2, NULL);
    int size = 1 << bits;
    for (int i = 0; i < size; i++) {
        int idx = storage[i].sym;   // read before overwriting (union aliases)
        int len = storage[i].len;
        int level, run;
        if (len == 0) {             // unused / illegal
            run = 66; level = MAX_LEVEL_MP4;
        } else if (idx == n) {      // ESC
            run = 66; level = 0;
        } else {
            run   = table_run[idx] + 1;
            level = table_level[idx] * qmul + qadd;
            if (idx >= last) run += 192;
        }
        storage[i].level = (int16_t)level;
        storage[i].len8  = (int8_t)len;
        storage[i].run   = (uint8_t)run;
    }
    // aux: max_level[last][run], max_run[last][level] (ff_rl_init)
    for (int la = 0; la < 2; la++) {
        for (int k = 0; k <= MAX_LEVEL_MP4; k++) rl->max_run[la][k] = 0;
        for (int k = 0; k <= MAX_RUN_MP4;   k++) rl->max_level[la][k] = 0;
        int start = la == 0 ? 0 : last;
        int end   = la == 0 ? last : n;
        for (int i = start; i < end; i++) {
            int r = table_run[i], lv = table_level[i];
            if (lv > rl->max_level[la][r]) rl->max_level[la][r] = (int8_t)lv;
            if (r  > rl->max_run[la][lv])  rl->max_run[la][lv]  = (int8_t)r;
        }
    }
    rl->table = storage; rl->bits = bits; rl->n = n; rl->last = last;
    rl->table_run = table_run; rl->table_level = table_level;
}

// Single-level get_vlc2: one show_bits(bits) lookup. Returns sym (-1 if invalid)
// and advances by the code length (0 for invalid, matching ffmpeg max_depth=1).
static inline int get_vlc2(GetBits* s, const VLCElem* table, int bits) {
    unsigned index = show_bits(s, bits);
    int sym = table[index].sym;
    int n   = table[index].len;
    skip_bits(s, n);
    return sym;
}

// Single-level GET_RL_VLC: writes level/run, advances by len8.
static inline void get_rl_vlc(GetBits* s, const RL_VLC_ELEM* table, int bits,
                              int* level, int* run) {
    unsigned index = show_bits(s, bits);
    *level = table[index].level;
    int n  = table[index].len8;
    *run   = table[index].run;
    skip_bits(s, n);
}

} // namespace mp4

#endif // MPEG4_MP4_VLC_H
