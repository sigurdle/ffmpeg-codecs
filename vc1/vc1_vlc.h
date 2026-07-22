/*
    Multi-level VLC (variable-length code) reader — a faithful port of FFmpeg
    release/8.1's libavcodec/vlc.c (build_table + the sort in ff_vlc_init_sparse)
    and the GET_VLC macro from get_bits.h. FFmpeg is LGPL-2.1+.

    The mpeg4/msmpeg4 sibling ports use a FLAT single-level LUT (mp4_vlc.h), which
    is fine for their short tables. VC-1's AC coefficient VLCs have codes up to 23
    bits — a flat 2^23 LUT would be 32 MB per table. So this reader keeps ffmpeg's
    real structure: a 1<<bits primary table whose long-code slots point at
    recursively-built subtables, walked by get_vlc2 up to `max_depth` levels.

    Reuses mp4::VLCElem {sym,len} and the mp4::GetBits reader.
*/

#ifndef VC1_VLC_H
#define VC1_VLC_H

#include <stdint.h>
#include "../mpeg4/mp4_bits.h"
#include "../mpeg4/mp4_vlc.h"   // mp4::VLCElem

namespace vc1 {

// A code awaiting placement: `code` is LEFT-justified into 32 bits (ffmpeg's form).
struct VLCcode { uint32_t code; int bits; int symbol; };

// Shared pool cursor across the VLCs built into it (mirrors VLCInitState). Each
// ml_init() takes a slice starting at `cursor`; subtable offsets are stored RELATIVE
// to that slice's base (as ffmpeg's `vlc.table = state->table` makes them), so the
// returned primary pointer indexes its own subtables directly. `ok` trips on overflow.
struct MLBuild { mp4::VLCElem* pool; int cursor; int cap; int ok; };

// build_table (vlc.c) — recursive, into `base[]` with a base-local cursor `*psize`.
// `codes` is mutated (prefixes stripped) as ffmpeg does. Returns the (sub)table's
// base-relative index.
static int ml_build_table(mp4::VLCElem* base, int* psize, int cap, int* ok,
                          int table_nb_bits, int nb_codes, VLCcode* codes) {
    int table_size  = 1 << table_nb_bits;
    int table_index = *psize;
    *psize += table_size;
    if (*psize > cap) { *ok = 0; return table_index; }
    mp4::VLCElem* table = &base[table_index];
    for (int i = 0; i < table_size; i++) { table[i].sym = 0; table[i].len = 0; }

    for (int i = 0; i < nb_codes; i++) {
        int n         = codes[i].bits;
        uint32_t code = codes[i].code;
        int symbol    = codes[i].symbol;
        if (n <= table_nb_bits) {
            int j  = code >> (32 - table_nb_bits);
            int nb = 1 << (table_nb_bits - n);
            for (int k = 0; k < nb; k++) { table[j].len = (int16_t)n; table[j].sym = (int16_t)symbol; j++; }
        } else {
            uint32_t code_prefix;
            int index, subtable_bits, j, k;
            n -= table_nb_bits;
            code_prefix = code >> (32 - table_nb_bits);
            subtable_bits = n;
            codes[i].bits = n;
            codes[i].code = code << table_nb_bits;
            for (k = i + 1; k < nb_codes; k++) {
                int n2 = codes[k].bits - table_nb_bits;
                if (n2 <= 0) break;
                uint32_t c2 = codes[k].code;
                if (c2 >> (32 - table_nb_bits) != code_prefix) break;
                codes[k].bits = n2;
                codes[k].code = c2 << table_nb_bits;
                if (n2 > subtable_bits) subtable_bits = n2;
            }
            if (subtable_bits > table_nb_bits) subtable_bits = table_nb_bits;
            j = code_prefix;
            table[j].len = (int16_t)(-subtable_bits);
            index = ml_build_table(base, psize, cap, ok, subtable_bits, k - i, codes + i);
            if (!*ok) return table_index;
            table = &base[table_index];   // reload after recursion (mirrors ffmpeg)
            table[j].sym = (int16_t)index;
            i = k - 1;
        }
    }
    for (int i = 0; i < table_size; i++)
        if (table[i].len == 0) table[i].sym = -1;
    return table_index;
}

static inline uint32_t ml_read(const void* base, int i, int wrap, int size) {
    const uint8_t* p = (const uint8_t*)base + (size_t)i * wrap;
    if (size == 1) return *p;
    if (size == 2) return *(const uint16_t*)p;
    return *(const uint32_t*)p;
}

// ff_vlc_init_sparse (symbols == NULL => symbol is the index). Gathers long codes
// (len > nb_bits) first, sorts them ascending by code, then appends the short codes
// in original order — exactly the two-phase COPY + AV_QSORT(compare_vlcspec) ffmpeg
// does — and builds. Returns the primary table pointer.
static inline const mp4::VLCElem* ml_init(MLBuild* b, int nb_bits, int nb_codes,
                                          const void* bits, int bits_wrap, int bits_size,
                                          const void* codes, int codes_wrap, int codes_size) {
    VLCcode buf[512];   // nb_codes <= 186 for the VC-1 AC tables
    int j = 0;
    for (int i = 0; i < nb_codes; i++) {
        int len = (int)ml_read(bits, i, bits_wrap, bits_size);
        if (len <= nb_bits) continue;                      // long codes only
        uint32_t code = ml_read(codes, i, codes_wrap, codes_size);
        buf[j].bits = len; buf[j].code = code << (32 - len); buf[j].symbol = i; j++;
    }
    // insertion sort long codes ascending by (code >> 1) — codes are distinct and,
    // for lengths <= 23, carry >= 9 low zero bits, so (code >> 1) is a total order.
    for (int a = 1; a < j; a++) {
        VLCcode key = buf[a]; int c = a - 1;
        while (c >= 0 && (buf[c].code >> 1) > (key.code >> 1)) { buf[c + 1] = buf[c]; c--; }
        buf[c + 1] = key;
    }
    for (int i = 0; i < nb_codes; i++) {
        int len = (int)ml_read(bits, i, bits_wrap, bits_size);
        if (!(len && len <= nb_bits)) continue;            // short codes, original order
        uint32_t code = ml_read(codes, i, codes_wrap, codes_size);
        buf[j].bits = len; buf[j].code = code << (32 - len); buf[j].symbol = i; j++;
    }
    mp4::VLCElem* slice = &b->pool[b->cursor];   // this VLC's own base (ffmpeg vlc.table)
    int localsize = 0;
    ml_build_table(slice, &localsize, b->cap - b->cursor, &b->ok, nb_bits, j, buf);
    b->cursor += localsize;
    return slice;
}

// GET_VLC (get_bits.h) — walk up to max_depth levels of subtables. Returns the
// symbol, or -1 for an invalid code. Advances the reader by the code length.
static inline int ml_get_vlc2(mp4::GetBits* gb, const mp4::VLCElem* table, int bits, int max_depth) {
    unsigned index = mp4::show_bits(gb, bits);
    int code = table[index].sym;
    int n    = table[index].len;
    if (max_depth > 1 && n < 0) {
        mp4::skip_bits(gb, bits);
        int nb = -n;
        index = mp4::show_bits(gb, nb) + code;
        code  = table[index].sym;
        n     = table[index].len;
        if (max_depth > 2 && n < 0) {
            mp4::skip_bits(gb, nb);
            nb = -n;
            index = mp4::show_bits(gb, nb) + code;
            code  = table[index].sym;
            n     = table[index].len;
        }
    }
    mp4::skip_bits(gb, n);
    return code;
}

} // namespace vc1

#endif // VC1_VLC_H
