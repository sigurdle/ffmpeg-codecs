/*
    Big-endian MSB-first bitstream reader — the subset of FFmpeg's get_bits.h
    (libavcodec/get_bits.h, LGPL) needed by the MPEG-4 Part 2 / H.263 decoder.

    Non-cached, index-based (matches ffmpeg's UNCHECKED_BITSTREAM_READER reads):
    show/get extract the next n MSB-first bits from a 32-bit window AV_RB32(buf +
    (index>>3)) << (index&7) >> (32-n). The CALLER must pad the input buffer with
    >= 8 trailing zero bytes so the 4-byte window reads never run off the end
    (ffmpeg relies on AV_INPUT_BUFFER_PADDING_SIZE for the same reason).

    -fwrapv is required (get_xbits sign arithmetic). See mpeg4_decode.cpp header.
*/

#ifndef MPEG4_MP4_BITS_H
#define MPEG4_MP4_BITS_H

#include <stdint.h>

namespace mp4 {

struct GetBits {
    const uint8_t* buf;
    int index;          // bit position
    int size_in_bits;
};

static inline void init_get_bits(GetBits* s, const uint8_t* buf, int byte_size) {
    s->buf = buf;
    s->index = 0;
    s->size_in_bits = byte_size * 8;
}

static inline int get_bits_count(const GetBits* s) { return s->index; }
static inline int get_bits_left(const GetBits* s)  { return s->size_in_bits - s->index; }

// 32-bit MSB-first window starting at the current byte (needs 4 readable bytes).
static inline uint32_t mp4_rb32(const uint8_t* p) {
    return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) | ((uint32_t)p[2] << 8) | p[3];
}

// Peek n bits (1..25) without advancing.
static inline unsigned show_bits(GetBits* s, int n) {
    uint32_t c = mp4_rb32(s->buf + (s->index >> 3));
    c <<= (s->index & 7);
    return c >> (32 - n);
}

static inline unsigned get_bits(GetBits* s, int n) {
    unsigned r = show_bits(s, n);
    s->index += n;
    return r;
}

static inline void skip_bits(GetBits* s, int n) { s->index += n; }
static inline void skip_bits1(GetBits* s)       { s->index += 1; }

static inline unsigned get_bits1(GetBits* s) {
    uint8_t r = s->buf[s->index >> 3];
    r <<= (s->index & 7);   // truncates to 8 bits (uint8_t) -> wanted bit to MSB
    r >>= 7;
    s->index += 1;
    return r;
}

// n in 1..32 (header fields are <=16, but VOS etc. may want more).
static inline unsigned get_bits_long(GetBits* s, int n) {
    if (n <= 25) return get_bits(s, n);
    unsigned hi = get_bits(s, 16);
    return (hi << (n - 16)) | get_bits(s, n - 16);
}

// Signed-extended n-bit field used for the intra-DC residual mantissa
// (ff_get_xbits). Reads the magnitude then sign-extends per the DC convention.
static inline int get_xbits(GetBits* s, int n) {
    int32_t cache = (int32_t)mp4_rb32(s->buf + (s->index >> 3));
    cache <<= (s->index & 7);
    int32_t sign = (~cache) >> 31;          // 0 (pos) or -1 (neg), arithmetic shift
    s->index += n;
    return (int)(((uint32_t)(sign ^ cache) >> (32 - n)) ^ (uint32_t)sign) - sign;
}

// 32-bit MSB-first cache at the current position (GET_CACHE), no advance.
static inline uint32_t get_cache(GetBits* s) {
    return mp4_rb32(s->buf + (s->index >> 3)) << (s->index & 7);
}

// Sign-extend the next n bits to int (advances). Used for the 12-bit escape level.
static inline int get_sbits(GetBits* s, int n) {
    int v = (int)((int32_t)get_cache(s) >> (32 - n));
    s->index += n;
    return v;
}

// SHOW_SBITS(1): the next bit sign-extended to 0 (bit 0) or -1 (bit 1), no advance.
static inline int show_sign1(GetBits* s) {
    return (int)((int32_t)get_cache(s) >> 31);
}

// Round the bit position up to the next byte boundary; return the byte pointer.
static inline const uint8_t* align_get_bits(GetBits* s) {
    int n = (-s->index) & 7;
    if (n) skip_bits(s, n);
    return s->buf + (s->index >> 3);
}

// Marker bit: consume exactly one bit; never errors on a 0 (ffmpeg check_marker
// only logs). Returns the bit.
static inline int check_marker(GetBits* s) { return (int)get_bits1(s); }

} // namespace mp4

#endif // MPEG4_MP4_BITS_H
