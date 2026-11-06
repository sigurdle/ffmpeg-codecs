/*
	The Opus range decoder: FFmpeg n8.1 libavcodec/opus/rc.c and rc.h (LGPL-2.1+), the decoder's half. The functions only CELT
	calls are here too, from the same file.

	Copyright (c) 2012 Andrew D'Addesio
	Copyright (c) 2013-2014 Mozilla Corporation
	Copyright (c) 2017 Rostislav Pehlivanov

	Departures from FFmpeg:
	  - The encoder is gone, and with it the encoder's fields of OpusRangeCoder.
	  - ff_sqrt's table walk is sqrt() in double, which gives the same floor for every 32-bit value.

	The bit reader is get_bits.h's checked one: past the end it stops 8 bits on, and the bytes there are the next frame's or
	the input's zeroed padding, as in FFmpeg.
*/

#pragma once

struct GetBitContext {
    const uint8_t *buffer;
    int index;
    int size_in_bits_plus8;
};

static inline void init_get_bits8(GetBitContext *gb, const uint8_t *buffer, int bytes)
{
    gb->buffer = buffer;
    gb->index = 0;
    gb->size_in_bits_plus8 = bytes * 8 + 8;
}

static inline uint32_t get_bits(GetBitContext *gb, int n)
{
    const uint8_t *p = gb->buffer + (gb->index >> 3);
    uint32_t cache = (uint32_t)p[0] << 24 | p[1] << 16 | p[2] << 8 | p[3];
    uint32_t value = (cache << (gb->index & 7)) >> (32 - n);
    gb->index = FFMIN(gb->size_in_bits_plus8, gb->index + n);
    return value;
}

#define opus_ilog(i) (av_log2(i) + !!(i))

typedef struct RawBitsContext {
    const uint8_t *position;
    uint32_t bytes;
    uint32_t cachelen;
    uint32_t cacheval;
} RawBitsContext;

typedef struct OpusRangeCoder {
    GetBitContext gb;
    RawBitsContext rb;
    uint32_t range;
    uint32_t value;
    uint32_t total_bits;
} OpusRangeCoder;

/**
 * CELT: estimate bits of entropy that have thus far been consumed for the
 *       current CELT frame, to integer and fractional (1/8th bit) precision
 */
static inline uint32_t opus_rc_tell(const OpusRangeCoder *rc)
{
    return rc->total_bits - av_log2(rc->range) - 1;
}

static inline uint32_t opus_rc_tell_frac(const OpusRangeCoder *rc)
{
    uint32_t i, total_bits, rcbuffer, range;

    total_bits = rc->total_bits << 3;
    rcbuffer   = av_log2(rc->range) + 1;
    range      = rc->range >> (rcbuffer-16);

    for (i = 0; i < 3; i++) {
        int bit;
        range = range * range >> 15;
        bit = range >> 16;
        rcbuffer = rcbuffer << 1 | bit;
        range >>= bit;
    }

    return total_bits - rcbuffer;
}

#define OPUS_RC_SYM  8
#define OPUS_RC_CEIL ((1 << OPUS_RC_SYM) - 1)
#define OPUS_RC_TOP (1u << 31)
#define OPUS_RC_BOT (OPUS_RC_TOP >> OPUS_RC_SYM)

static inline unsigned int ff_sqrt(unsigned int a)
{
    return (unsigned int)sqrt((double)a);
}

static inline void opus_rc_dec_normalize(OpusRangeCoder *rc)
{
    while (rc->range <= OPUS_RC_BOT) {
        rc->value = ((rc->value << OPUS_RC_SYM) | (get_bits(&rc->gb, OPUS_RC_SYM) ^ OPUS_RC_CEIL)) & (OPUS_RC_TOP - 1);
        rc->range     <<= OPUS_RC_SYM;
        rc->total_bits += OPUS_RC_SYM;
    }
}

static inline void opus_rc_dec_update(OpusRangeCoder *rc, uint32_t scale,
                                                uint32_t low, uint32_t high,
                                                uint32_t total)
{
    rc->value -= scale * (total - high);
    rc->range  = low ? scale * (high - low)
                      : rc->range - scale * (total - high);
    opus_rc_dec_normalize(rc);
}

static uint32_t ff_opus_rc_dec_cdf(OpusRangeCoder *rc, const uint16_t *cdf)
{
    unsigned int k, scale, total, symbol, low, high;

    total = *cdf++;

    scale   = rc->range / total;
    symbol = rc->value / scale + 1;
    symbol = total - FFMIN(symbol, total);

    for (k = 0; cdf[k] <= symbol; k++);
    high = cdf[k];
    low  = k ? cdf[k-1] : 0;

    opus_rc_dec_update(rc, scale, low, high, total);

    return k;
}

static uint32_t ff_opus_rc_dec_log(OpusRangeCoder *rc, uint32_t bits)
{
    uint32_t k, scale;
    scale = rc->range >> bits; // in this case, scale = symbol

    if (rc->value >= scale) {
        rc->value -= scale;
        rc->range -= scale;
        k = 0;
    } else {
        rc->range = scale;
        k = 1;
    }
    opus_rc_dec_normalize(rc);
    return k;
}

static uint32_t ff_opus_rc_get_raw(OpusRangeCoder *rc, uint32_t count)
{
    uint32_t value = 0;

    while (rc->rb.bytes && rc->rb.cachelen < count) {
        rc->rb.cacheval |= *--rc->rb.position << rc->rb.cachelen;
        rc->rb.cachelen += 8;
        rc->rb.bytes--;
    }

    value = (count ? rc->rb.cacheval & (0xFFFFFFFFu >> (32 - count)) : 0);
    rc->rb.cacheval    >>= count;
    rc->rb.cachelen     -= count;
    rc->total_bits      += count;

    return value;
}

static uint32_t ff_opus_rc_dec_uint(OpusRangeCoder *rc, uint32_t size)
{
    uint32_t bits, k, scale, total;

    bits  = opus_ilog(size - 1);
    total = (bits > 8) ? ((size - 1) >> (bits - 8)) + 1 : size;

    scale  = rc->range / total;
    k      = rc->value / scale + 1;
    k      = total - FFMIN(k, total);
    opus_rc_dec_update(rc, scale, k, k + 1, total);

    if (bits > 8) {
        k = k << (bits - 8) | ff_opus_rc_get_raw(rc, bits - 8);
        return FFMIN(k, size - 1);
    } else
        return k;
}

static uint32_t ff_opus_rc_dec_uint_step(OpusRangeCoder *rc, int k0)
{
    /* Use a probability of 3 up to itheta=8192 and then use 1 after */
    uint32_t k, scale, symbol, total = (k0+1)*3 + k0;
    scale  = rc->range / total;
    symbol = rc->value / scale + 1;
    symbol = total - FFMIN(symbol, total);

    k = (symbol < (k0+1)*3) ? symbol/3 : symbol - (k0+1)*2;

    opus_rc_dec_update(rc, scale, (k <= k0) ? 3*(k+0) : (k-1-k0) + 3*(k0+1),
                       (k <= k0) ? 3*(k+1) : (k-0-k0) + 3*(k0+1), total);
    return k;
}

static uint32_t ff_opus_rc_dec_uint_tri(OpusRangeCoder *rc, int qn)
{
    uint32_t k, scale, symbol, total, low, center;

    total = ((qn>>1) + 1) * ((qn>>1) + 1);
    scale   = rc->range / total;
    center = rc->value / scale + 1;
    center = total - FFMIN(center, total);

    if (center < total >> 1) {
        k      = (ff_sqrt(8 * center + 1) - 1) >> 1;
        low    = k * (k + 1) >> 1;
        symbol = k + 1;
    } else {
        k      = (2*(qn + 1) - ff_sqrt(8*(total - center - 1) + 1)) >> 1;
        low    = total - ((qn + 1 - k) * (qn + 2 - k) >> 1);
        symbol = qn + 1 - k;
    }

    opus_rc_dec_update(rc, scale, low, low + symbol, total);

    return k;
}

static int ff_opus_rc_dec_laplace(OpusRangeCoder *rc, uint32_t symbol, int decay)
{
    /* extends the range coder to model a Laplace distribution */
    int value = 0;
    uint32_t scale, low = 0, center;

    scale  = rc->range >> 15;
    center = rc->value / scale + 1;
    center = (1 << 15) - FFMIN(center, 1 << 15);

    if (center >= symbol) {
        value++;
        low = symbol;
        symbol = 1 + ((32768 - 32 - symbol) * (16384-decay) >> 15);

        while (symbol > 1 && center >= low + 2 * symbol) {
            value++;
            symbol *= 2;
            low    += symbol;
            symbol  = (((symbol - 2) * decay) >> 15) + 1;
        }

        if (symbol <= 1) {
            int distance = (center - low) >> 1;
            value += distance;
            low   += 2 * distance;
        }

        if (center < low + symbol)
            value *= -1;
        else
            low += symbol;
    }

    opus_rc_dec_update(rc, scale, low, FFMIN(low + symbol, 32768), 32768);

    return value;
}

static int ff_opus_rc_dec_init(OpusRangeCoder *rc, const uint8_t *data, int size)
{
    init_get_bits8(&rc->gb, data, size);

    rc->range = 128;
    rc->value = 127 - get_bits(&rc->gb, 7);
    rc->total_bits = 9;
    opus_rc_dec_normalize(rc);

    return 0;
}

static void ff_opus_rc_dec_raw_init(OpusRangeCoder *rc, const uint8_t *rightend, uint32_t bytes)
{
    rc->rb.position = rightend;
    rc->rb.bytes    = bytes;
    rc->rb.cachelen = 0;
    rc->rb.cacheval = 0;
}
