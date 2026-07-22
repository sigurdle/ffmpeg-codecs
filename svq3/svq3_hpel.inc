// Half-pel MC kernels, transliterated from ffmpeg release/8.1:
//   libavcodec/pel_template.c, libavcodec/hpel_template.c, libavcodec/hpeldsp.c
// 8-bit instantiation (BIT_DEPTH=8): pixel=uint8_t, pixel2=uint16_t, pixel4=uint32_t,
//   AV_RN2P=AV_RN16, AV_RN4P=AV_RN32, rnd_avg_pixel4=rnd_avg32.
// ROUNDING variants only (SVQ3 never touches the no_rnd tables). av_log/av_assert/ctx dropped.

// ---- rnd_avg.h helpers (rounding average only) -----------------------------
#ifndef SVQ3_HPEL_BYTE_VEC32
#define SVQ3_HPEL_BYTE_VEC32(c) ((c) * 0x01010101UL)
#endif

static inline uint32_t svq3_rnd_avg32(uint32_t a, uint32_t b)
{
    return (a | b) - (((a ^ b) & ~SVQ3_HPEL_BYTE_VEC32(0x01)) >> 1);
}

// AV_RN16 is not in the integration contract's provided set; define an unaligned
// 16-bit little-endian-native reader with the same semantics as ffmpeg's AV_RN16.
static inline uint16_t svq3_hpel_rn16(const void *p)
{
    uint16_t v;
    memcpy(&v, p, sizeof(v));
    return v;
}

// ============================================================================
// pel_template.c  --  DEF_PEL(put/avg): plain copy kernels (dxy == 0)
// pixel4 = uint32_t, AV_RN4P = AV_RN32 (== AV_RN32A on any-align via memcpy path;
// ffmpeg uses AV_RN32 which is unaligned). We use AV_RN32A/AV_WN32A provided by TU
// for the aligned 4-wide stores/loads, matching pel_template's *((pixel4*)block).
// ----------------------------------------------------------------------------

// put_pixels2 (dxy==0), 2 wide
static inline void svq3_put_pixels2_8_c(uint8_t *block, const uint8_t *pixels,
                                        ptrdiff_t line_size, int h)
{
    int i;
    for (i = 0; i < h; i++) {
        *((uint16_t *) block) = svq3_hpel_rn16(pixels);
        pixels += line_size;
        block  += line_size;
    }
}

// avg_pixels2 (dxy==0), 2 wide
static inline void svq3_avg_pixels2_8_c(uint8_t *block, const uint8_t *pixels,
                                        ptrdiff_t line_size, int h)
{
    int i;
    for (i = 0; i < h; i++) {
        // op_avg on a pixel2: rnd_avg_pixel4 over the 16-bit value (as in pel_template)
        *((uint16_t *) block) =
            (uint16_t) svq3_rnd_avg32(*((uint16_t *) block), svq3_hpel_rn16(pixels));
        pixels += line_size;
        block  += line_size;
    }
}

// put_pixels4 (dxy==0), 4 wide
static inline void svq3_put_pixels4_8_c(uint8_t *block, const uint8_t *pixels,
                                        ptrdiff_t line_size, int h)
{
    int i;
    for (i = 0; i < h; i++) {
        AV_WN32A(block, AV_RN32A(pixels));
        pixels += line_size;
        block  += line_size;
    }
}

// avg_pixels4 (dxy==0), 4 wide
static inline void svq3_avg_pixels4_8_c(uint8_t *block, const uint8_t *pixels,
                                        ptrdiff_t line_size, int h)
{
    int i;
    for (i = 0; i < h; i++) {
        AV_WN32A(block, svq3_rnd_avg32(AV_RN32A(block), AV_RN32A(pixels)));
        pixels += line_size;
        block  += line_size;
    }
}

// put_pixels8 (dxy==0), 8 wide
static inline void svq3_put_pixels8_8_c(uint8_t *block, const uint8_t *pixels,
                                        ptrdiff_t line_size, int h)
{
    int i;
    for (i = 0; i < h; i++) {
        AV_WN32A(block,     AV_RN32A(pixels));
        AV_WN32A(block + 4, AV_RN32A(pixels + 4));
        pixels += line_size;
        block  += line_size;
    }
}

// avg_pixels8 (dxy==0), 8 wide
static inline void svq3_avg_pixels8_8_c(uint8_t *block, const uint8_t *pixels,
                                        ptrdiff_t line_size, int h)
{
    int i;
    for (i = 0; i < h; i++) {
        AV_WN32A(block,     svq3_rnd_avg32(AV_RN32A(block),     AV_RN32A(pixels)));
        AV_WN32A(block + 4, svq3_rnd_avg32(AV_RN32A(block + 4), AV_RN32A(pixels + 4)));
        pixels += line_size;
        block  += line_size;
    }
}

// put_pixels16 (dxy==0), 16 wide  --  CALL_2X_PIXELS(put_pixels16, put_pixels8, 8)
static inline void svq3_put_pixels16_8_c(uint8_t *block, const uint8_t *pixels,
                                         ptrdiff_t line_size, int h)
{
    svq3_put_pixels8_8_c(block,     pixels,     line_size, h);
    svq3_put_pixels8_8_c(block + 8, pixels + 8, line_size, h);
}

// avg_pixels16 (dxy==0), 16 wide
static inline void svq3_avg_pixels16_8_c(uint8_t *block, const uint8_t *pixels,
                                         ptrdiff_t line_size, int h)
{
    svq3_avg_pixels8_8_c(block,     pixels,     line_size, h);
    svq3_avg_pixels8_8_c(block + 8, pixels + 8, line_size, h);
}

// ============================================================================
// hpel_template.c  --  DEF_HPEL(put/avg): _l2 (linear-average of two sources)
// Used to build the x2 (horizontal half) and y2 (vertical half) kernels.
// ----------------------------------------------------------------------------

// put_pixels8_l2 : dst = rnd_avg( src1, src2 ), 8 wide
static inline void svq3_put_pixels8_l2_8(uint8_t *dst,
                                         const uint8_t *src1, const uint8_t *src2,
                                         ptrdiff_t dst_stride,
                                         ptrdiff_t src_stride1, ptrdiff_t src_stride2,
                                         int h)
{
    int i;
    for (i = 0; i < h; i++) {
        uint32_t a, b;
        a = AV_RN32A(&src1[i * src_stride1]);
        b = AV_RN32A(&src2[i * src_stride2]);
        *((uint32_t *) &dst[i * dst_stride]) = svq3_rnd_avg32(a, b);
        a = AV_RN32A(&src1[i * src_stride1 + 4]);
        b = AV_RN32A(&src2[i * src_stride2 + 4]);
        *((uint32_t *) &dst[i * dst_stride + 4]) = svq3_rnd_avg32(a, b);
    }
}

// avg_pixels8_l2 : dst = rnd_avg( dst, rnd_avg(src1,src2) ), 8 wide
static inline void svq3_avg_pixels8_l2_8(uint8_t *dst,
                                         const uint8_t *src1, const uint8_t *src2,
                                         ptrdiff_t dst_stride,
                                         ptrdiff_t src_stride1, ptrdiff_t src_stride2,
                                         int h)
{
    int i;
    for (i = 0; i < h; i++) {
        uint32_t a, b;
        a = AV_RN32A(&src1[i * src_stride1]);
        b = AV_RN32A(&src2[i * src_stride2]);
        {
            uint32_t *d = (uint32_t *) &dst[i * dst_stride];
            *d = svq3_rnd_avg32(*d, svq3_rnd_avg32(a, b));
        }
        a = AV_RN32A(&src1[i * src_stride1 + 4]);
        b = AV_RN32A(&src2[i * src_stride2 + 4]);
        {
            uint32_t *d = (uint32_t *) &dst[i * dst_stride + 4];
            *d = svq3_rnd_avg32(*d, svq3_rnd_avg32(a, b));
        }
    }
}

// put_pixels4_l2 : 4 wide
static inline void svq3_put_pixels4_l2_8(uint8_t *dst,
                                         const uint8_t *src1, const uint8_t *src2,
                                         ptrdiff_t dst_stride,
                                         ptrdiff_t src_stride1, ptrdiff_t src_stride2,
                                         int h)
{
    int i;
    for (i = 0; i < h; i++) {
        uint32_t a, b;
        a = AV_RN32A(&src1[i * src_stride1]);
        b = AV_RN32A(&src2[i * src_stride2]);
        *((uint32_t *) &dst[i * dst_stride]) = svq3_rnd_avg32(a, b);
    }
}

// avg_pixels4_l2 : 4 wide
static inline void svq3_avg_pixels4_l2_8(uint8_t *dst,
                                         const uint8_t *src1, const uint8_t *src2,
                                         ptrdiff_t dst_stride,
                                         ptrdiff_t src_stride1, ptrdiff_t src_stride2,
                                         int h)
{
    int i;
    for (i = 0; i < h; i++) {
        uint32_t a, b;
        a = AV_RN32A(&src1[i * src_stride1]);
        b = AV_RN32A(&src2[i * src_stride2]);
        {
            uint32_t *d = (uint32_t *) &dst[i * dst_stride];
            *d = svq3_rnd_avg32(*d, svq3_rnd_avg32(a, b));
        }
    }
}

// put_pixels2_l2 : 2 wide  (AV_RN2P = AV_RN16; result written as pixel2)
static inline void svq3_put_pixels2_l2_8(uint8_t *dst,
                                         const uint8_t *src1, const uint8_t *src2,
                                         ptrdiff_t dst_stride,
                                         ptrdiff_t src_stride1, ptrdiff_t src_stride2,
                                         int h)
{
    int i;
    for (i = 0; i < h; i++) {
        uint32_t a, b;
        a = svq3_hpel_rn16(&src1[i * src_stride1]);
        b = svq3_hpel_rn16(&src2[i * src_stride2]);
        *((uint16_t *) &dst[i * dst_stride]) = (uint16_t) svq3_rnd_avg32(a, b);
    }
}

// avg_pixels2_l2 : 2 wide
static inline void svq3_avg_pixels2_l2_8(uint8_t *dst,
                                         const uint8_t *src1, const uint8_t *src2,
                                         ptrdiff_t dst_stride,
                                         ptrdiff_t src_stride1, ptrdiff_t src_stride2,
                                         int h)
{
    int i;
    for (i = 0; i < h; i++) {
        uint32_t a, b;
        a = svq3_hpel_rn16(&src1[i * src_stride1]);
        b = svq3_hpel_rn16(&src2[i * src_stride2]);
        {
            uint16_t *d = (uint16_t *) &dst[i * dst_stride];
            *d = (uint16_t) svq3_rnd_avg32(*d, svq3_rnd_avg32(a, b));
        }
    }
}

// put_pixels16_l2 / avg_pixels16_l2 : two 8-wide _l2 calls, offset 8
static inline void svq3_put_pixels16_l2_8(uint8_t *dst,
                                          const uint8_t *src1, const uint8_t *src2,
                                          ptrdiff_t dst_stride,
                                          ptrdiff_t src_stride1, ptrdiff_t src_stride2,
                                          int h)
{
    svq3_put_pixels8_l2_8(dst, src1, src2, dst_stride, src_stride1, src_stride2, h);
    svq3_put_pixels8_l2_8(dst + 8, src1 + 8, src2 + 8,
                          dst_stride, src_stride1, src_stride2, h);
}

static inline void svq3_avg_pixels16_l2_8(uint8_t *dst,
                                          const uint8_t *src1, const uint8_t *src2,
                                          ptrdiff_t dst_stride,
                                          ptrdiff_t src_stride1, ptrdiff_t src_stride2,
                                          int h)
{
    svq3_avg_pixels8_l2_8(dst, src1, src2, dst_stride, src_stride1, src_stride2, h);
    svq3_avg_pixels8_l2_8(dst + 8, src1 + 8, src2 + 8,
                          dst_stride, src_stride1, src_stride2, h);
}

// ============================================================================
// hpeldsp.c  --  PIXOP2(put/avg): x2 (xhalf), y2 (yhalf), xy2 (both) kernels.
// x2 == _l2(pixels, pixels+1); y2 == _l2(pixels, pixels+line_size).
// ----------------------------------------------------------------------------

// -------- x2 (horizontal half-pel), dxy == 1 --------
static void svq3_put_pixels8_x2_8_c(uint8_t *block, const uint8_t *pixels,
                                    ptrdiff_t line_size, int h)
{
    svq3_put_pixels8_l2_8(block, pixels, pixels + 1,
                          line_size, line_size, line_size, h);
}
static void svq3_avg_pixels8_x2_8_c(uint8_t *block, const uint8_t *pixels,
                                    ptrdiff_t line_size, int h)
{
    svq3_avg_pixels8_l2_8(block, pixels, pixels + 1,
                          line_size, line_size, line_size, h);
}
static void svq3_put_pixels4_x2_8_c(uint8_t *block, const uint8_t *pixels,
                                    ptrdiff_t line_size, int h)
{
    svq3_put_pixels4_l2_8(block, pixels, pixels + 1,
                          line_size, line_size, line_size, h);
}
static void svq3_avg_pixels4_x2_8_c(uint8_t *block, const uint8_t *pixels,
                                    ptrdiff_t line_size, int h)
{
    svq3_avg_pixels4_l2_8(block, pixels, pixels + 1,
                          line_size, line_size, line_size, h);
}
static void svq3_put_pixels2_x2_8_c(uint8_t *block, const uint8_t *pixels,
                                    ptrdiff_t line_size, int h)
{
    svq3_put_pixels2_l2_8(block, pixels, pixels + 1,
                          line_size, line_size, line_size, h);
}
static void svq3_avg_pixels2_x2_8_c(uint8_t *block, const uint8_t *pixels,
                                    ptrdiff_t line_size, int h)
{
    svq3_avg_pixels2_l2_8(block, pixels, pixels + 1,
                          line_size, line_size, line_size, h);
}

// -------- y2 (vertical half-pel), dxy == 2 --------
static void svq3_put_pixels8_y2_8_c(uint8_t *block, const uint8_t *pixels,
                                    ptrdiff_t line_size, int h)
{
    svq3_put_pixels8_l2_8(block, pixels, pixels + line_size,
                          line_size, line_size, line_size, h);
}
static void svq3_avg_pixels8_y2_8_c(uint8_t *block, const uint8_t *pixels,
                                    ptrdiff_t line_size, int h)
{
    svq3_avg_pixels8_l2_8(block, pixels, pixels + line_size,
                          line_size, line_size, line_size, h);
}
static void svq3_put_pixels4_y2_8_c(uint8_t *block, const uint8_t *pixels,
                                    ptrdiff_t line_size, int h)
{
    svq3_put_pixels4_l2_8(block, pixels, pixels + line_size,
                          line_size, line_size, line_size, h);
}
static void svq3_avg_pixels4_y2_8_c(uint8_t *block, const uint8_t *pixels,
                                    ptrdiff_t line_size, int h)
{
    svq3_avg_pixels4_l2_8(block, pixels, pixels + line_size,
                          line_size, line_size, line_size, h);
}
static void svq3_put_pixels2_y2_8_c(uint8_t *block, const uint8_t *pixels,
                                    ptrdiff_t line_size, int h)
{
    svq3_put_pixels2_l2_8(block, pixels, pixels + line_size,
                          line_size, line_size, line_size, h);
}
static void svq3_avg_pixels2_y2_8_c(uint8_t *block, const uint8_t *pixels,
                                    ptrdiff_t line_size, int h)
{
    svq3_avg_pixels2_l2_8(block, pixels, pixels + line_size,
                          line_size, line_size, line_size, h);
}

// -------- xy2 (both half-pel), dxy == 3 --------
// op_put(a,b)=a=b ; op_avg(a,b)=a=rnd_avg32(a,b). These are the exact byte-parallel
// bilinear kernels from hpeldsp.c (rounding: +2 bias in low bits).

// pixels2_xy2 : always a plain put (block[..] = (..)>>2, no OP indirection in ffmpeg)
static void svq3_put_pixels2_xy2_8_c(uint8_t *block, const uint8_t *pixels,
                                     ptrdiff_t line_size, int h)
{
    int i, a1, b1;
    int a0 = pixels[0];
    int b0 = pixels[1] + 2;

    a0 += b0;
    b0 += pixels[2];
    pixels += line_size;
    for (i = 0; i < h; i += 2) {
        a1  = pixels[0];
        b1  = pixels[1];
        a1 += b1;
        b1 += pixels[2];

        block[0] = (a1 + a0) >> 2;
        block[1] = (b1 + b0) >> 2;

        pixels += line_size;
        block  += line_size;

        a0  = pixels[0];
        b0  = pixels[1] + 2;
        a0 += b0;
        b0 += pixels[2];

        block[0] = (a1 + a0) >> 2;
        block[1] = (b1 + b0) >> 2;
        pixels  += line_size;
        block   += line_size;
    }
}
// avg variant shares the same body in ffmpeg (PIXOP2 emits avg_pixels2_xy2 with the
// SAME code — the OP is not applied here), so it is identical to put.
static void svq3_avg_pixels2_xy2_8_c(uint8_t *block, const uint8_t *pixels,
                                     ptrdiff_t line_size, int h)
{
    int i, a1, b1;
    int a0 = pixels[0];
    int b0 = pixels[1] + 2;

    a0 += b0;
    b0 += pixels[2];
    pixels += line_size;
    for (i = 0; i < h; i += 2) {
        a1  = pixels[0];
        b1  = pixels[1];
        a1 += b1;
        b1 += pixels[2];

        block[0] = (a1 + a0) >> 2;
        block[1] = (b1 + b0) >> 2;

        pixels += line_size;
        block  += line_size;

        a0  = pixels[0];
        b0  = pixels[1] + 2;
        a0 += b0;
        b0 += pixels[2];

        block[0] = (a1 + a0) >> 2;
        block[1] = (b1 + b0) >> 2;
        pixels  += line_size;
        block   += line_size;
    }
}

// pixels4_xy2
static void svq3_put_pixels4_xy2_8_c(uint8_t *block, const uint8_t *pixels,
                                     ptrdiff_t line_size, int h)
{
    int i;
    const uint32_t a = AV_RN32A(pixels);
    const uint32_t b = AV_RN32A(pixels + 1);
    uint32_t l0 = (a & 0x03030303UL) + (b & 0x03030303UL) + 0x02020202UL;
    uint32_t h0 = ((a & 0xFCFCFCFCUL) >> 2) + ((b & 0xFCFCFCFCUL) >> 2);
    uint32_t l1, h1;

    pixels += line_size;
    for (i = 0; i < h; i += 2) {
        uint32_t a = AV_RN32A(pixels);
        uint32_t b = AV_RN32A(pixels + 1);
        l1 = (a & 0x03030303UL) + (b & 0x03030303UL);
        h1 = ((a & 0xFCFCFCFCUL) >> 2) + ((b & 0xFCFCFCFCUL) >> 2);
        *((uint32_t *) block) = h0 + h1 + (((l0 + l1) >> 2) & 0x0F0F0F0FUL);
        pixels += line_size;
        block  += line_size;
        a  = AV_RN32A(pixels);
        b  = AV_RN32A(pixels + 1);
        l0 = (a & 0x03030303UL) + (b & 0x03030303UL) + 0x02020202UL;
        h0 = ((a & 0xFCFCFCFCUL) >> 2) + ((b & 0xFCFCFCFCUL) >> 2);
        *((uint32_t *) block) = h0 + h1 + (((l0 + l1) >> 2) & 0x0F0F0F0FUL);
        pixels += line_size;
        block  += line_size;
    }
}
static void svq3_avg_pixels4_xy2_8_c(uint8_t *block, const uint8_t *pixels,
                                     ptrdiff_t line_size, int h)
{
    int i;
    const uint32_t a = AV_RN32A(pixels);
    const uint32_t b = AV_RN32A(pixels + 1);
    uint32_t l0 = (a & 0x03030303UL) + (b & 0x03030303UL) + 0x02020202UL;
    uint32_t h0 = ((a & 0xFCFCFCFCUL) >> 2) + ((b & 0xFCFCFCFCUL) >> 2);
    uint32_t l1, h1;

    pixels += line_size;
    for (i = 0; i < h; i += 2) {
        uint32_t a = AV_RN32A(pixels);
        uint32_t b = AV_RN32A(pixels + 1);
        l1 = (a & 0x03030303UL) + (b & 0x03030303UL);
        h1 = ((a & 0xFCFCFCFCUL) >> 2) + ((b & 0xFCFCFCFCUL) >> 2);
        {
            uint32_t *d = (uint32_t *) block;
            *d = svq3_rnd_avg32(*d, h0 + h1 + (((l0 + l1) >> 2) & 0x0F0F0F0FUL));
        }
        pixels += line_size;
        block  += line_size;
        a  = AV_RN32A(pixels);
        b  = AV_RN32A(pixels + 1);
        l0 = (a & 0x03030303UL) + (b & 0x03030303UL) + 0x02020202UL;
        h0 = ((a & 0xFCFCFCFCUL) >> 2) + ((b & 0xFCFCFCFCUL) >> 2);
        {
            uint32_t *d = (uint32_t *) block;
            *d = svq3_rnd_avg32(*d, h0 + h1 + (((l0 + l1) >> 2) & 0x0F0F0F0FUL));
        }
        pixels += line_size;
        block  += line_size;
    }
}

// pixels8_xy2 : two 4-wide columns (j loop) with pointer fixups between them
static void svq3_put_pixels8_xy2_8_c(uint8_t *block, const uint8_t *pixels,
                                     ptrdiff_t line_size, int h)
{
    int j;
    for (j = 0; j < 2; j++) {
        int i;
        const uint32_t a = AV_RN32A(pixels);
        const uint32_t b = AV_RN32A(pixels + 1);
        uint32_t l0 = (a & 0x03030303UL) + (b & 0x03030303UL) + 0x02020202UL;
        uint32_t h0 = ((a & 0xFCFCFCFCUL) >> 2) + ((b & 0xFCFCFCFCUL) >> 2);
        uint32_t l1, h1;

        pixels += line_size;
        for (i = 0; i < h; i += 2) {
            uint32_t a = AV_RN32A(pixels);
            uint32_t b = AV_RN32A(pixels + 1);
            l1 = (a & 0x03030303UL) + (b & 0x03030303UL);
            h1 = ((a & 0xFCFCFCFCUL) >> 2) + ((b & 0xFCFCFCFCUL) >> 2);
            *((uint32_t *) block) = h0 + h1 + (((l0 + l1) >> 2) & 0x0F0F0F0FUL);
            pixels += line_size;
            block  += line_size;
            a  = AV_RN32A(pixels);
            b  = AV_RN32A(pixels + 1);
            l0 = (a & 0x03030303UL) + (b & 0x03030303UL) + 0x02020202UL;
            h0 = ((a & 0xFCFCFCFCUL) >> 2) + ((b & 0xFCFCFCFCUL) >> 2);
            *((uint32_t *) block) = h0 + h1 + (((l0 + l1) >> 2) & 0x0F0F0F0FUL);
            pixels += line_size;
            block  += line_size;
        }
        pixels += 4 - line_size * (h + 1);
        block  += 4 - line_size * h;
    }
}
static void svq3_avg_pixels8_xy2_8_c(uint8_t *block, const uint8_t *pixels,
                                     ptrdiff_t line_size, int h)
{
    int j;
    for (j = 0; j < 2; j++) {
        int i;
        const uint32_t a = AV_RN32A(pixels);
        const uint32_t b = AV_RN32A(pixels + 1);
        uint32_t l0 = (a & 0x03030303UL) + (b & 0x03030303UL) + 0x02020202UL;
        uint32_t h0 = ((a & 0xFCFCFCFCUL) >> 2) + ((b & 0xFCFCFCFCUL) >> 2);
        uint32_t l1, h1;

        pixels += line_size;
        for (i = 0; i < h; i += 2) {
            uint32_t a = AV_RN32A(pixels);
            uint32_t b = AV_RN32A(pixels + 1);
            l1 = (a & 0x03030303UL) + (b & 0x03030303UL);
            h1 = ((a & 0xFCFCFCFCUL) >> 2) + ((b & 0xFCFCFCFCUL) >> 2);
            {
                uint32_t *d = (uint32_t *) block;
                *d = svq3_rnd_avg32(*d, h0 + h1 + (((l0 + l1) >> 2) & 0x0F0F0F0FUL));
            }
            pixels += line_size;
            block  += line_size;
            a  = AV_RN32A(pixels);
            b  = AV_RN32A(pixels + 1);
            l0 = (a & 0x03030303UL) + (b & 0x03030303UL) + 0x02020202UL;
            h0 = ((a & 0xFCFCFCFCUL) >> 2) + ((b & 0xFCFCFCFCUL) >> 2);
            {
                uint32_t *d = (uint32_t *) block;
                *d = svq3_rnd_avg32(*d, h0 + h1 + (((l0 + l1) >> 2) & 0x0F0F0F0FUL));
            }
            pixels += line_size;
            block  += line_size;
        }
        pixels += 4 - line_size * (h + 1);
        block  += 4 - line_size * h;
    }
}

// pixels16_xy2 : CALL_2X_PIXELS(pixels16_xy2, pixels8_xy2, 8)
static void svq3_put_pixels16_xy2_8_c(uint8_t *block, const uint8_t *pixels,
                                      ptrdiff_t line_size, int h)
{
    svq3_put_pixels8_xy2_8_c(block,     pixels,     line_size, h);
    svq3_put_pixels8_xy2_8_c(block + 8, pixels + 8, line_size, h);
}
static void svq3_avg_pixels16_xy2_8_c(uint8_t *block, const uint8_t *pixels,
                                      ptrdiff_t line_size, int h)
{
    svq3_avg_pixels8_xy2_8_c(block,     pixels,     line_size, h);
    svq3_avg_pixels8_xy2_8_c(block + 8, pixels + 8, line_size, h);
}

// x2 / y2 for 16-wide : CALL_2X_PIXELS over the 8-wide x2/y2
static void svq3_put_pixels16_x2_8_c(uint8_t *block, const uint8_t *pixels,
                                     ptrdiff_t line_size, int h)
{
    svq3_put_pixels8_x2_8_c(block,     pixels,     line_size, h);
    svq3_put_pixels8_x2_8_c(block + 8, pixels + 8, line_size, h);
}
static void svq3_avg_pixels16_x2_8_c(uint8_t *block, const uint8_t *pixels,
                                     ptrdiff_t line_size, int h)
{
    svq3_avg_pixels8_x2_8_c(block,     pixels,     line_size, h);
    svq3_avg_pixels8_x2_8_c(block + 8, pixels + 8, line_size, h);
}
static void svq3_put_pixels16_y2_8_c(uint8_t *block, const uint8_t *pixels,
                                     ptrdiff_t line_size, int h)
{
    svq3_put_pixels8_y2_8_c(block,     pixels,     line_size, h);
    svq3_put_pixels8_y2_8_c(block + 8, pixels + 8, line_size, h);
}
static void svq3_avg_pixels16_y2_8_c(uint8_t *block, const uint8_t *pixels,
                                     ptrdiff_t line_size, int h)
{
    svq3_avg_pixels8_y2_8_c(block,     pixels,     line_size, h);
    svq3_avg_pixels8_y2_8_c(block + 8, pixels + 8, line_size, h);
}

// ============================================================================
// Tables + init  --  mirrors ff_hpeldsp_init (rounding tables only).
// svq3_*_pixels_tab[blocksize][dxy] :
//   blocksize 0=16, 1=8, 2=4, 3=2 ; dxy = xhalf + 2*yhalf (0..3).
// row layout per blocksize: { plain, x2, y2, xy2 }.
// ----------------------------------------------------------------------------
typedef void (*hpel_fn)(uint8_t* dst, const uint8_t* src, ptrdiff_t stride, int h);

static hpel_fn svq3_put_pixels_tab[4][4];
static hpel_fn svq3_avg_pixels_tab[4][4];

static void svq3_hpel_init(void)
{
    // put, blocksize 0 -> 16 wide
    svq3_put_pixels_tab[0][0] = svq3_put_pixels16_8_c;
    svq3_put_pixels_tab[0][1] = svq3_put_pixels16_x2_8_c;
    svq3_put_pixels_tab[0][2] = svq3_put_pixels16_y2_8_c;
    svq3_put_pixels_tab[0][3] = svq3_put_pixels16_xy2_8_c;
    // put, blocksize 1 -> 8 wide
    svq3_put_pixels_tab[1][0] = svq3_put_pixels8_8_c;
    svq3_put_pixels_tab[1][1] = svq3_put_pixels8_x2_8_c;
    svq3_put_pixels_tab[1][2] = svq3_put_pixels8_y2_8_c;
    svq3_put_pixels_tab[1][3] = svq3_put_pixels8_xy2_8_c;
    // put, blocksize 2 -> 4 wide
    svq3_put_pixels_tab[2][0] = svq3_put_pixels4_8_c;
    svq3_put_pixels_tab[2][1] = svq3_put_pixels4_x2_8_c;
    svq3_put_pixels_tab[2][2] = svq3_put_pixels4_y2_8_c;
    svq3_put_pixels_tab[2][3] = svq3_put_pixels4_xy2_8_c;
    // put, blocksize 3 -> 2 wide
    svq3_put_pixels_tab[3][0] = svq3_put_pixels2_8_c;
    svq3_put_pixels_tab[3][1] = svq3_put_pixels2_x2_8_c;
    svq3_put_pixels_tab[3][2] = svq3_put_pixels2_y2_8_c;
    svq3_put_pixels_tab[3][3] = svq3_put_pixels2_xy2_8_c;

    // avg, blocksize 0 -> 16 wide
    svq3_avg_pixels_tab[0][0] = svq3_avg_pixels16_8_c;
    svq3_avg_pixels_tab[0][1] = svq3_avg_pixels16_x2_8_c;
    svq3_avg_pixels_tab[0][2] = svq3_avg_pixels16_y2_8_c;
    svq3_avg_pixels_tab[0][3] = svq3_avg_pixels16_xy2_8_c;
    // avg, blocksize 1 -> 8 wide
    svq3_avg_pixels_tab[1][0] = svq3_avg_pixels8_8_c;
    svq3_avg_pixels_tab[1][1] = svq3_avg_pixels8_x2_8_c;
    svq3_avg_pixels_tab[1][2] = svq3_avg_pixels8_y2_8_c;
    svq3_avg_pixels_tab[1][3] = svq3_avg_pixels8_xy2_8_c;
    // avg, blocksize 2 -> 4 wide
    svq3_avg_pixels_tab[2][0] = svq3_avg_pixels4_8_c;
    svq3_avg_pixels_tab[2][1] = svq3_avg_pixels4_x2_8_c;
    svq3_avg_pixels_tab[2][2] = svq3_avg_pixels4_y2_8_c;
    svq3_avg_pixels_tab[2][3] = svq3_avg_pixels4_xy2_8_c;
    // avg, blocksize 3 -> 2 wide
    svq3_avg_pixels_tab[3][0] = svq3_avg_pixels2_8_c;
    svq3_avg_pixels_tab[3][1] = svq3_avg_pixels2_x2_8_c;
    svq3_avg_pixels_tab[3][2] = svq3_avg_pixels2_y2_8_c;
    svq3_avg_pixels_tab[3][3] = svq3_avg_pixels2_xy2_8_c;
}
