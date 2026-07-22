// Chroma-DC dequant IDCT, from ffmpeg release/8.1 libavcodec/h264idct_template.c (ff_h264_chroma_dc_dequant_idct, BIT_DEPTH=8)

// BIT_DEPTH=8 template instantiation:
//   dctcoef -> int16_t  (block is already int16_t, the pointer cast is a no-op)
//   SUINT   -> unsigned (production, non-CHECKED)
static void svq3_chroma_dc_dequant_idct(int16_t *_block, int qmul)
{
    const int stride = 16 * 2;
    const int xStride = 16;
    unsigned a, b, c, d, e;
    int16_t *block = _block;

    a = block[stride * 0 + xStride * 0];
    b = block[stride * 0 + xStride * 1];
    c = block[stride * 1 + xStride * 0];
    d = block[stride * 1 + xStride * 1];

    e = a - b;
    a = a + b;
    b = c - d;
    c = c + d;

    block[stride * 0 + xStride * 0] = (int)((a + c) * qmul) >> 7;
    block[stride * 0 + xStride * 1] = (int)((e + b) * qmul) >> 7;
    block[stride * 1 + xStride * 0] = (int)((a - c) * qmul) >> 7;
    block[stride * 1 + xStride * 1] = (int)((e - b) * qmul) >> 7;
}
