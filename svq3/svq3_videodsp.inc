// from ffmpeg release/8.1 libavcodec/videodsp_template.c: ff_emulated_edge_mc (BIT_DEPTH=8)

static void svq3_emulated_edge_mc(uint8_t *buf, const uint8_t *src,
                                  ptrdiff_t buf_linesize,
                                  ptrdiff_t src_linesize,
                                  int block_w, int block_h,
                                  int src_x, int src_y, int w, int h)
{
    int x, y;
    int start_y, start_x, end_y, end_x;

    if (!w || !h)
        return;

    // av_assert2(block_w * sizeof(pixel) <= FFABS(buf_linesize)); // dropped

    if (src_y >= h) {
        src -= src_y * src_linesize;
        src += (h - 1) * src_linesize;
        src_y = h - 1;
    } else if (src_y <= -block_h) {
        src -= src_y * src_linesize;
        src += (1 - block_h) * src_linesize;
        src_y = 1 - block_h;
    }
    if (src_x >= w) {
        // The subtracted expression has an unsigned type and must thus not be negative
        src  -= (1 + src_x - w) * sizeof(uint8_t);
        src_x = w - 1;
    } else if (src_x <= -block_w) {
        src  += (1 - block_w - src_x) * sizeof(uint8_t);
        src_x = 1 - block_w;
    }

    start_y = FFMAX(0, -src_y);
    start_x = FFMAX(0, -src_x);
    end_y = FFMIN(block_h, h-src_y);
    end_x = FFMIN(block_w, w-src_x);
    // av_assert2(start_y < end_y && block_h); // dropped
    // av_assert2(start_x < end_x && block_w); // dropped

    w    = end_x - start_x;
    src += start_y * src_linesize + start_x * (ptrdiff_t)sizeof(uint8_t);
    buf += start_x * sizeof(uint8_t);

    // top
    for (y = 0; y < start_y; y++) {
        memcpy(buf, src, w * sizeof(uint8_t));
        buf += buf_linesize;
    }

    // copy existing part
    for (; y < end_y; y++) {
        memcpy(buf, src, w * sizeof(uint8_t));
        src += src_linesize;
        buf += buf_linesize;
    }

    // bottom
    src -= src_linesize;
    for (; y < block_h; y++) {
        memcpy(buf, src, w * sizeof(uint8_t));
        buf += buf_linesize;
    }

    buf -= block_h * buf_linesize + start_x * (ptrdiff_t)sizeof(uint8_t);
    while (block_h--) {
        uint8_t *bufp = (uint8_t *) buf;

        // left
        for(x = 0; x < start_x; x++) {
            bufp[x] = bufp[start_x];
        }

        // right
        for (x = end_x; x < block_w; x++) {
            bufp[x] = bufp[end_x - 1];
        }
        buf += buf_linesize;
    }
}
