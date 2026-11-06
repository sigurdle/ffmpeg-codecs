/*
	QuickTime Planar RGB (eightbps::): a port of FFmpeg's libavcodec/8bps.c (release/8.1,
	LGPL). Included by codec_pack.cpp after RvCtx.

	Every frame is whole: PackBits-style rows, plane after plane, each plane's row lengths
	first. Depth 8 has one plane of palette indices, 24 R,G,B and 32 R,G,B,A. FFmpeg
	decodes into separate planes, here the planes interleave into the output picture:
	index, R,G,B or A,R,G,B (RvCtx.format).

	FFmpeg ends a row at its padded line size, here at the width. The two differ only for
	a row longer than the picture.
*/

#ifndef RV_EIGHTBPS_DECODE_H
#define RV_EIGHTBPS_DECODE_H

namespace eightbps {

static int plane_count(int depth) { return depth == 8 ? 1 : depth == 24 ? 3 : depth == 32 ? 4 : 0; }

static int decode(RvCtx* s, const uint8_t* buf, int buf_size) {
	const int width = s->width;
	const int height = s->height;
	const int planes = plane_count(s->depth);
	const int step = s->bpp;
	const int alphaFirst = s->depth == 32;

	if (buf_size < planes * height * (2 + 2 * ((width + 128) / 129))) return AVERROR_INVALIDDATA;

	const uint8_t* const ep = buf + buf_size;
	const uint8_t* dp = buf + planes * (height << 1);

	for (int p = 0; p < planes; p++) {
		// Stream order R,G,B,A; the output has A first at depth 32.
		const int offset = alphaFirst ? (p + 1) & 3 : p;
		const uint8_t* lp = buf + p * (height << 1);

		for (int row = 0; row < height; row++) {
			uint8_t* pix = s->frame + (size_t)row * s->stride + offset;
			int room = width;
			if (ep - lp < row * 2 + 2) return AVERROR_INVALIDDATA;
			unsigned dlen = (lp[row * 2] << 8) | lp[row * 2 + 1];
			while (dlen > 0) {
				if (ep - dp <= 1) return AVERROR_INVALIDDATA;
				int count = *dp++;
				if (count <= 127) {
					count++;
					dlen -= count + 1;
					if (room < count) break;
					if (ep - dp < count) return AVERROR_INVALIDDATA;
					for (int i = 0; i < count; i++, pix += step) *pix = dp[i];
					room -= count;
					dp += count;
				} else {
					count = 257 - count;
					if (room < count) break;
					for (int i = 0; i < count; i++, pix += step) *pix = dp[0];
					room -= count;
					dp++;
					dlen -= 2;
				}
			}
		}
	}
	return 0;
}

} // namespace eightbps

#endif
