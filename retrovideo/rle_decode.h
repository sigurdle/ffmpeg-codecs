/*
	Microsoft RLE (msrle::) and QuickTime Animation (qtrle::): ports of FFmpeg's
	libavcodec/msrle.c, msrledec.c and qtrle.c (release/8.1, LGPL). Included by
	codec_pack.cpp after RvCtx.

	Both paint over the persistent picture: skipped pixels keep the previous frame.

	The stored picture differs from FFmpeg's in two ways, both at the output:
	- Microsoft RLE 24 and 32 bit store R,G,B (the readers' RGB24), the stream has B,G,R[,X].
	- Paletted QuickTime Animation writes whole groups of 4, 8 or 16 pixels, past the
	  right edge when the width is not a multiple. FFmpeg's padded lines swallow that,
	  here it goes into RvCtx.rlePic, whose rows have room for a group, and each
	  decode copies the picture tight into `frame`.
*/

#ifndef RV_RLE_DECODE_H
#define RV_RLE_DECODE_H

namespace rle {

// bytestream2's checked reads: past the end reads 0 and does not advance.
struct ByteReader {
	const uint8_t* p;
	const uint8_t* end;
	int left() const { return (int)(end - p); }
	int byte() { return p < end ? *p++ : 0; }
	int be16() { int a = byte(); return (a << 8) | byte(); }
	void skip(int n) { p += n < left() ? n : left(); }
	void bytes(uint8_t* out, int n) { int c = n < left() ? n : left(); memcpy(out, p, c); p += c; }
};

} // namespace rle

namespace msrle {

static inline void put_pixel(uint8_t* out, const uint8_t* px, int depth) {
	switch (depth) {
	case 8:  out[0] = px[0]; break;
	case 16: out[0] = px[0]; out[1] = px[1]; break;
	default: out[0] = px[2]; out[1] = px[1]; out[2] = px[0]; break;   // 24, 32: B,G,R[,X] -> R,G,B
	}
}

static int decode_pal4(RvCtx* s, rle::ByteReader* gb) {
	int rle_code, stream_byte;
	int pixel_ptr = 0;
	int line = s->height - 1;
	const int width = s->width, stride = s->stride;
	uint8_t* pic = s->frame;

	while (line >= 0 && pixel_ptr <= width) {
		if (gb->left() <= 0) return AVERROR_INVALIDDATA;
		rle_code = stream_byte = gb->byte();
		if (rle_code == 0) {
			stream_byte = gb->byte();
			if (stream_byte == 0) {
				line--;
				pixel_ptr = 0;
			} else if (stream_byte == 1) {
				return 0;
			} else if (stream_byte == 2) {
				pixel_ptr += gb->byte();
				line -= gb->byte();
			} else {
				const int odd_pixel = stream_byte & 1;
				rle_code = (stream_byte + 1) / 2;
				const int extra_byte = rle_code & 1;
				if (pixel_ptr + 2 * rle_code - odd_pixel > width || gb->left() < rle_code)
					return AVERROR_INVALIDDATA;
				for (int i = 0; i < rle_code; i++) {
					if (pixel_ptr >= width) break;
					stream_byte = gb->byte();
					pic[line * stride + pixel_ptr] = stream_byte >> 4;
					pixel_ptr++;
					if (i + 1 == rle_code && odd_pixel) break;
					if (pixel_ptr >= width) break;
					pic[line * stride + pixel_ptr] = stream_byte & 0x0F;
					pixel_ptr++;
				}
				if (extra_byte) gb->skip(1);
			}
		} else {
			if (pixel_ptr + rle_code > width + 1) return AVERROR_INVALIDDATA;
			stream_byte = gb->byte();
			for (int i = 0; i < rle_code; i++) {
				if (pixel_ptr >= width) break;
				pic[line * stride + pixel_ptr] = (i & 1) == 0 ? stream_byte >> 4 : stream_byte & 0x0F;
				pixel_ptr++;
			}
		}
	}
	return gb->left() ? AVERROR_INVALIDDATA : 0;
}

static int decode_8_16_24_32(RvCtx* s, int depth, rle::ByteReader* gb) {
	const int stream_bpp = depth >> 3, out_bpp = s->bpp, stride = s->stride;
	const int width = s->width;
	int line = s->height - 1, pos = 0;
	uint8_t* pic = s->frame;
	uint8_t* output = pic + line * stride;
	uint8_t* output_end = output + stride;
	uint8_t px[4];

	while (gb->left() > 0) {
		int p1 = gb->byte();
		if (p1 == 0) {
			int p2 = gb->byte();
			if (p2 == 0) {
				if (--line < 0)
					return gb->be16() == 1 ? 0 : AVERROR_INVALIDDATA;
				output = pic + line * stride;
				output_end = output + stride;
				pos = 0;
				continue;
			} else if (p2 == 1) {
				return 0;
			} else if (p2 == 2) {
				p1 = gb->byte();
				p2 = gb->byte();
				line -= p2;
				pos += p1;
				if (line < 0 || pos >= width) return AVERROR_INVALIDDATA;
				output = pic + line * stride + pos * out_bpp;
				output_end = pic + line * stride + stride;
				continue;
			}
			// Copy data
			if (output + p2 * out_bpp > output_end) {
				gb->skip(2 * stream_bpp);
				continue;
			} else if (gb->left() < p2 * stream_bpp) {
				return AVERROR_INVALIDDATA;
			}
			for (int i = 0; i < p2; i++) {
				gb->bytes(px, stream_bpp);
				put_pixel(output, px, depth);
				output += out_bpp;
			}
			// RLE8 copy is actually padded - and runs are not!
			if (depth == 8 && (p2 & 1)) gb->skip(1);
			pos += p2;
		} else {
			if (output + p1 * out_bpp > output_end) continue;
			for (int i = 0; i < stream_bpp; i++) px[i] = (uint8_t)gb->byte();
			for (int i = 0; i < p1; i++) {
				put_pixel(output, px, depth);
				output += out_bpp;
			}
			pos += p1;
		}
	}
	return 0;   // no end-of-picture code
}

// A chunk the size of an uncompressed DIB is one (FFmpeg's "assume uncompressed").
static void copy_uncompressed(RvCtx* s, const uint8_t* buf, int istride) {
	const int depth = s->depth, height = s->height, width = s->width;
	const int stream_bpp = depth >> 3;
	for (int y = 0; y < height; y++) {
		const uint8_t* src = buf + (height - 1 - y) * istride;
		uint8_t* dst = s->frame + y * s->stride;
		if (depth == 4) {
			int j;
			for (j = 0; j < width - 1; j += 2) {
				dst[j] = src[j >> 1] >> 4;
				dst[j + 1] = src[j >> 1] & 0xF;
			}
			if (width & 1) dst[j] = src[j >> 1] >> 4;
		} else {
			for (int x = 0; x < width; x++)
				put_pixel(dst + x * s->bpp, src + x * stream_bpp, depth);
		}
	}
}

static int decode(RvCtx* s, const uint8_t* buf, int len) {
	const int depth = s->depth;
	if (depth != 4 && depth != 8 && depth != 16 && depth != 24 && depth != 32)
		return AVERROR_PATCHWELCOME;
	if (len < 2) return AVERROR_INVALIDDATA;   // minimally an end of picture code

	const int istride = ((s->width * depth + 31) & ~31) / 8;
	if (s->height * istride == len) {
		copy_uncompressed(s, buf, istride);
		return 0;
	}
	rle::ByteReader gb = { buf, buf + len };
	// msrle.c ignores this result: a damaged frame shows what was painted.
	if (depth == 4) decode_pal4(s, &gb);
	else decode_8_16_24_32(s, depth, &gb);
	return 0;
}

} // namespace msrle

namespace qtrle {

// Room for one whole group past the right edge of a paletted row.
enum { GROUP_PIXELS = 16 };

#define CHECK_PIXEL_PTR(n) \
	if ((pixel_ptr + (n) > pixel_limit) || (pixel_ptr + (n) < 0)) return

static void decode_1bpp(rle::ByteReader* g, uint8_t* rgb, int row_inc, int pixel_limit, int row_ptr, int lines_to_change) {
	int rle_code, pixel_ptr, skip;
	// skip & 0x80 means 'next line' in the loop but 'first line' at the start, so
	// start one line above and count one more (trac.ffmpeg.org/ticket/226).
	row_ptr -= row_inc;
	pixel_ptr = row_ptr;
	lines_to_change++;
	while (lines_to_change) {
		skip = g->byte();
		rle_code = (int8_t)g->byte();
		if (rle_code == 0) break;
		if (skip & 0x80) {
			lines_to_change--;
			row_ptr += row_inc;
			pixel_ptr = row_ptr + 2 * 8 * (skip & 0x7f);
		} else {
			pixel_ptr += 2 * 8 * skip;
		}
		CHECK_PIXEL_PTR(0);
		if (rle_code == -1) continue;

		if (rle_code < 0) {
			rle_code = -rle_code;
			const int pi0 = g->byte(), pi1 = g->byte();
			CHECK_PIXEL_PTR(rle_code * 2 * 8);
			while (rle_code--) {
				for (int b = 7; b >= 0; b--) rgb[pixel_ptr++] = (pi0 >> b) & 1;
				for (int b = 7; b >= 0; b--) rgb[pixel_ptr++] = (pi1 >> b) & 1;
			}
		} else {
			rle_code *= 2;
			CHECK_PIXEL_PTR(rle_code * 8);
			while (rle_code--) {
				const int x = g->byte();
				for (int b = 7; b >= 0; b--) rgb[pixel_ptr++] = (x >> b) & 1;
			}
		}
	}
}

static void decode_2n4bpp(rle::ByteReader* g, uint8_t* rgb, int row_inc, int pixel_limit, int row_ptr, int lines_to_change, int bpp) {
	int rle_code, pixel_ptr;
	uint8_t pi[16];
	const int num_pixels = (bpp == 4) ? 8 : 16;

	while (lines_to_change--) {
		pixel_ptr = row_ptr + (num_pixels * (g->byte() - 1));
		CHECK_PIXEL_PTR(0);

		while ((rle_code = (int8_t)g->byte()) != -1) {
			if (g->left() < 1) return;
			if (rle_code == 0) {
				pixel_ptr += (num_pixels * (g->byte() - 1));
				CHECK_PIXEL_PTR(0);
			} else if (rle_code < 0) {
				rle_code = -rle_code;
				for (int i = num_pixels - 1; i >= 0; i--) {
					const int peek = g->left() > 0 ? *g->p : 0;
					pi[num_pixels - 1 - i] = (peek >> ((i * bpp) & 0x07)) & ((1 << bpp) - 1);
					g->skip((i & ((num_pixels >> 2) - 1)) == 0);
				}
				CHECK_PIXEL_PTR(rle_code * num_pixels);
				while (rle_code--) {
					memcpy(&rgb[pixel_ptr], pi, num_pixels);
					pixel_ptr += num_pixels;
				}
			} else {
				rle_code *= 4;
				CHECK_PIXEL_PTR(rle_code * (num_pixels >> 2));
				while (rle_code--) {
					const int x = g->byte();
					if (bpp == 4) {
						rgb[pixel_ptr++] = (x >> 4) & 0x0f;
						rgb[pixel_ptr++] =  x       & 0x0f;
					} else {
						rgb[pixel_ptr++] = (x >> 6) & 0x03;
						rgb[pixel_ptr++] = (x >> 4) & 0x03;
						rgb[pixel_ptr++] = (x >> 2) & 0x03;
						rgb[pixel_ptr++] =  x       & 0x03;
					}
				}
			}
		}
		row_ptr += row_inc;
	}
}

static void decode_8bpp(rle::ByteReader* g, uint8_t* rgb, int row_inc, int pixel_limit, int row_ptr, int lines_to_change) {
	int rle_code, pixel_ptr;

	while (lines_to_change--) {
		pixel_ptr = row_ptr + (4 * (g->byte() - 1));
		CHECK_PIXEL_PTR(0);

		while ((rle_code = (int8_t)g->byte()) != -1) {
			if (g->left() < 1) return;
			if (rle_code == 0) {
				pixel_ptr += (4 * (g->byte() - 1));
				CHECK_PIXEL_PTR(0);
			} else if (rle_code < 0) {
				rle_code = -rle_code;
				const int pi1 = g->byte(), pi2 = g->byte(), pi3 = g->byte(), pi4 = g->byte();
				CHECK_PIXEL_PTR(rle_code * 4);
				while (rle_code--) {
					rgb[pixel_ptr++] = pi1;
					rgb[pixel_ptr++] = pi2;
					rgb[pixel_ptr++] = pi3;
					rgb[pixel_ptr++] = pi4;
				}
			} else {
				rle_code *= 4;
				CHECK_PIXEL_PTR(rle_code);
				g->bytes(&rgb[pixel_ptr], rle_code);
				pixel_ptr += rle_code;
			}
		}
		row_ptr += row_inc;
	}
}

// One pixel in stream order. 16 bit is big-endian in the stream and stored
// little-endian (RGB555, as the readers take it); 24 and 32 bit store as they come.
static inline void read_pixel(rle::ByteReader* g, uint8_t* px, int bpp) {
	if (bpp == 2) {
		const int v = g->be16();
		px[0] = v & 0xFF;
		px[1] = v >> 8;
	} else {
		for (int i = 0; i < bpp; i++) px[i] = (uint8_t)g->byte();
	}
}

// qtrle_decode_16bpp, _24bpp and _32bpp share this shape; bpp is the pixel size in bytes.
static void decode_truecolor(rle::ByteReader* g, uint8_t* rgb, int row_inc, int pixel_limit, int row_ptr, int lines_to_change, int bpp) {
	int rle_code, pixel_ptr;
	uint8_t px[4];

	while (lines_to_change--) {
		pixel_ptr = row_ptr + (g->byte() - 1) * bpp;
		CHECK_PIXEL_PTR(0);

		while ((rle_code = (int8_t)g->byte()) != -1) {
			if (g->left() < 1) return;
			if (rle_code == 0) {
				pixel_ptr += (g->byte() - 1) * bpp;
				CHECK_PIXEL_PTR(0);
			} else if (rle_code < 0) {
				rle_code = -rle_code;
				read_pixel(g, px, bpp);
				CHECK_PIXEL_PTR(rle_code * bpp);
				while (rle_code--) {
					memcpy(rgb + pixel_ptr, px, bpp);
					pixel_ptr += bpp;
				}
			} else {
				CHECK_PIXEL_PTR(rle_code * bpp);
				while (rle_code--) {
					read_pixel(g, rgb + pixel_ptr, bpp);
					pixel_ptr += bpp;
				}
			}
		}
		row_ptr += row_inc;
	}
}

#undef CHECK_PIXEL_PTR

// FFmpeg's discard_damaged_percentage default.
enum { DISCARD_DAMAGED_PERCENTAGE = 95 };

static int decode(RvCtx* s, const uint8_t* buf, int len) {
	// A frame too short to say anything repeats the previous one.
	if (len < 8) return 0;

	rle::ByteReader g = { buf, buf + len };
	int size = g.be16() << 16;
	size = (size | g.be16()) & 0x3FFFFFFF;
	if ((int64_t)size - len > (int64_t)size * DISCARD_DAMAGED_PERCENTAGE / 100)
		return AVERROR_INVALIDDATA;

	int start_line = 0, height = s->height;
	const int header = g.be16();
	if (header & 0x0008) {
		if (len < 14) return 0;
		start_line = g.be16();
		g.skip(2);
		height = g.be16();
		g.skip(2);
		if (height > s->height - start_line) return 0;
	}

	const bool paletted = s->depth <= 8;
	uint8_t* pic = paletted ? s->rlePic : s->frame;
	const int row_inc = paletted ? s->rleStride : s->stride;
	const int limit = row_inc * s->height;
	const int row_ptr = row_inc * start_line;

	switch (s->depth) {
	case 1:  decode_1bpp(&g, pic, row_inc, limit, row_ptr, height); break;
	case 2:  decode_2n4bpp(&g, pic, row_inc, limit, row_ptr, height, 2); break;
	case 4:  decode_2n4bpp(&g, pic, row_inc, limit, row_ptr, height, 4); break;
	case 8:  decode_8bpp(&g, pic, row_inc, limit, row_ptr, height); break;
	case 16: decode_truecolor(&g, pic, row_inc, limit, row_ptr, height, 2); break;
	case 24: decode_truecolor(&g, pic, row_inc, limit, row_ptr, height, 3); break;
	case 32: decode_truecolor(&g, pic, row_inc, limit, row_ptr, height, 4); break;
	default: return AVERROR_PATCHWELCOME;
	}

	if (paletted)
		for (int y = 0; y < s->height; y++)
			memcpy(s->frame + y * s->stride, s->rlePic + y * s->rleStride, s->width);
	return 0;
}

} // namespace qtrle

#endif
