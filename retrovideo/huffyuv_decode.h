/*
	HuffYUV and FFV Huffman (hfyu::): a port of FFmpeg's libavcodec/huffyuvdec.c, huffyuv.c
	and the 8 bit kernels of lossless_videodsp.c (release/8.1, LGPL). Included by
	codec_pack.cpp after RvCtx.

	Every frame is whole. Versions 0 and 1 use the classic tables, 2 and 3 (FFV Huffman's
	planar streams) read theirs from the extradata, and with the context flag each frame
	carries its own. Output: rv_set_extradata answers what comes out, and each decode
	fills RvCtx.frame with A,R,G,B or with Y then interleaved Cb,Cr.

	Left out: more than 8 bits, 4:4:4, 4:1:1, 4:4:0 and YCbCr with alpha (FFV Huffman only,
	no output path here), all refused at the extradata.

	Differs from FFmpeg only in speed: no joint two-symbol tables, one lookup per symbol.
*/

#ifndef RV_HUFFYUV_DECODE_H
#define RV_HUFFYUV_DECODE_H

namespace hfyu {

enum { LEFT = 0, PLANE = 1, MEDIAN = 2 };
enum { B = 0, G = 1, R = 2, A = 3 };   // AV_PIX_FMT_RGB32 on a little-endian host
enum { VLC_BITS = 12 };

// What rv_set_extradata answers: 0, 1, 3 are the Bitmap's chromaFormat.
enum { OUT_YCBCR420 = 0, OUT_YCBCR422 = 1, OUT_YCBCR410 = 3, OUT_RGB = 4, OUT_RGBA = 5 };

// A prefix code: codes up to VLC_BITS long by lookup, the few longer ones by search.
struct Vlc {
	uint8_t  sym[1 << VLC_BITS];
	uint8_t  len[1 << VLC_BITS];
	int      longCount;
	uint32_t longCode[256];   // left-aligned
	uint8_t  longLen[256];
	uint8_t  longSym[256];
};

static const uint8_t classic_shift_luma[] = {
    34, 36, 35, 69, 135, 232,   9, 16, 10, 24,  11,  23,  12,  16, 13, 10,
    14,  8, 15,  8,  16,   8,  17, 20, 16, 10, 207, 206, 205, 236, 11,  8,
    10, 21,  9, 23,   8,   8, 199, 70, 69, 68,
};

static const uint8_t classic_shift_chroma[] = {
    66, 36,  37,  38, 39, 40,  41,  75,  76,  77, 110, 239, 144, 81, 82,  83,
    84, 85, 118, 183, 56, 57,  88,  89,  56,  89, 154,  57,  58, 57, 26, 141,
    57, 56,  58,  57, 58, 57, 184, 119, 214, 245, 116,  83,  82, 49, 80,  79,
    78, 77,  44,  75, 41, 40,  39,  38,  37,  36,  34,
};

static const unsigned char classic_add_luma[256] = {
     3,  9,  5, 12, 10, 35, 32, 29, 27, 50, 48, 45, 44, 41, 39, 37,
    73, 70, 68, 65, 64, 61, 58, 56, 53, 50, 49, 46, 44, 41, 38, 36,
    68, 65, 63, 61, 58, 55, 53, 51, 48, 46, 45, 43, 41, 39, 38, 36,
    35, 33, 32, 30, 29, 27, 26, 25, 48, 47, 46, 44, 43, 41, 40, 39,
    37, 36, 35, 34, 32, 31, 30, 28, 27, 26, 24, 23, 22, 20, 19, 37,
    35, 34, 33, 31, 30, 29, 27, 26, 24, 23, 21, 20, 18, 17, 15, 29,
    27, 26, 24, 22, 21, 19, 17, 16, 14, 26, 25, 23, 21, 19, 18, 16,
    15, 27, 25, 23, 21, 19, 17, 16, 14, 26, 25, 23, 21, 18, 17, 14,
    12, 17, 19, 13,  4,  9,  2, 11,  1,  7,  8,  0, 16,  3, 14,  6,
    12, 10,  5, 15, 18, 11, 10, 13, 15, 16, 19, 20, 22, 24, 27, 15,
    18, 20, 22, 24, 26, 14, 17, 20, 22, 24, 27, 15, 18, 20, 23, 25,
    28, 16, 19, 22, 25, 28, 32, 36, 21, 25, 29, 33, 38, 42, 45, 49,
    28, 31, 34, 37, 40, 42, 44, 47, 49, 50, 52, 54, 56, 57, 59, 60,
    62, 64, 66, 67, 69, 35, 37, 39, 40, 42, 43, 45, 47, 48, 51, 52,
    54, 55, 57, 59, 60, 62, 63, 66, 67, 69, 71, 72, 38, 40, 42, 43,
    46, 47, 49, 51, 26, 28, 30, 31, 33, 34, 18, 19, 11, 13,  7,  8,
};

static const unsigned char classic_add_chroma[256] = {
     3,    1,   2,   2,   2,   2,   3,   3,   7,   5,   7,   5,   8,   6,  11,   9,
     7,   13,  11,  10,   9,   8,   7,   5,   9,   7,   6,   4,   7,   5,   8,   7,
     11,   8,  13,  11,  19,  15,  22,  23,  20,  33,  32,  28,  27,  29,  51,  77,
     43,  45,  76,  81,  46,  82,  75,  55,  56, 144,  58,  80,  60,  74, 147,  63,
    143,  65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,  78,  79,
     80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,  91,  27,  30,  21,  22,
     17,  14,   5,   6, 100,  54,  47,  50,  51,  53, 106, 107, 108, 109, 110, 111,
    112, 113, 114, 115,   4, 117, 118,  92,  94, 121, 122,   3, 124, 103,   2,   1,
      0, 129, 130, 131, 120, 119, 126, 125, 136, 137, 138, 139, 140, 141, 142, 134,
    135, 132, 133, 104,  64, 101,  62,  57, 102,  95,  93,  59,  61,  28,  97,  96,
     52,  49,  48,  29,  32,  25,  24,  46,  23,  98,  45,  44,  43,  20,  42,  41,
     19,  18,  99,  40,  15,  39,  38,  16,  13,  12,  11,  37,  10,   9,   8,  36,
      7, 128, 127, 105, 123, 116,  35,  34,  33, 145,  31,  79,  42, 146,  78,  26,
     83,  48,  49,  50,  44,  47,  26,  31,  30,  18,  17,  19,  21,  24,  25,  13,
     14,  16,  17,  18,  20,  21,  12,  14,  15,   9,  10,   6,   9,   6,   5,   8,
      6,  12,   8,  10,   7,   9,   6,   4,   6,   2,   2,   3,   3,   3,   3,   2,
};

} // namespace hfyu

struct HyuvCtx {
	int version, predictor, decorrelate, interlaced, bitstream_bpp;
	int alpha, chroma, yuv, chroma_h_shift, chroma_v_shift, context;
	int out;
	uint8_t  len[4][256];
	uint32_t bits[4][256];
	hfyu::Vlc vlc[4];
	uint8_t* temp[3];
	uint8_t* pix[4];
	int      ls[4];
	uint8_t* stream;
	size_t   streamCap;
	size_t   pos, sizeBits;   // the bit reader, over `stream`
};

namespace hfyu {

static int read_len_table(uint8_t* dst, rle::ByteReader* gb) {
	for (int i = 0; i < 256;) {
		if (gb->left() <= 0) return AVERROR_INVALIDDATA;
		int code = gb->byte();
		int repeat = code >> 5;
		const int val = code & 0x1F;
		if (repeat == 0) {
			if (gb->left() <= 0) return AVERROR_INVALIDDATA;
			repeat = gb->byte();
		}
		if (i + repeat > 256) return AVERROR_INVALIDDATA;
		while (repeat--) dst[i++] = val;
	}
	return 0;
}

static int generate_bits_table(uint32_t* dst, const uint8_t* len_table) {
	int lens[33] = { 0 };
	uint32_t codes[33];
	for (int i = 0; i < 256; i++) lens[len_table[i]]++;
	codes[32] = 0;
	for (int i = 32; i > 0; i--) {
		if ((lens[i] + codes[i]) & 1) return AVERROR_INVALIDDATA;
		codes[i - 1] = (lens[i] + codes[i]) >> 1;
	}
	for (int i = 0; i < 256; i++) {
		if (len_table[i]) dst[i] = codes[len_table[i]]++;
	}
	return 0;
}

static void build_vlc(Vlc* v, const uint8_t* len, const uint32_t* bits) {
	memset(v->len, 0, sizeof(v->len));
	v->longCount = 0;
	for (int i = 0; i < 256; i++) {
		const int l = len[i];
		if (!l) continue;
		const uint32_t code = bits[i] & (uint32_t)(((uint64_t)1 << l) - 1);
		if (l <= VLC_BITS) {
			const int first = code << (VLC_BITS - l);
			const int n = 1 << (VLC_BITS - l);
			memset(v->sym + first, i, n);
			memset(v->len + first, l, n);
		} else {
			v->longCode[v->longCount] = code << (32 - l);
			v->longLen[v->longCount] = l;
			v->longSym[v->longCount++] = i;
		}
	}
}

static int read_huffman_tables(HyuvCtx* c, const uint8_t* src, int length) {
	rle::ByteReader gb = { src, src + length };
	const int count = c->version > 2 ? 1 + c->alpha + 2 * c->chroma : 3;
	for (int i = 0; i < count; i++) {
		int ret;
		if ((ret = read_len_table(c->len[i], &gb)) < 0) return ret;
		if ((ret = generate_bits_table(c->bits[i], c->len[i])) < 0) return ret;
		build_vlc(&c->vlc[i], c->len[i], c->bits[i]);
	}
	return (int)(gb.p - src);
}

static void read_old_huffman_tables(HyuvCtx* c) {
	rle::ByteReader gb = { classic_shift_luma, classic_shift_luma + sizeof(classic_shift_luma) };
	read_len_table(c->len[0], &gb);
	gb = { classic_shift_chroma, classic_shift_chroma + sizeof(classic_shift_chroma) };
	read_len_table(c->len[1], &gb);

	for (int i = 0; i < 256; i++) c->bits[0][i] = classic_add_luma[i];
	for (int i = 0; i < 256; i++) c->bits[1][i] = classic_add_chroma[i];

	if (c->bitstream_bpp >= 24) {
		memcpy(c->bits[1], c->bits[0], sizeof(c->bits[0]));
		memcpy(c->len[1], c->len[0], sizeof(c->len[0]));
	}
	memcpy(c->bits[2], c->bits[1], sizeof(c->bits[0]));
	memcpy(c->len[2], c->len[1], sizeof(c->len[0]));

	for (int i = 0; i < 3; i++) build_vlc(&c->vlc[i], c->len[i], c->bits[i]);
}

// -- bit reader, MSB first --

static inline uint32_t show32(const HyuvCtx* c) {
	size_t i = c->pos >> 3;
	if (i > c->sizeBits >> 3) i = c->sizeBits >> 3;   // `stream` has 8 zero bytes past the end
	uint64_t v;
	memcpy(&v, c->stream + i, 8);
	return (uint32_t)((__builtin_bswap64(v) << (c->pos & 7)) >> 32);
}

static inline int get_bits8(HyuvCtx* c) {
	const int v = show32(c) >> 24;
	c->pos += 8;
	return v;
}

static inline int bits_left(const HyuvCtx* c) { return (int)((int64_t)c->sizeBits - (int64_t)c->pos); }

static inline int get_vlc(HyuvCtx* c, int table) {
	const Vlc* v = &c->vlc[table];
	const uint32_t w = show32(c);
	const int i = w >> (32 - VLC_BITS);
	const int l = v->len[i];
	if (l) {
		c->pos += l;
		return v->sym[i];
	}
	for (int k = 0; k < v->longCount; k++) {
		if (((w ^ v->longCode[k]) >> (32 - v->longLen[k])) == 0) {
			c->pos += v->longLen[k];
			return v->longSym[k];
		}
	}
	c->pos++;   // not a code: corrupt
	return 0;
}

// -- the bitstream of one row --

static void decode_422_bitstream(HyuvCtx* c, int count) {
	count /= 2;
	uint8_t* const t0 = c->temp[0];
	uint8_t* const t1 = c->temp[1];
	uint8_t* const t2 = c->temp[2];
	for (int i = 0; i < count; i++) {
		if (bits_left(c) <= 0) {
			t0[2 * i] = t1[i] = t0[2 * i + 1] = t2[i] = 0;
			continue;
		}
		t0[2 * i] = get_vlc(c, 0);
		t1[i] = get_vlc(c, 1);
		t0[2 * i + 1] = get_vlc(c, 0);
		t2[i] = get_vlc(c, 2);
	}
}

static void decode_gray_bitstream(HyuvCtx* c, int count) {
	count /= 2;
	uint8_t* const t0 = c->temp[0];
	for (int i = 0; i < count; i++) {
		if (bits_left(c) <= 0) {
			t0[2 * i] = t0[2 * i + 1] = 0;
			continue;
		}
		t0[2 * i] = get_vlc(c, 0);
		t0[2 * i + 1] = get_vlc(c, 0);
	}
}

static void decode_plane_bitstream(HyuvCtx* c, int width, int plane) {
	uint8_t* const t0 = c->temp[0];
	const int count = width / 2;
	for (int i = 0; i < count && bits_left(c) > 0; i++) {
		t0[2 * i] = get_vlc(c, plane);
		t0[2 * i + 1] = get_vlc(c, plane);
	}
	if ((width & 1) && bits_left(c) > 0) t0[width - 1] = get_vlc(c, plane);
}

static void decode_bgr_bitstream(HyuvCtx* c, int count) {
	const int alpha = c->bitstream_bpp != 24;
	const int decorrelate = c->decorrelate;
	uint8_t* const t0 = c->temp[0];
	for (int i = 0; i < count && bits_left(c) > 0; i++) {
		uint8_t* const px = t0 + 4 * i;
		if (decorrelate) {
			px[G] = get_vlc(c, 1);
			px[B] = get_vlc(c, 0) + px[G];
			px[R] = get_vlc(c, 2) + px[G];
		} else {
			px[B] = get_vlc(c, 0);
			px[G] = get_vlc(c, 1);
			px[R] = get_vlc(c, 2);
		}
		px[A] = alpha ? get_vlc(c, 2) : 0;
	}
}

// -- lossless_videodsp, huffyuvdsp --

static inline int add_left_pred(uint8_t* dst, const uint8_t* src, int w, int acc) {
	for (int i = 0; i < w; i++) {
		acc += src[i];
		dst[i] = acc;
	}
	return acc;
}

static inline void add_bytes(uint8_t* dst, const uint8_t* src, int w) {
	for (int i = 0; i < w; i++) dst[i] += src[i];
}

static inline int mid_pred(int a, int b, int c) {
	if (a > b) { const int t = a; a = b; b = t; }
	if (c < a) return a;
	return c > b ? b : c;
}

static void add_median_pred(uint8_t* dst, const uint8_t* src1, const uint8_t* diff, int w, int* left, int* left_top) {
	uint8_t l = *left, lt = *left_top;
	for (int i = 0; i < w; i++) {
		l = mid_pred(l, src1[i], (l + src1[i] - lt) & 0xFF) + diff[i];
		lt = src1[i];
		dst[i] = l;
	}
	*left = l;
	*left_top = lt;
}

static void add_left_pred_bgr32(uint8_t* dst, const uint8_t* src, int w, uint8_t* left) {
	uint8_t r = left[R], g = left[G], b = left[B], a = left[A];
	for (int i = 0; i < w; i++) {
		b += src[4 * i + B];
		g += src[4 * i + G];
		r += src[4 * i + R];
		a += src[4 * i + A];
		dst[4 * i + B] = b;
		dst[4 * i + G] = g;
		dst[4 * i + R] = r;
		dst[4 * i + A] = a;
	}
	left[B] = b;
	left[G] = g;
	left[R] = r;
	left[A] = a;
}

// -- the slice: decode_slice in huffyuvdec.c, one branch each --

static void decode_planar(RvCtx* s, HyuvCtx* c) {
	const int fake_mul = c->interlaced ? 2 : 1;
	for (int plane = 0; plane < 1 + 2 * c->chroma + c->alpha; plane++) {
		int left, lefttop, y;
		int w = s->width;
		int h = s->height;
		if (c->chroma && (plane == 1 || plane == 2)) {
			w >>= c->chroma_h_shift;
			h >>= c->chroma_v_shift;
		}
		uint8_t* const pic = c->pix[plane];
		const int ls = c->ls[plane];
		const int fake_stride = ls * fake_mul;

		switch (c->predictor) {
		case LEFT:
		case PLANE:
			decode_plane_bitstream(c, w, plane);
			left = add_left_pred(pic, c->temp[0], w, 0);
			for (y = 1; y < h; y++) {
				uint8_t* dst = pic + ls * y;
				decode_plane_bitstream(c, w, plane);
				left = add_left_pred(dst, c->temp[0], w, left);
				if (c->predictor == PLANE && y > c->interlaced) add_bytes(dst, dst - fake_stride, w);
			}
			break;
		case MEDIAN:
			decode_plane_bitstream(c, w, plane);
			left = add_left_pred(pic, c->temp[0], w, 0);
			y = 1;
			if (y >= h) break;
			if (c->interlaced) {
				decode_plane_bitstream(c, w, plane);
				left = add_left_pred(pic + ls, c->temp[0], w, left);
				y++;
				if (y >= h) break;
			}
			lefttop = pic[0];
			decode_plane_bitstream(c, w, plane);
			add_median_pred(pic + fake_stride, pic, c->temp[0], w, &left, &lefttop);
			y++;
			for (; y < h; y++) {
				decode_plane_bitstream(c, w, plane);
				uint8_t* dst = pic + ls * y;
				add_median_pred(dst, dst - fake_stride, c->temp[0], w, &left, &lefttop);
			}
			break;
		}
	}
}

static void decode_yuv(RvCtx* s, HyuvCtx* c) {
	const int width = s->width, height = s->height, width2 = width >> 1;
	uint8_t* const py = c->pix[0];
	uint8_t* const pu = c->pix[1];
	uint8_t* const pv = c->pix[2];
	const int lsy = c->ls[0], lsc = c->ls[1];
	const int fake_ystride = c->interlaced ? lsy * 2 : lsy;
	const int fake_cstride = c->interlaced ? lsc * 2 : lsc;
	uint8_t* const t0 = c->temp[0];
	uint8_t* const t1 = c->temp[1];
	uint8_t* const t2 = c->temp[2];
	int y, cy, lefty, leftu, leftv, lefttopy, lefttopu, lefttopv;

	leftv = pv[0] = get_bits8(c);
	lefty = py[1] = get_bits8(c);
	leftu = pu[0] = get_bits8(c);
	py[0] = get_bits8(c);

	decode_422_bitstream(c, width - 2);
	lefty = add_left_pred(py + 2, t0, width - 2, lefty);
	leftu = add_left_pred(pu + 1, t1, width2 - 1, leftu);
	leftv = add_left_pred(pv + 1, t2, width2 - 1, leftv);

	if (c->predictor != MEDIAN) {
		for (cy = y = 1; y < height; y++, cy++) {
			if (c->bitstream_bpp == 12) {
				decode_gray_bitstream(c, width);
				uint8_t* ydst = py + lsy * y;
				lefty = add_left_pred(ydst, t0, width, lefty);
				if (c->predictor == PLANE && y > c->interlaced) add_bytes(ydst, ydst - fake_ystride, width);
				y++;
				if (y >= height) break;
			}
			uint8_t* ydst = py + lsy * y;
			uint8_t* udst = pu + lsc * cy;
			uint8_t* vdst = pv + lsc * cy;
			decode_422_bitstream(c, width);
			lefty = add_left_pred(ydst, t0, width, lefty);
			leftu = add_left_pred(udst, t1, width2, leftu);
			leftv = add_left_pred(vdst, t2, width2, leftv);
			if (c->predictor == PLANE && cy > c->interlaced) {
				add_bytes(ydst, ydst - fake_ystride, width);
				add_bytes(udst, udst - fake_cstride, width2);
				add_bytes(vdst, vdst - fake_cstride, width2);
			}
		}
		return;
	}

	cy = y = 1;
	if (y >= height) return;

	if (c->interlaced) {
		decode_422_bitstream(c, width);
		lefty = add_left_pred(py + lsy, t0, width, lefty);
		leftu = add_left_pred(pu + lsc, t1, width2, leftu);
		leftv = add_left_pred(pv + lsc, t2, width2, leftv);
		y++;
		cy++;
		if (y >= height) return;
	}

	decode_422_bitstream(c, 4);
	lefty = add_left_pred(py + fake_ystride, t0, 4, lefty);
	leftu = add_left_pred(pu + fake_cstride, t1, 2, leftu);
	leftv = add_left_pred(pv + fake_cstride, t2, 2, leftv);

	lefttopy = py[3];
	decode_422_bitstream(c, width - 4);
	add_median_pred(py + fake_ystride + 4, py + 4, t0, width - 4, &lefty, &lefttopy);
	lefttopu = pu[1];
	lefttopv = pv[1];
	add_median_pred(pu + fake_cstride + 2, pu + 2, t1, width2 - 2, &leftu, &lefttopu);
	add_median_pred(pv + fake_cstride + 2, pv + 2, t2, width2 - 2, &leftv, &lefttopv);
	y++;
	cy++;

	for (; y < height; y++, cy++) {
		if (c->bitstream_bpp == 12) {
			while (2 * cy > y) {
				decode_gray_bitstream(c, width);
				uint8_t* ydst = py + lsy * y;
				add_median_pred(ydst, ydst - fake_ystride, t0, width, &lefty, &lefttopy);
				y++;
			}
			if (y >= height) break;
		}
		decode_422_bitstream(c, width);
		uint8_t* ydst = py + lsy * y;
		uint8_t* udst = pu + lsc * cy;
		uint8_t* vdst = pv + lsc * cy;
		add_median_pred(ydst, ydst - fake_ystride, t0, width, &lefty, &lefttopy);
		add_median_pred(udst, udst - fake_cstride, t1, width2, &leftu, &lefttopu);
		add_median_pred(vdst, vdst - fake_cstride, t2, width2, &leftv, &lefttopv);
	}
}

static int decode_bgr(RvCtx* s, HyuvCtx* c) {
	const int width = s->width, height = s->height;
	uint8_t* const pic = c->pix[0];
	const int ls = c->ls[0];
	const int fake_ystride = c->interlaced ? ls * 2 : ls;
	const int last_line = (height - 1) * ls;
	uint8_t left[4];

	if (c->predictor == MEDIAN) return AVERROR_PATCHWELCOME;

	if (c->bitstream_bpp == 32) {
		left[A] = pic[last_line + A] = get_bits8(c);
		left[R] = pic[last_line + R] = get_bits8(c);
		left[G] = pic[last_line + G] = get_bits8(c);
		left[B] = pic[last_line + B] = get_bits8(c);
	} else {
		left[R] = pic[last_line + R] = get_bits8(c);
		left[G] = pic[last_line + G] = get_bits8(c);
		left[B] = pic[last_line + B] = get_bits8(c);
		left[A] = pic[last_line + A] = 255;
		c->pos += 8;
	}

	decode_bgr_bitstream(c, width - 1);
	add_left_pred_bgr32(pic + last_line + 4, c->temp[0], width - 1, left);

	for (int y = height - 2; y >= 0; y--) {   // stored upside down
		decode_bgr_bitstream(c, width);
		add_left_pred_bgr32(pic + ls * y, c->temp[0], width, left);
		if (c->predictor == PLANE) {
			if (c->bitstream_bpp != 32) left[A] = 0;
			if (y < height - 1 - c->interlaced) add_bytes(pic + ls * y, pic + ls * y + fake_ystride, 4 * width);
		}
	}
	return 0;
}

// The finished picture into RvCtx.frame.
static void finish(RvCtx* s, HyuvCtx* c) {
	const int w = s->width, h = s->height;
	uint8_t* out = s->frame;

	if (c->out <= OUT_YCBCR410) {
		const int hmask = (1 << c->chroma_h_shift) - 1, vmask = (1 << c->chroma_v_shift) - 1;
		const int cw = (w + hmask) >> c->chroma_h_shift, ch = (h + vmask) >> c->chroma_v_shift;
		for (int y = 0; y < h; y++, out += w) memcpy(out, c->pix[0] + (size_t)y * c->ls[0], w);
		for (int y = 0; y < ch; y++) {
			const uint8_t* u = c->pix[1] + (size_t)y * c->ls[1];
			const uint8_t* v = c->pix[2] + (size_t)y * c->ls[2];
			for (int x = 0; x < cw; x++) {
				*out++ = u[x];
				*out++ = v[x];
			}
		}
		return;
	}

	const int opaque = c->out == OUT_RGB;
	if (c->version > 2) {
		// gray, or planes G,B,R[,A]
		const int rgb = c->chroma;
		for (int y = 0; y < h; y++) {
			const uint8_t* p0 = c->pix[0] + (size_t)y * c->ls[0];
			const uint8_t* p1 = rgb ? c->pix[1] + (size_t)y * c->ls[1] : p0;
			const uint8_t* p2 = rgb ? c->pix[2] + (size_t)y * c->ls[2] : p0;
			const uint8_t* p3 = opaque ? nullptr : c->pix[3] + (size_t)y * c->ls[3];
			for (int x = 0; x < w; x++) {
				*out++ = p3 ? p3[x] : 255;
				*out++ = p2[x];
				*out++ = p0[x];
				*out++ = p1[x];
			}
		}
		return;
	}
	for (int y = 0; y < h; y++) {
		const uint8_t* p = c->pix[0] + (size_t)y * c->ls[0];
		for (int x = 0; x < w; x++, p += 4) {
			*out++ = opaque ? 255 : p[A];
			*out++ = p[R];
			*out++ = p[G];
			*out++ = p[B];
		}
	}
}

static int decode(RvCtx* s, const uint8_t* buf, int buf_size) {
	HyuvCtx* const c = s->hyuv;
	if (!c) return AVERROR_PATCHWELCOME;   // the extradata was refused
	if (buf_size < (s->width * s->height + 7) / 8) return AVERROR_INVALIDDATA;

	if ((size_t)buf_size + 8 > c->streamCap) {
		free(c->stream);
		c->streamCap = (size_t)buf_size + 8;
		c->stream = (uint8_t*)malloc(c->streamCap);
		if (!c->stream) { c->streamCap = 0; return AVERROR_INVALIDDATA; }
	}
	memset(c->stream, 0, (size_t)buf_size + 8);
	for (int i = 0; i + 4 <= buf_size; i += 4) {
		c->stream[i] = buf[i + 3];
		c->stream[i + 1] = buf[i + 2];
		c->stream[i + 2] = buf[i + 1];
		c->stream[i + 3] = buf[i];
	}

	int table_size = 0;
	if (c->context) {
		table_size = read_huffman_tables(c, c->stream, buf_size);
		if (table_size < 0) return table_size;
	}
	c->pos = (size_t)table_size * 8;
	c->sizeBits = (size_t)buf_size * 8;

	if (c->version > 2) {
		decode_planar(s, c);
	} else if (c->bitstream_bpp < 24) {
		decode_yuv(s, c);
	} else {
		const int r = decode_bgr(s, c);
		if (r < 0) return r;
	}
	finish(s, c);
	return 0;
}

static void close(HyuvCtx* c) {
	if (!c) return;
	for (int i = 0; i < 3; i++) free(c->temp[i]);
	for (int i = 0; i < 4; i++) free(c->pix[i]);
	free(c->stream);
	free(c);
}

// decode_init: everything the extradata and the coded depth settle. >= 0: an OUT_ value.
static int open(RvCtx* s, const uint8_t* extradata, int extradata_size) {
	const int bits_per_coded_sample = s->depth;
	const int width = s->width, height = s->height;

	HyuvCtx* c = (HyuvCtx*)calloc(1, sizeof(HyuvCtx));
	if (!c) return AVERROR_INVALIDDATA;

	c->interlaced = height > 288;
	if (extradata_size) {
		if ((bits_per_coded_sample & 7) && bits_per_coded_sample != 12) c->version = 1;
		else if (extradata_size > 3 && extradata[3] == 0) c->version = 2;
		else c->version = 3;
	}

	int r = AVERROR_INVALIDDATA;
	c->chroma = 1;
	if (c->version >= 2) {
		if (extradata_size < 4) goto fail;
		c->decorrelate = extradata[0] & 64 ? 1 : 0;
		c->predictor = extradata[0] & 63;
		if (c->version == 2) {
			c->bitstream_bpp = extradata[1];
			if (c->bitstream_bpp == 0) c->bitstream_bpp = bits_per_coded_sample & ~7;
		} else {
			if ((extradata[1] >> 4) + 1 != 8) { r = AVERROR_PATCHWELCOME; goto fail; }
			c->chroma_h_shift = extradata[1] & 3;
			c->chroma_v_shift = (extradata[1] >> 2) & 3;
			c->yuv = !!(extradata[2] & 1);
			c->chroma = !!(extradata[2] & 3);
			c->alpha = !!(extradata[2] & 4);
		}
		{
			const int interlace = (extradata[2] & 0x30) >> 4;
			c->interlaced = interlace == 1 ? 1 : interlace == 2 ? 0 : c->interlaced;
		}
		c->context = extradata[2] & 0x40 ? 1 : 0;
		if (read_huffman_tables(c, extradata + 4, extradata_size - 4) < 0) goto fail;
	} else {
		switch (bits_per_coded_sample & 7) {
		case 1: c->predictor = LEFT; c->decorrelate = 0; break;
		case 2: c->predictor = LEFT; c->decorrelate = 1; break;
		case 3: c->predictor = PLANE; c->decorrelate = bits_per_coded_sample >= 24; break;
		case 4: c->predictor = MEDIAN; c->decorrelate = 0; break;
		default: c->predictor = LEFT; c->decorrelate = 0; break;
		}
		c->bitstream_bpp = bits_per_coded_sample & ~7;
		c->context = 0;
		read_old_huffman_tables(c);
	}

	if (c->version <= 2) {
		switch (c->bitstream_bpp) {
		case 12: c->out = OUT_YCBCR420; c->yuv = 1; c->chroma_h_shift = c->chroma_v_shift = 1; break;
		case 16: c->out = OUT_YCBCR422; c->yuv = 1; c->chroma_h_shift = 1; break;
		case 24: c->out = OUT_RGB; break;
		case 32: c->out = OUT_RGBA; c->alpha = 1; break;
		default: goto fail;
		}
	} else {
		switch ((c->chroma << 10) | (c->yuv << 9) | (c->alpha << 8) | 0x70 | c->chroma_h_shift | (c->chroma_v_shift << 2)) {
		case 0x070: c->out = OUT_RGB; break;        // gray
		case 0x470: c->out = OUT_RGB; break;        // G,B,R
		case 0x570: c->out = OUT_RGBA; break;       // G,B,R,A
		case 0x671: c->out = OUT_YCBCR422; break;
		case 0x675: c->out = OUT_YCBCR420; break;
		case 0x67A: c->out = OUT_YCBCR410; break;
		default: r = AVERROR_PATCHWELCOME; goto fail;
		}
	}

	if ((c->out == OUT_YCBCR422 || c->out == OUT_YCBCR420) && (width & 1)) goto fail;
	if (c->predictor == MEDIAN && c->out == OUT_YCBCR422 && (width % 4)) goto fail;

	for (int i = 0; i < 3; i++) {
		c->temp[i] = (uint8_t*)malloc(4 * width + 16);
		if (!c->temp[i]) goto fail;
	}
	{
		// Rows keep FFmpeg's slack: the first rows of a narrow picture write a fixed 2 or 4 pixels.
		const int packed = c->version <= 2 && c->bitstream_bpp >= 24;
		const int planes = c->version <= 2 ? (packed ? 1 : 3) : 1 + 2 * c->chroma + c->alpha;
		for (int i = 0; i < planes; i++) {
			const int sub = !packed && c->yuv && (i == 1 || i == 2);
			const int hs = sub ? c->chroma_h_shift : 0, vs = sub ? c->chroma_v_shift : 0;
			c->ls[i] = (packed ? width * 4 : (width + (1 << hs) - 1) >> hs) + 16;
			const size_t rows = (size_t)((height + (1 << vs) - 1) >> vs) + 1;
			c->pix[i] = (uint8_t*)calloc((size_t)c->ls[i] * rows, 1);
			if (!c->pix[i]) goto fail;
		}
	}

	close(s->hyuv);
	s->hyuv = c;
	return c->out;

fail:
	close(c);
	return r;
}

} // namespace hfyu

#endif
