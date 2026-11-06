/*
	"retrovideo" — a single WASM module bundling the esoteric AVI/QuickTime video
	codecs that don't warrant their own module: Cinepak (CVID), Microsoft Video-1
	(CRAM/MSVC), and (incrementally) the Indeo family (RT21 / IV3x / IV41 / IV50).
	Kept SEPARATE from the VP6 module on purpose.

	Each codec is a direct port of the corresponding FFmpeg decoder (libavcodec,
	release/8.1, LGPL); the surrounding libavcodec framework (AVCodecContext /
	AVFrame / AVPacket / the palette side-data plumbing) is replaced by a
	self-contained handle (RvCtx) that owns its output frame buffer across calls —
	these codecs decode inter-frames as partial updates of the previous picture, so
	the buffer MUST persist (FFmpeg achieves this with ff_reget_buffer). The decoder
	emits the codec's native surface (RGB24 / RGB555 / 8-bit index / — later —
	YCbCr 4:1:0); palette attachment and any RGBA expansion stay in JS/the readers,
	exactly as the existing JS decoders do.

	Conformance oracle: FFmpeg itself (tests decode the same samples with the ffmpeg
	CLI and compare pixels). The JS decoders in codecs/ffmpeg_js/ are not required.

	Boundary (extern "C", prefixed _ on the JS side; see retrovideo_decode_core.js):
	  RvCtx* rv_open(int codecId, int width, int height, int flags)
	  void   rv_close(RvCtx*)
	  void   rv_reset(RvCtx*)                          clear the persistent picture (seek)
	  int    rv_decode_frame(RvCtx*, const uint8_t* data, int len, int32_t info[8])
	           info = { codedWidth, codedHeight, format, bytesPerPixel, hasAlpha,
	                    ok, reserved, reserved }; returns 1/0
	  int    rv_set_extradata(RvCtx*, data, len)       HuffYUV: answers what its frames hold
	  uint8_t* rv_plane_ptr(RvCtx*, int idx)           idx 0 = Y / packed RGB / index
	                                                   idx 1,2 = Cb,Cr (YCbCr codecs)
	  void   rv_set_simd(int on)                       runtime SIMD toggle (Indeo only)
	  int    rv_test_transform(int kind, ...)          SIMD-vs-scalar hook (Indeo only)

	codecId / format / flags are defined in retrovideo_ids.h (shared with the JS).
*/

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

// Runtime SIMD toggle for the Indeo 4/5 inverse transforms. Declared BEFORE the
// transform header so its dispatch (`if (g_use_simd) ...`) can see it; written by
// rv_set_simd() and read by both the production transform path and the
// rv_test_transform() SIMD-vs-scalar conformance hook. Default ON; only the IVI
// slant/Haar transforms honor it (cinepak/msvideo1/indeo2/indeo3 are scalar).
static int g_use_simd = 1;

#include "ivi_transforms.h"   // iv45:: Indeo 4/5 inverse transforms + MC (scalar + i32x4 SIMD)
#include "ivi_core.h"         // ivi::  shared IVI band/tile/MB decode core + frontend hooks
#include "iv3_decode.h"       // iv3::  standalone Indeo 3 cell-tree / VQ decoder

// ── public enums (kept in lockstep with retrovideo_decode_core.js) ──
enum { RV_CVID = 0, RV_MSV1 = 1, RV_IV2 = 2, RV_IV3 = 3, RV_IV4 = 4, RV_IV5 = 5, RV_MSRLE = 6, RV_QTRLE = 7, RV_RPZA = 8, RV_SMC = 9, RV_8BPS = 10, RV_HFYU = 11 };
enum { RV_FMT_RGB24 = 0, RV_FMT_RGB555 = 1, RV_FMT_INDEX8 = 2, RV_FMT_YCBCR410 = 3, RV_FMT_ARGB32 = 4 };
enum { RV_FLAG_PALETTED = 1 };   // bit 0: 8-bit/paletted mode (cinepak & msvideo1)
// Bits 8 and up of the open flags: the coded depth, for the RLE codecs.
enum { RV_FLAG_DEPTH_SHIFT = 8 };

// info[6] warn codes — mapped to a callbacks.warn() string on the JS side so a
// failed/unsupported frame is surfaced to the user instead of silently dropped.
enum { RV_W_NONE = 0, RV_W_BADDATA = 1, RV_W_UNSUPPORTED = 2, RV_W_8BIT = 3, RV_W_HALFPEL = 4 };

// Translate a decoder's negative return code into an info[6] warn code.
static int rv_warn_code(int r) {
	switch (r) {
	case 0:  return RV_W_NONE;
	case -3: return RV_W_8BIT;       // iv3: 8-bit paletted pixel format
	case -4: return RV_W_HALFPEL;    // iv3: half-pel motion vectors
	case -2: return RV_W_UNSUPPORTED;// AVERROR_PATCHWELCOME (unsupported feature)
	default: return RV_W_BADDATA;    // AVERROR_INVALIDDATA / corrupt
	}
}

#define AVERROR_INVALIDDATA  (-1)
#define AVERROR_PATCHWELCOME (-2)

// ── small byte helpers (big/little-endian reads, clamp) ──
static inline int av_clip_uint8(int a) { return a < 0 ? 0 : (a > 255 ? 255 : a); }
static inline int FFMIN(int a, int b) { return a < b ? a : b; }
static inline uint32_t AV_RB16(const uint8_t* p) { return ((uint32_t)p[0] << 8) | p[1]; }
static inline uint32_t AV_RB24(const uint8_t* p) { return ((uint32_t)p[0] << 16) | ((uint32_t)p[1] << 8) | p[2]; }
static inline uint32_t AV_RB32(const uint8_t* p) { return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) | ((uint32_t)p[2] << 8) | p[3]; }
static inline uint32_t AV_RL16(const uint8_t* p) { return ((uint32_t)p[1] << 8) | p[0]; }

// ── unified handle ──
struct RvCtx {
	int codecId;
	int width, height;     // coded dimensions (codec-rounded)
	int format;            // RV_FMT_*
	int bpp;               // output bytes per pixel
	int stride;            // output stride in bytes
	uint8_t* frame;        // persistent output picture / Y plane (owned)
	size_t   frameSize;

	// Indeo YCbCr 4:1:0: chroma planes (plane 0 == `frame` == Y). Owned.
	uint8_t* plane1;       // Cb / U  (1/4 res each dim)
	uint8_t* plane2;       // Cr / V
	int      chromaW, chromaH;

	// Indeo 4/5 (IVI) decoder context (owned). Null for the other codecs.
	ivi::Ctx* ivi;

	// Indeo 3 decoder context (owned). Null for the other codecs.
	iv3::Ctx* iv3;

	// Soft warn code (RV_W_*) for a frame that still produced output (e.g. a
	// partially-decoded corrupt Indeo 3 frame); 0 = none. Set per decode.
	int warn;

	// Cinepak per-strip codebooks: 32 strips x (v4[256] + v1[256]) x 12 bytes.
	// Allocated lazily on first cinepak decode.
	uint8_t* cvid_codebooks;   // [MAX_STRIPS][2][256][12]
	int      cvid_strip_id[32];
	int      cvid_strip_x1[32], cvid_strip_y1[32], cvid_strip_x2[32], cvid_strip_y2[32];
	int      sega_film_skip_bytes;

	// Microsoft RLE and QuickTime Animation: coded bits per pixel; paletted
	// QuickTime Animation decodes into rlePic (owned, see rle_decode.h).
	int      depth;
	uint8_t* rlePic;
	int      rleStride;

	// Apple Video and Apple Graphics decode into blockPic (owned), whole 4x4 blocks:
	// rows of blockStride bytes, the height rounded up to a block.
	uint8_t* blockPic;
	int      blockStride;
	size_t   blockPicSize;

	// Apple Graphics color tables, kept across frames as FFmpeg keeps them.
	uint8_t  smcPairs[256 * 2];
	uint8_t  smcQuads[256 * 4];
	uint8_t  smcOctets[256 * 8];

	struct HyuvCtx* hyuv;   // HuffYUV, set up by rv_set_extradata
};

// A finished block picture, tight, into the output frame.
static void block_pic_finish(RvCtx* s) {
	const int rowBytes = s->width * s->bpp;
	for (int y = 0; y < s->height; y++) {
		memcpy(s->frame + (size_t)y * s->stride, s->blockPic + (size_t)y * s->blockStride, rowBytes);
	}
}

#include "rle_decode.h"   // msrle::, qtrle::
#include "rpza_decode.h"  // rpza::
#include "smc_decode.h"   // smc::
#include "eightbps_decode.h"   // eightbps::
#include "huffyuv_decode.h"    // hfyu::

// ════════════════════════════════════════════════════════════════════════════
//  Cinepak (CVID) — port of FFmpeg libavcodec/cinepak.c
// ════════════════════════════════════════════════════════════════════════════
namespace cvid {

static const int MAX_STRIPS = 32;

// codebook layout helpers: each strip owns v4 then v1, 256 entries of 12 bytes.
static inline uint8_t* v4_codebook(RvCtx* s, int strip) { return s->cvid_codebooks + (size_t)strip * 2 * 256 * 12; }
static inline uint8_t* v1_codebook(RvCtx* s, int strip) { return v4_codebook(s, strip) + 256 * 12; }

static void decode_codebook(uint8_t* codebook, int chunk_id, int size, const uint8_t* data) {
	const uint8_t* eod = data + size;
	uint32_t flag = 0, mask = 0;
	int n = (chunk_id & 0x04) ? 4 : 6;
	uint8_t* p = codebook;

	for (int i = 0; i < 256; i++) {
		if ((chunk_id & 0x01) && !(mask >>= 1)) {
			if (data + 4 > eod) break;
			flag = AV_RB32(data);
			data += 4;
			mask = 0x80000000u;
		}
		if (!(chunk_id & 0x01) || (flag & mask)) {
			if (data + n > eod) break;
			for (int k = 0; k < 4; k++) {
				int r = *data++;
				for (int kk = 0; kk < 3; kk++) *p++ = r;
			}
			if (n == 6) {
				int u = (int8_t)*data++;
				int v = (int8_t)*data++;
				p -= 12;
				for (int k = 0; k < 4; k++) {
					int r = *p++ + v * 2;
					int g = *p++ - (u / 2) - v;
					int b = *p   + u * 2;
					p -= 2;
					*p++ = av_clip_uint8(r);
					*p++ = av_clip_uint8(g);
					*p++ = av_clip_uint8(b);
				}
			}
		} else {
			p += 12;
		}
	}
}

static int decode_vectors(RvCtx* s, int strip, int chunk_id, int size, const uint8_t* data) {
	const uint8_t* eod = data + size;
	uint32_t flag = 0, mask = 0;
	const uint8_t* cb0; const uint8_t* cb1; const uint8_t* cb2; const uint8_t* cb3;
	const int palette = (s->format == RV_FMT_INDEX8);
	const int stride = s->stride;
	uint8_t* v1 = v1_codebook(s, strip);
	uint8_t* v4 = v4_codebook(s, strip);
	const int x1 = s->cvid_strip_x1[strip], y1 = s->cvid_strip_y1[strip];
	const int x2 = s->cvid_strip_x2[strip], y2 = s->cvid_strip_y2[strip];

	for (int y = y1; y < y2; y += 4) {
		// take care of y dimension not being a multiple of 4
		uint8_t* ip0; uint8_t* ip1; uint8_t* ip2; uint8_t* ip3;
		ip0 = ip1 = ip2 = ip3 = s->frame + (palette ? x1 : x1 * 3) + (size_t)y * stride;
		if (s->height - y > 1) {
			ip1 = ip0 + stride;
			if (s->height - y > 2) {
				ip2 = ip1 + stride;
				if (s->height - y > 3) {
					ip3 = ip2 + stride;
				}
			}
		}

		for (int x = x1; x < x2; x += 4) {
			if ((chunk_id & 0x01) && !(mask >>= 1)) {
				if (data + 4 > eod) return AVERROR_INVALIDDATA;
				flag = AV_RB32(data); data += 4; mask = 0x80000000u;
			}

			if (!(chunk_id & 0x01) || (flag & mask)) {
				if (!(chunk_id & 0x02) && !(mask >>= 1)) {
					if (data + 4 > eod) return AVERROR_INVALIDDATA;
					flag = AV_RB32(data); data += 4; mask = 0x80000000u;
				}

				if ((chunk_id & 0x02) || (~flag & mask)) {
					if (data >= eod) return AVERROR_INVALIDDATA;
					uint8_t* p = v1 + (size_t)(*data++) * 12;
					if (palette) {
						ip3[0] = ip3[1] = ip2[0] = ip2[1] = p[6];
						ip3[2] = ip3[3] = ip2[2] = ip2[3] = p[9];
						ip1[0] = ip1[1] = ip0[0] = ip0[1] = p[0];
						ip1[2] = ip1[3] = ip0[2] = ip0[3] = p[3];
					} else {
						p += 6;
						memcpy(ip3 + 0, p, 3); memcpy(ip3 + 3, p, 3);
						memcpy(ip2 + 0, p, 3); memcpy(ip2 + 3, p, 3);
						p += 3;
						memcpy(ip3 + 6, p, 3); memcpy(ip3 + 9, p, 3);
						memcpy(ip2 + 6, p, 3); memcpy(ip2 + 9, p, 3);
						p -= 9;
						memcpy(ip1 + 0, p, 3); memcpy(ip1 + 3, p, 3);
						memcpy(ip0 + 0, p, 3); memcpy(ip0 + 3, p, 3);
						p += 3;
						memcpy(ip1 + 6, p, 3); memcpy(ip1 + 9, p, 3);
						memcpy(ip0 + 6, p, 3); memcpy(ip0 + 9, p, 3);
					}
				} else if (flag & mask) {
					if (data + 4 > eod) return AVERROR_INVALIDDATA;
					cb0 = v4 + (size_t)(*data++) * 12;
					cb1 = v4 + (size_t)(*data++) * 12;
					cb2 = v4 + (size_t)(*data++) * 12;
					cb3 = v4 + (size_t)(*data++) * 12;
					if (palette) {
						uint8_t* p;
						p = ip3; *p++ = cb2[6]; *p++ = cb2[9]; *p++ = cb3[6]; *p = cb3[9];
						p = ip2; *p++ = cb2[0]; *p++ = cb2[3]; *p++ = cb3[0]; *p = cb3[3];
						p = ip1; *p++ = cb0[6]; *p++ = cb0[9]; *p++ = cb1[6]; *p = cb1[9];
						p = ip0; *p++ = cb0[0]; *p++ = cb0[3]; *p++ = cb1[0]; *p = cb1[3];
					} else {
						memcpy(ip3 + 0, cb2 + 6, 6);
						memcpy(ip3 + 6, cb3 + 6, 6);
						memcpy(ip2 + 0, cb2 + 0, 6);
						memcpy(ip2 + 6, cb3 + 0, 6);
						memcpy(ip1 + 0, cb0 + 6, 6);
						memcpy(ip1 + 6, cb1 + 6, 6);
						memcpy(ip0 + 0, cb0 + 0, 6);
						memcpy(ip0 + 6, cb1 + 0, 6);
					}
				}
			}

			if (palette) { ip0 += 4; ip1 += 4; ip2 += 4; ip3 += 4; }
			else         { ip0 += 12; ip1 += 12; ip2 += 12; ip3 += 12; }
		}
	}
	return 0;
}

static int decode_strip(RvCtx* s, int strip, const uint8_t* data, int size) {
	const uint8_t* eod = data + size;
	// coordinate sanity checks
	if (s->cvid_strip_x2[strip] > s->width  ||
	    s->cvid_strip_y2[strip] > s->height ||
	    s->cvid_strip_x1[strip] >= s->cvid_strip_x2[strip] ||
	    s->cvid_strip_y1[strip] >= s->cvid_strip_y2[strip])
		return AVERROR_INVALIDDATA;

	while (data + 4 <= eod) {
		int chunk_id   = data[0];
		int chunk_size = AV_RB24(&data[1]) - 4;
		if (chunk_size < 0) return AVERROR_INVALIDDATA;
		data += 4;
		chunk_size = (data + chunk_size > eod) ? (int)(eod - data) : chunk_size;

		switch (chunk_id) {
		case 0x20: case 0x21: case 0x24: case 0x25:
			decode_codebook(v4_codebook(s, strip), chunk_id, chunk_size, data);
			break;
		case 0x22: case 0x23: case 0x26: case 0x27:
			decode_codebook(v1_codebook(s, strip), chunk_id, chunk_size, data);
			break;
		case 0x30: case 0x31: case 0x32:
			return decode_vectors(s, strip, chunk_id, chunk_size, data);
		}
		data += chunk_size;
	}
	return AVERROR_INVALIDDATA;
}

// returns 0 on success, negative AVERROR otherwise.
static int decode(RvCtx* s, const uint8_t* buf, int buf_size) {
	if (buf_size < 10) return AVERROR_INVALIDDATA;

	int num_strips = AV_RB16(&buf[8]);
	// Empty frame (and no palette side-data, which we never have): nothing to do.
	if (!num_strips) return 0;

	// ── predecode check (cinepak_predecode_check) ──
	int encoded_buf_size = AV_RB24(&buf[1]);
	// discard_damaged_percentage defaults to 0 -> threshold is encoded_buf_size.
	if (buf_size < encoded_buf_size) return AVERROR_INVALIDDATA;

	if (s->sega_film_skip_bytes == -1) {
		if (!encoded_buf_size) return AVERROR_PATCHWELCOME;
		if (encoded_buf_size != buf_size && (buf_size % encoded_buf_size) != 0) {
			if (buf_size >= 16 && buf[10] == 0xFE && buf[11] == 0x00 && buf[12] == 0x00 &&
			    buf[13] == 0x06 && buf[14] == 0x00 && buf[15] == 0x00)
				s->sega_film_skip_bytes = 6;
			else
				s->sega_film_skip_bytes = 2;
		} else {
			s->sega_film_skip_bytes = 0;
		}
	}

	if (buf_size < 10 + s->sega_film_skip_bytes + num_strips * 12) return AVERROR_INVALIDDATA;
	{
		const uint8_t* d = buf + 10 + s->sega_film_skip_bytes;
		int strip_size = AV_RB24(d + 1);
		if (strip_size < 12 || strip_size > encoded_buf_size) return AVERROR_INVALIDDATA;
	}

	// ── decode (cinepak_decode) ──
	const uint8_t* eod = buf + buf_size;
	int frame_flags = buf[0];
	num_strips = FFMIN(num_strips, MAX_STRIPS);
	const uint8_t* data = buf + 10 + s->sega_film_skip_bytes;
	int y0 = 0;

	for (int i = 0; i < num_strips; i++) {
		if (data + 12 > eod) return AVERROR_INVALIDDATA;
		s->cvid_strip_id[i] = data[0];
		// zero y1 means "relative to the previous stripe"
		if (!(s->cvid_strip_y1[i] = AV_RB16(&data[4])))
			s->cvid_strip_y2[i] = (s->cvid_strip_y1[i] = y0) + AV_RB16(&data[8]);
		else
			s->cvid_strip_y2[i] = AV_RB16(&data[8]);
		s->cvid_strip_x1[i] = AV_RB16(&data[6]);
		s->cvid_strip_x2[i] = AV_RB16(&data[10]);

		int strip_size = AV_RB24(&data[1]) - 12;
		if (strip_size < 0) return AVERROR_INVALIDDATA;
		data += 12;
		strip_size = (data + strip_size > eod) ? (int)(eod - data) : strip_size;

		if (i > 0 && !(frame_flags & 0x01)) {
			// codebooks inherited from the previous strip
			memcpy(v4_codebook(s, i), v4_codebook(s, i - 1), 256 * 12);
			memcpy(v1_codebook(s, i), v1_codebook(s, i - 1), 256 * 12);
		}

		int result = decode_strip(s, i, data, strip_size);
		if (result != 0) return result;

		data += strip_size;
		y0 = s->cvid_strip_y2[i];
	}
	return 0;
}

} // namespace cvid

// ════════════════════════════════════════════════════════════════════════════
//  Microsoft Video-1 (CRAM/MSVC) — port of FFmpeg libavcodec/msvideo1.c
// ════════════════════════════════════════════════════════════════════════════
namespace msv1 {

// CHECK_STREAM_PTR: bail out of the decode (leaving the partial picture) if the
// stream pointer would run past the end — mirrors FFmpeg's early `return`.
#define CHECK_STREAM_PTR(n) do { if ((stream_ptr + (n)) > size) return; } while (0)

static void decode_8bit(RvCtx* s, const uint8_t* buf, int size) {
	const int width = s->width, height = s->height;
	const int stride = s->stride;                 // bytes per row == width
	uint8_t* pixels = s->frame;

	int stream_ptr = 0, skip_blocks = 0;
	int blocks_wide = width / 4, blocks_high = height / 4;
	int total_blocks = blocks_wide * blocks_high;
	const int block_inc = 4;
	const int row_dec = stride + 4;
	unsigned char colors[8];

	for (int block_y = blocks_high; block_y > 0; block_y--) {
		int block_ptr = ((block_y * 4) - 1) * stride;
		for (int block_x = blocks_wide; block_x > 0; block_x--) {
			if (skip_blocks) { block_ptr += block_inc; skip_blocks--; total_blocks--; continue; }
			int pixel_ptr = block_ptr;

			CHECK_STREAM_PTR(2);
			unsigned char byte_a = buf[stream_ptr++];
			unsigned char byte_b = buf[stream_ptr++];

			if (byte_a == 0 && byte_b == 0 && total_blocks == 0) return;
			else if ((byte_b & 0xFC) == 0x84) {
				skip_blocks = ((byte_b - 0x84) << 8) + byte_a - 1;
			} else if (byte_b < 0x80) {
				unsigned short flags = (byte_b << 8) | byte_a;
				CHECK_STREAM_PTR(2);
				colors[0] = buf[stream_ptr++];
				colors[1] = buf[stream_ptr++];
				for (int py = 0; py < 4; py++) {
					for (int px = 0; px < 4; px++, flags >>= 1)
						pixels[pixel_ptr++] = colors[(flags & 0x1) ^ 1];
					pixel_ptr -= row_dec;
				}
			} else if (byte_b >= 0x90) {
				unsigned short flags = (byte_b << 8) | byte_a;
				CHECK_STREAM_PTR(8);
				memcpy(colors, &buf[stream_ptr], 8);
				stream_ptr += 8;
				for (int py = 0; py < 4; py++) {
					for (int px = 0; px < 4; px++, flags >>= 1)
						pixels[pixel_ptr++] = colors[((py & 0x2) << 1) + (px & 0x2) + ((flags & 0x1) ^ 1)];
					pixel_ptr -= row_dec;
				}
			} else {
				colors[0] = byte_a;
				for (int py = 0; py < 4; py++) {
					for (int px = 0; px < 4; px++) pixels[pixel_ptr++] = colors[0];
					pixel_ptr -= row_dec;
				}
			}
			block_ptr += block_inc;
			total_blocks--;
		}
	}
}

static void decode_16bit(RvCtx* s, const uint8_t* buf, int size) {
	const int width = s->width, height = s->height;
	const int stride = s->width;                  // stride in uint16 units == width
	unsigned short* pixels = (unsigned short*)s->frame;

	int stream_ptr = 0, skip_blocks = 0;
	int blocks_wide = width / 4, blocks_high = height / 4;
	int total_blocks = blocks_wide * blocks_high;
	const int block_inc = 4;
	const int row_dec = stride + 4;
	unsigned short colors[8];

	for (int block_y = blocks_high; block_y > 0; block_y--) {
		int block_ptr = ((block_y * 4) - 1) * stride;
		for (int block_x = blocks_wide; block_x > 0; block_x--) {
			if (skip_blocks) { block_ptr += block_inc; skip_blocks--; total_blocks--; continue; }
			int pixel_ptr = block_ptr;

			CHECK_STREAM_PTR(2);
			unsigned char byte_a = buf[stream_ptr++];
			unsigned char byte_b = buf[stream_ptr++];

			if (byte_a == 0 && byte_b == 0 && total_blocks == 0) return;
			else if ((byte_b & 0xFC) == 0x84) {
				skip_blocks = ((byte_b - 0x84) << 8) + byte_a - 1;
			} else if (byte_b < 0x80) {
				unsigned short flags = (byte_b << 8) | byte_a;
				CHECK_STREAM_PTR(4);
				colors[0] = AV_RL16(&buf[stream_ptr]); stream_ptr += 2;
				colors[1] = AV_RL16(&buf[stream_ptr]); stream_ptr += 2;
				if (colors[0] & 0x8000) {
					CHECK_STREAM_PTR(12);
					colors[2] = AV_RL16(&buf[stream_ptr]); stream_ptr += 2;
					colors[3] = AV_RL16(&buf[stream_ptr]); stream_ptr += 2;
					colors[4] = AV_RL16(&buf[stream_ptr]); stream_ptr += 2;
					colors[5] = AV_RL16(&buf[stream_ptr]); stream_ptr += 2;
					colors[6] = AV_RL16(&buf[stream_ptr]); stream_ptr += 2;
					colors[7] = AV_RL16(&buf[stream_ptr]); stream_ptr += 2;
					for (int py = 0; py < 4; py++) {
						for (int px = 0; px < 4; px++, flags >>= 1)
							pixels[pixel_ptr++] = colors[((py & 0x2) << 1) + (px & 0x2) + ((flags & 0x1) ^ 1)];
						pixel_ptr -= row_dec;
					}
				} else {
					for (int py = 0; py < 4; py++) {
						for (int px = 0; px < 4; px++, flags >>= 1)
							pixels[pixel_ptr++] = colors[(flags & 0x1) ^ 1];
						pixel_ptr -= row_dec;
					}
				}
			} else {
				colors[0] = (byte_b << 8) | byte_a;
				for (int py = 0; py < 4; py++) {
					for (int px = 0; px < 4; px++) pixels[pixel_ptr++] = colors[0];
					pixel_ptr -= row_dec;
				}
			}
			block_ptr += block_inc;
			total_blocks--;
		}
	}
}

#undef CHECK_STREAM_PTR

// returns 0 on success.
static int decode(RvCtx* s, const uint8_t* buf, int buf_size) {
	// Discard frame if it's smaller than the minimum frame size.
	if (buf_size < (s->width / 4) * (s->height / 4) / 512) return AVERROR_INVALIDDATA;
	if (s->format == RV_FMT_INDEX8) decode_8bit(s, buf, buf_size);
	else                            decode_16bit(s, buf, buf_size);
	return 0;
}

} // namespace msv1

// ════════════════════════════════════════════════════════════════════════════
//  Intel Indeo 2 (RT21 / IF09) — port of FFmpeg libavcodec/indeo2.c
//  YUV410P; one 14-bit LE VLC + four 256-byte delta codebooks; intra/inter.
// ════════════════════════════════════════════════════════════════════════════
namespace iv2 {

#define IR2_CODES 143
// {symbol, length}; already grouped by ascending length (== a stable length sort).
static const uint8_t ir2_tab[IR2_CODES][2] = {
	{0x01,3},{0x02,3},{0x80,3},{0x03,3},{0x04,5},{0x81,5},{0x05,5},{0x06,5},{0x82,5},{0x83,5},
	{0x07,5},{0x08,5},{0x84,6},{0x09,6},{0x0A,6},{0x0B,6},{0x0C,6},{0x0D,6},{0x0E,6},{0x85,6},
	{0x0F,8},{0x10,8},{0x86,8},{0x87,8},{0x11,8},{0x12,8},{0x13,8},{0x14,8},{0x88,8},{0x15,8},
	{0x16,8},{0x89,8},{0x17,8},{0x18,8},{0x8A,8},{0x19,8},{0x1A,9},{0x8B,9},{0x1B,9},{0x1C,9},
	{0x8C,9},{0x1D,9},{0x1E,9},{0x8D,9},{0x1F,9},{0x20,9},{0x8E,9},{0x21,9},{0x22,9},{0x8F,9},
	{0x23,9},{0x24,9},{0x25,10},{0x26,10},{0x27,10},{0x28,10},{0x29,10},{0x2A,10},{0x2B,10},{0x2C,10},
	{0x2D,10},{0x2E,10},{0x2F,10},{0x30,10},{0x31,10},{0x32,10},{0x33,10},{0x34,10},{0x35,13},{0x36,13},
	{0x37,13},{0x38,13},{0x39,13},{0x3A,13},{0x3B,13},{0x3C,13},{0x3D,13},{0x3E,13},{0x3F,13},{0x40,13},
	{0x41,13},{0x42,13},{0x43,13},{0x44,13},{0x45,13},{0x46,13},{0x47,13},{0x48,13},{0x49,13},{0x4A,13},
	{0x4B,13},{0x4C,13},{0x4D,13},{0x4E,13},{0x4F,13},{0x50,13},{0x51,13},{0x52,13},{0x53,13},{0x54,13},
	{0x55,13},{0x56,13},{0x57,13},{0x58,13},{0x59,13},{0x5A,13},{0x5B,13},{0x5C,13},{0x5D,13},{0x5E,13},
	{0x5F,13},{0x60,13},{0x61,13},{0x62,13},{0x63,13},{0x64,13},{0x65,13},{0x66,13},{0x67,13},{0x68,13},
	{0x69,13},{0x6A,13},{0x6B,13},{0x6C,13},{0x6D,13},{0x6E,13},{0x6F,13},{0x70,13},{0x71,13},{0x72,13},
	{0x73,13},{0x74,13},{0x75,14},{0x76,14},{0x77,14},{0x78,14},{0x79,14},{0x7A,14},{0x7B,14},{0x7C,14},
	{0x7D,14},{0x7E,14},{0x7F,14},
};

static const uint8_t ir2_delta_table[4][256] = {
	{0x80,0x80,0x84,0x84,0x7C,0x7C,0x7F,0x85,0x81,0x7B,0x85,0x7F,0x7B,0x81,0x8C,0x8C,
	 0x74,0x74,0x83,0x8D,0x7D,0x73,0x8D,0x83,0x73,0x7D,0x77,0x89,0x89,0x77,0x89,0x77,
	 0x77,0x89,0x8C,0x95,0x74,0x6B,0x95,0x8C,0x6B,0x74,0x7C,0x90,0x84,0x70,0x90,0x7C,
	 0x70,0x84,0x96,0x96,0x6A,0x6A,0x82,0x98,0x7E,0x68,0x98,0x82,0x68,0x7E,0x97,0xA2,
	 0x69,0x5E,0xA2,0x97,0x5E,0x69,0xA2,0xA2,0x5E,0x5E,0x8B,0xA3,0x75,0x5D,0xA3,0x8B,
	 0x5D,0x75,0x71,0x95,0x8F,0x6B,0x95,0x71,0x6B,0x8F,0x78,0x9D,0x88,0x63,0x9D,0x78,
	 0x63,0x88,0x7F,0xA7,0x81,0x59,0xA7,0x7F,0x59,0x81,0xA4,0xB1,0x5C,0x4F,0xB1,0xA4,
	 0x4F,0x5C,0x96,0xB1,0x6A,0x4F,0xB1,0x96,0x4F,0x6A,0xB2,0xB2,0x4E,0x4E,0x65,0x9B,
	 0x9B,0x65,0x9B,0x65,0x65,0x9B,0x89,0xB4,0x77,0x4C,0xB4,0x89,0x4C,0x77,0x6A,0xA3,
	 0x96,0x5D,0xA3,0x6A,0x5D,0x96,0x73,0xAC,0x8D,0x54,0xAC,0x73,0x54,0x8D,0xB4,0xC3,
	 0x4C,0x3D,0xC3,0xB4,0x3D,0x4C,0xA4,0xC3,0x5C,0x3D,0xC3,0xA4,0x3D,0x5C,0xC4,0xC4,
	 0x3C,0x3C,0x96,0xC6,0x6A,0x3A,0xC6,0x96,0x3A,0x6A,0x7C,0xBA,0x84,0x46,0xBA,0x7C,
	 0x46,0x84,0x5B,0xAB,0xA5,0x55,0xAB,0x5B,0x55,0xA5,0x63,0xB4,0x9D,0x4C,0xB4,0x63,
	 0x4C,0x9D,0x86,0xCA,0x7A,0x36,0xCA,0x86,0x36,0x7A,0xB6,0xD7,0x4A,0x29,0xD7,0xB6,
	 0x29,0x4A,0xC8,0xD7,0x38,0x29,0xD7,0xC8,0x29,0x38,0xA4,0xD8,0x5C,0x28,0xD8,0xA4,
	 0x28,0x5C,0x6C,0xC1,0x94,0x3F,0xC1,0x6C,0x3F,0x94,0xD9,0xD9,0x27,0x27,0x80,0x80},
	{0x80,0x80,0x85,0x85,0x7B,0x7B,0x7E,0x87,0x82,0x79,0x87,0x7E,0x79,0x82,0x8F,0x8F,
	 0x71,0x71,0x84,0x8F,0x7C,0x71,0x8F,0x84,0x71,0x7C,0x75,0x8B,0x8B,0x75,0x8B,0x75,
	 0x75,0x8B,0x8E,0x9A,0x72,0x66,0x9A,0x8E,0x66,0x72,0x7B,0x93,0x85,0x6D,0x93,0x7B,
	 0x6D,0x85,0x9B,0x9B,0x65,0x65,0x82,0x9D,0x7E,0x63,0x9D,0x82,0x63,0x7E,0x9B,0xA8,
	 0x65,0x58,0xA8,0x9B,0x58,0x65,0xA9,0xA9,0x57,0x57,0x8D,0xAA,0x73,0x56,0xAA,0x8D,
	 0x56,0x73,0x6E,0x99,0x92,0x67,0x99,0x6E,0x67,0x92,0x76,0xA2,0x8A,0x5E,0xA2,0x76,
	 0x5E,0x8A,0x7F,0xAF,0x81,0x51,0xAF,0x7F,0x51,0x81,0xAB,0xBA,0x55,0x46,0xBA,0xAB,
	 0x46,0x55,0x9A,0xBB,0x66,0x45,0xBB,0x9A,0x45,0x66,0xBB,0xBB,0x45,0x45,0x60,0xA0,
	 0xA0,0x60,0xA0,0x60,0x60,0xA0,0x8B,0xBE,0x75,0x42,0xBE,0x8B,0x42,0x75,0x66,0xAA,
	 0x9A,0x56,0xAA,0x66,0x56,0x9A,0x70,0xB5,0x90,0x4B,0xB5,0x70,0x4B,0x90,0xBE,0xCF,
	 0x42,0x31,0xCF,0xBE,0x31,0x42,0xAB,0xD0,0x55,0x30,0xD0,0xAB,0x30,0x55,0xD1,0xD1,
	 0x2F,0x2F,0x9A,0xD3,0x66,0x2D,0xD3,0x9A,0x2D,0x66,0x7B,0xC5,0x85,0x3B,0xC5,0x7B,
	 0x3B,0x85,0x54,0xB4,0xAC,0x4C,0xB4,0x54,0x4C,0xAC,0x5E,0xBE,0xA2,0x42,0xBE,0x5E,
	 0x42,0xA2,0x87,0xD8,0x79,0x28,0xD8,0x87,0x28,0x79,0xC0,0xE8,0x40,0x18,0xE8,0xC0,
	 0x18,0x40,0xD5,0xE8,0x2B,0x18,0xE8,0xD5,0x18,0x2B,0xAB,0xE9,0x55,0x17,0xE9,0xAB,
	 0x17,0x55,0x68,0xCD,0x98,0x33,0xCD,0x68,0x33,0x98,0xEA,0xEA,0x16,0x16,0x80,0x80},
	{0x80,0x80,0x86,0x86,0x7A,0x7A,0x7E,0x88,0x82,0x78,0x88,0x7E,0x78,0x82,0x92,0x92,
	 0x6E,0x6E,0x85,0x92,0x7B,0x6E,0x92,0x85,0x6E,0x7B,0x73,0x8D,0x8D,0x73,0x8D,0x73,
	 0x73,0x8D,0x91,0x9E,0x6F,0x62,0x9E,0x91,0x62,0x6F,0x79,0x97,0x87,0x69,0x97,0x79,
	 0x69,0x87,0xA0,0xA0,0x60,0x60,0x83,0xA2,0x7D,0x5E,0xA2,0x83,0x5E,0x7D,0xA0,0xB0,
	 0x60,0x50,0xB0,0xA0,0x50,0x60,0xB1,0xB1,0x4F,0x4F,0x8F,0xB2,0x71,0x4E,0xB2,0x8F,
	 0x4E,0x71,0x6B,0x9E,0x95,0x62,0x9E,0x6B,0x62,0x95,0x74,0xA9,0x8C,0x57,0xA9,0x74,
	 0x57,0x8C,0x7F,0xB8,0x81,0x48,0xB8,0x7F,0x48,0x81,0xB4,0xC5,0x4C,0x3B,0xC5,0xB4,
	 0x3B,0x4C,0x9F,0xC6,0x61,0x3A,0xC6,0x9F,0x3A,0x61,0xC6,0xC6,0x3A,0x3A,0x59,0xA7,
	 0xA7,0x59,0xA7,0x59,0x59,0xA7,0x8D,0xCA,0x73,0x36,0xCA,0x8D,0x36,0x73,0x61,0xB2,
	 0x9F,0x4E,0xB2,0x61,0x4E,0x9F,0x6D,0xBF,0x93,0x41,0xBF,0x6D,0x41,0x93,0xCA,0xDF,
	 0x36,0x21,0xDF,0xCA,0x21,0x36,0xB3,0xDF,0x4D,0x21,0xDF,0xB3,0x21,0x4D,0xE1,0xE1,
	 0x1F,0x1F,0x9F,0xE3,0x61,0x1D,0xE3,0x9F,0x1D,0x61,0x7A,0xD3,0x86,0x2D,0xD3,0x7A,
	 0x2D,0x86,0x4C,0xBE,0xB4,0x42,0xBE,0x4C,0x42,0xB4,0x57,0xCA,0xA9,0x36,0xCA,0x57,
	 0x36,0xA9,0x88,0xE9,0x78,0x17,0xE9,0x88,0x17,0x78,0xCC,0xFB,0x34,0x05,0xFB,0xCC,
	 0x05,0x34,0xE6,0xFB,0x1A,0x05,0xFB,0xE6,0x05,0x1A,0xB4,0xFD,0x4C,0x03,0xFD,0xB4,
	 0x03,0x4C,0x63,0xDC,0x9D,0x24,0xDC,0x63,0x24,0x9D,0xFE,0xFE,0x02,0x02,0x80,0x80},
	{0x80,0x80,0x87,0x87,0x79,0x79,0x7E,0x89,0x82,0x77,0x89,0x7E,0x77,0x82,0x95,0x95,
	 0x6B,0x6B,0x86,0x96,0x7A,0x6A,0x96,0x86,0x6A,0x7A,0x70,0x90,0x90,0x70,0x90,0x70,
	 0x70,0x90,0x94,0xA4,0x6C,0x5C,0xA4,0x94,0x5C,0x6C,0x78,0x9B,0x88,0x65,0x9B,0x78,
	 0x65,0x88,0xA6,0xA6,0x5A,0x5A,0x83,0xA9,0x7D,0x57,0xA9,0x83,0x57,0x7D,0xA6,0xB9,
	 0x5A,0x47,0xB9,0xA6,0x47,0x5A,0xBA,0xBA,0x46,0x46,0x92,0xBC,0x6E,0x44,0xBC,0x92,
	 0x44,0x6E,0x67,0xA3,0x99,0x5D,0xA3,0x67,0x5D,0x99,0x72,0xB0,0x8E,0x50,0xB0,0x72,
	 0x50,0x8E,0x7F,0xC3,0x81,0x3D,0xC3,0x7F,0x3D,0x81,0xBE,0xD2,0x42,0x2E,0xD2,0xBE,
	 0x2E,0x42,0xA5,0xD4,0x5B,0x2C,0xD4,0xA5,0x2C,0x5B,0xD4,0xD4,0x2C,0x2C,0x52,0xAE,
	 0xAE,0x52,0xAE,0x52,0x52,0xAE,0x8F,0xD8,0x71,0x28,0xD8,0x8F,0x28,0x71,0x5B,0xBB,
	 0xA5,0x45,0xBB,0x5B,0x45,0xA5,0x69,0xCB,0x97,0x35,0xCB,0x69,0x35,0x97,0xD8,0xF0,
	 0x28,0x10,0xF0,0xD8,0x10,0x28,0xBD,0xF1,0x43,0x0F,0xF1,0xBD,0x0F,0x43,0xF3,0xF3,
	 0x0D,0x0D,0xA5,0xF6,0x5B,0x0A,0xF6,0xA5,0x0A,0x5B,0x78,0xE2,0x88,0x1E,0xE2,0x78,
	 0x1E,0x88,0x42,0xC9,0xBE,0x37,0xC9,0x42,0x37,0xBE,0x4F,0xD8,0xB1,0x28,0xD8,0x4F,
	 0x28,0xB1,0x8A,0xFD,0x76,0x03,0xFD,0x8A,0x03,0x76,0xDB,0xFF,0x25,0x01,0xFF,0xDB,
	 0x01,0x25,0xF9,0xFF,0x07,0x01,0xFF,0xF9,0x01,0x07,0xBE,0xFF,0x42,0x01,0xFF,0xBE,
	 0x01,0x42,0x5E,0xED,0xA2,0x13,0xED,0x5E,0x13,0xA2,0xFF,0xFF,0x01,0x01,0x80,0x80},
};

// VLC: 1<<14 peek-LUT of (symbol<<4)|length, built once from ir2_tab. Mirrors the
// JS buildIR2Lookup (== FFmpeg's VLC_INIT_STATIC_TABLE_FROM_LENGTHS, LE output):
// canonical code assignment by ascending length, bit-reversed for an LE peek.
static const int CODE_VLC_BITS = 14;
static uint16_t ir2_lut[1 << CODE_VLC_BITS];
static int ir2_lut_built = 0;

static void build_lut() {
	if (ir2_lut_built) return;
	memset(ir2_lut, 0, sizeof(ir2_lut));
	int code = 0, prevLen = ir2_tab[0][1];
	for (int i = 0; i < IR2_CODES; i++) {
		int sym = ir2_tab[i][0], len = ir2_tab[i][1];
		code <<= (len - prevLen);
		int revCode = 0;
		for (int b = 0; b < len; b++)
			if ((code >> (len - 1 - b)) & 1) revCode |= 1 << b;
		uint16_t value = (uint16_t)((sym << 4) | len);
		for (int high = revCode; high < (1 << CODE_VLC_BITS); high += (1 << len))
			ir2_lut[high] = value;
		code++;
		prevLen = len;
	}
	ir2_lut_built = 1;
}

// LSB-first bit reader (FFmpeg BITSTREAM_READER_LE).
struct BitLE {
	const uint8_t* data;
	int bitOff;
	int bitEnd;
};
static inline int peek14(BitLE* r) {
	int byteOff = r->bitOff >> 3, bit = r->bitOff & 7, len = r->bitEnd >> 3;
	int b0 = byteOff < len ? r->data[byteOff] : 0;
	int b1 = byteOff + 1 < len ? r->data[byteOff + 1] : 0;
	int b2 = byteOff + 2 < len ? r->data[byteOff + 2] : 0;
	return ((b0 | (b1 << 8) | (b2 << 16)) >> bit) & 0x3FFF;
}
static inline int get_code(BitLE* r) {
	int e = ir2_lut[peek14(r)];
	if (e == 0) return -1;
	r->bitOff += e & 0x0F;
	return e >> 4;
}
static inline int bits_left(BitLE* r) { return r->bitEnd - r->bitOff; }

static int decode_plane(BitLE* r, int width, int height, uint8_t* dst, int pitch, const uint8_t* table) {
	if ((width & 1) || width * height / (2 * (IR2_CODES - 0x7F)) > bits_left(r))
		return AVERROR_INVALIDDATA;
	int out = 0;
	// first row: absolute values
	while (out < width) {
		int c = get_code(r);
		if (c >= 0x80) {
			c -= 0x7F;
			if (out + c * 2 > width) return AVERROR_INVALIDDATA;
			for (int i = 0; i < c * 2; i++) dst[out++] = 0x80;
		} else {
			if (c <= 0) return AVERROR_INVALIDDATA;
			dst[out++] = table[c * 2];
			dst[out++] = table[c * 2 + 1];
		}
	}
	dst += pitch;
	for (int j = 1; j < height; j++) {
		out = 0;
		while (out < width) {
			if (bits_left(r) <= 0) return AVERROR_INVALIDDATA;
			int c = get_code(r);
			if (c >= 0x80) {
				c -= 0x7F;
				if (out + c * 2 > width) return AVERROR_INVALIDDATA;
				for (int i = 0; i < c * 2; i++) { dst[out] = dst[out - pitch]; out++; }
			} else {
				if (c <= 0) return AVERROR_INVALIDDATA;
				dst[out] = av_clip_uint8(dst[out - pitch] + (table[c * 2] - 128)); out++;
				dst[out] = av_clip_uint8(dst[out - pitch] + (table[c * 2 + 1] - 128)); out++;
			}
		}
		dst += pitch;
	}
	return 0;
}

static int decode_plane_inter(BitLE* r, int width, int height, uint8_t* dst, int pitch, const uint8_t* table) {
	if (width & 1) return AVERROR_INVALIDDATA;
	for (int j = 0; j < height; j++) {
		int out = 0;
		while (out < width) {
			if (bits_left(r) <= 0) return AVERROR_INVALIDDATA;
			int c = get_code(r);
			if (c >= 0x80) {
				out += (c - 0x7F) * 2;
				if (out > width) return AVERROR_INVALIDDATA;
			} else {
				if (c <= 0) return AVERROR_INVALIDDATA;
				dst[out] = av_clip_uint8(dst[out] + (((table[c * 2] - 128) * 3) >> 2)); out++;
				dst[out] = av_clip_uint8(dst[out] + (((table[c * 2 + 1] - 128) * 3) >> 2)); out++;
			}
		}
		dst += pitch;
	}
	return 0;
}

// returns 0 on success. Y -> frame (plane0); the bitstream order is Y, V, U
// (chroma swapped), so V -> plane2, U -> plane1.
static int decode(RvCtx* s, const uint8_t* buf, int buf_size) {
	const int start = 48;   // header is the first 48 bytes
	if (buf_size < 0x24 || start >= buf_size) return AVERROR_INVALIDDATA;

	int decode_delta = buf[18];
	int ltab = buf[0x22] & 3;
	int ctab = buf[0x22] >> 2;
	if (ctab > 3) return AVERROR_INVALIDDATA;

	BitLE r = { buf + start, 0, (buf_size - start) * 8 };
	const int w = s->width, h = s->height, p = s->stride;
	const int cw = s->width >> 2, ch = s->height >> 2, cp = s->chromaW;
	int ret;

	if (decode_delta) {   // intra
		if ((ret = decode_plane(&r, w, h, s->frame, p, ir2_delta_table[ltab])) < 0) return ret;
		if ((ret = decode_plane(&r, cw, ch, s->plane2, cp, ir2_delta_table[ctab])) < 0) return ret;
		if ((ret = decode_plane(&r, cw, ch, s->plane1, cp, ir2_delta_table[ctab])) < 0) return ret;
	} else {              // inter
		if ((ret = decode_plane_inter(&r, w, h, s->frame, p, ir2_delta_table[ltab])) < 0) return ret;
		if ((ret = decode_plane_inter(&r, cw, ch, s->plane2, cp, ir2_delta_table[ctab])) < 0) return ret;
		if ((ret = decode_plane_inter(&r, cw, ch, s->plane1, cp, ir2_delta_table[ctab])) < 0) return ret;
	}
	return 0;
}

} // namespace iv2

// ════════════════════════════════════════════════════════════════════════════
//  Intel Indeo 5 (IV50) — port of FFmpeg libavcodec/indeo5.c (frontend hooks).
//  The shared band/tile/MB decode loop lives in ivi_core.h (namespace ivi);
//  this installs the five decode hooks + wires per-band transform/quant tables.
// ════════════════════════════════════════════════════════════════════════════
namespace iv5 {

using namespace ivi;

enum { FRAMETYPE_INTRA = 0, FRAMETYPE_INTER = 1, FRAMETYPE_INTER_SCAL = 2,
       FRAMETYPE_INTER_NOREF = 3, FRAMETYPE_NULL = 4 };
enum { IVI5_PIC_SIZE_ESC = 15 };
enum { IVI_MB_HUFF = 0, IVI_BLK_HUFF = 1 };

// ff_zigzag_direct now lives in ivi_tables.h (shared with Indeo 4); reachable
// here as ivi::ff_zigzag_direct via `using namespace ivi` above.

static int pic_conf_changed(const IVIPicConfig* a, const IVIPicConfig* b) {
	return a->pic_width != b->pic_width || a->pic_height != b->pic_height ||
	       a->chroma_width != b->chroma_width || a->chroma_height != b->chroma_height ||
	       a->tile_width != b->tile_width || a->tile_height != b->tile_height ||
	       a->luma_bands != b->luma_bands || a->chroma_bands != b->chroma_bands;
}

static int skip_hdr_extension(GetBits* gb) {
	int i, len;
	do {
		len = get_bits(gb, 8);
		if (8 * len > get_bits_left(gb)) return AVERROR_INVALIDDATA;
		for (i = 0; i < len; i++) skip_bits(gb, 8);
	} while (len);
	return 0;
}

static int decode_gop_header(Ctx* ctx) {
	GetBits* gb = &ctx->gb;
	int result, i, p, tile_size, pic_size_indx, mb_size, blk_size, is_scalable;
	int quant_mat, blk_size_changed = 0;
	IVIBandDesc *band, *band1, *band2;
	IVIPicConfig pic_conf;

	ctx->gop_flags    = get_bits(gb, 8);
	ctx->gop_hdr_size = (ctx->gop_flags & 1) ? get_bits(gb, 16) : 0;
	if (ctx->gop_flags & IVI5_IS_PROTECTED) ctx->lock_word = get_bits(gb, 32);

	tile_size = (ctx->gop_flags & 0x40) ? 64 << get_bits(gb, 2) : 0;
	if (tile_size > 256) return AVERROR_INVALIDDATA;

	pic_conf.luma_bands   = get_bits(gb, 2) * 3 + 1;
	pic_conf.chroma_bands = get_bits1(gb) * 3 + 1;
	is_scalable = pic_conf.luma_bands != 1 || pic_conf.chroma_bands != 1;
	if (is_scalable && (pic_conf.luma_bands != 4 || pic_conf.chroma_bands != 1))
		return AVERROR_INVALIDDATA;

	pic_size_indx = get_bits(gb, 4);
	if (pic_size_indx == IVI5_PIC_SIZE_ESC) {
		pic_conf.pic_height = get_bits(gb, 13);
		pic_conf.pic_width  = get_bits(gb, 13);
	} else {
		pic_conf.pic_height = ivi5_common_pic_sizes[pic_size_indx * 2 + 1] << 2;
		pic_conf.pic_width  = ivi5_common_pic_sizes[pic_size_indx * 2    ] << 2;
	}

	if (ctx->gop_flags & 2) return AVERROR_PATCHWELCOME;   // YV12

	pic_conf.chroma_height = (pic_conf.pic_height + 3) >> 2;
	pic_conf.chroma_width  = (pic_conf.pic_width  + 3) >> 2;
	if (!tile_size) {
		pic_conf.tile_height = pic_conf.pic_height;
		pic_conf.tile_width  = pic_conf.pic_width;
	} else {
		pic_conf.tile_height = pic_conf.tile_width = tile_size;
	}

	if (pic_conf_changed(&pic_conf, &ctx->pic_conf) || ctx->gop_invalid) {
		result = init_planes(ctx, &pic_conf);
		if (result < 0) return result;
		ctx->pic_conf = pic_conf;
		ctx->is_scalable = is_scalable;
		blk_size_changed = 1;
	}

	for (p = 0; p <= 1; p++) {
		for (i = 0; i < (!p ? pic_conf.luma_bands : pic_conf.chroma_bands); i++) {
			band = &ctx->planes[p].bands[i];
			band->is_halfpel = get_bits1(gb);
			mb_size  = get_bits1(gb);
			blk_size = 8 >> get_bits1(gb);
			mb_size  = blk_size << !mb_size;

			if (p == 0 && blk_size == 4) return AVERROR_PATCHWELCOME;

			blk_size_changed = mb_size != band->mb_size || blk_size != band->blk_size;
			if (blk_size_changed) { band->mb_size = mb_size; band->blk_size = blk_size; }

			if (get_bits1(gb)) return AVERROR_PATCHWELCOME;   // extended transform info

			switch ((p << 2) + i) {
			case 0:
				band->inv_transform  = iv45::ff_ivi_inverse_slant_8x8;
				band->dc_transform   = iv45::ff_ivi_dc_slant_2d;
				band->scan           = ff_zigzag_direct;
				band->transform_size = 8;
				break;
			case 1:
				band->inv_transform  = iv45::ff_ivi_row_slant8;
				band->dc_transform   = iv45::ff_ivi_dc_row_slant;
				band->scan           = ff_ivi_vertical_scan_8x8;
				band->transform_size = 8;
				break;
			case 2:
				band->inv_transform  = iv45::ff_ivi_col_slant8;
				band->dc_transform   = iv45::ff_ivi_dc_col_slant;
				band->scan           = ff_ivi_horizontal_scan_8x8;
				band->transform_size = 8;
				break;
			case 3:
				band->inv_transform  = iv45::ff_ivi_put_pixels_8x8;
				band->dc_transform   = iv45::ff_ivi_put_dc_pixel_8x8;
				band->scan           = ff_ivi_horizontal_scan_8x8;
				band->transform_size = 8;
				break;
			case 4:
				band->inv_transform  = iv45::ff_ivi_inverse_slant_4x4;
				band->dc_transform   = iv45::ff_ivi_dc_slant_2d;
				band->scan           = ff_ivi_direct_scan_4x4;
				band->transform_size = 4;
				break;
			}

			band->is_2d_trans = band->inv_transform == iv45::ff_ivi_inverse_slant_8x8 ||
			                    band->inv_transform == iv45::ff_ivi_inverse_slant_4x4;
			if (band->transform_size != band->blk_size) return AVERROR_INVALIDDATA;

			quant_mat = !p ? ((pic_conf.luma_bands > 1) ? i + 1 : 0) : 5;
			if (band->blk_size == 8) {
				if (quant_mat >= 5) return AVERROR_INVALIDDATA;
				band->intra_base  = &ivi5_base_quant_8x8_intra[quant_mat][0];
				band->inter_base  = &ivi5_base_quant_8x8_inter[quant_mat][0];
				band->intra_scale = &ivi5_scale_quant_8x8_intra[quant_mat][0];
				band->inter_scale = &ivi5_scale_quant_8x8_inter[quant_mat][0];
			} else {
				band->intra_base  = ivi5_base_quant_4x4_intra;
				band->inter_base  = ivi5_base_quant_4x4_inter;
				band->intra_scale = ivi5_scale_quant_4x4_intra;
				band->inter_scale = ivi5_scale_quant_4x4_inter;
			}
			band->quant_mat = quant_mat;

			if (get_bits(gb, 2)) return AVERROR_INVALIDDATA;   // end marker
		}
	}

	for (i = 0; i < pic_conf.chroma_bands; i++) {
		band1 = &ctx->planes[1].bands[i];
		band2 = &ctx->planes[2].bands[i];
		band2->width          = band1->width;
		band2->height         = band1->height;
		band2->mb_size        = band1->mb_size;
		band2->blk_size       = band1->blk_size;
		band2->is_halfpel     = band1->is_halfpel;
		band2->intra_base     = band1->intra_base;
		band2->inter_base     = band1->inter_base;
		band2->intra_scale    = band1->intra_scale;
		band2->inter_scale    = band1->inter_scale;
		band2->scan           = band1->scan;
		band2->inv_transform  = band1->inv_transform;
		band2->dc_transform   = band1->dc_transform;
		band2->is_2d_trans    = band1->is_2d_trans;
		band2->transform_size = band1->transform_size;
		band2->quant_mat      = band1->quant_mat;
	}

	if (blk_size_changed) {
		result = init_tiles(ctx, pic_conf.tile_width, pic_conf.tile_height);
		if (result < 0) return result;
	}

	if (ctx->gop_flags & 8) {
		if (get_bits(gb, 3)) return AVERROR_INVALIDDATA;
		if (get_bits1(gb)) skip_bits(gb, 24);   // transparency fill color
	}

	align_get_bits(gb);
	skip_bits(gb, 23);   // FIXME: unknown meaning

	if (get_bits1(gb)) {   // skip GOP extension
		do { i = get_bits(gb, 16); } while (i & 0x8000);
	}

	align_get_bits(gb);
	return 0;
}

static int decode_pic_hdr(Ctx* ctx) {
	GetBits* gb = &ctx->gb;
	int ret;

	if (get_bits(gb, 5) != 0x1F) return AVERROR_INVALIDDATA;

	ctx->prev_frame_type = ctx->frame_type;
	ctx->frame_type      = get_bits(gb, 3);
	if (ctx->frame_type >= 5) { ctx->frame_type = FRAMETYPE_INTRA; return AVERROR_INVALIDDATA; }

	ctx->frame_num = get_bits(gb, 8);

	if (ctx->frame_type == FRAMETYPE_INTRA) {
		if ((ret = decode_gop_header(ctx)) < 0) { ctx->gop_invalid = 1; return ret; }
		ctx->gop_invalid = 0;
	}

	if (ctx->frame_type == FRAMETYPE_INTER_SCAL && !ctx->is_scalable) {
		ctx->frame_type = FRAMETYPE_INTER;
		return AVERROR_INVALIDDATA;
	}

	if (ctx->frame_type != FRAMETYPE_NULL) {
		ctx->frame_flags  = get_bits(gb, 8);
		ctx->pic_hdr_size = (ctx->frame_flags & 1) ? get_bits(gb, 24) : 0;
		ctx->checksum     = (ctx->frame_flags & 0x10) ? get_bits(gb, 16) : 0;
		if (ctx->frame_flags & 0x20) skip_hdr_extension(gb);
		ret = dec_huff_desc(ctx, ctx->frame_flags & 0x40, IVI_MB_HUFF, &ctx->mb_vlc);
		if (ret < 0) return ret;
		skip_bits(gb, 3);   // FIXME: unknown meaning
	}

	align_get_bits(gb);
	return 0;
}

static int decode_band_hdr(Ctx* ctx, IVIBandDesc* band) {
	GetBits* gb = &ctx->gb;
	int i, ret;
	uint8_t band_flags = get_bits(gb, 8);

	if (band_flags & 1) { band->is_empty = 1; return 0; }

	band->data_size = (ctx->frame_flags & 0x80) ? get_bits(gb, 24) : 0;
	band->inherit_mv     = band_flags & 2;
	band->inherit_qdelta = band_flags & 8;
	band->qdelta_present = band_flags & 4;
	if (!band->qdelta_present) band->inherit_qdelta = 1;

	band->num_corr = 0;
	if (band_flags & 0x10) {
		band->num_corr = get_bits(gb, 8);
		if (band->num_corr > 61) return AVERROR_INVALIDDATA;
		for (i = 0; i < band->num_corr * 2; i++) band->corr[i] = get_bits(gb, 8);
	}

	band->rvmap_sel = (band_flags & 0x40) ? get_bits(gb, 3) : 8;

	ret = dec_huff_desc(ctx, band_flags & 0x80, IVI_BLK_HUFF, &band->blk_vlc);
	if (ret < 0) return ret;

	band->checksum_present = get_bits1(gb);
	if (band->checksum_present) band->checksum = get_bits(gb, 16);

	band->glob_quant = get_bits(gb, 5);

	if (band_flags & 0x20) { align_get_bits(gb); skip_hdr_extension(gb); }

	align_get_bits(gb);
	return 0;
}

static int decode_mb_info(Ctx* ctx, IVIBandDesc* band, IVITile* tile) {
	GetBits* gb = &ctx->gb;
	int x, y, mv_x, mv_y, mv_delta, offs, mb_offset, mv_scale, blks_per_mb, s;
	IVIMbInfo *mb, *ref_mb;
	int row_offset = band->mb_size * band->pitch;

	mb     = tile->mbs;
	ref_mb = tile->ref_mbs;
	offs   = tile->ypos * band->pitch + tile->xpos;

	if (!ref_mb && ((band->qdelta_present && band->inherit_qdelta) || band->inherit_mv))
		return AVERROR_INVALIDDATA;

	if (tile->num_MBs != num_mbs_per_tile(tile->width, tile->height, band->mb_size))
		return AVERROR_INVALIDDATA;

	mv_scale = (ctx->planes[0].bands[0].mb_size >> 3) - (band->mb_size >> 3);
	mv_x = mv_y = 0;

	for (y = tile->ypos; y < tile->ypos + tile->height; y += band->mb_size) {
		mb_offset = offs;
		for (x = tile->xpos; x < tile->xpos + tile->width; x += band->mb_size) {
			mb->xpos = x; mb->ypos = y; mb->buf_offs = mb_offset;

			if (get_bits1(gb)) {
				if (ctx->frame_type == FRAMETYPE_INTRA) return AVERROR_INVALIDDATA;
				mb->type = 1; mb->cbp = 0; mb->q_delta = 0;
				if (!band->plane && !band->band_num && (ctx->frame_flags & 8)) {
					mb->q_delta = get_vlc(gb, ctx->mb_vlc.tab);
					mb->q_delta = to_signed(mb->q_delta);
				}
				mb->mv_x = mb->mv_y = 0;
				if (band->inherit_mv && ref_mb) {
					if (mv_scale) {
						mb->mv_x = scale_mv(ref_mb->mv_x, mv_scale);
						mb->mv_y = scale_mv(ref_mb->mv_y, mv_scale);
					} else {
						mb->mv_x = ref_mb->mv_x;
						mb->mv_y = ref_mb->mv_y;
					}
				}
			} else {
				if (band->inherit_mv && ref_mb) {
					mb->type = ref_mb->type;
				} else if (ctx->frame_type == FRAMETYPE_INTRA) {
					mb->type = 0;
				} else {
					mb->type = get_bits1(gb);
				}

				blks_per_mb = band->mb_size != band->blk_size ? 4 : 1;
				mb->cbp = get_bits(gb, blks_per_mb);

				mb->q_delta = 0;
				if (band->qdelta_present) {
					if (band->inherit_qdelta) {
						if (ref_mb) mb->q_delta = ref_mb->q_delta;
					} else if (mb->cbp || (!band->plane && !band->band_num && (ctx->frame_flags & 8))) {
						mb->q_delta = get_vlc(gb, ctx->mb_vlc.tab);
						mb->q_delta = to_signed(mb->q_delta);
					}
				}

				if (!mb->type) {
					mb->mv_x = mb->mv_y = 0;
				} else if (band->inherit_mv && ref_mb) {
					if (mv_scale) {
						mb->mv_x = scale_mv(ref_mb->mv_x, mv_scale);
						mb->mv_y = scale_mv(ref_mb->mv_y, mv_scale);
					} else {
						mb->mv_x = ref_mb->mv_x;
						mb->mv_y = ref_mb->mv_y;
					}
				} else {
					mv_delta = get_vlc(gb, ctx->mb_vlc.tab);
					mv_y += to_signed(mv_delta);
					mv_delta = get_vlc(gb, ctx->mb_vlc.tab);
					mv_x += to_signed(mv_delta);
					mb->mv_x = mv_x;
					mb->mv_y = mv_y;
				}
			}

			s = band->is_halfpel;
			if (mb->type)
			if (x + (mb->mv_x >> s) + (y + (mb->mv_y >> s)) * band->pitch < 0 ||
			    x + ((mb->mv_x + s) >> s) + band->mb_size - 1
			      + (y + band->mb_size - 1 + ((mb->mv_y + s) >> s)) * band->pitch > band->bufsize - 1)
				return AVERROR_INVALIDDATA;

			mb++;
			if (ref_mb) ref_mb++;
			mb_offset += band->mb_size;
		}
		offs += row_offset;
	}

	align_get_bits(gb);
	return 0;
}

static void switch_buffers(Ctx* ctx) {
	switch (ctx->prev_frame_type) {
	case FRAMETYPE_INTRA:
	case FRAMETYPE_INTER:
		ctx->buf_switch ^= 1;
		ctx->dst_buf = ctx->buf_switch;
		ctx->ref_buf = ctx->buf_switch ^ 1;
		break;
	case FRAMETYPE_INTER_SCAL:
		if (!ctx->inter_scal) { ctx->ref2_buf = 2; ctx->inter_scal = 1; }
		{ int t = ctx->dst_buf; ctx->dst_buf = ctx->ref2_buf; ctx->ref2_buf = t; }
		ctx->ref_buf = ctx->ref2_buf;
		break;
	case FRAMETYPE_INTER_NOREF:
	default:
		break;
	}

	switch (ctx->frame_type) {
	case FRAMETYPE_INTRA:
		ctx->buf_switch = 0;
		/* FALLTHROUGH */
	case FRAMETYPE_INTER:
		ctx->inter_scal = 0;
		ctx->dst_buf = ctx->buf_switch;
		ctx->ref_buf = ctx->buf_switch ^ 1;
		break;
	case FRAMETYPE_INTER_SCAL:
	case FRAMETYPE_INTER_NOREF:
	case FRAMETYPE_NULL:
	default:
		break;
	}
}

static int is_nonnull_frame(Ctx* ctx) { return ctx->frame_type != FRAMETYPE_NULL; }

// Create + initialize an Indeo 5 context for `width`x`height`.
static Ctx* open(int width, int height) {
	Ctx* ctx = ctx_alloc();
	if (!ctx) return nullptr;
	ctx->is_indeo4        = 0;
	ctx->decode_pic_hdr   = decode_pic_hdr;
	ctx->decode_band_hdr  = decode_band_hdr;
	ctx->decode_mb_info   = decode_mb_info;
	ctx->switch_buffers   = switch_buffers;
	ctx->is_nonnull_frame = is_nonnull_frame;

	ctx->pic_conf.pic_width     = width;
	ctx->pic_conf.pic_height    = height;
	ctx->pic_conf.chroma_width  = (width  + 3) >> 2;
	ctx->pic_conf.chroma_height = (height + 3) >> 2;
	ctx->pic_conf.tile_width    = width;
	ctx->pic_conf.tile_height   = height;
	ctx->pic_conf.luma_bands    = ctx->pic_conf.chroma_bands = 1;

	if (width > 0 && height > 0) {
		if (init_planes(ctx, &ctx->pic_conf) < 0) { ctx_free(ctx); return nullptr; }
	}
	return ctx;
}

} // namespace iv5

// ════════════════════════════════════════════════════════════════════════════
//  Intel Indeo 4 (IV41) — port of FFmpeg libavcodec/indeo4.c (frontend hooks).
//  Shares the ivi_core.h decode loop. Deltas vs Indeo5: bitstream-selected
//  transform (5-bit transform_id) + scan (4-bit scan_indx) per band, its own
//  pic-size/scan/quant tables, B-frames (4-value MB types + 4-buffer rotation),
//  no scale tables. The trailing-P "dual frame per packet" reorder is NOT
//  implemented (the JS oracle indeo4.js doesn't either — each chunk is one
//  frame, NULL_LAST repeats the previous), so this stays bit-exact with the JS.
// ════════════════════════════════════════════════════════════════════════════
namespace iv4 {

using namespace ivi;

enum { FRAMETYPE_INTRA = 0, FRAMETYPE_INTRA1 = 1, FRAMETYPE_INTER = 2,
       FRAMETYPE_BIDIR = 3, FRAMETYPE_INTER_NOREF = 4,
       FRAMETYPE_NULL_FIRST = 5, FRAMETYPE_NULL_LAST = 6 };
enum { IVI4_PIC_SIZE_ESC = 7 };
enum { IVI_MB_HUFF = 0, IVI_BLK_HUFF = 1 };

// transform_id -> (inv_transform, dc_transform, is_2d_trans). Mirrors ffmpeg's
// transforms[18]; NULL inv_trans entries are unsupported (DCT / no-transform).
struct TransformEntry { InvTransformPtr inv; DCTransformPtr dc; int is_2d; };
static const TransformEntry transforms[18] = {
	{ iv45::ff_ivi_inverse_haar_8x8,  iv45::ff_ivi_dc_haar_2d,       1 },
	{ iv45::ff_ivi_row_haar8,         iv45::ff_ivi_dc_haar_2d,       0 },
	{ iv45::ff_ivi_col_haar8,         iv45::ff_ivi_dc_haar_2d,       0 },
	{ iv45::ff_ivi_put_pixels_8x8,    iv45::ff_ivi_put_dc_pixel_8x8, 1 },
	{ iv45::ff_ivi_inverse_slant_8x8, iv45::ff_ivi_dc_slant_2d,      1 },
	{ iv45::ff_ivi_row_slant8,        iv45::ff_ivi_dc_row_slant,     1 },
	{ iv45::ff_ivi_col_slant8,        iv45::ff_ivi_dc_col_slant,     1 },
	{ nullptr, nullptr, 0 }, // inverse DCT 8x8
	{ nullptr, nullptr, 0 }, // inverse DCT 8x1
	{ nullptr, nullptr, 0 }, // inverse DCT 1x8
	{ iv45::ff_ivi_inverse_haar_4x4,  iv45::ff_ivi_dc_haar_2d,       1 },
	{ iv45::ff_ivi_inverse_slant_4x4, iv45::ff_ivi_dc_slant_2d,      1 },
	{ nullptr, nullptr, 0 }, // no transform 4x4
	{ iv45::ff_ivi_row_haar4,         iv45::ff_ivi_dc_haar_2d,       0 },
	{ iv45::ff_ivi_col_haar4,         iv45::ff_ivi_dc_haar_2d,       0 },
	{ iv45::ff_ivi_row_slant4,        iv45::ff_ivi_dc_row_slant,     0 },
	{ iv45::ff_ivi_col_slant4,        iv45::ff_ivi_dc_col_slant,     0 },
	{ nullptr, nullptr, 0 }, // inverse DCT 4x4
};

static int pic_conf_changed(const IVIPicConfig* a, const IVIPicConfig* b) {
	return a->pic_width != b->pic_width || a->pic_height != b->pic_height ||
	       a->tile_width != b->tile_width || a->tile_height != b->tile_height ||
	       a->luma_bands != b->luma_bands || a->chroma_bands != b->chroma_bands;
}

static int decode_plane_subdivision(GetBits* gb) {
	switch (get_bits(gb, 2)) {
	case 3:
		return 1;
	case 2:
		for (int i = 0; i < 4; i++)
			if (get_bits(gb, 2) != 3) return 0;
		return 4;
	default:
		return 0;
	}
}

static inline int scale_tile_size(int def_size, int size_factor) {
	return size_factor == 15 ? def_size : (size_factor + 1) << 5;
}

static int decode_pic_hdr(Ctx* ctx) {
	GetBits* gb = &ctx->gb;
	int pic_size_indx, i, p;
	IVIPicConfig pic_conf;

	if (get_bits(gb, 18) != 0x3FFF8) return AVERROR_INVALIDDATA;

	ctx->gop_invalid = 0;   // Indeo 4 has no GOP-validity flag

	ctx->prev_frame_type = ctx->frame_type;
	ctx->frame_type      = get_bits(gb, 3);
	if (ctx->frame_type == 7) return AVERROR_INVALIDDATA;

	if (ctx->frame_type == FRAMETYPE_BIDIR) ctx->has_b_frames = 1;

	ctx->has_transp = get_bits1(gb);
	if (get_bits1(gb)) return AVERROR_INVALIDDATA;   // sync bit must be 0

	ctx->data_size = get_bits1(gb) ? get_bits(gb, 24) : 0;

	if (ctx->frame_type >= FRAMETYPE_NULL_FIRST) return 0;   // null frame: no payload

	if (get_bits1(gb)) skip_bits(gb, 32);   // optional lock word (ignored)

	pic_size_indx = get_bits(gb, 3);
	if (pic_size_indx == IVI4_PIC_SIZE_ESC) {
		pic_conf.pic_height = get_bits(gb, 16);
		pic_conf.pic_width  = get_bits(gb, 16);
	} else {
		pic_conf.pic_height = ivi4_common_pic_sizes[pic_size_indx * 2 + 1];
		pic_conf.pic_width  = ivi4_common_pic_sizes[pic_size_indx * 2    ];
	}

	ctx->uses_tiling = get_bits1(gb);
	if (ctx->uses_tiling) {
		pic_conf.tile_height = scale_tile_size(pic_conf.pic_height, get_bits(gb, 4));
		pic_conf.tile_width  = scale_tile_size(pic_conf.pic_width,  get_bits(gb, 4));
	} else {
		pic_conf.tile_height = pic_conf.pic_height;
		pic_conf.tile_width  = pic_conf.pic_width;
	}

	if (get_bits(gb, 2)) return AVERROR_INVALIDDATA;   // only YVU9
	pic_conf.chroma_height = (pic_conf.pic_height + 3) >> 2;
	pic_conf.chroma_width  = (pic_conf.pic_width  + 3) >> 2;

	pic_conf.luma_bands = decode_plane_subdivision(gb);
	pic_conf.chroma_bands = 0;
	if (pic_conf.luma_bands)
		pic_conf.chroma_bands = decode_plane_subdivision(gb);
	if (!pic_conf.luma_bands) return AVERROR_INVALIDDATA;

	ctx->is_scalable = pic_conf.luma_bands != 1 || pic_conf.chroma_bands != 1;
	if (ctx->is_scalable && (pic_conf.luma_bands != 4 || pic_conf.chroma_bands != 1))
		return AVERROR_INVALIDDATA;

	if (pic_conf_changed(&pic_conf, &ctx->pic_conf)) {
		if (init_planes(ctx, &pic_conf) < 0) {
			ctx->pic_conf.luma_bands = 0;
			return AVERROR_INVALIDDATA;
		}
		ctx->pic_conf = pic_conf;

		for (p = 0; p <= 2; p++) {
			for (i = 0; i < (!p ? pic_conf.luma_bands : pic_conf.chroma_bands); i++) {
				ctx->planes[p].bands[i].mb_size  = !p ? (!ctx->is_scalable ? 16 : 8) : 4;
				ctx->planes[p].bands[i].blk_size = !p ? 8 : 4;
			}
		}
		if (init_tiles(ctx, pic_conf.tile_width, pic_conf.tile_height) < 0)
			return AVERROR_INVALIDDATA;
	}

	ctx->frame_num = get_bits1(gb) ? get_bits(gb, 20) : 0;

	if (get_bits1(gb)) skip_bits(gb, 8);   // skip decTimeEst

	if (dec_huff_desc(ctx, get_bits1(gb), IVI_MB_HUFF,  &ctx->mb_vlc)  ||
	    dec_huff_desc(ctx, get_bits1(gb), IVI_BLK_HUFF, &ctx->blk_vlc))
		return AVERROR_INVALIDDATA;

	ctx->rvmap_sel = get_bits1(gb) ? get_bits(gb, 3) : 8;

	ctx->in_imf = get_bits1(gb);
	ctx->in_q   = get_bits1(gb);
	ctx->pic_glob_quant = get_bits(gb, 5);
	ctx->unknown1 = get_bits1(gb) ? get_bits(gb, 3) : 0;
	ctx->checksum = get_bits1(gb) ? get_bits(gb, 16) : 0;

	while (get_bits1(gb)) {   // skip picture-header extensions
		if (get_bits_left(gb) < 10) return AVERROR_INVALIDDATA;
		skip_bits(gb, 8);
	}
	if (get_bits1(gb)) { /* "bad blocks" bit — ffmpeg only logs */ }

	align_get_bits(gb);
	return 0;
}

static int decode_band_hdr(Ctx* ctx, IVIBandDesc* band) {
	GetBits* gb = &ctx->gb;
	int plane = get_bits(gb, 2);
	int band_num = get_bits(gb, 4);
	if (band->plane != plane || band->band_num != band_num) return AVERROR_INVALIDDATA;

	band->is_empty = get_bits1(gb);
	if (band->is_empty) return 0;

	int old_blk_size = band->blk_size;

	if (get_bits1(gb)) skip_bits(gb, 16);   // optional header-size field (unused)

	band->is_halfpel = get_bits(gb, 2);
	if (band->is_halfpel >= 2) return AVERROR_INVALIDDATA;
	if (!band->is_halfpel) ctx->uses_fullpel = 1;

	band->checksum_present = get_bits1(gb);
	if (band->checksum_present) band->checksum = get_bits(gb, 16);

	int indx = get_bits(gb, 2);
	if (indx == 3) return AVERROR_INVALIDDATA;
	band->mb_size  = 16 >> indx;
	band->blk_size = 8 >> (indx >> 1);

	band->inherit_mv     = get_bits1(gb);
	band->inherit_qdelta = get_bits1(gb);
	band->glob_quant     = get_bits(gb, 5);

	int has_transform = !get_bits1(gb) || ctx->frame_type == FRAMETYPE_INTRA;
	if (has_transform) {
		int transform_id = get_bits(gb, 5);
		if (transform_id >= 18 || !transforms[transform_id].inv) return AVERROR_PATCHWELCOME;
		if ((transform_id >= 7 && transform_id <= 9) || transform_id == 17) return AVERROR_PATCHWELCOME;
		if (transform_id < 10 && band->blk_size < 8) return AVERROR_INVALIDDATA;
		if ((transform_id >= 0 && transform_id <= 2) || transform_id == 10) ctx->uses_haar = 1;

		band->inv_transform  = transforms[transform_id].inv;
		band->dc_transform   = transforms[transform_id].dc;
		band->is_2d_trans    = transforms[transform_id].is_2d;
		band->transform_size = transform_id < 10 ? 8 : 4;
		if (band->blk_size != band->transform_size) return AVERROR_INVALIDDATA;

		int scan_indx = get_bits(gb, 4);
		if (scan_indx == 15) return AVERROR_INVALIDDATA;
		if (scan_indx > 4 && scan_indx < 10) {
			if (band->blk_size != 4) return AVERROR_INVALIDDATA;
		} else if (band->blk_size != 8) {
			return AVERROR_INVALIDDATA;
		}
		band->scan = scan_index_to_tab[scan_indx];
		band->scan_size = band->blk_size;

		int quant_mat = get_bits(gb, 5);
		if (quant_mat == 31) return AVERROR_INVALIDDATA;
		if (quant_mat >= 22) return AVERROR_INVALIDDATA;
		band->quant_mat = quant_mat;
	} else {
		if (old_blk_size != band->blk_size) return AVERROR_INVALIDDATA;
	}

	if (quant_index_to_tab[band->quant_mat] > 4 && band->blk_size == 4) {
		band->quant_mat = 0;
		return AVERROR_INVALIDDATA;
	}
	if (band->scan_size != band->blk_size) return AVERROR_INVALIDDATA;
	if (band->transform_size == 8 && band->blk_size < 8) return AVERROR_INVALIDDATA;

	// Block Huffman: inherit the picture-level table by default.
	if (!get_bits1(gb))
		band->blk_vlc.tab = ctx->blk_vlc.tab;
	else if (dec_huff_desc(ctx, 1, IVI_BLK_HUFF, &band->blk_vlc))
		return AVERROR_INVALIDDATA;

	band->rvmap_sel = get_bits1(gb) ? get_bits(gb, 3) : 8;

	band->num_corr = 0;
	if (get_bits1(gb)) {
		band->num_corr = get_bits(gb, 8);
		if (band->num_corr > 61) return AVERROR_INVALIDDATA;
		for (int i = 0; i < band->num_corr * 2; i++) band->corr[i] = get_bits(gb, 8);
	}

	int tab = quant_index_to_tab[band->quant_mat];
	if (band->blk_size == 8) {
		band->intra_base = &ivi4_quant_8x8_intra[tab][0];
		band->inter_base = &ivi4_quant_8x8_inter[tab][0];
	} else {
		band->intra_base = &ivi4_quant_4x4_intra[tab][0];
		band->inter_base = &ivi4_quant_4x4_inter[tab][0];
	}
	band->intra_scale = nullptr;   // Indeo 4 has no scale tables
	band->inter_scale = nullptr;

	align_get_bits(gb);
	if (!band->scan) return AVERROR_INVALIDDATA;
	return 0;
}

static int decode_mb_info(Ctx* ctx, IVIBandDesc* band, IVITile* tile) {
	GetBits* gb = &ctx->gb;
	IVIMbInfo* mb = tile->mbs;
	IVIMbInfo* ref_mb = tile->ref_mbs;
	int offs = tile->ypos * band->pitch + tile->xpos;
	int blks_per_mb  = band->mb_size != band->blk_size ? 4 : 1;
	int mb_type_bits = ctx->frame_type == FRAMETYPE_BIDIR ? 2 : 1;
	int mv_scale = (ctx->planes[0].bands[0].mb_size >> 3) - (band->mb_size >> 3);
	int row_offset = band->mb_size * band->pitch;
	int mv_x = 0, mv_y = 0;

	if (((tile->width + band->mb_size - 1) / band->mb_size) *
	    ((tile->height + band->mb_size - 1) / band->mb_size) != tile->num_MBs)
		return AVERROR_INVALIDDATA;

	for (int y = tile->ypos; y < tile->ypos + tile->height; y += band->mb_size) {
		int mb_offset = offs;
		for (int x = tile->xpos; x < tile->xpos + tile->width; x += band->mb_size) {
			mb->xpos = x; mb->ypos = y; mb->buf_offs = mb_offset;
			mb->b_mv_x = mb->b_mv_y = 0;

			if (get_bits_left(gb) < 1) return AVERROR_INVALIDDATA;

			if (get_bits1(gb)) {
				if (ctx->frame_type == FRAMETYPE_INTRA) return AVERROR_INVALIDDATA;
				mb->type = 1; mb->cbp = 0; mb->q_delta = 0;
				if (!band->plane && !band->band_num && ctx->in_q)
					mb->q_delta = to_signed(get_vlc(gb, ctx->mb_vlc.tab));
				mb->mv_x = mb->mv_y = 0;
				if (band->inherit_mv && ref_mb) {
					if (mv_scale) {
						mb->mv_x = scale_mv(ref_mb->mv_x, mv_scale);
						mb->mv_y = scale_mv(ref_mb->mv_y, mv_scale);
					} else {
						mb->mv_x = ref_mb->mv_x;
						mb->mv_y = ref_mb->mv_y;
					}
				}
			} else {
				if (band->inherit_mv) {
					if (!ref_mb) return AVERROR_INVALIDDATA;
					mb->type = ref_mb->type;
				} else if (ctx->frame_type == FRAMETYPE_INTRA ||
				           ctx->frame_type == FRAMETYPE_INTRA1) {
					mb->type = 0;
				} else {
					mb->type = get_bits(gb, mb_type_bits);
				}

				mb->cbp = get_bits(gb, blks_per_mb);

				mb->q_delta = 0;
				if (band->inherit_qdelta) {
					if (ref_mb) mb->q_delta = ref_mb->q_delta;
				} else if (mb->cbp || (!band->plane && !band->band_num && ctx->in_q)) {
					mb->q_delta = to_signed(get_vlc(gb, ctx->mb_vlc.tab));
				}

				if (!mb->type) {
					mb->mv_x = mb->mv_y = 0;
				} else {
					if (band->inherit_mv) {
						if (ref_mb) {
							if (mv_scale) {
								mb->mv_x = scale_mv(ref_mb->mv_x, mv_scale);
								mb->mv_y = scale_mv(ref_mb->mv_y, mv_scale);
							} else {
								mb->mv_x = ref_mb->mv_x;
								mb->mv_y = ref_mb->mv_y;
							}
						}
					} else {
						mv_y += to_signed(get_vlc(gb, ctx->mb_vlc.tab));
						mv_x += to_signed(get_vlc(gb, ctx->mb_vlc.tab));
						mb->mv_x = mv_x;
						mb->mv_y = mv_y;
						if (mb->type == 3) {
							mv_y += to_signed(get_vlc(gb, ctx->mb_vlc.tab));
							mv_x += to_signed(get_vlc(gb, ctx->mb_vlc.tab));
							mb->b_mv_x = -mv_x;
							mb->b_mv_y = -mv_y;
						}
					}
					if (mb->type == 2) {
						mb->b_mv_x = -mb->mv_x;
						mb->b_mv_y = -mb->mv_y;
						mb->mv_x = 0;
						mb->mv_y = 0;
					}
				}
			}

			int s = band->is_halfpel;
			if (mb->type)
			if (x + (mb->mv_x >> s) + (y + (mb->mv_y >> s)) * band->pitch < 0 ||
			    x + ((mb->mv_x + s) >> s) + band->mb_size - 1
			      + (y + band->mb_size - 1 + ((mb->mv_y + s) >> s)) * band->pitch > band->bufsize - 1)
				return AVERROR_INVALIDDATA;

			mb++;
			if (ref_mb) ref_mb++;
			mb_offset += band->mb_size;
		}
		offs += row_offset;
	}

	align_get_bits(gb);
	return 0;
}

static void switch_buffers(Ctx* ctx) {
	int is_prev_ref = 0, is_ref = 0;
	switch (ctx->prev_frame_type) {
	case FRAMETYPE_INTRA: case FRAMETYPE_INTRA1: case FRAMETYPE_INTER: is_prev_ref = 1; break;
	}
	switch (ctx->frame_type) {
	case FRAMETYPE_INTRA: case FRAMETYPE_INTRA1: case FRAMETYPE_INTER: is_ref = 1; break;
	}
	if (is_prev_ref && is_ref) {
		int t = ctx->dst_buf; ctx->dst_buf = ctx->ref_buf; ctx->ref_buf = t;
	} else if (is_prev_ref) {
		int t = ctx->ref_buf; ctx->ref_buf = ctx->b_ref_buf; ctx->b_ref_buf = t;
		t = ctx->dst_buf; ctx->dst_buf = ctx->ref_buf; ctx->ref_buf = t;
	}
}

static int is_nonnull_frame(Ctx* ctx) { return ctx->frame_type < FRAMETYPE_NULL_FIRST; }

static Ctx* open(int width, int height) {
	Ctx* ctx = ctx_alloc();
	if (!ctx) return nullptr;
	ctx->is_indeo4        = 1;
	ctx->decode_pic_hdr   = decode_pic_hdr;
	ctx->decode_band_hdr  = decode_band_hdr;
	ctx->decode_mb_info   = decode_mb_info;
	ctx->switch_buffers   = switch_buffers;
	ctx->is_nonnull_frame = is_nonnull_frame;
	// Indeo 4 buffer rotation (decode_init): buffer 2 is reserved for scalability.
	ctx->dst_buf   = 0;
	ctx->ref_buf   = 1;
	ctx->b_ref_buf = 3;
	// pic_conf is all-zero (calloc) -> the first INTRA frame forces init_planes.
	(void)width; (void)height;
	return ctx;
}

} // namespace iv4

// Grow the Indeo output planes to the (GOP-decoded) coded size if needed, and
// update the handle's reported dimensions. The coded size comes from the
// bitstream's GOP header and can exceed the AVI-declared size used at rv_open.
static void iv_ensure_out(RvCtx* s, int w, int h) {
	int cw = (w + 3) >> 2, ch = (h + 3) >> 2;
	size_t ySize = (size_t)w * h;
	size_t cSize = (size_t)cw * ch;
	if (ySize > s->frameSize) {
		free(s->frame);
		s->frame = (uint8_t*)calloc(ySize ? ySize : 1, 1);
		s->frameSize = ySize;
	}
	if (cSize > (size_t)s->chromaW * s->chromaH) {
		free(s->plane1); free(s->plane2);
		s->plane1 = (uint8_t*)calloc(cSize + 1, 1);
		s->plane2 = (uint8_t*)calloc(cSize + 1, 1);
	}
	s->width = w; s->height = h; s->stride = w;
	s->chromaW = cw; s->chromaH = ch;
}

// Decode one Indeo 4/5 frame into the handle's Y/Cb/Cr planes. On a NULL frame
// (no new data) the previous picture is left in place (a repeat). 0 on success.
static int iv_decode(RvCtx* s, const uint8_t* buf, int len) {
	ivi::Ctx* ctx = s->ivi;
	int r = ivi::iv_decode_frame(ctx, buf, len);
	if (r < 0) return r;
	if (ctx->got_frame) {
		int w = ctx->planes[0].width, h = ctx->planes[0].height;
		iv_ensure_out(s, w, h);
		ivi::iv_output(ctx, s->frame, s->stride, s->plane1, s->chromaW, s->plane2, s->chromaW);
	}
	return 0;
}

// Decode one Indeo 3 frame into the handle's Y/Cb/Cr planes. Sync/null frames
// leave the previous picture in place (a repeat). 0 on success.
static int iv3_decode(RvCtx* s, const uint8_t* buf, int len) {
	iv3::Ctx* ctx = s->iv3;
	int r = iv3::decode(ctx, buf, len);
	if (r < 0) return r;   // hard error (8-bit / halfpel / corrupt header)
	if (ctx->got_frame) {
		int w = ctx->width, h = ctx->height;
		iv_ensure_out(s, w, h);
		iv3::output(ctx, s->frame, w, h, s->plane1, s->plane2, s->chromaW, s->chromaH);
	}
	// Partially-decoded frame still shown, but flag it for a one-shot warning.
	if (ctx->plane_failed) s->warn = RV_W_BADDATA;
	return 0;
}

// ════════════════════════════════════════════════════════════════════════════
//  C boundary
// ════════════════════════════════════════════════════════════════════════════
extern "C" {

RvCtx* rv_open(int codecId, int width, int height, int flags) {
	RvCtx* s = (RvCtx*)calloc(1, sizeof(RvCtx));
	if (!s) return nullptr;
	s->codecId = codecId;
	s->sega_film_skip_bytes = -1;

	const int paletted = (flags & RV_FLAG_PALETTED) != 0;
	switch (codecId) {
	case RV_CVID:
		s->width  = (width + 3) & ~3;
		s->height = (height + 3) & ~3;
		s->format = paletted ? RV_FMT_INDEX8 : RV_FMT_RGB24;
		s->bpp    = paletted ? 1 : 3;
		s->stride = s->width * s->bpp;
		s->cvid_codebooks = (uint8_t*)calloc((size_t)cvid::MAX_STRIPS * 2 * 256 * 12, 1);
		if (!s->cvid_codebooks) { free(s); return nullptr; }
		break;
	case RV_MSV1:
		s->width  = width;
		s->height = height;
		s->format = paletted ? RV_FMT_INDEX8 : RV_FMT_RGB555;
		s->bpp    = paletted ? 1 : 2;
		s->stride = s->width * s->bpp;
		break;
	case RV_MSRLE:
	case RV_QTRLE:
		s->width  = width;
		s->height = height;
		s->depth  = flags >> RV_FLAG_DEPTH_SHIFT;
		if (s->depth <= 8)       { s->format = RV_FMT_INDEX8; s->bpp = 1; }
		else if (s->depth == 16) { s->format = RV_FMT_RGB555; s->bpp = 2; }
		else if (s->depth == 32 && codecId == RV_QTRLE) { s->format = RV_FMT_ARGB32; s->bpp = 4; }
		else                     { s->format = RV_FMT_RGB24;  s->bpp = 3; }
		s->stride = s->width * s->bpp;
		if (codecId == RV_QTRLE && s->depth <= 8) {
			s->rleStride = s->width + qtrle::GROUP_PIXELS;
			s->rlePic = (uint8_t*)calloc((size_t)s->rleStride * s->height + 1, 1);
			if (!s->rlePic) { free(s); return nullptr; }
		}
		break;
	case RV_RPZA:
	case RV_SMC:
		s->width  = width;
		s->height = height;
		s->format = codecId == RV_RPZA ? RV_FMT_RGB555 : RV_FMT_INDEX8;
		s->bpp    = codecId == RV_RPZA ? 2 : 1;
		s->stride = s->width * s->bpp;
		s->blockStride  = ((s->width + 3) & ~3) * s->bpp;
		s->blockPicSize = (size_t)s->blockStride * ((s->height + 3) & ~3);
		s->blockPic = (uint8_t*)calloc(s->blockPicSize ? s->blockPicSize : 1, 1);
		if (!s->blockPic) { free(s); return nullptr; }
		break;
	case RV_8BPS:
		s->width  = width;
		s->height = height;
		s->depth  = flags >> RV_FLAG_DEPTH_SHIFT;
		if (!eightbps::plane_count(s->depth)) { free(s); return nullptr; }
		s->format = s->depth == 8 ? RV_FMT_INDEX8 : s->depth == 24 ? RV_FMT_RGB24 : RV_FMT_ARGB32;
		s->bpp    = s->depth / 8;
		s->stride = s->width * s->bpp;
		break;
	case RV_HFYU:
		// Room for A,R,G,B; YCbCr output takes less. rv_set_extradata decides which.
		s->width  = width;
		s->height = height;
		s->depth  = flags >> RV_FLAG_DEPTH_SHIFT;
		s->format = RV_FMT_ARGB32;
		s->bpp    = 4;
		s->stride = s->width * 4;
		break;
	case RV_IV2:
		s->width   = width;
		s->height  = height;
		s->format  = RV_FMT_YCBCR410;
		s->bpp     = 1;                       // Y plane bytes/pixel
		s->stride  = s->width;
		s->chromaW = (width + 3) >> 2;
		s->chromaH = (height + 3) >> 2;
		s->plane1 = (uint8_t*)calloc((size_t)s->chromaW * s->chromaH + 1, 1);
		s->plane2 = (uint8_t*)calloc((size_t)s->chromaW * s->chromaH + 1, 1);
		if (!s->plane1 || !s->plane2) { free(s->plane1); free(s->plane2); free(s); return nullptr; }
		iv2::build_lut();
		break;
	case RV_IV4:
	case RV_IV5:
		s->width   = width;
		s->height  = height;
		s->format  = RV_FMT_YCBCR410;
		s->bpp     = 1;
		s->stride  = s->width;
		s->chromaW = (width + 3) >> 2;
		s->chromaH = (height + 3) >> 2;
		s->plane1 = (uint8_t*)calloc((size_t)s->chromaW * s->chromaH + 1, 1);
		s->plane2 = (uint8_t*)calloc((size_t)s->chromaW * s->chromaH + 1, 1);
		s->ivi    = (codecId == RV_IV4) ? iv4::open(width, height) : iv5::open(width, height);
		if (!s->plane1 || !s->plane2 || !s->ivi) {
			free(s->plane1); free(s->plane2); ivi::ctx_free(s->ivi); free(s); return nullptr;
		}
		break;
	case RV_IV3:
		s->width   = width;
		s->height  = height;
		s->format  = RV_FMT_YCBCR410;
		s->bpp     = 1;
		s->stride  = s->width;
		s->chromaW = (width + 3) >> 2;
		s->chromaH = (height + 3) >> 2;
		s->plane1 = (uint8_t*)calloc((size_t)s->chromaW * s->chromaH + 1, 1);
		s->plane2 = (uint8_t*)calloc((size_t)s->chromaW * s->chromaH + 1, 1);
		s->iv3    = iv3::open(width, height);
		if (!s->plane1 || !s->plane2 || !s->iv3) {
			free(s->plane1); free(s->plane2); iv3::close(s->iv3); free(s); return nullptr;
		}
		break;
	default:
		free(s);
		return nullptr;
	}

	s->frameSize = (size_t)s->stride * s->height;
	s->frame = (uint8_t*)calloc(s->frameSize ? s->frameSize : 1, 1);
	if (!s->frame) { free(s->rlePic); free(s->blockPic); free(s->cvid_codebooks); free(s->plane1); free(s->plane2); free(s); return nullptr; }
	return s;
}

void rv_close(RvCtx* s) {
	if (!s) return;
	ivi::ctx_free(s->ivi);
	iv3::close(s->iv3);
	hfyu::close(s->hyuv);
	free(s->rlePic);
	free(s->blockPic);
	free(s->cvid_codebooks);
	free(s->plane1);
	free(s->plane2);
	free(s->frame);
	free(s);
}

// Seek: clear the persistent picture so a stale frame can't bleed through before
// the next keyframe rebuilds it. Cinepak's per-frame Sega-FILM probe is sticky
// (FFmpeg keeps it for the file's lifetime), so it is NOT reset here.
void rv_reset(RvCtx* s) {
	if (!s) return;
	if (s->frame) memset(s->frame, 0, s->frameSize);
	if (s->rlePic) memset(s->rlePic, 0, (size_t)s->rleStride * s->height);
	if (s->blockPic) memset(s->blockPic, 0, s->blockPicSize);
	if (s->plane1) memset(s->plane1, 0, (size_t)s->chromaW * s->chromaH);
	if (s->plane2) memset(s->plane2, 0, (size_t)s->chromaW * s->chromaH);
	if (s->ivi) ivi::ctx_reset(s->ivi);
	if (s->iv3) iv3::reset(s->iv3);
}

int rv_decode_frame(RvCtx* s, const uint8_t* data, int len, int32_t* info) {
	int ok = 0;
	if (s) {
		int r = AVERROR_INVALIDDATA;
		s->warn = RV_W_NONE;
		switch (s->codecId) {
		case RV_CVID: r = cvid::decode(s, data, len); break;
		case RV_MSV1: r = msv1::decode(s, data, len); break;
		case RV_IV2:  r = iv2::decode(s, data, len); break;
		case RV_IV3:  r = iv3_decode(s, data, len); break;
		case RV_IV4:  r = iv_decode(s, data, len); break;
		case RV_IV5:  r = iv_decode(s, data, len); break;
		case RV_MSRLE: r = msrle::decode(s, data, len); break;
		case RV_QTRLE: r = qtrle::decode(s, data, len); break;
		case RV_RPZA: r = rpza::decode(s, data, len); break;
		case RV_SMC:  r = smc::decode(s, data, len); break;
		case RV_8BPS: r = eightbps::decode(s, data, len); break;
		case RV_HFYU: r = hfyu::decode(s, data, len); break;
		}
		ok = (r == 0) ? 1 : 0;
		if (info) {
			info[0] = s->width;
			info[1] = s->height;
			info[2] = s->format;
			info[3] = s->bpp;
			info[4] = s->format == RV_FMT_ARGB32;   // hasAlpha
			info[5] = ok;
			info[6] = (r != 0) ? rv_warn_code(r) : s->warn;
			info[7] = 0;
		}
	} else if (info) {
		for (int i = 0; i < 8; i++) info[i] = 0;
	}
	return ok;
}

// HuffYUV's stream parameters and Huffman tables. Answers what its frames hold (hfyu::OUT_*),
// negative when the stream cannot be decoded.
int rv_set_extradata(RvCtx* s, const uint8_t* data, int len) {
	if (!s || s->codecId != RV_HFYU) return AVERROR_INVALIDDATA;
	return hfyu::open(s, data, len);
}

uint8_t* rv_plane_ptr(RvCtx* s, int idx) {
	if (!s) return nullptr;
	switch (idx) {
	case 0:  return s->frame;    // Y / packed RGB / index
	case 1:  return s->plane1;   // Cb / U
	case 2:  return s->plane2;   // Cr / V
	default: return nullptr;
	}
}

void rv_set_simd(int on) { g_use_simd = on; }

// Conformance/inspection hook: run one IVI inverse transform on a caller-provided
// int32[64] coefficient block + flags[8/4] into an int16 output (pitch in int16s).
// kind: 0 slant8, 1 slant4, 2 rowSlant8, 3 colSlant8, 4 rowSlant4, 5 colSlant4,
//       6 haar8, 7 haar4, 8 rowHaar8, 9 colHaar8, 10 rowHaar4, 11 colHaar4.
// Mirrors the InvTransformPtr signature; used by the SIMD-vs-scalar test (Step 6)
// and to validate the scalar port against the JS ivi_dsp oracle now.
int rv_test_transform(int kind, const int32_t* in, int16_t* out, int pitch, const uint8_t* flags) {
	switch (kind) {
	case 0:  iv45::ff_ivi_inverse_slant_8x8(in, out, pitch, flags); break;
	case 1:  iv45::ff_ivi_inverse_slant_4x4(in, out, pitch, flags); break;
	case 2:  iv45::ff_ivi_row_slant8(in, out, pitch, flags); break;
	case 3:  iv45::ff_ivi_col_slant8(in, out, pitch, flags); break;
	case 4:  iv45::ff_ivi_row_slant4(in, out, pitch, flags); break;
	case 5:  iv45::ff_ivi_col_slant4(in, out, pitch, flags); break;
	case 6:  iv45::ff_ivi_inverse_haar_8x8(in, out, pitch, flags); break;
	case 7:  iv45::ff_ivi_inverse_haar_4x4(in, out, pitch, flags); break;
	case 8:  iv45::ff_ivi_row_haar8(in, out, pitch, flags); break;
	case 9:  iv45::ff_ivi_col_haar8(in, out, pitch, flags); break;
	case 10: iv45::ff_ivi_row_haar4(in, out, pitch, flags); break;
	case 11: iv45::ff_ivi_col_haar4(in, out, pitch, flags); break;
	default: return 0;
	}
	return 1;
}

// Benchmark / equivalence helper for the SIMD step: run inverse transform `kind`
// `iters` times over an internal block in a tight in-wasm loop, and return a
// checksum of the accumulated output. The block is perturbed each iteration so the
// optimizer can't hoist the call out of the loop. Honors g_use_simd, so the test
// times scalar (rv_set_simd 0) vs SIMD (rv_set_simd 1) without per-block JS->wasm
// call overhead, and asserts both runs return the identical checksum.
int rv_bench_transform(int kind, int iters) {
	static int32_t in[64];
	static int16_t out[64];
	static uint8_t flags[8] = { 1, 1, 1, 1, 1, 1, 1, 1 };
	int blk = (kind == 1 || kind == 4 || kind == 5 ||
	           kind == 7 || kind == 10 || kind == 11) ? 4 : 8;
	int n = blk * blk;
	for (int i = 0; i < 64; i++) in[i] = ((i * 37) % 97) - 48;   // small signed spread
	uint32_t sum = 0;
	for (int it = 0; it < iters; it++) {
		in[0]     = it & 1023;       // perturb to defeat loop-invariant hoisting
		in[n - 1] = -(it & 511);
		rv_test_transform(kind, in, out, blk, flags);
		for (int i = 0; i < n; i++) sum = sum * 31u + (uint16_t)out[i];
	}
	return (int)sum;
}

} // extern "C"
