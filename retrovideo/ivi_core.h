/*
	Indeo Video Interactive (Indeo 4 / Indeo 5) shared decoder core — a direct
	port of FFmpeg libavcodec/ivi.c (+ the wavelet recompositions from
	ivi_dsp.c), release/8.1, LGPL. The libavcodec framework (AVCodecContext /
	AVFrame / GetBitContext / VLC) is replaced by a self-contained context
	(ivi::Ctx) and an LSB-first bit reader; the block transforms + motion
	compensation live in ivi_transforms.h (namespace iv45) and the static tables
	in ivi_tables.h (namespace ivi).

	A codec frontend (Indeo 5 here; Indeo 4 later) installs five hooks on the
	context — decode_pic_hdr / decode_band_hdr / decode_mb_info / switch_buffers /
	is_nonnull_frame — exactly as the C frontends set the corresponding function
	pointers. iv_decode_frame() then drives the common band/tile/MB decode loop.

	Header-only; #included once by codec_pack.cpp. Everything is `static` (single
	TU, no external linkage). -fwrapv keeps the slant/Haar two's-complement
	shifts byte-identical to FFmpeg.
*/

#ifndef IVI_CORE_H
#define IVI_CORE_H

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include "ivi_transforms.h"   // iv45:: block transforms + motion compensation
#include "ivi_tables.h"       // ivi::  scan / rvmap / huff-desc / indeo5 quant tables

namespace ivi {

static const int IVI_VLC_BITS = 13;            // max ivi huffman code length
static const int IVI_AVERR    = -1;            // generic AVERROR_INVALIDDATA
static const int IVI5_IS_PROTECTED = 0x20;     // gop_flags bit: password-protected
static const int IVI4_FRAMETYPE_BIDIR     = 3;
static const int IVI4_FRAMETYPE_NULL_LAST = 6;

// ── small arithmetic helpers (namespace-local; -fwrapv applies TU-wide) ──
static inline int iclip(int a, int lo, int hi) { return a < lo ? lo : (a > hi ? hi : a); }
static inline int iclip_u8(int a) { return a < 0 ? 0 : (a > 255 ? 255 : a); }
static inline int imin(int a, int b) { return a < b ? a : b; }
static inline int isign(int a) { return a > 0 ? 1 : -1; }   // FFSIGN: FFSIGN(0) == -1

// ════════════════════════════════════════════════════════════════════════════
//  Structures (mirror libavcodec/ivi.h)
// ════════════════════════════════════════════════════════════════════════════
typedef void (*InvTransformPtr)(const int32_t*, int16_t*, ptrdiff_t, const uint8_t*);
typedef void (*DCTransformPtr)(const int32_t*, int16_t*, ptrdiff_t, int);
typedef void (*ivi_mc_func)(int16_t*, const int16_t*, ptrdiff_t, int);
typedef void (*ivi_mc_avg_func)(int16_t*, const int16_t*, const int16_t*, ptrdiff_t, int, int);

struct IVIHuffTab {
	int             tab_sel;
	const uint16_t* tab;        // 1<<13 LUT of (sym<<5)|len; 0 == invalid
	IVIHuffDesc     cust_desc;  // custom codebook descriptor (tab_sel == 7)
	uint16_t*       cust_tab;   // owned dynamic LUT for the custom codebook, or null
};

struct IVIMbInfo {
	int16_t  xpos, ypos;
	uint32_t buf_offs;
	uint8_t  type, cbp;
	int8_t   q_delta, mv_x, mv_y, b_mv_x, b_mv_y;
};

struct IVITile {
	int        xpos, ypos, width, height, mb_size;
	int        is_empty, data_size, num_MBs;
	IVIMbInfo* mbs;
	IVIMbInfo* ref_mbs;
};

struct IVIBandDesc {
	int             plane, band_num, width, height, aheight;
	int             data_size;
	int16_t*        buf;
	int16_t*        ref_buf;
	int16_t*        b_ref_buf;
	int16_t*        bufs[4];
	ptrdiff_t       pitch;
	int             is_empty, mb_size, blk_size, is_halfpel;
	int             inherit_mv, inherit_qdelta, qdelta_present;
	int             quant_mat, glob_quant;
	const uint8_t*  scan;
	int             scan_size;
	IVIHuffTab      blk_vlc;
	int             num_corr;
	uint8_t         corr[61 * 2];
	int             rvmap_sel;
	RVMapDesc*      rv_map;
	int             num_tiles;
	IVITile*        tiles;
	InvTransformPtr inv_transform;
	int             transform_size;
	DCTransformPtr  dc_transform;
	int             is_2d_trans;
	int             checksum, checksum_present;
	int             bufsize;       // band buffer size in int16 elements
	const uint16_t* intra_base;
	const uint16_t* inter_base;
	const uint8_t*  intra_scale;
	const uint8_t*  inter_scale;
};

struct IVIPlaneDesc {
	uint16_t     width, height;
	uint8_t      num_bands;
	IVIBandDesc* bands;
};

struct IVIPicConfig {
	uint16_t pic_width, pic_height, chroma_width, chroma_height, tile_width, tile_height;
	uint8_t  luma_bands, chroma_bands;
};

// LSB-first bit reader (FFmpeg BITSTREAM_READER_LE GetBitContext).
struct GetBits {
	const uint8_t* data;
	int            bitOff;
	int            bitEnd;   // total bits == byteLen*8
};

struct Ctx {
	GetBits   gb;
	RVMapDesc rvmap_tabs[9];   // local corrected copy of the static rvmap tables

	uint32_t  frame_num;
	int       frame_type, prev_frame_type;
	uint32_t  data_size;
	int       is_scalable;
	const uint8_t* frame_data;
	uint32_t  frame_size;
	int       inter_scal;
	uint32_t  pic_hdr_size;
	uint8_t   frame_flags;
	uint16_t  checksum;

	IVIPicConfig pic_conf;
	IVIPlaneDesc planes[3];

	int       buf_switch, dst_buf, ref_buf, ref2_buf, b_ref_buf;

	IVIHuffTab mb_vlc, blk_vlc;

	uint16_t  gop_hdr_size;
	uint8_t   gop_flags;
	uint32_t  lock_word;

	int       is_indeo4;
	int       gop_invalid;
	int       buf_invalid[4];
	int       got_frame;       // 1 if this frame produced new output

	// Indeo 4 picture-header flags (set by the iv4 frontend).
	uint8_t   has_b_frames, has_transp, uses_tiling, uses_haar, uses_fullpel;
	uint8_t   in_imf, in_q, pic_glob_quant, unknown1, rvmap_sel;

	int (*decode_pic_hdr)  (Ctx*);
	int (*decode_band_hdr) (Ctx*, IVIBandDesc*);
	int (*decode_mb_info)  (Ctx*, IVIBandDesc*, IVITile*);
	void (*switch_buffers) (Ctx*);
	int (*is_nonnull_frame)(Ctx*);
};

// ════════════════════════════════════════════════════════════════════════════
//  LE bit reader  (mirrors get_bits.h with BITSTREAM_READER_LE)
// ════════════════════════════════════════════════════════════════════════════
static inline uint32_t get_bits(GetBits* gb, int n) {
	int off = gb->bitOff, byteOff = off >> 3, bit = off & 7, len = gb->bitEnd >> 3;
	uint32_t b0 = (byteOff     < len) ? gb->data[byteOff]     : 0;
	uint32_t b1 = (byteOff + 1 < len) ? gb->data[byteOff + 1] : 0;
	uint32_t b2 = (byteOff + 2 < len) ? gb->data[byteOff + 2] : 0;
	uint32_t b3 = (byteOff + 3 < len) ? gb->data[byteOff + 3] : 0;
	uint32_t bits = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
	gb->bitOff = off + n;
	uint32_t mask = (n >= 32) ? 0xFFFFFFFFu : ((1u << n) - 1u);
	return (bits >> bit) & mask;
}
static inline int get_bits1(GetBits* gb) { return (int)get_bits(gb, 1); }
static inline uint32_t show_bits(GetBits* gb, int n) {
	int save = gb->bitOff;
	uint32_t v = get_bits(gb, n);
	gb->bitOff = save;
	return v;
}
static inline int peek13(GetBits* gb) {
	int off = gb->bitOff, byteOff = off >> 3, bit = off & 7, len = gb->bitEnd >> 3;
	uint32_t b0 = (byteOff     < len) ? gb->data[byteOff]     : 0;
	uint32_t b1 = (byteOff + 1 < len) ? gb->data[byteOff + 1] : 0;
	uint32_t b2 = (byteOff + 2 < len) ? gb->data[byteOff + 2] : 0;
	return (int)(((b0 | (b1 << 8) | (b2 << 16)) >> bit) & 0x1FFF);
}
static inline void skip_bits(GetBits* gb, int n) { gb->bitOff += n; }
static inline void align_get_bits(GetBits* gb) { gb->bitOff = (gb->bitOff + 7) & ~7; }
static inline int get_bits_count(GetBits* gb) { return gb->bitOff; }
static inline int get_bits_left(GetBits* gb) { return gb->bitEnd - gb->bitOff; }

// VLC peek/advance for a complete prefix table. Returns the symbol. The IVI
// codebooks (the 8 static mb/blk tables and any well-formed custom one) are
// complete, so on a conformant stream every code is in the table and the len==0
// branch is unreachable. It exists only as a guard against a malformed custom
// codebook (tab_sel == 7 with LUT holes): there we deterministically reject by
// advancing one bit and returning 0xFFFF, which trips the caller's `sym >= 256`
// check (main symbol) or the scan_pos-overflow + post-loop check (escape sub-
// codes) into AVERROR — no infinite loop, no OOB. This differs from ffmpeg's
// get_vlc2 (returns the unfilled entry consuming 0 bits) and ivi.js getVlc
// (throws), but all three only diverge on this unreachable-on-valid-data path.
static inline int get_vlc(GetBits* gb, const uint16_t* table) {
	int e = table[peek13(gb)];
	int len = e & 0x1F;
	if (!len) { gb->bitOff += 1; return 0xFFFF; }
	gb->bitOff += len;
	return e >> 5;
}

// ════════════════════════════════════════════════════════════════════════════
//  VLC table construction  (ivi_create_huff_from_desc + vlc_init OUTPUT_LE)
// ════════════════════════════════════════════════════════════════════════════
// Build a 1<<13 LUT from a huffman descriptor. Returns 0, or -1 if the
// descriptor would need a code longer than IVI_VLC_BITS.
static int create_huff_from_desc(const IVIHuffDesc* cb, uint16_t* lookup) {
	uint16_t codewords[256];
	uint8_t  bits[256];
	int pos = 0;

	for (int i = 0; i < cb->num_rows; i++) {
		int codes_per_row = 1 << cb->xbits[i];
		int not_last_row  = (i != cb->num_rows - 1) ? 1 : 0;
		int prefix        = ((1 << i) - 1) << (cb->xbits[i] + not_last_row);
		for (int j = 0; j < codes_per_row; j++) {
			if (pos >= 256) break;     // Indeo5 codebooks may list >256; only 256 allowed
			int b = i + cb->xbits[i] + not_last_row;
			if (b > IVI_VLC_BITS) return -1;
			codewords[pos] = (uint16_t)(prefix | j);
			if (!b) b = 1;
			bits[pos] = (uint8_t)b;
			pos++;
		}
	}

	memset(lookup, 0, (size_t)(1 << IVI_VLC_BITS) * sizeof(uint16_t));
	for (int s = 0; s < pos; s++) {
		int code_len = bits[s], code = codewords[s], rev = 0;
		for (int b = 0; b < code_len; b++)
			if ((code >> (code_len - 1 - b)) & 1) rev |= 1 << b;
		uint16_t value = (uint16_t)((s << 5) | code_len);
		int fill_step = 1 << code_len;
		for (int h = rev; h < (1 << IVI_VLC_BITS); h += fill_step) lookup[h] = value;
	}
	return 0;
}

// Predefined macroblock/block tables, built once (BSS, not stored in the .wasm).
static uint16_t g_mb_vlc_tabs[8][1 << IVI_VLC_BITS];
static uint16_t g_blk_vlc_tabs[8][1 << IVI_VLC_BITS];
static int      g_static_vlc_inited = 0;

static void init_static_vlc() {
	if (g_static_vlc_inited) return;
	for (int i = 0; i < 8; i++) {
		create_huff_from_desc(&ivi_mb_huff_desc[i],  g_mb_vlc_tabs[i]);
		create_huff_from_desc(&ivi_blk_huff_desc[i], g_blk_vlc_tabs[i]);
	}
	g_static_vlc_inited = 1;
}

static int huff_desc_cmp(const IVIHuffDesc* a, const IVIHuffDesc* b) {
	return a->num_rows != b->num_rows || memcmp(a->xbits, b->xbits, a->num_rows);
}

// ff_ivi_dec_huff_desc: select a predefined table, or parse + build a custom one.
static int dec_huff_desc(Ctx* ctx, int desc_coded, int which_tab, IVIHuffTab* huff_tab) {
	GetBits* gb = &ctx->gb;
	if (!desc_coded) {
		huff_tab->tab = which_tab ? g_blk_vlc_tabs[7] : g_mb_vlc_tabs[7];
		return 0;
	}
	huff_tab->tab_sel = get_bits(gb, 3);
	if (huff_tab->tab_sel == 7) {
		IVIHuffDesc new_huff;
		new_huff.num_rows = get_bits(gb, 4);
		if (!new_huff.num_rows) return IVI_AVERR;
		for (int i = 0; i < new_huff.num_rows; i++) new_huff.xbits[i] = get_bits(gb, 4);
		if (huff_desc_cmp(&new_huff, &huff_tab->cust_desc) || !huff_tab->cust_tab) {
			huff_tab->cust_desc.num_rows = new_huff.num_rows;
			memcpy(huff_tab->cust_desc.xbits, new_huff.xbits, new_huff.num_rows);
			if (!huff_tab->cust_tab)
				huff_tab->cust_tab = (uint16_t*)malloc((size_t)(1 << IVI_VLC_BITS) * sizeof(uint16_t));
			if (!huff_tab->cust_tab) return IVI_AVERR;
			if (create_huff_from_desc(&huff_tab->cust_desc, huff_tab->cust_tab) < 0) {
				huff_tab->cust_desc.num_rows = 0;
				return IVI_AVERR;
			}
		}
		huff_tab->tab = huff_tab->cust_tab;
	} else {
		huff_tab->tab = which_tab ? g_blk_vlc_tabs[huff_tab->tab_sel]
		                          : g_mb_vlc_tabs[huff_tab->tab_sel];
	}
	return 0;
}

// ════════════════════════════════════════════════════════════════════════════
//  Plane / tile allocation  (ff_ivi_init_planes / ff_ivi_init_tiles)
// ════════════════════════════════════════════════════════════════════════════
static int num_mbs_per_tile(int tw, int th, int mb) {
	return ((tw + mb - 1) / mb) * ((th + mb - 1) / mb);
}

static void free_buffers(IVIPlaneDesc* planes) {
	for (int p = 0; p < 3; p++) {
		if (planes[p].bands) {
			for (int b = 0; b < planes[p].num_bands; b++) {
				IVIBandDesc* band = &planes[p].bands[b];
				for (int k = 0; k < 4; k++) { free(band->bufs[k]); band->bufs[k] = nullptr; }
				if (band->blk_vlc.cust_tab) { free(band->blk_vlc.cust_tab); band->blk_vlc.cust_tab = nullptr; }
				if (band->tiles) {
					for (int t = 0; t < band->num_tiles; t++) { free(band->tiles[t].mbs); band->tiles[t].mbs = nullptr; }
					free(band->tiles); band->tiles = nullptr;
				}
			}
		}
		free(planes[p].bands);
		planes[p].bands = nullptr;
		planes[p].num_bands = 0;
	}
}

static int init_planes(Ctx* ctx, const IVIPicConfig* cfg) {
	IVIPlaneDesc* planes = ctx->planes;
	free_buffers(planes);
	if (cfg->pic_width <= 0 || cfg->pic_height <= 0) return IVI_AVERR;
	if (cfg->luma_bands < 1 || cfg->chroma_bands < 1) return IVI_AVERR;

	planes[0].width     = cfg->pic_width;
	planes[0].height    = cfg->pic_height;
	planes[0].num_bands = cfg->luma_bands;
	planes[1].width = planes[2].width  = (cfg->pic_width  + 3) >> 2;
	planes[1].height = planes[2].height = (cfg->pic_height + 3) >> 2;
	planes[1].num_bands = planes[2].num_bands = cfg->chroma_bands;

	for (int p = 0; p < 3; p++) {
		planes[p].bands = (IVIBandDesc*)calloc(planes[p].num_bands, sizeof(IVIBandDesc));
		if (!planes[p].bands) return IVI_AVERR;

		int b_width  = planes[p].num_bands == 1 ? planes[p].width  : (planes[p].width  + 1) >> 1;
		int b_height = planes[p].num_bands == 1 ? planes[p].height : (planes[p].height + 1) >> 1;
		int align_fac      = p ? 8 : 16;
		int width_aligned  = ((b_width  + align_fac - 1) / align_fac) * align_fac;
		int height_aligned = ((b_height + align_fac - 1) / align_fac) * align_fac;
		int buf_size       = width_aligned * height_aligned;   // int16 elements

		for (int b = 0; b < planes[p].num_bands; b++) {
			IVIBandDesc* band = &planes[p].bands[b];
			band->plane    = p;
			band->band_num = b;
			band->width    = b_width;
			band->height   = b_height;
			band->pitch    = width_aligned;
			band->aheight  = height_aligned;
			band->bufsize  = buf_size;
		}
		planes[p].bands[0].blk_vlc.cust_desc.num_rows = 0;
	}
	return 0;
}

static int init_tiles_band(IVIBandDesc* band, IVITile* ref_tiles, int p, int b, int t_height, int t_width) {
	IVITile* tile = band->tiles;
	int ref_idx = 0;
	for (int y = 0; y < band->height; y += t_height) {
		for (int x = 0; x < band->width; x += t_width) {
			tile->xpos    = x;
			tile->ypos    = y;
			tile->mb_size = band->mb_size;
			tile->width   = imin(band->width  - x, t_width);
			tile->height  = imin(band->height - y, t_height);
			tile->is_empty = tile->data_size = 0;
			tile->num_MBs  = num_mbs_per_tile(tile->width, tile->height, band->mb_size);

			free(tile->mbs);
			tile->mbs = (IVIMbInfo*)calloc(tile->num_MBs, sizeof(IVIMbInfo));
			if (!tile->mbs) return IVI_AVERR;

			tile->ref_mbs = nullptr;
			if (p || b) {
				if (tile->num_MBs != ref_tiles[ref_idx].num_MBs) return IVI_AVERR;
				tile->ref_mbs = ref_tiles[ref_idx].mbs;
				ref_idx++;
			}
			tile++;
		}
	}
	return 0;
}

static int init_tiles(Ctx* ctx, int tile_width, int tile_height) {
	IVIPlaneDesc* planes = ctx->planes;
	for (int p = 0; p < 3; p++) {
		int t_width  = !p ? tile_width  : (tile_width  + 3) >> 2;
		int t_height = !p ? tile_height : (tile_height + 3) >> 2;

		if (!p && planes[0].num_bands == 4) {
			if ((t_width & 1) || (t_height & 1)) return -2;   // odd tiles: PATCHWELCOME
			t_width  >>= 1;
			t_height >>= 1;
		}
		if (t_width <= 0 || t_height <= 0) return IVI_AVERR;

		for (int b = 0; b < planes[p].num_bands; b++) {
			IVIBandDesc* band = &planes[p].bands[b];
			if (band->tiles) {
				for (int t = 0; t < band->num_tiles; t++) { free(band->tiles[t].mbs); band->tiles[t].mbs = nullptr; }
			}
			int x_tiles = (band->width  + t_width  - 1) / t_width;
			int y_tiles = (band->height + t_height - 1) / t_height;
			band->num_tiles = x_tiles * y_tiles;

			free(band->tiles);
			band->tiles = (IVITile*)calloc(band->num_tiles, sizeof(IVITile));
			if (!band->tiles) { band->num_tiles = 0; return IVI_AVERR; }

			int ret = init_tiles_band(band, planes[0].bands[0].tiles, p, b, t_height, t_width);
			if (ret < 0) return ret;
		}
	}
	return 0;
}

// Lazy band-buffer allocator (prepare_buf). i is a buffer slot (dst/ref/...).
static int16_t* prepare_buf(Ctx* ctx, IVIBandDesc* band, int i) {
	if (ctx->pic_conf.luma_bands <= 1 && i == 2) return nullptr;
	if (!band->bufs[i]) band->bufs[i] = (int16_t*)calloc(band->bufsize, sizeof(int16_t));
	return band->bufs[i];
}

// ════════════════════════════════════════════════════════════════════════════
//  Motion compensation dispatch  (ivi_mc)
// ════════════════════════════════════════════════════════════════════════════
static int ivi_mc(IVIBandDesc* band, ivi_mc_func mc, ivi_mc_avg_func mc_avg,
                  int offs, int mv_x, int mv_y, int mv_x2, int mv_y2,
                  int mc_type, int mc_type2) {
	int ref_offs = offs + mv_y * band->pitch + mv_x;
	int buf_size = band->pitch * band->aheight;
	int min_size = band->pitch * (band->blk_size - 1) + band->blk_size;
	int ref_size = (mc_type > 1) * band->pitch + (mc_type & 1);

	if (mc_type != -1) {
		if (offs < 0 || ref_offs < 0 || !band->ref_buf) return IVI_AVERR;
		if (buf_size - min_size < offs) return IVI_AVERR;
		if (buf_size - min_size - ref_size < ref_offs) return IVI_AVERR;
	}

	if (mc_type2 == -1) {
		mc(band->buf + offs, band->ref_buf + ref_offs, band->pitch, mc_type);
	} else {
		int ref_offs2 = offs + mv_y2 * band->pitch + mv_x2;
		int ref_size2 = (mc_type2 > 1) * band->pitch + (mc_type2 & 1);
		if (offs < 0 || ref_offs2 < 0 || !band->b_ref_buf) return IVI_AVERR;
		if (buf_size - min_size - ref_size2 < ref_offs2) return IVI_AVERR;
		if (mc_type == -1)
			mc(band->buf + offs, band->b_ref_buf + ref_offs2, band->pitch, mc_type2);
		else
			mc_avg(band->buf + offs, band->ref_buf + ref_offs,
			       band->b_ref_buf + ref_offs2, band->pitch, mc_type, mc_type2);
	}
	return 0;
}

// ════════════════════════════════════════════════════════════════════════════
//  Block decode  (ivi_decode_coded_blocks / ivi_decode_blocks)
// ════════════════════════════════════════════════════════════════════════════
// IVI_TOSIGNED: unsigned (sign in LSB) -> signed.
static inline int to_signed(int v) { return -(((v >> 1) ^ -(v & 1))); }

static int decode_coded_block(GetBits* gb, IVIBandDesc* band,
                              ivi_mc_func mc, ivi_mc_avg_func mc_avg,
                              int mv_x, int mv_y, int mv_x2, int mv_y2,
                              int* prev_dc, int is_intra, int mc_type, int mc_type2,
                              uint32_t quant, int offs) {
	const uint16_t* base_tab = is_intra ? band->intra_base : band->inter_base;
	RVMapDesc* rvmap = band->rv_map;
	uint8_t col_flags[8];
	int32_t trvec[64];
	uint32_t sym = 0, lo, hi, q;
	int pos, run, val;
	int blk_size   = band->blk_size;
	int num_coeffs = blk_size * blk_size;
	int col_mask   = blk_size - 1;
	int scan_pos   = -1;
	int min_size   = band->pitch * (band->transform_size - 1) + band->transform_size;
	int buf_size   = band->pitch * band->aheight - offs;

	if (min_size > buf_size) return IVI_AVERR;
	if (!band->scan) return IVI_AVERR;

	memset(trvec, 0, num_coeffs * sizeof(trvec[0]));
	memset(col_flags, 0, sizeof(col_flags));

	while (scan_pos <= num_coeffs) {
		sym = (uint32_t)get_vlc(gb, band->blk_vlc.tab);
		if (sym == rvmap->eob_sym) break;

		if (sym == rvmap->esc_sym) {
			run = get_vlc(gb, band->blk_vlc.tab) + 1;
			lo  = (uint32_t)get_vlc(gb, band->blk_vlc.tab);
			hi  = (uint32_t)get_vlc(gb, band->blk_vlc.tab);
			val = to_signed((hi << 6) | lo);
		} else {
			if (sym >= 256U) return IVI_AVERR;
			run = rvmap->runtab[sym];
			val = rvmap->valtab[sym];
		}

		scan_pos += run;
		if (scan_pos >= num_coeffs || scan_pos < 0) break;
		pos = band->scan[scan_pos];

		q = ((uint32_t)base_tab[pos] * quant) >> 9;
		if (q > 1)
			val = val * (int)q + isign(val) * ((int)((q ^ 1) - 1) >> 1);
		trvec[pos] = val;
		col_flags[pos & col_mask] |= !!val;
	}

	if (scan_pos < 0 || (scan_pos >= num_coeffs && sym != rvmap->eob_sym))
		return IVI_AVERR;

	if (is_intra && band->is_2d_trans) {
		*prev_dc     += trvec[0];
		trvec[0]      = *prev_dc;
		col_flags[0] |= !!*prev_dc;
	}

	if (band->transform_size > band->blk_size) return IVI_AVERR;

	band->inv_transform(trvec, band->buf + offs, band->pitch, col_flags);

	if (!is_intra)
		return ivi_mc(band, mc, mc_avg, offs, mv_x, mv_y, mv_x2, mv_y2, mc_type, mc_type2);
	return 0;
}

static int decode_blocks(Ctx* ctx, IVIBandDesc* band, IVITile* tile) {
	int blk_size   = band->blk_size;
	int num_blocks = (band->mb_size != blk_size) ? 4 : 1;
	int mc_type = 0, mc_type2 = -1;
	int mv_x = 0, mv_y = 0, mv_x2 = 0, mv_y2 = 0;
	int prev_dc = 0;
	GetBits* gb = &ctx->gb;
	ivi_mc_func mc_with_delta, mc_no_delta;
	ivi_mc_avg_func mc_avg_with_delta, mc_avg_no_delta;

	if (blk_size == 8) {
		mc_with_delta     = iv45::ff_ivi_mc_8x8_delta;
		mc_no_delta       = iv45::ff_ivi_mc_8x8_no_delta;
		mc_avg_with_delta = iv45::ff_ivi_mc_avg_8x8_delta;
		mc_avg_no_delta   = iv45::ff_ivi_mc_avg_8x8_no_delta;
	} else {
		mc_with_delta     = iv45::ff_ivi_mc_4x4_delta;
		mc_no_delta       = iv45::ff_ivi_mc_4x4_no_delta;
		mc_avg_with_delta = iv45::ff_ivi_mc_avg_4x4_delta;
		mc_avg_no_delta   = iv45::ff_ivi_mc_avg_4x4_no_delta;
	}

	for (int mbn = 0; mbn < tile->num_MBs; mbn++) {
		IVIMbInfo* mb = &tile->mbs[mbn];
		int is_intra = !mb->type;
		uint32_t cbp = mb->cbp;
		uint32_t buf_offs = mb->buf_offs;

		int quant = band->glob_quant + mb->q_delta;
		if (ctx->is_indeo4) quant = iclip(quant, 0, 31);   // av_clip_uintp2(.,5)
		else                quant = iclip(quant, 0, 23);
		const uint8_t* scale_tab = is_intra ? band->intra_scale : band->inter_scale;
		if (scale_tab) quant = scale_tab[quant];

		if (!is_intra) {
			mv_x  = mb->mv_x;  mv_y  = mb->mv_y;
			mv_x2 = mb->b_mv_x; mv_y2 = mb->b_mv_y;
			if (band->is_halfpel) {
				mc_type  = ((mv_y  & 1) << 1) | (mv_x  & 1);
				mc_type2 = ((mv_y2 & 1) << 1) | (mv_x2 & 1);
				mv_x  >>= 1; mv_y  >>= 1; mv_x2 >>= 1; mv_y2 >>= 1;
			}
			if (mb->type == 2) mc_type = -1;
			if (mb->type != 2 && mb->type != 3) mc_type2 = -1;
			if (mb->type) {
				int dmv_x = mb->mv_x >> band->is_halfpel;
				int dmv_y = mb->mv_y >> band->is_halfpel;
				int cx = mb->mv_x & band->is_halfpel;
				int cy = mb->mv_y & band->is_halfpel;
				if (mb->xpos + dmv_x < 0 || mb->xpos + dmv_x + band->mb_size + cx > band->pitch ||
				    mb->ypos + dmv_y < 0 || mb->ypos + dmv_y + band->mb_size + cy > band->aheight)
					return IVI_AVERR;
			}
			if (mb->type == 2 || mb->type == 3) {
				int dmv_x = mb->b_mv_x >> band->is_halfpel;
				int dmv_y = mb->b_mv_y >> band->is_halfpel;
				int cx = mb->b_mv_x & band->is_halfpel;
				int cy = mb->b_mv_y & band->is_halfpel;
				if (mb->xpos + dmv_x < 0 || mb->xpos + dmv_x + band->mb_size + cx > band->pitch ||
				    mb->ypos + dmv_y < 0 || mb->ypos + dmv_y + band->mb_size + cy > band->aheight)
					return IVI_AVERR;
			}
		}

		for (int blk = 0; blk < num_blocks; blk++) {
			if (blk & 1) {
				buf_offs += blk_size;
			} else if (blk == 2) {
				buf_offs -= blk_size;
				buf_offs += blk_size * band->pitch;
			}
			if (cbp & 1) {
				int ret = decode_coded_block(gb, band, mc_with_delta, mc_avg_with_delta,
				                             mv_x, mv_y, mv_x2, mv_y2, &prev_dc, is_intra,
				                             mc_type, mc_type2, (uint32_t)quant, (int)buf_offs);
				if (ret < 0) return ret;
			} else {
				int buf_sz = band->pitch * band->aheight - (int)buf_offs;
				int min_sz = (blk_size - 1) * band->pitch + blk_size;
				if (min_sz > buf_sz) return IVI_AVERR;
				if (is_intra) {
					band->dc_transform(&prev_dc, band->buf + buf_offs, band->pitch, blk_size);
				} else {
					int ret = ivi_mc(band, mc_no_delta, mc_avg_no_delta, (int)buf_offs,
					                 mv_x, mv_y, mv_x2, mv_y2, mc_type, mc_type2);
					if (ret < 0) return ret;
				}
			}
			cbp >>= 1;
		}
	}
	align_get_bits(gb);
	return 0;
}

// ════════════════════════════════════════════════════════════════════════════
//  Empty tile  (ivi_process_empty_tile)
// ════════════════════════════════════════════════════════════════════════════
static inline int scale_mv(int mv, int mv_scale) {
	return (mv + (mv > 0 ? 1 : 0) + (mv_scale - 1)) >> mv_scale;
}

static int process_empty_tile(Ctx* ctx, IVIBandDesc* band, IVITile* tile, int mv_scale) {
	int mb_size    = band->mb_size;
	int is_halfpel = band->is_halfpel;
	int pitch      = band->pitch;
	int xend       = tile->xpos + tile->width;
	int clear_first = !band->qdelta_present && !band->plane && !band->band_num;
	(void)ctx;

	if (tile->num_MBs != num_mbs_per_tile(tile->width, tile->height, mb_size))
		return IVI_AVERR;

	int offs = tile->ypos * pitch + tile->xpos;
	int row_offset = mb_size * pitch;
	int need_mc = 0;
	int mbi = 0, ref_idx = 0;
	IVIMbInfo* ref_mbs = tile->ref_mbs;

	for (int y = tile->ypos; y < tile->ypos + tile->height; y += mb_size) {
		int mb_offset = offs;
		for (int x = tile->xpos; x < xend; x += mb_size) {
			IVIMbInfo* mb = &tile->mbs[mbi++];
			IVIMbInfo* ref_mb = ref_mbs ? &ref_mbs[ref_idx++] : nullptr;
			mb->xpos = x; mb->ypos = y; mb->buf_offs = mb_offset;
			mb->type = 1; mb->cbp = 0;
			if (clear_first) { mb->q_delta = band->glob_quant; mb->mv_x = 0; mb->mv_y = 0; }
			if (ref_mb) {
				if (band->inherit_qdelta) mb->q_delta = ref_mb->q_delta;
				if (band->inherit_mv) {
					if (mv_scale) {
						mb->mv_x = scale_mv(ref_mb->mv_x, mv_scale);
						mb->mv_y = scale_mv(ref_mb->mv_y, mv_scale);
					} else {
						mb->mv_x = ref_mb->mv_x;
						mb->mv_y = ref_mb->mv_y;
					}
					need_mc |= mb->mv_x || mb->mv_y;
					int dmv_x = mb->mv_x >> is_halfpel, dmv_y = mb->mv_y >> is_halfpel;
					int cx = mb->mv_x & is_halfpel, cy = mb->mv_y & is_halfpel;
					if (mb->xpos + dmv_x < 0 || mb->xpos + dmv_x + mb_size + cx > pitch ||
					    mb->ypos + dmv_y < 0 || mb->ypos + dmv_y + mb_size + cy > band->aheight)
						return IVI_AVERR;
				}
			}
			mb_offset += mb_size;
		}
		offs += row_offset;
	}

	if (band->inherit_mv && need_mc) {
		int num_blocks = (mb_size != band->blk_size) ? 4 : 1;
		ivi_mc_func mc_no_delta = (band->blk_size == 8) ? iv45::ff_ivi_mc_8x8_no_delta
		                                                : iv45::ff_ivi_mc_4x4_no_delta;
		for (int m = 0; m < tile->num_MBs; m++) {
			IVIMbInfo* mb = &tile->mbs[m];
			int mv_x = mb->mv_x, mv_y = mb->mv_y, mc_type;
			if (!band->is_halfpel) {
				mc_type = 0;
			} else {
				mc_type = ((mv_y & 1) << 1) | (mv_x & 1);
				mv_x >>= 1; mv_y >>= 1;
			}
			for (int blk = 0; blk < num_blocks; blk++) {
				int off = mb->buf_offs + band->blk_size * ((blk & 1) + (!!(blk & 2)) * pitch);
				int ret = ivi_mc(band, mc_no_delta, nullptr, off, mv_x, mv_y, 0, 0, mc_type, -1);
				if (ret < 0) return ret;
			}
		}
	} else {
		const int16_t* src = band->ref_buf + tile->ypos * pitch + tile->xpos;
		int16_t* dst       = band->buf     + tile->ypos * pitch + tile->xpos;
		if (!band->ref_buf) return IVI_AVERR;
		for (int y = 0; y < tile->height; y++) {
			memcpy(dst, src, tile->width * sizeof(band->buf[0]));
			src += pitch; dst += pitch;
		}
	}
	return 0;
}

// ════════════════════════════════════════════════════════════════════════════
//  Tile data size  (ivi_dec_tile_data_size)
// ════════════════════════════════════════════════════════════════════════════
static int dec_tile_data_size(GetBits* gb) {
	int len = 0;
	if (get_bits1(gb)) {
		len = get_bits(gb, 8);
		if (len == 255) len = get_bits(gb, 24);
	}
	align_get_bits(gb);
	return len;
}

// ════════════════════════════════════════════════════════════════════════════
//  Band decode  (decode_band)
// ════════════════════════════════════════════════════════════════════════════
static void rvmap_swap(RVMapDesc* m, int a, int b) {
	uint8_t r = m->runtab[a]; m->runtab[a] = m->runtab[b]; m->runtab[b] = r;
	int8_t  v = m->valtab[a]; m->valtab[a] = m->valtab[b]; m->valtab[b] = v;
	if (a == m->eob_sym || b == m->eob_sym) m->eob_sym ^= a ^ b;
	if (a == m->esc_sym || b == m->esc_sym) m->esc_sym ^= a ^ b;
}

static int decode_band(Ctx* ctx, IVIBandDesc* band) {
	band->buf = prepare_buf(ctx, band, ctx->dst_buf);
	if (!band->buf) return IVI_AVERR;

	if (ctx->is_indeo4 && ctx->frame_type == IVI4_FRAMETYPE_BIDIR) {
		band->ref_buf   = prepare_buf(ctx, band, ctx->b_ref_buf);
		band->b_ref_buf = prepare_buf(ctx, band, ctx->ref_buf);
		if (!band->b_ref_buf) return IVI_AVERR;
	} else {
		band->ref_buf   = prepare_buf(ctx, band, ctx->ref_buf);
		band->b_ref_buf = nullptr;
	}
	if (!band->ref_buf) return IVI_AVERR;

	int result = ctx->decode_band_hdr(ctx, band);
	if (result) return result;
	if (band->is_empty) return IVI_AVERR;

	band->rv_map = &ctx->rvmap_tabs[band->rvmap_sel];

	for (int i = 0; i < band->num_corr; i++)
		rvmap_swap(band->rv_map, band->corr[i * 2], band->corr[i * 2 + 1]);

	int pos = get_bits_count(&ctx->gb);
	result = 0;

	for (int t = 0; t < band->num_tiles; t++) {
		IVITile* tile = &band->tiles[t];
		if (tile->mb_size != band->mb_size || ctx->planes[0].bands[0].mb_size < band->mb_size) {
			result = IVI_AVERR; break;
		}
		tile->is_empty = get_bits1(&ctx->gb);
		if (tile->is_empty) {
			int mv_scale = (ctx->planes[0].bands[0].mb_size >> 3) - (band->mb_size >> 3);
			result = process_empty_tile(ctx, band, tile, mv_scale);
			if (result < 0) break;
		} else {
			tile->data_size = dec_tile_data_size(&ctx->gb);
			if (!tile->data_size) { result = IVI_AVERR; break; }
			result = ctx->decode_mb_info(ctx, band, tile);
			if (result < 0) break;
			result = decode_blocks(ctx, band, tile);
			if (result < 0) break;
			if (((get_bits_count(&ctx->gb) - pos) >> 3) != tile->data_size) { result = IVI_AVERR; break; }
			pos += tile->data_size << 3;
		}
	}

	// Always undo rvmap corrections (in reverse).
	for (int i = band->num_corr - 1; i >= 0; i--)
		rvmap_swap(band->rv_map, band->corr[i * 2], band->corr[i * 2 + 1]);

	align_get_bits(&ctx->gb);
	return result;
}

// ════════════════════════════════════════════════════════════════════════════
//  Output  (ivi_output_plane + wavelet recompositions)
// ════════════════════════════════════════════════════════════════════════════
static void output_plane(IVIPlaneDesc* plane, uint8_t* dst, int dst_pitch) {
	const int16_t* src = plane->bands[0].buf;
	if (!src) return;
	ptrdiff_t pitch = plane->bands[0].pitch;
	for (int y = 0; y < plane->height; y++) {
		for (int x = 0; x < plane->width; x++) dst[x] = (uint8_t)iclip_u8(src[x] + 128);
		src += pitch;
		dst += dst_pitch;
	}
}

// ff_ivi_recompose53 — 5/3 wavelet recomposition over 4 bands (scalable Indeo5).
static void recompose53(const IVIPlaneDesc* plane, uint8_t* dst, int dst_pitch) {
	int32_t p0, p1, p2, p3, tmp0, tmp1, tmp2;
	int32_t b0_1, b0_2, b1_1, b1_2, b1_3, b2_1, b2_2, b2_3, b2_4, b2_5, b2_6;
	int32_t b3_1, b3_2, b3_3, b3_4, b3_5, b3_6, b3_7, b3_8, b3_9;
	ptrdiff_t pitch = plane->bands[0].pitch, back_pitch = 0;
	const int16_t* b0_ptr = plane->bands[0].buf;
	const int16_t* b1_ptr = plane->bands[1].buf;
	const int16_t* b2_ptr = plane->bands[2].buf;
	const int16_t* b3_ptr = plane->bands[3].buf;

	for (int y = 0; y < plane->height; y += 2) {
		if (y + 2 >= plane->height) pitch = 0;
		b0_1 = b0_ptr[0];
		b0_2 = b0_ptr[pitch];
		b1_1 = b1_ptr[back_pitch];
		b1_2 = b1_ptr[0];
		b1_3 = b1_1 - b1_2 * 6 + b1_ptr[pitch];
		b2_2 = b2_ptr[0]; b2_3 = b2_2;
		b2_5 = b2_ptr[pitch]; b2_6 = b2_5;
		b3_2 = b3_ptr[back_pitch]; b3_3 = b3_2;
		b3_5 = b3_ptr[0]; b3_6 = b3_5;
		b3_8 = b3_2 - b3_5 * 6 + b3_ptr[pitch]; b3_9 = b3_8;

		for (int x = 0, indx = 0; x < plane->width; x += 2, indx++) {
			if (x + 2 >= plane->width) { b0_ptr--; b1_ptr--; b2_ptr--; b3_ptr--; }
			b2_1 = b2_2; b2_2 = b2_3; b2_4 = b2_5; b2_5 = b2_6;
			b3_1 = b3_2; b3_2 = b3_3; b3_4 = b3_5; b3_5 = b3_6; b3_7 = b3_8; b3_8 = b3_9;

			tmp0 = b0_1; tmp2 = b0_2;
			b0_1 = b0_ptr[indx + 1];
			b0_2 = b0_ptr[pitch + indx + 1];
			tmp1 = tmp0 + b0_1;
			p0 = tmp0 * 16;
			p1 = tmp1 * 8;
			p2 = (tmp0 + tmp2) * 8;
			p3 = (tmp1 + tmp2 + b0_2) * 4;

			tmp0 = b1_2; tmp1 = b1_1;
			b1_2 = b1_ptr[indx + 1];
			b1_1 = b1_ptr[back_pitch + indx + 1];
			tmp2 = tmp1 - tmp0 * 6 + b1_3;
			b1_3 = b1_1 - b1_2 * 6 + b1_ptr[pitch + indx + 1];
			p0 += (tmp0 + tmp1) * 8;
			p1 += (tmp0 + tmp1 + b1_1 + b1_2) * 4;
			p2 += tmp2 * 4;
			p3 += (tmp2 + b1_3) * 2;

			b2_3 = b2_ptr[indx + 1];
			b2_6 = b2_ptr[pitch + indx + 1];
			tmp0 = b2_1 + b2_2;
			tmp1 = b2_1 - b2_2 * 6 + b2_3;
			p0 += tmp0 * 8;
			p1 += tmp1 * 4;
			p2 += (tmp0 + b2_4 + b2_5) * 4;
			p3 += (tmp1 + b2_4 - b2_5 * 6 + b2_6) * 2;

			b3_6 = b3_ptr[indx + 1];
			b3_3 = b3_ptr[back_pitch + indx + 1];
			tmp0 = b3_1 + b3_4;
			tmp1 = b3_2 + b3_5;
			tmp2 = b3_3 + b3_6;
			b3_9 = b3_3 - b3_6 * 6 + b3_ptr[pitch + indx + 1];
			p0 += (tmp0 + tmp1) * 4;
			p1 += (tmp0 - tmp1 * 6 + tmp2) * 2;
			p2 += (b3_7 + b3_8) * 2;
			p3 += b3_7 - b3_8 * 6 + b3_9;

			dst[x]               = (uint8_t)iclip_u8((p0 >> 6) + 128);
			dst[x + 1]           = (uint8_t)iclip_u8((p1 >> 6) + 128);
			dst[dst_pitch + x]   = (uint8_t)iclip_u8((p2 >> 6) + 128);
			dst[dst_pitch + x + 1] = (uint8_t)iclip_u8((p3 >> 6) + 128);
		}
		dst += dst_pitch << 1;
		back_pitch = -pitch;
		b0_ptr += pitch + 1;
		b1_ptr += pitch + 1;
		b2_ptr += pitch + 1;
		b3_ptr += pitch + 1;
	}
}

// ff_ivi_recompose_haar — Haar wavelet recomposition (scalable Indeo4).
static void recompose_haar(const IVIPlaneDesc* plane, uint8_t* dst, int dst_pitch) {
	ptrdiff_t pitch = plane->bands[0].pitch;
	const int16_t* b0_ptr = plane->bands[0].buf;
	const int16_t* b1_ptr = plane->bands[1].buf;
	const int16_t* b2_ptr = plane->bands[2].buf;
	const int16_t* b3_ptr = plane->bands[3].buf;

	for (int y = 0; y < plane->height; y += 2) {
		for (int x = 0, indx = 0; x < plane->width; x += 2, indx++) {
			int b0 = b0_ptr[indx], b1 = b1_ptr[indx], b2 = b2_ptr[indx], b3 = b3_ptr[indx];
			int p0 = (b0 + b1 + b2 + b3 + 2) >> 2;
			int p1 = (b0 + b1 - b2 - b3 + 2) >> 2;
			int p2 = (b0 - b1 + b2 - b3 + 2) >> 2;
			int p3 = (b0 - b1 - b2 + b3 + 2) >> 2;
			dst[x]                 = (uint8_t)iclip_u8(p0 + 128);
			dst[x + 1]             = (uint8_t)iclip_u8(p1 + 128);
			dst[dst_pitch + x]     = (uint8_t)iclip_u8(p2 + 128);
			dst[dst_pitch + x + 1] = (uint8_t)iclip_u8(p3 + 128);
		}
		dst += dst_pitch << 1;
		b0_ptr += pitch; b1_ptr += pitch; b2_ptr += pitch; b3_ptr += pitch;
	}
}

// Write the decoded frame to three planar 8-bit destinations (Y, Cb, Cr).
// Indeo is natively YVU9, so Cb == planes[2] and Cr == planes[1].
static void iv_output(Ctx* ctx, uint8_t* ydst, int ypitch,
                      uint8_t* cbdst, int cbpitch, uint8_t* crdst, int crpitch) {
	if (ctx->is_scalable && ctx->planes[0].num_bands == 4) {
		// Match FFmpeg exactly: scalable Indeo 4 ALWAYS uses the Haar recompose,
		// scalable Indeo 5 the 5/3 recompose (ff_ivi_decode_frame keys this on
		// is_indeo4, not on uses_haar — uses_haar is only a debug log there). The
		// JS oracle indeo4.js keys it on uses_haar, which is a latent bug on the
		// (untested) scalable-slant Indeo 4 path; we follow FFmpeg, not the JS.
		if (ctx->is_indeo4) recompose_haar(&ctx->planes[0], ydst, ypitch);
		else                recompose53  (&ctx->planes[0], ydst, ypitch);
	} else {
		output_plane(&ctx->planes[0], ydst, ypitch);
	}
	output_plane(&ctx->planes[2], cbdst, cbpitch);
	output_plane(&ctx->planes[1], crdst, crpitch);
}

// ════════════════════════════════════════════════════════════════════════════
//  Top-level frame driver  (ff_ivi_decode_frame, indeo5 subset)
// ════════════════════════════════════════════════════════════════════════════
static int iv_decode_frame(Ctx* ctx, const uint8_t* buf, int buf_size) {
	ctx->gb.data = buf;
	ctx->gb.bitOff = 0;
	ctx->gb.bitEnd = buf_size * 8;
	ctx->frame_data = buf;
	ctx->frame_size = buf_size;
	ctx->got_frame = 0;

	int result = ctx->decode_pic_hdr(ctx);
	if (result) return result;
	if (ctx->gop_invalid) return IVI_AVERR;

	if (ctx->is_indeo4 && ctx->frame_type == IVI4_FRAMETYPE_NULL_LAST)
		return 0;
	if (ctx->gop_flags & IVI5_IS_PROTECTED)
		return -2;   // password-protected: PATCHWELCOME
	if (!ctx->planes[0].bands)
		return IVI_AVERR;

	ctx->switch_buffers(ctx);

	if (ctx->is_nonnull_frame(ctx)) {
		ctx->buf_invalid[ctx->dst_buf] = 1;
		for (int p = 0; p < 3; p++) {
			for (int b = 0; b < ctx->planes[p].num_bands; b++) {
				result = decode_band(ctx, &ctx->planes[p].bands[b]);
				if (result < 0) return result;
			}
		}
		ctx->buf_invalid[ctx->dst_buf] = 0;
	} else {
		if (ctx->is_scalable) return IVI_AVERR;
		for (int p = 0; p < 3; p++)
			if (!ctx->planes[p].bands[0].buf) return IVI_AVERR;
	}
	if (ctx->buf_invalid[ctx->dst_buf]) return -1;

	ctx->got_frame = ctx->is_nonnull_frame(ctx);
	return 0;
}

// ════════════════════════════════════════════════════════════════════════════
//  Context lifecycle
// ════════════════════════════════════════════════════════════════════════════
static Ctx* ctx_alloc() {
	Ctx* ctx = (Ctx*)calloc(1, sizeof(Ctx));
	if (!ctx) return nullptr;
	init_static_vlc();
	memcpy(ctx->rvmap_tabs, ff_ivi_rvmap_tabs, sizeof(ff_ivi_rvmap_tabs));
	ctx->buf_switch = 0;
	ctx->inter_scal = 0;
	ctx->gop_invalid = 1;
	return ctx;
}

static void ctx_free(Ctx* ctx) {
	if (!ctx) return;
	free_buffers(ctx->planes);
	if (ctx->mb_vlc.cust_tab)  free(ctx->mb_vlc.cust_tab);
	if (ctx->blk_vlc.cust_tab) free(ctx->blk_vlc.cust_tab);
	free(ctx);
}

// Clear the band buffers (seek): zero every allocated band buffer so a stale
// inter frame can't bleed through before the next INTRA rebuilds the picture.
static void ctx_reset(Ctx* ctx) {
	if (!ctx) return;
	for (int p = 0; p < 3; p++) {
		if (!ctx->planes[p].bands) continue;
		for (int b = 0; b < ctx->planes[p].num_bands; b++) {
			IVIBandDesc* band = &ctx->planes[p].bands[b];
			for (int k = 0; k < 4; k++)
				if (band->bufs[k]) memset(band->bufs[k], 0, (size_t)band->bufsize * sizeof(int16_t));
		}
	}
	ctx->buf_invalid[0] = ctx->buf_invalid[1] = ctx->buf_invalid[2] = ctx->buf_invalid[3] = 0;
	ctx->buf_switch = 0;
	ctx->inter_scal = 0;
	ctx->prev_frame_type = 0;
}

} // namespace ivi

#endif // IVI_CORE_H
