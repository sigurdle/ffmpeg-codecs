/*
	Intel Indeo Video v3 (IV31 / IV32) decoder — a direct port of FFmpeg
	libavcodec/indeo3.c (release/8.1, LGPL). Standalone: a recursive cell tree
	(H/V binary splits) terminating in INTRA_NULL or INTER_DATA leaves; each cell
	is decoded by run-length + vector-quantized pixel-pair deltas applied with
	SWAR adds (carry between adjacent pixels is part of the encoding). Pixels are
	stored 7-bit; output_plane expands them to 8-bit with a left shift.

	Differences from the IVI codecs: a big-endian (MSB-first) bit reader for the
	2-bit tree opcodes interleaved with a byte-aligned cursor (next_cell_data) for
	cell payloads, deferred bit-reader resync (skip_bits / need_resync), and its
	own double-buffered plane layout (one extra top line for INTRA prediction).

	Header-only; #included once by codec_pack.cpp. -fwrapv keeps the SWAR adds and
	signed shifts byte-identical to FFmpeg. The VQ delta codebooks live in
	iv3_tables.h (namespace iv3, generated from the JS oracle).
*/

#ifndef IV3_DECODE_H
#define IV3_DECODE_H

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include "iv3_tables.h"   // iv3:: vqEntry + vq_tab

namespace iv3 {

static const int IV3_AVERR   = -1;   // AVERROR_INVALIDDATA
static const int IV3_E_8BIT  = -3;   // 8-bit paletted pixel format (unsupported)
static const int IV3_E_HALFPEL = -4; // half-pel motion vectors (unsupported)

// RLE opcodes.
enum { RLE_ESC_F9 = 249, RLE_ESC_FA = 250, RLE_ESC_FB = 251, RLE_ESC_FC = 252,
       RLE_ESC_FD = 253, RLE_ESC_FE = 254, RLE_ESC_FF = 255 };
// Bitstream flags.
enum { BS_8BIT_PEL = 1 << 1, BS_KEYFRAME = 1 << 2, BS_MV_Y_HALF = 1 << 4,
       BS_MV_X_HALF = 1 << 5, BS_NONREF = 1 << 8, BS_BUFFER = 9 };
// Cell-tree opcodes.
enum { H_SPLIT = 0, V_SPLIT = 1, INTRA_NULL = 2, INTER_DATA = 3 };
// Cell-decode error codes.
enum { IV3_NOERR = 0, IV3_BAD_RLE = 1, IV3_BAD_DATA = 2, IV3_BAD_COUNTER = 3,
       IV3_UNSUPPORTED = 4, IV3_OUT_OF_DATA = 5 };

#define IV3_OS_HDR_ID  0x46524D48u   // 'FRMH' big-endian
#define CELL_STACK_MAX 20

static inline int iv3_min(int a, int b) { return a < b ? a : b; }
static inline int iv3_align(int v, int a) { return (v + a - 1) & ~(a - 1); }

// ── unaligned little-endian read/write (wasm is LE; compiler folds to one op) ──
static inline uint32_t RL16(const uint8_t* p) { return (uint32_t)p[0] | ((uint32_t)p[1] << 8); }
static inline uint32_t RL32(const uint8_t* p) { uint32_t v; memcpy(&v, p, 4); return v; }
static inline uint32_t RN16(const uint8_t* p) { uint16_t v; memcpy(&v, p, 2); return v; }
static inline uint32_t RN32(const uint8_t* p) { uint32_t v; memcpy(&v, p, 4); return v; }
static inline uint64_t RN64(const uint8_t* p) { uint64_t v; memcpy(&v, p, 8); return v; }
static inline void WN16(uint8_t* p, uint32_t v) { uint16_t t = (uint16_t)v; memcpy(p, &t, 2); }
static inline void WN32(uint8_t* p, uint32_t v) { memcpy(p, &v, 4); }
static inline void WN64(uint8_t* p, uint64_t v) { memcpy(p, &v, 8); }

// Replicate even bytes: [b0,b1,b2,b3] -> [b0,b0,b2,b2] (LE).
static inline uint32_t replicate32(uint32_t a) { uint32_t t = a & 0x00FF00FFu; return t | (t << 8); }
static inline uint64_t replicate64(uint64_t a) { uint64_t t = a & 0x00FF00FF00FF00FFull; return t | (t << 8); }

static inline void avg32(uint8_t* dst, const uint8_t* src, const uint8_t* ref) {
	WN32(dst, ((RN32(src) + RN32(ref)) >> 1) & 0x7F7F7F7Fu);
}
static inline void avg64(uint8_t* dst, const uint8_t* src, const uint8_t* ref) {
	WN64(dst, ((RN64(src) + RN64(ref)) >> 1) & 0x7F7F7F7F7F7F7F7Full);
}
static inline void fill_64(uint8_t* dst, uint64_t pix, int n, ptrdiff_t row_offset) {
	for (; n > 0; dst += row_offset, n--) WN64(dst, pix);
}
static inline void copy_block4(uint8_t* dst, const uint8_t* src, ptrdiff_t dpitch, ptrdiff_t spitch, int h) {
	for (int i = 0; i < h; i++) { memcpy(dst, src, 4); dst += dpitch; src += spitch; }
}

// ── requantization table (built once) ──
static uint8_t requant_tab[8][128];
static int requant_built = 0;
static void build_requant_tab() {
	if (requant_built) return;
	static const int8_t offsets[8] = { 1, 1, 2, -3, -3, 3, 4, 4 };
	static const int8_t deltas[8]  = { 0, 1, 0,  4,  4, 1, 0, 1 };
	for (int i = 0; i < 8; i++) {
		int step = i + 2;
		for (int j = 0; j < 128; j++)
			requant_tab[i][j] = (uint8_t)((j + offsets[i]) / step * step + deltas[i]);
	}
	requant_tab[0][127] = 126;
	requant_tab[1][119] = 118; requant_tab[1][120] = 118;
	requant_tab[2][126] = 124; requant_tab[2][127] = 124;
	requant_tab[6][124] = 120; requant_tab[6][125] = 120;
	requant_tab[6][126] = 120; requant_tab[6][127] = 120;
	requant_tab[1][7] = 10; requant_tab[4][8] = 10;
	requant_built = 1;
}

// ── structures ──
struct Plane {
	uint8_t*  buffers[2];
	uint8_t*  pixels[2];   // buffers[i] + pitch (skip the INTRA prediction line)
	int       width, height;
	ptrdiff_t pitch;
};

struct Cell {
	int16_t      xpos, ypos, width, height;
	uint8_t      tree;       // 0 = MC tree, 1 = VQ tree
	const int8_t* mv_ptr;    // motion vector (mv_y, mv_x), or null for INTRA
};

// MSB-first bit reader (FFmpeg default get_bits.h). Only 2-bit reads occur.
struct BitBE {
	const uint8_t* base;
	int            bitOff;
	int            bitEnd;
};

struct Ctx {
	BitBE          gb;
	int            need_resync, skip_bits;
	const uint8_t* next_cell_data;
	const uint8_t* last_byte;
	const int8_t*  mc_vectors;
	unsigned       num_vectors;

	int            width, height;
	uint32_t       frame_num;
	int            data_size;
	uint16_t       frame_flags;
	uint8_t        cb_offset, buf_sel;
	const uint8_t* y_data_ptr;
	const uint8_t* v_data_ptr;
	const uint8_t* u_data_ptr;
	int            y_data_size, v_data_size, u_data_size;
	const uint8_t* alt_quant;
	Plane          planes[3];

	int            got_frame;
	int            plane_failed;   // a plane decode hit corrupt data this frame
};

static inline int get_bits_count(BitBE* gb) { return gb->bitOff; }
static inline int get_bits_left(BitBE* gb) { return gb->bitEnd - gb->bitOff; }
static inline void skip_bits_long(BitBE* gb, int n) { gb->bitOff += n; }
static inline int get_bits(BitBE* gb, int n) {
	int val = 0;
	for (int i = 0; i < n; i++) {
		int byteIdx = gb->bitOff >> 3;
		int bit = 7 - (gb->bitOff & 7);
		val = (val << 1) | ((gb->base[byteIdx] >> bit) & 1);
		gb->bitOff++;
	}
	return val;
}

// ── frame buffers ──
static void free_frame_buffers(Ctx* ctx) {
	ctx->width = ctx->height = 0;
	for (int p = 0; p < 3; p++) {
		free(ctx->planes[p].buffers[0]); ctx->planes[p].buffers[0] = nullptr;
		free(ctx->planes[p].buffers[1]); ctx->planes[p].buffers[1] = nullptr;
		ctx->planes[p].pixels[0] = ctx->planes[p].pixels[1] = nullptr;
	}
}

static int allocate_frame_buffers(Ctx* ctx, int luma_width, int luma_height) {
	luma_width  = iv3_align(luma_width, 2);
	luma_height = iv3_align(luma_height, 2);
	if (luma_width < 16 || luma_width > 640 || luma_height < 16 || luma_height > 480 ||
	    (luma_width & 1) || (luma_height & 1))
		return IV3_AVERR;

	ctx->width  = luma_width;
	ctx->height = luma_height;

	int chroma_width  = iv3_align(luma_width  >> 2, 4);
	int chroma_height = iv3_align(luma_height >> 2, 4);
	int luma_pitch    = iv3_align(luma_width, 16);
	int chroma_pitch  = iv3_align(chroma_width, 16);
	int luma_size     = luma_pitch * (luma_height + 1);
	int chroma_size   = chroma_pitch * (chroma_height + 1);

	for (int p = 0; p < 3; p++) {
		ctx->planes[p].pitch  = !p ? luma_pitch  : chroma_pitch;
		ctx->planes[p].width  = !p ? luma_width  : chroma_width;
		ctx->planes[p].height = !p ? luma_height : chroma_height;
		int sz = !p ? luma_size : chroma_size;
		ctx->planes[p].buffers[0] = (uint8_t*)malloc(sz);
		ctx->planes[p].buffers[1] = (uint8_t*)malloc(sz);
		if (!ctx->planes[p].buffers[0] || !ctx->planes[p].buffers[1]) return IV3_AVERR;
		memset(ctx->planes[p].buffers[0], 0x40, ctx->planes[p].pitch);
		memset(ctx->planes[p].buffers[1], 0x40, ctx->planes[p].pitch);
		ctx->planes[p].pixels[0] = ctx->planes[p].buffers[0] + ctx->planes[p].pitch;
		ctx->planes[p].pixels[1] = ctx->planes[p].buffers[1] + ctx->planes[p].pitch;
		memset(ctx->planes[p].pixels[0], 0, (size_t)ctx->planes[p].pitch * ctx->planes[p].height);
		memset(ctx->planes[p].pixels[1], 0, (size_t)ctx->planes[p].pitch * ctx->planes[p].height);
	}
	return 0;
}

// ── motion-compensated cell copy (full-pel; halfpel is rejected upstream) ──
static int copy_cell(Ctx* ctx, Plane* plane, Cell* cell) {
	int offset_dst = (cell->ypos << 2) * plane->pitch + (cell->xpos << 2);
	uint8_t* dst = plane->pixels[ctx->buf_sel] + offset_dst;
	int mv_y = 0, mv_x = 0;
	if (cell->mv_ptr) { mv_y = cell->mv_ptr[0]; mv_x = cell->mv_ptr[1]; }

	if ((cell->ypos << 2) + mv_y < -1 || (cell->xpos << 2) + mv_x < 0 ||
	    ((cell->ypos + cell->height) << 2) + mv_y > plane->height ||
	    ((cell->xpos + cell->width)  << 2) + mv_x > plane->width)
		return IV3_AVERR;

	int offset = offset_dst + mv_y * plane->pitch + mv_x;
	const uint8_t* src = plane->pixels[ctx->buf_sel ^ 1] + offset;
	int h = cell->height << 2;
	int w = cell->width << 2;
	for (int y = 0; y < h; y++) {
		memcpy(dst + y * plane->pitch, src + y * plane->pitch, w);
	}
	return 0;
}

// ── per-cell RLE/VQ data decode ──
static int decode_cell_data(Ctx* ctx, Cell* cell, uint8_t* block, const uint8_t* ref_block,
                            ptrdiff_t row_offset, int h_zoom, int v_zoom, int mode,
                            const vqEntry* delta[2], int swap_quads[2],
                            const uint8_t** data_ptr, const uint8_t* last_ptr) {
	int rle_blocks = 0;
	int skip_flag = 0, is_first_row = 1;
	const ptrdiff_t blk_row_offset = (row_offset << (2 + v_zoom)) - (cell->width << 2);
	const ptrdiff_t line_offset = v_zoom ? row_offset : 0;

	if ((cell->height & v_zoom) || (cell->width & h_zoom)) return IV3_BAD_DATA;

	for (int y = 0; y < cell->height; is_first_row = 0, y += 1 + v_zoom) {
		for (int x = 0; x < cell->width; x += 1 + h_zoom) {
			const uint8_t* ref = ref_block;
			uint8_t* dst = block;

			if (rle_blocks > 0) {
				if (mode <= 4) {
					if (cell->mv_ptr || !skip_flag)
						copy_block4(dst, ref, row_offset, row_offset, 4 << v_zoom);
				} else if (mode == 10 && !cell->mv_ptr) {
					uint64_t pix64 = RN64(ref);
					if (is_first_row) {
						pix64 = replicate64(pix64);
						fill_64(dst + row_offset, pix64, 7, row_offset);
						avg64(dst, ref, dst + row_offset);
					} else fill_64(dst, pix64, 8, row_offset);
				}
				rle_blocks--;
			} else {
				for (int line = 0; line < 4;) {
					int num_lines = 1;
					int is_top_of_cell = is_first_row && !line;
					const vqEntry* delta_tab = (mode <= 4) ? delta[line & 1] : delta[1];

					if (*data_ptr >= last_ptr) return IV3_OUT_OF_DATA;
					uint8_t code = *(*data_ptr)++;
					if (code < 248) {
						unsigned dyad1, dyad2;
						if (code < delta_tab->num_dyads) {
							if (*data_ptr >= last_ptr) return IV3_OUT_OF_DATA;
							dyad1 = *(*data_ptr)++;
							dyad2 = code;
							if (dyad1 >= delta_tab->num_dyads || dyad1 >= 248) return IV3_BAD_DATA;
						} else {
							code -= delta_tab->num_dyads;
							dyad1 = code / delta_tab->quad_exp;
							dyad2 = code % delta_tab->quad_exp;
							if (swap_quads[line & 1]) { unsigned t = dyad1; dyad1 = dyad2; dyad2 = t; }
						}
						if (mode <= 4) {
							/* APPLY_DELTA_4 */
							WN16(dst + line_offset,     (RN16(ref)     + delta_tab->deltas[dyad1]) & 0x7F7F);
							WN16(dst + line_offset + 2, (RN16(ref + 2) + delta_tab->deltas[dyad2]) & 0x7F7F);
							if (mode >= 3) {
								if (is_top_of_cell && !cell->ypos) memcpy(dst, dst + row_offset, 4);
								else avg32(dst, ref, dst + row_offset);
							}
						} else if (mode == 10 && !cell->mv_ptr) {
							/* APPLY_DELTA_8 */
							if (is_top_of_cell) {
								WN32(dst + row_offset,     (replicate32(RN32(ref))     + delta_tab->deltas_m10[dyad1]) & 0x7F7F7F7Fu);
								WN32(dst + row_offset + 4, (replicate32(RN32(ref + 4)) + delta_tab->deltas_m10[dyad2]) & 0x7F7F7F7Fu);
							} else {
								WN32(dst + row_offset,     (RN32(ref)     + delta_tab->deltas_m10[dyad1]) & 0x7F7F7F7Fu);
								WN32(dst + row_offset + 4, (RN32(ref + 4) + delta_tab->deltas_m10[dyad2]) & 0x7F7F7F7Fu);
							}
							if (is_top_of_cell && !cell->ypos) memcpy(dst, dst + row_offset, 8);
							else avg64(dst, ref, dst + row_offset);
						} else {
							/* APPLY_DELTA_1011_INTER */
							if (mode == 10) {
								WN32(dst,                  (RN32(dst)                  + delta_tab->deltas_m10[dyad1]) & 0x7F7F7F7Fu);
								WN32(dst + 4,              (RN32(dst + 4)              + delta_tab->deltas_m10[dyad2]) & 0x7F7F7F7Fu);
								WN32(dst + row_offset,     (RN32(dst + row_offset)     + delta_tab->deltas_m10[dyad1]) & 0x7F7F7F7Fu);
								WN32(dst + row_offset + 4, (RN32(dst + row_offset + 4) + delta_tab->deltas_m10[dyad2]) & 0x7F7F7F7Fu);
							} else {
								WN16(dst,                  (RN16(dst)                  + delta_tab->deltas[dyad1]) & 0x7F7F);
								WN16(dst + 2,              (RN16(dst + 2)              + delta_tab->deltas[dyad2]) & 0x7F7F);
								WN16(dst + row_offset,     (RN16(dst + row_offset)     + delta_tab->deltas[dyad1]) & 0x7F7F);
								WN16(dst + row_offset + 2, (RN16(dst + row_offset + 2) + delta_tab->deltas[dyad2]) & 0x7F7F);
							}
						}
					} else {
						switch (code) {
						case RLE_ESC_FC:
							skip_flag  = 0;
							rle_blocks = 1;
							code       = 253;
							/* fallthrough */
						case RLE_ESC_FF:
						case RLE_ESC_FE:
						case RLE_ESC_FD:
							num_lines = 257 - code - line;
							if (num_lines <= 0) return IV3_BAD_RLE;
							if (mode <= 4) {
								copy_block4(dst, ref, row_offset, row_offset, num_lines << v_zoom);
							} else if (mode == 10 && !cell->mv_ptr) {
								uint64_t pix64 = RN64(ref);
								if (is_top_of_cell) {
									pix64 = replicate64(pix64);
									fill_64(dst + row_offset, pix64, (num_lines << 1) - 1, row_offset);
									avg64(dst, ref, dst + row_offset);
								} else fill_64(dst, pix64, num_lines << 1, row_offset);
							}
							break;
						case RLE_ESC_FB: {
							if (*data_ptr >= last_ptr) return IV3_OUT_OF_DATA;
							code = *(*data_ptr)++;
							rle_blocks = (code & 0x1F) - 1;
							if (code >= 64 || rle_blocks < 0) return IV3_BAD_COUNTER;
							skip_flag = code & 0x20;
							num_lines = 4 - line;
							if (mode >= 10 || (cell->mv_ptr || !skip_flag)) {
								if (mode <= 4) {
									copy_block4(dst, ref, row_offset, row_offset, num_lines << v_zoom);
								} else if (mode == 10 && !cell->mv_ptr) {
									uint64_t pix64 = RN64(ref);
									if (is_top_of_cell) {
										pix64 = replicate64(pix64);
										fill_64(dst + row_offset, pix64, (num_lines << 1) - 1, row_offset);
										avg64(dst, ref, dst + row_offset);
									} else fill_64(dst, pix64, num_lines << 1, row_offset);
								}
							}
							break;
						}
						case RLE_ESC_F9:
							skip_flag  = 1;
							rle_blocks = 1;
							/* fallthrough */
						case RLE_ESC_FA:
							if (line) return IV3_BAD_RLE;
							num_lines = 4;
							if (cell->mv_ptr) {
								if (mode <= 4) {
									copy_block4(dst, ref, row_offset, row_offset, num_lines << v_zoom);
								} else if (mode == 10 && !cell->mv_ptr) {
									uint64_t pix64 = RN64(ref);
									if (is_top_of_cell) {
										pix64 = replicate64(pix64);
										fill_64(dst + row_offset, pix64, (num_lines << 1) - 1, row_offset);
										avg64(dst, ref, dst + row_offset);
									} else fill_64(dst, pix64, num_lines << 1, row_offset);
								}
							}
							break;
						default:
							return IV3_UNSUPPORTED;
						}
					}

					line += num_lines;
					ref  += row_offset * (num_lines << v_zoom);
					dst  += row_offset * (num_lines << v_zoom);
				}
			}

			block     += 4 << h_zoom;
			ref_block += 4 << h_zoom;
		}
		ref_block += blk_row_offset;
		block     += blk_row_offset;
	}
	return IV3_NOERR;
}

// ── cell-level mode dispatch + VQ table / requant selection ──
static int decode_cell(Ctx* ctx, Plane* plane, Cell* cell,
                       const uint8_t* data_ptr, const uint8_t* last_ptr) {
	const uint8_t* data_start = data_ptr;
	uint8_t code = *data_ptr++;
	int mode     = code >> 4;
	int vq_index = code & 0xF;

	int offset = (cell->ypos << 2) * plane->pitch + (cell->xpos << 2);
	uint8_t* block = plane->pixels[ctx->buf_sel] + offset;
	uint8_t* ref_block = nullptr;

	if (!cell->mv_ptr) {
		ref_block = block - plane->pitch;   // INTRA: predict from previous line
	} else if (mode >= 10) {
		int ret = copy_cell(ctx, plane, cell);   // 10/11 INTER: copy then deltas in place
		if (ret < 0) return ret;
	} else {
		int mv_y = cell->mv_ptr[0], mv_x = cell->mv_ptr[1];
		if ((cell->ypos << 2) + mv_y < -1 || (cell->xpos << 2) + mv_x < 0 ||
		    ((cell->ypos + cell->height) << 2) + mv_y > plane->height ||
		    ((cell->xpos + cell->width)  << 2) + mv_x > plane->width)
			return IV3_AVERR;
		offset += mv_y * plane->pitch + mv_x;
		ref_block = plane->pixels[ctx->buf_sel ^ 1] + offset;
	}

	int prim_indx, second_indx;
	if (mode == 1 || mode == 4) {
		code        = ctx->alt_quant[vq_index];
		prim_indx   = (code >> 4)  + ctx->cb_offset;
		second_indx = (code & 0xF) + ctx->cb_offset;
	} else {
		vq_index += ctx->cb_offset;
		prim_indx = second_indx = vq_index;
	}
	if (prim_indx >= 24 || second_indx >= 24) return IV3_AVERR;

	const vqEntry* delta[2] = { &vq_tab[second_indx], &vq_tab[prim_indx] };
	int swap_quads[2] = { second_indx >= 16, prim_indx >= 16 };

	if (vq_index >= 8 && ref_block) {
		for (int xx = 0; xx < (cell->width << 2); xx++)
			ref_block[xx] = requant_tab[vq_index & 7][ref_block[xx] & 127];
	}

	int error = IV3_NOERR;
	switch (mode) {
	case 0: case 1:
	case 3: case 4: {
		if (mode >= 3 && cell->mv_ptr) return IV3_AVERR;   // Mode 3/4 on INTER cell
		int zoom_fac = mode >= 3;
		error = decode_cell_data(ctx, cell, block, ref_block, plane->pitch,
		                         0, zoom_fac, mode, delta, swap_quads, &data_ptr, last_ptr);
		break;
	}
	case 10: case 11:
		if (mode == 10 && !cell->mv_ptr) {
			error = decode_cell_data(ctx, cell, block, ref_block, plane->pitch,
			                         1, 1, mode, delta, swap_quads, &data_ptr, last_ptr);
		} else {
			if (mode == 11 && !cell->mv_ptr) return IV3_AVERR;
			int zoom_fac = mode == 10;
			error = decode_cell_data(ctx, cell, block, block /* dummy ref */, plane->pitch,
			                         zoom_fac, 1, mode, delta, swap_quads, &data_ptr, last_ptr);
		}
		break;
	default:
		return IV3_AVERR;
	}

	if (error != IV3_NOERR) return IV3_AVERR;
	return (int)(data_ptr - data_start);
}

// ── recursive cell tree ──
#define IV3_SPLIT_CELL(size, new_size) (new_size) = ((size) > 2) ? ((((size) + 2) >> 2) << 1) : 1

static void update_bitpos(Ctx* ctx, int n) { ctx->skip_bits += n; ctx->need_resync = 1; }
static void resync_bitstream(Ctx* ctx) {
	if (ctx->need_resync && !(get_bits_count(&ctx->gb) & 7)) {
		skip_bits_long(&ctx->gb, ctx->skip_bits);
		ctx->skip_bits = 0;
		ctx->need_resync = 0;
	}
}
static int check_cell(Plane* plane, Cell* c) {
	return !(c->xpos + c->width > (plane->width >> 2) || c->ypos + c->height > (plane->height >> 2));
}

static int parse_bintree(Ctx* ctx, Plane* plane, int code, Cell* ref_cell,
                         int depth, int strip_width) {
	if (depth <= 0) return IV3_AVERR;

	Cell curr_cell = *ref_cell;   // clone parent
	if (code == H_SPLIT) {
		IV3_SPLIT_CELL(ref_cell->height, curr_cell.height);
		ref_cell->ypos   += curr_cell.height;
		ref_cell->height -= curr_cell.height;
		if (ref_cell->height <= 0 || curr_cell.height <= 0) return IV3_AVERR;
	} else if (code == V_SPLIT) {
		if (curr_cell.width > strip_width)
			curr_cell.width = (curr_cell.width <= (strip_width << 1) ? 1 : 2) * strip_width;
		else
			IV3_SPLIT_CELL(ref_cell->width, curr_cell.width);
		ref_cell->xpos  += curr_cell.width;
		ref_cell->width -= curr_cell.width;
		if (ref_cell->width <= 0 || curr_cell.width <= 0) return IV3_AVERR;
	}

	while (get_bits_left(&ctx->gb) >= 2) {
		resync_bitstream(ctx);
		code = get_bits(&ctx->gb, 2);
		switch (code) {
		case H_SPLIT:
		case V_SPLIT:
			if (parse_bintree(ctx, plane, code, &curr_cell, depth - 1, strip_width))
				return IV3_AVERR;
			break;
		case INTRA_NULL:
			if (!curr_cell.tree) {
				curr_cell.mv_ptr = nullptr;   // MC tree INTRA marker
				curr_cell.tree   = 1;
			} else {
				resync_bitstream(ctx);
				code = get_bits(&ctx->gb, 2);
				if (code >= 2) return IV3_AVERR;
				// code == 1 = SkipCell (not implemented; ffmpeg only logs)
				if (!check_cell(plane, &curr_cell)) return IV3_AVERR;
				if (!curr_cell.mv_ptr) return IV3_AVERR;
				return copy_cell(ctx, plane, &curr_cell);
			}
			break;
		case INTER_DATA:
			if (!curr_cell.tree) {
				if (!ctx->need_resync)
					ctx->next_cell_data = &ctx->gb.base[(get_bits_count(&ctx->gb) + 7) >> 3];
				if (ctx->next_cell_data >= ctx->last_byte) return IV3_AVERR;
				unsigned mv_idx = *(ctx->next_cell_data++);
				if (mv_idx >= ctx->num_vectors) return IV3_AVERR;
				curr_cell.mv_ptr = &ctx->mc_vectors[mv_idx << 1];
				curr_cell.tree   = 1;
				update_bitpos(ctx, 8);
			} else {
				if (!ctx->need_resync)
					ctx->next_cell_data = &ctx->gb.base[(get_bits_count(&ctx->gb) + 7) >> 3];
				if (!check_cell(plane, &curr_cell)) return IV3_AVERR;
				int bytes_used = decode_cell(ctx, plane, &curr_cell, ctx->next_cell_data, ctx->last_byte);
				if (bytes_used < 0) return IV3_AVERR;
				update_bitpos(ctx, bytes_used << 3);
				ctx->next_cell_data += bytes_used;
				return 0;
			}
			break;
		}
	}
	return IV3_AVERR;
}

static int decode_plane(Ctx* ctx, Plane* plane, const uint8_t* data, int data_size, int strip_width) {
	if (data_size < 4) return IV3_AVERR;
	unsigned num_vectors = RL32(data); data += 4; data_size -= 4;
	if (num_vectors > 256) return IV3_AVERR;
	if (num_vectors * 2 > (unsigned)data_size) return IV3_AVERR;

	ctx->num_vectors = num_vectors;
	ctx->mc_vectors  = num_vectors ? (const int8_t*)data : nullptr;

	ctx->gb.base   = &data[num_vectors * 2];
	ctx->gb.bitOff = 0;
	ctx->gb.bitEnd = (data_size - num_vectors * 2) << 3;
	ctx->skip_bits   = 0;
	ctx->need_resync = 0;
	ctx->last_byte   = data + data_size;

	Cell curr_cell;
	curr_cell.xpos = curr_cell.ypos = 0;
	curr_cell.width  = plane->width  >> 2;
	curr_cell.height = plane->height >> 2;
	curr_cell.tree   = 0;
	curr_cell.mv_ptr = nullptr;

	return parse_bintree(ctx, plane, INTRA_NULL, &curr_cell, CELL_STACK_MAX, strip_width);
}

// ── frame headers (OS header + bitstream header + plane offsets) ──
static int valid_dims(int w, int h) {
	return w >= 16 && w <= 640 && h >= 16 && h <= 480 && !(w & 3) && !(h & 3);
}

static int decode_frame_headers(Ctx* ctx, const uint8_t* buf, int buf_size) {
	if (buf_size < 16) return IV3_AVERR;
	const uint8_t* p = buf;
	uint32_t frame_num = RL32(p); p += 4;
	uint32_t word2     = RL32(p); p += 4;
	uint32_t check_sum = RL32(p); p += 4;
	uint32_t data_size = RL32(p); p += 4;
	if ((frame_num ^ word2 ^ data_size ^ IV3_OS_HDR_ID) != check_sum) return IV3_AVERR;

	const uint8_t* bs_hdr = p;
	if (buf_size < 16 + 16) return IV3_AVERR;
	if (RL16(p) != 32) return IV3_AVERR;
	p += 2;

	ctx->frame_num   = frame_num;
	ctx->frame_flags = RL16(p); p += 2;
	ctx->data_size   = (RL32(p) + 7) >> 3; p += 4;
	ctx->cb_offset   = *p++;

	if (ctx->data_size == 16) return 4;
	ctx->data_size = iv3_min(ctx->data_size, buf_size - 16);
	p += 3;   // reserved byte + checksum

	uint16_t height = RL16(p); p += 2;
	uint16_t width  = RL16(p); p += 2;
	if (!valid_dims(width, height)) return IV3_AVERR;

	if (width != ctx->width || height != ctx->height) {
		if (width < 16 || width > 640 || height < 16 || height > 480 || (width & 3) || (height & 3))
			return IV3_AVERR;
		free_frame_buffers(ctx);
		int res = allocate_frame_buffers(ctx, width, height);
		if (res < 0) return res;
	}

	int y_offset = (int32_t)RL32(p); p += 4;
	int v_offset = (int32_t)RL32(p); p += 4;
	int u_offset = (int32_t)RL32(p); p += 4;
	p += 4;

	uint32_t starts[3] = { (uint32_t)y_offset, (uint32_t)v_offset, (uint32_t)u_offset };
	uint32_t ends[3];
	for (int j = 0; j < 3; j++) {
		ends[j] = ctx->data_size;
		for (int i = 2; i >= 0; i--)
			if (starts[i] < ends[j] && starts[i] > starts[j]) ends[j] = starts[i];
	}
	ctx->y_data_size = ends[0] - starts[0];
	ctx->v_data_size = ends[1] - starts[1];
	ctx->u_data_size = ends[2] - starts[2];

	int min_off = y_offset < v_offset ? (y_offset < u_offset ? y_offset : u_offset)
	                                  : (v_offset < u_offset ? v_offset : u_offset);
	int max_off = y_offset > v_offset ? (y_offset > u_offset ? y_offset : u_offset)
	                                  : (v_offset > u_offset ? v_offset : u_offset);
	int min_size = ctx->y_data_size < ctx->v_data_size ? ctx->y_data_size : ctx->v_data_size;
	if (ctx->u_data_size < min_size) min_size = ctx->u_data_size;

	if (min_off < 0 || max_off >= ctx->data_size - 16 ||
	    min_off < (int)(p - bs_hdr) + 16 || min_size <= 0)
		return IV3_AVERR;

	ctx->y_data_ptr = bs_hdr + y_offset;
	ctx->v_data_ptr = bs_hdr + v_offset;
	ctx->u_data_ptr = bs_hdr + u_offset;
	ctx->alt_quant  = p;

	if (ctx->data_size == 16) return 16;   // sync frame
	if (ctx->frame_flags & BS_8BIT_PEL) return IV3_E_8BIT;
	if (ctx->frame_flags & (BS_MV_X_HALF | BS_MV_Y_HALF)) return IV3_E_HALFPEL;
	return 0;
}

// ── public-ish API used by codec_pack.cpp ──
static Ctx* open(int width, int height) {
	Ctx* ctx = (Ctx*)calloc(1, sizeof(Ctx));
	if (!ctx) return nullptr;
	build_requant_tab();
	if (width > 0 && height > 0) {
		if (allocate_frame_buffers(ctx, width, height) < 0) { free(ctx); return nullptr; }
	}
	return ctx;
}

static void close(Ctx* ctx) {
	if (!ctx) return;
	free_frame_buffers(ctx);
	free(ctx);
}

static void reset(Ctx* ctx) {
	if (!ctx) return;
	for (int p = 0; p < 3; p++)
		for (int b = 0; b < 2; b++)
			if (ctx->planes[p].pixels[b])
				memset(ctx->planes[p].pixels[b], 0, (size_t)ctx->planes[p].pitch * ctx->planes[p].height);
}

// Returns 0 on success (sets got_frame), negative on hard error.
static int decode(Ctx* ctx, const uint8_t* buf, int buf_size) {
	ctx->got_frame = 0;
	int res = decode_frame_headers(ctx, buf, buf_size);
	if (res < 0) return res;
	ctx->buf_sel = (ctx->frame_flags >> BS_BUFFER) & 1;
	if (res) return 0;   // sync/null frame: no plane data, leave previous picture

	if (!ctx->planes[0].buffers[0]) return IV3_AVERR;

	// Decode Y, U, V. On a per-plane error we still output the partial picture
	// (partial > nothing), matching the JS oracle's per-plane try/continue, but
	// flag it so the caller can surface a "corrupt frame" warning.
	ctx->plane_failed = 0;
	if (decode_plane(ctx, &ctx->planes[0], ctx->y_data_ptr, ctx->y_data_size, 40) < 0) ctx->plane_failed = 1;
	if (decode_plane(ctx, &ctx->planes[1], ctx->u_data_ptr, ctx->u_data_size, 10) < 0) ctx->plane_failed = 1;
	if (decode_plane(ctx, &ctx->planes[2], ctx->v_data_ptr, ctx->v_data_size, 10) < 0) ctx->plane_failed = 1;

	ctx->got_frame = 1;
	return 0;
}

// Write the three planes (7-bit -> 8-bit via <<1) to the YCbCr output buffers.
// Indeo3 plane order is Y, U, V (no YVU9 swap): Cb = planes[1], Cr = planes[2].
static void output_plane_to(const Plane* plane, int buf_sel, uint8_t* dst, int out_w, int out_h) {
	const uint8_t* src = plane->pixels[buf_sel];
	ptrdiff_t pitch = plane->pitch;
	int w = iv3_min(out_w, plane->width);
	int h = iv3_min(out_h, plane->height);
	for (int y = 0; y < h; y++) {
		for (int x = 0; x < w; x++) dst[x] = (uint8_t)((src[x] & 0x7F) << 1);
		src += pitch;
		dst += out_w;
	}
}

static void output(Ctx* ctx, uint8_t* y, int yw, int yh,
                   uint8_t* cb, uint8_t* cr, int cw, int ch) {
	output_plane_to(&ctx->planes[0], ctx->buf_sel, y, yw, yh);
	output_plane_to(&ctx->planes[1], ctx->buf_sel, cb, cw, ch);
	output_plane_to(&ctx->planes[2], ctx->buf_sel, cr, cw, ch);
}

} // namespace iv3

#endif // IV3_DECODE_H
