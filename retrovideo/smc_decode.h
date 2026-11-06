/*
	Apple Graphics (smc::): a port of FFmpeg's libavcodec/smc.c (release/8.1, LGPL).
	Included by codec_pack.cpp after RvCtx.

	Paints 4x4 blocks of palette indices over the persistent picture, skipped blocks
	keep the previous frame. The last block of a row or column can reach past the
	picture, so it decodes into RvCtx.blockPic (see block_pic_finish in codec_pack.cpp).

	One departure: FFmpeg finds the previous blocks across a row start from the width,
	which is right only when its line size equals a width that is a multiple of 4; here
	they are the last blocks of the previous row at any width.
*/

#ifndef RV_SMC_DECODE_H
#define RV_SMC_DECODE_H

namespace smc {

enum { CPAIR = 2, CQUAD = 4, COCTET = 8, COLORS_PER_TABLE = 256 };

static inline int block_count(int opcode, rle::ByteReader* gb) {
	return (opcode & 0x10) ? 1 + gb->byte() : 1 + (opcode & 0x0F);
}

static inline uint32_t be32(rle::ByteReader* gb) {
	const uint32_t a = (uint32_t)gb->be16();
	return (a << 16) | (uint32_t)gb->be16();
}

// Next entry of a color table: read from the stream and stored (a "new colors" opcode),
// or a stored one by index. Returns the entry's base in `table`.
static int table_entry(rle::ByteReader* gb, bool isNew, uint8_t* table, int entrySize, int* nextEntry) {
	if (!isNew) return entrySize * gb->byte();
	const int base = entrySize * *nextEntry;
	for (int i = 0; i < entrySize; i++) table[base + i] = (uint8_t)gb->byte();
	*nextEntry = (*nextEntry + 1) % COLORS_PER_TABLE;
	return base;
}

static inline void copy_block(uint8_t* pixels, int block_ptr, int prev_block_ptr, int row_inc) {
	for (int y = 0; y < 4; y++) {
		for (int x = 0; x < 4; x++) pixels[block_ptr++] = pixels[prev_block_ptr++];
		block_ptr += row_inc;
		prev_block_ptr += row_inc;
	}
}

static int decode_stream(RvCtx* s, rle::ByteReader* gb) {
	const int width = s->width;
	const int stride = s->blockStride;
	const int row_inc = stride - 4;
	uint8_t* const pixels = s->blockPic;
	const int lastColumn = ((width + 3) / 4 - 1) * 4;

	int row_ptr = 0, pixel_ptr = 0, block_ptr;
	int n_blocks;
	int color_pair_index = 0, color_quad_index = 0, color_octet_index = 0;
	int total_blocks = ((width + 3) / 4) * ((s->height + 3) / 4);

	gb->skip(4);   // a byte and the chunk size; the size from the container wins

	// False when the blocks ran out with n_blocks still to paint.
	const auto advance_block = [&]() {
		pixel_ptr += 4;
		if (pixel_ptr >= width) {
			pixel_ptr = 0;
			row_ptr += stride * 4;
		}
		total_blocks--;
		return total_blocks >= (n_blocks != 0 ? 1 : 0);
	};

	while (total_blocks) {
		if (gb->left() < 1) return AVERROR_INVALIDDATA;
		const int opcode = gb->byte();

		switch (opcode & 0xF0) {
		case 0x00:   // skip n blocks
		case 0x10:
			n_blocks = block_count(opcode, gb);
			while (n_blocks--) {
				if (!advance_block()) return AVERROR_INVALIDDATA;
			}
			break;

		case 0x20:   // repeat the last block n times
		case 0x30: {
			n_blocks = block_count(opcode, gb);
			if (row_ptr == 0 && pixel_ptr == 0) return AVERROR_INVALIDDATA;

			const int prev = pixel_ptr == 0
				? row_ptr - stride * 4 + lastColumn
				: row_ptr + pixel_ptr - 4;
			while (n_blocks--) {
				copy_block(pixels, row_ptr + pixel_ptr, prev, row_inc);
				if (!advance_block()) return AVERROR_INVALIDDATA;
			}
			break;
		}

		case 0x40:   // repeat the previous pair of blocks n times
		case 0x50: {
			n_blocks = block_count(opcode, gb) * 2;
			if (row_ptr == 0 && pixel_ptr < 2 * 4) return AVERROR_INVALIDDATA;

			const int prevRow = row_ptr - stride * 4;
			int prev1, prev2;
			if (pixel_ptr == 0) {
				prev1 = prevRow + lastColumn - 4;
				prev2 = prevRow + lastColumn;
			} else if (pixel_ptr == 4) {
				prev1 = prevRow + lastColumn;
				prev2 = row_ptr;
			} else {
				prev1 = row_ptr + pixel_ptr - 4 * 2;
				prev2 = row_ptr + pixel_ptr - 4;
			}
			if (prev1 < 0) return AVERROR_INVALIDDATA;   // a one block wide picture

			bool second = false;
			while (n_blocks--) {
				copy_block(pixels, row_ptr + pixel_ptr, second ? prev2 : prev1, row_inc);
				second = !second;
				if (!advance_block()) return AVERROR_INVALIDDATA;
			}
			break;
		}

		case 0x60:   // 1-color blocks
		case 0x70: {
			n_blocks = block_count(opcode, gb);
			const int pixel = gb->byte();
			while (n_blocks--) {
				block_ptr = row_ptr + pixel_ptr;
				for (int y = 0; y < 4; y++) {
					for (int x = 0; x < 4; x++) pixels[block_ptr++] = (uint8_t)pixel;
					block_ptr += row_inc;
				}
				if (!advance_block()) return AVERROR_INVALIDDATA;
			}
			break;
		}

		case 0x80:   // 2-color blocks
		case 0x90: {
			n_blocks = (opcode & 0x0F) + 1;
			const int base = table_entry(gb, (opcode & 0xF0) == 0x80, s->smcPairs, CPAIR, &color_pair_index);
			while (n_blocks--) {
				const unsigned color_flags = (unsigned)gb->be16();
				unsigned flag_mask = 0x8000;
				block_ptr = row_ptr + pixel_ptr;
				for (int y = 0; y < 4; y++) {
					for (int x = 0; x < 4; x++) {
						pixels[block_ptr++] = s->smcPairs[base + ((color_flags & flag_mask) ? 1 : 0)];
						flag_mask >>= 1;
					}
					block_ptr += row_inc;
				}
				if (!advance_block()) return AVERROR_INVALIDDATA;
			}
			break;
		}

		case 0xA0:   // 4-color blocks
		case 0xB0: {
			n_blocks = (opcode & 0x0F) + 1;
			const int base = table_entry(gb, (opcode & 0xF0) == 0xA0, s->smcQuads, CQUAD, &color_quad_index);
			while (n_blocks--) {
				const uint32_t color_flags = be32(gb);
				int shift = 30;
				block_ptr = row_ptr + pixel_ptr;
				for (int y = 0; y < 4; y++) {
					for (int x = 0; x < 4; x++) {
						pixels[block_ptr++] = s->smcQuads[base + ((color_flags >> shift) & 0x03)];
						shift -= 2;
					}
					block_ptr += row_inc;
				}
				if (!advance_block()) return AVERROR_INVALIDDATA;
			}
			break;
		}

		case 0xC0:   // 8-color blocks
		case 0xD0: {
			n_blocks = (opcode & 0x0F) + 1;
			const int base = table_entry(gb, (opcode & 0xF0) == 0xC0, s->smcOctets, COCTET, &color_octet_index);
			while (n_blocks--) {
				// Hex bytes 01 23 45 67 89 AB give flags_a xx012456, flags_b xx89A37B.
				const unsigned val1 = (unsigned)gb->be16();
				const unsigned val2 = (unsigned)gb->be16();
				const unsigned val3 = (unsigned)gb->be16();
				const unsigned color_flags_a = ((val1 & 0xFFF0) << 8) | (val2 >> 4);
				const unsigned color_flags_b = ((val3 & 0xFFF0) << 8) |
					((val1 & 0x0F) << 8) | ((val2 & 0x0F) << 4) | (val3 & 0x0F);

				unsigned color_flags = color_flags_a;
				int shift = 21;
				block_ptr = row_ptr + pixel_ptr;
				for (int y = 0; y < 4; y++) {
					if (y == 2) {
						color_flags = color_flags_b;
						shift = 21;
					}
					for (int x = 0; x < 4; x++) {
						pixels[block_ptr++] = s->smcOctets[base + ((color_flags >> shift) & 0x07)];
						shift -= 3;
					}
					block_ptr += row_inc;
				}
				if (!advance_block()) return AVERROR_INVALIDDATA;
			}
			break;
		}

		default:     // 16-color blocks, every pixel its own color
			n_blocks = (opcode & 0x0F) + 1;
			while (n_blocks--) {
				block_ptr = row_ptr + pixel_ptr;
				for (int y = 0; y < 4; y++) {
					for (int x = 0; x < 4; x++) pixels[block_ptr++] = (uint8_t)gb->byte();
					block_ptr += row_inc;
				}
				if (!advance_block()) return AVERROR_INVALIDDATA;
			}
			break;
		}
	}
	return 0;
}

static int decode(RvCtx* s, const uint8_t* buf, int buf_size) {
	const int total_blocks = ((s->width + 3) / 4) * ((s->height + 3) / 4);
	if (total_blocks / 1024 > buf_size) return AVERROR_INVALIDDATA;
	rle::ByteReader gb = { buf, buf + buf_size };
	const int r = decode_stream(s, &gb);
	if (r == 0) block_pic_finish(s);
	return r;
}

} // namespace smc

#endif
