/*
	Apple Video (rpza::): a port of FFmpeg's libavcodec/rpza.c (release/8.1, LGPL).
	Included by codec_pack.cpp after RvCtx.

	Paints 4x4 blocks over the persistent picture, skipped blocks keep the previous
	frame. The last block of a row or column can reach past the picture, so it decodes
	into RvCtx.blockPic (see block_pic_finish in codec_pack.cpp). RGB555 in host order.
*/

#ifndef RV_RPZA_DECODE_H
#define RV_RPZA_DECODE_H

namespace rpza {

static int decode_stream(RvCtx* s, rle::ByteReader* gb) {
	const int width = s->width;
	const int stride = s->blockStride / 2;
	const int row_inc = stride - 4;
	uint16_t* const pixels = (uint16_t*)s->blockPic;
	uint16_t colorA = 0, colorB;
	uint16_t color4[4];
	int row_ptr = 0, pixel_ptr = 0, block_ptr;

	gb->skip(4);   // 0xe1 and the chunk size; the size from the container wins

	int total_blocks = ((width + 3) / 4) * ((s->height + 3) / 4);
	if (total_blocks / 32 > gb->left()) return AVERROR_INVALIDDATA;

	const auto advance_block = [&]() {
		pixel_ptr += 4;
		if (pixel_ptr >= width) {
			pixel_ptr = 0;
			row_ptr += stride * 4;
		}
		total_blocks--;
	};

	while (gb->left() > 0) {
		int opcode = gb->byte();
		int n_blocks = (opcode & 0x1f) + 1;

		// Most significant bit clear: more data decides what this is.
		if ((opcode & 0x80) == 0) {
			colorA = (uint16_t)((opcode << 8) | gb->byte());
			opcode = 0;
			if (gb->left() > 0 && (*gb->p & 0x80) != 0) {
				// Behaves as 110xxxxx with the colorA above; 0x20 enters the switch there.
				opcode = 0x20;
				n_blocks = 1;
			}
		}

		n_blocks = FFMIN(n_blocks, total_blocks);

		switch (opcode & 0xe0) {
		case 0x80:   // skip blocks
			while (n_blocks--) {
				if (total_blocks < 1) return AVERROR_INVALIDDATA;
				advance_block();
			}
			break;

		case 0xa0:   // fill blocks with one color
			colorA = (uint16_t)gb->be16();
			while (n_blocks--) {
				if (total_blocks < 1) return AVERROR_INVALIDDATA;
				block_ptr = row_ptr + pixel_ptr;
				for (int y = 0; y < 4; y++) {
					for (int x = 0; x < 4; x++) pixels[block_ptr++] = colorA;
					block_ptr += row_inc;
				}
				advance_block();
			}
			break;

		case 0xc0:   // fill blocks with 4 colors
			colorA = (uint16_t)gb->be16();
			// fall through
		case 0x20: {
			colorB = (uint16_t)gb->be16();

			color4[0] = colorB;
			color4[1] = 0;
			color4[2] = 0;
			color4[3] = colorA;

			for (int shift = 10; shift >= 0; shift -= 5) {
				const int ta = (colorA >> shift) & 0x1F;
				const int tb = (colorB >> shift) & 0x1F;
				color4[1] |= ((11 * ta + 21 * tb) >> 5) << shift;
				color4[2] |= ((21 * ta + 11 * tb) >> 5) << shift;
			}

			if (gb->left() < n_blocks * 4) return AVERROR_INVALIDDATA;
			while (n_blocks--) {
				if (total_blocks < 1) return AVERROR_INVALIDDATA;
				block_ptr = row_ptr + pixel_ptr;
				for (int y = 0; y < 4; y++) {
					const int index = gb->byte();
					for (int x = 0; x < 4; x++) {
						pixels[block_ptr++] = color4[(index >> (2 * (3 - x))) & 0x03];
					}
					block_ptr += row_inc;
				}
				advance_block();
			}
			break;
		}

		case 0x00:   // fill a block with 16 colors
			if (gb->left() < 30) return AVERROR_INVALIDDATA;
			if (total_blocks < 1) return AVERROR_INVALIDDATA;
			block_ptr = row_ptr + pixel_ptr;
			for (int y = 0; y < 4; y++) {
				for (int x = 0; x < 4; x++) {
					// The upper left pixel's color is already read.
					if (y != 0 || x != 0) colorA = (uint16_t)gb->be16();
					pixels[block_ptr++] = colorA;
				}
				block_ptr += row_inc;
			}
			advance_block();
			break;

		default:
			return AVERROR_INVALIDDATA;
		}
	}
	return 0;
}

static int decode(RvCtx* s, const uint8_t* buf, int buf_size) {
	rle::ByteReader gb = { buf, buf + buf_size };
	const int r = decode_stream(s, &gb);
	if (r == 0) block_pic_finish(s);
	return r;
}

} // namespace rpza

#endif
