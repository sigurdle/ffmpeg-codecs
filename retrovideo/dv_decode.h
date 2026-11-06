/*
	DV (dv::): a port of FFmpeg's libavcodec/dvdec.c, dv.c, dv_internal.h, dv_profile.c and the
	8 bit kernels of simple_idct.c and simple_idct_template.c (release/8.1, LGPL). Included by
	codec_pack.cpp after RvCtx.

	Every frame is whole and says what it is: its header picks the profile, as FFmpeg's does, so
	the container's tag and dimensions only matter for the SL25 case. All of FFmpeg's profiles
	are here: DV25 (4:2:0 and 4:1:1), DVCPRO50 (4:2:2), DVCPRO HD (4:2:2).

	Output: each decode fills RvCtx.frame with Y then interleaved Cb,Cr. There is no 4:1:1 output
	format, so 4:1:1 chroma is doubled into 4:2:2. RvCtx.format says which came out.

	Differs from FFmpeg in structure only: no threads, no lowres, the VLC table is built here
	rather than by ff_vlc_init_from_lengths, and error concealment runs as at its default (on).
*/

#ifndef RV_DV_DECODE_H
#define RV_DV_DECODE_H

#include "dv_tables.h"

namespace dv {

enum { PIX_411, PIX_420, PIX_422 };

static const uint8_t block_sizes_dv2550[8] = { 112, 112, 112, 112, 80, 80, 0, 0 };
static const uint8_t block_sizes_dv100[8]  = { 80, 80, 80, 80, 80, 80, 64, 64 };

struct Profile {
	int dsf, video_stype, frame_size, difseg_size, n_difchan, height, width, pix_fmt, bpm;
	const uint8_t* block_sizes;
};

static const Profile profiles[] = {
	{ 0, 0x00, 120000, 10, 1,  480,  720, PIX_411, 6, block_sizes_dv2550 },   // 525/60
	{ 1, 0x00, 144000, 12, 1,  576,  720, PIX_420, 6, block_sizes_dv2550 },   // 625/50
	{ 1, 0x00, 144000, 12, 1,  576,  720, PIX_411, 6, block_sizes_dv2550 },   // 625/50, SMPTE 314M
	{ 0, 0x04, 240000, 10, 2,  480,  720, PIX_422, 6, block_sizes_dv2550 },   // DVCPRO50 525/60
	{ 1, 0x04, 288000, 12, 2,  576,  720, PIX_422, 6, block_sizes_dv2550 },   // DVCPRO50 625/50
	{ 0, 0x14, 480000, 10, 4, 1080, 1280, PIX_422, 8, block_sizes_dv100 },    // 1080i60
	{ 1, 0x14, 576000, 12, 4, 1080, 1440, PIX_422, 8, block_sizes_dv100 },    // 1080i50
	{ 0, 0x18, 240000, 10, 2,  720,  960, PIX_422, 8, block_sizes_dv100 },    // 720p60
	{ 1, 0x18, 288000, 12, 2,  720,  960, PIX_422, 8, block_sizes_dv100 },    // 720p50
	{ 1, 0x01, 144000, 12, 1,  576,  720, PIX_420, 6, block_sizes_dv2550 },   // IEC 61883-5 625/50
};
enum { PROFILE_COUNT = sizeof(profiles) / sizeof(profiles[0]) };

static inline bool is_hd(const Profile* p)      { return (p->video_stype & 0x10) != 0; }
static inline bool is_1080i50(const Profile* p) { return p->video_stype == 0x14 && p->dsf == 1; }
static inline bool is_720p50(const Profile* p)  { return p->video_stype == 0x18 && p->dsf == 1; }

// ff_dv_frame_profile. sl25: the stream's tag is SL25 and it is 720x576.
static const Profile* frame_profile(const Profile* sys, const uint8_t* frame, int buf_size, bool sl25) {
	if (buf_size < 6 * 80) return nullptr;

	const int dsf   = (frame[3] & 0x80) >> 7;
	const int stype = frame[80 * 5 + 48 + 3] & 0x1f;
	const int pal   = !!(frame[80 * 5 + 48 + 3] & 0x20);

	// 576i50 25Mbps 4:1:1 is a special case
	if ((dsf == 1 && stype == 0 && (frame[4] & 0x07)) || (stype == 31 && sl25)) return &profiles[2];

	// PAL files with a dsf flag of 0: FFmpeg trac #8333 and #2177
	if (dsf == 0 && pal == 1 && stype == profiles[1].video_stype && buf_size == profiles[1].frame_size) return &profiles[1];

	for (int i = 0; i < PROFILE_COUNT; i++) {
		if (dsf == profiles[i].dsf && stype == profiles[i].video_stype) return &profiles[i];
	}

	// corrupted input the size of the last frame
	if (sys && buf_size == sys->frame_size) return sys;

	// QuickTime 3: FFmpeg trac #217
	if ((frame[3] & 0x7f) == 0x3f && frame[80 * 5 + 48 + 3] == 0xff) return &profiles[dsf];

	return nullptr;
}

struct WorkChunk {
	uint16_t buf_offset;
	uint16_t mb_coordinates[5];
};

// dv_calc_mb_coordinates
static void calc_mb_coordinates(const Profile* d, int chan, int seq, int slot, uint16_t* tbl) {
	static const uint8_t off[]   = {  2,  6,  8, 0,  4 };
	static const uint8_t shuf1[] = { 36, 18, 54, 0, 72 };
	static const uint8_t shuf2[] = { 24, 12, 36, 0, 48 };
	static const uint8_t shuf3[] = { 18,  9, 27, 0, 36 };

	static const uint8_t l_start[]          = { 0, 4, 9, 13, 18, 22, 27, 31, 36, 40 };
	static const uint8_t l_start_shuffled[] = { 9, 4, 13, 0, 18 };

	static const uint8_t serpent1[] = {
		0, 1, 2, 2, 1, 0,
		0, 1, 2, 2, 1, 0,
		0, 1, 2, 2, 1, 0,
		0, 1, 2, 2, 1, 0,
		0, 1, 2
	};
	static const uint8_t serpent2[] = {
		0, 1, 2, 3, 4, 5, 5, 4, 3, 2, 1, 0,
		0, 1, 2, 3, 4, 5, 5, 4, 3, 2, 1, 0,
		0, 1, 2, 3, 4, 5
	};

	static const uint8_t remap[][2] = {
		{  0,  0 }, {  0,  0 }, {  0,  0 }, {  0,  0 }, /* dummy */
		{  0,  0 }, {  0,  1 }, {  0,  2 }, {  0,  3 }, { 10,  0 },
		{ 10,  1 }, { 10,  2 }, { 10,  3 }, { 20,  0 }, { 20,  1 },
		{ 20,  2 }, { 20,  3 }, { 30,  0 }, { 30,  1 }, { 30,  2 },
		{ 30,  3 }, { 40,  0 }, { 40,  1 }, { 40,  2 }, { 40,  3 },
		{ 50,  0 }, { 50,  1 }, { 50,  2 }, { 50,  3 }, { 60,  0 },
		{ 60,  1 }, { 60,  2 }, { 60,  3 }, { 70,  0 }, { 70,  1 },
		{ 70,  2 }, { 70,  3 }, {  0, 64 }, {  0, 65 }, {  0, 66 },
		{ 10, 64 }, { 10, 65 }, { 10, 66 }, { 20, 64 }, { 20, 65 },
		{ 20, 66 }, { 30, 64 }, { 30, 65 }, { 30, 66 }, { 40, 64 },
		{ 40, 65 }, { 40, 66 }, { 50, 64 }, { 50, 65 }, { 50, 66 },
		{ 60, 64 }, { 60, 65 }, { 60, 66 }, { 70, 64 }, { 70, 65 },
		{ 70, 66 }, {  0, 67 }, { 20, 67 }, { 40, 67 }, { 60, 67 }
	};

	int i, k, x, y, blk;

	for (int m = 0; m < 5; m++) {
		switch (d->width) {
		case 1440:
			blk = (chan * 11 + seq) * 27 + slot;

			if (chan == 0 && seq == 11) {
				x = m * 27 + slot;
				if (x < 90) {
					y = 0;
				} else {
					x = (x - 90) * 2;
					y = 67;
				}
			} else {
				i = (4 * chan + blk + off[m]) % 11;
				k = (blk / 11) % 27;

				x = shuf1[m] + (chan & 1) * 9 + k % 9;
				y = (i * 3 + k / 9) * 2 + (chan >> 1) + 1;
			}
			tbl[m] = (x << 1) | (y << 9);
			break;
		case 1280:
			blk = (chan * 10 + seq) * 27 + slot;

			i = (4 * chan + (seq / 5) + 2 * blk + off[m]) % 10;
			k = (blk / 5) % 27;

			x = shuf1[m] + (chan & 1) * 9 + k % 9;
			y = (i * 3 + k / 9) * 2 + (chan >> 1) + 4;

			if (x >= 80) {
				x = remap[y][0] + ((x - 80) << (y > 59));
				y = remap[y][1];
			}
			tbl[m] = (x << 1) | (y << 9);
			break;
		case 960:
			blk = (chan * 10 + seq) * 27 + slot;

			i = (4 * chan + (seq / 5) + 2 * blk + off[m]) % 10;
			k = (blk / 5) % 27 + (i & 1) * 3;

			x      = shuf2[m]   + k % 6 +  6 * (chan  & 1);
			y      = l_start[i] + k / 6 + 45 * (chan >> 1);
			tbl[m] = (x << 1) | (y << 9);
			break;
		case 720:
			switch (d->pix_fmt) {
			case PIX_422:
				x = shuf3[m] + slot / 3;
				y = serpent1[slot] + ((((seq + off[m]) % d->difseg_size) << 1) + chan) * 3;
				tbl[m] = (x << 1) | (y << 8);
				break;
			case PIX_420:
				x = shuf3[m] + slot / 3;
				y = serpent1[slot] + ((seq + off[m]) % d->difseg_size) * 3;
				tbl[m] = (x << 1) | (y << 9);
				break;
			case PIX_411:
				i = (seq + off[m]) % d->difseg_size;
				k = slot + ((m == 1 || m == 2) ? 3 : 0);

				x = l_start_shuffled[m] + k / 6;
				y = serpent2[k] + i * 6;
				if (x > 21) y = y * 2 - i * 6;
				tbl[m] = (x << 2) | (y << 8);
				break;
			}
			break;
		default:
			break;
		}
	}
}

// ff_dv_init_dynamic_tables
static void init_dynamic_tables(WorkChunk* work_chunks, const Profile* d) {
	int p = 0, i = 0;
	for (int c = 0; c < d->n_difchan; c++) {
		for (int s = 0; s < d->difseg_size; s++) {
			p += 6;
			for (int j = 0; j < 27; j++) {
				p += !(j % 3);
				if (!(is_1080i50(d) && c != 0 && s == 11) && !(is_720p50(d) && s > 9)) {
					calc_mb_coordinates(d, c, s, j, &work_chunks[i].mb_coordinates[0]);
					work_chunks[i++].buf_offset = p;
				}
				p += 5;
			}
		}
	}
}

static inline int work_pool_size(const Profile* d) {
	int size = d->n_difchan * d->difseg_size * 27;
	if (is_1080i50(d)) size -= 3 * 27;
	if (is_720p50(d))  size -= 4 * 27;
	return size;
}

// The IDCTs: simple_idct_template.c at 8 bits.

enum { W1 = 22725, W2 = 21407, W3 = 19266, W4 = 16383, W5 = 12873, W6 = 8867, W7 = 4520 };
enum { ROW_SHIFT = 11, COL_SHIFT = 20, DC_SHIFT = 3 };

static inline void idct_row(int16_t* row) {
	if (!(row[1] | row[2] | row[3] | row[4] | row[5] | row[6] | row[7])) {
		const int16_t dc = (int16_t)(row[0] * (1 << DC_SHIFT));
		for (int i = 0; i < 8; i++) row[i] = dc;
		return;
	}

	int a0 = W4 * row[0] + (1 << (ROW_SHIFT - 1));
	int a1 = a0, a2 = a0, a3 = a0;

	a0 += W2 * row[2];
	a1 += W6 * row[2];
	a2 -= W6 * row[2];
	a3 -= W2 * row[2];

	int b0 = W1 * row[1] + W3 * row[3];
	int b1 = W3 * row[1] - W7 * row[3];
	int b2 = W5 * row[1] - W1 * row[3];
	int b3 = W7 * row[1] - W5 * row[3];

	if (row[4] | row[5] | row[6] | row[7]) {
		a0 +=  W4 * row[4] + W6 * row[6];
		a1 += -W4 * row[4] - W2 * row[6];
		a2 += -W4 * row[4] + W2 * row[6];
		a3 +=  W4 * row[4] - W6 * row[6];

		b0 +=  W5 * row[5] + W7 * row[7];
		b1 += -W1 * row[5] - W5 * row[7];
		b2 +=  W7 * row[5] + W3 * row[7];
		b3 +=  W3 * row[5] - W1 * row[7];
	}

	row[0] = (int16_t)((a0 + b0) >> ROW_SHIFT);
	row[7] = (int16_t)((a0 - b0) >> ROW_SHIFT);
	row[1] = (int16_t)((a1 + b1) >> ROW_SHIFT);
	row[6] = (int16_t)((a1 - b1) >> ROW_SHIFT);
	row[2] = (int16_t)((a2 + b2) >> ROW_SHIFT);
	row[5] = (int16_t)((a2 - b2) >> ROW_SHIFT);
	row[3] = (int16_t)((a3 + b3) >> ROW_SHIFT);
	row[4] = (int16_t)((a3 - b3) >> ROW_SHIFT);
}

// One column's a[] and b[]: the column stage both the put and the plain IDCT share.
static inline void idct_col_terms(const int16_t* col, int* a, int* b) {
	a[0] = W4 * (col[8 * 0] + ((1 << (COL_SHIFT - 1)) / W4));
	a[1] = a[0];
	a[2] = a[0];
	a[3] = a[0];

	a[0] += W2 * col[8 * 2];
	a[1] += W6 * col[8 * 2];
	a[2] -= W6 * col[8 * 2];
	a[3] -= W2 * col[8 * 2];

	b[0] = W1 * col[8 * 1] + W3 * col[8 * 3];
	b[1] = W3 * col[8 * 1] - W7 * col[8 * 3];
	b[2] = W5 * col[8 * 1] - W1 * col[8 * 3];
	b[3] = W7 * col[8 * 1] - W5 * col[8 * 3];

	if (col[8 * 4]) {
		a[0] += W4 * col[8 * 4];
		a[1] -= W4 * col[8 * 4];
		a[2] -= W4 * col[8 * 4];
		a[3] += W4 * col[8 * 4];
	}
	if (col[8 * 5]) {
		b[0] += W5 * col[8 * 5];
		b[1] -= W1 * col[8 * 5];
		b[2] += W7 * col[8 * 5];
		b[3] += W3 * col[8 * 5];
	}
	if (col[8 * 6]) {
		a[0] += W6 * col[8 * 6];
		a[1] -= W2 * col[8 * 6];
		a[2] += W2 * col[8 * 6];
		a[3] -= W6 * col[8 * 6];
	}
	if (col[8 * 7]) {
		b[0] += W7 * col[8 * 7];
		b[1] -= W5 * col[8 * 7];
		b[2] += W3 * col[8 * 7];
		b[3] -= W1 * col[8 * 7];
	}
}

static void simple_idct_put(uint8_t* dest, int line_size, int16_t* block) {
	for (int i = 0; i < 8; i++) idct_row(block + i * 8);

	for (int i = 0; i < 8; i++) {
		int a[4], b[4];
		idct_col_terms(block + i, a, b);
		uint8_t* p = dest + i;
		p[0]             = av_clip_uint8((a[0] + b[0]) >> COL_SHIFT);
		p[line_size]     = av_clip_uint8((a[1] + b[1]) >> COL_SHIFT);
		p[line_size * 2] = av_clip_uint8((a[2] + b[2]) >> COL_SHIFT);
		p[line_size * 3] = av_clip_uint8((a[3] + b[3]) >> COL_SHIFT);
		p[line_size * 4] = av_clip_uint8((a[3] - b[3]) >> COL_SHIFT);
		p[line_size * 5] = av_clip_uint8((a[2] - b[2]) >> COL_SHIFT);
		p[line_size * 6] = av_clip_uint8((a[1] - b[1]) >> COL_SHIFT);
		p[line_size * 7] = av_clip_uint8((a[0] - b[0]) >> COL_SHIFT);
	}
}

// The plain IDCT, in place: DVCPRO HD's short last row.
static void simple_idct(int16_t* block) {
	for (int i = 0; i < 8; i++) idct_row(block + i * 8);

	for (int i = 0; i < 8; i++) {
		int a[4], b[4];
		int16_t* col = block + i;
		idct_col_terms(col, a, b);
		col[0]  = (int16_t)((a[0] + b[0]) >> COL_SHIFT);
		col[8]  = (int16_t)((a[1] + b[1]) >> COL_SHIFT);
		col[16] = (int16_t)((a[2] + b[2]) >> COL_SHIFT);
		col[24] = (int16_t)((a[3] + b[3]) >> COL_SHIFT);
		col[32] = (int16_t)((a[3] - b[3]) >> COL_SHIFT);
		col[40] = (int16_t)((a[2] - b[2]) >> COL_SHIFT);
		col[48] = (int16_t)((a[1] - b[1]) >> COL_SHIFT);
		col[56] = (int16_t)((a[0] - b[0]) >> COL_SHIFT);
	}
}

// The 2-4-8 IDCT: simple_idct.c.

enum { CN_SHIFT = 12, C1 = 2676, C2 = 1108, C_SHIFT = 4 + 1 + 12 };   // C1, C2: 0.6532814824, 0.2705980501 fixed

static inline void idct4col_put(uint8_t* dest, int line_size, const int16_t* col) {
	const int a0 = col[8 * 0], a1 = col[8 * 2], a2 = col[8 * 4], a3 = col[8 * 6];
	const int c0 = ((a0 + a2) * (1 << (CN_SHIFT - 1))) + (1 << (C_SHIFT - 1));
	const int c2 = ((a0 - a2) * (1 << (CN_SHIFT - 1))) + (1 << (C_SHIFT - 1));
	const int c1 = a1 * C1 + a3 * C2;
	const int c3 = a1 * C2 - a3 * C1;
	dest[0]             = av_clip_uint8((c0 + c1) >> C_SHIFT);
	dest[line_size]     = av_clip_uint8((c2 + c3) >> C_SHIFT);
	dest[line_size * 2] = av_clip_uint8((c2 - c3) >> C_SHIFT);
	dest[line_size * 3] = av_clip_uint8((c0 - c1) >> C_SHIFT);
}

// The input must be interlaced: the butterfly splits it into the fields' sum and difference.
static void simple_idct248_put(uint8_t* dest, int line_size, int16_t* block) {
	int16_t* ptr = block;
	for (int i = 0; i < 4; i++) {
		for (int k = 0; k < 8; k++) {
			const int a0 = ptr[k], a1 = ptr[8 + k];
			ptr[k]     = (int16_t)(a0 + a1);
			ptr[8 + k] = (int16_t)(a0 - a1);
		}
		ptr += 2 * 8;
	}

	for (int i = 0; i < 8; i++) idct_row(block + i * 8);

	for (int i = 0; i < 8; i++) {
		idct4col_put(dest + i, 2 * line_size, block + i);
		idct4col_put(dest + line_size + i, 2 * line_size, block + 8 + i);
	}
}

static inline void idct_put(bool idct248, uint8_t* dest, int line_size, int16_t* block) {
	if (idct248) simple_idct248_put(dest, line_size, block);
	else simple_idct_put(dest, line_size, block);
}

// The AC code.

enum { TEX_VLC_BITS = 10, NB_VLC = 409, VLC_TABLE_SIZE = 1664, IWEIGHT_BITS = 14 };

struct RlVlcElem {
	int16_t level;
	int8_t  len8;
	uint8_t run;
};

static RlVlcElem rl_vlc[VLC_TABLE_SIZE];

// The table dv_init_static gets from ff_vlc_init_from_lengths. As in FFmpeg no code is unused, and
// a code past TEX_VLC_BITS consumes the depth of its subtable: the one 15 bit code, run 0 level 0,
// shares a subtable with 16 bit ones and so takes 16 bits, as it does there.
static void init_vlc() {
	if (rl_vlc[0].len8 != 0) return;

	uint8_t  subBits[1 << TEX_VLC_BITS];
	uint16_t subStart[1 << TEX_VLC_BITS];
	memset(subBits, 0, sizeof(subBits));

	for (int pass = 0; pass < 2; pass++) {
		uint32_t code = 0;
		for (int i = 0; i < NB_VLC; i++) {
			for (int sign = 0; sign < (vlc_level[i] ? 2 : 1); sign++) {
				const int len    = vlc_len[i] + (vlc_level[i] ? 1 : 0);
				const int prefix = code >> (32 - TEX_VLC_BITS);
				if (pass == 0) {
					if (len > TEX_VLC_BITS && len - TEX_VLC_BITS > subBits[prefix]) subBits[prefix] = len - TEX_VLC_BITS;
				} else {
					RlVlcElem e;
					e.len8  = (int8_t)len;
					e.run   = vlc_run[i] + 1;
					e.level = sign ? -vlc_level[i] : vlc_level[i];
					if (len <= TEX_VLC_BITS) {
						for (int k = 0; k < (1 << (TEX_VLC_BITS - len)); k++) rl_vlc[prefix + k] = e;
					} else {
						const int bits  = subBits[prefix];
						const int first = subStart[prefix] + ((code << TEX_VLC_BITS) >> (32 - bits));
						for (int k = 0; k < (1 << (bits - (len - TEX_VLC_BITS))); k++) rl_vlc[first + k] = e;
					}
				}
				code += 1u << (32 - len);
			}
		}
		if (pass == 0) {
			int next = 1 << TEX_VLC_BITS;
			for (int p = 0; p < (1 << TEX_VLC_BITS); p++) {
				if (!subBits[p]) continue;
				subStart[p] = (uint16_t)next;
				rl_vlc[p].len8  = (int8_t)-subBits[p];
				rl_vlc[p].level = (int16_t)next;
				rl_vlc[p].run   = 0;
				next += 1 << subBits[p];
			}
		}
	}
}

struct GetBits {
	const uint8_t* buf;
	int      size_in_bits;
	unsigned index;
};

static inline void gb_init(GetBits* gb, const uint8_t* buf, int bits) {
	gb->buf = buf;
	gb->size_in_bits = bits;
	gb->index = 0;
}

static inline uint32_t gb_cache(const uint8_t* buf, unsigned index) {
	return AV_RB32(buf + (index >> 3)) << (index & 7);
}

// n is up to 25
static inline uint32_t gb_get(GetBits* gb, int n) {
	const uint32_t v = gb_cache(gb->buf, gb->index) >> (32 - n);
	gb->index += n;
	return v;
}

static inline int gb_left(const GetBits* gb) { return gb->size_in_bits - (int)gb->index; }

struct PutBits {
	uint8_t* start;
	uint8_t* ptr;
	uint32_t acc;
	int      left;
};

static inline void pb_init(PutBits* pb, uint8_t* buf) {
	pb->start = pb->ptr = buf;
	pb->acc = 0;
	pb->left = 32;
}

static inline void pb_store(PutBits* pb, uint32_t v) {
	pb->ptr[0] = (uint8_t)(v >> 24);
	pb->ptr[1] = (uint8_t)(v >> 16);
	pb->ptr[2] = (uint8_t)(v >> 8);
	pb->ptr[3] = (uint8_t)v;
	pb->ptr += 4;
}

// n is up to 25
static inline void pb_put(PutBits* pb, int n, uint32_t v) {
	if (n < pb->left) {
		pb->acc = (pb->acc << n) | v;
		pb->left -= n;
	} else {
		pb->acc = (pb->acc << pb->left) | (v >> (n - pb->left));
		pb_store(pb, pb->acc);
		pb->left += 32 - n;
		pb->acc = v;
	}
}

static inline int pb_count(const PutBits* pb) { return (int)(pb->ptr - pb->start) * 8 + 32 - pb->left; }

// What was written, then zeros past it: a code may be looked up across the end.
static inline void pb_flush_zeroed(PutBits* pb) {
	pb_store(pb, pb->left == 32 ? 0 : pb->acc << pb->left);
	pb_store(pb, 0);
}

static inline void bit_copy(PutBits* pb, GetBits* gb) {
	int bits_left = gb_left(gb);
	while (bits_left >= 25) {
		pb_put(pb, 25, gb_get(gb, 25));
		bits_left -= 25;
	}
	if (bits_left > 0) pb_put(pb, bits_left, gb_get(gb, bits_left));
}

struct BlockInfo {
	const uint32_t* factor_table;
	const uint8_t*  scan_table;
	uint8_t  pos;   // position in block
	bool     idct248;
	uint8_t  partial_bit_count;
	uint32_t partial_bit_buffer;
};

// dv_decode_ac
static void decode_ac(GetBits* gb, BlockInfo* mb, int16_t* block) {
	const int last_index = gb->size_in_bits;
	const uint8_t*  scan_table   = mb->scan_table;
	const uint32_t* factor_table = mb->factor_table;
	int pos = mb->pos;
	const int partial_bit_count = mb->partial_bit_count;

	unsigned re_index = gb->index;
	uint32_t re_cache = gb_cache(gb->buf, re_index);

	// if we must parse a partial VLC, we do it here
	if (partial_bit_count > 0) {
		re_cache = (re_cache >> partial_bit_count) | mb->partial_bit_buffer;
		re_index -= partial_bit_count;
		mb->partial_bit_count = 0;
	}

	for (;;) {
		int index   = re_cache >> (32 - TEX_VLC_BITS);
		int vlc_len = rl_vlc[index].len8;
		if (vlc_len < 0) {
			index = ((re_cache << TEX_VLC_BITS) >> (32 + vlc_len)) + rl_vlc[index].level;
			vlc_len = TEX_VLC_BITS - vlc_len;
		}
		const int level = rl_vlc[index].level;
		const int run   = rl_vlc[index].run;

		// a code that crosses the end continues in the next pass's bits
		if (re_index + vlc_len > (unsigned)last_index) {
			mb->partial_bit_count  = (uint8_t)(last_index - re_index);
			mb->partial_bit_buffer = re_cache & ~(0xffffffffu >> mb->partial_bit_count);
			re_index = last_index;
			break;
		}
		re_index += vlc_len;

		pos += run;
		if (pos >= 64) break;

		block[scan_table[pos]] = (int16_t)(((uint32_t)level * factor_table[pos] + (1u << (IWEIGHT_BITS - 1))) >> IWEIGHT_BITS);

		re_cache = gb_cache(gb->buf, re_index);
	}
	gb->index = re_index;
	mb->pos = (uint8_t)pos;
}

enum { MAX_BPM = 8, SLACK = 16 };

struct Ctx {
	const Profile* sys;
	WorkChunk work_chunks[4 * 12 * 27];
	uint32_t  idct_factor[2 * 4 * 16 * 64];

	uint8_t* cb;           // planes, the Cr's follows in the same allocation
	uint8_t* cr;
	int      chromaW, chromaH;

	bool     sl25;

	alignas(16) int16_t sblock[5 * MAX_BPM][64];
	alignas(16) uint8_t mb_bit_buffer[80 + 32];
	alignas(16) uint8_t vs_bit_buffer[80 * 5 + 32];
	BlockInfo mb_data[5 * MAX_BPM];
};

// dv_init_weight_tables
static void init_weight_tables(Ctx* c, const Profile* d) {
	uint32_t* factor1 = &c->idct_factor[0];
	uint32_t* factor2 = &c->idct_factor[is_hd(d) ? 4096 : 2816];

	if (is_hd(d)) {
		// quantization quanta by QNO for DV100: 0 and 1 both have no quantization
		static const uint8_t dv100_qstep[16] = { 1, 1, 2, 3, 4, 5, 6, 7, 8, 16, 18, 20, 22, 24, 28, 52 };
		const uint16_t* iweight1 = d->height == 720 ? iweight_720_y : iweight_1080_y;
		const uint16_t* iweight2 = d->height == 720 ? iweight_720_c : iweight_1080_c;
		for (int k = 0; k < 4; k++) {
			for (int s = 0; s < 16; s++) {
				for (int i = 0; i < 64; i++) {
					*factor1++ = (dv100_qstep[s] << (k + 9)) * iweight1[i];
					*factor2++ = (dv100_qstep[s] << (k + 9)) * iweight2[i];
				}
			}
		}
	} else {
		static const uint8_t quant_areas[4] = { 6, 21, 43, 64 };
		const uint16_t* iweight1 = iweight_88;
		for (int j = 0; j < 2; j++, iweight1 = iweight_248) {
			for (int s = 0; s < 22; s++) {
				for (int i = 0, k = 0; k < 4; k++) {
					for (; i < quant_areas[k]; i++) {
						*factor1   = iweight1[i] << (quant_shifts[s * 4 + k] + 1);
						*factor2++ = (*factor1++) << 1;
					}
				}
			}
		}
	}
}

static inline void put_block_8x4(const int16_t* block, uint8_t* p, int stride) {
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 8; j++) p[j] = av_clip_uint8(block[j]);
		block += 8;
		p += stride;
	}
}

static void dv100_idct_put_last_row_field_chroma(uint8_t* data, int stride, int16_t* blocks) {
	simple_idct(blocks + 0 * 64);
	simple_idct(blocks + 1 * 64);

	put_block_8x4(blocks + 0 * 64,         data,              stride << 1);
	put_block_8x4(blocks + 0 * 64 + 4 * 8, data + 8,          stride << 1);
	put_block_8x4(blocks + 1 * 64,         data + stride,     stride << 1);
	put_block_8x4(blocks + 1 * 64 + 4 * 8, data + 8 + stride, stride << 1);
}

static void dv100_idct_put_last_row_field_luma(uint8_t* data, int stride, int16_t* blocks) {
	simple_idct(blocks + 0 * 64);
	simple_idct(blocks + 1 * 64);
	simple_idct(blocks + 2 * 64);
	simple_idct(blocks + 3 * 64);

	put_block_8x4(blocks + 0 * 64,         data,               stride << 1);
	put_block_8x4(blocks + 0 * 64 + 4 * 8, data + 16,          stride << 1);
	put_block_8x4(blocks + 1 * 64,         data + 8,           stride << 1);
	put_block_8x4(blocks + 1 * 64 + 4 * 8, data + 24,          stride << 1);
	put_block_8x4(blocks + 2 * 64,         data + stride,      stride << 1);
	put_block_8x4(blocks + 2 * 64 + 4 * 8, data + 16 + stride, stride << 1);
	put_block_8x4(blocks + 3 * 64,         data + 8  + stride, stride << 1);
	put_block_8x4(blocks + 3 * 64 + 4 * 8, data + 24 + stride, stride << 1);
}

// dv_calculate_mb_xy
static inline void calculate_mb_xy(const Profile* sys, const uint8_t* buf, const WorkChunk* chunk, int m, int* mb_x, int* mb_y) {
	*mb_x = chunk->mb_coordinates[m] & 0xff;
	*mb_y = chunk->mb_coordinates[m] >> 8;

	// 720p frames come split in half and the odd half (chan 2 and 3) is displaced
	if (sys->height == 720 && !(buf[1] & 0x0C)) *mb_y -= (*mb_y > 17) ? 18 : -72;
}

// dv_decode_video_segment. mb_x and mb_y are in units of 8 pixels.
static void decode_video_segment(RvCtx* rv, Ctx* s, const uint8_t* buf, const WorkChunk* work_chunk) {
	const Profile* sys = s->sys;
	const int bpm = sys->bpm;
	const bool hd = is_hd(sys);
	const int ysize = sys->width, csize = s->chromaW;
	int is_field_mode[5];
	int vs_bit_buffer_damaged = 0;
	int mb_bit_buffer_damaged[5] = { 0, 0, 0, 0, 0 };
	int retried = 0;
	int sta = 0;

retry:
	memset(s->sblock, 0, sizeof(s->sblock));

	// pass 1: read DC and AC coefficients in blocks
	const uint8_t* buf_ptr = &buf[work_chunk->buf_offset * 80];
	int16_t* block1 = &s->sblock[0][0];
	BlockInfo* mb1 = s->mb_data;
	PutBits vs_pb;
	pb_init(&vs_pb, s->vs_bit_buffer);
	for (int mb_index = 0; mb_index < 5; mb_index++, mb1 += bpm, block1 += bpm * 64) {
		// the header
		const int quant = buf_ptr[3] & 0x0f;
		if ((buf_ptr[3] >> 4) == 0x0E) vs_bit_buffer_damaged = 1;
		if (!mb_index) {
			sta = buf_ptr[3] >> 4;
		} else if (sta != (buf_ptr[3] >> 4)) {
			vs_bit_buffer_damaged = 1;
		}
		buf_ptr += 4;

		PutBits pb;
		pb_init(&pb, s->mb_bit_buffer);
		BlockInfo* mb = mb1;
		int16_t* block = block1;
		is_field_mode[mb_index] = 0;
		for (int j = 0; j < bpm; j++) {
			const int last_index = sys->block_sizes[j];
			GetBits gb;
			gb_init(&gb, buf_ptr, last_index);

			// the DC, 9 bits signed
			int dc = (int32_t)(gb_cache(buf_ptr, 0)) >> (32 - 9);
			gb.index = 9;
			const int dct_mode = gb_get(&gb, 1);
			const int class1   = gb_get(&gb, 2);
			if (hd) {
				mb->idct248      = false;
				mb->scan_table   = zigzag_direct;
				mb->factor_table = &s->idct_factor[(j >= 4) * 4 * 16 * 64 + class1 * 16 * 64 + quant * 64];
				is_field_mode[mb_index] |= !j && dct_mode;
			} else {
				mb->idct248      = !!dct_mode;
				mb->scan_table   = dct_mode ? zigzag248_direct : zigzag_direct;
				mb->factor_table = &s->idct_factor[(class1 == 3) * 2 * 22 * 64 + dct_mode * 22 * 64 + (quant + quant_offset[class1]) * 64];
			}
			// unsigned, because 128 is not added in the standard IDCT
			block[0] = (int16_t)(dc * 4 + 1024);
			buf_ptr += last_index >> 3;
			mb->pos = 0;
			mb->partial_bit_count = 0;

			decode_ac(&gb, mb, block);

			// the rest of a finished block is for the others
			if (mb->pos >= 64) bit_copy(&pb, &gb);
			if (mb->pos >= 64 && mb->pos < 127) vs_bit_buffer_damaged = mb_bit_buffer_damaged[mb_index] = 1;

			block += 64;
			mb++;
		}

		if (mb_bit_buffer_damaged[mb_index] > 0) continue;

		// pass 2: the macroblock's spare bits
		block = block1;
		mb = mb1;
		GetBits gb;
		gb_init(&gb, s->mb_bit_buffer, pb_count(&pb));
		pb_flush_zeroed(&pb);
		int j;
		for (j = 0; j < bpm; j++, block += 64, mb++) {
			if (mb->pos < 64 && gb_left(&gb) > 0) {
				decode_ac(&gb, mb, block);
				// if still not finished, no need to parse other blocks
				if (mb->pos < 64) break;
				if (mb->pos < 127) vs_bit_buffer_damaged = mb_bit_buffer_damaged[mb_index] = 1;
			}
		}
		// all blocks are finished, so the extra bytes can be used at the video segment level
		if (j >= bpm) bit_copy(&vs_pb, &gb);
	}

	// pass 3: the whole video segment's spare bits
	{
		int16_t* block = &s->sblock[0][0];
		BlockInfo* mb = s->mb_data;
		GetBits gb;
		gb_init(&gb, s->vs_bit_buffer, pb_count(&vs_pb));
		pb_flush_zeroed(&vs_pb);
		for (int mb_index = 0; mb_index < 5; mb_index++) {
			for (int j = 0; j < bpm; j++) {
				if (mb->pos < 64 && gb_left(&gb) > 0 && !vs_bit_buffer_damaged) decode_ac(&gb, mb, block);
				if (mb->pos >= 64 && mb->pos < 127) vs_bit_buffer_damaged = 1;
				block += 64;
				mb++;
			}
		}
	}
	if (vs_bit_buffer_damaged && !retried) {
		rv->warn = RV_W_BADDATA;
		retried = 1;
		goto retry;
	}

	// compute the IDCT and place the blocks
	int16_t* block = &s->sblock[0][0];
	BlockInfo* mb = s->mb_data;
	uint8_t* const luma = rv->frame;
	for (int mb_index = 0; mb_index < 5; mb_index++) {
		int mb_x, mb_y;
		calculate_mb_xy(sys, buf, work_chunk, mb_index, &mb_x, &mb_y);
		const int field = is_field_mode[mb_index];

		// luma
		int y_stride;
		if (sys->pix_fmt == PIX_420 || (sys->pix_fmt == PIX_411 && mb_x >= (704 / 8)) || (sys->height >= 720 && mb_y != 134)) {
			y_stride = ysize << ((!field) * 3);
		} else {
			y_stride = 2 << 3;
		}
		uint8_t* y_ptr = luma + ((mb_y * ysize + mb_x) << 3);
		if (mb_y == 134 && field) {
			dv100_idct_put_last_row_field_luma(y_ptr, ysize, block);
		} else {
			const int linesize = ysize << field;
			idct_put(mb[0].idct248, y_ptr, linesize, block + 0 * 64);
			if (sys->video_stype == 4) {   // SD 4:2:2
				idct_put(mb[2].idct248, y_ptr + 8, linesize, block + 2 * 64);
			} else {
				idct_put(mb[1].idct248, y_ptr + 8,            linesize, block + 1 * 64);
				idct_put(mb[2].idct248, y_ptr     + y_stride, linesize, block + 2 * 64);
				idct_put(mb[3].idct248, y_ptr + 8 + y_stride, linesize, block + 3 * 64);
			}
		}
		mb    += 4;
		block += 4 * 64;

		// chroma, Cr first
		const int c_offset = ((mb_y >> (sys->pix_fmt == PIX_420 ? 1 : 0)) * csize + (mb_x >> (sys->pix_fmt == PIX_411 ? 2 : 1))) << 3;
		for (int j = 2; j; j--) {
			uint8_t* c_ptr = (j == 2 ? s->cr : s->cb) + c_offset;
			if (sys->pix_fmt == PIX_411 && mb_x >= (704 / 8)) {
				// the right edge: one block split into two stacked 4x8
				alignas(8) uint8_t pixels[64];
				idct_put(mb->idct248, pixels, 8, block);
				uint8_t* row = pixels;
				for (int y = 0; y < 8; y++, c_ptr += csize, row += 8) {
					uint8_t* c_ptr1 = c_ptr + (csize << 3);
					for (int x = 0; x < 4; x++) {
						c_ptr[x]  = row[x];
						c_ptr1[x] = row[4 + x];
					}
				}
				block += 64;
				mb++;
			} else {
				const int c_stride = (mb_y == 134) ? 8 : csize << ((!field) * 3);
				if (mb_y == 134 && field) {
					dv100_idct_put_last_row_field_chroma(c_ptr, csize, block);
					mb += 2;
					block += 2 * 64;
				} else {
					const int linesize = csize << field;
					idct_put(mb->idct248, c_ptr, linesize, block);
					mb++;
					block += 64;
					if (bpm == 8) {
						idct_put(mb->idct248, c_ptr + c_stride, linesize, block);
						mb++;
						block += 64;
					}
				}
			}
		}
	}
}

static int set_profile(RvCtx* rv, Ctx* s, const Profile* sys) {
	init_dynamic_tables(s->work_chunks, sys);
	init_weight_tables(s, sys);
	s->sys = sys;

	const int w = sys->width, h = sys->height;
	s->chromaW = w >> (sys->pix_fmt == PIX_411 ? 2 : 1);
	s->chromaH = h >> (sys->pix_fmt == PIX_420 ? 1 : 0);

	free(s->cb);
	s->cb = (uint8_t*)calloc((size_t)(s->chromaW * (s->chromaH + SLACK)) * 2, 1);
	if (!s->cb) return AVERROR_INVALIDDATA;
	s->cr = s->cb + s->chromaW * (s->chromaH + SLACK);

	// Y, then Cb,Cr as 4:2:0 or 4:2:2. The rows past the picture are DVCPRO HD's.
	const size_t size = (size_t)w * (h + SLACK) * 2;
	if (size > rv->frameSize) {
		free(rv->frame);
		rv->frame = (uint8_t*)calloc(size, 1);
		rv->frameSize = rv->frame ? size : 0;
		if (!rv->frame) return AVERROR_INVALIDDATA;
	}
	rv->width  = w;
	rv->height = h;
	rv->stride = w;
	rv->format = sys->pix_fmt == PIX_420 ? RV_FMT_YCBCR420 : RV_FMT_YCBCR422;
	return 0;
}

static void store_chroma(RvCtx* rv, const Ctx* s) {
	const Profile* sys = s->sys;
	uint8_t* out = rv->frame + (size_t)sys->width * sys->height;
	const uint8_t* cb = s->cb;
	const uint8_t* cr = s->cr;
	const int cw = s->chromaW;
	for (int y = 0; y < s->chromaH; y++) {
		if (sys->pix_fmt == PIX_411) {
			for (int x = 0; x < cw; x++) {
				out[0] = out[2] = cb[x];
				out[1] = out[3] = cr[x];
				out += 4;
			}
		} else {
			for (int x = 0; x < cw; x++) {
				out[0] = cb[x];
				out[1] = cr[x];
				out += 2;
			}
		}
		cb += cw;
		cr += cw;
	}
}

// A picture before the first frame that decodes: black at the open dimensions, as 4:2:0.
static void black(RvCtx* rv) {
	memset(rv->frame, 128, rv->frameSize);
	memset(rv->frame, 16, (size_t)rv->width * rv->height);
}

static void open(RvCtx* rv, bool sl25) {
	init_vlc();
	Ctx* s = (Ctx*)calloc(1, sizeof(Ctx));
	rv->dv = s;
	if (s) s->sl25 = sl25;
}

static void close(Ctx* s) {
	if (!s) return;
	free(s->cb);
	free(s);
}

static int decode(RvCtx* rv, const uint8_t* buf, int buf_size) {
	Ctx* s = rv->dv;
	if (!s) return AVERROR_INVALIDDATA;

	const Profile* sys = frame_profile(s->sys, buf, buf_size, s->sl25);
	if (!sys || buf_size < sys->frame_size) return AVERROR_INVALIDDATA;

	if (sys != s->sys) {
		const int r = set_profile(rv, s, sys);
		if (r < 0) {
			s->sys = nullptr;
			return r;
		}
	}

	const int chunks = work_pool_size(sys);
	for (int i = 0; i < chunks; i++) decode_video_segment(rv, s, buf, &s->work_chunks[i]);

	store_chroma(rv, s);
	return 0;
}

} // namespace dv

#endif
