/*
	AC-3 (Dolby Digital / ATSC A/52) audio decoder — WASM C++.

	This is a direct port of FFmpeg's AC-3 decoder (libavcodec, release/8.1, LGPL):
	  ac3dec.c / ac3dec_float.c / ac3.c / ac3tab.c / ac3dsp.c / ac3dec_data.c /
	  ac3_parser.c / kbdwin.c + libavutil/lfg.c + libavutil/float_dsp.c
	The algorithm (header parse, exponent decode, bit allocation, mantissa decode,
	coupling, rematrixing, dither, IMDCT, windowed overlap-add) is lifted from those
	files; the surrounding libavcodec framework (AVCodecContext / AVFrame / AVPacket /
	the DSP function-pointer dispatch / AVTXContext / threading / AVERROR) has been
	replaced by a self-contained handle (AC3Ctx) that owns its own planar float
	output buffer.

	It REPLACES the earlier from-spec JavaScript decoder (helpers/mpeg/ac3decode.js),
	which was AI-written from ATSC A/52 rather than ported, and which evaluated the
	IMDCT as a direct O(N^2) cosine sum. Deleting it follows the VP6 precedent: an
	AI-derived reimplementation is not an independent oracle, so FFmpeg itself is the
	oracle and we port FFmpeg directly.

	Scope — plain AC-3 only, matching what the JS decoder covered and what the
	containers we read actually carry (DVD / MPEG-PS / AVI 0x2000 / MOV ac-3):
	  - bsid <= 10 (A/52 including the Annex D low-sample-rate shift). NOT E-AC-3.
	  - No spectral extension (SPX), no adaptive hybrid transform (AHT), no enhanced
	    coupling, no dependent substreams — all E-AC-3-only features.
	  - No fixed-point path (USE_FIXED); float only.

	Differences from FFmpeg, all behavior-preserving for that scope:
	  - Output is always the stream's own channel count. FFmpeg's optional downmix to
	    mono/stereo (set_downmix_coeffs / ac3_upmix_delay / ff_ac3dsp_downmix) is
	    dropped, so out_channels == channels, `downmix_output` is always false and the
	    two arms of the block-switch branch collapse to a single do_imdct() call.
	    Channels are still emitted in FFmpeg's output order via ff_ac3_dec_channel_map
	    (5.1 coded L,C,R,Ls,Rs,LFE -> FL,FR,FC,LFE,SL,SR), so PCM is directly
	    comparable to `ffmpeg -f f32le` reference output.
	  - The IMDCT is computed here as a DCT-IV via a complex FFT (see imdct_core)
	    instead of FFmpeg's generic libavutil/tx codelet framework, which would be far
	    larger than the whole decoder. The transform is the plain A/52 7.9.4
	    definition; ac3_test_imdct() checks it against a direct O(N^2) evaluation of
	    that definition, and the decoder as a whole is checked against FFmpeg's PCM.
	  - av_lfg_init(0) needs MD5, so the dither RNG's seed-0 state is embedded as a
	    constant table (LFG_INIT_STATE) instead of being derived at run time. FFmpeg
	    only ever seeds this generator with 0 here, and it leaves state[0..7] zero —
	    that quirk is reproduced, because dithered bins must match bit-for-bit.

	Boundary (extern "C", prefixed _ on the JS side; see ac3_decode_core.js):
	  AC3Ctx*  ac3_open(void)
	  void     ac3_close(AC3Ctx*)
	  void     ac3_reset(AC3Ctx*)                    drop overlap-add delay (seek)
	  void     ac3_set_drc_scale(AC3Ctx*, float)     0 = no DRC (ffmpeg -drc_scale)
	  void     ac3_set_dither(AC3Ctx*, int)          0 = zero bap=0 bins instead
	  int      ac3_probe_frame(const uint8_t*, int, int info[6])
	             parse just the syncinfo/bsi; info = {frameSize, sampleRate, channels,
	             bitRate, bsid, acmod}; returns 1/0 without touching decoder state
	  int      ac3_decode_frame(AC3Ctx*, const uint8_t*, int, int info[5])
	             decode ONE frame; info = {frameSize, sampleRate, channels, samples,
	             shortBlocks}; returns 1/0
	  float*   ac3_channel_ptr(AC3Ctx*, int ch)      planar output, 1536 samples
*/

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ─── limits (ac3defs.h) ──────────────────────────────────────────────────────
#define AC3_MAX_CHANNELS   7	// including the coupling channel at index 0
#define CPL_CH             0
#define AC3_MAX_COEFS      256
#define AC3_BLOCK_SIZE     256
#define AC3_MAX_BLOCKS     6
#define AC3_FRAME_SIZE     (AC3_MAX_BLOCKS * AC3_BLOCK_SIZE)	// 1536
#define AC3_CRITICAL_BANDS 50
#define AC3_MAX_CPL_BANDS  18
#define AC3_OUTPUT_LFEON   8

#define EXP_REUSE 0
#define EXP_D15   1
#define EXP_D25   2
#define EXP_D45   3

#define DBA_REUSE    0
#define DBA_NEW      1
#define DBA_NONE     2
#define DBA_RESERVED 3

#define AC3_CHMODE_DUALMONO 0
#define AC3_CHMODE_MONO     1
#define AC3_CHMODE_STEREO   2
#define AC3_CHMODE_3F       3
#define AC3_CHMODE_2F1R     4
#define AC3_CHMODE_3F1R     5
#define AC3_CHMODE_2F2R     6
#define AC3_CHMODE_3F2R     7

static inline int FFMIN(int a, int b) { return a < b ? a : b; }
static inline int FFMAX(int a, int b) { return a > b ? a : b; }

// libavutil av_clip_uintp2(a, 6): clamp to [0, 63].
static inline int clip_uintp2_6(int a) { return a & ~63 ? (~a) >> 31 & 63 : a; }

// ─── bit reader (libavcodec/get_bits.h, MSB-first) ───────────────────────────
// Reads past the end yield zero bits rather than reading out of bounds; FFmpeg
// gets the same effect from AV_INPUT_BUFFER_PADDING_SIZE on the input buffer.
struct GetBits {
	const uint8_t* buf;
	int size_in_bits;
	int index;
};

static void init_get_bits(GetBits* g, const uint8_t* buf, int byte_size)
{
	g->buf = buf;
	g->size_in_bits = byte_size * 8;
	g->index = 0;
}

static inline unsigned get_bits(GetBits* g, int n)
{
	unsigned v = 0;
	for (int i = 0; i < n; i++) {
		int idx = g->index++;
		unsigned bit = 0;
		if (idx < g->size_in_bits)
			bit = (g->buf[idx >> 3] >> (7 - (idx & 7))) & 1;
		v = (v << 1) | bit;
	}
	return v;
}

static inline unsigned get_bits1(GetBits* g) { return get_bits(g, 1); }
static inline void skip_bits(GetBits* g, int n) { g->index += n; }

// Signed n-bit value, MSB-first (get_sbits).
static inline int get_sbits(GetBits* g, int n)
{
	unsigned v = get_bits(g, n);
	unsigned sign = 1u << (n - 1);
	return (int)((v ^ sign) - sign);
}

// show_bits without consuming.
static inline unsigned show_bits(GetBits* g, int n)
{
	int save = g->index;
	unsigned v = get_bits(g, n);
	g->index = save;
	return v;
}

#include "ac3_tables.h"

///////////////////////////////////////////////////////////////////////////////
// Transform: IMDCT (A/52 7.9.4) as a DCT-IV via a complex FFT.
//
// FFmpeg routes this through libavutil/tx (av_tx_init(AV_TX_FLOAT_MDCT, ...)), a
// generic codelet framework much larger than this whole decoder, whose imdct_half
// output is in a folded order that vector_fmul_window then unfolds. We instead
// compute the transform in its plain spec form
//
//   z[n] = sum_k X[k] * cos((pi/M) * (n + 0.5 + M/2) * (k + 0.5)),  n = 0..2M-1
//
// and do the windowed overlap-add directly on it. Writing the core as a DCT-IV:
// substituting n = M/2 + p and using cos(pi*(k+1/2)) = 0 gives
//
//   z[M/2 + p] = -DCTIV(X)[M-1-p],   DCTIV(X)[q] = sum_k X[k] cos((pi/M)(q+.5)(k+.5))
//
// and the two standard MDCT symmetries recover the rest:
//   z[j]        = -z[M-1-j]      for j < M/2      (antisymmetric about M/2)
//   z[3M/2 + n] =  z[3M/2 -1 -n]                  (symmetric about 3M/2)
//
// ac3_test_imdct() checks this against a direct O(N^2) evaluation of the formula
// above, so the FFT factoring is verified against the definition it optimizes.

struct Cpx { double re, im; };

// Iterative radix-2 complex DFT, out-of-place-safe (operates in place on buf).
// len must be a power of two <= 128. Sign convention: exp(-2*pi*i*j*m/len).
static void fft_dit(Cpx* buf, int len)
{
	// bit-reversal permutation
	for (int i = 1, j = 0; i < len; i++) {
		int bit = len >> 1;
		for (; j & bit; bit >>= 1) j ^= bit;
		j ^= bit;
		if (i < j) { Cpx t = buf[i]; buf[i] = buf[j]; buf[j] = t; }
	}
	for (int size = 2; size <= len; size <<= 1) {
		double ang = -2.0 * M_PI / size;
		double wr = cos(ang), wi = sin(ang);
		for (int i = 0; i < len; i += size) {
			double cr = 1.0, ci = 0.0;
			for (int j = 0; j < size / 2; j++) {
				Cpx u = buf[i + j];
				Cpx v = buf[i + j + size / 2];
				double vr = v.re * cr - v.im * ci;
				double vi = v.re * ci + v.im * cr;
				buf[i + j].re            = u.re + vr;
				buf[i + j].im            = u.im + vi;
				buf[i + j + size / 2].re = u.re - vr;
				buf[i + j + size / 2].im = u.im - vi;
				double ncr = cr * wr - ci * wi;
				ci = cr * wi + ci * wr;
				cr = ncr;
			}
		}
	}
}

// DCT-IV of size N (N = 256 or 128 here) via an N/2-point complex FFT:
//   c[j]        = (x[2j] + i*x[N-1-2j]) * exp(-i*pi*(4j+1)/(4N))
//   C           = DFT_{N/2}(c)
//   y[m]        = C[m] * exp(-i*pi*m/N)
//   out[2m]     =  Re(y[m])
//   out[N-1-2m] = -Im(y[m])
//
// The three phases must sum to exactly pi*(4m+1)(4j+1)/(4N), which is DCT-IV's
// kernel: the DFT contributes 16jm, the pre-twiddle 4j+1 and the post-twiddle 4m.
// Splitting the constant 1 across BOTH twiddles (the tempting symmetric form
// exp(-i*pi*(4m+1)/(4N))) over-rotates by pi/(4N) and is subtly wrong everywhere.
static void dct4(const double* x, double* out, int N)
{
	const int L = N / 2;
	Cpx c[128];

	for (int j = 0; j < L; j++) {
		double a = x[2 * j];
		double b = x[N - 1 - 2 * j];
		double ang = -M_PI * (4.0 * j + 1.0) / (4.0 * N);
		double wr = cos(ang), wi = sin(ang);
		c[j].re = a * wr - b * wi;
		c[j].im = a * wi + b * wr;
	}

	fft_dit(c, L);

	for (int m = 0; m < L; m++) {
		double ang = -M_PI * (double)m / (double)N;
		double wr = cos(ang), wi = sin(ang);
		double yr = c[m].re * wr - c[m].im * wi;
		double yi = c[m].re * wi + c[m].im * wr;
		out[2 * m]         =  yr;
		out[N - 1 - 2 * m] = -yi;
	}
}

/*
	The independent core of the IMDCT: core[p] = z[M/2 + p], p = 0..M-1.

	Of the 2M outputs only these M are independent (the two symmetries above recover
	the rest), and this is also exactly what FFmpeg's av_tx MDCT-inverse of length M
	returns — which is why the short-block path below is written in terms of cores
	rather than full transforms.
*/
static void imdct_core(const double* X, double* core, int M)
{
	double dc[256];
	dct4(X, dc, M);
	for (int p = 0; p < M; p++)
		core[p] = -dc[M - 1 - p];
}

// Full IMDCT: M coefficients in, 2M time samples out (A/52 7.9.4).
static void imdct(const double* X, double* z, int M)
{
	double core[256];
	imdct_core(X, core, M);

	const int half = M / 2;
	for (int j = 0; j < half; j++)      z[j] = -core[half - 1 - j];
	for (int p = 0; p < M; p++)         z[half + p] = core[p];
	for (int n = 0; n < half; n++)      z[half + M + n] = core[M - 1 - n];
}

///////////////////////////////////////////////////////////////////////////////
// Kaiser-Bessel-derived window (libavcodec/kbdwin.c, alpha = 5, n = 256).
// These 256 values are the RISING half of the 512-sample A/52 window; the falling
// half is w[511-n], which is how the overlap-add below indexes it.

static double bessel_i0(double x)
{
	double sum = 1.0, term = 1.0;
	const double x2 = x * x / 4.0;
	for (int k = 1; k < 200; k++) {
		term *= x2 / ((double)k * (double)k);
		sum += term;
		if (term < sum * 1e-20) break;
	}
	return sum;
}

static void kbd_window_init(double* window, double alpha, int n)
{
	double temp[257];
	double sum = 0.0, scale = 0.0;
	const double alpha2 = 4.0 * (alpha * M_PI / n) * (alpha * M_PI / n);

	for (int i = 0; i <= n / 2; i++) {
		double tmp = (double)i * (double)(n - i) * alpha2;
		temp[i] = bessel_i0(sqrt(tmp));
		scale += temp[i] * (1 + (i && i < n / 2));
	}
	scale = 1.0 / (scale + 1);

	int i = 0;
	for (; i <= n / 2; i++) {
		sum += temp[i];
		window[i] = sqrt(sum * scale);
	}
	for (; i < n; i++) {
		sum += temp[n - i];
		window[i] = sqrt(sum * scale);
	}
}

///////////////////////////////////////////////////////////////////////////////
// Dither RNG — libavutil's lagged Fibonacci generator (av_lfg_get):
//   state[i] = state[i-24] + state[i-55]  (mod 2^32)
// The A/52 dither for zero-bit (bap=0) mantissas is decoder-defined, so matching
// FFmpeg's generator exactly is what lets dithered bins compare bit-for-bit.

struct LFG {
	uint32_t state[64];
	unsigned index;
};

static inline void lfg_init(LFG* c)
{
	memcpy(c->state, LFG_INIT_STATE, sizeof(c->state));
	c->index = 0;
}

static inline uint32_t lfg_get(LFG* c)
{
	uint32_t a = c->state[c->index & 63] =
		c->state[(c->index - 24) & 63] + c->state[(c->index - 55) & 63];
	c->index += 1u;
	return a;
}

///////////////////////////////////////////////////////////////////////////////
// Decoder context (FFmpeg's AC3DecodeContext, minus everything E-AC-3 and minus
// the downmix machinery — see the file header for why those are gone).

struct AC3Ctx {
	GetBits gbc;

	// bit stream information
	int frame_size, bit_rate, sample_rate, num_blocks;
	int bitstream_id, bitstream_mode, channel_mode, lfe_on;
	int dialog_normalization[2], compression_exists[2];
	double heavy_dynamic_range[2];

	// bit allocation parameters
	int sr_code, sr_shift;
	int slow_gain, slow_decay, fast_decay, db_per_bit, floor_;
	int cpl_fast_leak, cpl_slow_leak;

	// coupling
	int cpl_in_use[AC3_MAX_BLOCKS];
	int channel_in_cpl[AC3_MAX_CHANNELS];
	int phase_flags_in_use, phase_flags[AC3_MAX_CPL_BANDS];
	int num_cpl_bands;
	uint8_t cpl_band_struct[AC3_MAX_CPL_BANDS];
	uint8_t cpl_band_sizes[AC3_MAX_CPL_BANDS];
	int first_cpl_coords[AC3_MAX_CHANNELS];
	int cpl_coords[AC3_MAX_CHANNELS][AC3_MAX_CPL_BANDS];
	int first_cpl_leak;

	// channels
	int fbw_channels, channels, lfe_ch, output_mode, out_channels;

	// dynamic range
	double dynamic_range[2];
	float drc_scale;
	int heavy_compression;

	// bandwidth
	int start_freq[AC3_MAX_CHANNELS], end_freq[AC3_MAX_CHANNELS];

	// rematrixing
	int num_rematrixing_bands, rematrixing_flags[4];

	// exponents. dexps is written at [start_freq + 1 .. ] in groups, so it carries a
	// little slack past AC3_MAX_COEFS: decode_exponents fills ngrps*3*group_size
	// entries and the last group can run a few bins past end_freq.
	int num_exp_groups[AC3_MAX_CHANNELS];
	int8_t dexps[AC3_MAX_CHANNELS][AC3_MAX_COEFS + 16];
	int exp_strategy[AC3_MAX_BLOCKS][AC3_MAX_CHANNELS];

	// bit allocation
	int snr_offset[AC3_MAX_CHANNELS], fast_gain[AC3_MAX_CHANNELS];
	uint8_t bap[AC3_MAX_CHANNELS][AC3_MAX_COEFS];
	int16_t psd[AC3_MAX_CHANNELS][AC3_MAX_COEFS];
	int16_t band_psd[AC3_MAX_CHANNELS][AC3_CRITICAL_BANDS];
	int16_t mask[AC3_MAX_CHANNELS][AC3_CRITICAL_BANDS];
	int dba_mode[AC3_MAX_CHANNELS], dba_nsegs[AC3_MAX_CHANNELS];
	uint8_t dba_offsets[AC3_MAX_CHANNELS][8];
	uint8_t dba_lengths[AC3_MAX_CHANNELS][8];
	uint8_t dba_values[AC3_MAX_CHANNELS][8];

	// dither
	int dither_flag[AC3_MAX_CHANNELS];
	int dither_enabled;
	LFG dith_state;

	// See ac3_upmix_delay(). Starts at 1, exactly as FFmpeg's ac3_decode_init does.
	int downmixed;

	// transform
	int block_switch[AC3_MAX_CHANNELS];
	// Count of block-switched (short) channel-blocks in the frame just decoded.
	// Reported through info[4] so a conformance run can prove it actually exercised
	// the short transform rather than only the long one.
	int short_blocks;
	int32_t fixed_coeffs[AC3_MAX_CHANNELS][AC3_MAX_COEFS];
	double transform_coeffs[AC3_MAX_CHANNELS][AC3_MAX_COEFS];
	double delay[AC3_MAX_CHANNELS][AC3_BLOCK_SIZE];
	double window[AC3_BLOCK_SIZE];

	// planar float output, FFmpeg channel order, AC3_FRAME_SIZE samples per channel
	float out[6][AC3_FRAME_SIZE];
};

// Grouped mantissas for 3-, 5- and 11-level quantization (FFmpeg mant_groups).
struct MantGroups {
	int b1_mant[2], b2_mant[2], b4_mant;
	int b1, b2, b4;
};

///////////////////////////////////////////////////////////////////////////////
// Frame header — syncinfo + bsi (libavcodec/ac3_parser.c, plain AC-3 path only).

// Mix level code tables (A/52 5.4.2.4 / 5.4.2.6). Parsed for bitstream fidelity;
// we never downmix, so the values themselves go unused.
static const uint8_t center_levels[4]   = { 4, 5, 6, 5 };
static const uint8_t surround_levels[4] = { 4, 6, 7, 6 };

#define AC3_PARSE_OK              0
#define AC3_PARSE_ERROR_SYNC     -1
#define AC3_PARSE_ERROR_BSID     -2
#define AC3_PARSE_ERROR_SR       -3
#define AC3_PARSE_ERROR_FRAMESZ  -4

struct AC3Header {
	int sr_code, sr_shift, sample_rate, bit_rate, frame_size;
	int bitstream_id, bitstream_mode, channel_mode, lfe_on, channels;
	int center_mix_level, surround_mix_level;
	int dialog_normalization[2], compression_exists[2], heavy_dynamic_range[2];
};

// Find the 0x0B77 syncword, tolerating the byte-swapped 0x770B form's alignment
// (ff_ac3_find_syncword). Returns the offset, or -1.
static int ac3_find_syncword(const uint8_t* buf, int buf_size)
{
	int i;
	for (i = 1; i < buf_size; i += 2) {
		if (buf[i] == 0x77 || buf[i] == 0x0B) {
			if ((buf[i] ^ buf[i - 1]) == (0x77 ^ 0x0B)) { i--; break; }
			else if (i + 1 < buf_size && (buf[i] ^ buf[i + 1]) == (0x77 ^ 0x0B)) break;
		}
	}
	if (i >= buf_size) return -1;
	return i;
}

static int parse_frame_header(GetBits* gbc, AC3Header* h)
{
	memset(h, 0, sizeof(*h));

	if (get_bits(gbc, 16) != 0x0B77)
		return AC3_PARSE_ERROR_SYNC;

	// read ahead to bsid to tell AC-3 from E-AC-3
	h->bitstream_id = show_bits(gbc, 29) & 0x1F;
	if (h->bitstream_id > 16)  return AC3_PARSE_ERROR_BSID;
	if (h->bitstream_id > 10)  return AC3_PARSE_ERROR_BSID;	// E-AC-3: not supported

	// defaults
	h->center_mix_level   = 5;	// -4.5 dB
	h->surround_mix_level = 6;	// -6.0 dB

	skip_bits(gbc, 16);		// crc1
	h->sr_code = get_bits(gbc, 2);
	if (h->sr_code == 3) return AC3_PARSE_ERROR_SR;

	int frame_size_code = get_bits(gbc, 6);
	if (frame_size_code > 37) return AC3_PARSE_ERROR_FRAMESZ;
	int bit_rate_code = frame_size_code >> 1;

	skip_bits(gbc, 5);		// bsid, already read above
	h->bitstream_mode = get_bits(gbc, 3);
	h->channel_mode   = get_bits(gbc, 3);

	if (h->channel_mode == AC3_CHMODE_STEREO) {
		skip_bits(gbc, 2);	// dsurmod
	} else {
		if ((h->channel_mode & 1) && h->channel_mode != AC3_CHMODE_MONO)
			h->center_mix_level = center_levels[get_bits(gbc, 2)];
		if (h->channel_mode & 4)
			h->surround_mix_level = surround_levels[get_bits(gbc, 2)];
	}
	h->lfe_on = get_bits1(gbc);

	h->sr_shift    = FFMAX(h->bitstream_id, 8) - 8;
	h->sample_rate = ff_ac3_sample_rate_tab[h->sr_code] >> h->sr_shift;
	h->bit_rate    = (ff_ac3_bitrate_tab[bit_rate_code] * 1000) >> h->sr_shift;
	h->channels    = ff_ac3_channels_tab[h->channel_mode] + h->lfe_on;
	h->frame_size  = ff_ac3_frame_size_tab[frame_size_code][h->sr_code] * 2;

	// rest of the bsi; read twice in dual-mono
	for (int i = 0; i < (h->channel_mode ? 1 : 2); i++) {
		h->dialog_normalization[i] = -(int)get_bits(gbc, 5);
		h->compression_exists[i] = get_bits1(gbc);
		if (h->compression_exists[i])
			h->heavy_dynamic_range[i] = get_bits(gbc, 8);
		if (get_bits1(gbc)) skip_bits(gbc, 8);	// language code
		if (get_bits1(gbc)) skip_bits(gbc, 7);	// audio production info
	}

	skip_bits(gbc, 2);	// copyright + original bitstream

	if (h->bitstream_id != 6) {
		if (get_bits1(gbc)) skip_bits(gbc, 14);	// timecode1
		if (get_bits1(gbc)) skip_bits(gbc, 14);	// timecode2
	} else {
		// Alternate Bit Stream Syntax (A/52 Annex D)
		if (get_bits1(gbc)) {
			skip_bits(gbc, 2);					// dmixmod
			skip_bits(gbc, 3);					// ltrtcmixlev
			skip_bits(gbc, 3);					// ltrtsurmixlev
			h->center_mix_level   = get_bits(gbc, 3);
			h->surround_mix_level = get_bits(gbc, 3);
		}
		if (get_bits1(gbc)) skip_bits(gbc, 14);	// dsurexmod, dheadphonmod, +10
	}

	// additional bitstream info
	if (get_bits1(gbc)) {
		int i = get_bits(gbc, 6);
		do { skip_bits(gbc, 8); } while (i--);
	}

	return AC3_PARSE_OK;
}

///////////////////////////////////////////////////////////////////////////////
// Exponents (A/52 7.1.3), bit allocation (A/52 7.2) — ac3dec.c / ac3.c / ac3dsp.c

static int decode_exponents(GetBits* gbc, int exp_strategy, int ngrps,
                            uint8_t absexp, int8_t* dexps)
{
	int dexp[256];
	int group_size = exp_strategy + (exp_strategy == EXP_D45);

	if (ngrps < 0 || ngrps * 3 > 256) return -1;

	for (int grp = 0, i = 0; grp < ngrps; grp++) {
		int expacc = get_bits(gbc, 7);
		if (expacc >= 125) return -1;
		dexp[i++] = ff_ac3_ungroup_3_in_7_bits_tab[expacc][0];
		dexp[i++] = ff_ac3_ungroup_3_in_7_bits_tab[expacc][1];
		dexp[i++] = ff_ac3_ungroup_3_in_7_bits_tab[expacc][2];
	}

	int prevexp = absexp;
	for (int i = 0, j = 0; i < ngrps * 3; i++) {
		prevexp += dexp[i] - 2;
		if ((unsigned)prevexp > 24u) return -1;
		switch (group_size) {
		case 4: dexps[j++] = prevexp;
		        dexps[j++] = prevexp;
		        // fall through
		case 2: dexps[j++] = prevexp;
		        // fall through
		case 1: dexps[j++] = prevexp;
		}
	}
	return 0;
}

static void bit_alloc_calc_psd(int8_t* exp_, int start, int end, int16_t* psd, int16_t* band_psd)
{
	for (int bin = start; bin < end; bin++)
		psd[bin] = (int16_t)(3072 - (exp_[bin] << 7));

	int bin = start;
	int band = ff_ac3_bin_to_band_tab[start];
	do {
		int v = psd[bin++];
		int band_end = FFMIN(ff_ac3_band_start_tab[band + 1], end);
		for (; bin < band_end; bin++) {
			int mx = FFMAX(v, psd[bin]);
			int adr = FFMIN(mx - ((v + psd[bin] + 1) >> 1), 255);
			v = mx + ac3_log_add_tab[adr];
		}
		band_psd[band++] = (int16_t)v;
	} while (end > ff_ac3_band_start_tab[band]);
}

static inline int calc_lowcomp1(int a, int b0, int b1, int c)
{
	if ((b0 + 256) == b1)   a = c;
	else if (b0 > b1)       a = FFMAX(a - 64, 0);
	return a;
}

static inline int calc_lowcomp(int a, int b0, int b1, int bin)
{
	if (bin < 7)       return calc_lowcomp1(a, b0, b1, 384);
	else if (bin < 20) return calc_lowcomp1(a, b0, b1, 320);
	else               return FFMAX(a - 128, 0);
}

static int bit_alloc_calc_mask(AC3Ctx* s, int16_t* band_psd, int start, int end,
                               int fast_gain, int is_lfe, int dba_mode, int dba_nsegs,
                               uint8_t* dba_offsets, uint8_t* dba_lengths,
                               uint8_t* dba_values, int16_t* mask)
{
	int16_t excite[AC3_CRITICAL_BANDS];
	int band, band_start, band_end, begin, end1;
	int lowcomp, fastleak, slowleak;

	if (end <= 0) return -1;

	band_start = ff_ac3_bin_to_band_tab[start];
	band_end   = ff_ac3_bin_to_band_tab[end - 1] + 1;

	if (band_start == 0) {
		lowcomp = 0;
		lowcomp = calc_lowcomp1(lowcomp, band_psd[0], band_psd[1], 384);
		excite[0] = (int16_t)(band_psd[0] - fast_gain - lowcomp);
		lowcomp = calc_lowcomp1(lowcomp, band_psd[1], band_psd[2], 384);
		excite[1] = (int16_t)(band_psd[1] - fast_gain - lowcomp);
		begin = 7;
		fastleak = slowleak = 0;
		for (band = 2; band < 7; band++) {
			if (!(is_lfe && band == 6))
				lowcomp = calc_lowcomp1(lowcomp, band_psd[band], band_psd[band + 1], 384);
			fastleak = band_psd[band] - fast_gain;
			slowleak = band_psd[band] - s->slow_gain;
			excite[band] = (int16_t)(fastleak - lowcomp);
			if (!(is_lfe && band == 6)) {
				if (band_psd[band] <= band_psd[band + 1]) { begin = band + 1; break; }
			}
		}

		end1 = FFMIN(band_end, 22);
		for (band = begin; band < end1; band++) {
			if (!(is_lfe && band == 6))
				lowcomp = calc_lowcomp(lowcomp, band_psd[band], band_psd[band + 1], band);
			fastleak = FFMAX(fastleak - s->fast_decay, band_psd[band] - fast_gain);
			slowleak = FFMAX(slowleak - s->slow_decay, band_psd[band] - s->slow_gain);
			excite[band] = (int16_t)FFMAX(fastleak - lowcomp, slowleak);
		}
		begin = 22;
	} else {
		// coupling channel
		begin = band_start;
		fastleak = (s->cpl_fast_leak << 8) + 768;
		slowleak = (s->cpl_slow_leak << 8) + 768;
	}

	for (band = begin; band < band_end; band++) {
		fastleak = FFMAX(fastleak - s->fast_decay, band_psd[band] - fast_gain);
		slowleak = FFMAX(slowleak - s->slow_decay, band_psd[band] - s->slow_gain);
		excite[band] = (int16_t)FFMAX(fastleak, slowleak);
	}

	for (band = band_start; band < band_end; band++) {
		int tmp = s->db_per_bit - band_psd[band];
		if (tmp > 0) excite[band] += tmp >> 2;
		mask[band] = (int16_t)FFMAX(ac3_hearing_threshold_tab[band >> s->sr_shift][s->sr_code], excite[band]);
	}

	if (dba_mode == DBA_REUSE || dba_mode == DBA_NEW) {
		if (dba_nsegs > 8) return -1;
		band = band_start;
		for (int seg = 0; seg < dba_nsegs; seg++) {
			band += dba_offsets[seg];
			if (band >= AC3_CRITICAL_BANDS || dba_lengths[seg] > AC3_CRITICAL_BANDS - band)
				return -1;
			int delta = (dba_values[seg] >= 4) ? (dba_values[seg] - 3) * 128
			                                   : (dba_values[seg] - 4) * 128;
			for (int i = 0; i < dba_lengths[seg]; i++)
				mask[band++] += delta;
		}
	}
	return 0;
}

static void bit_alloc_calc_bap(int16_t* mask, int16_t* psd, int start, int end,
                               int snr_offset, int floor_, uint8_t* bap)
{
	// special case: snr offset of -960 means every bap is zero
	if (snr_offset == -960) { memset(bap, 0, AC3_MAX_COEFS); return; }

	int bin = start;
	int band = ff_ac3_bin_to_band_tab[start];
	int band_end;
	do {
		int m = (FFMAX(mask[band] - snr_offset - floor_, 0) & 0x1FE0) + floor_;
		band_end = ff_ac3_band_start_tab[++band];
		band_end = FFMIN(band_end, end);
		for (; bin < band_end; bin++) {
			int address = clip_uintp2_6((psd[bin] - m) >> 5);
			bap[bin] = ff_ac3_bap_tab[address];
		}
	} while (end > band_end);
}

///////////////////////////////////////////////////////////////////////////////
// Mantissas (A/52 7.3), coupling (A/52 7.4), rematrixing (A/52 7.5) — ac3dec.c

// libavcodec/mathops.h MULH: high 32 bits of a 32x32 -> 64 signed product.
static inline int MULH(int a, int b) { return (int)(((int64_t)a * (int64_t)b) >> 32); }

// A/52 7.4.3: rebuild coupled channels from the coupling channel and coordinates.
static void calc_transform_coeffs_cpl(AC3Ctx* s)
{
	int bin = s->start_freq[CPL_CH];
	for (int band = 0; band < s->num_cpl_bands; band++) {
		int band_start = bin;
		int band_end = bin + s->cpl_band_sizes[band];
		if (band_end > AC3_MAX_COEFS) band_end = AC3_MAX_COEFS;
		for (int ch = 1; ch <= s->fbw_channels; ch++) {
			if (s->channel_in_cpl[ch]) {
				int cpl_coord = s->cpl_coords[ch][band] << 5;
				for (bin = band_start; bin < band_end; bin++)
					s->fixed_coeffs[ch][bin] = MULH(s->fixed_coeffs[CPL_CH][bin] * (1 << 4), cpl_coord);
				if (ch == 2 && s->phase_flags[band]) {
					for (bin = band_start; bin < band_end; bin++)
						s->fixed_coeffs[2][bin] = -s->fixed_coeffs[2][bin];
				}
			}
		}
		bin = band_end;
	}
}

static void decode_transform_coeffs_ch(AC3Ctx* s, int ch_index, MantGroups* m)
{
	int start_freq = s->start_freq[ch_index];
	int end_freq   = s->end_freq[ch_index];
	uint8_t* baps   = s->bap[ch_index];
	int8_t*  exps   = s->dexps[ch_index];
	int32_t* coeffs = s->fixed_coeffs[ch_index];
	int dither = (ch_index == CPL_CH) || s->dither_flag[ch_index];
	GetBits* gbc = &s->gbc;

	for (int freq = start_freq; freq < end_freq; freq++) {
		int bap = baps[freq];
		int mantissa;
		switch (bap) {
		case 0:
			// random noise, approximately -0.707 .. 0.707
			if (dither && s->dither_enabled)
				mantissa = (int)((((lfg_get(&s->dith_state) >> 8) * 181) >> 8)) - 5931008;
			else
				mantissa = 0;
			break;
		case 1:
			if (m->b1) { m->b1--; mantissa = m->b1_mant[m->b1]; }
			else {
				int bits = get_bits(gbc, 5);
				mantissa      = ff_ac3_bap1_mantissas[bits][0];
				m->b1_mant[1] = ff_ac3_bap1_mantissas[bits][1];
				m->b1_mant[0] = ff_ac3_bap1_mantissas[bits][2];
				m->b1 = 2;
			}
			break;
		case 2:
			if (m->b2) { m->b2--; mantissa = m->b2_mant[m->b2]; }
			else {
				int bits = get_bits(gbc, 7);
				mantissa      = ff_ac3_bap2_mantissas[bits][0];
				m->b2_mant[1] = ff_ac3_bap2_mantissas[bits][1];
				m->b2_mant[0] = ff_ac3_bap2_mantissas[bits][2];
				m->b2 = 2;
			}
			break;
		case 3:
			mantissa = ff_ac3_bap3_mantissas[get_bits(gbc, 3)];
			break;
		case 4:
			if (m->b4) { m->b4 = 0; mantissa = m->b4_mant; }
			else {
				int bits = get_bits(gbc, 7);
				mantissa   = ff_ac3_bap4_mantissas[bits][0];
				m->b4_mant = ff_ac3_bap4_mantissas[bits][1];
				m->b4 = 1;
			}
			break;
		case 5:
			mantissa = ff_ac3_bap5_mantissas[get_bits(gbc, 4)];
			break;
		default:	// 6..15: plain signed mantissa, left-justified into 24 bits
			if (bap > 15) bap = 15;
			mantissa = (int)((unsigned)get_sbits(gbc, ff_ac3_quantization_tab[bap])
			                 << (24 - ff_ac3_quantization_tab[bap]));
			break;
		}
		coeffs[freq] = mantissa >> exps[freq];
	}
}

// A/52 7.3.4: coupled channels that do not dither must not inherit the coupling
// channel's dithered zero-bit bins.
static void remove_dithering(AC3Ctx* s)
{
	for (int ch = 1; ch <= s->fbw_channels; ch++) {
		if (!s->dither_flag[ch] && s->channel_in_cpl[ch]) {
			for (int i = s->start_freq[CPL_CH]; i < s->end_freq[CPL_CH]; i++)
				if (!s->bap[CPL_CH][i]) s->fixed_coeffs[ch][i] = 0;
		}
	}
}

static void decode_transform_coeffs(AC3Ctx* s)
{
	MantGroups m;
	int got_cplchan = 0;
	m.b1 = m.b2 = m.b4 = 0;

	for (int ch = 1; ch <= s->channels; ch++) {
		int end;
		decode_transform_coeffs_ch(s, ch, &m);
		// the coupling channel's coefficients follow the first coupled channel's
		if (s->channel_in_cpl[ch]) {
			if (!got_cplchan) {
				decode_transform_coeffs_ch(s, CPL_CH, &m);
				calc_transform_coeffs_cpl(s);
				got_cplchan = 1;
			}
			end = s->end_freq[CPL_CH];
		} else {
			end = s->end_freq[ch];
		}
		do { s->fixed_coeffs[ch][end] = 0; } while (++end < 256);
	}

	remove_dithering(s);
}

// A/52 7.5.4 stereo rematrixing.
static void do_rematrixing(AC3Ctx* s)
{
	int end = FFMIN(s->end_freq[1], s->end_freq[2]);

	for (int bnd = 0; bnd < s->num_rematrixing_bands; bnd++) {
		if (s->rematrixing_flags[bnd]) {
			int bndend = FFMIN(end, ff_ac3_rematrix_band_tab[bnd + 1]);
			for (int i = ff_ac3_rematrix_band_tab[bnd]; i < bndend; i++) {
				int tmp0 = s->fixed_coeffs[1][i];
				s->fixed_coeffs[1][i] += s->fixed_coeffs[2][i];
				s->fixed_coeffs[2][i]  = tmp0 - s->fixed_coeffs[2][i];
			}
		}
	}
}

///////////////////////////////////////////////////////////////////////////////
// Coupling strategy / coordinates and band structure (A/52 7.4) — ac3dec.c.
// The E-AC-3 arms (enhanced coupling, stereo-implied coupling, the band-structure
// presence bit) are gone: in plain AC-3 the band structure is always transmitted.

// Table E2.16 Default Coupling Banding Structure.
static const uint8_t ff_ac3_default_cpl_band_struct[18] = {
	0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1
};

static void decode_band_structure(GetBits* gbc, int blk, int start_subband, int end_subband,
                                  const uint8_t* default_band_struct, int* num_bands,
                                  uint8_t* band_sizes, uint8_t* band_struct, int band_struct_size)
{
	uint8_t bnd_sz[22];
	int n_subbands = end_subband - start_subband;
	int n_bands = 0;

	if (!blk) memcpy(band_struct, default_band_struct, band_struct_size);
	band_struct += start_subband + 1;

	for (int subbnd = 0; subbnd < n_subbands - 1; subbnd++)
		band_struct[subbnd] = get_bits1(gbc);

	if (num_bands || band_sizes) {
		n_bands = n_subbands;
		bnd_sz[0] = 12;
		for (int bnd = 0, subbnd = 1; subbnd < n_subbands; subbnd++) {
			if (band_struct[subbnd - 1]) { n_bands--; bnd_sz[bnd] += 12; }
			else                         { bnd_sz[++bnd] = 12; }
		}
	}
	if (num_bands)  *num_bands = n_bands;
	if (band_sizes) memcpy(band_sizes, bnd_sz, n_bands);
}

static int coupling_strategy(AC3Ctx* s, int blk, uint8_t* bit_alloc_stages)
{
	GetBits* bc = &s->gbc;

	memset(bit_alloc_stages, 3, AC3_MAX_CHANNELS);
	s->cpl_in_use[blk] = get_bits1(bc);

	if (s->cpl_in_use[blk]) {
		if (s->channel_mode < AC3_CHMODE_STEREO)
			return -1;	// coupling is not allowed in mono or dual-mono

		for (int ch = 1; ch <= s->fbw_channels; ch++)
			s->channel_in_cpl[ch] = get_bits1(bc);

		if (s->channel_mode == AC3_CHMODE_STEREO)
			s->phase_flags_in_use = get_bits1(bc);

		int cpl_start_subband = get_bits(bc, 4);
		int cpl_end_subband   = get_bits(bc, 4) + 3;
		if (cpl_start_subband >= cpl_end_subband) return -1;

		s->start_freq[CPL_CH] = cpl_start_subband * 12 + 37;
		s->end_freq[CPL_CH]   = cpl_end_subband   * 12 + 37;

		decode_band_structure(bc, blk, cpl_start_subband, cpl_end_subband,
		                      ff_ac3_default_cpl_band_struct,
		                      &s->num_cpl_bands, s->cpl_band_sizes,
		                      s->cpl_band_struct, sizeof(s->cpl_band_struct));
	} else {
		for (int ch = 1; ch <= s->fbw_channels; ch++) {
			s->channel_in_cpl[ch] = 0;
			s->first_cpl_coords[ch] = 1;
		}
		s->first_cpl_leak = 0;
		s->phase_flags_in_use = 0;
	}
	return 0;
}

static int coupling_coordinates(AC3Ctx* s, int blk)
{
	GetBits* bc = &s->gbc;
	int cpl_coords_exist = 0;

	for (int ch = 1; ch <= s->fbw_channels; ch++) {
		if (s->channel_in_cpl[ch]) {
			if (get_bits1(bc)) {
				s->first_cpl_coords[ch] = 0;
				cpl_coords_exist = 1;
				int master_cpl_coord = 3 * get_bits(bc, 2);
				for (int bnd = 0; bnd < s->num_cpl_bands; bnd++) {
					int cpl_coord_exp  = get_bits(bc, 4);
					int cpl_coord_mant = get_bits(bc, 4);
					if (cpl_coord_exp == 15) s->cpl_coords[ch][bnd] = cpl_coord_mant << 22;
					else                     s->cpl_coords[ch][bnd] = (cpl_coord_mant + 16) << 21;
					s->cpl_coords[ch][bnd] >>= (cpl_coord_exp + master_cpl_coord);
				}
			} else if (!blk) {
				return -1;	// new coupling coordinates must be present in block 0
			}
		} else {
			s->first_cpl_coords[ch] = 1;
		}
	}

	if (s->channel_mode == AC3_CHMODE_STEREO && cpl_coords_exist) {
		for (int bnd = 0; bnd < s->num_cpl_bands; bnd++)
			s->phase_flags[bnd] = s->phase_flags_in_use ? get_bits1(bc) : 0;
	}
	return 0;
}

///////////////////////////////////////////////////////////////////////////////
// Transform + windowed overlap-add (A/52 7.9.4)

/*
	Coefficient headroom scale (A/52 7.3.3 mantissas are 24-bit fixed point).

	FFmpeg multiplies the integer mantissas by `dynrng * 1/4194304` (2^-22) and feeds
	them to av_tx. Our transform is instead the plain A/52 7.9.4 form, whose leading
	-(2/N) factor imdct_channel() applies as -(1/M). Composing the two, a long block
	(M=256) ends up scaling the integer mantissas by

	    2^-14 * (1/256) = 2^-22

	which is exactly FFmpeg's headroom factor — i.e. av_tx's IMDCT already carries the
	spec's -(1/M), and this constant is not a fudge factor but the 2^-22 written as
	the part that does NOT depend on the transform length. Measured against FFmpeg's
	reference PCM the least-squares scale is 1.0 (tests/AC3.test.js asserts it to
	within 5%, so a drift here fails loudly rather than quietly changing loudness).

	Because the length-dependent half lives in imdct_channel(), long (M=256) and short
	(M=128) blocks automatically get the 2:1 relative level A/52 7.9.4 gives them. The
	superseded JS decoder folded everything into one constant used for both, which was
	only ever exercised on long blocks.
*/
static const double AC3_COEFF_SCALE = 1.0 / 16384.0;	// 2^-14

// One channel's IMDCT + KBD window + overlap-add into `pcm` (256 samples).
static void imdct_channel(AC3Ctx* s, int ch, float* pcm)
{
	double z[512];
	double* delay = s->delay[ch];
	const double* w = s->window;

	/*
		A/52 7.9.4's leading -(2/N) factor. N is the WINDOW length, 512, for both
		block types — not the transform's coefficient count — so short blocks use the
		same -1/256 as long ones. (Cross-check: FFmpeg passes scale=1.0 to both
		av_tx_init(len=128) and av_tx_init(len=256) and applies the same 2^-22
		coefficient gain either way, so its IMDCT gain is length-independent too.)

		Getting this wrong is nearly invisible: a per-length -(1/M) still gives
		bit-exact output on any stream without block switching, and FFmpeg's AC-3
		ENCODER never emits short blocks, so no stream it produces can catch it. It
		took real Dolby-encoded DVD audio to expose — hence the short-block coverage
		assertion in tests/AC3.test.js.
	*/
	const double norm = -1.0 / 256.0;

	if (s->block_switch[ch]) {
		/*
			Two interleaved 128-coefficient transforms tile the 512-sample window: the
			EVEN coefficients supply the half that is windowed against the previous
			block's tail, the ODD ones the half carried into the next block (A/52
			7.9.4.2 — and exactly FFmpeg's two tx_128 calls, one into tmp_output and
			one straight into delay[]).

			These must be spliced as CORES, not as two full 256-sample transforms
			concatenated. A 128-coefficient transform's 256 outputs are antisymmetric
			about 63.5 and symmetric about 191.5, whereas z[0..255] here has to be
			antisymmetric about 127.5 — a different structure, so concatenating the
			full transforms silently produces the wrong signal in exactly the frames
			that use block switching.
		*/
		double even[128], odd[128], ce[128], co[128];
		for (int k = 0; k < 128; k++) {
			even[k] = s->transform_coeffs[ch][2 * k];
			odd[k]  = s->transform_coeffs[ch][2 * k + 1];
		}
		imdct_core(even, ce, 128);
		imdct_core(odd,  co, 128);
		for (int n = 0; n < 128; n++) {
			z[n]       = -ce[127 - n];
			z[128 + n] =  ce[n];
			z[256 + n] =  co[n];
			z[384 + n] =  co[127 - n];
		}
	} else {
		imdct(s->transform_coeffs[ch], z, 256);
	}
	for (int n = 0; n < 512; n++) z[n] *= norm;

	for (int n = 0; n < 256; n++)
		pcm[n] = (float)(w[n] * z[n] + delay[n]);
	for (int n = 0; n < 256; n++)
		delay[n] = w[255 - n] * z[256 + n];
}

/*
	FFmpeg's ac3_upmix_delay() — reproduced deliberately, quirk and all.

	FFmpeg sets `downmixed = 1` in ac3_decode_init() unconditionally, meaning "the
	overlap-add delay currently holds downmixed channels". The first block that mixes
	long and short transforms then "upmixes" the delay back before transforming:

	    if (different_transforms) {
	        if (s->downmixed) { s->downmixed = 0; ac3_upmix_delay(s); }

	When no downmix is configured — our case, and FFmpeg's own default for 5.1 out —
	the delay was never downmixed, so this zeroes and shuffles a block of real audio
	history exactly once per stream, at the first block-switching block. It is an
	upstream bug, but it is audible in the reference decoder's output, so matching it
	is what makes bit-exactness a meaningful claim: without it one frame per stream
	sits ~45 dB below the reference while every other frame is exact.

	To prefer clean audio over reference fidelity, drop the call — nothing else
	depends on `downmixed`. FFmpeg indexes delay[ch-1]; ours is 1-based, hence the
	shifted indices.
*/
static void ac3_upmix_delay(AC3Ctx* s)
{
	const size_t n = sizeof(s->delay[0]);
	switch (s->channel_mode) {
	case AC3_CHMODE_DUALMONO:
	case AC3_CHMODE_STEREO:
		memcpy(s->delay[2], s->delay[1], n);	// upmix mono to stereo
		break;
	case AC3_CHMODE_2F2R:
		memset(s->delay[4], 0, n);
		// fall through
	case AC3_CHMODE_2F1R:
		memset(s->delay[3], 0, n);
		break;
	case AC3_CHMODE_3F2R:
		memset(s->delay[5], 0, n);
		// fall through
	case AC3_CHMODE_3F1R:
		memset(s->delay[4], 0, n);
		// fall through
	case AC3_CHMODE_3F:
		memcpy(s->delay[3], s->delay[2], n);
		memset(s->delay[2], 0, n);
		break;
	}
}

///////////////////////////////////////////////////////////////////////////////
// One audio block (A/52 5.4.2 audblk) — ac3dec.c decode_audio_block(), AC-3 only.

static int decode_audio_block(AC3Ctx* s, int blk, float* const* outch)
{
	GetBits* gbc = &s->gbc;
	int fbw_channels = s->fbw_channels;
	int channel_mode = s->channel_mode;
	uint8_t bit_alloc_stages[AC3_MAX_CHANNELS] = { 0 };
	int i, bnd, seg, ch;

	// block switch flags
	int different_transforms = 0;
	for (ch = 1; ch <= fbw_channels; ch++) {
		s->block_switch[ch] = get_bits1(gbc);
		if (ch > 1 && s->block_switch[ch] != s->block_switch[1])
			different_transforms = 1;
	}

	// dither flags
	for (ch = 1; ch <= fbw_channels; ch++)
		s->dither_flag[ch] = get_bits1(gbc);

	// dynamic range (A/52 7.7.1); read twice in dual-mono
	i = !s->channel_mode;
	do {
		if (get_bits1(gbc)) {
			int range_bits = get_bits(gbc, 8);
			double range = dynamic_range_tab[range_bits];
			// asymmetric DRC is allowed when drc_scale > 1 (quiet sounds boosted)
			if (range_bits <= 127 || s->drc_scale <= 1.0f)
				s->dynamic_range[i] = pow(range, (double)s->drc_scale);
			else
				s->dynamic_range[i] = range;
		} else if (blk == 0) {
			s->dynamic_range[i] = 1.0;
		}
	} while (i--);

	// coupling strategy
	if (get_bits1(gbc)) {
		if (coupling_strategy(s, blk, bit_alloc_stages) < 0) return -1;
	} else {
		if (!blk) return -1;	// new coupling strategy must be present in block 0
		s->cpl_in_use[blk] = s->cpl_in_use[blk - 1];
	}
	int cpl_in_use = s->cpl_in_use[blk];

	// coupling coordinates
	if (cpl_in_use) {
		if (coupling_coordinates(s, blk) < 0) return -1;
	}

	// stereo rematrixing strategy
	if (channel_mode == AC3_CHMODE_STEREO) {
		if (get_bits1(gbc)) {
			s->num_rematrixing_bands = 4;
			if (cpl_in_use && s->start_freq[CPL_CH] <= 61)
				s->num_rematrixing_bands -= 1 + (s->start_freq[CPL_CH] == 37);
			for (bnd = 0; bnd < s->num_rematrixing_bands; bnd++)
				s->rematrixing_flags[bnd] = get_bits1(gbc);
		} else if (!blk) {
			s->num_rematrixing_bands = 0;
		}
	}

	// exponent strategies
	for (ch = !cpl_in_use; ch <= s->channels; ch++) {
		s->exp_strategy[blk][ch] = get_bits(gbc, 2 - (ch == s->lfe_ch));
		if (s->exp_strategy[blk][ch] != EXP_REUSE) bit_alloc_stages[ch] = 3;
	}

	// channel bandwidth
	for (ch = 1; ch <= fbw_channels; ch++) {
		s->start_freq[ch] = 0;
		if (s->exp_strategy[blk][ch] != EXP_REUSE) {
			int prev = s->end_freq[ch];
			if (s->channel_in_cpl[ch]) {
				s->end_freq[ch] = s->start_freq[CPL_CH];
			} else {
				int bandwidth_code = get_bits(gbc, 6);
				if (bandwidth_code > 60) return -1;
				s->end_freq[ch] = bandwidth_code * 3 + 73;
			}
			int group_size = 3 << (s->exp_strategy[blk][ch] - 1);
			s->num_exp_groups[ch] = (s->end_freq[ch] + group_size - 4) / group_size;
			if (blk > 0 && s->end_freq[ch] != prev)
				memset(bit_alloc_stages, 3, AC3_MAX_CHANNELS);
		}
	}
	if (cpl_in_use && s->exp_strategy[blk][CPL_CH] != EXP_REUSE) {
		s->num_exp_groups[CPL_CH] = (s->end_freq[CPL_CH] - s->start_freq[CPL_CH]) /
		                            (3 << (s->exp_strategy[blk][CPL_CH] - 1));
	}

	// exponents
	for (ch = !cpl_in_use; ch <= s->channels; ch++) {
		if (s->exp_strategy[blk][ch] != EXP_REUSE) {
			s->dexps[ch][0] = (int8_t)(get_bits(gbc, 4) << !ch);
			if (decode_exponents(gbc, s->exp_strategy[blk][ch], s->num_exp_groups[ch],
			                     s->dexps[ch][0], &s->dexps[ch][s->start_freq[ch] + !!ch]))
				return -1;
			if (ch != CPL_CH && ch != s->lfe_ch)
				skip_bits(gbc, 2);	// gainrng
		}
	}

	// bit allocation parameters
	if (get_bits1(gbc)) {
		s->slow_decay = ff_ac3_slow_decay_tab[get_bits(gbc, 2)] >> s->sr_shift;
		s->fast_decay = ff_ac3_fast_decay_tab[get_bits(gbc, 2)] >> s->sr_shift;
		s->slow_gain  = ff_ac3_slow_gain_tab[get_bits(gbc, 2)];
		s->db_per_bit = ff_ac3_db_per_bit_tab[get_bits(gbc, 2)];
		s->floor_     = ff_ac3_floor_tab[get_bits(gbc, 3)];
		for (ch = !cpl_in_use; ch <= s->channels; ch++)
			bit_alloc_stages[ch] = (uint8_t)FFMAX(bit_alloc_stages[ch], 2);
	} else if (!blk) {
		return -1;	// new bit allocation info must be present in block 0
	}

	// SNR offsets and fast gains
	if (get_bits1(gbc)) {
		int snr = 0;
		int csnr = ((int)get_bits(gbc, 6) - 15) << 4;
		for (ch = !cpl_in_use; ch <= s->channels; ch++) {
			// snroffststr is fixed at 2 for plain AC-3, so every channel carries its
			// own 4-bit offset (FFmpeg's `ch == i || snr_offset_strategy == 2`).
			snr = (csnr + (int)get_bits(gbc, 4)) << 2;
			if (blk && s->snr_offset[ch] != snr)
				bit_alloc_stages[ch] = (uint8_t)FFMAX(bit_alloc_stages[ch], 1);
			s->snr_offset[ch] = snr;

			int prev = s->fast_gain[ch];
			s->fast_gain[ch] = ff_ac3_fast_gain_tab[get_bits(gbc, 3)];
			if (blk && prev != s->fast_gain[ch])
				bit_alloc_stages[ch] = (uint8_t)FFMAX(bit_alloc_stages[ch], 2);
		}
	} else if (!blk) {
		return -1;	// new snr offsets must be present in block 0
	}

	// coupling leak information
	if (cpl_in_use) {
		if (s->first_cpl_leak || get_bits1(gbc)) {
			int fl = get_bits(gbc, 3);
			int sl = get_bits(gbc, 3);
			if (blk && (fl != s->cpl_fast_leak || sl != s->cpl_slow_leak))
				bit_alloc_stages[CPL_CH] = (uint8_t)FFMAX(bit_alloc_stages[CPL_CH], 2);
			s->cpl_fast_leak = fl;
			s->cpl_slow_leak = sl;
		} else if (!blk) {
			return -1;	// new coupling leak info must be present in block 0
		}
		s->first_cpl_leak = 0;
	}

	// delta bit allocation
	if (get_bits1(gbc)) {
		for (ch = !cpl_in_use; ch <= fbw_channels; ch++) {
			s->dba_mode[ch] = get_bits(gbc, 2);
			if (s->dba_mode[ch] == DBA_RESERVED) return -1;
			bit_alloc_stages[ch] = (uint8_t)FFMAX(bit_alloc_stages[ch], 2);
		}
		for (ch = !cpl_in_use; ch <= fbw_channels; ch++) {
			if (s->dba_mode[ch] == DBA_NEW) {
				s->dba_nsegs[ch] = get_bits(gbc, 3) + 1;
				for (seg = 0; seg < s->dba_nsegs[ch]; seg++) {
					s->dba_offsets[ch][seg] = get_bits(gbc, 5);
					s->dba_lengths[ch][seg] = get_bits(gbc, 4);
					s->dba_values[ch][seg]  = get_bits(gbc, 3);
				}
				bit_alloc_stages[ch] = (uint8_t)FFMAX(bit_alloc_stages[ch], 2);
			}
		}
	} else if (blk == 0) {
		for (ch = 0; ch <= s->channels; ch++) s->dba_mode[ch] = DBA_NONE;
	}

	// bit allocation
	for (ch = !cpl_in_use; ch <= s->channels; ch++) {
		if (bit_alloc_stages[ch] > 2)
			bit_alloc_calc_psd(s->dexps[ch], s->start_freq[ch], s->end_freq[ch],
			                   s->psd[ch], s->band_psd[ch]);
		if (bit_alloc_stages[ch] > 1) {
			if (bit_alloc_calc_mask(s, s->band_psd[ch], s->start_freq[ch], s->end_freq[ch],
			                        s->fast_gain[ch], (ch == s->lfe_ch),
			                        s->dba_mode[ch], s->dba_nsegs[ch], s->dba_offsets[ch],
			                        s->dba_lengths[ch], s->dba_values[ch], s->mask[ch]))
				return -1;
		}
		if (bit_alloc_stages[ch] > 0)
			bit_alloc_calc_bap(s->mask[ch], s->psd[ch], s->start_freq[ch], s->end_freq[ch],
			                   s->snr_offset[ch], s->floor_, s->bap[ch]);
	}

	// skipped dummy data
	if (get_bits1(gbc)) {
		int skipl = get_bits(gbc, 9);
		skip_bits(gbc, 8 * skipl);
	}

	// mantissas (this also uncouples channels when coupling is in use)
	decode_transform_coeffs(s);

	if (channel_mode == AC3_CHMODE_STEREO)
		do_rematrixing(s);

	// scale coefficients (headroom + dynrng), then transform
	for (ch = 1; ch <= s->channels; ch++) {
		int audio_channel = 0;
		if (channel_mode == AC3_CHMODE_DUALMONO && ch <= 2) audio_channel = 2 - ch;
		double gain = (s->heavy_compression && s->compression_exists[audio_channel])
		            ? s->heavy_dynamic_range[audio_channel]
		            : s->dynamic_range[audio_channel];
		gain *= AC3_COEFF_SCALE;
		for (i = 0; i < AC3_MAX_COEFS; i++)
			s->transform_coeffs[ch][i] = (double)s->fixed_coeffs[ch][i] * gain;
	}

	// The delay samples have already been "downmixed", so upmix them before the
	// transform — see ac3_upmix_delay() for why this fires once even with no downmix.
	if (different_transforms && s->downmixed) {
		s->downmixed = 0;
		ac3_upmix_delay(s);
	}

	for (ch = 1; ch <= s->channels; ch++) {
		if (s->block_switch[ch]) s->short_blocks++;
		imdct_channel(s, ch, outch[ch - 1] + blk * AC3_BLOCK_SIZE);
	}

	return 0;
}

///////////////////////////////////////////////////////////////////////////////
// Frame decode and the extern "C" boundary

extern "C" {

AC3Ctx* ac3_open(void)
{
	ac3_init_static();
	AC3Ctx* s = (AC3Ctx*)calloc(1, sizeof(AC3Ctx));
	if (!s) return 0;
	kbd_window_init(s->window, 5.0, AC3_BLOCK_SIZE);
	lfg_init(&s->dith_state);
	s->drc_scale = 1.0f;
	s->dither_enabled = 1;
	s->downmixed = 1;		// FFmpeg ac3_decode_init(); see ac3_upmix_delay()
	return s;
}

void ac3_close(AC3Ctx* s) { free(s); }

// Drop inter-frame state (the overlap-add delay and the dither RNG) — used on seek
// so a new position does not inherit the previous one's tail. Mirrors FFmpeg's
// ac3_decode_flush().
void ac3_reset(AC3Ctx* s)
{
	if (!s) return;
	memset(s->delay, 0, sizeof(s->delay));
	memset(s->cpl_in_use, 0, sizeof(s->cpl_in_use));
	memset(s->channel_in_cpl, 0, sizeof(s->channel_in_cpl));
	memset(s->exp_strategy, 0, sizeof(s->exp_strategy));
	memset(s->dexps, 0, sizeof(s->dexps));
	lfg_init(&s->dith_state);
	s->downmixed = 1;
}

void ac3_set_drc_scale(AC3Ctx* s, float scale) { if (s) s->drc_scale = scale; }
void ac3_set_dither(AC3Ctx* s, int on)         { if (s) s->dither_enabled = !!on; }

// Parse just the syncinfo/bsi of the frame at `buf`, without touching decoder state.
// info = { frameSize, sampleRate, channels, bitRate, bitstreamId, channelMode }.
int ac3_probe_frame(const uint8_t* buf, int len, int* info)
{
	GetBits gbc;
	AC3Header h;

	if (!buf || len < 8) return 0;
	init_get_bits(&gbc, buf, len);
	if (parse_frame_header(&gbc, &h) != AC3_PARSE_OK) return 0;

	info[0] = h.frame_size;
	info[1] = h.sample_rate;
	info[2] = h.channels;
	info[3] = h.bit_rate;
	info[4] = h.bitstream_id;
	info[5] = h.channel_mode;
	return 1;
}

// Decode ONE AC-3 frame starting at buf[0] (which must be the 0x0B77 syncword).
// info = { frameSize, sampleRate, channels, samplesPerChannel }. Returns 1 on
// success. On a mid-frame error the already-decoded blocks are kept and the
// remaining ones are zeroed, and 0 is returned — FFmpeg instead repeats the last
// good block, which needs error-concealment state we deliberately do not carry.
int ac3_decode_frame(AC3Ctx* s, const uint8_t* buf, int len, int* info)
{
	AC3Header h;
	float* outch[AC3_MAX_CHANNELS];

	if (!s || !buf || len < 8) return 0;

	init_get_bits(&s->gbc, buf, len);
	if (parse_frame_header(&s->gbc, &h) != AC3_PARSE_OK) return 0;
	if (h.frame_size > len) return 0;	// incomplete frame

	s->sr_code       = h.sr_code;
	s->sr_shift      = h.sr_shift;
	s->bitstream_id  = h.bitstream_id;
	s->bitstream_mode= h.bitstream_mode;
	s->channel_mode  = h.channel_mode;
	s->lfe_on        = h.lfe_on;
	s->sample_rate   = h.sample_rate;
	s->bit_rate      = h.bit_rate;
	s->channels      = h.channels;
	s->fbw_channels  = s->channels - s->lfe_on;
	s->lfe_ch        = s->fbw_channels + 1;
	s->frame_size    = h.frame_size;
	s->num_blocks    = AC3_MAX_BLOCKS;
	s->first_cpl_leak = 0;

	if (s->lfe_on) {
		s->start_freq[s->lfe_ch]     = 0;
		s->end_freq[s->lfe_ch]       = 7;
		s->num_exp_groups[s->lfe_ch] = 2;
		s->channel_in_cpl[s->lfe_ch] = 0;
	}

	for (int i = 0; i < (s->channel_mode ? 1 : 2); i++) {
		s->dialog_normalization[i] = h.dialog_normalization[i] ? h.dialog_normalization[i] : -31;
		s->compression_exists[i]   = h.compression_exists[i];
		if (s->compression_exists[i])
			s->heavy_dynamic_range[i] = heavy_dynamic_range_tab[h.heavy_dynamic_range[i]];
	}

	// No downmix: out_channels == channels, and the coded channels are routed to
	// FFmpeg's output order (see ff_ac3_dec_channel_map).
	s->out_channels = s->channels;
	s->output_mode  = s->channel_mode | (s->lfe_on ? AC3_OUTPUT_LFEON : 0);
	if (s->out_channels > 6) return 0;

	const uint8_t* channel_map = ff_ac3_dec_channel_map[s->output_mode & ~AC3_OUTPUT_LFEON][s->lfe_on];
	for (int slot = 0; slot < s->out_channels; slot++)
		outch[channel_map[slot]] = s->out[slot];

	int ok = 1;
	s->short_blocks = 0;
	for (int blk = 0; blk < s->num_blocks; blk++) {
		if (decode_audio_block(s, blk, outch)) {
			for (int b = blk; b < s->num_blocks; b++)
				for (int ch = 0; ch < s->out_channels; ch++)
					memset(s->out[ch] + b * AC3_BLOCK_SIZE, 0, AC3_BLOCK_SIZE * sizeof(float));
			ok = 0;
			break;
		}
	}

	info[0] = s->frame_size;
	info[1] = s->sample_rate;
	info[2] = s->out_channels;
	info[3] = s->num_blocks * AC3_BLOCK_SIZE;
	info[4] = s->short_blocks;
	return ok;
}

float* ac3_channel_ptr(AC3Ctx* s, int ch)
{
	if (!s || ch < 0 || ch >= 6) return 0;
	return s->out[ch];
}

/*
	Self-check for the FFT factoring of the IMDCT.

	Fills M coefficients from a fixed PRNG, evaluates the transform both ways — the
	direct O(N^2) sum straight out of A/52 7.9.4 and the DCT-IV/FFT path imdct() uses
	— and returns the largest absolute difference over all 2M outputs, scaled by 1e9
	so it can cross the WASM boundary as an int. Anything above a few ULPs of the
	sum's own magnitude means the factoring is wrong.
*/
int ac3_test_imdct(int M)
{
	double X[256], z[512];
	if (M != 128 && M != 256) return -1;

	uint32_t st = 12345u;
	for (int k = 0; k < M; k++) {
		st = st * 1664525u + 1013904223u;
		X[k] = (double)(int32_t)st / 2147483648.0;
	}

	imdct(X, z, M);

	double maxerr = 0.0;
	for (int n = 0; n < 2 * M; n++) {
		double ref = 0.0;
		for (int k = 0; k < M; k++)
			ref += X[k] * cos((M_PI / M) * (n + 0.5 + M / 2.0) * (k + 0.5));
		double e = fabs(ref - z[n]);
		if (e > maxerr) maxerr = e;
	}
	return (int)(maxerr * 1e9);
}

} // extern "C"
