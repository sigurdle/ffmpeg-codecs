/*
** DVD-Video LPCM (MPEG-PS private stream 1, substreams 0xA0-0xA7).
**
** A port of FFmpeg's libavcodec/pcm-dvd.c. Each private-stream-1 LPCM packet starts with
** the substream id, 3 bytes of frame count / first access unit pointer, then the 3-byte
** LPCM header this file parses:
**
**   byte 0  emphasis (1), mute (1), reserved (1), frame number (5)
**   byte 1  quantization (2), sample frequency (2), reserved (1), channels - 1 (3)
**   byte 2  dynamic range control (0x80 = off)
**
** Samples are big-endian. 16-bit samples are plain interleaved words; 20- and 24-bit
** samples come in groups of 4: the top 16 bits of the 4 samples first, then the low
** bits (two nibbles a byte for 20-bit, one byte each for 24-bit). A block is the groups
** that complete a set of samples for every channel; a block can straddle two packets.
*/

const FREQUENCIES = [48000, 96000, 44100, 32000];

/**
 * The LPCM header (pcm_dvd_parse_header), or null for the 28-bit depth FFmpeg refuses.
 * @returns {{bitsPerSample:number, sampleRate:number, channels:number, blockSize:number,
 *            samplesPerBlock:number, groupsPerBlock:number}|null}
 */
export function parseDvdLpcmHeader(b0, b1, b2)
{
	const bitsPerSample = 16 + ((b1 >> 6) & 3) * 4;
	if (bitsPerSample === 28) {
		return null;
	}
	const sampleRate = FREQUENCIES[(b1 >> 4) & 3];
	const channels = 1 + (b1 & 7);

	let blockSize, samplesPerBlock, groupsPerBlock = 0;
	if (bitsPerSample === 16) {
		samplesPerBlock = 1;
		blockSize = channels * 2;
	} else if (channels === 1 || channels === 2 || channels === 4) {
		// one group has all the samples needed
		blockSize = 4 * bitsPerSample / 8;
		samplesPerBlock = 4 / channels;
		groupsPerBlock = 1;
	} else if (channels === 8) {
		// two groups have all the samples needed
		blockSize = 8 * bitsPerSample / 8;
		samplesPerBlock = 1;
		groupsPerBlock = 2;
	} else {
		// need `channels` groups
		blockSize = 4 * channels * bitsPerSample / 8;
		samplesPerBlock = 4;
		groupsPerBlock = channels;
	}
	return {bitsPerSample, sampleRate, channels, blockSize, samplesPerBlock, groupsPerBlock};
}

function be16(src, o) { return (src[o] << 8) | src[o + 1]; }

/**
 * pcm_dvd_decode_samples: `blocks` blocks from src[off..] into `dst` (interleaved, int16
 * range for 16-bit, int32 range for 20/24-bit). Returns the next dst index.
 */
function decodeSamples(h, src, off, blocks, dst, d)
{
	let o = off;
	if (h.bitsPerSample === 16) {
		let samples = blocks * h.channels;
		while (samples--) {
			const v = be16(src, o);
			dst[d++] = (v << 16) >> 16;
			o += 2;
		}
		return d;
	}
	const twenty = h.bitsPerSample === 20;
	if (h.channels === 1) {
		while (blocks--) {
			for (let i = 2; i; i--) {
				const a = be16(src, o) << 16;
				const b = be16(src, o + 2) << 16;
				if (twenty) {
					const t = src[o + 4];
					dst[d++] = a + ((t & 0xf0) << 8);
					dst[d++] = b + ((t & 0x0f) << 12);
					o += 5;
				} else {
					dst[d++] = a + (src[o + 4] << 8);
					dst[d++] = b + (src[o + 5] << 8);
					o += 6;
				}
			}
		}
		return d;
	}
	while (blocks--) {
		for (let i = h.groupsPerBlock; i; i--) {
			const s0 = be16(src, o) << 16;
			const s1 = be16(src, o + 2) << 16;
			const s2 = be16(src, o + 4) << 16;
			const s3 = be16(src, o + 6) << 16;
			o += 8;
			if (twenty) {
				const t0 = src[o], t1 = src[o + 1];
				dst[d++] = s0 + ((t0 & 0xf0) << 8);
				dst[d++] = s1 + ((t0 & 0x0f) << 12);
				dst[d++] = s2 + ((t1 & 0xf0) << 8);
				dst[d++] = s3 + ((t1 & 0x0f) << 12);
				o += 2;
			} else {
				dst[d++] = s0 + (src[o] << 8);
				dst[d++] = s1 + (src[o + 1] << 8);
				dst[d++] = s2 + (src[o + 2] << 8);
				dst[d++] = s3 + (src[o + 3] << 8);
				o += 4;
			}
		}
	}
	return d;
}

/**
 * Decodes the packets of one LPCM substream in order (pcm_dvd_decode_frame), carrying a block
 * split across two packets. reset() at a seek.
 */
export class DvdLpcmDecoder
{
	constructor()
	{
		this._headerKey = -1;
		this._header = null;
		this._extra = new Uint8Array(8 * 3 * 4);
		this._extraCount = 0;
	}

	reset()
	{
		this._extraCount = 0;
	}

	/**
	 * @param {number} headerKey - the packet's 3 LPCM header bytes, b0 << 16 | b1 << 8 | b2
	 * @param {Uint8Array} data - the samples after the header
	 * @returns {{channelData: Float32Array[], sampleRate:number, channelCount:number, sampleCount:number}|null}
	 */
	decode(headerKey, data)
	{
		// The frame number (low 5 bits of byte 0) changes every packet; the rest is the format.
		const key = headerKey & 0xe0ffff;
		if (key !== this._headerKey) {
			this._header = parseDvdLpcmHeader((headerKey >> 16) & 0xff, (headerKey >> 8) & 0xff, headerKey & 0xff);
			this._headerKey = key;
			this._extraCount = 0;	// leftovers of another layout
		}
		const h = this._header;
		if (!h) {
			return null;
		}

		let blocks = ((data.length + this._extraCount) / h.blockSize) | 0;
		const out = new Int32Array(blocks * h.samplesPerBlock * h.channels);
		let d = 0;
		let src = 0;
		let left = data.length;

		if (this._extraCount) {
			const missing = h.blockSize - this._extraCount;
			if (left < missing) {
				this._extra.set(data, this._extraCount);
				this._extraCount += left;
				return null;
			}
			this._extra.set(data.subarray(0, missing), this._extraCount);
			d = decodeSamples(h, this._extra, 0, 1, out, d);
			src += missing;
			left -= missing;
			this._extraCount = 0;
			--blocks;
		}
		if (blocks > 0) {
			d = decodeSamples(h, data, src, blocks, out, d);
			src += blocks * h.blockSize;
			left -= blocks * h.blockSize;
		}
		if (left > 0) {
			this._extra.set(data.subarray(src, src + left), 0);
			this._extraCount = left;
		}

		const channels = h.channels;
		const sampleCount = (d / channels) | 0;
		if (!sampleCount) {
			return null;
		}
		const scale = h.bitsPerSample === 16 ? 1 / 32768 : 1 / 2147483648;
		const channelData = [];
		for (let c = 0; c < channels; ++c) {
			const plane = new Float32Array(sampleCount);
			for (let i = 0, k = c; i < sampleCount; ++i, k += channels) {
				plane[i] = out[k] * scale;
			}
			channelData.push(plane);
		}
		return {channelData, sampleRate: h.sampleRate, channelCount: channels, sampleCount};
	}
}
