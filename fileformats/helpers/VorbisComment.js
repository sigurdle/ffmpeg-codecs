import * as io from "@sigurd/core/io.js";

/*
 * A Vorbis comment block, the text of a FLAC, Ogg Vorbis or Opus file, into a reader's flat `meta`: title, creator, copyright
 * and description, the keys of every other audio reader. It takes the block's bytes and knows nothing of the file around them.
 *
 * Follows FFmpeg's ff_vorbis_comment (libavformat/oggparsevorbis.c) with ff_vorbiscomment_metadata_conv (vorbiscomment.c):
 *   - Names match in any case. A comment without "=", or with an empty name or value, is left out.
 *   - A name given twice joins its values with ";".
 *   - DESCRIPTION is FFmpeg's "comment", beside COMMENT itself, and the one that came first in the block answers a look-up of
 *     "comment". OBSERVED in ffprobe on both orders.
 *   - A count too large for what is left ends the read, and what came before stands.
 * Departures: METADATA_BLOCK_PICTURE is not read and FFmpeg reads it as cover art; CHAPTERnnn comments are not read as
 * chapters and FFmpeg does. Only the count this function returns can see that: FFmpeg leaves either out of its count.
 */

const LENGTH_BYTES = 4;

const EQUALS = 0x3D;

const utf8 = new TextDecoder("utf-8");

/** Where each name goes in `meta`; DESCRIPTION and COMMENT are one. */
const META_KEYS = new Map([
	["TITLE", "title"],
	["ARTIST", "creator"],
	["COPYRIGHT", "copyright"],
	["DESCRIPTION", "description"],
	["COMMENT", "description"],
]);

/** The bytes up to a NUL: FFmpeg's values are C strings. */
function beforeNul(bytes)
{
	const nul = bytes.indexOf(0);
	return nul < 0 ? bytes : bytes.subarray(0, nul);
}

/**
 * Reads the block into `meta`, filling only what it lacks. The bytes of a block that does not parse add nothing.
 *
 * @param {Uint8Array} bytes the vendor string's length onward
 * @param {object} meta
 * @returns {number} the comments the block held under any name, the four here or not
 */
export function readVorbisComment(bytes, meta)
{
	if (bytes.length < 2 * LENGTH_BYTES) {
		return 0;
	}
	const stream = new io.LittleEndianStream(bytes);
	const vendorLength = stream.readUint32();
	if (bytes.length - LENGTH_BYTES - LENGTH_BYTES < vendorLength) {
		return 0;
	}
	stream.filePos += vendorLength;
	let remaining = stream.readUint32();

	const values = new Map();		// by upper-case name, in order of first sight
	let count = 0;
	while (bytes.length - stream.filePos >= LENGTH_BYTES && remaining > 0) {
		const length = stream.readUint32();
		const start = stream.filePos;
		if (bytes.length - start < length) {
			break;
		}
		const comment = bytes.subarray(start, start + length);
		stream.filePos += length;
		--remaining;

		const equals = comment.indexOf(EQUALS);
		if (equals > 0 && equals + 1 < length) {
			const name = utf8.decode(comment.subarray(0, equals)).toUpperCase();
			if (name !== "METADATA_BLOCK_PICTURE") {
				const value = utf8.decode(beforeNul(comment.subarray(equals + 1)));
				values.set(name, values.has(name) ? `${values.get(name)};${value}` : value);
				++count;
			}
		}
	}

	for (const [name, value] of values) {
		const key = META_KEYS.get(name);
		if (key) {
			meta[key] ||= value;
		}
	}
	return count;
}
