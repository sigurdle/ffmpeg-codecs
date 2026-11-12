import * as io from "@sigurd/core/io.js";

/*
 * ID3 tags, the text of an MP3 file, into a reader's flat `meta`: title, creator, copyright and description, the keys of
 * every other audio reader, which the one merge turns into doc.metadata. Album, track, genre and date have no key there
 * and are not read. Each read fills only what `meta` lacks, so the tag read first wins field by field.
 *
 * ID3v2 follows FFmpeg's libavformat/id3v2.c, with its answer to files whose ID3v2.4 frame sizes are not syncsafe. Three
 * departures:
 *   - A frame compressed with zlib is skipped; FFmpeg inflates it. No sample holds one.
 *   - ID3v2.3's compression and encryption flags are its own bits (0x80, 0x40), as in the spec and mutagen: FFmpeg tests
 *     ID3v2.4's bits there.
 *   - A frame ending exactly at the tag's end is a frame: FFmpeg looks for a next id in the audio, and stops.
 * A COMM frame with a description is a program's own data (iTunes' iTunNORM), not the comment: FFmpeg files it apart too.
 * A TXXX frame named "comment" is one: FFmpeg writes its own comments that way.
 */

const FLAG_UNSYNCHRONIZED = 0x80;
const FLAG_V22_COMPRESSED = 0x40;
const FLAG_EXTENDED_HEADER = 0x40;

const FRAME_V23_COMPRESSED_OR_ENCRYPTED = 0x00C0;
const FRAME_V24_COMPRESSED_OR_ENCRYPTED = 0x000C;
const FRAME_V24_UNSYNCHRONIZED = 0x0002;
const FRAME_V24_DATA_LENGTH = 0x0001;

const FRAME_HEADER_V22_BYTES = 6;
const FRAME_HEADER_BYTES = 10;
const FRAME_FLAGS_BYTES = 2;
const DATA_LENGTH_BYTES = 4;
const EXTENDED_SIZE_BYTES = 4;
// A comment's text encoding and language, before its description.
const COMMENT_DESCRIPTION_OFFSET = 1 + 3;

const ENCODING_UTF16_BOM = 1;
const ENCODING_UTF16_BE = 2;
/** The decoder of each ID3v2 text encoding, by its number. */
const ENCODING_LABELS = ["latin1", "utf-16be", "utf-16be", "utf-8"];
const ENCODING_LATIN1 = 0;
const BOM_BIG_ENDIAN = 0xFEFF;
const BOM_LITTLE_ENDIAN = 0xFFFE;

/** The text frames `meta` holds. ID3v2.2's ids have three letters; they end in a space here. */
const TEXT_FRAMES = new Map([
	[FOURCC("TIT2"), "title"],
	[FOURCC("TT2 "), "title"],
	[FOURCC("TPE1"), "creator"],
	[FOURCC("TP1 "), "creator"],
	[FOURCC("TCOP"), "copyright"],
	[FOURCC("TCR "), "copyright"],
]);
const COMMENT_FRAMES = [FOURCC("COMM"), FOURCC("COM ")];
const USER_TEXT_FRAMES = [FOURCC("TXXX"), FOURCC("TXX ")];

/** The fields of an ID3v1 tag: [meta key, first byte, past-last byte]. */
const ID3V1_FIELDS = [["title", 3, 33], ["creator", 33, 63], ["description", 97, 127]];

/**
 * The NUL-ended string at `start` of a frame's `bytes`, as FFmpeg's decode_str reads it.
 *
 * @returns {{text: string, end: number}|null} `end` is where the next string starts; null when it cannot be read
 */
function readString(bytes, start, encoding)
{
	let label = ENCODING_LABELS[encoding];
	if (!label) {
		return null;
	}
	let first = start;
	if (encoding === ENCODING_UTF16_BOM) {
		const bom = (bytes[first] << 8) | bytes[first + 1];
		if (bom === 0) {
			return {text: "", end: first + 2};
		}
		if (bom !== BOM_BIG_ENDIAN && bom !== BOM_LITTLE_ENDIAN) {
			return null;
		}
		label = bom === BOM_LITTLE_ENDIAN ? "utf-16le" : label;
		first += 2;
	}
	const wide = encoding === ENCODING_UTF16_BOM || encoding === ENCODING_UTF16_BE;
	let stop = first;
	while (stop < bytes.length && (bytes[stop] || (wide && bytes[stop + 1]))) {
		stop += wide ? 2 : 1;
	}
	return {text: new TextDecoder(label).decode(bytes.subarray(first, stop)).trim(), end: stop + (wide ? 2 : 1)};
}

/** Frame `id`'s text, into `meta`. */
function readFrame(id, body, meta)
{
	const encoding = body[0];
	const field = TEXT_FRAMES.get(id);
	if (field) {
		const value = readString(body, 1, encoding);
		if (value?.text) {
			meta[field] ??= value.text;
		}
		return;
	}
	const isComment = COMMENT_FRAMES.includes(id);
	const description = readString(body, isComment ? COMMENT_DESCRIPTION_OFFSET : 1, encoding);
	if (description && (isComment ? !description.text : description.text.toLowerCase() === "comment")) {
		const value = readString(body, description.end, encoding);
		if (value?.text) {
			meta.description ??= value.text;
		}
	}
}

/** The bytes with the 0 that follows each 0xFF gone. */
function withoutUnsynchronization(bytes)
{
	const out = new Uint8Array(bytes.length);
	let length = 0;
	for (let i = 0; i < bytes.length; ++i) {
		out[length++] = bytes[i];
		if (bytes[i] === 0xFF && bytes[i + 1] === 0) {
			++i;
		}
	}
	return out.subarray(0, length);
}

/** Whether a frame, or the padding after the last, can start at `at`: FFmpeg's check_tag. */
function isFrameStart(bytes, at)
{
	if (at === bytes.length) {
		return true;
	}
	if (at + 4 > bytes.length) {
		return false;
	}
	const id = bytes.subarray(at, at + 4);
	return id.every((byte) => byte === 0)
		|| id.every((byte) => (byte >= 0x41 && byte <= 0x5A) || (byte >= 0x30 && byte <= 0x39));
}

/** The four 7-bit groups of a syncsafe size, as a number. */
function syncsafeValue(size)
{
	return (size & 0x7F) + ((size & 0x7F00) >> 1) + ((size & 0x7F0000) >> 2) + ((size & 0x7F000000) >>> 3);
}

/**
 * An ID3v2.4 frame's size, at the stream's position. Some writers put ID3v2.3's plain size there: the size under which a
 * frame follows is the true one. Negative when neither gives one.
 */
function readFrameSizeV24(stream, bytes, bytesLeft)
{
	const size = stream.readUint32();
	if (size <= 0x7F) {
		return size;
	}
	const syncsafe = syncsafeValue(size);
	if (size >= bytesLeft) {
		return syncsafe;
	}
	const data = stream.filePos + FRAME_FLAGS_BYTES;
	if (isFrameStart(bytes, data + syncsafe)) {
		return syncsafe;
	}
	return isFrameStart(bytes, data + size) ? size : -1;
}

/**
 * The text of one ID3v2 tag, into `meta`.
 *
 * @param {Uint8Array} header the tag's 10 bytes of header
 * @param {Uint8Array} bytes what its header's size covers
 * @param {Object} meta
 */
export function readID3v2(header, bytes, meta)
{
	const version = header[3];
	const flags = header[5];
	if (version < 2 || version > 4 || (version === 2 && flags & FLAG_V22_COMPRESSED)) {
		return;
	}

	const stream = new io.BigEndianStream(bytes);
	if (version > 2 && flags & FLAG_EXTENDED_HEADER && bytes.length >= EXTENDED_SIZE_BYTES) {
		const size = syncsafeValue(stream.readUint32());
		const rest = version === 4 ? size - EXTENDED_SIZE_BYTES : size;
		if (rest < 0) {
			return;
		}
		stream.filePos += rest;
	}

	const headerBytes = version === 2 ? FRAME_HEADER_V22_BYTES : FRAME_HEADER_BYTES;
	// Padding fills what is left: a frame's id never starts with a 0.
	while (stream.filePos + headerBytes <= bytes.length && bytes[stream.filePos] !== 0) {
		const frameStart = stream.filePos;
		let id;
		let size;
		let frameFlags = 0;
		if (version === 2) {
			id = ((stream.readUint16() << 16) | (stream.readUint8() << 8) | 0x20) >>> 0;
			size = (stream.readUint16() << 8) | stream.readUint8();
		} else {
			id = stream.readFourcc();
			size = version === 3 ? stream.readUint32() : readFrameSizeV24(stream, bytes, bytes.length - frameStart);
			frameFlags = stream.readUint16();
		}
		const data = stream.filePos;
		if (size < 0 || data + size > bytes.length) {
			return;
		}
		stream.filePos = data + size;

		const hidden = version === 3 ? FRAME_V23_COMPRESSED_OR_ENCRYPTED : FRAME_V24_COMPRESSED_OR_ENCRYPTED;
		if (size && !(frameFlags & hidden) && (TEXT_FRAMES.has(id) || COMMENT_FRAMES.includes(id) || USER_TEXT_FRAMES.includes(id))) {
			const v24Flags = version === 4 ? frameFlags : 0;
			const body = bytes.subarray(data + (v24Flags & FRAME_V24_DATA_LENGTH ? DATA_LENGTH_BYTES : 0), data + size);
			const unsynchronized = flags & FLAG_UNSYNCHRONIZED || v24Flags & FRAME_V24_UNSYNCHRONIZED;
			readFrame(id, unsynchronized ? withoutUnsynchronization(body) : body, meta);
		}
	}
}

/** The text of an ID3v1 tag's 128 bytes, into `meta`. */
export function readID3v1(bytes, meta)
{
	for (const [field, start, end] of ID3V1_FIELDS) {
		const {text} = readString(bytes.subarray(start, end), 0, ENCODING_LATIN1);
		if (text) {
			meta[field] ??= text;
		}
	}
}
