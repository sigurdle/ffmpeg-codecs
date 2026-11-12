/*
 * The pages of an Ogg file, and the packets of one logical stream across them. It knows nothing of what the packets hold.
 *
 * Follows FFmpeg's libavformat/oggdec.c, ogg_read_page and ogg_packet:
 *   - A page is sought from where the last one ended, within MAX_PAGE_BYTES; past a good page where no "OggS" stands, the search
 *     goes back to 4 bytes into that page, once.
 *   - A page whose CRC or version is wrong is passed over, and the search resumes 4 bytes into it.
 *   - A page cut off by the file's end is no page.
 *   - A packet ends at the first lacing value under 255 and may cross pages. A lost page leaves what came before it joined to what
 *     comes after, as FFmpeg's buffer does; a continuation with nothing before it is dropped; a packet the file ends inside is no
 *     packet.
 * Departures: a page of another serial is passed over, so nothing replaces the stream as a chained file does in FFmpeg, and the
 * stream ends at its EOS page. An empty packet is left out: FFmpeg hands it on and a decoder takes nothing from it.
 */

const SYNC = [0x4F, 0x67, 0x67, 0x53];		// "OggS"
const SYNC_BYTES = 4;

const HEADER_BYTES = 27;
const VERSION_OFFSET = 4;
const FLAGS_OFFSET = 5;
const GRANULE_OFFSET = 6;
const SERIAL_OFFSET = 14;
const CRC_OFFSET = 22;
const SEGMENT_COUNT_OFFSET = 26;

/** As oggdec.c: the search for a page gives up after this. */
const MAX_PAGE_BYTES = 65307;

const LACING_MAX = 255;

export const OGG_FLAG_CONTINUED = 1;
export const OGG_FLAG_BOS = 2;
export const OGG_FLAG_EOS = 4;

/** A page that holds no complete packet's end carries this. */
const NO_GRANULE = -1;

/** CRC-32, polynomial 0x04C11DB7, most significant bit first. */
const CRC_TABLE = Int32Array.from({length: 256}, (_, index) => {
	let crc = index << 24;
	for (let bit = 0; bit < 8; ++bit) {
		crc = crc & 0x80000000 ? (crc << 1) ^ 0x04C11DB7 : crc << 1;
	}
	return crc;
});

/** The CRC of a page, its own CRC field read as zero. */
function pageCrc(page)
{
	let crc = 0;
	for (let i = 0; i < page.length; ++i) {
		const byte = i >= CRC_OFFSET && i < CRC_OFFSET + 4 ? 0 : page[i];
		crc = (crc << 8) ^ CRC_TABLE[((crc >>> 24) ^ byte) & 0xFF];
	}
	return crc >>> 0;
}

/**
 * Walks a file's pages, whatever their serial. A page's `bytes` stay valid until the next read of the file.
 */
export class OggPageReader
{
	/** @param {import("./ChunkedFileReader.js").ChunkedFileReader} bytes */
	constructor(bytes)
	{
		this.bytes = bytes;
		this.position = 0;			// where the next page is sought
		this.pagePosition = -1;		// the last good page, and -1 once a search has gone back to it
	}

	/** The next "OggS" from `this.position`, -1 when there is none. */
	findSync()
	{
		const at = this.position;
		if (this.isSync(at)) {
			return at;
		}
		let from = at;
		if (this.pagePosition > 0) {
			from = this.pagePosition + SYNC_BYTES;
			this.pagePosition = -1;
		}
		const length = Math.min(MAX_PAGE_BYTES + SYNC_BYTES, this.bytes.size - from);
		const window = length > 0 ? this.bytes.read(from, length) : new Uint8Array(0);
		for (let i = 0; i + SYNC_BYTES <= window.length; ++i) {
			if (window[i] === SYNC[0] && window[i + 1] === SYNC[1] && window[i + 2] === SYNC[2] && window[i + 3] === SYNC[3]) {
				return from + i;
			}
		}
		return -1;
	}

	isSync(position)
	{
		if (position + SYNC_BYTES > this.bytes.size) {
			return false;
		}
		const head = this.bytes.read(position, SYNC_BYTES);
		return head[0] === SYNC[0] && head[1] === SYNC[1] && head[2] === SYNC[2] && head[3] === SYNC[3];
	}

	/**
	 * @returns {{flags: number, granule: number, serial: number, segments: Uint8Array, body: Uint8Array, bodyPosition: number}|null}
	 *   null past the last page. `granule` is -1 for none.
	 */
	next()
	{
		for (;;) {
			const position = this.findSync();
			if (position < 0 || position + HEADER_BYTES > this.bytes.size) {
				return null;
			}
			const head = this.bytes.read(position, HEADER_BYTES);
			const segmentCount = head[SEGMENT_COUNT_OFFSET];
			if (position + HEADER_BYTES + segmentCount > this.bytes.size) {
				return null;
			}
			const segments = this.bytes.read(position + HEADER_BYTES, segmentCount);
			let bodyBytes = 0;
			for (let i = 0; i < segmentCount; ++i) {
				bodyBytes += segments[i];
			}
			const total = HEADER_BYTES + segmentCount + bodyBytes;
			if (position + total > this.bytes.size) {
				return null;
			}

			const page = this.bytes.read(position, total);
			const view = new DataView(page.buffer, page.byteOffset, total);
			if (pageCrc(page) !== view.getUint32(CRC_OFFSET, true) || page[VERSION_OFFSET] !== 0) {
				this.position = position + SYNC_BYTES;
				continue;
			}

			this.position = position + total;
			this.pagePosition = position;
			return {
				flags: page[FLAGS_OFFSET],
				granule: Number(view.getBigInt64(GRANULE_OFFSET, true)),
				serial: view.getUint32(SERIAL_OFFSET, true),
				segments: page.subarray(HEADER_BYTES, HEADER_BYTES + segmentCount),
				body: page.subarray(HEADER_BYTES + segmentCount),
				bodyPosition: position + HEADER_BYTES + segmentCount,
			};
		}
	}
}

/**
 * The packets of the logical stream `serial`, whole, by number. The walk goes only as far as a read needs, and what it found
 * stays, so a read of any packet costs one read of the file.
 */
export class OggPackets
{
	/**
	 * @param {import("./ChunkedFileReader.js").ChunkedFileReader} bytes
	 * @param {number} serial
	 */
	constructor(bytes, serial)
	{
		this.bytes = bytes;
		this.serial = serial;
		this.pages = new OggPageReader(bytes);
		this.ended = false;

		// Per packet where its first piece stands and its whole size; pieces only for one that crosses pages.
		this.positions = [];
		this.sizes = [];
		this.crossingPieces = new Map();

		this.pieces = [];			// the packet the walk is inside, [position, length]
		this.partialBytes = 0;
		this.incomplete = false;

		/** The last granule a page gave, for a stream that needs it at its end. */
		this.lastGranule = 0;
		/** Per page where a packet ended: {packetEnd, granule, ended}, packetEnd the first packet past that page, ended its EOS flag. */
		this.pageEnds = [];
		/** The EOS page: {granule, firstPacket, previousGranule}, firstPacket the first to end in it. Null without one. */
		this.endPage = null;
	}

	/** The packets found so far. */
	get count() { return this.positions.length; }

	/** Walks until packet `index` is found. False when the stream ends first. */
	walkTo(index)
	{
		while (this.positions.length <= index && !this.ended) {
			const page = this.pages.next();
			if (!page) {
				this.ended = true;
			} else if (page.serial === this.serial) {
				this.takePage(page);
			}
		}
		return index < this.positions.length;
	}

	/** Walks to the stream's end. */
	walkAll()
	{
		this.walkTo(Infinity);
	}

	/** What ogg_packet makes of a page's lacing values, and ogg_read_page of its continuation flag. */
	takePage(page)
	{
		const {segments, bodyPosition} = page;
		const firstPacket = this.count;
		let segment = 0;
		let offset = 0;

		if (page.flags & OGG_FLAG_CONTINUED || this.incomplete) {
			if (!this.partialBytes) {
				while (segment < segments.length) {
					const value = segments[segment++];
					offset += value;
					if (value < LACING_MAX) {
						break;
					}
				}
			}
		} else {
			this.pieces = [];
			this.partialBytes = 0;
		}

		let pieceStart = offset;
		while (segment < segments.length) {
			const value = segments[segment++];
			offset += value;
			if (value < LACING_MAX) {
				this.endPacket(bodyPosition + pieceStart, offset - pieceStart);
				pieceStart = offset;
			}
		}
		if (offset > pieceStart) {
			this.pieces.push([bodyPosition + pieceStart, offset - pieceStart]);
			this.partialBytes += offset - pieceStart;
		}
		this.incomplete = this.partialBytes > 0;

		if (this.count > firstPacket) {
			this.pageEnds.push({packetEnd: this.count, granule: page.granule, ended: !!(page.flags & OGG_FLAG_EOS)});
		}
		if (page.flags & OGG_FLAG_EOS) {
			this.endPage = {granule: page.granule, firstPacket, previousGranule: this.lastGranule};
			this.ended = true;
		} else if (this.count > firstPacket && page.granule !== NO_GRANULE) {
			this.lastGranule = page.granule;
		}
	}

	endPacket(position, length)
	{
		const size = this.partialBytes + length;
		if (size) {
			if (this.pieces.length) {
				this.crossingPieces.set(this.positions.length, [...this.pieces, [position, length]]);
			}
			this.positions.push(this.pieces.length ? this.pieces[0][0] : position);
			this.sizes.push(size);
		}
		this.pieces = [];
		this.partialBytes = 0;
	}

	/**
	 * The bytes of packet `index`, null past the last. A view stays valid until the next read of the file.
	 * @param {number} index
	 * @returns {Uint8Array|null}
	 */
	read(index)
	{
		if (!this.walkTo(index)) {
			return null;
		}
		const crossing = this.crossingPieces.get(index);
		if (!crossing) {
			return this.bytes.read(this.positions[index], this.sizes[index]);
		}
		const packet = new Uint8Array(this.sizes[index]);
		let at = 0;
		for (const [position, length] of crossing) {
			packet.set(this.bytes.read(position, length), at);
			at += length;
		}
		return packet;
	}
}
