/*
	QuickTime's default color tables, for a paletted sample description that holds none:
	FFmpeg's libavformat/qtpalette.c (release/8.1, LGPL), table and rules.
*/

import {RGBA8} from "@sigurd/core/imaging/PixelFormat.js";

// R,G,B of each entry, as hex.
const DEFAULT_TABLES = {
	1: "ffffff000000",
	2: "ffffffacacac555555000000",
	4: "fffffffcf305ff6402dd0806f208844600a50000d402abea1fb714006411562c0590713ac0c0c0808080404040000000",
	8:
	"ffffffffffccffff99ffff66ffff33ffff00ffccffffccccffcc99ffcc66ffcc33ffcc00ff99ffff99ccff9999ff9966" +
	"ff9933ff9900ff66ffff66ccff6699ff6666ff6633ff6600ff33ffff33ccff3399ff3366ff3333ff3300ff00ffff00cc" +
	"ff0099ff0066ff0033ff0000ccffffccffccccff99ccff66ccff33ccff00ccccffcccccccccc99cccc66cccc33cccc00" +
	"cc99ffcc99cccc9999cc9966cc9933cc9900cc66ffcc66cccc6699cc6666cc6633cc6600cc33ffcc33cccc3399cc3366" +
	"cc3333cc3300cc00ffcc00cccc0099cc0066cc0033cc000099ffff99ffcc99ff9999ff6699ff3399ff0099ccff99cccc" +
	"99cc9999cc6699cc3399cc009999ff9999cc9999999999669999339999009966ff9966cc996699996666996633996600" +
	"9933ff9933cc9933999933669933339933009900ff9900cc99009999006699003399000066ffff66ffcc66ff9966ff66" +
	"66ff3366ff0066ccff66cccc66cc9966cc6666cc3366cc006699ff6699cc6699996699666699336699006666ff6666cc" +
	"6666996666666666336666006633ff6633cc6633996633666633336633006600ff6600cc660099660066660033660000" +
	"33ffff33ffcc33ff9933ff6633ff3333ff0033ccff33cccc33cc9933cc6633cc3333cc003399ff3399cc339999339966" +
	"3399333399003366ff3366cc3366993366663366333366003333ff3333cc3333993333663333333333003300ff3300cc" +
	"33009933006633003333000000ffff00ffcc00ff9900ff6600ff3300ff0000ccff00cccc00cc9900cc6600cc3300cc00" +
	"0099ff0099cc0099990099660099330099000066ff0066cc0066990066660066330066000033ff0033cc003399003366" +
	"0033330033000000ff0000cc000099000066000033ee0000dd0000bb0000aa0000880000770000550000440000220000" +
	"11000000ee0000dd0000bb0000aa000088000077000055000044000022000011000000ee0000dd0000bb0000aa000088" +
	"000077000055000044000022000011eeeeeeddddddbbbbbbaaaaaa888888777777555555444444222222111111000000",
};

/**
 * The color table of a sample description with a nonzero color table id.
 * @param depth 1, 2, 4 or 8, the grayscale bit taken off
 * @param isGrayScale the sample description's depth had it; ignored at depth 1
 */
export function qtDefaultColorTable(depth, isGrayScale)
{
	const numColors = 1 << depth;
	const data = new Uint8Array(numColors*4);
	const hex = DEFAULT_TABLES[depth];
	let grayLevel = 255;
	const grayStep = Math.floor(256 / (numColors - 1));
	for (let i = 0; i < numColors; ++i) {
		for (let c = 0; c < 3; ++c) {
			data[i*4 + c] = isGrayScale && depth > 1 ? grayLevel : parseInt(hex.substr((i*3 + c)*2, 2), 16);
		}
		data[i*4 + 3] = 255;
		grayLevel = Math.max(0, grayLevel - grayStep);
	}
	return {numColors, pixelFormat: RGBA8, data};
}
