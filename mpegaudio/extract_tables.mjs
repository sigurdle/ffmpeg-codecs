/*
 * Extracts the MPEG audio constant tables verbatim from FFmpeg's sources into
 * mpegaudio_tables.h.
 *
 * They are large, purely numeric, and transcribing them by hand would be both
 * tedious and a good way to introduce a silent one-digit bug, so the port generates
 * them instead. Re-run only when re-basing on a newer FFmpeg:
 *
 *   node extract_tables.mjs [path-to-ffmpeg/libavcodec]
 *
 * The generated header is committed, so a normal build does not need FFmpeg's tree.
 */
import fs from 'node:fs';
import path from 'node:path';

const SRC = process.argv[2] || 'd:/dev/monorepo/tmp/ffmpeg_src/libavcodec';
const read = f => fs.readFileSync(path.join(SRC, f), 'utf8');

// [source file, symbol, declaration to emit]
const WANT = [
  ['mpegaudiotabs.h',       'ff_mpa_freq_tab',          'static const uint16_t ff_mpa_freq_tab[3]'],
  ['mpegaudiotabs.h',       'ff_mpa_bitrate_tab',       'static const uint16_t ff_mpa_bitrate_tab[2][3][15]'],
  ['mpegaudiodata.c',       'ff_mpa_sblimit_table',     'static const int ff_mpa_sblimit_table[5]'],
  ['mpegaudiodata.c',       'ff_mpa_quant_steps',       'static const int ff_mpa_quant_steps[17]'],
  ['mpegaudiodata.c',       'ff_mpa_quant_bits',        'static const int ff_mpa_quant_bits[17]'],
  ['mpegaudiodata.c',       'alloc_table_1',            'static const unsigned char alloc_table_1[]'],
  ['mpegaudiodata.c',       'alloc_table_3',            'static const unsigned char alloc_table_3[]'],
  ['mpegaudiodata.c',       'alloc_table_4',            'static const unsigned char alloc_table_4[]'],
  // NOT ff_scale_factor_modshift: FFmpeg fills it at run time in
  // mpegaudiodec_common_init_static(), so there is no initializer to lift. The port
  // computes it the same way (see mpegaudio_init_static).
  ['mpegaudiodec_common.c', 'ff_band_size_long',        'static const uint8_t ff_band_size_long[9][22]'],
  ['mpegaudiodec_common.c', 'ff_band_size_short',       'static const uint8_t ff_band_size_short[9][13]'],
  ['mpegaudiodec_common.c', 'ff_mpa_pretab',            'static const uint8_t ff_mpa_pretab[2][22]'],
  ['mpegaudiodec_common.c', 'ff_slen_table',            'static const uint8_t ff_slen_table[2][16]'],
  ['mpegaudiodec_common.c', 'ff_lsf_nsf_table',         'static const uint8_t ff_lsf_nsf_table[6][3][4]'],
  ['mpegaudiodec_common.c', 'mpa_hufflens',             'static const uint8_t mpa_hufflens[]'],
  ['mpegaudiodec_common.c', 'mpa_huffsymbols',          'static const uint8_t mpa_huffsymbols[]'],
  ['mpegaudiodec_common.c', 'mpa_huff_sizes_minus_one', 'static const uint8_t mpa_huff_sizes_minus_one[]'],
  ['mpegaudiodec_common.c', 'ff_mpa_huff_data',         'static const uint8_t ff_mpa_huff_data[32][2]'],
  ['mpegaudiodec_common.c', 'mpa_quad_codes',           'static const uint8_t mpa_quad_codes[2][16]'],
  ['mpegaudiodec_common.c', 'mpa_quad_bits',            'static const uint8_t mpa_quad_bits[2][16]'],
  ['mpegaudiodsp_data.c',   'ff_mpa_enwindow',          'static const int32_t ff_mpa_enwindow[257]'],
];

const isWordChar = c => c !== undefined && /[A-Za-z0-9_]/.test(c);

/**
 * Return the `{ ... }` initializer body of `sym`, brace-matched.
 *
 * Plain scanning rather than a regex: the definition can be preceded by any
 * declarator and separated from `=` by array bounds and whitespace. Requiring an
 * actual `=` means a mere declaration in a header is skipped, and a symbol that has
 * moved or been renamed upstream fails loudly instead of matching the wrong thing.
 */
function initializer(text, sym) {
  let at = -1;
  for (let k = text.indexOf(sym); k >= 0; k = text.indexOf(sym, k + 1)) {
    if (isWordChar(text[k - 1]) || isWordChar(text[k + sym.length])) continue;
    let p = k + sym.length;
    const skipSpace = () => { while (p < text.length && (text[p] === ' ' || text[p] === '\t')) p++; };
    skipSpace();
    while (text[p] === '[') {
      const close = text.indexOf(']', p);
      if (close < 0) break;
      p = close + 1;
      skipSpace();
    }
    if (text[p] === '=') { at = k; break; }
  }
  if (at < 0) throw new Error(`definition not found: ${sym}`);

  const start = text.indexOf('{', at);
  if (start < 0) throw new Error(`no initializer for ${sym}`);
  let depth = 0, end = start;
  for (; end < text.length; end++) {
    if (text[end] === '{') depth++;
    else if (text[end] === '}') { depth--; if (!depth) { end++; break; } }
  }
  if (depth) throw new Error(`unbalanced initializer for ${sym}`);
  const body = text.slice(start, end).replace(/\r\n/g, '\n');

  /*
   * Reject anything that is not a pure data initializer.
   *
   * The `=` test above also matches an ASSIGNMENT to an element, e.g. FFmpeg fills
   * ff_scale_factor_modshift[i] in a loop rather than initializing it — that match
   * would happily brace-capture the surrounding function body and emit it as a
   * "table". Real initializers contain only numbers, separators and comments, so
   * anything with a statement in it is a mis-extraction, not a table.
   */
  const stripped = body.replace(/\/\*[\s\S]*?\*\//g, '').replace(/\/\/[^\n]*/g, '');
  if (/[;()]/.test(stripped) || /\b(?:for|if|int|return|uint\w*)\b/.test(stripped))
    throw new Error(`${sym}: matched a statement, not a data initializer — is it runtime-filled upstream?`);

  return body;
}

let out = `/*
	MPEG audio (MPEG-1/2/2.5 Layer I/II/III) constant tables.

	GENERATED — do not edit by hand. Extracted verbatim from FFmpeg's libavcodec
	(release/8.1, LGPL) by extract_tables.mjs, which lists the symbol and source file
	for each one. Re-run it when re-basing on a newer FFmpeg:

	    node extract_tables.mjs <path-to-ffmpeg/libavcodec>

	These are the numeric tables of ISO/IEC 11172-3 and 13818-3 themselves: the
	bitrate and sample-rate tables, the Layer II bit-allocation tables, the Layer III
	scalefactor band layouts and Huffman code books, and the 257-tap polyphase
	synthesis window.
*/

#ifndef MPEGAUDIO_TABLES_H
#define MPEGAUDIO_TABLES_H

#include <stdint.h>

`;

const cache = new Map();
for (const [file, sym, decl] of WANT) {
  if (!cache.has(file)) cache.set(file, read(file));
  out += `${decl} =\n${initializer(cache.get(file), sym)};\n\n`;
}

out += `/* Layer II allocation table by table index (ff_mpa_alloc_tables). */
static const unsigned char * const ff_mpa_alloc_tables[5] =
{ alloc_table_1, alloc_table_1, alloc_table_3, alloc_table_3, alloc_table_4, };

#endif /* MPEGAUDIO_TABLES_H */
`;

fs.writeFileSync(path.join(import.meta.dirname, 'mpegaudio_tables.h'), out);
console.log(`wrote mpegaudio_tables.h (${out.length} bytes, ${WANT.length} tables)`);

/*
 * dct32_float.h — the 32-point DCT of the polyphase synthesis filterbank.
 *
 * Taken verbatim rather than rewritten: FFmpeg's dct32 is a fully unrolled
 * butterfly network driven by macros (BF/BF0/BF1/BF2/ADD over 32 named locals),
 * roughly 200 lines in which a single transposed index would be inaudible in
 * casual listening but wrong. Its only external dependencies are the four macros
 * the float build defines, so the body compiles unchanged under the prelude below.
 */
{
  const src = read('dct32_template.c').replace(/\r\n/g, '\n');
  const from = src.indexOf('/* tab[i][j]');
  if (from < 0) throw new Error('dct32_template.c: body marker not found');

  const dct32 = `/*
	32-point DCT for the MPEG audio polyphase synthesis filterbank.

	GENERATED — do not edit by hand. The body below is FFmpeg's dct32_template.c
	(libavcodec, release/8.1, LGPL) copied verbatim by extract_tables.mjs, with the
	float-build macro prelude it expects supplied here instead of by FFmpeg's build
	system. It is an unrolled butterfly network; transcribing it by hand would risk a
	silent index error, so it is lifted whole.
*/

#ifndef DCT32_FLOAT_H
#define DCT32_FLOAT_H

// FFmpeg's DCT32_FLOAT configuration (dct32_template.c + mathops.h).
#define SUINT     unsigned
#define SUINT32   uint32_t
#define dct32     ff_dct32_float
#define FIXHR(x)       ((float)(x))
#define MULH3(x, y, s) ((s)*(y)*(x))
#define INTFLOAT  float
#define SUINTFLOAT float

${src.slice(from)}
#undef SUINT
#undef SUINT32
#undef dct32
#undef FIXHR
#undef MULH3
#undef INTFLOAT
#undef SUINTFLOAT

#endif /* DCT32_FLOAT_H */
`;
  fs.writeFileSync(path.join(import.meta.dirname, 'dct32_float.h'), dct32);
  console.log(`wrote dct32_float.h (${dct32.length} bytes)`);
}
