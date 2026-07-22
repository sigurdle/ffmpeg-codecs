# ffmpeg-codecs

WebAssembly video decoders transcribed / ported from **FFmpeg** `libavcodec`
(release/8.1). Because they are derived from FFmpeg source, they are licensed under
the **GNU LGPL-2.1-or-later** (see [`LICENSE`](./LICENSE) and [`NOTICE`](./NOTICE)).

This repository exists to make the decoder **source** available for the compiled
`.wasm` binaries shipped inside a closed-source application — i.e. it is the
"corresponding source" required by LGPL-2.1 §6. The application itself is a separate
work that merely *uses* these decoders and is not covered by this license.

## What's here

| Directory | Codec(s) |
|-----------|----------|
| `vc1/` | VC-1 / WMV3 (Simple & Main profile) |
| `msmpeg4/` | Windows Media Video 7 (WMV1) |
| `mpeg4/` | MPEG-4 Part 2 (DivX/XviD) + H.263 |
| `vp6/` | On2 VP6 (VP6, VP6F) |
| `retrovideo/` | Cinepak, Microsoft Video-1, Intel Indeo 2/3/4/5 |
| `svq3/` | Sorenson Video 3 (SVQ3) |
| `common/` | shared kernels (`wasm_simd_idct.h`) |

Only the transcribed C/C++ sources and their build scripts are included. The
application-side JavaScript glue that loads and drives the modules is *not* part of
this library and lives with the application.

## Building

Each codec directory has a `build.sh` that compiles the sources to a standalone
`.wasm` module with [Emscripten](https://emscripten.org/) (`emcc`). The scripts
assume an `emsdk` install; adjust `EMSDK_DIR` at the top. Cross-codec `#include`s use
sibling relative paths (e.g. `vc1/` includes `../mpeg4/mp4_bits.h`), so keep the
directory layout intact.

```sh
cd vc1 && ./build.sh        # -> vc1_decode.{wasm,mjs}
```

## Relinking (LGPL §6)

The decoders are distributed as **separate `.wasm` modules** loaded at runtime, which
approximates dynamic linking: a recipient may rebuild any decoder from the source
here (`build.sh`) and replace the corresponding `.wasm` in the application. The
module's exported C ABI is declared at the bottom of each `*_decode.cpp` (the
`EXPORTS=` list in `build.sh`).

## Provenance

Each source file carries a header naming the FFmpeg file(s) it was transcribed from.
FFmpeg © the FFmpeg developers, https://ffmpeg.org/. See `NOTICE`.
