#!/usr/bin/env bash
#
# Build the Apple ProRes WASM decoder from prores_decode.cpp.
#
# A port of FFmpeg's ProRes decoder (libavcodec, release/8.1, LGPL) — proresdec.c,
# proresdsp.c, proresdata.c and the 10-bit extra-shift / 12-bit simple_idct_template.c
# instances. The libavcodec framework is replaced by a self-contained handle (ProresCtx)
# owning one frame's planes, as in the SVQ3 port.
#
# -fwrapv: the IDCT and the dequant rely on two's-complement wrap to be byte-identical to
# FFmpeg. -msimd128: the IDCT and the RGBA16 conversion have i32x4 / f32x4 kernels
# (prores_set_simd(0) selects the scalar ones), byte-identical to the scalar code.
#
# Outputs:
#   ./prores_decode.{mjs,wasm}                 shipped decoder (web,worker)
#   ../../tests/wasm/prores_codec.{mjs,wasm}   node build for the conformance test
#
# Usage:
#   ./build.sh            # speed build (-O3)
#   OPT=-Oz ./build.sh    # size build
#
set -euo pipefail

HERE_W="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -W)"
EMSDK_DIR="${EMSDK_DIR:-/d/dev/tools/emsdk}"
OPT="${OPT:--O3}"
LINK_OPT="${LINK_OPT:--O3}"

EMSDK_W="$(cd "$EMSDK_DIR" && pwd -W)"
PYBIN="$(ls "$EMSDK_DIR"/python/*/python.exe | head -1)"
export EM_CONFIG="$EMSDK_W/.emscripten"
export EMSDK_PYTHON="$(cd "$(dirname "$PYBIN")" && pwd -W)/python.exe"
EMCC="$EMSDK_W/upstream/emscripten/emcc.py"
emcc() { "$PYBIN" "$EMCC" "$@"; }

SRC="$HERE_W/prores_decode.cpp"
OBJ="$HERE_W/prores_decode.o"

echo "compiling prores ($OPT)..."
emcc $OPT -msimd128 -fwrapv -ffunction-sections -fdata-sections -fno-exceptions -fno-rtti -c "$SRC" -o "$OBJ"

EXPORTS=_prores_open,_prores_close,_prores_input,_prores_parse_header,_prores_decode_frame,_prores_info,_prores_plane,_prores_stride,_prores_to_rgba16,_prores_set_simd,_malloc,_free

LINK_COMMON=( $LINK_OPT -Wl,--gc-sections -sMALLOC=emmalloc -sFILESYSTEM=0
              -sALLOW_MEMORY_GROWTH=1
              -sMODULARIZE=1 -sEXPORT_ES6=1
              -sEXPORTED_RUNTIME_METHODS=HEAPU8,HEAPU16,HEAP32 --no-entry )

echo "=== link: shipped decoder (web,worker) ==="
emcc "${LINK_COMMON[@]}" -sENVIRONMENT=web,worker -sEXPORT_NAME=ProresDecodeModule \
	-sEXPORTED_FUNCTIONS=$EXPORTS "$OBJ" -o "$HERE_W/prores_decode.mjs"

TESTDIR_W="$(cd "$HERE_W/../../tests" && pwd -W)/wasm"
mkdir -p "$TESTDIR_W"
echo "=== link: node test build ==="
emcc "${LINK_COMMON[@]}" -sENVIRONMENT=web,worker,node -sEXPORT_NAME=ProresCodecModule \
	-sEXPORTED_FUNCTIONS=$EXPORTS "$OBJ" -o "$TESTDIR_W/prores_codec.mjs"

echo "BUILD_OK"
ls -la "$HERE_W"/prores_decode.wasm "$TESTDIR_W"/prores_codec.wasm
