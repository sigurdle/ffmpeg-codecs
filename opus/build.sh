#!/usr/bin/env bash
#
# Build the Opus WASM audio decoder from opus_decode.cpp and its headers.
#
# Same Emscripten setup as the video codecs here (plugins/vp6/build.sh documents it):
# the machine's `python` is a Microsoft Store stub, so emcc is driven through the
# emsdk-bundled interpreter, and emcc gets Windows-form (D:/...) paths.
#
# opus_tables.h comes from extract_tables.mjs and is committed.
#
# -fwrapv is REQUIRED, as in FFmpeg's build.
#
# Outputs:
#   ./opus_decode.{mjs,wasm}                 shipped decoder (web,worker)
#   ../../tests/wasm/opus_codec.{mjs,wasm}   node build for the vitest test
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

SRC="$HERE_W/opus_decode.cpp"
OBJ="$HERE_W/opus_decode.o"

echo "compiling opus ($OPT)..."
emcc $OPT -fwrapv -ffunction-sections -fdata-sections -fno-exceptions -fno-rtti -c "$SRC" -o "$OBJ"

EXPORTS=_opus_open,_opus_map_channel,_opus_close,_opus_reset,_opus_packet_samples,_opus_decode_packet,_opus_flush,_opus_channel_ptr,_malloc,_free

LINK_COMMON=( $LINK_OPT -Wl,--gc-sections -sMALLOC=emmalloc -sFILESYSTEM=0
              -sALLOW_MEMORY_GROWTH=1
              -sMODULARIZE=1 -sEXPORT_ES6=1
              -sEXPORTED_RUNTIME_METHODS=HEAPU8,HEAPF32 --no-entry )

echo "=== link: shipped decoder (web,worker) ==="
emcc "${LINK_COMMON[@]}" -sENVIRONMENT=web,worker -sEXPORT_NAME=OpusDecodeModule \
	-sEXPORTED_FUNCTIONS=$EXPORTS "$OBJ" -o "$HERE_W/opus_decode.mjs"

TESTDIR_W="$(cd "$HERE_W/../../tests" && pwd -W)/wasm"
mkdir -p "$TESTDIR_W"
echo "=== link: node test build ==="
emcc "${LINK_COMMON[@]}" -sENVIRONMENT=web,worker,node -sEXPORT_NAME=OpusCodecModule \
	-sEXPORTED_FUNCTIONS=$EXPORTS "$OBJ" -o "$TESTDIR_W/opus_codec.mjs"

rm -f "$OBJ"
echo "BUILD_OK"
ls -la "$HERE_W"/opus_decode.wasm "$TESTDIR_W"/opus_codec.wasm
