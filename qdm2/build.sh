#!/usr/bin/env bash
#
# Build the QDM2 WASM audio decoder from qdm2_decode.cpp.
#
# Same Emscripten setup as the other codecs here (plugins/vp6/build.sh documents it):
# the machine's `python` is a Microsoft Store stub, so emcc is driven through the
# emsdk-bundled interpreter, and emcc gets Windows-form (D:/...) paths.
#
# -fwrapv: the checksum and the noise generators rely on two's-complement wrap, as FFmpeg does.
#
# The synthesis filterbank is ../mpegaudio/mpegaudio_dsp.h, included from there.
#
# Outputs:
#   ./qdm2_decode.{mjs,wasm}                 shipped decoder (web,worker)
#   ../../tests/wasm/qdm2_codec.{mjs,wasm}   node build for the vitest test
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

SRC="$HERE_W/qdm2_decode.cpp"
OBJ="$HERE_W/qdm2_decode.o"

echo "compiling qdm2 ($OPT)..."
emcc $OPT -fwrapv -ffunction-sections -fdata-sections -fno-exceptions -fno-rtti -c "$SRC" -o "$OBJ"

EXPORTS=_qdm2_open,_qdm2_close,_qdm2_reset,_qdm2_configure,_qdm2_decode_packet,_qdm2_channel_ptr,_malloc,_free

LINK_COMMON=( $LINK_OPT -Wl,--gc-sections -sMALLOC=emmalloc -sFILESYSTEM=0
              -sALLOW_MEMORY_GROWTH=1
              -sMODULARIZE=1 -sEXPORT_ES6=1
              -sEXPORTED_RUNTIME_METHODS=HEAPU8,HEAPU32,HEAPF32 --no-entry )

echo "=== link: shipped decoder (web,worker) ==="
emcc "${LINK_COMMON[@]}" -sENVIRONMENT=web,worker -sEXPORT_NAME=Qdm2DecodeModule \
	-sEXPORTED_FUNCTIONS=$EXPORTS "$OBJ" -o "$HERE_W/qdm2_decode.mjs"

TESTDIR_W="$(cd "$HERE_W/../../tests" && pwd -W)/wasm"
mkdir -p "$TESTDIR_W"
echo "=== link: node test build ==="
emcc "${LINK_COMMON[@]}" -sENVIRONMENT=web,worker,node -sEXPORT_NAME=Qdm2CodecModule \
	-sEXPORTED_FUNCTIONS=$EXPORTS "$OBJ" -o "$TESTDIR_W/qdm2_codec.mjs"

echo "BUILD_OK"
ls -la "$HERE_W"/qdm2_decode.wasm "$TESTDIR_W"/qdm2_codec.wasm
