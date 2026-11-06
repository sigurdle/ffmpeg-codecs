#!/usr/bin/env bash
#
# Build the FLAC WASM audio decoder from flac_decode.cpp.
#
# Same Emscripten setup as the video codecs here (plugins/vp6/build.sh documents it):
# the machine's `python` is a Microsoft Store stub, so emcc is driven through the
# emsdk-bundled interpreter, and emcc gets Windows-form (D:/...) paths.
#
# -fwrapv is REQUIRED: the predictors rely on two's-complement wrap, as FFmpeg's SUINT does.
#
# Outputs:
#   ./flac_decode.{mjs,wasm}                 shipped decoder (web,worker)
#   ../../tests/wasm/flac_codec.{mjs,wasm}   node build for the vitest test
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

SRC="$HERE_W/flac_decode.cpp"
OBJ="$HERE_W/flac_decode.o"

echo "compiling flac ($OPT)..."
emcc $OPT -fwrapv -ffunction-sections -fdata-sections -fno-exceptions -fno-rtti -c "$SRC" -o "$OBJ"

EXPORTS=_flac_open,_flac_close,_flac_set_streaminfo,_flac_frame_size,_flac_packet_samples,_flac_decode_frame,_flac_channel_ptr,_malloc,_free

LINK_COMMON=( $LINK_OPT -Wl,--gc-sections -sMALLOC=emmalloc -sFILESYSTEM=0
              -sALLOW_MEMORY_GROWTH=1
              -sMODULARIZE=1 -sEXPORT_ES6=1
              -sEXPORTED_RUNTIME_METHODS=HEAPU8,HEAPF32 --no-entry )

echo "=== link: shipped decoder (web,worker) ==="
emcc "${LINK_COMMON[@]}" -sENVIRONMENT=web,worker -sEXPORT_NAME=FlacDecodeModule \
	-sEXPORTED_FUNCTIONS=$EXPORTS "$OBJ" -o "$HERE_W/flac_decode.mjs"

TESTDIR_W="$(cd "$HERE_W/../../tests" && pwd -W)/wasm"
mkdir -p "$TESTDIR_W"
echo "=== link: node test build ==="
emcc "${LINK_COMMON[@]}" -sENVIRONMENT=web,worker,node -sEXPORT_NAME=FlacCodecModule \
	-sEXPORTED_FUNCTIONS=$EXPORTS "$OBJ" -o "$TESTDIR_W/flac_codec.mjs"

rm -f "$OBJ"
echo "BUILD_OK"
ls -la "$HERE_W"/flac_decode.wasm "$TESTDIR_W"/flac_codec.wasm
