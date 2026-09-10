#!/usr/bin/env bash
#
# Build the MPEG-1/2/2.5 audio (MP1/MP2/MP3) WASM decoder from mpegaudio_decode.cpp.
#
# Same Emscripten setup as the video codecs here (plugins/vp6/build.sh documents it):
# the machine's `python` is a Microsoft Store stub, so emcc is driven through the
# emsdk-bundled interpreter, and emcc gets Windows-form (D:/...) paths. Smart App
# Control is enforced but does not block Binaryen, so the link runs at -O3 and
# wasm-opt/metadce get to trim the module.
#
# -fwrapv is REQUIRED: FFmpeg computes the Layer I/II unscaling and the Layer III
# mantissa/exponent arithmetic on wrapping signed types (SUINT), and the DCT32/IMDCT
#
# Outputs:
#   ./mpegaudio_decode.{mjs,wasm}                 shipped decoder (web,worker)
#   ../../tests/wasm/mpegaudio_codec.{mjs,wasm}   node build for the vitest conformance test
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

SRC="$HERE_W/mpegaudio_decode.cpp"
OBJ="$HERE_W/mpegaudio_decode.o"

echo "compiling mpegaudio ($OPT)..."
emcc $OPT -fwrapv -ffunction-sections -fdata-sections -fno-exceptions -fno-rtti -c "$SRC" -o "$OBJ"

EXPORTS=_mpa_open,_mpa_close,_mpa_reset,_mpa_probe_frame,_mpa_decode_frame,_mpa_channel_ptr,_malloc,_free

LINK_COMMON=( $LINK_OPT -Wl,--gc-sections -sMALLOC=emmalloc -sFILESYSTEM=0
              -sALLOW_MEMORY_GROWTH=1
              -sMODULARIZE=1 -sEXPORT_ES6=1
              -sEXPORTED_RUNTIME_METHODS=HEAPU8,HEAPU32,HEAPF32 --no-entry )

echo "=== link: shipped decoder (web,worker) ==="
emcc "${LINK_COMMON[@]}" -sENVIRONMENT=web,worker -sEXPORT_NAME=MpaDecodeModule \
	-sEXPORTED_FUNCTIONS=$EXPORTS "$OBJ" -o "$HERE_W/mpegaudio_decode.mjs"

TESTDIR_W="$(cd "$HERE_W/../../tests" && pwd -W)/wasm"
mkdir -p "$TESTDIR_W"
echo "=== link: node test build ==="
emcc "${LINK_COMMON[@]}" -sENVIRONMENT=web,worker,node -sEXPORT_NAME=MpaCodecModule \
	-sEXPORTED_FUNCTIONS=$EXPORTS "$OBJ" -o "$TESTDIR_W/mpegaudio_codec.mjs"

echo "BUILD_OK"
ls -la "$HERE_W"/mpegaudio_decode.wasm "$TESTDIR_W"/mpegaudio_codec.wasm
