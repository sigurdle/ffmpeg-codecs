#!/usr/bin/env bash
#
# Build the AC-3 (Dolby Digital / ATSC A/52) WASM audio decoder from ac3_decode.cpp.
#
# Same Emscripten setup as the video codecs here (plugins/vp6/build.sh documents it):
# the machine's `python` is a Microsoft Store stub, so emcc is driven through the
# emsdk-bundled interpreter, and emcc gets Windows-form (D:/...) paths. Smart App
# Control is enforced but does not block Binaryen, so the link runs at -O3 and
# wasm-opt/metadce get to trim the module.
#
# -fwrapv is REQUIRED: mantissa expansion (`(unsigned)get_sbits(...) << (24 - n)`)
# and the coupling MULH path rely on two's-complement wrap, as FFmpeg does.
#
# Outputs:
#   ./ac3_decode.{mjs,wasm}                 shipped decoder (web,worker)
#   ../../tests/wasm/ac3_codec.{mjs,wasm}   node build for the vitest conformance test
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

SRC="$HERE_W/ac3_decode.cpp"
OBJ="$HERE_W/ac3_decode.o"

echo "compiling ac3 ($OPT)..."
emcc $OPT -fwrapv -ffunction-sections -fdata-sections -fno-exceptions -fno-rtti -c "$SRC" -o "$OBJ"

EXPORTS=_ac3_open,_ac3_close,_ac3_reset,_ac3_set_drc_scale,_ac3_set_dither,_ac3_probe_frame,_ac3_decode_frame,_ac3_channel_ptr,_ac3_test_imdct,_malloc,_free

LINK_COMMON=( $LINK_OPT -Wl,--gc-sections -sMALLOC=emmalloc -sFILESYSTEM=0
              -sALLOW_MEMORY_GROWTH=1
              -sMODULARIZE=1 -sEXPORT_ES6=1
              -sEXPORTED_RUNTIME_METHODS=HEAPU8,HEAPU32,HEAPF32 --no-entry )

echo "=== link: shipped decoder (web,worker) ==="
emcc "${LINK_COMMON[@]}" -sENVIRONMENT=web,worker -sEXPORT_NAME=Ac3DecodeModule \
	-sEXPORTED_FUNCTIONS=$EXPORTS "$OBJ" -o "$HERE_W/ac3_decode.mjs"

TESTDIR_W="$(cd "$HERE_W/../../tests" && pwd -W)/wasm"
mkdir -p "$TESTDIR_W"
echo "=== link: node test build ==="
emcc "${LINK_COMMON[@]}" -sENVIRONMENT=web,worker,node -sEXPORT_NAME=Ac3CodecModule \
	-sEXPORTED_FUNCTIONS=$EXPORTS "$OBJ" -o "$TESTDIR_W/ac3_codec.mjs"

echo "BUILD_OK"
ls -la "$HERE_W"/ac3_decode.wasm "$TESTDIR_W"/ac3_codec.wasm
