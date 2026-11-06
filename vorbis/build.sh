#!/usr/bin/env bash
#
# Build the Vorbis WASM audio decoder from vorbis_decode.cpp.
#
# Same Emscripten setup as the video codecs here (plugins/vp6/build.sh documents it):
# the machine's `python` is a Microsoft Store stub, so emcc is driven through the
# emsdk-bundled interpreter, and emcc gets Windows-form (D:/...) paths.
#
# vorbis_tables.h comes from extract_tables.mjs and is committed.
#
# -fwrapv is REQUIRED, as in FFmpeg's build.
#
# Outputs:
#   ./vorbis_decode.{mjs,wasm}                 shipped decoder (web,worker)
#   ../../tests/wasm/vorbis_codec.{mjs,wasm}   node build for the vitest test
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

SRC="$HERE_W/vorbis_decode.cpp"
OBJ="$HERE_W/vorbis_decode.o"

echo "compiling vorbis ($OPT)..."
emcc $OPT -fwrapv -ffunction-sections -fdata-sections -fno-exceptions -fno-rtti -c "$SRC" -o "$OBJ"

EXPORTS=_vorbis_open,_vorbis_close,_vorbis_set_headers,_vorbis_sample_rate,_vorbis_channels,_vorbis_blocksize,_vorbis_mode_count,_vorbis_mode_blockflag,_vorbis_reset,_vorbis_decode_packet,_vorbis_channel_ptr,_malloc,_free

LINK_COMMON=( $LINK_OPT -Wl,--gc-sections -sMALLOC=emmalloc -sFILESYSTEM=0
              -sALLOW_MEMORY_GROWTH=1
              -sMODULARIZE=1 -sEXPORT_ES6=1
              -sEXPORTED_RUNTIME_METHODS=HEAPU8,HEAPF32 --no-entry )

echo "=== link: shipped decoder (web,worker) ==="
emcc "${LINK_COMMON[@]}" -sENVIRONMENT=web,worker -sEXPORT_NAME=VorbisDecodeModule \
	-sEXPORTED_FUNCTIONS=$EXPORTS "$OBJ" -o "$HERE_W/vorbis_decode.mjs"

TESTDIR_W="$(cd "$HERE_W/../../tests" && pwd -W)/wasm"
mkdir -p "$TESTDIR_W"
echo "=== link: node test build ==="
emcc "${LINK_COMMON[@]}" -sENVIRONMENT=web,worker,node -sEXPORT_NAME=VorbisCodecModule \
	-sEXPORTED_FUNCTIONS=$EXPORTS "$OBJ" -o "$TESTDIR_W/vorbis_codec.mjs"

rm -f "$OBJ"
echo "BUILD_OK"
ls -la "$HERE_W"/vorbis_decode.wasm "$TESTDIR_W"/vorbis_codec.wasm
