#!/usr/bin/env bash
#
# Build the WAVE audio WASM decoders (MS ADPCM, IMA ADPCM, A-law, mu-law) from waveaudio_decode.cpp.
#
# Same Emscripten setup as the video codecs here (plugins/vp6/build.sh documents it):
# the machine's `python` is a Microsoft Store stub, so emcc is driven through the
# emsdk-bundled interpreter, and emcc gets Windows-form (D:/...) paths.
#
# -fwrapv is REQUIRED: FFmpeg's MS ADPCM predictor multiplies header values on plain ints.
#
# Outputs:
#   ./waveaudio_decode.{mjs,wasm}                 shipped decoder (web,worker)
#   ../../tests/wasm/waveaudio_codec.{mjs,wasm}   node build for the vitest conformance test
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

SRC="$HERE_W/waveaudio_decode.cpp"
OBJ="$HERE_W/waveaudio_decode.o"

echo "compiling waveaudio ($OPT)..."
emcc $OPT -fwrapv -ffunction-sections -fdata-sections -fno-exceptions -fno-rtti -c "$SRC" -o "$OBJ"

EXPORTS=_wa_open,_wa_close,_wa_reset,_wa_input_ptr,_wa_decode,_wa_channel_ptr,_malloc,_free

LINK_COMMON=( $LINK_OPT -Wl,--gc-sections -sMALLOC=emmalloc -sFILESYSTEM=0
              -sALLOW_MEMORY_GROWTH=1
              -sMODULARIZE=1 -sEXPORT_ES6=1
              -sEXPORTED_RUNTIME_METHODS=HEAPU8,HEAPF32 --no-entry )

echo "=== link: shipped decoder (web,worker) ==="
emcc "${LINK_COMMON[@]}" -sENVIRONMENT=web,worker -sEXPORT_NAME=WaveAudioDecodeModule \
	-sEXPORTED_FUNCTIONS=$EXPORTS "$OBJ" -o "$HERE_W/waveaudio_decode.mjs"

TESTDIR_W="$(cd "$HERE_W/../../tests" && pwd -W)/wasm"
mkdir -p "$TESTDIR_W"
echo "=== link: node test build ==="
emcc "${LINK_COMMON[@]}" -sENVIRONMENT=web,worker,node -sEXPORT_NAME=WaveAudioCodecModule \
	-sEXPORTED_FUNCTIONS=$EXPORTS "$OBJ" -o "$TESTDIR_W/waveaudio_codec.mjs"

echo "BUILD_OK"
ls -la "$HERE_W"/waveaudio_decode.wasm "$TESTDIR_W"/waveaudio_codec.wasm
