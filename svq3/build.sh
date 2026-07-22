#!/usr/bin/env bash
#
# Build the Sorenson Video 3 (SVQ3) WASM decoder from svq3_decode.cpp.
#
# SVQ3 is a draft-H.264 codec (QuickTime/MOV). This is a port of FFmpeg's SVQ3
# decoder (libavcodec, release/8.1, LGPL) — svq3.c plus the reachable 8-bit
# subset of the H.264 intra-prediction / hpel / third-pel / videodsp / chroma-DC
# kernels it calls. The libavcodec framework (AVFrame / AVCodecContext / the DSP
# function-pointer dispatch / threading / AVERROR) is replaced by a self-contained
# handle (SVQ3Ctx) owning a 3-slot Y/U/V frame pool, mirroring the VP6 port.
#
# Same Smart App Control / Binaryen workaround as plugins/vp6/build.sh: clang
# (allowed by SAC) optimizes at -O3/-Oz; the link runs at -O0 (no wasm-opt/metadce)
# and wasm-ld --gc-sections strips unreferenced code.
#
# -fwrapv is REQUIRED: the SVQ3 (13,13,7,17) inverse transform and the dequant
# rely on two's-complement signed-overflow wrap to be byte-identical to FFmpeg.
#
# Outputs:
#   ./svq3_decode.{mjs,wasm}                 shipped decoder (web,worker)
#   ../../tests/wasm/svq3_codec.{mjs,wasm}   node build for the vitest conformance test
#
# Usage:
#   ./build.sh            # speed build (-O3)
#   OPT=-Oz ./build.sh    # size build
#
set -euo pipefail

HERE_W="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -W)"
EMSDK_DIR="${EMSDK_DIR:-/d/dev/tools/emsdk}"
OPT="${OPT:--O3}"

EMSDK_W="$(cd "$EMSDK_DIR" && pwd -W)"
PYBIN="$(ls "$EMSDK_DIR"/python/*/python.exe | head -1)"
export EM_CONFIG="$EMSDK_W/.emscripten"
export EMSDK_PYTHON="$(cd "$(dirname "$PYBIN")" && pwd -W)/python.exe"
EMCC="$EMSDK_W/upstream/emscripten/emcc.py"
emcc() { "$PYBIN" "$EMCC" "$@"; }

SRC="$HERE_W/svq3_decode.cpp"
OBJ="$HERE_W/svq3_decode.o"

echo "compiling svq3 ($OPT)..."
emcc $OPT -msimd128 -fwrapv -ffunction-sections -fdata-sections -fno-exceptions -fno-rtti -c "$SRC" -o "$OBJ"

EXPORTS=_svq3_open,_svq3_parse_extradata,_svq3_low_delay,_svq3_close,_svq3_reset,_svq3_decode_frame,_svq3_y_ptr,_svq3_u_ptr,_svq3_v_ptr,_svq3_set_simd,_svq3_test_idct,_malloc,_free

LINK_COMMON=( -O0 -Wl,--gc-sections -sMALLOC=emmalloc -sFILESYSTEM=0
              -sALLOW_MEMORY_GROWTH=1
              -sMODULARIZE=1 -sEXPORT_ES6=1
              -sEXPORTED_RUNTIME_METHODS=HEAPU8,HEAPU32 --no-entry )

echo "=== link: shipped decoder (web,worker) ==="
emcc "${LINK_COMMON[@]}" -sENVIRONMENT=web,worker -sEXPORT_NAME=Svq3DecodeModule \
	-sEXPORTED_FUNCTIONS=$EXPORTS "$OBJ" -o "$HERE_W/svq3_decode.mjs"

TESTDIR_W="$(cd "$HERE_W/../../tests" && pwd -W)/wasm"
mkdir -p "$TESTDIR_W"
echo "=== link: node test build ==="
emcc "${LINK_COMMON[@]}" -sENVIRONMENT=web,worker,node -sEXPORT_NAME=Svq3CodecModule \
	-sEXPORTED_FUNCTIONS=$EXPORTS "$OBJ" -o "$TESTDIR_W/svq3_codec.mjs"

echo "BUILD_OK"
ls -la "$HERE_W"/svq3_decode.wasm "$TESTDIR_W"/svq3_codec.wasm
