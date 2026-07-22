#!/usr/bin/env bash
#
# Build the MPEG-4 Part 2 ASP (DivX/XviD/mp4v) + H.263 WASM decoder from
# mpeg4_decode.cpp.
#
# This is an LGPL-derived module ported from FFmpeg release/8.1 (see the header
# of mpeg4_decode.cpp) — kept in its OWN wasm binary, separate from the author's
# MPEG-1/2 + Sorenson kernels.
#
# Same Smart App Control / Binaryen workaround as plugins/vp6/build.sh and
# plugins/mpeg/build.sh: clang (allowed by SAC) optimizes at -O3/-Oz; the link
# runs at -O0 (no wasm-opt/metadce) and wasm-ld --gc-sections strips unreferenced
# code.
#
# -fwrapv is REQUIRED: the ported simple_idct8 + dequant rely on two's-complement
# signed-overflow wrap to be byte-identical to FFmpeg (which wraps the same way).
#
# Outputs:
#   ./mpeg4_decode.{mjs,wasm}                 shipped decoder (web,worker)
#   ../../tests/wasm/mpeg4_codec.{mjs,wasm}   node build for the vitest conformance test
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

SRC="$HERE_W/mpeg4_decode.cpp"
OBJ="$HERE_W/mpeg4_decode.o"

echo "compiling mpeg4 ($OPT)..."
emcc $OPT -msimd128 -fwrapv -ffunction-sections -fdata-sections -fno-exceptions -fno-rtti -c "$SRC" -o "$OBJ"

EXPORTS=_mp4v_open,_mp4v_close,_mp4v_reset,_mp4v_decode_frame,_mp4v_y_ptr,_mp4v_u_ptr,_mp4v_v_ptr,_mp4v_set_simd,_mp4v_test_idct,_mp4v_test_idct_simd,_mp4v_ref_idct,_mp4v_test_qpel,_mp4v_test_getbits,_mp4v_test_vlc,_mp4v_test_rlvlc,_malloc,_free

LINK_COMMON=( -O0 -Wl,--gc-sections -sMALLOC=emmalloc -sFILESYSTEM=0
              -sALLOW_MEMORY_GROWTH=1
              -sMODULARIZE=1 -sEXPORT_ES6=1
              -sEXPORTED_RUNTIME_METHODS=HEAPU8,HEAPU32 --no-entry )

echo "=== link: shipped decoder (web,worker) ==="
emcc "${LINK_COMMON[@]}" -sENVIRONMENT=web,worker -sEXPORT_NAME=Mpeg4DecodeModule \
	-sEXPORTED_FUNCTIONS=$EXPORTS "$OBJ" -o "$HERE_W/mpeg4_decode.mjs"

TESTDIR_W="$(cd "$HERE_W/../../tests" && pwd -W)/wasm"
mkdir -p "$TESTDIR_W"
echo "=== link: node test build ==="
emcc "${LINK_COMMON[@]}" -sENVIRONMENT=web,worker,node -sEXPORT_NAME=Mpeg4CodecModule \
	-sEXPORTED_FUNCTIONS=$EXPORTS "$OBJ" -o "$TESTDIR_W/mpeg4_codec.mjs"

echo "BUILD_OK"
ls -la "$HERE_W"/mpeg4_decode.wasm "$TESTDIR_W"/mpeg4_codec.wasm
