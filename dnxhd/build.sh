#!/usr/bin/env bash
#
# Build the Avid DNxHD / DNxHR WASM encoder from dnxhd_encode.cpp.
#
# A port of FFmpeg's VC-3 encoder (libavcodec, release/8.1, LGPL; the encoder sources are
# unchanged in release/9.0) -- dnxhdenc.c, dnxhddata.c and the mpegvideo encoder pieces
# it runs (dct_quantize_c, ff_convert_matrix, the islow fdct, put_bits). The libavcodec
# framework is replaced by a self-contained handle (DnxCtx), as in the ProRes decoder port.
#
# -fwrapv: the fdct and the quantizers rely on two's-complement wrap to be byte-identical
# to FFmpeg.
#
# Outputs:
#   ./dnxhd_encode.{mjs,wasm}                 shipped encoder (web,worker)
#   ../../tests/wasm/dnxhd_codec.{mjs,wasm}   node build for the conformance test
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

SRC="$HERE_W/dnxhd_encode.cpp"
OBJ="$HERE_W/dnxhd_encode.o"

echo "compiling dnxhd ($OPT)..."
emcc $OPT -fwrapv -Wall -ffunction-sections -fdata-sections -fno-exceptions -fno-rtti -c "$SRC" -o "$OBJ"

EXPORTS=_dnxhd_open,_dnxhd_close,_dnxhd_plane,_dnxhd_stride,_dnxhd_bit_depth,_dnxhd_frame_size,_dnxhd_cid,_dnxhd_rgba16,_dnxhd_from_rgba16,_dnxhd_encode,_dnxhd_packet,_malloc,_free

LINK_COMMON=( $LINK_OPT -Wl,--gc-sections -sMALLOC=emmalloc -sFILESYSTEM=0
              -sALLOW_MEMORY_GROWTH=1
              -sMODULARIZE=1 -sEXPORT_ES6=1
              -sEXPORTED_RUNTIME_METHODS=HEAPU8,HEAPU16,HEAP32 --no-entry )

echo "=== link: shipped encoder (web,worker) ==="
emcc "${LINK_COMMON[@]}" -sENVIRONMENT=web,worker -sEXPORT_NAME=DnxhdEncodeModule \
	-sEXPORTED_FUNCTIONS=$EXPORTS "$OBJ" -o "$HERE_W/dnxhd_encode.mjs"

TESTDIR_W="$(cd "$HERE_W/../../tests" && pwd -W)/wasm"
mkdir -p "$TESTDIR_W"
echo "=== link: node test build ==="
emcc "${LINK_COMMON[@]}" -sENVIRONMENT=web,worker,node -sEXPORT_NAME=DnxhdCodecModule \
	-sEXPORTED_FUNCTIONS=$EXPORTS "$OBJ" -o "$TESTDIR_W/dnxhd_codec.mjs"

rm -f "$OBJ"
echo "BUILD_OK"
ls -la "$HERE_W"/dnxhd_encode.wasm "$TESTDIR_W"/dnxhd_codec.wasm
