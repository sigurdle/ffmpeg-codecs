#!/usr/bin/env bash
#
# Build the Microsoft MPEG-4 / WMV1-2 (msmpeg4) WASM decoder from
# msmpeg4_decode.cpp. LGPL-derived (transcription of FFmpeg release/8.1's msmpeg4
# decoder — see msmpeg4_decode.h) — kept in its OWN wasm binary.
#
# Reuses the sibling mpeg4 port's kernels (get_bits / VLC / simple_idct8) via
# relative #include; no extra include paths are needed since they live one dir up.
#
# Same Smart App Control / Binaryen workaround as plugins/mpeg4/build.sh: clang
# (allowed by SAC) optimizes at -O3/-Oz; the link runs at -O0 (no wasm-opt/metadce)
# and wasm-ld --gc-sections strips unreferenced code.
#
# -fwrapv is REQUIRED (dequant / sign arithmetic byte-identical to FFmpeg).
#
# Outputs:
#   ./msmpeg4_decode.{mjs,wasm}                 shipped decoder (web,worker)
#   ../../tests/wasm/msmpeg4_codec.{mjs,wasm}   node build for the vitest conformance test
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

SRC="$HERE_W/msmpeg4_decode.cpp"
OBJ="$HERE_W/msmpeg4_decode.o"

echo "compiling msmpeg4 ($OPT)..."
emcc $OPT -msimd128 -fwrapv -ffunction-sections -fdata-sections -fno-exceptions -fno-rtti -c "$SRC" -o "$OBJ"

EXPORTS=_msm_open,_msm_close,_msm_test_header,_msm_decode,_msm_y_ptr,_msm_u_ptr,_msm_v_ptr,_malloc,_free

LINK_COMMON=( -O0 -Wl,--gc-sections -sMALLOC=emmalloc -sFILESYSTEM=0
              -sALLOW_MEMORY_GROWTH=1
              -sMODULARIZE=1 -sEXPORT_ES6=1
              -sEXPORTED_RUNTIME_METHODS=HEAPU8,HEAPU32 --no-entry )

echo "=== link: shipped decoder (web,worker) ==="
emcc "${LINK_COMMON[@]}" -sENVIRONMENT=web,worker -sEXPORT_NAME=MsMpeg4DecodeModule \
	-sEXPORTED_FUNCTIONS=$EXPORTS "$OBJ" -o "$HERE_W/msmpeg4_decode.mjs"

TESTDIR_W="$(cd "$HERE_W/../../tests" && pwd -W)/wasm"
mkdir -p "$TESTDIR_W"
echo "=== link: node test build ==="
emcc "${LINK_COMMON[@]}" -sENVIRONMENT=web,worker,node -sEXPORT_NAME=MsMpeg4CodecModule \
	-sEXPORTED_FUNCTIONS=$EXPORTS "$OBJ" -o "$TESTDIR_W/msmpeg4_codec.mjs"

echo "BUILD_OK"
ls -la "$HERE_W"/msmpeg4_decode.wasm "$TESTDIR_W"/msmpeg4_codec.wasm
