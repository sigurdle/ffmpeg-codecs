#!/usr/bin/env bash
#
# Build the Microsoft MPEG-4 / WMV1-2 (vc1) WASM decoder from
# vc1_decode.cpp. LGPL-derived (transcription of FFmpeg release/8.1's vc1
# decoder — see vc1_decode.h) — kept in its OWN wasm binary.
#
# Reuses the sibling mpeg4 port's kernels (get_bits / VLC / simple_idct8) via
# relative #include; no extra include paths are needed since they live one dir up.
#
# NOTE (2026-08-02): the paragraph below was wrong. Smart App Control IS enforced
# here (VerifiedAndReputablePolicyState=1), but wasm-opt.exe runs fine under it —
# verified by running it. So the link is NOT pinned to -O0 any more: LINK_OPT
# (default -O3) lets Binaryen run, which is ~12% SMALLER at identical decode speed.
# Measured on JPEG: TU -O3/link -O0 = 582KB, TU -O3/link -O3 = 511KB, 0.999x speed.
# Same Smart App Control / Binaryen workaround as plugins/mpeg4/build.sh: clang
# (allowed by SAC) optimizes at -O3/-Oz; the link runs at -O0 (no wasm-opt/metadce)
# and wasm-ld --gc-sections strips unreferenced code.
#
# -fwrapv is REQUIRED (dequant / sign arithmetic byte-identical to FFmpeg).
#
# Outputs:
#   ./vc1_decode.{mjs,wasm}                 shipped decoder (web,worker)
#   ../../tests/wasm/vc1_codec.{mjs,wasm}   node build for the vitest conformance test
#
# Usage:
#   ./build.sh            # speed build (-O3)
#   OPT=-Oz ./build.sh    # size build
#
set -euo pipefail

HERE_W="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -W)"
EMSDK_DIR="${EMSDK_DIR:-/d/dev/tools/emsdk}"
OPT="${OPT:--O3}"
# Link-time optimization level — this is the step that runs Binaryen (wasm-opt).
# It used to be pinned to -O0 because Smart App Control was thought to block the
# unsigned Binaryen binaries; measured 2026-08-02, wasm-opt runs fine under SAC
# (enforced, state 1) and -O3 linking is ~12% smaller at identical decode speed.
LINK_OPT="${LINK_OPT:--O3}"

EMSDK_W="$(cd "$EMSDK_DIR" && pwd -W)"
PYBIN="$(ls "$EMSDK_DIR"/python/*/python.exe | head -1)"
export EM_CONFIG="$EMSDK_W/.emscripten"
export EMSDK_PYTHON="$(cd "$(dirname "$PYBIN")" && pwd -W)/python.exe"
EMCC="$EMSDK_W/upstream/emscripten/emcc.py"
emcc() { "$PYBIN" "$EMCC" "$@"; }

SRC="$HERE_W/vc1_decode.cpp"
OBJ="$HERE_W/vc1_decode.o"

echo "compiling vc1 ($OPT)..."
emcc $OPT -msimd128 -fwrapv -ffunction-sections -fdata-sections -fno-exceptions -fno-rtti -c "$SRC" -o "$OBJ"

EXPORTS=_vc1_open,_vc1_close,_vc1_test_seqhdr,_vc1_test_pichdr,_vc1_test_adv_headers,_vc1_decode_adv,_vc1_decode_iframe,_vc1_decode_pframe,_vc1_seqhdr,_vc1_decode_frame,_vc1_coded_h,_vc1_plane_y,_vc1_plane_u,_vc1_plane_v,_vc1_stride_y,_vc1_stride_c,_malloc,_free

LINK_COMMON=( $LINK_OPT -Wl,--gc-sections -sMALLOC=emmalloc -sFILESYSTEM=0
              -sALLOW_MEMORY_GROWTH=1
              -sMODULARIZE=1 -sEXPORT_ES6=1
              -sEXPORTED_RUNTIME_METHODS=HEAPU8,HEAPU32 --no-entry )

echo "=== link: shipped decoder (web,worker) ==="
emcc "${LINK_COMMON[@]}" -sENVIRONMENT=web,worker -sEXPORT_NAME=Vc1DecodeModule \
	-sEXPORTED_FUNCTIONS=$EXPORTS "$OBJ" -o "$HERE_W/vc1_decode.mjs"

TESTDIR_W="$(cd "$HERE_W/../../tests" && pwd -W)/wasm"
mkdir -p "$TESTDIR_W"
echo "=== link: node test build ==="
emcc "${LINK_COMMON[@]}" -sENVIRONMENT=web,worker,node -sEXPORT_NAME=Vc1CodecModule \
	-sEXPORTED_FUNCTIONS=$EXPORTS "$OBJ" -o "$TESTDIR_W/vc1_codec.mjs"

echo "BUILD_OK"
ls -la "$HERE_W"/vc1_decode.wasm "$TESTDIR_W"/vc1_codec.wasm
