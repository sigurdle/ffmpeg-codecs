#!/usr/bin/env bash
#
# Build the "retrovideo" WASM decoder pack (Cinepak / Microsoft Video-1, and
# incrementally the Indeo family) from codec_pack.cpp.
#
# NOTE (2026-08-02): the paragraph below was wrong. Smart App Control IS enforced
# here (VerifiedAndReputablePolicyState=1), but wasm-opt.exe runs fine under it —
# verified by running it. So the link is NOT pinned to -O0 any more: LINK_OPT
# (default -O3) lets Binaryen run, which is ~12% SMALLER at identical decode speed.
# Measured on JPEG: TU -O3/link -O0 = 582KB, TU -O3/link -O3 = 511KB, 0.999x speed.
# Same Smart App Control / Binaryen workaround as plugins/mpeg/build.sh and
# plugins/vp6/build.sh: clang (allowed by SAC) optimizes at -O3/-Oz; the link
# runs at -O0 (no wasm-opt/metadce) and wasm-ld --gc-sections strips unreferenced
# code. These are self-contained ports of FFmpeg's decoders (LGPL, release/8.1) —
# no third-party library to compile.
#
# -fwrapv is REQUIRED once the Indeo slant/Haar transforms (and Indeo3 SWAR) land:
# they rely on two's-complement signed-overflow wrap to be byte-identical to the
# FFmpeg reference. Cinepak/MSVideo1 don't need it, but we keep it on for the
# whole TU so the invariant holds as codecs are added.
#
# Outputs:
#   ./retrovideo_decode.{mjs,wasm}                 shipped decoder (web,worker)
#   ../../tests/wasm/retrovideo_codec.{mjs,wasm}   node build for the vitest tests
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

SRC="$HERE_W/codec_pack.cpp"
OBJ="$HERE_W/codec_pack.o"
COMMON_W="$(cd "$HERE_W/../common" && pwd -W)"

echo "compiling retrovideo ($OPT)..."
emcc $OPT -msimd128 -fwrapv -I"$COMMON_W" -ffunction-sections -fdata-sections -fno-exceptions -fno-rtti -c "$SRC" -o "$OBJ"

EXPORTS=_rv_open,_rv_close,_rv_reset,_rv_decode_frame,_rv_plane_ptr,_rv_set_simd,_rv_test_transform,_rv_bench_transform,_malloc,_free

LINK_COMMON=( $LINK_OPT -Wl,--gc-sections -sMALLOC=emmalloc -sFILESYSTEM=0
              -sALLOW_MEMORY_GROWTH=1
              -sMODULARIZE=1 -sEXPORT_ES6=1
              -sEXPORTED_RUNTIME_METHODS=HEAPU8,HEAPU32 --no-entry )

echo "=== link: shipped decoder (web,worker) ==="
emcc "${LINK_COMMON[@]}" -sENVIRONMENT=web,worker -sEXPORT_NAME=RetroVideoDecodeModule \
	-sEXPORTED_FUNCTIONS=$EXPORTS "$OBJ" -o "$HERE_W/retrovideo_decode.mjs"

TESTDIR_W="$(cd "$HERE_W/../../tests" && pwd -W)/wasm"
mkdir -p "$TESTDIR_W"
echo "=== link: node test build ==="
emcc "${LINK_COMMON[@]}" -sENVIRONMENT=web,worker,node -sEXPORT_NAME=RetroVideoCodecModule \
	-sEXPORTED_FUNCTIONS=$EXPORTS "$OBJ" -o "$TESTDIR_W/retrovideo_codec.mjs"

echo "BUILD_OK"
ls -la "$HERE_W"/retrovideo_decode.wasm "$TESTDIR_W"/retrovideo_codec.wasm
