#!/usr/bin/env bash
# tools/glm/build_oracle.sh - builds the GLM-5.3-Flash parity tests' oracle: ggml-cpu (shared, CPU only) from a
# llama.cpp checkout at f1cee99 (b11342, the commit whose src/models/glm5-next.cpp defines the model).
#
#   tools/glm/build_oracle.sh <llama.cpp f1cee99 checkout> <build dir>
#
# then configure Strata with -DSTRATA_GLM_ORACLE_DIR=<checkout> -DSTRATA_GLM_ORACLE_BUILD_DIR=<build dir>.
# Only the ggml, ggml-base and ggml-cpu targets are built; the checkout is not modified (out-of-tree build).
set -euo pipefail
src=${1:?llama.cpp checkout}
out=${2:?build dir}
want=f1cee9941b0e843ea260bf8dd9a090fbd9711b6a
have=$(git -C "$src" rev-parse HEAD)
if [[ "$have" != "$want" ]]; then
  echo "build_oracle: $src is at $have, the oracle must be $want" >&2
  exit 1
fi
cmake -S "$src" -B "$out" -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
  -DGGML_CUDA=OFF -DGGML_HIP=OFF -DGGML_VULKAN=OFF -DGGML_OPENMP=OFF -DGGML_NATIVE=ON -DGGML_BACKEND_DL=OFF \
  -DGGML_CCACHE=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=OFF \
  -DLLAMA_BUILD_SERVER=OFF -DLLAMA_CURL=OFF
cmake --build "$out" --target ggml ggml-base ggml-cpu -j "${JOBS:-12}"
