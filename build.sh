#!/bin/bash
set -e

GINKGO_DIR="${GINKGO_DIR:-/workspace/ICTC/install/ginkgo}"

rm -rf build
mkdir build
cd build
cmake .. -DGinkgo_DIR="${GINKGO_DIR}/lib/cmake/Ginkgo" -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
