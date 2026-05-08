#!/bin/bash
set -e

GINKGO_DIR="${GINKGO_DIR:-$HOME/ginkgo/install}"

rm -rf build
mkdir build
cd build
cmake .. -DGinkgo_DIR="${GINKGO_DIR}/lib/cmake/Ginkgo"
make -j$(nproc)
