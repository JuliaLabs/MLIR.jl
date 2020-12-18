#!/bin/bash

DEV_DIR=mlir_dev
DEVPATH=$(pwd)/$DEV_DIR
JULIA_PATH="$DEVPATH/julia"

# Create development dir.
mkdir $DEV_DIR
cd $DEV_DIR

# Build a development version of Julia with LLVM 12 and MLIR.
git clone https://github.com/JuliaLang/julia
cd julia
git checkout $JULIA_COMMIT_HEAD
make -j `nproc` \
    USE_BINARYBUILDER_LLVM=0 \
    LLVM_VER=svn \
    LLVM_DEBUG=2 \
    USE_MLIR=1

cd ..

# Build MLIR.jl.
git clone https://github.com/vchuravy/MLIR.jl
cd MLIR.jl
$JULIA_PATH create_bindings.jl
