# MLIR.jl

Note that the package requires that you build Julia with LLVM 12 and MLIR enabled.

To build Julia with LLVM 12 + MLIR, [clone Julia](https://github.com/JuliaLang/julia) and run:

```sh
cd julia
make -j `nproc` \
    USE_BINARYBUILDER_LLVM=0 \
    LLVM_VER=svn \
    LLVM_DEBUG=0 \
    USE_MLIR=1
cd ..
```

Alternatively - you can run the [create_dev.sh](https://github.com/femtomc/MLIR.jl/blob/main/create_dev.sh) script.
