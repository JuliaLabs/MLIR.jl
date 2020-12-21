<p align="center">
<img height="200px" src="logo.png">
</p>
<br>

> **WARNING**: This package requires that you build Julia with LLVM 12 and MLIR enabled.

`MLIR.jl` presents high-level tools to manipulate MLIR dialects through [the MLIR C API](https://mlir.llvm.org/docs/CAPI/).

---

### Development

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

Alternatively - you can run the [create_dev.sh](https://github.com/femtomc/MLIR.jl/blob/main/create_dev.sh) script which should clone Julia, checkout the correct version and build with the correct version of LLVM.

#### Known working version

CI currently tests against Julia `a328cb65c9649d92170ec56a7c103482d8286c1e` and LLVM `01d1de81963d91773c92b29e2d08605293c59750`.
