module MLIR

using Libdl
using LLVM
using LLVM: @runtime_ccall, @checked, refcheck

const libmlir = Ref{Symbol}()
const libmlir_private = Ref{Symbol}()

# ------------ Initialization ------------ #

function __init__()

    # Find the libMLIR public *.so.
    libmlir[] = if VERSION >= v"1.6.0-DEV.1429"
        path = joinpath(String(Base.libllvm_path()), "..", "libMLIRPublicAPI.so") |> normpath
        if path === nothing
            error("""Cannot find the LLVM library loaded by Julia.
                  Please use a version of Julia that has been built with USE_LLVM_SHLIB=1 (like the official binaries).
                  If you are, please file an issue and attach the output of `Libdl.dllist()`.""")
        end
        Symbol(path)
    else
        error("Please use a version of Julia with `versioninfo` > 1.6, built with LLVM 12 + MLIR enabled.")
    end

    # Find the private *.so.
    libmlir_private[] = if VERSION >= v"1.6.0-DEV.1429"
        path = joinpath(String(Base.libllvm_path()), "..", "libMLIR.so") |> normpath
        if path === nothing
            error("""Cannot find the LLVM library loaded by Julia.
                  Please use a version of Julia that has been built with USE_LLVM_SHLIB=1 (like the official binaries).
                  If you are, please file an issue and attach the output of `Libdl.dllist()`.""")
        end
        Symbol(path)
    else
        error("Please use a version of Julia with `versioninfo` > 1.6, built with LLVM 12 + MLIR enabled.")
    end
end

# ------------ C API ------------ #

module API
using CEnum
using ..MLIR
using ..MLIR: libmlir, libmlir_private
using ..LLVM: @checked, refcheck
const intptr_t = Int
libdir = joinpath(@__DIR__, "..", "lib")
include(joinpath(libdir, "libmlir.jl"))
end

# ----------- Core ------------ #

# Builds off C API.

include("core/IR.jl")

# ------------ Standard dialects ------------ #

include("dialects/affine.jl")

end # module
