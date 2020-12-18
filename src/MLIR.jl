module MLIR

using Libdl
using LLVM
using LLVM: @runtime_ccall, @checked, refcheck

const libmlir = Ref{Symbol}()

# ------------ Initialization ------------ #

function __init__()
    # Find the libMLIR public API shared object file to call into.
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
end

# ------------ C API ------------ #

module API
using CEnum
using ..MLIR
using ..MLIR: libmlir
const intptr_t = Ptr{Csize_t}
libdir = joinpath(@__DIR__, "..", "lib")
include(joinpath(libdir, "ctypes.jl"))
export Ctm, Ctime_t, Cclock_t
include(joinpath(libdir, "libmlir_common.jl"))
include(joinpath(libdir, "libmlir_api.jl"))
end

# ----------- Core ------------ #

import Base: ==, insert!, append!

include("core/utils.jl")
include("core/ir.jl")
include("core/pass_manager.jl")

# ------------ Standard dialects ------------ #

include("dialects/affine.jl")

end # module
