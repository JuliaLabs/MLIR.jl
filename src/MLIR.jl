module MLIR

using Libdl
using LLVM
using LLVM: @runtime_ccall

const libmlir = Ref{String}()

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

# ------------ Initialization ------------ #

function __init__()
    # find the libLLVM and libMLIR loaded by Julia
    libmlir[] = if VERSION >= v"1.6.0-DEV.1429"
        path = joinpath(Base.libllvm_path(), "..", "libMLIR.so")
        if path === nothing
            error("""Cannot find the LLVM library loaded by Julia.
                     Please use a version of Julia that has been built with USE_LLVM_SHLIB=1 (like the official binaries).
                     If you are, please file an issue and attach the output of `Libdl.dllist()`.""")
        end
        String(path)
    else
        error("Please use a version of Julia with `versioninfo` > 1.6, built with LLVM 12 + MLIR enabled.")
    end
end

end # module
