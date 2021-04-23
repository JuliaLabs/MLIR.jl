module MLIR

using Libdl
using LLVM

const libmlir = Ref{Symbol}()
const libmlir_private = Ref{Symbol}()

include("init.jl")

#####
##### C API
#####

module API
using CEnum
using ..MLIR
using ..MLIR: libmlir, libmlir_private
using LLVM: @checked, refcheck
const intptr_t = Int
libdir = joinpath(@__DIR__, "..", "lib")
include(joinpath(libdir, "ctypes.jl"))
export Ctm, Ctime_t, Cclock_t
include(joinpath(libdir, "libMLIR_common.jl"))
include(joinpath(libdir, "libMLIR_h.jl"))
end

#####
##### Core IR
#####

include("core/IR.jl")

end # module
