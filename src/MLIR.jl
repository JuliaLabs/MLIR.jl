module MLIR

using Libdl

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

end # module
