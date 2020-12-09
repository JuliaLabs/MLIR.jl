module MLIR

using Libdl

const libmlir = Ref{String}()

module API
using CEnum
using ..MLIR
using ..MLIR: libmlir
const off_t = Csize_t
libdir = joinpath(@__DIR__, "..", "lib")
include(joinpath(libdir, "ctypes.jl"))
include(joinpath(libdir, "libmlir_common.jl"))
include(joinpath(libdir, "libmlir_api.jl"))
end

end # module
