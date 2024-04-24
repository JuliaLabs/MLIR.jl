module API

using ..MLIR: VersionDispatcher
using CEnum

include("Types.jl")
using .Types

# generate versioned API modules
for dir in Base.Filesystem.readdir(joinpath(@__DIR__))
    isdir(joinpath(@__DIR__, dir)) || continue
    @eval module $(Symbol(:v, dir))
    using ...MLIR: MLIR_VERSION, MLIR_C_PATH
    using ...API.Types
    include(joinpath(@__DIR__, $dir, "libMLIR_h.jl"))
    end
end

const Dispatcher = VersionDispatcher(@__MODULE__)

end
