using MLIR
using Test

include("examples.jl")
include("executionengine.jl")

@testset "MlirStringRef conversion" begin
    s = "mlir 😄 α γ 🍕"

    ms = Base.unsafe_convert(MLIR.API.MlirStringRef, s)
    reconstructed = unsafe_string(Ptr{Cchar}(ms.data), ms.length)

    @test s == reconstructed
end
