using MLIR
using Test

include("examples.jl")
include("executionengine.jl")
include("ir.jl")

@testset "MlirStringRef conversion" begin
    s = "mlir 😄 α γ 🍕"

    ms = Base.unsafe_convert(MLIR.API.MlirStringRef, s)
    reconstructed = unsafe_string(Ptr{Cchar}(ms.data), ms.length)

    @test s == reconstructed
end

@testset "show" begin
    MLIR.IR.context!(MLIR.IR.Context()) do
        dialect = MLIR.IR.get_or_load_dialect!("llvm")
        @test sprint(show, dialect) == "Dialect(\"llvm\")"
    end
end
