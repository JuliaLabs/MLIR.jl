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

@testset "show" begin
    MLIR.IR.context!(MLIR.IR.Context()) do
        dialect = MLIR.IR.get_or_load_dialect!("func")
        @teest sprint(show, dialect) == "Dialect(\"func\")"
    end
end
