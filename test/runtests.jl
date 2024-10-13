using MLIR
using Test

@testset "$file" for file in (
    "examples.jl",
    "executionengine.jl",
    "ir.jl"
)
    @info "testing" file
    include(file)
end

@testset "MlirStringRef conversion" begin
    s = "mlir 😄 α γ 🍕"

    ms = Base.unsafe_convert(MLIR.API.MlirStringRef, s)
    reconstructed = unsafe_string(Ptr{Cchar}(ms.data), ms.length)

    @test s == reconstructed
end

@testset "show" begin
    MLIR.IR.context!(MLIR.IR.Context()) do
        dialect = IR.get_or_load_dialect!(IR.DialectHandle(:llvm))
        @test sprint(show, dialect) == "Dialect(\"llvm\")"
    end
end
