using MLIR.Dialects: arith
using MLIR.IR, LLVM

@testset "operation introspection" begin
    IR.context!(IR.Context()) do
        IR.get_or_load_dialect!("linalg")
        op = arith.constant(; value=true, result=MLIRType(Bool))

        @test IR.name(op) == "arith.constant"
        @test IR.dialect(op) === :arith
        @test IR.get_attribute_by_name(op, "value") |> IR.bool_value
    end
end
