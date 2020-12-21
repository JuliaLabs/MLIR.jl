@testset "Context" begin

@testset "Create and destroy" begin
    MLIR.IR.Context() do ctx
        @test MLIR.IR.num_registered_dialects(ctx) == 0
        @test MLIR.IR.num_loaded_dialects(ctx) == 1
    end
end

@testset "Create and register the standard dialect" begin
    MLIR.IR.Context() do ctx
        @test MLIR.IR.num_registered_dialects(ctx) == 0
        MLIR.IR.register_standard_dialect!(ctx)
        @test MLIR.IR.num_registered_dialects(ctx) == 1

        prev_loaded = MLIR.IR.num_loaded_dialects(ctx)
        MLIR.IR.load_standard_dialect!(ctx)
        @test MLIR.IR.num_loaded_dialects(ctx) > prev_loaded
    end
end

end 