module Simple

using MLIR

test1 = () -> begin
    # Create and destroy.
    ctx = MLIR.IR.create_context()
    MLIR.IR.num_loaded_dialects(ctx) |> println
    MLIR.IR.num_registered_dialects(ctx) |> println
    MLIR.IR.destroy!(ctx)
end

test2 = () -> begin
    # Create and register standard.
    ctx = MLIR.IR.create_context()
    MLIR.IR.register_standard_dialect!(ctx)
    MLIR.IR.load_standard_dialect!(ctx)
    MLIR.IR.num_loaded_dialects(ctx) |> println
    MLIR.IR.num_registered_dialects(ctx) |> println
    MLIR.IR.destroy!(ctx)
end

test3 = () -> begin
    ctx = MLIR.IR.create_context()
    loc = MLIR.IR.create_unknown_location(ctx)
    st = MLIR.IR.OperationState("std.add", loc)
    index_type = MLIR.IR.parse_type(ctx, "index")
    MLIR.IR.push!(st, 1, [index_type])
    op = MLIR.IR.Operation(st)
    MLIR.IR.dump(op)
end

test1()
test2()
test3()

end # module
