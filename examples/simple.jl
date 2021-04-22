module Simple

using MLIR

test0 = () -> begin
    println("---- TEST 0 ----\n")

    # Constructors.
    ctx = MLIR.IR.Context()
    println(ctx)
    loc = MLIR.IR.Location(ctx)
    println(loc)
    mod = MLIR.IR.Module(loc)
    println(mod)
    op_state = MLIR.IR.OperationState("foo", loc)
    println(op_state)
    op = MLIR.IR.Operation(op_state)
    println(op)
    reg = MLIR.IR.Region()
    println(reg)
    t = MLIR.IR.Type(ctx, "index")
    println(t)
    blk = MLIR.IR.Block(t)
    arg = MLIR.IR.get_arg(blk, 0)
    println(arg)
    attr = MLIR.IR.Attribute(ctx, "\"add\"")
    println(attr)
    ident = MLIR.IR.Identifier(ctx, "type")
    println(ident)
    named_attr = MLIR.IR.NamedAttribute(ident, attr)
    println(named_attr)
end

test1 = () -> begin
    println("\n---- TEST 1 ----\n")

    # Create and destroy.
    ctx = MLIR.IR.create_context()
    MLIR.IR.num_loaded_dialects(ctx) |> y -> println("Num loaded dialects: $y")
    MLIR.IR.num_registered_dialects(ctx) |> y -> println("Num registered dialects: $y")
    MLIR.IR.destroy!(ctx)
end

test2 = () -> begin
    println("\n---- TEST 2 ----\n")

    # Create and register standard.
    ctx = MLIR.IR.create_context()
    MLIR.IR.register_standard_dialect!(ctx)
    MLIR.IR.load_standard_dialect!(ctx)
    MLIR.IR.num_loaded_dialects(ctx) |> y -> println("Num loaded dialects: $y")
    MLIR.IR.num_registered_dialects(ctx) |> y -> println("Num registered dialects: $y")
    MLIR.IR.destroy!(ctx)
end

test3 = () -> begin
    println("\n---- TEST 3 ----\n")

    # Create and dump an operation.
    ctx = MLIR.IR.create_context()
    loc = MLIR.IR.create_unknown_location(ctx)
    st = MLIR.IR.OperationState("std.add", loc)
    index_type = MLIR.IR.parse_type(ctx, "index")
    MLIR.IR.push_results!(st, index_type)
    op = MLIR.IR.Operation(st)
    MLIR.IR.dump(op)
end

test4 = () -> begin
    println("\n---- TEST 4 ----\n")

    # Create an operation and verify.
    ctx = MLIR.IR.create_context()
    loc = MLIR.IR.create_unknown_location(ctx)
    func_state = MLIR.IR.OperationState("func", loc)
    func_region = MLIR.IR.create_region()
    sym_name_ref = MLIR.IR.Identifier(ctx, "sym_name")
    func_name_attr = MLIR.IR.NamedAttribute(sym_name_ref, MLIR.IR.Attribute(ctx, "\"add\""))
    MLIR.IR.push_regions!(func_state, func_region)
    MLIR.IR.push_attributes!(func_state, func_name_attr)
    type_ref = MLIR.IR.Identifier(ctx, "type")
    func_type_attr = MLIR.IR.Attribute(ctx, "(f32, f32) -> f32")
    named_func_type_attr = MLIR.IR.NamedAttribute(type_ref, func_type_attr)
    MLIR.IR.push_attributes!(func_state, named_func_type_attr)
    func = MLIR.IR.Operation(func_state)
    MLIR.IR.verify(func)
    MLIR.IR.dump(func)
end

test5 = () -> begin
    println("\n---- TEST 5 ----\n")

    # Create a more complex operation and verify.
    ctx = MLIR.IR.create_context()
    loc = MLIR.IR.create_unknown_location(ctx)
    module_op = MLIR.IR.Module(loc)
    module_body = MLIR.IR.get_body(module_op)
    memref_type = MLIR.IR.parse_type(ctx, "memref<?xf32>")
    func_body_arg_types = [memref_type, memref_type]
    func_region = MLIR.IR.create_region()
    func_body = MLIR.IR.create_block(func_body_arg_types)
    MLIR.IR.push!(func_region, func_body)
    func_type_attr = MLIR.IR.Attribute(ctx, "(memref<?xf32>, memref<?xf32>) -> ()")
    func_name_attr = MLIR.IR.Attribute(ctx, "\"add\"")
    type_ref = MLIR.IR.Identifier(ctx, "type")
    sym_name_ref = MLIR.IR.Identifier(ctx, "sym_name")
    func_attrs = [MLIR.IR.NamedAttribute(type_ref, func_type_attr), MLIR.IR.NamedAttribute(sym_name_ref, func_name_attr)]
    func_state = MLIR.IR.OperationState("func", loc)
    MLIR.IR.push_attributes!(func_state, func_attrs)
    MLIR.IR.push_regions!(func_state, func_region)
    func = MLIR.IR.Operation(func_state)
    MLIR.IR.verify(func)
    MLIR.IR.dump(func)
end

test6 = () -> begin
    println("\n---- TEST 6 ----\n")

    # Do syntax.
    loc = MLIR.IR.Context() do ctx
        loc = MLIR.IR.Location(ctx)
        loc
    end
    println(loc)
end

test0()
test1()
test2()
test3()
test4()
test5()
test6()

end # module
