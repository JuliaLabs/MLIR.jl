module Simple

using MLIR

test0 = () -> begin
    println("---- TEST 0 ----\n")
    # Constructors.
    ctx = MLIR.IR.Context()
    println(typeof(ctx))
    loc = MLIR.IR.Location(ctx)
    println(typeof(loc))
    mod = MLIR.IR.Module(loc)
    println(typeof(mod))
    op_state = MLIR.IR.OperationState("foo", loc)
    println(typeof(op_state))
    op = MLIR.IR.Operation(op_state)
    println(typeof(op))
    reg = MLIR.IR.Region()
    println(typeof(reg))
    t = MLIR.IR.Type(ctx, "index")
    println(typeof(t))
    blk = MLIR.IR.Block(t)
    arg = MLIR.IR.get_arg(blk, 0)
    try
        blk[1]
    catch e
        println("Correctly caught:\n$e")
    end
    println(typeof(arg))
    attr = MLIR.IR.Attribute(ctx, "\"add\"")
    println(typeof(attr))
    named_attr = MLIR.IR.NamedAttribute("type", attr)
    println(typeof(named_attr))
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
    MLIR.IR.push!(st, index_type)
    op = MLIR.IR.Operation(st)
    MLIR.IR.dump(op)
end

test4 = () -> begin
    println("\n---- TEST 4 ----\n")
    ctx = MLIR.IR.create_context()
    loc = MLIR.IR.create_unknown_location(ctx)
    func_state = MLIR.IR.OperationState("func", loc)
    func = MLIR.IR.Operation(func_state)
    MLIR.IR.dump(func)
end

test5 = () -> begin
    println("\n---- TEST 5 ----\n")
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
    func_attrs = map(x -> MLIR.IR.NamedAttribute(x, func_type_attr), ["type", "sym_name"])
    func_state = MLIR.IR.OperationState("func", loc)
    MLIR.IR.push!(func_state, func_attrs)
    MLIR.IR.push!(func_state, func_region)
    display(func_state)
    func = MLIR.IR.Operation(func_state)
    MLIR.IR.dump(func)
end

test0()
test1()
test2()
test3()
test4()

end # module
