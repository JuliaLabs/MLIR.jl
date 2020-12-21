module BuildingAddIR

# Translated from: https://github.com/tachukao/ocaml-mlir/blob/master/tests/ir.ml

using MLIR
using MLIR.IR

function populate_loop_body(ctx, loop_body, loc, func_body)
    iv = IR.get_arg(loop_body, 0)
    func_arg0 = IR.get_arg(func_body, 0)
    func_arg1 = IR.get_arg(func_body, 1)
    f32 = IR.parse_type(ctx, "f32")

    # (lhs)
    load_lhs_state = IR.create_operation_state("std.load", loc)
    load_lhs_operands = [func_arg0, iv]
    IR.push!(load_lhs_state, load_lhs_operands)
    IR.push!(load_lhs_state, f32)
    load_lhs = IR.Operation(load_lhs_state)
    IR.push!(loop_body, load_lhs)

    # (rhs)
    load_rhs_state = IR.create_operation_state("std.load", loc)
    load_rhs_operands = [func_arg1, iv]
    IR.push!(load_rhs_state, load_rhs_operands)
    IR.push!(load_rhs_state, f32)
    load_rhs = IR.Operation(load_rhs_state)
    IR.push!(loop_body, load_rhs)

    # (add)
    add_state = IR.create_operation_state("std.addf", loc)
    add_operands = [IR.get_result(load_lhs, 0), IR.get_result(load_rhs, 0)]
    IR.push!(add_state, add_operands)
    IR.push!(add_state, f32)
    add = IR.Operation(add_state)
    IR.push!(loop_body, add)

    # (store state)
    store_state = IR.create_operation_state("std.store", loc)
    store_operands = [IR.get_result(add, 0), func_arg0, iv]
    IR.push!(store_state, store_operands)
    store = IR.Operation(store_state)
    IR.push!(loop_body, store)

    # (yield state)
    yield_state = IR.create_operation_state("scf.yield", loc)
    yield = IR.Operation(yield_state)
    IR.push!(loop_body, yield)
end

function make_and_dump_add(ctx, loc)
    module_op = IR.Module(loc)
    module_body = IR.get_body(module_op)
    memref_type = IR.parse_type(ctx, "memref<?xf32>")
    func_body_arg_types = [memref_type, memref_type]
    func_region = IR.create_region()
    func_body = IR.create_block(func_body_arg_types)
    IR.push!(func_region, func_body)
    func_type_attr = IR.parse_attribute(ctx, "(memref<?xf32>, memref<?xf32>) -> ()")
    func_name_attr = IR.parse_attribute(ctx, "\"add\"")
    func_attrs = [IR.NamedAttribute(ctx, "type", func_type_attr), IR.NamedAttribute(ctx, "sym_name", func_name_attr)]
    func_state = IR.create_operation_state("func", loc)
    IR.push!(func_state, func_attrs)
    IR.push!(func_state, func_region)
    func = IR.Operation(func_state)
    IR.insert!(module_body, 0, func)
    index_type = IR.parse_type(ctx, "index")
    index_zero_literal = IR.parse_attribute(ctx, "0: index")
    index_zero_value_attr = IR.NamedAttribute(ctx, "value", index_zero_literal)
    const_zero_state = IR.OperationState("std.constant", loc)
    IR.push!(const_zero_state, index_type)
    IR.push!(const_zero_state, index_zero_value_attr)
    const_zero = IR.Operation(const_zero_state)
    IR.push!(func_body, const_zero)
    func_arg0 = IR.get_arg(func_body, 0)
    const_zero_value = IR.get_result(const_zero, 0)
    dim_operands = [func_arg0, const_zero_value]
    dim_state = IR.OperationState("std.dim", loc)
    IR.push!(dim_state, dim_operands)
    IR.push!(dim_state, index_type)
    dim = IR.Operation(dim_state)
    IR.push!(func_body, dim)
    loop_body_region = IR.create_region()
    loop_body = IR.Block(index_type)
    IR.push!(loop_body_region, loop_body)
    index_one_literal = IR.parse_attribute(ctx, "1: index")
    index_one_value_attr = IR.NamedAttribute(ctx, "value", index_one_literal)
    const_one_state = IR.OperationState("std.constant", loc)
    IR.push!(const_one_state, index_type)
    IR.push!(const_one_state, index_one_value_attr)
    const_one = IR.Operation(const_one_state)
    IR.push!(func_body, const_one)
    dim_value = IR.get_result(dim, 0)
    const_one_value = IR.get_result(const_one, 0)
    loop_operands = [const_zero_value, dim_value, const_one_value]
    loop_state = IR.OperationState("scf.for", loc)
    IR.push!(loop_state, loop_operands)
    IR.push!(loop_state, loop_body_region)
    loop = IR.Operation(loop_state)
    IR.push!(func_body, loop)
    populate_loop_body(ctx, loop_body, loc, func_body)
    ret_state = IR.OperationState("std.return", loc)
    ret = IR.Operation(ret_state)
    IR.push!(func_body, ret)
    modu = IR.get_operation(module_op)
    modu
end

construct_and_traverse_ir = () -> begin
    ctx = IR.create_context()
    loc = IR.create_unknown_location(ctx)
    MLIR.IR.register_standard_dialect!(ctx)
    modu = make_and_dump_add(ctx, loc)
    IR.verify(modu)
    IR.dump(modu)
end

modu = construct_and_traverse_ir()

end # module
