module Blocks

using MLIR
import MLIR.IR as IR

# Minimal reproducer.

test1 = () -> begin
    ctx = IR.Context()
    IR.register_all_dialects!(ctx)
    IR.set_allow_unregistered_dialects!(ctx, true)
    @assert(IR.get_allow_unregistered_dialects(ctx))
    loc = MLIR.IR.create_unknown_location(ctx)
    st = MLIR.IR.OperationState("add", loc)
    r = IR.create_region()
    IR.push_regions!(st, r)
    b1 = IR.Block()
    IR.push!(r, b1)
    b2 = IR.Block()
    IR.push!(r, b2)

    br = MLIR.IR.OperationState("std.br", IR.Location(ctx))
    IR.push_successors!(br, b2)
    brop = IR.Operation(br)
    IR.verify(brop)
    MLIR.IR.dump(brop)
    IR.push!(b1, brop)
    println()

    index_type = MLIR.IR.parse_type(ctx, "index")
    MLIR.IR.push_results!(st, index_type)
    op = MLIR.IR.Operation(st)
    IR.verify(op)
    MLIR.IR.dump(op)
end

test1()

end # module
