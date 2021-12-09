# ------------ Block alias and APIs ------------ #

const Block = API.MlirBlock

create_block(n::Int, ts::Vector{Type}) = API.mlirBlockCreate(n, ts)
create_block(ts::Vector{Type}) = create_block(length(ts), ts)
destroy!(b::Block) = API.mlirBlockDestroy(b)
is_null(b::Block) = API.mlirBlockIsNull(b)
Base.:(==)(b1::Block, b2::Block) = API.mlirBlockEqual(b1, b2)
get_arg(b::Block, pos::Int) = API.mlirBlockGetArgument(b, pos)
get_num_args(b::Block) = API.mlirBlockGetNumArguments(b)
get_next_in_region(b::Block) = API.mlirBlockGetNextInRegion(b)
get_first_operation(b::Block) = API.mlirBlockGetFirstOperation(b)
get_terminator(b::Block) = API.mlirBlockGetTerminator(b)
insert!(b::Block, pos::Int, op::Operation) = API.mlirBlockInsertOwnedOperation(b, pos, unwrap(op))
insertbefore!(b::Block, ref::Operation, op::Operation) = API.mlirBlockInsertOwnedOperationAfter(b, unwrap(ref), unwrap(op))
insertafter!(b::Block, ref::Operation, op::Operation) = API.mlirBlockInsertOwnedOperation(b, unwrap(ref), unwrap(op))

function Base.getindex(blk::Block, pos::Int)
    arg = get_arg(blk, pos)
    unwrap(arg) == C_NULL && error("Pointer to SSA argument at block position $pos is NULL.")
    arg
end

# Constructor.
Block(ts::Type...) = create_block(collect(ts))

@doc(
"""
const Block = API.MlirBlock
""", Block)
