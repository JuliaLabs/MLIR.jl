#####
##### MlirBlock alias and APIs
#####

const Block = MLIR.API.MlirBlock

create_block(n::Int, ts::Vector{Type}) = MLIR.API.mlirBlockCreate(n, ts)
create_block(ts::Vector{Type}) = create_block(length(ts), ts)

@doc(
"""
    create_block(n::Int, ts::Vector{Type}) = MLIR.API.mlirBlockCreate(n, ts)
    create_block(ts::Vector{Type}) = create_block(length(ts), ts)

Create a block with number of arguments `n` and argument types `ts`. Returns a MLIR.API.MlirBlock, which wraps an opaque handle to the block in memory.
""", create_block)

destroy!(b::Block) = MLIR.API.mlirBlockDestroy(b)
is_null(b::Block) = MLIR.API.mlirBlockIsNull(b)
Base.:(==)(b1::Block, b2::Block) = MLIR.API.mlirBlockEqual(b1, b2)
get_arg(b::Block, pos::Int) = MLIR.API.mlirBlockGetArgument(b, pos)
get_num_args(b::Block) = MLIR.API.mlirBlockGetNumArguments(b)
get_next_in_region(b::Block) = MLIR.API.mlirBlockGetNextInRegion(b)
get_first_operation(b::Block) = MLIR.API.mlirBlockGetFirstOperation(b)
get_terminator(b::Block) = Operation(MLIR.API.mlirBlockGetTerminator(b))
insert!(b::Block, pos::Int, op::Operation) = MLIR.API.mlirBlockInsertOwnedOperation(b, pos, unwrap(op))
insertbefore!(b::Block, ref::Operation, op::Operation) = MLIR.API.mlirBlockInsertOwnedOperationBefore(b, unwrap(ref), unwrap(op))
insertafter!(b::Block, ref::Operation, op::Operation) = MLIR.API.mlirBlockInsertOwnedOperationAfter(b, unwrap(ref), unwrap(op))

function Base.getindex(blk::Block, pos::Int)
    arg = get_arg(blk, pos)
    unwrap(arg) == C_NULL && error("Pointer to SSA argument at block position $pos is NULL.")
    arg
end

# Constructor.
Block(ts::Type...) = create_block(Type[ts...])
Block() = MLIR.API.mlirBlockCreate(0, Ref{Type}())

# Builders for blocks. Accepts Operation instances.
push!(b::Block, op::Operation) = MLIR.API.mlirBlockAppendOwnedOperation(b, unwrap(op))
push_operation!(b::Block, op::Operation) = push!(b, op)

@doc(
"""
const Block = MLIR.API.MlirBlock
""", Block)
