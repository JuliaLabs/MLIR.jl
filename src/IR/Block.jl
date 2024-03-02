mutable struct Block
    block::API.MlirBlock
    @atomic owned::Bool

    function Block(block::API.MlirBlock, owned::Bool=true)
        @assert !mlirIsNull(block) "cannot create Block with null MlirBlock"
        finalizer(new(block, owned)) do block
            if block.owned
                API.mlirBlockDestroy(block.block)
            end
        end
    end
end

Block() = Block(Type[], Location[])

"""
    Block(args, locs)

Creates a new empty block with the given argument types and transfers ownership to the caller.
"""
function Block(args::Vector{Type}, locs::Vector{Location})
    @assert length(args) == length(locs) "there should be one args for each locs (got $(length(args)) & $(length(locs)))"
    Block(API.mlirBlockCreate(length(args), args, locs))
end

"""
    ==(block, other)

Checks whether two blocks handles point to the same block. This does not perform deep comparison.
"""
Base.:(==)(a::Block, b::Block) = API.mlirBlockEqual(a, b)
Base.cconvert(::Core.Type{API.MlirBlock}, block::Block) = block
Base.unsafe_convert(::Core.Type{API.MlirBlock}, block::Block) = block.block

"""
    parent_op(block)

Returns the closest surrounding operation that contains this block.
"""
parent_op(block::Block) = Operation(API.mlirBlockGetParentOperation(block))

"""
    parent_region(block)

Returns the region that contains this block.
"""
parent_region(block::Block) = Region(API.mlirBlockGetParentRegion(block))

Base.parent(block::Block) = parent_region(block)

"""
    next(block)

Returns the block immediately following the given block in its parent region or `nothing` if last.
"""
function next(block::Block)
    block = API.mlirBlockGetNextInRegion(block)
    mlirIsNull(block) && return nothing
    Block(block)
end

"""
    nargs(block)

Returns the number of arguments of the block.
"""
nargs(block::Block) = API.mlirBlockGetNumArguments(block)

"""
    argument(block, i)

Returns `i`-th argument of the block.
"""
function argument(block::Block, i)
    i ∉ 1:nargs(block) && throw(BoundsError(block, i))
    Value(API.mlirBlockGetArgument(block, i - 1))
end

"""
    push_argument!(block, type, loc)

Appends an argument of the specified type to the block. Returns the newly added argument.
"""
push_argument!(block::Block, type, loc) = Value(API.mlirBlockAddArgument(block, type, loc))

"""
    first_op(block)

Returns the first operation in the block or `nothing` if empty.
"""
function first_op(block::Block)
    op = API.mlirBlockGetFirstOperation(block)
    mlirIsNull(op) && return nothing
    Operation(op, false)
end
Base.first(block::Block) = first_op(block)

"""
    terminator(block)

Returns the terminator operation in the block or `nothing` if no terminator.
"""
function terminator(block::Block)
    op = API.mlirBlockGetTerminator(block)
    mlirIsNull(op) && return nothing
    Operation(op, false)
end

"""
    push!(block, operation)

Takes an operation owned by the caller and appends it to the block.
"""
function Base.push!(block::Block, op::Operation)
    API.mlirBlockAppendOwnedOperation(block, lose_ownership!(op))
    op
end

"""
    insert!(block, index, operation)

Takes an operation owned by the caller and inserts it as `index` to the block.
This is an expensive operation that scans the block linearly, prefer insertBefore/After instead.
"""
function Base.insert!(block::Block, index, op::Operation)
    API.mlirBlockInsertOwnedOperation(block, index - 1, lose_ownership!(op))
    op
end

function Base.pushfirst!(block::Block, op::Operation)
    insert!(block, 1, op)
    op
end

"""
    insert_after!(block, reference, operation)

Takes an operation owned by the caller and inserts it after the (non-owned) reference operation in the given block. If the reference is null, prepends the operation. Otherwise, the reference must belong to the block.
"""
function insert_after!(block::Block, reference::Operation, op::Operation)
    API.mlirBlockInsertOwnedOperationAfter(block, reference, lose_ownership!(op))
    op
end

"""
    insert_before!(block, reference, operation)

Takes an operation owned by the caller and inserts it before the (non-owned) reference operation in the given block. If the reference is null, appends the operation. Otherwise, the reference must belong to the block.
"""
function insert_before!(block::Block, reference::Operation, op::Operation)
    API.mlirBlockInsertOwnedOperationBefore(block, reference, lose_ownership!(op))
    op
end

function lose_ownership!(block::Block)
    @assert block.owned
    # API.mlirBlockDetach(block)
    @atomic block.owned = false
    block
end

function Base.show(io::IO, block::Block)
    c_print_callback = @cfunction(print_callback, Cvoid, (API.MlirStringRef, Any))
    ref = Ref(io)
    API.mlirBlockPrint(block, c_print_callback, ref)
end
