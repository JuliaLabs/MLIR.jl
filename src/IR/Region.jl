mutable struct Region
    region::API.MlirRegion
    @atomic owned::Bool

    function Region(region, owned=true)
        @assert !mlirIsNull(region)
        finalizer(new(region, owned)) do region
            if region.owned
                API.mlirRegionDestroy(region.region)
            end
        end
    end
end

"""
    Region()

Creates a new empty region and transfers ownership to the caller.
"""
Region() = Region(API.mlirRegionCreate())

Base.cconvert(::Core.Type{API.MlirRegion}, region::Region) = region
Base.unsafe_convert(::Core.Type{API.MlirRegion}, region::Region) = region.region

"""
    ==(region, other)

Checks whether two region handles point to the same region. This does not perform deep comparison.
"""
Base.:(==)(a::Region, b::Region) = API.mlirRegionEqual(a, b)

"""
    Base.first(region)

Gets the first block in the region.
"""
function Base.first(region::Region)
    block = API.mlirRegionGetFirstBlock(region)
    mlirIsNull(block) && return nothing
    return Block(block, false)
end

next(region::Region) = Region(API.mlirRegionGetNextInOperation(region), false) # TODO check `owned=false` is correct

"""
    Base.push!(region, block)

Takes a block owned by the caller and appends it to the given region.
"""
function Base.push!(region::Region, block::Block)
    API.mlirRegionAppendOwnedBlock(region, lose_ownership!(block))
    return block
end

"""
    Base.insert!(region, index, block)

Takes a block owned by the caller and inserts it at `index` to the given region. This is an expensive operation that linearly scans the region, prefer insertAfter/Before instead.
"""
function Base.insert!(region::Region, index, block::Block)
    API.mlirRegionInsertOwnedBlock(region, index - 1, lose_ownership!(block))
    return block
end

function Base.pushfirst!(region::Region, block)
    block1 = first(region)
    insert_before!(region, block1, block)
    return block
end

"""
    insert_after!(region, reference, block)

Takes a block owned by the caller and inserts it after the (non-owned) reference block in the given region. The reference block must belong to the region. If the reference block is null, prepends the block to the region.
"""
insert_after!(region::Region, reference::Block, block::Block) =
    API.mlirRegionInsertOwnedBlockAfter(region, reference, lose_ownership!(block))

"""
    insert_before!(region, reference, block)

Takes a block owned by the caller and inserts it before the (non-owned) reference block in the given region. The reference block must belong to the region. If the reference block is null, appends the block to the region.
"""
insert_before!(region::Region, reference::Block, block::Block) =
    API.mlirRegionInsertOwnedBlockBefore(region, reference, lose_ownership!(block))

function lose_ownership!(region::Region)
    @assert region.owned
    @atomic region.owned = false
    return region
end

Base.IteratorSize(::Core.Type{Region}) = Base.SizeUnknown()
Base.eltype(::Region) = Block

function Base.iterate(region::Region)
    b = first(region)
    return (b, b)
end

function Base.iterate(::Region, block)
    raw_block = API.mlirBlockGetNextInRegion(block)
    if mlirIsNull(raw_block)
        nothing
    else
        b = Block(raw_block, false)
        (b, b)
    end
end
