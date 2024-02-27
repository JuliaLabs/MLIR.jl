mutable struct Region
    region::API.MlirRegion
    @atomic owned::Bool

    Region(region, owned=true) = begin
        @assert !mlirRegionIsNull(region)
        finalizer(new(region, owned)) do region
            if region.owned
                API.mlirRegionDestroy(region.region)
            end
        end
    end
end

Region() = Region(API.mlirRegionCreate())

Base.cconvert(::Core.Type{API.MlirRegion}, region::Region) = region
Base.unsafe_convert(::Core.Type{API.MlirRegion}, region::Region) = region.region
Base.:(==)(a::Region, b::Region) = API.mlirRegionEqual(a, b)

function Base.push!(region::Region, block::Block)
    API.mlirRegionAppendOwnedBlock(region, lose_ownership!(block))
    block
end

function Base.insert!(region::Region, pos, block::Block)
    API.mlirRegionInsertOwnedBlock(region, pos - 1, lose_ownership!(block))
    block
end

function Base.pushfirst!(region::Region, block)
    insert!(region, 1, block)
    block
end

insert_after!(region::Region, reference::Block, block::Block) = API.mlirRegionInsertOwnedBlockAfter(region, reference, lose_ownership!(block))
insert_before!(region::Region, reference::Block, block::Block) = API.mlirRegionInsertOwnedBlockBefore(region, reference, lose_ownership!(block))

function first_block(region::Region)
    block = mlirRegionGetFirstBlock(region)
    mlirBlockIsNull(block) && return nothing
    Block(block, false)
end
Base.first(region::Region) = first_block(region)

function lose_ownership!(region::Region)
    @assert region.owned
    @atomic region.owned = false
    region
end

"""
    RegionIterator(::Operation)

Iterates over all sub-regions for the given operation.
"""
struct RegionIterator
    op::Operation
end

function Base.iterate(it::RegionIterator)
    raw_region = API.mlirOperationGetFirstRegion(it.op)
    if mlirRegionIsNull(raw_region)
        nothing
    else
        region = Region(raw_region, false)
        (region, region)
    end
end

function Base.iterate(it::RegionIterator, region)
    raw_region = API.mlirRegionGetNextInOperation(region)
    if mlirRegionIsNull(raw_region)
        nothing
    else
        region = Region(raw_region, false)
        (region, region)
    end
end
