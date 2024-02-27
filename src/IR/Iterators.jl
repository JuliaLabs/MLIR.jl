"""
    BlockIterator(region::Region)

Iterates over all blocks in the given region.
"""
struct BlockIterator
    region::Region
end

function Base.iterate(it::BlockIterator)
    reg = it.region
    raw_block = API.mlirRegionGetFirstBlock(reg)
    if mlirBlockIsNull(raw_block)
        nothing
    else
        b = Block(raw_block, false)
        (b, b)
    end
end

function Base.iterate(::BlockIterator, block)
    raw_block = API.mlirBlockGetNextInRegion(block)
    if mlirBlockIsNull(raw_block)
        nothing
    else
        b = Block(raw_block, false)
        (b, b)
    end
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

"""
    OperationIterator(block::Block)

Iterates over all operations for the given block.
"""
struct OperationIterator
    block::Block
end

function Base.iterate(it::OperationIterator)
    raw_op = API.mlirBlockGetFirstOperation(it.block)
    if mlirOperationIsNull(raw_op)
        nothing
    else
        op = Operation(raw_op, false)
        (op, op)
    end
end

function Base.iterate(::OperationIterator, op)
    raw_op = API.mlirOperationGetNextInBlock(op)
    if mlirOperationIsNull(raw_op)
        nothing
    else
        op = Operation(raw_op, false)
        (op, op)
    end
end
