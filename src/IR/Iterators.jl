"""
    RegionIterator(::Operation)

Iterates over all sub-regions for the given operation.
"""
struct RegionIterator
    op::Operation
end

Base.eltype(::RegionIterator) = Region
Base.length(it::RegionIterator) = nregions(it.op)

function Base.iterate(it::RegionIterator)
    raw_region = API.mlirOperationGetFirstRegion(it.op)
    if mlirIsNull(raw_region)
        nothing
    else
        region = Region(raw_region, false)
        (region, region)
    end
end

function Base.iterate(it::RegionIterator, region)
    raw_region = API.mlirRegionGetNextInOperation(region)
    if mlirIsNull(raw_region)
        nothing
    else
        region = Region(raw_region, false)
        (region, region)
    end
end
