#####
##### MlirRegion alias and APIs
#####

const Region = MLIR.API.MlirRegion

create_region() = MLIR.API.mlirRegionCreate()
destroy!(r::Region) = MLIR.API.mlirRegionDestroy(r)
is_null(r::Region) = MLIR.API.mlirRegionIsNull(r)
get_first_block(r::Region) = MLIR.API.mlirRegionGetFirstBlock(r)
insert!(r::Region, pos::Int, b::Block) = MLIR.API.mlirRegionAppendOwnedBlock(r, pos, b)
insertafter!(r::Region, ref::Block, b::Block) = MLIR.API.mlirRegionInsertOwnedBlockAfter(r, ref, b)
insertbefore!(r::Region, ref::Block, b::Block) = MLIR.API.mlirRegionInsertOwnedBlockBefore(r, ref, b)

# Constructor.
Region() = create_region()

# Builders for regions.
push!(r::Region, b::Block) = MLIR.API.mlirRegionAppendOwnedBlock(r, b)
push_block!(r::Region, b::Block) = push!(r, b)

@doc(
"""
const Region = MLIR.API.MlirRegion
""", Region)
