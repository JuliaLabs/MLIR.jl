# ------------ Region alias and APIs ------------ #

const Region = API.MlirRegion

create_region() = API.mlirRegionCreate()
destroy!(r::Region) = API.mlirRegionDestroy(r)
is_null(r::Region) = API.mlirRegionIsNull(r)
get_first_block(r::Region) = API.mlirRegionGetFirstBlock(r)
insert!(r::Region, pos::Int, b::Block) = API.mlirRegionAppendOwnedBlock(r, pos, b)
insertafter!(r::Region, ref::Block, b::Block) = API.mlirRegionInsertOwnedBlockAfter(r, ref, b)
insertbefore!(r::Region, ref::Block, b::Block) = API.mlirRegionInsertOwnedBlockBefore(r, ref, b)

# Constructor.
Region() = create_region()

@doc(
"""
const Region = API.MlirRegion
""", Region)
