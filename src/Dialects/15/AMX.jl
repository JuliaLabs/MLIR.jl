module amx

import ...IR: NamedAttribute, MLIRType, get_value, Location, Block, Region, Attribute, create_operation, context, IndexType
import ..Dialects: namedattribute, operandsegmentsizes
import ...API


"""
`tdpbf16ps`

"""
function tdpbf16ps(operand_0, operand_1, operand_2, operand_3, operand_4, operand_5; res::MLIRType, location=Location())
    results = MLIRType[res, ]
    operands = API.MlirValue[get_value(operand_0), get_value(operand_1), get_value(operand_2), get_value(operand_3), get_value(operand_4), get_value(operand_5), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "amx.tdpbf16ps", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`tdpbssd`

"""
function tdpbssd(operand_0, operand_1, operand_2, operand_3, operand_4, operand_5; res::MLIRType, location=Location())
    results = MLIRType[res, ]
    operands = API.MlirValue[get_value(operand_0), get_value(operand_1), get_value(operand_2), get_value(operand_3), get_value(operand_4), get_value(operand_5), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "amx.tdpbssd", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`tdpbsud`

"""
function tdpbsud(operand_0, operand_1, operand_2, operand_3, operand_4, operand_5; res::MLIRType, location=Location())
    results = MLIRType[res, ]
    operands = API.MlirValue[get_value(operand_0), get_value(operand_1), get_value(operand_2), get_value(operand_3), get_value(operand_4), get_value(operand_5), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "amx.tdpbsud", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`tdpbusd`

"""
function tdpbusd(operand_0, operand_1, operand_2, operand_3, operand_4, operand_5; res::MLIRType, location=Location())
    results = MLIRType[res, ]
    operands = API.MlirValue[get_value(operand_0), get_value(operand_1), get_value(operand_2), get_value(operand_3), get_value(operand_4), get_value(operand_5), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "amx.tdpbusd", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`tdpbuud`

"""
function tdpbuud(operand_0, operand_1, operand_2, operand_3, operand_4, operand_5; res::MLIRType, location=Location())
    results = MLIRType[res, ]
    operands = API.MlirValue[get_value(operand_0), get_value(operand_1), get_value(operand_2), get_value(operand_3), get_value(operand_4), get_value(operand_5), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "amx.tdpbuud", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`tileloadd64`

"""
function tileloadd64(operand_0, operand_1, operand_2, operand_3; res::MLIRType, location=Location())
    results = MLIRType[res, ]
    operands = API.MlirValue[get_value(operand_0), get_value(operand_1), get_value(operand_2), get_value(operand_3), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "amx.tileloadd64", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`tilestored64`

"""
function tilestored64(operand_0, operand_1, operand_2, operand_3, operand_4; location=Location())
    results = MLIRType[]
    operands = API.MlirValue[get_value(operand_0), get_value(operand_1), get_value(operand_2), get_value(operand_3), get_value(operand_4), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "amx.tilestored64", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`tilezero`

"""
function tilezero(operand_0, operand_1; res::MLIRType, location=Location())
    results = MLIRType[res, ]
    operands = API.MlirValue[get_value(operand_0), get_value(operand_1), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "amx.tilezero", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`tile_load`

Loads a tile from memory defined by a base and indices, with the
shape defined by the 2-dim vector type of the result. This is
eventually lowered into the \"tileloadd\" instruction with the
corresponding tile configuration.

# Example

```mlir
  %0 = amx.tile_load %arg0[%c0, %c0] : memref<?x?xi8> into vector<16x64xi8>
```
"""
function tile_load(base, indices; res::MLIRType, location=Location())
    results = MLIRType[res, ]
    operands = API.MlirValue[get_value(base), get_value.(indices)..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "amx.tile_load", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`tile_mulf`

Multiplies a \"m x k\" tile with a \"k x n\" tile and accumulates the results
into a \"m x n\" destination tile. Supports \"f32 <- bf16 x bf16\" (with
pairs of \"bf16\"). The operation is eventually lowered into the
\"tdpbf16ps\" instruction with the corresponding tile configuration.

# Example

```mlir
  %0 = amx.tile_mulf %a, %b, %c
    : vector<16x32xbf16>, vector<16x32xbf16>, vector<16x16xf32>
```
"""
function tile_mulf(lhs, rhs, acc; res::MLIRType, location=Location())
    results = MLIRType[res, ]
    operands = API.MlirValue[get_value(lhs), get_value(rhs), get_value(acc), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "amx.tile_mulf", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`tile_muli`

Multiplies a \"m x k\" tile with a \"k x n\" tile and accumulates the results
into a \"m x n\" destination tile. Supports all \"si32 <- s/ui8 x s/ui8\"
combinations (4 bytes packed into dwords in the columns of both the
source operand tiles; the zero or sign extension is specified with
the attributes and default to sign extended). The operation is eventually
lowered into one of the \"tdpbssd\", \"tdpbsud\", \"tdpbusd\", or \"tdpbuud\"
instructions with the corresponding tile configuration.

# Example

```mlir
  %0 = amx.tile_muli %a zext, %b zext, %c
    : vector<16x64xi8>, vector<16x64xi8>, vector<16x16xi32>
```
"""
function tile_muli(lhs, rhs, acc; res::MLIRType, isZextLhs=nothing, isZextRhs=nothing, location=Location())
    results = MLIRType[res, ]
    operands = API.MlirValue[get_value(lhs), get_value(rhs), get_value(acc), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (isZextLhs != nothing) && push!(attributes, namedattribute("isZextLhs", isZextLhs))
    (isZextRhs != nothing) && push!(attributes, namedattribute("isZextRhs", isZextRhs))
    
    create_operation(
        "amx.tile_muli", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`tile_store`

Stores a tile to memory defined by a base and indices, with the
shape defined by the 2-dim vector type of the value. This is
eventually lowered into the \"tilestored\" instruction with the
corresponding tile configuration.

# Example

```mlir
  amx.tile_store %arg1[%c0, %c0], %0 : memref<?x?xi8>, vector<16x64xi8>
```
"""
function tile_store(base, indices, val; location=Location())
    results = MLIRType[]
    operands = API.MlirValue[get_value(base), get_value.(indices)..., get_value(val), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "amx.tile_store", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`tile_zero`

Zeroes the destination tile, with the shape defined by the 2-dim
vector type of the result. This is eventually lowered into the
\"tilezero\" instruction with the corresponding tile configuration.

# Example

```mlir
  %0 = amx.tile_zero : vector<16x16xbf16>
```
"""
function tile_zero(; res::MLIRType, location=Location())
    results = MLIRType[res, ]
    operands = API.MlirValue[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "amx.tile_zero", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

end # amx
