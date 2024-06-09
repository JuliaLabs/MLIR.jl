module amx

import ...IR: IR, NamedAttribute, Value, value, Location, Block, Region, Attribute, context, IndexType
import ..Dialects: namedattribute, operandsegmentsizes

"""
`tdpbf16ps`

"""
function tdpbf16ps(operand_0, operand_1, operand_2, operand_3, operand_4, operand_5; res::IR.Type, location=Location())
    results = IR.Type[res, ]
    operands = Value[value(operand_0), value(operand_1), value(operand_2), value(operand_3), value(operand_4), value(operand_5), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "amx.tdpbf16ps", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`tdpbssd`

"""
function tdpbssd(operand_0, operand_1, operand_2, operand_3, operand_4, operand_5; res::IR.Type, location=Location())
    results = IR.Type[res, ]
    operands = Value[value(operand_0), value(operand_1), value(operand_2), value(operand_3), value(operand_4), value(operand_5), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "amx.tdpbssd", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`tdpbsud`

"""
function tdpbsud(operand_0, operand_1, operand_2, operand_3, operand_4, operand_5; res::IR.Type, location=Location())
    results = IR.Type[res, ]
    operands = Value[value(operand_0), value(operand_1), value(operand_2), value(operand_3), value(operand_4), value(operand_5), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "amx.tdpbsud", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`tdpbusd`

"""
function tdpbusd(operand_0, operand_1, operand_2, operand_3, operand_4, operand_5; res::IR.Type, location=Location())
    results = IR.Type[res, ]
    operands = Value[value(operand_0), value(operand_1), value(operand_2), value(operand_3), value(operand_4), value(operand_5), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "amx.tdpbusd", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`tdpbuud`

"""
function tdpbuud(operand_0, operand_1, operand_2, operand_3, operand_4, operand_5; res::IR.Type, location=Location())
    results = IR.Type[res, ]
    operands = Value[value(operand_0), value(operand_1), value(operand_2), value(operand_3), value(operand_4), value(operand_5), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "amx.tdpbuud", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`tileloadd64`

"""
function tileloadd64(operand_0, operand_1, operand_2, operand_3; res::IR.Type, location=Location())
    results = IR.Type[res, ]
    operands = Value[value(operand_0), value(operand_1), value(operand_2), value(operand_3), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
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
    results = IR.Type[]
    operands = Value[value(operand_0), value(operand_1), value(operand_2), value(operand_3), value(operand_4), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "amx.tilestored64", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`tilezero`

"""
function tilezero(operand_0, operand_1; res::IR.Type, location=Location())
    results = IR.Type[res, ]
    operands = Value[value(operand_0), value(operand_1), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
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
function tile_load(base, indices; res::IR.Type, location=Location())
    results = IR.Type[res, ]
    operands = Value[value(base), value.(indices)..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
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
function tile_mulf(lhs, rhs, acc; res::IR.Type, location=Location())
    results = IR.Type[res, ]
    operands = Value[value(lhs), value(rhs), value(acc), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
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
function tile_muli(lhs, rhs, acc; res::IR.Type, isZextLhs=nothing, isZextRhs=nothing, location=Location())
    results = IR.Type[res, ]
    operands = Value[value(lhs), value(rhs), value(acc), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(isZextLhs) && push!(attributes, namedattribute("isZextLhs", isZextLhs))
    !isnothing(isZextRhs) && push!(attributes, namedattribute("isZextRhs", isZextRhs))
    
    IR.create_operation(
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
    results = IR.Type[]
    operands = Value[value(base), value.(indices)..., value(val), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
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
function tile_zero(; res::IR.Type, location=Location())
    _results = IR.Type[res,]
    _operands = Value[]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "amx.tile_zero",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

end # amx
