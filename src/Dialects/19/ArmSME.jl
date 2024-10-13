module arm_sme

import ...IR:
    IR, NamedAttribute, Value, Location, Block, Region, Attribute, context, IndexType
import ..Dialects: namedattribute, operandsegmentsizes

"""
`copy_tile`

Copies an SME \"virtual tile\" value to a new SSA value. This operation is
primarily intended to be used to normalize the IR prior to tile allocation.

# Example

```mlir
%copy = arm_sme.copy_tile %tile : vector<[4]x[4]xf32>
```
"""
function copy_tile(tile::Value; result=nothing::Union{Nothing,IR.Type}, location=Location())
    _results = IR.Type[]
    _operands = Value[tile,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(result) && push!(_results, result)

    return IR.create_operation(
        "arm_sme.copy_tile",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=(length(_results) == 0 ? nothing : _results),
        result_inference=(length(_results) == 0 ? true : false),
    )
end

"""
`fmopa_2way`

This operation represents a sum of 2 widened outer products. It takes 2 1-D
scalable vectors as input and a 2-D scalable vector (ZA tile) as output.

For example (fp16 to fp32):

```mlir
%result = arm_sme.fmopa_2way %lhs, %rhs :
  vector<[8]xf16>, vector<[8]xf16> into vector<[4]x[4]xf32>
```

The `lhs` encodes a matrix of shape SVLSx2 and the `rhs` a matrix of
2xSVLS, where SVLS (spec [1], section B2.1) is the number of 32-bit
elements in a vector of SVL bits. To illustrate, below is a breakdown of
this operation for fp16 to fp32, SVL=128 (i.e., vscale=1):

```
                      LHS                          RHS
           [A0 A1 A2 A3 A4 A5 A6 A7]    [B0 B1 B2 B3 B4 B5 B6 B7]

----------------------------------------------------------------------------

                              implicit layout

                          [A0 A1]    |
                          [A2 A3]    |    [B0 B2 B4 B6]
                          [A4 A5]    |    [B1 B3 B5 B7]
                          [A6 A7]    |

----------------------------------------------------------------------------

                              2 outer products

                  Acol0 ⊗ Brow0      |           Acol1 ⊗ Brow1
                  -------------      |           -------------
                                     |
              [B0 B2 B4 B6]          |       [B1 B3 B5 B7]
                                     |
         [A0  [A0B0 A0B2 A0B4 A0B6]  |  [A1  [A1B1 A1B3 A1B5 A1B7]
          A2  [A2B0 A2B2 A2B4 A2B6]  |   A3  [A3B1 A3B3 A3B5 A3B7]
          A4  [A4B0 A4B2 A4B4 A4B6]  |   A5  [A5B1 A5B3 A5B5 A5B7]
          A6] [A6B0 A6B2 A6B4 A6B6]  |   A7] [A7B1 A7B3 A7B5 A7B7]
                                     |

----------------------------------------------------------------------------

                          sum of 2 outer products

                       Acol0 ⊗ Brow0 + Acol1 ⊗ Brow1

             [A0B0 + A1B1 A0B2 + A1B3 A0B4 + A1B5 A0B6 + A1B7]
             [A2B0 + A3B1 A2B2 + A3B3 A2B4 + A3B5 A2B6 + A3B7]
             [A4B0 + A5B1 A4B2 + A5B3 A4B4 + A5B5 A4B6 + A5B7]
             [A6B0 + A7B1 A6B2 + A7B3 A6B4 + A7B5 A6B6 + A7B7]

----------------------------------------------------------------------------
```

This operation enables the folding of 2 outer products chained via the
accumulator into a single outer product.

For example:

```mlir
%a0_ext = arith.extf %a0 : vector<[4]xf16> to vector<[4]xf32>
%b0_ext = arith.extf %b0 : vector<[4]xf16> to vector<[4]xf32>
%a1_ext = arith.extf %a1 : vector<[4]xf16> to vector<[4]xf32>
%b1_ext = arith.extf %b1 : vector<[4]xf16> to vector<[4]xf32>

%0 = arm_sme.outerproduct %a0_ext, %b0_ext : vector<[4]xf32>, vector<[4]xf32>
%1 = arm_sme.outerproduct %a1_ext, %b1_ext acc(%0) : vector<[4]xf32>, vector<[4]xf32>
```

The 2 outer products in the example above can be fused into a single outer
product as follows:

	```mlir
%a_packed = \"llvm.intr.experimental.vector.interleave2\"(%a0, %a1) : (vector<[4]xf16>, vector<[4]xf16>) -> vector<[8]xf16>
%b_packed = \"llvm.intr.experimental.vector.interleave2\"(%b0, %b1) : (vector<[4]xf16>, vector<[4]xf16>) -> vector<[8]xf16>
%0 = arm_sme.fmopa_2way %a_packed, %b_packed : vector<[8]xf16>, vector<[8]xf16> into vector<[4]x[4]xf32>
	```

This is implemented in the `-arm-sme-outer-product-fusion` pass.

# Example FP16 to FP32
```mlir
%result = arm_sme.fmopa_2way \$lhs, \$rhs : vector<[8]xf16>, vector<[8]xf16> into vector<[4]x[4]xf32>
```

# Example BF16 to FP32
```mlir
%result = arm_sme.fmopa_2way \$lhs, \$rhs : vector<[8]xbf16>, vector<[8]xbf16> into vector<[4]x[4]xf32>
```

| Spec | Features |
| ---- | -------- |
| [FMOPA (widening, 2-way, FP16 to FP32)](https://developer.arm.com/documentation/ddi0602/2023-09/SME-Instructions/FMOPA--widening--2-way--FP16-to-FP32---Half-precision-floating-point-sum-of-outer-products-and-accumulate-) | +sme |
| [BFMOPA (widening, 2-way, BF16 to FP32)](https://developer.arm.com/documentation/ddi0602/2023-09/SME-Instructions/BFMOPA--widening---BFloat16-sum-of-outer-products-and-accumulate-) | +sme |

[1] https://developer.arm.com/documentation/ddi0616
"""
function fmopa_2way(
    lhs::Value,
    rhs::Value,
    lhsMask=nothing::Union{Nothing,Value};
    rhsMask=nothing::Union{Nothing,Value},
    acc=nothing::Union{Nothing,Value},
    result::IR.Type,
    location=Location(),
)
    _results = IR.Type[result,]
    _operands = Value[lhs, rhs]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(lhsMask) && push!(_operands, lhsMask)
    !isnothing(rhsMask) && push!(_operands, rhsMask)
    !isnothing(acc) && push!(_operands, acc)
    push!(
        _attributes,
        operandsegmentsizes([
            1,
            1,
            isnothing(lhsMask) ? 0 : 1,
            isnothing(rhsMask) ? 0 : 1,
            isnothing(acc) ? 0 : 1,
        ]),
    )

    return IR.create_operation(
        "arm_sme.fmopa_2way",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`fmops_2way`

Equivalent to `fmopa_2way` but outer products are subtracted from
destination `result`.

# Example FP16 to FP32
```mlir
%result = arm_sme.fmops_2way \$lhs, \$rhs : vector<[8]xf16>, vector<[8]xf16> into vector<[4]x[4]xf32>
```

# Example BF16 to FP32
```mlir
%result = arm_sme.fmops_2way \$lhs, \$rhs : vector<[8]xbf16>, vector<[8]xbf16> into vector<[4]x[4]xf32>
```

Refer to
[fmopa_2way](#arm_smefmopa_2way-arm_smefmopa2wayop) for a detailed
description of 2-way outer products.

| Spec | Features |
| ---- | -------- |
| [FMOPS (widening, 2-way, FP16 to FP32)](https://developer.arm.com/documentation/ddi0602/2023-09/SME-Instructions/FMOPS--widening---Half-precision-floating-point-sum-of-outer-products-and-subtract-) | +sme |
| [BFMOPS (widening, 2-way, BF16 to FP32)](https://developer.arm.com/documentation/ddi0602/2023-09/SME-Instructions/BMOPS--Bitwise-exclusive-NOR-population-count-outer-product-and-subtract-) | +sme |
"""
function fmops_2way(
    lhs::Value,
    rhs::Value,
    lhsMask=nothing::Union{Nothing,Value};
    rhsMask=nothing::Union{Nothing,Value},
    acc=nothing::Union{Nothing,Value},
    result::IR.Type,
    location=Location(),
)
    _results = IR.Type[result,]
    _operands = Value[lhs, rhs]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(lhsMask) && push!(_operands, lhsMask)
    !isnothing(rhsMask) && push!(_operands, rhsMask)
    !isnothing(acc) && push!(_operands, acc)
    push!(
        _attributes,
        operandsegmentsizes([
            1,
            1,
            isnothing(lhsMask) ? 0 : 1,
            isnothing(rhsMask) ? 0 : 1,
            isnothing(acc) ? 0 : 1,
        ]),
    )

    return IR.create_operation(
        "arm_sme.fmops_2way",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`get_tile`

Creates a new SME \"virtual tile\" value within a function. The contents of
the tile returned from this operation are undefined.

Example 1:

```mlir
// Create an 8-bit element \"virtual tile\" value:
%za0_b = arm_sme.get_tile: vector<[16]x[16]xi8>
```

Example 2:

```mlir
// Create two 16-bit element \"virtual tiles\" values:
%za0_h = arm_sme.get_tile : vector<[8]x[8]xi16>
%za1_h = arm_sme.get_tile : vector<[8]x[8]xi16>
```

Example 3:
```mlir
// Create an 128-bit element \"virtual tile\" value:
%za0_q = arm_sme.get_tile : vector<[1]x[1]xi128>
```
"""
function get_tile(; tile::IR.Type, location=Location())
    _results = IR.Type[tile,]
    _operands = Value[]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "arm_sme.get_tile",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`load_tile_slice`

Loads a 1D tile slice from memory into a 2D SME \"virtual tile\". The tile
slice is defined by the dimension of the 2D scalable vector type pointed by
the index. A tile slice index describes where in the input tile the tile
slice is loaded to. An optional tile slice layout attribute specifies
whether the tile slice being loaded at the given index is horizontal
(default) or vertical. The updated tile is returned as the result.

The slice of memory read is defined by a base and indices and must be
contiguous. The memref must be either rank 1 or rank 2, have dynamic
dimensions since the operation is scalable, and the element type must be a
scalar that matches the element type of the result.

The provided `mask` is used to specify which elements of the tile slice
will be loaded.

Example 1: Load a vector<[16]xi8> tile slice from memory into tile horizontally (default) at given index.
```mlir
%tile_update = arm_sme.load_tile_slice %base[%c0], %mask, %tile, %tile_slice_index : memref<?x?xi8>, vector<[16]xi1>, vector<[16]x[16]xi8>
```

Example 2: Load a vector<[4]xf32> tile slice from memory into tile vertically at given index.
```mlir
%tile_update = arm_sme.load_tile_slice %base[%c0], %mask, %tile, %tile_slice_index layout<vertical> : memref<?x?xf32>, vector<[4]xi1>, vector<[4]x[4]xf32>
```

Example 3: Load a vector<[1]xi128> tile slice from memory into tile vertically at given index.
```mlir
%tile_update = arm_sme.load_tile_slice %base[%c0], %mask, %tile, %tile_slice_index layout<vertical> : memref<?x?xi128>, vector<[1]xi1>, vector<[1]x[1]xi128>
```
"""
function load_tile_slice(
    base::Value,
    mask::Value,
    tile::Value,
    indices::Vector{Value},
    tile_slice_index::Value;
    result=nothing::Union{Nothing,IR.Type},
    layout=nothing,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[base, mask, tile, indices..., tile_slice_index]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(result) && push!(_results, result)
    !isnothing(layout) && push!(_attributes, namedattribute("layout", layout))

    return IR.create_operation(
        "arm_sme.load_tile_slice",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=(length(_results) == 0 ? nothing : _results),
        result_inference=(length(_results) == 0 ? true : false),
    )
end

"""
`move_tile_slice_to_vector`

The tile slice to vector operation extracts a 1-D scalable slice from a 2-D
scalable tile at the given index. A tile slice is a 1-D vector of
horizontally or vertically contiguous elements within a ZA tile.

An optional tile slice layout attribute specifies whether the tile slice is
horizontal (default) or vertical.

Example 1: Extract `vector<[16]xi8>` from tile horizontally at the given index.
```mlir
%slice = arm_sme.move_tile_slice_to_vector %tile[%tile_slice_index] : vector<[16]xi8> from vector<[16]x[16]xi8>
```

Example 2: Extract `vector<[2]xf64>` from tile vertically at the given index.
```mlir
%slice = arm_sme.move_tile_slice_to_vector %tile[%tile_slice_index] layout<vertical> : vector<[2]xf64> from vector<[2]x[2]xf64>
```
"""
function move_tile_slice_to_vector(
    tile::Value,
    tile_slice_index::Value;
    result=nothing::Union{Nothing,IR.Type},
    layout=nothing,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[tile, tile_slice_index]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(result) && push!(_results, result)
    !isnothing(layout) && push!(_attributes, namedattribute("layout", layout))

    return IR.create_operation(
        "arm_sme.move_tile_slice_to_vector",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=(length(_results) == 0 ? nothing : _results),
        result_inference=(length(_results) == 0 ? true : false),
    )
end

"""
`move_vector_to_tile_slice`

The vector to tile slice operation moves a 1-D scalable vector to a slice
of a 2-D scalable vector tile at the given index. The type of the 1-D
scalable vector to be moved must match the type of the tile slice. A tile
slice is a 1-D vector of horizontally or vertically contiguous elements
within a ZA tile. The updated tile is returned as the result.

An optional tile slice layout attribute specifies whether the tile slice is
horizontal (default) or vertical.

Example 1: Move a vector<[16]xi8> into tile horizontally (default) at given index.
```mlir
%tile_update = arm_sme.move_vector_to_tile_slice %vector, %tile, %tile_slice_index : vector<[16]xi8> into vector<[16]x[16]xi8>
```

Example 2: Move a vector<[2]xf64> into tile vertically at given index.
```mlir
%tile_update = arm_sme.move_vector_to_tile_slice %vector, %tile, %tile_slice_index layout<vertical> : vector<[2]xf64> into vector<[2]x[2]xf64>
```
"""
function move_vector_to_tile_slice(
    vector::Value,
    tile::Value,
    tile_slice_index::Value;
    result=nothing::Union{Nothing,IR.Type},
    layout=nothing,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[vector, tile, tile_slice_index]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(result) && push!(_results, result)
    !isnothing(layout) && push!(_attributes, namedattribute("layout", layout))

    return IR.create_operation(
        "arm_sme.move_vector_to_tile_slice",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=(length(_results) == 0 ? nothing : _results),
        result_inference=(length(_results) == 0 ? true : false),
    )
end

"""
`outerproduct`

This operation represents an outer product that fits within an SME tile.
All operands must be SVE vectors and the result a SME tile. Unlike
`vector.outerproduct` masking is on the operands (rather than the result),
which mirrors the SME instructions.

Example 1: Unmasked outerproduct (without accumulator)
```mlir
// Not specifying an accumulator implicitly zeros the destination tile.
%result = arm_sme.outerproduct \$lhs, \$rhs : vector<[4]xf32>, vector<[4]xf32>
```

Example 2: Unmasked outerproduct (with accumulator)
```mlir
%result = arm_sme.outerproduct \$lhs, \$rhs acc(\$accumulator)
            : vector<[4]xf32>, vector<[4]xf32>
```

Example 3: Masked outerproduct
```mlir
%result = arm_sme.outerproduct \$lhs, \$rhs masks(\$lhsMask, \$rhsMask)
            : vector<[4]xf32>, vector<[4]xf32>
```

Example 4: Masked outerproduct (with accumulator)
```mlir
%result = arm_sme.outerproduct \$lhs, \$rhs acc(\$accumulator) masks(\$lhsMask, \$rhsMask)
            : vector<[4]xf32>, vector<[4]xf32>
```
"""
function outerproduct(
    lhs::Value,
    rhs::Value,
    lhsMask=nothing::Union{Nothing,Value};
    rhsMask=nothing::Union{Nothing,Value},
    acc=nothing::Union{Nothing,Value},
    result=nothing::Union{Nothing,IR.Type},
    kind=nothing,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[lhs, rhs]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(lhsMask) && push!(_operands, lhsMask)
    !isnothing(rhsMask) && push!(_operands, rhsMask)
    !isnothing(acc) && push!(_operands, acc)
    push!(
        _attributes,
        operandsegmentsizes([
            1,
            1,
            isnothing(lhsMask) ? 0 : 1,
            isnothing(rhsMask) ? 0 : 1,
            isnothing(acc) ? 0 : 1,
        ]),
    )
    !isnothing(result) && push!(_results, result)
    !isnothing(kind) && push!(_attributes, namedattribute("kind", kind))

    return IR.create_operation(
        "arm_sme.outerproduct",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=(length(_results) == 0 ? nothing : _results),
        result_inference=(length(_results) == 0 ? true : false),
    )
end

"""
`smopa_2way`

# Example
```mlir
%result = arm_sme.smopa_2way \$lhs, \$rhs : vector<[8]xi16>, vector<[8]xi16> into vector<[4]x[4]xi32>
```

Refer to
[fmopa_2way](#arm_smefmopa_2way-arm_smefmopa2wayop) for a detailed
description of 2-way outer products.

| Spec | Features |
| ---- | -------- |
| [SMOPA (2-way)](https://developer.arm.com/documentation/ddi0602/2023-09/SME-Instructions/SMOPA--2-way---Signed-integer-sum-of-outer-products-and-accumulate-) | +sme2 |
"""
function smopa_2way(
    lhs::Value,
    rhs::Value,
    lhsMask=nothing::Union{Nothing,Value};
    rhsMask=nothing::Union{Nothing,Value},
    acc=nothing::Union{Nothing,Value},
    result::IR.Type,
    location=Location(),
)
    _results = IR.Type[result,]
    _operands = Value[lhs, rhs]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(lhsMask) && push!(_operands, lhsMask)
    !isnothing(rhsMask) && push!(_operands, rhsMask)
    !isnothing(acc) && push!(_operands, acc)
    push!(
        _attributes,
        operandsegmentsizes([
            1,
            1,
            isnothing(lhsMask) ? 0 : 1,
            isnothing(rhsMask) ? 0 : 1,
            isnothing(acc) ? 0 : 1,
        ]),
    )

    return IR.create_operation(
        "arm_sme.smopa_2way",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`smopa_4way`

This operation represents a sum of 4 widened outer products. It takes 2 1-D
scalable vectors as input and a 2-D scalable vector (ZA tile) as output.

For example (i8 to i32):

```mlir
%result = arm_sme.smopa_4way \$lhs, \$rhs :
  vector<[16]xi8>, vector<[16]xi8> into vector<[4]x[4]xi32>
```

The `lhs` encodes a matrix of shape SVLSx4 and the `rhs` a matrix of
4xSVLS, where SVLS (spec [1], section B2.1) is the number of 32-bit
elements in a vector of SVL bits. To illustrate, below is a breakdown of
this operation for i8 to i32, SVL=128 (i.e., vscale=1):

```
                                    LHS
          [A0 A1 A2 A3 A4 A5 A6 A7 A8 A9 A10 A11 A12 A15 A14 A15]

                                    RHS
          [B0 B1 B2 B3 B4 B5 B6 B7 B8 B9 B10 B11 B12 B13 B14 B15]

----------------------------------------------------------------------------

                              implicit layout

                [A0   A1  A2  A3]    |    [B0 B4  B8 B12]
                [A4   A5  A6  A7]    |    [B1 B5  B9 B13]
                [A8   A9 A10 A11]    |    [B2 B6 B10 B14]
                [A12 A13 A14 A15]    |    [B3 B7 B11 B15]

----------------------------------------------------------------------------

                              4 outer products

             Acol0 ⊗ Brow0           |            Acol1 ⊗ Brow1
             -------------           |            -------------
                                     |
         [B0 B4 B8 B12]              |        [B1 B5 B9 B13]
                                     |
   [A0   [ A0B0  A0B4  A0B8  A0B12]  |  [A1   [ A1B1  A1B5  A1B9  A1B13]
    A4   [ A4B0  A4B4  A4B8  A4B12]  |   A5   [ A5B1  A5B5  A5B9  A5B13]
    A8   [ A8B0  A8B4  A8B8  A8B12]  |   A9   [ A9B1  A9B5  A9B9  A9B13]
    A12] [A12B0 A12B4 A12B8 A12B12]  |   A13] [A13B1 A13B5 A13B9 A13B13]
                                     |
             Acol2 ⊗ Brow2           |            Acol3 ⊗ Brow3
             -------------           |            -------------
                                     |
         [B2, B6, B10, B14]          |        [B3 B7 B11 B15]
                                     |
   [A2   [ A2B2  A2B6  A2B10  A2B14] |  [A3   [ A3B3  A3B7  A3B11  A3B15]
    A6   [ A6B2  A6B6  A6B10  A6B14] |   A7   [ A7B3  A7B7  A7B11  A7B15]
    A10  [A10B2 A10B6 A10B10 A10B14] |   A11  [A11B3 A11B7 A11B11 A11B15]
    A14] [A14B2 A14B6 A14B10 A14B14] |   A15] [A15B3 A15B7 A15B11 A15B15]
                                     |

----------------------------------------------------------------------------

                          sum of 4 outer products

       Acol0 ⊗ Brow0 + Acol1 ⊗ Brow1 + Acol2 ⊗ Brow2 + Acol3 ⊗ Brow3

 [ A0B0 +  A1B1 +  A2B2 +  A3B3 ... ...  A0B12 +  A1B13 +  A2B14 +  A3B15]
 [ A4B0 +  A5B1 +  A6B2 +  A7B3 ... ...  A4B12 +  A5B13 +  A6B14 +  A7B15]
 [ A8B0 +  A9B1 + A10B2 + A11B3 ... ...  A8B12 +  A9B13 + A10B14 + A11B15]
 [A12B0 + A13B1 + A14B2 + A15B3 ... ... A12B12 + A13B13 + A14B14 + A15B15]

----------------------------------------------------------------------------
```

This operation enables the folding of 4 outer products chained via the
accumulator into a single outer product.

For example:

```mlir
%a0_ext = arith.extsi %a0 : vector<[4]xi8> to vector<[4]xi32>
%b0_ext = arith.extsi %b0 : vector<[4]xi8> to vector<[4]xi32>

%a1_ext = arith.extsi %a1 : vector<[4]xi8> to vector<[4]xi32>
%b1_ext = arith.extsi %b1 : vector<[4]xi8> to vector<[4]xi32>

%a2_ext = arith.extsi %a2 : vector<[4]xi8> to vector<[4]xi32>
%b2_ext = arith.extsi %b2 : vector<[4]xi8> to vector<[4]xi32>

%a3_ext = arith.extsi %a3 : vector<[4]xi8> to vector<[4]xi32>
%b3_ext = arith.extsi %b3 : vector<[4]xi8> to vector<[4]xi32>

%0 = arm_sme.outerproduct %a0_ext, %b0_ext : vector<[4]xi32>, vector<[4]xi32>
%1 = arm_sme.outerproduct %a1_ext, %b1_ext acc(%0) : vector<[4]xi32>, vector<[4]xi32>
%2 = arm_sme.outerproduct %a2_ext, %b2_ext acc(%1) : vector<[4]xi32>, vector<[4]xi32>
%3 = arm_sme.outerproduct %a3_ext, %b3_ext acc(%2) : vector<[4]xi32>, vector<[4]xi32>
```

The 4 outer products in the example above can be fused into a single outer
product as follows:

```mlir
%lhs0 = \"llvm.intr.experimental.vector.interleave2\"(%a0, %a2) : (vector<[4]xi8>, vector<[4]xi8>) -> vector<[8]xi8>
%lhs1 = \"llvm.intr.experimental.vector.interleave2\"(%a1, %a3) : (vector<[4]xi8>, vector<[4]xi8>) -> vector<[8]xi8>
%lhs = \"llvm.intr.experimental.vector.interleave2\"(%lhs0, %lhs1) : (vector<[8]xi8>, vector<[8]xi8>) -> vector<[16]xi8>

%rhs0 = \"llvm.intr.experimental.vector.interleave2\"(%b0, %b2) : (vector<[4]xi8>, vector<[4]xi8>) -> vector<[8]xi8>
%rhs1 = \"llvm.intr.experimental.vector.interleave2\"(%b1, %b3) : (vector<[4]xi8>, vector<[4]xi8>) -> vector<[8]xi8>
%rhs = \"llvm.intr.experimental.vector.interleave2\"(%rhs0, %rhs1) : (vector<[8]xi8>, vector<[8]xi8>) -> vector<[16]xi8>

%0 = arm_sme.smopa_4way %lhs, %rhs : vector<[16]xi8>, vector<[16]xi8> into vector<[4]x[4]xi32>
```

This is implemented in the `-arm-sme-outer-product-fusion` pass.

# Example I8 to I32
```mlir
%result = arm_sme.smopa_4way \$lhs, \$rhs : vector<[16]xi8>, vector<[16]xi8> into vector<[4]x[4]xi32>
```

# Example I16 to I64
```mlir
%result = arm_sme.smopa_4way \$lhs, \$rhs : vector<[8]xi16>, vector<[8]xi16> into vector<[2]x[2]xi64>
```

| Spec | Features |
| ---- | -------- |
| [SMOPA (4-way)](https://developer.arm.com/documentation/ddi0602/2023-09/SME-Instructions/SMOPA--4-way---Signed-integer-sum-of-outer-products-and-accumulate-) | +sme (32-bit), +sme-i16i64 (64-bit)|
"""
function smopa_4way(
    lhs::Value,
    rhs::Value,
    lhsMask=nothing::Union{Nothing,Value};
    rhsMask=nothing::Union{Nothing,Value},
    acc=nothing::Union{Nothing,Value},
    result::IR.Type,
    location=Location(),
)
    _results = IR.Type[result,]
    _operands = Value[lhs, rhs]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(lhsMask) && push!(_operands, lhsMask)
    !isnothing(rhsMask) && push!(_operands, rhsMask)
    !isnothing(acc) && push!(_operands, acc)
    push!(
        _attributes,
        operandsegmentsizes([
            1,
            1,
            isnothing(lhsMask) ? 0 : 1,
            isnothing(rhsMask) ? 0 : 1,
            isnothing(acc) ? 0 : 1,
        ]),
    )

    return IR.create_operation(
        "arm_sme.smopa_4way",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`smops_2way`

# Example
```mlir
%result = arm_sme.smops_2way \$lhs, \$rhs : vector<[8]xi16>, vector<[8]xi16> into vector<[4]x[4]xi32>
```

Refer to
[fmopa_2way](#arm_smefmopa_2way-arm_smefmopa2wayop) for a detailed
description of 2-way outer products.

| Spec | Features |
| ---- | -------- |
| [SMOPS (2-way)](https://developer.arm.com/documentation/ddi0602/2023-09/SME-Instructions/SMOPS--2-way---Signed-integer-sum-of-outer-products-and-subtract-) | +sme2 |
"""
function smops_2way(
    lhs::Value,
    rhs::Value,
    lhsMask=nothing::Union{Nothing,Value};
    rhsMask=nothing::Union{Nothing,Value},
    acc=nothing::Union{Nothing,Value},
    result::IR.Type,
    location=Location(),
)
    _results = IR.Type[result,]
    _operands = Value[lhs, rhs]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(lhsMask) && push!(_operands, lhsMask)
    !isnothing(rhsMask) && push!(_operands, rhsMask)
    !isnothing(acc) && push!(_operands, acc)
    push!(
        _attributes,
        operandsegmentsizes([
            1,
            1,
            isnothing(lhsMask) ? 0 : 1,
            isnothing(rhsMask) ? 0 : 1,
            isnothing(acc) ? 0 : 1,
        ]),
    )

    return IR.create_operation(
        "arm_sme.smops_2way",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`smops_4way`

Equivalent to `smopa_4way` but outer products are subtracted from
destination `result`.

# Example I8 to I32
```mlir
%result = arm_sme.smops_4way \$lhs, \$rhs : vector<[16]xi8>, vector<[16]xi8> into vector<[4]x[4]xi32>
```

# Example I16 to I64
```mlir
%result = arm_sme.smops_4way \$lhs, \$rhs : vector<[8]xi16>, vector<[8]xi16> into vector<[2]x[2]xi64>
```

Refer to [smopa_4way](#arm_smesmopa_4way-arm_smesmopa4wayop) for a
detailed description of 4-way outer products.

| Spec | Features |
| ---- | -------- |
| [SMOPS (4-way)](https://developer.arm.com/documentation/ddi0602/2023-09/SME-Instructions/SMOPS--4-way---Signed-integer-sum-of-outer-products-and-subtract-) | +sme (32-bit), +sme-i16i64 (64-bit)|
"""
function smops_4way(
    lhs::Value,
    rhs::Value,
    lhsMask=nothing::Union{Nothing,Value};
    rhsMask=nothing::Union{Nothing,Value},
    acc=nothing::Union{Nothing,Value},
    result::IR.Type,
    location=Location(),
)
    _results = IR.Type[result,]
    _operands = Value[lhs, rhs]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(lhsMask) && push!(_operands, lhsMask)
    !isnothing(rhsMask) && push!(_operands, rhsMask)
    !isnothing(acc) && push!(_operands, acc)
    push!(
        _attributes,
        operandsegmentsizes([
            1,
            1,
            isnothing(lhsMask) ? 0 : 1,
            isnothing(rhsMask) ? 0 : 1,
            isnothing(acc) ? 0 : 1,
        ]),
    )

    return IR.create_operation(
        "arm_sme.smops_4way",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`store_tile_slice`

Stores a 1D tile slice from a 2D SME \"virtual tile\" into memory. The tile
slice is defined by the dimension of the 2D scalable vector type pointed by
the index. A tile slice index describes where in the input tile the tile
slice is stored from. An optional tile slice layout attribute specifies
whether the tile slice being stored from the given index is horizontal
(default) or vertical.

The slice of memory written is defined by a base and indices and must be
contiguous. The memref must be either rank 1 or rank 2, have dynamic
dimensions since the operation is scalable, and the element type must be a
scalar that matches the element type of the input tile.

The provided `mask` is used to specify which elements of the tile slice
will be stored.

Example 1: Store vector<[16]xi8> horizontal (default) tile slice from tile at given index to memory.
```mlir
arm_sme.store_tile_slice %tile, %tile_slice_index, %mask, %base[%c0] : vector<[16]x[16]xi8>, vector<[16]xi1>, memref<?x?xi8>
```

Example 2: Store vector<[4]xf32> vertical tile slice from tile at given index to memory.
```mlir
arm_sme.store_tile_slice %tile, %tile_slice_index, %mask, %base[%c0] layout<vertical> : vector<[4]x[4]xf32>, vector<[4]xi1>, memref<?x?xf32>
```

Example 3: Store a vector<[1]xi128> vertical tile slice from tile at given index to memory.
```mlir
arm_sme.store_tile_slice %tile, %tile_slice_index, %mask, %base[%c0] layout<vertical> : vector<[1]x[1]xi128>, vector<[1]xi1>, memref<?x?xi128>
```
"""
function store_tile_slice(
    tile::Value,
    tile_slice_index::Value,
    mask::Value,
    base::Value,
    indices::Vector{Value};
    layout=nothing,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[tile, tile_slice_index, mask, base, indices...]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(layout) && push!(_attributes, namedattribute("layout", layout))

    return IR.create_operation(
        "arm_sme.store_tile_slice",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`streaming_vl`

This operation returns the streaming vector length (SVL) for a given type
size. Unlike `vector.vscale` the value returned is invariant to the
streaming mode.

# Example
```mlir
// Streaming vector length in:
// - bytes (8-bit, SVL.B)
%svl_b = arm_sme.streaming_vl <byte>
// - half words (16-bit, SVL.H)
%svl_h = arm_sme.streaming_vl <half>
// - words (32-bit, SVL.W)
%svl_w = arm_sme.streaming_vl <word>
// - double words (64-bit, SVL.D)
%svl_d = arm_sme.streaming_vl <double>
```
"""
function streaming_vl(;
    result_0=nothing::Union{Nothing,IR.Type}, type_size, location=Location()
)
    _results = IR.Type[]
    _operands = Value[]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[namedattribute("type_size", type_size),]
    !isnothing(result_0) && push!(_results, result_0)

    return IR.create_operation(
        "arm_sme.streaming_vl",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=(length(_results) == 0 ? nothing : _results),
        result_inference=(length(_results) == 0 ? true : false),
    )
end

"""
`sumopa_4way`

# Example I8 to I32
```mlir
%result = arm_sme.sumopa_4way \$lhs, \$rhs : vector<[16]xi8>, vector<[16]xi8> into vector<[4]x[4]xi32>
```

# Example I16 to I64
```mlir
%result = arm_sme.sumopa_4way \$lhs, \$rhs : vector<[8]xi16>, vector<[8]xi16> into vector<[2]x[2]xi64>
```

Refer to [smopa_4way](#arm_smesmopa_4way-arm_smesmopa4wayop) for a
detailed description of 4-way outer products.

| Spec | Features |
| ---- | -------- |
| [SUMOPA (4-way)](https://developer.arm.com/documentation/ddi0602/2023-09/SME-Instructions/SUMOPA--Signed-by-unsigned-integer-sum-of-outer-products-and-accumulate-) | +sme (32-bit), +sme-i16i64 (64-bit)|
"""
function sumopa_4way(
    lhs::Value,
    rhs::Value,
    lhsMask=nothing::Union{Nothing,Value};
    rhsMask=nothing::Union{Nothing,Value},
    acc=nothing::Union{Nothing,Value},
    result::IR.Type,
    location=Location(),
)
    _results = IR.Type[result,]
    _operands = Value[lhs, rhs]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(lhsMask) && push!(_operands, lhsMask)
    !isnothing(rhsMask) && push!(_operands, rhsMask)
    !isnothing(acc) && push!(_operands, acc)
    push!(
        _attributes,
        operandsegmentsizes([
            1,
            1,
            isnothing(lhsMask) ? 0 : 1,
            isnothing(rhsMask) ? 0 : 1,
            isnothing(acc) ? 0 : 1,
        ]),
    )

    return IR.create_operation(
        "arm_sme.sumopa_4way",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`sumops_4way`

# Example I8 to I32
```mlir
%result = arm_sme.sumops_4way \$lhs, \$rhs : vector<[16]xi8>, vector<[16]xi8> into vector<[4]x[4]xi32>
```

# Example I16 to I64
```mlir
%result = arm_sme.sumops_4way \$lhs, \$rhs : vector<[8]xi16>, vector<[8]xi16> into vector<[2]x[2]xi64>
```

Refer to [smopa_4way](#arm_smesmopa_4way-arm_smesmopa4wayop) for a
detailed description of 4-way outer products.

| Spec | Features |
| ---- | -------- |
| [SUMOPS (4-way)](https://developer.arm.com/documentation/ddi0602/2023-09/SME-Instructions/SUMOPS--Signed-by-unsigned-integer-sum-of-outer-products-and-subtract-) | +sme (32-bit), +sme-i16i64 (64-bit)|
"""
function sumops_4way(
    lhs::Value,
    rhs::Value,
    lhsMask=nothing::Union{Nothing,Value};
    rhsMask=nothing::Union{Nothing,Value},
    acc=nothing::Union{Nothing,Value},
    result::IR.Type,
    location=Location(),
)
    _results = IR.Type[result,]
    _operands = Value[lhs, rhs]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(lhsMask) && push!(_operands, lhsMask)
    !isnothing(rhsMask) && push!(_operands, rhsMask)
    !isnothing(acc) && push!(_operands, acc)
    push!(
        _attributes,
        operandsegmentsizes([
            1,
            1,
            isnothing(lhsMask) ? 0 : 1,
            isnothing(rhsMask) ? 0 : 1,
            isnothing(acc) ? 0 : 1,
        ]),
    )

    return IR.create_operation(
        "arm_sme.sumops_4way",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`tile_load`

Loads a 2D SME \"virtual tile\" from memory defined by a base and indices,
with the shape defined by the 2D scalable vector type of the result tile.
An optional tile slice layout attribute specifies whether the slices of the
tile being loaded are horizontal (default) or vertical. The slice of memory
must be contiguous. The memref must be either rank 1 or rank 2 with dynamic
dimensions, since the operation is scalable, and the element type must be a
scalar that matches the element type of the result.

An optional SSA value `padding` of the same elemental type as the MemRef is
provided to specify a fallback value in the case of masking.

An optional SSA value `mask` may be specified to mask out elements read
from the MemRef. The `mask` type is an `i1` vector with a shape that
matches how elements are read from the MemRef. Elements whose corresponding
mask element is `0` are masked out and replaced with `padding`.

If either `padding` or `mask` are specified, both must be specified.

Example 1: Load an 8-bit element ZA tile with horizontal layout (default) from memory (ZA0.B).
```mlir
%tile = arm_sme.tile_load %base[%c0, %c0] : memref<?x?xi8>, vector<[16]x[16]xi8>
```

Example 2: Load a FP 32-bit element ZA tile with vertical layout from memory.
```mlir
%tile = arm_sme.tile_load %base[%c0, %c0] layout<vertical> : memref<?x?xf32>, vector<[4]x[4]xf32>
```

Example 3: Load a 128-bit element ZA tile with horizontal layout (default) from memory.
```mlir
%tile = arm_sme.tile_load %base[%c0, %c0] layout<horizontal> : memref<?x?xi128>, vector<[1]x[1]xi128>
```

Example 4: Masked load of int 32-bit element ZA tile with horizontal layout (default) from memory.
```mlir
%tile = arm_sme.tile_load %base[%c0, %c0], %pad, %mask : memref<?x?xf32>, vector<[4]x[4]xf32>
```
"""
function tile_load(
    base::Value,
    indices::Vector{Value},
    padding=nothing::Union{Nothing,Value};
    mask=nothing::Union{Nothing,Value},
    result::IR.Type,
    layout=nothing,
    location=Location(),
)
    _results = IR.Type[result,]
    _operands = Value[base, indices...]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(padding) && push!(_operands, padding)
    !isnothing(mask) && push!(_operands, mask)
    push!(
        _attributes,
        operandsegmentsizes([
            1, length(indices), isnothing(padding) ? 0 : 1, isnothing(mask) ? 0 : 1
        ]),
    )
    !isnothing(layout) && push!(_attributes, namedattribute("layout", layout))

    return IR.create_operation(
        "arm_sme.tile_load",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`tile_store`

Stores a 2D SME \"virtual tile\" to memory defined by a base and indices,
with the shape defined by the 2D scalable vector type of the tile being
stored. An optional tile slice layout attribute specifies whether the
slices of the tile being stored are horizontal (default) or vertical. The
slice of memory must be contiguous. The memref must be either rank 1 or
rank 2 with dynamic dimensions, since the operation is scalable, and the
element type must be a scalar that matches the element type of the result.

An optional `mask` may be provided, the shape of which corresponds to the
`tile`, and selects which elements of the tile will be stored.

Example 1: Store an 8-bit element ZA tile with horizontal (default) layout to memory (ZA0.B).
```mlir
arm_sme.tile_store %tile, %base[%c0, %c0] : vector<[16]x[16]xi8>, memref<?x?xi8>
```

Example 2: Store a FP 32-bit element ZA tile with vertical layout to memory.
```mlir
arm_sme.tile_store %tile, %base[%c0, %c0] layout<vertical> : vector<[4]x[4]xf32>, memref<?x?xf32>
```

Example 3: Store a 128-bit element ZA tile with horizontal (default) layout to memory.
```mlir
arm_sme.tile_store %tile, %base[%c0, %c0] layout<horizontal> : vector<[1]x[1]xi128>, memref<?x?xi128>
```

Example 4: Masked store a int 32-bit element ZA tile with vertical layout to memory.
```mlir
arm_sme.tile_store %tile, %base[%c0, %c0], %mask layout<vertical> : vector<[4]x[4]xf32>, memref<?x?xf32>
```
"""
function tile_store(
    valueToStore::Value,
    base::Value,
    indices::Vector{Value},
    mask=nothing::Union{Nothing,Value};
    layout=nothing,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[valueToStore, base, indices...]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(mask) && push!(_operands, mask)
    push!(
        _attributes, operandsegmentsizes([1, 1, length(indices), isnothing(mask) ? 0 : 1])
    )
    !isnothing(layout) && push!(_attributes, namedattribute("layout", layout))

    return IR.create_operation(
        "arm_sme.tile_store",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`umopa_2way`

# Example
```mlir
%result = arm_sme.umopa_2way \$lhs, \$rhs : vector<[8]xi16>, vector<[8]xi16> into vector<[4]x[4]xi32>
```

Refer to
[fmopa_2way](#arm_smefmopa_2way-arm_smefmopa2wayop) for a detailed
description of 2-way outer products.

| Spec | Features |
| ---- | -------- |
| [UMOPA (2-way)](https://developer.arm.com/documentation/ddi0602/2023-09/SME-Instructions/UMOPA--2-way---Unsigned-integer-sum-of-outer-products-and-accumulate-) | +sme2 |
"""
function umopa_2way(
    lhs::Value,
    rhs::Value,
    lhsMask=nothing::Union{Nothing,Value};
    rhsMask=nothing::Union{Nothing,Value},
    acc=nothing::Union{Nothing,Value},
    result::IR.Type,
    location=Location(),
)
    _results = IR.Type[result,]
    _operands = Value[lhs, rhs]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(lhsMask) && push!(_operands, lhsMask)
    !isnothing(rhsMask) && push!(_operands, rhsMask)
    !isnothing(acc) && push!(_operands, acc)
    push!(
        _attributes,
        operandsegmentsizes([
            1,
            1,
            isnothing(lhsMask) ? 0 : 1,
            isnothing(rhsMask) ? 0 : 1,
            isnothing(acc) ? 0 : 1,
        ]),
    )

    return IR.create_operation(
        "arm_sme.umopa_2way",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`umopa_4way`

# Example I8 to I32
```mlir
%result = arm_sme.umopa_4way \$lhs, \$rhs : vector<[16]xi8>, vector<[16]xi8> into vector<[4]x[4]xi32>
```

# Example I16 to I64
```mlir
%result = arm_sme.umopa_4way \$lhs, \$rhs : vector<[8]xi16>, vector<[8]xi16> into vector<[2]x[2]xi64>
```

Refer to [smopa_4way](#arm_smesmopa_4way-arm_smesmopa4wayop) for a
detailed description of 4-way outer products.

| Spec | Features |
| ---- | -------- |
| [UMOPA (4-way)](https://developer.arm.com/documentation/ddi0602/2023-09/SME-Instructions/UMOPA--4-way---Unsigned-integer-sum-of-outer-products-and-accumulate-) | +sme (32-bit), +sme-i16i64 (64-bit)|
"""
function umopa_4way(
    lhs::Value,
    rhs::Value,
    lhsMask=nothing::Union{Nothing,Value};
    rhsMask=nothing::Union{Nothing,Value},
    acc=nothing::Union{Nothing,Value},
    result::IR.Type,
    location=Location(),
)
    _results = IR.Type[result,]
    _operands = Value[lhs, rhs]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(lhsMask) && push!(_operands, lhsMask)
    !isnothing(rhsMask) && push!(_operands, rhsMask)
    !isnothing(acc) && push!(_operands, acc)
    push!(
        _attributes,
        operandsegmentsizes([
            1,
            1,
            isnothing(lhsMask) ? 0 : 1,
            isnothing(rhsMask) ? 0 : 1,
            isnothing(acc) ? 0 : 1,
        ]),
    )

    return IR.create_operation(
        "arm_sme.umopa_4way",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`umops_2way`

# Example
```mlir
%result = arm_sme.umops_2way \$lhs, \$rhs : vector<[8]xi16>, vector<[8]xi16> into vector<[4]x[4]xi32>
```

Refer to
[fmopa_2way](#arm_smefmopa_2way-arm_smefmopa2wayop) for a detailed
description of 2-way outer products.

| Spec | Features |
| ---- | -------- |
| [UMOPS (2-way)](https://developer.arm.com/documentation/ddi0602/2023-09/SME-Instructions/UMOPS--2-way---Unsigned-integer-sum-of-outer-products-and-subtract-) | +sme2 |
"""
function umops_2way(
    lhs::Value,
    rhs::Value,
    lhsMask=nothing::Union{Nothing,Value};
    rhsMask=nothing::Union{Nothing,Value},
    acc=nothing::Union{Nothing,Value},
    result::IR.Type,
    location=Location(),
)
    _results = IR.Type[result,]
    _operands = Value[lhs, rhs]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(lhsMask) && push!(_operands, lhsMask)
    !isnothing(rhsMask) && push!(_operands, rhsMask)
    !isnothing(acc) && push!(_operands, acc)
    push!(
        _attributes,
        operandsegmentsizes([
            1,
            1,
            isnothing(lhsMask) ? 0 : 1,
            isnothing(rhsMask) ? 0 : 1,
            isnothing(acc) ? 0 : 1,
        ]),
    )

    return IR.create_operation(
        "arm_sme.umops_2way",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`umops_4way`

# Example I8 to I32
```mlir
%result = arm_sme.umops_4way \$lhs, \$rhs : vector<[16]xi8>, vector<[16]xi8> into vector<[4]x[4]xi32>
```

# Example I16 to I64
```mlir
%result = arm_sme.umops_4way \$lhs, \$rhs : vector<[8]xi16>, vector<[8]xi16> into vector<[2]x[2]xi64>
```

Refer to [smopa_4way](#arm_smesmopa_4way-arm_smesmopa4wayop) for a
detailed description of 4-way outer products.

| Spec | Features |
| ---- | -------- |
| [UMOPS (4-way)](https://developer.arm.com/documentation/ddi0602/2023-09/SME-Instructions/UMOPS--4-way---Unsigned-integer-sum-of-outer-products-and-subtract-) | +sme (32-bit), +sme-i16i64 (64-bit)|
"""
function umops_4way(
    lhs::Value,
    rhs::Value,
    lhsMask=nothing::Union{Nothing,Value};
    rhsMask=nothing::Union{Nothing,Value},
    acc=nothing::Union{Nothing,Value},
    result::IR.Type,
    location=Location(),
)
    _results = IR.Type[result,]
    _operands = Value[lhs, rhs]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(lhsMask) && push!(_operands, lhsMask)
    !isnothing(rhsMask) && push!(_operands, rhsMask)
    !isnothing(acc) && push!(_operands, acc)
    push!(
        _attributes,
        operandsegmentsizes([
            1,
            1,
            isnothing(lhsMask) ? 0 : 1,
            isnothing(rhsMask) ? 0 : 1,
            isnothing(acc) ? 0 : 1,
        ]),
    )

    return IR.create_operation(
        "arm_sme.umops_4way",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`usmopa_4way`

# Example I8 to I32
```mlir
%result = arm_sme.usmopa_4way \$lhs, \$rhs : vector<[16]xi8>, vector<[16]xi8> into vector<[4]x[4]xi32>
```

# Example I16 to I64
```mlir
%result = arm_sme.usmopa_4way \$lhs, \$rhs : vector<[8]xi16>, vector<[8]xi16> into vector<[2]x[2]xi64>
```

Refer to [smopa_4way](#arm_smesmopa_4way-arm_smesmopa4wayop) for a
detailed description of 4-way outer products.

| Spec | Features |
| ---- | -------- |
| [USMOPA (4-way)](https://developer.arm.com/documentation/ddi0602/2023-09/SME-Instructions/USMOPA--Unsigned-by-signed-integer-sum-of-outer-products-and-accumulate-) | +sme (32-bit), +sme-i16i64 (64-bit)|
"""
function usmopa_4way(
    lhs::Value,
    rhs::Value,
    lhsMask=nothing::Union{Nothing,Value};
    rhsMask=nothing::Union{Nothing,Value},
    acc=nothing::Union{Nothing,Value},
    result::IR.Type,
    location=Location(),
)
    _results = IR.Type[result,]
    _operands = Value[lhs, rhs]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(lhsMask) && push!(_operands, lhsMask)
    !isnothing(rhsMask) && push!(_operands, rhsMask)
    !isnothing(acc) && push!(_operands, acc)
    push!(
        _attributes,
        operandsegmentsizes([
            1,
            1,
            isnothing(lhsMask) ? 0 : 1,
            isnothing(rhsMask) ? 0 : 1,
            isnothing(acc) ? 0 : 1,
        ]),
    )

    return IR.create_operation(
        "arm_sme.usmopa_4way",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`usmops_4way`

# Example I8 to I32
```mlir
%result = arm_sme.usmops_4way \$lhs, \$rhs : vector<[16]xi8>, vector<[16]xi8> into vector<[4]x[4]xi32>
```

# Example I16 to I64
```mlir
%result = arm_sme.usmops_4way \$lhs, \$rhs : vector<[8]xi16>, vector<[8]xi16> into vector<[2]x[2]xi64>
```

Refer to [smopa_4way](#arm_smesmopa_4way-arm_smesmopa4wayop) for a
detailed description of 4-way outer products.

| Spec | Features |
| ---- | -------- |
| [USMOPS (4-way)](https://developer.arm.com/documentation/ddi0602/2023-09/SME-Instructions/USMOPS--Unsigned-by-signed-integer-sum-of-outer-products-and-subtract-) | +sme (32-bit), +sme-i16i64 (64-bit)|
"""
function usmops_4way(
    lhs::Value,
    rhs::Value,
    lhsMask=nothing::Union{Nothing,Value};
    rhsMask=nothing::Union{Nothing,Value},
    acc=nothing::Union{Nothing,Value},
    result::IR.Type,
    location=Location(),
)
    _results = IR.Type[result,]
    _operands = Value[lhs, rhs]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(lhsMask) && push!(_operands, lhsMask)
    !isnothing(rhsMask) && push!(_operands, rhsMask)
    !isnothing(acc) && push!(_operands, acc)
    push!(
        _attributes,
        operandsegmentsizes([
            1,
            1,
            isnothing(lhsMask) ? 0 : 1,
            isnothing(rhsMask) ? 0 : 1,
            isnothing(acc) ? 0 : 1,
        ]),
    )

    return IR.create_operation(
        "arm_sme.usmops_4way",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`zero`

Creates a new SME \"virtual tile\" value within a function. The contents of
the tile returned from this operation are zero-initialized.

Example 1: Zero an 8-bit element ZA tile.

```mlir
%0 = arm_sme.zero : vector<[16]x[16]xi8>
```

Example 2: Zero a 64-bit element ZA tile.

```mlir
%0 = arm_sme.zero : vector<[2]x[2]xi64>
```
"""
function zero(; res::IR.Type, location=Location())
    _results = IR.Type[res,]
    _operands = Value[]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "arm_sme.zero",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

import ...IR:
    IR, NamedAttribute, Value, Location, Block, Region, Attribute, context, IndexType
import ..Dialects: namedattribute, operandsegmentsizes

"""
`intr_cntsb`

"""
function intr_cntsb(; res::IR.Type, location=Location())
    _results = IR.Type[res,]
    _operands = Value[]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "arm_sme.intr.cntsb",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_cntsd`

"""
function intr_cntsd(; res::IR.Type, location=Location())
    _results = IR.Type[res,]
    _operands = Value[]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "arm_sme.intr.cntsd",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_cntsh`

"""
function intr_cntsh(; res::IR.Type, location=Location())
    _results = IR.Type[res,]
    _operands = Value[]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "arm_sme.intr.cntsh",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_cntsw`

"""
function intr_cntsw(; res::IR.Type, location=Location())
    _results = IR.Type[res,]
    _operands = Value[]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "arm_sme.intr.cntsw",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_ld1b_horiz`

"""
function intr_ld1b_horiz(
    predicate::Value,
    load_address::Value,
    tile_slice_index::Value;
    tile_id,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[predicate, load_address, tile_slice_index]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[namedattribute("tile_id", tile_id),]

    return IR.create_operation(
        "arm_sme.intr.ld1b.horiz",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_ld1b_vert`

"""
function intr_ld1b_vert(
    predicate::Value,
    load_address::Value,
    tile_slice_index::Value;
    tile_id,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[predicate, load_address, tile_slice_index]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[namedattribute("tile_id", tile_id),]

    return IR.create_operation(
        "arm_sme.intr.ld1b.vert",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_ld1d_horiz`

"""
function intr_ld1d_horiz(
    predicate::Value,
    load_address::Value,
    tile_slice_index::Value;
    tile_id,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[predicate, load_address, tile_slice_index]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[namedattribute("tile_id", tile_id),]

    return IR.create_operation(
        "arm_sme.intr.ld1d.horiz",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_ld1d_vert`

"""
function intr_ld1d_vert(
    predicate::Value,
    load_address::Value,
    tile_slice_index::Value;
    tile_id,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[predicate, load_address, tile_slice_index]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[namedattribute("tile_id", tile_id),]

    return IR.create_operation(
        "arm_sme.intr.ld1d.vert",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_ld1h_horiz`

"""
function intr_ld1h_horiz(
    predicate::Value,
    load_address::Value,
    tile_slice_index::Value;
    tile_id,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[predicate, load_address, tile_slice_index]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[namedattribute("tile_id", tile_id),]

    return IR.create_operation(
        "arm_sme.intr.ld1h.horiz",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_ld1h_vert`

"""
function intr_ld1h_vert(
    predicate::Value,
    load_address::Value,
    tile_slice_index::Value;
    tile_id,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[predicate, load_address, tile_slice_index]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[namedattribute("tile_id", tile_id),]

    return IR.create_operation(
        "arm_sme.intr.ld1h.vert",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_ld1q_horiz`

"""
function intr_ld1q_horiz(
    predicate::Value,
    load_address::Value,
    tile_slice_index::Value;
    tile_id,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[predicate, load_address, tile_slice_index]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[namedattribute("tile_id", tile_id),]

    return IR.create_operation(
        "arm_sme.intr.ld1q.horiz",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_ld1q_vert`

"""
function intr_ld1q_vert(
    predicate::Value,
    load_address::Value,
    tile_slice_index::Value;
    tile_id,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[predicate, load_address, tile_slice_index]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[namedattribute("tile_id", tile_id),]

    return IR.create_operation(
        "arm_sme.intr.ld1q.vert",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_ld1w_horiz`

"""
function intr_ld1w_horiz(
    predicate::Value,
    load_address::Value,
    tile_slice_index::Value;
    tile_id,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[predicate, load_address, tile_slice_index]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[namedattribute("tile_id", tile_id),]

    return IR.create_operation(
        "arm_sme.intr.ld1w.horiz",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_ld1w_vert`

"""
function intr_ld1w_vert(
    predicate::Value,
    load_address::Value,
    tile_slice_index::Value;
    tile_id,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[predicate, load_address, tile_slice_index]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[namedattribute("tile_id", tile_id),]

    return IR.create_operation(
        "arm_sme.intr.ld1w.vert",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_mopa`

"""
function intr_mopa(
    lhs_predicate::Value,
    rhs_predicate::Value,
    lhs_vector::Value,
    rhs_vector::Value;
    tile_id,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[lhs_predicate, rhs_predicate, lhs_vector, rhs_vector]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[namedattribute("tile_id", tile_id),]

    return IR.create_operation(
        "arm_sme.intr.mopa",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_mopa_wide`

"""
function intr_mopa_wide(
    lhs_predicate::Value,
    rhs_predicate::Value,
    lhs_vector::Value,
    rhs_vector::Value;
    tile_id,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[lhs_predicate, rhs_predicate, lhs_vector, rhs_vector]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[namedattribute("tile_id", tile_id),]

    return IR.create_operation(
        "arm_sme.intr.mopa.wide",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_mops`

"""
function intr_mops(
    lhs_predicate::Value,
    rhs_predicate::Value,
    lhs_vector::Value,
    rhs_vector::Value;
    tile_id,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[lhs_predicate, rhs_predicate, lhs_vector, rhs_vector]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[namedattribute("tile_id", tile_id),]

    return IR.create_operation(
        "arm_sme.intr.mops",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_mops_wide`

"""
function intr_mops_wide(
    lhs_predicate::Value,
    rhs_predicate::Value,
    lhs_vector::Value,
    rhs_vector::Value;
    tile_id,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[lhs_predicate, rhs_predicate, lhs_vector, rhs_vector]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[namedattribute("tile_id", tile_id),]

    return IR.create_operation(
        "arm_sme.intr.mops.wide",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_read_horiz`

"""
function intr_read_horiz(
    vector::Value,
    predicate::Value,
    tile_slice_index::Value;
    res::IR.Type,
    tile_id,
    location=Location(),
)
    _results = IR.Type[res,]
    _operands = Value[vector, predicate, tile_slice_index]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[namedattribute("tile_id", tile_id),]

    return IR.create_operation(
        "arm_sme.intr.read.horiz",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_read_vert`

"""
function intr_read_vert(
    vector::Value,
    predicate::Value,
    tile_slice_index::Value;
    res::IR.Type,
    tile_id,
    location=Location(),
)
    _results = IR.Type[res,]
    _operands = Value[vector, predicate, tile_slice_index]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[namedattribute("tile_id", tile_id),]

    return IR.create_operation(
        "arm_sme.intr.read.vert",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_smopa_wide`

"""
function intr_smopa_wide(
    lhs_predicate::Value,
    rhs_predicate::Value,
    lhs_vector::Value,
    rhs_vector::Value;
    tile_id,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[lhs_predicate, rhs_predicate, lhs_vector, rhs_vector]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[namedattribute("tile_id", tile_id),]

    return IR.create_operation(
        "arm_sme.intr.smopa.wide",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_smopa_za32`

"""
function intr_smopa_za32(
    lhs_predicate::Value,
    rhs_predicate::Value,
    lhs_vector::Value,
    rhs_vector::Value;
    tile_id,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[lhs_predicate, rhs_predicate, lhs_vector, rhs_vector]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[namedattribute("tile_id", tile_id),]

    return IR.create_operation(
        "arm_sme.intr.smopa.za32",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_smops_wide`

"""
function intr_smops_wide(
    lhs_predicate::Value,
    rhs_predicate::Value,
    lhs_vector::Value,
    rhs_vector::Value;
    tile_id,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[lhs_predicate, rhs_predicate, lhs_vector, rhs_vector]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[namedattribute("tile_id", tile_id),]

    return IR.create_operation(
        "arm_sme.intr.smops.wide",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_smops_za32`

"""
function intr_smops_za32(
    lhs_predicate::Value,
    rhs_predicate::Value,
    lhs_vector::Value,
    rhs_vector::Value;
    tile_id,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[lhs_predicate, rhs_predicate, lhs_vector, rhs_vector]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[namedattribute("tile_id", tile_id),]

    return IR.create_operation(
        "arm_sme.intr.smops.za32",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_st1b_horiz`

"""
function intr_st1b_horiz(
    predicate::Value,
    store_address::Value,
    tile_slice_index::Value;
    tile_id,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[predicate, store_address, tile_slice_index]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[namedattribute("tile_id", tile_id),]

    return IR.create_operation(
        "arm_sme.intr.st1b.horiz",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_st1b_vert`

"""
function intr_st1b_vert(
    predicate::Value,
    store_address::Value,
    tile_slice_index::Value;
    tile_id,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[predicate, store_address, tile_slice_index]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[namedattribute("tile_id", tile_id),]

    return IR.create_operation(
        "arm_sme.intr.st1b.vert",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_st1d_horiz`

"""
function intr_st1d_horiz(
    predicate::Value,
    store_address::Value,
    tile_slice_index::Value;
    tile_id,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[predicate, store_address, tile_slice_index]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[namedattribute("tile_id", tile_id),]

    return IR.create_operation(
        "arm_sme.intr.st1d.horiz",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_st1d_vert`

"""
function intr_st1d_vert(
    predicate::Value,
    store_address::Value,
    tile_slice_index::Value;
    tile_id,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[predicate, store_address, tile_slice_index]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[namedattribute("tile_id", tile_id),]

    return IR.create_operation(
        "arm_sme.intr.st1d.vert",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_st1h_horiz`

"""
function intr_st1h_horiz(
    predicate::Value,
    store_address::Value,
    tile_slice_index::Value;
    tile_id,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[predicate, store_address, tile_slice_index]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[namedattribute("tile_id", tile_id),]

    return IR.create_operation(
        "arm_sme.intr.st1h.horiz",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_st1h_vert`

"""
function intr_st1h_vert(
    predicate::Value,
    store_address::Value,
    tile_slice_index::Value;
    tile_id,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[predicate, store_address, tile_slice_index]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[namedattribute("tile_id", tile_id),]

    return IR.create_operation(
        "arm_sme.intr.st1h.vert",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_st1q_horiz`

"""
function intr_st1q_horiz(
    predicate::Value,
    store_address::Value,
    tile_slice_index::Value;
    tile_id,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[predicate, store_address, tile_slice_index]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[namedattribute("tile_id", tile_id),]

    return IR.create_operation(
        "arm_sme.intr.st1q.horiz",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_st1q_vert`

"""
function intr_st1q_vert(
    predicate::Value,
    store_address::Value,
    tile_slice_index::Value;
    tile_id,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[predicate, store_address, tile_slice_index]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[namedattribute("tile_id", tile_id),]

    return IR.create_operation(
        "arm_sme.intr.st1q.vert",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_st1w_horiz`

"""
function intr_st1w_horiz(
    predicate::Value,
    store_address::Value,
    tile_slice_index::Value;
    tile_id,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[predicate, store_address, tile_slice_index]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[namedattribute("tile_id", tile_id),]

    return IR.create_operation(
        "arm_sme.intr.st1w.horiz",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_st1w_vert`

"""
function intr_st1w_vert(
    predicate::Value,
    store_address::Value,
    tile_slice_index::Value;
    tile_id,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[predicate, store_address, tile_slice_index]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[namedattribute("tile_id", tile_id),]

    return IR.create_operation(
        "arm_sme.intr.st1w.vert",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_str`

"""
function intr_str(index::Value, store_address::Value, offset::Value; location=Location())
    _results = IR.Type[]
    _operands = Value[index, store_address, offset]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "arm_sme.intr.str",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_sumopa_wide`

"""
function intr_sumopa_wide(
    lhs_predicate::Value,
    rhs_predicate::Value,
    lhs_vector::Value,
    rhs_vector::Value;
    tile_id,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[lhs_predicate, rhs_predicate, lhs_vector, rhs_vector]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[namedattribute("tile_id", tile_id),]

    return IR.create_operation(
        "arm_sme.intr.sumopa.wide",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_sumops_wide`

"""
function intr_sumops_wide(
    lhs_predicate::Value,
    rhs_predicate::Value,
    lhs_vector::Value,
    rhs_vector::Value;
    tile_id,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[lhs_predicate, rhs_predicate, lhs_vector, rhs_vector]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[namedattribute("tile_id", tile_id),]

    return IR.create_operation(
        "arm_sme.intr.sumops.wide",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_umopa_wide`

"""
function intr_umopa_wide(
    lhs_predicate::Value,
    rhs_predicate::Value,
    lhs_vector::Value,
    rhs_vector::Value;
    tile_id,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[lhs_predicate, rhs_predicate, lhs_vector, rhs_vector]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[namedattribute("tile_id", tile_id),]

    return IR.create_operation(
        "arm_sme.intr.umopa.wide",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_umopa_za32`

"""
function intr_umopa_za32(
    lhs_predicate::Value,
    rhs_predicate::Value,
    lhs_vector::Value,
    rhs_vector::Value;
    tile_id,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[lhs_predicate, rhs_predicate, lhs_vector, rhs_vector]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[namedattribute("tile_id", tile_id),]

    return IR.create_operation(
        "arm_sme.intr.umopa.za32",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_umops_wide`

"""
function intr_umops_wide(
    lhs_predicate::Value,
    rhs_predicate::Value,
    lhs_vector::Value,
    rhs_vector::Value;
    tile_id,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[lhs_predicate, rhs_predicate, lhs_vector, rhs_vector]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[namedattribute("tile_id", tile_id),]

    return IR.create_operation(
        "arm_sme.intr.umops.wide",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_umops_za32`

"""
function intr_umops_za32(
    lhs_predicate::Value,
    rhs_predicate::Value,
    lhs_vector::Value,
    rhs_vector::Value;
    tile_id,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[lhs_predicate, rhs_predicate, lhs_vector, rhs_vector]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[namedattribute("tile_id", tile_id),]

    return IR.create_operation(
        "arm_sme.intr.umops.za32",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_usmopa_wide`

"""
function intr_usmopa_wide(
    lhs_predicate::Value,
    rhs_predicate::Value,
    lhs_vector::Value,
    rhs_vector::Value;
    tile_id,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[lhs_predicate, rhs_predicate, lhs_vector, rhs_vector]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[namedattribute("tile_id", tile_id),]

    return IR.create_operation(
        "arm_sme.intr.usmopa.wide",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_usmops_wide`

"""
function intr_usmops_wide(
    lhs_predicate::Value,
    rhs_predicate::Value,
    lhs_vector::Value,
    rhs_vector::Value;
    tile_id,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[lhs_predicate, rhs_predicate, lhs_vector, rhs_vector]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[namedattribute("tile_id", tile_id),]

    return IR.create_operation(
        "arm_sme.intr.usmops.wide",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_write_horiz`

"""
function intr_write_horiz(
    tile_slice_index::Value, predicate::Value, vector::Value; tile_id, location=Location()
)
    _results = IR.Type[]
    _operands = Value[tile_slice_index, predicate, vector]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[namedattribute("tile_id", tile_id),]

    return IR.create_operation(
        "arm_sme.intr.write.horiz",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_write_vert`

"""
function intr_write_vert(
    tile_slice_index::Value, predicate::Value, vector::Value; tile_id, location=Location()
)
    _results = IR.Type[]
    _operands = Value[tile_slice_index, predicate, vector]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[namedattribute("tile_id", tile_id),]

    return IR.create_operation(
        "arm_sme.intr.write.vert",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_zero`

"""
function intr_zero(; tile_mask, location=Location())
    _results = IR.Type[]
    _operands = Value[]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[namedattribute("tile_mask", tile_mask),]

    return IR.create_operation(
        "arm_sme.intr.zero",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

end # arm_sme
