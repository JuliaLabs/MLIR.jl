module arm_sme

import ...IR:
    IR, NamedAttribute, Value, Location, Block, Region, Attribute, context, IndexType
import ..Dialects: namedattribute, operandsegmentsizes

"""
`get_tile`

Allocates a new SME \"virtual tile\" within a function. The contents of the
tile returned from this operation are undefined.

Example 1:

```mlir
// Allocate an 8-bit element \"virtual tile\"
%za0_b = arm_sme.get_tile: vector<[16]x[16]xi8>
```

Example 2:

```mlir
// Allocate two 16-bit element \"virtual tiles\"
%za0_h = arm_sme.get_tile : vector<[8]x[8]xi16>
%za1_h = arm_sme.get_tile : vector<[8]x[8]xi16>
```

Example 3:
```mlir
// Allocate an 128-bit element \"virtual tile\"
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
`materialize_ssa_tile`

A placeholder to preserve dataflow while lowering to SME intrinsics (which
do not take or return SME virtual tile values). This operation is intended
to be DCE\'d once all ArmSME operations have been lowered.

This operation is not intended to be used outside of the ArmSME -> LLVM
conversion.
"""
function materialize_ssa_tile(; tile::IR.Type, location=Location())
    _results = IR.Type[tile,]
    _operands = Value[]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "arm_sme.materialize_ssa_tile",
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
`zero`

Initialise ZA with 0. This operation is convenient wrapper for the SME
`zero` intrinsic and instruction.

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
