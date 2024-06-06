module arm_sme

import ...IR: IR, NamedAttribute, Value, Location, Block, Region, Attribute, context, IndexType
import ..Dialects: namedattribute, operandsegmentsizes


"""
`cast_tile_to_vector`

A `cast_tile_to_vector` operation does a cast from a tile id to a 2-d
scalable vector type, which represents an SME \"virtual tile\". This would
normally be used when lowering operations that return \"virtual tile\" vector
types to model the output. This is required to preserve dataflow as SME
intrinsics have no return values.

# Example

Input:
```mlir
%tile = vector.load %mem1[%c0] : memref<?xi32>, vector<[4]x[4]xi32>
vector.store %tile, %mem2[%c0] : memref<?xi32>, vector<[4]x[4]xi32>
```

After lowering `vector.load`:
```mlir
%tile_id = arm_sme.get_tile_id : i32
scf.for %vnum = %c0 to %num_vectors step %c1 {
  // ...
  \"arm_sme.intr.ld1w.horiz\"(%pg, %ptr, %tile_id, %vnum) : (vector<[4]xi1>, !llvm.ptr, i32, i32) -> ()
}
%tile = arm_sme.cast_tile_to_vector %tile_id : i32 to vector<[4]x[4]xi32>
vector.store %tile, %mem2[%c0] : memref<?xi32>, vector<[4]x[4]xi32>
```

In the example above, the `vector.load` can\'t be replaced with an SME
intrinsic that has no outputs since it is used by the `vector.store`.
However, by inserting a `cast_tile_to_vector` op after the load intrinsics
the `vector.load` can be replaced. This enables \"local\" rewrites on
individual vector ops, rather than \"global\" rewrites that would have to
look at the vector op uses and also lower them.

Canonicalization will look through `arm_sme.cast_tile_to_vector` and fold
the cast away if it comes from a `arm_sme.cast_vector_to_tile`.
"""
function cast_tile_to_vector(tile_id::Value; vector::IR.Type, location=Location())
    results = IR.Type[vector, ]
    operands = Value[tile_id, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "arm_sme.cast_tile_to_vector", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`cast_vector_to_tile`

A `cast_vector_to_tile` operation does a cast from a 2-d scalable vector
type, which represents an SME \"virtual tile\", to a tile id. This is
required to preserve dataflow as the SME intrinsics have no return values.

# Example

Input:
```mlir
%tile = vector.load %mem1[%c0] : memref<?xi32>, vector<[4]x[4]xi32>
vector.store %tile, %mem2[%c0] : memref<?xi32>, vector<[4]x[4]xi32>
```

After lowering `vector.store`:
```mlir
%tile = vector.load %mem1[%c0] : memref<?xi32>, vector<[4]x[4]xi32>
scf.for %vnum = %c0 to %num_vectors step %c1 {
  // ...
  %tile_id = arm_sme.cast_vector_to_tile %tile : (vector<[4]x[4]xi32>) -> i32
  \"arm_sme.intr.st1w.horiz\"(%pg, %ptr, %tile_id, %vnum) : (vector<[4]xi1>, !llvm.ptr, i32, i32) -> ()
}
```

Canonicalization will look through `arm_sme.cast_vector_to_tile` and fold
the cast away if it comes from a `arm_sme.cast_tile_to_vector`.
"""
function cast_vector_to_tile(vector::Value; tile_id::IR.Type, location=Location())
    results = IR.Type[tile_id, ]
    operands = Value[vector, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "arm_sme.cast_vector_to_tile", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`get_tile_id`

A `get_tile_id` operation returns a scalar integer representing an SME
\"virtual tile\" id. The bitwidth of the scalar indicates the element
bitwidth of the \"virtual tile\".

The scope of a tile id is a function and cannot be passed or returned from
functions.

# Example
```mlir
// Allocate and return an 8-bit element \"virtual tile\" id
%za0_b = arm_sme.get_tile_id : i8
```

# Example
```
// Allocate and return two 16-bit element \"virtual tile\" ids
%za0_h = arm_sme.get_tile_id : i16
%za1_h = arm_sme.get_tile_id : i16
```

# Example
```
// Allocate and return an 128-bit element \"virtual tile\" id
%za0_q = arm_sme.get_tile_id : i128
```
"""
function get_tile_id(; tile_id::IR.Type, location=Location())
    results = IR.Type[tile_id, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "arm_sme.get_tile_id", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`intr_ld1b_horiz`

"""
function intr_ld1b_horiz(operand_0::Value, operand_1::Value, operand_2::Value, operand_3::Value; location=Location())
    results = IR.Type[]
    operands = Value[operand_0, operand_1, operand_2, operand_3, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "arm_sme.intr.ld1b.horiz", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`intr_ld1b_vert`

"""
function intr_ld1b_vert(operand_0::Value, operand_1::Value, operand_2::Value, operand_3::Value; location=Location())
    results = IR.Type[]
    operands = Value[operand_0, operand_1, operand_2, operand_3, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "arm_sme.intr.ld1b.vert", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`intr_ld1d_horiz`

"""
function intr_ld1d_horiz(operand_0::Value, operand_1::Value, operand_2::Value, operand_3::Value; location=Location())
    results = IR.Type[]
    operands = Value[operand_0, operand_1, operand_2, operand_3, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "arm_sme.intr.ld1d.horiz", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`intr_ld1d_vert`

"""
function intr_ld1d_vert(operand_0::Value, operand_1::Value, operand_2::Value, operand_3::Value; location=Location())
    results = IR.Type[]
    operands = Value[operand_0, operand_1, operand_2, operand_3, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "arm_sme.intr.ld1d.vert", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`intr_ld1h_horiz`

"""
function intr_ld1h_horiz(operand_0::Value, operand_1::Value, operand_2::Value, operand_3::Value; location=Location())
    results = IR.Type[]
    operands = Value[operand_0, operand_1, operand_2, operand_3, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "arm_sme.intr.ld1h.horiz", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`intr_ld1h_vert`

"""
function intr_ld1h_vert(operand_0::Value, operand_1::Value, operand_2::Value, operand_3::Value; location=Location())
    results = IR.Type[]
    operands = Value[operand_0, operand_1, operand_2, operand_3, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "arm_sme.intr.ld1h.vert", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`intr_ld1q_horiz`

"""
function intr_ld1q_horiz(operand_0::Value, operand_1::Value, operand_2::Value, operand_3::Value; location=Location())
    results = IR.Type[]
    operands = Value[operand_0, operand_1, operand_2, operand_3, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "arm_sme.intr.ld1q.horiz", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`intr_ld1q_vert`

"""
function intr_ld1q_vert(operand_0::Value, operand_1::Value, operand_2::Value, operand_3::Value; location=Location())
    results = IR.Type[]
    operands = Value[operand_0, operand_1, operand_2, operand_3, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "arm_sme.intr.ld1q.vert", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`intr_ld1w_horiz`

"""
function intr_ld1w_horiz(operand_0::Value, operand_1::Value, operand_2::Value, operand_3::Value; location=Location())
    results = IR.Type[]
    operands = Value[operand_0, operand_1, operand_2, operand_3, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "arm_sme.intr.ld1w.horiz", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`intr_ld1w_vert`

"""
function intr_ld1w_vert(operand_0::Value, operand_1::Value, operand_2::Value, operand_3::Value; location=Location())
    results = IR.Type[]
    operands = Value[operand_0, operand_1, operand_2, operand_3, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "arm_sme.intr.ld1w.vert", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`intr_mopa`

"""
function intr_mopa(operand_0::Value, operand_1::Value, operand_2::Value, operand_3::Value, operand_4::Value; location=Location())
    results = IR.Type[]
    operands = Value[operand_0, operand_1, operand_2, operand_3, operand_4, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "arm_sme.intr.mopa", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`intr_mopa_wide`

"""
function intr_mopa_wide(operand_0::Value, operand_1::Value, operand_2::Value, operand_3::Value, operand_4::Value; location=Location())
    results = IR.Type[]
    operands = Value[operand_0, operand_1, operand_2, operand_3, operand_4, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "arm_sme.intr.mopa.wide", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`intr_mops`

"""
function intr_mops(operand_0::Value, operand_1::Value, operand_2::Value, operand_3::Value, operand_4::Value; location=Location())
    results = IR.Type[]
    operands = Value[operand_0, operand_1, operand_2, operand_3, operand_4, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "arm_sme.intr.mops", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`intr_mops_wide`

"""
function intr_mops_wide(operand_0::Value, operand_1::Value, operand_2::Value, operand_3::Value, operand_4::Value; location=Location())
    results = IR.Type[]
    operands = Value[operand_0, operand_1, operand_2, operand_3, operand_4, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "arm_sme.intr.mops.wide", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`intr_smopa_wide`

"""
function intr_smopa_wide(operand_0::Value, operand_1::Value, operand_2::Value, operand_3::Value, operand_4::Value; location=Location())
    results = IR.Type[]
    operands = Value[operand_0, operand_1, operand_2, operand_3, operand_4, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "arm_sme.intr.smopa.wide", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`intr_smops_wide`

"""
function intr_smops_wide(operand_0::Value, operand_1::Value, operand_2::Value, operand_3::Value, operand_4::Value; location=Location())
    results = IR.Type[]
    operands = Value[operand_0, operand_1, operand_2, operand_3, operand_4, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "arm_sme.intr.smops.wide", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`intr_st1b_horiz`

"""
function intr_st1b_horiz(operand_0::Value, operand_1::Value, operand_2::Value, operand_3::Value; location=Location())
    results = IR.Type[]
    operands = Value[operand_0, operand_1, operand_2, operand_3, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "arm_sme.intr.st1b.horiz", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`intr_st1b_vert`

"""
function intr_st1b_vert(operand_0::Value, operand_1::Value, operand_2::Value, operand_3::Value; location=Location())
    results = IR.Type[]
    operands = Value[operand_0, operand_1, operand_2, operand_3, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "arm_sme.intr.st1b.vert", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`intr_st1d_horiz`

"""
function intr_st1d_horiz(operand_0::Value, operand_1::Value, operand_2::Value, operand_3::Value; location=Location())
    results = IR.Type[]
    operands = Value[operand_0, operand_1, operand_2, operand_3, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "arm_sme.intr.st1d.horiz", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`intr_st1d_vert`

"""
function intr_st1d_vert(operand_0::Value, operand_1::Value, operand_2::Value, operand_3::Value; location=Location())
    results = IR.Type[]
    operands = Value[operand_0, operand_1, operand_2, operand_3, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "arm_sme.intr.st1d.vert", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`intr_st1h_horiz`

"""
function intr_st1h_horiz(operand_0::Value, operand_1::Value, operand_2::Value, operand_3::Value; location=Location())
    results = IR.Type[]
    operands = Value[operand_0, operand_1, operand_2, operand_3, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "arm_sme.intr.st1h.horiz", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`intr_st1h_vert`

"""
function intr_st1h_vert(operand_0::Value, operand_1::Value, operand_2::Value, operand_3::Value; location=Location())
    results = IR.Type[]
    operands = Value[operand_0, operand_1, operand_2, operand_3, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "arm_sme.intr.st1h.vert", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`intr_st1q_horiz`

"""
function intr_st1q_horiz(operand_0::Value, operand_1::Value, operand_2::Value, operand_3::Value; location=Location())
    results = IR.Type[]
    operands = Value[operand_0, operand_1, operand_2, operand_3, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "arm_sme.intr.st1q.horiz", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`intr_st1q_vert`

"""
function intr_st1q_vert(operand_0::Value, operand_1::Value, operand_2::Value, operand_3::Value; location=Location())
    results = IR.Type[]
    operands = Value[operand_0, operand_1, operand_2, operand_3, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "arm_sme.intr.st1q.vert", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`intr_st1w_horiz`

"""
function intr_st1w_horiz(operand_0::Value, operand_1::Value, operand_2::Value, operand_3::Value; location=Location())
    results = IR.Type[]
    operands = Value[operand_0, operand_1, operand_2, operand_3, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "arm_sme.intr.st1w.horiz", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`intr_st1w_vert`

"""
function intr_st1w_vert(operand_0::Value, operand_1::Value, operand_2::Value, operand_3::Value; location=Location())
    results = IR.Type[]
    operands = Value[operand_0, operand_1, operand_2, operand_3, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "arm_sme.intr.st1w.vert", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`intr_str`

"""
function intr_str(operand_0::Value, operand_1::Value; location=Location())
    results = IR.Type[]
    operands = Value[operand_0, operand_1, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "arm_sme.intr.str", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`intr_sumopa_wide`

"""
function intr_sumopa_wide(operand_0::Value, operand_1::Value, operand_2::Value, operand_3::Value, operand_4::Value; location=Location())
    results = IR.Type[]
    operands = Value[operand_0, operand_1, operand_2, operand_3, operand_4, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "arm_sme.intr.sumopa.wide", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`intr_sumops_wide`

"""
function intr_sumops_wide(operand_0::Value, operand_1::Value, operand_2::Value, operand_3::Value, operand_4::Value; location=Location())
    results = IR.Type[]
    operands = Value[operand_0, operand_1, operand_2, operand_3, operand_4, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "arm_sme.intr.sumops.wide", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`intr_umopa_wide`

"""
function intr_umopa_wide(operand_0::Value, operand_1::Value, operand_2::Value, operand_3::Value, operand_4::Value; location=Location())
    results = IR.Type[]
    operands = Value[operand_0, operand_1, operand_2, operand_3, operand_4, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "arm_sme.intr.umopa.wide", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`intr_umops_wide`

"""
function intr_umops_wide(operand_0::Value, operand_1::Value, operand_2::Value, operand_3::Value, operand_4::Value; location=Location())
    results = IR.Type[]
    operands = Value[operand_0, operand_1, operand_2, operand_3, operand_4, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "arm_sme.intr.umops.wide", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`intr_usmopa_wide`

"""
function intr_usmopa_wide(operand_0::Value, operand_1::Value, operand_2::Value, operand_3::Value, operand_4::Value; location=Location())
    results = IR.Type[]
    operands = Value[operand_0, operand_1, operand_2, operand_3, operand_4, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "arm_sme.intr.usmopa.wide", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`intr_usmops_wide`

"""
function intr_usmops_wide(operand_0::Value, operand_1::Value, operand_2::Value, operand_3::Value, operand_4::Value; location=Location())
    results = IR.Type[]
    operands = Value[operand_0, operand_1, operand_2, operand_3, operand_4, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "arm_sme.intr.usmops.wide", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`intr_za_disable`

"""
function intr_za_disable(; location=Location())
    results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "arm_sme.intr.za.disable", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`intr_za_enable`

"""
function intr_za_enable(; location=Location())
    results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "arm_sme.intr.za.enable", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`intr_zero`

"""
function intr_zero(operand_0::Value; location=Location())
    results = IR.Type[]
    operands = Value[operand_0, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "arm_sme.intr.zero", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`tile_load`

Loads a 2D SME \"virtual tile\" from memory defined by a base and indices,
with the shape defined by the 2D scalable vector type of the result tile.
The slice of memory must be contiguous. The memref must be either rank 1 or
rank 2 with dynamic dimensions, since the operation is scalable, and the
element type must be a scalar that matches the element type of the result.

Example 1: Load an 8-bit element ZA tile from memory (ZA0.B).
```mlir
%tile = arm_sme.tile_load %base[%c0, %c0] : memref<?x?xi8>, vector<[16]x[16]xi8>
```

Example 2: Load a FP 32-bit element ZA tile from memory.
```mlir
%tile = arm_sme.tile_load %base[%c0, %c0] : memref<?x?xf32>, vector<[4]x[4]xf32>
```

Example 3: Load a 128-bit element ZA tile from memory.
```mlir
%tile = arm_sme.tile_load %base[%c0, %c0] : memref<?x?xi128>, vector<[1]x[1]xi128>
```
"""
function tile_load(base::Value, indices::Vector{Value}; result::IR.Type, location=Location())
    results = IR.Type[result, ]
    operands = Value[base, indices..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "arm_sme.tile_load", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`tile_store`

Stores a 2D SME \"virtual tile\" to memory defined by a base and indices,
with the shape defined by the 2D scalable vector type of the tile being
stored. The slice of memory must be contiguous. The memref must be either
rank 1 or rank 2 with dynamic dimensions, since the operation is scalable,
and the element type must be a scalar that matches the element type of the
result.

Example 1: Store an 8-bit element ZA tile to memory (ZA0.B).
```mlir
arm_sme.tile_store %tile, %base[%c0, %c0] : vector<[16]x[16]xi8>, memref<?x?xi8>
```

Example 2: Store a FP 32-bit element ZA tile to memory.
```mlir
arm_sme.tile_store %tile, %base[%c0, %c0] : vector<[4]x[4]xf32>, memref<?x?xf32>
```

Example 3: Store a 128-bit element ZA tile to memory.
```mlir
arm_sme.tile_store %tile, %base[%c0, %c0] : vector<[1]x[1]xi128>, memref<?x?xi128>
```
"""
function tile_store(valueToStore::Value, base::Value, indices::Vector{Value}; location=Location())
    results = IR.Type[]
    operands = Value[valueToStore, base, indices..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "arm_sme.tile_store", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`zero`

Initialise ZA with 0. This operation is convenient wrapper for the SME
`zero` intrinsic and instruction. 

NOTE: At the moment it is assumed that the element type is `i8` and that
there\'s only one \"virtual tile\".

# Example

```mlir
%0 = arm_sme.zero : vector<[16]x[16]xi8>
```
"""
function zero(; res::IR.Type, location=Location())
    results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "arm_sme.zero", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

end # arm_sme
