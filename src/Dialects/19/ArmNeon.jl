module arm_neon

import ...IR:
    IR, NamedAttribute, Value, Location, Block, Region, Attribute, context, IndexType
import ..Dialects: namedattribute, operandsegmentsizes

"""
`intr_smull`

Signed Multiply Long (vector). This instruction multiplies corresponding
signed integer values in the lower or upper half of the vectors of the two
source SIMD&FP registers, places the results in a vector, and writes the
vector to the destination SIMD&FP register.

Source:
https://developer.arm.com/architectures/instruction-sets/simd-isas/neon/intrinsics
"""
function intr_smull(a::Value, b::Value; res::IR.Type, location=Location())
    _results = IR.Type[res,]
    _operands = Value[a, b]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "arm_neon.intr.smull",
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
`_2d_sdot`

The two input vectors `b` and `c` have a 2D shape, consisting of either 2
or 4 rows, each row having length 4. This operation computes the pair-wise
dot-products of the rows of `b` and `c` and accumulates them with the
corresponding entry of `a`:

```
res[i] := a[i] + dot_product(b[i, ...], c[i, ...])
```
"""
function _2d_sdot(a::Value, b::Value, c::Value; res::IR.Type, location=Location())
    _results = IR.Type[res,]
    _operands = Value[a, b, c]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "arm_neon.2d.sdot",
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
`intr_sdot`

Signed integer addition of dot product (vector). This instruction performs
the following operation on signed integer vectors: res = dot(b, c) + a,
where vector operands are partitioned into groups of four elements.

Source:
https://developer.arm.com/architectures/instruction-sets/simd-isas/neon/intrinsics
"""
function intr_sdot(a::Value, b::Value, c::Value; res::IR.Type, location=Location())
    _results = IR.Type[res,]
    _operands = Value[a, b, c]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "arm_neon.intr.sdot",
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
`intr_smmla`

SMMLA: Signed integer matrix multiply-accumulate.

Signed 8-bit integer matrix multiply-accumulate. This instruction multiplies
the 2x8 matrix of signed 8-bit integer values in the first source vector by
the 8x2 matrix of signed 8-bit integer values in the second source vector.
The resulting 2x2 32-bit integer matrix product is destructively added to
the 32-bit integer matrix accumulator in the destination vector. This is
equivalent to performing an 8-way dot product per destination element.

Source:
https://developer.arm.com/architectures/instruction-sets/intrinsics/#f:@navigationhierarchiessimdisa=[Neon]&q=smmla
"""
function intr_smmla(acc::Value, src1::Value, src2::Value; res::IR.Type, location=Location())
    _results = IR.Type[res,]
    _operands = Value[acc, src1, src2]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "arm_neon.intr.smmla",
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
`intr_ummla`

UMMLA: Signed integer matrix multiply-accumulate.

Unsigned 8-bit integer matrix multiply-accumulate. This instruction
multiplies the 2x8 matrix of unsigned 8-bit integer values in the first
source vector by the 8x2 matrix of unsigned 8-bit integer values in the
second source vector. The resulting 2x2 32-bit integer matrix product is
destructively added to the 32-bit integer matrix accumulator in the
destination vector. This is equivalent to performing an 8-way dot product
per destination element.

Source:
https://developer.arm.com/architectures/instruction-sets/intrinsics/#f:@navigationhierarchiessimdisa=[Neon]&q=ummla
"""
function intr_ummla(acc::Value, src1::Value, src2::Value; res::IR.Type, location=Location())
    _results = IR.Type[res,]
    _operands = Value[acc, src1, src2]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "arm_neon.intr.ummla",
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
`intr_usmmla`

USMMLA: Signed integer matrix multiply-accumulate.

Unsigned and signed 8-bit integer matrix multiply-accumulate. This
instruction multiplies the 2x8 matrix of unsigned 8-bit integer values in
the first source vector by the 8x2 matrix of signed 8-bit integer values in
the second source vector. The resulting 2x2 32-bit integer matrix product is
destructively added to the 32-bit integer matrix accumulator in the
destination vector. This is equivalent to performing an 8-way dot product
 per destination element.


Source:
https://developer.arm.com/architectures/instruction-sets/intrinsics/#f:@navigationhierarchiessimdisa=[Neon]&q=usmmla
"""
function intr_usmmla(
    acc::Value, src1::Value, src2::Value; res::IR.Type, location=Location()
)
    _results = IR.Type[res,]
    _operands = Value[acc, src1, src2]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "arm_neon.intr.usmmla",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

end # arm_neon
