module x86vector

import ...IR:
    IR, NamedAttribute, Value, Location, Block, Region, Attribute, context, IndexType
import ..Dialects: namedattribute, operandsegmentsizes

"""
`avx_intr_dp_ps_256`

"""
function avx_intr_dp_ps_256(
    a::Value, b::Value, c::Value; res=nothing::Union{Nothing,IR.Type}, location=Location()
)
    _results = IR.Type[]
    _operands = Value[a, b, c]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(res) && push!(_results, res)

    return IR.create_operation(
        "x86vector.avx.intr.dp.ps.256",
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
`avx_intr_dot`

Computes the 4-way dot products of the lower and higher parts of the source
vectors and broadcasts the two results to the lower and higher elements of
the destination vector, respectively. Adding one element of the lower part
to one element of the higher part in the destination vector yields the full
dot product of the two source vectors.

# Example

```mlir
%0 = x86vector.avx.intr.dot %a, %b : vector<8xf32>
%1 = vector.extractelement %0[%i0 : i32]: vector<8xf32>
%2 = vector.extractelement %0[%i4 : i32]: vector<8xf32>
%d = arith.addf %1, %2 : f32
```
"""
function avx_intr_dot(
    a::Value, b::Value; res=nothing::Union{Nothing,IR.Type}, location=Location()
)
    _results = IR.Type[]
    _operands = Value[a, b]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(res) && push!(_results, res)

    return IR.create_operation(
        "x86vector.avx.intr.dot",
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
`avx512_intr_mask_compress`

"""
function avx512_intr_mask_compress(
    a::Value, src::Value, k::Value; res=nothing::Union{Nothing,IR.Type}, location=Location()
)
    _results = IR.Type[]
    _operands = Value[a, src, k]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(res) && push!(_results, res)

    return IR.create_operation(
        "x86vector.avx512.intr.mask.compress",
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
`avx512_mask_compress`

The mask.compress op is an AVX512 specific op that can lower to the
`llvm.mask.compress` instruction. Instead of `src`, a constant vector
vector attribute `constant_src` may be specified. If neither `src` nor
`constant_src` is specified, the remaining elements in the result vector are
set to zero.

#### From the Intel Intrinsics Guide:

Contiguously store the active integer/floating-point elements in `a` (those
with their respective bit set in writemask `k`) to `dst`, and pass through the
remaining elements from `src`.
"""
function avx512_mask_compress(
    k::Value,
    a::Value,
    src=nothing::Union{Nothing,Value};
    dst=nothing::Union{Nothing,IR.Type},
    constant_src=nothing,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[k, a]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(src) && push!(_operands, src)
    !isnothing(dst) && push!(_results, dst)
    !isnothing(constant_src) &&
        push!(_attributes, namedattribute("constant_src", constant_src))

    return IR.create_operation(
        "x86vector.avx512.mask.compress",
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
`avx512_mask_rndscale`

The mask.rndscale op is an AVX512 specific op that can lower to the proper
LLVMAVX512 operation: `llvm.mask.rndscale.ps.512` or
`llvm.mask.rndscale.pd.512` instruction depending on the type of vectors it
is applied to.

#### From the Intel Intrinsics Guide:

Round packed floating-point elements in `a` to the number of fraction bits
specified by `imm`, and store the results in `dst` using writemask `k`
(elements are copied from src when the corresponding mask bit is not set).
"""
function avx512_mask_rndscale(
    src::Value,
    k::Value,
    a::Value,
    imm::Value,
    rounding::Value;
    dst=nothing::Union{Nothing,IR.Type},
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[src, k, a, imm, rounding]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(dst) && push!(_results, dst)

    return IR.create_operation(
        "x86vector.avx512.mask.rndscale",
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
`avx512_intr_mask_rndscale_pd_512`

"""
function avx512_intr_mask_rndscale_pd_512(
    src::Value,
    k::Value,
    a::Value,
    imm::Value,
    rounding::Value;
    res=nothing::Union{Nothing,IR.Type},
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[src, k, a, imm, rounding]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(res) && push!(_results, res)

    return IR.create_operation(
        "x86vector.avx512.intr.mask.rndscale.pd.512",
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
`avx512_intr_mask_rndscale_ps_512`

"""
function avx512_intr_mask_rndscale_ps_512(
    src::Value,
    k::Value,
    a::Value,
    imm::Value,
    rounding::Value;
    res=nothing::Union{Nothing,IR.Type},
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[src, k, a, imm, rounding]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(res) && push!(_results, res)

    return IR.create_operation(
        "x86vector.avx512.intr.mask.rndscale.ps.512",
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
`avx512_mask_scalef`

The `mask.scalef` op is an AVX512 specific op that can lower to the proper
LLVMAVX512 operation: `llvm.mask.scalef.ps.512` or
`llvm.mask.scalef.pd.512` depending on the type of MLIR vectors it is
applied to.

#### From the Intel Intrinsics Guide:

Scale the packed floating-point elements in `a` using values from `b`, and
store the results in `dst` using writemask `k` (elements are copied from src
when the corresponding mask bit is not set).
"""
function avx512_mask_scalef(
    src::Value,
    a::Value,
    b::Value,
    k::Value,
    rounding::Value;
    dst=nothing::Union{Nothing,IR.Type},
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[src, a, b, k, rounding]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(dst) && push!(_results, dst)

    return IR.create_operation(
        "x86vector.avx512.mask.scalef",
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
`avx512_intr_mask_scalef_pd_512`

"""
function avx512_intr_mask_scalef_pd_512(
    src::Value,
    a::Value,
    b::Value,
    k::Value,
    rounding::Value;
    res=nothing::Union{Nothing,IR.Type},
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[src, a, b, k, rounding]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(res) && push!(_results, res)

    return IR.create_operation(
        "x86vector.avx512.intr.mask.scalef.pd.512",
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
`avx512_intr_mask_scalef_ps_512`

"""
function avx512_intr_mask_scalef_ps_512(
    src::Value,
    a::Value,
    b::Value,
    k::Value,
    rounding::Value;
    res=nothing::Union{Nothing,IR.Type},
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[src, a, b, k, rounding]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(res) && push!(_results, res)

    return IR.create_operation(
        "x86vector.avx512.intr.mask.scalef.ps.512",
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
`avx_intr_rsqrt_ps_256`

"""
function avx_intr_rsqrt_ps_256(
    a::Value; res=nothing::Union{Nothing,IR.Type}, location=Location()
)
    _results = IR.Type[]
    _operands = Value[a,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(res) && push!(_results, res)

    return IR.create_operation(
        "x86vector.avx.intr.rsqrt.ps.256",
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
`avx_rsqrt`

"""
function avx_rsqrt(a::Value; b=nothing::Union{Nothing,IR.Type}, location=Location())
    _results = IR.Type[]
    _operands = Value[a,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(b) && push!(_results, b)

    return IR.create_operation(
        "x86vector.avx.rsqrt",
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
`avx512_intr_vp2intersect_d_512`

"""
function avx512_intr_vp2intersect_d_512(
    a::Value, b::Value; res::IR.Type, location=Location()
)
    _results = IR.Type[res,]
    _operands = Value[a, b]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "x86vector.avx512.intr.vp2intersect.d.512",
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
`avx512_vp2intersect`

The `vp2intersect` op is an AVX512 specific op that can lower to the proper
LLVMAVX512 operation: `llvm.vp2intersect.d.512` or
`llvm.vp2intersect.q.512` depending on the type of MLIR vectors it is
applied to.

#### From the Intel Intrinsics Guide:

Compute intersection of packed integer vectors `a` and `b`, and store
indication of match in the corresponding bit of two mask registers
specified by `k1` and `k2`. A match in corresponding elements of `a` and
`b` is indicated by a set bit in the corresponding bit of the mask
registers.
"""
function avx512_vp2intersect(
    a::Value,
    b::Value;
    k1=nothing::Union{Nothing,IR.Type},
    k2=nothing::Union{Nothing,IR.Type},
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[a, b]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(k1) && push!(_results, k1)
    !isnothing(k2) && push!(_results, k2)

    return IR.create_operation(
        "x86vector.avx512.vp2intersect",
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
`avx512_intr_vp2intersect_q_512`

"""
function avx512_intr_vp2intersect_q_512(
    a::Value, b::Value; res::IR.Type, location=Location()
)
    _results = IR.Type[res,]
    _operands = Value[a, b]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "x86vector.avx512.intr.vp2intersect.q.512",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

end # x86vector
