module xegpu

import ...IR:
    IR, NamedAttribute, Value, Location, Block, Region, Attribute, context, IndexType
import ..Dialects: namedattribute, operandsegmentsizes

"""
`alloc_nbarrier`
AllocNbarrier is to create a set of named barriers as
  specified by `nbarrier_num`. Named barriers are workgroup level resources,
    and are shared by all threads in the workgroup. For example, there are
    up to 32 barriers (range 0-31) for each XeCore on PVC. A typical use case
    is that a workgroup is partitioned into N subgroups of threads (N <= 32),
    and each subgroup coordinating their work with a separate barrier with id
    range from 0 to N respectively.
"""
function alloc_nbarrier(; nbarrier_num, location=Location())
    _results = IR.Type[]
    _operands = Value[]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[namedattribute("nbarrier_num", nbarrier_num),]

    return IR.create_operation(
        "xegpu.alloc_nbarrier",
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
`atomic_rmw`

The `xegpu.atomic_rmw` operation provides a way to perform a read-modify-write
operation on the region described by the `TensorDesc` free from data races. The
`kind` enumeration specifies the modification to be performed, The `mask` operand
has the same shape with `TensorDesc`, and is used to enable or disable specific
data points of the `TensorDesc`. The `value` operand represents the new value to
be applied during the modification.
"""
function atomic_rmw(
    tensorDesc::Value, mask::Value, value::Value; result::IR.Type, kind, location=Location()
)
    _results = IR.Type[result,]
    _operands = Value[tensorDesc, mask, value]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[namedattribute("kind", kind),]

    return IR.create_operation(
        "xegpu.atomic_rmw",
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
`create_tdesc`

\"create_tdesc\" is similar to \"create_nd_tdesc\" in terms that it creates
a Tensor Descriptor (TensorDescType) for a memory region. While \"create_nd_tdesc\"
is for creating continuous subviews, \"create_tdesc\" is for creating non-continuous
(scattered) subviews, allowing each work-item in a subgroup specifying their own offset.
It accepts the following parameters:

* source: a 1D memref or pointer (uint64_t) represents the flattened memory object.
* offsets: a array containing offsets of each access point. Its size
  is fixed to the hardware supportted subgroup size, e.g., 16 on PVC,
  implying each element in the array corresponds to a work-item (SIMT lane)
  in the subgroup.
* chunk_size: [optional attribute] indicates number of continious
  elements accessed for each offset, default is 1.

Example 1. It assumes subgroup size is 4, and accesses a[0], a[16], a[32], a[64]
```
%a = memref.alloc() : memref<1024xf32>
%1 = xegpu.create_tdesc %a[0, 16, 32, 64]: memref<1024xf32> -> TensorDesc<4xf32>
```

Example 2. It assumes subgroup size is 4, and each workitem access 8 elements.
           It will access totally 32 data elements: a[0:7], a[16:23], a[32:39], a[64:71]
```
%0 = memref.alloc() : memref<1024xf32>
%1 = xegpu.create_tdesc %0[0, 16, 32, 64] {chunk_size = 8}: memref<1024xf32> -> TensorDesc<4x8xf32>
```

Example 3. It is similar to Example 2, but there is some overlaps among workitems.
           It accesses: a[0:7], a[4:11], a[8:15], a[12:19]
```
%0 = memref.alloc() : memref<1024xf32>
%1 = xegpu.create_tdesc %0[0, 4, 8, 12] {chunk_size = 8}: memref<1024xf32> -> TensorDesc<4x8xf32>
```
"""
function create_tdesc(
    source::Value,
    offsets::Vector{Value};
    TensorDesc::IR.Type,
    const_offsets,
    chunk_size=nothing,
    location=Location(),
)
    _results = IR.Type[TensorDesc,]
    _operands = Value[source, offsets...]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[namedattribute("const_offsets", const_offsets),]
    !isnothing(chunk_size) && push!(_attributes, namedattribute("chunk_size", chunk_size))

    return IR.create_operation(
        "xegpu.create_tdesc",
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
`create_nd_tdesc`

The \"create_nd_tdesc\" operation creates a TensorDescType which represents
a sub-view of a 2D memory region (It can be extended to support n-D memory
region if needed in future). Elements in the subview continuous in each
dimension. It encodes the following important information for supporting
Intel hardware features:

* source: an object representing (starting address/pointer of) a 2D memory region.
    It can be either a 2D memref object, or simply a pointer represented by uint64_t type.
    for the later case, the shape and layout information of the 2D memory region should
    be explicitly passed via `shape` and `strides` parameters.
* offsets: two index values represents offsets from the \"source\" at the each dimension
    at which the subview of the target memory will be created. It is encoded via two
    variables, including \"offsets\" and \"const_offsets\", such that it can
    accept various forms, such as, operands (e.g., [%c0, %c]) and attributes (e.g., [2, 4]).
* shape: the shape information of the memory region pointed by the \"source\".  It is
    typically encoded via the MemRefType of the source, e.g., memref<4096x4096xf16>.
    But if \"source\" is simply a pointer represented as uint64_t type, or a memref
    type without shape information e.g., memref<?x?xf16>, the shape information has
    to be explicitly passed via the \"shape\" and \"const_shape\" arguments.
* strides: the strides of the memory region pointed by the \"source\". Similar to shape,
    it is typically encoded via the MemRefType of the source too. But if \"source\" is
    simply a pointer represented as uint64_t type, or a memref type without shape
    information e.g., memref<?x?xf16>, the strides information has to be explicitly
    passed via the \"strides\" and \"const_strides\" argument.

Example 1 (suppose the tensor shape inferred by the compiler is 8x16):
%0 = memref.alloc() : memref<1024x1024xf32>
%c0 = arith.constant 0 : index
%c1 = arith.constant 1 : index
%1 = xegpu.create_nd_tdesc %0[%c0, %c0]: memref<1024x1024xf32> -> TensorDesc<8x16xf32>

Example 2 (suppose the tensor shape inferred by the compiler is 8x16):
%0 = memref.alloc(%h, %w) : memref<?x?xf32>
%c0 = arith.constant 0 : index
%c1 = arith.constant 1 : index
%1 = xegpu.create_nd_tdesc %0[%c0, %c0], [%h, %w], [%w, %c1]: memref<?x?xf32> -> TensorDesc<8x16xf32>

Example 3 (suppose the tensor shape inferred by the compiler is 8x16):
%0 = ... : ui64
%c0 = arith.constant 0 : index
%c1 = arith.constant 1 : index
%1 = xegpu.create_nd_tdesc %0[%c0, %c0], [%h, %w], [%w, %c1]: ui64 -> TensorDesc<8x16xf32>
"""
function create_nd_tdesc(
    source::Value,
    offsets::Vector{Value},
    shape::Vector{Value},
    strides::Vector{Value};
    TensorDesc::IR.Type,
    const_offsets,
    const_shape=nothing,
    const_strides=nothing,
    location=Location(),
)
    _results = IR.Type[TensorDesc,]
    _operands = Value[source, offsets..., shape..., strides...]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[namedattribute("const_offsets", const_offsets),]
    push!(
        _attributes,
        operandsegmentsizes([1, length(offsets), length(shape), length(strides)]),
    )
    !isnothing(const_shape) &&
        push!(_attributes, namedattribute("const_shape", const_shape))
    !isnothing(const_strides) &&
        push!(_attributes, namedattribute("const_strides", const_strides))

    return IR.create_operation(
        "xegpu.create_nd_tdesc",
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
`dpas`
DPAS performs matrix multiplication on matrix A of `mxk`
    size, B of `kxn` size, and accumulate on matrix C of `mxn` to the same size
    matrix , `m=8`, `n=16` and `k=8 * 32/bit_width_of_elem_type`. So for fp16
    data type, the matrices are `A: vector<8x16xf16>`, `B: vector<16x16xf16>`,
    and `C/D: vector<8x16xf32>`. Besides the matrix size requirements, DPAS
    also requires A and B to be loaded with the required data layout. Specially,
    VNNI layout is required for B operand. It is achieved via setting `vnni_axis = 0`
    of the corresponding `load_nd` operator. To keep both operands as 3D vector,
    operand A is loaded via setting `vnni_axis = 1` without impacting the
    physical layouts change in register. Due to the VNNI transformation, A and B operands
    are represented as 3D vector, with the last dimension representing the VNNI factor,
    which is computed as `32/bit_width_of_elem_type`. Therefore, `A: vector<8x16xf16>`
    is represented as `A: vector<8x8x2xf16>`, and `B: vector<16x16xf16>` is
    represented as `B: vector<8x16x2xf16>`.

    Note: on PVC, the hardware can perform load with VNNI transformation when data
          element type is 16-bit or lower precision, taking 2 or 4 elements from
          the first dimension and inserted into the newly added innermost dimension.
"""
function dpas(
    lhs::Value,
    rhs::Value,
    acc=nothing::Union{Nothing,Value};
    result::IR.Type,
    location=Location(),
)
    _results = IR.Type[result,]
    _operands = Value[lhs, rhs]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(acc) && push!(_operands, acc)

    return IR.create_operation(
        "xegpu.dpas",
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
`fence`
It synchronizes the memory access between
    write and following read or write.
    1. `Memory_kind` describes the memory kind. \"global\" means the global memory,
        \"slm\" means the share local memory.
    2. `Fence_scope` describes the scope of fence. \"Workgroup\" means that the scope would be
        within each workgroup. \"GPU\" means the scope would be across workgroups within the GPU.
"""
function fence(; memory_kind, fence_scope, location=Location())
    _results = IR.Type[]
    _operands = Value[]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[
        namedattribute("memory_kind", memory_kind),
        namedattribute("fence_scope", fence_scope),
    ]

    return IR.create_operation(
        "xegpu.fence",
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
`init_nbarrier`
InitNbarrierOp assigns the named barrier with the specified
      barrier ID (0~31) to the current thread. Multiple threads may bind to the
      same named barrier, and the `participant_thread_num` specifies the total
      number of threads associated with the nbarrier. It returns an object of
      NbarrierType representing the barrier
"""
function init_nbarrier(
    nbarrier_id::Value, participant_thread_num::Value; result::IR.Type, location=Location()
)
    _results = IR.Type[result,]
    _operands = Value[nbarrier_id, participant_thread_num]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "xegpu.init_nbarrier",
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
`load`
 It (aka. load) load data per each work-item. The output
   describes the data being loaded at the subgroup level, so its size is
   consistent with the number of work-items in a subgroup. When `chunk_size_per_lane`
   attribute is larger than 1 in TensorDesc, the output vector will be 2D vector,
   with dim-1 correspoding to the chunk size.

   The mask operand masks out memory access so that it is safe to pass out-of-boundary
   addresses/offsets as long as they are masked. It applies to slots of SIMD lanes.

 Example:
 ```
   %2 = xegpu.load %1, %0 {transpose = [1, 0],
                           l1_hint = #xegpu.cache_hint<cached>,
                           l2_hint = #xegpu.cache_hint<uncached>,
                           l3_hint = #xegpu.cache_hint<uncached>}
         : !xegpu.tensor_desc<16xf32, #xegpu.tdesc_attr<scattered=true>>, vector<16xi1>
           -> vector<16xf32>
 ```
"""
function load(
    TensorDesc::Value,
    mask::Value;
    value::IR.Type,
    transpose=nothing,
    l1_hint=nothing,
    l2_hint=nothing,
    l3_hint=nothing,
    location=Location(),
)
    _results = IR.Type[value,]
    _operands = Value[TensorDesc, mask]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(transpose) && push!(_attributes, namedattribute("transpose", transpose))
    !isnothing(l1_hint) && push!(_attributes, namedattribute("l1_hint", l1_hint))
    !isnothing(l2_hint) && push!(_attributes, namedattribute("l2_hint", l2_hint))
    !isnothing(l3_hint) && push!(_attributes, namedattribute("l3_hint", l3_hint))

    return IR.create_operation(
        "xegpu.load",
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
`load_nd`

LoadNdOp essentially mimics the hardware block read instruction to read
a block of data from memory to register. It takes a set of optional cache
hints for each level of cache, L1, L2 and L3. If hardware does not have a
correspoding cache, Corresponding cache hint attribute will be masked.
VNNI transformation is an hardware feature for Intel GPU, which is used to
do data packing during the load for B operand of matrix operation, if
the bit width of the data type is less then 32 bits, e.g., fp16. And
transpose is another Intel hardware feature, which will do transpose
operation when loading the data if the bit width of the data type is
fp32 or fp64. It implies that vnni and transpose cannot exit at the
same time.

# Example
```
  xegpu.load_nd %1 {transpose = [1, 0],
                    l1_hint = #xegpu.cache_hint<cached>,
                    l2_hint = #xegpu.cache_hint<uncached>,
                    l3_hint = #xegpu.cache_hint<streaming>}
          : !xegpu.tensor_desc<8x16xf32> -> vector<16x8xf32>
```
"""
function load_nd(
    TensorDesc::Value;
    value::IR.Type,
    vnni_axis=nothing,
    transpose=nothing,
    l1_hint=nothing,
    l2_hint=nothing,
    l3_hint=nothing,
    location=Location(),
)
    _results = IR.Type[value,]
    _operands = Value[TensorDesc,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(vnni_axis) && push!(_attributes, namedattribute("vnni_axis", vnni_axis))
    !isnothing(transpose) && push!(_attributes, namedattribute("transpose", transpose))
    !isnothing(l1_hint) && push!(_attributes, namedattribute("l1_hint", l1_hint))
    !isnothing(l2_hint) && push!(_attributes, namedattribute("l2_hint", l2_hint))
    !isnothing(l3_hint) && push!(_attributes, namedattribute("l3_hint", l3_hint))

    return IR.create_operation(
        "xegpu.load_nd",
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
`nbarrier_arrive`
NbarrierArriveOp signals the hardware (or other threads)
    that the current thread has produced its data for the consumer threads. When
    the hardware signalled by `participant_thread_num` threads for the named barrier,
    it will notify the threads waiting for the named barrier to continue their work.
"""
function nbarrier_arrive(nbarrier::Value; location=Location())
    _results = IR.Type[]
    _operands = Value[nbarrier,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "xegpu.nbarrier_arrive",
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
`nbarrier_wait`
NbarrierWaitOp signals the hardware which named barrier
    the current thread is waiting for, such that it can get notified when the
    named barrier is completed.
"""
function nbarrier_wait(nbarrier::Value; location=Location())
    _results = IR.Type[]
    _operands = Value[nbarrier,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "xegpu.nbarrier_wait",
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
`prefetch_nd`

It issues an instruction to prefetch a block of data from continuous
memory regions to each level of the cache based on their cache policy.

# Example
```
  xegpu.prefetch_nd %tdesc {l1_hint = #xegpu.cache_hint<cached>,
                            l2_hint = #xegpu.cache_hint<cached>,
                            l3_hint = #xegpu.cache_hint<cached>}
    : !xegpu.tensor_desc<8x16xf16>
```
"""
function prefetch_nd(
    TensorDesc::Value;
    l1_hint=nothing,
    l2_hint=nothing,
    l3_hint=nothing,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[TensorDesc,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(l1_hint) && push!(_attributes, namedattribute("l1_hint", l1_hint))
    !isnothing(l2_hint) && push!(_attributes, namedattribute("l2_hint", l2_hint))
    !isnothing(l3_hint) && push!(_attributes, namedattribute("l3_hint", l3_hint))

    return IR.create_operation(
        "xegpu.prefetch_nd",
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
`prefetch`

It issues instructions to prefetch a set of scattered data points
from memory to each level of the cache based on their cache policy.
As compared to prefetch_nd, which works on non-scattered TensorDesc,
it works on scattered TensorDesc instead.

# Example
```
  xegpu.prefetch %tdesc {l1_hint = #xegpu.cache_hint<cached>,
                         l2_hint = #xegpu.cache_hint<cached>,
                         l3_hint = #xegpu.cache_hint<cached>}
    : !xegpu.tensor_desc<16xf16>
```
"""
function prefetch(
    TensorDesc::Value;
    l1_hint=nothing,
    l2_hint=nothing,
    l3_hint=nothing,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[TensorDesc,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(l1_hint) && push!(_attributes, namedattribute("l1_hint", l1_hint))
    !isnothing(l2_hint) && push!(_attributes, namedattribute("l2_hint", l2_hint))
    !isnothing(l3_hint) && push!(_attributes, namedattribute("l3_hint", l3_hint))

    return IR.create_operation(
        "xegpu.prefetch",
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
`store_nd`

StoreNdOp essentially mimics the hardware block write instruction io
write a block of data from register into the memory region as described
by the TensorDesc. It takes a set of optional cache hints for each level
of cache, L1, L2 and L3. If hardware does not have a correspoding cache,
Corresponding cache hint attribute will be masked.

# Example
```
  xegpu.store_nd %3, %2 {l1_hint = #xegpu.cache_hint<uncached>,
                         l2_hint = #xegpu.cache_hint<write_back>,
                         l3_hint = #xegpu.cache_hint<write_through>}
                         : vector<8x16xf16>, !xegpu.tensor_desc<8x16xf16>
```
"""
function store_nd(
    value::Value,
    TensorDesc::Value;
    l1_hint=nothing,
    l2_hint=nothing,
    l3_hint=nothing,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[value, TensorDesc]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(l1_hint) && push!(_attributes, namedattribute("l1_hint", l1_hint))
    !isnothing(l2_hint) && push!(_attributes, namedattribute("l2_hint", l2_hint))
    !isnothing(l3_hint) && push!(_attributes, namedattribute("l3_hint", l3_hint))

    return IR.create_operation(
        "xegpu.store_nd",
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
`store`
 It (aka. store) stores data to scattered memory locations.
 It has similar semantic to `load_gather`.

 Example:
 ```
   %3 = xegpu.store %0, %1, %2 {l1_hint = #xegpu.cache_hint<uncached>,
                                l2_hint = #xegpu.cache_hint<write_back>,
                                l3_hint = #xegpu.cache_hint<write_through>}
         : vector<16xf32>, !xegpu.tensor_desc<16xf32, #xegpu.tdesc_attr<scattered=true>>, vector<16xi1>
 ```
"""
function store(
    value::Value,
    TensorDesc::Value,
    mask::Value;
    l1_hint=nothing,
    l2_hint=nothing,
    l3_hint=nothing,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[value, TensorDesc, mask]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(l1_hint) && push!(_attributes, namedattribute("l1_hint", l1_hint))
    !isnothing(l2_hint) && push!(_attributes, namedattribute("l2_hint", l2_hint))
    !isnothing(l3_hint) && push!(_attributes, namedattribute("l3_hint", l3_hint))

    return IR.create_operation(
        "xegpu.store",
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
`update_nd_offset`
The op updates the offset of the given TensorDesc.
    The offsets are relative offset to the current position in the number
    of elements. It will result in a same type TensorDesc as the input.

  example:
  ```
    %2 = xegpu.update_nd_offset %1, [0, 16]: !xegpu.tensor_desc<8x16xf32>
  ```
"""
function update_nd_offset(
    TensorDesc::Value,
    offsets::Vector{Value};
    result::IR.Type,
    const_offsets,
    location=Location(),
)
    _results = IR.Type[result,]
    _operands = Value[TensorDesc, offsets...]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[namedattribute("const_offsets", const_offsets),]

    return IR.create_operation(
        "xegpu.update_nd_offset",
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
`update_offset`
It behaves similar to `update_nd_offset` in terms that
    it updates offset of a TensorDesc, and the offsets are relative offset to
    the current position in the number of elements. However, `update_nd_offset`
    is to update the start point of a 2D block, so its offset constains two
    elements representing the shift in each dimension. `update_offset` is to
    update the offset per work-item, so its offsets contains values representing
    shifts for each work-item.

    Example:
    ```
      %2 = xegpu.update_offset %1, [32, 32, 32, 32]
            : !xegpu.tensor_desc<4x2xf32, #xegpu.tdesc_attr<scattered = true>>
    ```
"""
function update_offset(
    TensorDesc::Value,
    offsets::Vector{Value};
    result::IR.Type,
    const_offsets,
    location=Location(),
)
    _results = IR.Type[result,]
    _operands = Value[TensorDesc, offsets...]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[namedattribute("const_offsets", const_offsets),]

    return IR.create_operation(
        "xegpu.update_offset",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

end # xegpu
