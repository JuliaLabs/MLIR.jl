module nvgpu

import ...IR:
    IR, NamedAttribute, Value, Location, Block, Region, Attribute, context, IndexType
import ..Dialects: namedattribute, operandsegmentsizes

"""
`device_async_copy`

The `nvgpu.device_async_copy` op initiates an asynchronous copy operation of
elements from source (global memory) to the destination (shared memory)
without blocking the thread. The async copy is added to a group.

This op is meant to be used with `nvgpu.device_async_create_group` and
`nvgpu.device_async_wait` to synchronize copies as explained in those ops
descriptions.

`bypassL1` attribute is hint to the hardware to bypass the L1 cache during
async copy, this hint may be ignored by the hardware.

`dstElements` attribute is the total number of elements written to
destination (shared memory).

`srcElements` argument is the total number of elements read from
source (global memory).

`srcElements` is an optional argument and when present the op only reads
`srcElements` number of elements from the source (global memory) and zero fills
the rest of the elements in the destination (shared memory).

In order to do a copy and wait for the result we need the following
combination:
```
// copy 1.
%cp1 = nvgpu.device_async_copy %A[%c0], %B[%c0], 4 :memref<16xf32> to memref<16xf32, 3>
// copy 2.
%cp2 = nvgpu.device_async_copy %C[%c0], %D[%c0], 4 : memref<16xf32> to memref<16xf32, 3>
// group 1 contains copy 1 and copy 2.
%token1 = nvgpu.device_async_create_group %cp1, %cp2
// copy 3.
%cp3 = nvgpu.device_async_copy %E[%c0], %F[%c0], 4 : memref<16xf32> to memref<16xf32, 3>
// group 2 contains copy 3.
%token2 = nvgpu.device_async_create_group %cp3
// after the wait copy 1 and copy 2 are complete.
nvgpu.device_async_wait %token1
// after the wait copy 3 is complete.
nvgpu.device_async_wait %token2
```

# Example

```mlir
%0 = nvgpu.device_async_copy %src[%c0, %c0], %dst[%c0, %c0, %c0], 4 :
  memref<4x5xf32> to memref<2x7x5xf32, 3>
```
"""
function device_async_copy(
    dst::Value,
    dstIndices::Vector{Value},
    src::Value,
    srcIndices::Vector{Value},
    srcElements=nothing::Union{Nothing,Value};
    asyncToken::IR.Type,
    dstElements,
    bypassL1=nothing,
    location=Location(),
)
    _results = IR.Type[asyncToken,]
    _operands = Value[dst, dstIndices..., src, srcIndices...]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[namedattribute("dstElements", dstElements),]
    !isnothing(srcElements) && push!(_operands, srcElements)
    push!(
        _attributes,
        operandsegmentsizes([
            1, length(dstIndices), 1, length(srcIndices), isnothing(srcElements) ? 0 : 1
        ]),
    )
    !isnothing(bypassL1) && push!(_attributes, namedattribute("bypassL1", bypassL1))

    return IR.create_operation(
        "nvgpu.device_async_copy",
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
`device_async_create_group`

The `nvgpu.device_async_create_group` op creates a group of memory accesses
containing all the pending `device_async_copy` operations associated with
argument tokens. Each token can only be part of one group.

It returns a token that can be use to wait until the group fully completes.

This is meant to be used with `nvgpu.device_async_wait` to synchronize copies
as explained in those ops descriptions.

Groups are executed in the order they are created.

# Example

```mlir
%0 = nvgpu.device_async_create_group
  ```
"""
function device_async_create_group(
    inputTokens::Vector{Value}; asyncToken::IR.Type, location=Location()
)
    _results = IR.Type[asyncToken,]
    _operands = Value[inputTokens...,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "nvgpu.device_async_create_group",
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
`device_async_wait`

The `nvgpu.device_async_wait` op will block the execution thread until the group
associated with the source token is fully completed.

The optional `\$numGroups` attribute gives an upper bound of the number of
groups uncompleted when the wait can unblock the thread. For example,  if
16 async groups are pushe and `\$numGroups` is set to 12, then the thread
will unblock when 12 groups or fewer are in flight (4 groups have
completed).

# Example

```mlir
nvgpu.device_async_wait %0
```
"""
function device_async_wait(asyncDependencies::Value; numGroups=nothing, location=Location())
    _results = IR.Type[]
    _operands = Value[asyncDependencies,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(numGroups) && push!(_attributes, namedattribute("numGroups", numGroups))

    return IR.create_operation(
        "nvgpu.device_async_wait",
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
`ldmatrix`

The `nvgpu.ldmatrix` op represents loading a matrix fragment from
memory to registers. The source and result type must be compatible
with lowering to the `nvvm.ldmatrix` instruction. This op represents
the distributed version of a `vector.transfer_read` as an intermediate
step between lowering from `vector.transfer_read` to `nvvm.ldmatrix`.

This operation is meant to follow the semantic of described here:
https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-ldmatrix

# Example
```mlir
%0 = nvgpu.ldmatrix %sm[%c0, %c0] {numTiles = 4 : i32, transpose = false} :
  memref<?x?xf16, 3> -> vector<4x2xf16>
```
"""
function ldmatrix(
    srcMemref::Value,
    indices::Vector{Value};
    res::IR.Type,
    transpose,
    numTiles,
    location=Location(),
)
    _results = IR.Type[res,]
    _operands = Value[srcMemref, indices...]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[
        namedattribute("transpose", transpose), namedattribute("numTiles", numTiles)
    ]

    return IR.create_operation(
        "nvgpu.ldmatrix",
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
`mbarrier_arrive_expect_tx`

A thread executing the Op performs an expect-tx operation on the mbarrier 
object at the location specified by the address operand \$barrier. The 
expect-tx operation, with an \$txcount argument, increases the tx-count of 
an mbarrier object by the value specified by \$txcount. This makes the 
current phase of the mbarrier object to expect and track the completion of 
additional asynchronous transactions.

The `\$txCount` specifies the number of element to the expect-tx operation.

# Example
```mlir
  nvgpu.mbarrier.arrive.expect_tx %barrier, %ic0 : !nvgpu.mbarrier.barrier<memorySpace = #gpu.address_space<workgroup>>
```
"""
function mbarrier_arrive_expect_tx(
    barriers::Value,
    txcount::Value,
    mbarId::Value,
    predicate=nothing::Union{Nothing,Value};
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[barriers, txcount, mbarId]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(predicate) && push!(_operands, predicate)

    return IR.create_operation(
        "nvgpu.mbarrier.arrive.expect_tx",
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
`mbarrier_arrive_nocomplete`

The Op performs arrive-on operation on the `mbarrier` object and returns a 
`nvgpu.mbarrier.token`.

The Op does not cause the `nvgpu.mbarrier` to complete its current phase.

# Example
```mlir
  %token = nvgpu.mbarrier.arrive.noComplete %barrier, %count : !nvgpu.mbarrier.barrier<memorySpace = #gpu.address_space<workgroup>> -> !nvgpu.mbarrier.token
```
"""
function mbarrier_arrive_nocomplete(
    barriers::Value, mbarId::Value, count::Value; token::IR.Type, location=Location()
)
    _results = IR.Type[token,]
    _operands = Value[barriers, mbarId, count]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "nvgpu.mbarrier.arrive.nocomplete",
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
`mbarrier_arrive`

The Op performs arrive-on operation on the `mbarrier` object and returns a 
`nvgpu.mbarrier.token`.

For more information, see
https://docs.nvidia.com/cuda/parallel-thread-execution/#arrive-on-operation-on-mbarrier-object

# Example
```mlir
  %token = nvgpu.mbarrier.arrive %barrier : !nvgpu.mbarrier.barrier<memorySpace = #gpu.address_space<workgroup>> -> !nvgpu.mbarrier.token
```
"""
function mbarrier_arrive(
    barriers::Value, mbarId::Value; token::IR.Type, location=Location()
)
    _results = IR.Type[token,]
    _operands = Value[barriers, mbarId]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "nvgpu.mbarrier.arrive",
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
`mbarrier_create`

The Op generates one or more `mbarrier` object, which is a barrier created in 
shared memory and supports various synchronization behaviors for threads.

The `mbarrier` object has the following type and alignment requirements:
  Type: .b64, Alignment: 8, Memory space: .shared

# Example
```mlir
  %barrier = nvgpu.mbarrier.create -> !nvgpu.mbarrier.barrier<memorySpace = #gpu.address_space<workgroup>>
```
"""
function mbarrier_create(; barriers::IR.Type, location=Location())
    _results = IR.Type[barriers,]
    _operands = Value[]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "nvgpu.mbarrier.create",
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
`mbarrier_init`

The Op initializes the `mbarrier` object with the given number of threads.

# Example
```mlir
  %num_threads = gpu.block_dim x
  %barrier = nvgpu.mbarrier.create -> !nvgpu.mbarrier.barrier<memorySpace = #gpu.address_space<workgroup>>
  nvgpu.mbarrier.init %barrier, %num_threads : !nvgpu.mbarrier.barrier<memorySpace = #gpu.address_space<workgroup>>
```
"""
function mbarrier_init(
    barriers::Value,
    count::Value,
    mbarId::Value,
    predicate=nothing::Union{Nothing,Value};
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[barriers, count, mbarId]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(predicate) && push!(_operands, predicate)

    return IR.create_operation(
        "nvgpu.mbarrier.init",
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
`mbarrier_test_wait`

Checks whether the mbarrier object has completed the phase. It is is a 
non-blocking instruction which tests for the completion of the phase.

# Example
```mlir
  %isComplete = nvgpu.mbarrier.test.wait %barrier, %token : !nvgpu.mbarrier.barrier<memorySpace = #gpu.address_space<workgroup>>, !nvgpu.mbarrier.token
```
"""
function mbarrier_test_wait(
    barriers::Value, token::Value, mbarId::Value; waitComplete::IR.Type, location=Location()
)
    _results = IR.Type[waitComplete,]
    _operands = Value[barriers, token, mbarId]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "nvgpu.mbarrier.test.wait",
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
`mbarrier_try_wait_parity`

Checks whether the mbarrier object has completed the phase. It is is a 
potentially blocking instruction which tests for the completion of the 
phase. Suspended thread resumes execution when the specified phase completes 
OR before the phase completes following a system-dependent time limit. 

The `\$phaseParity` specifies either even phase (0) or odd phase (1) to 
wait.

# Example
```mlir
  nvgpu.mbarrier.try_wait.parity %barrier, %phaseParity, %ticks : !nvgpu.mbarrier.barrier<memorySpace = #gpu.address_space<workgroup>>
```
"""
function mbarrier_try_wait_parity(
    barriers::Value, phaseParity::Value, ticks::Value, mbarId::Value; location=Location()
)
    _results = IR.Type[]
    _operands = Value[barriers, phaseParity, ticks, mbarId]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "nvgpu.mbarrier.try_wait.parity",
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
`mma_sp_sync`

The `nvgu.mma.sp.sync` operation performs a warp-distributed MMA operation
where operand A is \"structured sparse\". In this case, the `matrixA` operand
represents the (warp-distributed) non-zero values of operand A, and the
`sparse_metadata` operand provides the indices.

The full description of the sparsity storage format and distribution scheme is
described in the PTX docs. This operation is meant to follow the semantic
described in the PTX documentation here:
https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-for-sparse-mma

The way the indices are distributed among the threads in a warp is controlled
by the optional `sparsity_selector` operand, which is `0` by default. For
more information, please consult the PTX documentation linked above.

Example (targetingthe f16 16x8x32 `mma.sp` PTX instruction):

```mlir
nvgpu.mma.sp.sync (%a, %b, %c) metadata (%meta) {mmaShape = [16, 8, 32]} :
  (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf16>) -> vector<2x2xf16>
```
"""
function mma_sp_sync(
    matrixA::Value,
    matrixB::Value,
    matrixC::Value,
    sparseMetadata::Value;
    res::IR.Type,
    mmaShape,
    sparsitySelector=nothing,
    tf32Enabled=nothing,
    location=Location(),
)
    _results = IR.Type[res,]
    _operands = Value[matrixA, matrixB, matrixC, sparseMetadata]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[namedattribute("mmaShape", mmaShape),]
    !isnothing(sparsitySelector) &&
        push!(_attributes, namedattribute("sparsitySelector", sparsitySelector))
    !isnothing(tf32Enabled) &&
        push!(_attributes, namedattribute("tf32Enabled", tf32Enabled))

    return IR.create_operation(
        "nvgpu.mma.sp.sync",
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
`mma_sync`

The `nvgpu.mma.sync` op represents the warp-level matrix-multiply-and-
accumulate (mma) operation that is compatible with `nvvm.mma.sync`.
The operands and results vector sizes are thread-level onwership to
the warp-level mma operation shape. `mmaShape` attribute holds the
warp-level matrix-multiply shape.

The `nvgpu.mma.sync` op serves as an intermediate point between lowering from
`vector.contract` to `nvvm.mma.sync`.

This operation is meant to follow the semantic of described here:
  https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-mma

# Example

```mlir
%res = nvgpu.mma.sync (%matrixA, %matrixB, %matrixC) {mmaShape = [16, 8, 16]} :
    (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
```
"""
function mma_sync(
    matrixA::Value,
    matrixB::Value,
    matrixC::Value;
    res::IR.Type,
    mmaShape,
    tf32Enabled=nothing,
    location=Location(),
)
    _results = IR.Type[res,]
    _operands = Value[matrixA, matrixB, matrixC]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[namedattribute("mmaShape", mmaShape),]
    !isnothing(tf32Enabled) &&
        push!(_attributes, namedattribute("tf32Enabled", tf32Enabled))

    return IR.create_operation(
        "nvgpu.mma.sync",
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
`tma_async_load`

The Op loads a tile memory region from global memory to shared memory by 
Tensor Memory Access (TMA).

`\$tensorMapDescriptor` is tensor map descriptor which has information about
tile shape. The descriptor is created by `nvgpu.tma.create.descriptor`

The Op uses `\$barrier` mbarrier based completion mechanism.
"""
function tma_async_load(
    dst::Value,
    barriers::Value,
    tensorMapDescriptor::Value,
    coordinates::Vector{Value},
    mbarId::Value,
    multicastMask=nothing::Union{Nothing,Value};
    predicate=nothing::Union{Nothing,Value},
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[dst, barriers, tensorMapDescriptor, coordinates..., mbarId]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(multicastMask) && push!(_operands, multicastMask)
    !isnothing(predicate) && push!(_operands, predicate)
    push!(
        _attributes,
        operandsegmentsizes([
            1,
            1,
            1,
            length(coordinates),
            1,
            isnothing(multicastMask) ? 0 : 1,
            isnothing(predicate) ? 0 : 1,
        ]),
    )

    return IR.create_operation(
        "nvgpu.tma.async.load",
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
`tma_async_store`

The Op store a tile memory region from global memory to shared memory by 
Tensor Memory Access (TMA).

`\$tensorMapDescriptor` is tensor map descriptor which has information about
tile shape. The descriptor is created by `nvgpu.tma.create.descriptor`
"""
function tma_async_store(
    src::Value,
    tensorMapDescriptor::Value,
    coordinates::Vector{Value},
    predicate=nothing::Union{Nothing,Value};
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[src, tensorMapDescriptor, coordinates...]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(predicate) && push!(_operands, predicate)
    push!(
        _attributes,
        operandsegmentsizes([1, 1, length(coordinates), isnothing(predicate) ? 0 : 1]),
    )

    return IR.create_operation(
        "nvgpu.tma.async.store",
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
`tma_create_descriptor`

The Op creates a tensor map descriptor object representing tiled memory 
region. To do that it calls CUDA Driver\'s `cuTensorMapEncodeTiled`. The 
descriptor is used by Tensor Memory Access (TMA).

The `tensor` is the source tensor to be tiled. 

The `boxDimensions` is the size of the tiled memory region in each dimension.

For more information see below:
https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html
"""
function tma_create_descriptor(
    tensor::Value, boxDimensions::Vector{Value}; tensorMap::IR.Type, location=Location()
)
    _results = IR.Type[tensorMap,]
    _operands = Value[tensor, boxDimensions...]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "nvgpu.tma.create.descriptor",
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
`tma_prefetch_descriptor`

The Op brings the cache line containing the given `\$tmaDescriptor` for 
subsequent use by the `tma.async.load` instruction.
"""
function tma_prefetch_descriptor(
    tensorMapDescriptor::Value, predicate=nothing::Union{Nothing,Value}; location=Location()
)
    _results = IR.Type[]
    _operands = Value[tensorMapDescriptor,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(predicate) && push!(_operands, predicate)

    return IR.create_operation(
        "nvgpu.tma.prefetch.descriptor",
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
`warpgroup_generate_descriptor`

This Op builds a `nvgpu.warpgroup.descriptor` that is used by 
`nvgpu.warpgroup.mma` to perform warpgroup-level matrix multiply and 
accumulate.

The descriptor specifies the properties of the matrix in shared memory that 
is a multiplicand in the matrix multiply and accumulate operation.
"""
function warpgroup_generate_descriptor(
    tensor::Value, tensorMap::Value; descriptor::IR.Type, location=Location()
)
    _results = IR.Type[descriptor,]
    _operands = Value[tensor, tensorMap]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "nvgpu.warpgroup.generate.descriptor",
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
`warpgroup_mma_init_accumulator`

This Op generates and initializes the accumulator matrix for 
`nvgpu.warpgroup.mma` op to perform matrix-multiply-and-accumulate.
"""
function warpgroup_mma_init_accumulator(; matrixC::IR.Type, location=Location())
    _results = IR.Type[matrixC,]
    _operands = Value[]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "nvgpu.warpgroup.mma.init.accumulator",
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
`warpgroup_mma`

The `nvgpu.warpgroup.mma` op performs the warpgroup-level (4 warps) 
matrix-multiply-and-accumulate (mma) operation that results in 
`nvvm.wgmma.mma_async`. 

The operands are `descriptorA` and `descriptorB` that are wgmma matrix 
descriptors that shows the properties of the matrix in shared memory. The 
results are thread-level ownership to the warpgroup-level mma operation 
shape. The shape is deduced from the descriptor types and output vector.

The Op encapsulates multiple `nvvm.wgmma.mma_async` operations to complete 
the given shape. As `nvvm.wgmma.async` Op, or its corresponding PTX 
instruction, is asynchronous, this Op groups the `nvvm.wgmma.async` and 
surrounds them between `wgmma.fence.aligned` and 
`wgmma.commit.group.sync.aligned`, `wgmma.wait.group.sync.aligned` Ops.

# Example
```mlir
  %r1,%r2 = nvgpu.warpgroup.mma %descA, %descB, %acc1, %acc2: 
             !nvgpu.warpgroup.descriptor<tensor = memref<128x64xf16, 3>>, 
             !nvgpu.warpgroup.descriptor<tensor = memref<64x128xf16, 3>>, 
             !nvgpu.warpgroup.accumulator<fragmented = vector<64x128xf32>>,
             !nvgpu.warpgroup.accumulator<fragmented = vector<64x128xf32>>
             -> 
             !nvgpu.warpgroup.accumulator<fragmented = vector<64x128xf32>>,
             !nvgpu.warpgroup.accumulator<fragmented = vector<64x128xf32>>
```
"""
function warpgroup_mma(
    descriptorA::Value,
    descriptorB::Value,
    matrixC::Value;
    matrixD::IR.Type,
    waitGroup=nothing,
    transposeA=nothing,
    transposeB=nothing,
    location=Location(),
)
    _results = IR.Type[matrixD,]
    _operands = Value[descriptorA, descriptorB, matrixC]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(waitGroup) && push!(_attributes, namedattribute("waitGroup", waitGroup))
    !isnothing(transposeA) && push!(_attributes, namedattribute("transposeA", transposeA))
    !isnothing(transposeB) && push!(_attributes, namedattribute("transposeB", transposeB))

    return IR.create_operation(
        "nvgpu.warpgroup.mma",
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
`warpgroup_mma_store`

The `nvgpu.warpgroup.mma.store` op performs the store of fragmented result 
in \$matrixD to given memref. 

[See the details of register fragment layout for accumulator matrix D]
(https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#wgmma-64n16-d) 

Note that, the op must be run with warp group.
"""
function warpgroup_mma_store(matrixD::Value, dstMemref::Value; location=Location())
    _results = IR.Type[]
    _operands = Value[matrixD, dstMemref]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "nvgpu.warpgroup.mma.store",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

end # nvgpu
