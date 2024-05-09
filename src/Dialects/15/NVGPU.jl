module nvgpu

import ...IR:
    IR, NamedAttribute, Value, Location, Block, Region, Attribute, context, IndexType
import ..Dialects: namedattribute, operandsegmentsizes
import ...API

"""
`device_async_copy`

The `gpu.device_async_copy` op initiates an asynchronous copy operation of
`\$size` elements from source to the destination without blocking the thread.
The destination has to be in shared memory.

This is memory access will be pending to be added to a group.

This op is meant to be used with `gpu.device_async_create_group` and
`gpu.device_async_wait` to synchronize copies as explained in those ops
descriptions. 
`bypassL1` attribute is hint to the backend and hardware that
the copy should by pass the L1 cache, this may be dropped by the backend or
hardware. 

In order to do a copy and wait for the result we need the following
combination:
```
// copy 1.
%cp1 = gpu.device_async_copy %A[%c0], %B[%c0], 4 :memref<16xf32> to memref<16xf32, 3>
// copy 2.
%cp2 = gpu.device_async_copy %C[%c0], %D[%c0], 4 : memref<16xf32> to memref<16xf32, 3>
// group 1 contains copy 1 and copy 2.
%token1 = gpu.device_async_create_group %cp1, %cp2
// copy 3.
%cp3 = gpu.device_async_copy %E[%c0], %F[%c0], 4 : memref<16xf32> to memref<16xf32, 3>
// group 2 contains copy 3.
%token2 = gpu.device_async_create_group %cp3
// after the wait copy 1 and copy 2 are complete.
gpu.device_async_wait %token1
// after the wait copy 3 is complete.
gpu.device_async_wait %token2
```

# Example

```mlir
%0 = gpu.device_async_copy %src[%c0, %c0], %dst[%c0, %c0, %c0], 4 :
  memref<4x5xf32> to memref<2x7x5xf32, 3>
```
"""
function device_async_copy(
    dst::Value,
    dstIndices::Vector{Value},
    src::Value,
    srcIndices::Vector{Value};
    asyncToken::IR.Type,
    numElements,
    bypassL1=nothing,
    location=Location(),
)
    results = IR.Type[asyncToken,]
    operands = Value[dst, dstIndices..., src, srcIndices...]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("numElements", numElements),]
    push!(attributes, operandsegmentsizes([1, length(dstIndices), 1, length(srcIndices)]))
    !isnothing(bypassL1) && push!(attributes, namedattribute("bypassL1", bypassL1))

    return IR.create_operation(
        "nvgpu.device_async_copy",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=results,
        result_inference=false,
    )
end

"""
`device_async_create_group`

The `gpu.device_async_create_group` op creates a group of memory accesses
containing all the pending `device_async_copy` operations associated with
argument tokens. Each token can only be part of one group.

It returns a token that can be use to wait until the group fully completes.

This is meant to be used with `gpu.device_async_wait` to synchronize copies
as explained in those ops descriptions.

Groups are executed in the order they are created.

# Example

```mlir
%0 = gpu.device_async_create_group
```
"""
function device_async_create_group(
    inputTokens::Vector{Value}; asyncToken::IR.Type, location=Location()
)
    results = IR.Type[asyncToken,]
    operands = Value[inputTokens...,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return IR.create_operation(
        "nvgpu.device_async_create_group",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=results,
        result_inference=false,
    )
end

"""
`device_async_wait`

The `gpu.device_async_wait` op will block the execution thread until the group
associated with the source token is fully completed.

  The optional `\$numGroup` attribute gives a lower bound of the number of
  groups uncompleted when the wait can unblock the thread.
# Example

```mlir
gpu.device_async_wait %0
```
"""
function device_async_wait(asyncDependencies::Value; numGroups=nothing, location=Location())
    results = IR.Type[]
    operands = Value[asyncDependencies,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(numGroups) && push!(attributes, namedattribute("numGroups", numGroups))

    return IR.create_operation(
        "nvgpu.device_async_wait",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=results,
        result_inference=false,
    )
end

"""
`ldmatrix`

The `nvgpu.ldmatrix` op represents loading a matrix fragment from
memory. The load source and result type must be compatible with lowering
to the `nvvm.ldmatrix` instruction. This op is meant to represent
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
    results = IR.Type[res,]
    operands = Value[srcMemref, indices...]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("transpose", transpose), namedattribute("numTiles", numTiles)
    ]

    return IR.create_operation(
        "nvgpu.ldmatrix",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=results,
        result_inference=false,
    )
end

"""
`mma_sync`

The `nvgpu.mma.sync` op represents the distributed form of a collective
matrix-multiply-and-accumulate (mma) operation that is compatible with
`nvvm.mma.sync`. The operands and results are fragments of the full matrix
operands. The full shape of the distributed mma operation is given by the
`mmaShape` attribute in the form of a list of dimensions `[m, n, k]`.

This operation is meant to be lowered to the `nvvm.mma.sync` instruction, and
is an intermediate point between lowering from `vector.contract` to
`nvvm.mma.sync`.

This operation is meant to follow the semantic of described here:
  https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-mma

# Example

```mlir
nvgpu.mma.sync (%a, %b, %c) :
  (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf16>) -> vector<2x2xf16>
```
"""
function mma_sync(
    matrixA::Value,
    matrixB::Value,
    matrixC::Value;
    res::IR.Type,
    mmaShape,
    location=Location(),
)
    results = IR.Type[res,]
    operands = Value[matrixA, matrixB, matrixC]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("mmaShape", mmaShape),]

    return IR.create_operation(
        "nvgpu.mma.sync",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=results,
        result_inference=false,
    )
end

end # nvgpu
