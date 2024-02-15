module nvgpu

import ...IR: NamedAttribute, MLIRType, Value, Location, Block, Region, Attribute, create_operation, context, IndexType
import ..Dialects: namedattribute, operandsegmentsizes
import ...API


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
function device_async_copy(dst::Value, dstIndices::Vector{Value}, src::Value, srcIndices::Vector{Value}, srcElements=nothing::Union{Nothing, Value}; asyncToken::MLIRType, dstElements, bypassL1=nothing, location=Location())
    results = MLIRType[asyncToken, ]
    operands = Value[dst, dstIndices..., src, srcIndices..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("dstElements", dstElements), ]
    !isnothing(srcElements) && push!(operands, srcElements)
    push!(attributes, operandsegmentsizes([1, length(dstIndices), 1, length(srcIndices), (srcElements==nothing) ? 0 : 1]))
    !isnothing(bypassL1) && push!(attributes, namedattribute("bypassL1", bypassL1))
    
    create_operation(
        "nvgpu.device_async_copy", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
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
function device_async_create_group(inputTokens::Vector{Value}; asyncToken::MLIRType, location=Location())
    results = MLIRType[asyncToken, ]
    operands = Value[inputTokens..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "nvgpu.device_async_create_group", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`device_async_wait`

The `nvgpu.device_async_wait` op will block the execution thread until the group
associated with the source token is fully completed.

The optional `\$numGroup` attribute gives a lower bound of the number of
groups uncompleted when the wait can unblock the thread.

# Example

```mlir
nvgpu.device_async_wait %0
```
"""
function device_async_wait(asyncDependencies::Value; numGroups=nothing, location=Location())
    results = MLIRType[]
    operands = Value[asyncDependencies, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(numGroups) && push!(attributes, namedattribute("numGroups", numGroups))
    
    create_operation(
        "nvgpu.device_async_wait", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
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
function ldmatrix(srcMemref::Value, indices::Vector{Value}; res::MLIRType, transpose, numTiles, location=Location())
    results = MLIRType[res, ]
    operands = Value[srcMemref, indices..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("transpose", transpose), namedattribute("numTiles", numTiles), ]
    
    create_operation(
        "nvgpu.ldmatrix", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
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
function mma_sp_sync(matrixA::Value, matrixB::Value, matrixC::Value, sparseMetadata::Value; res::MLIRType, mmaShape, sparsitySelector=nothing, tf32Enabled=nothing, location=Location())
    results = MLIRType[res, ]
    operands = Value[matrixA, matrixB, matrixC, sparseMetadata, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("mmaShape", mmaShape), ]
    !isnothing(sparsitySelector) && push!(attributes, namedattribute("sparsitySelector", sparsitySelector))
    !isnothing(tf32Enabled) && push!(attributes, namedattribute("tf32Enabled", tf32Enabled))
    
    create_operation(
        "nvgpu.mma.sp.sync", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
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
function mma_sync(matrixA::Value, matrixB::Value, matrixC::Value; res::MLIRType, mmaShape, tf32Enabled=nothing, location=Location())
    results = MLIRType[res, ]
    operands = Value[matrixA, matrixB, matrixC, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("mmaShape", mmaShape), ]
    !isnothing(tf32Enabled) && push!(attributes, namedattribute("tf32Enabled", tf32Enabled))
    
    create_operation(
        "nvgpu.mma.sync", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

end # nvgpu
