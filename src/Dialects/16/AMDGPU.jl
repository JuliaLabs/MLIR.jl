module amdgpu

import ...IR: NamedAttribute, MLIRType, get_value, Location, Block, Region, Attribute, create_operation, context, IndexType
import ..Dialects: namedattribute, operandsegmentsizes
import ...API


"""
`lds_barrier`

`amdgpu.lds_barrier` is both a barrier (all workitems in a workgroup must reach
the barrier before any of them may proceed past it) and a wait for all
operations that affect the Local Data Store (LDS) issued from that wrokgroup
to complete before the workgroup may continue. Since the LDS is per-workgroup
memory, this barrier may be used, for example, to ensure all workitems have
written data to LDS before any workitem attempts to read from it.

Note that `lds_barrier` does **not** force reads to or from global memory
to complete before execution continues. Therefore, it should be used when
operations on global memory can be issued far in advance of when their results
are used (for example, by writing them to LDS).
"""
function lds_barrier(; location=Location())
    results = MLIRType[]
    operands = API.MlirValue[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "amdgpu.lds_barrier", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`mfma`

The `amdgpu.mfma` op is an MLIR wrapper around intrinsics
for various `mfma` instructions in the CDNA architecture, which perform
multiple outer products in order to allow fast matrix multiplication.

The wrapper will select an appropriate `mfma` instruction, if one is available,
based on the provided `m`, `k`, `n`, and `nBlks` attributes, along with the
types of the source and destination arguments.

For information on the layouts of the input and output matrces (which are stored
in `sourceA`, `sourceB`, `destC`, and `destD`), see the CDNA ISA documentation.

The `cbsz`, `abid`, and `blgp` parameters control how the lanes of the wave
are permuted when matrix data is being loaded: `blgp` can be any number of
fixed permutations, `cbsz` specifies the log_2 of the number of chunks the lanes
holding sourceA are split into, and `abid` selects one of those chunks.

Note, this wrapper allows specifying `vector<4Kxi8>` arguments to MFMA
intrinsics that take an integer type of width `4K`. For example,
one can provide a vector<4xi8> as an argument to an MFMA instruction that
logically takes 4 i8s but whose intrinsics are specified to take an i32.
In these cases, the bytes in the vector will be concatenated in little-endian
order (that is, v[0] will go to arg[7:0], v[1] to arg[15:8] and so on).

The negateA, negateB, and negateC flags are only supported for double-precision
operations on gfx940+.
"""
function mfma(sourceA, sourceB, destC; destD::MLIRType, m, n, k, blocks, cbsz=nothing, abid=nothing, blgp=nothing, reducePrecision=nothing, negateA=nothing, negateB=nothing, negateC=nothing, location=Location())
    results = MLIRType[destD, ]
    operands = API.MlirValue[get_value(sourceA), get_value(sourceB), get_value(destC), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("m", m), namedattribute("n", n), namedattribute("k", k), namedattribute("blocks", blocks), ]
    (cbsz != nothing) && push!(attributes, namedattribute("cbsz", cbsz))
    (abid != nothing) && push!(attributes, namedattribute("abid", abid))
    (blgp != nothing) && push!(attributes, namedattribute("blgp", blgp))
    (reducePrecision != nothing) && push!(attributes, namedattribute("reducePrecision", reducePrecision))
    (negateA != nothing) && push!(attributes, namedattribute("negateA", negateA))
    (negateB != nothing) && push!(attributes, namedattribute("negateB", negateB))
    (negateC != nothing) && push!(attributes, namedattribute("negateC", negateC))
    
    create_operation(
        "amdgpu.mfma", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`raw_buffer_atomic_fadd`

The `amdgpu.raw_buffer_atomic_fadd` op is a wrapper around the
buffer-based atomic floating point addition available on the MI-* series
of AMD GPUs.

The index into the buffer is computed as for `memref.store` with the addition
of `indexOffset` (which is used to aid in emitting vectorized code) and,
if present `sgprOffset` (which is added after bounds checks and includes
any non-zero offset on the memref type).

All indexing components are given in terms of the memref\'s element size, not
the byte lengths required by the intrinsic.

Out of bounds atomic operations are ignored in hardware.

See `amdgpu.raw_buffer_load` for a description of how the underlying
instruction is constructed.
"""
function raw_buffer_atomic_fadd(value, memref, indices, sgprOffset=nothing; boundsCheck=nothing, indexOffset=nothing, location=Location())
    results = MLIRType[]
    operands = API.MlirValue[get_value(value), get_value(memref), get_value.(indices)..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (sgprOffset != nothing) && push!(operands, get_valuesgprOffset)
    push!(attributes, operandsegmentsizes([1, 1, length(indices), (sgprOffset==nothing) ? 0 : 1]))
    (boundsCheck != nothing) && push!(attributes, namedattribute("boundsCheck", boundsCheck))
    (indexOffset != nothing) && push!(attributes, namedattribute("indexOffset", indexOffset))
    
    create_operation(
        "amdgpu.raw_buffer_atomic_fadd", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`raw_buffer_load`

The `amdgpu.raw_buffer_load` op is a wrapper around the buffer load intrinsics
available on AMD GPUs, including extensions in newer GPUs.

The index into the buffer is computed as for `memref.load` with the additon
of `indexOffset` and `sgprOffset` (which **may or may not** be considered
in bounds checks and includes any offset present on the memref type if it\'s
non-zero).

All indices and offsets are in units of the memref\'s data type and are
converted to bytes during lowering.

When a load is out of bounds, the instruction returns zero.
Partially-out of bounds have chipset-dependent behavior: whether reading
2 elements starting at index 7 of a `memref<8xf32>` returns the last element
in the first vector component depends on the architecture.

The memref struct is converted into a buffer resource (a V#) and the arguments
are translated to intrinsic arguments as follows:
- The base address of the buffer is the base address of the memref
- The stride is 0 to enable raw mode
- The number of records is the size of the memref, in bytes
  In the case of dynamically-shaped memrefs, this is computed at runtime
  as max_d (size(d) * stride(d)) * sizeof(elementType(memref))
- The offset enable bit is 1, the index enable bit is 0.
- The thread ID addition bit is off
- If `boundsCheck` is false and the target chipset is RDNA, OOB_SELECT is set
  to 2 to disable bounds checks, otherwise it is 3
- The cache coherency bits are off
"""
function raw_buffer_load(memref, indices, sgprOffset=nothing; value::MLIRType, boundsCheck=nothing, indexOffset=nothing, location=Location())
    results = MLIRType[value, ]
    operands = API.MlirValue[get_value(memref), get_value.(indices)..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (sgprOffset != nothing) && push!(operands, get_valuesgprOffset)
    push!(attributes, operandsegmentsizes([1, length(indices), (sgprOffset==nothing) ? 0 : 1]))
    (boundsCheck != nothing) && push!(attributes, namedattribute("boundsCheck", boundsCheck))
    (indexOffset != nothing) && push!(attributes, namedattribute("indexOffset", indexOffset))
    
    create_operation(
        "amdgpu.raw_buffer_load", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`raw_buffer_store`

The `amdgpu.raw_buffer_store` op is a wrapper around the buffer store
intrinsics available on AMD GPUs, including extensions in newer GPUs.

The store index is computed as in `memref.store` with the addition of
`indexOffset` (which is included for uniformity with atomics and may be useful
when writing vectorized code) and `sgprOffset` (which is added after bounds
checks and implicitly includes the offset of the memref type if non-zero).
All index components are in terms of the elements of the memref, not bytes,
and are scaled up appropriately.

Out of bounds stores are ignored in hardware.
Wthether a vector write that includes some in-bounds and soeme out-of-bounds
components is partically completed is chipset-dependent.

See `amdgpu.raw_buffer_load` for a description of how the underlying
instruction is constructed.
"""
function raw_buffer_store(value, memref, indices, sgprOffset=nothing; boundsCheck=nothing, indexOffset=nothing, location=Location())
    results = MLIRType[]
    operands = API.MlirValue[get_value(value), get_value(memref), get_value.(indices)..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (sgprOffset != nothing) && push!(operands, get_valuesgprOffset)
    push!(attributes, operandsegmentsizes([1, 1, length(indices), (sgprOffset==nothing) ? 0 : 1]))
    (boundsCheck != nothing) && push!(attributes, namedattribute("boundsCheck", boundsCheck))
    (indexOffset != nothing) && push!(attributes, namedattribute("indexOffset", indexOffset))
    
    create_operation(
        "amdgpu.raw_buffer_store", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

end # amdgpu
