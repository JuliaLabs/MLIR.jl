module gpu

import ...IR: IR, NamedAttribute, Value, Location, Block, Region, Attribute, create_operation, context, IndexType
import ..Dialects: namedattribute, operandsegmentsizes
import ...API


"""
`all_reduce`

The `all_reduce` op reduces the value of every work item across a local
workgroup. The result is equal for all work items of a workgroup.

For example, both

```mlir
%1 = gpu.all_reduce add %0 {} : (f32) -> (f32)
%2 = gpu.all_reduce %0 {
^bb(%lhs : f32, %rhs : f32):
  %sum = arith.addf %lhs, %rhs : f32
  \"gpu.yield\"(%sum) : (f32) -> ()
} : (f32) -> (f32)
```

compute the sum of each work item\'s %0 value. The first version specifies
the accumulation as operation, whereas the second version specifies the
accumulation as code region. The accumulation operation must be one of:
`add`, `and`, `max`, `min`, `mul`, `or`, `xor`.

Either none or all work items of a workgroup need to execute this op
in convergence.
"""
function all_reduce(value::Value; result_0=nothing::Union{Nothing,IR.Type}, op=nothing, body::Region, location=Location())
    results = IR.Type[]
    operands = Value[value,]
    owned_regions = Region[body,]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result_0) && push!(results, result_0)
    !isnothing(op) && push!(attributes, namedattribute("op", op))

    create_operation(
        "gpu.all_reduce", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`alloc`

The `gpu.alloc` operation allocates a region of memory on the GPU. It is
similar to the `memref.alloc` op, but supports asynchronous GPU execution.

The op does not execute before all async dependencies have finished
executing.

If the `async` keyword is present, the op is executed asynchronously (i.e.
it does not block until the execution has finished on the device). In
that case, it also returns a !gpu.async.token.

# Example

```mlir
%memref, %token = gpu.alloc async [%dep] (%width) : memref<64x?xf32, 1>
```
"""
function alloc(asyncDependencies::Vector{Value}, dynamicSizes::Vector{Value}, symbolOperands::Vector{Value}; memref::IR.Type, asyncToken=nothing::Union{Nothing,IR.Type}, location=Location())
    results = IR.Type[memref,]
    operands = Value[asyncDependencies..., dynamicSizes..., symbolOperands...,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    push!(attributes, operandsegmentsizes([length(asyncDependencies), length(dynamicSizes), length(symbolOperands),]))
    !isnothing(asyncToken) && push!(results, asyncToken)

    create_operation(
        "gpu.alloc", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`barrier`

The \"barrier\" op synchronizes all work items of a workgroup. It is used
to coordinate communication between the work items of the workgroup.

```mlir
gpu.barrier
```

waits until all work items in the workgroup have reached this point
and all memory accesses made by these work items prior to the op are
visible to all work items in the workgroup. Data hazards between work items
accessing the same memory can be avoided by synchronizing work items
in-between these accesses.

Either none or all work items of a workgroup need to execute this op
in convergence.
"""
function barrier(; location=Location())
    results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    create_operation(
        "gpu.barrier", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`block_dim`

Returns the number of threads in the thread block (aka the block size) along
the x, y, or z `dimension`.

# Example

```mlir
%bDimX = gpu.block_dim x
```
"""
function block_dim(; result_0=nothing::Union{Nothing,IR.Type}, dimension, location=Location())
    results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("dimension", dimension),]
    !isnothing(result_0) && push!(results, result_0)

    create_operation(
        "gpu.block_dim", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`block_id`

Returns the block id, i.e. the index of the current block within the grid
along the x, y, or z `dimension`.

# Example

```mlir
%bIdY = gpu.block_id y
```
"""
function block_id(; result_0=nothing::Union{Nothing,IR.Type}, dimension, location=Location())
    results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("dimension", dimension),]
    !isnothing(result_0) && push!(results, result_0)

    create_operation(
        "gpu.block_id", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`dealloc`

The `gpu.dealloc` operation frees the region of memory referenced by a
memref which was originally created by the `gpu.alloc` operation. It is
similar to the `memref.dealloc` op, but supports asynchronous GPU execution.

The op does not execute before all async dependencies have finished
executing.

If the `async` keyword is present, the op is executed asynchronously (i.e.
it does not block until the execution has finished on the device). In
that case, it returns a !gpu.async.token.

# Example

```mlir
%token = gpu.dealloc async [%dep] %memref : memref<8x64xf32, 1>
```
"""
function dealloc(asyncDependencies::Vector{Value}, memref::Value; asyncToken=nothing::Union{Nothing,IR.Type}, location=Location())
    results = IR.Type[]
    operands = Value[asyncDependencies..., memref,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(asyncToken) && push!(results, asyncToken)

    create_operation(
        "gpu.dealloc", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`func`

Defines a function that can be executed on a GPU. This supports memory
attribution and its body has a particular execution model.

GPU functions are either kernels (as indicated by the `kernel` attribute) or
regular functions. The former can be launched from the host side, while the
latter are device side only.

The memory attribution defines SSA values that correspond to memory buffers
allocated in the memory hierarchy of the GPU (see below).

The operation has one attached region that corresponds to the body of the
function. The region arguments consist of the function arguments without
modification, followed by buffers defined in memory annotations. The body of
a GPU function, when launched, is executed by multiple work items. There are
no guarantees on the order in which work items execute, or on the connection
between them. In particular, work items are not necessarily executed in
lock-step. Synchronization ops such as \"gpu.barrier\" should be used to
coordinate work items. Declarations of GPU functions, i.e. not having the
body region, are not supported.

# Syntax

```
op ::= `gpu.func` symbol-ref-id `(` argument-list `)` (`->`
function-result-list)?
       memory-attribution `kernel`? function-attributes? region

memory-attribution ::= (`workgroup` `(` ssa-id-and-type-list `)`)?
                       (`private` `(` ssa-id-and-type-list `)`)?
```

# Example

```mlir
gpu.func @foo(%arg0: index)
    workgroup(%workgroup: memref<32xf32, 3>)
    private(%private: memref<1xf32, 5>)
    kernel
    attributes {qux: \"quux\"} {
  gpu.return
}
```

The generic form illustrates the concept

```mlir
\"gpu.func\"(%arg: index) {sym_name: \"foo\", kernel, qux: \"quux\"} ({
^bb0(%arg0: index, %workgroup: memref<32xf32, 3>,
     %private: memref<1xf32, 5>):
  \"gpu.return\"() : () -> ()
}) : (index) -> ()
```

Note the non-default memory spaces used in memref types in memory
attribution.
"""
function func(; body::Region, location=Location())
    results = IR.Type[]
    operands = Value[]
    owned_regions = Region[body,]
    successors = Block[]
    attributes = NamedAttribute[]

    create_operation(
        "gpu.func", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`module_`

GPU module contains code that is intended to be run on a GPU. A host device
can launch this code through a gpu.launc_func that creates a fully
qualified symbol through the gpu.module\'s symbol and a gpu.func symbol
contained in the gpu.module.

The module\'s top-level scope is modeled by a single region with a single
block. GPU modules are required to have a name that is used for symbol
resolution by the gpu.launch_func operation.

Using an op with a region to define a GPU module enables \"embedding\" GPU
modules with SIMT execution models in other dialects in a clean manner and
allows filtering of code regions to execute passes on only code intended to
or not intended to be run on the separate device.

```
  gpu.module @symbol_name {
  gpu.func {}
    ...
  gpu.module_end
}

```
"""
function module_(; body::Region, location=Location())
    results = IR.Type[]
    operands = Value[]
    owned_regions = Region[body,]
    successors = Block[]
    attributes = NamedAttribute[]

    create_operation(
        "gpu.module", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`grid_dim`

Returns the number of thread blocks in the grid along the x, y, or z
`dimension`.

# Example

```mlir
%gDimZ = gpu.grid_dim z
```
"""
function grid_dim(; result_0=nothing::Union{Nothing,IR.Type}, dimension, location=Location())
    results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("dimension", dimension),]
    !isnothing(result_0) && push!(results, result_0)

    create_operation(
        "gpu.grid_dim", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`host_register`

This op maps the provided host buffer into the device address space.

This operation may not be supported in every environment, there is not yet a
way to check at runtime whether this feature is supported.

Writes from the host are guaranteed to be visible to device kernels that are
launched afterwards. Writes from the device are guaranteed to be visible on
the host after synchronizing with the device kernel completion.
"""
function host_register(value::Value; location=Location())
    results = IR.Type[]
    operands = Value[value,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    create_operation(
        "gpu.host_register", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`launch_func`

Launch a kernel function on the specified grid of thread blocks.
`gpu.launch` operations are lowered to `gpu.launch_func` operations by
outlining the kernel body into a function in a dedicated module, which
reflects the separate compilation process. The kernel function is required
to have the `gpu.kernel` attribute. The module containing the kernel
function is required to be a gpu.module. And finally, the module containing
the kernel module (which thus cannot be the top-level module) is required
to have the `gpu.container_module` attribute. The `gpu.launch_func`
operation has a symbol attribute named `kernel` to identify the fully
specified kernel function to launch (both the gpu.module and func).

The `gpu.launch_func` supports async dependencies: the kernel does not start
executing until the ops producing those async dependencies have completed.

By the default, the host implicitly blocks until kernel execution has
completed. If the `async` keyword is present, the host does not block but
instead a `!gpu.async.token` is returned. Other async GPU ops can take this
token as dependency.

The operation requires at least the grid and block sizes along the x,y,z
dimensions as arguments. When a lower-dimensional kernel is required,
unused sizes must be explicitly set to `1`.

The remaining operands are optional. The first optional operand corresponds
to the amount of dynamic shared memory a kernel\'s workgroup should be
allocated; when this operand is not present, a zero size is assumed.

The remaining operands if present are passed as arguments to the kernel
function.

# Example

```mlir
module attributes {gpu.container_module} {

  // This module creates a separate compilation unit for the GPU compiler.
  gpu.module @kernels {
    func @kernel_1(%arg0 : f32, %arg1 : memref<?xf32, 1>)
        attributes { nvvm.kernel = true } {

      // Operations that produce block/thread IDs and dimensions are
      // injected when outlining the `gpu.launch` body to a function called
      // by `gpu.launch_func`.
      %tIdX = gpu.thread_id x
      %tIdY = gpu.thread_id y
      %tIdZ = gpu.thread_id z

      %bDimX = gpu.block_dim x
      %bDimY = gpu.block_dim y
      %bDimZ = gpu.block_dim z

      %bIdX = gpu.block_id x
      %bIdY = gpu.block_id y
      %bIdZ = gpu.block_id z

      %gDimX = gpu.grid_dim x
      %gDimY = gpu.grid_dim y
      %gDimZ = gpu.grid_dim z

      \"some_op\"(%bx, %tx) : (index, index) -> ()
      %42 = load %arg1[%bx] : memref<?xf32, 1>
    }
  }

  %t0 = gpu.wait async
  gpu.launch_func
      async                           // (Optional) Don\'t block host, return token.
      [%t0]                           // (Optional) Execute only after %t0 has completed.
      @kernels::@kernel_1             // Kernel function.
      blocks in (%cst, %cst, %cst)    // Grid size.
      threads in (%cst, %cst, %cst)   // Block size.
      dynamic_shared_memory_size %s   // (Optional) Amount of dynamic shared
                                      // memory to allocate for a workgroup.
      args(%arg0 : f32,               // (Optional) Kernel arguments.
           %arg1 : memref<?xf32, 1>)
}
```
"""
function launch_func(asyncDependencies::Vector{Value}, gridSizeX::Value, gridSizeY::Value, gridSizeZ::Value, blockSizeX::Value, blockSizeY::Value, blockSizeZ::Value, dynamicSharedMemorySize=nothing::Union{Nothing,Value}; operands::Vector{Value}, asyncToken=nothing::Union{Nothing,IR.Type}, kernel, location=Location())
    results = IR.Type[]
    operands = Value[asyncDependencies..., gridSizeX, gridSizeY, gridSizeZ, blockSizeX, blockSizeY, blockSizeZ, operands...,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("kernel", kernel),]
    !isnothing(dynamicSharedMemorySize) && push!(operands, dynamicSharedMemorySize)
    push!(attributes, operandsegmentsizes([length(asyncDependencies), 1, 1, 1, 1, 1, 1, (dynamicSharedMemorySize == nothing) ? 0 : 1length(operands),]))
    !isnothing(asyncToken) && push!(results, asyncToken)

    create_operation(
        "gpu.launch_func", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`launch`

Launch a kernel on the specified grid of thread blocks. The body of the
kernel is defined by the single region that this operation contains. The
operation takes six operands followed by an optional operand: the first
three operands are grid sizes along the x,y,z dimensions and the following
three are block sizes along the x,y,z dimensions. The last operand is
optional and corresponds to the amount of dynamic shared memory a kernel\'s
workgroup should be allocated; when this operand is not present, a zero size
is assumed.

When a lower-dimensional kernel is required, unused sizes must
be explicitly set to `1`.

The body region has _twelve_ arguments, grouped as follows:

-   three arguments that contain block identifiers along x,y,z dimensions;
-   three arguments that contain thread identifiers along x,y,z dimensions;
-   operands of the `gpu.launch` operation as is (i.e. the operands for
    grid and block sizes).

# Syntax

```
operation ::= `gpu.launch` `block` `(` ssa-id-list `)` `in` ssa-reassignment
                         `threads` `(` ssa-id-list `)` `in` ssa-reassignment
                         (dynamic_shared_memory_size ssa-use)?
                         region attr-dict?
ssa-reassignment ::= `(` ssa-id `=` ssa-use (`,` ssa-id `=` ssa-use)* `)`
```

# Example

```mlir
gpu.launch blocks(%bx, %by, %bz) in (%sz_bx = %0, %sz_by = %1, %sz_bz = %2)
           threads(%tx, %ty, %tz) in (%sz_tx = %3, %sz_ty = %4, %sz_tz = %5) {
  // Block and thread identifiers, as well as block/grid sizes are
  // immediately usable inside body region.
  \"some_op\"(%bx, %tx) : (index, index) -> ()
  // Assuming %val1 is defined outside the gpu.launch region.
  %42 = load %val1[%bx] : memref<?xf32, 1>
}

// Generic syntax explains how the pretty syntax maps to the IR structure.
\"gpu.launch\"(%cst, %cst, %c1,  // Grid sizes.
             %cst, %c1, %c1)   // Block sizes.

    {/*attributes*/}
    // All sizes and identifiers have \"index\" size.
    : (index, index, index, index, index, index) -> () {
// The operation passes block and thread identifiers, followed by grid and
// block sizes.
^bb0(%bx : index, %by : index, %bz : index,
     %tx : index, %ty : index, %tz : index,
     %num_bx : index, %num_by : index, %num_bz : index,
     %num_tx : index, %num_ty : index, %num_tz : index)
  \"some_op\"(%bx, %tx) : (index, index) -> ()
  %3 = \"memref.load\"(%val1, %bx) : (memref<?xf32, 1>, index) -> f32
}
```

Rationale: using operation/block arguments gives analyses a clear way of
understanding that a value has additional semantics (e.g., we will need to
know what value corresponds to threadIdx.x for coalescing). We can recover
these properties by analyzing the operations producing values, but it is
easier just to have that information by construction.
"""
function launch(gridSizeX::Value, gridSizeY::Value, gridSizeZ::Value, blockSizeX::Value, blockSizeY::Value, blockSizeZ::Value, dynamicSharedMemorySize=nothing::Union{Nothing,Value}; body::Region, location=Location())
    results = IR.Type[]
    operands = Value[gridSizeX, gridSizeY, gridSizeZ, blockSizeX, blockSizeY, blockSizeZ,]
    owned_regions = Region[body,]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(dynamicSharedMemorySize) && push!(operands, dynamicSharedMemorySize)

    create_operation(
        "gpu.launch", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`memcpy`

The `gpu.memcpy` operation copies the content of one memref to another.

The op does not execute before all async dependencies have finished
executing.

If the `async` keyword is present, the op is executed asynchronously (i.e.
it does not block until the execution has finished on the device). In
that case, it returns a !gpu.async.token.

# Example

```mlir
%token = gpu.memcpy async [%dep] %dst, %src : memref<?xf32, 1>, memref<?xf32>
```
"""
function memcpy(asyncDependencies::Vector{Value}, dst::Value, src::Value; asyncToken=nothing::Union{Nothing,IR.Type}, location=Location())
    results = IR.Type[]
    operands = Value[asyncDependencies..., dst, src,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(asyncToken) && push!(results, asyncToken)

    create_operation(
        "gpu.memcpy", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`memset`

The `gpu.memset` operation sets the content of memref to a scalar value.

The op does not execute before all async dependencies have finished
executing.

If the `async` keyword is present, the op is executed asynchronously (i.e.
it does not block until the execution has finished on the device). In
that case, it returns a !gpu.async.token.

# Example

```mlir
%token = gpu.memset async [%dep] %dst, %value : memref<?xf32, 1>, f32
```
"""
function memset(asyncDependencies::Vector{Value}, dst::Value, value::Value; asyncToken=nothing::Union{Nothing,IR.Type}, location=Location())
    results = IR.Type[]
    operands = Value[asyncDependencies..., dst, value,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(asyncToken) && push!(results, asyncToken)

    create_operation(
        "gpu.memset", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`module_end`

This op terminates the only block inside the only region of a `gpu.module`.
"""
function module_end(; location=Location())
    results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    create_operation(
        "gpu.module_end", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`num_subgroups`

Returns the number of subgroups within a workgroup.

# Example

```mlir
%numSg = gpu.num_subgroups : index
```
"""
function num_subgroups(; result=nothing::Union{Nothing,IR.Type}, location=Location())
    results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(results, result)

    create_operation(
        "gpu.num_subgroups", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`printf`

`gpu.printf` takes a literal format string `format` and an arbitrary number of
scalar arguments that should be printed.

The format string is a C-style printf string, subject to any restrictions
imposed by one\'s target platform.
"""
function printf(args::Vector{Value}; format, location=Location())
    results = IR.Type[]
    operands = Value[args...,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("format", format),]

    create_operation(
        "gpu.printf", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`return_`

A terminator operation for regions that appear in the body of  `gpu.func`
functions. The operands to the `gpu.return` are the result values returned
by an invocation of the `gpu.func`.
"""
function return_(operands::Vector{Value}; location=Location())
    results = IR.Type[]
    operands = Value[operands...,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    create_operation(
        "gpu.return", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`shuffle`

The \"shuffle\" op moves values to a different invocation within the same
subgroup.

# Example

```mlir
%1, %2 = gpu.shuffle %0, %offset, %width xor : f32
```

For lane k returns the value from lane `k ^ offset` and `true` if that lane
is smaller than %width. Otherwise it returns an unspecified value and
`false`. A lane is the index of an invocation relative to its subgroup.

The width specifies the number of invocations that participate in the
shuffle. The width needs to be the same for all invocations that participate
in the shuffle. Exactly the first `width` invocations of a subgroup need to
execute this op in convergence.
"""
function shuffle(value::Value, offset::Value, width::Value; result=nothing::Union{Nothing,IR.Type}, valid=nothing::Union{Nothing,IR.Type}, mode, location=Location())
    results = IR.Type[]
    operands = Value[value, offset, width,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("mode", mode),]
    !isnothing(result) && push!(results, result)
    !isnothing(valid) && push!(results, valid)

    create_operation(
        "gpu.shuffle", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`subgroup_id`

Returns the subgroup id, i.e. the index of the current subgroup within the
workgroup.

# Example

```mlir
%sgId = gpu.subgroup_id : index
```
"""
function subgroup_id(; result=nothing::Union{Nothing,IR.Type}, location=Location())
    results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(results, result)

    create_operation(
        "gpu.subgroup_id", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`subgroup_mma_compute`

The `gpu.subgroup_mma_compute` operation performs a matrix-multiply accumulate (mma)
operation using all the threads in a subgroup.

This operation takes three `!gpu.mma_matrix`s as arguments: these hold `A`,
 `B` and `C`operands for the mma operation. The operation performed is represented
as `C += A * B`. The op returns a `!gpu.mma_matrix` which contains the result of
the operation held by all threads in a subgroup.

This op is meant to be used along with `gpu.subgroup_mma_store_matrix` and
`gpu.subgroup_mma_load_matrix` ops.

# Example

```mlir
%D = gpu.subgroup_mma_compute_matrix %A, %B, %C :
  !gpu.mma_matrix<16x16xf16, \"AOp\">, !gpu.mma_matrix<16x16xf16, \"BOp\">>
  -> !gpu.mma_matrix<16x16xf16, \"COp\">
```
"""
function subgroup_mma_compute(opA::Value, opB::Value, opC::Value; res=nothing::Union{Nothing,IR.Type}, location=Location())
    results = IR.Type[]
    operands = Value[opA, opB, opC,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(results, res)

    create_operation(
        "gpu.subgroup_mma_compute", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`subgroup_mma_constant_matrix`

The `gpu.subgroup_mma_constant_matrix` creates a `!gpu.mma_matrix` with
constant elements.

The operation takes a scalar input and return a `!gpu.mma_matrix` where each
element of is equal to the operand constant. The destination mma_matrix type
must have elememt type equal to the constant type. Since the layout of
`!gpu.mma_matrix` is opaque this only support setting all the elements to
the same value.

This op is meant to be used along with `gpu.subgroup_mma_compute`.

# Example

```mlir
 %0 = gpu.subgroup_mma_constant_matrix %a :
   !gpu.mma_matrix<16x16xf16, \"AOp\">
 %1 = gpu.subgroup_mma_constant_matrix %b :
   !gpu.mma_matrix<16x16xf32, \"COp\">
```
"""
function subgroup_mma_constant_matrix(value::Value; res::IR.Type, location=Location())
    results = IR.Type[res,]
    operands = Value[value,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    create_operation(
        "gpu.subgroup_mma_constant_matrix", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`subgroup_mma_elementwise`

The `gpu.subgroup_mma_elementwise` takes `!gpu.mma_matrix` inputs and
compute a new `!gpu.mma_matrix` by applying an elementwise operation to each
element.

Since the operation is elementwise and the matrix type must match, the
matrix elements are processed independently of the matrix layout.

This op is meant to be used along with `gpu.subgroup_mma_compute`.

# Example

```mlir
 %0 =  %A, %B { operation = \"ADD\" } :
  (!gpu.mma_matrix<16x16xf16, \"COp\">, !gpu.mma_matrix<16x16xf16, \"COp\">)
  -> !gpu.mma_matrix<16x16xf16, \"COp\">
```
"""
function subgroup_mma_elementwise(args::Vector{Value}; res::IR.Type, operation, location=Location())
    results = IR.Type[res,]
    operands = Value[args...,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("operation", operation),]

    create_operation(
        "gpu.subgroup_mma_elementwise", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`subgroup_mma_load_matrix`

The `gpu.subgroup_mma_load_matrix` operation loads a matrix collectively
using all the threads in a subgroup.

This operation takes a memref as its first operand: it is the source matrix
from which data is to be loaded. The op returns a `!gpu.mma_matrix`. The
source memref can be in global memory or shared memory. The load address is
determined using `indices`. The matrix being loaded into is the result.  The
`leadDimension` attribute specifies the leading dimension size of the source
matrix which eventually allows the lowering to determine the size of each
row.

This op is often meant to be used along with `gpu.subgroup_mma_store_matrix` and
`gpu.subgroup_mma_compute`.

# Example

```mlir
 %0 = gpu.subgroup_mma_load_matrix src[%i,%j] : {leadDimension = 32 : i32}
      : memref<32x32xf16, 3>, !gpu.mma_matrix<16x16xf16, \"AOp\">
```
"""
function subgroup_mma_load_matrix(srcMemref::Value, indices::Vector{Value}; res::IR.Type, leadDimension, location=Location())
    results = IR.Type[res,]
    operands = Value[srcMemref, indices...,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("leadDimension", leadDimension),]

    create_operation(
        "gpu.subgroup_mma_load_matrix", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`subgroup_mma_store_matrix`

The `gpu.subgroup_mma_store_matrix` operation stores a matrix collectively
using all the threads in a subgroup.

This operation takes a `!gpu.mma_matrix` and a memref as operands.
`!gpu.mma_matrix` is the source value containing the data to be stored into the
destination memref which can be in global or shared memory.  The store address
is determined using the indices provided. The `leadDimension` attribute
specifies the leading dimension of the destination matrix.

This op is often meant to be used along with `gpu.subgroup_mma_load_matrix` and
`gpu.subgroup_mma_compute`.

# Example

```mlir
gpu.subgroup_mma_store_matrix %D, %sg[%i,%j] : { leadDimension = 32 : i32}
                : !gpu.mma_matrix<16x16xf16, \"COp\">, memref<32x32xf16, 3>
```
"""
function subgroup_mma_store_matrix(src::Value, dstMemref::Value, indices::Vector{Value}; leadDimension, location=Location())
    results = IR.Type[]
    operands = Value[src, dstMemref, indices...,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("leadDimension", leadDimension),]

    create_operation(
        "gpu.subgroup_mma_store_matrix", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`subgroup_size`

Returns the number of threads within a subgroup.

# Example

```mlir
%sgSz = gpu.subgroup_size : index
```
"""
function subgroup_size(; result=nothing::Union{Nothing,IR.Type}, location=Location())
    results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(results, result)

    create_operation(
        "gpu.subgroup_size", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`terminator`

A terminator operation for regions that appear in the body of `gpu.launch`
operation.  These regions are not expected to return any value so the
terminator takes no operands.
"""
function terminator(; location=Location())
    results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    create_operation(
        "gpu.terminator", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`thread_id`

Returns the thread id, i.e. the index of the current thread within the block
along the x, y, or z `dimension`.

# Example

```mlir
%tIdX = gpu.thread_id x
```
"""
function thread_id(; result_0=nothing::Union{Nothing,IR.Type}, dimension, location=Location())
    results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("dimension", dimension),]
    !isnothing(result_0) && push!(results, result_0)

    create_operation(
        "gpu.thread_id", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`wait`

This op synchronizes the host or the device with a list of dependent ops.

If the op contains the `async` keyword, it returns a new async token which
is synchronized with the op arguments. This new token is merely a shortcut
to the argument list, and one could replace the uses of the result with the
arguments for the same effect. The async version of this op is primarily
used to make each async token have a single use during lowering and
thereby make forks in async execution explicit. Example usage:

```mlir
%t0 = gpu.foo async : !gpu.async.token
%t1 = gpu.bar async : !gpu.async.token
%t2 = gpu.wait async [%t0, %t1]
// gpu.baz doesn\'t run until gpu.foo and gpu.bar have both completed, just
// as if the async dependencies were [%t0, %t1].
%t3 = gpu.baz async [%t2]
```

If the op does not contain the `async` keyword, it does not return a new
async token but blocks until all ops producing the async dependency tokens
finished execution. All dependent memory operations are visible to the host
once this op completes. Example usage:

```mlir
%t0 = gpu.foo async : !gpu.async.token
%t1 = gpu.bar async : !gpu.async.token
// The gpu.wait op blocks until gpu.foo and gpu.bar have completed.
gpu.wait [%t0, %t1]
```
"""
function wait(asyncDependencies::Vector{Value}; asyncToken=nothing::Union{Nothing,IR.Type}, location=Location())
    results = IR.Type[]
    operands = Value[asyncDependencies...,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(asyncToken) && push!(results, asyncToken)

    create_operation(
        "gpu.wait", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`yield`

gpu.yield` is a special terminator operation for blocks inside regions
in gpu ops. It returns values to the immediately enclosing gpu op.

# Example

```mlir
gpu.yield %f0, %f1 : f32, f32
```
"""
function yield(values::Vector{Value}; location=Location())
    results = IR.Type[]
    operands = Value[values...,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    create_operation(
        "gpu.yield", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

end # gpu
