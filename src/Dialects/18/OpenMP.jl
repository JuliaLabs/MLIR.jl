module omp

import ...IR:
    IR, NamedAttribute, Value, Location, Block, Region, Attribute, context, IndexType
import ..Dialects: namedattribute, operandsegmentsizes

"""
`atomic_capture`

This operation performs an atomic capture.

`hint` is the value of hint (as used in the hint clause). It is a compile
time constant. As the name suggests, this is just a hint for optimization.

`memory_order` indicates the memory ordering behavior of the construct. It
can be one of `seq_cst`, `acq_rel`, `release`, `acquire` or `relaxed`.

The region has the following allowed forms:

```
  omp.atomic.capture {
    omp.atomic.update ...
    omp.atomic.read ...
    omp.terminator
  }

  omp.atomic.capture {
    omp.atomic.read ...
    omp.atomic.update ...
    omp.terminator
  }

  omp.atomic.capture {
    omp.atomic.read ...
    omp.atomic.write ...
    omp.terminator
  }
```
"""
function atomic_capture(;
    hint_val=nothing, memory_order_val=nothing, region::Region, location=Location()
)
    _results = IR.Type[]
    _operands = Value[]
    _owned_regions = Region[region,]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(hint_val) && push!(_attributes, namedattribute("hint_val", hint_val))
    !isnothing(memory_order_val) &&
        push!(_attributes, namedattribute("memory_order_val", memory_order_val))

    return IR.create_operation(
        "omp.atomic.capture",
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
`atomic_read`

This operation performs an atomic read.

The operand `x` is the address from where the value is atomically read.
The operand `v` is the address where the value is stored after reading.

`hint` is the value of hint (as specified in the hint clause). It is a
compile time constant. As the name suggests, this is just a hint for
optimization.

`memory_order` indicates the memory ordering behavior of the construct. It
can be one of `seq_cst`, `acquire` or `relaxed`.
"""
function atomic_read(
    x::Value,
    v::Value;
    element_type,
    hint_val=nothing,
    memory_order_val=nothing,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[x, v]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[namedattribute("element_type", element_type),]
    !isnothing(hint_val) && push!(_attributes, namedattribute("hint_val", hint_val))
    !isnothing(memory_order_val) &&
        push!(_attributes, namedattribute("memory_order_val", memory_order_val))

    return IR.create_operation(
        "omp.atomic.read",
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
`atomic_update`

This operation performs an atomic update.

The operand `x` is exactly the same as the operand `x` in the OpenMP
Standard (OpenMP 5.0, section 2.17.7). It is the address of the variable
that is being updated. `x` is atomically read/written.

`hint` is the value of hint (as used in the hint clause). It is a compile
time constant. As the name suggests, this is just a hint for optimization.

`memory_order` indicates the memory ordering behavior of the construct. It
can be one of `seq_cst`, `release` or `relaxed`.

The region describes how to update the value of `x`. It takes the value at
`x` as an input and must yield the updated value. Only the update to `x` is
atomic. Generally the region must have only one instruction, but can
potentially have more than one instructions too. The update is sematically
similar to a compare-exchange loop based atomic update.

The syntax of atomic update operation is different from atomic read and
atomic write operations. This is because only the host dialect knows how to
appropriately update a value. For example, while generating LLVM IR, if
there are no special `atomicrmw` instructions for the operation-type
combination in atomic update, a compare-exchange loop is generated, where
the core update operation is directly translated like regular operations by
the host dialect. The front-end must handle semantic checks for allowed
operations.
"""
function atomic_update(
    x::Value;
    hint_val=nothing,
    memory_order_val=nothing,
    region::Region,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[x,]
    _owned_regions = Region[region,]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(hint_val) && push!(_attributes, namedattribute("hint_val", hint_val))
    !isnothing(memory_order_val) &&
        push!(_attributes, namedattribute("memory_order_val", memory_order_val))

    return IR.create_operation(
        "omp.atomic.update",
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
`atomic_write`

This operation performs an atomic write.

The operand `x` is the address to where the `expr` is atomically
written w.r.t. multiple threads. The evaluation of `expr` need not be
atomic w.r.t. the write to address. In general, the type(x) must
dereference to type(expr).

`hint` is the value of hint (as specified in the hint clause). It is a
compile time constant. As the name suggests, this is just a hint for
optimization.

`memory_order` indicates the memory ordering behavior of the construct. It
can be one of `seq_cst`, `release` or `relaxed`.
"""
function atomic_write(
    x::Value, expr::Value; hint_val=nothing, memory_order_val=nothing, location=Location()
)
    _results = IR.Type[]
    _operands = Value[x, expr]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(hint_val) && push!(_attributes, namedattribute("hint_val", hint_val))
    !isnothing(memory_order_val) &&
        push!(_attributes, namedattribute("memory_order_val", memory_order_val))

    return IR.create_operation(
        "omp.atomic.write",
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
`barrier`

The barrier construct specifies an explicit barrier at the point at which
the construct appears.
"""
function barrier(; location=Location())
    _results = IR.Type[]
    _operands = Value[]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "omp.barrier",
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
`cancel`

The cancel construct activates cancellation of the innermost enclosing
region of the type specified.
"""
function cancel(
    if_expr=nothing::Union{Nothing,Value};
    cancellation_construct_type_val,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[namedattribute(
        "cancellation_construct_type_val", cancellation_construct_type_val
    ),]
    !isnothing(if_expr) && push!(_operands, if_expr)

    return IR.create_operation(
        "omp.cancel",
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
`cancellationpoint`

The cancellation point construct introduces a user-defined cancellation
point at which implicit or explicit tasks check if cancellation of the
innermost enclosing region of the type specified has been activated.
"""
function cancellationpoint(; cancellation_construct_type_val, location=Location())
    _results = IR.Type[]
    _operands = Value[]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[namedattribute(
        "cancellation_construct_type_val", cancellation_construct_type_val
    ),]

    return IR.create_operation(
        "omp.cancellationpoint",
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
`critical_declare`

Declares a named critical section.

The name can be used in critical constructs in the dialect.
"""
function critical_declare(; sym_name, hint_val=nothing, location=Location())
    _results = IR.Type[]
    _operands = Value[]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[namedattribute("sym_name", sym_name),]
    !isnothing(hint_val) && push!(_attributes, namedattribute("hint_val", hint_val))

    return IR.create_operation(
        "omp.critical.declare",
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
`critical`

The critical construct imposes a restriction on the associated structured
block (region) to be executed by only a single thread at a time.
"""
function critical(; name=nothing, region::Region, location=Location())
    _results = IR.Type[]
    _operands = Value[]
    _owned_regions = Region[region,]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(name) && push!(_attributes, namedattribute("name", name))

    return IR.create_operation(
        "omp.critical",
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
`bounds`

This operation is a variation on the OpenACC dialects DataBoundsOp. Within
the OpenMP dialect it stores the bounds/range of data to be mapped to a
device specified by map clauses on target directives. Within,
the OpenMP dialect the DataBoundsOp is associated with MapInfoOp,
helping to store bounds information for the mapped variable.

It is used to support OpenMP array sectioning, Fortran pointer and
allocatable mapping and pointer/allocatable member of derived types.
In all cases the DataBoundsOp holds information on the section of
data to be mapped. Such as the upper bound and lower bound of the
section of data to be mapped. This information is currently
utilised by the LLVM-IR lowering to help generate instructions to
copy data to and from the device when processing target operations.

The example below copys a section of a 10-element array; all except the
first element, utilising OpenMP array sectioning syntax where array
subscripts are provided to specify the bounds to be mapped to device.
To simplify the examples, the constants are used directly, in reality
they will be MLIR SSA values.

C++:
```
int array[10];
#pragma target map(array[1:9])
```
=>
```mlir
omp.bounds lower_bound(1) upper_bound(9) extent(9) start_idx(0)
```

Fortran:
```
integer :: array(1:10)
!\$target map(array(2:10))
```
=>
```mlir
omp.bounds lower_bound(1) upper_bound(9) extent(9) start_idx(1)
```

For Fortran pointers and allocatables (as well as those that are
members of derived types) the bounds information is provided by
the Fortran compiler and runtime through descriptor information.

A basic pointer example can be found below (constants again
provided for simplicity, where in reality SSA values will be
used, in this case that point to data yielded by Fortran\'s
descriptors):

Fortran:
```
integer, pointer :: ptr(:)
allocate(ptr(10))
!\$target map(ptr)
```
=>
```mlir
omp.bounds lower_bound(0) upper_bound(9) extent(10) start_idx(1)
```

This operation records the bounds information in a normalized fashion
(zero-based). This works well with the `PointerLikeType`
requirement in data clauses - since a `lower_bound` of 0 means looking
at data at the zero offset from pointer.

This operation must have an `upper_bound` or `extent` (or both are allowed -
but not checked for consistency). When the source language\'s arrays are
not zero-based, the `start_idx` must specify the zero-position index.
"""
function bounds(
    lower_bound=nothing::Union{Nothing,Value};
    upper_bound=nothing::Union{Nothing,Value},
    extent=nothing::Union{Nothing,Value},
    stride=nothing::Union{Nothing,Value},
    start_idx=nothing::Union{Nothing,Value},
    result::IR.Type,
    stride_in_bytes=nothing,
    location=Location(),
)
    _results = IR.Type[result,]
    _operands = Value[]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(lower_bound) && push!(_operands, lower_bound)
    !isnothing(upper_bound) && push!(_operands, upper_bound)
    !isnothing(extent) && push!(_operands, extent)
    !isnothing(stride) && push!(_operands, stride)
    !isnothing(start_idx) && push!(_operands, start_idx)
    push!(
        _attributes,
        operandsegmentsizes([
            isnothing(lower_bound) ? 0 : 1,
            isnothing(upper_bound) ? 0 : 1,
            isnothing(extent) ? 0 : 1,
            isnothing(stride) ? 0 : 1,
            isnothing(start_idx) ? 0 : 1,
        ]),
    )
    !isnothing(stride_in_bytes) &&
        push!(_attributes, namedattribute("stride_in_bytes", stride_in_bytes))

    return IR.create_operation(
        "omp.bounds",
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
`flush`

The flush construct executes the OpenMP flush operation. This operation
makes a thread’s temporary view of memory consistent with memory and
enforces an order on the memory operations of the variables explicitly
specified or implied.
"""
function flush(varList::Vector{Value}; location=Location())
    _results = IR.Type[]
    _operands = Value[varList...,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "omp.flush",
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
`map_info`

The MapInfoOp captures information relating to individual OpenMP map clauses
that are applied to certain OpenMP directives such as Target and Target Data.

For example, the map type modifier; such as from, tofrom and to, the variable
being captured or the bounds of an array section being mapped.

It can be used to capture both implicit and explicit map information, where
explicit is an argument directly specified to an OpenMP map clause or implicit
where a variable is utilised in a target region but is defined externally to
the target region.

This map information is later used to aid the lowering of the target operations
they are attached to providing argument input and output context for kernels
generated or the target data mapping environment.

Example (Fortran):

```
integer :: index
!\$target map(to: index)
```
=>
```mlir
omp.map_info var_ptr(%index_ssa) map_type(to) map_capture_type(ByRef)
  name(index)
```

Description of arguments:
- `var_ptr`: The address of variable to copy.
- `var_type`: The type of the variable to copy.
- `var_ptr_ptr`: Used when the variable copied is a member of a class, structure
  or derived type and refers to the originating struct.
- `bounds`: Used when copying slices of array\'s, pointers or pointer members of
  objects (e.g. derived types or classes), indicates the bounds to be copied
  of the variable. When it\'s an array slice it is in rank order where rank 0
  is the inner-most dimension.
- \'map_clauses\': OpenMP map type for this map capture, for example: from, to and
  always. It\'s a bitfield composed of the OpenMP runtime flags stored in
  OpenMPOffloadMappingFlags.
- \'map_capture_type\': Capture type for the variable e.g. this, byref, byvalue, byvla
   this can affect how the variable is lowered.
- `name`: Holds the name of variable as specified in user clause (including bounds).
"""
function map_info(
    var_ptr::Value,
    var_ptr_ptr=nothing::Union{Nothing,Value};
    bounds::Vector{Value},
    omp_ptr::IR.Type,
    var_type,
    map_type=nothing,
    map_capture_type=nothing,
    name=nothing,
    location=Location(),
)
    _results = IR.Type[omp_ptr,]
    _operands = Value[var_ptr, bounds...]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[namedattribute("var_type", var_type),]
    !isnothing(var_ptr_ptr) && push!(_operands, var_ptr_ptr)
    push!(
        _attributes,
        operandsegmentsizes([1, isnothing(var_ptr_ptr) ? 0 : 1, length(bounds)]),
    )
    !isnothing(map_type) && push!(_attributes, namedattribute("map_type", map_type))
    !isnothing(map_capture_type) &&
        push!(_attributes, namedattribute("map_capture_type", map_capture_type))
    !isnothing(name) && push!(_attributes, namedattribute("name", name))

    return IR.create_operation(
        "omp.map_info",
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
`master`

The master construct specifies a structured block that is executed by
the master thread of the team.
"""
function master(; region::Region, location=Location())
    _results = IR.Type[]
    _operands = Value[]
    _owned_regions = Region[region,]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "omp.master",
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
`ordered`

The ordered construct without region is a stand-alone directive that
specifies cross-iteration dependences in a doacross loop nest.

The `depend_type_val` attribute refers to either the DEPEND(SOURCE) clause
or the DEPEND(SINK: vec) clause.

The `num_loops_val` attribute specifies the number of loops in the doacross
nest.

The `depend_vec_vars` is a variadic list of operands that specifies the index
of the loop iterator in the doacross nest for the DEPEND(SOURCE) clause or
the index of the element of \"vec\" for the DEPEND(SINK: vec) clause. It
contains the operands in multiple \"vec\" when multiple DEPEND(SINK: vec)
clauses exist in one ORDERED directive.
"""
function ordered(
    depend_vec_vars::Vector{Value};
    depend_type_val=nothing,
    num_loops_val=nothing,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[depend_vec_vars...,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(depend_type_val) &&
        push!(_attributes, namedattribute("depend_type_val", depend_type_val))
    !isnothing(num_loops_val) &&
        push!(_attributes, namedattribute("num_loops_val", num_loops_val))

    return IR.create_operation(
        "omp.ordered",
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
`ordered_region`

The ordered construct with region specifies a structured block in a
worksharing-loop, SIMD, or worksharing-loop SIMD region that is executed in
the order of the loop iterations.

The `simd` attribute corresponds to the SIMD clause specified. If it is not
present, it behaves as if the THREADS clause is specified or no clause is
specified.
"""
function ordered_region(; simd=nothing, region::Region, location=Location())
    _results = IR.Type[]
    _operands = Value[]
    _owned_regions = Region[region,]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(simd) && push!(_attributes, namedattribute("simd", simd))

    return IR.create_operation(
        "omp.ordered_region",
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
`parallel`

The parallel construct includes a region of code which is to be executed
by a team of threads.

The optional \$if_expr_var parameter specifies a boolean result of a
conditional check. If this value is 1 or is not provided then the parallel
region runs as normal, if it is 0 then the parallel region is executed with
one thread.

The optional \$num_threads_var parameter specifies the number of threads which
should be used to execute the parallel region.

The \$allocators_vars and \$allocate_vars parameters are a variadic list of values
that specify the memory allocator to be used to obtain storage for private values.

Reductions can be performed in a parallel construct by specifying reduction
accumulator variables in `reduction_vars` and symbols referring to reduction
declarations in the `reductions` attribute. Each reduction is identified
by the accumulator it uses and accumulators must not be repeated in the same
reduction. The `omp.reduction` operation accepts the accumulator and a
partial value which is considered to be produced by the thread for the
given reduction. If multiple values are produced for the same accumulator,
i.e. there are multiple `omp.reduction`s, the last value is taken. The
reduction declaration specifies how to combine the values from each thread
into the final value, which is available in the accumulator after all the
threads complete.

The optional \$proc_bind_val attribute controls the thread affinity for the execution
of the parallel region.
"""
function parallel(
    if_expr_var=nothing::Union{Nothing,Value};
    num_threads_var=nothing::Union{Nothing,Value},
    allocate_vars::Vector{Value},
    allocators_vars::Vector{Value},
    reduction_vars::Vector{Value},
    reductions=nothing,
    proc_bind_val=nothing,
    region::Region,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[allocate_vars..., allocators_vars..., reduction_vars...]
    _owned_regions = Region[region,]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(if_expr_var) && push!(_operands, if_expr_var)
    !isnothing(num_threads_var) && push!(_operands, num_threads_var)
    push!(
        _attributes,
        operandsegmentsizes([
            isnothing(if_expr_var) ? 0 : 1,
            isnothing(num_threads_var) ? 0 : 1,
            length(allocate_vars),
            length(allocators_vars),
            length(reduction_vars),
        ]),
    )
    !isnothing(reductions) && push!(_attributes, namedattribute("reductions", reductions))
    !isnothing(proc_bind_val) &&
        push!(_attributes, namedattribute("proc_bind_val", proc_bind_val))

    return IR.create_operation(
        "omp.parallel",
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
`reduction_declare`

Declares an OpenMP reduction kind. This requires two mandatory and one
optional region.

  1. The initializer region specifies how to initialize the thread-local
     reduction value. This is usually the neutral element of the reduction.
     For convenience, the region has an argument that contains the value
     of the reduction accumulator at the start of the reduction. It is
     expected to `omp.yield` the new value on all control flow paths.
  2. The reduction region specifies how to combine two values into one, i.e.
     the reduction operator. It accepts the two values as arguments and is
     expected to `omp.yield` the combined value on all control flow paths.
  3. The atomic reduction region is optional and specifies how two values
     can be combined atomically given local accumulator variables. It is
     expected to store the combined value in the first accumulator variable.

Note that the MLIR type system does not allow for type-polymorphic
reductions. Separate reduction declarations should be created for different
element and accumulator types.

For initializer and reduction regions, the operand to `omp.yield` must
match the parent operation\'s results.
"""
function reduction_declare(;
    sym_name,
    type,
    initializerRegion::Region,
    reductionRegion::Region,
    atomicReductionRegion::Region,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[]
    _owned_regions = Region[initializerRegion, reductionRegion, atomicReductionRegion]
    _successors = Block[]
    _attributes = NamedAttribute[
        namedattribute("sym_name", sym_name), namedattribute("type", type)
    ]

    return IR.create_operation(
        "omp.reduction.declare",
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
`reduction`

Indicates the value that is produced by the current reduction-participating
entity for a reduction requested in some ancestor. The reduction is
identified by the accumulator, but the value of the accumulator may not be
updated immediately.
"""
function reduction(operand::Value, accumulator::Value; location=Location())
    _results = IR.Type[]
    _operands = Value[operand, accumulator]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "omp.reduction",
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
`section`

A section operation encloses a region which represents one section in a
sections construct. A section op should always be surrounded by an
`omp.sections` operation.
"""
function section(; region::Region, location=Location())
    _results = IR.Type[]
    _operands = Value[]
    _owned_regions = Region[region,]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "omp.section",
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
`sections`

The sections construct is a non-iterative worksharing construct that
contains `omp.section` operations. The `omp.section` operations are to be
distributed among and executed by the threads in a team. Each `omp.section`
is executed once by one of the threads in the team in the context of its
implicit task.

Reductions can be performed in a sections construct by specifying reduction
accumulator variables in `reduction_vars` and symbols referring to reduction
declarations in the `reductions` attribute. Each reduction is identified
by the accumulator it uses and accumulators must not be repeated in the same
reduction. The `omp.reduction` operation accepts the accumulator and a
partial value which is considered to be produced by the section for the
given reduction. If multiple values are produced for the same accumulator,
i.e. there are multiple `omp.reduction`s, the last value is taken. The
reduction declaration specifies how to combine the values from each section
into the final value, which is available in the accumulator after all the
sections complete.

The \$allocators_vars and \$allocate_vars parameters are a variadic list of values
that specify the memory allocator to be used to obtain storage for private values.

The `nowait` attribute, when present, signifies that there should be no
implicit barrier at the end of the construct.
"""
function sections(
    reduction_vars::Vector{Value},
    allocate_vars::Vector{Value},
    allocators_vars::Vector{Value};
    reductions=nothing,
    nowait=nothing,
    region::Region,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[reduction_vars..., allocate_vars..., allocators_vars...]
    _owned_regions = Region[region,]
    _successors = Block[]
    _attributes = NamedAttribute[]
    push!(
        _attributes,
        operandsegmentsizes([
            length(reduction_vars), length(allocate_vars), length(allocators_vars)
        ]),
    )
    !isnothing(reductions) && push!(_attributes, namedattribute("reductions", reductions))
    !isnothing(nowait) && push!(_attributes, namedattribute("nowait", nowait))

    return IR.create_operation(
        "omp.sections",
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
`simdloop`

The simd construct can be applied to a loop to indicate that the loop can be
transformed into a SIMD loop (that is, multiple iterations of the loop can
be executed concurrently using SIMD instructions).. The lower and upper
bounds specify a half-open range: the range includes the lower bound but
does not include the upper bound. If the `inclusive` attribute is specified
then the upper bound is also included.

The body region can contain any number of blocks. The region is terminated
by \"omp.yield\" instruction without operands.

Collapsed loops are represented by the simd-loop having a list of indices,
bounds and steps where the size of the list is equal to the collapse value.

The `alignment_values` attribute additionally specifies alignment of each
corresponding aligned operand. Note that `\$aligned_vars` and
`alignment_values` should contain the same number of elements.

When an if clause is present and evaluates to false, the preferred number of
iterations to be executed concurrently is one, regardless of whether
a simdlen clause is speciﬁed.

The optional `nontemporal` attribute specifies variables which have low
temporal locality across the iterations where they are accessed.

The optional `order` attribute specifies which order the iterations of the
associate loops are executed in. Currently the only option for this
attribute is \"concurrent\".

When a simdlen clause is present, the preferred number of iterations to be
executed concurrently is the value provided to the simdlen clause.

The safelen clause specifies that no two concurrent iterations within a
SIMD chunk can have a distance in the logical iteration space that is
greater than or equal to the value given in the clause.
```
omp.simdloop <clauses>
for (%i1, %i2) : index = (%c0, %c0) to (%c10, %c10) step (%c1, %c1) {
  // block operations
  omp.yield
}
```
"""
function simdloop(
    lowerBound::Vector{Value},
    upperBound::Vector{Value},
    step::Vector{Value},
    aligned_vars::Vector{Value},
    if_expr=nothing::Union{Nothing,Value};
    nontemporal_vars::Vector{Value},
    alignment_values=nothing,
    order_val=nothing,
    simdlen=nothing,
    safelen=nothing,
    inclusive=nothing,
    region::Region,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[
        lowerBound..., upperBound..., step..., aligned_vars..., nontemporal_vars...
    ]
    _owned_regions = Region[region,]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(if_expr) && push!(_operands, if_expr)
    push!(
        _attributes,
        operandsegmentsizes([
            length(lowerBound),
            length(upperBound),
            length(step),
            length(aligned_vars),
            isnothing(if_expr) ? 0 : 1,
            length(nontemporal_vars),
        ]),
    )
    !isnothing(alignment_values) &&
        push!(_attributes, namedattribute("alignment_values", alignment_values))
    !isnothing(order_val) && push!(_attributes, namedattribute("order_val", order_val))
    !isnothing(simdlen) && push!(_attributes, namedattribute("simdlen", simdlen))
    !isnothing(safelen) && push!(_attributes, namedattribute("safelen", safelen))
    !isnothing(inclusive) && push!(_attributes, namedattribute("inclusive", inclusive))

    return IR.create_operation(
        "omp.simdloop",
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
`single`

The single construct specifies that the associated structured block is
executed by only one of the threads in the team (not necessarily the
master thread), in the context of its implicit task. The other threads
in the team, which do not execute the block, wait at an implicit barrier
at the end of the single construct unless a nowait clause is specified.
"""
function single(
    allocate_vars::Vector{Value},
    allocators_vars::Vector{Value};
    nowait=nothing,
    region::Region,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[allocate_vars..., allocators_vars...]
    _owned_regions = Region[region,]
    _successors = Block[]
    _attributes = NamedAttribute[]
    push!(
        _attributes, operandsegmentsizes([length(allocate_vars), length(allocators_vars)])
    )
    !isnothing(nowait) && push!(_attributes, namedattribute("nowait", nowait))

    return IR.create_operation(
        "omp.single",
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
`target`

The target construct includes a region of code which is to be executed
on a device.

The optional \$if_expr parameter specifies a boolean result of a
conditional check. If this value is 1 or is not provided then the target
region runs on a device, if it is 0 then the target region is executed on the
host device.

The optional \$device parameter specifies the device number for the target region.

The optional \$thread_limit specifies the limit on the number of threads

The optional \$nowait elliminates the implicit barrier so the parent task can make progress
even if the target task is not yet completed.

TODO:  is_device_ptr, depend, defaultmap, in_reduction
"""
function target(
    if_expr=nothing::Union{Nothing,Value};
    device=nothing::Union{Nothing,Value},
    thread_limit=nothing::Union{Nothing,Value},
    map_operands::Vector{Value},
    nowait=nothing,
    region::Region,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[map_operands...,]
    _owned_regions = Region[region,]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(if_expr) && push!(_operands, if_expr)
    !isnothing(device) && push!(_operands, device)
    !isnothing(thread_limit) && push!(_operands, thread_limit)
    push!(
        _attributes,
        operandsegmentsizes([
            isnothing(if_expr) ? 0 : 1,
            isnothing(device) ? 0 : 1,
            isnothing(thread_limit) ? 0 : 1,
            length(map_operands),
        ]),
    )
    !isnothing(nowait) && push!(_attributes, namedattribute("nowait", nowait))

    return IR.create_operation(
        "omp.target",
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
`target_data`

Map variables to a device data environment for the extent of the region.

The omp target data directive maps variables to a device data
environment, and defines the lexical scope of the data environment
that is created. The omp target data directive can reduce data copies
to and from the offloading device when multiple target regions are using
the same data.

The optional \$if_expr parameter specifies a boolean result of a
conditional check. If this value is 1 or is not provided then the target
region runs on a device, if it is 0 then the target region is executed
on the host device.

The optional \$device parameter specifies the device number for the target
region.

The optional \$use_device_ptr specifies the device pointers to the
corresponding list items in the device data environment.

The optional \$use_device_addr specifies the address of the objects in the
device data enviornment.

The \$map_operands specifies the locator-list operands of the map clause.

The \$map_types specifies the types and modifiers for the map clause.

TODO:  depend clause and map_type_modifier values iterator and mapper.
"""
function target_data(
    if_expr=nothing::Union{Nothing,Value};
    device=nothing::Union{Nothing,Value},
    use_device_ptr::Vector{Value},
    use_device_addr::Vector{Value},
    map_operands::Vector{Value},
    region::Region,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[use_device_ptr..., use_device_addr..., map_operands...]
    _owned_regions = Region[region,]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(if_expr) && push!(_operands, if_expr)
    !isnothing(device) && push!(_operands, device)
    push!(
        _attributes,
        operandsegmentsizes([
            isnothing(if_expr) ? 0 : 1,
            isnothing(device) ? 0 : 1,
            length(use_device_ptr),
            length(use_device_addr),
            length(map_operands),
        ]),
    )

    return IR.create_operation(
        "omp.target_data",
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
`target_enter_data`

The target enter data directive specifies that variables are mapped to
a device data environment. The target enter data directive is a
stand-alone directive.

The optional \$if_expr parameter specifies a boolean result of a
conditional check. If this value is 1 or is not provided then the target
region runs on a device, if it is 0 then the target region is executed on
the host device.

The optional \$device parameter specifies the device number for the
target region.

The optional \$nowait eliminates the implicit barrier so the parent task
can make progress even if the target task is not yet completed.

The \$map_operands specifies the locator-list operands of the map clause.

The \$map_types specifies the types and modifiers for the map clause.

TODO:  depend clause and map_type_modifier values iterator and mapper.
"""
function target_enter_data(
    if_expr=nothing::Union{Nothing,Value};
    device=nothing::Union{Nothing,Value},
    map_operands::Vector{Value},
    nowait=nothing,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[map_operands...,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(if_expr) && push!(_operands, if_expr)
    !isnothing(device) && push!(_operands, device)
    push!(
        _attributes,
        operandsegmentsizes([
            isnothing(if_expr) ? 0 : 1, isnothing(device) ? 0 : 1, length(map_operands)
        ]),
    )
    !isnothing(nowait) && push!(_attributes, namedattribute("nowait", nowait))

    return IR.create_operation(
        "omp.target_enter_data",
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
`target_exit_data`

The target exit data directive specifies that variables are mapped to a
device data environment. The target exit data directive is
a stand-alone directive.

The optional \$if_expr parameter specifies a boolean result of a
conditional check. If this value is 1 or is not provided then the target
region runs on a device, if it is 0 then the target region is executed
on the host device.

The optional \$device parameter specifies the device number for the
target region.

The optional \$nowait eliminates the implicit barrier so the parent
task can make progress even if the target task is not yet completed.

The \$map_operands specifies the locator-list operands of the map clause.

The \$map_types specifies the types and modifiers for the map clause.

TODO:  depend clause and map_type_modifier values iterator and mapper.
"""
function target_exit_data(
    if_expr=nothing::Union{Nothing,Value};
    device=nothing::Union{Nothing,Value},
    map_operands::Vector{Value},
    nowait=nothing,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[map_operands...,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(if_expr) && push!(_operands, if_expr)
    !isnothing(device) && push!(_operands, device)
    push!(
        _attributes,
        operandsegmentsizes([
            isnothing(if_expr) ? 0 : 1, isnothing(device) ? 0 : 1, length(map_operands)
        ]),
    )
    !isnothing(nowait) && push!(_attributes, namedattribute("nowait", nowait))

    return IR.create_operation(
        "omp.target_exit_data",
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
`target_update_data`

The target update directive makes the corresponding list items in the device
data environment consistent with their original list items, according to the
specified motion clauses. The target update construct is a stand-alone
directive.

The optional \$if_expr parameter specifies a boolean result of a
conditional check. If this value is 1 or is not provided then the target
region runs on a device, if it is 0 then the target region is executed
on the host device.

The optional \$device parameter specifies the device number for the
target region.

The optional \$nowait eliminates the implicit barrier so the parent
task can make progress even if the target task is not yet completed.

We use `MapInfoOp` to model the motion clauses and their modifiers. Even
though the spec differentiates between map-types & map-type-modifiers vs.
motion-clauses & motion-modifiers, the motion clauses and their modifiers are
a subset of map types and their modifiers. The subset relation is handled in
during verification to make sure the restrictions for target update are
respected.

TODO: depend clause
"""
function target_update_data(
    if_expr=nothing::Union{Nothing,Value};
    device=nothing::Union{Nothing,Value},
    motion_operands::Vector{Value},
    nowait=nothing,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[motion_operands...,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(if_expr) && push!(_operands, if_expr)
    !isnothing(device) && push!(_operands, device)
    push!(
        _attributes,
        operandsegmentsizes([
            isnothing(if_expr) ? 0 : 1, isnothing(device) ? 0 : 1, length(motion_operands)
        ]),
    )
    !isnothing(nowait) && push!(_attributes, namedattribute("nowait", nowait))

    return IR.create_operation(
        "omp.target_update_data",
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
`taskgroup`

The taskgroup construct specifies a wait on completion of child tasks of the
current task and their descendent tasks.

When a thread encounters a taskgroup construct, it starts executing the
region. All child tasks generated in the taskgroup region and all of their
descendants that bind to the same parallel region as the taskgroup region
are part of the taskgroup set associated with the taskgroup region. There is
an implicit task scheduling point at the end of the taskgroup region. The
current task is suspended at the task scheduling point until all tasks in
the taskgroup set complete execution.

The `task_reduction` clause specifies a reduction among tasks. For each list
item, the number of copies is unspecified. Any copies associated with the
reduction are initialized before they are accessed by the tasks
participating in the reduction. After the end of the region, the original
list item contains the result of the reduction.

The `allocators_vars` and `allocate_vars` arguments are a variadic list of
values that specify the memory allocator to be used to obtain storage for
private values.
"""
function taskgroup(
    task_reduction_vars::Vector{Value},
    allocate_vars::Vector{Value},
    allocators_vars::Vector{Value};
    task_reductions=nothing,
    region::Region,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[task_reduction_vars..., allocate_vars..., allocators_vars...]
    _owned_regions = Region[region,]
    _successors = Block[]
    _attributes = NamedAttribute[]
    push!(
        _attributes,
        operandsegmentsizes([
            length(task_reduction_vars), length(allocate_vars), length(allocators_vars)
        ]),
    )
    !isnothing(task_reductions) &&
        push!(_attributes, namedattribute("task_reductions", task_reductions))

    return IR.create_operation(
        "omp.taskgroup",
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
`taskloop`

The taskloop construct specifies that the iterations of one or more
associated loops will be executed in parallel using explicit tasks. The
iterations are distributed across tasks generated by the construct and
scheduled to be executed.

The `lowerBound` and `upperBound` specify a half-open range: the range
includes the lower bound but does not include the upper bound. If the
`inclusive` attribute is specified then the upper bound is also included.
The `step` specifies the loop step.

The body region can contain any number of blocks.

```
omp.taskloop <clauses>
for (%i1, %i2) : index = (%c0, %c0) to (%c10, %c10) step (%c1, %c1) {
  %a = load %arrA[%i1, %i2] : memref<?x?xf32>
  %b = load %arrB[%i1, %i2] : memref<?x?xf32>
  %sum = arith.addf %a, %b : f32
  store %sum, %arrC[%i1, %i2] : memref<?x?xf32>
  omp.terminator
}
```

For definitions of \"undeferred task\", \"included task\", \"final task\" and
\"mergeable task\", please check OpenMP Specification.

When an `if` clause is present on a taskloop construct, and if the `if`
clause expression evaluates to `false`, undeferred tasks are generated. The
use of a variable in an `if` clause expression of a taskloop construct
causes an implicit reference to the variable in all enclosing constructs.

When a `final` clause is present on a taskloop construct and the `final`
clause expression evaluates to `true`, the generated tasks will be final
tasks. The use of a variable in a `final` clause expression of a taskloop
construct causes an implicit reference to the variable in all enclosing
constructs.

If the `untied` clause is specified, all tasks generated by the taskloop
construct are untied tasks.

When the `mergeable` clause is present on a taskloop construct, each
generated task is a mergeable task.

Reductions can be performed in a loop by specifying reduction accumulator
variables in `reduction_vars` or `in_reduction_vars` and symbols referring
to reduction declarations in the `reductions` or `in_reductions` attribute.
Each reduction is identified by the accumulator it uses and accumulators
must not be repeated in the same reduction. The `omp.reduction` operation
accepts the accumulator and a partial value which is considered to be
produced by the current loop iteration for the given reduction. If multiple
values are produced for the same accumulator, i.e. there are multiple
`omp.reduction`s, the last value is taken. The reduction declaration
specifies how to combine the values from each iteration into the final
value, which is available in the accumulator after the loop completes.

If an `in_reduction` clause is present on the taskloop construct, the
behavior is as if each generated task was defined by a task construct on
which an `in_reduction` clause with the same reduction operator and list
items is present. Thus, the generated tasks are participants of a reduction
previously defined by a reduction scoping clause.

If a `reduction` clause is present on the taskloop construct, the behavior
is as if a `task_reduction` clause with the same reduction operator and list
items was applied to the implicit taskgroup construct enclosing the taskloop
construct. The taskloop construct executes as if each generated task was
defined by a task construct on which an `in_reduction` clause with the same
reduction operator and list items is present. Thus, the generated tasks are
participants of the reduction defined by the `task_reduction` clause that
was applied to the implicit taskgroup construct.

When a `priority` clause is present on a taskloop construct, the generated
tasks use the `priority-value` as if it was specified for each individual
task. If the `priority` clause is not specified, tasks generated by the
taskloop construct have the default task priority (zero).

The `allocators_vars` and `allocate_vars` arguments are a variadic list of
values that specify the memory allocator to be used to obtain storage for
private values.

If a `grainsize` clause is present on the taskloop construct, the number of
logical loop iterations assigned to each generated task is greater than or
equal to the minimum of the value of the grain-size expression and the
number of logical loop iterations, but less than two times the value of the
grain-size expression.

If `num_tasks` is specified, the taskloop construct creates as many tasks as
the minimum of the num-tasks expression and the number of logical loop
iterations. Each task must have at least one logical loop iteration.

By default, the taskloop construct executes as if it was enclosed in a
taskgroup construct with no statements or directives outside of the taskloop
construct. Thus, the taskloop construct creates an implicit taskgroup
region. If the `nogroup` clause is present, no implicit taskgroup region is
created.
"""
function taskloop(
    lowerBound::Vector{Value},
    upperBound::Vector{Value},
    step::Vector{Value},
    if_expr=nothing::Union{Nothing,Value};
    final_expr=nothing::Union{Nothing,Value},
    in_reduction_vars::Vector{Value},
    reduction_vars::Vector{Value},
    priority=nothing::Union{Nothing,Value},
    allocate_vars::Vector{Value},
    allocators_vars::Vector{Value},
    grain_size=nothing::Union{Nothing,Value},
    num_tasks=nothing::Union{Nothing,Value},
    inclusive=nothing,
    untied=nothing,
    mergeable=nothing,
    in_reductions=nothing,
    reductions=nothing,
    nogroup=nothing,
    region::Region,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[
        lowerBound...,
        upperBound...,
        step...,
        in_reduction_vars...,
        reduction_vars...,
        allocate_vars...,
        allocators_vars...,
    ]
    _owned_regions = Region[region,]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(if_expr) && push!(_operands, if_expr)
    !isnothing(final_expr) && push!(_operands, final_expr)
    !isnothing(priority) && push!(_operands, priority)
    !isnothing(grain_size) && push!(_operands, grain_size)
    !isnothing(num_tasks) && push!(_operands, num_tasks)
    push!(
        _attributes,
        operandsegmentsizes([
            length(lowerBound),
            length(upperBound),
            length(step),
            isnothing(if_expr) ? 0 : 1,
            isnothing(final_expr) ? 0 : 1,
            length(in_reduction_vars),
            length(reduction_vars),
            isnothing(priority) ? 0 : 1,
            length(allocate_vars),
            length(allocators_vars),
            isnothing(grain_size) ? 0 : 1,
            isnothing(num_tasks) ? 0 : 1,
        ]),
    )
    !isnothing(inclusive) && push!(_attributes, namedattribute("inclusive", inclusive))
    !isnothing(untied) && push!(_attributes, namedattribute("untied", untied))
    !isnothing(mergeable) && push!(_attributes, namedattribute("mergeable", mergeable))
    !isnothing(in_reductions) &&
        push!(_attributes, namedattribute("in_reductions", in_reductions))
    !isnothing(reductions) && push!(_attributes, namedattribute("reductions", reductions))
    !isnothing(nogroup) && push!(_attributes, namedattribute("nogroup", nogroup))

    return IR.create_operation(
        "omp.taskloop",
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
`task`

The task construct defines an explicit task.

For definitions of \"undeferred task\", \"included task\", \"final task\" and
\"mergeable task\", please check OpenMP Specification.

When an `if` clause is present on a task construct, and the value of
`if_expr` evaluates to `false`, an \"undeferred task\" is generated, and the
encountering thread must suspend the current task region, for which
execution cannot be resumed until execution of the structured block that is
associated with the generated task is completed.

When a `final` clause is present on a task construct and the `final_expr`
evaluates to `true`, the generated task will be a \"final task\". All task
constructs encountered during execution of a final task will generate final
and included tasks.

If the `untied` clause is present on a task construct, any thread in the
team can resume the task region after a suspension. The `untied` clause is
ignored if a `final` clause is present on the same task construct and the
`final_expr` evaluates to `true`, or if a task is an included task.

When the `mergeable` clause is present on a task construct, the generated
task is a \"mergeable task\".

The `in_reduction` clause specifies that this particular task (among all the
tasks in current taskgroup, if any) participates in a reduction.

The `priority` clause is a hint for the priority of the generated task.
The `priority` is a non-negative integer expression that provides a hint for
task execution order. Among all tasks ready to be executed, higher priority
tasks (those with a higher numerical value in the priority clause
expression) are recommended to execute before lower priority ones. The
default priority-value when no priority clause is specified should be
assumed to be zero (the lowest priority).

The `depends` and `depend_vars` arguments are variadic lists of values
that specify the dependencies of this particular task in relation to
other tasks.

The `allocators_vars` and `allocate_vars` arguments are a variadic list of
values that specify the memory allocator to be used to obtain storage for
private values.
"""
function task(
    if_expr=nothing::Union{Nothing,Value};
    final_expr=nothing::Union{Nothing,Value},
    in_reduction_vars::Vector{Value},
    priority=nothing::Union{Nothing,Value},
    depend_vars::Vector{Value},
    allocate_vars::Vector{Value},
    allocators_vars::Vector{Value},
    untied=nothing,
    mergeable=nothing,
    in_reductions=nothing,
    depends=nothing,
    region::Region,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[
        in_reduction_vars..., depend_vars..., allocate_vars..., allocators_vars...
    ]
    _owned_regions = Region[region,]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(if_expr) && push!(_operands, if_expr)
    !isnothing(final_expr) && push!(_operands, final_expr)
    !isnothing(priority) && push!(_operands, priority)
    push!(
        _attributes,
        operandsegmentsizes([
            isnothing(if_expr) ? 0 : 1,
            isnothing(final_expr) ? 0 : 1,
            length(in_reduction_vars),
            isnothing(priority) ? 0 : 1,
            length(depend_vars),
            length(allocate_vars),
            length(allocators_vars),
        ]),
    )
    !isnothing(untied) && push!(_attributes, namedattribute("untied", untied))
    !isnothing(mergeable) && push!(_attributes, namedattribute("mergeable", mergeable))
    !isnothing(in_reductions) &&
        push!(_attributes, namedattribute("in_reductions", in_reductions))
    !isnothing(depends) && push!(_attributes, namedattribute("depends", depends))

    return IR.create_operation(
        "omp.task",
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
`taskwait`

The taskwait construct specifies a wait on the completion of child tasks
of the current task.
"""
function taskwait(; location=Location())
    _results = IR.Type[]
    _operands = Value[]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "omp.taskwait",
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
`taskyield`

The taskyield construct specifies that the current task can be suspended
in favor of execution of a different task.
"""
function taskyield(; location=Location())
    _results = IR.Type[]
    _operands = Value[]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "omp.taskyield",
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
`teams`

The teams construct defines a region of code that triggers the creation of a
league of teams. Once created, the number of teams remains constant for the
duration of its code region.

The optional \$num_teams_upper and \$num_teams_lower specify the limit on the
number of teams to be created. If only the upper bound is specified, it acts
as if the lower bound was set to the same value. It is not supported to set
\$num_teams_lower if \$num_teams_upper is not specified. They define a closed
range, where both the lower and upper bounds are included.

If the \$if_expr is present and it evaluates to `false`, the number of teams
created is one.

The optional \$thread_limit specifies the limit on the number of threads.

The \$allocators_vars and \$allocate_vars parameters are a variadic list of
values that specify the memory allocator to be used to obtain storage for
private values.
"""
function teams(
    num_teams_lower=nothing::Union{Nothing,Value};
    num_teams_upper=nothing::Union{Nothing,Value},
    if_expr=nothing::Union{Nothing,Value},
    thread_limit=nothing::Union{Nothing,Value},
    allocate_vars::Vector{Value},
    allocators_vars::Vector{Value},
    reduction_vars::Vector{Value},
    reductions=nothing,
    region::Region,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[allocate_vars..., allocators_vars..., reduction_vars...]
    _owned_regions = Region[region,]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(num_teams_lower) && push!(_operands, num_teams_lower)
    !isnothing(num_teams_upper) && push!(_operands, num_teams_upper)
    !isnothing(if_expr) && push!(_operands, if_expr)
    !isnothing(thread_limit) && push!(_operands, thread_limit)
    push!(
        _attributes,
        operandsegmentsizes([
            isnothing(num_teams_lower) ? 0 : 1,
            isnothing(num_teams_upper) ? 0 : 1,
            isnothing(if_expr) ? 0 : 1,
            isnothing(thread_limit) ? 0 : 1,
            length(allocate_vars),
            length(allocators_vars),
            length(reduction_vars),
        ]),
    )
    !isnothing(reductions) && push!(_attributes, namedattribute("reductions", reductions))

    return IR.create_operation(
        "omp.teams",
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
`terminator`

A terminator operation for regions that appear in the body of OpenMP
operation.  These regions are not expected to return any value so the
terminator takes no operands. The terminator op returns control to the
enclosing op.
"""
function terminator(; location=Location())
    _results = IR.Type[]
    _operands = Value[]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "omp.terminator",
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
`threadprivate`

The threadprivate directive specifies that variables are replicated, with
each thread having its own copy.

The current implementation uses the OpenMP runtime to provide thread-local
storage (TLS). Using the TLS feature of the LLVM IR will be supported in
future.

This operation takes in the address of a symbol that represents the original
variable and returns the address of its TLS. All occurrences of
threadprivate variables in a parallel region should use the TLS returned by
this operation.

The `sym_addr` refers to the address of the symbol, which is a pointer to
the original variable.
"""
function threadprivate(sym_addr::Value; tls_addr::IR.Type, location=Location())
    _results = IR.Type[tls_addr,]
    _operands = Value[sym_addr,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "omp.threadprivate",
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
`wsloop`

The worksharing-loop construct specifies that the iterations of the loop(s)
will be executed in parallel by threads in the current context. These
iterations are spread across threads that already exist in the enclosing
parallel region. The lower and upper bounds specify a half-open range: the
range includes the lower bound but does not include the upper bound. If the
`inclusive` attribute is specified then the upper bound is also included.

The body region can contain any number of blocks. The region is terminated
by \"omp.yield\" instruction without operands.

```
omp.wsloop <clauses>
for (%i1, %i2) : index = (%c0, %c0) to (%c10, %c10) step (%c1, %c1) {
  %a = load %arrA[%i1, %i2] : memref<?x?xf32>
  %b = load %arrB[%i1, %i2] : memref<?x?xf32>
  %sum = arith.addf %a, %b : f32
  store %sum, %arrC[%i1, %i2] : memref<?x?xf32>
  omp.yield
}
```

The `linear_step_vars` operand additionally specifies the step for each
associated linear operand. Note that the `linear_vars` and
`linear_step_vars` variadic lists should contain the same number of
elements.

Reductions can be performed in a worksharing-loop by specifying reduction
accumulator variables in `reduction_vars` and symbols referring to reduction
declarations in the `reductions` attribute. Each reduction is identified
by the accumulator it uses and accumulators must not be repeated in the same
reduction. The `omp.reduction` operation accepts the accumulator and a
partial value which is considered to be produced by the current loop
iteration for the given reduction. If multiple values are produced for the
same accumulator, i.e. there are multiple `omp.reduction`s, the last value
is taken. The reduction declaration specifies how to combine the values from
each iteration into the final value, which is available in the accumulator
after the loop completes.

The optional `schedule_val` attribute specifies the loop schedule for this
loop, determining how the loop is distributed across the parallel threads.
The optional `schedule_chunk_var` associated with this determines further
controls this distribution.

Collapsed loops are represented by the worksharing-loop having a list of
indices, bounds and steps where the size of the list is equal to the
collapse value.

The `nowait` attribute, when present, signifies that there should be no
implicit barrier at the end of the loop.

The optional `ordered_val` attribute specifies how many loops are associated
with the worksharing-loop construct. The value of zero refers to the ordered
clause specified without parameter.

The optional `order` attribute specifies which order the iterations of the
associate loops are executed in. Currently the only option for this
attribute is \"concurrent\".
"""
function wsloop(
    lowerBound::Vector{Value},
    upperBound::Vector{Value},
    step::Vector{Value},
    linear_vars::Vector{Value},
    linear_step_vars::Vector{Value},
    reduction_vars::Vector{Value},
    schedule_chunk_var=nothing::Union{Nothing,Value};
    reductions=nothing,
    schedule_val=nothing,
    schedule_modifier=nothing,
    simd_modifier=nothing,
    nowait=nothing,
    ordered_val=nothing,
    order_val=nothing,
    inclusive=nothing,
    region::Region,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[
        lowerBound...,
        upperBound...,
        step...,
        linear_vars...,
        linear_step_vars...,
        reduction_vars...,
    ]
    _owned_regions = Region[region,]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(schedule_chunk_var) && push!(_operands, schedule_chunk_var)
    push!(
        _attributes,
        operandsegmentsizes([
            length(lowerBound),
            length(upperBound),
            length(step),
            length(linear_vars),
            length(linear_step_vars),
            length(reduction_vars),
            isnothing(schedule_chunk_var) ? 0 : 1,
        ]),
    )
    !isnothing(reductions) && push!(_attributes, namedattribute("reductions", reductions))
    !isnothing(schedule_val) &&
        push!(_attributes, namedattribute("schedule_val", schedule_val))
    !isnothing(schedule_modifier) &&
        push!(_attributes, namedattribute("schedule_modifier", schedule_modifier))
    !isnothing(simd_modifier) &&
        push!(_attributes, namedattribute("simd_modifier", simd_modifier))
    !isnothing(nowait) && push!(_attributes, namedattribute("nowait", nowait))
    !isnothing(ordered_val) &&
        push!(_attributes, namedattribute("ordered_val", ordered_val))
    !isnothing(order_val) && push!(_attributes, namedattribute("order_val", order_val))
    !isnothing(inclusive) && push!(_attributes, namedattribute("inclusive", inclusive))

    return IR.create_operation(
        "omp.wsloop",
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
`yield`

\"omp.yield\" yields SSA values from the OpenMP dialect op region and
terminates the region. The semantics of how the values are yielded is
defined by the parent operation.
"""
function yield(results::Vector{Value}; location=Location())
    _results = IR.Type[]
    _operands = Value[results...,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "omp.yield",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

end # omp
