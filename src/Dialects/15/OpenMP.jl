module omp

import ...IR:
    IR, NamedAttribute, Value, Location, Block, Region, Attribute, context, IndexType
import ..Dialects: namedattribute, operandsegmentsizes
import ...API

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
    results = IR.Type[]
    operands = Value[]
    owned_regions = Region[region,]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(hint_val) && push!(attributes, namedattribute("hint_val", hint_val))
    !isnothing(memory_order_val) &&
        push!(attributes, namedattribute("memory_order_val", memory_order_val))

    return IR.create_operation(
        "omp.atomic.capture",
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
    x::Value, v::Value; hint_val=nothing, memory_order_val=nothing, location=Location()
)
    results = IR.Type[]
    operands = Value[x, v]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(hint_val) && push!(attributes, namedattribute("hint_val", hint_val))
    !isnothing(memory_order_val) &&
        push!(attributes, namedattribute("memory_order_val", memory_order_val))

    return IR.create_operation(
        "omp.atomic.read",
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
    results = IR.Type[]
    operands = Value[x,]
    owned_regions = Region[region,]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(hint_val) && push!(attributes, namedattribute("hint_val", hint_val))
    !isnothing(memory_order_val) &&
        push!(attributes, namedattribute("memory_order_val", memory_order_val))

    return IR.create_operation(
        "omp.atomic.update",
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
`atomic_write`

This operation performs an atomic write.

The operand `address` is the address to where the `value` is atomically
written w.r.t. multiple threads. The evaluation of `value` need not be
atomic w.r.t. the write to address. In general, the type(address) must
dereference to type(value).

`hint` is the value of hint (as specified in the hint clause). It is a
compile time constant. As the name suggests, this is just a hint for
optimization.

`memory_order` indicates the memory ordering behavior of the construct. It
can be one of `seq_cst`, `release` or `relaxed`.
"""
function atomic_write(
    address::Value,
    value::Value;
    hint_val=nothing,
    memory_order_val=nothing,
    location=Location(),
)
    results = IR.Type[]
    operands = Value[address, value]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(hint_val) && push!(attributes, namedattribute("hint_val", hint_val))
    !isnothing(memory_order_val) &&
        push!(attributes, namedattribute("memory_order_val", memory_order_val))

    return IR.create_operation(
        "omp.atomic.write",
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
`barrier`

The barrier construct specifies an explicit barrier at the point at which
the construct appears.
"""
function barrier(; location=Location())
    results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return IR.create_operation(
        "omp.barrier",
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
`cancel`

The cancel construct activates cancellation of the innermost enclosing
region of the type specified.
"""
function cancel(
    if_expr=nothing::Union{Nothing,Value};
    cancellation_construct_type_val,
    location=Location(),
)
    results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute(
        "cancellation_construct_type_val", cancellation_construct_type_val
    ),]
    !isnothing(if_expr) && push!(operands, if_expr)

    return IR.create_operation(
        "omp.cancel",
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
`cancellationpoint`

The cancellation point construct introduces a user-defined cancellation
point at which implicit or explicit tasks check if cancellation of the
innermost enclosing region of the type specified has been activated.
"""
function cancellationpoint(; cancellation_construct_type_val, location=Location())
    results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute(
        "cancellation_construct_type_val", cancellation_construct_type_val
    ),]

    return IR.create_operation(
        "omp.cancellationpoint",
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
`critical_declare`

Declares a named critical section.

The name can be used in critical constructs in the dialect.
"""
function critical_declare(; sym_name, hint_val=nothing, location=Location())
    results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("sym_name", sym_name),]
    !isnothing(hint_val) && push!(attributes, namedattribute("hint_val", hint_val))

    return IR.create_operation(
        "omp.critical.declare",
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
`critical`

The critical construct imposes a restriction on the associated structured
block (region) to be executed by only a single thread at a time.
"""
function critical(; name=nothing, region::Region, location=Location())
    results = IR.Type[]
    operands = Value[]
    owned_regions = Region[region,]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(name) && push!(attributes, namedattribute("name", name))

    return IR.create_operation(
        "omp.critical",
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
`flush`

The flush construct executes the OpenMP flush operation. This operation
makes a thread’s temporary view of memory consistent with memory and
enforces an order on the memory operations of the variables explicitly
specified or implied.
"""
function flush(varList::Vector{Value}; location=Location())
    results = IR.Type[]
    operands = Value[varList...,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return IR.create_operation(
        "omp.flush",
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
`master`

The master construct specifies a structured block that is executed by
the master thread of the team.
"""
function master(; region::Region, location=Location())
    results = IR.Type[]
    operands = Value[]
    owned_regions = Region[region,]
    successors = Block[]
    attributes = NamedAttribute[]

    return IR.create_operation(
        "omp.master",
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
    results = IR.Type[]
    operands = Value[depend_vec_vars...,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(depend_type_val) &&
        push!(attributes, namedattribute("depend_type_val", depend_type_val))
    !isnothing(num_loops_val) &&
        push!(attributes, namedattribute("num_loops_val", num_loops_val))

    return IR.create_operation(
        "omp.ordered",
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
`ordered_region`

The ordered construct with region specifies a structured block in a
worksharing-loop, SIMD, or worksharing-loop SIMD region that is executed in
the order of the loop iterations.

The `simd` attribute corresponds to the SIMD clause specified. If it is not
present, it behaves as if the THREADS clause is specified or no clause is
specified.
"""
function ordered_region(; simd=nothing, region::Region, location=Location())
    results = IR.Type[]
    operands = Value[]
    owned_regions = Region[region,]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(simd) && push!(attributes, namedattribute("simd", simd))

    return IR.create_operation(
        "omp.ordered_region",
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
    results = IR.Type[]
    operands = Value[allocate_vars..., allocators_vars..., reduction_vars...]
    owned_regions = Region[region,]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(if_expr_var) && push!(operands, if_expr_var)
    !isnothing(num_threads_var) && push!(operands, num_threads_var)
    push!(
        attributes,
        operandsegmentsizes([
            if (if_expr_var == nothing)
                0
            elseif 1(num_threads_var == nothing)
                0
            else
                1length(allocate_vars)
            end,
            length(allocators_vars),
            length(reduction_vars),
        ]),
    )
    !isnothing(reductions) && push!(attributes, namedattribute("reductions", reductions))
    !isnothing(proc_bind_val) &&
        push!(attributes, namedattribute("proc_bind_val", proc_bind_val))

    return IR.create_operation(
        "omp.parallel",
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
    results = IR.Type[]
    operands = Value[]
    owned_regions = Region[initializerRegion, reductionRegion, atomicReductionRegion]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("sym_name", sym_name), namedattribute("type", type)
    ]

    return IR.create_operation(
        "omp.reduction.declare",
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
`reduction`

Indicates the value that is produced by the current reduction-participating
entity for a reduction requested in some ancestor. The reduction is
identified by the accumulator, but the value of the accumulator may not be
updated immediately.
"""
function reduction(operand::Value, accumulator::Value; location=Location())
    results = IR.Type[]
    operands = Value[operand, accumulator]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return IR.create_operation(
        "omp.reduction",
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
`section`

A section operation encloses a region which represents one section in a
sections construct. A section op should always be surrounded by an
`omp.sections` operation.
"""
function section(; region::Region, location=Location())
    results = IR.Type[]
    operands = Value[]
    owned_regions = Region[region,]
    successors = Block[]
    attributes = NamedAttribute[]

    return IR.create_operation(
        "omp.section",
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
    results = IR.Type[]
    operands = Value[reduction_vars..., allocate_vars..., allocators_vars...]
    owned_regions = Region[region,]
    successors = Block[]
    attributes = NamedAttribute[]
    push!(
        attributes,
        operandsegmentsizes([
            length(reduction_vars), length(allocate_vars), length(allocators_vars)
        ]),
    )
    !isnothing(reductions) && push!(attributes, namedattribute("reductions", reductions))
    !isnothing(nowait) && push!(attributes, namedattribute("nowait", nowait))

    return IR.create_operation(
        "omp.sections",
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
`simdloop`

The simd construct can be applied to a loop to indicate that the loop can be 
transformed into a SIMD loop (that is, multiple iterations of the loop can 
be executed concurrently using SIMD instructions).. The lower and upper 
bounds specify a half-open range: the range includes the lower bound but 
does not include the upper bound. If the `inclusive` attribute is specified
then the upper bound is also included.

The body region can contain any number of blocks. The region is terminated
by \"omp.yield\" instruction without operands.

When an if clause is present and evaluates to false, the preferred number of
iterations to be executed concurrently is one, regardless of whether
a simdlen clause is speciﬁed.
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
    if_expr=nothing::Union{Nothing,Value};
    inclusive=nothing,
    region::Region,
    location=Location(),
)
    results = IR.Type[]
    operands = Value[lowerBound..., upperBound..., step...]
    owned_regions = Region[region,]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(if_expr) && push!(operands, if_expr)
    push!(
        attributes,
        operandsegmentsizes([
            length(lowerBound),
            length(upperBound),
            length(step),
            (if_expr == nothing) ? 0 : 1,
        ]),
    )
    !isnothing(inclusive) && push!(attributes, namedattribute("inclusive", inclusive))

    return IR.create_operation(
        "omp.simdloop",
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
    results = IR.Type[]
    operands = Value[allocate_vars..., allocators_vars...]
    owned_regions = Region[region,]
    successors = Block[]
    attributes = NamedAttribute[]
    push!(attributes, operandsegmentsizes([length(allocate_vars), length(allocators_vars)]))
    !isnothing(nowait) && push!(attributes, namedattribute("nowait", nowait))

    return IR.create_operation(
        "omp.single",
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

TODO:  map, is_device_ptr, depend, defaultmap, in_reduction
"""
function target(
    if_expr=nothing::Union{Nothing,Value};
    device=nothing::Union{Nothing,Value},
    thread_limit=nothing::Union{Nothing,Value},
    nowait=nothing,
    region::Region,
    location=Location(),
)
    results = IR.Type[]
    operands = Value[]
    owned_regions = Region[region,]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(if_expr) && push!(operands, if_expr)
    !isnothing(device) && push!(operands, device)
    !isnothing(thread_limit) && push!(operands, thread_limit)
    push!(
        attributes,
        operandsegmentsizes([
            if (if_expr == nothing)
                0
            elseif 1(device == nothing)
                0
            elseif 1(thread_limit == nothing)
                0
            else
                1
            end,
        ]),
    )
    !isnothing(nowait) && push!(attributes, namedattribute("nowait", nowait))

    return IR.create_operation(
        "omp.target",
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
    results = IR.Type[]
    operands = Value[task_reduction_vars..., allocate_vars..., allocators_vars...]
    owned_regions = Region[region,]
    successors = Block[]
    attributes = NamedAttribute[]
    push!(
        attributes,
        operandsegmentsizes([
            length(task_reduction_vars), length(allocate_vars), length(allocators_vars)
        ]),
    )
    !isnothing(task_reductions) &&
        push!(attributes, namedattribute("task_reductions", task_reductions))

    return IR.create_operation(
        "omp.taskgroup",
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
    results = IR.Type[]
    operands = Value[
        lowerBound...,
        upperBound...,
        step...,
        in_reduction_vars...,
        reduction_vars...,
        allocate_vars...,
        allocators_vars...,
    ]
    owned_regions = Region[region,]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(if_expr) && push!(operands, if_expr)
    !isnothing(final_expr) && push!(operands, final_expr)
    !isnothing(priority) && push!(operands, priority)
    !isnothing(grain_size) && push!(operands, grain_size)
    !isnothing(num_tasks) && push!(operands, num_tasks)
    push!(
        attributes,
        operandsegmentsizes([
            length(lowerBound),
            length(upperBound),
            length(step),
            if (if_expr == nothing)
                0
            elseif 1(final_expr == nothing)
                0
            else
                1length(in_reduction_vars)
            end,
            length(reduction_vars),
            (priority == nothing) ? 0 : 1length(allocate_vars),
            length(allocators_vars),
            if (grain_size == nothing)
                0
            elseif 1(num_tasks == nothing)
                0
            else
                1
            end,
        ]),
    )
    !isnothing(inclusive) && push!(attributes, namedattribute("inclusive", inclusive))
    !isnothing(untied) && push!(attributes, namedattribute("untied", untied))
    !isnothing(mergeable) && push!(attributes, namedattribute("mergeable", mergeable))
    !isnothing(in_reductions) &&
        push!(attributes, namedattribute("in_reductions", in_reductions))
    !isnothing(reductions) && push!(attributes, namedattribute("reductions", reductions))
    !isnothing(nogroup) && push!(attributes, namedattribute("nogroup", nogroup))

    return IR.create_operation(
        "omp.taskloop",
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

The `allocators_vars` and `allocate_vars` arguments are a variadic list of
values that specify the memory allocator to be used to obtain storage for
private values.
"""
function task(
    if_expr=nothing::Union{Nothing,Value};
    final_expr=nothing::Union{Nothing,Value},
    in_reduction_vars::Vector{Value},
    priority=nothing::Union{Nothing,Value},
    allocate_vars::Vector{Value},
    allocators_vars::Vector{Value},
    untied=nothing,
    mergeable=nothing,
    in_reductions=nothing,
    region::Region,
    location=Location(),
)
    results = IR.Type[]
    operands = Value[in_reduction_vars..., allocate_vars..., allocators_vars...]
    owned_regions = Region[region,]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(if_expr) && push!(operands, if_expr)
    !isnothing(final_expr) && push!(operands, final_expr)
    !isnothing(priority) && push!(operands, priority)
    push!(
        attributes,
        operandsegmentsizes([
            if (if_expr == nothing)
                0
            elseif 1(final_expr == nothing)
                0
            else
                1length(in_reduction_vars)
            end,
            (priority == nothing) ? 0 : 1length(allocate_vars),
            length(allocators_vars),
        ]),
    )
    !isnothing(untied) && push!(attributes, namedattribute("untied", untied))
    !isnothing(mergeable) && push!(attributes, namedattribute("mergeable", mergeable))
    !isnothing(in_reductions) &&
        push!(attributes, namedattribute("in_reductions", in_reductions))

    return IR.create_operation(
        "omp.task",
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
`taskwait`

The taskwait construct specifies a wait on the completion of child tasks
of the current task.
"""
function taskwait(; location=Location())
    results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return IR.create_operation(
        "omp.taskwait",
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
`taskyield`

The taskyield construct specifies that the current task can be suspended
in favor of execution of a different task.
"""
function taskyield(; location=Location())
    results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return IR.create_operation(
        "omp.taskyield",
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
`terminator`

A terminator operation for regions that appear in the body of OpenMP
operation.  These regions are not expected to return any value so the
terminator takes no operands. The terminator op returns control to the
enclosing op.
"""
function terminator(; location=Location())
    results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return IR.create_operation(
        "omp.terminator",
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
    results = IR.Type[tls_addr,]
    operands = Value[sym_addr,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return IR.create_operation(
        "omp.threadprivate",
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
    results = IR.Type[]
    operands = Value[
        lowerBound...,
        upperBound...,
        step...,
        linear_vars...,
        linear_step_vars...,
        reduction_vars...,
    ]
    owned_regions = Region[region,]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(schedule_chunk_var) && push!(operands, schedule_chunk_var)
    push!(
        attributes,
        operandsegmentsizes([
            length(lowerBound),
            length(upperBound),
            length(step),
            length(linear_vars),
            length(linear_step_vars),
            length(reduction_vars),
            (schedule_chunk_var == nothing) ? 0 : 1,
        ]),
    )
    !isnothing(reductions) && push!(attributes, namedattribute("reductions", reductions))
    !isnothing(schedule_val) &&
        push!(attributes, namedattribute("schedule_val", schedule_val))
    !isnothing(schedule_modifier) &&
        push!(attributes, namedattribute("schedule_modifier", schedule_modifier))
    !isnothing(simd_modifier) &&
        push!(attributes, namedattribute("simd_modifier", simd_modifier))
    !isnothing(nowait) && push!(attributes, namedattribute("nowait", nowait))
    !isnothing(ordered_val) && push!(attributes, namedattribute("ordered_val", ordered_val))
    !isnothing(order_val) && push!(attributes, namedattribute("order_val", order_val))
    !isnothing(inclusive) && push!(attributes, namedattribute("inclusive", inclusive))

    return IR.create_operation(
        "omp.wsloop",
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
`yield`

\"omp.yield\" yields SSA values from the OpenMP dialect op region and
terminates the region. The semantics of how the values are yielded is
defined by the parent operation.
"""
function yield(results::Vector{Value}; location=Location())
    results = IR.Type[]
    operands = Value[results...,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return IR.create_operation(
        "omp.yield",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=results,
        result_inference=false,
    )
end

end # omp
