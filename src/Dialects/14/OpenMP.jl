module omp

import ...IR: IR, NamedAttribute, Value, Location, Block, Region, Attribute, context, IndexType
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
function atomic_capture(; hint=nothing, memory_order=nothing, region::Region, location=Location())
    results = IR.Type[]
    operands = Value[]
    owned_regions = Region[region,]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(hint) && push!(attributes, namedattribute("hint", hint))
    !isnothing(memory_order) && push!(attributes, namedattribute("memory_order", memory_order))

    IR.create_operation(
        "omp.atomic.capture", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
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
can be one of `seq_cst`, `acq_rel`, `release`, `acquire` or `relaxed`.
"""
function atomic_read(x::Value, v::Value; hint=nothing, memory_order=nothing, location=Location())
    results = IR.Type[]
    operands = Value[x, v,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(hint) && push!(attributes, namedattribute("hint", hint))
    !isnothing(memory_order) && push!(attributes, namedattribute("memory_order", memory_order))

    IR.create_operation(
        "omp.atomic.read", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`atomic_update`

This operation performs an atomic update.

The operands `x` and `expr` are exactly the same as the operands `x` and
`expr` in the OpenMP Standard. The operand `x` is the address of the
variable that is being updated. `x` is atomically read/written. The
evaluation of `expr` need not be atomic w.r.t the read or write of the
location designated by `x`. In general, type(x) must dereference to
type(expr).

The attribute `isXBinopExpr` is
  - true when the expression is of the form `x binop expr` on RHS
  - false when the expression is of the form `expr binop x` on RHS

The attribute `binop` is the binary operation being performed atomically.

`hint` is the value of hint (as used in the hint clause). It is a compile
time constant. As the name suggests, this is just a hint for optimization.

`memory_order` indicates the memory ordering behavior of the construct. It
can be one of `seq_cst`, `acq_rel`, `release`, `acquire` or `relaxed`.
"""
function atomic_update(x::Value, expr::Value; isXBinopExpr=nothing, binop, hint=nothing, memory_order=nothing, location=Location())
    results = IR.Type[]
    operands = Value[x, expr,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("binop", binop),]
    !isnothing(isXBinopExpr) && push!(attributes, namedattribute("isXBinopExpr", isXBinopExpr))
    !isnothing(hint) && push!(attributes, namedattribute("hint", hint))
    !isnothing(memory_order) && push!(attributes, namedattribute("memory_order", memory_order))

    IR.create_operation(
        "omp.atomic.update", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
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
can be one of `seq_cst`, `acq_rel`, `release`, `acquire` or `relaxed`.
"""
function atomic_write(address::Value, value::Value; hint=nothing, memory_order=nothing, location=Location())
    results = IR.Type[]
    operands = Value[address, value,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(hint) && push!(attributes, namedattribute("hint", hint))
    !isnothing(memory_order) && push!(attributes, namedattribute("memory_order", memory_order))

    IR.create_operation(
        "omp.atomic.write", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
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

    IR.create_operation(
        "omp.barrier", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`critical_declare`

Declares a named critical section.

The name can be used in critical constructs in the dialect.
"""
function critical_declare(; sym_name, hint=nothing, location=Location())
    results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("sym_name", sym_name),]
    !isnothing(hint) && push!(attributes, namedattribute("hint", hint))

    IR.create_operation(
        "omp.critical.declare", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
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

    IR.create_operation(
        "omp.critical", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
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

    IR.create_operation(
        "omp.flush", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
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

    IR.create_operation(
        "omp.master", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
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
function ordered(depend_vec_vars::Vector{Value}; depend_type_val=nothing, num_loops_val=nothing, location=Location())
    results = IR.Type[]
    operands = Value[depend_vec_vars...,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(depend_type_val) && push!(attributes, namedattribute("depend_type_val", depend_type_val))
    !isnothing(num_loops_val) && push!(attributes, namedattribute("num_loops_val", num_loops_val))

    IR.create_operation(
        "omp.ordered", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
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

    IR.create_operation(
        "omp.ordered_region", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
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

The optional \$default_val attribute specifies the default data sharing attribute
of values used in the parallel region that are not passed explicitly as parameters
to the operation.

The \$private_vars, \$firstprivate_vars, \$shared_vars and \$copyin_vars parameters
are a variadic list of values that specify the data sharing attribute of
those values.

The \$allocators_vars and \$allocate_vars parameters are a variadic list of values
that specify the memory allocator to be used to obtain storage for private values.

The optional \$proc_bind_val attribute controls the thread affinity for the execution
of the parallel region.
"""
function parallel(if_expr_var=nothing::Union{Nothing,Value}; num_threads_var=nothing::Union{Nothing,Value}, private_vars::Vector{Value}, firstprivate_vars::Vector{Value}, shared_vars::Vector{Value}, copyin_vars::Vector{Value}, allocate_vars::Vector{Value}, allocators_vars::Vector{Value}, default_val=nothing, proc_bind_val=nothing, region::Region, location=Location())
    results = IR.Type[]
    operands = Value[private_vars..., firstprivate_vars..., shared_vars..., copyin_vars..., allocate_vars..., allocators_vars...,]
    owned_regions = Region[region,]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(if_expr_var) && push!(operands, if_expr_var)
    !isnothing(num_threads_var) && push!(operands, num_threads_var)
    push!(attributes, operandsegmentsizes([(if_expr_var == nothing) ? 0 : 1(num_threads_var == nothing) ? 0 : 1length(private_vars), length(firstprivate_vars), length(shared_vars), length(copyin_vars), length(allocate_vars), length(allocators_vars),]))
    !isnothing(default_val) && push!(attributes, namedattribute("default_val", default_val))
    !isnothing(proc_bind_val) && push!(attributes, namedattribute("proc_bind_val", proc_bind_val))

    IR.create_operation(
        "omp.parallel", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
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
"""
function reduction_declare(; sym_name, type, initializerRegion::Region, reductionRegion::Region, atomicReductionRegion::Region, location=Location())
    results = IR.Type[]
    operands = Value[]
    owned_regions = Region[initializerRegion, reductionRegion, atomicReductionRegion,]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("sym_name", sym_name), namedattribute("type", type),]

    IR.create_operation(
        "omp.reduction.declare", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
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
    operands = Value[operand, accumulator,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    IR.create_operation(
        "omp.reduction", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
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

    IR.create_operation(
        "omp.section", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`sections`

The sections construct is a non-iterative worksharing construct that
contains `omp.section` operations. The `omp.section` operations are to be
distributed among and executed by the threads in a team. Each `omp.section`
is executed once by one of the threads in the team in the context of its
implicit task.

`private_vars`, `firstprivate_vars` and`lastprivate_vars` arguments are
variadic list of operands that specify the data sharing attributes of the
list of values. They are optional.

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
function sections(private_vars::Vector{Value}, firstprivate_vars::Vector{Value}, lastprivate_vars::Vector{Value}, reduction_vars::Vector{Value}, allocate_vars::Vector{Value}, allocators_vars::Vector{Value}; reductions=nothing, nowait=nothing, region::Region, location=Location())
    results = IR.Type[]
    operands = Value[private_vars..., firstprivate_vars..., lastprivate_vars..., reduction_vars..., allocate_vars..., allocators_vars...,]
    owned_regions = Region[region,]
    successors = Block[]
    attributes = NamedAttribute[]
    push!(attributes, operandsegmentsizes([length(private_vars), length(firstprivate_vars), length(lastprivate_vars), length(reduction_vars), length(allocate_vars), length(allocators_vars),]))
    !isnothing(reductions) && push!(attributes, namedattribute("reductions", reductions))
    !isnothing(nowait) && push!(attributes, namedattribute("nowait", nowait))

    IR.create_operation(
        "omp.sections", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
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

TODO:  private, map, is_device_ptr, firstprivate, depend, defaultmap, in_reduction
"""
function target(if_expr=nothing::Union{Nothing,Value}; device=nothing::Union{Nothing,Value}, thread_limit=nothing::Union{Nothing,Value}, nowait=nothing, region::Region, location=Location())
    results = IR.Type[]
    operands = Value[]
    owned_regions = Region[region,]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(if_expr) && push!(operands, if_expr)
    !isnothing(device) && push!(operands, device)
    !isnothing(thread_limit) && push!(operands, thread_limit)
    push!(attributes, operandsegmentsizes([(if_expr == nothing) ? 0 : 1(device == nothing) ? 0 : 1(thread_limit == nothing) ? 0 : 1]))
    !isnothing(nowait) && push!(attributes, namedattribute("nowait", nowait))

    IR.create_operation(
        "omp.target", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
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

    IR.create_operation(
        "omp.taskwait", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
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

    IR.create_operation(
        "omp.taskyield", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
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

    IR.create_operation(
        "omp.terminator", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`wsloop`

The workshare loop construct specifies that the iterations of the loop(s)
will be executed in parallel by threads in the current context. These
iterations are spread across threads that already exist in the enclosing
parallel region. The lower and upper bounds specify a half-open range: the
range includes the lower bound but does not include the upper bound. If the
`inclusive` attribute is specified then the upper bound is also included.

The body region can contain any number of blocks. The region is terminated
by \"omp.yield\" instruction without operands.

```
omp.wsloop (%i1, %i2) : index = (%c0, %c0) to (%c10, %c10) step (%c1, %c1) {
  %a = load %arrA[%i1, %i2] : memref<?x?xf32>
  %b = load %arrB[%i1, %i2] : memref<?x?xf32>
  %sum = arith.addf %a, %b : f32
  store %sum, %arrC[%i1, %i2] : memref<?x?xf32>
  omp.yield
}
```

`private_vars`, `firstprivate_vars`, `lastprivate_vars` and `linear_vars`
arguments are variadic list of operands that specify the data sharing
attributes of the list of values. The `linear_step_vars` operand
additionally specifies the step for each associated linear operand. Note
that the `linear_vars` and `linear_step_vars` variadic lists should contain
the same number of elements.

Reductions can be performed in a workshare loop by specifying reduction
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

The optional `collapse_val` attribute specifies the number of loops which
are collapsed to form the worksharing loop.

The `nowait` attribute, when present, signifies that there should be no
implicit barrier at the end of the loop.

The optional `ordered_val` attribute specifies how many loops are associated
with the do loop construct.

The optional `order` attribute specifies which order the iterations of the
associate loops are executed in. Currently the only option for this
attribute is \"concurrent\".
"""
function wsloop(lowerBound::Vector{Value}, upperBound::Vector{Value}, step::Vector{Value}, private_vars::Vector{Value}, firstprivate_vars::Vector{Value}, lastprivate_vars::Vector{Value}, linear_vars::Vector{Value}, linear_step_vars::Vector{Value}, reduction_vars::Vector{Value}, schedule_chunk_var=nothing::Union{Nothing,Value}; reductions=nothing, schedule_val=nothing, schedule_modifier=nothing, simd_modifier=nothing, collapse_val=nothing, nowait=nothing, ordered_val=nothing, order_val=nothing, inclusive=nothing, region::Region, location=Location())
    results = IR.Type[]
    operands = Value[lowerBound..., upperBound..., step..., private_vars..., firstprivate_vars..., lastprivate_vars..., linear_vars..., linear_step_vars..., reduction_vars...,]
    owned_regions = Region[region,]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(schedule_chunk_var) && push!(operands, schedule_chunk_var)
    push!(attributes, operandsegmentsizes([length(lowerBound), length(upperBound), length(step), length(private_vars), length(firstprivate_vars), length(lastprivate_vars), length(linear_vars), length(linear_step_vars), length(reduction_vars), (schedule_chunk_var == nothing) ? 0 : 1]))
    !isnothing(reductions) && push!(attributes, namedattribute("reductions", reductions))
    !isnothing(schedule_val) && push!(attributes, namedattribute("schedule_val", schedule_val))
    !isnothing(schedule_modifier) && push!(attributes, namedattribute("schedule_modifier", schedule_modifier))
    !isnothing(simd_modifier) && push!(attributes, namedattribute("simd_modifier", simd_modifier))
    !isnothing(collapse_val) && push!(attributes, namedattribute("collapse_val", collapse_val))
    !isnothing(nowait) && push!(attributes, namedattribute("nowait", nowait))
    !isnothing(ordered_val) && push!(attributes, namedattribute("ordered_val", ordered_val))
    !isnothing(order_val) && push!(attributes, namedattribute("order_val", order_val))
    !isnothing(inclusive) && push!(attributes, namedattribute("inclusive", inclusive))

    IR.create_operation(
        "omp.wsloop", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`yield`

\"omp.yield\" yields SSA values from the OpenMP dialect op region and
terminates the region. The semantics of how the values are yielded is
defined by the parent operation.
If \"omp.yield\" has any operands, the operands must match the parent
operation\'s results.
"""
function yield(results::Vector{Value}; location=Location())
    results = IR.Type[]
    operands = Value[results...,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    IR.create_operation(
        "omp.yield", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

end # omp
