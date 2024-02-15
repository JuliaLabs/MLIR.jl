module transform

import ...IR: NamedAttribute, MLIRType, Value, Location, Block, Region, Attribute, create_operation, context, IndexType
import ..Dialects: namedattribute, operandsegmentsizes
import ...API


"""
`alternatives`

This op may have an arbitrary number of regions, each of which represents a
sequence of transform operations to be applied to the same payload IR. The
regions are visited in order of appearance, and transforms in them are
applied in their respective order of appearance. If one of these transforms
fails to apply, the remaining ops in the same region are skipped an the next
region is attempted. If all transformations in a region succeed, the
remaining regions are skipped and the entire \"alternatives\" transformation
succeeds. If all regions contained a failing transformation, the entire
\"alternatives\" transformation fails.

It is up to the nested operations to define which errors are \"recoverable\"
(or \"silenceable\") and allow another alternatives to be attempted, and which
errors should be propagated without attempting the other alternatives.

The single operand of this operation is the scope in which the alternative
transformation sequences are attempted, that is, an operation in the payload
IR that contains all the other operations that may be modified by the
transformations. The scope operation must be isolated from above. There is
no check that the transforms are indeed scoped as their \"apply\" methods can
be arbitrarily complex. Therefore it is the responsibility of the user to
ensure that the transforms are scoped correctly, or to produce an
irrecoverable error and thus abort the execution without attempting the
remaining alternatives. Note that the payload IR outside of the given scope
is not necessarily in the valid state, or even accessible to the
transformation.

The changes to the IR within the scope performed by transforms in the failed
alternative region are reverted before attempting the next region.
Practically, this is achieved by cloning the scope. Therefore it is advised
to limit the scope as much as possible and place the most likely
alternatives early in the region list. The operation is also isolated from
above and requires rediscovering the operations within the given scope to
avoid additional handle invalidation. The latter restriction may be lifted
in the future.

Each of the regions may yield transform IR handles. The handles of the first
successful alternative region are returned as the results of the
\"alternatives\" op. Therefore, each alternative region must yield the same
number of results, which should also match the number and the types of the
\"alternatives\" op results.

Remark: this op allows one to implement a simple \"try\" construct as follows:

```mlir
%result = transform.alternatives %scope {
^bb0(%arg0: !pdl.operation):
  // Try a fallible transformation.
  %0 = transform.fallible %arg0 // ...
  // If succeeded, yield the the result of the transformation.
  transform.yield %0 : !pdl.operation
}, {
^bb0(%arg0: !pdl.operation):
  // Otherwise, the second alternative is tried and it always succeeds by
  // returning the original handle.
  transform.yield %arg0 : !pdl.operation
}
```
"""
function alternatives(scope=nothing::Union{Nothing, Value}; results::Vector{MLIRType}, alternatives::Vector{Region}, location=Location())
    results = MLIRType[results..., ]
    operands = Value[]
    owned_regions = Region[alternatives..., ]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(scope) && push!(operands, scope)
    
    create_operation(
        "transform.alternatives", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`cast`

"""
function cast(input::Value; output::MLIRType, location=Location())
    results = MLIRType[output, ]
    operands = Value[input, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "transform.cast", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`foreach`

This op has exactly one region with exactly one block (\"body\"). The body is
executed for each payload op that is associated to the target operand in an
unbatched fashion. I.e., the block argument (\"iteration variable\") is always
mapped to exactly one payload op.

This op always reads the target handle. Furthermore, it consumes the handle
if there is a transform op in the body that consumes the iteration variable.
This op does not return anything.

The transformations inside the body are applied in order of their
appearance. During application, if any transformation in the sequence fails,
the entire sequence fails immediately leaving the payload IR in potentially
invalid state, i.e., this operation offers no transformation rollback
capabilities.

This op generates as many handles as the terminating YieldOp has operands.
For each result, the payload ops of the corresponding YieldOp operand are
merged and mapped to the same resulting handle.
"""
function foreach(target::Value; results::Vector{MLIRType}, body::Region, location=Location())
    results = MLIRType[results..., ]
    operands = Value[target, ]
    owned_regions = Region[body, ]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "transform.foreach", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`get_closest_isolated_parent`

The handles defined by this Transform op correspond to the closest isolated
from above ancestor of the Payload IR operations associated with its
operand. If any of the given Payload IR ops has no such parent (unlikely as
there usually is a top-level ModuleOp), the transformation is considered to
have failed.

Ancestor ops follow the same order as the ops associated with the
operand, except for potential duplicates (multiple Payload IR ops associated
with the operand have the same parent) for which the ancestor will only be
listed once for the first time it occurs. For example, given the list
\"(childof(A), childof(B), childof(B), childof(A), childof(B))\", the
resulting list will be just \"(A, B)\". Note that no other semantic ordering
is applied, e.g., \"B\" may itself be a parent of \"A\". This may have an impact
on the further transformation applied to the handle produced here.
"""
function get_closest_isolated_parent(target::Value; parent::MLIRType, location=Location())
    results = MLIRType[parent, ]
    operands = Value[target, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "transform.get_closest_isolated_parent", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`get_consumers_of_result`

The handle defined by this Transform op corresponds to all operations that
consume the SSA value defined by the `target` and `result_number`
arguments.
This operation applies to a single payload operation, otherwise it 
definitely fails.
The return handle points to the consuming operations operations, which can
be empty.
"""
function get_consumers_of_result(target::Value; consumers::MLIRType, result_number, location=Location())
    results = MLIRType[consumers, ]
    operands = Value[target, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("result_number", result_number), ]
    
    create_operation(
        "transform.get_consumers_of_result", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`get_producer_of_operand`

The handle defined by this Transform op corresponds to operation that
produces the SSA value defined by the `target` and `operand_number`
arguments. If the origin of the SSA value is not an operations (i.e. it is
a block argument), the transform silently fails.
The return handle points to only the subset of successfully produced
computational operations, which can be empty.
"""
function get_producer_of_operand(target::Value; producer::MLIRType, operand_number, location=Location())
    results = MLIRType[producer, ]
    operands = Value[target, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("operand_number", operand_number), ]
    
    create_operation(
        "transform.get_producer_of_operand", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`merge_handles`

Creates a new Transform IR handle value that points to the same Payload IR
operations as the operand handles. The Payload IR operations are listed
in the same order as they are in the operand handles, grouped by operand
handle, e.g., all Payload IR operations associated with the first handle
come first, then all Payload IR operations associated with the second handle
and so on. If `deduplicate` is set, do not add the given Payload IR
operation more than once to the final list regardless of it coming from the
same or different handles. Consumes the operands and produces a new handle.
"""
function merge_handles(handles::Vector{Value}; result=nothing::Union{Nothing, MLIRType}, deduplicate=nothing, location=Location())
    results = MLIRType[]
    operands = Value[handles..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(results, result)
    !isnothing(deduplicate) && push!(attributes, namedattribute("deduplicate", deduplicate))
    
    create_operation(
        "transform.merge_handles", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`pdl_match`

Find Payload IR ops nested within the Payload IR op associated with the
operand that match the PDL pattern identified by its name. The pattern is
expected to be defined in the closest surrounding `WithPDLPatternsOp`.

Produces a Transform IR value associated with the list of Payload IR ops
that matched the pattern. The order of results in the list is that of the
Operation::walk, clients are advised not to rely on a specific order though.
If the operand is associated with multiple Payload IR ops, finds matching
ops nested within each of those and produces a single list containing all
of the matched ops.

The transformation is considered successful regardless of whether some
Payload IR ops actually matched the pattern and only fails if the pattern
could not be looked up or compiled.
"""
function pdl_match(root::Value; matched::MLIRType, pattern_name, location=Location())
    results = MLIRType[matched, ]
    operands = Value[root, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("pattern_name", pattern_name), ]
    
    create_operation(
        "transform.pdl_match", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`print`

This op dumps each payload op that is associated with the `target` operand
to stderr. It also prints the `name` string attribute. If no target is
specified, the top-level op is dumped.

This op is useful for printf-style debugging.
"""
function print(target=nothing::Union{Nothing, Value}; name=nothing, location=Location())
    results = MLIRType[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(target) && push!(operands, target)
    !isnothing(name) && push!(attributes, namedattribute("name", name))
    
    create_operation(
        "transform.print", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`replicate`

Produces a new handle associated with a list of payload IR ops that is
computed by repeating the list of payload IR ops associated with the
operand handle as many times as the \"pattern\" handle has associated
operations. For example, if pattern is associated with [op1, op2] and the
operand handle is associated with [op3, op4, op5], the resulting handle
will be associated with [op3, op4, op5, op3, op4, op5].

This transformation is useful to \"align\" the sizes of payload IR lists
before a transformation that expects, e.g., identically-sized lists. For
example, a transformation may be parameterized by same notional per-target
size computed at runtime and supplied as another handle, the replication
allows this size to be computed only once and used for every target instead
of replicating the computation itself.

Note that it is undesirable to pass a handle with duplicate operations to
an operation that consumes the handle. Handle consumption often indicates
that the associated payload IR ops are destroyed, so having the same op
listed more than once will lead to double-free. Single-operand
MergeHandlesOp may be used to deduplicate the associated list of payload IR
ops when necessary. Furthermore, a combination of ReplicateOp and
MergeHandlesOp can be used to construct arbitrary lists with repetitions.
"""
function replicate(pattern::Value, handles::Vector{Value}; replicated::Vector{MLIRType}, location=Location())
    results = MLIRType[replicated..., ]
    operands = Value[pattern, handles..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "transform.replicate", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`sequence`

The transformations indicated by the sequence are applied in order of their
appearance. Each value produced by a transformation within the sequence
corresponds to an operation or a group of operations in the payload IR.
The behavior of the operation when a nested transformation produces a
silenceable error is controlled by the `failure_propagation_mode` attribute.
When set to `propagate`, the failure of any nested transformation in the
sequence implies immediate failure of the entire sequence with a silenceable
error, and no further transformation is attempted. When set to `suppress`,
silenceable errors in nested operations are ignored and further
transformations are applied. Beware that even silenceable errors may leave
the payload IR in a state unsuitable for further transformations. It is
the responsibility of the caller to ensure the following transformations
are robust enough when errors are suppressed. Definite errors reported by
nested transformations abort the sequence regardless of the propagation
mode. The set of modes may be extended in the future, e.g., to collect
silenceable errors and report them after attempting all transformations in
the sequence.

The entry block of this operation has a single argument that maps to either
the operand if provided or the top-level container operation of the payload
IR, typically the root operation of the pass interpreting the transform
dialect. Operand omission is only allowed for sequences not contained in
another sequence.

The body of the sequence terminates with an implicit or explicit
`transform.yield` op. The operands of the terminator are returned as the
results of the sequence op.
"""
function sequence(root=nothing::Union{Nothing, Value}; results::Vector{MLIRType}, failure_propagation_mode, body::Region, location=Location())
    results = MLIRType[results..., ]
    operands = Value[]
    owned_regions = Region[body, ]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("failure_propagation_mode", failure_propagation_mode), ]
    !isnothing(root) && push!(operands, root)
    
    create_operation(
        "transform.sequence", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`split_handles`

Creates `num_result_handles` transform IR handles extracted from the
`handle` operand. The resulting Payload IR operation handles are listed
in the same order as the operations appear in the source `handle`.
This is useful for ensuring a statically known number of operations are
tracked by the source `handle` and to extract them into individual handles
that can be further manipulated in isolation.

This operation succeeds and returns `num_result_handles` if the statically
specified `num_result_handles` corresponds to the dynamic number of
operations contained in the source `handle`. Otherwise it silently fails.
"""
function split_handles(handle::Value; results::Vector{MLIRType}, num_result_handles, location=Location())
    results = MLIRType[results..., ]
    operands = Value[handle, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("num_result_handles", num_result_handles), ]
    
    create_operation(
        "transform.split_handles", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`with_pdl_patterns`

This op contains a set of named PDL patterns that are available for the
Transform dialect operations to be used for pattern matching. For example,
PDLMatchOp can be used to produce a Transform IR value associated with all
Payload IR operations that match the pattern as follows:

```mlir
transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  pdl.pattern @my_pattern : benefit(1) {
    %0 = pdl.operation //...
    // Regular PDL goes here.
    pdl.rewrite %0 with \"transform.dialect\"
  }

  sequence %arg0 failures(propagate) {
  ^bb0(%arg1: !pdl.operation):
    %1 = pdl_match @my_pattern in %arg1
    // Use %1 as handle
  }
}
```

Note that the pattern is expected to finish with a `pdl.rewrite` terminator
that points to the custom rewriter named \"transform.dialect\". The rewriter
actually does nothing, but the transform application will keep track of the
operations that matched the pattern.

This op is expected to contain `pdl.pattern` operations and exactly one
another Transform dialect operation that gets executed with all patterns
available. This op is a possible top-level Transform IR op, the argument of
its entry block corresponds to either the root op of the payload IR or the
ops associated with its operand when provided.
"""
function with_pdl_patterns(root=nothing::Union{Nothing, Value}; body::Region, location=Location())
    results = MLIRType[]
    operands = Value[]
    owned_regions = Region[body, ]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(root) && push!(operands, root)
    
    create_operation(
        "transform.with_pdl_patterns", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`yield`

This terminator operation yields operation handles from regions of the
transform IR ops back to the containing op. It is not itself associated with
any transformation on the payload IR and is used for flow purposes only.
"""
function yield(operands::Vector{Value}; location=Location())
    results = MLIRType[]
    operands = Value[operands..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "transform.yield", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

import ...IR: NamedAttribute, MLIRType, Value, Location, Block, Region, Attribute, create_operation, context, IndexType
import ..Dialects: namedattribute, operandsegmentsizes
import ...API


"""
`affine_simplify_bounded_affine_ops`

Simplify the targeted affine.min / affine.max ops given the supplied
lower and upper bounds for values that may be used as target op operands.

# Example
```
%0 = transform.structured.match ops{[\"affine.min\", \"affine.max\"]} in %arg1
%1 = transform.structured.match ops{[\"gpu.lane_id\"]} in %arg1
transform.affine.simplify_bounded_affine_ops %0 with [%1] within [0] and [32]

// Multiple bounds can be specified.
transform.affine.simplify_bounded_affine_ops %0 with [%1, %2] within [0, 5] and [32, 50]
```

Bounded op handles (`%1` and `%2) must be mapped to ops that have a single
result of index type. The sets of target ops and bounded ops must not
overlap.

#### Return modes

Target ops must be affine.min or affine.max ops. This transform consumes the
target handle and does not produce any handle. It reads the bounded op
handles.

TODO: Support affine.apply targets.
TODO: Allow mixed PDL_Operation/int64_t for lower_bounds and upper_bounds.
"""
function affine_simplify_bounded_affine_ops(target::Value, bounded_values::Vector{Value}; lower_bounds, upper_bounds, location=Location())
    results = MLIRType[]
    operands = Value[target, bounded_values..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("lower_bounds", lower_bounds), namedattribute("upper_bounds", upper_bounds), ]
    
    create_operation(
        "transform.affine.simplify_bounded_affine_ops", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

import ...IR: NamedAttribute, MLIRType, Value, Location, Block, Region, Attribute, create_operation, context, IndexType
import ..Dialects: namedattribute, operandsegmentsizes
import ...API


"""
`bufferization_empty_tensor_to_alloc_tensor`

Replace a tensor.empty with a bufferization.tensor_alloc.

#### Return modes

This operation consumes the `target` handle and produces the `transformed`
handle. `target` is expected to be a `tensor.empty` operation. The transform
always succeeds.
"""
function bufferization_empty_tensor_to_alloc_tensor(target::Value; transformed::MLIRType, location=Location())
    results = MLIRType[transformed, ]
    operands = Value[target, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "transform.bufferization.empty_tensor_to_alloc_tensor", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`bufferization_one_shot_bufferize`

Indicates that the given `target` op should be bufferized with One-Shot
Bufferize. The bufferization can be configured with various attributes that
corresponding to options in `BufferizationOptions` and the
`one-shot-bufferize` pass. More information can be found in the pass
documentation.

If `target_is_module` is set, `target` must be a module. In that case the
`target` handle can be reused by other transform ops. When bufferizing other
ops, the `target` handled is freed after bufferization and can no longer be
used.

Note: Only ops that implement `BufferizableOpInterface` are bufferized. All
other ops are ignored if `allow_unknown_ops`. If `allow_unknown_ops` is
unset, this transform fails when an unknown/non-bufferizable op is found.
Many ops implement `BufferizableOpInterface` via an external model. These
external models must be registered when applying this transform op;
otherwise, said ops would be considered non-bufferizable.
"""
function bufferization_one_shot_bufferize(target::Value; function_boundary_type_conversion=nothing, allow_return_allocs=nothing, allow_unknown_ops=nothing, bufferize_function_boundaries=nothing, create_deallocs=nothing, target_is_module=nothing, test_analysis_only=nothing, print_conflicts=nothing, location=Location())
    results = MLIRType[]
    operands = Value[target, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(function_boundary_type_conversion) && push!(attributes, namedattribute("function_boundary_type_conversion", function_boundary_type_conversion))
    !isnothing(allow_return_allocs) && push!(attributes, namedattribute("allow_return_allocs", allow_return_allocs))
    !isnothing(allow_unknown_ops) && push!(attributes, namedattribute("allow_unknown_ops", allow_unknown_ops))
    !isnothing(bufferize_function_boundaries) && push!(attributes, namedattribute("bufferize_function_boundaries", bufferize_function_boundaries))
    !isnothing(create_deallocs) && push!(attributes, namedattribute("create_deallocs", create_deallocs))
    !isnothing(target_is_module) && push!(attributes, namedattribute("target_is_module", target_is_module))
    !isnothing(test_analysis_only) && push!(attributes, namedattribute("test_analysis_only", test_analysis_only))
    !isnothing(print_conflicts) && push!(attributes, namedattribute("print_conflicts", print_conflicts))
    
    create_operation(
        "transform.bufferization.one_shot_bufferize", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

import ...IR: NamedAttribute, MLIRType, Value, Location, Block, Region, Attribute, create_operation, context, IndexType
import ..Dialects: namedattribute, operandsegmentsizes
import ...API


"""
`gpu_map_foreach_to_blocks`

Target the gpu_launch op and rewrite the top level `scf.foreach_thread`
to distributed gpu.block_id attribute. If `generate_gpu_launch` attribute
is set, then first generates `gpu_launch` and moves the top level
`scf.foreach_thread` inside.

The operation searches top level `scf.foreach_thread` ops under
`gpu_launch` and maps each such op to GPU blocks. Mapping is
one-to-one and the induction variables of `scf.foreach_thread` are
rewritten to gpu.block_id according to the `thread_dim_mapping` attribute.

Dynamic, `scf.foreach_thread` trip counts are currently not supported.
Dynamic block dim sizes are currently not supported.

Only **bufferized** scf.foreach_thread are currently supported.
Only scf.foreach_thread distributed to **at most 3 dimensions** are
currently supported.

The operation alters the block size of the given gpu_launch using
gridDim argument.

#### Return modes:

This operation ignores non-gpu_launch ops and drops them in the return.

If any scf.foreach_thread with tensors is found, the transform definitely
fails.

If all the scf.foreach_thread operations contained within the LaunchOp
referred to by the `target` PDLOperation lower to GPU properly, the
transform succeeds. Otherwise the transform definitely fails.

The returned handle points to the same LaunchOp operand, consuming it and
producing a new SSA value to satisfy chaining and linearity of the IR
properties.
"""
function gpu_map_foreach_to_blocks(target::Value; result::MLIRType, gridDim=nothing, generate_gpu_launch=nothing, location=Location())
    results = MLIRType[result, ]
    operands = Value[target, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(gridDim) && push!(attributes, namedattribute("gridDim", gridDim))
    !isnothing(generate_gpu_launch) && push!(attributes, namedattribute("generate_gpu_launch", generate_gpu_launch))
    
    create_operation(
        "transform.gpu.map_foreach_to_blocks", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`gpu_map_nested_foreach_to_threads`

Target the `gpu.launch op` and rewrite all `scf.foreach_thread`
nested in it to distributed `gpu.thread_id` attribute.

The operation searches for `scf.foreach_thread` ops nested under `target`
and maps each such op to GPU threads. Mapping is one-to-one and the
induction variables of `scf.foreach_thread` are rewritten to
`gpu.thread_id` according to the `mapping` attribute.

Sibling `scf.foreach_thread` are supported in which case, the union of
the number of threads is computed and may result in predication.

Multiple scf.foreach_thread are supported per `gpu.launch` in which case,
the max of all the threads is computed and taken for the global
`gpu.thread_id`. If necessary, `scf.foreach_thread` that do not use the
whole thread range result in predicated computations.

Dynamic `scf.foreach_thread` trip counts are currently not supported.
Dynamic block dim sizes are currently not supported.

Only **bufferized** `scf.foreach_thread` are currently supported.
Only `scf.foreach_thread` distributed to **at most 3 dimensions** are
currently supported.

Barriers are inserted after each scf.foreach_thread op for now.

The operation alters the block size of the given gpu_launch using
blockDim argument.

#### Return modes:

This operation ignores non-gpu_launch ops and drops them in the return.

If any scf.foreach_thread with tensors is found, the transform definitely
fails.

If all the scf.foreach_thread operations contained within the LaunchOp
referred to by the `target` PDLOperation lower to GPU properly, the
transform succeeds. Otherwise the transform definitely fails.

The returned handle points to the same LaunchOp operand, consuming it and
producing a new SSA value to satisfy chaining and linearity of the IR
properties.

#### Example:

```
gpu.launch blocks(%bx, %by, %bz) in (%x = %0, %y = %1, %z = %2)
           threads(%tx, %ty, %tz) in (%tx = %3, %ty = %4, %tz = %5) {
  scf.foreach_thread (%i, %j) in (7, 9) {
    ... // body 1
  } {mapping = [#gpu.thread<x>, #gpu.thread<y>, #gpu.thread<z>]}
  scf.foreach_thread (%i) in (12) {
    ... // body 2
  } {mapping = [#gpu.thread<x>]}
  gpu.terminator
}
```
is translated to:

```
%bdimX = arith.constant 12 : index
%bdimY = arith.constant 9 : index
gpu.launch blocks(%bx, %by, %bz) in (%x = %0, %y = %1, %z = %2)
       threads(%tx, %ty, %tz) in (%tx = %bdimX, %ty = %bdimY, %tz = %5) {
  if (threadIdx.x < 9 && threadIdx.y < 7) {
    ... // body 1
  }
  gpu.barrier
  if (threadIdx.y < 1) {
    ... // body 2
  }
  gpu.barrier
  gpu.terminator
}
```
"""
function gpu_map_nested_foreach_to_threads(target::Value; result::MLIRType, blockDim=nothing, syncAfterDistribute=nothing, location=Location())
    results = MLIRType[result, ]
    operands = Value[target, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(blockDim) && push!(attributes, namedattribute("blockDim", blockDim))
    !isnothing(syncAfterDistribute) && push!(attributes, namedattribute("syncAfterDistribute", syncAfterDistribute))
    
    create_operation(
        "transform.gpu.map_nested_foreach_to_threads", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

import ...IR: NamedAttribute, MLIRType, Value, Location, Block, Region, Attribute, create_operation, context, IndexType
import ..Dialects: namedattribute, operandsegmentsizes
import ...API


"""
`structured_decompose`

Decomposes named complex operations, such as higher-dimensional
(depthwise) convolutions, into combinations of lower-dimensional equivalents
when possible.

#### Return modes

This operation ignores non-Linalg ops and drops them in the return.
If all the operations referred to by the `target` PDLOperation decompose
properly, the transform succeeds. Otherwise the transform silently fails.
The return handle points to only the subset of successfully produced
computational operations, which can be empty.
"""
function structured_decompose(target::Value; transformed::MLIRType, location=Location())
    results = MLIRType[transformed, ]
    operands = Value[target, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "transform.structured.decompose", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`structured_fuse_into_containing_op`

Fuses the `producer_op` into the `containing_op`.
Returns a handle to the fused ops.

The producer is typically a slice of a tileable op (i.e., implements
TilingInterface). In that case, this transform computes the accessed
producer slice inside of the containing op (\"tile and fuse\"). Otherwise,
the entire producer is cloned inside the containing op (\"clone and fuse\").

The containing op handle must be associated with exactly one payload op. The
producer op handle may be associated with multiple payload ops. This
transform fuses producers one-by-one, always picking an unspecified producer
that has at least one use inside the containing op among the
producers.

Note: If a producer has multiple uses inside the containing op, it is
currently tiled and/or cloned multiple times into the containing op.
TODO: Reuse already fused OpResults instead of tiling/cloning a second time
when possible. Fuse producers according to a topological sorting to achieve
the largest amount of reuse.

#### Return modes

If at least one producer could not be fused, this operation fails silently.
This is the case when tiling fails or when no producer op could be found
among the remaining producers that has at least one use within the
containing op. I.e., \"producers\" that are not consumed within the containing
op are rejected by this operation.

This operation reads and frees the producer handle.
This operation reads the containing op handle.
"""
function structured_fuse_into_containing_op(producer_op::Value, containing_op::Value; fused_op::MLIRType, location=Location())
    results = MLIRType[fused_op, ]
    operands = Value[producer_op, containing_op, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "transform.structured.fuse_into_containing_op", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`structured_fuse`

Tiles the operations pointed to by the target handle and fuses their
producers greedily using the options provided as attributes.
"""
function structured_fuse(target::Value; transformed::MLIRType, loops::Vector{MLIRType}, tile_sizes=nothing, tile_interchange=nothing, location=Location())
    results = MLIRType[transformed, loops..., ]
    operands = Value[target, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(tile_sizes) && push!(attributes, namedattribute("tile_sizes", tile_sizes))
    !isnothing(tile_interchange) && push!(attributes, namedattribute("tile_interchange", tile_interchange))
    
    create_operation(
        "transform.structured.fuse", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`structured_generalize`

Transforms a named structured operation into the generic form with the
explicit attached region.

#### Return modes

This operation ignores non-Linalg ops and drops them in the return.
If all the operations referred to by the `target` PDLOperation generalize
properly, the transform succeeds. Otherwise the transform silently fails.
The return handle points to only the subset of successfully produced
equivalent generic operations, which can be empty or contain the original
ops if they were already in generic form.
"""
function structured_generalize(target::Value; transformed::MLIRType, location=Location())
    results = MLIRType[transformed, ]
    operands = Value[target, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "transform.structured.generalize", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`structured_interchange`

Interchanges the iterators of the operations pointed to by the target handle
using the iterator interchange attribute.

#### Return modes

This operation ignores non-linalg::Generic ops and drops them in the return.
This operation fails if the interchange attribute is invalid.
If all the operations referred to by the `target` PDLOperation interchange
properly, the transform succeeds.
If any interchange fails, the transform definitely fails.
The return handle points to only the subset of successfully produced
interchanged operations, which can be empty.
"""
function structured_interchange(target::Value; transformed::MLIRType, iterator_interchange=nothing, location=Location())
    results = MLIRType[transformed, ]
    operands = Value[target, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(iterator_interchange) && push!(attributes, namedattribute("iterator_interchange", iterator_interchange))
    
    create_operation(
        "transform.structured.interchange", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`structured_masked_vectorize`

Vectorize the target ops, which must be Linalg ops, with masked vectors
of the specified size.

The vector sizes can be either static or dynamic (SSA values). In case of
SSA values, the handle must be mapped to exactly one payload op with
exactly one index-typed result.

#### Return modes:

This operation produces a definite failure if the dynamic vector sizes (SSA
values) do not satify the constraints mentioned above. It produces a
silenceable failure if at least one target op is not a Linalg op or fails to
vectorize.
"""
function structured_masked_vectorize(target::Value, vector_sizes::Vector{Value}; static_vector_sizes=nothing, location=Location())
    results = MLIRType[]
    operands = Value[target, vector_sizes..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(static_vector_sizes) && push!(attributes, namedattribute("static_vector_sizes", static_vector_sizes))
    
    create_operation(
        "transform.structured.masked_vectorize", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`structured_match`

Match op with the specified constraints, within the target op.

The following constraints are supported:
  - interface: an optional MatchInterfaceEnum specifying an enum
    representation for an interface to target.
  - ops: an optional StrArrayAttr specifying the concrete name of an op.
    Multiple names can be specified. Matched ops must have one of specified
    names.
  - attribute: the matched op must have all specified attributes (with their
    specified values).
  - filter_result_type: the matched op must return exactly this one type.

Note: Only ops that satisfy all specified constraints are matched.

TODO: Extend with regions to allow a limited form of constraints.

#### Return modes

This op traverses the ops nested under `target` and returns the handles to
all the operations that match the requirements.

This op fails if the target is not a handle to exactly one operation.
Otherwise it succeeds.

This operation does not consume the target handle and produces new handles:
it is a navigation op.
"""
function structured_match(target::Value; results::MLIRType, ops=nothing, interface=nothing, op_attrs=nothing, filter_result_type=nothing, location=Location())
    results = MLIRType[results, ]
    operands = Value[target, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(ops) && push!(attributes, namedattribute("ops", ops))
    !isnothing(interface) && push!(attributes, namedattribute("interface", interface))
    !isnothing(op_attrs) && push!(attributes, namedattribute("op_attrs", op_attrs))
    !isnothing(filter_result_type) && push!(attributes, namedattribute("filter_result_type", filter_result_type))
    
    create_operation(
        "transform.structured.match", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`structured_multitile_sizes`

Emits the IR computing the tile sizes `s1` and `s2` such that:

  - there exists a combination of `n` tiles of size `s1` and `m` tiles of
    size `s2` that covers the entirety of the iteration space `dimension` of
    the target structured op;
  - `s1`, `s2` is less than or equal to `target_size`;
  - `s1` and `s2` are divisible by `divisor.

For example, for a dimension of size 54 with target size 12 and divisor 2,
this can emit the IR computing the tile size 10, used for 3 tiles, and 12,
used for 2 tiles, totally 10*3 + 12*2 = 54. Note that when the divisor does
not divide the original dimension size, it is impossible to compute such
tile sizes. An assertion is emitted to guard against this in the dynamic
case.

Expects the target size and the divisor to be strictly positive. Folds the
IR as much as possible, normally obtaining constant sizes and numbers of
tiles for a statically known dimension.

This does *not* consume the target handle and produces three handles each
pointing to single-result index-typed operations (which may be arithmetic
constant operations) defining the two respective tile sizes and the product
of the first tile size with the number of tiles of that size (useful for
splitting the iteration space).

This operation composes with the regular tiling when applied per-dimension:

```mlir
%sz1, %sz2, %split = structured.multitile_sizes %target
                     { target_size = 10, dimension = 1 }
                   : !transform.any_op, !transform.param<i64>,
                     !transform.param<i64>, !transform.param<i64>
%low, %high = structured.split %target after %split { dimension = 1 }
            : !transform.any_op, !transform.param<i64>
%tiled_low, %loop1 = structured.tile %low [0, %sz1]
                   : (!transform.any_op, !transform.param<i64>)
                  -> (!transform.any_op, !transform.any_op)
%tiled_high, %loop2 = structured.tile %high [0, %sz2]
                    : (!transform.any_op, !transform.param<i64>)
                   -> (!transform.any_op, !transform.any_op)
%common = merge_handles %tiled_low, %tiled_high : !transform.any_op

%sz3, %sz4, %split = structured.multitile_size %target
                     { target_size = 42, dimension = 0 }
                   : !transform.any_op, !transform.any_op,
                     !transform.any_op, !transform.any_op
%sz3r, %sz4r, %splitr = replicate num(%common) %sz3, %sz4, %splitr
         : !transform.any_op, !transform.any_op, !transform.any_op
structured.split %common after %splitr { dimension = 0 }
         : !transform.any_op, !transform.any_op
// ...
```
"""
function structured_multitile_sizes(target::Value; low_size::MLIRType, high_size::MLIRType, split_point::MLIRType, dimension, target_size, divisor=nothing, location=Location())
    results = MLIRType[low_size, high_size, split_point, ]
    operands = Value[target, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("dimension", dimension), namedattribute("target_size", target_size), ]
    !isnothing(divisor) && push!(attributes, namedattribute("divisor", divisor))
    
    create_operation(
        "transform.structured.multitile_sizes", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`structured_pack`

Pack a LinalgOp by applying a data tiling transformation on the op and
packing the operands according to the `packed_sizes` specification.

Iterator dimensions are tiled in their canonical order in the op spec.
Operands are packed according to the same canonical order of the op iterator
dimensions.

Specifying a packed size of 0 for an iterator removes it from consideration
for packing.

`tensor.pack` (resp. `tensor.unpack`) operations are inserted for the operands
(resp. results) that need to be packed (resp. unpacked) according to the
`packed_sizes` specification.

#### Example

Consider a `linalg.matmul` with indexing maps:
```
  //              M   N   K       M   K
  // affine_map<(d0, d1, d2) -> (d0, d2)>
  //                              K   N
  // affine_map<(d0, d1, d2) -> (d2, d1)>
  //                              M   N
  // affine_map<(d0, d1, d2) -> (d0, d1)>
  %0 = linalg.matmul  ins(%A, %B: tensor<?x?xf32>, tensor<?x?xf32>)
                     outs(    %C: tensor<?x?xf32>)
```

Specifying packed_sizes [2, 3, 4] results in tiling the iterator dimensions
M, N and K, in this order, in both the op and its operands.
```
  //              M   N   K   m   n   k       M   K   m   k
  // affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
  //                                          K   N   n   k
  // affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d1, d4, d5)>
  //                                          M   N   m   n
  // affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>
  %0 = linalg.generic_representing_some_higher_d_matmul  
        ins(%A, %B: tensor<?x?x2x4xf32>, tensor<?x?x4x3xf32>)
       outs(    %C: tensor<?x?x2x3xf32>)
```
In particular, note that the second operand `B` has shape `KxNxnxk` (and not
`KxNxkxn` as one could expect by looking **only** at the operand).

Other layouts can be obtained unsurprisingly from this canonical 
transformation by composing the resulting operation with a (future) 
`transform.structured.pack_transpose` op.
This composition allows separating concerns and composes better compared
to adding additional permutation attributes to this transform op.

#### Return modes

This operation applies to a single Linalg op, otherwise it fails.
This operation may produce a definiteFailure if the packing fails for any
reason.

The returned handle point to the packed LinalgOp.
"""
function structured_pack(target::Value, packed_sizes::Vector{Value}; packed_op::MLIRType, static_packed_sizes=nothing, location=Location())
    results = MLIRType[packed_op, ]
    operands = Value[target, packed_sizes..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(static_packed_sizes) && push!(attributes, namedattribute("static_packed_sizes", static_packed_sizes))
    
    create_operation(
        "transform.structured.pack", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`structured_pack_transpose`

Apply a transposition to a single `tensor.pack` (resp. `tensor.unpack`) and 
update the `linalg.generic` op that consumes (resp. produces) the operation.

This transform allows composing a simple `structured.pack` with additional
transpositions to e.g. match the data format required by a specific library
call or ISA instruction.

The transpose spec must specify at least one of `outer_perm` or `inner_perm`
attributes, which will act upon the `outer_dims_perm` or `inner_dims_pos` of
the specified `tensor.pack` or `tensor.unpack` op.

If the `target` of this op is a `tensor.pack` then a new `tensor.empty` will
be created along with transposed versions of the `tensor.pack` and the 
consuming `linalg.generic`, which is expected to be the sole consumer.

If the `target` of this op is a `tensor.unpack` then the whole pack / compute
/ unpack chain will be transposed and transposed clones of `tensor.pack`,
the consuming `linalg.generic` and the tail `tensor.pack` will be created.

#### Return modes

This operation targets a single `tensor.pack` / `tensor.unpack` op and a
single matching `linalg.generic` that consumes / produces the op. Otherwise,
it produces a silenceableFailure.

This operation may produce a silenceableFailure if the transpose spec is
ill-formed (i.e. `outer_perm` or `inner_perm` are not permutations of the
proper rank) or if the tranposition of all involved operations fails for any
reason.

This operation returns 3 handles, one to the transformed LinalgOp, one to
the transformed `tensor.pack` and one to the transformed `tensor.unpack`.
The last handle for `tensor.unpack` is empty if `target_pack_or_unpack_op` 
was not itself a `tensor.unpack`.
"""
function structured_pack_transpose(target_pack_or_un_pack_op::Value, target_linalg_op::Value; packed_op::MLIRType, pack_op::MLIRType, un_pack_op::MLIRType, outer_perm=nothing, inner_perm=nothing, location=Location())
    results = MLIRType[packed_op, pack_op, un_pack_op, ]
    operands = Value[target_pack_or_un_pack_op, target_linalg_op, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(outer_perm) && push!(attributes, namedattribute("outer_perm", outer_perm))
    !isnothing(inner_perm) && push!(attributes, namedattribute("inner_perm", inner_perm))
    
    create_operation(
        "transform.structured.pack_transpose", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`structured_pad`

Pads the operations pointed to by the target handle using the options
provides as operation attributes.

#### Return modes

This operation ignores non-Linalg ops and drops them in the return.
This operation may produce a definiteFailure if the padding fails for any
reason.
If all the operations referred to by the `target` PDLOperation pad
properly, the transform succeeds. Otherwise the transform silently fails.
The return handle points to only the subset of successfully produced
padded operations, which can be empty.
"""
function structured_pad(target::Value; transformed::MLIRType, padding_values=nothing, padding_dimensions=nothing, pack_paddings=nothing, hoist_paddings=nothing, transpose_paddings=nothing, location=Location())
    results = MLIRType[transformed, ]
    operands = Value[target, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(padding_values) && push!(attributes, namedattribute("padding_values", padding_values))
    !isnothing(padding_dimensions) && push!(attributes, namedattribute("padding_dimensions", padding_dimensions))
    !isnothing(pack_paddings) && push!(attributes, namedattribute("pack_paddings", pack_paddings))
    !isnothing(hoist_paddings) && push!(attributes, namedattribute("hoist_paddings", hoist_paddings))
    !isnothing(transpose_paddings) && push!(attributes, namedattribute("transpose_paddings", transpose_paddings))
    
    create_operation(
        "transform.structured.pad", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`structured_promote`

Promotes the specified operands of the target into a separate memory buffer.

At this point, this transform does not allow customizing alloc/dealloc
functions nor the behavior on copy in/out operations.

#### Return modes

This operation applies to a single Linalg op that satisfies the
`promoteSubviewsPrecondition`, otherwise it fails.

If the operations referred to by the `target` PDLOperation promote
properly, the transform succeeds.

When successful, the return handle points to the \$target operation that
was modified inplace.
"""
function structured_promote(target::Value; transformed::MLIRType, operands_to_promote=nothing, use_full_tile_buffers=nothing, use_full_tiles_by_default=nothing, use_alloca=nothing, alignment=nothing, location=Location())
    results = MLIRType[transformed, ]
    operands = Value[target, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(operands_to_promote) && push!(attributes, namedattribute("operands_to_promote", operands_to_promote))
    !isnothing(use_full_tile_buffers) && push!(attributes, namedattribute("use_full_tile_buffers", use_full_tile_buffers))
    !isnothing(use_full_tiles_by_default) && push!(attributes, namedattribute("use_full_tiles_by_default", use_full_tiles_by_default))
    !isnothing(use_alloca) && push!(attributes, namedattribute("use_alloca", use_alloca))
    !isnothing(alignment) && push!(attributes, namedattribute("alignment", alignment))
    
    create_operation(
        "transform.structured.promote", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`structured_replace`

Replace all `target` payload ops with the single op that is contained in
this op\'s region. All targets must have zero arguments and must be isolated
from above.

This op is for debugging/experiments only.

#### Return modes

This operation consumes the `target` handle.
"""
function structured_replace(target::Value; replacement::MLIRType, bodyRegion::Region, location=Location())
    results = MLIRType[replacement, ]
    operands = Value[target, ]
    owned_regions = Region[bodyRegion, ]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "transform.structured.replace", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`structured_scalarize`

Indicates that ops of a specific kind in the given function should be
scalarized (i.e. their dynamic dimensions tiled by 1).

#### Return modes:

This operation ignores non-Linalg ops and drops them in the return.
This operation produces `definiteFailure` if the scalarization fails for any
reason.
If all the operations referred to by the `target` PDLOperation scalarize
properly, the transform succeeds. Otherwise the transform silently fails.

The return handle points to only the subset of successfully produced
tiled-by-1 operations, which can be empty.

This operation does not return handles to the tiled loop.
We make this design choice because it is hard to know ahead of time the
number of loops that will be produced (it depends on the number of dynamic
dimensions after multiple transformations have been applied).
Loops can always be recovered by navigating from the tiled operations if
needed.
"""
function structured_scalarize(target::Value; result::MLIRType, location=Location())
    results = MLIRType[result, ]
    operands = Value[target, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "transform.structured.scalarize", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`structured_split`

Indicates that the given `target` op should be split into two complementary
parts, which combined cover the entire iteration domain of the original op.
The split is performed along the iteration space dimension provided as
attribute. In case of dimension overflow, the transformation fails. The
split is performed at the dimension iterator value specified as either the
static split point attribute when it is known at transform IR construction
time or as the handle to an operation producing a single index-typed value
when it is computed by payload IR. In the latter case, the static split
point must be set to `ShapedType::kDynamic` and the dynamic size handle
must point to as many value-producing operations as there are structured
operations pointed to by the target handle.

The operation consumes the target handle, but preserves the split point
handle if provided. It produces two new handles pointing to the two parts
of the structured op after splitting, in the same order as the target
operand, with the first handle corresponding to the part with lower
iteration space indices.
"""
function structured_split(target::Value, dynamic_split_point=nothing::Union{Nothing, Value}; first::MLIRType, second::MLIRType, dimension, static_split_point, location=Location())
    results = MLIRType[first, second, ]
    operands = Value[target, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("dimension", dimension), namedattribute("static_split_point", static_split_point), ]
    !isnothing(dynamic_split_point) && push!(operands, dynamic_split_point)
    
    create_operation(
        "transform.structured.split", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`structured_split_reduction`

Indicates that the given `target` op should be transformed with the
`splitReduction` transformation and split factor provided as attribute.

The `splitReduction` transformation splits the first single linalg op
reduction into a parallel and reduction dimension.
A new `linalg.generic` op is created to perform the rest of the reduction.

The transformation supports different configurations attributes:
  - split_factor: the factor by which to split (i.e. the size of the
    remaining reduction after splitting).
  - insert_split_dimension: the dimension in the temporary tensor into
    which the new parallel dimension is inserted.
  - inner_parallel: specifies whether the parallel dimension is before or
    after the reduction dimension in the splitting op.
  - use_scaling_algorithm: whether to use a scaling based formulation that
    does not create an ExpandShapeOp (default: do not use scaling)
  - use_alloc: whether to use an alloc op to allocate the temporary
    tensor (default: do not use alloc op)

#### Return modes

This operation ignores non-Linalg ops and drops them in the return.
This operation produces `definiteFailure` if the splitting fails for any
reason.

If all the operations referred to by the `target` PDLOperation split
properly, the transform succeeds. Otherwise the transform silently fails.
The 4 returned handles points to only the subset of successfully produced
computational operations, which can all be empty.
This 4 returned handles point to:
  - the init op (or tensor_alloc op if use_alloc = true),
  - the fill op used to initialize the neutral element,
  - the split op and
  - the result-combining op.

#### Example (default: `use_scaling_algorithm = false, use_alloc = false`):

```
  %r = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>,
                                        affine_map<(d0) -> ()>],
        iterator_types = [\"reduction\"]}
  ins(%in : tensor<32xf32>)
  outs(%out : tensor<f32>) {
  ^bb0(%arg1: f32, %arg2: f32):
    %y = arith.addf %arg1, %arg2 : f32
    linalg.yield %y : f32
  } -> tensor<f32>
```

is split into:

```
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.expand_shape %in [[0, 1]] : tensor<32xf32> into tensor<4x8xf32>
  %1 = tensor.empty() : tensor<4xf32>
  %2 = linalg.fill ins(%cst : f32) outs(%1 : tensor<4xf32>) -> tensor<4xf32>
  %3 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                                        affine_map<(d0, d1) -> (d0)>],
    iterator_types = [\"parallel\", \"reduction\"]}
    ins(%0 : tensor<4x8xf32>) outs(%2 : tensor<4xf32>) {
    ^bb0(%arg3: f32, %arg5: f32):
    %5 = arith.addf %arg3, %arg4 : f32
    linalg.yield %5 : f32
  } -> tensor<4xf32>
  %r = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>,
                                        affine_map<(d0) -> ()>],
    iterator_types = [\"reduction\"]}
    ins(%3 : tensor<4xf32>) outs(%out : tensor<f32>) {
    ^bb0(%arg3: f32, %arg4: f32):
    %5 = arith.addf %arg3, %arg4 : f32
    linalg.yield %5 : f32
  } -> tensor<f32>
```

#### Example (`use_scaling_algorithm = true, use_alloc = true`):

Instead of introducing an ExpandShapeOp, this scaling-based implementation
rewrites a reduction dimension `k` into `k * split_factor + kk`.
The dimension `kk` is added as an extra parallel dimension to the
intermediate output tensor at position `insert_split_dimension`.

Consider a minimal example where `k` is reduced:
    O(i, j) += I(i, j, k)
Assume i=3, j=5, k=128, split_factor=16 and insert_split_dimension=0.
The compute is rewritten as:
  a. O_i(kk, i, j) += I(i, j, 16 * k + kk)
  b. O(i, j) += O_i(kk, i, j)
The intermediate tensor O_i is of shape (128/16)x3x5 == 8x3x5.

#### Example:

```
 %0 = linalg.matmul ins(%A, %B: tensor<16x256xf32>, tensor<256x32xf32>)
   outs(%C: tensor<16x32xf32>) -> tensor<16x32xf32>
```

Is transformed to:

```
 #map0 = affine_map<(d0, d1, d2, d3) -> (d0, d2 * 4 + d3)>
 #map1 = affine_map<(d0, d1, d2, d3) -> (d2 * 4 + d3, d1)>
 #map2 = affine_map<(d0, d1, d2, d3) -> (d2, d3)>
 #map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
 #map4 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
 #map5 = affine_map<(d0, d1, d2) -> (d0, d1)>
 %0 = tensor.empty() : tensor<16x32x64xf32>
 %cst = arith.constant 0.000000e+00 : f32
 %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<16x32x64xf32>) ->
    tensor<16x32x64xf32>
 %2 = tensor.empty() : tensor<64x4xi1>

 %3 = linalg.generic {indexing_maps = [#map0, #map1, #map2, #map3],
   iterator_types = [\"parallel\", \"parallel\", \"parallel\", \"reduction\"]}
   ins(%A, %B, %2 : tensor<16x256xf32>, tensor<256x32xf32>, tensor<64x4xi1>)
   outs(%1 : tensor<16x32x64xf32>) {
     ^bb0(%arg3: f32, %arg4: f32, %arg5: i1, %arg6: f32):
       %5 = arith.mulf %arg3, %arg4 : f32
       %6 = arith.addf %arg6, %5 : f32
       linalg.yield %6 : f32
 } -> tensor<16x32x64xf32>

 %4 = linalg.generic {indexing_maps = [#map4, #map5],
   iterator_types = [\"parallel\", \"parallel\", \"reduction\"]}
   ins(%3 : tensor<16x32x64xf32>)
   outs(%C : tensor<16x32xf32>) {
     ^bb0(%arg3: f32, %arg4: f32):
       %5 = arith.addf %arg3, %arg4 : f32
       linalg.yield %5 : f32
 } -> tensor<16x32xf32>

 return %4 : tensor<16x32xf32>
```
"""
function structured_split_reduction(target::Value; init_or_alloc_op::MLIRType, fill_op::MLIRType, split_linalg_op::MLIRType, combining_linalg_op::MLIRType, split_factor=nothing, insert_split_dimension=nothing, inner_parallel=nothing, use_scaling_algorithm=nothing, use_alloc=nothing, location=Location())
    results = MLIRType[init_or_alloc_op, fill_op, split_linalg_op, combining_linalg_op, ]
    operands = Value[target, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(split_factor) && push!(attributes, namedattribute("split_factor", split_factor))
    !isnothing(insert_split_dimension) && push!(attributes, namedattribute("insert_split_dimension", insert_split_dimension))
    !isnothing(inner_parallel) && push!(attributes, namedattribute("inner_parallel", inner_parallel))
    !isnothing(use_scaling_algorithm) && push!(attributes, namedattribute("use_scaling_algorithm", use_scaling_algorithm))
    !isnothing(use_alloc) && push!(attributes, namedattribute("use_alloc", use_alloc))
    
    create_operation(
        "transform.structured.split_reduction", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`structured_tile`

Indicates that the given `target` op should be tiled with the given sizes.
This transform generates a loop nest with a smaller (\"tiled\") target
operation in its body. Currently limited to LinalgOps.

Tile sizes may be known at transformation time, in which case they are
expected to be provided in the `static_size` attribute, or not, in which
case the tile value must be computed by the payload IR and the handle to the
operation computing it must be provided through `dynamic_sizes`. When the
sizes are not known statically, the corresponding entry in the
`static_sizes` attribute must be set to `ShapedType::kDynamic`. Only
the dynamic sizes must be provided in `dynamic_sizes`, i.e., there should
be as many handles as `ShapedType::kDynamic` values in the
`static_sizes` attribute. A static size of `0` indicates that the dimension
should not be tiled. No loop will be generated for such dimensions. If all
tile sizes are `0`, this transform is effectively a no-op.

This op returns handles to the tiled op (in the generated loop nest) and the
generated loops. The number of loops is the number of tile sizes that are
statically known to be non-zero.

#### Return modes

On success, the resulting handles are associated with co-indexed lists of
tiled operations and loops around them.

This operation only supports Linalg ops and produces a silenceable failure
if the input contains any non-Linalg ops. The ops preceding it in the list
associated with the `target` handle will have been tiled.

This operation produces a silenceable failure if the `dynamic_sizes` handles
are associated with lists of payload operations of a size different than
that of the list associated with the `target` handle.

If the internal implementation of tiling for any of the operations fails,
produces a definite failure.
"""
function structured_tile(target::Value, dynamic_sizes::Vector{Value}; tiled_linalg_op::MLIRType, loops::Vector{MLIRType}, static_sizes=nothing, interchange=nothing, location=Location())
    results = MLIRType[tiled_linalg_op, loops..., ]
    operands = Value[target, dynamic_sizes..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(static_sizes) && push!(attributes, namedattribute("static_sizes", static_sizes))
    !isnothing(interchange) && push!(attributes, namedattribute("interchange", interchange))
    
    create_operation(
        "transform.structured.tile", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`structured_tile_reduction_using_foreach_thread`

Tile a PartialReductionOpInterface op to a tiled `scf.foreach_thread` doing
partial reduction.

This transformation tiles the `target` along the reduction dimensions. It
creates a tensor initialized with the identity value. Then it creates a
`scf.foreach_thread` loops with the number threads given by `num_threads`.
The op is tiled op with a size equal to `floordiv(size, num_threads)`.
All the partial reduction value is are parallel inserted to create a new
tensor. After the loop a merge operation is created to do a final reduction
with the partial reductions tensor.
If an extra `tile_sizes` parameter is passed the tiles are cyclically
distributed on the threads of the `scf.foreach_threads` loop.

#### Return modes

This 4 returned handles point to:
  - the parent foreach_thread op,
  - the fill op used to initialize the neutral element,
  - the parallel tiled op and
  - the result-combining op.

#### Example:

```
  %red = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                                          affine_map<(d0, d1) -> (d0)>],
  iterator_types = [\"parallel\", \"reduction\"]}
  ins(%arg0 : tensor<?x?xf32>)
  outs(%out : tensor<?xf32>) {
    ^bb0(%arg7: f32, %arg9: f32):
    %1 = arith.addf %arg7, %arg9 : f32
    linalg.yield %1 : f32
  } -> tensor<?xf32>
  return %red : tensor<?xf32>
```

is transformed into:

```
  %0 = tensor.empty(%dim_1) : tensor<?x5xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<?x5xf32>) -> tensor<?x5xf32>
  %2 = scf.foreach_thread (%arg2) in (%c5) shared_outs(%arg3 = %1) -> (tensor<?x5xf32>) {
    %4 = affine.min #map(%arg2)[%dim_0]
    %5 = affine.max #map1(%4)
    %extracted_slice = tensor.extract_slice %arg3[0, %arg2] [%dim, 1] [1, 1] : tensor<?x5xf32> to tensor<?xf32>
    %6 = affine.apply #map2(%arg2)[%dim_0]
    %extracted_slice_2 = tensor.extract_slice %arg0[0, %6] [%dim, %5] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
    %extracted_slice_3 = tensor.extract_slice %extracted_slice[0] [%dim] [1] : tensor<?xf32> to tensor<?xf32>
    %7 = linalg.generic {indexing_maps = [#map3, #map4], iterator_types = [\"parallel\", \"reduction\"]} ins(%extracted_slice_2 : tensor<?x?xf32>) outs(%extracted_slice_3 : tensor<?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %9 = arith.addf %in, %out : f32
      linalg.yield %9 : f32
    } -> tensor<?xf32>
    scf.foreach_thread.perform_concurrently {
      tensor.parallel_insert_slice %7 into %arg3[0, %arg2] [%dim, 1] [1, 1] : tensor<?xf32> into tensor<?x5xf32>
    }
  } {mapping = []}
  %3 = linalg.generic {indexing_maps = [#map3, #map4], iterator_types = [\"parallel\", \"reduction\"]} ins(%2 : tensor<?x5xf32>) outs(%arg1 : tensor<?xf32>) {
  ^bb0(%in: f32, %out: f32):
    %4 = arith.addf %in, %out : f32
    linalg.yield %4 : f32
  } -> tensor<?xf32>
```
"""
function structured_tile_reduction_using_foreach_thread(target::Value; foreach_thread_op::MLIRType, fill_op::MLIRType, split_linalg_op::MLIRType, combining_linalg_op::MLIRType, num_threads=nothing, tile_sizes=nothing, mapping=nothing, location=Location())
    results = MLIRType[foreach_thread_op, fill_op, split_linalg_op, combining_linalg_op, ]
    operands = Value[target, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(num_threads) && push!(attributes, namedattribute("num_threads", num_threads))
    !isnothing(tile_sizes) && push!(attributes, namedattribute("tile_sizes", tile_sizes))
    !isnothing(mapping) && push!(attributes, namedattribute("mapping", mapping))
    
    create_operation(
        "transform.structured.tile_reduction_using_foreach_thread", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`structured_tile_reduction_using_scf`

Indicates that the given `target` op should be transformed with the
`tileReduction` transformation with the tile size provided as attribute.

This transformation tiles the `target` along the reduction dimensions. It
creates a tensor initialized with the identity value. Then it creates nested
loops with a parallel version of `target` op inside. The parallel op
dimensions are less or equal to the tile size passed by user.
After the loop a merge operation is created to do a final reduction with the
partial reductions.
The initial tensor always uses the tile size dimension. This may overallocate
if the tile size is greater than the reduction dimension.

#### Return modes

This 4 returned handles point to:
  - the parent for op,
  - the fill op used to initialize the neutral element,
  - the parallel tiled op and
  - the result-combining op.

#### Example:

```
  %red = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                                          affine_map<(d0, d1) -> (d0)>],
  iterator_types = [\"parallel\", \"reduction\"]}
  ins(%arg0 : tensor<?x?xf32>)
  outs(%out : tensor<?xf32>) {
    ^bb0(%arg7: f32, %arg9: f32):
    %1 = arith.addf %arg7, %arg9 : f32
    linalg.yield %1 : f32
  } -> tensor<?xf32>
  return %red : tensor<?xf32>
```

is transformed into:

```
  %0 = tensor.empty(%dim_1) : tensor<?x5xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<?x5xf32>) -> tensor<?x5xf32>
  %2 = scf.for %arg2 = %c0 to %dim_0 step %c5 iter_args(%arg3 = %1) -> (tensor<?x5xf32>) {
    %extracted_slice = tensor.extract_slice %1[0, 0] [%dim, 5] [1, 1] : tensor<?x5xf32> to tensor<?x5xf32>
    %extracted_slice_2 = tensor.extract_slice %arg0[0, %arg2] [%dim, 5] [1, 1] : tensor<?x?xf32> to tensor<?x5xf32>
    %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                                          affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = [\"parallel\", \"parallel\"]}
    ins(%extracted_slice_2 : tensor<?x5xf32>)
    outs(%extracted_slice : tensor<?x5xf32>) {
    ^bb0(%in: f32, %out: f32):
      %5 = arith.addf %in, %out : f32
      linalg.yield %5 : f32
    } -> tensor<?x5xf32>
    %dim_3 = tensor.dim %1, %c0 : tensor<?x5xf32>
    %inserted_slice = tensor.insert_slice %4 into %arg3[0, 0] [%dim_3, 5] [1, 1] : tensor<?x5xf32> into tensor<?x5xf32>
    scf.yield %inserted_slice : tensor<?x5xf32>
  }
  %3 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                                        affine_map<(d0, d1) -> (d0)>],
  iterator_types = [\"parallel\", \"reduction\"]}
  ins(%2 : tensor<?x5xf32>)
  outs(%arg1 : tensor<?xf32>) {
  ^bb0(%in: f32, %out: f32):
    %4 = arith.addf %in, %out : f32
    linalg.yield %4 : f32
  } -> tensor<?xf32>
```
"""
function structured_tile_reduction_using_scf(target::Value; for_op::MLIRType, fill_op::MLIRType, split_linalg_op::MLIRType, combining_linalg_op::MLIRType, tile_sizes=nothing, location=Location())
    results = MLIRType[for_op, fill_op, split_linalg_op, combining_linalg_op, ]
    operands = Value[target, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(tile_sizes) && push!(attributes, namedattribute("tile_sizes", tile_sizes))
    
    create_operation(
        "transform.structured.tile_reduction_using_scf", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`structured_tile_to_foreach_thread_op`

Tile a TilingInterface op to a tiled `scf.foreach_thread`.

Tiling is applied by either specifying `num_threads` or `tile_size`. If
`num_threads` is specified, then the tile size for each dimension `i` is
calculated dynamically via `ceilDiv(dimSize[i], num_threads[i])`.
`num_threads` and `tile_size` can be either static index attributes or SSA
values of PDL operation handle type (or a mix thereof). Operation handles
must be mapped to exactly one op that has exactly one result of index type.

Static zero tile sizes indicate that the dimension is not tiled and can be
thought of as tiling by the full size of data.

It is the user\'s responsibility to ensure that `num_threads/tile_sizes` is
a valid tiling specification (i.e. that only tiles parallel dimensions,
e.g. in the Linalg case).

If non-empty, the `mapping` is added as an attribute to the
resulting `scf.foreach_thread`.

Note: `tile_sizes` and `num_threads` are variadic. Each tile size/number of
threads can be an index attribute or a transform handle that is mapped to
exactly one payload op with exactly one index result.

#### Return modes

This operation ignores ops that do not implement the TilingInterface and
drops them in the return.

If all the operations referred to by the `target` PDLOperation tile
successfully, the transform succeeds.
Otherwise the transform silently fails.

The two returned handles point to only the subset of successfully produced
tiled operations, which can all be empty.

These two returned handles point to:
  - the new scf.foreach_thread op,
  - the tiled op that implements TilingInterface.

#### Example using `num_threads`

```
%0 = pdl_match @match_matmul in %arg1
%3:2 = transform.structured.tile_to_foreach_thread_op %0 num_threads [10, 20]
```

#### Example using `tile_sizes`

```
%0 = pdl_match @match_matmul in %arg1
%sz = pdl_match @match_size_op in %arg1
%3:2 = transform.structured.tile_to_foreach_thread_op %0 tile_sizes [0, %sz, 20]
```
"""
function structured_tile_to_foreach_thread_op(target::Value, num_threads::Vector{Value}, tile_sizes::Vector{Value}, packed_num_threads=nothing::Union{Nothing, Value}; packed_tile_sizes=nothing::Union{Nothing, Value}, foreach_thread_op::MLIRType, tiled_op::MLIRType, static_num_threads=nothing, static_tile_sizes=nothing, mapping=nothing, location=Location())
    results = MLIRType[foreach_thread_op, tiled_op, ]
    operands = Value[target, num_threads..., tile_sizes..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(packed_num_threads) && push!(operands, packed_num_threads)
    !isnothing(packed_tile_sizes) && push!(operands, packed_tile_sizes)
    push!(attributes, operandsegmentsizes([1, length(num_threads), length(tile_sizes), (packed_num_threads==nothing) ? 0 : 1(packed_tile_sizes==nothing) ? 0 : 1]))
    !isnothing(static_num_threads) && push!(attributes, namedattribute("static_num_threads", static_num_threads))
    !isnothing(static_tile_sizes) && push!(attributes, namedattribute("static_tile_sizes", static_tile_sizes))
    !isnothing(mapping) && push!(attributes, namedattribute("mapping", mapping))
    
    create_operation(
        "transform.structured.tile_to_foreach_thread_op", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`structured_tile_to_scf_for`

Indicates that the given `target` op should be tiled with the given sizes.
This transform generates a loop nest with a smaller (\"tiled\") target
operation in its body. The target must implement TilingInterface.

Tile sizes may be known at transformation time, in which case they are
expected to be provided in the `static_size` attribute, or not, in which
case the tile value must be computed by the payload IR and the handle to the
operation computing it must be provided through `dynamic_sizes`. When the
sizes are not known statically, the corresponding entry in the
`static_sizes` attribute must be set to `ShapedType::kDynamic`. Only
the dynamic sizes must be provided in `dynamic_sizes`, i.e., there should
be as many handles as `ShapedType::kDynamic` values in the
`static_sizes` attribute. A static size of `0` indicates that the dimension
should not be tiled. No loop will be generated for such dimensions. If all
tile sizes are `0`, this transform is effectively a no-op.

This op returns handles to the tiled op (in the generated loop nest) and the
generated loops. The number of loops is the number of tile sizes that are
statically known to be non-zero.

#### Return modes

On success, the resulting handles are associated with co-indexed lists of
tiled operations and loops around them.

This operation only supports TilingInterface ops and produces a silenceable
failure if the input contains any non-TilingInterface ops. The ops preceding
it in the list associated with the `target` handle will have been tiled.

This operation produces a silenceable failure if the `dynamic_sizes` handles
are associated with lists of payload operations of a size different than
that of the list associated with the `target` handle.

If the internal implementation of tiling for any of the operations fails,
produces a definite failure.
"""
function structured_tile_to_scf_for(target::Value, dynamic_sizes::Vector{Value}; tiled_linalg_op::MLIRType, loops::Vector{MLIRType}, static_sizes=nothing, interchange=nothing, location=Location())
    results = MLIRType[tiled_linalg_op, loops..., ]
    operands = Value[target, dynamic_sizes..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(static_sizes) && push!(attributes, namedattribute("static_sizes", static_sizes))
    !isnothing(interchange) && push!(attributes, namedattribute("interchange", interchange))
    
    create_operation(
        "transform.structured.tile_to_scf_for", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`structured_vectorize`

Indicates that the given `target` op all the ops it contains should be
vectorized with the configuration specified by the attributes of this op.
This vectorization only handles structured ops that operate on shaped types
and does not vectorize loops or straight-line. Internally, it applies a
set of rewrite patterns, some of which enable vectorization and some of
which clean up the results. Therefore, it can only be applied to an op with
the \"isolated from above property\". If finer granularity is required, it can
be achieved by outlining the target part of the payload IR into, e.g., a
function, performing the transformation, and inlining it back. This
transformation only fails if the entire pattern rewriting failed, i.e., it
does **not** fail when no ops were vectorized.

Note that this transformation is invalidating the handles to any payload IR
operation that is contained inside the vectorization target.

This transformation supports the following attributes:
  - `vectorize_padding`: a UnitAttr to activate the vectorization of
  `tensor.pad` ops. Different pipelines may prefer to lower such ops to
  loops.
  - `disable_multi_reduction_to_contract_patterns`: a UnitAttr to deactivate
  the rewrite of `vector.multi_reduction` to `vector.contract`. This is
  intended to be used in tests only.
  - `disable_transfer_permutation_map_lowering_patterns`: a UnitAttr to
  deactivate the rewrite of `vector.transfer` with permutation maps into
  explicit `vector.transpose` operations. This is intended to be used in
  tests only but may be promotoed to a first class attribute in the future.

#### Return modes:

This operation produces `definiteFailure` if vectorization fails for any
reason.
The operation always returns the handle to the target op that is expected
to be isolated from above.
"""
function structured_vectorize(target::Value; transformed::MLIRType, vectorize_padding=nothing, vectorize_nd_extract=nothing, disable_multi_reduction_to_contract_patterns=nothing, disable_transfer_permutation_map_lowering_patterns=nothing, location=Location())
    results = MLIRType[transformed, ]
    operands = Value[target, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(vectorize_padding) && push!(attributes, namedattribute("vectorize_padding", vectorize_padding))
    !isnothing(vectorize_nd_extract) && push!(attributes, namedattribute("vectorize_nd_extract", vectorize_nd_extract))
    !isnothing(disable_multi_reduction_to_contract_patterns) && push!(attributes, namedattribute("disable_multi_reduction_to_contract_patterns", disable_multi_reduction_to_contract_patterns))
    !isnothing(disable_transfer_permutation_map_lowering_patterns) && push!(attributes, namedattribute("disable_transfer_permutation_map_lowering_patterns", disable_transfer_permutation_map_lowering_patterns))
    
    create_operation(
        "transform.structured.vectorize", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

import ...IR: NamedAttribute, MLIRType, Value, Location, Block, Region, Attribute, create_operation, context, IndexType
import ..Dialects: namedattribute, operandsegmentsizes
import ...API


"""
`memref_multibuffer`

Transformation to do multi-buffering/array expansion to remove
dependencies on the temporary allocation between consecutive loop
iterations. This transform expands the size of an allocation by
a given multiplicative factor and fixes up any users of the
multibuffered allocation.

#### Return modes

This operation returns the new allocation if multi-buffering
succeeds, and failure otherwise.
"""
function memref_multibuffer(target::Value; transformed::MLIRType, factor, location=Location())
    results = MLIRType[transformed, ]
    operands = Value[target, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("factor", factor), ]
    
    create_operation(
        "transform.memref.multibuffer", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

import ...IR: NamedAttribute, MLIRType, Value, Location, Block, Region, Attribute, create_operation, context, IndexType
import ..Dialects: namedattribute, operandsegmentsizes
import ...API


"""
`loop_get_parent_for`

Produces a handle to the n-th (default 1) parent `scf.for` or `affine.for`
(when the affine flag is true) loop for each Payload IR operation
associated with the operand. Fails if such a loop cannot be found. The list
of operations associated with the handle contains parent operations in the
same order as the list associated with the operand, except for operations
that are parents to more than one input which are only present once.
"""
function loop_get_parent_for(target::Value; parent::MLIRType, num_loops=nothing, affine=nothing, location=Location())
    results = MLIRType[parent, ]
    operands = Value[target, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(num_loops) && push!(attributes, namedattribute("num_loops", num_loops))
    !isnothing(affine) && push!(attributes, namedattribute("affine", affine))
    
    create_operation(
        "transform.loop.get_parent_for", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`loop_coalesce`

Given a perfect loop nest identified by the outermost loop,
perform loop coalescing in a bottom-up one-by-one manner.

#### Return modes

The return handle points to the coalesced loop if coalescing happens, or
the given input loop if coalescing does not happen.
"""
function loop_coalesce(target::Value; transformed::MLIRType, location=Location())
    results = MLIRType[transformed, ]
    operands = Value[target, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "transform.loop.coalesce", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`loop_outline`

Moves the loop into a separate function with the specified name and
replaces the loop in the Payload IR with a call to that function. Takes
care of forwarding values that are used in the loop as function arguments.
If the operand is associated with more than one loop, each loop will be
outlined into a separate function. The provided name is used as a _base_
for forming actual function names following SymbolTable auto-renaming
scheme to avoid duplicate symbols. Expects that all ops in the Payload IR
have a SymbolTable ancestor (typically true because of the top-level
module). Returns the handle to the list of outlined functions in the same
order as the operand handle.
"""
function loop_outline(target::Value; transformed::MLIRType, func_name, location=Location())
    results = MLIRType[transformed, ]
    operands = Value[target, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("func_name", func_name), ]
    
    create_operation(
        "transform.loop.outline", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`loop_peel`

Updates the given loop so that its step evenly divides its range and puts
the remaining iteration into a separate loop or a conditional.

In the absence of sufficient static information, this op may peel a loop,
even if the step always divides the range evenly at runtime.

#### Return modes

This operation ignores non-scf::ForOp ops and drops them in the return.

This operation always succeeds and returns the scf::ForOp with the
postcondition: \"the loop trip count is divisible by the step\".
This operation may return the same unmodified loop handle when peeling did
not modify the IR (i.e. the loop trip count was already divisible).

Note that even though the Payload IR modification may be performed
in-place, this operation consumes the operand handle and produces a new
one.

TODO: Return both the peeled loop and the remainder loop.
"""
function loop_peel(target::Value; transformed::MLIRType, fail_if_already_divisible=nothing, location=Location())
    results = MLIRType[transformed, ]
    operands = Value[target, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(fail_if_already_divisible) && push!(attributes, namedattribute("fail_if_already_divisible", fail_if_already_divisible))
    
    create_operation(
        "transform.loop.peel", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`loop_pipeline`

Transforms the given loops one by one to achieve software pipelining for
each of them. That is, performs some amount of reads from memory before the
loop rather than inside the loop, the same amount of writes into memory
after the loop, and updates each iteration to read the data for a following
iteration rather than the current one.

The amount is specified by the attributes.

The values read and about to be stored are transferred as loop iteration
arguments. Currently supports memref and vector transfer operations as
memory reads/writes.

#### Return modes

This operation ignores non-scf::For ops and drops them in the return.
If all the operations referred to by the `target` PDLOperation pipeline
properly, the transform succeeds. Otherwise the transform silently fails.
The return handle points to only the subset of successfully produced
pipelined loops, which can be empty.
"""
function loop_pipeline(target::Value; transformed::MLIRType, iteration_interval=nothing, read_latency=nothing, location=Location())
    results = MLIRType[transformed, ]
    operands = Value[target, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(iteration_interval) && push!(attributes, namedattribute("iteration_interval", iteration_interval))
    !isnothing(read_latency) && push!(attributes, namedattribute("read_latency", read_latency))
    
    create_operation(
        "transform.loop.pipeline", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`loop_unroll`

Unrolls each loop associated with the given handle to have up to the given
number of loop body copies per iteration. If the unroll factor is larger
than the loop trip count, the latter is used as the unroll factor instead.

#### Return modes

This operation ignores non-scf::For, non-affine::For ops and drops them in
the return.  If all the operations referred to by the `target` PDLOperation
unroll properly, the transform succeeds. Otherwise the transform silently
fails.

Does not return handles as the operation may result in the loop being
removed after a full unrolling.
"""
function loop_unroll(target::Value; factor, location=Location())
    results = MLIRType[]
    operands = Value[target, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("factor", factor), ]
    
    create_operation(
        "transform.loop.unroll", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

import ...IR: NamedAttribute, MLIRType, Value, Location, Block, Region, Attribute, create_operation, context, IndexType
import ..Dialects: namedattribute, operandsegmentsizes
import ...API


"""
`vector_lower_vectors`

Indicates that the vector operations nested under the isolated from above op
`target` should be lowered to finer-grained vector primitives.

At this time, the transform is all or nothing.

This is usally a late step that is run after bufferization as part of the
process of lowering to e.g. LLVM or NVVM.
"""
function vector_lower_vectors(target::Value; results::MLIRType, contraction_lowering=nothing, multireduction_lowering=nothing, split_transfers=nothing, transpose_lowering=nothing, transpose_avx2_lowering=nothing, unroll_vector_transfers=nothing, location=Location())
    results = MLIRType[results, ]
    operands = Value[target, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(contraction_lowering) && push!(attributes, namedattribute("contraction_lowering", contraction_lowering))
    !isnothing(multireduction_lowering) && push!(attributes, namedattribute("multireduction_lowering", multireduction_lowering))
    !isnothing(split_transfers) && push!(attributes, namedattribute("split_transfers", split_transfers))
    !isnothing(transpose_lowering) && push!(attributes, namedattribute("transpose_lowering", transpose_lowering))
    !isnothing(transpose_avx2_lowering) && push!(attributes, namedattribute("transpose_avx2_lowering", transpose_avx2_lowering))
    !isnothing(unroll_vector_transfers) && push!(attributes, namedattribute("unroll_vector_transfers", unroll_vector_transfers))
    
    create_operation(
        "transform.vector.lower_vectors", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

end # transform
