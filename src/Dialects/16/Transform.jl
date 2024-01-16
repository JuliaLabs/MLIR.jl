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
    (scope != nothing) && push!(operands, scope)
    
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
    (result != nothing) && push!(results, result)
    (deduplicate != nothing) && push!(attributes, namedattribute("deduplicate", deduplicate))
    
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
    (target != nothing) && push!(operands, target)
    (name != nothing) && push!(attributes, namedattribute("name", name))
    
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
    (root != nothing) && push!(operands, root)
    
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
    (root != nothing) && push!(operands, root)
    
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

end # transform
