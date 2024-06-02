module transform

import ...IR: IR, NamedAttribute, Value, Location, Block, Region, Attribute, context, IndexType
import ..Dialects: namedattribute, operandsegmentsizes


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
    results = IR.Type[]
    operands = Value[target, bounded_values..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("lower_bounds", lower_bounds), namedattribute("upper_bounds", upper_bounds), ]
    
    IR.create_operation(
        "transform.affine.simplify_bounded_affine_ops", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

import ...IR: IR, NamedAttribute, Value, Location, Block, Region, Attribute, context, IndexType
import ..Dialects: namedattribute, operandsegmentsizes


"""
`bufferization_buffer_loop_hoisting`

Hoist buffer allocations (\"memref.alloc\" and \"memref.alloca\") from loops
within the targeted op. This transform assumes that there are no buffer
deallocation ops in the IR.

This transform reads the `target` handle and modifies the payload.
"""
function bufferization_buffer_loop_hoisting(target::Value; location=Location())
    results = IR.Type[]
    operands = Value[target, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "transform.bufferization.buffer_loop_hoisting", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`bufferization_eliminate_empty_tensors`

Try to eliminate all `tensor.empty` ops within the targeted op by replacing
them with a destination tensor.

`tensor.empty` ops cannot be bufferizes. They can either be converted to
`bufferization.alloc_tensor` or replaced with another tensor (via this
transform). `tensor.empty` does not specify the contents of the returned
tensor so their results can be replaced with arbitrary tensor values as long
as the dimensions match.

This transform looks for `tensor.empty` ops where the SSA use-def chain of
the result ends in a supported \"anchor op\" (always following the aliasing
OpOperand/OpResult chain). Currently supported anchor ops are:
- `tensor.insert_slice`
- `bufferization.yield` (inside `bufferization.alloc_tensor`)

# Example

```
%0 = tensor.empty() : tensor<5xf32>
%1 = linalg.fill ... outs(%0)
%2 = tensor.insert_slice %1 into %t[1][5][1]
```

Is rewritten with:
```
%0 = tensor.extract_slice %t[1][5][1]
%1 = linalg.fill ... outs(%0)
%2 = tensor.insert_slice %1 into %t[1][5][1]
```

The above example can bufferize without an allocation (in the absence of
other conflicts) because there is no longer a `tensor.empty` op.

See `-eliminate-empty-tensors` for more details.

#### Return modes

This transform reads the target handle and modifies the payload. It does
not produce any handle.
"""
function bufferization_eliminate_empty_tensors(target::Value; location=Location())
    results = IR.Type[]
    operands = Value[target, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "transform.bufferization.eliminate_empty_tensors", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`bufferization_empty_tensor_to_alloc_tensor`

Replace a tensor.empty with a bufferization.tensor_alloc.

#### Return modes

This operation consumes the `target` handle and produces the `transformed`
handle. `target` is expected to be a `tensor.empty` operation. The transform
always succeeds.
"""
function bufferization_empty_tensor_to_alloc_tensor(target::Value; transformed::IR.Type, location=Location())
    results = IR.Type[transformed, ]
    operands = Value[target, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
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

The targeted ops must be modules or functions. This is because there is
always a single, bufferized replacement op for such targets.

Note: Only ops that implement `BufferizableOpInterface` are bufferized. All
other ops are ignored if `allow_unknown_ops`. If `allow_unknown_ops` is
unset, this transform fails when an unknown/non-bufferizable op is found.
Many ops implement `BufferizableOpInterface` via an external model. These
external models must be registered when applying this transform op;
otherwise, said ops would be considered non-bufferizable.

#### Return modes

This operation consumes the `target` handle and produces the `transformed`
handle.
"""
function bufferization_one_shot_bufferize(target::Value; transformed::IR.Type, function_boundary_type_conversion=nothing, allow_return_allocs=nothing, allow_unknown_ops=nothing, bufferize_function_boundaries=nothing, create_deallocs=nothing, test_analysis_only=nothing, print_conflicts=nothing, memcpy_op=nothing, location=Location())
    results = IR.Type[transformed, ]
    operands = Value[target, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(function_boundary_type_conversion) && push!(attributes, namedattribute("function_boundary_type_conversion", function_boundary_type_conversion))
    !isnothing(allow_return_allocs) && push!(attributes, namedattribute("allow_return_allocs", allow_return_allocs))
    !isnothing(allow_unknown_ops) && push!(attributes, namedattribute("allow_unknown_ops", allow_unknown_ops))
    !isnothing(bufferize_function_boundaries) && push!(attributes, namedattribute("bufferize_function_boundaries", bufferize_function_boundaries))
    !isnothing(create_deallocs) && push!(attributes, namedattribute("create_deallocs", create_deallocs))
    !isnothing(test_analysis_only) && push!(attributes, namedattribute("test_analysis_only", test_analysis_only))
    !isnothing(print_conflicts) && push!(attributes, namedattribute("print_conflicts", print_conflicts))
    !isnothing(memcpy_op) && push!(attributes, namedattribute("memcpy_op", memcpy_op))
    
    IR.create_operation(
        "transform.bufferization.one_shot_bufferize", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

import ...IR: IR, NamedAttribute, Value, Location, Block, Region, Attribute, context, IndexType
import ..Dialects: namedattribute, operandsegmentsizes


"""
`apply_patterns_gpu_eliminate_barriers`

Removes unnecessary GPU barriers from the function. If a barrier does not
enforce any conflicting pair of memory effects, including a pair that is
enforced by another barrier, it is unnecessary and can be removed.

The approach is based on \"High-Performance GPU-to-CPU Transpilation and
Optimization via High-Level Parallel Constructs\" by  Moses, Ivanov,
Domke, Endo, Doerfert, and Zinenko in PPoPP 2023. Specifically, it
analyzes the memory effects of the operations before and after the given
barrier and checks if the barrier enforces any of the memory
effect-induced dependencies that aren\'t already enforced by another
barrier.

For example, in the following code

```mlir
  store %A
  barrier  // enforces load-after-store
  load %A
  barrier  // load-after-store already enforced by the previous barrier
  load %A
```

the second barrier can be removed.
"""
function apply_patterns_gpu_eliminate_barriers(; location=Location())
    results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "transform.apply_patterns.gpu.eliminate_barriers", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`gpu_map_forall_to_blocks`

Target the gpu_launch op and rewrite the top level `scf.forall`
to distributed gpu.block_id attribute. If `generate_gpu_launch` attribute
is set, then first generates `gpu_launch` and moves the top level
`scf.forall` inside.

The operation searches top level `scf.forall` ops under
`gpu_launch` and maps each such op to GPU blocks. Mapping is
one-to-one and the induction variables of `scf.forall` are
rewritten to gpu.block_id according to the `thread_dim_mapping` attribute.

Dynamic, `scf.forall` trip counts are currently not supported.
Dynamic block dim sizes are currently not supported.

Only **bufferized** scf.forall are currently supported.
Only scf.forall distributed to **at most 3 dimensions** are
currently supported.

The operation alters the block size of the given gpu_launch using the 
grid_dims argument.

#### Return modes:

This operation ignores non-gpu_launch ops and drops them in the return.

If any scf.forall with tensors is found, the transform definitely
fails.

If all the scf.forall operations contained within the LaunchOp
referred to by the `target` PDLOperation lower to GPU properly, the
transform succeeds. Otherwise the transform definitely fails.

The returned handle points to the same LaunchOp operand, consuming it and
producing a new SSA value to satisfy chaining and linearity of the IR
properties.
"""
function gpu_map_forall_to_blocks(target::Value; result::IR.Type, grid_dims=nothing, generate_gpu_launch=nothing, location=Location())
    results = IR.Type[result, ]
    operands = Value[target, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(grid_dims) && push!(attributes, namedattribute("grid_dims", grid_dims))
    !isnothing(generate_gpu_launch) && push!(attributes, namedattribute("generate_gpu_launch", generate_gpu_launch))
    
    IR.create_operation(
        "transform.gpu.map_forall_to_blocks", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`gpu_map_nested_forall_to_threads`

Target the `gpu.launch op` and rewrite all `scf.forall` nested in it to 
distributed `gpu.thread_id` attribute.

The operation searches for `scf.forall` ops nested under `target` and maps
each such op to GPU threads. 

`scf.forall` induction variables are rewritten to `gpu.thread_id` according
to the `mapping` attribute.

Different types of mappings attributes are supported:
  - the block_dims is a list of integers that specifies the number of
    threads in each dimension. This is a mandatory attribute that is used
    to constrain the number of threads in each dimension. If an 
    `scf.forall` op is mapped to fewer threads, predication occurs.
  - the warp_dims is a list of integers that specifies the number of
    warps in each dimension. This is an optional attribute that is used
    to constrain the number of warps in each dimension. When present, this
    attribute must be specified in a way that is compatible with the 
    block_dims attribute. If an `scf.forall` op is mapped to fewer warps,
    predication occurs.

Dynamic `scf.forall` trip counts are currently not supported.
Dynamic block dim sizes are currently not supported.

Only **bufferized** `scf.forall` are currently supported.
Only `scf.forall` distributed to **at most 3 dimensions** are
currently supported.

The `sync_after_distribute`attribute controls whether a `gpu.barrier` is
inserted after each scf.forall op. At this time, this is an all or nothing
choice. This will need to be tightened in the future.

The operation alters the block size of the given gpu_launch using the 
mandatory block_dims argument.

#### Return modes:

This operation ignores non-gpu_launch ops and drops them in the return.

If any scf.forall with tensors is found, the transform definitely
fails.

If all the scf.forall operations with gpu.thread mapping contained
within the LaunchOp referred to by the `target` PDLOperation lower to GPU
properly, the transform succeeds. Otherwise the transform definitely
fails.

scf.forall operations with mappings other than gpu.thread are
ignored.

The returned handle points to the same LaunchOp operand, consuming it and
producing a new SSA value to satisfy chaining and linearity of the IR
properties.

#### Example:

```
gpu.launch blocks(%bx, %by, %bz) in (%x = %0, %y = %1, %z = %2)
           threads(%tx, %ty, %tz) in (%tx = %3, %ty = %4, %tz = %5) {
  scf.forall (%i, %j) in (7, 9) {
    ... // body 1
  } {mapping = [#gpu.thread<x>, #gpu.thread<y>, #gpu.thread<z>]}
  scf.forall (%i) in (12) {
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
function gpu_map_nested_forall_to_threads(target::Value; result::IR.Type, block_dims=nothing, warp_dims=nothing, sync_after_distribute=nothing, location=Location())
    results = IR.Type[result, ]
    operands = Value[target, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(block_dims) && push!(attributes, namedattribute("block_dims", block_dims))
    !isnothing(warp_dims) && push!(attributes, namedattribute("warp_dims", warp_dims))
    !isnothing(sync_after_distribute) && push!(attributes, namedattribute("sync_after_distribute", sync_after_distribute))
    
    IR.create_operation(
        "transform.gpu.map_nested_forall_to_threads", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

import ...IR: IR, NamedAttribute, Value, Location, Block, Region, Attribute, context, IndexType
import ..Dialects: namedattribute, operandsegmentsizes


"""
`match_structured_body`

Checks if the body of the structured payload op satisfies one of the
following mutually exclusive criteria specified by attributes:

  * `reduction_position`: the body of the structured payload op implements
    a reduction of the `n`-th operand (`n` is the value of the attribute)
    using a single combiner operation;

  * `passthrough`: the body of the structured payload op only forwards
    inputs to the outputs (copy or broadcast).

  * `contraction`: the body of the structured payload op is a contraction
    of the form `<red>(<elem>(bbarg0, bbarg1), bbarg2)` where `<elem>` and
    `<red>` are binary operations whose names are specified in the attribute
    and operands can be permuted and optionally forwarded through a chain of
    unary side effect-free operations.

  
This op can only appear immediately inside a `transform.match.structured`
op and apply to its first block argument because it assumes the payload
to have been already checked for being a single structured op.
  

#### Return modes

Succeeds if the operation body satisfies the specified criteria, produces a
silenceable failure otherwise. Produces a definite failure if the operand is
not associated with a single payload op.
"""
function match_structured_body(operand_handle::Value; reduction_position=nothing, passthrough=nothing, contraction=nothing, location=Location())
    results = IR.Type[]
    operands = Value[operand_handle, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(reduction_position) && push!(attributes, namedattribute("reduction_position", reduction_position))
    !isnothing(passthrough) && push!(attributes, namedattribute("passthrough", passthrough))
    !isnothing(contraction) && push!(attributes, namedattribute("contraction", contraction))
    
    IR.create_operation(
        "transform.match.structured.body", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`match_structured_classify_contraction_dims`

Checks if the structured payload op has contraction-like dimensions as
follows:

  C(batch, m, n) += A(batch, m, k) * B(batch, k, n)

That is:

  - \'batch\' are parallel dimensions used in inputs and result;
  - \'m\' are parallel dimensions used in the LHS and result;
  - \'n\' are parallel dimensions used in rhe RHS and result;
  - \'k\' are reduction dimensions present only in LHS and RHS.

Note that this doesn\'t check the operation in the body.

  
This op can only appear immediately inside a `transform.match.structured`
op and apply to its first block argument because it assumes the payload
to have been already checked for being a single structured op.
  

#### Return modes

Succeeds if the operation has the contraction-like dimensions, produces a
silenceable failure otherwise.
"""
function match_structured_classify_contraction_dims(operand_handle::Value; batch::IR.Type, m::IR.Type, n::IR.Type, k::IR.Type, location=Location())
    results = IR.Type[batch, m, n, k, ]
    operands = Value[operand_handle, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "transform.match.structured.classify_contraction_dims", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`match_structured_dim`

Checks if the dimensions (loop ranges) of the structured payload op satisfy
the criteria specified as attributes. May capture the numeric value of the
dimension into a parameter that it returns.


 The following dimension specifications are supported:

  * `all`: all dimensions are checked and captured;
  * list of integers: the listed dimensions are checked and captured;
  * `except(` list of integers `)`: all dimensions except the
    specified ones are checked and captured.

Negative indexes are interpreted by counting values from the last one
(similarly to Python). For example, `-1` means the last dimension and
`except(-1)` means all dimensions but the last. Indexes must be unique,
including after interpretation of negative ones.

Produces a silenceable failure in case of index overflow, including backward
counting.
  

The following mutually exclusive conditions are available as unit
attributes:

  * `parallel`: the dimension corresponds to a parallel loop;
  * `reduction`: the dimension corresponds to a reduction loop.

If the result type is specified, associates the parameter with the (static)
values of dimensions in the same order as listed and preserving the natural
order for `all` and `except`. Specifically, if `-1, -2` are specified, the
parameter will be associated with the value of the second-to-last dimension
followed by the last dimension. If the dimension is dynamic, the parameter
will contain a negative value corresponding to kDynamic in C++.


This op can only appear immediately inside a `transform.match.structured`
op and apply to its first block argument because it assumes the payload
to have been already checked for being a single structured op.
  

#### Return modes

Succeeds if the specified dimensions satisfy the specified criteria,
produces a silenceable failure otherwise. Produces a definite failure if
the operand is not associated with a single payload op.
"""
function match_structured_dim(operand_handle::Value; result=nothing::Union{Nothing, IR.Type}, raw_dim_list, is_inverted=nothing, is_all=nothing, parallel=nothing, reduction=nothing, location=Location())
    results = IR.Type[]
    operands = Value[operand_handle, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("raw_dim_list", raw_dim_list), ]
    !isnothing(result) && push!(results, result)
    !isnothing(is_inverted) && push!(attributes, namedattribute("is_inverted", is_inverted))
    !isnothing(is_all) && push!(attributes, namedattribute("is_all", is_all))
    !isnothing(parallel) && push!(attributes, namedattribute("parallel", parallel))
    !isnothing(reduction) && push!(attributes, namedattribute("reduction", reduction))
    
    IR.create_operation(
        "transform.match.structured.dim", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`match_structured_elemental_bitwidth`

Produces a transform dialect parameter associated with the bitwidth of the
elemental type of the payload value passed as the operand.
This op can only appear immediately inside a `transform.match.structured`
op and apply to its first block argument because it assumes the payload
to have been already checked for being a single structured op.
  

#### Return modes

Succeeds if the operand is associated with exactly one payload value of
`ShapedType`. Produces a silenceable failure otherwise.
"""
function match_structured_elemental_bitwidth(operand_handle::Value; result::IR.Type, location=Location())
    results = IR.Type[result, ]
    operands = Value[operand_handle, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "transform.match.structured.elemental_bitwidth", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`match_structured_init`

Produces a transform dialect value depending on the result type:
  - If the result type is a value handle, it will be associated with the init
    operand(s) of the payload operation associated with the operand handle.
  - If the result type is an operation handle, it will be associated with the
    operation defining the init operand(s) of the payload operation associated
    with the operand handle.
  - If the result type is an affine map parameter type, it will be associated
    with the indexing map that corresponds to the init operand(s) of the
    payload operation associated with the operand handle.

For example, given the following operation:

```mlir
%arg3 = linalg.fill
linalg.matmul ins(%arg1, %arg2 : ...) outs(%arg3 : ...)
```

in case of a successful match for init operand 0 this operation will return,
for each of the respective cases above:

  - A handle to `%arg3` if the result is a value handle.
  - A handle to `linalg.fill` if the result is an operation handle.
  - A parameter containing the result map of the matrix multiplication, i.e.
    `affine_map<(d0, d1, d2) -> (d0, d1)>` if the result is an affine
    map parameter.

The match succeeds if the conditions specified as attributes succeed.


 The following init specifications are supported:

  * `all`: all inits are checked and captured;
  * list of integers: the listed inits are checked and captured;
  * `except(` list of integers `)`: all inits except the
    specified ones are checked and captured.

Negative indexes are interpreted by counting values from the last one
(similarly to Python). For example, `-1` means the last init and
`except(-1)` means all inits but the last. Indexes must be unique,
including after interpretation of negative ones.

Produces a silenceable failure in case of index overflow, including backward
counting.
  


This op can only appear immediately inside a `transform.match.structured`
op and apply to its first block argument because it assumes the payload
to have been already checked for being a single structured op.
  

#### Return modes

Succeeds if all init(outs) indexes are in bounds, produces a silenceable
failure otherwise. Additionally, when the result is an operation handle,
produces a silenceable failure if the init(outs) specification defines
more than one init(outs) or if the operand is not an operation result.
"""
function match_structured_init(operand_handle::Value; result=nothing::Union{Nothing, IR.Type}, raw_position_list, is_inverted=nothing, is_all=nothing, permutation=nothing, projected_permutation=nothing, location=Location())
    results = IR.Type[]
    operands = Value[operand_handle, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("raw_position_list", raw_position_list), ]
    !isnothing(result) && push!(results, result)
    !isnothing(is_inverted) && push!(attributes, namedattribute("is_inverted", is_inverted))
    !isnothing(is_all) && push!(attributes, namedattribute("is_all", is_all))
    !isnothing(permutation) && push!(attributes, namedattribute("permutation", permutation))
    !isnothing(projected_permutation) && push!(attributes, namedattribute("projected_permutation", projected_permutation))
    
    IR.create_operation(
        "transform.match.structured.init", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`match_structured_input`

Produces a transform dialect value depending on the result type:

  - If the result type is a value handle, it will be associated with the input
    operand(s) of the payload operation associated with the operand handle.
  - If the result type is an operation handle, it will be associated with the
    operation defining the input operand(s) of the payload operation associated
    with the operand handle.
  - If the result type is an affine map parameter type, it will be associated
    with the indexing map that corresponds to the input operand(s) of the
    payload operation associated with the operand handle.

For example, given the following operation:

```mlir
%arg1 = some.op
linalg.matmul ins(%arg1, %arg2 : ...) outs(%arg3 : ...)
```

in case of a successful match for operand 0 this operation will return, for
each of the respective cases above:

  - A handle to `%arg1` if the result is a value handle.
  - A handle to `some.op` if the result is an operation handle.
  - A parameter containing the LHS map of the matrix multiplication, i.e.
    `affine_map<(d0, d1, d2) -> (d0, d2)>` if the result is an affine
    map parameter.

The match succeeds if the conditions specified as attributes succeed.


 The following input specifications are supported:

  * `all`: all inputs are checked and captured;
  * list of integers: the listed inputs are checked and captured;
  * `except(` list of integers `)`: all inputs except the
    specified ones are checked and captured.

Negative indexes are interpreted by counting values from the last one
(similarly to Python). For example, `-1` means the last input and
`except(-1)` means all inputs but the last. Indexes must be unique,
including after interpretation of negative ones.

Produces a silenceable failure in case of index overflow, including backward
counting.
  


This op can only appear immediately inside a `transform.match.structured`
op and apply to its first block argument because it assumes the payload
to have been already checked for being a single structured op.
  

#### Return modes

Succeeds if all input indexes are in bounds, produces a silenceable failure
otherwise. Additionally, when the result is an operation handle, produces a
silenceable failure if the input specification defines more than one input
or if the operand is not an operation result.
"""
function match_structured_input(operand_handle::Value; result=nothing::Union{Nothing, IR.Type}, raw_position_list, is_inverted=nothing, is_all=nothing, permutation=nothing, projected_permutation=nothing, location=Location())
    results = IR.Type[]
    operands = Value[operand_handle, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("raw_position_list", raw_position_list), ]
    !isnothing(result) && push!(results, result)
    !isnothing(is_inverted) && push!(attributes, namedattribute("is_inverted", is_inverted))
    !isnothing(is_all) && push!(attributes, namedattribute("is_all", is_all))
    !isnothing(permutation) && push!(attributes, namedattribute("permutation", permutation))
    !isnothing(projected_permutation) && push!(attributes, namedattribute("projected_permutation", projected_permutation))
    
    IR.create_operation(
        "transform.match.structured.input", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`match_structured_num_inits`

Produces a transform dialect parameter value associated with an integer
attribute containing the number of init(outs) operands of the payload
operation associated with the operand handle.


This op can only appear immediately inside a `transform.match.structured`
op and apply to its first block argument because it assumes the payload
to have been already checked for being a single structured op.
  

#### Return modes

Succeeds if the operand is associated with exactly one structured payload
operation. Produces a silenceable failure otherwise.
"""
function match_structured_num_inits(operand_handle::Value; result::IR.Type, location=Location())
    results = IR.Type[result, ]
    operands = Value[operand_handle, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "transform.match.structured.num_inits", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`match_structured_num_inputs`

Produces a transform dialect parameter value associated with an integer
attribute containing the number of input operands of the payload operation
associated with the operand handle.


This op can only appear immediately inside a `transform.match.structured`
op and apply to its first block argument because it assumes the payload
to have been already checked for being a single structured op.
  

#### Return modes

Succeeds if the operand is associated with exactly one structured payload
operation. Produces a silenceable failure otherwise.
"""
function match_structured_num_inputs(operand_handle::Value; result::IR.Type, location=Location())
    results = IR.Type[result, ]
    operands = Value[operand_handle, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "transform.match.structured.num_inputs", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`match_structured`

Checks if the payload operation associated with the operand handle is a
structured operation, that is, an operation that implements
`LinalgOpInterface`, and that all conditions listed in the body of this
operation are satisfied. Produces a silenceable failure if the payload
operation is not structured.

The transform operations nested in the body region are applied one by one.
If any of them produces a failure, silenceable or definite, the following
operations are not applied. If the failure propagation mode is \"propagate\",
silenceable failures are forwarded as the result of this operation. If it is
\"suppress\", they are ignored and this operation immediately succeeds.
Definite failures are always propagated immediately.

In case of success, the transform values produced by this operation are
associated with the same payload as the operands of the block terminator. If
any of the nested operations produced a silenceable failure, regardless of
the failure propagation mode, the transform values produced by this
operation that correspond to the already defined terminator operands are
associated with the same payload as the already defined terminator operands.
Other values produced by this operation are associated with empty payloads.

If the failure propagation mode is not specified, it is considered
\"propagate\" by default. The \"suppress\" mode can be used to specify optional
matches.

#### Return modes

This operation only reads all operand handles and produces all resulting
handles. It succeeds in \"propagate\" mode if the payload operation is a
structured operation and if all the nested operations succeed. It succeeds
in \"suppress\" mode as long as the operand handle is associated with exactly
one payload operation. It produces a definite failure when the handle is
not associated with exactly one payload operation.
"""
function match_structured(current::Value; outputs::Vector{IR.Type}, failure_propagation_mode=nothing, body_region::Region, location=Location())
    results = IR.Type[outputs..., ]
    operands = Value[current, ]
    owned_regions = Region[body_region, ]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(failure_propagation_mode) && push!(attributes, namedattribute("failure_propagation_mode", failure_propagation_mode))
    
    IR.create_operation(
        "transform.match.structured", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`match_structured_rank`

Produces a transform dialect parameter value associated with an integer
attribute containing the rank of the structured payload operation associated
with the operand handle.


This op can only appear immediately inside a `transform.match.structured`
op and apply to its first block argument because it assumes the payload
to have been already checked for being a single structured op.
  

#### Return modes

Succeeds if the operand is associated with exactly one structured payload
operation. Produces a silenceable failure otherwise.
"""
function match_structured_rank(operand_handle::Value; rank::IR.Type, location=Location())
    results = IR.Type[rank, ]
    operands = Value[operand_handle, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "transform.match.structured.rank", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`match_structured_result`

Produces a transform dialect value handle associated with the payload value
defined as a result of the payload operation associated with the operand
handle, or an operation handle to an operation using the produced result
with additional constraints specified by the attributes as follows.

  * If `any` is specified, binds the resulting handle to any operation using
    the result and succeeds.
  * If `single` is specified, binds the resulting handle to the only
    operation using the result or fails if there is more than one (or no)
    such operation.

The number of the result is specified as `position` attribute. It may take
positive and negative values. Negative values are interpreted as counting
results from backwards, e.g., `-1` means the last result and `-2` means the
second-to-last result. In any case, the position must be in bounds for the
given payload operation. A silenceable failure is produced for out-of-bounds
positions.

  
This op can only appear immediately inside a `transform.match.structured`
op and apply to its first block argument because it assumes the payload
to have been already checked for being a single structured op.
  

#### Return modes

Succeeds if the position is in bounds and if the user operation could be
found when requested. Produces a silenceable failure otherwise.
"""
function match_structured_result(operand_handle::Value; result::IR.Type, position, any=nothing, single=nothing, location=Location())
    results = IR.Type[result, ]
    operands = Value[operand_handle, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("position", position), ]
    !isnothing(any) && push!(attributes, namedattribute("any", any))
    !isnothing(single) && push!(attributes, namedattribute("single", single))
    
    IR.create_operation(
        "transform.match.structured.result", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`match_structured_yield`

Forwards the payload association from the operands to the results of the
parent op. Always succeeds.
"""
function match_structured_yield(handles::Vector{Value}; location=Location())
    results = IR.Type[]
    operands = Value[handles..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "transform.match.structured.yield", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

import ...IR: IR, NamedAttribute, Value, Location, Block, Region, Attribute, context, IndexType
import ..Dialects: namedattribute, operandsegmentsizes


"""
`apply_patterns_linalg_erase_unnecessary_inputs`

Collects patterns that promote inputs to outputs and remove unused inputs of
`linalg.generic` ops.
"""
function apply_patterns_linalg_erase_unnecessary_inputs(; location=Location())
    results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "transform.apply_patterns.linalg.erase_unnecessary_inputs", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`apply_patterns_linalg_fold_unit_extent_dims_via_reshapes`

Collects patterns to fold unit-extent dimensions in operands/results of
linalg ops on tensors via reassociative reshape ops.
"""
function apply_patterns_linalg_fold_unit_extent_dims_via_reshapes(; location=Location())
    results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "transform.apply_patterns.linalg.fold_unit_extent_dims_via_reshapes", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`apply_patterns_linalg_fold_unit_extent_dims_via_slices`

Collects patterns to fold unit-extent dimensions in operands/results of
linalg ops on tensors via rank-reducing slices.
"""
function apply_patterns_linalg_fold_unit_extent_dims_via_slices(; location=Location())
    results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "transform.apply_patterns.linalg.fold_unit_extent_dims_via_slices", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`apply_patterns_linalg_tiling_canonicalization`

Collects canonicalization patterns relevant to apply after tiling patterns.
"""
function apply_patterns_linalg_tiling_canonicalization(; location=Location())
    results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "transform.apply_patterns.linalg.tiling_canonicalization", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`structured_bufferize_to_allocation`

This transform bufferizes the targeted operation and materializes the
result in a new allocation. It replaces all original uses of the target
result with the newly allocated buffer, wrapped in a
`bufferization.to_tensor` op. It returns a handle to the newly allocated
buffer. Furthermore, it returns a handle that is mapped to all newly created
ops.

Only bufferizable ops are that bufferize to a memory write or have an
aliasing OpOperand (and do not themselves bufferize to an allocation) are
supported. They are bufferized using their BufferizableOpInterface
implementation. E.g.:

```
%0 = tensor.insert %f into %dest[%pos] : tensor<10xf32>
```

Is bufferized to:

```
%alloc = memref.alloc() : memref<10xf32>
memref.tensor_store %dest, %alloc : memref<10xf32>
memref.store %f, %alloc[%pos] : memref<10xf32>
%0 = bufferization.to_tensor %alloc restrict writable : memref<10xf32>
```

Selected ops that bufferize to an allocation (or need special handling) are
also supported:
- `tensor.pad` is lowered to an allocation, followed by a `linalg.fill` and
  and a buffer copy (all on memrefs).
- `vector.mask` is bufferized together with its region. The allocation is
  placed in front of the `vector.mask` op.

An optional memory space attribute can be specified for the materialized
buffer allocation.

If a memory copy is needed, a \"memref.tensor_store\" is used when possible.
This is an op with tensor semantics that will bufferize to a memory copy
later. Which concrete op will be used for the memory copy is up to the
bufferization framework. Alternatively, a custom memcpy op can be specified
via `memcpy_op`. Currently supported are \"memref.copy\" and \"linalg.copy\".
In that case, the source of each memcpy must not have a custom memory space.
Furthermore, because the future buffer layout unknown for a given tensor,
a fully dynamic layout is assumed for best compatibility. Users should use
\"memref.tensor_store\" when possible.

\"memref.alloc\" is used for new buffer allocations. The buffer is deallocated
at the end of the block. Custom allocation ops can be specified via
`alloc_op`. Currently supported are \"memref.alloc\" and \"memref.alloca\". In
case of a \"memref.alloca\", the buffer is not deallocated.

If `bufferize_destination_only` is set, only the destination operands of the
op are bufferized to a new memory allocation, but not the op itself.

#### Return modes

This operation consumes the `target` handle and produces the
`allocated_buffer` and `new_ops` handles. It always succeeds.
"""
function structured_bufferize_to_allocation(target::Value; allocated_buffer::IR.Type, new_ops::IR.Type, memory_space=nothing, memcpy_op=nothing, alloc_op=nothing, bufferize_destination_only=nothing, location=Location())
    results = IR.Type[allocated_buffer, new_ops, ]
    operands = Value[target, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(memory_space) && push!(attributes, namedattribute("memory_space", memory_space))
    !isnothing(memcpy_op) && push!(attributes, namedattribute("memcpy_op", memcpy_op))
    !isnothing(alloc_op) && push!(attributes, namedattribute("alloc_op", alloc_op))
    !isnothing(bufferize_destination_only) && push!(attributes, namedattribute("bufferize_destination_only", bufferize_destination_only))
    
    IR.create_operation(
        "transform.structured.bufferize_to_allocation", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`structured_convert_conv2d_to_img2col`

Convert linalg.conv_2d_xxx into linalg.generic (for img2col packing)
and linalg.matmul.

A convolution operation can be written as a matrix-matrix multiplication by
unfolding the cross-correlation between input and filter and explicitly copy
overlapped sliding window inputs.

Consider 2D input X with single channel input and output and 2x2 filter W:
```
[x(0, 0)  , x(0, 1)  , ...,   x(0, n)  ]
[x(1, 0)  , x(1, 1)  , ...,   x(1, n)  ]
[.        ,  .       ,.   ,      .     ]            [w(0, 0), w(0, 1)]
[.        ,  .       , .  ,      .     ]    (conv)  [w(1, 0), w(1, 1)]
[.        ,  .       ,   .,      .     ]
[x(n-1, 0), x(n-1, 1), ..., x(n-1, n-1)]
```

The packed input data (img2col) is a matrix with |rows| = output spatial
size, |columns| = filter spatial size. To compute the output Y(i, j) we need
to calculate the dot product between filter window at input X(x, y)) and the
filter which will look like the following where r.h.s is the img2col matrix
and l.h.s is the flattned filter:
```
[x(0,0), x(0,1), x(1,0), x(1,1)]
[x(0,1), x(1,1), x(0,2), x(1,2)] (matmul) [w(0,0), w(0,1), w(1,0), w(1,1)]
[x(0,1), x(1,1), x(0,2), x(1,2)]
[   .  ,    .  ,    .  ,    .  ]
```

In general for 2D case with (N, H, W, C) input and (Kh, Kw, C, D) filter
and output (N, Ho, Wo, D) the convolution is the following matrix-matrix
multiplication (Ho x Wo, Kh x Kw x C) * (Kh x Kw x C, D) for each input in
the N input. For the case where N > 1 its a batched matrxi-matrix
multplication.

Returns two handles:
- One on the operation that produces the img2col tensor.
- One on the final operation of the sequence that replaces the original
  convolution.

#### Return modes:

Returns a definite failure if target is not isolated from above.
Returns a silenceable failure if the pattern application failed.
"""
function structured_convert_conv2d_to_img2col(target::Value; img2col_tensor::IR.Type, transformed::IR.Type, location=Location())
    results = IR.Type[img2col_tensor, transformed, ]
    operands = Value[target, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "transform.structured.convert_conv2d_to_img2col", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`structured_decompose_interface`

TODO
"""
function structured_decompose_interface(target::Value; transformed::IR.Type, location=Location())
    results = IR.Type[transformed, ]
    operands = Value[target, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "transform.structured.decompose_interface", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`structured_decompose`

Decomposes named complex operations, such as higher-dimensional
(depthwise) convolutions, into combinations of lower-dimensional equivalents
when possible.

#### Return modes

This operation ignores non-Linalg ops and drops them in the return.
If all the operations referred to by the `target` handle decompose
properly, the transform succeeds. Otherwise the transform silently fails.
The return handle points to only the subset of successfully produced
computational operations, which can be empty.
"""
function structured_decompose(target::Value; transformed::IR.Type, location=Location())
    results = IR.Type[transformed, ]
    operands = Value[target, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "transform.structured.decompose", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`structured_eliminate_empty_tensors`

Try to eliminate all `tensor.empty` op uses that are anchored on a LinalgOp
within the targeted op.

This op is similar to `bufferization.eliminate_empty_tensors`, but specific
to LinalgOps.

`tensor.empty` ops cannot be bufferized. They can either be converted to
`bufferization.alloc_tensor` or replaced with another tensor (via this
transform). `tensor.empty` does not specify the contents of the returned
tensor so their results can be replaced with arbitrary tensor values as long
as the dimensions match.

This transform looks for `tensor.empty` ops where the SSA use-def chain of
the result ends in a supported LinalgOp (always following the aliasing
OpOperand/OpResult chain). The following LinalgOps are supported:
- Only parallel iterator types.
- The use-def chain ends in an input operand of the LinalgOp.
- The LinalgOp has an unused output operand with the same shape and
  indexing map.

# Example

```
%0 = tensor.empty()
%1 = linalg.matmul ins(...) outs(%0)
%2 = linalg.generic ins(%1) outs(%dest) {
  ^bb0(%in: f32, %out: f32):
  // out not used
}
```

Is rewritten with:
```
%0 = tensor.empty()
%1 = linalg.matmul ins(...) outs(%dest)
%2 = linalg.generic ins(%0) outs(%1) {
  ^bb0(%in: f32, %out: f32):
  // Use %out instead of %in
}
```

After this transformation, the \"ins\" operand has no uses inside the body of
the LinalgOp and can be folded away with existing cleanup patterns.
Afterwards, the tensor::EmptyOp can also fold away, so that the example can
bufferize without an allocation (in the absence of other conflicts).

#### Return modes

This transform reads the target handle and modifies the payload. It does
not produce any handle.
"""
function structured_eliminate_empty_tensors(target::Value; location=Location())
    results = IR.Type[]
    operands = Value[target, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "transform.structured.eliminate_empty_tensors", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`structured_fuse_into_containing_op`

Fuses the `producer_op` into the `containing_op`.
Returns a handle to the fused ops and the `new_containing_op`.

The producer is typically a slice of a tileable op (i.e., implements
TilingInterface). In that case, this transform computes the accessed
producer slice inside of the containing op (\"tile and fuse\") and if required,
creates a new containing op with outputs from the fused producer. Otherwise,
the entire producer is cloned inside the containing op (\"clone and fuse\").

The containing op handle must be associated with exactly one payload op. The
producer op handle may be associated with multiple payload ops. This
transform fuses producers one-by-one, always picking an unspecified producer
that has at least one use inside the containing op among the
producers. A producer can be listed multiple times in the handle.

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

This operation consumes the producer handle.
This operation only reads the containing op handle.
"""
function structured_fuse_into_containing_op(producer_op::Value, containing_op::Value; fused_op::IR.Type, new_containing_op::IR.Type, location=Location())
    results = IR.Type[fused_op, new_containing_op, ]
    operands = Value[producer_op, containing_op, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
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
function structured_fuse(target::Value; transformed::IR.Type, loops::Vector{IR.Type}, tile_sizes=nothing, tile_interchange=nothing, location=Location())
    results = IR.Type[transformed, loops..., ]
    operands = Value[target, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(tile_sizes) && push!(attributes, namedattribute("tile_sizes", tile_sizes))
    !isnothing(tile_interchange) && push!(attributes, namedattribute("tile_interchange", tile_interchange))
    
    IR.create_operation(
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
If all the operations referred to by the `target` handle generalize
properly, the transform succeeds. Otherwise the transform silently fails.
The return handle points to only the subset of successfully produced
equivalent generic operations, which can be empty or contain the original
ops if they were already in generic form.
"""
function structured_generalize(target::Value; transformed::IR.Type, location=Location())
    results = IR.Type[transformed, ]
    operands = Value[target, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "transform.structured.generalize", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`structured_hoist_pad_build_packing_loop_nest`

Helper transform used to hoist a tensor.pad target operation. This operation
creates the packing loop nest required by the hoist_pad operation and makes
that functionality available independently.

TODO: In the future, we should consider rewriting as a tensor.pack after
hoisting since this abstraction is now available.

#### Return modes

This operation ignores non-tensor.pad ops and drops them in the result.
If any non-tensor.pad is passed, the transform emits a silenceable failure.

The return handle points to only the subset of successfully created packing
loop nests, which can be empty.
"""
function structured_hoist_pad_build_packing_loop_nest(target::Value, loop::Value; packing_loop::IR.Type, transpose=nothing, location=Location())
    results = IR.Type[packing_loop, ]
    operands = Value[target, loop, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(transpose) && push!(attributes, namedattribute("transpose", transpose))
    
    IR.create_operation(
        "transform.structured.hoist_pad.build_packing_loop_nest", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`structured_hoist_pad`

Hoist the tensor.pad target operation by at most the given number of loops.
Optionally apply the transpose attribute to the inner dimensions.

TODO: In the future, we should consider rewriting as a tensor.pack after 
hoisting since this abstraction is now available.
TODO: Maybe also return the linalg.generic transpose created at some point.

#### Return modes

This operation ignores non-tensor.pad ops and drops them in the result.
If any non-tensor.pad is passed, the transform emits a silenceable failure.

If all the operations referred to by the `target` handle padproperly, the
transform succeeds. Otherwise the transform silently fails.

The return handle points to only the subset of successfully hoisted 
tensor.pad operations, which can be empty.
"""
function structured_hoist_pad(target::Value; transformed::IR.Type, num_loops, transpose=nothing, location=Location())
    results = IR.Type[transformed, ]
    operands = Value[target, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("num_loops", num_loops), ]
    !isnothing(transpose) && push!(attributes, namedattribute("transpose", transpose))
    
    IR.create_operation(
        "transform.structured.hoist_pad", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`structured_hoist_redundant_tensor_subsets`

Hoists supported tensor subset extract/insert operation pairs out of 
immediately enclosing loop iteratively, if the following conditions
are true:
   1. The 2 ops access the same tensor subset.
   2. All operands are invariant under the enclosing loop.

The supported subset extract/insert operation pairs currently comprise:
   - tensor.extract_slice / tensor.insert_slice
   - vector.transfer_read / vector.transfer_write on tensors

Only scf.for loops are currently supported.

When applied to:
   1. an scf.for loop, hoist out of this loop only.
   2. a non-loop op, apply hoisting to all the contained loop ops.

#### Return modes:

The operation always succeeds and returns nothing.
"""
function structured_hoist_redundant_tensor_subsets(target::Value; location=Location())
    results = IR.Type[]
    operands = Value[target, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "transform.structured.hoist_redundant_tensor_subsets", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`structured_hoist_redundant_vector_transfers`

Hoist vector.transfer_read / vector.transfer_write pairs out of immediately
enclosing scf::ForOp iteratively, if the following conditions are true:
   1. The 2 ops access the same memref with the same indices.
   2. All operands are invariant under the enclosing scf::ForOp.
   3. No uses of the memref either dominate the transfer_read or are
   dominated by the transfer_write (i.e. no aliasing between the write and
   the read across the loop)

WARNING: This hoisting does not model parallelism and is generally incorrect
when used on distributed loops with memref semantics!
TODO: obsolete and should be retired.

#### Return modes:

The operation always succeeds and returns a handle to the transformed
function op.
"""
function structured_hoist_redundant_vector_transfers(target::Value; transformed::IR.Type, location=Location())
    results = IR.Type[transformed, ]
    operands = Value[target, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "transform.structured.hoist_redundant_vector_transfers", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`structured_insert_slice_to_copy`

Targeted rewrite of an tensor.insert_slice to linalg.copy.
This is useful to materialize copies explicitly before bufferization and 
transform them, avoiding the need to rediscover them after bufferization.

If the insert_slice source is already a linalg.copy, only return the source
op (i.e. do not create an additional linalg.copy op).

#### Return modes:

The operation always succeeds and returns a handle to the relevant 
linalg.copy op.
"""
function structured_insert_slice_to_copy(target::Value; transformed::IR.Type, location=Location())
    results = IR.Type[transformed, ]
    operands = Value[target, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "transform.structured.insert_slice_to_copy", location;
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
If all the operations referred to by the `target` handle interchange
properly, the transform succeeds.
If any interchange fails, the transform definitely fails.
The return handle points to only the subset of successfully produced
interchanged operations, which can be empty.
"""
function structured_interchange(target::Value; transformed::IR.Type, iterator_interchange=nothing, location=Location())
    results = IR.Type[transformed, ]
    operands = Value[target, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(iterator_interchange) && push!(attributes, namedattribute("iterator_interchange", iterator_interchange))
    
    IR.create_operation(
        "transform.structured.interchange", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`structured_lower_pack`

Rewrite a tensor.pack into tensor.pad + tensor.expand_shape + linalg.transpose.

#### Return modes

This operation ignores non-pack ops and drops them in the return.
This operation produces a silenceableFailure if the rewrite fails for any
reason.
If all the operations referred to by the `target` are rewritten, the
transform succeeds.
Return handles to the newly produced pad, expand_shape and transpose ops.
"""
function structured_lower_pack(target::Value; pad_op::IR.Type, expand_shape_op::IR.Type, transpose_op::IR.Type, location=Location())
    results = IR.Type[pad_op, expand_shape_op, transpose_op, ]
    operands = Value[target, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "transform.structured.lower_pack", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`structured_lower_unpack`

Lower a tensor.unpack into empty + linalg.transpose + tensor.collapse_shape + 
tensor.extract_slice.

#### Return modes

This operation ignores non-unpack ops and drops them in the return.
This operation produces a silenceableFailure if the rewrite fails for any
reason.
If all the operations referred to by the `target` are rewritten, the
transform succeeds.
Return handles to the newly produced empty, transpose, collapse_shape and extract_slice ops.
"""
function structured_lower_unpack(target::Value; empty_op::IR.Type, transpose_op::IR.Type, collapse_shape_op::IR.Type, extract_slice_op::IR.Type, location=Location())
    results = IR.Type[empty_op, transpose_op, collapse_shape_op, extract_slice_op, ]
    operands = Value[target, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "transform.structured.lower_unpack", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`structured_gpu_map_copy_to_threads`

Targeted mapping of a linalg.copy / tensor.pad operation on tensors to a GPU
thread mapping.

This operation implements a greedy heuristic that determines a good
distribution of threads to break down the copy/pad operation into.
The heuristic is driven by considerations related to the underlying
architecture for which good high-level decisions are needed assuming certain
hardware features. Relevant features are exposed via first-class attributes
to control the behavior of the transformation at a high level.

For now, a single heuristic is implemented and can be extended on a per-need
basis.

#### Return modes

This operation fails definitely if there is an unsupported op (i.e., not
linalg.copy / tensor.pad) among the targeted op. Otherwise, the operation
always succeeds and returns a handle to the relevant tiled linalg.copy /
tensor.pad op and the enclosing scf.forall op.
"""
function structured_gpu_map_copy_to_threads(target::Value; forall_op::IR.Type, tiled_op::IR.Type, total_num_threads, desired_bit_alignment, location=Location())
    results = IR.Type[forall_op, tiled_op, ]
    operands = Value[target, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("total_num_threads", total_num_threads), namedattribute("desired_bit_alignment", desired_bit_alignment), ]
    
    IR.create_operation(
        "transform.structured.gpu.map_copy_to_threads", location;
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

Note: The input vector sizes must be bigger than or equal to their
counterpart iteration space sizes.

Typically this operator should be applied to linalg operations that have
already be tiled to the appropriate sizes.

#### Return modes:

This operation produces a definite failure if the dynamic vector sizes (SSA
values) do not satisfy the constraints mentioned above. It produces a
silenceable failure if at least one target op is not a Linalg op or fails to
vectorize.
"""
function structured_masked_vectorize(target::Value, vector_sizes::Vector{Value}; vectorize_nd_extract=nothing, scalable_sizes=nothing, static_vector_sizes=nothing, location=Location())
    results = IR.Type[]
    operands = Value[target, vector_sizes..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(vectorize_nd_extract) && push!(attributes, namedattribute("vectorize_nd_extract", vectorize_nd_extract))
    !isnothing(scalable_sizes) && push!(attributes, namedattribute("scalable_sizes", scalable_sizes))
    !isnothing(static_vector_sizes) && push!(attributes, namedattribute("static_vector_sizes", static_vector_sizes))
    
    IR.create_operation(
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
function structured_match(target::Value; results_::IR.Type, ops=nothing, interface=nothing, op_attrs=nothing, filter_result_type=nothing, location=Location())
    results = IR.Type[results_, ]
    operands = Value[target, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(ops) && push!(attributes, namedattribute("ops", ops))
    !isnothing(interface) && push!(attributes, namedattribute("interface", interface))
    !isnothing(op_attrs) && push!(attributes, namedattribute("op_attrs", op_attrs))
    !isnothing(filter_result_type) && push!(attributes, namedattribute("filter_result_type", filter_result_type))
    
    IR.create_operation(
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
function structured_multitile_sizes(target::Value; low_size::IR.Type, high_size::IR.Type, split_point::IR.Type, dimension, target_size, divisor=nothing, location=Location())
    results = IR.Type[low_size, high_size, split_point, ]
    operands = Value[target, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("dimension", dimension), namedattribute("target_size", target_size), ]
    !isnothing(divisor) && push!(attributes, namedattribute("divisor", divisor))
    
    IR.create_operation(
        "transform.structured.multitile_sizes", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`structured_pack_greedily`

Target a Linalg op and rewrite it into packed LinalgOp form by trying to
infer whether a known suboperation is embedded

Different packing strategies are applied in order, when one applies 
successfully, the transform returns:
  1. Matmul packing: Try to infer a matmul operation embedded in the target op.
     Specifically, this looks for 2 parallel dimensions that participate in
     an outer-product and 1 reduction dimension.
     These dimensions are referred as (m, n, k) to match canonical matmul
     terminology.
     
     The packed sizes for (m, n, k) are specified by `matmul_packed_sizes`
     and the optional `matmul_padded_sizes_next_multiple_of`.
     When an entry `matmul_packed_sizes[i]` is non-0, the corresponding 
     dimension is packed by `matmul_packed_sizes[i]`.
     Otherwise, the dimension is merely padded to the next multiple of
     `matmul_padded_sizes_next_multiple_of[i]`.

     `matmul_padded_sizes_next_multiple_of` is optional and is expected to
     either be empty or of size `3`, matching the size of `matmul_packed_sizes`.
     For each individual element of `matmul_packed_sizes` and 
     `matmul_padded_sizes_next_multiple_of`, only one of them is allowed to
     be non-zero.
     
     The ordering of the packed dimensions (mm, nn, kk) is specified by the
     `matmul_inner_dims_order` attribute.

Packing occurs as follows:
  1. Find the dimensions to pack according to the strategy.
  2. The target is converted to linalg.generic form.
  3. An interchange transform is applied to isolate the dimensions to pack as
     the most minor indexing dimensions of the linalg.generic. The most minor
     dimensions are themselves ordered according to `inner_dims_order`.
  4. An elementwise traversal of `matmul_packed_sizes` and
     `matmul_padded_sizes_next_multiple_of` is performed and for each 
     dimension `d`, either pack to `matmul_packed_sizes[d]` or pad to the
     `matmul_padded_sizes_next_multiple_of[d]`.
  5. Packing/padding is performed by the amounts determined in step 4. and
     following `inner_dims_order`.

By normalizing the most minor dimensions to `inner_dims_order`, the transform
guarantees that packing immediately generates inner dimensions in a desirable
layout.

Outer dimension layout permutations are not controlled by this transform op
at the moment and can be obtained by composing with the pack_transpose
transformation.

#### Return modes

This operation ignores non-Linalg ops and drops them in the return.
It returns the list of packed Linalg ops or the original op when all available
packing strategies failed to apply.
"""
function structured_pack_greedily(target::Value, matmul_packed_sizes::Vector{Value}; packed_op::IR.Type, static_matmul_packed_sizes=nothing, matmul_padded_sizes_next_multiple_of=nothing, matmul_inner_dims_order=nothing, location=Location())
    results = IR.Type[packed_op, ]
    operands = Value[target, matmul_packed_sizes..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(static_matmul_packed_sizes) && push!(attributes, namedattribute("static_matmul_packed_sizes", static_matmul_packed_sizes))
    !isnothing(matmul_padded_sizes_next_multiple_of) && push!(attributes, namedattribute("matmul_padded_sizes_next_multiple_of", matmul_padded_sizes_next_multiple_of))
    !isnothing(matmul_inner_dims_order) && push!(attributes, namedattribute("matmul_inner_dims_order", matmul_inner_dims_order))
    
    IR.create_operation(
        "transform.structured.pack_greedily", location;
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
transformation by composing the resulting operation with a
`transform.structured.pack_transpose` op.
This composition allows separating concerns and composes better compared
to adding additional permutation attributes to this transform op.

#### Return modes

This operation applies to a single Linalg op, otherwise it fails.
This operation may produce a definiteFailure if the packing fails for any
reason.

The returned handle point to the packed LinalgOp.
"""
function structured_pack(target::Value, packed_sizes::Vector{Value}; packed_op::IR.Type, static_packed_sizes=nothing, location=Location())
    results = IR.Type[packed_op, ]
    operands = Value[target, packed_sizes..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(static_packed_sizes) && push!(attributes, namedattribute("static_packed_sizes", static_packed_sizes))
    
    IR.create_operation(
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
function structured_pack_transpose(target_pack_or_un_pack_op::Value, target_linalg_op::Value; packed_op::IR.Type, pack_op::IR.Type, un_pack_op::IR.Type, outer_perm=nothing, inner_perm=nothing, location=Location())
    results = IR.Type[packed_op, pack_op, un_pack_op, ]
    operands = Value[target_pack_or_un_pack_op, target_linalg_op, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(outer_perm) && push!(attributes, namedattribute("outer_perm", outer_perm))
    !isnothing(inner_perm) && push!(attributes, namedattribute("inner_perm", inner_perm))
    
    IR.create_operation(
        "transform.structured.pack_transpose", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`structured_pad`

Pads the operations pointed to by the target handle using the options
provides as operation attributes. The operation returns a handle to the
padded operation and to the padding operation (\"tensor.pad\").

#### Return modes

This operation ignores non-Linalg ops and drops them in the return.
This operation may produce a definiteFailure if the padding fails for any
reason.

If all the operations referred to by the `target` handle pad
properly, the transform succeeds. Otherwise the transform silently fails.
The return handle points to only the subset of successfully produced
padded operations, which can be empty.
"""
function structured_pad(target::Value; padded::IR.Type, pad::IR.Type, padding_values=nothing, padding_dimensions=nothing, pad_to_multiple_of=nothing, pack_paddings=nothing, transpose_paddings=nothing, copy_back=nothing, location=Location())
    results = IR.Type[padded, pad, ]
    operands = Value[target, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(padding_values) && push!(attributes, namedattribute("padding_values", padding_values))
    !isnothing(padding_dimensions) && push!(attributes, namedattribute("padding_dimensions", padding_dimensions))
    !isnothing(pad_to_multiple_of) && push!(attributes, namedattribute("pad_to_multiple_of", pad_to_multiple_of))
    !isnothing(pack_paddings) && push!(attributes, namedattribute("pack_paddings", pack_paddings))
    !isnothing(transpose_paddings) && push!(attributes, namedattribute("transpose_paddings", transpose_paddings))
    !isnothing(copy_back) && push!(attributes, namedattribute("copy_back", copy_back))
    
    IR.create_operation(
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

If the operations referred to by the `target` handle promote
properly, the transform succeeds.

When successful, the return handle points to the \$target operation that
was modified inplace.
"""
function structured_promote(target::Value; transformed::IR.Type, operands_to_promote=nothing, use_full_tile_buffers=nothing, use_full_tiles_by_default=nothing, use_alloca=nothing, mapping=nothing, alignment=nothing, location=Location())
    results = IR.Type[transformed, ]
    operands = Value[target, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(operands_to_promote) && push!(attributes, namedattribute("operands_to_promote", operands_to_promote))
    !isnothing(use_full_tile_buffers) && push!(attributes, namedattribute("use_full_tile_buffers", use_full_tile_buffers))
    !isnothing(use_full_tiles_by_default) && push!(attributes, namedattribute("use_full_tiles_by_default", use_full_tiles_by_default))
    !isnothing(use_alloca) && push!(attributes, namedattribute("use_alloca", use_alloca))
    !isnothing(mapping) && push!(attributes, namedattribute("mapping", mapping))
    !isnothing(alignment) && push!(attributes, namedattribute("alignment", alignment))
    
    IR.create_operation(
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
function structured_replace(target::Value; replacement::IR.Type, bodyRegion::Region, location=Location())
    results = IR.Type[replacement, ]
    operands = Value[target, ]
    owned_regions = Region[bodyRegion, ]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "transform.structured.replace", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`structured_rewrite_in_destination_passing_style`

Rewrite a supported tensor operation that is not in destination-passing style
into a form that is in destination-passing style.
Currently supported operations are:
  - tensor.pad
  - tensor.generate
  - tensor.from_elements
This dichotomy hints at a future interface, for now the implementation just 
switches between different implementation.

#### Return modes

This operation ignores non-unsupported ops and drops them from the return.
If all the operations referred to by the `target` handle generalize
properly, the transform succeeds. Otherwise the transform silently fails.
The return handle points to a subset of successfully produced operations:
  - `tensor.pad` case, the returned handle points to the tensor.insert_slice.
  - `tensor.generate` case, the returned handle points to the linalg.generic.
  - `tensor.from_elements` case, the returned handle points to the last 
    `tensor.insert`.
"""
function structured_rewrite_in_destination_passing_style(target::Value; transformed::IR.Type, location=Location())
    results = IR.Type[transformed, ]
    operands = Value[target, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "transform.structured.rewrite_in_destination_passing_style", location;
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
If all the operations referred to by the `target` handle scalarize
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
function structured_scalarize(target::Value; result::IR.Type, location=Location())
    results = IR.Type[result, ]
    operands = Value[target, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
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
function structured_split(target::Value, dynamic_split_point=nothing::Union{Nothing, Value}; first::IR.Type, second::IR.Type, dimension, static_split_point, location=Location())
    results = IR.Type[first, second, ]
    operands = Value[target, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("dimension", dimension), namedattribute("static_split_point", static_split_point), ]
    !isnothing(dynamic_split_point) && push!(operands, dynamic_split_point)
    
    IR.create_operation(
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

If all the operations referred to by the `target` handle split
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
function structured_split_reduction(target::Value; init_or_alloc_op::IR.Type, fill_op::IR.Type, split_linalg_op::IR.Type, combining_linalg_op::IR.Type, split_factor=nothing, insert_split_dimension=nothing, inner_parallel=nothing, use_scaling_algorithm=nothing, use_alloc=nothing, location=Location())
    results = IR.Type[init_or_alloc_op, fill_op, split_linalg_op, combining_linalg_op, ]
    operands = Value[target, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(split_factor) && push!(attributes, namedattribute("split_factor", split_factor))
    !isnothing(insert_split_dimension) && push!(attributes, namedattribute("insert_split_dimension", insert_split_dimension))
    !isnothing(inner_parallel) && push!(attributes, namedattribute("inner_parallel", inner_parallel))
    !isnothing(use_scaling_algorithm) && push!(attributes, namedattribute("use_scaling_algorithm", use_scaling_algorithm))
    !isnothing(use_alloc) && push!(attributes, namedattribute("use_alloc", use_alloc))
    
    IR.create_operation(
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
function structured_tile(target::Value, dynamic_sizes::Vector{Value}; tiled_linalg_op::IR.Type, loops::Vector{IR.Type}, static_sizes=nothing, interchange=nothing, scalable_sizes=nothing, location=Location())
    results = IR.Type[tiled_linalg_op, loops..., ]
    operands = Value[target, dynamic_sizes..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(static_sizes) && push!(attributes, namedattribute("static_sizes", static_sizes))
    !isnothing(interchange) && push!(attributes, namedattribute("interchange", interchange))
    !isnothing(scalable_sizes) && push!(attributes, namedattribute("scalable_sizes", scalable_sizes))
    
    IR.create_operation(
        "transform.structured.tile", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`structured_tile_reduction_using_forall`

Tile a PartialReductionOpInterface op to a tiled `scf.forall` doing
partial reduction.

This transformation tiles the `target` along the reduction dimensions. It
creates a tensor initialized with the identity value. Then it creates a
`scf.forall` loops with the number threads given by `num_threads`.
The op is tiled op with a size equal to `floordiv(size, num_threads)`.
All the partial reduction value is are parallel inserted to create a new
tensor. After the loop a merge operation is created to do a final reduction
with the partial reductions tensor.
If an extra `tile_sizes` parameter is passed the tiles are cyclically
distributed on the threads of the `scf.foralls` loop.

#### Return modes

This 4 returned handles point to:
  - the parent forall op,
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
  %2 = scf.forall (%arg2) in (%c5) shared_outs(%arg3 = %1) -> (tensor<?x5xf32>) {
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
    scf.forall.in_parallel {
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
function structured_tile_reduction_using_forall(target::Value; forall_op::IR.Type, fill_op::IR.Type, split_linalg_op::IR.Type, combining_linalg_op::IR.Type, num_threads=nothing, tile_sizes=nothing, mapping=nothing, location=Location())
    results = IR.Type[forall_op, fill_op, split_linalg_op, combining_linalg_op, ]
    operands = Value[target, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(num_threads) && push!(attributes, namedattribute("num_threads", num_threads))
    !isnothing(tile_sizes) && push!(attributes, namedattribute("tile_sizes", tile_sizes))
    !isnothing(mapping) && push!(attributes, namedattribute("mapping", mapping))
    
    IR.create_operation(
        "transform.structured.tile_reduction_using_forall", location;
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
function structured_tile_reduction_using_scf(target::Value; for_op::IR.Type, fill_op::IR.Type, split_linalg_op::IR.Type, combining_linalg_op::IR.Type, tile_sizes=nothing, location=Location())
    results = IR.Type[for_op, fill_op, split_linalg_op, combining_linalg_op, ]
    operands = Value[target, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(tile_sizes) && push!(attributes, namedattribute("tile_sizes", tile_sizes))
    
    IR.create_operation(
        "transform.structured.tile_reduction_using_scf", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`structured_tile_to_forall_op`

Tile a TilingInterface op to a tiled `scf.forall`.

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
resulting `scf.forall`.

Note: `tile_sizes` and `num_threads` are variadic. Each tile size/number of
threads can be an index attribute or a transform handle that is mapped to
exactly one payload op with exactly one index result.

#### Return modes

This operation ignores ops that do not implement the TilingInterface and
drops them in the return.

If all the operations referred to by the `target` handle tile
successfully, the transform succeeds.
Otherwise the transform silently fails.

The two returned handles point to only the subset of successfully produced
tiled operations, which can all be empty.

These two returned handles point to:
  - the new scf.forall op,
  - the tiled op that implements TilingInterface.

#### Example using `num_threads`

```
%0 = pdl_match @match_matmul in %arg1
%3:2 = transform.structured.tile_to_forall_op %0 num_threads [10, 20]
```

#### Example using `tile_sizes`

```
%0 = pdl_match @match_matmul in %arg1
%sz = pdl_match @match_size_op in %arg1
%3:2 = transform.structured.tile_to_forall_op %0 tile_sizes [0, %sz, 20]
```
"""
function structured_tile_to_forall_op(target::Value, num_threads::Vector{Value}, tile_sizes::Vector{Value}, packed_num_threads=nothing::Union{Nothing, Value}; packed_tile_sizes=nothing::Union{Nothing, Value}, forall_op::IR.Type, tiled_op::IR.Type, static_num_threads=nothing, static_tile_sizes=nothing, mapping=nothing, location=Location())
    results = IR.Type[forall_op, tiled_op, ]
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
    
    IR.create_operation(
        "transform.structured.tile_to_forall_op", location;
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
function structured_tile_to_scf_for(target::Value, dynamic_sizes::Vector{Value}; tiled_linalg_op::IR.Type, loops::Vector{IR.Type}, static_sizes=nothing, interchange=nothing, location=Location())
    results = IR.Type[tiled_linalg_op, loops..., ]
    operands = Value[target, dynamic_sizes..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(static_sizes) && push!(attributes, namedattribute("static_sizes", static_sizes))
    !isnothing(interchange) && push!(attributes, namedattribute("interchange", interchange))
    
    IR.create_operation(
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
  tests only but may be promoted to a first class attribute in the future.

#### Return modes:

This operation produces `definiteFailure` if vectorization fails for any
reason.
The operation always returns the handle to the target op that is expected
to be isolated from above.
"""
function structured_vectorize(target::Value; transformed::IR.Type, vectorize_padding=nothing, vectorize_nd_extract=nothing, disable_multi_reduction_to_contract_patterns=nothing, disable_transfer_permutation_map_lowering_patterns=nothing, location=Location())
    results = IR.Type[transformed, ]
    operands = Value[target, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(vectorize_padding) && push!(attributes, namedattribute("vectorize_padding", vectorize_padding))
    !isnothing(vectorize_nd_extract) && push!(attributes, namedattribute("vectorize_nd_extract", vectorize_nd_extract))
    !isnothing(disable_multi_reduction_to_contract_patterns) && push!(attributes, namedattribute("disable_multi_reduction_to_contract_patterns", disable_multi_reduction_to_contract_patterns))
    !isnothing(disable_transfer_permutation_map_lowering_patterns) && push!(attributes, namedattribute("disable_transfer_permutation_map_lowering_patterns", disable_transfer_permutation_map_lowering_patterns))
    
    IR.create_operation(
        "transform.structured.vectorize", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

import ...IR: IR, NamedAttribute, Value, Location, Block, Region, Attribute, context, IndexType
import ..Dialects: namedattribute, operandsegmentsizes


"""
`apply_patterns_memref_expand_ops`

Collects patterns to rewrite ops within the memref dialect.

- Converts `atomic_rmw` that cannot be lowered to a simple atomic op with
  AtomicRMWOpLowering pattern, e.g. with \"minf\" or \"maxf\" attributes, to
  `memref.generic_atomic_rmw` with the expanded code.
- Converts `memref.reshape` that has a target shape of a statically-known
  size to `memref.reinterpret_cast`.
"""
function apply_patterns_memref_expand_ops(; location=Location())
    results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "transform.apply_patterns.memref.expand_ops", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`apply_patterns_memref_expand_strided_metadata`

Collects patterns for expanding memref operations that modify the metadata
(sizes, offset, strides) of a memref into easier to analyze constructs.
"""
function apply_patterns_memref_expand_strided_metadata(; location=Location())
    results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "transform.apply_patterns.memref.expand_strided_metadata", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`apply_patterns_memref_extract_address_computations`

Collects patterns for extracting address computations from operations
with memory accesses such that these memory accesses use only a base
pointer.

For instance,
```mlir
memref.load %base[%off0, ...]
```

Will be rewritten in:
```mlir
%new_base = memref.subview %base[%off0,...][1,...][1,...]
memref.load %new_base[%c0,...]
```
"""
function apply_patterns_memref_extract_address_computations(; location=Location())
    results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "transform.apply_patterns.memref.extract_address_computations", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`apply_patterns_memref_fold_memref_alias_ops`

Collects patterns for folding memref aliasing ops (memref.subview) into
consumer load/store ops (affine.load, memref.load, nvgpu.ldmatrix,
vector.load, vector.transfer_read, affine.store, memref.store, etc.) and
other ops (e.g., memref.subview).
"""
function apply_patterns_memref_fold_memref_alias_ops(; location=Location())
    results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "transform.apply_patterns.memref.fold_memref_alias_ops", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`apply_patterns_memref_resolve_ranked_shaped_type_result_dims`

Collects patterns that resolve `memref.dim` operations with values that are
defined by operations that implement the `ReifyRankedShapedTypeOpInterface`,
in terms of shapes of its input operands.
"""
function apply_patterns_memref_resolve_ranked_shaped_type_result_dims(; location=Location())
    results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "transform.apply_patterns.memref.resolve_ranked_shaped_type_result_dims", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`memref_make_loop_independent`

Rewrite the targeted ops such that their index-typed operands no longer
depend on any loop induction variable of the `num_loop` enclosing `scf.for`
loops. I.e., compute an upper bound that is independent of any such loop IV
for every tensor dimension. The transformed op could then be hoisted from
the `num_loop` enclosing loops. To preserve the original semantics, place a
`memref.subview` inside the loop.

Currently supported operations are:
- memref.alloca: Replaced with a new memref.alloca with upper bound sizes,
  followed by a memref.subview.

#### Return modes

This operation fails if at least one induction variable could not be
eliminated. In case the targeted op is already independent of induction
variables, this transform succeeds and returns the unmodified target op.

Otherwise, the returned handle points to a subset of the produced ops:
- memref.alloca: The returned handle points to the memref.subview op.

This transform op consumes the target handle and produces a result handle.
"""
function memref_make_loop_independent(target::Value; transformed::IR.Type, num_loops, location=Location())
    results = IR.Type[transformed, ]
    operands = Value[target, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("num_loops", num_loops), ]
    
    IR.create_operation(
        "transform.memref.make_loop_independent", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`memref_multibuffer`

Transformation to do multi-buffering/array expansion to remove
dependencies on the temporary allocation between consecutive loop
iterations. This transform expands the size of an allocation by
a given multiplicative factor and fixes up any users of the
multibuffered allocation.
If skip analysis is not set the transformation will only apply
if it can prove that there is no data being carried across loop
iterations.

#### Return modes

This operation returns the new allocation if multi-buffering
succeeds, and failure otherwise.
"""
function memref_multibuffer(target::Value; transformed::IR.Type, factor, skip_analysis=nothing, location=Location())
    results = IR.Type[transformed, ]
    operands = Value[target, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("factor", factor), ]
    !isnothing(skip_analysis) && push!(attributes, namedattribute("skip_analysis", skip_analysis))
    
    IR.create_operation(
        "transform.memref.multibuffer", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

import ...IR: IR, NamedAttribute, Value, Location, Block, Region, Attribute, context, IndexType
import ..Dialects: namedattribute, operandsegmentsizes


"""
`nvgpu_create_async_groups`

Look for global to shared memory copies within the targeted op in the form
of vector transfer ops and convert them to async copies when possible.
Consecutive copies are put into the same group. A \"wait\" operation is
inserted right at the of end the group.

`bypass_l1` specifies whether `bypassL1` attributes should be added to
the async copies. `bypass_l1` is a compiler hint: only 16 byte transfers
can bypass the L1 cache, so this attribute is not set for any other transfer
sizes.

#### Return modes

This op consumes the `target` handle and produces the `result` handle, which
is mapped to the same payload operations as the `target` handle. The op
modifies the payload.
"""
function nvgpu_create_async_groups(target::Value; result::IR.Type, bypass_l1=nothing, location=Location())
    results = IR.Type[result, ]
    operands = Value[target, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(bypass_l1) && push!(attributes, namedattribute("bypass_l1", bypass_l1))
    
    IR.create_operation(
        "transform.nvgpu.create_async_groups", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`nvgpu_pipeline_shared_memory_copies`

Applies software pipelining to a given scf.for loop. The pipelining
strategy will look for a load into shared memory and pipeline it to overlap
it with the rest of the loop.

NOTE: It is user responsibility to ensure that there are no dependency
between `depth` iterations of the loop by using multi-buffering. It is
also user responsibility to ensure a sufficient amount of shared memory
is allocated to cover eventual writes by `depth-1` speculative
iterations.

`depth` will indicate how many stages the software pipeline should have.
`peel_epilogue` allows to force the epilogue to be peeled out instead of
potentially using predicated operations for the epilogue phase.

#### Return modes

Consumes the operand handle and produces a result handle pointing to the
loop, which may or may not have been pipelined. Produces a definite failure
if the loop pipeliner mutated the IR before failing to pipeline, in
particular if `peel_epilogue` is not set and the loop body doesn\'t support
predication. If failure propagation mode is set to \"propagate\", produces a
silenceable failure when pipelining preconditions, e.g., loop bound being
static, are not met or when the loop wasn\'t pipelined because due to the
lack of loads into shared memory. If the failure propagation mode is set
to \"suppress\" (default), succeeds in these case and associates the result
handle with the original loop.

TODO: the shared memory part and behavior specific to NVGPU should be
made orthogonal to pipelining so that `transform.loop.pipeline` becomes
usable here.
"""
function nvgpu_pipeline_shared_memory_copies(for_op::Value; result::IR.Type, depth, peel_epilogue=nothing, failure_propagation_mode=nothing, location=Location())
    results = IR.Type[result, ]
    operands = Value[for_op, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("depth", depth), ]
    !isnothing(peel_epilogue) && push!(attributes, namedattribute("peel_epilogue", peel_epilogue))
    !isnothing(failure_propagation_mode) && push!(attributes, namedattribute("failure_propagation_mode", failure_propagation_mode))
    
    IR.create_operation(
        "transform.nvgpu.pipeline_shared_memory_copies", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`nvgpu_rewrite_matmul_as_mma_sync`

Rewrite a matmul operation on memref to an mma.sync operation on vectors.

Memory copies with the required access patterns are automatically inserted.
Operations that do not have a 1-1 mapping to mma.sync operations are left
unchanged.
"""
function nvgpu_rewrite_matmul_as_mma_sync(target::Value; location=Location())
    results = IR.Type[]
    operands = Value[target, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "transform.nvgpu.rewrite_matmul_as_mma_sync", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

import ...IR: IR, NamedAttribute, Value, Location, Block, Region, Attribute, context, IndexType
import ..Dialects: namedattribute, operandsegmentsizes


"""
`apply_patterns_scf_for_loop_canonicalization`

Collects patterns for canonicalizing operations inside SCF loop bodies.
At the moment, only affine.min/max computations with iteration variables,
loop bounds and loop steps are canonicalized.
"""
function apply_patterns_scf_for_loop_canonicalization(; location=Location())
    results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "transform.apply_patterns.scf.for_loop_canonicalization", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`loop_get_parent_for`

Produces a handle to the n-th (default 1) parent `scf.for` or `affine.for`
(when the affine flag is true) loop for each Payload IR operation
associated with the operand. Fails if such a loop cannot be found. The list
of operations associated with the handle contains parent operations in the
same order as the list associated with the operand, except for operations
that are parents to more than one input which are only present once.
"""
function loop_get_parent_for(target::Value; parent::IR.Type, num_loops=nothing, affine=nothing, location=Location())
    results = IR.Type[parent, ]
    operands = Value[target, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(num_loops) && push!(attributes, namedattribute("num_loops", num_loops))
    !isnothing(affine) && push!(attributes, namedattribute("affine", affine))
    
    IR.create_operation(
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
function loop_coalesce(target::Value; transformed::IR.Type, location=Location())
    results = IR.Type[transformed, ]
    operands = Value[target, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "transform.loop.coalesce", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`loop_outline`

Moves the loop into a separate function with the specified name and replaces
the loop in the Payload IR with a call to that function. Takes care of
forwarding values that are used in the loop as function arguments. If the
operand is associated with more than one loop, each loop will be outlined
into a separate function. The provided name is used as a _base_ for forming
actual function names following `SymbolTable` auto-renaming scheme to avoid
duplicate symbols. Expects that all ops in the Payload IR have a
`SymbolTable` ancestor (typically true because of the top-level module).

#### Return Modes

Returns a handle to the list of outlined functions and a handle to the
corresponding function call operations in the same order as the operand
handle.

Produces a definite failure if outlining failed for any of the targets.
"""
function loop_outline(target::Value; function_::IR.Type, call::IR.Type, func_name, location=Location())
    results = IR.Type[function_, call, ]
    operands = Value[target, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("func_name", func_name), ]
    
    IR.create_operation(
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
function loop_peel(target::Value; transformed::IR.Type, fail_if_already_divisible=nothing, location=Location())
    results = IR.Type[transformed, ]
    operands = Value[target, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(fail_if_already_divisible) && push!(attributes, namedattribute("fail_if_already_divisible", fail_if_already_divisible))
    
    IR.create_operation(
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
function loop_pipeline(target::Value; transformed::IR.Type, iteration_interval=nothing, read_latency=nothing, location=Location())
    results = IR.Type[transformed, ]
    operands = Value[target, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(iteration_interval) && push!(attributes, namedattribute("iteration_interval", iteration_interval))
    !isnothing(read_latency) && push!(attributes, namedattribute("read_latency", read_latency))
    
    IR.create_operation(
        "transform.loop.pipeline", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`loop_promote_if_one_iteration`

Promotes the given target loop op if it has a single iteration. I.e., the
loop op is removed and only the body remains.

#### Return modes

This transform fails if the target is mapped to ops that are loops. Ops are
considered loops if they implement the `LoopLikeOpInterface`. Otherwise,
this transform always succeeds. The transform consumes the target handle and
modifies the payload.
"""
function loop_promote_if_one_iteration(target::Value; location=Location())
    results = IR.Type[]
    operands = Value[target, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "transform.loop.promote_if_one_iteration", location;
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
    results = IR.Type[]
    operands = Value[target, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("factor", factor), ]
    
    IR.create_operation(
        "transform.loop.unroll", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`scf_take_assumed_branch`

Given an scf.if conditional, inject user-defined information that it is
always safe to execute only the if or else branch. 

This is achieved by just replacing the scf.if by the content of one of its
branches.

This is particularly useful for user-controlled rewriting of conditionals
that exist solely to guard against out-of-bounds behavior.

At the moment, no assume or assert operation is emitted as it is not always
desirable. In the future, this may be controlled by a dedicated attribute.

#### Return modes

The transform only consumes its operand and does not produce any result.
The transform definitely fails if `take_else_branch` is specified and the
`else` region is empty.
"""
function scf_take_assumed_branch(target::Value; take_else_branch=nothing, location=Location())
    results = IR.Type[]
    operands = Value[target, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(take_else_branch) && push!(attributes, namedattribute("take_else_branch", take_else_branch))
    
    IR.create_operation(
        "transform.scf.take_assumed_branch", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

import ...IR: IR, NamedAttribute, Value, Location, Block, Region, Attribute, context, IndexType
import ..Dialects: namedattribute, operandsegmentsizes


"""
`apply_patterns_tensor_drop_redundant_insert_slice_rank_expansion`

Indicates that redundant tensor.insert_slice rank reductions should be
dropped. E.g., cases where a tensor.extract_slice rank reduction immediately
follows an inverse tensor.insert_slice rank expansion.
"""
function apply_patterns_tensor_drop_redundant_insert_slice_rank_expansion(; location=Location())
    results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "transform.apply_patterns.tensor.drop_redundant_insert_slice_rank_expansion", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`apply_patterns_tensor_fold_into_pack_and_unpack`

Indicates that operations like tensor.pad and tensor.extract_slice should
be folded into tensor.pack and tensor.unpack operations, respectively.
"""
function apply_patterns_tensor_fold_into_pack_and_unpack(; location=Location())
    results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "transform.apply_patterns.tensor.fold_into_pack_and_unpack", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`apply_patterns_tensor_fold_tensor_empty`

Indicates that tensor.extract_slice and reassociative reshapes should be
folded into tensor.empty.

If `fold_single_use_only` is set to \"true\", only tensor.empty that have a
single use are folded.
"""
function apply_patterns_tensor_fold_tensor_empty(; fold_single_use_only=nothing, location=Location())
    results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(fold_single_use_only) && push!(attributes, namedattribute("fold_single_use_only", fold_single_use_only))
    
    IR.create_operation(
        "transform.apply_patterns.tensor.fold_tensor_empty", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`apply_patterns_tensor_fold_tensor_subset_ops_into_vector_transfers`

Indicates that tensor.extract_slice -> vector.transfer_read and
vector.transfer_write -> tensor.insert_slice op chains should be folded into
vector tranfer read and write ops
"""
function apply_patterns_tensor_fold_tensor_subset_ops_into_vector_transfers(; location=Location())
    results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "transform.apply_patterns.tensor.fold_tensor_subset_ops_into_vector_transfers", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`apply_patterns_tensor_fold_tensor_subset_ops`

Indicates that tensor.empty should be folded with tensor.extract_slice,
tensor.expand_shape and tensor.collapse_shape.
"""
function apply_patterns_tensor_fold_tensor_subset_ops(; location=Location())
    results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "transform.apply_patterns.tensor.fold_tensor_subset_ops", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`apply_patterns_tensor_merge_consecutive_insert_extract_slice`

Indicates that consecutive tensor.extract_slice/tensor.insert_slice ops
should be merged into a single op. These patterns are not canonicalizations
because the bufferization is sensitive to IR structure.
"""
function apply_patterns_tensor_merge_consecutive_insert_extract_slice(; location=Location())
    results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "transform.apply_patterns.tensor.merge_consecutive_insert_extract_slice", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`apply_patterns_tensor_reassociative_reshape_folding`

Indicates that reassociative reshapes (tensor.collapse_shape /
tensor.expand_shape) should be folded with inverse rank expansions / rank
reductions (via tensor.insert_slice / tensor.extract_slice).
"""
function apply_patterns_tensor_reassociative_reshape_folding(; location=Location())
    results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "transform.apply_patterns.tensor.reassociative_reshape_folding", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`apply_patterns_tensor_rewrite_as_constant`

Indicates that tensor ops (such as tensor.generate) should be replaced with
constants (arith.constant) when possible.
"""
function apply_patterns_tensor_rewrite_as_constant(; location=Location())
    results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "transform.apply_patterns.tensor.rewrite_as_constant", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`tensor_make_loop_independent`

Rewrite the targeted ops such that their index-typed operands no longer
depend on any loop induction variable of the `num_loop` enclosing `scf.for`
loops. I.e., compute an upper bound that is independent of any such loop IV
for every tensor dimension. The transformed op could then be hoisted from
the `num_loop` enclosing loops. To preserve the original semantics, place a
`tensor.extract_slice` inside the loop.

Currently supported operations are:
- tensor.empty: Replaced with a new tensor.empty with upper bound sizes,
  followed by a tensor.extract_slice.
- tensor.pad: Replaced by an upper bound padding, followed by a
  tensor.extract_slice.

#### Return modes

This operation fails if at least one induction variable could not be
eliminated. In case the targeted op is already independent of induction
variables, this transform succeeds and returns the unmodified target op.

Otherwise, the returned handle points to a subset of the produced ops:
- tensor.empty: The returned handle points to the tensor.extract_slice op.
- tensor.pad: The returned handle points to the tensor.extract_slice op.

This transform op consumes the target handle and produces a result handle.
"""
function tensor_make_loop_independent(target::Value; transformed::IR.Type, num_loops, location=Location())
    results = IR.Type[transformed, ]
    operands = Value[target, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("num_loops", num_loops), ]
    
    IR.create_operation(
        "transform.tensor.make_loop_independent", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

import ...IR: IR, NamedAttribute, Value, Location, Block, Region, Attribute, context, IndexType
import ..Dialects: namedattribute, operandsegmentsizes


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
^bb0(%arg0: !transform.any_op):
  // Try a fallible transformation.
  %0 = transform.fallible %arg0 // ...
  // If succeeded, yield the the result of the transformation.
  transform.yield %0 : !transform.any_op
}, {
^bb0(%arg0: !transform.any_op):
  // Otherwise, the second alternative is tried and it always succeeds by
  // returning the original handle.
  transform.yield %arg0 : !transform.any_op
}
```
"""
function alternatives(scope=nothing::Union{Nothing, Value}; results_::Vector{IR.Type}, alternatives::Vector{Region}, location=Location())
    results = IR.Type[results_..., ]
    operands = Value[]
    owned_regions = Region[alternatives..., ]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(scope) && push!(operands, scope)
    
    IR.create_operation(
        "transform.alternatives", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`annotate`

Adds an attribute with the given `name` to the `target` operation. An
optional `param` handle can be provided to give the attribute a specific
value, else a UnitAttr is added. A single attribute will be broadcasted to
all target operations, otherwise the attributes will be mapped 1:1 based on
the order within the handles.

Fails silently if the length of the parameter payload does not match the length of
the target payload. Does not consume the provided handles.
"""
function annotate(target::Value, param=nothing::Union{Nothing, Value}; name, location=Location())
    results = IR.Type[]
    operands = Value[target, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("name", name), ]
    !isnothing(param) && push!(operands, param)
    
    IR.create_operation(
        "transform.annotate", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`apply_patterns_canonicalization`

This op populates all canonicalization patterns of all loaded dialects in
an `apply_patterns` transform.
"""
function apply_patterns_canonicalization(; location=Location())
    results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "transform.apply_patterns.canonicalization", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`apply_cse`

This transform applies common subexpression elimination (CSE) to the body
of the targeted op.

This transform reads the target handle and modifies the payload. Existing
handles to operations inside of the targeted op are retained and updated if
necessary. Note that this can lead to situations where a handle, that was
previously mapped to multiple distinct (but equivalent) operations, is now
mapped to the same operation multiple times.
"""
function apply_cse(target::Value; location=Location())
    results = IR.Type[]
    operands = Value[target, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "transform.apply_cse", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`apply_dce`

This transform applies dead code elimination (DCE) to the body of the
targeted op.

Note: \"transform.apply_patterns\" with an empty region can also be used to
remove dead ops. However, that op applies additional simplifications such as
op folding and region simplification.

This transform reads the target handle and modifies the payload. Note that
this transform may silently remove payload ops from handles.
"""
function apply_dce(target::Value; location=Location())
    results = IR.Type[]
    operands = Value[target, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "transform.apply_dce", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`apply_licm`

This transform moves side-effect free, loop invariant code out of the
targeted loop-like op. The targeted op must implement the
`LoopLikeOpInterface`.

Note: To move invariant ops from a loop nest, this transform must be applied
to each loop of the loop nest, starting with the inner-most loop.

This transform reads the target handle and modifies the payload.
"""
function apply_licm(target::Value; location=Location())
    results = IR.Type[]
    operands = Value[target, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "transform.apply_licm", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`apply_patterns`

This transform greedily applies the specified patterns to the body of the
targeted op until a fixpoint was reached. Patterns are not applied to the
targeted op itself.

The patterns that should be applied are specified in the graph region of
this op. They must implement the `PatternDescriptorOpInterface`. The order
in which patterns are applied is unspecified; i.e., the ordering of ops in
the region of this op is irrelevant.

If `apple_cse` is set, the greedy pattern rewrite is interleaved with
common subexpression elimination (CSE): both are repeated until a fixpoint
is reached.

This transform only reads the target handle and modifies the payload. If a
pattern erases or replaces a tracked op, the mapping is updated accordingly.

Only replacements via `RewriterBase::replaceOp` or `replaceOpWithNewOp` are
considered \"payload op replacements\". Furthermore, only if the replacement
values are defined by the same op and that op has the same type as the
original op, the mapping is updated. Otherwise, this transform fails
silently. More details can be found at the documentation site of
`TrackingListener`.

This transform also fails silently if the pattern application did not
converge within the default number of iterations/rewrites of the greedy
pattern rewrite driver.
"""
function apply_patterns(target::Value; apply_cse=nothing, patterns::Region, location=Location())
    results = IR.Type[]
    operands = Value[target, ]
    owned_regions = Region[patterns, ]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(apply_cse) && push!(attributes, namedattribute("apply_cse", apply_cse))
    
    IR.create_operation(
        "transform.apply_patterns", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`apply_registered_pass`

This transform applies the specified pass to the targeted ops. The name of
the pass is specified as a string attribute, as set during pass
registration. Optionally, pass options may be specified as a string
attribute. The pass options syntax is identical to the one used with
\"mlir-opt\".

This transform consumes the target handle and produces a new handle that is
mapped to the same op. Passes are not allowed to remove/modify the operation
that they operate on, so the target op is guaranteed to still exist. The
target handle is invalidated because a pass may arbitrarily modify the body
of targeted ops.
"""
function apply_registered_pass(target::Value; result::IR.Type, pass_name, options=nothing, location=Location())
    results = IR.Type[result, ]
    operands = Value[target, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("pass_name", pass_name), ]
    !isnothing(options) && push!(attributes, namedattribute("options", options))
    
    IR.create_operation(
        "transform.apply_registered_pass", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`cast`

"""
function cast(input::Value; output::IR.Type, location=Location())
    results = IR.Type[output, ]
    operands = Value[input, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "transform.cast", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`foreach_match`

Given a pair of co-indexed lists of transform dialect symbols (such as
`transform.named_sequence`), walks the payload IR associated with the root
handle and interprets the symbols as matcher/action pairs by applying the
body of the corresponding symbol definition. The symbol from the first list
is the matcher part: if it results in a silenceable error, the error is
silenced and the next matcher is attempted. Definite failures from any
matcher stop the application immediately and are propagated unconditionally.
If none of the matchers succeeds, the next payload operation in walk order
(post-order at the moment of writing, double check `Operation::walk`) is
matched. If a matcher succeeds, the co-indexed action symbol is applied and
the following matchers are not applied to the same payload operation. If the
action succeeds, the next payload operation in walk order is matched. If it
fails, both silenceable and definite errors are propagated as the result of
this op.

The matcher symbol must take one operand of a type that implements the same
transform dialect interface as the `root` operand (a check is performed at
application time to see if the associated payload satisfies the constraints
of the actual type). It must not consume the operand as multiple matchers
may be applied. The matcher may produce any number of results. The action
symbol paired with the matcher must take the same number of arguments as the
matcher has results, and these arguments must implement the same transform
dialect interfaces, but not necessarily have the exact same type (again, a
check is performed at application time to see if the associated payload
satisfies the constraints of actual types on both sides). The action symbol
may not have results. The actions are expected to only modify payload
operations nested in the `root` payload operations associated with the
operand of this transform operation. Furhermore, the actions may not modify
operations outside of the currently matched payload operation, e.g., they
may not modify sibling or parent operations. If such behavior is desired,
the parent must be matched first and the nested operations obtained by
traversing the IR from the parent. This is due to the matching being
performed as a post-order IR walk.

This operation consumes the operand and produces a new handle associated
with the same payload. This is necessary to trigger invalidation of handles
to any of the payload operations nested in the payload operations associated
with the operand, as those are likely to be modified by actions. Note that
the root payload operation associated with the operand are not matched.

The operation succeeds if none of the matchers produced a definite failure
during application and if all of the applied actions produced success. Note
that it also succeeds if all the matchers failed on all payload operations,
i.e. failure to apply is not an error. The operation produces a silenceable
failure if any applied action produced a silenceable failure. In this case,
the resulting handle is associated with an empty payload. The operation
produces a definite failure if any of the applied matchers or actions
produced a definite failure.
"""
function foreach_match(root::Value; updated::IR.Type, matchers, actions, location=Location())
    results = IR.Type[updated, ]
    operands = Value[root, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("matchers", matchers), namedattribute("actions", actions), ]
    
    IR.create_operation(
        "transform.foreach_match", location;
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
function foreach(target::Value; results_::Vector{IR.Type}, body::Region, location=Location())
    results = IR.Type[results_..., ]
    operands = Value[target, ]
    owned_regions = Region[body, ]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "transform.foreach", location;
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
function get_consumers_of_result(target::Value; consumers::IR.Type, result_number, location=Location())
    results = IR.Type[consumers, ]
    operands = Value[target, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("result_number", result_number), ]
    
    IR.create_operation(
        "transform.get_consumers_of_result", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`get_defining_op`

The handle defined by this Transform op corresponds to the defining op of
the targeted value.

This transform fails silently if the targeted value is a block argument.
"""
function get_defining_op(target::Value; result::IR.Type, location=Location())
    results = IR.Type[result, ]
    operands = Value[target, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "transform.get_defining_op", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`get_parent_op`

The handle defined by this Transform op corresponds to the parents of the
targeted payload ops (in the same order).

Requirements that parent ops must fulfill can be optionally specified. In
that case for each target op, the closest parent op that fulfills all
requirements, is returned.
- `isolated_from_above`: the parent op must be isolated from above
- `op_name`: the parent op must have the specified name

If `deduplicate` is set, the result handle does not contain any duplicate
ops. For example, given the list
\"(childof(A), childof(B), childof(B), childof(A), childof(B))\", the
resulting list will be just \"(A, B)\". Note that no other semantic ordering
is applied, e.g., \"B\" may itself be a parent of \"A\". This may have an impact
on the further transformation applied to the handle produced here.

If any of the given Payload IR ops has no such suitable parent, the
transformation fails silently.
"""
function get_parent_op(target::Value; parent::IR.Type, isolated_from_above=nothing, op_name=nothing, deduplicate=nothing, location=Location())
    results = IR.Type[parent, ]
    operands = Value[target, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(isolated_from_above) && push!(attributes, namedattribute("isolated_from_above", isolated_from_above))
    !isnothing(op_name) && push!(attributes, namedattribute("op_name", op_name))
    !isnothing(deduplicate) && push!(attributes, namedattribute("deduplicate", deduplicate))
    
    IR.create_operation(
        "transform.get_parent_op", location;
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
function get_producer_of_operand(target::Value; producer::IR.Type, operand_number, location=Location())
    results = IR.Type[producer, ]
    operands = Value[target, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("operand_number", operand_number), ]
    
    IR.create_operation(
        "transform.get_producer_of_operand", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`get_result`

The handle defined by this Transform op corresponds to the OpResult with
`result_number` that is defined by the given `target` operation.

This transform fails silently if the targeted operation does not have enough
results. It reads the target handle and produces the result handle.
"""
function get_result(target::Value; result::IR.Type, result_number, location=Location())
    results = IR.Type[result, ]
    operands = Value[target, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("result_number", result_number), ]
    
    IR.create_operation(
        "transform.get_result", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`get_type`

This operation creates a new Transform parameter containing the
type(s) of the value(s) associated with the operand handle.

This transform never fails.
"""
function get_type(value::Value; type_param::IR.Type, elemental=nothing, location=Location())
    results = IR.Type[type_param, ]
    operands = Value[value, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(elemental) && push!(attributes, namedattribute("elemental", elemental))
    
    IR.create_operation(
        "transform.get_type", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`include_`

The application of this transform operation is equivalent to applying the
operations contained in the named transform sequence with operands being
remapped to block arguments. The behavior of the operation when a
transformation in the included named sequence produces a silenceable error
is controlled by the `failure_propagation_mode` attribute. When set to
`propagate`, the failure of any nested transformation in the sequence
implies immediate failure of the entire sequence with a silenceable error,
and no further transformation is attempted. When set to `suppress`,
silenceable errors in nested operations are ignored and further
transformations are applied. Beware that even silenceable errors may leave
the payload IR in a state unsuitable for further transformations. It is the
responsibility of the user to ensure the following transformations are
robust enough when errors are suppressed. Definite errors are propagated
immediately regardless of the mode. The objects associated with the results
of this operation are the same as those associated with the operands of the
`transform.yield` in the referenced named sequence.
"""
function include_(operands_::Vector{Value}; results_::Vector{IR.Type}, target, failure_propagation_mode, location=Location())
    results = IR.Type[results_..., ]
    operands = Value[operands_..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("target", target), namedattribute("failure_propagation_mode", failure_propagation_mode), ]
    
    IR.create_operation(
        "transform.include", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`match_operation_name`

Succeeds if the operation associated with the operand handle has one of the
given operation names. Produces a silenceable failure otherwise.

If more than one payload operation is associated with the operand handle,
produces a definite failure.
"""
function match_operation_name(operand_handle::Value; op_names, location=Location())
    results = IR.Type[]
    operands = Value[operand_handle, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("op_names", op_names), ]
    
    IR.create_operation(
        "transform.match.operation_name", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`match_param_cmpi`

Succeeds if all of the co-indexed values associated with the given
parameters relate as specified by the predicate (greater than, less than,
equal to, or their combinations). Comparison treats all values as signed.
Produces a silenceable failure otherwise.
"""
function match_param_cmpi(param::Value, reference::Value; predicate, location=Location())
    results = IR.Type[]
    operands = Value[param, reference, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("predicate", predicate), ]
    
    IR.create_operation(
        "transform.match.param.cmpi", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`merge_handles`

Creates a new Transform IR handle value that points to the same Payload IR
operations/values/parameters as the operand handles. The Payload IR elements
are listed in the same order as they are in the operand handles, grouped by
operand handle, e.g., all Payload IR associated with the first handle comes
first, then all Payload IR associated with the second handle and so on. If
`deduplicate` is set, do not add the given Payload IR operation, value, or
parameter more than once to the final list regardless of it coming from the
same or different handles. Consumes the operands and produces a new handle.
"""
function merge_handles(handles::Vector{Value}; result=nothing::Union{Nothing, IR.Type}, deduplicate=nothing, location=Location())
    results = IR.Type[]
    operands = Value[handles..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(results, result)
    !isnothing(deduplicate) && push!(attributes, namedattribute("deduplicate", deduplicate))
    
    IR.create_operation(
        "transform.merge_handles", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`named_sequence`

Defines a named (callable, function-like) sequence of other Transform
dialect operations that can be included using `transform.include` as part of
another Transform dialect construct. This sequence is not processed
immediately but rather dispatched to when the inclusion is processed. The
arguments and results can be used to communicate a subset of mapping into
the named sequence. The sequence must consist of a single block and end with
a `transform.yield` terminator. The operands of the terminator become the
results of the `transform.include`.

When dispatched to, the operations in the named sequence are executed one by
one, similarly to the regular unnamed sequence. The failure propagation mode
is specified on the `transform.include`. Different inclusions may use
different failure propagation modes. This transform operation always
succeeds by itself, but the inclusion may fail if any of the operations
fail.

Named sequences can only appear at the top-level of the Transform dialect
nesting structure. That is, they cannot be nested in other Transform dialect
operations. Furthermore, one of the ancestors must have the `SymbolTable`
trait and have the `transform.with_named_sequence` attribute attached.

Named sequences may include other named sequences via `transform.include`,
but recursion is *not* allowed.
"""
function named_sequence(; sym_name, function_type, sym_visibility=nothing, arg_attrs=nothing, res_attrs=nothing, body::Region, location=Location())
    results = IR.Type[]
    operands = Value[]
    owned_regions = Region[body, ]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("sym_name", sym_name), namedattribute("function_type", function_type), ]
    !isnothing(sym_visibility) && push!(attributes, namedattribute("sym_visibility", sym_visibility))
    !isnothing(arg_attrs) && push!(attributes, namedattribute("arg_attrs", arg_attrs))
    !isnothing(res_attrs) && push!(attributes, namedattribute("res_attrs", res_attrs))
    
    IR.create_operation(
        "transform.named_sequence", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`param_constant`

Produces a new transform dialect parameter associated with the singleton
list containing the given attribute. The operation itself always succeeds,
but the general association check may fail if the parameter type does not
accept the given kind of attribute as valid.
"""
function param_constant(; param::IR.Type, value, location=Location())
    results = IR.Type[param, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("value", value), ]
    
    IR.create_operation(
        "transform.param.constant", location;
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
    results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(target) && push!(operands, target)
    !isnothing(name) && push!(attributes, namedattribute("name", name))
    
    IR.create_operation(
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
function replicate(pattern::Value, handles::Vector{Value}; replicated::Vector{IR.Type}, location=Location())
    results = IR.Type[replicated..., ]
    operands = Value[pattern, handles..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "transform.replicate", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`select`

The handle defined by this Transform op corresponds to all operations among
`target` that have the specified properties. Currently the following
properties are supported:

- `op_name`: The op must have the specified name.

The result payload ops are in the same relative order as the targeted ops.
This transform op reads the `target` handle and produces the `result`
handle. It reads the payload, but does not modify it.
"""
function select(target::Value; result::IR.Type, op_name, location=Location())
    results = IR.Type[result, ]
    operands = Value[target, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("op_name", op_name), ]
    
    IR.create_operation(
        "transform.select", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`sequence`

The transformations indicated by the sequence are applied in order of their
appearance. Each value produced by a transformation within the sequence
corresponds to a group of operations or values in the payload IR, or to a
group of parameters, depending on the type of the value. The behavior of the
operation when a nested transformation produces a silenceable error is
controlled by the `failure_propagation_mode` attribute. When set to
`propagate`, the failure of any nested transformation in the sequence
implies immediate failure of the entire sequence with a silenceable error,
and no further transformation is attempted. When set to `suppress`,
silenceable errors in nested operations are ignored and further
transformations are applied. Beware that even silenceable errors may leave
the payload IR in a state unsuitable for further transformations. It is the
responsibility of the caller to ensure the following transformations are
robust enough when errors are suppressed. Definite errors reported by nested
transformations abort the sequence regardless of the propagation mode. The
set of modes may be extended in the future, e.g., to collect silenceable
errors and report them after attempting all transformations in the sequence.

The entry block of this operation has a single argument that maps to either
the operand if provided or the top-level container operation of the payload
IR, typically the root operation of the pass interpreting the transform
dialect. Operand omission is only allowed for sequences not contained in
another sequence.

The type of the block argument must match the type of the operand. If the
sequence is a top-level transform (without an operand), it can be used for
matching operations if the specified type within the top-level container
payload IR (including the container op itself). E.g.:

```mlir
transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  // %arg1 is mapped to the top-level container of the payload IR, which is
  // typically a module
}

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.op<\"func.func>\"):
  // %arg1 is mapped to all \"func.func\" ops within and including the
  // top-level container of the payload IR. Nested operations that have the
  // specified op type are not included.
}
```

The body of the sequence terminates with an implicit or explicit
`transform.yield` op. The operands of the terminator are returned as the
results of the sequence op.
"""
function sequence(root=nothing::Union{Nothing, Value}; extra_bindings::Vector{Value}, results_::Vector{IR.Type}, failure_propagation_mode, body::Region, location=Location())
    results = IR.Type[results_..., ]
    operands = Value[extra_bindings..., ]
    owned_regions = Region[body, ]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("failure_propagation_mode", failure_propagation_mode), ]
    !isnothing(root) && push!(operands, root)
    push!(attributes, operandsegmentsizes([(root==nothing) ? 0 : 1length(extra_bindings), ]))
    
    IR.create_operation(
        "transform.sequence", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`split_handle`

Splits `handle` into one or multiple handles, as specified by the number
of results of this operation. `handle` should be mapped to as many payload
ops as there are results. Otherwise, this transform will fail silently by
default. Each result handle is mapped to exactly one payload op. The order
of the payload ops is preserved, i.e., the i-th payload op is mapped to the
i-th result handle.

This operation is useful for ensuring a statically known number of
operations are tracked by the source `handle` and to extract them into
individual handles that can be further manipulated in isolation.

If there are more payload ops than results, the remaining ops are mapped to
the result with index `overflow_result`. If no `overflow_result` is
specified, the transform fails silently.

If there are fewer payload ops than results, the transform fails silently
if `fail_on_payload_too_small` is set to \"true\". Otherwise, it succeeds and
the remaining result handles are not mapped to any op. It also succeeds if
`handle` is empty and `pass_through_empty_handle` is set to \"true\",
regardless of `fail_on_payload_too_small`.
"""
function split_handle(handle::Value; results_::Vector{IR.Type}, pass_through_empty_handle=nothing, fail_on_payload_too_small=nothing, overflow_result=nothing, location=Location())
    results = IR.Type[results_..., ]
    operands = Value[handle, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(pass_through_empty_handle) && push!(attributes, namedattribute("pass_through_empty_handle", pass_through_empty_handle))
    !isnothing(fail_on_payload_too_small) && push!(attributes, namedattribute("fail_on_payload_too_small", fail_on_payload_too_small))
    !isnothing(overflow_result) && push!(attributes, namedattribute("overflow_result", overflow_result))
    
    IR.create_operation(
        "transform.split_handle", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`verify`

This transform verifies the targeted ops. If at least one op fails to
verify, the transform fails definitely.

Note: This op was designed for debugging purposes and should be used like an
assertion. It is intentional that this op produces a definite failure and
not a silenceable one. Correctness of the program should not depend on this
op.

This transform reads the target handle.
"""
function verify(target::Value; location=Location())
    results = IR.Type[]
    operands = Value[target, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "transform.verify", location;
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
function yield(operands_::Vector{Value}; location=Location())
    results = IR.Type[]
    operands = Value[operands_..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "transform.yield", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

import ...IR: IR, NamedAttribute, Value, Location, Block, Region, Attribute, context, IndexType
import ..Dialects: namedattribute, operandsegmentsizes


"""
`apply_patterns_vector_cast_away_vector_leading_one_dim`

Collect a set of leading one dimension removal patterns.

These patterns insert vector.shape_cast to remove leading one dimensions
to expose more canonical forms of read/write/insert/extract operations.
With them, there are more chances that we can cancel out extract-insert
pairs or forward write-read pairs.
"""
function apply_patterns_vector_cast_away_vector_leading_one_dim(; location=Location())
    results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "transform.apply_patterns.vector.cast_away_vector_leading_one_dim", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`apply_patterns_vector_lower_broadcast`

Indicates that vector broadcast operations should be lowered to
finer-grained vector primitives.

This is usally a late step that is run after bufferization as part of the
process of lowering to e.g. LLVM or NVVM.
"""
function apply_patterns_vector_lower_broadcast(; location=Location())
    results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "transform.apply_patterns.vector.lower_broadcast", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`apply_patterns_vector_lower_contraction`

Indicates that vector contraction-like operations should be lowered to
finer-grained vector primitives.

This is usually a late step that is run after bufferization as part of the
process of lowering to e.g. LLVM or NVVM.
"""
function apply_patterns_vector_lower_contraction(; lowering_strategy=nothing, location=Location())
    results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(lowering_strategy) && push!(attributes, namedattribute("lowering_strategy", lowering_strategy))
    
    IR.create_operation(
        "transform.apply_patterns.vector.lower_contraction", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`apply_patterns_vector_lower_gather`

Indicates that vector.gather operations should be lowered to
finer-grained vector primitives.
"""
function apply_patterns_vector_lower_gather(; location=Location())
    results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "transform.apply_patterns.vector.lower_gather", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`apply_patterns_vector_lower_masked_transfers`

Indicates that masked vector.transfer and vector.gather operations should
be lowered to finer-grained vector primitives.

This is usually a late step that is run after bufferization as part of the
process of lowering to e.g. LLVM or NVVM.
"""
function apply_patterns_vector_lower_masked_transfers(; location=Location())
    results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "transform.apply_patterns.vector.lower_masked_transfers", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`apply_patterns_vector_lower_masks`

Indicates that vector.create_mask and vector.constant_mask operations
should be lowered to finer-grained vector primitives.

This is usually a late step that is run after bufferization as part of the
process of lowering to e.g. LLVM or NVVM.
"""
function apply_patterns_vector_lower_masks(; location=Location())
    results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "transform.apply_patterns.vector.lower_masks", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`apply_patterns_vector_lower_multi_reduction`

Indicates that vector multi_reduction-like operations should be lowered to
finer-grained vector primitives.

This is usually a late step that is run after bufferization as part of the
process of lowering to e.g. LLVM or NVVM.
"""
function apply_patterns_vector_lower_multi_reduction(; lowering_strategy=nothing, location=Location())
    results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(lowering_strategy) && push!(attributes, namedattribute("lowering_strategy", lowering_strategy))
    
    IR.create_operation(
        "transform.apply_patterns.vector.lower_multi_reduction", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`apply_patterns_vector_lower_outerproduct`

Indicates that the vector outerproduct operations should be lowered to
finer-grained vector primitives.

This is usually a late step that is run after bufferization as part of the
process of lowering to e.g. LLVM or NVVM.
"""
function apply_patterns_vector_lower_outerproduct(; location=Location())
    results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "transform.apply_patterns.vector.lower_outerproduct", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`apply_patterns_vector_lower_scan`

Indicates that vector.scan operations should be lowered to
finer-grained vector primitives.
"""
function apply_patterns_vector_lower_scan(; location=Location())
    results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "transform.apply_patterns.vector.lower_scan", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`apply_patterns_vector_lower_shape_cast`

Indicates that vector shape_cast operations should be lowered to
finer-grained vector primitives.

This is usually a late step that is run after bufferization as part of the
process of lowering to e.g. LLVM or NVVM.
"""
function apply_patterns_vector_lower_shape_cast(; location=Location())
    results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "transform.apply_patterns.vector.lower_shape_cast", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`apply_patterns_vector_lower_transfer`

Indicates that vector transfer operations should be lowered to finer-grained
vector primitives.

This is usually a late step that is run after bufferization as part of the
process of lowering to e.g. LLVM or NVVM.
"""
function apply_patterns_vector_lower_transfer(; max_transfer_rank=nothing, location=Location())
    results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(max_transfer_rank) && push!(attributes, namedattribute("max_transfer_rank", max_transfer_rank))
    
    IR.create_operation(
        "transform.apply_patterns.vector.lower_transfer", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`apply_patterns_vector_lower_transpose`

Indicates that vector transpose-like operations should be lowered to
finer-grained vector primitives.

This is usually a late step that is run after bufferization as part of the
process of lowering to e.g. LLVM or NVVM.
"""
function apply_patterns_vector_lower_transpose(; lowering_strategy=nothing, avx2_lowering_strategy=nothing, location=Location())
    results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(lowering_strategy) && push!(attributes, namedattribute("lowering_strategy", lowering_strategy))
    !isnothing(avx2_lowering_strategy) && push!(attributes, namedattribute("avx2_lowering_strategy", avx2_lowering_strategy))
    
    IR.create_operation(
        "transform.apply_patterns.vector.lower_transpose", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`apply_patterns_vector_materialize_masks`

Indicates that mask operations should be lowered to fine-grained arithemtic
operations.

This is usually the last step that is run after bufferization as part of the
process of lowering to e.g. LLVM or NVVM.
"""
function apply_patterns_vector_materialize_masks(; location=Location())
    results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "transform.apply_patterns.vector.materialize_masks", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`apply_patterns_vector_rank_reducing_subview_patterns`

Apply opt-in vector transfer permutation patterns that include:
  - TransferReadDropUnitDimsPattern
  - TransferWriteDropUnitDimsPattern

These patterns have the effect of rewriting a vector.transfer with unit
dimensions into a rank-reduced version thanks to subview operations.
This is complemented by shape_cast folding patterns.
"""
function apply_patterns_vector_rank_reducing_subview_patterns(; location=Location())
    results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "transform.apply_patterns.vector.rank_reducing_subview_patterns", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`apply_patterns_vector_split_transfer_full_partial`

Indicates that vector transfer operations should be split to full and
partial parts.

This is usually a late step that is run after bufferization as part of the
process of lowering to e.g. LLVM or NVVM.
"""
function apply_patterns_vector_split_transfer_full_partial(; split_transfer_strategy=nothing, location=Location())
    results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(split_transfer_strategy) && push!(attributes, namedattribute("split_transfer_strategy", split_transfer_strategy))
    
    IR.create_operation(
        "transform.apply_patterns.vector.split_transfer_full_partial", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`apply_patterns_vector_transfer_permutation_patterns`

Apply opt-in vector transfer permutation patterns that include:
  - TransferReadPermutationLowering
  - TransferWritePermutationLowering
  - TransferOpReduceRank
  - TransferWriteNonPermutationLowering

These patterns have the effect of rewriting a vector.transfer with an
arbitrary permutation_map to a vector.transfer with a permutation_map that
is a minor identity followed by a vector.transpose.

In other words, this makes the vector.transfer contiguous on the most minor
dimensions and materializes the permutation_map as a vector.transpose.
"""
function apply_patterns_vector_transfer_permutation_patterns(; location=Location())
    results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "transform.apply_patterns.vector.transfer_permutation_patterns", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`apply_patterns_vector_transfer_to_scf`

Indicates that vector transfer operations should be rewritten with scf.for
loops over finer-grained vector primitives.

This is usually a late step that is run after bufferization as part of the
process of lowering to e.g. LLVM or NVVM.
"""
function apply_patterns_vector_transfer_to_scf(; max_transfer_rank=nothing, full_unroll=nothing, location=Location())
    results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(max_transfer_rank) && push!(attributes, namedattribute("max_transfer_rank", max_transfer_rank))
    !isnothing(full_unroll) && push!(attributes, namedattribute("full_unroll", full_unroll))
    
    IR.create_operation(
        "transform.apply_patterns.vector.transfer_to_scf", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

end # transform
