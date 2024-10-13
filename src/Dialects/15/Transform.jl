module transform

import ...IR: IR, NamedAttribute, Value, value, Location, Block, Region, Attribute, context, IndexType
import ..Dialects: namedattribute, operandsegmentsizes

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
function bufferization_one_shot_bufferize(target; allow_return_allocs=nothing, allow_unknown_ops=nothing, bufferize_function_boundaries=nothing, create_deallocs=nothing, target_is_module=nothing, test_analysis_only=nothing, print_conflicts=nothing, location=Location())
    results = IR.Type[]
    operands = Value[value(target), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(allow_return_allocs) && push!(attributes, namedattribute("allow_return_allocs", allow_return_allocs))
    !isnothing(allow_unknown_ops) && push!(attributes, namedattribute("allow_unknown_ops", allow_unknown_ops))
    !isnothing(bufferize_function_boundaries) && push!(attributes, namedattribute("bufferize_function_boundaries", bufferize_function_boundaries))
    !isnothing(create_deallocs) && push!(attributes, namedattribute("create_deallocs", create_deallocs))
    !isnothing(target_is_module) && push!(attributes, namedattribute("target_is_module", target_is_module))
    !isnothing(test_analysis_only) && push!(attributes, namedattribute("test_analysis_only", test_analysis_only))
    !isnothing(print_conflicts) && push!(attributes, namedattribute("print_conflicts", print_conflicts))
    
    IR.create_operation(
        "transform.bufferization.one_shot_bufferize", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

import ...IR: IR, NamedAttribute, Value, value, Location, Block, Region, Attribute, context, IndexType
import ..Dialects: namedattribute, operandsegmentsizes

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
function structured_decompose(target; transformed::IR.Type, location=Location())
    results = IR.Type[transformed, ]
    operands = Value[value(target), ]
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
`structured_fuse_into_containing_op`
Fuse a producer into a containing operation.
"""
function structured_fuse_into_containing_op(producer_op, containing_op; fused_op::IR.Type, location=Location())
    results = IR.Type[fused_op, ]
    operands = Value[value(producer_op), value(containing_op), ]
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
function structured_fuse(target; transformed::IR.Type, loops::Vector{IR.Type}, tile_sizes=nothing, tile_interchange=nothing, location=Location())
    results = IR.Type[transformed, loops..., ]
    operands = Value[value(target), ]
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
If all the operations referred to by the `target` PDLOperation generalize
properly, the transform succeeds. Otherwise the transform silently fails.
The return handle points to only the subset of successfully produced 
equivalent generic operations, which can be empty or contain the original
ops if they were already in generic form.
"""
function structured_generalize(target; transformed::IR.Type, location=Location())
    results = IR.Type[transformed, ]
    operands = Value[value(target), ]
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
function structured_interchange(target; transformed::IR.Type, iterator_interchange=nothing, location=Location())
    results = IR.Type[transformed, ]
    operands = Value[value(target), ]
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
`structured_match`

Match op with the specified constraints, within the target op.

The following constraints are supported:
  - interface: an optional MatchInterfaceEnum specifying an enum
    representation for an interface to target.
  - ops: an optional StrArrayAttr specifying the concrete name of an op.
    Multiple names can be specified. Matched ops must have one of specified
    names.
  - attribute: an optional Str specifying the name of an attribute that
    matched ops must have.
  
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
function structured_match(target; results_::IR.Type, ops=nothing, interface=nothing, attribute=nothing, location=Location())
    results = IR.Type[results_, ]
    operands = Value[value(target), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(ops) && push!(attributes, namedattribute("ops", ops))
    !isnothing(interface) && push!(attributes, namedattribute("interface", interface))
    !isnothing(attribute) && push!(attributes, namedattribute("attribute", attribute))
    
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
%low, %high = structured.split %target after %split { dimension = 1 }
%tiled_low = structured.tile %low [0, %sz1]
%tiled_high = structured.tile %high [0, %sz2]
%common = merge_handles %tiled_low, %tiled_high

%sz3, %sz4, %split = structured.multitile_size %target
                     { target_size = 42, dimension = 0 }
%sz3r, %sz4r, %splitr = replicate num(%common) %sz3, %sz4, %splitr
structured.split %common after %splitr { dimension = 0 }
// ...
```
"""
function structured_multitile_sizes(target; low_size::IR.Type, high_size::IR.Type, split_point::IR.Type, dimension, target_size, divisor=nothing, location=Location())
    results = IR.Type[low_size, high_size, split_point, ]
    operands = Value[value(target), ]
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
function structured_pad(target; transformed::IR.Type, padding_values=nothing, padding_dimensions=nothing, pack_paddings=nothing, hoist_paddings=nothing, transpose_paddings=nothing, location=Location())
    results = IR.Type[transformed, ]
    operands = Value[value(target), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(padding_values) && push!(attributes, namedattribute("padding_values", padding_values))
    !isnothing(padding_dimensions) && push!(attributes, namedattribute("padding_dimensions", padding_dimensions))
    !isnothing(pack_paddings) && push!(attributes, namedattribute("pack_paddings", pack_paddings))
    !isnothing(hoist_paddings) && push!(attributes, namedattribute("hoist_paddings", hoist_paddings))
    !isnothing(transpose_paddings) && push!(attributes, namedattribute("transpose_paddings", transpose_paddings))
    
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

If the operations referred to by the `target` PDLOperation promote
properly, the transform succeeds. 

When successful, the return handle points to the \$target operation that 
was modified inplace.
"""
function structured_promote(target; transformed::IR.Type, operands_to_promote=nothing, use_full_tile_buffers=nothing, use_full_tiles_by_default=nothing, use_alloca=nothing, alignment=nothing, location=Location())
    results = IR.Type[transformed, ]
    operands = Value[value(target), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(operands_to_promote) && push!(attributes, namedattribute("operands_to_promote", operands_to_promote))
    !isnothing(use_full_tile_buffers) && push!(attributes, namedattribute("use_full_tile_buffers", use_full_tile_buffers))
    !isnothing(use_full_tiles_by_default) && push!(attributes, namedattribute("use_full_tiles_by_default", use_full_tiles_by_default))
    !isnothing(use_alloca) && push!(attributes, namedattribute("use_alloca", use_alloca))
    !isnothing(alignment) && push!(attributes, namedattribute("alignment", alignment))
    
    IR.create_operation(
        "transform.structured.promote", location;
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
function structured_scalarize(target; result::IR.Type, location=Location())
    results = IR.Type[result, ]
    operands = Value[value(target), ]
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
point must be set to `ShapedType::kDynamicSize` and the dynamic size handle
must point to as many value-producing operations as there are structured
operations pointed to by the target handle.

The operation consumes the target handle, but preserves the split point
handle if provided. It produces two new handles pointing to the two parts
of the structured op after splitting, in the same order as the target
operand, with the first handle corresponding to the part with lower
iteration space indices.
"""
function structured_split(target, dynamic_split_point=nothing; first::IR.Type, second::IR.Type, dimension, static_split_point, location=Location())
    results = IR.Type[first, second, ]
    operands = Value[value(target), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("dimension", dimension), namedattribute("static_split_point", static_split_point), ]
    !isnothing(dynamic_split_point) && push!(operands, value(dynamic_split_point))
    
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
  %1 = linalg.init_tensor [4] : tensor<4xf32>
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
 %0 = linalg.init_tensor [16, 32, 64] : tensor<16x32x64xf32>
 %cst = arith.constant 0.000000e+00 : f32
 %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<16x32x64xf32>) ->
    tensor<16x32x64xf32>
 %2 = linalg.init_tensor [64, 4] : tensor<64x4xi1>

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
function structured_split_reduction(target; init_or_alloc_op::IR.Type, fill_op::IR.Type, split_linalg_op::IR.Type, combining_linalg_op::IR.Type, split_factor=nothing, insert_split_dimension=nothing, use_scaling_algorithm=nothing, use_alloc=nothing, location=Location())
    results = IR.Type[init_or_alloc_op, fill_op, split_linalg_op, combining_linalg_op, ]
    operands = Value[value(target), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(split_factor) && push!(attributes, namedattribute("split_factor", split_factor))
    !isnothing(insert_split_dimension) && push!(attributes, namedattribute("insert_split_dimension", insert_split_dimension))
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
`static_sizes` attribute must be set to `ShapedType::kDynamicSize`. Only
the dynamic sizes must be provided in `dynamic_sizes`, i.e., there should
be as many handles as `ShapedType::kDynamicSize` values in the
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
function structured_tile(target, dynamic_sizes; tiled_linalg_op::IR.Type, loops::Vector{IR.Type}, static_sizes=nothing, interchange=nothing, location=Location())
    results = IR.Type[tiled_linalg_op, loops..., ]
    operands = Value[value(target), value.(dynamic_sizes)..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(static_sizes) && push!(attributes, namedattribute("static_sizes", static_sizes))
    !isnothing(interchange) && push!(attributes, namedattribute("interchange", interchange))
    
    IR.create_operation(
        "transform.structured.tile", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`structured_tile_to_foreach_thread_op`

Tile a TilingInterface op to a tiled `scf.foreach_thread`. Tiling is
applied by either specifying `num_threads` or `tile_size`. If `num_threads`
is specified, then the tile size for each dimension `i` is calculated
dynamically via `ceilDiv(dimSize[i], num_threads[i])`.
If non-empty, the `thread_dim_mapping` is added as an attribute to the
resulting `scf.foreach_thread`.
Zero tile sizes indicate that the dimension is not tiled and can be
thought of as tiling by the full size of data.
It is the user\'s responsibility to ensure that `num_threads/tile_sizes` is
a valid tiling specification (i.e. that only tiles parallel dimensions, 
e.g. in the Linalg case).

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

### Example using `num_threads`

```
%0 = pdl_match @match_matmul in %arg1    
%3:2 = transform.structured.tile_to_foreach_thread_op %0 num_threads [10, 20]
```

### Example using `tile_sizes`

```
%0 = pdl_match @match_matmul in %arg1    
%3:2 = transform.structured.tile_to_foreach_thread_op %0 tile_sizes [10, 20, 0]
```
"""
function structured_tile_to_foreach_thread_op(target; foreach_thread_op::IR.Type, tiled_op::IR.Type, num_threads=nothing, tile_sizes=nothing, thread_dim_mapping=nothing, location=Location())
    results = IR.Type[foreach_thread_op, tiled_op, ]
    operands = Value[value(target), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(num_threads) && push!(attributes, namedattribute("num_threads", num_threads))
    !isnothing(tile_sizes) && push!(attributes, namedattribute("tile_sizes", tile_sizes))
    !isnothing(thread_dim_mapping) && push!(attributes, namedattribute("thread_dim_mapping", thread_dim_mapping))
    
    IR.create_operation(
        "transform.structured.tile_to_foreach_thread_op", location;
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

#### Return modes:

This operation produces `definiteFailure` if vectorization fails for any
reason.
The operation always returns the handle to the target op that is expected 
to be isolated from above.
"""
function structured_vectorize(target; transformed::IR.Type, vectorize_padding=nothing, location=Location())
    results = IR.Type[transformed, ]
    operands = Value[value(target), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(vectorize_padding) && push!(attributes, namedattribute("vectorize_padding", vectorize_padding))
    
    IR.create_operation(
        "transform.structured.vectorize", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

import ...IR: IR, NamedAttribute, Value, value, Location, Block, Region, Attribute, context, IndexType
import ..Dialects: namedattribute, operandsegmentsizes

"""
`loop_get_parent_for`

Produces a handle to the n-th (default 1) parent `scf.for` loop for each
Payload IR operation associated with the operand. Fails if such a loop
cannot be found. The list of operations associated with the handle contains
parent operations in the same order as the list associated with the operand,
except for operations that are parents to more than one input which are only
present once.
"""
function loop_get_parent_for(target; parent::IR.Type, num_loops=nothing, location=Location())
    results = IR.Type[parent, ]
    operands = Value[value(target), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(num_loops) && push!(attributes, namedattribute("num_loops", num_loops))
    
    IR.create_operation(
        "transform.loop.get_parent_for", location;
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
function loop_outline(target; transformed::IR.Type, func_name, location=Location())
    results = IR.Type[transformed, ]
    operands = Value[value(target), ]
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
function loop_peel(target; transformed::IR.Type, fail_if_already_divisible=nothing, location=Location())
    results = IR.Type[transformed, ]
    operands = Value[value(target), ]
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
function loop_pipeline(target; transformed::IR.Type, iteration_interval=nothing, read_latency=nothing, location=Location())
    results = IR.Type[transformed, ]
    operands = Value[value(target), ]
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
`loop_unroll`

Unrolls each loop associated with the given handle to have up to the given
number of loop body copies per iteration. If the unroll factor is larger
than the loop trip count, the latter is used as the unroll factor instead.

#### Return modes

This operation ignores non-scf::For ops and drops them in the return.
If all the operations referred to by the `target` PDLOperation unroll
properly, the transform succeeds. Otherwise the transform silently fails.

Does not return handles as the operation may result in the loop being
removed after a full unrolling.
"""
function loop_unroll(target; factor, location=Location())
    results = IR.Type[]
    operands = Value[value(target), ]
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

import ...IR: IR, NamedAttribute, Value, value, Location, Block, Region, Attribute, context, IndexType
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
function alternatives(scope=nothing; results_::Vector{IR.Type}, alternatives::Vector{Region}, location=Location())
    results = IR.Type[results_..., ]
    operands = Value[]
    owned_regions = Region[alternatives..., ]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(scope) && push!(operands, value(scope))
    
    IR.create_operation(
        "transform.alternatives", location;
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
"""
function foreach(target; body::Region, location=Location())
    results = IR.Type[]
    operands = Value[value(target), ]
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
function get_closest_isolated_parent(target; parent::IR.Type, location=Location())
    results = IR.Type[parent, ]
    operands = Value[value(target), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "transform.get_closest_isolated_parent", location;
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
function merge_handles(handles; result::IR.Type, deduplicate=nothing, location=Location())
    results = IR.Type[result, ]
    operands = Value[value.(handles)..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(deduplicate) && push!(attributes, namedattribute("deduplicate", deduplicate))
    
    IR.create_operation(
        "transform.merge_handles", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
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
function pdl_match(root; matched::IR.Type, pattern_name, location=Location())
    results = IR.Type[matched, ]
    operands = Value[value(root), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("pattern_name", pattern_name), ]
    
    IR.create_operation(
        "transform.pdl_match", location;
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
function replicate(pattern, handles; replicated::Vector{IR.Type}, location=Location())
    results = IR.Type[replicated..., ]
    operands = Value[value(pattern), value.(handles)..., ]
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
`sequence`

The transformations indicated by the sequence are applied in order of their
appearance. Each value produced by a transformation within the sequence
corresponds to an operation or a group of operations in the payload IR.
During application, if any transformation in the sequence fails, the entire
sequence fails immediately leaving the payload IR in potentially invalid
state, i.e., this operation offers no transformation rollback capabilities.

The entry block of this operation has a single argument that maps to either
the operand if provided or the top-level container operation of the payload
IR, typically the root operation of the pass interpreting the transform
dialect. Operand omission is only allowed for sequences not contained in
another sequence.
"""
function sequence(root=nothing; results_::Vector{IR.Type}, body::Region, location=Location())
    results = IR.Type[results_..., ]
    operands = Value[]
    owned_regions = Region[body, ]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(root) && push!(operands, value(root))
    
    IR.create_operation(
        "transform.sequence", location;
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

  sequence %arg0 {
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
function with_pdl_patterns(root=nothing; body::Region, location=Location())
    results = IR.Type[]
    operands = Value[]
    owned_regions = Region[body, ]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(root) && push!(operands, value(root))
    
    IR.create_operation(
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
function yield(operands_; location=Location())
    results = IR.Type[]
    operands = Value[value.(operands_)..., ]
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

end # transform
