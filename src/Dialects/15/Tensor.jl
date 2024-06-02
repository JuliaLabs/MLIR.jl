module tensor

import ...IR: IR, NamedAttribute, Value, Location, Block, Region, Attribute, context, IndexType
import ..Dialects: namedattribute, operandsegmentsizes


"""
`cast`

Convert a tensor from one type to an equivalent type without changing any
data elements. The source and destination types must both be tensor types
with the same element type. If both are ranked, then the rank should be the
same and static dimensions should match. The operation is invalid if
converting to a mismatching constant dimension.

# Example

```mlir
// Convert from unknown rank to rank 2 with unknown dimension sizes.
%2 = tensor.cast %1 : tensor<*xf32> to tensor<?x?xf32>

// Convert to a type with more known dimensions.
%3 = tensor.cast %2 : tensor<?x?xf32> to tensor<4x?xf32>

// Discard static dimension and rank information.
%4 = tensor.cast %3 : tensor<4x?xf32> to tensor<?x?xf32>
%5 = tensor.cast %4 : tensor<?x?xf32> to tensor<*xf32>
```
"""
function cast(source::Value; dest::IR.Type, location=Location())
    results = IR.Type[dest, ]
    operands = Value[source, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "tensor.cast", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`collapse_shape`

The `tensor.collapse_shape` op produces a new tensor with a smaller
rank whose sizes are a reassociation of the original `src`.

A reassociation is defined as a continuous grouping of dimensions and is
represented with an array of I64ArrayAttr attribute.

The verification rule is that the reassociation maps are applied to the
operand tensor with the higher rank to obtain the result tensor with the
smaller rank.

The result tensor type of a reshape can be zero-ranked if the operand
tensor type is statically shaped with all dimensions being unit extent. In
such case the reassociation map is empty.

Examples:

```mlir
// Dimension collapse (i, j) -> i\' and k -> k\'
%b = tensor.collapse_shape %a [[0, 1], [2]]
    : tensor<?x?x?xf32> into tensor<?x?xf32>
```
"""
function collapse_shape(src::Value; result::IR.Type, reassociation, location=Location())
    results = IR.Type[result, ]
    operands = Value[src, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("reassociation", reassociation), ]
    
    IR.create_operation(
        "tensor.collapse_shape", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`dim`

The `tensor.dim` operation takes a tensor and a dimension operand of type
`index`. It returns the size of the requested dimension of the given
tensor. If the dimension index is out of bounds, the behavior is undefined.

The specified tensor type is that of the first operand.

# Example

```mlir
// Always returns 4, can be constant folded:
%c0 = arith.constant 0 : index
%x = tensor.dim %A, %c0 : tensor<4x?xf32>

// Returns the dynamic dimension of %A.
%c1 = arith.constant 1 : index
%y = tensor.dim %A, %c1 : memref<4x?xf32>

// Equivalent generic form:
%x = \"tensor.dim\"(%A, %c0) : (memref<4x?xf32>, index) -> index
%y = \"tensor.dim\"(%A, %c1) : (memref<4x?xf32>, index) -> index
```
"""
function dim(source::Value, index::Value; result=nothing::Union{Nothing, IR.Type}, location=Location())
    results = IR.Type[]
    operands = Value[source, index, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(results, result)
    
    IR.create_operation(
        "tensor.dim", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`expand_shape`

The `tensor.expand_shape` op produces a new tensor with a higher
rank whose sizes are a reassociation of the original `src`.

A reassociation is defined as a continuous grouping of dimensions and is
represented with an array of I64ArrayAttr attribute.

The verification rule is that the reassociation maps are applied to the
result tensor with the higher rank to obtain the operand tensor with the
smaller rank.

The operand tensor type of a reshape can be zero-ranked if the result
tensor type is statically shaped with all dimensions being unit extent. In
such cases the reassociation map is empty.

Examples:

```mlir
// Dimension expansion i -> (i\', j\') and (k) -> (k\')
%b = tensor.expand_shape %a [[0, 1], [2]]
    : tensor<?x?xf32> into tensor<?x?x?xf32>
```
"""
function expand_shape(src::Value; result::IR.Type, reassociation, location=Location())
    results = IR.Type[result, ]
    operands = Value[src, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("reassociation", reassociation), ]
    
    IR.create_operation(
        "tensor.expand_shape", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`extract`

The `tensor.extract` op reads a tensor and returns one
element from it specified by an index list. The output of the op is a
new value with the same type as the elements of the tensor. The
arity of indices must match the rank of the accessed value (i.e., if a
tensor is of rank 3, then 3 indices are required for the extract. The
indices should all be of `index` type.

# Example

```mlir
%4 = tensor.extract %t[%1, %2] : tensor<4x4xi32>
%5 = tensor.extract %rt[%1, %2] : tensor<?x?xi32>
%6 = tensor.extract %ut[%1, %2] : tensor<*xi32>
```
"""
function extract(tensor::Value, indices::Vector{Value}; result::IR.Type, location=Location())
    results = IR.Type[result, ]
    operands = Value[tensor, indices..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "tensor.extract", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`extract_slice`

The \"extract_slice\" operation extract a tensor from another tensor as
specified by the operation\'s offsets, sizes and strides arguments.

The extract_slice operation supports the following arguments:

* source: the \"base\" tensor from which to extract a slice.
* offsets: tensor-rank number of offsets into the \"base\" tensor from which
           to extract the slice.
* sizes: tensor-rank number of sizes which specify the sizes of the result
         tensor type.
* strides: tensor-rank number of strides specifying subsampling in each
           dimension.

The representation based on offsets, sizes and strides support a
partially-static specification via attributes specified through the
`static_offsets`, `static_sizes` and `static_strides` arguments. A special
sentinel value ShapedType::kDynamicSize and
ShapedType::kDynamicStrideOrOffset encodes that the corresponding entry has
a dynamic value.

After buffer allocation, the \"extract_slice\" op is expected to lower into a
memref.subview op.

An extract_slice operation may additionally reduce the rank of the resulting
tensor by removing dimensions that are statically known to be of size 1.
This rank-reduction behavior is not required by the op semantics: this
flexibility allows to progressively drop unit dimensions while lowering
between different flavors of ops on that operate on tensors.

Verification vs Inference in the rank-reduced case:
===================================================
Note that there may be multiple ways to infer a resulting rank-reduced type.
  e.g. 1x6x1 could potentially rank-reduce to either 1x6 or 6x1 2-D shapes.

To disambiguate, the inference helpers `inferCanonicalRankReducedResultType`
only drop the first unit dimensions, in order:
  e.g. 1x6x1 rank-reduced to 2-D will infer the 6x1 2-D shape, but not 1x6.

Verification however has access to result type and does not need to infer.
The verifier calls `isRankReducedType(getSource(), getResult())` to 
determine whether the result type is rank-reduced from the source type.
This computes a so-called rank-reduction mask, consisting of dropped unit 
dims, to map the rank-reduced type to the source type by dropping ones:
  e.g. 1x6 is a rank-reduced version of 1x6x1 by mask {2}
       6x1 is a rank-reduced version of 1x6x1 by mask {0}
       1x2x1x4 is a rank-reduced version of 1x1x2x1x1x4x1 by mask {1, 4, 6}
         (remaining common 1 dimensions are matched eagerly)

# Example

```
// Rank-reducing extract_slice.
%1 = tensor.extract_slice %0[0, 0, 0][1, 16, 4][1, 1, 1] :
  tensor<8x16x4xf32> to tensor<16x4xf32>
%3 = tensor.extract_slice %2[%o0, 4, %o2][1, %sz1, 1][1, %st1, 1] :
  tensor<8x16x4xf32> to tensor<1x?xf32>
```
"""
function extract_slice(source::Value, offsets::Vector{Value}, sizes::Vector{Value}, strides::Vector{Value}; result::IR.Type, static_offsets, static_sizes, static_strides, location=Location())
    results = IR.Type[result, ]
    operands = Value[source, offsets..., sizes..., strides..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("static_offsets", static_offsets), namedattribute("static_sizes", static_sizes), namedattribute("static_strides", static_strides), ]
    push!(attributes, operandsegmentsizes([1, length(offsets), length(sizes), length(strides), ]))
    
    IR.create_operation(
        "tensor.extract_slice", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`from_elements`

Create a N-D tensor from a range of same-type arguments. The number of
provided `elements` should equal to the number of the elements in the
result type. The `elements` correspond to a flattened tensor.

# Example

```mlir
tensor.from_elements %a, %b, %c, %d, %e, %f :  tensor<2x3xindex>
```

will result in a tensor

[[%a, %b, %c]
 [%d, %e, %f]]
"""
function from_elements(elements::Vector{Value}; result::IR.Type, location=Location())
    results = IR.Type[result, ]
    operands = Value[elements..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "tensor.from_elements", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`generate`

This operation creates a dynamically sized tensor with elements of any type.
It expects one index operand per dynamic extent of the result tensor.

The body region defines the tensor\'s elements. It takes index operands as
its region arguments that span the index space. The element at the given
position is yielded with the `yield` operation (see `YieldOp`). There is
no defined ordering to the invocations of the body. It is conceptually
a \"parallel map\" operation.

# Example

```mlir
  %tnsr = tensor.generate %m, %n {
  ^bb0(%i : index, %j : index, %k : index):
    ...
    yield %elem : f32
  } : tensor<?x3x?f32>
```
"""
function generate(dynamicExtents::Vector{Value}; result::IR.Type, body::Region, location=Location())
    results = IR.Type[result, ]
    operands = Value[dynamicExtents..., ]
    owned_regions = Region[body, ]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "tensor.generate", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`insert`

The `tensor.insert` op writes a tensor into a tensor `dest`as specified by
the operation\'s indices.

It returns a copy of `dest` with the proper slice updated with the value
of `scalar`.

The arity of indices must match the rank of the tensor `dest` (i.e., if a
tensor is of rank 3, then 3 indices are required for the extract. The
indices should all be of `index` type.

# Example

```mlir
%4 = tensor.insert %t into %dest[%1, %2] : tensor<4x4xi32>
%5 = tensor.insert %rt into %dest[%1, %2] : tensor<?x?xi32>
%6 = tensor.insert %ut into %dest[%1, %2] : tensor<*xi32>
```
"""
function insert(scalar::Value, dest::Value, indices::Vector{Value}; result::IR.Type, location=Location())
    results = IR.Type[result, ]
    operands = Value[scalar, dest, indices..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "tensor.insert", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`insert_slice`

The \"insert_slice\" operation insert a tensor `source` into another
tensor `dest` as specified by the operation\'s offsets, sizes and strides
arguments.

It returns a copy of `dest` with the proper slice updated with the value
of `source`.

The insert_slice operation supports the following arguments:

* source: the tensor that is inserted.
* dest: the tensor into which the source tensor is inserted.
* offsets: tensor-rank number of offsets into the `dest` tensor into which
           the slice is inserted.
* sizes: tensor-rank number of sizes which specify the sizes of the source
         tensor type.
* strides: tensor-rank number of strides that specify subsampling in each
           dimension.

The representation based on offsets, sizes and strides support a
partially-static specification via attributes specified through the
`static_offsets`, `static_sizes` and `static_strides` arguments. A special
sentinel value ShapedType::kDynamicSize and
ShapedType::kDynamicStrideOrOffset encodes that the corresponding entry has
a dynamic value.

After buffer allocation, the \"insert_slice\" op is expected to lower into a
memref.subview op.

An insert_slice operation may additionally specify insertion into a tensor
of higher rank than the source tensor, along dimensions that are statically
known to be of size 1.
This rank-altering behavior is not required by the op semantics: this
flexibility allows to progressively drop unit dimensions while lowering
between different flavors of ops on that operate on tensors.
The rank-altering behavior of tensor.insert_slice matches the rank-reducing
behavior of tensor.extract_slice.

Verification in the rank-reduced case:
======================================
The same verification discussion and mechanisms apply as for ExtractSliceOp.
Unlike ExtractSliceOp however, there is no need for a specific inference.

# Example

```
// Rank-altering insert_slice.
%1 = tensor.insert_slice %t into %0[0, 0, 0][1, 16, 4][1, 1, 1] :
  tensor<16x4xf32> into tensor<8x16x4xf32>
%3 = tensor.insert_slice %tt into %2[%o0, 4, %o2][1, %sz1, 1][1, %st1, 1] :
  tensor<1x?xf32> into tensor<8x16x4xf32>
```
"""
function insert_slice(source::Value, dest::Value, offsets::Vector{Value}, sizes::Vector{Value}, strides::Vector{Value}; result::IR.Type, static_offsets, static_sizes, static_strides, location=Location())
    results = IR.Type[result, ]
    operands = Value[source, dest, offsets..., sizes..., strides..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("static_offsets", static_offsets), namedattribute("static_sizes", static_sizes), namedattribute("static_strides", static_strides), ]
    push!(attributes, operandsegmentsizes([1, 1, length(offsets), length(sizes), length(strides), ]))
    
    IR.create_operation(
        "tensor.insert_slice", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`pad`

`tensor.pad` is an operation that pads the `source` tensor
with given `low` and `high` padding config.

The PadOp operation supports the following arguments:

* source: the \"base\" tensor on which to pad.
* low: A list contains the padding along the start of each
       dimension, i.e `low`.
* high: A list contains the padding along the end of each
        dimension, i.e. `high`.
* nofold: indicates that the operation should not be folded when source and
          result types are equal.

The result tensor dimensions are `low` + `dim` + `high` along that
dimension. The number of elements of `low` and `high` must match
the rank of the input tensor. They can be either a constant or a
dynamic value.

The region of the `tensor.pad` operation returns the value to use
for the padding. The arguments of the region represent the index
of the source being accessed. There should be as many arguments as
the rank of the `source` tensor. The value `yield`-ed by the
region is used as the value of the view at the given position.

If `nofold` is set, the padding operation will not be folded away even
if the source type and the padded type have the same static shape. This can
be used, e.g., for packing or promotion to faster memory.

Example 1:

```mlir
  %pad_value = ... : f32
  %0 = tensor.pad %0 low[1, 2] high[2, 3] {
  ^bb0(%arg0 : index, %arg1 : index):
    tensor.yield %pad_value : f32
  } : tensor<?x?xf32> to tensor<?x?xf32>
```

Example 2:

```mlir
  %pad_value = ... : f32
  %0 = tensor.pad %arg0 low[2, %arg1, 3, 3] high[3, 3, %arg1, 2] {
  ^bb0(%arg2: index, %arg3: index, %arg4: index, %arg5: index):
      tensor.yield %pad_value : f32
  } : tensor<1x2x2x?xf32> to tensor<6x?x?x?xf32>
```

Example 3:

```mlir
  %pad_value = ... : f32
  %0 = tensor.pad %arg0 low[0, 0] high[%ub0, %ub1] {
  ^bb0(%arg1: index, %arg2: index):
    tensor.yield %pad_value : f32
  } : tensor<2x3xf32> to tensor<?x?xf32>
```

Example 4:

```mlir
  // Force a padded value to be always exist with `nofold`.
  %pad_value = ... : f32
  %0 = tensor.pad %arg0 nofold low[0, 0] high[0, 0] {
  ^bb0(%arg1: index, %arg2: index):
    tensor.yield %pad_value : f32
  } : tensor<2x3xf32> to tensor<2x3xf32>
```
"""
function pad(source::Value, low::Vector{Value}, high::Vector{Value}; result::IR.Type, static_low, static_high, nofold=nothing, region::Region, location=Location())
    results = IR.Type[result, ]
    operands = Value[source, low..., high..., ]
    owned_regions = Region[region, ]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("static_low", static_low), namedattribute("static_high", static_high), ]
    push!(attributes, operandsegmentsizes([1, length(low), length(high), ]))
    !isnothing(nofold) && push!(attributes, namedattribute("nofold", nofold))
    
    IR.create_operation(
        "tensor.pad", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`parallel_insert_slice`

The `parallel_insert_slice` yields a subset tensor value to its parent 
ParallelCombiningOpInterface. These subset tensor values are aggregated to
in some unspecified order into a full tensor value returned by the parent 
parallel iterating op. 
The `parallel_insert_slice` is one such op allowed in the 
ParallelCombiningOpInterface op.

Conflicting writes result in undefined semantics, in that the indices written
to by multiple parallel updates might contain data from any of the updates,
or even a malformed bit pattern.

If an index is updated exactly once, the value contained at that index
in the resulting tensor will be equal to the value at a corresponding index
of a slice that was used for the updated. If an index is not updated at all,
its value will be equal to the one in the original tensor.

This op does not create a new value, which allows maintaining a clean
separation between the subset and full tensor.

Note that we cannot mark this operation as pure (NoSideEffects), even
though it has no side effects, because it will get DCEd during
canonicalization.

The parallel_insert_slice operation supports the following arguments:

* source: the tensor that is inserted.
* dest: the tensor into which the source tensor is inserted.
* offsets: tensor-rank number of offsets into the `dest` tensor into which
           the slice is inserted.
* sizes: tensor-rank number of sizes which specify the sizes of the source
         tensor type.
* strides: tensor-rank number of strides that specify subsampling in each
           dimension.

The representation based on offsets, sizes and strides support a
partially-static specification via attributes specified through the
`static_offsets`, `static_sizes` and `static_strides` arguments. A special
sentinel value ShapedType::kDynamicSize and
ShapedType::kDynamicStrideOrOffset encodes that the corresponding entry has
a dynamic value.

After buffer allocation, the \"parallel_insert_slice\" op is expected to lower
into a memref.subview op.

A parallel_insert_slice operation may additionally specify insertion into a
tensor of higher rank than the source tensor, along dimensions that are 
statically known to be of size 1.
This rank-altering behavior is not required by the op semantics: this
flexibility allows to progressively drop unit dimensions while lowering
between different flavors of ops on that operate on tensors.
The rank-altering behavior of tensor.parallel_insert_slice matches the 
rank-reducing behavior of tensor.insert_slice and tensor.extract_slice.

Verification in the rank-reduced case:
======================================
The same verification discussion and mechanisms apply as for ExtractSliceOp.
Unlike ExtractSliceOp however, there is no need for a specific inference.
"""
function parallel_insert_slice(source::Value, dest::Value, offsets::Vector{Value}, sizes::Vector{Value}, strides::Vector{Value}; static_offsets, static_sizes, static_strides, location=Location())
    results = IR.Type[]
    operands = Value[source, dest, offsets..., sizes..., strides..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("static_offsets", static_offsets), namedattribute("static_sizes", static_sizes), namedattribute("static_strides", static_strides), ]
    push!(attributes, operandsegmentsizes([1, 1, length(offsets), length(sizes), length(strides), ]))
    
    IR.create_operation(
        "tensor.parallel_insert_slice", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`rank`

The `tensor.rank` operation takes a tensor operand and returns its rank.

# Example

```mlir
%0 = tensor.rank %arg0 : tensor<*xf32>
%1 = tensor.rank %arg1 : tensor<?x?xf32>
```
"""
function rank(tensor::Value; result_0=nothing::Union{Nothing, IR.Type}, location=Location())
    results = IR.Type[]
    operands = Value[tensor, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result_0) && push!(results, result_0)
    
    IR.create_operation(
        "tensor.rank", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`reshape`

The `reshape` operation converts a tensor from one type to an equivalent
type with a provided shape. The source and destination types are compatible
if both have the same element type, same number of elements. The following
combinations are possible:

a. Source type is ranked or unranked. Shape argument has static size.
Result type is ranked.

```mlir
// Reshape statically-shaped tensor.
%dst = tensor.reshape %src(%shape)
         : (tensor<4x1xf32>, tensor<1xi32>) -> tensor<4xf32>
%dst0 = tensor.reshape %src(%shape0)
         : (tensor<4x1xf32>, tensor<2xi32>) -> tensor<2x2xf32>
// Flatten unranked tensor.
%dst = tensor.reshape %src(%shape)
         : (tensor<*xf32>, tensor<1xi32>) -> tensor<?xf32>
```

b. Source type is ranked or unranked. Shape argument has dynamic size.
Result type is unranked.

```mlir
// Reshape dynamically-shaped 1D tensor.
%dst = tensor.reshape %src(%shape)
         : (tensor<?xf32>, tensor<?xi32>) -> tensor<*xf32>
// Reshape unranked tensor.
%dst = tensor.reshape %src(%shape)
         : (tensor<*xf32>, tensor<?xi32>) -> tensor<*xf32>
```
"""
function reshape(source::Value, shape::Value; result::IR.Type, location=Location())
    results = IR.Type[result, ]
    operands = Value[source, shape, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "tensor.reshape", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`splat`

Broadcast the operand to all elements of the result tensor. The operand is
required to be of integer/index/float type, and the result tensor must be
statically shaped.

# Example

```mlir
%s = arith.constant 10.1 : f32
%t = tensor.splat %s : tensor<8x16xf32>
```

TODO: This operation is easy to extend to broadcast to dynamically shaped
      tensors:

```mlir
// Broadcasts %s to a 2-d dynamically shaped tensor, with %m, %n binding
// to the sizes of the two dynamic dimensions.
%m = \"foo\"() : () -> (index)
%n = \"bar\"() : () -> (index)
%t = tensor.splat %s [%m, %n] : tensor<?x?xf32>
```
"""
function splat(input::Value; aggregate::IR.Type, location=Location())
    results = IR.Type[aggregate, ]
    operands = Value[input, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "tensor.splat", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`yield`

This operation is used to yield a single value from a within a region. It
is used to create dynamically sized tensors
(see `tensor.generate` and `tensor.pad` ops).
"""
function yield(value::Value; location=Location())
    results = IR.Type[]
    operands = Value[value, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "tensor.yield", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

end # tensor
