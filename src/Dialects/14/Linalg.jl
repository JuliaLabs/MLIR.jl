module linalg

import ...IR: NamedAttribute, MLIRType, Value, Location, Block, Region, Attribute, create_operation, context, IndexType
import ..Dialects: namedattribute, operandsegmentsizes
import ...API


"""
`index`

The `linalg.index` operation returns the iteration index of the immediately
enclosing linalg structured operation for the iteration dimension `dim`. The
`dim` attribute specifies the position of the accessed dimension in the
indexing map domain.

# Example

```mlir
#map = affine_map<(i, j) -> (i, j)>
linalg.generic {indexing_maps = [#map, #map],
                iterator_types = [\"parallel\", \"parallel\"]}
  outs(%I, %J : memref<?x?xindex>, memref<?x?xindex>) {
  ^bb0(%arg0 : index, %arg1 : index):
  // Access the outer iteration dimension i
  %i = linalg.index 0 : index
  // Access the inner iteration dimension j
  %j = linalg.index 1 : index
  linalg.yield %i, %j : index, index
}
```

This may lower to IR resembling:

```mlir
%0 = dim %I, %c0 : memref<?x?xindex>
%1 = dim %I, %c1 : memref<?x?xindex>
scf.for %i = %c0 to %0 step %c1 {
  scf.for %j = %c0 to %1 step %c1 {
    store %i, %I[%i, %j] : memref<?x?xindex>
    store %j, %J[%i, %j] : memref<?x?xindex>
  }
}
```
"""
function index(; result=nothing::Union{Nothing, MLIRType}, dim, location=Location())
    results = MLIRType[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("dim", dim), ]
    (result != nothing) && push!(results, result)
    
    create_operation(
        "linalg.index", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`init_tensor`

`linalg.init_tensor` is an operation that materializes a tensor of
a given shape. The shape could be dynamic or static.
"""
function init_tensor(sizes::Vector{Value}; result::MLIRType, static_sizes, location=Location())
    results = MLIRType[result, ]
    operands = Value[sizes..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("static_sizes", static_sizes), ]
    
    create_operation(
        "linalg.init_tensor", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`tiled_loop`

This is a loop-like operation with additional properties. The arguments
also include the input and the output tensors or memrefs and the attributes
to specify the iterator types.

Parsing TiledLoopOp will set all elements of the `iterator_types` attribute
to \"parallel\" type, when it is absent from the custom format.

Tensor-based version:

The body region of the loop contains `extract_slice` operations applied to
every tensor argument of TiledLoopOp.

The body region must contain exactly one block that terminates with
`linalg.yield` with the operands resulting from `insert_slice` operations.

# Example

```mlir
%0 = linalg.tiled_loop (%i) = (%c0) to (%c24) step (%c4)
    ins(%lhs, %rhs : tensor<24x64xi8>, tensor<24x64xi8>)
    outs(%out : tensor<24x64xi8>)
    iterators(\"parallel\")
    distribution(\"block_x\") {
  %lhs_sub = tensor.extract_slice %lhs[%i, 0] [%c4, %c64] [1, 1]
      : tensor<24x64xi8> to tensor<?x?xi8>
  %rhs_sub = tensor.extract_slice %rhs[%i, 0] [%c4, %c64] [1, 1]
      : tensor<24x64xi8> to tensor<?x?xi8>
  %out_sub = tensor.extract_slice %out[%i, 0] [%c4, %c64] [1, 1]
      : tensor<24x64xi8> to tensor<?x?xi8>

  %result_sub = linalg.generic ...

  %result = tensor.insert_slice %result_sub into %out[%i, 0][%c4, %c64][1, 1]
    : tensor<?x?xi8> into tensor<24x64xi8>
  linalg.yield %result : tensor<24x64xi8>
}
```

MemRef-based version:

The body region of the loop contains `subview` operations applied to
every memref argument of TiledLoopOp.

The body region must contain exactly one block that terminates with
`linalg.yield` with no operands.

# Example

```mlir
linalg.tiled_loop (%i) = (%c0) to (%c24) step (%c4)
    ins(%lhs, %rhs : memref<24x64xi8>, memref<24x64xi8>)
    outs(%out : memref<24x64xi8>)
    iterators(\"parallel\")
    distribution(\"block_x\") {
  %lhs_sub = subview %lhs[%i, 0] [%c4, %c64] [1, 1]
      : memref<24x64xi8> to memref<?x?xi8>
  %rhs_sub = subview %rhs[%i, 0] [%c4, %c64] [1, 1]
      : memref<24x64xi8> to memref<?x?xi8>
  %out_sub = subview %out[%i, 0] [%c4, %c64] [1, 1]
      : memref<24x64xi8> to memref<?x?xi8>

  %result_sub = linalg.generic ...
  linalg.yield
}
```
"""
function tiled_loop(lowerBound::Vector{Value}, upperBound::Vector{Value}, step::Vector{Value}, inputs::Vector{Value}, outputs::Vector{Value}; results::Vector{MLIRType}, iterator_types, distribution_types=nothing, region::Region, location=Location())
    results = MLIRType[results..., ]
    operands = Value[lowerBound..., upperBound..., step..., inputs..., outputs..., ]
    owned_regions = Region[region, ]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("iterator_types", iterator_types), ]
    push!(attributes, operandsegmentsizes([length(lowerBound), length(upperBound), length(step), length(inputs), length(outputs), ]))
    (distribution_types != nothing) && push!(attributes, namedattribute("distribution_types", distribution_types))
    
    create_operation(
        "linalg.tiled_loop", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`yield`

`linalg.yield` is a special terminator operation for blocks inside regions
in `linalg` generic ops. It returns values to the immediately enclosing
`linalg` generic op.

# Example

```mlir
linalg.yield %f0, %f1 : f32, f32
```
"""
function yield(values::Vector{Value}; location=Location())
    results = MLIRType[]
    operands = Value[values..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "linalg.yield", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

end # linalg
