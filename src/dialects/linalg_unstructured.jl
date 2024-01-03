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
`softmax`

linalg.softmax computes a numerically stable version of softmax.

For a given input tensor and a specified dimension `d`, compute:
  1. the max `m` along that dimension `d`
  2. f(x) = exp(x - m)
  3. sum f(x) along dimension d to get l(x).
  4. compute the final result f(x) / l(x).

This is an aggregate linalg operation that further reduces to a small DAG of
structured operations.

Warning: Regarding the tiling capabilities, the implementation doesn\'t
check that the provided dimensions make sense. This is the responsability
of the transformation calling the tiling to ensure that the provided
sizes for each dimension make sense with respect to the semantic of
softmax.
"""
function softmax(input::Value, output::Value; result::Vector{MLIRType}, dimension, location=Location())
    results = MLIRType[result..., ]
    operands = Value[input, output, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("dimension", dimension), ]
    
    create_operation(
        "linalg.softmax", location;
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
