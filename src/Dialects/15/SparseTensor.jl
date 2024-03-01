module sparse_tensor

import ...IR: IR, NamedAttribute, Value, Location, Block, Region, Attribute, create_operation, context, IndexType
import ..Dialects: namedattribute, operandsegmentsizes
import ...API


"""
`binary`

Defines a computation within a `linalg.generic` operation that takes two
operands and executes one of the regions depending on whether both operands
or either operand is nonzero (i.e. stored explicitly in the sparse storage
format).

Three regions are defined for the operation and must appear in this order:
- overlap (elements present in both sparse tensors)
- left (elements only present in the left sparse tensor)
- right (element only present in the right sparse tensor)

Each region contains a single block describing the computation and result.
Every non-empty block must end with a sparse_tensor.yield and the return
type must match the type of `output`. The primary region\'s block has two
arguments, while the left and right region\'s block has only one argument.

A region may also be declared empty (i.e. `left={}`), indicating that the
region does not contribute to the output. For example, setting both
`left={}` and `right={}` is equivalent to the intersection of the two
inputs as only the overlap region will contribute values to the output.

As a convenience, there is also a special token `identity` which can be
used in place of the left or right region. This token indicates that
the return value is the input value (i.e. func(%x) => return %x).
As a practical example, setting `left=identity` and `right=identity`
would be equivalent to a union operation where non-overlapping values
in the inputs are copied to the output unchanged.

Example of isEqual applied to intersecting elements only:

```mlir
%C = bufferization.alloc_tensor...
%0 = linalg.generic #trait
  ins(%A: tensor<?xf64, #SparseVec>, %B: tensor<?xf64, #SparseVec>)
  outs(%C: tensor<?xi8, #SparseVec>) {
  ^bb0(%a: f64, %b: f64, %c: i8) :
    %result = sparse_tensor.binary %a, %b : f64, f64 to i8
      overlap={
        ^bb0(%arg0: f64, %arg1: f64):
          %cmp = arith.cmpf \"oeq\", %arg0, %arg1 : f64
          %ret_i8 = arith.extui %cmp : i1 to i8
          sparse_tensor.yield %ret_i8 : i8
      }
      left={}
      right={}
    linalg.yield %result : i8
} -> tensor<?xi8, #SparseVec>
```

Example of A+B in upper triangle, A-B in lower triangle:

```mlir
%C = bufferization.alloc_tensor...
%1 = linalg.generic #trait
  ins(%A: tensor<?x?xf64, #CSR>, %B: tensor<?x?xf64, #CSR>
  outs(%C: tensor<?x?xf64, #CSR> {
  ^bb0(%a: f64, %b: f64, %c: f64) :
    %row = linalg.index 0 : index
    %col = linalg.index 1 : index
    %result = sparse_tensor.binary %a, %b : f64, f64 to f64
      overlap={
        ^bb0(%x: f64, %y: f64):
          %cmp = arith.cmpi \"uge\", %col, %row : index
          %upperTriangleResult = arith.addf %x, %y : f64
          %lowerTriangleResult = arith.subf %x, %y : f64
          %ret = arith.select %cmp, %upperTriangleResult, %lowerTriangleResult : f64
          sparse_tensor.yield %ret : f64
      }
      left=identity
      right={
        ^bb0(%y: f64):
          %cmp = arith.cmpi \"uge\", %col, %row : index
          %lowerTriangleResult = arith.negf %y : f64
          %ret = arith.select %cmp, %y, %lowerTriangleResult : f64
          sparse_tensor.yield %ret : f64
      }
    linalg.yield %result : f64
} -> tensor<?x?xf64, #CSR>
```

Example of set difference. Returns a copy of A where its sparse structure
is *not* overlapped by B. The element type of B can be different than A
because we never use its values, only its sparse structure:

```mlir
%C = bufferization.alloc_tensor...
%2 = linalg.generic #trait
  ins(%A: tensor<?x?xf64, #CSR>, %B: tensor<?x?xi32, #CSR>
  outs(%C: tensor<?x?xf64, #CSR> {
  ^bb0(%a: f64, %b: i32, %c: f64) :
    %result = sparse_tensor.binary %a, %b : f64, i32 to f64
      overlap={}
      left=identity
      right={}
    linalg.yield %result : f64
} -> tensor<?x?xf64, #CSR>
```
"""
function binary(x::Value, y::Value; output::IR.Type, left_identity=nothing, right_identity=nothing, overlapRegion::Region, leftRegion::Region, rightRegion::Region, location=Location())
  results = IR.Type[output,]
  operands = Value[x, y,]
  owned_regions = Region[overlapRegion, leftRegion, rightRegion,]
  successors = Block[]
  attributes = NamedAttribute[]
  !isnothing(left_identity) && push!(attributes, namedattribute("left_identity", left_identity))
  !isnothing(right_identity) && push!(attributes, namedattribute("right_identity", right_identity))

  create_operation(
    "sparse_tensor.binary", location;
    operands, owned_regions, successors, attributes,
    results=results,
    result_inference=false
  )
end

"""
`compress`

Finishes a single access pattern expansion by moving inserted elements
into the sparse storage scheme. The values and filled array are reset
in a *sparse* fashion by only iterating over set elements through an
indirection using the added array, so that the operations are kept
proportional to the number of nonzeros. See the \'expand\' operation
for more details.

Note that this operation is \"impure\" in the sense that its behavior
is solely defined by side-effects and not SSA values. The semantics
may be refined over time as our sparse abstractions evolve.

# Example

```mlir
sparse_tensor.compress %0, %1, %values, %filled, %added, %2
    : tensor<4x4xf64, #CSR>, memref<?xindex>, memref<?xf64>,
	  memref<?xi1>, memref<?xindex>, index
```
"""
function compress(tensor::Value, indices::Value, values::Value, filled::Value, added::Value, count::Value; location=Location())
  results = IR.Type[]
  operands = Value[tensor, indices, values, filled, added, count,]
  owned_regions = Region[]
  successors = Block[]
  attributes = NamedAttribute[]

  create_operation(
    "sparse_tensor.compress", location;
    operands, owned_regions, successors, attributes,
    results=results,
    result_inference=false
  )
end

"""
`convert`

Converts one sparse or dense tensor type to another tensor type. The rank
of the source and destination types must match exactly, and the dimension
sizes must either match exactly or relax from a static to a dynamic size.
The sparse encoding of the two types can obviously be completely different.
The name `convert` was preferred over `cast`, since the operation may incur
a non-trivial cost.

When converting between two different sparse tensor types, only explicitly
stored values are moved from one underlying sparse storage format to
the other. When converting from an unannotated dense tensor type to a
sparse tensor type, an explicit test for nonzero values is used. When
converting to an unannotated dense tensor type, implicit zeroes in the
sparse storage format are made explicit. Note that the conversions can have
non-trivial costs associated with them, since they may involve elaborate
data structure transformations. Also, conversions from sparse tensor types
into dense tensor types may be infeasible in terms of storage requirements.

Examples:

```mlir
%0 = sparse_tensor.convert %a : tensor<32x32xf32> to tensor<32x32xf32, #CSR>
%1 = sparse_tensor.convert %a : tensor<32x32xf32> to tensor<?x?xf32, #CSR>
%2 = sparse_tensor.convert %b : tensor<8x8xi32, #CSC> to tensor<8x8xi32, #CSR>
%3 = sparse_tensor.convert %c : tensor<4x8xf64, #CSR> to tensor<4x?xf64, #CSC>

// The following conversion is not allowed (since it would require a
// runtime assertion that the source\'s dimension size is actually 100).
%4 = sparse_tensor.convert %d : tensor<?xf64> to tensor<100xf64, #SV>
```
"""
function convert(source::Value; dest::IR.Type, location=Location())
  results = IR.Type[dest,]
  operands = Value[source,]
  owned_regions = Region[]
  successors = Block[]
  attributes = NamedAttribute[]

  create_operation(
    "sparse_tensor.convert", location;
    operands, owned_regions, successors, attributes,
    results=results,
    result_inference=false
  )
end

"""
`expand`

Performs an access pattern expansion for the innermost dimensions of the
given tensor. This operation is useful to implement kernels in which a
sparse tensor appears as output. This technique is known under several
different names and using several alternative implementations,
for example, phase counter [Gustavson72], expanded or switch array
[Pissanetzky84], in phase scan [Duff90], access pattern expansion [Bik96],
and workspaces [Kjolstad19].

The values and filled array have sizes that suffice for a *dense* innermost
dimension (e.g. a full row for matrices). The added array and count are used
to store new indices when a false value is encountered in the filled array.
All arrays should be allocated before the loop (possibly even shared between
loops in a future optimization) so that their *dense* initialization can be
amortized over many iterations. Setting and resetting the dense arrays in
the loop nest itself is kept *sparse* by only iterating over set elements
through an indirection using the added array, so that the operations are
kept proportional to the number of nonzeros.

Note that this operation is \"impure\" in the sense that its behavior
is solely defined by side-effects and not SSA values. The semantics
may be refined over time as our sparse abstractions evolve.

# Example

```mlir
%values, %filled, %added, %count = sparse_tensor.expand %0
  : tensor<4x4xf64, #CSR> to memref<?xf64>, memref<?xi1>, memref<?xindex>, index
```
"""
function expand(tensor::Value; values::IR.Type, filled::IR.Type, added::IR.Type, count::IR.Type, location=Location())
  results = IR.Type[values, filled, added, count,]
  operands = Value[tensor,]
  owned_regions = Region[]
  successors = Block[]
  attributes = NamedAttribute[]

  create_operation(
    "sparse_tensor.expand", location;
    operands, owned_regions, successors, attributes,
    results=results,
    result_inference=false
  )
end

"""
`lex_insert`

Inserts the given value at given indices into the underlying sparse
storage format of the given tensor with the given indices. This
operation can only be applied when a tensor materializes unintialized
with a `bufferization.alloc_tensor` operation, the insertions occur in
strict lexicographical index order, and the final tensor is constructed
with a `load` operation that has the `hasInserts` attribute set.

Note that this operation is \"impure\" in the sense that its behavior
is solely defined by side-effects and not SSA values. The semantics
may be refined over time as our sparse abstractions evolve.

# Example

```mlir
sparse_tensor.lex_insert %tensor, %indices, %val
  : tensor<1024x1024xf64, #CSR>, memref<?xindex>, memref<f64>
```
"""
function lex_insert(tensor::Value, indices::Value, value::Value; location=Location())
  results = IR.Type[]
  operands = Value[tensor, indices, value,]
  owned_regions = Region[]
  successors = Block[]
  attributes = NamedAttribute[]

  create_operation(
    "sparse_tensor.lex_insert", location;
    operands, owned_regions, successors, attributes,
    results=results,
    result_inference=false
  )
end

"""
`load`

Rematerializes a tensor from the underlying sparse storage format of the
given tensor. This is similar to the `bufferization.to_tensor` operation
in the sense that it provides a bridge between a bufferized world view
and a tensor world view. Unlike the `bufferization.to_tensor` operation,
however, this sparse operation is used only temporarily to maintain a
correctly typed intermediate representation during progressive
bufferization.

The `hasInserts` attribute denote whether insertions to the underlying
sparse storage format may have occurred, in which case the underlying
sparse storage format needs to be finalized. Otherwise, the operation
simply folds away.

Note that this operation is \"impure\" in the sense that its behavior
is solely defined by side-effects and not SSA values. The semantics
may be refined over time as our sparse abstractions evolve.

# Example

```mlir
%1 = sparse_tensor.load %0 : tensor<8xf64, #SV>
```
"""
function load(tensor::Value; result=nothing::Union{Nothing,IR.Type}, hasInserts=nothing, location=Location())
  results = IR.Type[]
  operands = Value[tensor,]
  owned_regions = Region[]
  successors = Block[]
  attributes = NamedAttribute[]
  !isnothing(result) && push!(results, result)
  !isnothing(hasInserts) && push!(attributes, namedattribute("hasInserts", hasInserts))

  create_operation(
    "sparse_tensor.load", location;
    operands, owned_regions, successors, attributes,
    results=(length(results) == 0 ? nothing : results),
    result_inference=(length(results) == 0 ? true : false)
  )
end

"""
`new`

Materializes a sparse tensor with contents taken from an opaque pointer
provided by `source`. For targets that have access to a file system,
for example, this pointer may be a filename (or file) of a sparse
tensor in a particular external storage format. The form of the operation
is kept deliberately very general to allow for alternative implementations
in the future, such as pointers to buffers or runnable initialization
code. The operation is provided as an anchor that materializes a properly
typed sparse tensor with inital contents into a computation.

# Example

```mlir
sparse_tensor.new %source : !Source to tensor<1024x1024xf64, #CSR>
```
"""
function new(source::Value; result::IR.Type, location=Location())
  results = IR.Type[result,]
  operands = Value[source,]
  owned_regions = Region[]
  successors = Block[]
  attributes = NamedAttribute[]

  create_operation(
    "sparse_tensor.new", location;
    operands, owned_regions, successors, attributes,
    results=results,
    result_inference=false
  )
end

"""
`out`

Outputs the contents of a sparse tensor to the destination defined by an
opaque pointer provided by `dest`. For targets that have access to a file
system, for example, this pointer may specify a filename (or file) for output.
The form of the operation is kept deliberately very general to allow for
alternative implementations in the future, such as sending the contents to
a buffer defined by a pointer.

# Example

```mlir
sparse_tensor.out %t, %dest : tensor<1024x1024xf64, #CSR>, !Dest
```
"""
function out(tensor::Value, dest::Value; location=Location())
  results = IR.Type[]
  operands = Value[tensor, dest,]
  owned_regions = Region[]
  successors = Block[]
  attributes = NamedAttribute[]

  create_operation(
    "sparse_tensor.out", location;
    operands, owned_regions, successors, attributes,
    results=results,
    result_inference=false
  )
end

"""
`reduce`

Defines a computation with a `linalg.generic` operation that takes two
operands and an identity value and reduces all values down to a single
result based on the computation in the region.

The region must contain exactly one block taking two arguments. The block
must end with a sparse_tensor.yield and the output must match the input
argument types.

Note that this operation is only required for custom reductions beyond the

standard operations (add, mul, and, or, etc). The `linalg.generic`
`iterator_types` defines which indices are being reduced. When the associated
operands are used in an operation, a reduction will occur. The use of this
explicit `reduce` operation is not required in most cases.

Example of Matrix->Vector reduction using max(product(x_i), 100):

```mlir
%cf1 = arith.constant 1.0 : f64
%cf100 = arith.constant 100.0 : f64
%C = bufferization.alloc_tensor...
%0 = linalg.generic #trait
  ins(%A: tensor<?x?xf64, #SparseMatrix>)
  outs(%C: tensor<?xf64, #SparseVec>) {
  ^bb0(%a: f64, %c: f64) :
    %result = sparse_tensor.reduce %c, %a, %cf1 : f64 {
        ^bb0(%arg0: f64, %arg1: f64):
          %0 = arith.mulf %arg0, %arg1 : f64
          %cmp = arith.cmpf \"ogt\", %0, %cf100 : f64
          %ret = arith.select %cmp, %cf100, %0 : f64
          sparse_tensor.yield %ret : f64
      }
    linalg.yield %result : f64
} -> tensor<?xf64, #SparseVec>
```
"""
function reduce(x::Value, y::Value, identity::Value; output=nothing::Union{Nothing,IR.Type}, region::Region, location=Location())
  results = IR.Type[]
  operands = Value[x, y, identity,]
  owned_regions = Region[region,]
  successors = Block[]
  attributes = NamedAttribute[]
  !isnothing(output) && push!(results, output)

  create_operation(
    "sparse_tensor.reduce", location;
    operands, owned_regions, successors, attributes,
    results=(length(results) == 0 ? nothing : results),
    result_inference=(length(results) == 0 ? true : false)
  )
end

"""
`indices`

Returns the indices array of the sparse storage format at the
given dimension for the given sparse tensor. This is similar to the
`bufferization.to_memref` operation in the sense that it provides a bridge
between a tensor world view and a bufferized world view. Unlike the
`bufferization.to_memref` operation, however, this sparse operation actually
lowers into a call into a support library to obtain access to the
indices array.

# Example

```mlir
%1 = sparse_tensor.indices %0, %c1
   : tensor<64x64xf64, #CSR> to memref<?xindex>
```
"""
function indices(tensor::Value, dim::Value; result::IR.Type, location=Location())
  results = IR.Type[result,]
  operands = Value[tensor, dim,]
  owned_regions = Region[]
  successors = Block[]
  attributes = NamedAttribute[]

  create_operation(
    "sparse_tensor.indices", location;
    operands, owned_regions, successors, attributes,
    results=results,
    result_inference=false
  )
end

"""
`pointers`

Returns the pointers array of the sparse storage format at the
given dimension for the given sparse tensor. This is similar to the
`bufferization.to_memref` operation in the sense that it provides a bridge
between a tensor world view and a bufferized world view. Unlike the
`bufferization.to_memref` operation, however, this sparse operation actually
lowers into a call into a support library to obtain access to the
pointers array.

# Example

```mlir
%1 = sparse_tensor.pointers %0, %c1
   : tensor<64x64xf64, #CSR> to memref<?xindex>
```
"""
function pointers(tensor::Value, dim::Value; result::IR.Type, location=Location())
  results = IR.Type[result,]
  operands = Value[tensor, dim,]
  owned_regions = Region[]
  successors = Block[]
  attributes = NamedAttribute[]

  create_operation(
    "sparse_tensor.pointers", location;
    operands, owned_regions, successors, attributes,
    results=results,
    result_inference=false
  )
end

"""
`values`

Returns the values array of the sparse storage format for the given
sparse tensor, independent of the actual dimension. This is similar to
the `bufferization.to_memref` operation in the sense that it provides a bridge
between a tensor world view and a bufferized world view. Unlike the
`bufferization.to_memref` operation, however, this sparse operation actually
lowers into a call into a support library to obtain access to the
values array.

# Example

```mlir
%1 = sparse_tensor.values %0 : tensor<64x64xf64, #CSR> to memref<?xf64>
```
"""
function values(tensor::Value; result::IR.Type, location=Location())
  results = IR.Type[result,]
  operands = Value[tensor,]
  owned_regions = Region[]
  successors = Block[]
  attributes = NamedAttribute[]

  create_operation(
    "sparse_tensor.values", location;
    operands, owned_regions, successors, attributes,
    results=results,
    result_inference=false
  )
end

"""
`unary`

Defines a computation with a `linalg.generic` operation that takes a single
operand and executes one of two regions depending on whether the operand is
nonzero (i.e. stored explicitly in the sparse storage format).

Two regions are defined for the operation must appear in this order:
- present (elements present in the sparse tensor)
- absent (elements not present in the sparse tensor)

Each region contains a single block describing the computation and result.
A non-empty block must end with a sparse_tensor.yield and the return type
must match the type of `output`. The primary region\'s block has one
argument, while the missing region\'s block has zero arguments.

A region may also be declared empty (i.e. `absent={}`), indicating that the
region does not contribute to the output.

Example of A+1, restricted to existing elements:

```mlir
%C = bufferization.alloc_tensor...
%0 = linalg.generic #trait
  ins(%A: tensor<?xf64, #SparseVec>)
  outs(%C: tensor<?xf64, #SparseVec>) {
  ^bb0(%a: f64, %c: f64) :
    %result = sparse_tensor.unary %a : f64 to f64
      present={
        ^bb0(%arg0: f64):
          %cf1 = arith.constant 1.0 : f64
          %ret = arith.addf %arg0, %cf1 : f64
          sparse_tensor.yield %ret : f64
      }
      absent={}
    linalg.yield %result : f64
} -> tensor<?xf64, #SparseVec>
```

Example returning +1 for existing values and -1 for missing values:
```mlir
%result = sparse_tensor.unary %a : f64 to i32
  present={
    ^bb0(%x: f64):
      %ret = arith.constant 1 : i32
      sparse_tensor.yield %ret : i32
  }
  absent={
    %ret = arith.constant -1 : i32
    sparse_tensor.yield %ret : i32
  }
```

Example showing a structural inversion (existing values become missing in
the output, while missing values are filled with 1):
```mlir
%result = sparse_tensor.unary %a : f64 to i64
  present={}
  absent={
    %ret = arith.constant 1 : i64
    sparse_tensor.yield %ret : i64
  }
```
"""
function unary(x::Value; output::IR.Type, presentRegion::Region, absentRegion::Region, location=Location())
  results = IR.Type[output,]
  operands = Value[x,]
  owned_regions = Region[presentRegion, absentRegion,]
  successors = Block[]
  attributes = NamedAttribute[]

  create_operation(
    "sparse_tensor.unary", location;
    operands, owned_regions, successors, attributes,
    results=results,
    result_inference=false
  )
end

"""
`yield`

Yields a value from within a `binary` or `unary` block.

# Example

```
%0 = sparse_tensor.unary %a : i64 to i64 {
  ^bb0(%arg0: i64):
    %cst = arith.constant 1 : i64
    %ret = arith.addi %arg0, %cst : i64
    sparse_tensor.yield %ret : i64
}
```
"""
function yield(result::Value; location=Location())
  results = IR.Type[]
  operands = Value[result,]
  owned_regions = Region[]
  successors = Block[]
  attributes = NamedAttribute[]

  create_operation(
    "sparse_tensor.yield", location;
    operands, owned_regions, successors, attributes,
    results=results,
    result_inference=false
  )
end

end # sparse_tensor
