module sparse_tensor

import ...IR:
    IR, NamedAttribute, Value, Location, Block, Region, Attribute, context, IndexType
import ..Dialects: namedattribute, operandsegmentsizes

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

Due to the possibility of empty regions, i.e. lack of a value for certain
cases, the result of this operation may only feed directly into the output
of the `linalg.generic` operation or into into a custom reduction
`sparse_tensor.reduce` operation that follows in the same region.

Example of isEqual applied to intersecting elements only:

```mlir
%C = bufferization.alloc_tensor...
%0 = linalg.generic #trait
  ins(%A: tensor<?xf64, #SparseVector>,
      %B: tensor<?xf64, #SparseVector>)
  outs(%C: tensor<?xi8, #SparseVector>) {
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
} -> tensor<?xi8, #SparseVector>
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
function binary(
    x::Value,
    y::Value;
    output::IR.Type,
    left_identity=nothing,
    right_identity=nothing,
    overlapRegion::Region,
    leftRegion::Region,
    rightRegion::Region,
    location=Location(),
)
    _results = IR.Type[output,]
    _operands = Value[x, y]
    _owned_regions = Region[overlapRegion, leftRegion, rightRegion]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(left_identity) &&
        push!(_attributes, namedattribute("left_identity", left_identity))
    !isnothing(right_identity) &&
        push!(_attributes, namedattribute("right_identity", right_identity))

    return IR.create_operation(
        "sparse_tensor.binary",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`compress`

Finishes a single access pattern expansion by moving inserted elements
into the sparse storage scheme of the given tensor with the given
level-coordinates.  The arity of `lvlCoords` is one less than the
level-rank of the tensor, with the coordinate of the innermost
level defined through the `added` array.  The `values` and `filled`
arrays are reset in a *sparse* fashion by only iterating over set
elements through an indirection using the `added` array, so that
the operations are kept proportional to the number of nonzeros.
See the `sparse_tensor.expand` operation for more details.

Note that this operation is \"impure\" in the sense that even though
the result is modeled through an SSA value, the insertion is eventually
done \"in place\", and referencing the old SSA value is undefined behavior.

# Example

```mlir
%result = sparse_tensor.compress %values, %filled, %added, %count into %tensor[%i]
  : memref<?xf64>, memref<?xi1>, memref<?xindex>, tensor<4x4xf64, #CSR>
```
"""
function compress(
    values::Value,
    filled::Value,
    added::Value,
    count::Value,
    tensor::Value,
    lvlCoords::Vector{Value};
    result=nothing::Union{Nothing,IR.Type},
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[values, filled, added, count, tensor, lvlCoords...]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(result) && push!(_results, result)

    return IR.create_operation(
        "sparse_tensor.compress",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=(length(_results) == 0 ? nothing : _results),
        result_inference=(length(_results) == 0 ? true : false),
    )
end

"""
`concatenate`

Concatenates a list input tensors and the output tensor with the same
dimension-rank.  The concatenation happens on the specified `dimension`
(0 <= dimension < dimRank).  The resulting `dimension` size is the
sum of all the input sizes for that dimension, while all the other
dimensions should have the same size in the input and output tensors.

Only statically-sized input tensors are accepted, while the output tensor
can be dynamically-sized.

# Example

```mlir
%0 = sparse_tensor.concatenate %1, %2 { dimension = 0 : index }
  : tensor<64x64xf64, #CSR>, tensor<64x64xf64, #CSR> to tensor<128x64xf64, #CSR>
```
"""
function concatenate(inputs::Vector{Value}; result::IR.Type, dimension, location=Location())
    _results = IR.Type[result,]
    _operands = Value[inputs...,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[namedattribute("dimension", dimension),]

    return IR.create_operation(
        "sparse_tensor.concatenate",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
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

Trivial dense-to-dense convert will be removed by canonicalization while
trivial sparse-to-sparse convert will be removed by the sparse codegen. This
is because we use trivial sparse-to-sparse convert to tell bufferization
that the sparse codegen will expand the tensor buffer into sparse tensor
storage.

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
    _results = IR.Type[dest,]
    _operands = Value[source,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "sparse_tensor.convert",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`expand`

Performs an access pattern expansion for the innermost levels of the
given tensor. This operation is useful to implement kernels in which a
sparse tensor appears as output. This technique is known under several
different names and using several alternative implementations,
for example, phase counter [Gustavson72], expanded or switch array
[Pissanetzky84], in phase scan [Duff90], access pattern expansion [Bik96],
and workspaces [Kjolstad19].

The `values` and `filled` arrays must have lengths equal to the
level-size of the innermost level (i.e., as if the innermost level
were *dense*).  The `added` array and `count` are used to store new
level-coordinates when a false value is encountered in the `filled`
array.  All arrays should be allocated before the loop (possibly even
shared between loops in a future optimization) so that their *dense*
initialization can be amortized over many iterations.  Setting and
resetting the dense arrays in the loop nest itself is kept *sparse*
by only iterating over set elements through an indirection using
the added array, so that the operations are kept proportional to
the number of nonzeros.

Note that this operation is \"impure\" in the sense that even though the
results are modeled through SSA values, the operation relies on a proper
side-effecting context that sets and resets the expanded arrays.

# Example

```mlir
%values, %filled, %added, %count = sparse_tensor.expand %tensor
  : tensor<4x4xf64, #CSR> to memref<?xf64>, memref<?xi1>, memref<?xindex>
```
"""
function expand(
    tensor::Value;
    values::IR.Type,
    filled::IR.Type,
    added::IR.Type,
    count::IR.Type,
    location=Location(),
)
    _results = IR.Type[values, filled, added, count]
    _operands = Value[tensor,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "sparse_tensor.expand",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`foreach`

Iterates over stored elements in a tensor (which are typically, but not always,
non-zero for sparse tensors) and executes the block.

`tensor`: the input tensor to iterate over.
`initArgs`: the initial loop argument to carry and update during each iteration.
`order`: an optional permutation affine map that specifies the order in which
the dimensions are visited (e.g., row first or column first). This is only
applicable when the input tensor is a non-annotated dense tensor.

For an input tensor with dim-rank `n`, the block must take `n + 1`
arguments (plus additional loop-carried variables as described below).
The first `n` arguments provide the dimension-coordinates of the element
being visited, and must all have `index` type.  The `(n+1)`-th argument
provides the element\'s value, and must have the tensor\'s element type.

`sparse_tensor.foreach` can also operate on loop-carried variables and returns
the final values after loop termination. The initial values of the variables are
passed as additional SSA operands to the \"sparse_tensor.foreach\" following the n + 1
SSA values mentioned above (n coordinates and 1 value).

The region must terminate with a \"sparse_tensor.yield\" that passes the current
values of all loop-carried variables to the next iteration, or to the
result, if at the last iteration. The number and static types of loop-carried
variables may not change with iterations.

For example:
```mlir
%c0 = arith.constant 0 : i32
%ret = sparse_tensor.foreach in %0 init(%c0): tensor<?x?xi32, #DCSR>, i32 -> i32 do {
 ^bb0(%arg1: index, %arg2: index, %arg3: i32, %iter: i32):
   %sum = arith.add %iter, %arg3
   sparse_tensor.yield %sum
}
```

It is important to note that the generated loop iterates over
elements in their storage order.  However, regardless of the
storage scheme used by the tensor, the block is always given
the dimension-coordinates.

For example:
```mlir
#COL_MAJOR = #sparse_tensor.encoding<{
  dimLevelType = [ \"compressed\", \"compressed\" ],
  dimOrdering = affine_map<(i,j) -> (j,i)>
}>

// foreach on a column-major sparse tensor
sparse_tensor.foreach in %0 : tensor<2x3xf64, #COL_MAJOR> do {
 ^bb0(%row: index, %col: index, %arg3: f64):
    // [%row, %col] -> [0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1]
}

#ROW_MAJOR = #sparse_tensor.encoding<{
  dimLevelType = [ \"compressed\", \"compressed\" ],
}>

// foreach on a row-major sparse tensor
sparse_tensor.foreach in %0 : tensor<2x3xf64, #ROW_MAJOR> do {
 ^bb0(%row: index, %col: index, %arg3: f64):
    // [%row, %col] -> [0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1]
}

// foreach on a row-major dense tensor but visit column first
sparse_tensor.foreach in %0 {order=affine_map<(i,j)->(j,i)>}: tensor<2x3xf64> do {
 ^bb0(%row: index, %col: index, %arg3: f64):
    // [%row, %col] -> [0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1]
}

```
"""
function foreach(
    tensor::Value,
    initArgs::Vector{Value};
    results::Vector{IR.Type},
    order=nothing,
    region::Region,
    location=Location(),
)
    _results = IR.Type[results...,]
    _operands = Value[tensor, initArgs...]
    _owned_regions = Region[region,]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(order) && push!(_attributes, namedattribute("order", order))

    return IR.create_operation(
        "sparse_tensor.foreach",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`storage_specifier_get`

Returns the requested field of the given storage_specifier.

Example of querying the size of the coordinates array for level 0:

```mlir
%0 = sparse_tensor.storage_specifier.get %arg0 crd_mem_sz at 0
     : !sparse_tensor.storage_specifier<#COO>
```
"""
function storage_specifier_get(
    specifier::Value;
    result=nothing::Union{Nothing,IR.Type},
    specifierKind,
    level=nothing,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[specifier,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[namedattribute("specifierKind", specifierKind),]
    !isnothing(result) && push!(_results, result)
    !isnothing(level) && push!(_attributes, namedattribute("level", level))

    return IR.create_operation(
        "sparse_tensor.storage_specifier.get",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=(length(_results) == 0 ? nothing : _results),
        result_inference=(length(_results) == 0 ? true : false),
    )
end

"""
`insert`

Inserts the value into the underlying storage of the tensor at the
given level-coordinates.  The arity of `lvlCoords` must match the
level-rank of the tensor.  This operation can only be applied when
the tensor materializes unintialized from a `bufferization.alloc_tensor`
operation and the final tensor is constructed with a `load` operation
which has the `hasInserts` attribute set.

The level-properties of the sparse tensor type fully describe what
kind of insertion order is allowed.  When all levels have \"unique\"
and \"ordered\" properties, for example, insertions should occur in
strict lexicographical level-coordinate order.  Other properties
define different insertion regimens.  Inserting in a way contrary
to these properties results in undefined behavior.

Note that this operation is \"impure\" in the sense that even though
the result is modeled through an SSA value, the insertion is eventually
done \"in place\", and referencing the old SSA value is undefined behavior.
This operation is scheduled to be unified with the dense counterpart
`tensor.insert` that has pure SSA semantics.

# Example

```mlir
%result = sparse_tensor.insert %val into %tensor[%i,%j] : tensor<1024x1024xf64, #CSR>
```
"""
function insert(
    value::Value,
    tensor::Value,
    lvlCoords::Vector{Value};
    result=nothing::Union{Nothing,IR.Type},
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[value, tensor, lvlCoords...]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(result) && push!(_results, result)

    return IR.create_operation(
        "sparse_tensor.insert",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=(length(_results) == 0 ? nothing : _results),
        result_inference=(length(_results) == 0 ? true : false),
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

Note that this operation is \"impure\" in the sense that even though
the result is modeled through an SSA value, the operation relies on
a proper context of materializing and inserting the tensor value.

Examples:

```mlir
%result = sparse_tensor.load %tensor : tensor<8xf64, #SV>

%1 = sparse_tensor.load %0 hasInserts : tensor<16x32xf32, #CSR>
```
"""
function load(
    tensor::Value;
    result=nothing::Union{Nothing,IR.Type},
    hasInserts=nothing,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[tensor,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(result) && push!(_results, result)
    !isnothing(hasInserts) && push!(_attributes, namedattribute("hasInserts", hasInserts))

    return IR.create_operation(
        "sparse_tensor.load",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=(length(_results) == 0 ? nothing : _results),
        result_inference=(length(_results) == 0 ? true : false),
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

Reading in a symmetric matrix will result in just the lower/upper triangular
part of the matrix (so that only relevant information is stored). Proper
symmetry support for operating on symmetric matrices is still TBD.

# Example

```mlir
sparse_tensor.new %source : !Source to tensor<1024x1024xf64, #CSR>
```
"""
function new(source::Value; result::IR.Type, location=Location())
    _results = IR.Type[result,]
    _operands = Value[source,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "sparse_tensor.new",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`number_of_entries`

Returns the number of entries that are stored in the given sparse tensor.
Note that this is typically the number of nonzero elements in the tensor,
but since explicit zeros may appear in the storage formats, the more
accurate nomenclature is used.

# Example

```mlir
%noe = sparse_tensor.number_of_entries %tensor : tensor<64x64xf64, #CSR>
```
"""
function number_of_entries(
    tensor::Value; result=nothing::Union{Nothing,IR.Type}, location=Location()
)
    _results = IR.Type[]
    _operands = Value[tensor,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(result) && push!(_results, result)

    return IR.create_operation(
        "sparse_tensor.number_of_entries",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=(length(_results) == 0 ? nothing : _results),
        result_inference=(length(_results) == 0 ? true : false),
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

Note that this operation is \"impure\" in the sense that its behavior
is solely defined by side-effects and not SSA values.

# Example

```mlir
sparse_tensor.out %t, %dest : tensor<1024x1024xf64, #CSR>, !Dest
```
"""
function out(tensor::Value, dest::Value; location=Location())
    _results = IR.Type[]
    _operands = Value[tensor, dest]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "sparse_tensor.out",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`pack`

Packs the values and per-level coordinate or postion arrays into a sparse tensor.
The order and types of provided levels must be consistent with the actual storage
layout of the returned sparse tensor described below.

- `values : tensor<? x V>`
  supplies the value for each stored element in the sparse tensor.
- `levels: [tensor<? x iType>, ...]`
  each supplies the sparse tensor coordinates scheme in the sparse tensor for
  the corresponding level as specifed by `sparse_tensor::StorageLayout`.

This operation can be used to materialize a sparse tensor from external
sources; e.g., when passing two numpy arrays from Python.

Disclaimer: This is the user\'s responsibility to provide input that can be
correctly interpreted by the sparse compiler, which does not perform
any sanity test during runtime to verify data integrity.

TODO: The returned tensor is allowed (in principle) to have non-identity
dimOrdering/higherOrdering mappings.  However, the current implementation
does not yet support them.


# Example

```mlir
%values      = arith.constant dense<[ 1.1,   2.2,   3.3 ]> : tensor<3xf64>
%coordinates = arith.constant dense<[[0,0], [1,2], [1,3]]> : tensor<3x2xindex>
%st = sparse_tensor.pack %values, %coordinates
    : tensor<3xf64>, tensor<3x2xindex> to tensor<3x4xf64, #COO>
// yields COO format |1.1, 0.0, 0.0, 0.0|
//     of 3x4 matrix |0.0, 0.0, 2.2, 3.3|
//                   |0.0, 0.0, 0.0, 0.0|
```
"""
function pack(values::Value, levels::Vector{Value}; result::IR.Type, location=Location())
    _results = IR.Type[result,]
    _operands = Value[values, levels...]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "sparse_tensor.pack",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`push_back`

Pushes `value` to the end of the given sparse tensor storage buffer
`inBuffer` as indicated by the value of `curSize` and returns the
new size of the buffer in `newSize` (`newSize = curSize + n`).
The capacity of the buffer is recorded in the memref type of `inBuffer`.
If the current buffer is full, then `inBuffer.realloc` is called before
pushing the data to the buffer. This is similar to std::vector push_back.

The optional input `n` specifies the number of times to repeately push
the value to the back of the tensor. When `n` is a compile-time constant,
its value can\'t be less than 1. If `n` is a runtime value that is less
than 1, the behavior is undefined. Although using input `n` is semantically
equivalent to calling push_back n times, it gives compiler more chances to
to optimize the memory reallocation and the filling of the memory with the
same value.

The `inbounds` attribute tells the compiler that the insertion won\'t go
beyond the current storage buffer. This allows the compiler to not generate
the code for capacity check and reallocation. The typical usage will be for
\"dynamic\" sparse tensors for which a capacity can be set beforehand.

Note that this operation is \"impure\" in the sense that even though
the result is modeled through an SSA value, referencing the memref
through the old SSA value after this operation is undefined behavior.

# Example

```mlir
%buf, %newSize = sparse_tensor.push_back %curSize, %buffer, %val
   : index, memref<?xf64>, f64
```

```mlir
%buf, %newSize = sparse_tensor.push_back inbounds %curSize, %buffer, %val
   : xindex, memref<?xf64>, f64
```

```mlir
%buf, %newSize = sparse_tensor.push_back inbounds %curSize, %buffer, %val, %n
   : xindex, memref<?xf64>, f64
```
"""
function push_back(
    curSize::Value,
    inBuffer::Value,
    value::Value,
    n=nothing::Union{Nothing,Value};
    outBuffer=nothing::Union{Nothing,IR.Type},
    newSize=nothing::Union{Nothing,IR.Type},
    inbounds=nothing,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[curSize, inBuffer, value]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(n) && push!(_operands, n)
    !isnothing(outBuffer) && push!(_results, outBuffer)
    !isnothing(newSize) && push!(_results, newSize)
    !isnothing(inbounds) && push!(_attributes, namedattribute("inbounds", inbounds))

    return IR.create_operation(
        "sparse_tensor.push_back",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=(length(_results) == 0 ? nothing : _results),
        result_inference=(length(_results) == 0 ? true : false),
    )
end

"""
`reduce`

Defines a computation with a `linalg.generic` operation that takes two
operands and an identity value and reduces all stored values down to a
single result based on the computation in the region.

The region must contain exactly one block taking two arguments. The block
must end with a sparse_tensor.yield and the output must match the input
argument types.

Note that this operation is only required for custom reductions beyond
the standard reduction operations (add, sub, or, xor) that can be
sparsified by merely reducing the stored values. More elaborate reduction
operations (mul, and, min, max, etc.) would need to account for implicit
zeros as well. They can still be handled using this custom reduction
operation. The `linalg.generic` `iterator_types` defines which indices
are being reduced. When the associated operands are used in an operation,
a reduction will occur. The use of this explicit `reduce` operation
is not required in most cases.

Example of Matrix->Vector reduction using max(product(x_i), 100):

```mlir
%cf1 = arith.constant 1.0 : f64
%cf100 = arith.constant 100.0 : f64
%C = bufferization.alloc_tensor...
%0 = linalg.generic #trait
   ins(%A: tensor<?x?xf64, #SparseMatrix>)
  outs(%C: tensor<?xf64, #SparseVector>) {
  ^bb0(%a: f64, %c: f64) :
    %result = sparse_tensor.reduce %c, %a, %cf1 : f64 {
        ^bb0(%arg0: f64, %arg1: f64):
          %0 = arith.mulf %arg0, %arg1 : f64
          %cmp = arith.cmpf \"ogt\", %0, %cf100 : f64
          %ret = arith.select %cmp, %cf100, %0 : f64
          sparse_tensor.yield %ret : f64
      }
    linalg.yield %result : f64
} -> tensor<?xf64, #SparseVector>
```
"""
function reduce(
    x::Value,
    y::Value,
    identity::Value;
    output=nothing::Union{Nothing,IR.Type},
    region::Region,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[x, y, identity]
    _owned_regions = Region[region,]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(output) && push!(_results, output)

    return IR.create_operation(
        "sparse_tensor.reduce",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=(length(_results) == 0 ? nothing : _results),
        result_inference=(length(_results) == 0 ? true : false),
    )
end

"""
`select`

Defines an evaluation within a `linalg.generic` operation that takes a single
operand and decides whether or not to keep that operand in the output.

A single region must contain exactly one block taking one argument. The block
must end with a sparse_tensor.yield and the output type must be boolean.

Value threshold is an obvious usage of the select operation. However, by using
`linalg.index`, other useful selection can be achieved, such as selecting the
upper triangle of a matrix.

Example of selecting A >= 4.0:

```mlir
%C = bufferization.alloc_tensor...
%0 = linalg.generic #trait
   ins(%A: tensor<?xf64, #SparseVector>)
  outs(%C: tensor<?xf64, #SparseVector>) {
  ^bb0(%a: f64, %c: f64) :
    %result = sparse_tensor.select %a : f64 {
        ^bb0(%arg0: f64):
          %cf4 = arith.constant 4.0 : f64
          %keep = arith.cmpf \"uge\", %arg0, %cf4 : f64
          sparse_tensor.yield %keep : i1
      }
    linalg.yield %result : f64
} -> tensor<?xf64, #SparseVector>
```

Example of selecting lower triangle of a matrix:

```mlir
%C = bufferization.alloc_tensor...
%1 = linalg.generic #trait
   ins(%A: tensor<?x?xf64, #CSR>)
  outs(%C: tensor<?x?xf64, #CSR>) {
  ^bb0(%a: f64, %c: f64) :
    %row = linalg.index 0 : index
    %col = linalg.index 1 : index
    %result = sparse_tensor.select %a : f64 {
        ^bb0(%arg0: f64):
          %keep = arith.cmpf \"olt\", %col, %row : f64
          sparse_tensor.yield %keep : i1
      }
    linalg.yield %result : f64
} -> tensor<?x?xf64, #CSR>
```
"""
function select(
    x::Value; output=nothing::Union{Nothing,IR.Type}, region::Region, location=Location()
)
    _results = IR.Type[]
    _operands = Value[x,]
    _owned_regions = Region[region,]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(output) && push!(_results, output)

    return IR.create_operation(
        "sparse_tensor.select",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=(length(_results) == 0 ? nothing : _results),
        result_inference=(length(_results) == 0 ? true : false),
    )
end

"""
`storage_specifier_set`

Set the field of the storage specifier to the given input value. Returns
the updated storage_specifier as a new SSA value.

Example of updating the sizes of the coordinates array for level 0:

```mlir
%0 = sparse_tensor.storage_specifier.set %arg0 crd_mem_sz at 0 with %new_sz
   : !sparse_tensor.storage_specifier<#COO>
```
"""
function storage_specifier_set(
    specifier::Value,
    value::Value;
    result=nothing::Union{Nothing,IR.Type},
    specifierKind,
    level=nothing,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[specifier, value]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[namedattribute("specifierKind", specifierKind),]
    !isnothing(result) && push!(_results, result)
    !isnothing(level) && push!(_attributes, namedattribute("level", level))

    return IR.create_operation(
        "sparse_tensor.storage_specifier.set",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=(length(_results) == 0 ? nothing : _results),
        result_inference=(length(_results) == 0 ? true : false),
    )
end

"""
`sort_coo`

Sparse_tensor.sort_coo is similar to sparse_tensor.sort, except that all the
`xs` values and some `ys` values are put in the linear buffer `xy`. The
optional index attribute `nx` provides the number of `xs` values in `xy`.
When `nx` is not explicitly specified, its value is 1. The optional index
attribute `ny` provides the number of `ys` values in `xy`. When `ny` is not
explicitly specified, its value is 0. This instruction supports a more
efficient way to store the COO definition in sparse tensor type.

The buffer xy should have a dimension not less than n * (nx + ny) while the
buffers in `ys` should have a dimension not less than `n`. The behavior of
the operator is undefined if this condition is not met.

# Example

```mlir
sparse_tensor.sort_coo insertion_sort_stable %n, %x { nx = 2 : index}
  : memref<?xindex>
```

```mlir
sparse_tensor.sort hybrid_quick_sort %n, %xy jointly %y1
  { nx = 2 : index, ny = 2 : index}
  : memref<?xi64> jointly memref<?xf32>
```
"""
function sort_coo(
    n::Value,
    xy::Value,
    ys::Vector{Value};
    nx=nothing,
    ny=nothing,
    algorithm,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[n, xy, ys...]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[namedattribute("algorithm", algorithm),]
    !isnothing(nx) && push!(_attributes, namedattribute("nx", nx))
    !isnothing(ny) && push!(_attributes, namedattribute("ny", ny))

    return IR.create_operation(
        "sparse_tensor.sort_coo",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`sort`

Lexicographically sort the first `n` values in `xs` along with the values in
`ys`. Conceptually, the values being sorted are tuples produced by
`zip(zip(xs), zip(ys))`. In particular, values in `ys` needed to be sorted
along with values in `xs`, but values in `ys` don\'t affect the
lexicographical order. The order in which arrays appear in `xs` affects the
sorting result. The operator updates `xs` and `ys` in place with the result
of the sorting.

For example, assume x1=[4, 3], x2=[1, 2], y1=[10, 5], then the output of
\"sort 2, x1, x2 jointly y1\" are x1=[3, 4], x2=[2, 1], y1=[5, 10] while the
output of \"sort 2, x2, x1, jointly y1\" are x2=[1, 2], x1=[4, 3], y1=[10, 5].

Buffers in `xs` needs to have the same integral element type while buffers
in `ys` can have different numeric element types. All buffers in `xs` and
`ys` should have a dimension not less than `n`. The behavior of the operator
is undefined if this condition is not met. The operator requires at least
one buffer in `xs` while `ys` can be empty.

The enum attribute `algorithm` indicates the sorting algorithm used to
implement the operator: hybrid_quick_sort, insertion_sort_stable,
quick_sort, or heap_sort.

Note that this operation is \"impure\" in the sense that its behavior is
solely defined by side-effects and not SSA values.

# Example

```mlir
sparse_tensor.sort insertion_sort_stable %n, %x1, %x2 jointly y1, %y2
  : memref<?xindex>, memref<?xindex> jointly memref<?xindex>, memref<?xf32>
```

```mlir
sparse_tensor.sort hybrid_quick_sort %n, %x1, %x2 jointly y1, %y2
  { alg=1 : index}
  : memref<?xindex>, memref<?xindex> jointly memref<?xindex>, memref<?xf32>
```
"""
function sort(
    n::Value, xs::Vector{Value}, ys::Vector{Value}; algorithm, location=Location()
)
    _results = IR.Type[]
    _operands = Value[n, xs..., ys...]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[namedattribute("algorithm", algorithm),]
    push!(_attributes, operandsegmentsizes([1, length(xs), length(ys)]))

    return IR.create_operation(
        "sparse_tensor.sort",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`storage_specifier_init`

Returns an initial storage specifier value.  A storage specifier
value holds the level-sizes, position arrays, coordinate arrays,
and the value array.
If this is a specifier for slices, it also holds the extra strides/offsets
for each tensor dimension.

TODO: The sparse tensor slice support is currently in a unstable state, and
is subject to change in the future.

# Example

```mlir
#CSR = #sparse_tensor.encoding<{ dimLevelType = [ \"dense\", \"compressed\" ]}>
#CSR_SLICE = #sparse_tensor.encoding<{
  dimLevelType = [ \"dense\", \"compressed\" ],
  slice = [ (1, 4, 1), (1, 4, 2) ]
}>

%0 = sparse_tensor.storage_specifier.init :  !sparse_tensor.storage_specifier<#CSR>
%1 = sparse_tensor.storage_specifier.init with %src
     : !sparse_tensor.storage_specifier<#CSR> to
       !sparse_tensor.storage_specifier<#CSR_SLICE>
```
"""
function storage_specifier_init(
    source=nothing::Union{Nothing,Value}; result::IR.Type, location=Location()
)
    _results = IR.Type[result,]
    _operands = Value[]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(source) && push!(_operands, source)

    return IR.create_operation(
        "sparse_tensor.storage_specifier.init",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`coordinates_buffer`

Returns the linear coordinates array for a sparse tensor with
a trailing COO region with at least two levels.  It is an error
if the tensor doesn\'t contain such a COO region.  This is similar
to the `bufferization.to_memref` operation in the sense that it
provides a bridge between a tensor world view and a bufferized
world view.  Unlike the `bufferization.to_memref` operation,
however, this operation actually lowers into code that extracts
the linear coordinates array from the sparse storage scheme that
stores the coordinates for the COO region as an array of structures.
For example, a 2D COO sparse tensor with two non-zero elements at
coordinates (1, 3) and (4, 6) are stored in a linear buffer as
(1, 4, 3, 6) instead of two buffer as (1, 4) and (3, 6).

Writing into the result of this operation is undefined behavior.

# Example

```mlir
%1 = sparse_tensor.coordinates_buffer %0
   : tensor<64x64xf64, #COO> to memref<?xindex>
```
"""
function coordinates_buffer(tensor::Value; result::IR.Type, location=Location())
    _results = IR.Type[result,]
    _operands = Value[tensor,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "sparse_tensor.coordinates_buffer",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`coordinates`

Returns the coordinates array of the tensor\'s storage at the given
level.  This is similar to the `bufferization.to_memref` operation
in the sense that it provides a bridge between a tensor world view
and a bufferized world view.  Unlike the `bufferization.to_memref`
operation, however, this sparse operation actually lowers into code
that extracts the coordinates array from the sparse storage itself
(either by calling a support library or through direct code).

Writing into the result of this operation is undefined behavior.

# Example

```mlir
%1 = sparse_tensor.coordinates %0 { level = 1 : index }
   : tensor<64x64xf64, #CSR> to memref<?xindex>
```
"""
function coordinates(tensor::Value; result::IR.Type, level, location=Location())
    _results = IR.Type[result,]
    _operands = Value[tensor,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[namedattribute("level", level),]

    return IR.create_operation(
        "sparse_tensor.coordinates",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`positions`

Returns the positions array of the tensor\'s storage at the given
level.  This is similar to the `bufferization.to_memref` operation
in the sense that it provides a bridge between a tensor world view
and a bufferized world view.  Unlike the `bufferization.to_memref`
operation, however, this sparse operation actually lowers into code
that extracts the positions array from the sparse storage itself
(either by calling a support library or through direct code).

Writing into the result of this operation is undefined behavior.

# Example

```mlir
%1 = sparse_tensor.positions %0 { level = 1 : index }
   : tensor<64x64xf64, #CSR> to memref<?xindex>
```
"""
function positions(tensor::Value; result::IR.Type, level, location=Location())
    _results = IR.Type[result,]
    _operands = Value[tensor,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[namedattribute("level", level),]

    return IR.create_operation(
        "sparse_tensor.positions",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`slice_offset`

Extracts the offset of the sparse tensor slice at the given dimension.

Currently, sparse tensor slices are still a work in progress, and only
works when runtime library is disabled (i.e., running sparse compiler
with `enable-runtime-library=false`).

# Example

```mlir
%0 = tensor.extract_slice %s[%v1, %v2][64, 64][1, 1] : tensor<128x128xf64, #DCSR>
                                                    to tensor<64x64xf64, #Slice>

%1 = sparse_tensor.slice.offset %0 at 0 : tensor<64x64xf64, #Slice>
%2 = sparse_tensor.slice.offset %0 at 1 : tensor<64x64xf64, #Slice>
// %1 = %v1
// %2 = %v2
```
"""
function slice_offset(
    slice::Value; offset=nothing::Union{Nothing,IR.Type}, dim, location=Location()
)
    _results = IR.Type[]
    _operands = Value[slice,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[namedattribute("dim", dim),]
    !isnothing(offset) && push!(_results, offset)

    return IR.create_operation(
        "sparse_tensor.slice.offset",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=(length(_results) == 0 ? nothing : _results),
        result_inference=(length(_results) == 0 ? true : false),
    )
end

"""
`slice_stride`

Extracts the stride of the sparse tensor slice at the given dimension.

Currently, sparse tensor slices are still a work in progress, and only
works when runtime library is disabled (i.e., running sparse compiler
with `enable-runtime-library=false`).

# Example

```mlir
%0 = tensor.extract_slice %s[%v1, %v2][64, 64][%s1, %s2] : tensor<128x128xf64, #DCSR>
                                                        to tensor<64x64xf64, #Slice>

%1 = sparse_tensor.slice.stride %0 at 0 : tensor<64x64xf64, #Slice>
%2 = sparse_tensor.slice.stride %0 at 1 : tensor<64x64xf64, #Slice>
// %1 = %s1
// %2 = %s2

```
"""
function slice_stride(
    slice::Value; stride=nothing::Union{Nothing,IR.Type}, dim, location=Location()
)
    _results = IR.Type[]
    _operands = Value[slice,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[namedattribute("dim", dim),]
    !isnothing(stride) && push!(_results, stride)

    return IR.create_operation(
        "sparse_tensor.slice.stride",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=(length(_results) == 0 ? nothing : _results),
        result_inference=(length(_results) == 0 ? true : false),
    )
end

"""
`values`

Returns the values array of the sparse storage format for the given
sparse tensor, independent of the actual dimension. This is similar to
the `bufferization.to_memref` operation in the sense that it provides a bridge
between a tensor world view and a bufferized world view. Unlike the
`bufferization.to_memref` operation, however, this sparse operation actually
lowers into code that extracts the values array from the sparse storage
scheme (either by calling a support library or through direct code).

Writing into the result of this operation is undefined behavior.

# Example

```mlir
%1 = sparse_tensor.values %0 : tensor<64x64xf64, #CSR> to memref<?xf64>
```
"""
function values(tensor::Value; result::IR.Type, location=Location())
    _results = IR.Type[result,]
    _operands = Value[tensor,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "sparse_tensor.values",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
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

Due to the possibility of empty regions, i.e. lack of a value for certain
cases, the result of this operation may only feed directly into the output
of the `linalg.generic` operation or into into a custom reduction
`sparse_tensor.reduce` operation that follows in the same region.

Example of A+1, restricted to existing elements:

```mlir
%C = bufferization.alloc_tensor...
%0 = linalg.generic #trait
   ins(%A: tensor<?xf64, #SparseVector>)
  outs(%C: tensor<?xf64, #SparseVector>) {
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
} -> tensor<?xf64, #SparseVector>
```

Example returning +1 for existing values and -1 for missing values:

```mlir
%C = bufferization.alloc_tensor...
%1 = linalg.generic #trait
   ins(%A: tensor<?xf64, #SparseVector>)
  outs(%C: tensor<?xf64, #SparseVector>) {
  ^bb0(%a: f64, %c: f64) :
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
    linalg.yield %result : f64
} -> tensor<?xf64, #SparseVector>
```

Example showing a structural inversion (existing values become missing in
the output, while missing values are filled with 1):

```mlir
%C = bufferization.alloc_tensor...
%2 = linalg.generic #trait
    ins(%A: tensor<?xf64, #SparseVector>)
    outs(%C: tensor<?xf64, #SparseVector>) {
      %result = sparse_tensor.unary %a : f64 to i64
        present={}
        absent={
          %ret = arith.constant 1 : i64
          sparse_tensor.yield %ret : i64
        }
    linalg.yield %result : f64
} -> tensor<?xf64, #SparseVector>
```
"""
function unary(
    x::Value;
    output::IR.Type,
    presentRegion::Region,
    absentRegion::Region,
    location=Location(),
)
    _results = IR.Type[output,]
    _operands = Value[x,]
    _owned_regions = Region[presentRegion, absentRegion]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "sparse_tensor.unary",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`unpack`

The unpack operation is the inverse of `sparse_tensor::pack`.  It returns
the values and per-level position and coordinate array to the user
from the sparse tensor along with the actual length of the memory used in
each returned buffer. This operation can be used for returning an
unpacked MLIR sparse tensor to frontend; e.g., returning two numpy arrays to Python.

Disclaimer: This is the user\'s responsibility to allocate large enough buffers
to hold the sparse tensor. The sparse compiler simply copies each fields
of the sparse tensor into the user-supplied buffer without bound checking.

TODO: the current implementation does not yet support non-identity mappings.

# Example

```mlir
// input COO format |1.1, 0.0, 0.0, 0.0|
//    of 3x4 matrix |0.0, 0.0, 2.2, 3.3|
//                  |0.0, 0.0, 0.0, 0.0|
%v, %p, %c, %v_len, %p_len, %c_len = sparse_tensor.unpack %sp : tensor<3x4xf64, #SparseVector>
                                     outs(%od, %op, %oi : tensor<3xf64>, tensor<2xindex>, tensor<3x2xindex>)
                                  -> tensor<3xf64>, (tensor<2xindex>, tensor<3x2xindex>), index, (index, index)
// %v = arith.constant dense<[ 1.1,   2.2,   3.3 ]> : tensor<3xf64>
// %p = arith.constant dense<[ 0,              3 ]> : tensor<2xindex>
// %c = arith.constant dense<[[0,0], [1,2], [1,3]]> : tensor<3x2xindex>
// %v_len = 3
// %p_len = 2
// %c_len = 6 (3x2)
```
"""
function unpack(
    tensor::Value,
    out_values::Value,
    out_levels::Vector{Value};
    ret_values::IR.Type,
    ret_levels::Vector{IR.Type},
    val_len::IR.Type,
    lvl_lens::Vector{IR.Type},
    location=Location(),
)
    _results = IR.Type[ret_values, ret_levels..., val_len, lvl_lens...]
    _operands = Value[tensor, out_values, out_levels...]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "sparse_tensor.unpack",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`yield`

Yields a value from within a `binary`, `unary`, `reduce`,
`select` or `foreach` block.

# Example

```mlir
%0 = sparse_tensor.unary %a : i64 to i64 {
  present={
    ^bb0(%arg0: i64):
      %cst = arith.constant 1 : i64
      %ret = arith.addi %arg0, %cst : i64
      sparse_tensor.yield %ret : i64
  }
}
```
"""
function yield(result=nothing::Union{Nothing,Value}; location=Location())
    _results = IR.Type[]
    _operands = Value[]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(result) && push!(_operands, result)

    return IR.create_operation(
        "sparse_tensor.yield",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

end # sparse_tensor
