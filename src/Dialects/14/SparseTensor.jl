module sparse_tensor

import ...IR: IR, NamedAttribute, Value, value, Location, Block, Region, Attribute, context, IndexType
import ..Dialects: namedattribute, operandsegmentsizes

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
function compress(tensor, indices, values, filled, added, count; location=Location())
    results = IR.Type[]
    operands = Value[value(tensor), value(indices), value(values), value(filled), value(added), value(count), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
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
function convert(source; dest::IR.Type, location=Location())
    results = IR.Type[dest, ]
    operands = Value[value(source), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
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
function expand(tensor; values::IR.Type, filled::IR.Type, added::IR.Type, count::IR.Type, location=Location())
    results = IR.Type[values, filled, added, count, ]
    operands = Value[value(tensor), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "sparse_tensor.expand", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`init`

Materializes an uninitialized sparse tensor with given shape (either static
or dynamic). The operation is provided as an anchor that materializes a
properly typed but uninitialized sparse tensor into the output clause of
a subsequent operation that yields a sparse tensor as the result.

# Example

```mlir
%c = sparse_tensor.init_tensor [%d1, %d2] : tensor<?x?xf32, #SparseMatrix>
%0 = linalg.matmul
  ins(%a, %b: tensor<?x?xf32>, tensor<?x?xf32>)
  outs(%c: tensor<?x?xf32, #SparseMatrix>) -> tensor<?x?xf32, #SparseMatrix>
```
"""
function init(sizes; result::IR.Type, location=Location())
    results = IR.Type[result, ]
    operands = Value[value.(sizes)..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "sparse_tensor.init", location;
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
with an `init` operation, the insertions occur in strict lexicographical
index order, and the final tensor is constructed with a `tensor`
operation that has the `hasInserts` attribute set.

Note that this operation is \"impure\" in the sense that its behavior
is solely defined by side-effects and not SSA values. The semantics
may be refined over time as our sparse abstractions evolve.

```mlir
sparse_tensor.lex_insert %tensor, %indices, %val
  : tensor<1024x1024xf64, #CSR>, memref<?xindex>, f64
```
"""
function lex_insert(tensor, indices, value; location=Location())
    results = IR.Type[]
    operands = Value[value(tensor), value(indices), value(value), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
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
function load(tensor; result=nothing::Union{Nothing, IR.Type}, hasInserts=nothing, location=Location())
    results = IR.Type[]
    operands = Value[value(tensor), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(results, result)
    !isnothing(hasInserts) && push!(attributes, namedattribute("hasInserts", hasInserts))
    
    IR.create_operation(
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
function new(source; result::IR.Type, location=Location())
    results = IR.Type[result, ]
    operands = Value[value(source), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
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
function out(tensor, dest; location=Location())
    results = IR.Type[]
    operands = Value[value(tensor), value(dest), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "sparse_tensor.out", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`release`

Releases the underlying sparse storage format for a tensor that
materialized earlier through a `new` operator, `init` operator, or a
`convert` operator with an annotated tensor type as destination (unless
that convert is folded away since the source and destination types were
identical). This operation should only be called once for any materialized
tensor.  Also, after this operation, any subsequent `memref` querying
operation on the tensor returns undefined results.

Note that this operation is \"impure\" in the sense that its behavior
is solely defined by side-effects and not SSA values. The semantics
may be refined over time as our sparse abstractions evolve.

# Example

```mlir
sparse_tensor.release %tensor : tensor<1024x1024xf64, #CSR>
```
"""
function release(tensor; location=Location())
    results = IR.Type[]
    operands = Value[value(tensor), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "sparse_tensor.release", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
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
function indices(tensor, dim; result::IR.Type, location=Location())
    results = IR.Type[result, ]
    operands = Value[value(tensor), value(dim), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
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
function pointers(tensor, dim; result::IR.Type, location=Location())
    results = IR.Type[result, ]
    operands = Value[value(tensor), value(dim), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
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
function values(tensor; result::IR.Type, location=Location())
    results = IR.Type[result, ]
    operands = Value[value(tensor), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "sparse_tensor.values", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

end # sparse_tensor
