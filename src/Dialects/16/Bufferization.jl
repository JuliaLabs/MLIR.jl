module bufferization

import ...IR: IR, NamedAttribute, Value, Location, Block, Region, Attribute, context, IndexType
import ..Dialects: namedattribute, operandsegmentsizes


"""
`alloc_tensor`

`bufferization.alloc_tensor` materializes an uninitialized tensor with a
given shape (dynamic or static). It always bufferizes to a new buffer
allocation of the given shape. The optional `copy` operand specifies the
contents of the tensors. If no `copy` operand is specified, reading from the
result of an `alloc_tensor` op yields an undefined value.

If `copy` is specified, no dynamic sizes should be passed, since they are
the same as the dynamic sizes of the `copy` operand.

`alloc_tensor` is a helper op for bufferization. The operation is provided
as an anchor that marks the beginning of a new tensor SSA use-def chain. It
can be used to control in-place bufferization decisions during One-Shot
Bufferize: The bufferized result of a `bufferization.alloc_tensor` does not
alias with any other buffer, so it can be used to resolve read-after-write
conflicts that would have been introduced by the in-place bufferization of
another op.

The optional `memory_space` attribute specifies the memory space when
bufferizing this op. The memory space is inferred from `copy` if specified.
If neither `copy` nor `memory_space` is specified, the default memory space
is used during bufferization.

The optional `size_hint` operand specifies the number of non-zero elements
for sparse tensors. The value of `size_hint` should be not less than 1 and
not larger than the linear size of the corresponding dense tensor type. If
this requirement is not met, the behavior of the operator is undefined.

Both dense and sparse tensor types are supported. The result of a
`bufferization.alloc_tensor` is a tensor value that can be used like any
other tensor value. In practice, it is often used as the \"out\" operand of
another op. Sparse tensor allocations should always be used in a local
construction operation and never escape the function boundary directly.

# Example

```mlir
%c = bufferization.alloc_tensor(%d1, %d2) : tensor<?x?xf32, #SparseMatrix>
%0 = linalg.matmul
  ins(%a, %b: tensor<?x?xf32, #SparseMatrix>, tensor<?x?xf32, #SparseMatrix>)
  outs(%c: tensor<?x?xf32, #SparseMatrix>) -> tensor<?x?xf32, #SparseMatrix>
return %0 : tensor<?x?xf32, #SparseMatrix>
```

```mlir
%c = bufferization.alloc_tensor(%d1, %d2) size_hint = %noe
  : tensor<?x?xf32, #SparseMatrix>
```
"""
function alloc_tensor(dynamic_sizes::Vector{Value}, copy=nothing::Union{Nothing, Value}; size_hint=nothing::Union{Nothing, Value}, result::IR.Type, memory_space=nothing, location=Location())
    results = IR.Type[result, ]
    operands = Value[dynamic_sizes..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(copy) && push!(operands, copy)
    !isnothing(size_hint) && push!(operands, size_hint)
    push!(attributes, operandsegmentsizes([length(dynamic_sizes), isnothing(copy) ? 0 : 1, isnothing(size_hint) ? 0 : 1, ]))
    !isnothing(memory_space) && push!(attributes, namedattribute("memory_space", memory_space))
    
    IR.create_operation(
        "bufferization.alloc_tensor", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`clone`

Clones the data in the input view into an implicitly defined output view.

Usage:

```mlir
%arg1 = bufferization.clone %arg0 : memref<?xf32> to memref<?xf32>
```

Valid implementations of this operation may alias the input and output
views or create an actual copy. Mutating the source or result
of the clone operation after the clone operation thus leads to undefined
behavior.
"""
function clone(input::Value; output::IR.Type, location=Location())
    results = IR.Type[output, ]
    operands = Value[input, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "bufferization.clone", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`dealloc_tensor`

`bufferization.dealloc_tensor` is a buffer deallocation in tensor land. This
op can be used for manual buffer deallocation. Some bufferizations (such as
One-Shot Bufferize) take care of buffer deallocation, in which case this op
is usually not needed. Details can be found in the documentation of the
respective bufferization passes.

In case of a dense tensor, this op lowers to a `memref.dealloc` op during
bufferization.

In case of a sparse tensor, this op releases the underlying sparse storage
format for a tensor that materialized earlier through a `new` operation, a
`convert` operation with annotated destination tensor type (unless the
convert is folded away), or a `bufferization.alloc_tensor` operation. The
release operation should only be called once for any materialized tensor.
After this operation, any subsequent `memref` querying operation on the
tensor returns undefined results.

# Example

```mlir
bufferization.dealloc_tensor %tensor : tensor<1024x1024xf64, #CSR>
```
"""
function dealloc_tensor(tensor::Value; location=Location())
    results = IR.Type[]
    operands = Value[tensor, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "bufferization.dealloc_tensor", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`to_memref`

Casts a tensor to a memref.

```mlir
// Result type is memref<4x?xf32, #layout, 42>
%12 = bufferization.to_memref %10 : memref<4x?xf32, #layout, 42>
```

Note, that mutating the result of the `to_memref` operation leads to
undefined behavior.

This operation is a specialized variant of the built-in
`unrealized_conversion_cast` and is intended for use in the context of
gradual bufferization.
"""
function to_memref(tensor::Value; memref::IR.Type, location=Location())
    results = IR.Type[memref, ]
    operands = Value[tensor, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "bufferization.to_memref", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`to_tensor`

Create a tensor from a `memref`, making an independent copy of the element
data. The result value is a tensor whose shape and element type match the
memref operand.

The opposite of this op is `to_memref`. Together, these two ops are
useful for source/target materializations when doing type conversions
involving tensors and memrefs.

# Example

```mlir
// Produces a value of tensor<4x?xf32> type.
%12 = bufferization.to_tensor %10 : memref<4x?xf32, #layout, memspace0>
```

If tensor load is used in the bufferization steps, mutating the source
buffer after loading leads to undefined behavior.
"""
function to_tensor(memref::Value; result::IR.Type, location=Location())
    results = IR.Type[result, ]
    operands = Value[memref, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "bufferization.to_tensor", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

end # bufferization
