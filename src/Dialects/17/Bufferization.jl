module bufferization

import ...IR:
    IR, NamedAttribute, Value, Location, Block, Region, Attribute, context, IndexType
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

Note: An `alloc_tensor` with a `copy` should also be expressed as an
`alloc_tensor` without `copy`, followed by a `copy_tensor`.
"""
function alloc_tensor(
    dynamic_sizes::Vector{Value},
    copy=nothing::Union{Nothing,Value};
    size_hint=nothing::Union{Nothing,Value},
    result::IR.Type,
    memory_space=nothing,
    location=Location(),
)
    results = IR.Type[result,]
    operands = Value[dynamic_sizes...,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(copy) && push!(operands, copy)
    !isnothing(size_hint) && push!(operands, size_hint)
    push!(
        attributes,
        operandsegmentsizes([
            length(dynamic_sizes), isnothing(copy) ? 0 : 1, isnothing(size_hint) ? 0 : 1
        ]),
    )
    !isnothing(memory_space) &&
        push!(attributes, namedattribute("memory_space", memory_space))

    return IR.create_operation(
        "bufferization.alloc_tensor",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=results,
        result_inference=false,
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
    results = IR.Type[output,]
    operands = Value[input,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return IR.create_operation(
        "bufferization.clone",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=results,
        result_inference=false,
    )
end

"""
`copy_tensor`

Copy the contents of the source tensor into the destination tensor. This
operation is guaranteed to bufferize to a memory copy.
"""
function copy_tensor(
    source::Value, dest::Value; result=nothing::Union{Nothing,IR.Type}, location=Location()
)
    results = IR.Type[]
    operands = Value[source, dest]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(results, result)

    return IR.create_operation(
        "bufferization.copy_tensor",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false),
    )
end

"""
`dealloc`

This operation deallocates each of the given memrefs if there is no alias
to that memref in the list of retained memrefs and the corresponding
condition value is set. This condition can be used to indicate and pass on
ownership of memref values (or in other words, the responsibility of
deallocating that memref). If two memrefs alias each other, only one will be
deallocated to avoid double free situations.

The memrefs to be deallocated must be the originally allocated memrefs,
however, the memrefs to be retained may be arbitrary memrefs.

Returns a list of conditions corresponding to the list of memrefs which
indicates the new ownerships, i.e., if the memref was deallocated the
ownership was dropped (set to \'false\') and otherwise will be the same as the
input condition.

# Example
```mlir
%0:2 = bufferization.dealloc %a0, %a1 if %cond0, %cond1 retain %r0, %r1 :
  memref<2xf32>, memref<4xi32> retain memref<?xf32>, memref<f64>
```
Deallocation will be called on `%a0` if `%cond0` is \'true\' and neither `%r0`
or `%r1` are aliases of `%a0`. `%a1` will be deallocated when `%cond1` is
set to \'true\' and none of `%r0`, %r1` and `%a0` are aliases.
"""
function dealloc(
    memrefs::Vector{Value},
    conditions::Vector{Value},
    retained::Vector{Value};
    updatedConditions=nothing::Union{Nothing,Vector{IR.Type}},
    location=Location(),
)
    results = IR.Type[]
    operands = Value[memrefs..., conditions..., retained...]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    push!(
        attributes,
        operandsegmentsizes([length(memrefs), length(conditions), length(retained)]),
    )
    !isnothing(updatedConditions) && push!(results, updatedConditions...)

    return IR.create_operation(
        "bufferization.dealloc",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false),
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
    operands = Value[tensor,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return IR.create_operation(
        "bufferization.dealloc_tensor",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=results,
        result_inference=false,
    )
end

"""
`to_memref`

An operation that returns the future buffer of a `tensor`.

```mlir
// Result type is memref<4x?xf32, #layout, 0>
%m = bufferization.to_memref %t : memref<4x?xf32, #layout, 0>
```

This operation is a specialized variant of the built-in
`unrealized_conversion_cast` and is used to make sure that the IR stays
valid at any point during the bufferization.

The `read_only` attribute can optionally be set, indicating to the
bufferization that the buffer returned by this op (or an alias created from
the returned buffer) will not be written to.
"""
function to_memref(tensor::Value; memref::IR.Type, read_only=nothing, location=Location())
    results = IR.Type[memref,]
    operands = Value[tensor,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(read_only) && push!(attributes, namedattribute("read_only", read_only))

    return IR.create_operation(
        "bufferization.to_memref",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=results,
        result_inference=false,
    )
end

"""
`to_tensor`

An operation that creates a tensor from a `memref`. The result value is a
tensor whose shape and element type match the memref operand.

The opposite of this op is `to_memref`. Together, these two ops are
useful for source/target materializations when doing type conversions
involving tensors and memrefs.

# Example

```mlir
// Produces a value of tensor<4x?xf32> type.
%t = bufferization.to_tensor %m : memref<4x?xf32, #layout, 0>
```

If the `writable` unit attribute is set, the produced tensor is considered
\"writable\" during bufferization. Otherwise, every OpOperand that bufferizes
to a write to the future buffer of the resulting tensor (or an alias
thereof) will bufferize out-of-place to prevent emitting any writes to
`memref` during bufferization.

If the given memref does not alias with any other memref passed to another
`to_tensor` op, the `restrict` unit attribute can be set. Only such
operations are supported by One-Shot Bufferize. (Otherwise, potential memref
aliasing relationships would have to be captured in One-Shot Bufferize.)

# Example

```
%t = bufferization.to_tensor %m restrict writable : memref<4xf32>

// %t is writable, so the tensor.insert may bufferize in-place in the
// absence of other conflicts.
%r = tensor.insert %f into %t[%idx] : tensor<4xf32>
```

`to_tensor` ops are not bufferized. They are expected to fold away after
bufferization. If there are non-bufferizable ops in the IR and
`allowUnknownOps` is set, they may be part of the resulting IR and not fold
away. However, such IR is no longer bufferizable with One-Shot Bufferize.
"""
function to_tensor(
    memref::Value; result::IR.Type, restrict=nothing, writable=nothing, location=Location()
)
    results = IR.Type[result,]
    operands = Value[memref,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(restrict) && push!(attributes, namedattribute("restrict", restrict))
    !isnothing(writable) && push!(attributes, namedattribute("writable", writable))

    return IR.create_operation(
        "bufferization.to_tensor",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=results,
        result_inference=false,
    )
end

end # bufferization
