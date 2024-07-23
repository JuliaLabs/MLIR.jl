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
    _results = IR.Type[result,]
    _operands = Value[dynamic_sizes...,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(copy) && push!(_operands, copy)
    !isnothing(size_hint) && push!(_operands, size_hint)
    push!(
        _attributes,
        operandsegmentsizes([
            length(dynamic_sizes), isnothing(copy) ? 0 : 1, isnothing(size_hint) ? 0 : 1
        ]),
    )
    !isnothing(memory_space) &&
        push!(_attributes, namedattribute("memory_space", memory_space))

    return IR.create_operation(
        "bufferization.alloc_tensor",
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
    _results = IR.Type[output,]
    _operands = Value[input,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "bufferization.clone",
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
`dealloc`

This operation deallocates each of the given memrefs if there is no alias
to that memref in the list of retained memrefs and the corresponding
condition value is set. This condition can be used to indicate and pass on
ownership of memref values (or in other words, the responsibility of
deallocating that memref). If two memrefs alias each other, only one will be
deallocated to avoid double free situations.

The number of variadic `memref` operands (the memrefs to be deallocated)
must equal the number of variadic `condition` operands and correspond to
each other element-wise.

The `memref` operands must be the originally allocated memrefs, however, the
`retained` memref operands may be arbitrary memrefs.

This operation returns a variadic number of `updatedConditions` operands,
one updated condition per retained memref. An updated condition indicates
the ownership of the respective retained memref. It is computed as the
disjunction of all `conditions` operands where the corresponding to
`memrefs` operand aliases with the retained memref. If the retained memref
has no aliases among `memrefs`, the resulting updated condition is \'false\'.
This is because all memrefs that need to be deallocated within one basic
block should be added to the same `bufferization.dealloc` operation at the
end of the block; if no aliasing memref is present, then it does not have to
be deallocated and thus we don\'t need to claim ownership. If the memrefs to
be deallocated are split over multiple dealloc operations (e.g., to avoid
aliasing checks at runtime between the `memref` operands), then the results
have to be manually combined using an `arith.ori` operation and all of them
still require the same list of `retained` memref operands unless the
(potentially empty) set of aliasing memrefs can be determined statically. In
that case, the `updatedCondition` operand can be replaced accordingly (e.g.,
by a canonicalizer).

# Example
```mlir
%0:3 = bufferization.dealloc (%a0, %a1 : memref<2xf32>, memref<4xi32>)
  if (%cond0, %cond1) retain (%r0, %r1, %r2 : memref<?xf32>, memref<f64>,
  memref<2xi32>)
```
Deallocation will be called on `%a0` if `%cond0` is \'true\' and neither
`%r0`, `%r1`, or `%r2` are aliases of `%a0`. `%a1` will be deallocated when
`%cond1` is set to \'true\' and none of `%r0`, %r1`, `%r2`, and `%a0` are
aliases.

Note that this can be an expensive operation if there are many operands that
cannot be optimized away. The runtime cost of this operation (assuming that
nothing is optimized away) is `O(|memrefs|^2+|memrefs|*|retained|)`. The
cost in terms of memory space is `O(|memrefs|+|retained|)`. As a result, it
is recommended to place it carefully in the IR such that most operands can
be optimized away by running the `buffer-deallocation-simplification` pass.
"""
function dealloc(
    memrefs::Vector{Value},
    conditions::Vector{Value},
    retained::Vector{Value};
    updatedConditions=nothing::Union{Nothing,Vector{IR.Type}},
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[memrefs..., conditions..., retained...]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    push!(
        _attributes,
        operandsegmentsizes([length(memrefs), length(conditions), length(retained)]),
    )
    !isnothing(updatedConditions) && push!(_results, updatedConditions...)

    return IR.create_operation(
        "bufferization.dealloc",
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
    _results = IR.Type[]
    _operands = Value[tensor,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "bufferization.dealloc_tensor",
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
`materialize_in_destination`

This op indicates that the data of the `source` tensor is guaranteed to
materialize in `dest`, which can be a tensor or a memref. In case of a
tensor, `source` materializes in the future buffer of `dest` and a the
updated destination tensor is returned. If this is not possible, e.g.,
because the destination tensor is read-only or because its original
contents are still read later, the input IR fails to bufferize. In case of a
memref, `source` materializes in `dest`, which is already a buffer. The op
has no results in that case.

`source`, `dest` and `result` (if present) must have the same shape and
element type. If the op has a result, the types of `result` and `dest` must
match exactly (e.g., including any tensor encodings).

By default, this op bufferizes to a memcpy from the future buffer of the
`source` tensor to the future buffer of the `dest` tensor or to the `dest`
buffer. However, transformations such as \"empty tensor elimination\" may
rewrite IR such that a computation is performed directly in `dest` and no
memcpy is needed.

If `dest` is a buffer, the `writable` attribute must be specified and the
`restrict` keyword can be specified. These attributes have the same meaning
as the respective attributes of `bufferization.to_tensor`.

`writable` indicates that the `dest` buffer is considered writable. It does
not make sense to materialize a computation in a read-only buffer, so
`writable` is required.

`restrict` indicates that there is no `bufferization.to_tensor` op and no
other `bufferization.materialize_in_destination` op with `dest` (or an alias
thereof) and \"restrict\". Only ops with this attribute are considered for
\"empty tensor elimination\". As part of empty tensor elimination, a new
`to_tensor` op with `dest` may be inserted and the `restrict` attribute is
transferred from this op to the new `to_tensor` op. Having \"restrict\" on
this op guarantees that performing empty tensor elimination would not create
invalid IR (i.e., having multiple `to_tensor restrict` with aliasing
buffers).

Note: `writable` could be removed from this op because it must always be set
for memref destinations. This op has that attribute to make clear the
requirements on the `dest` operand in the op assembly format.

Note: If `dest` is a tensor, `tensor.insert_slice` could be used for the
same purpose, but since tensor dialect ops only indicate *what* should be
computed but not *where*, it could fold away, causing the computation to
materialize in a different buffer.
"""
function materialize_in_destination(
    source::Value,
    dest::Value;
    result=nothing::Union{Nothing,IR.Type},
    restrict=nothing,
    writable=nothing,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[source, dest]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(result) && push!(_results, result)
    !isnothing(restrict) && push!(_attributes, namedattribute("restrict", restrict))
    !isnothing(writable) && push!(_attributes, namedattribute("writable", writable))

    return IR.create_operation(
        "bufferization.materialize_in_destination",
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
    _results = IR.Type[memref,]
    _operands = Value[tensor,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(read_only) && push!(_attributes, namedattribute("read_only", read_only))

    return IR.create_operation(
        "bufferization.to_memref",
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

The `restrict` unit attribute (similar to the C `restrict` keyword)
indicates that the produced tensor result is the only way for the tensor
IR to gain access to the `memref` operand (or an alias thereof). E.g.,
there must be no other `to_tensor` op with the same or with an aliasing
`memref` operand.

Note: Only `to_tensor` ops with the `restrict` unit attribute are supported
by One-Shot Bufferize. Other IR is rejected. (To support `to_tensor`
without `restrict`, One-Shot Bufferize would have to analyze memref IR.)
Ops that have incorrect usage of `restrict` may bufferize incorrectly.

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
    memref::Value;
    result=nothing::Union{Nothing,IR.Type},
    restrict=nothing,
    writable=nothing,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[memref,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(result) && push!(_results, result)
    !isnothing(restrict) && push!(_attributes, namedattribute("restrict", restrict))
    !isnothing(writable) && push!(_attributes, namedattribute("writable", writable))

    return IR.create_operation(
        "bufferization.to_tensor",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=(length(_results) == 0 ? nothing : _results),
        result_inference=(length(_results) == 0 ? true : false),
    )
end

end # bufferization
