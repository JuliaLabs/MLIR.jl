module bufferization

import ...IR:
    IR, NamedAttribute, Value, Location, Block, Region, Attribute, context, IndexType
import ..Dialects: namedattribute, operandsegmentsizes

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
`to_memref`

Casts a tensor to a memref.

```mlir
// Result type is tensor<4x?xf32>
%12 = bufferization.to_memref %10 : memref<4x?xf32, #map0, 42>
```

Note, that mutating the result of the to_memref operation leads to
undefined behavior.

This operation is a specialized variant of the built-in
unrealized_conversion_cast and is intended for use in the context of
gradual bufferization.
"""
function to_memref(tensor::Value; memref::IR.Type, location=Location())
    results = IR.Type[memref,]
    operands = Value[tensor,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

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

Create a tensor from a memref, making an independent copy of the element
data. The result value is a tensor whose shape and element type match the
memref operand.

The opposite of this op is to_memref. Together, these two ops are
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
    results = IR.Type[result,]
    operands = Value[memref,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

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
