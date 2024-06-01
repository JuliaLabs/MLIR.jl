module irdl

import ...IR:
    IR, NamedAttribute, Value, Location, Block, Region, Attribute, context, IndexType
import ..Dialects: namedattribute, operandsegmentsizes
import ...API

"""
`all_of`

`irdl.all_of` defines a constraint that accepts any type or attribute that
satisfies all of its provided constraints.

# Example

```mlir
irdl.dialect cmath {
  irdl.type complex_f32 {
    %0 = irdl.is i32
    %1 = irdl.is f32
    %2 = irdl.any_of(%0, %1) // is 32-bit

    %3 = irdl.is f32
    %4 = irdl.is f64
    %5 = irdl.any_of(%3, %4) // is a float

    %6 = irdl.all_of(%2, %5) // is a 32-bit float
    irdl.parameters(%6)
  }
}
```

The above program defines a type `complex` inside the dialect `cmath` that
can has one parameter that must be 32-bit long and a float (in other
words, that must be `f32`).
"""
function all_of(
    args::Vector{Value}; output=nothing::Union{Nothing,IR.Type}, location=Location()
)
    results = IR.Type[]
    operands = Value[args...,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(output) && push!(results, output)

    return IR.create_operation(
        "irdl.all_of",
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
`any_of`

`irdl.any_of` defines a constraint that accepts any type or attribute that
satisfies at least one of its provided type constraints.

# Example

```mlir
irdl.dialect cmath {
  irdl.type complex {
    %0 = irdl.is i32
    %1 = irdl.is i64
    %2 = irdl.is f32
    %3 = irdl.is f64
    %4 = irdl.any_of(%0, %1, %2, %3)
    irdl.parameters(%4)
  }
}
```

The above program defines a type `complex` inside the dialect `cmath` that
can have a single type parameter that can be either `i32`, `i64`, `f32` or
`f32`.
"""
function any_of(
    args::Vector{Value}; output=nothing::Union{Nothing,IR.Type}, location=Location()
)
    results = IR.Type[]
    operands = Value[args...,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(output) && push!(results, output)

    return IR.create_operation(
        "irdl.any_of",
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
`any`

`irdl.any` defines a constraint that accepts any type or attribute.

# Example

```mlir
irdl.dialect @cmath {
  irdl.type @complex_flexible {
    %0 = irdl.any
    irdl.parameters(%0)
  }
}
```

The above program defines a type `complex_flexible` inside the dialect
`cmath` that has a single parameter that can be any attribute.
"""
function any(; output=nothing::Union{Nothing,IR.Type}, location=Location())
    results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(output) && push!(results, output)

    return IR.create_operation(
        "irdl.any",
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
`attribute`

`irdl.attribute` defines a new attribute belonging to the `irdl.dialect`
parent.

The attribute parameters can be defined with an `irdl.parameters` operation
in the optional region.

# Example

```mlir
irdl.dialect @testd {
  irdl.attribute @enum_attr {
    %0 = irdl.is \"foo\"
    %1 = irdl.is \"bar\"
    %2 = irdl.any_of(%0, %1)
    irdl.parameters(%2)
  }
}
```

The above program defines an `enum_attr` attribute inside the `testd`
dialect. The attribute has one `StringAttr` parameter that should be
either a `\"foo\"` or a `\"bar\"`.
"""
function attribute(; sym_name, body::Region, location=Location())
    results = IR.Type[]
    operands = Value[]
    owned_regions = Region[body,]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("sym_name", sym_name),]

    return IR.create_operation(
        "irdl.attribute",
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
`attributes`

`irdl.attributes` defines the attributes of the `irdl.operation` parent 
operation definition.

In the following example, `irdl.attributes` defines the attributes of the
`attr_op` operation:

```mlir
irdl.dialect @example {

  irdl.operation @attr_op {
    %0 = irdl.any
    %1 = irdl.is i64
    irdl.attibutes {
      \"attr1\" = %0,
      \"attr2\" = %1
    }
  }
}
```

The operation will expect an arbitrary attribute \"attr1\" and an
attribute \"attr2\" with value `i64`.
"""
function attributes(
    attributeValues::Vector{Value}; attributeValueNames, location=Location()
)
    results = IR.Type[]
    operands = Value[attributeValues...,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("attributeValueNames", attributeValueNames),]

    return IR.create_operation(
        "irdl.attributes",
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
`c_pred`

`irdl.c_pred` defines a constraint that is written in C++.

Dialects using this operation cannot be registered at runtime, as it relies
on C++ code.

Special placeholders can be used to refer to entities in the context where
this predicate is used. They serve as \"hooks\" to the enclosing environment.
The following special placeholders are supported in constraints for an op:

* `\$_builder` will be replaced by a mlir::Builder instance.
* `\$_op` will be replaced by the current operation.
* `\$_self` will be replaced with the entity this predicate is attached to.
   Compared to ODS, `\$_self` is always of type `mlir::Attribute`, and types
   are manipulated as `TypeAttr` attributes.

# Example
```mlir
irdl.type @op_with_attr {
  %0 = irdl.c_pred \"::llvm::isa<::mlir::IntegerAttr>(\$_self)\"
  irdl.parameters(%0)
}
```

In this example, @op_with_attr is defined as a type with a single
parameter, which is an `IntegerAttr`, as constrained by the C++ predicate.
"""
function c_pred(; output=nothing::Union{Nothing,IR.Type}, pred, location=Location())
    results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("pred", pred),]
    !isnothing(output) && push!(results, output)

    return IR.create_operation(
        "irdl.c_pred",
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
`dialect`

The `irdl.dialect` operation defines a dialect. All operations, attributes,
and types defined inside its region will be part of the dialect.

# Example

```mlir
irdl.dialect @cmath {
  ...
}
```

The above program defines a `cmath` dialect.
"""
function dialect(; sym_name, body::Region, location=Location())
    results = IR.Type[]
    operands = Value[]
    owned_regions = Region[body,]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("sym_name", sym_name),]

    return IR.create_operation(
        "irdl.dialect",
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
`is`

`irdl.is` defines a constraint that only accepts a specific instance of a
type or attribute.

# Example

```mlir
irdl.dialect @cmath {
  irdl.type @complex_i32 {
    %0 = irdl.is i32
    irdl.parameters(%0)
  }
}
```

The above program defines a `complex_i32` type inside the dialect `cmath`
that can only have a `i32` as its parameter.
"""
function is(; output=nothing::Union{Nothing,IR.Type}, expected, location=Location())
    results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("expected", expected),]
    !isnothing(output) && push!(results, output)

    return IR.create_operation(
        "irdl.is",
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
`operands`

`irdl.operands` define the operands of the `irdl.operation` parent operation
definition.

In the following example, `irdl.operands` defines the operands of the
`norm` operation:

```mlir
irdl.dialect @cmath {

  irdl.type @complex { /* ... */ }

  irdl.operation @mul {
    %0 = irdl.any
    %1 = irdl.parametric @complex<%0>
    irdl.results(%1)
    irdl.operands(%1, %1)
  }
}
```

The `mul` operation will expect two operands of type `cmath.complex`, that
have the same type, and return a result of the same type.

The operands can also be marked as variadic or optional:
```mlir
irdl.operands(%0, single %1, optional %2, variadic %3)
```

Here, %0 and %1 are required single operands, %2 is an optional operand,
and %3 is a variadic operand.

When more than one operand is marked as optional or variadic, the operation
will expect a \'operandSegmentSizes\' attribute that defines the number of
operands in each segment.
"""
function operands(args::Vector{Value}; location=Location())
    results = IR.Type[]
    operands = Value[args...,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return IR.create_operation(
        "irdl.operands",
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
`operation`

`irdl.operation` defines a new operation belonging to the `irdl.dialect`
parent.

Operations can define constraints on their operands and results with the
`irdl.results` and `irdl.operands` operations. If these operations are not
present in the region, the results or operands are expected to be empty.

# Example

```mlir
irdl.dialect @cmath {

  irdl.type @complex { /* ... */ }

  irdl.operation @norm {
    %0 = irdl.any
    %1 = irdl.parametric @complex<%0>
    irdl.results(%0)
    irdl.operands(%1)
  }
}
```

The above program defines an operation `norm` inside the dialect `cmath`.
The operation expects a single operand of base type `cmath.complex`, and
returns a single result of the element type of the operand.
"""
function operation(; sym_name, body::Region, location=Location())
    results = IR.Type[]
    operands = Value[]
    owned_regions = Region[body,]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("sym_name", sym_name),]

    return IR.create_operation(
        "irdl.operation",
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
`parameters`

`irdl.parameters` defines the constraints on parameters of a type or
attribute definition.

# Example

```mlir
irdl.dialect @cmath {
  irdl.type @complex {
    %0 = irdl.is i32
    %1 = irdl.is i64
    %2 = irdl.any_of(%0, %1)
    irdl.parameters(%2)
  }
}
```

The above program defines a type `complex` inside the dialect `cmath`. The
type has a single parameter that should be either `i32` or `i64`.
"""
function parameters(args::Vector{Value}; location=Location())
    results = IR.Type[]
    operands = Value[args...,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return IR.create_operation(
        "irdl.parameters",
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
`parametric`

`irdl.parametric` defines a constraint that accepts only a single type
or attribute base. The attribute base is defined by a symbolic reference
to the corresponding definition. It will additionally constraint the
parameters of the type/attribute.

# Example

```mlir
irdl.dialect @cmath {

  irdl.type @complex { /* ... */ }

  irdl.operation @norm {
    %0 = irdl.any
    %1 = irdl.parametric @complex<%0>
    irdl.operands(%1)
    irdl.results(%0)
  }
}
```

The above program defines an operation `norm` inside the dialect `cmath` that
for any `T` takes a `cmath.complex` with parameter `T` and returns a `T`.
"""
function parametric(
    args::Vector{Value};
    output=nothing::Union{Nothing,IR.Type},
    base_type,
    location=Location(),
)
    results = IR.Type[]
    operands = Value[args...,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("base_type", base_type),]
    !isnothing(output) && push!(results, output)

    return IR.create_operation(
        "irdl.parametric",
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
`results`

`irdl.results` define the results of the `irdl.operation` parent operation
definition.

In the following example, `irdl.results` defines the results of the
`norm` operation:

```mlir
irdl.dialect @cmath {

  irdl.type @complex { /* ... */ }

  irdl.operation @get_values {
    %0 = irdl.any
    %1 = irdl.parametric @complex<%0>
    irdl.results(%0, %0)
    irdl.operands(%1)
  }
}
```

The operation will expect one operand of the `cmath.complex` type, and two
results that have the underlying type of the `cmath.complex`.

The results can also be marked as variadic or optional:
```mlir
irdl.results(%0, single %1, optional %2, variadic %3)
```

Here, %0 and %1 are required single results, %2 is an optional result,
and %3 is a variadic result.

When more than one result is marked as optional or variadic, the operation
will expect a \'resultSegmentSizes\' attribute that defines the number of
results in each segment.
"""
function results(args::Vector{Value}; location=Location())
    results = IR.Type[]
    operands = Value[args...,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return IR.create_operation(
        "irdl.results",
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
`type`

`irdl.type` defines a new type belonging to the `irdl.dialect` parent.

The type parameters can be defined with an `irdl.parameters` operation in
the optional region.

# Example

```mlir
irdl.dialect @cmath {
  irdl.type @complex {
    %0 = irdl.is i32
    %1 = irdl.is i64
    %2 = irdl.any_of(%0, %1)
    irdl.parameters(%2)
  }
}
```

The above program defines a type `complex` inside the dialect `cmath`. The
type has a single parameter that should be either `i32` or `i64`.
"""
function type(; sym_name, body::Region, location=Location())
    results = IR.Type[]
    operands = Value[]
    owned_regions = Region[body,]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("sym_name", sym_name),]

    return IR.create_operation(
        "irdl.type",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=results,
        result_inference=false,
    )
end

end # irdl
