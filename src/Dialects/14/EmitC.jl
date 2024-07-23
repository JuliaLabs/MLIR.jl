module emitc

import ...IR:
    IR, NamedAttribute, Value, Location, Block, Region, Attribute, context, IndexType
import ..Dialects: namedattribute, operandsegmentsizes

"""
`apply`

With the `apply` operation the operators & (address of) and * (contents of)
can be applied to a single operand.

# Example

```mlir
// Custom form of applying the & operator.
%0 = emitc.apply \"&\"(%arg0) : (i32) -> !emitc.opaque<\"int32_t*\">

// Generic form of the same operation.
%0 = \"emitc.apply\"(%arg0) {applicableOperator = \"&\"}
    : (i32) -> !emitc.opaque<\"int32_t*\">

```
"""
function apply(operand::Value; result::IR.Type, applicableOperator, location=Location())
    _results = IR.Type[result,]
    _operands = Value[operand,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[namedattribute("applicableOperator", applicableOperator),]

    return IR.create_operation(
        "emitc.apply",
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
`call`

The `call` operation represents a C++ function call. The call allows
specifying order of operands and attributes in the call as follows:

- integer value of index type refers to an operand;
- attribute which will get lowered to constant value in call;

# Example

```mlir
// Custom form defining a call to `foo()`.
%0 = emitc.call \"foo\" () : () -> i32

// Generic form of the same operation.
%0 = \"emitc.call\"() {callee = \"foo\"} : () -> i32
```
"""
function call(
    operands::Vector{Value};
    result_0::Vector{IR.Type},
    callee,
    args=nothing,
    template_args=nothing,
    location=Location(),
)
    _results = IR.Type[result_0...,]
    _operands = Value[operands...,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[namedattribute("callee", callee),]
    !isnothing(args) && push!(attributes, namedattribute("args", args))
    !isnothing(template_args) &&
        push!(attributes, namedattribute("template_args", template_args))

    return IR.create_operation(
        "emitc.call",
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
`constant`

The `constant` operation produces an SSA value equal to some constant
specified by an attribute. This can be used to form simple integer and
floating point constants, as well as more exotic things like tensor
constants. The `constant` operation also supports the EmitC opaque
attribute and the EmitC opaque type.

# Example

```mlir
// Integer constant
%0 = \"emitc.constant\"(){value = 42 : i32} : () -> i32

// Constant emitted as `int32_t* = NULL;`
%1 = \"emitc.constant\"()
    {value = #emitc.opaque<\"NULL\"> : !emitc.opaque<\"int32_t*\">}
    : () -> !emitc.opaque<\"int32_t*\">
```
"""
function constant(; result_0::IR.Type, value, location=Location())
    _results = IR.Type[result_0,]
    _operands = Value[]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[namedattribute("value", value),]

    return IR.create_operation(
        "emitc.constant",
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
`_include`

The `include` operation allows to define a source file inclusion via the
`#include` directive.

# Example

```mlir
// Custom form defining the inclusion of `<myheader>`.
emitc.include <\"myheader.h\">

// Generic form of the same operation.
\"emitc.include\" (){include = \"myheader.h\", is_standard_include} : () -> ()

// Custom form defining the inclusion of `\"myheader\"`.
emitc.include \"myheader.h\"

// Generic form of the same operation.
\"emitc.include\" (){include = \"myheader.h\"} : () -> ()
```
"""
function _include(; _include, is_standard_include=nothing, location=Location())
    _results = IR.Type[]
    _operands = Value[]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[namedattribute("include", _include),]
    !isnothing(is_standard_include) &&
        push!(attributes, namedattribute("is_standard_include", is_standard_include))

    return IR.create_operation(
        "emitc.include",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

end # emitc
