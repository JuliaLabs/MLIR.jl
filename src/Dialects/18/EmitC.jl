module emitc

import ...IR:
    IR, NamedAttribute, Value, Location, Block, Region, Attribute, context, IndexType
import ..Dialects: namedattribute, operandsegmentsizes

"""
`add`

With the `add` operation the arithmetic operator + (addition) can
be applied.

# Example

```mlir
// Custom form of the addition operation.
%0 = emitc.add %arg0, %arg1 : (i32, i32) -> i32
%1 = emitc.add %arg2, %arg3 : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
```
```c++
// Code emitted for the operations above.
int32_t v5 = v1 + v2;
float* v6 = v3 + v4;
```
"""
function add(lhs::Value, rhs::Value; result_0::IR.Type, location=Location())
    results = IR.Type[result_0,]
    operands = Value[lhs, rhs]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return IR.create_operation(
        "emitc.add",
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
`apply`

With the `apply` operation the operators & (address of) and * (contents of)
can be applied to a single operand.

# Example

```mlir
// Custom form of applying the & operator.
%0 = emitc.apply \"&\"(%arg0) : (i32) -> !emitc.ptr<i32>

// Generic form of the same operation.
%0 = \"emitc.apply\"(%arg0) {applicableOperator = \"&\"}
    : (i32) -> !emitc.ptr<i32>

```
"""
function apply(operand::Value; result::IR.Type, applicableOperator, location=Location())
    results = IR.Type[result,]
    operands = Value[operand,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("applicableOperator", applicableOperator),]

    return IR.create_operation(
        "emitc.apply",
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
`assign`

The `assign` operation stores an SSA value to the location designated by an
EmitC variable. This operation doesn\'t return any value. The assigned value
must be of the same type as the variable being assigned. The operation is
emitted as a C/C++ \'=\' operator.

# Example

```mlir
// Integer variable
%0 = \"emitc.variable\"(){value = 42 : i32} : () -> i32
%1 = emitc.call_opaque \"foo\"() : () -> (i32)

// Assign emitted as `... = ...;`
\"emitc.assign\"(%0, %1) : (i32, i32) -> ()
```
"""
function assign(var::Value, value::Value; location=Location())
    results = IR.Type[]
    operands = Value[var, value]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return IR.create_operation(
        "emitc.assign",
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
`call_opaque`

The `call_opaque` operation represents a C++ function call. The callee
can be an arbitrary non-empty string. The call allows specifying order
of operands and attributes in the call as follows:

- integer value of index type refers to an operand;
- attribute which will get lowered to constant value in call;

# Example

```mlir
// Custom form defining a call to `foo()`.
%0 = emitc.call_opaque \"foo\" () : () -> i32

// Generic form of the same operation.
%0 = \"emitc.call_opaque\"() {callee = \"foo\"} : () -> i32
```
"""
function call_opaque(
    operands::Vector{Value};
    result_0::Vector{IR.Type},
    callee,
    args=nothing,
    template_args=nothing,
    location=Location(),
)
    results = IR.Type[result_0...,]
    operands = Value[operands...,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("callee", callee),]
    !isnothing(args) && push!(attributes, namedattribute("args", args))
    !isnothing(template_args) &&
        push!(attributes, namedattribute("template_args", template_args))

    return IR.create_operation(
        "emitc.call_opaque",
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
`cast`

The `cast` operation performs an explicit type conversion and is emitted
as a C-style cast expression. It can be applied to integer, float, index
and EmitC types.

# Example

```mlir
// Cast from `int32_t` to `float`
%0 = emitc.cast %arg0: i32 to f32

// Cast from `void` to `int32_t` pointer
%1 = emitc.cast %arg1 :
    !emitc.ptr<!emitc.opaque<\"void\">> to !emitc.ptr<i32>
```
"""
function cast(source::Value; dest::IR.Type, location=Location())
    results = IR.Type[dest,]
    operands = Value[source,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return IR.create_operation(
        "emitc.cast",
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
`cmp`

With the `cmp` operation the comparison operators ==, !=, <, <=, >, >=, <=> 
can be applied.

Its first argument is an attribute that defines the comparison operator:

- equal to (mnemonic: `\"eq\"`; integer value: `0`)
- not equal to (mnemonic: `\"ne\"`; integer value: `1`)
- less than (mnemonic: `\"lt\"`; integer value: `2`)
- less than or equal to (mnemonic: `\"le\"`; integer value: `3`)
- greater than (mnemonic: `\"gt\"`; integer value: `4`)
- greater than or equal to (mnemonic: `\"ge\"`; integer value: `5`)
- three-way-comparison (mnemonic: `\"three_way\"`; integer value: `6`)

# Example
```mlir
// Custom form of the cmp operation.
%0 = emitc.cmp eq, %arg0, %arg1 : (i32, i32) -> i1
%1 = emitc.cmp lt, %arg2, %arg3 : 
    (
      !emitc.opaque<\"std::valarray<float>\">,
      !emitc.opaque<\"std::valarray<float>\">
    ) -> !emitc.opaque<\"std::valarray<bool>\">
```
```c++
// Code emitted for the operations above.
bool v5 = v1 == v2;
std::valarray<bool> v6 = v3 < v4;
```
"""
function cmp(lhs::Value, rhs::Value; result_0::IR.Type, predicate, location=Location())
    results = IR.Type[result_0,]
    operands = Value[lhs, rhs]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("predicate", predicate),]

    return IR.create_operation(
        "emitc.cmp",
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
`constant`

The `constant` operation produces an SSA value equal to some constant
specified by an attribute. This can be used to form simple integer and
floating point constants, as well as more exotic things like tensor
constants. The `constant` operation also supports the EmitC opaque
attribute and the EmitC opaque type. Since folding is supported,
it should not be used with pointers.

# Example

```mlir
// Integer constant
%0 = \"emitc.constant\"(){value = 42 : i32} : () -> i32

// Constant emitted as `char = CHAR_MIN;`
%1 = \"emitc.constant\"()
    {value = #emitc.opaque<\"CHAR_MIN\"> : !emitc.opaque<\"char\">}
    : () -> !emitc.opaque<\"char\">
```
"""
function constant(; result_0::IR.Type, value, location=Location())
    results = IR.Type[result_0,]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("value", value),]

    return IR.create_operation(
        "emitc.constant",
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
`div`

With the `div` operation the arithmetic operator / (division) can
be applied.

# Example

```mlir
// Custom form of the division operation.
%0 = emitc.div %arg0, %arg1 : (i32, i32) -> i32
%1 = emitc.div %arg2, %arg3 : (f32, f32) -> f32
```
```c++
// Code emitted for the operations above.
int32_t v5 = v1 / v2;
float v6 = v3 / v4;
```
"""
function div(operand_0::Value, operand_1::Value; result_0::IR.Type, location=Location())
    results = IR.Type[result_0,]
    operands = Value[operand_0, operand_1]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return IR.create_operation(
        "emitc.div",
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
`expression`

The `expression` operation returns a single SSA value which is yielded by
its single-basic-block region. The operation doesn\'t take any arguments.

As the operation is to be emitted as a C expression, the operations within
its body must form a single Def-Use tree of emitc ops whose result is
yielded by a terminating `yield`.

# Example

```mlir
%r = emitc.expression : () -> i32 {
  %0 = emitc.add %a, %b : (i32, i32) -> i32
  %1 = emitc.call \"foo\"(%0) : () -> i32
  %2 = emitc.add %c, %d : (i32, i32) -> i32
  %3 = emitc.mul %1, %2 : (i32, i32) -> i32
  yield %3
}
```

May be emitted as

```c++
int32_t v7 = foo(v1 + v2) * (v3 + v4);
```

The operations allowed within expression body are emitc.add, emitc.apply,
emitc.call, emitc.cast, emitc.cmp, emitc.div, emitc.mul, emitc.rem and
emitc.sub.

When specified, the optional `do_not_inline` indicates that the expression is
to be emitted as seen above, i.e. as the rhs of an EmitC SSA value
definition. Otherwise, the expression may be emitted inline, i.e. directly
at its use.
"""
function expression(;
    result::IR.Type, do_not_inline=nothing, region::Region, location=Location()
)
    results = IR.Type[result,]
    operands = Value[]
    owned_regions = Region[region,]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(do_not_inline) &&
        push!(attributes, namedattribute("do_not_inline", do_not_inline))

    return IR.create_operation(
        "emitc.expression",
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
`for_`

The `emitc.for` operation represents a C loop of the following form:

```c++
for (T i = lb; i < ub; i += step) { /* ... */ } // where T is typeof(lb)
```

The operation takes 3 SSA values as operands that represent the lower bound,
upper bound and step respectively, and defines an SSA value for its
induction variable. It has one region capturing the loop body. The induction
variable is represented as an argument of this region. This SSA value is a
signless integer or index. The step is a value of same type.

This operation has no result. The body region must contain exactly one block
that terminates with `emitc.yield`. Calling ForOp::build will create such a
region and insert the terminator implicitly if none is defined, so will the
parsing even in cases when it is absent from the custom format. For example:

```mlir
// Index case.
emitc.for %iv = %lb to %ub step %step {
  ... // body
}
...
// Integer case.
emitc.for %iv_32 = %lb_32 to %ub_32 step %step_32 : i32 {
  ... // body
}
```
"""
function for_(
    lowerBound::Value, upperBound::Value, step::Value; region::Region, location=Location()
)
    results = IR.Type[]
    operands = Value[lowerBound, upperBound, step]
    owned_regions = Region[region,]
    successors = Block[]
    attributes = NamedAttribute[]

    return IR.create_operation(
        "emitc.for",
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
`if_`

The `if` operation represents an if-then-else construct for
conditionally executing two regions of code. The operand to an if operation
is a boolean value. For example:

```mlir
emitc.if %b  {
  ...
} else {
  ...
}
```

The \"then\" region has exactly 1 block. The \"else\" region may have 0 or 1
blocks. The blocks are always terminated with `emitc.yield`, which can be
left out to be inserted implicitly. This operation doesn\'t produce any
results.
"""
function if_(condition::Value; thenRegion::Region, elseRegion::Region, location=Location())
    results = IR.Type[]
    operands = Value[condition,]
    owned_regions = Region[thenRegion, elseRegion]
    successors = Block[]
    attributes = NamedAttribute[]

    return IR.create_operation(
        "emitc.if",
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
`include_`

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
function include_(; include_, is_standard_include=nothing, location=Location())
    results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("include", include_),]
    !isnothing(is_standard_include) &&
        push!(attributes, namedattribute("is_standard_include", is_standard_include))

    return IR.create_operation(
        "emitc.include",
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
`literal`

The `literal` operation produces an SSA value equal to some constant
specified by an attribute.
"""
function literal(; result::IR.Type, value, location=Location())
    results = IR.Type[result,]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("value", value),]

    return IR.create_operation(
        "emitc.literal",
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
`mul`

With the `mul` operation the arithmetic operator * (multiplication) can
be applied.

# Example

```mlir
// Custom form of the multiplication operation.
%0 = emitc.mul %arg0, %arg1 : (i32, i32) -> i32
%1 = emitc.mul %arg2, %arg3 : (f32, f32) -> f32
```
```c++
// Code emitted for the operations above.
int32_t v5 = v1 * v2;
float v6 = v3 * v4;
```
"""
function mul(operand_0::Value, operand_1::Value; result_0::IR.Type, location=Location())
    results = IR.Type[result_0,]
    operands = Value[operand_0, operand_1]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return IR.create_operation(
        "emitc.mul",
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
`rem`

With the `rem` operation the arithmetic operator % (remainder) can
be applied.

# Example

```mlir
// Custom form of the remainder operation.
%0 = emitc.rem %arg0, %arg1 : (i32, i32) -> i32
```
```c++
// Code emitted for the operation above.
int32_t v5 = v1 % v2;
```
"""
function rem(operand_0::Value, operand_1::Value; result_0::IR.Type, location=Location())
    results = IR.Type[result_0,]
    operands = Value[operand_0, operand_1]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return IR.create_operation(
        "emitc.rem",
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
`sub`

With the `sub` operation the arithmetic operator - (subtraction) can
be applied.

# Example

```mlir
// Custom form of the substraction operation.
%0 = emitc.sub %arg0, %arg1 : (i32, i32) -> i32
%1 = emitc.sub %arg2, %arg3 : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
%2 = emitc.sub %arg4, %arg5 : (!emitc.ptr<i32>, !emitc.ptr<i32>)
    -> !emitc.opaque<\"ptrdiff_t\">
```
```c++
// Code emitted for the operations above.
int32_t v7 = v1 - v2;
float* v8 = v3 - v4;
ptrdiff_t v9 = v5 - v6;
```
"""
function sub(lhs::Value, rhs::Value; result_0::IR.Type, location=Location())
    results = IR.Type[result_0,]
    operands = Value[lhs, rhs]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return IR.create_operation(
        "emitc.sub",
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
`variable`

The `variable` operation produces an SSA value equal to some value
specified by an attribute. This can be used to form simple integer and
floating point variables, as well as more exotic things like tensor
variables. The `variable` operation also supports the EmitC opaque
attribute and the EmitC opaque type. If further supports the EmitC
pointer type, whereas folding is not supported.
The `variable` is emitted as a C/C++ local variable.

# Example

```mlir
// Integer variable
%0 = \"emitc.variable\"(){value = 42 : i32} : () -> i32

// Variable emitted as `int32_t* = NULL;`
%1 = \"emitc.variable\"()
    {value = #emitc.opaque<\"NULL\"> : !emitc.opaque<\"int32_t*\">}
    : () -> !emitc.opaque<\"int32_t*\">
```

Since folding is not supported, it can be used with pointers.
As an example, it is valid to create pointers to `variable` operations
by using `apply` operations and pass these to a `call` operation.
```mlir
%0 = \"emitc.variable\"() {value = 0 : i32} : () -> i32
%1 = \"emitc.variable\"() {value = 0 : i32} : () -> i32
%2 = emitc.apply \"&\"(%0) : (i32) -> !emitc.ptr<i32>
%3 = emitc.apply \"&\"(%1) : (i32) -> !emitc.ptr<i32>
emitc.call_opaque \"write\"(%2, %3)
  : (!emitc.ptr<i32>, !emitc.ptr<i32>) -> ()
```
"""
function variable(; result_0::IR.Type, value, location=Location())
    results = IR.Type[result_0,]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("value", value),]

    return IR.create_operation(
        "emitc.variable",
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
`yield`

\"yield\" terminates its parent EmitC op\'s region, optionally yielding
an SSA value. The semantics of how the values are yielded is defined by the
parent operation.
If \"yield\" has an operand, the operand must match the parent operation\'s
result. If the parent operation defines no values, then the \"emitc.yield\"
may be left out in the custom syntax and the builders will insert one
implicitly. Otherwise, it has to be present in the syntax to indicate which
value is yielded.
"""
function yield(result=nothing::Union{Nothing,Value}; location=Location())
    results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(operands, result)

    return IR.create_operation(
        "emitc.yield",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=results,
        result_inference=false,
    )
end

end # emitc
