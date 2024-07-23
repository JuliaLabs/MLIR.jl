module math

import ...IR:
    IR, NamedAttribute, Value, Location, Block, Region, Attribute, context, IndexType
import ..Dialects: namedattribute, operandsegmentsizes

"""
`absf`

The `absf` operation computes the absolute value. It takes one operand of
floating point type (i.e., scalar, tensor or vector) and returns one result
of the same type.

# Example

```mlir
// Scalar absolute value.
%a = math.absf %b : f64
```
"""
function absf(
    operand::Value;
    result=nothing::Union{Nothing,IR.Type},
    fastmath=nothing,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[operand,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(result) && push!(_results, result)
    !isnothing(fastmath) && push!(_attributes, namedattribute("fastmath", fastmath))

    return IR.create_operation(
        "math.absf",
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
`absi`

The `absi` operation computes the absolute value. It takes one operand of
integer type (i.e., scalar, tensor or vector) and returns one result of the
same type.

# Example

```mlir
// Scalar absolute value.
%a = math.absi %b : i64
```
"""
function absi(operand::Value; result=nothing::Union{Nothing,IR.Type}, location=Location())
    _results = IR.Type[]
    _operands = Value[operand,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(result) && push!(_results, result)

    return IR.create_operation(
        "math.absi",
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
`atan2`

# Syntax

```
operation ::= ssa-id `=` `math.atan2` ssa-use `,` ssa-use `:` type
```

The `atan2` operation takes two operands and returns one result, all of
which must be of the same type.  The operands must be of floating point type
(i.e., scalar, tensor or vector).

The 2-argument arcus tangent `atan2(y, x)` returns the angle in the
Euclidian plane between the positive x-axis and the ray through the point
(x, y).  It is a generalization of the 1-argument arcus tangent which
returns the angle on the basis of the ratio y/x.

See also https://en.wikipedia.org/wiki/Atan2

# Example

```mlir
// Scalar variant.
%a = math.atan2 %b, %c : f32
```
"""
function atan2(
    lhs::Value,
    rhs::Value;
    result=nothing::Union{Nothing,IR.Type},
    fastmath=nothing,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[lhs, rhs]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(result) && push!(_results, result)
    !isnothing(fastmath) && push!(_attributes, namedattribute("fastmath", fastmath))

    return IR.create_operation(
        "math.atan2",
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
`atan`

# Syntax

```
operation ::= ssa-id `=` `math.atan` ssa-use `:` type
```

The `atan` operation computes the arcus tangent of a given value.  It takes
one operand of floating point type (i.e., scalar, tensor or vector) and returns
one result of the same type. It has no standard attributes.

# Example

```mlir
// Arcus tangent of scalar value.
%a = math.atan %b : f64
```
"""
function atan(
    operand::Value;
    result=nothing::Union{Nothing,IR.Type},
    fastmath=nothing,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[operand,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(result) && push!(_results, result)
    !isnothing(fastmath) && push!(_attributes, namedattribute("fastmath", fastmath))

    return IR.create_operation(
        "math.atan",
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
`cbrt`

The `cbrt` operation computes the cube root. It takes one operand of
floating point type (i.e., scalar, tensor or vector) and returns one result
of the same type. It has no standard attributes.

# Example

```mlir
// Scalar cube root value.
%a = math.cbrt %b : f64
```

Note: This op is not equivalent to powf(..., 1/3.0).
"""
function cbrt(
    operand::Value;
    result=nothing::Union{Nothing,IR.Type},
    fastmath=nothing,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[operand,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(result) && push!(_results, result)
    !isnothing(fastmath) && push!(_attributes, namedattribute("fastmath", fastmath))

    return IR.create_operation(
        "math.cbrt",
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
`ceil`

# Syntax

```
operation ::= ssa-id `=` `math.ceil` ssa-use `:` type
```

The `ceil` operation computes the ceiling of a given value. It takes one
operand of floating point type (i.e., scalar, tensor or vector) and returns one
result of the same type.  It has no standard attributes.

# Example

```mlir
// Scalar ceiling value.
%a = math.ceil %b : f64
```
"""
function ceil(
    operand::Value;
    result=nothing::Union{Nothing,IR.Type},
    fastmath=nothing,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[operand,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(result) && push!(_results, result)
    !isnothing(fastmath) && push!(_attributes, namedattribute("fastmath", fastmath))

    return IR.create_operation(
        "math.ceil",
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
`copysign`

# Syntax

```
operation ::= ssa-id `=` `math.copysign` ssa-use `,` ssa-use `:` type
```

The `copysign` returns a value with the magnitude of the first operand and
the sign of the second operand. It takes two operands and returns one result of
the same type. The operands must be of floating point type (i.e., scalar,
tensor or vector). It has no standard attributes.

# Example

```mlir
// Scalar copysign value.
%a = math.copysign %b, %c : f64
```
"""
function copysign(
    lhs::Value,
    rhs::Value;
    result=nothing::Union{Nothing,IR.Type},
    fastmath=nothing,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[lhs, rhs]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(result) && push!(_results, result)
    !isnothing(fastmath) && push!(_attributes, namedattribute("fastmath", fastmath))

    return IR.create_operation(
        "math.copysign",
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
`cos`

# Syntax

```
operation ::= ssa-id `=` `math.cos` ssa-use `:` type
```

The `cos` operation computes the cosine of a given value. It takes one
operand of floating point type (i.e., scalar, tensor or vector) and returns one
result of the same type.  It has no standard attributes.

# Example

```mlir
// Scalar cosine value.
%a = math.cos %b : f64
```
"""
function cos(
    operand::Value;
    result=nothing::Union{Nothing,IR.Type},
    fastmath=nothing,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[operand,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(result) && push!(_results, result)
    !isnothing(fastmath) && push!(_attributes, namedattribute("fastmath", fastmath))

    return IR.create_operation(
        "math.cos",
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
`ctlz`

The `ctlz` operation computes the number of leading zeros of an integer value.
It operates on scalar, tensor or vector.

# Example

```mlir
// Scalar ctlz function value.
%a = math.ctlz %b : i32
```
"""
function ctlz(operand::Value; result=nothing::Union{Nothing,IR.Type}, location=Location())
    _results = IR.Type[]
    _operands = Value[operand,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(result) && push!(_results, result)

    return IR.create_operation(
        "math.ctlz",
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
`cttz`

The `cttz` operation computes the number of trailing zeros of an integer value.
It operates on scalar, tensor or vector.

# Example

```mlir
// Scalar cttz function value.
%a = math.cttz %b : i32
```
"""
function cttz(operand::Value; result=nothing::Union{Nothing,IR.Type}, location=Location())
    _results = IR.Type[]
    _operands = Value[operand,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(result) && push!(_results, result)

    return IR.create_operation(
        "math.cttz",
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
`ctpop`

The `ctpop` operation computes the number of set bits of an integer value.
It operates on scalar, tensor or vector.

# Example

```mlir
// Scalar ctpop function value.
%a = math.ctpop %b : i32
```
"""
function ctpop(operand::Value; result=nothing::Union{Nothing,IR.Type}, location=Location())
    _results = IR.Type[]
    _operands = Value[operand,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(result) && push!(_results, result)

    return IR.create_operation(
        "math.ctpop",
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
`erf`

# Syntax

```
operation ::= ssa-id `=` `math.erf` ssa-use `:` type
```

The `erf` operation computes the error function. It takes one operand of
floating point type (i.e., scalar, tensor or vector) and returns one result of
the same type. It has no standard attributes.

# Example

```mlir
// Scalar error function value.
%a = math.erf %b : f64
```
"""
function erf(
    operand::Value;
    result=nothing::Union{Nothing,IR.Type},
    fastmath=nothing,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[operand,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(result) && push!(_results, result)
    !isnothing(fastmath) && push!(_attributes, namedattribute("fastmath", fastmath))

    return IR.create_operation(
        "math.erf",
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
`exp2`

# Syntax

```
operation ::= ssa-id `=` `math.exp2` ssa-use `:` type
```

The `exp` operation takes one operand of floating point type (i.e., scalar,
tensor or vector) and returns one result of the same type. It has no standard
attributes.

# Example

```mlir
// Scalar natural exponential.
%a = math.exp2 %b : f64
```
"""
function exp2(
    operand::Value;
    result=nothing::Union{Nothing,IR.Type},
    fastmath=nothing,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[operand,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(result) && push!(_results, result)
    !isnothing(fastmath) && push!(_attributes, namedattribute("fastmath", fastmath))

    return IR.create_operation(
        "math.exp2",
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
`expm1`

# Syntax

```
operation ::= ssa-id `=` `math.expm1` ssa-use `:` type
```

expm1(x) := exp(x) - 1

The `expm1` operation takes one operand of floating point type (i.e.,
scalar, tensor or vector) and returns one result of the same type. It has no
standard attributes.

# Example

```mlir
// Scalar natural exponential minus 1.
%a = math.expm1 %b : f64
```
"""
function expm1(
    operand::Value;
    result=nothing::Union{Nothing,IR.Type},
    fastmath=nothing,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[operand,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(result) && push!(_results, result)
    !isnothing(fastmath) && push!(_attributes, namedattribute("fastmath", fastmath))

    return IR.create_operation(
        "math.expm1",
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
`exp`

# Syntax

```
operation ::= ssa-id `=` `math.exp` ssa-use `:` type
```

The `exp` operation takes one operand of floating point type (i.e., scalar,
tensor or vector) and returns one result of the same type. It has no standard
attributes.

# Example

```mlir
// Scalar natural exponential.
%a = math.exp %b : f64
```
"""
function exp(
    operand::Value;
    result=nothing::Union{Nothing,IR.Type},
    fastmath=nothing,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[operand,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(result) && push!(_results, result)
    !isnothing(fastmath) && push!(_attributes, namedattribute("fastmath", fastmath))

    return IR.create_operation(
        "math.exp",
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
`fpowi`

# Syntax

```
operation ::= ssa-id `=` `math.fpowi` ssa-use `,` ssa-use `:` type
```

The `fpowi` operation takes a `base` operand of floating point type
(i.e. scalar, tensor or vector) and a `power` operand of integer type
(also scalar, tensor or vector) and returns one result of the same type
as `base`. The result is `base` raised to the power of `power`.
The operation is elementwise for non-scalars, e.g.:

```mlir
%v = math.fpowi %base, %power : vector<2xf32>, vector<2xi32
```

The result is a vector of:

```
[<math.fpowi %base[0], %power[0]>, <math.fpowi %base[1], %power[1]>]
```

# Example

```mlir
// Scalar exponentiation.
%a = math.fpowi %base, %power : f64, i32
```
"""
function fpowi(
    lhs::Value,
    rhs::Value;
    result=nothing::Union{Nothing,IR.Type},
    fastmath=nothing,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[lhs, rhs]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(result) && push!(_results, result)
    !isnothing(fastmath) && push!(_attributes, namedattribute("fastmath", fastmath))

    return IR.create_operation(
        "math.fpowi",
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
`floor`

# Syntax

```
operation ::= ssa-id `=` `math.floor` ssa-use `:` type
```

The `floor` operation computes the floor of a given value. It takes one
operand of floating point type (i.e., scalar, tensor or vector) and returns one
result of the same type.  It has no standard attributes.

# Example

```mlir
// Scalar floor value.
%a = math.floor %b : f64
```
"""
function floor(
    operand::Value;
    result=nothing::Union{Nothing,IR.Type},
    fastmath=nothing,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[operand,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(result) && push!(_results, result)
    !isnothing(fastmath) && push!(_attributes, namedattribute("fastmath", fastmath))

    return IR.create_operation(
        "math.floor",
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
`fma`

# Syntax

```
operation ::= ssa-id `=` `math.fma` ssa-use `,` ssa-use `,` ssa-use `:` type
```

The `fma` operation takes three operands and returns one result, each of
these is required to be the same type. Operands must be of floating point type
(i.e., scalar, tensor or vector).

# Example

```mlir
// Scalar fused multiply-add: d = a*b + c
%d = math.fma %a, %b, %c : f64
```

The semantics of the operation correspond to those of the `llvm.fma`
[intrinsic](https://llvm.org/docs/LangRef.html#llvm-fma-intrinsic). In the
particular case of lowering to LLVM, this is guaranteed to lower
to the `llvm.fma.*` intrinsic.
"""
function fma(
    a::Value,
    b::Value,
    c::Value;
    result=nothing::Union{Nothing,IR.Type},
    fastmath=nothing,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[a, b, c]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(result) && push!(_results, result)
    !isnothing(fastmath) && push!(_attributes, namedattribute("fastmath", fastmath))

    return IR.create_operation(
        "math.fma",
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
`ipowi`

# Syntax

```
operation ::= ssa-id `=` `math.ipowi` ssa-use `,` ssa-use `:` type
```

The `ipowi` operation takes two operands of integer type (i.e., scalar,
tensor or vector) and returns one result of the same type. Operands
must have the same type.

# Example

```mlir
// Scalar signed integer exponentiation.
%a = math.ipowi %b, %c : i32
```
"""
function ipowi(
    lhs::Value, rhs::Value; result=nothing::Union{Nothing,IR.Type}, location=Location()
)
    _results = IR.Type[]
    _operands = Value[lhs, rhs]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(result) && push!(_results, result)

    return IR.create_operation(
        "math.ipowi",
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
`log10`

Computes the base-10 logarithm of the given value. It takes one operand of
floating point type (i.e., scalar, tensor or vector) and returns one result of
the same type.

# Example

```mlir
// Scalar log10 operation.
%y = math.log10 %x : f64
```
"""
function log10(
    operand::Value;
    result=nothing::Union{Nothing,IR.Type},
    fastmath=nothing,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[operand,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(result) && push!(_results, result)
    !isnothing(fastmath) && push!(_attributes, namedattribute("fastmath", fastmath))

    return IR.create_operation(
        "math.log10",
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
`log1p`

Computes the base-e logarithm of one plus the given value. It takes one
operand of floating point type (i.e., scalar, tensor or vector) and returns one
result of the same type.

log1p(x) := log(1 + x)

# Example

```mlir
// Scalar log1p operation.
%y = math.log1p %x : f64
```
"""
function log1p(
    operand::Value;
    result=nothing::Union{Nothing,IR.Type},
    fastmath=nothing,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[operand,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(result) && push!(_results, result)
    !isnothing(fastmath) && push!(_attributes, namedattribute("fastmath", fastmath))

    return IR.create_operation(
        "math.log1p",
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
`log2`

Computes the base-2 logarithm of the given value. It takes one operand of
floating point type (i.e., scalar, tensor or vector) and returns one result of
the same type.

# Example

```mlir
// Scalar log2 operation.
%y = math.log2 %x : f64
```
"""
function log2(
    operand::Value;
    result=nothing::Union{Nothing,IR.Type},
    fastmath=nothing,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[operand,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(result) && push!(_results, result)
    !isnothing(fastmath) && push!(_attributes, namedattribute("fastmath", fastmath))

    return IR.create_operation(
        "math.log2",
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
`log`

Computes the base-e logarithm of the given value. It takes one operand of
floating point type (i.e., scalar, tensor or vector) and returns one result of
the same type.

# Example

```mlir
// Scalar log operation.
%y = math.log %x : f64
```
"""
function log(
    operand::Value;
    result=nothing::Union{Nothing,IR.Type},
    fastmath=nothing,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[operand,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(result) && push!(_results, result)
    !isnothing(fastmath) && push!(_attributes, namedattribute("fastmath", fastmath))

    return IR.create_operation(
        "math.log",
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
`powf`

# Syntax

```
operation ::= ssa-id `=` `math.powf` ssa-use `,` ssa-use `:` type
```

The `powf` operation takes two operands of floating point type (i.e.,
scalar, tensor or vector) and returns one result of the same type. Operands
must have the same type.

# Example

```mlir
// Scalar exponentiation.
%a = math.powf %b, %c : f64
```
"""
function powf(
    lhs::Value,
    rhs::Value;
    result=nothing::Union{Nothing,IR.Type},
    fastmath=nothing,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[lhs, rhs]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(result) && push!(_results, result)
    !isnothing(fastmath) && push!(_attributes, namedattribute("fastmath", fastmath))

    return IR.create_operation(
        "math.powf",
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
`roundeven`

# Syntax

```
operation ::= ssa-id `=` `math.roundeven` ssa-use `:` type
```

The `roundeven` operation returns the operand rounded to the nearest integer
value in floating-point format. It takes one operand of floating point type
(i.e., scalar, tensor or vector) and produces one result of the same type.  The
operation rounds the argument to the nearest integer value in floating-point
format, rounding halfway cases to even, regardless of the current
rounding direction.

# Example

```mlir
// Scalar round operation.
%a = math.roundeven %b : f64
```
"""
function roundeven(
    operand::Value;
    result=nothing::Union{Nothing,IR.Type},
    fastmath=nothing,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[operand,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(result) && push!(_results, result)
    !isnothing(fastmath) && push!(_attributes, namedattribute("fastmath", fastmath))

    return IR.create_operation(
        "math.roundeven",
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
`round`

# Syntax

```
operation ::= ssa-id `=` `math.round` ssa-use `:` type
```

The `round` operation returns the operand rounded to the nearest integer
value in floating-point format. It takes one operand of floating point type
(i.e., scalar, tensor or vector) and produces one result of the same type.  The
operation rounds the argument to the nearest integer value in floating-point
format, rounding halfway cases away from zero, regardless of the current
rounding direction.

# Example

```mlir
// Scalar round operation.
%a = math.round %b : f64
```
"""
function round(
    operand::Value;
    result=nothing::Union{Nothing,IR.Type},
    fastmath=nothing,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[operand,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(result) && push!(_results, result)
    !isnothing(fastmath) && push!(_attributes, namedattribute("fastmath", fastmath))

    return IR.create_operation(
        "math.round",
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
`rsqrt`

The `rsqrt` operation computes the reciprocal of the square root. It takes
one operand of floating point type (i.e., scalar, tensor or vector) and returns
one result of the same type. It has no standard attributes.

# Example

```mlir
// Scalar reciprocal square root value.
%a = math.rsqrt %b : f64
```
"""
function rsqrt(
    operand::Value;
    result=nothing::Union{Nothing,IR.Type},
    fastmath=nothing,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[operand,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(result) && push!(_results, result)
    !isnothing(fastmath) && push!(_attributes, namedattribute("fastmath", fastmath))

    return IR.create_operation(
        "math.rsqrt",
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
`sin`

# Syntax

```
operation ::= ssa-id `=` `math.sin` ssa-use `:` type
```

The `sin` operation computes the sine of a given value. It takes one
operand of floating point type (i.e., scalar, tensor or vector) and returns one
result of the same type.  It has no standard attributes.

# Example

```mlir
// Scalar sine value.
%a = math.sin %b : f64
```
"""
function sin(
    operand::Value;
    result=nothing::Union{Nothing,IR.Type},
    fastmath=nothing,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[operand,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(result) && push!(_results, result)
    !isnothing(fastmath) && push!(_attributes, namedattribute("fastmath", fastmath))

    return IR.create_operation(
        "math.sin",
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
`sqrt`

The `sqrt` operation computes the square root. It takes one operand of
floating point type (i.e., scalar, tensor or vector) and returns one result of
the same type. It has no standard attributes.

# Example

```mlir
// Scalar square root value.
%a = math.sqrt %b : f64
```
"""
function sqrt(
    operand::Value;
    result=nothing::Union{Nothing,IR.Type},
    fastmath=nothing,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[operand,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(result) && push!(_results, result)
    !isnothing(fastmath) && push!(_attributes, namedattribute("fastmath", fastmath))

    return IR.create_operation(
        "math.sqrt",
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
`tan`

The `tan` operation computes the tangent. It takes one operand
of floating point type (i.e., scalar, tensor or vector) and returns one
result of the same type. It has no standard attributes.

# Example

```mlir
// Scalar tangent value.
%a = math.tan %b : f64
```
"""
function tan(
    operand::Value;
    result=nothing::Union{Nothing,IR.Type},
    fastmath=nothing,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[operand,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(result) && push!(_results, result)
    !isnothing(fastmath) && push!(_attributes, namedattribute("fastmath", fastmath))

    return IR.create_operation(
        "math.tan",
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
`tanh`

The `tanh` operation computes the hyperbolic tangent. It takes one operand
of floating point type (i.e., scalar, tensor or vector) and returns one
result of the same type. It has no standard attributes.

# Example

```mlir
// Scalar hyperbolic tangent value.
%a = math.tanh %b : f64
```
"""
function tanh(
    operand::Value;
    result=nothing::Union{Nothing,IR.Type},
    fastmath=nothing,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[operand,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(result) && push!(_results, result)
    !isnothing(fastmath) && push!(_attributes, namedattribute("fastmath", fastmath))

    return IR.create_operation(
        "math.tanh",
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
`trunc`

# Syntax

```
operation ::= ssa-id `=` `math.trunc` ssa-use `:` type
```

The `trunc` operation returns the operand rounded to the nearest integer
value in floating-point format. It takes one operand of floating point type
(i.e., scalar, tensor or vector) and produces one result of the same type.
The operation always rounds to the nearest integer not larger in magnitude
than the operand, regardless of the current rounding direction.

# Example

```mlir
// Scalar trunc operation.
%a = math.trunc %b : f64
```
"""
function trunc(
    operand::Value;
    result=nothing::Union{Nothing,IR.Type},
    fastmath=nothing,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[operand,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(result) && push!(_results, result)
    !isnothing(fastmath) && push!(_attributes, namedattribute("fastmath", fastmath))

    return IR.create_operation(
        "math.trunc",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=(length(_results) == 0 ? nothing : _results),
        result_inference=(length(_results) == 0 ? true : false),
    )
end

end # math
