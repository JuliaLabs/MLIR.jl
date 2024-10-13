module math

import ...IR: IR, NamedAttribute, Value, value, Location, Block, Region, Attribute, context, IndexType
import ..Dialects: namedattribute, operandsegmentsizes

"""
`abs`

The `abs` operation computes the absolute value. It takes one operand and
returns one result of the same type. This type may be a float scalar type,
a vector whose element type is float, or a tensor of floats.

# Example

```mlir
// Scalar absolute value.
%a = math.abs %b : f64

// SIMD vector element-wise absolute value.
%f = math.abs %g : vector<4xf32>

// Tensor element-wise absolute value.
%x = math.abs %y : tensor<4x?xf8>
```
"""
function abs(operand; result=nothing::Union{Nothing, IR.Type}, location=Location())
    results = IR.Type[]
    operands = Value[value(operand), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(results, result)
    
    IR.create_operation(
        "math.abs", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`atan2`

# Syntax

```
operation ::= ssa-id `=` `math.atan2` ssa-use `,` ssa-use `:` type
```

The `atan2` operation takes two operands and returns one result, all of
which must be of the same type.  This type may be a floating point scalar
type, a vector whose element type is a floating point type, or a floating
point tensor.

The 2-argument arcus tangent `atan2(y, x)` returns the angle in the
Euclidian plane between the positive x-axis and the ray through the point
(x, y).  It is a generalization of the 1-argument arcus tangent which
returns the angle on the basis of the ratio y/x.

See also https://en.wikipedia.org/wiki/Atan2

# Example

```mlir
// Scalar variant.
%a = math.atan2 %b, %c : f32

// SIMD vector variant.
%f = math.atan2 %g, %h : vector<4xf32>

// Tensor variant.
%x = math.atan2 %y, %z : tensor<4x?xf32>
```
"""
function atan2(lhs, rhs; result=nothing::Union{Nothing, IR.Type}, location=Location())
    results = IR.Type[]
    operands = Value[value(lhs), value(rhs), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(results, result)
    
    IR.create_operation(
        "math.atan2", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`atan`

# Syntax

```
operation ::= ssa-id `=` `math.atan` ssa-use `:` type
```

The `atan` operation computes the arcus tangent of a given value.  It takes
one operand and returns one result of the same type.  This type may be a
float scalar type, a vector whose element type is float, or a tensor of
floats.  It has no standard attributes.

# Example

```mlir
// Arcus tangent of scalar value.
%a = math.atan %b : f64

// SIMD vector element-wise arcus tangent.
%f = math.atan %g : vector<4xf32>

// Tensor element-wise arcus tangent.
%x = math.atan %y : tensor<4x?xf8>
```
"""
function atan(operand; result=nothing::Union{Nothing, IR.Type}, location=Location())
    results = IR.Type[]
    operands = Value[value(operand), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(results, result)
    
    IR.create_operation(
        "math.atan", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`ceil`

# Syntax

```
operation ::= ssa-id `=` `math.ceil` ssa-use `:` type
```

The `ceil` operation computes the ceiling of a given value. It takes one
operand and returns one result of the same type. This type may be a float
scalar type, a vector whose element type is float, or a tensor of floats.
It has no standard attributes.

# Example

```mlir
// Scalar ceiling value.
%a = math.ceil %b : f64

// SIMD vector element-wise ceiling value.
%f = math.ceil %g : vector<4xf32>

// Tensor element-wise ceiling value.
%x = math.ceil %y : tensor<4x?xf8>
```
"""
function ceil(operand; result=nothing::Union{Nothing, IR.Type}, location=Location())
    results = IR.Type[]
    operands = Value[value(operand), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(results, result)
    
    IR.create_operation(
        "math.ceil", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`copysign`

# Syntax

```
operation ::= ssa-id `=` `math.copysign` ssa-use `,` ssa-use `:` type
```

The `copysign` returns a value with the magnitude of the first operand and
the sign of the second operand. It takes two operands and returns one
result of the same type. This type may be a float scalar type, a vector
whose element type is float, or a tensor of floats. It has no standard
attributes.

# Example

```mlir
// Scalar copysign value.
%a = math.copysign %b, %c : f64

// SIMD vector element-wise copysign value.
%f = math.copysign %g, %h : vector<4xf32>

// Tensor element-wise copysign value.
%x = math.copysign %y, %z : tensor<4x?xf8>
```
"""
function copysign(lhs, rhs; result=nothing::Union{Nothing, IR.Type}, location=Location())
    results = IR.Type[]
    operands = Value[value(lhs), value(rhs), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(results, result)
    
    IR.create_operation(
        "math.copysign", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`cos`

# Syntax

```
operation ::= ssa-id `=` `math.cos` ssa-use `:` type
```

The `cos` operation computes the cosine of a given value. It takes one
operand and returns one result of the same type. This type may be a float
scalar type, a vector whose element type is float, or a tensor of floats.
It has no standard attributes.

# Example

```mlir
// Scalar cosine value.
%a = math.cos %b : f64

// SIMD vector element-wise cosine value.
%f = math.cos %g : vector<4xf32>

// Tensor element-wise cosine value.
%x = math.cos %y : tensor<4x?xf8>
```
"""
function cos(operand; result=nothing::Union{Nothing, IR.Type}, location=Location())
    results = IR.Type[]
    operands = Value[value(operand), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(results, result)
    
    IR.create_operation(
        "math.cos", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`ctlz`

The `ctlz` operation computes the number of leading zeros of an integer value.

# Example

```mlir
// Scalar ctlz function value.
%a = math.ctlz %b : i32

// SIMD vector element-wise ctlz function value.
%f = math.ctlz %g : vector<4xi16>

// Tensor element-wise ctlz function value.
%x = math.ctlz %y : tensor<4x?xi8>
```
"""
function ctlz(operand; result=nothing::Union{Nothing, IR.Type}, location=Location())
    results = IR.Type[]
    operands = Value[value(operand), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(results, result)
    
    IR.create_operation(
        "math.ctlz", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`cttz`

The `cttz` operation computes the number of trailing zeros of an integer value.

# Example

```mlir
// Scalar cttz function value.
%a = math.cttz %b : i32

// SIMD vector element-wise cttz function value.
%f = math.cttz %g : vector<4xi16>

// Tensor element-wise cttz function value.
%x = math.cttz %y : tensor<4x?xi8>
```
"""
function cttz(operand; result=nothing::Union{Nothing, IR.Type}, location=Location())
    results = IR.Type[]
    operands = Value[value(operand), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(results, result)
    
    IR.create_operation(
        "math.cttz", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`ctpop`

The `ctpop` operation computes the number of set bits of an integer value.

# Example

```mlir
// Scalar ctpop function value.
%a = math.ctpop %b : i32

// SIMD vector element-wise ctpop function value.
%f = math.ctpop %g : vector<4xi16>

// Tensor element-wise ctpop function value.
%x = math.ctpop %y : tensor<4x?xi8>
```
"""
function ctpop(operand; result=nothing::Union{Nothing, IR.Type}, location=Location())
    results = IR.Type[]
    operands = Value[value(operand), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(results, result)
    
    IR.create_operation(
        "math.ctpop", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`erf`

# Syntax

```
operation ::= ssa-id `=` `math.erf` ssa-use `:` type
```

The `erf` operation computes the error function. It takes one operand
and returns one result of the same type. This type may be a float scalar
type, a vector whose element type is float, or a tensor of floats. It has
no standard attributes.

# Example

```mlir
// Scalar error function value.
%a = math.erf %b : f64

// SIMD vector element-wise error function value.
%f = math.erf %g : vector<4xf32>

// Tensor element-wise error function value.
%x = math.erf %y : tensor<4x?xf8>
```
"""
function erf(operand; result=nothing::Union{Nothing, IR.Type}, location=Location())
    results = IR.Type[]
    operands = Value[value(operand), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(results, result)
    
    IR.create_operation(
        "math.erf", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`exp2`

# Syntax

```
operation ::= ssa-id `=` `math.exp2` ssa-use `:` type
```

The `exp` operation takes one operand and returns one result of the same
type. This type may be a float scalar type, a vector whose element type is
float, or a tensor of floats. It has no standard attributes.

# Example

```mlir
// Scalar natural exponential.
%a = math.exp2 %b : f64

// SIMD vector element-wise natural exponential.
%f = math.exp2 %g : vector<4xf32>

// Tensor element-wise natural exponential.
%x = math.exp2 %y : tensor<4x?xf8>
```
"""
function exp2(operand; result=nothing::Union{Nothing, IR.Type}, location=Location())
    results = IR.Type[]
    operands = Value[value(operand), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(results, result)
    
    IR.create_operation(
        "math.exp2", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`expm1`

# Syntax

```
operation ::= ssa-id `=` `math.expm1` ssa-use `:` type
```

expm1(x) := exp(x) - 1

The `expm1` operation takes one operand and returns one result of the same
type. This type may be a float scalar type, a vector whose element type is
float, or a tensor of floats. It has no standard attributes.

# Example

```mlir
// Scalar natural exponential minus 1.
%a = math.expm1 %b : f64

// SIMD vector element-wise natural exponential minus 1.
%f = math.expm1 %g : vector<4xf32>

// Tensor element-wise natural exponential minus 1.
%x = math.expm1 %y : tensor<4x?xf8>
```
"""
function expm1(operand; result=nothing::Union{Nothing, IR.Type}, location=Location())
    results = IR.Type[]
    operands = Value[value(operand), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(results, result)
    
    IR.create_operation(
        "math.expm1", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`exp`

# Syntax

```
operation ::= ssa-id `=` `math.exp` ssa-use `:` type
```

The `exp` operation takes one operand and returns one result of the same
type. This type may be a float scalar type, a vector whose element type is
float, or a tensor of floats. It has no standard attributes.

# Example

```mlir
// Scalar natural exponential.
%a = math.exp %b : f64

// SIMD vector element-wise natural exponential.
%f = math.exp %g : vector<4xf32>

// Tensor element-wise natural exponential.
%x = math.exp %y : tensor<4x?xf8>
```
"""
function exp(operand; result=nothing::Union{Nothing, IR.Type}, location=Location())
    results = IR.Type[]
    operands = Value[value(operand), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(results, result)
    
    IR.create_operation(
        "math.exp", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`floor`

# Syntax

```
operation ::= ssa-id `=` `math.floor` ssa-use `:` type
```

The `floor` operation computes the floor of a given value. It takes one
operand and returns one result of the same type. This type may be a float
scalar type, a vector whose element type is float, or a tensor of floats.
It has no standard attributes.

# Example

```mlir
// Scalar floor value.
%a = math.floor %b : f64

// SIMD vector element-wise floor value.
%f = math.floor %g : vector<4xf32>

// Tensor element-wise floor value.
%x = math.floor %y : tensor<4x?xf8>
```
"""
function floor(operand; result=nothing::Union{Nothing, IR.Type}, location=Location())
    results = IR.Type[]
    operands = Value[value(operand), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(results, result)
    
    IR.create_operation(
        "math.floor", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`fma`

# Syntax

```
operation ::= ssa-id `=` `math.fma` ssa-use `,` ssa-use `,` ssa-use `:` type
```

The `fma` operation takes three operands and returns one result, each of
these is required to be the same type. This type may be a floating point
scalar type, a vector whose element type is a floating point type, or a
floating point tensor.

# Example

```mlir
// Scalar fused multiply-add: d = a*b + c
%d = math.fma %a, %b, %c : f64

// SIMD vector fused multiply-add, e.g. for Intel SSE.
%i = math.fma %f, %g, %h : vector<4xf32>

// Tensor fused multiply-add.
%w = math.fma %x, %y, %z : tensor<4x?xbf16>
```

The semantics of the operation correspond to those of the `llvm.fma`
[intrinsic](https://llvm.org/docs/LangRef.html#llvm-fma-intrinsic). In the
particular case of lowering to LLVM, this is guaranteed to lower
to the `llvm.fma.*` intrinsic.
"""
function fma(a, b, c; result=nothing::Union{Nothing, IR.Type}, location=Location())
    results = IR.Type[]
    operands = Value[value(a), value(b), value(c), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(results, result)
    
    IR.create_operation(
        "math.fma", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`log10`

Computes the base-10 logarithm of the given value. It takes one operand and
returns one result of the same type.

# Example

```mlir
%y = math.log10 %x : f64
```
"""
function log10(operand; result=nothing::Union{Nothing, IR.Type}, location=Location())
    results = IR.Type[]
    operands = Value[value(operand), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(results, result)
    
    IR.create_operation(
        "math.log10", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`log1p`

Computes the base-e logarithm of one plus the given value. It takes one
operand and returns one result of the same type.

log1p(x) := log(1 + x)

# Example

```mlir
%y = math.log1p %x : f64
```
"""
function log1p(operand; result=nothing::Union{Nothing, IR.Type}, location=Location())
    results = IR.Type[]
    operands = Value[value(operand), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(results, result)
    
    IR.create_operation(
        "math.log1p", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`log2`

Computes the base-2 logarithm of the given value. It takes one operand and
returns one result of the same type.

# Example

```mlir
%y = math.log2 %x : f64
```
"""
function log2(operand; result=nothing::Union{Nothing, IR.Type}, location=Location())
    results = IR.Type[]
    operands = Value[value(operand), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(results, result)
    
    IR.create_operation(
        "math.log2", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`log`

Computes the base-e logarithm of the given value. It takes one operand and
returns one result of the same type.

# Example

```mlir
%y = math.log %x : f64
```
"""
function log(operand; result=nothing::Union{Nothing, IR.Type}, location=Location())
    results = IR.Type[]
    operands = Value[value(operand), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(results, result)
    
    IR.create_operation(
        "math.log", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`powf`

# Syntax

```
operation ::= ssa-id `=` `math.powf` ssa-use `,` ssa-use `:` type
```

The `powf` operation takes two operands and returns one result, each of
these is required to be the same type. This type may be a floating point
scalar type, a vector whose element type is a floating point type, or a
floating point tensor.

# Example

```mlir
// Scalar exponentiation.
%a = math.powf %b, %c : f64

// SIMD pointwise vector exponentiation
%f = math.powf %g, %h : vector<4xf32>

// Tensor pointwise exponentiation.
%x = math.powf %y, %z : tensor<4x?xbf16>
```
"""
function powf(lhs, rhs; result=nothing::Union{Nothing, IR.Type}, location=Location())
    results = IR.Type[]
    operands = Value[value(lhs), value(rhs), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(results, result)
    
    IR.create_operation(
        "math.powf", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`rsqrt`

The `rsqrt` operation computes the reciprocal of the square root. It takes
one operand and returns one result of the same type. This type may be a
float scalar type, a vector whose element type is float, or a tensor of
floats. It has no standard attributes.
"""
function rsqrt(operand; result=nothing::Union{Nothing, IR.Type}, location=Location())
    results = IR.Type[]
    operands = Value[value(operand), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(results, result)
    
    IR.create_operation(
        "math.rsqrt", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`sin`

# Syntax

```
operation ::= ssa-id `=` `math.sin` ssa-use `:` type
```

The `sin` operation computes the sine of a given value. It takes one
operand and returns one result of the same type. This type may be a float
scalar type, a vector whose element type is float, or a tensor of floats.
It has no standard attributes.

# Example

```mlir
// Scalar sine value.
%a = math.sin %b : f64

// SIMD vector element-wise sine value.
%f = math.sin %g : vector<4xf32>

// Tensor element-wise sine value.
%x = math.sin %y : tensor<4x?xf8>
```
"""
function sin(operand; result=nothing::Union{Nothing, IR.Type}, location=Location())
    results = IR.Type[]
    operands = Value[value(operand), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(results, result)
    
    IR.create_operation(
        "math.sin", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`sqrt`

The `sqrt` operation computes the square root. It takes one operand and
returns one result of the same type. This type may be a float scalar type, a
vector whose element type is float, or a tensor of floats. It has no standard
attributes.

# Example

```mlir
// Scalar square root value.
%a = math.sqrt %b : f64
// SIMD vector element-wise square root value.
%f = math.sqrt %g : vector<4xf32>
// Tensor element-wise square root value.
%x = math.sqrt %y : tensor<4x?xf32>
```
"""
function sqrt(operand; result=nothing::Union{Nothing, IR.Type}, location=Location())
    results = IR.Type[]
    operands = Value[value(operand), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(results, result)
    
    IR.create_operation(
        "math.sqrt", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`tanh`

# Syntax

```
operation ::= ssa-id `=` `std.tanh` ssa-use `:` type
```

The `tanh` operation computes the hyperbolic tangent. It takes one operand
and returns one result of the same type. This type may be a float scalar
type, a vector whose element type is float, or a tensor of floats. It has
no standard attributes.

# Example

```mlir
// Scalar hyperbolic tangent value.
%a = math.tanh %b : f64

// SIMD vector element-wise hyperbolic tangent value.
%f = math.tanh %g : vector<4xf32>

// Tensor element-wise hyperbolic tangent value.
%x = math.tanh %y : tensor<4x?xf8>
```
"""
function tanh(operand; result=nothing::Union{Nothing, IR.Type}, location=Location())
    results = IR.Type[]
    operands = Value[value(operand), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(results, result)
    
    IR.create_operation(
        "math.tanh", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

end # math
