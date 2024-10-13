module polynomial

import ...IR:
    IR, NamedAttribute, Value, Location, Block, Region, Attribute, context, IndexType
import ..Dialects: namedattribute, operandsegmentsizes

"""
`add`

Performs polynomial addition on the operands. The operands may be single
polynomials or containers of identically-typed polynomials, i.e., polynomials
from the same underlying ring with the same coefficient types.

Addition is defined to occur in the ring defined by the ring attribute of
the two operands, meaning the addition is taken modulo the coefficientModulus
and the polynomialModulus of the ring.

# Example

```mlir
// add two polynomials modulo x^1024 - 1
#poly = #polynomial.int_polynomial<x**1024 - 1>
#ring = #polynomial.ring<coefficientType=i32, coefficientModulus=65536:i32, polynomialModulus=#poly>
%0 = polynomial.constant int<1 + x**2> : !polynomial.polynomial<#ring>
%1 = polynomial.constant int<x**5 - x + 1> : !polynomial.polynomial<#ring>
%2 = polynomial.add %0, %1 : !polynomial.polynomial<#ring>
```
"""
function add(
    lhs::Value, rhs::Value; result=nothing::Union{Nothing,IR.Type}, location=Location()
)
    _results = IR.Type[]
    _operands = Value[lhs, rhs]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(result) && push!(_results, result)

    return IR.create_operation(
        "polynomial.add",
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
`constant`

# Example

```mlir
!int_poly_ty = !polynomial.polynomial<ring=<coefficientType=i32>>
%0 = polynomial.constant int<1 + x**2> : !int_poly_ty

!float_poly_ty = !polynomial.polynomial<ring=<coefficientType=f32>>
%1 = polynomial.constant float<0.5 + 1.3e06 x**2> : !float_poly_ty
```
"""
function constant(; output=nothing::Union{Nothing,IR.Type}, value, location=Location())
    _results = IR.Type[]
    _operands = Value[]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[namedattribute("value", value),]
    !isnothing(output) && push!(_results, output)

    return IR.create_operation(
        "polynomial.constant",
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
`from_tensor`

`polynomial.from_tensor` creates a polynomial value from a tensor of coefficients.
The input tensor must list the coefficients in degree-increasing order.

The input one-dimensional tensor may have size at most the degree of the
ring\'s polynomialModulus generator polynomial, with smaller dimension implying that
all higher-degree terms have coefficient zero.

# Example

```mlir
#poly = #polynomial.int_polynomial<x**1024 - 1>
#ring = #polynomial.ring<coefficientType=i32, coefficientModulus=65536:i32, polynomialModulus=#poly>
%two = arith.constant 2 : i32
%five = arith.constant 5 : i32
%coeffs = tensor.from_elements %two, %two, %five : tensor<3xi32>
%poly = polynomial.from_tensor %coeffs : tensor<3xi32> -> !polynomial.polynomial<#ring>
```
"""
function from_tensor(input::Value; output::IR.Type, location=Location())
    _results = IR.Type[output,]
    _operands = Value[input,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "polynomial.from_tensor",
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
`intt`

`polynomial.intt` computes the reverse integer Number Theoretic Transform
(INTT) on the input tensor. This is the inverse operation of the
`polynomial.ntt` operation.

The input tensor is interpreted as a point-value representation of the
output polynomial at powers of a primitive `n`-th root of unity (see
`polynomial.ntt`). The ring of the polynomial is taken from the required
encoding attribute of the tensor.

The choice of primitive root may be optionally specified.
"""
function intt(input::Value; output::IR.Type, root=nothing, location=Location())
    _results = IR.Type[output,]
    _operands = Value[input,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(root) && push!(_attributes, namedattribute("root", root))

    return IR.create_operation(
        "polynomial.intt",
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
`leading_term`

The degree of a polynomial is the largest \$k\$ for which the coefficient
`a_k` of `x^k` is nonzero. The leading term is the term `a_k * x^k`, which
this op represents as a pair of results. The first is the degree `k` as an
index, and the second is the coefficient, whose type matches the
coefficient type of the polynomial\'s ring attribute.

# Example

```mlir
#poly = #polynomial.int_polynomial<x**1024 - 1>
#ring = #polynomial.ring<coefficientType=i32, coefficientModulus=65536:i32, polynomialModulus=#poly>
%0 = polynomial.constant int<1 + x**2> : !polynomial.polynomial<#ring>
%1, %2 = polynomial.leading_term %0 : !polynomial.polynomial<#ring> -> (index, i32)
```
"""
function leading_term(
    input::Value; degree::IR.Type, coefficient::IR.Type, location=Location()
)
    _results = IR.Type[degree, coefficient]
    _operands = Value[input,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "polynomial.leading_term",
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
`monic_monomial_mul`

Multiply a polynomial by a monic monomial, meaning a polynomial of the form
`1 * x^k` for an index operand `k`.

In some special rings of polynomials, such as a ring of polynomials
modulo `x^n - 1`, `monomial_mul` can be interpreted as a cyclic shift of
the coefficients of the polynomial. For some rings, this results in
optimized lowerings that involve rotations and rescaling of the
coefficients of the input.
"""
function monic_monomial_mul(
    input::Value,
    monomialDegree::Value;
    output=nothing::Union{Nothing,IR.Type},
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[input, monomialDegree]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(output) && push!(_results, output)

    return IR.create_operation(
        "polynomial.monic_monomial_mul",
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
`monomial`

Construct a polynomial that consists of a single monomial term, from its
degree and coefficient as dynamic inputs.

The coefficient type of the output polynomial\'s ring attribute must match
the `coefficient` input type.

# Example

```mlir
#poly = #polynomial.int_polynomial<x**1024 - 1>
#ring = #polynomial.ring<coefficientType=i32, coefficientModulus=65536:i32, polynomialModulus=#poly>
%deg = arith.constant 1023 : index
%five = arith.constant 5 : i32
%0 = polynomial.monomial %five, %deg : (i32, index) -> !polynomial.polynomial<#ring>
```
"""
function monomial(coefficient::Value, degree::Value; output::IR.Type, location=Location())
    _results = IR.Type[output,]
    _operands = Value[coefficient, degree]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "polynomial.monomial",
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
`mul`

Performs polynomial multiplication on the operands. The operands may be single
polynomials or containers of identically-typed polynomials, i.e., polynomials
from the same underlying ring with the same coefficient types.

Multiplication is defined to occur in the ring defined by the ring attribute of
the two operands, meaning the multiplication is taken modulo the coefficientModulus
and the polynomialModulus of the ring.

# Example

```mlir
// multiply two polynomials modulo x^1024 - 1
#poly = #polynomial.int_polynomial<x**1024 - 1>
#ring = #polynomial.ring<coefficientType=i32, coefficientModulus=65536:i32, polynomialModulus=#poly>
%0 = polynomial.constant int<1 + x**2> : !polynomial.polynomial<#ring>
%1 = polynomial.constant int<x**5 - x + 1> : !polynomial.polynomial<#ring>
%2 = polynomial.mul %0, %1 : !polynomial.polynomial<#ring>
```
"""
function mul(
    lhs::Value, rhs::Value; result=nothing::Union{Nothing,IR.Type}, location=Location()
)
    _results = IR.Type[]
    _operands = Value[lhs, rhs]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(result) && push!(_results, result)

    return IR.create_operation(
        "polynomial.mul",
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
`mul_scalar`

Multiplies the polynomial operand\'s coefficients by a given scalar value.
The operation is defined to occur in the ring defined by the ring attribute
of the two operands, meaning the multiplication is taken modulo the
coefficientModulus of the ring.

The `scalar` input must have the same type as the polynomial ring\'s
coefficientType.

# Example

```mlir
// multiply two polynomials modulo x^1024 - 1
#poly = #polynomial.int_polynomial<x**1024 - 1>
#ring = #polynomial.ring<coefficientType=i32, coefficientModulus=65536:i32, polynomialModulus=#poly>
%0 = polynomial.constant int<1 + x**2> : !polynomial.polynomial<#ring>
%1 = arith.constant 3 : i32
%2 = polynomial.mul_scalar %0, %1 : !polynomial.polynomial<#ring>, i32
```
"""
function mul_scalar(
    polynomial::Value,
    scalar::Value;
    output=nothing::Union{Nothing,IR.Type},
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[polynomial, scalar]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(output) && push!(_results, output)

    return IR.create_operation(
        "polynomial.mul_scalar",
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
`ntt`

`polynomial.ntt` computes the forward integer Number Theoretic Transform
(NTT) on the input polynomial. It returns a tensor containing a point-value
representation of the input polynomial. The output tensor has shape equal
to the degree of the ring\'s `polynomialModulus`. The polynomial\'s RingAttr
is embedded as the encoding attribute of the output tensor.

Given an input polynomial `F(x)` over a ring whose `polynomialModulus` has
degree `n`, and a primitive `n`-th root of unity `omega_n`, the output is
the list of \$n\$ evaluations

  `f[k] = F(omega[n]^k) ; k = {0, ..., n-1}`

The choice of primitive root may be optionally specified.
"""
function ntt(input::Value; output::IR.Type, root=nothing, location=Location())
    _results = IR.Type[output,]
    _operands = Value[input,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(root) && push!(_attributes, namedattribute("root", root))

    return IR.create_operation(
        "polynomial.ntt",
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
`sub`

Performs polynomial subtraction on the operands. The operands may be single
polynomials or containers of identically-typed polynomials, i.e., polynomials
from the same underlying ring with the same coefficient types.

Subtraction is defined to occur in the ring defined by the ring attribute of
the two operands, meaning the subtraction is taken modulo the coefficientModulus
and the polynomialModulus of the ring.

# Example

```mlir
// subtract two polynomials modulo x^1024 - 1
#poly = #polynomial.int_polynomial<x**1024 - 1>
#ring = #polynomial.ring<coefficientType=i32, coefficientModulus=65536:i32, polynomialModulus=#poly>
%0 = polynomial.constant int<1 + x**2> : !polynomial.polynomial<#ring>
%1 = polynomial.constant int<x**5 - x + 1> : !polynomial.polynomial<#ring>
%2 = polynomial.sub %0, %1 : !polynomial.polynomial<#ring>
```
"""
function sub(
    lhs::Value, rhs::Value; result=nothing::Union{Nothing,IR.Type}, location=Location()
)
    _results = IR.Type[]
    _operands = Value[lhs, rhs]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(result) && push!(_results, result)

    return IR.create_operation(
        "polynomial.sub",
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
`to_tensor`

`polynomial.to_tensor` creates a dense tensor value containing the
coefficients of the input polynomial. The output tensor contains the
coefficients in degree-increasing order.

Operations that act on the coefficients of a polynomial, such as extracting
a specific coefficient or extracting a range of coefficients, should be
implemented by composing `to_tensor` with the relevant `tensor` dialect
ops.

The output tensor has shape equal to the degree of the polynomial ring
attribute\'s polynomialModulus, including zeroes.

# Example

```mlir
#poly = #polynomial.int_polynomial<x**1024 - 1>
#ring = #polynomial.ring<coefficientType=i32, coefficientModulus=65536:i32, polynomialModulus=#poly>
%two = arith.constant 2 : i32
%five = arith.constant 5 : i32
%coeffs = tensor.from_elements %two, %two, %five : tensor<3xi32>
%poly = polynomial.from_tensor %coeffs : tensor<3xi32> -> !polynomial.polynomial<#ring>
%tensor = polynomial.to_tensor %poly : !polynomial.polynomial<#ring> -> tensor<1024xi32>
```
"""
function to_tensor(input::Value; output::IR.Type, location=Location())
    _results = IR.Type[output,]
    _operands = Value[input,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "polynomial.to_tensor",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

end # polynomial
