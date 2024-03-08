module index

import ...IR: IR, NamedAttribute, Value, Location, Block, Region, Attribute, create_operation, context, IndexType
import ..Dialects: namedattribute, operandsegmentsizes
import ...API


"""
`add`

The `index.add` operation takes two index values and computes their sum.

# Example

```mlir
// c = a + b
%c = index.add %a, %b
```
"""
function add(lhs::Value, rhs::Value; result=nothing::Union{Nothing,IR.Type}, location=Location())
    results = IR.Type[]
    operands = Value[lhs, rhs,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(results, result)

    create_operation(
        "index.add", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`and`

The `index.and` operation takes two index values and computes their bitwise
and.

# Example

```mlir
// c = a & b
%c = index.and %a, %b
```
"""
function and(lhs::Value, rhs::Value; result=nothing::Union{Nothing,IR.Type}, location=Location())
    results = IR.Type[]
    operands = Value[lhs, rhs,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(results, result)

    create_operation(
        "index.and", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`bool_constant`

The `index.bool.constant` operation produces an bool-typed SSA value equal
to either `true` or `false`.

This operation is used to materialize bool constants that arise when folding
`index.cmp`.

# Example

```mlir
%0 = index.bool.constant true
```
"""
function bool_constant(; result=nothing::Union{Nothing,IR.Type}, value, location=Location())
    results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("value", value),]
    !isnothing(result) && push!(results, result)

    create_operation(
        "index.bool.constant", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`casts`

The `index.casts` operation enables conversions between values of index type
and concrete fixed-width integer types. If casting to a wider integer, the
value is sign-extended. If casting to a narrower integer, the value is
truncated.

# Example

```mlir
// Cast to i32
%0 = index.casts %a : index to i32

// Cast from i64
%1 = index.casts %b : i64 to index
```
"""
function casts(input::Value; output::IR.Type, location=Location())
    results = IR.Type[output,]
    operands = Value[input,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    create_operation(
        "index.casts", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`castu`

The `index.castu` operation enables conversions between values of index type
and concrete fixed-width integer types. If casting to a wider integer, the
value is zero-extended. If casting to a narrower integer, the value is
truncated.

# Example

```mlir
// Cast to i32
%0 = index.castu %a : index to i32

// Cast from i64
%1 = index.castu %b : i64 to index
```
"""
function castu(input::Value; output::IR.Type, location=Location())
    results = IR.Type[output,]
    operands = Value[input,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    create_operation(
        "index.castu", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`ceildivs`

The `index.ceildivs` operation takes two index values and computes their
signed quotient. Treats the leading bit as the sign and rounds towards
positive infinity, i.e. `7 / -2 = -3`.

Note: division by zero and signed division overflow are undefined behaviour.

# Example

```mlir
// c = ceil(a / b)
%c = index.ceildivs %a, %b
```
"""
function ceildivs(lhs::Value, rhs::Value; result=nothing::Union{Nothing,IR.Type}, location=Location())
    results = IR.Type[]
    operands = Value[lhs, rhs,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(results, result)

    create_operation(
        "index.ceildivs", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`ceildivu`

The `index.ceildivu` operation takes two index values and computes their
unsigned quotient. Treats the leading bit as the most significant and rounds
towards positive infinity, i.e. `6 / -2 = 1`.

Note: division by zero is undefined behaviour.

# Example

```mlir
// c = ceil(a / b)
%c = index.ceildivu %a, %b
```
"""
function ceildivu(lhs::Value, rhs::Value; result=nothing::Union{Nothing,IR.Type}, location=Location())
    results = IR.Type[]
    operands = Value[lhs, rhs,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(results, result)

    create_operation(
        "index.ceildivu", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`cmp`

The `index.cmp` operation takes two index values and compares them according
to the comparison predicate and returns an `i1`. The following comparisons
are supported:

-   `eq`:  equal
-   `ne`:  not equal
-   `slt`: signed less than
-   `sle`: signed less than or equal
-   `sgt`: signed greater than
-   `sge`: signed greater than or equal
-   `ult`: unsigned less than
-   `ule`: unsigned less than or equal
-   `ugt`: unsigned greater than
-   `uge`: unsigned greater than or equal

The result is `1` if the comparison is true and `0` otherwise.

# Example

```mlir
// Signed less than comparison.
%0 = index.cmp slt(%a, %b)

// Unsigned greater than or equal comparison.
%1 = index.cmp uge(%a, %b)

// Not equal comparison.
%2 = index.cmp ne(%a, %b)
```
"""
function cmp(lhs::Value, rhs::Value; result=nothing::Union{Nothing,IR.Type}, pred, location=Location())
    results = IR.Type[]
    operands = Value[lhs, rhs,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("pred", pred),]
    !isnothing(result) && push!(results, result)

    create_operation(
        "index.cmp", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`constant`

The `index.constant` operation produces an index-typed SSA value equal to
some index-typed integer constant.

# Example

```mlir
%0 = index.constant 42
```
"""
function constant(; result=nothing::Union{Nothing,IR.Type}, value, location=Location())
    results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("value", value),]
    !isnothing(result) && push!(results, result)

    create_operation(
        "index.constant", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`divs`

The `index.divs` operation takes two index values and computes their signed
quotient. Treats the leading bit as the sign and rounds towards zero, i.e.
`6 / -2 = -3`.

Note: division by zero and signed division overflow are undefined behaviour.

# Example

```mlir
// c = a / b
%c = index.divs %a, %b
```
"""
function divs(lhs::Value, rhs::Value; result=nothing::Union{Nothing,IR.Type}, location=Location())
    results = IR.Type[]
    operands = Value[lhs, rhs,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(results, result)

    create_operation(
        "index.divs", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`divu`

The `index.divu` operation takes two index values and computes their
unsigned quotient. Treats the leading bit as the most significant and rounds
towards zero, i.e. `6 / -2 = 0`.

Note: division by zero is undefined behaviour.

# Example

```mlir
// c = a / b
%c = index.divu %a, %b
```
"""
function divu(lhs::Value, rhs::Value; result=nothing::Union{Nothing,IR.Type}, location=Location())
    results = IR.Type[]
    operands = Value[lhs, rhs,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(results, result)

    create_operation(
        "index.divu", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`floordivs`

The `index.floordivs` operation takes two index values and computes their
signed quotient. Treats the leading bit as the sign and rounds towards
negative infinity, i.e. `5 / -2 = -3`.

Note: division by zero and signed division overflow are undefined behaviour.

# Example

```mlir
// c = floor(a / b)
%c = index.floordivs %a, %b
```
"""
function floordivs(lhs::Value, rhs::Value; result=nothing::Union{Nothing,IR.Type}, location=Location())
    results = IR.Type[]
    operands = Value[lhs, rhs,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(results, result)

    create_operation(
        "index.floordivs", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`maxs`

The `index.maxs` operation takes two index values and computes their signed
maximum value. Treats the leading bit as the sign, i.e. `max(-2, 6) = 6`.

# Example

```mlir
// c = max(a, b)
%c = index.maxs %a, %b
```
"""
function maxs(lhs::Value, rhs::Value; result=nothing::Union{Nothing,IR.Type}, location=Location())
    results = IR.Type[]
    operands = Value[lhs, rhs,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(results, result)

    create_operation(
        "index.maxs", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`maxu`

The `index.maxu` operation takes two index values and computes their
unsigned maximum value. Treats the leading bit as the most significant, i.e.
`max(15, 6) = 15` or `max(-2, 6) = -2`.

# Example

```mlir
// c = max(a, b)
%c = index.maxu %a, %b
```
"""
function maxu(lhs::Value, rhs::Value; result=nothing::Union{Nothing,IR.Type}, location=Location())
    results = IR.Type[]
    operands = Value[lhs, rhs,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(results, result)

    create_operation(
        "index.maxu", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`mins`

The `index.mins` operation takes two index values and computes their signed
minimum value. Treats the leading bit as the sign, i.e. `min(-2, 6) = -2`.

# Example

```mlir
// c = min(a, b)
%c = index.mins %a, %b
```
"""
function mins(lhs::Value, rhs::Value; result=nothing::Union{Nothing,IR.Type}, location=Location())
    results = IR.Type[]
    operands = Value[lhs, rhs,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(results, result)

    create_operation(
        "index.mins", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`minu`

The `index.minu` operation takes two index values and computes their
unsigned minimum value. Treats the leading bit as the most significant, i.e.
`min(15, 6) = 6` or `min(-2, 6) = 6`.

# Example

```mlir
// c = min(a, b)
%c = index.minu %a, %b
```
"""
function minu(lhs::Value, rhs::Value; result=nothing::Union{Nothing,IR.Type}, location=Location())
    results = IR.Type[]
    operands = Value[lhs, rhs,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(results, result)

    create_operation(
        "index.minu", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`mul`

The `index.mul` operation takes two index values and computes their product.

# Example

```mlir
// c = a * b
%c = index.mul %a, %b
```
"""
function mul(lhs::Value, rhs::Value; result=nothing::Union{Nothing,IR.Type}, location=Location())
    results = IR.Type[]
    operands = Value[lhs, rhs,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(results, result)

    create_operation(
        "index.mul", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`or`

The `index.or` operation takes two index values and computes their bitwise
or.

# Example

```mlir
// c = a | b
%c = index.or %a, %b
```
"""
function or(lhs::Value, rhs::Value; result=nothing::Union{Nothing,IR.Type}, location=Location())
    results = IR.Type[]
    operands = Value[lhs, rhs,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(results, result)

    create_operation(
        "index.or", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`rems`

The `index.rems` operation takes two index values and computes their signed
remainder. Treats the leading bit as the sign, i.e. `6 % -2 = 0`.

# Example

```mlir
// c = a % b
%c = index.rems %a, %b
```
"""
function rems(lhs::Value, rhs::Value; result=nothing::Union{Nothing,IR.Type}, location=Location())
    results = IR.Type[]
    operands = Value[lhs, rhs,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(results, result)

    create_operation(
        "index.rems", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`remu`

The `index.remu` operation takes two index values and computes their
unsigned remainder. Treats the leading bit as the most significant, i.e.
`6 % -2 = 6`.

# Example

```mlir
// c = a % b
%c = index.remu %a, %b
```
"""
function remu(lhs::Value, rhs::Value; result=nothing::Union{Nothing,IR.Type}, location=Location())
    results = IR.Type[]
    operands = Value[lhs, rhs,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(results, result)

    create_operation(
        "index.remu", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`shl`

The `index.shl` operation shifts an index value to the left by a variable
amount. The low order bits are filled with zeroes. The RHS operand is always
treated as unsigned. If the RHS operand is equal to or greater than the
index bitwidth, the operation is undefined.

# Example

```mlir
// c = a << b
%c = index.shl %a, %b
```
"""
function shl(lhs::Value, rhs::Value; result=nothing::Union{Nothing,IR.Type}, location=Location())
    results = IR.Type[]
    operands = Value[lhs, rhs,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(results, result)

    create_operation(
        "index.shl", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`shrs`

The `index.shrs` operation shifts an index value to the right by a variable
amount. The LHS operand is treated as signed. The high order bits are filled
with copies of the most significant bit. If the RHS operand is equal to or
greater than the index bitwidth, the operation is undefined.

# Example

```mlir
// c = a >> b
%c = index.shrs %a, %b
```
"""
function shrs(lhs::Value, rhs::Value; result=nothing::Union{Nothing,IR.Type}, location=Location())
    results = IR.Type[]
    operands = Value[lhs, rhs,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(results, result)

    create_operation(
        "index.shrs", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`shru`

The `index.shru` operation shifts an index value to the right by a variable
amount. The LHS operand is treated as unsigned. The high order bits are
filled with zeroes. If the RHS operand is equal to or greater than the index
bitwidth, the operation is undefined.

# Example

```mlir
// c = a >> b
%c = index.shru %a, %b
```
"""
function shru(lhs::Value, rhs::Value; result=nothing::Union{Nothing,IR.Type}, location=Location())
    results = IR.Type[]
    operands = Value[lhs, rhs,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(results, result)

    create_operation(
        "index.shru", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`sizeof`

The `index.sizeof` operation produces an index-typed SSA value equal to the
size in bits of the `index` type. For example, on 32-bit systems, the result
is `32 : index`, and on 64-bit systems, the result is `64 : index`.

# Example

```mlir
%0 = index.sizeof
```
"""
function sizeof(; result=nothing::Union{Nothing,IR.Type}, location=Location())
    results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(results, result)

    create_operation(
        "index.sizeof", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`sub`

The `index.sub` operation takes two index values and computes the difference
of the first from the second operand.

# Example

```mlir
// c = a - b
%c = index.sub %a, %b
```
"""
function sub(lhs::Value, rhs::Value; result=nothing::Union{Nothing,IR.Type}, location=Location())
    results = IR.Type[]
    operands = Value[lhs, rhs,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(results, result)

    create_operation(
        "index.sub", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`xor`

The `index.xor` operation takes two index values and computes their bitwise
xor.

# Example

```mlir
// c = a ^ b
%c = index.xor %a, %b
```
"""
function xor(lhs::Value, rhs::Value; result=nothing::Union{Nothing,IR.Type}, location=Location())
    results = IR.Type[]
    operands = Value[lhs, rhs,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(results, result)

    create_operation(
        "index.xor", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

end # index
