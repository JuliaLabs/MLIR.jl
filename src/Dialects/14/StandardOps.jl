module std

import ...IR: IR, NamedAttribute, Value, Location, Block, Region, Attribute, create_operation, context, IndexType
import ..Dialects: namedattribute, operandsegmentsizes
import ...API


"""
`assert`

Assert operation with single boolean operand and an error message attribute.
If the argument is `true` this operation has no effect. Otherwise, the
program execution will abort. The provided error message may be used by a
runtime to propagate the error to the user.

# Example

```mlir
assert %b, \"Expected ... to be true\"
```
"""
function assert(arg::Value; msg, location=Location())
    results = IR.Type[]
    operands = Value[arg,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("msg", msg),]

    create_operation(
        "std.assert", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`br`

The `br` operation represents a branch operation in a function.
The operation takes variable number of operands and produces no results.
The operand number and types for each successor must match the arguments of
the block successor.

# Example

```mlir
^bb2:
  %2 = call @someFn()
  br ^bb3(%2 : tensor<*xf32>)
^bb3(%3: tensor<*xf32>):
```
"""
function br(destOperands::Vector{Value}; dest::Block, location=Location())
    results = IR.Type[]
    operands = Value[destOperands...,]
    owned_regions = Region[]
    successors = Block[dest,]
    attributes = NamedAttribute[]

    create_operation(
        "std.br", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`call_indirect`

The `call_indirect` operation represents an indirect call to a value of
function type. Functions are first class types in MLIR, and may be passed as
arguments and merged together with block arguments. The operands and result
types of the call must match the specified function type.

Function values can be created with the
[`constant` operation](#stdconstant-constantop).

# Example

```mlir
%31 = call_indirect %15(%0, %1)
        : (tensor<16xf32>, tensor<16xf32>) -> tensor<16xf32>
```
"""
function call_indirect(callee::Value, callee_operands::Vector{Value}; results::Vector{IR.Type}, location=Location())
    results = IR.Type[results...,]
    operands = Value[callee, callee_operands...,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    create_operation(
        "std.call_indirect", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`call`

The `call` operation represents a direct call to a function that is within
the same symbol scope as the call. The operands and result types of the
call must match the specified function type. The callee is encoded as a
symbol reference attribute named \"callee\".

# Example

```mlir
%2 = call @my_add(%0, %1) : (f32, f32) -> f32
```
"""
function call(operands::Vector{Value}; result_0::Vector{IR.Type}, callee, location=Location())
    results = IR.Type[result_0...,]
    operands = Value[operands...,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("callee", callee),]

    create_operation(
        "std.call", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`cond_br`

The `cond_br` terminator operation represents a conditional branch on a
boolean (1-bit integer) value. If the bit is set, then the first destination
is jumped to; if it is false, the second destination is chosen. The count
and types of operands must align with the arguments in the corresponding
target blocks.

The MLIR conditional branch operation is not allowed to target the entry
block for a region. The two destinations of the conditional branch operation
are allowed to be the same.

The following example illustrates a function with a conditional branch
operation that targets the same block.

# Example

```mlir
func @select(%a: i32, %b: i32, %flag: i1) -> i32 {
  // Both targets are the same, operands differ
  cond_br %flag, ^bb1(%a : i32), ^bb1(%b : i32)

^bb1(%x : i32) :
  return %x : i32
}
```
"""
function cond_br(condition::Value, trueDestOperands::Vector{Value}, falseDestOperands::Vector{Value}; trueDest::Block, falseDest::Block, location=Location())
    results = IR.Type[]
    operands = Value[condition, trueDestOperands..., falseDestOperands...,]
    owned_regions = Region[]
    successors = Block[trueDest, falseDest,]
    attributes = NamedAttribute[]
    push!(attributes, operandsegmentsizes([1, length(trueDestOperands), length(falseDestOperands),]))

    create_operation(
        "std.cond_br", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`constant`

# Syntax

```
operation ::= ssa-id `=` `std.constant` attribute-value `:` type
```

The `constant` operation produces an SSA value equal to some constant
specified by an attribute. This is the way that MLIR uses to form simple
integer and floating point constants, as well as more exotic things like
references to functions and tensor/vector constants.

# Example

```mlir
// Complex constant
%1 = constant [1.0 : f32, 1.0 : f32] : complex<f32>

// Reference to function @myfn.
%2 = constant @myfn : (tensor<16xf32>, f32) -> tensor<16xf32>

// Equivalent generic forms
%1 = \"std.constant\"() {value = [1.0 : f32, 1.0 : f32] : complex<f32>}
   : () -> complex<f32>
%2 = \"std.constant\"() {value = @myfn}
   : () -> ((tensor<16xf32>, f32) -> tensor<16xf32>)
```

MLIR does not allow direct references to functions in SSA operands because
the compiler is multithreaded, and disallowing SSA values to directly
reference a function simplifies this
([rationale](../Rationale/Rationale.md#multithreading-the-compiler)).
"""
function constant(; result_0::IR.Type, value, location=Location())
    results = IR.Type[result_0,]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("value", value),]

    create_operation(
        "std.constant", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`return_`

The `return` operation represents a return operation within a function.
The operation takes variable number of operands and produces no results.
The operand number and types must match the signature of the function
that contains the operation.

# Example

```mlir
func @foo() : (i32, f8) {
  ...
  return %0, %1 : i32, f8
}
```
"""
function return_(operands::Vector{Value}; location=Location())
    results = IR.Type[]
    operands = Value[operands...,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    create_operation(
        "std.return", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`select`

The `select` operation chooses one value based on a binary condition
supplied as its first operand. If the value of the first operand is `1`,
the second operand is chosen, otherwise the third operand is chosen.
The second and the third operand must have the same type.

The operation applies to vectors and tensors elementwise given the _shape_
of all operands is identical. The choice is made for each element
individually based on the value at the same position as the element in the
condition operand. If an i1 is provided as the condition, the entire vector
or tensor is chosen.

The `select` operation combined with [`cmpi`](#stdcmpi-cmpiop) can be used
to implement `min` and `max` with signed or unsigned comparison semantics.

# Example

```mlir
// Custom form of scalar selection.
%x = select %cond, %true, %false : i32

// Generic form of the same operation.
%x = \"std.select\"(%cond, %true, %false) : (i1, i32, i32) -> i32

// Element-wise vector selection.
%vx = std.select %vcond, %vtrue, %vfalse : vector<42xi1>, vector<42xf32>

// Full vector selection.
%vx = std.select %cond, %vtrue, %vfalse : vector<42xf32>
```
"""
function select(condition::Value, true_value::Value, false_value::Value; result=nothing::Union{Nothing,IR.Type}, location=Location())
    results = IR.Type[]
    operands = Value[condition, true_value, false_value,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(results, result)

    create_operation(
        "std.select", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`splat`

Broadcast the operand to all elements of the result vector or tensor. The
operand has to be of integer/index/float type. When the result is a tensor,
it has to be statically shaped.

# Example

```mlir
%s = load %A[%i] : memref<128xf32>
%v = splat %s : vector<4xf32>
%t = splat %s : tensor<8x16xi32>
```

TODO: This operation is easy to extend to broadcast to dynamically shaped
tensors in the same way dynamically shaped memrefs are handled.

```mlir
// Broadcasts %s to a 2-d dynamically shaped tensor, with %m, %n binding
// to the sizes of the two dynamic dimensions.
%m = \"foo\"() : () -> (index)
%n = \"bar\"() : () -> (index)
%t = splat %s [%m, %n] : tensor<?x?xi32>
```
"""
function splat(input::Value; aggregate::IR.Type, location=Location())
    results = IR.Type[aggregate,]
    operands = Value[input,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    create_operation(
        "std.splat", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`switch`

The `switch` terminator operation represents a switch on a signless integer
value. If the flag matches one of the specified cases, then the
corresponding destination is jumped to. If the flag does not match any of
the cases, the default destination is jumped to. The count and types of
operands must align with the arguments in the corresponding target blocks.

# Example

```mlir
switch %flag : i32, [
  default: ^bb1(%a : i32),
  42: ^bb1(%b : i32),
  43: ^bb3(%c : i32)
]
```
"""
function switch(flag::Value, defaultOperands::Vector{Value}, caseOperands::Vector{Value}; case_values=nothing, case_operand_segments, defaultDestination::Block, caseDestinations::Vector{Block}, location=Location())
    results = IR.Type[]
    operands = Value[flag, defaultOperands..., caseOperands...,]
    owned_regions = Region[]
    successors = Block[defaultDestination, caseDestinations...,]
    attributes = NamedAttribute[namedattribute("case_operand_segments", case_operand_segments),]
    push!(attributes, operandsegmentsizes([1, length(defaultOperands), length(caseOperands),]))
    !isnothing(case_values) && push!(attributes, namedattribute("case_values", case_values))

    create_operation(
        "std.switch", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

end # std
