module builtin

import ...IR: NamedAttribute, MLIRType, Value, Location, Block, Region, Attribute, create_operation, context, IndexType
import ..Dialects: namedattribute, operandsegmentsizes
import ...API


"""
`func`

Operations within the function cannot implicitly capture values defined
outside of the function, i.e. Functions are `IsolatedFromAbove`. All
external references must use function arguments or attributes that establish
a symbolic connection (e.g. symbols referenced by name via a string
attribute like SymbolRefAttr). An external function declaration (used when
referring to a function declared in some other module) has no body. While
the MLIR textual form provides a nice inline syntax for function arguments,
they are internally represented as “block arguments” to the first block in
the region.

Only dialect attribute names may be specified in the attribute dictionaries
for function arguments, results, or the function itself.

# Example

```mlir
// External function definitions.
func @abort()
func @scribble(i32, i64, memref<? x 128 x f32, #layout_map0>) -> f64

// A function that returns its argument twice:
func @count(%x: i64) -> (i64, i64)
  attributes {fruit: \"banana\"} {
  return %x, %x: i64, i64
}

// A function with an argument attribute
func @example_fn_arg(%x: i32 {swift.self = unit})

// A function with a result attribute
func @example_fn_result() -> (f64 {dialectName.attrName = 0 : i64})

// A function with an attribute
func @example_fn_attr() attributes {dialectName.attrName = false}
```
"""
function func(; sym_name, type, sym_visibility=nothing, body::Region, location=Location())
    results = MLIRType[]
    operands = Value[]
    owned_regions = Region[body, ]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("sym_name", sym_name), namedattribute("type", type), ]
    (sym_visibility != nothing) && push!(attributes, namedattribute("sym_visibility", sym_visibility))
    
    create_operation(
        "builtin.func", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`module_`

A `module` represents a top-level container operation. It contains a single
[graph region](../LangRef.md#control-flow-and-ssacfg-regions) containing a single block
which can contain any operations and does not have a terminator. Operations
within this region cannot implicitly capture values defined outside the module,
i.e. Modules are [IsolatedFromAbove](../Traits.md#isolatedfromabove). Modules have
an optional [symbol name](../SymbolsAndSymbolTables.md) which can be used to refer
to them in operations.

# Example

```mlir
module {
  func @foo()
}
```
"""
function module_(; sym_name=nothing, sym_visibility=nothing, body::Region, location=Location())
    results = MLIRType[]
    operands = Value[]
    owned_regions = Region[body, ]
    successors = Block[]
    attributes = NamedAttribute[]
    (sym_name != nothing) && push!(attributes, namedattribute("sym_name", sym_name))
    (sym_visibility != nothing) && push!(attributes, namedattribute("sym_visibility", sym_visibility))
    
    create_operation(
        "builtin.module", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`unrealized_conversion_cast`

An `unrealized_conversion_cast` operation represents an unrealized
conversion from one set of types to another, that is used to enable the
inter-mixing of different type systems. This operation should not be
attributed any special representational or execution semantics, and is
generally only intended to be used to satisfy the temporary intermixing of
type systems during the conversion of one type system to another.

This operation may produce results of arity 1-N, and accept as input
operands of arity 0-N.

# Example

```mlir
// An unrealized 0-1 conversion. These types of conversions are useful in
// cases where a type is removed from the type system, but not all uses have
// been converted. For example, imagine we have a tuple type that is
// expanded to its element types. If only some uses of an empty tuple type
// instance are converted we still need an instance of the tuple type, but
// have no inputs to the unrealized conversion.
%result = unrealized_conversion_cast to !bar.tuple_type<>

// An unrealized 1-1 conversion.
%result1 = unrealized_conversion_cast %operand : !foo.type to !bar.lowered_type

// An unrealized 1-N conversion.
%results2:2 = unrealized_conversion_cast %tuple_operand : !foo.tuple_type<!foo.type, !foo.type> to !foo.type, !foo.type

// An unrealized N-1 conversion.
%result3 = unrealized_conversion_cast %operand, %operand : !foo.type, !foo.type to !bar.tuple_type<!foo.type, !foo.type>
```
"""
function unrealized_conversion_cast(inputs::Vector{Value}; outputs::Vector{MLIRType}, location=Location())
    results = MLIRType[outputs..., ]
    operands = Value[inputs..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "builtin.unrealized_conversion_cast", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

end # builtin
