module ml_program

import ...IR: NamedAttribute, MLIRType, Value, Location, Block, Region, Attribute, create_operation, context, IndexType
import ..Dialects: namedattribute, operandsegmentsizes
import ...API


"""
`func`

This simple function container represents callables in an ML program where
the body is an `SSACFG` region. It must be terminated by a `return` op which
yields values with the same arity and types as the `FunctionType` results
of the containing `func`.

This op is a `Symbol` but does not introduce a new `SymbolTable`. As such,
it cannot represent nested symbols.

# Example

```mlir
ml_program.func private @some_extern(i32) -> i32
ml_program.func @compute(%arg0 : i32) -> i32 {
  ml_program.return %arg0 : i32
}
```
"""
function func(; sym_name, function_type, arg_attrs=nothing, res_attrs=nothing, sym_visibility=nothing, body::Region, location=Location())
    results = MLIRType[]
    operands = Value[]
    owned_regions = Region[body, ]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("sym_name", sym_name), namedattribute("function_type", function_type), ]
    !isnothing(arg_attrs) && push!(attributes, namedattribute("arg_attrs", arg_attrs))
    !isnothing(res_attrs) && push!(attributes, namedattribute("res_attrs", res_attrs))
    !isnothing(sym_visibility) && push!(attributes, namedattribute("sym_visibility", sym_visibility))
    
    create_operation(
        "ml_program.func", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`global_load_const`

Loads a constant (immutable) value from a global directly by symbol.

This op is only legal for globals that are not mutable and exists because
such a load can be considered to have no side effects.

# Example

```mlir
%0 = ml_program.global_load_const @foobar : tensor<?xi32>
```
"""
function global_load_const(; result::MLIRType, global_, location=Location())
    results = MLIRType[result, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("global", global_), ]
    
    create_operation(
        "ml_program.global_load_const", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`global_load_graph`

Performs a non-atomic, non-volatile, non-synchronized load from a global
that may be mutable.

It is fully expected that these constraints are not suitable for all
situations, and alternative ops should be defined and used for more advanced
cases.

This op is side effecting and may not be valid to use in graph regions
without additional consideration to evaluation order constraints.

# Example

```mlir
%0, %cstr = ml_program.global_load_graph @foobar
  ordering (%token -> !ml_program.token) : tensor<?xi32>
```
"""
function global_load_graph(consumeTokens::Vector{Value}; result::MLIRType, produceToken::MLIRType, global_, location=Location())
    results = MLIRType[result, produceToken, ]
    operands = Value[consumeTokens..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("global", global_), ]
    
    create_operation(
        "ml_program.global_load_graph", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`global_load`

Performs a non-atomic, non-volatile, non-synchronized load from a global
that may be mutable.

It is fully expected that these constraints are not suitable for
all situations, and alternative ops should be defined and used for more
advanced cases.

This op is side effecting and may not be valid to use in graph regions
without additional consideration to evaluation order constraints. See
`global_load_graph` for op which allows for explicit ordering constraints.

# Example

```mlir
%0 = ml_program.global_load @foobar : tensor<?xi32>
```
"""
function global_load(; result::MLIRType, global_, location=Location())
    results = MLIRType[result, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("global", global_), ]
    
    create_operation(
        "ml_program.global_load", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`global_`

Declares a named global variable (or constant).

A global contains a value of a specified type which can be accessed at
runtime via appropriate load/store operations. It can be mutable or
constant, optionally taking an initial value or declared as
extern (in which case, the initial value is found in external storage
by symbol name).

Generally, the type of the global and the type of the initial value
will be the same. However, for type hierarchies which can have a more
generalized bounding type that can be assigned from a narrow type, this
is allowed (but not verified).

Examples:

```mlir
// Constant global.
ml_program.global @foobar(dense<4> : tensor<4xi32>) : tensor<?xi32>

// Constant with external linkage.
ml_program.global mutable @foobar(#ml_program.extern<tensor<4xi32>>)
  : tensor<?xi32>

// Mutable global with an undefined initial value.
ml_program.global mutable @foobar : tensor<?xi32>
```
"""
function global_(; sym_name, type, is_mutable=nothing, value=nothing, sym_visibility=nothing, location=Location())
    results = MLIRType[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("sym_name", sym_name), namedattribute("type", type), ]
    !isnothing(is_mutable) && push!(attributes, namedattribute("is_mutable", is_mutable))
    !isnothing(value) && push!(attributes, namedattribute("value", value))
    !isnothing(sym_visibility) && push!(attributes, namedattribute("sym_visibility", sym_visibility))
    
    create_operation(
        "ml_program.global", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`global_store_graph`

Performs a non-atomic, non-volatile, non-synchronized store to a mutable
global.

It is fully expected that these constraints are not suitable for
all situations, and alternative ops should be defined and used for more
advanced cases.

This op is side effecting and may not be valid to use in graph regions
without additional consideration to evaluation order constraints.

# Example

```mlir
%token = ml_program.global_store @foobar = %0 : tensor<?xi32>
  ordering (%in_token -> !ml_program.token) : tensor<?xi32>
```
"""
function global_store_graph(value::Value, consumeTokens::Vector{Value}; produceToken::MLIRType, global_, location=Location())
    results = MLIRType[produceToken, ]
    operands = Value[value, consumeTokens..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("global", global_), ]
    
    create_operation(
        "ml_program.global_store_graph", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`global_store`

Performs a non-atomic, non-volatile, non-synchronized store to a mutable
global.

It is fully expected that these constraints are not suitable for
all situations, and alternative ops should be defined and used for more
advanced cases.

This op is side effecting and may not be valid to use in graph regions
without additional consideration to evaluation order constraints. See
`global_store_graph` for op which allows for explicit ordering constraints.

# Example

```mlir
ml_program.global_store @foobar = %0 : tensor<?xi32>
```
"""
function global_store(value::Value; global_, location=Location())
    results = MLIRType[]
    operands = Value[value, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("global", global_), ]
    
    create_operation(
        "ml_program.global_store", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`output`

The `output` operation terminates a subgraph by yielding values
to the caller.
The operation takes variable number of operands and produces no results.
The operand number and types must match the signature of the function
that contains the operation.
"""
function output(operands::Vector{Value}; location=Location())
    results = MLIRType[]
    operands = Value[operands..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "ml_program.output", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`return_`

The `return` operation terminates a `func` function by yielding values
to the caller.
The operation takes variable number of operands and produces no results.
The operand number and types must match the signature of the function
that contains the operation.
"""
function return_(operands::Vector{Value}; location=Location())
    results = MLIRType[]
    operands = Value[operands..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "ml_program.return", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`subgraph`

This simple function container represents callables in an ML program where
the body is a `Graph` region containing a single block. It must be
terminated by an `output` op which yields values with the same arity and
types as the `FunctionType` results of the containing `subgraph`.

This op is a `Symbol` but does not introduce a new `SymbolTable`. As such,
it cannot represented nested symbols.

# Example

```mlir
ml_program.subgraph private @some_extern(i32) -> i32
ml_program.subgraph @compute(%arg0 : i32) -> i32 {
  ml_program.output %arg0 : i32
}
```
"""
function subgraph(; sym_name, function_type, arg_attrs=nothing, res_attrs=nothing, sym_visibility=nothing, body::Region, location=Location())
    results = MLIRType[]
    operands = Value[]
    owned_regions = Region[body, ]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("sym_name", sym_name), namedattribute("function_type", function_type), ]
    !isnothing(arg_attrs) && push!(attributes, namedattribute("arg_attrs", arg_attrs))
    !isnothing(res_attrs) && push!(attributes, namedattribute("res_attrs", res_attrs))
    !isnothing(sym_visibility) && push!(attributes, namedattribute("sym_visibility", sym_visibility))
    
    create_operation(
        "ml_program.subgraph", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`token`

Token values are used to chain side effecting ops in a graph so as to
establish an execution order. This op produces a token.
"""
function token(; token::MLIRType, location=Location())
    results = MLIRType[token, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "ml_program.token", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

end # ml_program
