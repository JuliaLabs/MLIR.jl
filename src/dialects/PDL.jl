module PDL

import ...IR: NamedAttribute, MLIRType, Value, Location, Block, Region, Attribute, create_operation, context, IndexType
import ..Dialects: namedattribute, operandsegmentsizes
import ...API


"""
apply_native_constraint

`pdl.apply_native_constraint` operations apply a native C++ constraint, that
has been registered externally with the consumer of PDL, to a given set of
entities.

Example:

```mlir
// Apply `myConstraint` to the entities defined by `input`, `attr`, and `op`.
pdl.apply_native_constraint \"myConstraint\"(%input, %attr, %op : !pdl.value, !pdl.attribute, !pdl.operation)
```
  
"""
function apply_native_constraint(args::Vector{Value}; name::Union{Attribute, NamedAttribute}, location=Location())
    results = MLIRType[]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("name", name), ]
    
    create_operation(
        "pdl.apply_native_constraint", location,
        results=results,
        operands=operands,
        owned_regions=owned_regions,
        successors=successors,
        attributes=attributes,
        result_inference=false
    )
end

"""
apply_native_rewrite

`pdl.apply_native_rewrite` operations apply a native C++ function, that has
been registered externally with the consumer of PDL, to perform a rewrite
and optionally return a number of values. The native function may accept any
number of arguments. This operation is used within a pdl.rewrite region to enable
the interleaving of native rewrite methods with other pdl constructs.

Example:

```mlir
// Apply a native rewrite method that returns an attribute.
%ret = pdl.apply_native_rewrite \"myNativeFunc\"(%arg0, %attr1) : !pdl.attribute
```

```c++
// The native rewrite as defined in C++:
static Attribute myNativeFunc(PatternRewriter &rewriter, Value arg0, Attribute arg1) {
  // Just return the second arg.
  return arg1;
}

void registerNativeRewrite(PDLPatternModule &pdlModule) {
  pdlModule.registerRewriteFunction(\"myNativeFunc\", myNativeFunc);
}
```
  
"""
function apply_native_rewrite(args::Vector{Value}; results::Vector{MLIRType}, name::Union{Attribute, NamedAttribute}, location=Location())
    results = MLIRType[results..., ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("name", name), ]
    
    create_operation(
        "pdl.apply_native_rewrite", location,
        results=results,
        operands=operands,
        owned_regions=owned_regions,
        successors=successors,
        attributes=attributes,
        result_inference=false
    )
end

"""
attribute

`pdl.attribute` operations capture named attribute edges into an operation.
Instances of this operation define, and partially constrain, attributes of a
given operation. A `pdl.attribute` may partially constrain the input by
specifying an expected attribute value type (via a `pdl.type` operation), or
a constant value for the attribute (via `val`). Only one of these may be set
for a given input, as the type of the constant value provides the type. When
defined within a `pdl.rewrite` region, the constant value must be specified.

Example:

```mlir
// Define an attribute:
%attr = pdl.attribute

// Define an attribute with an expected type:
%type = pdl.type : i32
%attr = pdl.attribute : %type

// Define an attribute with a constant value:
%attr = pdl.attribute = \"hello\"
```
  
"""
function attribute(valueType=nothing::Union{Nothing, Value}; attr::MLIRType, value=nothing::Union{Nothing, Union{Attribute, NamedAttribute}}, location=Location())
    results = MLIRType[attr, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (valueType != nothing) && push!(operands, valueType)
    (value != nothing) && push!(attributes, namedattribute("value", value))
    
    create_operation(
        "pdl.attribute", location,
        results=results,
        operands=operands,
        owned_regions=owned_regions,
        successors=successors,
        attributes=attributes,
        result_inference=false
    )
end

"""
erase

`pdl.erase` operations are used within `pdl.rewrite` regions to specify that
an input operation should be marked as erased. The semantics of this
operation correspond with the `eraseOp` method on a `PatternRewriter`.

Example:

```mlir
pdl.erase %root
```
  
"""
function erase(opValue::Value; location=Location())
    results = MLIRType[]
    operands = Value[opValue, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "pdl.erase", location,
        results=results,
        operands=operands,
        owned_regions=owned_regions,
        successors=successors,
        attributes=attributes,
        result_inference=false
    )
end

"""
operand

`pdl.operand` operations capture external operand edges into an operation
node that originate from operations or block arguments not otherwise
specified within the pattern (i.e. via `pdl.result` or `pdl.results`). These
operations define individual operands of a given operation. A `pdl.operand`
may partially constrain an operand by specifying an expected value type
(via a `pdl.type` operation).

Example:

```mlir
// Define an external operand:
%operand = pdl.operand

// Define an external operand with an expected type:
%type = pdl.type : i32
%operand = pdl.operand : %type
```
  
"""
function operand(valueType=nothing::Union{Nothing, Value}; value::MLIRType, location=Location())
    results = MLIRType[value, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (valueType != nothing) && push!(operands, valueType)
    
    create_operation(
        "pdl.operand", location,
        results=results,
        operands=operands,
        owned_regions=owned_regions,
        successors=successors,
        attributes=attributes,
        result_inference=false
    )
end

"""
operands

`pdl.operands` operations capture external operand range edges into an
operation node that originate from operations or block arguments not
otherwise specified within the pattern (i.e. via `pdl.result` or
`pdl.results`). These operations define groups of input operands into a
given operation. A `pdl.operands` may partially constrain a set of input
operands by specifying expected value types (via `pdl.types` operations).

Example:

```mlir
// Define a range of input operands:
%operands = pdl.operands

// Define a range of input operands with expected types:
%types = pdl.types : [i32, i64, i32]
%typed_operands = pdl.operands : %types
```
  
"""
function operands(valueType=nothing::Union{Nothing, Value}; value::MLIRType, location=Location())
    results = MLIRType[value, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (valueType != nothing) && push!(operands, valueType)
    
    create_operation(
        "pdl.operands", location,
        results=results,
        operands=operands,
        owned_regions=owned_regions,
        successors=successors,
        attributes=attributes,
        result_inference=false
    )
end

"""
operation

`pdl.operation` operations define operation nodes within a pattern. Within
a match sequence, i.e. when directly nested within a `pdl.pattern`, these
operations correspond to input operations, or those that already existing
within the MLIR module. Inside of a `pdl.rewrite`, these operations
correspond to operations that should be created as part of the replacement
sequence.

`pdl.operation`s are composed of a name, and a set of attribute, operand,
and result type values, that map to what those that would be on a
constructed instance of that operation. The results of a `pdl.operation` are
a handle to the operation itself. Handles to the results of the operation
can be extracted via `pdl.result`.

Example:

```mlir
// Define an instance of a `foo.op` operation.
%op = pdl.operation \"foo.op\"(%arg0, %arg1 : !pdl.value, !pdl.value)
  {\"attrA\" = %attr0} -> (%type, %type : !pdl.type, !pdl.type)
```

When used within a matching context, the name of the operation may be
omitted.

When used within a rewriting context, i.e. when defined within a
`pdl.rewrite`, all of the result types must be \"inferable\". This means that
the type must be attributable to either a constant type value or the result
type of another entity, such as an attribute, the result of a
`apply_native_rewrite`, or the result type of another operation. If the
result type value does not meet any of these criteria, the operation must
override the `InferTypeOpInterface` to ensure that the result types can be
inferred.

The operands of the operation are interpreted in the following ways:

1) A single !pdl.range<value>:

In this case, the single range is treated as all of the operands of the
operation.

```mlir
// Define an instance with single range of operands.
%op = pdl.operation \"func.return\"(%allArgs : !pdl.range<value>)
```

2) A variadic number of either !pdl.value or !pdl.range<value>:

In this case, the inputs are expected to correspond with the operand groups
defined on the operation in ODS.

```tablgen
// Given the following operation definition in ODS:
def MyIndirectCallOp {
  let results = (outs FunctionType:\$call, Variadic<AnyType>:\$args);
}
```

```mlir
// We can match the operands as so:
%op = pdl.operation \"my.indirect_call\"(%call, %args : !pdl.value, !pdl.range<value>)
```

The results of the operation are interpreted in the following ways:

1) A single !pdl.range<type>:

In this case, the single range is treated as all of the result types of the
operation.

```mlir
// Define an instance with single range of types.
%allResultTypes = pdl.types
%op = pdl.operation \"builtin.unrealized_conversion_cast\" -> (%allResultTypes : !pdl.types)
```

2) A variadic number of either !pdl.type or !pdl.range<type>:

In this case, the inputs are expected to correspond with the result groups
defined on the operation in ODS.

```tablgen
// Given the following operation definition in ODS:
def MyOp {
  let results = (outs SomeType:\$result, Variadic<SomeType>:\$otherResults);
}
```

```mlir
// We can match the results as so:
%result = pdl.type
%otherResults = pdl.types
%op = pdl.operation \"foo.op\" -> (%result, %otherResults : !pdl.type, !pdl.range<type>)
```
  
"""
function operation(operandValues::Vector{Value}, attributeValues::Vector{Value}, typeValues::Vector{Value}; op::MLIRType, opName=nothing::Union{Nothing, Union{Attribute, NamedAttribute}}, attributeValueNames::Union{Attribute, NamedAttribute}, location=Location())
    results = MLIRType[op, ]
    operands = Value[operandValues..., attributeValues..., typeValues..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("attributeValueNames", attributeValueNames), ]
    push!(attributes, operandsegmentsizes([length(operandValues), length(attributeValues), length(typeValues), ]))
    (opName != nothing) && push!(attributes, namedattribute("opName", opName))
    
    create_operation(
        "pdl.operation", location,
        results=results,
        operands=operands,
        owned_regions=owned_regions,
        successors=successors,
        attributes=attributes,
        result_inference=false
    )
end

"""
pattern

`pdl.pattern` operations provide a transformable representation for a
`RewritePattern`. The attributes on this operation correspond to the various
metadata on a `RewritePattern`, such as the benefit. The match section of
the pattern is specified within the region body, with the rewrite provided
by a terminating `pdl.rewrite`.

Example:

```mlir
// Provide a pattern matching \"foo.op\" that replaces the root with its
// operand.
pdl.pattern : benefit(1) {
  %resultType = pdl.type
  %inputOperand = pdl.operand
  %root = pdl.operation \"foo.op\"(%inputOperand) -> (%resultType)
  pdl.rewrite %root {
    pdl.replace %root with (%inputOperand)
  }
}
```
  
"""
function pattern(; benefit::Union{Attribute, NamedAttribute}, sym_name=nothing::Union{Nothing, Union{Attribute, NamedAttribute}}, bodyRegion::Region, location=Location())
    results = MLIRType[]
    operands = Value[]
    owned_regions = Region[bodyRegion, ]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("benefit", benefit), ]
    (sym_name != nothing) && push!(attributes, namedattribute("sym_name", sym_name))
    
    create_operation(
        "pdl.pattern", location,
        results=results,
        operands=operands,
        owned_regions=owned_regions,
        successors=successors,
        attributes=attributes,
        result_inference=false
    )
end

"""
range

`pdl.range` operations construct a range from a given set of PDL entities,
which all share the same underlying element type. For example, a
`!pdl.range<value>` may be constructed from a list of `!pdl.value`
or `!pdl.range<value>` entities.

Example:

```mlir
// Construct a range of values.
%valueRange = pdl.range %inputValue, %inputRange : !pdl.value, !pdl.range<value>

// Construct a range of types.
%typeRange = pdl.range %inputType, %inputRange : !pdl.type, !pdl.range<type>

// Construct an empty range of types.
%valueRange = pdl.range : !pdl.range<type>
```

TODO: Range construction is currently limited to rewrites, but it could
be extended to constraints under certain circustances; i.e., if we can
determine how to extract the underlying elements. If we can\'t, e.g. if
there are multiple sub ranges used for construction, we won\'t be able
to determine their sizes during constraint time.
  
"""
function range(arguments::Vector{Value}; result::MLIRType, location=Location())
    results = MLIRType[result, ]
    operands = Value[arguments..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "pdl.range", location,
        results=results,
        operands=operands,
        owned_regions=owned_regions,
        successors=successors,
        attributes=attributes,
        result_inference=false
    )
end

"""
replace

`pdl.replace` operations are used within `pdl.rewrite` regions to specify
that an input operation should be marked as replaced. The semantics of this
operation correspond with the `replaceOp` method on a `PatternRewriter`. The
set of replacement values can be either:
* a single `Operation` (`replOperation` should be populated)
  - The operation will be replaced with the results of this operation.
* a set of `Value`s (`replValues` should be populated)
  - The operation will be replaced with these values.

Example:

```mlir
// Replace root node with 2 values:
pdl.replace %root with (%val0, %val1 : !pdl.value, !pdl.value)

// Replace root node with a range of values:
pdl.replace %root with (%vals : !pdl.range<value>)

// Replace root with another operation:
pdl.replace %root with %otherOp
```
  
"""
function replace(opValue::Value, replOperation=nothing::Union{Nothing, Value}, replValues::Vector{Value}; location=Location())
    results = MLIRType[]
    operands = Value[opValue, replValues..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (replOperation != nothing) && push!(operands, replOperation)
    push!(attributes, operandsegmentsizes([1, (replOperation==nothing) ? 0 : 1length(replValues), ]))
    
    create_operation(
        "pdl.replace", location,
        results=results,
        operands=operands,
        owned_regions=owned_regions,
        successors=successors,
        attributes=attributes,
        result_inference=false
    )
end

"""
result

`pdl.result` operations extract result edges from an operation node within
a pattern or rewrite region. The provided index is zero-based, and
represents the concrete result to extract, i.e. this is not the result index
as defined by the ODS definition of the operation.

Example:

```mlir
// Extract a result:
%operation = pdl.operation ...
%pdl_result = pdl.result 1 of %operation

// Imagine the following IR being matched:
%result_0, %result_1 = foo.op ...

// If the example pattern snippet above were matching against `foo.op` in
// the IR snippet, `%pdl_result` would correspond to `%result_1`.
```
  
"""
function result(parent::Value; val::MLIRType, index::Union{Attribute, NamedAttribute}, location=Location())
    results = MLIRType[val, ]
    operands = Value[parent, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("index", index), ]
    
    create_operation(
        "pdl.result", location,
        results=results,
        operands=operands,
        owned_regions=owned_regions,
        successors=successors,
        attributes=attributes,
        result_inference=false
    )
end

"""
results

`pdl.results` operations extract a result group from an operation within a
pattern or rewrite region. If an index is provided, this operation extracts
a result group as defined by the ODS definition of the operation. In this
case the result of this operation may be either a single `pdl.value` or
a `pdl.range<value>`, depending on the constraint of the result in ODS. If
no index is provided, this operation extracts the full result range of the
operation.

Example:

```mlir
// Extract all of the results of an operation:
%operation = pdl.operation ...
%results = pdl.results of %operation

// Extract the results in the first result group of an operation, which is
// variadic:
%operation = pdl.operation ...
%results = pdl.results 0 of %operation -> !pdl.range<value>

// Extract the results in the second result group of an operation, which is
// not variadic:
%operation = pdl.operation ...
%results = pdl.results 1 of %operation -> !pdl.value
```
  
"""
function results(parent::Value; val::MLIRType, index=nothing::Union{Nothing, Union{Attribute, NamedAttribute}}, location=Location())
    results = MLIRType[val, ]
    operands = Value[parent, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (index != nothing) && push!(attributes, namedattribute("index", index))
    
    create_operation(
        "pdl.results", location,
        results=results,
        operands=operands,
        owned_regions=owned_regions,
        successors=successors,
        attributes=attributes,
        result_inference=false
    )
end

"""
rewrite

`pdl.rewrite` operations terminate the region of a `pdl.pattern` and specify
the main rewrite of a `pdl.pattern`, on the optional root operation. The
rewrite is specified either via a string name (`name`) to a native
rewrite function, or via the region body. The rewrite region, if specified,
must contain a single block. If the rewrite is external it functions
similarly to `pdl.apply_native_rewrite`, and takes a set of additional
positional values defined within the matcher as arguments. If the rewrite is
external, the root operation is passed to the native function as the leading
arguments. The root operation, if provided, specifies the starting point in
the pattern for the subgraph isomorphism search. Pattern matching will proceed
from this node downward (towards the defining operation) or upward
(towards the users) until all the operations in the pattern have been matched.
If the root is omitted, the pdl_interp lowering will automatically select
the best root of the pdl.rewrite among all the operations in the pattern.

Example:

```mlir
// Specify an external rewrite function:
pdl.rewrite %root with \"myExternalRewriter\"(%value : !pdl.value)

// Specify a rewrite inline using PDL with the given root:
pdl.rewrite %root {
  %op = pdl.operation \"foo.op\"(%arg0, %arg1)
  pdl.replace %root with %op
}

// Specify a rewrite inline using PDL, automatically selecting root:
pdl.rewrite {
  %op1 = pdl.operation \"foo.op\"(%arg0, %arg1)
  %op2 = pdl.operation \"bar.op\"(%arg0, %arg1)
  pdl.replace %root1 with %op1
  pdl.replace %root2 with %op2
}
```
  
"""
function rewrite(root=nothing::Union{Nothing, Value}, externalArgs::Vector{Value}; name=nothing::Union{Nothing, Union{Attribute, NamedAttribute}}, bodyRegion::Region, location=Location())
    results = MLIRType[]
    operands = Value[externalArgs..., ]
    owned_regions = Region[bodyRegion, ]
    successors = Block[]
    attributes = NamedAttribute[]
    (root != nothing) && push!(operands, root)
    push!(attributes, operandsegmentsizes([(root==nothing) ? 0 : 1length(externalArgs), ]))
    (name != nothing) && push!(attributes, namedattribute("name", name))
    
    create_operation(
        "pdl.rewrite", location,
        results=results,
        operands=operands,
        owned_regions=owned_regions,
        successors=successors,
        attributes=attributes,
        result_inference=false
    )
end

"""
type

`pdl.type` operations capture result type constraints of `Attributes`,
`Values`, and `Operations`. Instances of this operation define, and
partially constrain, results types of a given entity. A `pdl.type` may
partially constrain the result by specifying a constant `Type`.

Example:

```mlir
// Define a type:
%type = pdl.type

// Define a type with a constant value:
%type = pdl.type : i32
```
  
"""
function type(; result::MLIRType, constantType=nothing::Union{Nothing, Union{Attribute, NamedAttribute}}, location=Location())
    results = MLIRType[result, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (constantType != nothing) && push!(attributes, namedattribute("constantType", constantType))
    
    create_operation(
        "pdl.type", location,
        results=results,
        operands=operands,
        owned_regions=owned_regions,
        successors=successors,
        attributes=attributes,
        result_inference=false
    )
end

"""
types

`pdl.types` operations capture result type constraints of `Value`s, and
`Operation`s. Instances of this operation define results types of a given
entity. A `pdl.types` may partially constrain the results by specifying
an array of `Type`s.

Example:

```mlir
// Define a range of types:
%types = pdl.types

// Define a range of types with a range of constant values:
%types = pdl.types : [i32, i64, i32]
```
  
"""
function types(; result::MLIRType, constantTypes=nothing::Union{Nothing, Union{Attribute, NamedAttribute}}, location=Location())
    results = MLIRType[result, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (constantTypes != nothing) && push!(attributes, namedattribute("constantTypes", constantTypes))
    
    create_operation(
        "pdl.types", location,
        results=results,
        operands=operands,
        owned_regions=owned_regions,
        successors=successors,
        attributes=attributes,
        result_inference=false
    )
end

end # PDL
