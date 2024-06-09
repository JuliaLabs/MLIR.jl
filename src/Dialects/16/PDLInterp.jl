module pdl_interp

import ...IR: IR, NamedAttribute, Value, value, Location, Block, Region, Attribute, context, IndexType
import ..Dialects: namedattribute, operandsegmentsizes

"""
`apply_constraint`

`pdl_interp.apply_constraint` operations apply a generic constraint, that
has been registered with the interpreter, with a given set of positional
values. On success, this operation branches to the true destination,
otherwise the false destination is taken.

# Example

```mlir
// Apply `myConstraint` to the entities defined by `input`, `attr`, and
// `op`.
pdl_interp.apply_constraint \"myConstraint\"(%input, %attr, %op : !pdl.value, !pdl.attribute, !pdl.operation) -> ^matchDest, ^failureDest
```
"""
function apply_constraint(args; name, trueDest::Block, falseDest::Block, location=Location())
    results = IR.Type[]
    operands = Value[value.(args)..., ]
    owned_regions = Region[]
    successors = Block[trueDest, falseDest, ]
    attributes = NamedAttribute[namedattribute("name", name), ]
    
    IR.create_operation(
        "pdl_interp.apply_constraint", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`apply_rewrite`

`pdl_interp.apply_rewrite` operations invoke an external rewriter that has
been registered with the interpreter to perform the rewrite after a
successful match. The rewrite is passed a set of positional arguments. The
rewrite function may return any number of results.

# Example

```mlir
// Rewriter operating solely on the root operation.
pdl_interp.apply_rewrite \"rewriter\"(%root : !pdl.operation)

// Rewriter operating solely on the root operation and return an attribute.
%attr = pdl_interp.apply_rewrite \"rewriter\"(%root : !pdl.operation) : !pdl.attribute

// Rewriter operating on the root operation along with additional arguments
// from the matcher.
pdl_interp.apply_rewrite \"rewriter\"(%root : !pdl.operation, %value : !pdl.value)
```
"""
function apply_rewrite(args; results_::Vector{IR.Type}, name, location=Location())
    results = IR.Type[results_..., ]
    operands = Value[value.(args)..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("name", name), ]
    
    IR.create_operation(
        "pdl_interp.apply_rewrite", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`are_equal`

`pdl_interp.are_equal` operations compare two positional values for
equality. On success, this operation branches to the true destination,
otherwise the false destination is taken.

# Example

```mlir
pdl_interp.are_equal %result1, %result2 : !pdl.value -> ^matchDest, ^failureDest
```
"""
function are_equal(lhs, rhs; trueDest::Block, falseDest::Block, location=Location())
    results = IR.Type[]
    operands = Value[value(lhs), value(rhs), ]
    owned_regions = Region[]
    successors = Block[trueDest, falseDest, ]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "pdl_interp.are_equal", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`branch`

`pdl_interp.branch` operations expose general branch functionality to the
interpreter, and are generally used to branch from one pattern match
sequence to another.

# Example

```mlir
pdl_interp.branch ^dest
```
"""
function branch(; dest::Block, location=Location())
    _results = IR.Type[]
    _operands = Value[]
    _owned_regions = Region[]
    _successors = Block[dest,]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "pdl_interp.branch",
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
`check_attribute`

`pdl_interp.check_attribute` operations compare the value of a given
attribute with a constant value. On success, this operation branches to the
true destination, otherwise the false destination is taken.

# Example

```mlir
pdl_interp.check_attribute %attr is 10 -> ^matchDest, ^failureDest
```
"""
function check_attribute(attribute; constantValue, trueDest::Block, falseDest::Block, location=Location())
    results = IR.Type[]
    operands = Value[value(attribute), ]
    owned_regions = Region[]
    successors = Block[trueDest, falseDest, ]
    attributes = NamedAttribute[namedattribute("constantValue", constantValue), ]
    
    IR.create_operation(
        "pdl_interp.check_attribute", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`check_operand_count`

`pdl_interp.check_operand_count` operations compare the number of operands
of a given operation value with a constant. The comparison is either exact
or at_least, with the latter used to compare against a minimum number of
expected operands. On success, this operation branches to the true
destination, otherwise the false destination is taken.

# Example

```mlir
// Check for exact equality.
pdl_interp.check_operand_count of %op is 2 -> ^matchDest, ^failureDest

// Check for at least N operands.
pdl_interp.check_operand_count of %op is at_least 2 -> ^matchDest, ^failureDest
```
"""
function check_operand_count(inputOp; count, compareAtLeast=nothing, trueDest::Block, falseDest::Block, location=Location())
    results = IR.Type[]
    operands = Value[value(inputOp), ]
    owned_regions = Region[]
    successors = Block[trueDest, falseDest, ]
    attributes = NamedAttribute[namedattribute("count", count), ]
    !isnothing(compareAtLeast) && push!(attributes, namedattribute("compareAtLeast", compareAtLeast))
    
    IR.create_operation(
        "pdl_interp.check_operand_count", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`check_operation_name`

`pdl_interp.check_operation_name` operations compare the name of a given
operation with a known name. On success, this operation branches to the true
destination, otherwise the false destination is taken.

# Example

```mlir
pdl_interp.check_operation_name of %op is \"foo.op\" -> ^matchDest, ^failureDest
```
"""
function check_operation_name(inputOp; name, trueDest::Block, falseDest::Block, location=Location())
    results = IR.Type[]
    operands = Value[value(inputOp), ]
    owned_regions = Region[]
    successors = Block[trueDest, falseDest, ]
    attributes = NamedAttribute[namedattribute("name", name), ]
    
    IR.create_operation(
        "pdl_interp.check_operation_name", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`check_result_count`

`pdl_interp.check_result_count` operations compare the number of results
of a given operation value with a constant. The comparison is either exact
or at_least, with the latter used to compare against a minimum number of
expected results. On success, this operation branches to the true
destination, otherwise the false destination is taken.

# Example

```mlir
// Check for exact equality.
pdl_interp.check_result_count of %op is 2 -> ^matchDest, ^failureDest

// Check for at least N results.
pdl_interp.check_result_count of %op is at_least 2 -> ^matchDest, ^failureDest
```
"""
function check_result_count(inputOp; count, compareAtLeast=nothing, trueDest::Block, falseDest::Block, location=Location())
    results = IR.Type[]
    operands = Value[value(inputOp), ]
    owned_regions = Region[]
    successors = Block[trueDest, falseDest, ]
    attributes = NamedAttribute[namedattribute("count", count), ]
    !isnothing(compareAtLeast) && push!(attributes, namedattribute("compareAtLeast", compareAtLeast))
    
    IR.create_operation(
        "pdl_interp.check_result_count", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`check_type`

`pdl_interp.check_type` operations compare a type with a statically known
type. On success, this operation branches to the true destination, otherwise
the false destination is taken.

# Example

```mlir
pdl_interp.check_type %type is i32 -> ^matchDest, ^failureDest
```
"""
function check_type(value; type, trueDest::Block, falseDest::Block, location=Location())
    results = IR.Type[]
    operands = Value[value(value), ]
    owned_regions = Region[]
    successors = Block[trueDest, falseDest, ]
    attributes = NamedAttribute[namedattribute("type", type), ]
    
    IR.create_operation(
        "pdl_interp.check_type", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`check_types`

`pdl_interp.check_types` operations compare a range of types with a
statically known range of types. On success, this operation branches
to the true destination, otherwise the false destination is taken.

# Example

```mlir
pdl_interp.check_types %type are [i32, i64] -> ^matchDest, ^failureDest
```
"""
function check_types(value; types, trueDest::Block, falseDest::Block, location=Location())
    results = IR.Type[]
    operands = Value[value(value), ]
    owned_regions = Region[]
    successors = Block[trueDest, falseDest, ]
    attributes = NamedAttribute[namedattribute("types", types), ]
    
    IR.create_operation(
        "pdl_interp.check_types", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`continue_`

`pdl_interp.continue` operation breaks the current iteration within the
`pdl_interp.foreach` region and continues with the next iteration from
the beginning of the region.

# Example

```mlir
pdl_interp.continue
```
"""
function continue_(; location=Location())
    _results = IR.Type[]
    _operands = Value[]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "pdl_interp.continue",
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
`create_attribute`

`pdl_interp.create_attribute` operations generate a handle within the
interpreter for a specific constant attribute value.

# Example

```mlir
%attr = pdl_interp.create_attribute 10 : i64
```
"""
function create_attribute(; attribute::IR.Type, value, location=Location())
    _results = IR.Type[attribute,]
    _operands = Value[]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[namedattribute("value", value),]

    return IR.create_operation(
        "pdl_interp.create_attribute",
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
`create_operation`

`pdl_interp.create_operation` operations create an `Operation` instance with
the specified attributes, operands, and result types. See `pdl.operation`
for a more detailed description on the general interpretation of the arguments
to this operation.

# Example

```mlir
// Create an instance of a `foo.op` operation.
%op = pdl_interp.create_operation \"foo.op\"(%arg0 : !pdl.value) {\"attrA\" = %attr0} -> (%type : !pdl.type)

// Create an instance of a `foo.op` operation that has inferred result types
// (using the InferTypeOpInterface).
%op = pdl_interp.create_operation \"foo.op\"(%arg0 : !pdl.value) {\"attrA\" = %attr0} -> <inferred>
```
"""
function create_operation(inputOperands, inputAttributes, inputResultTypes; resultOp::IR.Type, name, inputAttributeNames, inferredResultTypes=nothing, location=Location())
    results = IR.Type[resultOp, ]
    operands = Value[value.(inputOperands)..., value.(inputAttributes)..., value.(inputResultTypes)..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("name", name), namedattribute("inputAttributeNames", inputAttributeNames), ]
    push!(attributes, operandsegmentsizes([length(inputOperands), length(inputAttributes), length(inputResultTypes), ]))
    !isnothing(inferredResultTypes) && push!(attributes, namedattribute("inferredResultTypes", inferredResultTypes))
    
    IR.create_operation(
        "pdl_interp.create_operation", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`create_range`

`pdl_interp.create_range` operations construct a range from a given set of PDL
entities, which all share the same underlying element type. For example, a
`!pdl.range<value>` may be constructed from a list of `!pdl.value`
or `!pdl.range<value>` entities.

# Example

```mlir
// Construct a range of values.
%valueRange = pdl_interp.create_range %inputValue, %inputRange : !pdl.value, !pdl.range<value>

// Construct a range of types.
%typeRange = pdl_interp.create_range %inputType, %inputRange : !pdl.type, !pdl.range<type>

// Construct an empty range of types.
%valueRange = pdl_interp.create_range : !pdl.range<type>
```
"""
function create_range(arguments; result::IR.Type, location=Location())
    results = IR.Type[result, ]
    operands = Value[value.(arguments)..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "pdl_interp.create_range", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`create_type`

`pdl_interp.create_type` operations generate a handle within the interpreter
for a specific constant type value.

# Example

```mlir
pdl_interp.create_type i64
```
"""
function create_type(; result::IR.Type, value, location=Location())
    _results = IR.Type[result,]
    _operands = Value[]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[namedattribute("value", value),]

    return IR.create_operation(
        "pdl_interp.create_type",
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
`create_types`

`pdl_interp.create_types` operations generate a handle within the
interpreter for a specific range of constant type values.

# Example

```mlir
pdl_interp.create_types [i64, i64]
```
"""
function create_types(; result::IR.Type, value, location=Location())
    _results = IR.Type[result,]
    _operands = Value[]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[namedattribute("value", value),]

    return IR.create_operation(
        "pdl_interp.create_types",
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
`erase`

`pdl.erase` operations are used to specify that an operation should be
marked as erased. The semantics of this operation correspond with the
`eraseOp` method on a `PatternRewriter`.

# Example

```mlir
pdl_interp.erase %root
```
"""
function erase(inputOp; location=Location())
    results = IR.Type[]
    operands = Value[value(inputOp), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "pdl_interp.erase", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`extract`

`pdl_interp.extract` operations are used to extract an item from a range
at the specified index. If the index is out of range, returns null.

# Example

```mlir
// Extract the value at index 1 from a range of values.
%ops = pdl_interp.extract 1 of %values : !pdl.value
```
"""
function extract(range; result::IR.Type, index, location=Location())
    results = IR.Type[result, ]
    operands = Value[value(range), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("index", index), ]
    
    IR.create_operation(
        "pdl_interp.extract", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`finalize`

`pdl_interp.finalize` is used to denote the termination of a match or
rewrite sequence.

# Example

```mlir
pdl_interp.finalize
```
"""
function finalize(; location=Location())
    _results = IR.Type[]
    _operands = Value[]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "pdl_interp.finalize",
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
`foreach`

`pdl_interp.foreach` iteratively selects an element from a range of values
and executes the region until pdl.continue is reached.

In the bytecode interpreter, this operation is implemented by looping over
the values and, for each selection, running the bytecode until we reach
pdl.continue. This may result in multiple matches being reported. Note
that the input range is mutated (popped from).

# Example

```mlir
pdl_interp.foreach %op : !pdl.operation in %ops {
  pdl_interp.continue
} -> ^next
```
"""
function foreach(values; region::Region, successor::Block, location=Location())
    results = IR.Type[]
    operands = Value[value(values), ]
    owned_regions = Region[region, ]
    successors = Block[successor, ]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "pdl_interp.foreach", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`func`

`pdl_interp.func` operations act as interpreter functions. These are
callable SSA-region operations that contain other interpreter operations.
Interpreter functions are used for both the matching and the rewriting
portion of the interpreter.

# Example

```mlir
pdl_interp.func @rewriter(%root: !pdl.operation) {
  %op = pdl_interp.create_operation \"foo.new_operation\"
  pdl_interp.erase %root
  pdl_interp.finalize
}
```
"""
function func(;
    sym_name,
    function_type,
    arg_attrs=nothing,
    res_attrs=nothing,
    body::Region,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[]
    _owned_regions = Region[body,]
    _successors = Block[]
    _attributes = NamedAttribute[
        namedattribute("sym_name", sym_name), namedattribute("function_type", function_type)
    ]
    !isnothing(arg_attrs) && push!(_attributes, namedattribute("arg_attrs", arg_attrs))
    !isnothing(res_attrs) && push!(_attributes, namedattribute("res_attrs", res_attrs))

    return IR.create_operation(
        "pdl_interp.func",
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
`get_attribute`

`pdl_interp.get_attribute` operations try to get a specific attribute from
an operation. If the operation does not have that attribute, a null value is
returned.

# Example

```mlir
%attr = pdl_interp.get_attribute \"attr\" of %op
```
"""
function get_attribute(inputOp; attribute::IR.Type, name, location=Location())
    results = IR.Type[attribute, ]
    operands = Value[value(inputOp), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("name", name), ]
    
    IR.create_operation(
        "pdl_interp.get_attribute", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`get_attribute_type`

`pdl_interp.get_attribute_type` operations get the resulting type of a
specific attribute.

# Example

```mlir
%type = pdl_interp.get_attribute_type of %attr
```
"""
function get_attribute_type(value; result::IR.Type, location=Location())
    results = IR.Type[result, ]
    operands = Value[value(value), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "pdl_interp.get_attribute_type", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`get_defining_op`

`pdl_interp.get_defining_op` operations try to get the defining operation
of a specific value or range of values. In the case of range, the defining
op of the first value is returned. If the value is not an operation result
or range of operand results, null is returned.

# Example

```mlir
%op = pdl_interp.get_defining_op of %value : !pdl.value
```
"""
function get_defining_op(value; inputOp::IR.Type, location=Location())
    results = IR.Type[inputOp, ]
    operands = Value[value(value), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "pdl_interp.get_defining_op", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`get_operand`

`pdl_interp.get_operand` operations try to get a specific operand from an
operation If the operation does not have an operand for the given index, a
null value is returned.

# Example

```mlir
%operand = pdl_interp.get_operand 1 of %op
```
"""
function get_operand(inputOp; value::IR.Type, index, location=Location())
    results = IR.Type[value, ]
    operands = Value[value(inputOp), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("index", index), ]
    
    IR.create_operation(
        "pdl_interp.get_operand", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`get_operands`

`pdl_interp.get_operands` operations try to get a specific operand
group from an operation. If the expected result is a single Value, null is
returned if the operand group is not of size 1. If a range is expected,
null is returned if the operand group is invalid. If no index is provided,
the returned operand group corresponds to all operands of the operation.

# Example

```mlir
// Get the first group of operands from an operation, and expect a single
// element.
%operand = pdl_interp.get_operands 0 of %op : !pdl.value

// Get the first group of operands from an operation.
%operands = pdl_interp.get_operands 0 of %op : !pdl.range<value>

// Get all of the operands from an operation.
%operands = pdl_interp.get_operands of %op : !pdl.range<value>
```
"""
function get_operands(inputOp; value::IR.Type, index=nothing, location=Location())
    results = IR.Type[value, ]
    operands = Value[value(inputOp), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(index) && push!(attributes, namedattribute("index", index))
    
    IR.create_operation(
        "pdl_interp.get_operands", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`get_result`

`pdl_interp.get_result` operations try to get a specific result from an
operation. If the operation does not have a result for the given index, a
null value is returned.

# Example

```mlir
%result = pdl_interp.get_result 1 of %op
```
"""
function get_result(inputOp; value::IR.Type, index, location=Location())
    results = IR.Type[value, ]
    operands = Value[value(inputOp), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("index", index), ]
    
    IR.create_operation(
        "pdl_interp.get_result", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`get_results`

`pdl_interp.get_results` operations try to get a specific result group
from an operation. If the expected result is a single Value, null is
returned if the result group is not of size 1. If a range is expected,
null is returned if the result group is invalid. If no index is provided,
the returned operand group corresponds to all results of the operation.

# Example

```mlir
// Get the first group of results from an operation, and expect a single
// element.
%result = pdl_interp.get_results 0 of %op : !pdl.value

// Get the first group of results from an operation.
%results = pdl_interp.get_results 0 of %op : !pdl.range<value>

// Get all of the results from an operation.
%results = pdl_interp.get_results of %op : !pdl.range<value>
```
"""
function get_results(inputOp; value::IR.Type, index=nothing, location=Location())
    results = IR.Type[value, ]
    operands = Value[value(inputOp), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(index) && push!(attributes, namedattribute("index", index))
    
    IR.create_operation(
        "pdl_interp.get_results", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`get_users`

`pdl_interp.get_users` extracts the users that accept this value. In the
case of a range, the union of users of the all the values are returned,
similarly to ResultRange::getUsers.

# Example

```mlir
// Get all the users of a single value.
%ops = pdl_interp.get_users of %value : !pdl.value

// Get all the users of the first value in a range.
%ops = pdl_interp.get_users of %values : !pdl.range<value>
```
"""
function get_users(value; operations::IR.Type, location=Location())
    results = IR.Type[operations, ]
    operands = Value[value(value), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "pdl_interp.get_users", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`get_value_type`

`pdl_interp.get_value_type` operations get the resulting type of a specific
value or range thereof.

# Example

```mlir
// Get the type of a single value.
%type = pdl_interp.get_value_type of %value : !pdl.type

// Get the types of a value range.
%type = pdl_interp.get_value_type of %values : !pdl.range<type>
```
"""
function get_value_type(value; result::IR.Type, location=Location())
    results = IR.Type[result, ]
    operands = Value[value(value), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "pdl_interp.get_value_type", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`is_not_null`

`pdl_interp.is_not_null` operations check that a positional value or range
exists. For ranges, this does not mean that the range was simply empty. On
success, this operation branches to the true destination. Otherwise, the
false destination is taken.

# Example

```mlir
pdl_interp.is_not_null %value : !pdl.value -> ^matchDest, ^failureDest
```
"""
function is_not_null(value; trueDest::Block, falseDest::Block, location=Location())
    results = IR.Type[]
    operands = Value[value(value), ]
    owned_regions = Region[]
    successors = Block[trueDest, falseDest, ]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "pdl_interp.is_not_null", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`record_match`

`pdl_interp.record_match` operations record a successful pattern match with
the interpreter and branch to the next part of the matcher. The metadata
recorded by these operations correspond to a specific `pdl.pattern`, as well
as what values were used during that match that should be propagated to the
rewriter.

# Example

```mlir
pdl_interp.record_match @rewriters::myRewriter(%root : !pdl.operation) : benefit(1), loc([%root, %op1]), root(\"foo.op\") -> ^nextDest
```
"""
function record_match(inputs, matchedOps; rewriter, rootKind=nothing, generatedOps=nothing, benefit, dest::Block, location=Location())
    results = IR.Type[]
    operands = Value[value.(inputs)..., value.(matchedOps)..., ]
    owned_regions = Region[]
    successors = Block[dest, ]
    attributes = NamedAttribute[namedattribute("rewriter", rewriter), namedattribute("benefit", benefit), ]
    push!(attributes, operandsegmentsizes([length(inputs), length(matchedOps), ]))
    !isnothing(rootKind) && push!(attributes, namedattribute("rootKind", rootKind))
    !isnothing(generatedOps) && push!(attributes, namedattribute("generatedOps", generatedOps))
    
    IR.create_operation(
        "pdl_interp.record_match", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`replace`

`pdl_interp.replaced` operations are used to specify that an operation
should be marked as replaced. The semantics of this operation correspond
with the `replaceOp` method on a `PatternRewriter`. The set of replacement
values must match the number of results specified by the operation.

# Example

```mlir
// Replace root node with 2 values:
pdl_interp.replace %root with (%val0, %val1 : !pdl.type, !pdl.type)
```
"""
function replace(inputOp, replValues; location=Location())
    results = IR.Type[]
    operands = Value[value(inputOp), value.(replValues)..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "pdl_interp.replace", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`switch_attribute`

`pdl_interp.switch_attribute` operations compare the value of a given
attribute with a set of constant attributes. If the value matches one of the
provided case values the destination for that case value is taken, otherwise
the default destination is taken.

# Example

```mlir
pdl_interp.switch_attribute %attr to [10, true](^10Dest, ^trueDest) -> ^defaultDest
```
"""
function switch_attribute(attribute; caseValues, defaultDest::Block, cases::Vector{Block}, location=Location())
    results = IR.Type[]
    operands = Value[value(attribute), ]
    owned_regions = Region[]
    successors = Block[defaultDest, cases..., ]
    attributes = NamedAttribute[namedattribute("caseValues", caseValues), ]
    
    IR.create_operation(
        "pdl_interp.switch_attribute", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`switch_operand_count`

`pdl_interp.switch_operand_count` operations compare the operand count of a
given operation with a set of potential counts. If the value matches one of
the provided case values the destination for that case value is taken,
otherwise the default destination is taken.

# Example

```mlir
pdl_interp.switch_operand_count of %op to [10, 2] -> ^10Dest, ^2Dest, ^defaultDest
```
"""
function switch_operand_count(inputOp; caseValues, defaultDest::Block, cases::Vector{Block}, location=Location())
    results = IR.Type[]
    operands = Value[value(inputOp), ]
    owned_regions = Region[]
    successors = Block[defaultDest, cases..., ]
    attributes = NamedAttribute[namedattribute("caseValues", caseValues), ]
    
    IR.create_operation(
        "pdl_interp.switch_operand_count", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`switch_operation_name`

`pdl_interp.switch_operation_name` operations compare the name of a given
operation with a set of known names. If the value matches one of the
provided case values the destination for that case value is taken, otherwise
the default destination is taken.

# Example

```mlir
pdl_interp.switch_operation_name of %op to [\"foo.op\", \"bar.op\"](^fooDest, ^barDest) -> ^defaultDest
```
"""
function switch_operation_name(inputOp; caseValues, defaultDest::Block, cases::Vector{Block}, location=Location())
    results = IR.Type[]
    operands = Value[value(inputOp), ]
    owned_regions = Region[]
    successors = Block[defaultDest, cases..., ]
    attributes = NamedAttribute[namedattribute("caseValues", caseValues), ]
    
    IR.create_operation(
        "pdl_interp.switch_operation_name", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`switch_result_count`

`pdl_interp.switch_result_count` operations compare the result count of a
given operation with a set of potential counts. If the value matches one of
the provided case values the destination for that case value is taken,
otherwise the default destination is taken.

# Example

```mlir
pdl_interp.switch_result_count of %op to [0, 2](^0Dest, ^2Dest) -> ^defaultDest
```
"""
function switch_result_count(inputOp; caseValues, defaultDest::Block, cases::Vector{Block}, location=Location())
    results = IR.Type[]
    operands = Value[value(inputOp), ]
    owned_regions = Region[]
    successors = Block[defaultDest, cases..., ]
    attributes = NamedAttribute[namedattribute("caseValues", caseValues), ]
    
    IR.create_operation(
        "pdl_interp.switch_result_count", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`switch_type`

`pdl_interp.switch_type` operations compare a type with a set of statically
known types. If the value matches one of the provided case values the
destination for that case value is taken, otherwise the default destination
is taken.

# Example

```mlir
pdl_interp.switch_type %type to [i32, i64] -> ^i32Dest, ^i64Dest, ^defaultDest
```
"""
function switch_type(value; caseValues, defaultDest::Block, cases::Vector{Block}, location=Location())
    results = IR.Type[]
    operands = Value[value(value), ]
    owned_regions = Region[]
    successors = Block[defaultDest, cases..., ]
    attributes = NamedAttribute[namedattribute("caseValues", caseValues), ]
    
    IR.create_operation(
        "pdl_interp.switch_type", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`switch_types`

`pdl_interp.switch_types` operations compare a range of types with a set of
statically known ranges. If the value matches one of the provided case
values the destination for that case value is taken, otherwise the default
destination is taken.

# Example

```mlir
pdl_interp.switch_types %type is [[i32], [i64, i64]] -> ^i32Dest, ^i64Dest, ^defaultDest
```
"""
function switch_types(value; caseValues, defaultDest::Block, cases::Vector{Block}, location=Location())
    results = IR.Type[]
    operands = Value[value(value), ]
    owned_regions = Region[]
    successors = Block[defaultDest, cases..., ]
    attributes = NamedAttribute[namedattribute("caseValues", caseValues), ]
    
    IR.create_operation(
        "pdl_interp.switch_types", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

end # pdl_interp
