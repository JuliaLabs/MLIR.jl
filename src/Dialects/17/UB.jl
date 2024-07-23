module ub

import ...IR:
    IR, NamedAttribute, Value, Location, Block, Region, Attribute, context, IndexType
import ..Dialects: namedattribute, operandsegmentsizes

"""
`poison`

The `poison` operation materializes a compile-time poisoned constant value
to indicate deferred undefined behavior.
`value` attirbute is needed to indicate an optional additional poison
semantics (e.g. partially poisoned vectors), default value indicates results
is fully poisoned.

# Syntax

```
poison-op ::= `poison` (`<` value `>`)? `:` type
```

Examples:

```
// Short form
%0 = ub.poison : i32
// Long form
%1 = ub.poison <#custom_poison_elements_attr> : vector<4xi64>
```
"""
function poison(; result::IR.Type, value=nothing, location=Location())
    _results = IR.Type[result,]
    _operands = Value[]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(value) && push!(_attributes, namedattribute("value", value))

    return IR.create_operation(
        "ub.poison",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

end # ub
