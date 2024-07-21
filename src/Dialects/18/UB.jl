module ub

import ...IR:
    IR, NamedAttribute, Value, Location, Block, Region, Attribute, context, IndexType
import ..Dialects: namedattribute, operandsegmentsizes

"""
`poison`

The `poison` operation materializes a compile-time poisoned constant value
to indicate deferred undefined behavior.
`value` attribute is needed to indicate an optional additional poison
semantics (e.g. partially poisoned vectors), default value indicates results
is fully poisoned.

Examples:

```
// Short form
%0 = ub.poison : i32
// Long form
%1 = ub.poison <#custom_poison_elements_attr> : vector<4xi64>
```
"""
function poison(; result::IR.Type, value=nothing, location=Location())
    results = IR.Type[result,]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(value) && push!(attributes, namedattribute("value", value))

    return IR.create_operation(
        "ub.poison",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=results,
        result_inference=false,
    )
end

end # ub
