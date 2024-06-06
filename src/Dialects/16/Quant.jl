module quant

import ...IR: IR, NamedAttribute, Value, Location, Block, Region, Attribute, context, IndexType
import ..Dialects: namedattribute, operandsegmentsizes


"""
`dcast`

"""
function dcast(arg::Value; result_0::IR.Type, location=Location())
    results = IR.Type[result_0, ]
    operands = Value[arg, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "quant.dcast", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`qcast`

"""
function qcast(arg::Value; result_0::IR.Type, location=Location())
    results = IR.Type[result_0, ]
    operands = Value[arg, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "quant.qcast", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`scast`

"""
function scast(arg::Value; result_0::IR.Type, location=Location())
    results = IR.Type[result_0, ]
    operands = Value[arg, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "quant.scast", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

end # quant
