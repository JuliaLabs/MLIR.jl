module quant

import ...IR: NamedAttribute, MLIRType, Value, Location, Block, Region, Attribute, create_operation, context, IndexType
import ..Dialects: namedattribute, operandsegmentsizes
import ...API


"""
`dcast`

"""
function dcast(arg::Value; result_0::MLIRType, location=Location())
    results = MLIRType[result_0, ]
    operands = Value[arg, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "quant.dcast", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`qcast`

"""
function qcast(arg::Value; result_0::MLIRType, location=Location())
    results = MLIRType[result_0, ]
    operands = Value[arg, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "quant.qcast", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`scast`

"""
function scast(arg::Value; result_0::MLIRType, location=Location())
    results = MLIRType[result_0, ]
    operands = Value[arg, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "quant.scast", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

end # quant
