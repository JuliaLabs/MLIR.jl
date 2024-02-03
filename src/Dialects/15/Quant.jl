module quant

import ...IR: NamedAttribute, MLIRType, get_value, Location, Block, Region, Attribute, create_operation, context, IndexType
import ..Dialects: namedattribute, operandsegmentsizes
import ...API


"""
`const_fake_quant`

Given a const min, max, num_bits and narrow_range attribute, applies the
same uniform quantization simulation as is done by the TensorFlow
fake_quant_with_min_max_args op. See the fakeQuantAttrsToType() utility
method and the quant-convert-simulated-quantization pass for further details.
"""
function const_fake_quant(inputs; outputs=nothing::Union{Nothing, MLIRType}, min, max, num_bits, narrow_range=nothing, is_signed=nothing, location=Location())
    results = MLIRType[]
    operands = API.MlirValue[get_value(inputs), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("min", min), namedattribute("max", max), namedattribute("num_bits", num_bits), ]
    (outputs != nothing) && push!(results, outputs)
    (narrow_range != nothing) && push!(attributes, namedattribute("narrow_range", narrow_range))
    (is_signed != nothing) && push!(attributes, namedattribute("is_signed", is_signed))
    
    create_operation(
        "quant.const_fake_quant", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`const_fake_quant_per_axis`

Given a const min, max, num_bits and narrow_range attribute, applies the
same per axis uniform quantization simulation as is done by the TensorFlow
fake_quant_with_min_max_vars_per_channel op. See the fakeQuantAttrsToType()
utility method and the quant-convert-simulated-quantization pass for further
details.
"""
function const_fake_quant_per_axis(inputs; outputs=nothing::Union{Nothing, MLIRType}, min, max, axis, num_bits, narrow_range=nothing, is_signed=nothing, location=Location())
    results = MLIRType[]
    operands = API.MlirValue[get_value(inputs), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("min", min), namedattribute("max", max), namedattribute("axis", axis), namedattribute("num_bits", num_bits), ]
    (outputs != nothing) && push!(results, outputs)
    (narrow_range != nothing) && push!(attributes, namedattribute("narrow_range", narrow_range))
    (is_signed != nothing) && push!(attributes, namedattribute("is_signed", is_signed))
    
    create_operation(
        "quant.const_fake_quant_per_axis", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`coupled_ref`

Ordinarily, relationships between ops for the purposes of determining
compatible quantized types is explicit based on the use-def chain. However,
in some situations, a use may be separated from its def by arbitrary
external connections. In such a case, during analysis, all coupled_ref
nodes in a module which share a coupledKey will be considered to be
directly connected as via an identity op for the purpose of type inference.
"""
function coupled_ref(arg; result_0=nothing::Union{Nothing, MLIRType}, coupledKey, location=Location())
    results = MLIRType[]
    operands = API.MlirValue[get_value(arg), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("coupledKey", coupledKey), ]
    (result_0 != nothing) && push!(results, result_0)
    
    create_operation(
        "quant.coupled_ref", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`dcast`

"""
function dcast(arg; result_0::MLIRType, location=Location())
    results = MLIRType[result_0, ]
    operands = API.MlirValue[get_value(arg), ]
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
function qcast(arg; result_0::MLIRType, location=Location())
    results = MLIRType[result_0, ]
    operands = API.MlirValue[get_value(arg), ]
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
`region`

"""
function region(inputs; outputs::Vector{MLIRType}, input_specs, output_specs, logical_kernel, body::Region, location=Location())
    results = MLIRType[outputs..., ]
    operands = API.MlirValue[get_value.(inputs)..., ]
    owned_regions = Region[body, ]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("input_specs", input_specs), namedattribute("output_specs", output_specs), namedattribute("logical_kernel", logical_kernel), ]
    
    create_operation(
        "quant.region", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`return_`

"""
function return_(results; location=Location())
    results = MLIRType[]
    operands = API.MlirValue[get_value.(results)..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "quant.return", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`stats`

Associates statistics about the runtime ranges of values observed for
evaluations of this node.

Statistics about the entire type are reported in the \'layerStats\' attribute
and those for each axis, in the (optional) `axisStats` attribute. The
interpretation of each is determined by the last dimension of its shape.
Currently, only dim=2 is supported, which is interpreted as [min, max].

`layerStats` must be a rank 1 tensor: [2]
`axisStats` must be a rank 2 tensor: [N, 2], where N=the slice size
  splitted by the `axis` dimension. For example:

```
<?x?x3x2>, axis=3 => N=2
<?x?x3x2>, axis=2 => N=6
```
"""
function stats(arg; result_0=nothing::Union{Nothing, MLIRType}, layerStats, axisStats=nothing, axis=nothing, location=Location())
    results = MLIRType[]
    operands = API.MlirValue[get_value(arg), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("layerStats", layerStats), ]
    (result_0 != nothing) && push!(results, result_0)
    (axisStats != nothing) && push!(attributes, namedattribute("axisStats", axisStats))
    (axis != nothing) && push!(attributes, namedattribute("axis", axis))
    
    create_operation(
        "quant.stats", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`stats_ref`

This op acts as an identity that, when encountered at runtime, should result
in statistics being collected about about the value of its operand/result.
Such statistics will be stored with the provided key, allowing this node
to later be converted to a \'stats\' op if statistics with that key have been
encountered.
"""
function stats_ref(arg; result_0=nothing::Union{Nothing, MLIRType}, statsKey, location=Location())
    results = MLIRType[]
    operands = API.MlirValue[get_value(arg), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("statsKey", statsKey), ]
    (result_0 != nothing) && push!(results, result_0)
    
    create_operation(
        "quant.stats_ref", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`scast`

"""
function scast(arg; result_0::MLIRType, location=Location())
    results = MLIRType[result_0, ]
    operands = API.MlirValue[get_value(arg), ]
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
