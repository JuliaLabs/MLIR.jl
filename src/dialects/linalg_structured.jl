module linalg

import ...IR: NamedAttribute, MLIRType, Value, Location, Block, Region, Attribute, create_operation, context, IndexType
import ..Dialects: namedattribute, operandsegmentsizes
import ...API


"""
`abs`

No numeric casting is performed on the input operand.
"""
function abs(inputs::Vector{Value}, outputs::Vector{Value}; result_tensors::Vector{MLIRType}, region::Region, location=Location())
    results = MLIRType[result_tensors..., ]
    operands = Value[inputs..., outputs..., ]
    owned_regions = Region[region, ]
    successors = Block[]
    attributes = NamedAttribute[]
    push!(attributes, operandsegmentsizes([length(inputs), length(outputs), ]))
    
    create_operation(
        "linalg.abs", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`add`

The shapes and element types must be identical. The appropriate casts,
broadcasts and reductions should be done previously to calling this op.

This means reduction/broadcast/element cast semantics is explicit. Further
passes can take that into account when lowering this code. For example,
a `linalg.broadcast` + `linalg.add` sequence can be lowered to a
`linalg.generic` with different affine maps for the two operands.
"""
function add(inputs::Vector{Value}, outputs::Vector{Value}; result_tensors::Vector{MLIRType}, region::Region, location=Location())
    results = MLIRType[result_tensors..., ]
    operands = Value[inputs..., outputs..., ]
    owned_regions = Region[region, ]
    successors = Block[]
    attributes = NamedAttribute[]
    push!(attributes, operandsegmentsizes([length(inputs), length(outputs), ]))
    
    create_operation(
        "linalg.add", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`batch_matmul`

Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output.
"""
function batch_matmul(inputs::Vector{Value}, outputs::Vector{Value}; result_tensors::Vector{MLIRType}, region::Region, location=Location())
    results = MLIRType[result_tensors..., ]
    operands = Value[inputs..., outputs..., ]
    owned_regions = Region[region, ]
    successors = Block[]
    attributes = NamedAttribute[]
    push!(attributes, operandsegmentsizes([length(inputs), length(outputs), ]))
    
    create_operation(
        "linalg.batch_matmul", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`batch_matmul_transpose_a`

has its non-batch dimensions transposed.

Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output.
"""
function batch_matmul_transpose_a(inputs::Vector{Value}, outputs::Vector{Value}; result_tensors::Vector{MLIRType}, region::Region, location=Location())
    results = MLIRType[result_tensors..., ]
    operands = Value[inputs..., outputs..., ]
    owned_regions = Region[region, ]
    successors = Block[]
    attributes = NamedAttribute[]
    push!(attributes, operandsegmentsizes([length(inputs), length(outputs), ]))
    
    create_operation(
        "linalg.batch_matmul_transpose_a", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`batch_matmul_transpose_b`

has its non-batch dimensions transposed.

Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output.
"""
function batch_matmul_transpose_b(inputs::Vector{Value}, outputs::Vector{Value}; result_tensors::Vector{MLIRType}, region::Region, location=Location())
    results = MLIRType[result_tensors..., ]
    operands = Value[inputs..., outputs..., ]
    owned_regions = Region[region, ]
    successors = Block[]
    attributes = NamedAttribute[]
    push!(attributes, operandsegmentsizes([length(inputs), length(outputs), ]))
    
    create_operation(
        "linalg.batch_matmul_transpose_b", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`batch_matvec`

Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output.
"""
function batch_matvec(inputs::Vector{Value}, outputs::Vector{Value}; result_tensors::Vector{MLIRType}, region::Region, location=Location())
    results = MLIRType[result_tensors..., ]
    operands = Value[inputs..., outputs..., ]
    owned_regions = Region[region, ]
    successors = Block[]
    attributes = NamedAttribute[]
    push!(attributes, operandsegmentsizes([length(inputs), length(outputs), ]))
    
    create_operation(
        "linalg.batch_matvec", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`batch_reduce_matmul`

The partial multiplication results are reduced into a 2D output.

Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output.
"""
function batch_reduce_matmul(inputs::Vector{Value}, outputs::Vector{Value}; result_tensors::Vector{MLIRType}, region::Region, location=Location())
    results = MLIRType[result_tensors..., ]
    operands = Value[inputs..., outputs..., ]
    owned_regions = Region[region, ]
    successors = Block[]
    attributes = NamedAttribute[]
    push!(attributes, operandsegmentsizes([length(inputs), length(outputs), ]))
    
    create_operation(
        "linalg.batch_reduce_matmul", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`broadcast`

Broadcast the input into the given shape by adding `dimensions`.

# Example
```
  %bcast = linalg.broadcast
      ins(%input:tensor<16xf32>)
      inits(%init:tensor<16x64xf32>)
      dimensions = [1]
```
"""
function broadcast(input::Value, init::Value; result::Vector{MLIRType}, dimensions, region::Region, location=Location())
    results = MLIRType[result..., ]
    operands = Value[input, init, ]
    owned_regions = Region[region, ]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("dimensions", dimensions), ]
    
    create_operation(
        "linalg.broadcast", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`ceil`

No numeric casting is performed on the input operand.
"""
function ceil(inputs::Vector{Value}, outputs::Vector{Value}; result_tensors::Vector{MLIRType}, region::Region, location=Location())
    results = MLIRType[result_tensors..., ]
    operands = Value[inputs..., outputs..., ]
    owned_regions = Region[region, ]
    successors = Block[]
    attributes = NamedAttribute[]
    push!(attributes, operandsegmentsizes([length(inputs), length(outputs), ]))
    
    create_operation(
        "linalg.ceil", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`conv_1d_ncw_fcw`

Layout:
  * Input: NCW.
  * Kernel: FCW.

Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output.
"""
function conv_1d_ncw_fcw(inputs::Vector{Value}, outputs::Vector{Value}; result_tensors::Vector{MLIRType}, strides=nothing, dilations=nothing, region::Region, location=Location())
    results = MLIRType[result_tensors..., ]
    operands = Value[inputs..., outputs..., ]
    owned_regions = Region[region, ]
    successors = Block[]
    attributes = NamedAttribute[]
    push!(attributes, operandsegmentsizes([length(inputs), length(outputs), ]))
    (strides != nothing) && push!(attributes, namedattribute("strides", strides))
    (dilations != nothing) && push!(attributes, namedattribute("dilations", dilations))
    
    create_operation(
        "linalg.conv_1d_ncw_fcw", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`conv_1d_nwc_wcf`

Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output.
"""
function conv_1d_nwc_wcf(inputs::Vector{Value}, outputs::Vector{Value}; result_tensors::Vector{MLIRType}, strides=nothing, dilations=nothing, region::Region, location=Location())
    results = MLIRType[result_tensors..., ]
    operands = Value[inputs..., outputs..., ]
    owned_regions = Region[region, ]
    successors = Block[]
    attributes = NamedAttribute[]
    push!(attributes, operandsegmentsizes([length(inputs), length(outputs), ]))
    (strides != nothing) && push!(attributes, namedattribute("strides", strides))
    (dilations != nothing) && push!(attributes, namedattribute("dilations", dilations))
    
    create_operation(
        "linalg.conv_1d_nwc_wcf", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`conv_1d`

Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output.
"""
function conv_1d(inputs::Vector{Value}, outputs::Vector{Value}; result_tensors::Vector{MLIRType}, region::Region, location=Location())
    results = MLIRType[result_tensors..., ]
    operands = Value[inputs..., outputs..., ]
    owned_regions = Region[region, ]
    successors = Block[]
    attributes = NamedAttribute[]
    push!(attributes, operandsegmentsizes([length(inputs), length(outputs), ]))
    
    create_operation(
        "linalg.conv_1d", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`conv_2d_nchw_fchw`

Layout:
  * Input: NCHW.
  * Kernel: FCHW.

Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output.
"""
function conv_2d_nchw_fchw(inputs::Vector{Value}, outputs::Vector{Value}; result_tensors::Vector{MLIRType}, strides=nothing, dilations=nothing, region::Region, location=Location())
    results = MLIRType[result_tensors..., ]
    operands = Value[inputs..., outputs..., ]
    owned_regions = Region[region, ]
    successors = Block[]
    attributes = NamedAttribute[]
    push!(attributes, operandsegmentsizes([length(inputs), length(outputs), ]))
    (strides != nothing) && push!(attributes, namedattribute("strides", strides))
    (dilations != nothing) && push!(attributes, namedattribute("dilations", dilations))
    
    create_operation(
        "linalg.conv_2d_nchw_fchw", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`conv_2d_ngchw_fgchw`

Layout:
  * Input: NGCHW.
  * Kernel: FGCHW.

Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output.
"""
function conv_2d_ngchw_fgchw(inputs::Vector{Value}, outputs::Vector{Value}; result_tensors::Vector{MLIRType}, strides=nothing, dilations=nothing, region::Region, location=Location())
    results = MLIRType[result_tensors..., ]
    operands = Value[inputs..., outputs..., ]
    owned_regions = Region[region, ]
    successors = Block[]
    attributes = NamedAttribute[]
    push!(attributes, operandsegmentsizes([length(inputs), length(outputs), ]))
    (strides != nothing) && push!(attributes, namedattribute("strides", strides))
    (dilations != nothing) && push!(attributes, namedattribute("dilations", dilations))
    
    create_operation(
        "linalg.conv_2d_ngchw_fgchw", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`conv_2d_nhwc_fhwc`

Layout:
  * Input: NHWC.
  * Kernel: FHWC.

Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output.
"""
function conv_2d_nhwc_fhwc(inputs::Vector{Value}, outputs::Vector{Value}; result_tensors::Vector{MLIRType}, strides=nothing, dilations=nothing, region::Region, location=Location())
    results = MLIRType[result_tensors..., ]
    operands = Value[inputs..., outputs..., ]
    owned_regions = Region[region, ]
    successors = Block[]
    attributes = NamedAttribute[]
    push!(attributes, operandsegmentsizes([length(inputs), length(outputs), ]))
    (strides != nothing) && push!(attributes, namedattribute("strides", strides))
    (dilations != nothing) && push!(attributes, namedattribute("dilations", dilations))
    
    create_operation(
        "linalg.conv_2d_nhwc_fhwc", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`conv_2d_nhwc_hwcf`

Layout:
  * Input: NHWC.
  * Kernel: HWCF.

Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output.
"""
function conv_2d_nhwc_hwcf(inputs::Vector{Value}, outputs::Vector{Value}; result_tensors::Vector{MLIRType}, strides=nothing, dilations=nothing, region::Region, location=Location())
    results = MLIRType[result_tensors..., ]
    operands = Value[inputs..., outputs..., ]
    owned_regions = Region[region, ]
    successors = Block[]
    attributes = NamedAttribute[]
    push!(attributes, operandsegmentsizes([length(inputs), length(outputs), ]))
    (strides != nothing) && push!(attributes, namedattribute("strides", strides))
    (dilations != nothing) && push!(attributes, namedattribute("dilations", dilations))
    
    create_operation(
        "linalg.conv_2d_nhwc_hwcf", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`conv_2d_nhwc_hwcf_q`

Layout:
  * Input: NHWC.
  * Kernel: HWCF.

Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output. This includes the zero
point offsets common to quantized operations.
"""
function conv_2d_nhwc_hwcf_q(inputs::Vector{Value}, outputs::Vector{Value}; result_tensors::Vector{MLIRType}, strides=nothing, dilations=nothing, region::Region, location=Location())
    results = MLIRType[result_tensors..., ]
    operands = Value[inputs..., outputs..., ]
    owned_regions = Region[region, ]
    successors = Block[]
    attributes = NamedAttribute[]
    push!(attributes, operandsegmentsizes([length(inputs), length(outputs), ]))
    (strides != nothing) && push!(attributes, namedattribute("strides", strides))
    (dilations != nothing) && push!(attributes, namedattribute("dilations", dilations))
    
    create_operation(
        "linalg.conv_2d_nhwc_hwcf_q", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`conv_2d`

Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output.
"""
function conv_2d(inputs::Vector{Value}, outputs::Vector{Value}; result_tensors::Vector{MLIRType}, region::Region, location=Location())
    results = MLIRType[result_tensors..., ]
    operands = Value[inputs..., outputs..., ]
    owned_regions = Region[region, ]
    successors = Block[]
    attributes = NamedAttribute[]
    push!(attributes, operandsegmentsizes([length(inputs), length(outputs), ]))
    
    create_operation(
        "linalg.conv_2d", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`conv_3d_ncdhw_fcdhw`

Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output.
"""
function conv_3d_ncdhw_fcdhw(inputs::Vector{Value}, outputs::Vector{Value}; result_tensors::Vector{MLIRType}, strides=nothing, dilations=nothing, region::Region, location=Location())
    results = MLIRType[result_tensors..., ]
    operands = Value[inputs..., outputs..., ]
    owned_regions = Region[region, ]
    successors = Block[]
    attributes = NamedAttribute[]
    push!(attributes, operandsegmentsizes([length(inputs), length(outputs), ]))
    (strides != nothing) && push!(attributes, namedattribute("strides", strides))
    (dilations != nothing) && push!(attributes, namedattribute("dilations", dilations))
    
    create_operation(
        "linalg.conv_3d_ncdhw_fcdhw", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`conv_3d_ndhwc_dhwcf`

Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output.
"""
function conv_3d_ndhwc_dhwcf(inputs::Vector{Value}, outputs::Vector{Value}; result_tensors::Vector{MLIRType}, strides=nothing, dilations=nothing, region::Region, location=Location())
    results = MLIRType[result_tensors..., ]
    operands = Value[inputs..., outputs..., ]
    owned_regions = Region[region, ]
    successors = Block[]
    attributes = NamedAttribute[]
    push!(attributes, operandsegmentsizes([length(inputs), length(outputs), ]))
    (strides != nothing) && push!(attributes, namedattribute("strides", strides))
    (dilations != nothing) && push!(attributes, namedattribute("dilations", dilations))
    
    create_operation(
        "linalg.conv_3d_ndhwc_dhwcf", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`conv_3d_ndhwc_dhwcf_q`

Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output. This includes the zero
point offsets common to quantized operations.
"""
function conv_3d_ndhwc_dhwcf_q(inputs::Vector{Value}, outputs::Vector{Value}; result_tensors::Vector{MLIRType}, strides=nothing, dilations=nothing, region::Region, location=Location())
    results = MLIRType[result_tensors..., ]
    operands = Value[inputs..., outputs..., ]
    owned_regions = Region[region, ]
    successors = Block[]
    attributes = NamedAttribute[]
    push!(attributes, operandsegmentsizes([length(inputs), length(outputs), ]))
    (strides != nothing) && push!(attributes, namedattribute("strides", strides))
    (dilations != nothing) && push!(attributes, namedattribute("dilations", dilations))
    
    create_operation(
        "linalg.conv_3d_ndhwc_dhwcf_q", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`conv_3d`

Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output.
"""
function conv_3d(inputs::Vector{Value}, outputs::Vector{Value}; result_tensors::Vector{MLIRType}, region::Region, location=Location())
    results = MLIRType[result_tensors..., ]
    operands = Value[inputs..., outputs..., ]
    owned_regions = Region[region, ]
    successors = Block[]
    attributes = NamedAttribute[]
    push!(attributes, operandsegmentsizes([length(inputs), length(outputs), ]))
    
    create_operation(
        "linalg.conv_3d", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`copy`

Numeric casting is performed on the input operand, promoting it to the same
data type as the accumulator/output.
"""
function copy(inputs::Vector{Value}, outputs::Vector{Value}; result_tensors::Vector{MLIRType}, cast=nothing, region::Region, location=Location())
    results = MLIRType[result_tensors..., ]
    operands = Value[inputs..., outputs..., ]
    owned_regions = Region[region, ]
    successors = Block[]
    attributes = NamedAttribute[]
    push!(attributes, operandsegmentsizes([length(inputs), length(outputs), ]))
    (cast != nothing) && push!(attributes, namedattribute("cast", cast))
    
    create_operation(
        "linalg.copy", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`depthwise_conv_1d_ncw_cw`

Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output. Multiplier is set to 1
which is a special case for most depthwise convolutions.
"""
function depthwise_conv_1d_ncw_cw(inputs::Vector{Value}, outputs::Vector{Value}; result_tensors::Vector{MLIRType}, strides=nothing, dilations=nothing, region::Region, location=Location())
    results = MLIRType[result_tensors..., ]
    operands = Value[inputs..., outputs..., ]
    owned_regions = Region[region, ]
    successors = Block[]
    attributes = NamedAttribute[]
    push!(attributes, operandsegmentsizes([length(inputs), length(outputs), ]))
    (strides != nothing) && push!(attributes, namedattribute("strides", strides))
    (dilations != nothing) && push!(attributes, namedattribute("dilations", dilations))
    
    create_operation(
        "linalg.depthwise_conv_1d_ncw_cw", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`depthwise_conv_1d_nwc_wc`

Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output. Multiplier is set to 1
which is a special case for most depthwise convolutions.
"""
function depthwise_conv_1d_nwc_wc(inputs::Vector{Value}, outputs::Vector{Value}; result_tensors::Vector{MLIRType}, strides=nothing, dilations=nothing, region::Region, location=Location())
    results = MLIRType[result_tensors..., ]
    operands = Value[inputs..., outputs..., ]
    owned_regions = Region[region, ]
    successors = Block[]
    attributes = NamedAttribute[]
    push!(attributes, operandsegmentsizes([length(inputs), length(outputs), ]))
    (strides != nothing) && push!(attributes, namedattribute("strides", strides))
    (dilations != nothing) && push!(attributes, namedattribute("dilations", dilations))
    
    create_operation(
        "linalg.depthwise_conv_1d_nwc_wc", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`depthwise_conv_1d_nwc_wcm`

Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output.
"""
function depthwise_conv_1d_nwc_wcm(inputs::Vector{Value}, outputs::Vector{Value}; result_tensors::Vector{MLIRType}, strides=nothing, dilations=nothing, region::Region, location=Location())
    results = MLIRType[result_tensors..., ]
    operands = Value[inputs..., outputs..., ]
    owned_regions = Region[region, ]
    successors = Block[]
    attributes = NamedAttribute[]
    push!(attributes, operandsegmentsizes([length(inputs), length(outputs), ]))
    (strides != nothing) && push!(attributes, namedattribute("strides", strides))
    (dilations != nothing) && push!(attributes, namedattribute("dilations", dilations))
    
    create_operation(
        "linalg.depthwise_conv_1d_nwc_wcm", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`depthwise_conv_2d_nchw_chw`

Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output. Multiplier is set to 1
which is a special case for most depthwise convolutions.
"""
function depthwise_conv_2d_nchw_chw(inputs::Vector{Value}, outputs::Vector{Value}; result_tensors::Vector{MLIRType}, strides=nothing, dilations=nothing, region::Region, location=Location())
    results = MLIRType[result_tensors..., ]
    operands = Value[inputs..., outputs..., ]
    owned_regions = Region[region, ]
    successors = Block[]
    attributes = NamedAttribute[]
    push!(attributes, operandsegmentsizes([length(inputs), length(outputs), ]))
    (strides != nothing) && push!(attributes, namedattribute("strides", strides))
    (dilations != nothing) && push!(attributes, namedattribute("dilations", dilations))
    
    create_operation(
        "linalg.depthwise_conv_2d_nchw_chw", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`depthwise_conv_2d_nhwc_hwc`

Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output. Multiplier is set to 1
which is a special case for most depthwise convolutions.
"""
function depthwise_conv_2d_nhwc_hwc(inputs::Vector{Value}, outputs::Vector{Value}; result_tensors::Vector{MLIRType}, strides=nothing, dilations=nothing, region::Region, location=Location())
    results = MLIRType[result_tensors..., ]
    operands = Value[inputs..., outputs..., ]
    owned_regions = Region[region, ]
    successors = Block[]
    attributes = NamedAttribute[]
    push!(attributes, operandsegmentsizes([length(inputs), length(outputs), ]))
    (strides != nothing) && push!(attributes, namedattribute("strides", strides))
    (dilations != nothing) && push!(attributes, namedattribute("dilations", dilations))
    
    create_operation(
        "linalg.depthwise_conv_2d_nhwc_hwc", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`depthwise_conv_2d_nhwc_hwc_q`

Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output.
"""
function depthwise_conv_2d_nhwc_hwc_q(inputs::Vector{Value}, outputs::Vector{Value}; result_tensors::Vector{MLIRType}, strides=nothing, dilations=nothing, region::Region, location=Location())
    results = MLIRType[result_tensors..., ]
    operands = Value[inputs..., outputs..., ]
    owned_regions = Region[region, ]
    successors = Block[]
    attributes = NamedAttribute[]
    push!(attributes, operandsegmentsizes([length(inputs), length(outputs), ]))
    (strides != nothing) && push!(attributes, namedattribute("strides", strides))
    (dilations != nothing) && push!(attributes, namedattribute("dilations", dilations))
    
    create_operation(
        "linalg.depthwise_conv_2d_nhwc_hwc_q", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`depthwise_conv_2d_nhwc_hwcm`

Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output.
"""
function depthwise_conv_2d_nhwc_hwcm(inputs::Vector{Value}, outputs::Vector{Value}; result_tensors::Vector{MLIRType}, strides=nothing, dilations=nothing, region::Region, location=Location())
    results = MLIRType[result_tensors..., ]
    operands = Value[inputs..., outputs..., ]
    owned_regions = Region[region, ]
    successors = Block[]
    attributes = NamedAttribute[]
    push!(attributes, operandsegmentsizes([length(inputs), length(outputs), ]))
    (strides != nothing) && push!(attributes, namedattribute("strides", strides))
    (dilations != nothing) && push!(attributes, namedattribute("dilations", dilations))
    
    create_operation(
        "linalg.depthwise_conv_2d_nhwc_hwcm", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`depthwise_conv_2d_nhwc_hwcm_q`

Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output.
"""
function depthwise_conv_2d_nhwc_hwcm_q(inputs::Vector{Value}, outputs::Vector{Value}; result_tensors::Vector{MLIRType}, strides=nothing, dilations=nothing, region::Region, location=Location())
    results = MLIRType[result_tensors..., ]
    operands = Value[inputs..., outputs..., ]
    owned_regions = Region[region, ]
    successors = Block[]
    attributes = NamedAttribute[]
    push!(attributes, operandsegmentsizes([length(inputs), length(outputs), ]))
    (strides != nothing) && push!(attributes, namedattribute("strides", strides))
    (dilations != nothing) && push!(attributes, namedattribute("dilations", dilations))
    
    create_operation(
        "linalg.depthwise_conv_2d_nhwc_hwcm_q", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`depthwise_conv_3d_ncdhw_cdhw`

Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output. Multiplier is set to 1
which is a special case for most depthwise convolutions.
"""
function depthwise_conv_3d_ncdhw_cdhw(inputs::Vector{Value}, outputs::Vector{Value}; result_tensors::Vector{MLIRType}, strides=nothing, dilations=nothing, region::Region, location=Location())
    results = MLIRType[result_tensors..., ]
    operands = Value[inputs..., outputs..., ]
    owned_regions = Region[region, ]
    successors = Block[]
    attributes = NamedAttribute[]
    push!(attributes, operandsegmentsizes([length(inputs), length(outputs), ]))
    (strides != nothing) && push!(attributes, namedattribute("strides", strides))
    (dilations != nothing) && push!(attributes, namedattribute("dilations", dilations))
    
    create_operation(
        "linalg.depthwise_conv_3d_ncdhw_cdhw", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`depthwise_conv_3d_ndhwc_dhwc`

Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output. Multiplier is set to 1
which is a special case for most depthwise convolutions.
"""
function depthwise_conv_3d_ndhwc_dhwc(inputs::Vector{Value}, outputs::Vector{Value}; result_tensors::Vector{MLIRType}, strides=nothing, dilations=nothing, region::Region, location=Location())
    results = MLIRType[result_tensors..., ]
    operands = Value[inputs..., outputs..., ]
    owned_regions = Region[region, ]
    successors = Block[]
    attributes = NamedAttribute[]
    push!(attributes, operandsegmentsizes([length(inputs), length(outputs), ]))
    (strides != nothing) && push!(attributes, namedattribute("strides", strides))
    (dilations != nothing) && push!(attributes, namedattribute("dilations", dilations))
    
    create_operation(
        "linalg.depthwise_conv_3d_ndhwc_dhwc", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`depthwise_conv_3d_ndhwc_dhwcm`

Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output.
"""
function depthwise_conv_3d_ndhwc_dhwcm(inputs::Vector{Value}, outputs::Vector{Value}; result_tensors::Vector{MLIRType}, strides=nothing, dilations=nothing, region::Region, location=Location())
    results = MLIRType[result_tensors..., ]
    operands = Value[inputs..., outputs..., ]
    owned_regions = Region[region, ]
    successors = Block[]
    attributes = NamedAttribute[]
    push!(attributes, operandsegmentsizes([length(inputs), length(outputs), ]))
    (strides != nothing) && push!(attributes, namedattribute("strides", strides))
    (dilations != nothing) && push!(attributes, namedattribute("dilations", dilations))
    
    create_operation(
        "linalg.depthwise_conv_3d_ndhwc_dhwcm", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`div`

types, performs a signed division.

The shapes and element types must be identical. The appropriate casts,
broadcasts and reductions should be done previously to calling this op.

This means reduction/broadcast/element cast semantics is explicit. Further
passes can take that into account when lowering this code. For example,
a `linalg.broadcast` + `linalg.div` sequence can be lowered to a
`linalg.generic` with different affine maps for the two operands.
"""
function div(inputs::Vector{Value}, outputs::Vector{Value}; result_tensors::Vector{MLIRType}, region::Region, location=Location())
    results = MLIRType[result_tensors..., ]
    operands = Value[inputs..., outputs..., ]
    owned_regions = Region[region, ]
    successors = Block[]
    attributes = NamedAttribute[]
    push!(attributes, operandsegmentsizes([length(inputs), length(outputs), ]))
    
    create_operation(
        "linalg.div", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`div_unsigned`

types, performs an unsigned division.

The shapes and element types must be identical. The appropriate casts,
broadcasts and reductions should be done previously to calling this op.

This means reduction/broadcast/element cast semantics is explicit. Further
passes can take that into account when lowering this code. For example,
a `linalg.broadcast` + `linalg.div` sequence can be lowered to a
`linalg.generic` with different affine maps for the two operands.
"""
function div_unsigned(inputs::Vector{Value}, outputs::Vector{Value}; result_tensors::Vector{MLIRType}, region::Region, location=Location())
    results = MLIRType[result_tensors..., ]
    operands = Value[inputs..., outputs..., ]
    owned_regions = Region[region, ]
    successors = Block[]
    attributes = NamedAttribute[]
    push!(attributes, operandsegmentsizes([length(inputs), length(outputs), ]))
    
    create_operation(
        "linalg.div_unsigned", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`dot`

Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output.
"""
function dot(inputs::Vector{Value}, outputs::Vector{Value}; result_tensors::Vector{MLIRType}, region::Region, location=Location())
    results = MLIRType[result_tensors..., ]
    operands = Value[inputs..., outputs..., ]
    owned_regions = Region[region, ]
    successors = Block[]
    attributes = NamedAttribute[]
    push!(attributes, operandsegmentsizes([length(inputs), length(outputs), ]))
    
    create_operation(
        "linalg.dot", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`elemwise_binary`

Numeric casting is performed on the input operand, promoting it to the same
data type as the accumulator/output.
"""
function elemwise_binary(inputs::Vector{Value}, outputs::Vector{Value}; result_tensors::Vector{MLIRType}, fun=nothing, cast=nothing, region::Region, location=Location())
    results = MLIRType[result_tensors..., ]
    operands = Value[inputs..., outputs..., ]
    owned_regions = Region[region, ]
    successors = Block[]
    attributes = NamedAttribute[]
    push!(attributes, operandsegmentsizes([length(inputs), length(outputs), ]))
    (fun != nothing) && push!(attributes, namedattribute("fun", fun))
    (cast != nothing) && push!(attributes, namedattribute("cast", cast))
    
    create_operation(
        "linalg.elemwise_binary", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`elemwise_unary`

Numeric casting is performed on the input operand, promoting it to the same
data type as the accumulator/output.
"""
function elemwise_unary(inputs::Vector{Value}, outputs::Vector{Value}; result_tensors::Vector{MLIRType}, fun=nothing, cast=nothing, region::Region, location=Location())
    results = MLIRType[result_tensors..., ]
    operands = Value[inputs..., outputs..., ]
    owned_regions = Region[region, ]
    successors = Block[]
    attributes = NamedAttribute[]
    push!(attributes, operandsegmentsizes([length(inputs), length(outputs), ]))
    (fun != nothing) && push!(attributes, namedattribute("fun", fun))
    (cast != nothing) && push!(attributes, namedattribute("cast", cast))
    
    create_operation(
        "linalg.elemwise_unary", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`exp`

No numeric casting is performed on the input operand.
"""
function exp(inputs::Vector{Value}, outputs::Vector{Value}; result_tensors::Vector{MLIRType}, region::Region, location=Location())
    results = MLIRType[result_tensors..., ]
    operands = Value[inputs..., outputs..., ]
    owned_regions = Region[region, ]
    successors = Block[]
    attributes = NamedAttribute[]
    push!(attributes, operandsegmentsizes([length(inputs), length(outputs), ]))
    
    create_operation(
        "linalg.exp", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`fill`

Works for arbitrary ranked output tensors since the operation performs scalar
accesses only and is thus rank polymorphic. Numeric casting is performed on
the value operand, promoting it to the same data type as the output.
"""
function fill(inputs::Vector{Value}, outputs::Vector{Value}; result_tensors::Vector{MLIRType}, region::Region, location=Location())
    results = MLIRType[result_tensors..., ]
    operands = Value[inputs..., outputs..., ]
    owned_regions = Region[region, ]
    successors = Block[]
    attributes = NamedAttribute[]
    push!(attributes, operandsegmentsizes([length(inputs), length(outputs), ]))
    
    create_operation(
        "linalg.fill", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`fill_rng_2d`

The operation generations pseudo random numbers using a linear congruential
generator. It provides no guarantees regarding the distribution of the
generated random numbers. Instead of generating the random numbers
sequentially, it instantiates one random number generator per data element
and runs them in parallel. The seed operand and the indices of the data
element seed the random number generation. The min and max operands limit
the range of the generated random numbers.
"""
function fill_rng_2d(inputs::Vector{Value}, outputs::Vector{Value}; result_tensors::Vector{MLIRType}, region::Region, location=Location())
    results = MLIRType[result_tensors..., ]
    operands = Value[inputs..., outputs..., ]
    owned_regions = Region[region, ]
    successors = Block[]
    attributes = NamedAttribute[]
    push!(attributes, operandsegmentsizes([length(inputs), length(outputs), ]))
    
    create_operation(
        "linalg.fill_rng_2d", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`floor`

No numeric casting is performed on the input operand.
"""
function floor(inputs::Vector{Value}, outputs::Vector{Value}; result_tensors::Vector{MLIRType}, region::Region, location=Location())
    results = MLIRType[result_tensors..., ]
    operands = Value[inputs..., outputs..., ]
    owned_regions = Region[region, ]
    successors = Block[]
    attributes = NamedAttribute[]
    push!(attributes, operandsegmentsizes([length(inputs), length(outputs), ]))
    
    create_operation(
        "linalg.floor", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`generic`

Generic Linalg op form where the key properties of the computation are
specified as attributes. In pretty form, a `linalg.generic` op is written
as:

  ```mlir
  linalg.generic #trait_attribute
      ins(%A, %B : memref<?x?xf32, stride_specification>,
                   memref<?x?xf32, stride_specification>)
      outs(%C : memref<?x?xf32, stride_specification>)
      attrs = {other-optional-attributes}
      {region}
  ```

Where #trait_attributes is an alias of a dictionary attribute containing:
  - doc [optional]: a documentation string
  - indexing_maps: a list of AffineMapAttr, one AffineMapAttr per each input
    and output view. Such AffineMapAttr specifies the mapping between the
    loops and the indexing within each view.
  - library_call [optional]: a StringAttr containing the name of an
    external library function that the linalg.generic operation maps to.
    The external library is assumed to be dynamically linked and no strong
    compile-time guarantees are provided. In the absence of such a library
    call, linalg.generic will always lower to loops.
  - iterator_types: an ArrayAttr specifying the type of the enclosing loops.
    Each element of the list represents and iterator of one of the following
    types:
      parallel, reduction, window

# Example
Defining a #matmul_trait attribute in MLIR can be done as follows:
  ```mlir
  #matmul_accesses = [
    (m, n, k) -> (m, k),
    (m, n, k) -> (k, n),
    (m, n, k) -> (m, n)
  ]
  #matmul_trait = {
    doc = \"C(m, n) += A(m, k) * B(k, n)\",
    indexing_maps = #matmul_accesses,
    library_call = \"linalg_matmul\",
    iterator_types = [\"parallel\", \"parallel\", \"reduction\"]
  }
  ```

And can be reused in multiple places as:
  ```mlir
  linalg.generic #matmul_trait
    ins(%A, %B : memref<?x?xf32, stride_specification>,
                 memref<?x?xf32, stride_specification>)
    outs(%C : memref<?x?xf32, stride_specification>)
    {other-optional-attributes} {
    ^bb0(%a: f32, %b: f32, %c: f32) :
      %d = arith.mulf %a, %b: f32
      %e = arith.addf %c, %d: f32
      linalg.yield %e : f32
  }
  ```

This may lower to either:
  ```mlir
  call @linalg_matmul(%A, %B, %C) :
    (memref<?x?xf32, stride_specification>,
     memref<?x?xf32, stride_specification>,
     memref<?x?xf32, stride_specification>)
    -> ()
  ```

or IR resembling:
```mlir
scf.for %m = %c0 to %M step %c1 {
  scf.for %n = %c0 to %N step %c1 {
    scf.for %k = %c0 to %K step %c1 {
      %a = load %A[%m, %k] : memref<?x?xf32, stride_specification>
      %b = load %B[%k, %n] : memref<?x?xf32, stride_specification>
      %c = load %C[%m, %n] : memref<?x?xf32, stride_specification>
      %d = arith.mulf %a, %b: f32
      %e = arith.addf %c, %d: f32
      store %e, %C[%m, %n] : memref<?x?x?xf32, stride_specification>
    }
  }
}
```

To allow progressive lowering from the value world (a.k.a tensor values) to
the buffer world (a.k.a memref values), a `linalg.generic` op allows mixing
tensors and buffers operands and tensor results.

```mlir
%C = linalg.generic #trait_attribute
  ins(%A, %B : tensor<?x?xf32>, memref<?x?xf32, stride_specification>)
  outs(%C : tensor<?x?xf32>)
  {other-optional-attributes}
  {region}
  -> (tensor<?x?xf32>)
```
"""
function generic(inputs::Vector{Value}, outputs::Vector{Value}; result_tensors::Vector{MLIRType}, indexing_maps, iterator_types, doc=nothing, library_call=nothing, region::Region, location=Location())
    results = MLIRType[result_tensors..., ]
    operands = Value[inputs..., outputs..., ]
    owned_regions = Region[region, ]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("indexing_maps", indexing_maps), namedattribute("iterator_types", iterator_types), ]
    push!(attributes, operandsegmentsizes([length(inputs), length(outputs), ]))
    (doc != nothing) && push!(attributes, namedattribute("doc", doc))
    (library_call != nothing) && push!(attributes, namedattribute("library_call", library_call))
    
    create_operation(
        "linalg.generic", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`log`

No numeric casting is performed on the input operand.
"""
function log(inputs::Vector{Value}, outputs::Vector{Value}; result_tensors::Vector{MLIRType}, region::Region, location=Location())
    results = MLIRType[result_tensors..., ]
    operands = Value[inputs..., outputs..., ]
    owned_regions = Region[region, ]
    successors = Block[]
    attributes = NamedAttribute[]
    push!(attributes, operandsegmentsizes([length(inputs), length(outputs), ]))
    
    create_operation(
        "linalg.log", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`map`

Models elementwise operations on tensors in terms of arithmetic operations
on the corresponding elements.

# Example
```
  %add = linalg.map
      ins(%lhs, %rhs : tensor<64xf32>, tensor<64xf32>)
      outs(%init: tensor<64xf32>)
      (%lhs_elem: f32, %rhs_elem: f32) {
        %0 = arith.addf %lhs_elem, %rhs_elem: f32
        linalg.yield %0: f32
      }
```

Shortened print form is available. Applies to simple maps with one 
non-yield operation inside the body.

The example above will be printed as:
```
  %add = linalg.map { arith.addf }
      ins(%lhs, %rhs : tensor<64xf32>, tensor<64xf32>)
      outs(%init: tensor<64xf32>)
```
"""
function map(inputs::Vector{Value}, init::Value; result::Vector{MLIRType}, mapper::Region, location=Location())
    results = MLIRType[result..., ]
    operands = Value[inputs..., init, ]
    owned_regions = Region[mapper, ]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "linalg.map", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`matmul`

Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output.
"""
function matmul(inputs::Vector{Value}, outputs::Vector{Value}; result_tensors::Vector{MLIRType}, cast=nothing, region::Region, location=Location())
    results = MLIRType[result_tensors..., ]
    operands = Value[inputs..., outputs..., ]
    owned_regions = Region[region, ]
    successors = Block[]
    attributes = NamedAttribute[]
    push!(attributes, operandsegmentsizes([length(inputs), length(outputs), ]))
    (cast != nothing) && push!(attributes, namedattribute("cast", cast))
    
    create_operation(
        "linalg.matmul", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`matmul_transpose_a`

transposed.

Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output.
"""
function matmul_transpose_a(inputs::Vector{Value}, outputs::Vector{Value}; result_tensors::Vector{MLIRType}, cast=nothing, region::Region, location=Location())
    results = MLIRType[result_tensors..., ]
    operands = Value[inputs..., outputs..., ]
    owned_regions = Region[region, ]
    successors = Block[]
    attributes = NamedAttribute[]
    push!(attributes, operandsegmentsizes([length(inputs), length(outputs), ]))
    (cast != nothing) && push!(attributes, namedattribute("cast", cast))
    
    create_operation(
        "linalg.matmul_transpose_a", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`matmul_transpose_b`

transposed.

Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output.
"""
function matmul_transpose_b(inputs::Vector{Value}, outputs::Vector{Value}; result_tensors::Vector{MLIRType}, cast=nothing, region::Region, location=Location())
    results = MLIRType[result_tensors..., ]
    operands = Value[inputs..., outputs..., ]
    owned_regions = Region[region, ]
    successors = Block[]
    attributes = NamedAttribute[]
    push!(attributes, operandsegmentsizes([length(inputs), length(outputs), ]))
    (cast != nothing) && push!(attributes, namedattribute("cast", cast))
    
    create_operation(
        "linalg.matmul_transpose_b", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`matmul_unsigned`

Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output.
"""
function matmul_unsigned(inputs::Vector{Value}, outputs::Vector{Value}; result_tensors::Vector{MLIRType}, region::Region, location=Location())
    results = MLIRType[result_tensors..., ]
    operands = Value[inputs..., outputs..., ]
    owned_regions = Region[region, ]
    successors = Block[]
    attributes = NamedAttribute[]
    push!(attributes, operandsegmentsizes([length(inputs), length(outputs), ]))
    
    create_operation(
        "linalg.matmul_unsigned", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`matvec`

Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output.
"""
function matvec(inputs::Vector{Value}, outputs::Vector{Value}; result_tensors::Vector{MLIRType}, region::Region, location=Location())
    results = MLIRType[result_tensors..., ]
    operands = Value[inputs..., outputs..., ]
    owned_regions = Region[region, ]
    successors = Block[]
    attributes = NamedAttribute[]
    push!(attributes, operandsegmentsizes([length(inputs), length(outputs), ]))
    
    create_operation(
        "linalg.matvec", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`max`

The shapes and element types must be identical. The appropriate casts,
broadcasts and reductions should be done previously to calling this op.

This means reduction/broadcast/element cast semantics is explicit. Further
passes can take that into account when lowering this code. For example,
a `linalg.broadcast` + `linalg.div` sequence can be lowered to a
`linalg.generic` with different affine maps for the two operands.
"""
function max(inputs::Vector{Value}, outputs::Vector{Value}; result_tensors::Vector{MLIRType}, region::Region, location=Location())
    results = MLIRType[result_tensors..., ]
    operands = Value[inputs..., outputs..., ]
    owned_regions = Region[region, ]
    successors = Block[]
    attributes = NamedAttribute[]
    push!(attributes, operandsegmentsizes([length(inputs), length(outputs), ]))
    
    create_operation(
        "linalg.max", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`mmt4d`

Differences from linalg.matmul:
* The right hand side is transposed, whence the \'t\' in \'mmt\'.
* The input and output tensors have a 4D shape instead of a 2D shape. They
  are interpreted as 2D matrices with one level of 2D tile subdivision,
  whence the 2+2=4 dimensions. The inner tile dimensions are identified with
  \'0\' suffixes below, for instance the LHS matrix shape (M, K, M0, K0) reads
  as: MxK tiles, each of shape M0xK0.
"""
function mmt4d(inputs::Vector{Value}, outputs::Vector{Value}; result_tensors::Vector{MLIRType}, region::Region, location=Location())
    results = MLIRType[result_tensors..., ]
    operands = Value[inputs..., outputs..., ]
    owned_regions = Region[region, ]
    successors = Block[]
    attributes = NamedAttribute[]
    push!(attributes, operandsegmentsizes([length(inputs), length(outputs), ]))
    
    create_operation(
        "linalg.mmt4d", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`mul`

The shapes and element types must be identical. The appropriate casts,
broadcasts and reductions should be done previously to calling this op.

This means reduction/broadcast/element cast semantics is explicit. Further
passes can take that into account when lowering this code. For example,
a `linalg.broadcast` + `linalg.mul` sequence can be lowered to a
`linalg.generic` with different affine maps for the two operands.
"""
function mul(inputs::Vector{Value}, outputs::Vector{Value}; result_tensors::Vector{MLIRType}, region::Region, location=Location())
    results = MLIRType[result_tensors..., ]
    operands = Value[inputs..., outputs..., ]
    owned_regions = Region[region, ]
    successors = Block[]
    attributes = NamedAttribute[]
    push!(attributes, operandsegmentsizes([length(inputs), length(outputs), ]))
    
    create_operation(
        "linalg.mul", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`negf`

No numeric casting is performed on the input operand.
"""
function negf(inputs::Vector{Value}, outputs::Vector{Value}; result_tensors::Vector{MLIRType}, region::Region, location=Location())
    results = MLIRType[result_tensors..., ]
    operands = Value[inputs..., outputs..., ]
    owned_regions = Region[region, ]
    successors = Block[]
    attributes = NamedAttribute[]
    push!(attributes, operandsegmentsizes([length(inputs), length(outputs), ]))
    
    create_operation(
        "linalg.negf", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`pooling_nchw_max`

Numeric casting is performed on the input operand, promoting it to the same
data type as the accumulator/output.
"""
function pooling_nchw_max(inputs::Vector{Value}, outputs::Vector{Value}; result_tensors::Vector{MLIRType}, strides=nothing, dilations=nothing, region::Region, location=Location())
    results = MLIRType[result_tensors..., ]
    operands = Value[inputs..., outputs..., ]
    owned_regions = Region[region, ]
    successors = Block[]
    attributes = NamedAttribute[]
    push!(attributes, operandsegmentsizes([length(inputs), length(outputs), ]))
    (strides != nothing) && push!(attributes, namedattribute("strides", strides))
    (dilations != nothing) && push!(attributes, namedattribute("dilations", dilations))
    
    create_operation(
        "linalg.pooling_nchw_max", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`pooling_nchw_sum`

Layout:
  * Input: NCHW.
  * Kernel: HW.

Numeric casting is performed on the input operand, promoting it to the same
data type as the accumulator/output.
"""
function pooling_nchw_sum(inputs::Vector{Value}, outputs::Vector{Value}; result_tensors::Vector{MLIRType}, strides=nothing, dilations=nothing, region::Region, location=Location())
    results = MLIRType[result_tensors..., ]
    operands = Value[inputs..., outputs..., ]
    owned_regions = Region[region, ]
    successors = Block[]
    attributes = NamedAttribute[]
    push!(attributes, operandsegmentsizes([length(inputs), length(outputs), ]))
    (strides != nothing) && push!(attributes, namedattribute("strides", strides))
    (dilations != nothing) && push!(attributes, namedattribute("dilations", dilations))
    
    create_operation(
        "linalg.pooling_nchw_sum", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`pooling_ncw_max`

Numeric casting is performed on the input operand, promoting it to the same
data type as the accumulator/output.
"""
function pooling_ncw_max(inputs::Vector{Value}, outputs::Vector{Value}; result_tensors::Vector{MLIRType}, strides=nothing, dilations=nothing, region::Region, location=Location())
    results = MLIRType[result_tensors..., ]
    operands = Value[inputs..., outputs..., ]
    owned_regions = Region[region, ]
    successors = Block[]
    attributes = NamedAttribute[]
    push!(attributes, operandsegmentsizes([length(inputs), length(outputs), ]))
    (strides != nothing) && push!(attributes, namedattribute("strides", strides))
    (dilations != nothing) && push!(attributes, namedattribute("dilations", dilations))
    
    create_operation(
        "linalg.pooling_ncw_max", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`pooling_ncw_sum`

Layout:
  * Input: NCW.
  * Kernel: W.

Numeric casting is performed on the input operand, promoting it to the same
data type as the accumulator/output.
"""
function pooling_ncw_sum(inputs::Vector{Value}, outputs::Vector{Value}; result_tensors::Vector{MLIRType}, strides=nothing, dilations=nothing, region::Region, location=Location())
    results = MLIRType[result_tensors..., ]
    operands = Value[inputs..., outputs..., ]
    owned_regions = Region[region, ]
    successors = Block[]
    attributes = NamedAttribute[]
    push!(attributes, operandsegmentsizes([length(inputs), length(outputs), ]))
    (strides != nothing) && push!(attributes, namedattribute("strides", strides))
    (dilations != nothing) && push!(attributes, namedattribute("dilations", dilations))
    
    create_operation(
        "linalg.pooling_ncw_sum", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`pooling_ndhwc_max`

Numeric casting is performed on the input operand, promoting it to the same
data type as the accumulator/output.
"""
function pooling_ndhwc_max(inputs::Vector{Value}, outputs::Vector{Value}; result_tensors::Vector{MLIRType}, strides=nothing, dilations=nothing, region::Region, location=Location())
    results = MLIRType[result_tensors..., ]
    operands = Value[inputs..., outputs..., ]
    owned_regions = Region[region, ]
    successors = Block[]
    attributes = NamedAttribute[]
    push!(attributes, operandsegmentsizes([length(inputs), length(outputs), ]))
    (strides != nothing) && push!(attributes, namedattribute("strides", strides))
    (dilations != nothing) && push!(attributes, namedattribute("dilations", dilations))
    
    create_operation(
        "linalg.pooling_ndhwc_max", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`pooling_ndhwc_min`

Numeric casting is performed on the input operand, promoting it to the same
data type as the accumulator/output.
"""
function pooling_ndhwc_min(inputs::Vector{Value}, outputs::Vector{Value}; result_tensors::Vector{MLIRType}, strides=nothing, dilations=nothing, region::Region, location=Location())
    results = MLIRType[result_tensors..., ]
    operands = Value[inputs..., outputs..., ]
    owned_regions = Region[region, ]
    successors = Block[]
    attributes = NamedAttribute[]
    push!(attributes, operandsegmentsizes([length(inputs), length(outputs), ]))
    (strides != nothing) && push!(attributes, namedattribute("strides", strides))
    (dilations != nothing) && push!(attributes, namedattribute("dilations", dilations))
    
    create_operation(
        "linalg.pooling_ndhwc_min", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`pooling_ndhwc_sum`

Numeric casting is performed on the input operand, promoting it to the same
data type as the accumulator/output.
"""
function pooling_ndhwc_sum(inputs::Vector{Value}, outputs::Vector{Value}; result_tensors::Vector{MLIRType}, strides=nothing, dilations=nothing, region::Region, location=Location())
    results = MLIRType[result_tensors..., ]
    operands = Value[inputs..., outputs..., ]
    owned_regions = Region[region, ]
    successors = Block[]
    attributes = NamedAttribute[]
    push!(attributes, operandsegmentsizes([length(inputs), length(outputs), ]))
    (strides != nothing) && push!(attributes, namedattribute("strides", strides))
    (dilations != nothing) && push!(attributes, namedattribute("dilations", dilations))
    
    create_operation(
        "linalg.pooling_ndhwc_sum", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`pooling_nhwc_max`

Numeric casting is performed on the input operand, promoting it to the same
data type as the accumulator/output.
"""
function pooling_nhwc_max(inputs::Vector{Value}, outputs::Vector{Value}; result_tensors::Vector{MLIRType}, strides=nothing, dilations=nothing, region::Region, location=Location())
    results = MLIRType[result_tensors..., ]
    operands = Value[inputs..., outputs..., ]
    owned_regions = Region[region, ]
    successors = Block[]
    attributes = NamedAttribute[]
    push!(attributes, operandsegmentsizes([length(inputs), length(outputs), ]))
    (strides != nothing) && push!(attributes, namedattribute("strides", strides))
    (dilations != nothing) && push!(attributes, namedattribute("dilations", dilations))
    
    create_operation(
        "linalg.pooling_nhwc_max", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`pooling_nhwc_max_unsigned`

Numeric casting is performed on the input operand, promoting it to the same
data type as the accumulator/output.
"""
function pooling_nhwc_max_unsigned(inputs::Vector{Value}, outputs::Vector{Value}; result_tensors::Vector{MLIRType}, strides=nothing, dilations=nothing, region::Region, location=Location())
    results = MLIRType[result_tensors..., ]
    operands = Value[inputs..., outputs..., ]
    owned_regions = Region[region, ]
    successors = Block[]
    attributes = NamedAttribute[]
    push!(attributes, operandsegmentsizes([length(inputs), length(outputs), ]))
    (strides != nothing) && push!(attributes, namedattribute("strides", strides))
    (dilations != nothing) && push!(attributes, namedattribute("dilations", dilations))
    
    create_operation(
        "linalg.pooling_nhwc_max_unsigned", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`pooling_nhwc_min`

Numeric casting is performed on the input operand, promoting it to the same
data type as the accumulator/output.
"""
function pooling_nhwc_min(inputs::Vector{Value}, outputs::Vector{Value}; result_tensors::Vector{MLIRType}, strides=nothing, dilations=nothing, region::Region, location=Location())
    results = MLIRType[result_tensors..., ]
    operands = Value[inputs..., outputs..., ]
    owned_regions = Region[region, ]
    successors = Block[]
    attributes = NamedAttribute[]
    push!(attributes, operandsegmentsizes([length(inputs), length(outputs), ]))
    (strides != nothing) && push!(attributes, namedattribute("strides", strides))
    (dilations != nothing) && push!(attributes, namedattribute("dilations", dilations))
    
    create_operation(
        "linalg.pooling_nhwc_min", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`pooling_nhwc_min_unsigned`

Numeric casting is performed on the input operand, promoting it to the same
data type as the accumulator/output.
"""
function pooling_nhwc_min_unsigned(inputs::Vector{Value}, outputs::Vector{Value}; result_tensors::Vector{MLIRType}, strides=nothing, dilations=nothing, region::Region, location=Location())
    results = MLIRType[result_tensors..., ]
    operands = Value[inputs..., outputs..., ]
    owned_regions = Region[region, ]
    successors = Block[]
    attributes = NamedAttribute[]
    push!(attributes, operandsegmentsizes([length(inputs), length(outputs), ]))
    (strides != nothing) && push!(attributes, namedattribute("strides", strides))
    (dilations != nothing) && push!(attributes, namedattribute("dilations", dilations))
    
    create_operation(
        "linalg.pooling_nhwc_min_unsigned", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`pooling_nhwc_sum`

Layout:
  * Input: NHWC.
  * Kernel: HW.

Numeric casting is performed on the input operand, promoting it to the same
data type as the accumulator/output.
"""
function pooling_nhwc_sum(inputs::Vector{Value}, outputs::Vector{Value}; result_tensors::Vector{MLIRType}, strides=nothing, dilations=nothing, region::Region, location=Location())
    results = MLIRType[result_tensors..., ]
    operands = Value[inputs..., outputs..., ]
    owned_regions = Region[region, ]
    successors = Block[]
    attributes = NamedAttribute[]
    push!(attributes, operandsegmentsizes([length(inputs), length(outputs), ]))
    (strides != nothing) && push!(attributes, namedattribute("strides", strides))
    (dilations != nothing) && push!(attributes, namedattribute("dilations", dilations))
    
    create_operation(
        "linalg.pooling_nhwc_sum", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`pooling_nwc_max`

Numeric casting is performed on the input operand, promoting it to the same
data type as the accumulator/output.
"""
function pooling_nwc_max(inputs::Vector{Value}, outputs::Vector{Value}; result_tensors::Vector{MLIRType}, strides=nothing, dilations=nothing, region::Region, location=Location())
    results = MLIRType[result_tensors..., ]
    operands = Value[inputs..., outputs..., ]
    owned_regions = Region[region, ]
    successors = Block[]
    attributes = NamedAttribute[]
    push!(attributes, operandsegmentsizes([length(inputs), length(outputs), ]))
    (strides != nothing) && push!(attributes, namedattribute("strides", strides))
    (dilations != nothing) && push!(attributes, namedattribute("dilations", dilations))
    
    create_operation(
        "linalg.pooling_nwc_max", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`pooling_nwc_max_unsigned`

Numeric casting is performed on the input operand, promoting it to the same
data type as the accumulator/output.
"""
function pooling_nwc_max_unsigned(inputs::Vector{Value}, outputs::Vector{Value}; result_tensors::Vector{MLIRType}, strides=nothing, dilations=nothing, region::Region, location=Location())
    results = MLIRType[result_tensors..., ]
    operands = Value[inputs..., outputs..., ]
    owned_regions = Region[region, ]
    successors = Block[]
    attributes = NamedAttribute[]
    push!(attributes, operandsegmentsizes([length(inputs), length(outputs), ]))
    (strides != nothing) && push!(attributes, namedattribute("strides", strides))
    (dilations != nothing) && push!(attributes, namedattribute("dilations", dilations))
    
    create_operation(
        "linalg.pooling_nwc_max_unsigned", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`pooling_nwc_min`

Numeric casting is performed on the input operand, promoting it to the same
data type as the accumulator/output.
"""
function pooling_nwc_min(inputs::Vector{Value}, outputs::Vector{Value}; result_tensors::Vector{MLIRType}, strides=nothing, dilations=nothing, region::Region, location=Location())
    results = MLIRType[result_tensors..., ]
    operands = Value[inputs..., outputs..., ]
    owned_regions = Region[region, ]
    successors = Block[]
    attributes = NamedAttribute[]
    push!(attributes, operandsegmentsizes([length(inputs), length(outputs), ]))
    (strides != nothing) && push!(attributes, namedattribute("strides", strides))
    (dilations != nothing) && push!(attributes, namedattribute("dilations", dilations))
    
    create_operation(
        "linalg.pooling_nwc_min", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`pooling_nwc_min_unsigned`

Numeric casting is performed on the input operand, promoting it to the same
data type as the accumulator/output.
"""
function pooling_nwc_min_unsigned(inputs::Vector{Value}, outputs::Vector{Value}; result_tensors::Vector{MLIRType}, strides=nothing, dilations=nothing, region::Region, location=Location())
    results = MLIRType[result_tensors..., ]
    operands = Value[inputs..., outputs..., ]
    owned_regions = Region[region, ]
    successors = Block[]
    attributes = NamedAttribute[]
    push!(attributes, operandsegmentsizes([length(inputs), length(outputs), ]))
    (strides != nothing) && push!(attributes, namedattribute("strides", strides))
    (dilations != nothing) && push!(attributes, namedattribute("dilations", dilations))
    
    create_operation(
        "linalg.pooling_nwc_min_unsigned", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`pooling_nwc_sum`

Layout:
  * Input: NWC.
  * Kernel: W.

Numeric casting is performed on the input operand, promoting it to the same
data type as the accumulator/output.
"""
function pooling_nwc_sum(inputs::Vector{Value}, outputs::Vector{Value}; result_tensors::Vector{MLIRType}, strides=nothing, dilations=nothing, region::Region, location=Location())
    results = MLIRType[result_tensors..., ]
    operands = Value[inputs..., outputs..., ]
    owned_regions = Region[region, ]
    successors = Block[]
    attributes = NamedAttribute[]
    push!(attributes, operandsegmentsizes([length(inputs), length(outputs), ]))
    (strides != nothing) && push!(attributes, namedattribute("strides", strides))
    (dilations != nothing) && push!(attributes, namedattribute("dilations", dilations))
    
    create_operation(
        "linalg.pooling_nwc_sum", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`quantized_batch_matmul`

Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output. The quantized variant
includes zero-point adjustments for the left and right operands of the
matmul.
"""
function quantized_batch_matmul(inputs::Vector{Value}, outputs::Vector{Value}; result_tensors::Vector{MLIRType}, region::Region, location=Location())
    results = MLIRType[result_tensors..., ]
    operands = Value[inputs..., outputs..., ]
    owned_regions = Region[region, ]
    successors = Block[]
    attributes = NamedAttribute[]
    push!(attributes, operandsegmentsizes([length(inputs), length(outputs), ]))
    
    create_operation(
        "linalg.quantized_batch_matmul", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`quantized_matmul`

Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output. The quantized variant
includes zero-point adjustments for the left and right operands of the
matmul.
"""
function quantized_matmul(inputs::Vector{Value}, outputs::Vector{Value}; result_tensors::Vector{MLIRType}, region::Region, location=Location())
    results = MLIRType[result_tensors..., ]
    operands = Value[inputs..., outputs..., ]
    owned_regions = Region[region, ]
    successors = Block[]
    attributes = NamedAttribute[]
    push!(attributes, operandsegmentsizes([length(inputs), length(outputs), ]))
    
    create_operation(
        "linalg.quantized_matmul", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`reduce`

Executes `combiner` on the `dimensions` of `inputs` and returns the
reduced result. The `dimensions` attribute needs to list the reduction
dimensions in increasing order.

# Example
```
  %reduce = linalg.reduce
      ins(%input:tensor<16x32x64xf32>)
      outs(%init:tensor<16x64xf32>)
      dimensions = [1]
      (%in: f32, %out: f32) {
        %0 = arith.addf %out, %in: f32
        linalg.yield %0: f32
      }
```

Shortened print form is available. Applies to simple (not variadic) reduces
with one non-yield operation inside the body. Applies only if the operation
takes `%out` as the first argument.

The example above will be printed as:
```
      %reduce = linalg.reduce { arith.addf }
      ins(%input:tensor<16x32x64xf32>)
      outs(%init:tensor<16x64xf32>)
      dimensions = [1]
```
"""
function reduce(inputs::Vector{Value}, inits::Vector{Value}; result_0::Vector{MLIRType}, dimensions, combiner::Region, location=Location())
    results = MLIRType[result_0..., ]
    operands = Value[inputs..., inits..., ]
    owned_regions = Region[combiner, ]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("dimensions", dimensions), ]
    
    create_operation(
        "linalg.reduce", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`sub`

The shapes and element types must be identical. The appropriate casts,
broadcasts and reductions should be done previously to calling this op.

This means reduction/broadcast/element cast semantics is explicit. Further
passes can take that into account when lowering this code. For example,
a `linalg.broadcast` + `linalg.sub` sequence can be lowered to a
`linalg.generic` with different affine maps for the two operands.
"""
function sub(inputs::Vector{Value}, outputs::Vector{Value}; result_tensors::Vector{MLIRType}, region::Region, location=Location())
    results = MLIRType[result_tensors..., ]
    operands = Value[inputs..., outputs..., ]
    owned_regions = Region[region, ]
    successors = Block[]
    attributes = NamedAttribute[]
    push!(attributes, operandsegmentsizes([length(inputs), length(outputs), ]))
    
    create_operation(
        "linalg.sub", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`transpose`

Permutes the dimensions of `input` according to the given `permutation`.
  `dim(result, i) = dim(input, permutation[i])`

This op actually moves data, unlike `memref.transpose` which is a metadata
operation only that produces a transposed \"view\".

# Example
```
  %transpose = linalg.transpose
      ins(%input:tensor<16x64xf32>)
      outs(%init:tensor<64x16xf32>)
      permutation = [1, 0]
```
"""
function transpose(input::Value, init::Value; result::Vector{MLIRType}, permutation, region::Region, location=Location())
    results = MLIRType[result..., ]
    operands = Value[input, init, ]
    owned_regions = Region[region, ]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("permutation", permutation), ]
    
    create_operation(
        "linalg.transpose", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`vecmat`

Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output.
"""
function vecmat(inputs::Vector{Value}, outputs::Vector{Value}; result_tensors::Vector{MLIRType}, region::Region, location=Location())
    results = MLIRType[result_tensors..., ]
    operands = Value[inputs..., outputs..., ]
    owned_regions = Region[region, ]
    successors = Block[]
    attributes = NamedAttribute[]
    push!(attributes, operandsegmentsizes([length(inputs), length(outputs), ]))
    
    create_operation(
        "linalg.vecmat", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

end # linalg
