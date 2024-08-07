module linalg

import ...IR:
    IR, NamedAttribute, Value, Location, Block, Region, Attribute, context, IndexType
import ..Dialects: namedattribute, operandsegmentsizes

"""
`index`

The `linalg.index` operation returns the iteration index of the immediately
enclosing linalg structured operation for the iteration dimension `dim`. The
`dim` attribute specifies the position of the accessed dimension in the
indexing map domain.

# Example

```mlir
#map = affine_map<(i, j) -> (i, j)>
linalg.generic {indexing_maps = [#map, #map],
                iterator_types = [\"parallel\", \"parallel\"]}
  outs(%I, %J : memref<?x?xindex>, memref<?x?xindex>) {
  ^bb0(%arg0 : index, %arg1 : index):
  // Access the outer iteration dimension i
  %i = linalg.index 0 : index
  // Access the inner iteration dimension j
  %j = linalg.index 1 : index
  linalg.yield %i, %j : index, index
}
```

This may lower to IR resembling:

```mlir
%0 = dim %I, %c0 : memref<?x?xindex>
%1 = dim %I, %c1 : memref<?x?xindex>
scf.for %i = %c0 to %0 step %c1 {
  scf.for %j = %c0 to %1 step %c1 {
    store %i, %I[%i, %j] : memref<?x?xindex>
    store %j, %J[%i, %j] : memref<?x?xindex>
  }
}
```
"""
function index(; result=nothing::Union{Nothing,IR.Type}, dim, location=Location())
    _results = IR.Type[]
    _operands = Value[]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[namedattribute("dim", dim),]
    !isnothing(result) && push!(_results, result)

    return IR.create_operation(
        "linalg.index",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=(length(_results) == 0 ? nothing : _results),
        result_inference=(length(_results) == 0 ? true : false),
    )
end

"""
`init_tensor`

`linalg.init_tensor` is an operation that defines a tensor of a particular
shape. The shape could be dynamic or static. The contents of the tensor are
unspecified and the only purpose of the op result is to materialize the
specified shape in IR and make it available to other transformations.

Note: This op can be lowered to a `bufferization.alloc_tensor`, at which
point it turns into an explicit buffer allocation.
"""
function init_tensor(
    sizes::Vector{Value}; result::IR.Type, static_sizes, location=Location()
)
    _results = IR.Type[result,]
    _operands = Value[sizes...,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[namedattribute("static_sizes", static_sizes),]

    return IR.create_operation(
        "linalg.init_tensor",
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
`yield`

`linalg.yield` is a special terminator operation for blocks inside regions
in `linalg` generic ops. It returns values to the immediately enclosing
`linalg` generic op.

# Example

```mlir
linalg.yield %f0, %f1 : f32, f32
```
"""
function yield(values::Vector{Value}; location=Location())
    _results = IR.Type[]
    _operands = Value[values...,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "linalg.yield",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

import ...IR:
    IR, NamedAttribute, Value, Location, Block, Region, Attribute, context, IndexType
import ..Dialects: namedattribute, operandsegmentsizes

"""
`batch_matmul`

Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output.
"""
function batch_matmul(
    inputs::Vector{Value},
    outputs::Vector{Value};
    result_tensors::Vector{IR.Type},
    region::Region,
    location=Location(),
)
    _results = IR.Type[result_tensors...,]
    _operands = Value[inputs..., outputs...]
    _owned_regions = Region[region,]
    _successors = Block[]
    _attributes = NamedAttribute[]
    push!(_attributes, operandsegmentsizes([length(inputs), length(outputs)]))

    return IR.create_operation(
        "linalg.batch_matmul",
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
`batch_matvec`

Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output.
"""
function batch_matvec(
    inputs::Vector{Value},
    outputs::Vector{Value};
    result_tensors::Vector{IR.Type},
    region::Region,
    location=Location(),
)
    _results = IR.Type[result_tensors...,]
    _operands = Value[inputs..., outputs...]
    _owned_regions = Region[region,]
    _successors = Block[]
    _attributes = NamedAttribute[]
    push!(_attributes, operandsegmentsizes([length(inputs), length(outputs)]))

    return IR.create_operation(
        "linalg.batch_matvec",
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
`conv_1d_nwc_wcf`

Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output.
"""
function conv_1d_nwc_wcf(
    inputs::Vector{Value},
    outputs::Vector{Value};
    result_tensors::Vector{IR.Type},
    strides=nothing,
    dilations=nothing,
    region::Region,
    location=Location(),
)
    _results = IR.Type[result_tensors...,]
    _operands = Value[inputs..., outputs...]
    _owned_regions = Region[region,]
    _successors = Block[]
    _attributes = NamedAttribute[]
    push!(_attributes, operandsegmentsizes([length(inputs), length(outputs)]))
    !isnothing(strides) && push!(_attributes, namedattribute("strides", strides))
    !isnothing(dilations) && push!(_attributes, namedattribute("dilations", dilations))

    return IR.create_operation(
        "linalg.conv_1d_nwc_wcf",
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
`conv_1d`

Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output.
"""
function conv_1d(
    inputs::Vector{Value},
    outputs::Vector{Value};
    result_tensors::Vector{IR.Type},
    region::Region,
    location=Location(),
)
    _results = IR.Type[result_tensors...,]
    _operands = Value[inputs..., outputs...]
    _owned_regions = Region[region,]
    _successors = Block[]
    _attributes = NamedAttribute[]
    push!(_attributes, operandsegmentsizes([length(inputs), length(outputs)]))

    return IR.create_operation(
        "linalg.conv_1d",
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
`conv_2d_nchw_fchw`

Layout:
  * Input: NCHW.
  * Kernel: FCHW.

Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output.
"""
function conv_2d_nchw_fchw(
    inputs::Vector{Value},
    outputs::Vector{Value};
    result_tensors::Vector{IR.Type},
    strides=nothing,
    dilations=nothing,
    region::Region,
    location=Location(),
)
    _results = IR.Type[result_tensors...,]
    _operands = Value[inputs..., outputs...]
    _owned_regions = Region[region,]
    _successors = Block[]
    _attributes = NamedAttribute[]
    push!(_attributes, operandsegmentsizes([length(inputs), length(outputs)]))
    !isnothing(strides) && push!(_attributes, namedattribute("strides", strides))
    !isnothing(dilations) && push!(_attributes, namedattribute("dilations", dilations))

    return IR.create_operation(
        "linalg.conv_2d_nchw_fchw",
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
`conv_2d_ngchw_fgchw`

Layout:
  * Input: NGCHW.
  * Kernel: FGCHW.

Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output.
"""
function conv_2d_ngchw_fgchw(
    inputs::Vector{Value},
    outputs::Vector{Value};
    result_tensors::Vector{IR.Type},
    strides=nothing,
    dilations=nothing,
    region::Region,
    location=Location(),
)
    _results = IR.Type[result_tensors...,]
    _operands = Value[inputs..., outputs...]
    _owned_regions = Region[region,]
    _successors = Block[]
    _attributes = NamedAttribute[]
    push!(_attributes, operandsegmentsizes([length(inputs), length(outputs)]))
    !isnothing(strides) && push!(_attributes, namedattribute("strides", strides))
    !isnothing(dilations) && push!(_attributes, namedattribute("dilations", dilations))

    return IR.create_operation(
        "linalg.conv_2d_ngchw_fgchw",
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
`conv_2d_nhwc_fhwc`

Layout:
  * Input: NHWC.
  * Kernel: FHWC.

Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output.
"""
function conv_2d_nhwc_fhwc(
    inputs::Vector{Value},
    outputs::Vector{Value};
    result_tensors::Vector{IR.Type},
    strides=nothing,
    dilations=nothing,
    region::Region,
    location=Location(),
)
    _results = IR.Type[result_tensors...,]
    _operands = Value[inputs..., outputs...]
    _owned_regions = Region[region,]
    _successors = Block[]
    _attributes = NamedAttribute[]
    push!(_attributes, operandsegmentsizes([length(inputs), length(outputs)]))
    !isnothing(strides) && push!(_attributes, namedattribute("strides", strides))
    !isnothing(dilations) && push!(_attributes, namedattribute("dilations", dilations))

    return IR.create_operation(
        "linalg.conv_2d_nhwc_fhwc",
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
`conv_2d_nhwc_hwcf`

Layout:
  * Input: NHWC.
  * Kernel: HWCF.

Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output.
"""
function conv_2d_nhwc_hwcf(
    inputs::Vector{Value},
    outputs::Vector{Value};
    result_tensors::Vector{IR.Type},
    strides=nothing,
    dilations=nothing,
    region::Region,
    location=Location(),
)
    _results = IR.Type[result_tensors...,]
    _operands = Value[inputs..., outputs...]
    _owned_regions = Region[region,]
    _successors = Block[]
    _attributes = NamedAttribute[]
    push!(_attributes, operandsegmentsizes([length(inputs), length(outputs)]))
    !isnothing(strides) && push!(_attributes, namedattribute("strides", strides))
    !isnothing(dilations) && push!(_attributes, namedattribute("dilations", dilations))

    return IR.create_operation(
        "linalg.conv_2d_nhwc_hwcf",
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
`conv_2d_nhwc_hwcf_q`

Layout:
  * Input: NHWC.
  * Kernel: HWCF.

Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output. This includes the zero
point offsets common to quantized operations.
"""
function conv_2d_nhwc_hwcf_q(
    inputs::Vector{Value},
    outputs::Vector{Value};
    result_tensors::Vector{IR.Type},
    strides=nothing,
    dilations=nothing,
    region::Region,
    location=Location(),
)
    _results = IR.Type[result_tensors...,]
    _operands = Value[inputs..., outputs...]
    _owned_regions = Region[region,]
    _successors = Block[]
    _attributes = NamedAttribute[]
    push!(_attributes, operandsegmentsizes([length(inputs), length(outputs)]))
    !isnothing(strides) && push!(_attributes, namedattribute("strides", strides))
    !isnothing(dilations) && push!(_attributes, namedattribute("dilations", dilations))

    return IR.create_operation(
        "linalg.conv_2d_nhwc_hwcf_q",
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
`conv_2d`

Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output.
"""
function conv_2d(
    inputs::Vector{Value},
    outputs::Vector{Value};
    result_tensors::Vector{IR.Type},
    region::Region,
    location=Location(),
)
    _results = IR.Type[result_tensors...,]
    _operands = Value[inputs..., outputs...]
    _owned_regions = Region[region,]
    _successors = Block[]
    _attributes = NamedAttribute[]
    push!(_attributes, operandsegmentsizes([length(inputs), length(outputs)]))

    return IR.create_operation(
        "linalg.conv_2d",
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
`conv_3d_ndhwc_dhwcf`

Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output.
"""
function conv_3d_ndhwc_dhwcf(
    inputs::Vector{Value},
    outputs::Vector{Value};
    result_tensors::Vector{IR.Type},
    strides=nothing,
    dilations=nothing,
    region::Region,
    location=Location(),
)
    _results = IR.Type[result_tensors...,]
    _operands = Value[inputs..., outputs...]
    _owned_regions = Region[region,]
    _successors = Block[]
    _attributes = NamedAttribute[]
    push!(_attributes, operandsegmentsizes([length(inputs), length(outputs)]))
    !isnothing(strides) && push!(_attributes, namedattribute("strides", strides))
    !isnothing(dilations) && push!(_attributes, namedattribute("dilations", dilations))

    return IR.create_operation(
        "linalg.conv_3d_ndhwc_dhwcf",
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
`conv_3d`

Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output.
"""
function conv_3d(
    inputs::Vector{Value},
    outputs::Vector{Value};
    result_tensors::Vector{IR.Type},
    region::Region,
    location=Location(),
)
    _results = IR.Type[result_tensors...,]
    _operands = Value[inputs..., outputs...]
    _owned_regions = Region[region,]
    _successors = Block[]
    _attributes = NamedAttribute[]
    push!(_attributes, operandsegmentsizes([length(inputs), length(outputs)]))

    return IR.create_operation(
        "linalg.conv_3d",
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
`copy`

Numeric casting is performed on the input operand, promoting it to the same
data type as the accumulator/output.
"""
function copy(
    inputs::Vector{Value},
    outputs::Vector{Value};
    result_tensors::Vector{IR.Type},
    cast=nothing,
    region::Region,
    location=Location(),
)
    _results = IR.Type[result_tensors...,]
    _operands = Value[inputs..., outputs...]
    _owned_regions = Region[region,]
    _successors = Block[]
    _attributes = NamedAttribute[]
    push!(_attributes, operandsegmentsizes([length(inputs), length(outputs)]))
    !isnothing(cast) && push!(_attributes, namedattribute("cast", cast))

    return IR.create_operation(
        "linalg.copy",
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
`depthwise_conv_1d_nwc_wc`

Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output. Multiplier is set to 1
which is a special case for most depthwise convolutions.
"""
function depthwise_conv_1d_nwc_wc(
    inputs::Vector{Value},
    outputs::Vector{Value};
    result_tensors::Vector{IR.Type},
    strides=nothing,
    dilations=nothing,
    region::Region,
    location=Location(),
)
    _results = IR.Type[result_tensors...,]
    _operands = Value[inputs..., outputs...]
    _owned_regions = Region[region,]
    _successors = Block[]
    _attributes = NamedAttribute[]
    push!(_attributes, operandsegmentsizes([length(inputs), length(outputs)]))
    !isnothing(strides) && push!(_attributes, namedattribute("strides", strides))
    !isnothing(dilations) && push!(_attributes, namedattribute("dilations", dilations))

    return IR.create_operation(
        "linalg.depthwise_conv_1d_nwc_wc",
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
`depthwise_conv_1d_nwc_wcm`

Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output.
"""
function depthwise_conv_1d_nwc_wcm(
    inputs::Vector{Value},
    outputs::Vector{Value};
    result_tensors::Vector{IR.Type},
    strides=nothing,
    dilations=nothing,
    region::Region,
    location=Location(),
)
    _results = IR.Type[result_tensors...,]
    _operands = Value[inputs..., outputs...]
    _owned_regions = Region[region,]
    _successors = Block[]
    _attributes = NamedAttribute[]
    push!(_attributes, operandsegmentsizes([length(inputs), length(outputs)]))
    !isnothing(strides) && push!(_attributes, namedattribute("strides", strides))
    !isnothing(dilations) && push!(_attributes, namedattribute("dilations", dilations))

    return IR.create_operation(
        "linalg.depthwise_conv_1d_nwc_wcm",
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
`depthwise_conv_2d_nchw_chw`

Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output. Multiplier is set to 1
which is a special case for most depthwise convolutions.
"""
function depthwise_conv_2d_nchw_chw(
    inputs::Vector{Value},
    outputs::Vector{Value};
    result_tensors::Vector{IR.Type},
    strides=nothing,
    dilations=nothing,
    region::Region,
    location=Location(),
)
    _results = IR.Type[result_tensors...,]
    _operands = Value[inputs..., outputs...]
    _owned_regions = Region[region,]
    _successors = Block[]
    _attributes = NamedAttribute[]
    push!(_attributes, operandsegmentsizes([length(inputs), length(outputs)]))
    !isnothing(strides) && push!(_attributes, namedattribute("strides", strides))
    !isnothing(dilations) && push!(_attributes, namedattribute("dilations", dilations))

    return IR.create_operation(
        "linalg.depthwise_conv_2d_nchw_chw",
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
`depthwise_conv_2d_nhwc_hwc`

Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output. Multiplier is set to 1
which is a special case for most depthwise convolutions.
"""
function depthwise_conv_2d_nhwc_hwc(
    inputs::Vector{Value},
    outputs::Vector{Value};
    result_tensors::Vector{IR.Type},
    strides=nothing,
    dilations=nothing,
    region::Region,
    location=Location(),
)
    _results = IR.Type[result_tensors...,]
    _operands = Value[inputs..., outputs...]
    _owned_regions = Region[region,]
    _successors = Block[]
    _attributes = NamedAttribute[]
    push!(_attributes, operandsegmentsizes([length(inputs), length(outputs)]))
    !isnothing(strides) && push!(_attributes, namedattribute("strides", strides))
    !isnothing(dilations) && push!(_attributes, namedattribute("dilations", dilations))

    return IR.create_operation(
        "linalg.depthwise_conv_2d_nhwc_hwc",
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
`depthwise_conv_2d_nhwc_hwc_q`

Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output.
"""
function depthwise_conv_2d_nhwc_hwc_q(
    inputs::Vector{Value},
    outputs::Vector{Value};
    result_tensors::Vector{IR.Type},
    strides=nothing,
    dilations=nothing,
    region::Region,
    location=Location(),
)
    _results = IR.Type[result_tensors...,]
    _operands = Value[inputs..., outputs...]
    _owned_regions = Region[region,]
    _successors = Block[]
    _attributes = NamedAttribute[]
    push!(_attributes, operandsegmentsizes([length(inputs), length(outputs)]))
    !isnothing(strides) && push!(_attributes, namedattribute("strides", strides))
    !isnothing(dilations) && push!(_attributes, namedattribute("dilations", dilations))

    return IR.create_operation(
        "linalg.depthwise_conv_2d_nhwc_hwc_q",
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
`depthwise_conv_2d_nhwc_hwcm`

Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output.
"""
function depthwise_conv_2d_nhwc_hwcm(
    inputs::Vector{Value},
    outputs::Vector{Value};
    result_tensors::Vector{IR.Type},
    strides=nothing,
    dilations=nothing,
    region::Region,
    location=Location(),
)
    _results = IR.Type[result_tensors...,]
    _operands = Value[inputs..., outputs...]
    _owned_regions = Region[region,]
    _successors = Block[]
    _attributes = NamedAttribute[]
    push!(_attributes, operandsegmentsizes([length(inputs), length(outputs)]))
    !isnothing(strides) && push!(_attributes, namedattribute("strides", strides))
    !isnothing(dilations) && push!(_attributes, namedattribute("dilations", dilations))

    return IR.create_operation(
        "linalg.depthwise_conv_2d_nhwc_hwcm",
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
`depthwise_conv_2d_nhwc_hwcm_q`

Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output.
"""
function depthwise_conv_2d_nhwc_hwcm_q(
    inputs::Vector{Value},
    outputs::Vector{Value};
    result_tensors::Vector{IR.Type},
    strides=nothing,
    dilations=nothing,
    region::Region,
    location=Location(),
)
    _results = IR.Type[result_tensors...,]
    _operands = Value[inputs..., outputs...]
    _owned_regions = Region[region,]
    _successors = Block[]
    _attributes = NamedAttribute[]
    push!(_attributes, operandsegmentsizes([length(inputs), length(outputs)]))
    !isnothing(strides) && push!(_attributes, namedattribute("strides", strides))
    !isnothing(dilations) && push!(_attributes, namedattribute("dilations", dilations))

    return IR.create_operation(
        "linalg.depthwise_conv_2d_nhwc_hwcm_q",
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
`depthwise_conv_3d_ndhwc_dhwc`

Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output. Multiplier is set to 1
which is a special case for most depthwise convolutions.
"""
function depthwise_conv_3d_ndhwc_dhwc(
    inputs::Vector{Value},
    outputs::Vector{Value};
    result_tensors::Vector{IR.Type},
    strides=nothing,
    dilations=nothing,
    region::Region,
    location=Location(),
)
    _results = IR.Type[result_tensors...,]
    _operands = Value[inputs..., outputs...]
    _owned_regions = Region[region,]
    _successors = Block[]
    _attributes = NamedAttribute[]
    push!(_attributes, operandsegmentsizes([length(inputs), length(outputs)]))
    !isnothing(strides) && push!(_attributes, namedattribute("strides", strides))
    !isnothing(dilations) && push!(_attributes, namedattribute("dilations", dilations))

    return IR.create_operation(
        "linalg.depthwise_conv_3d_ndhwc_dhwc",
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
`depthwise_conv_3d_ndhwc_dhwcm`

Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output.
"""
function depthwise_conv_3d_ndhwc_dhwcm(
    inputs::Vector{Value},
    outputs::Vector{Value};
    result_tensors::Vector{IR.Type},
    strides=nothing,
    dilations=nothing,
    region::Region,
    location=Location(),
)
    _results = IR.Type[result_tensors...,]
    _operands = Value[inputs..., outputs...]
    _owned_regions = Region[region,]
    _successors = Block[]
    _attributes = NamedAttribute[]
    push!(_attributes, operandsegmentsizes([length(inputs), length(outputs)]))
    !isnothing(strides) && push!(_attributes, namedattribute("strides", strides))
    !isnothing(dilations) && push!(_attributes, namedattribute("dilations", dilations))

    return IR.create_operation(
        "linalg.depthwise_conv_3d_ndhwc_dhwcm",
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
`dot`

Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output.
"""
function dot(
    inputs::Vector{Value},
    outputs::Vector{Value};
    result_tensors::Vector{IR.Type},
    region::Region,
    location=Location(),
)
    _results = IR.Type[result_tensors...,]
    _operands = Value[inputs..., outputs...]
    _owned_regions = Region[region,]
    _successors = Block[]
    _attributes = NamedAttribute[]
    push!(_attributes, operandsegmentsizes([length(inputs), length(outputs)]))

    return IR.create_operation(
        "linalg.dot",
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
`elemwise_binary`

Numeric casting is performed on the input operand, promoting it to the same
data type as the accumulator/output.
"""
function elemwise_binary(
    inputs::Vector{Value},
    outputs::Vector{Value};
    result_tensors::Vector{IR.Type},
    fun=nothing,
    cast=nothing,
    region::Region,
    location=Location(),
)
    _results = IR.Type[result_tensors...,]
    _operands = Value[inputs..., outputs...]
    _owned_regions = Region[region,]
    _successors = Block[]
    _attributes = NamedAttribute[]
    push!(_attributes, operandsegmentsizes([length(inputs), length(outputs)]))
    !isnothing(fun) && push!(_attributes, namedattribute("fun", fun))
    !isnothing(cast) && push!(_attributes, namedattribute("cast", cast))

    return IR.create_operation(
        "linalg.elemwise_binary",
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
`elemwise_unary`

Numeric casting is performed on the input operand, promoting it to the same
data type as the accumulator/output.
"""
function elemwise_unary(
    inputs::Vector{Value},
    outputs::Vector{Value};
    result_tensors::Vector{IR.Type},
    fun=nothing,
    cast=nothing,
    region::Region,
    location=Location(),
)
    _results = IR.Type[result_tensors...,]
    _operands = Value[inputs..., outputs...]
    _owned_regions = Region[region,]
    _successors = Block[]
    _attributes = NamedAttribute[]
    push!(_attributes, operandsegmentsizes([length(inputs), length(outputs)]))
    !isnothing(fun) && push!(_attributes, namedattribute("fun", fun))
    !isnothing(cast) && push!(_attributes, namedattribute("cast", cast))

    return IR.create_operation(
        "linalg.elemwise_unary",
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
`fill`

Works for arbitrary ranked output tensors since the operation performs scalar
accesses only and is thus rank polymorphic. Numeric casting is performed on
the value operand, promoting it to the same data type as the output.
"""
function fill(
    inputs::Vector{Value},
    outputs::Vector{Value};
    result_tensors::Vector{IR.Type},
    region::Region,
    location=Location(),
)
    _results = IR.Type[result_tensors...,]
    _operands = Value[inputs..., outputs...]
    _owned_regions = Region[region,]
    _successors = Block[]
    _attributes = NamedAttribute[]
    push!(_attributes, operandsegmentsizes([length(inputs), length(outputs)]))

    return IR.create_operation(
        "linalg.fill",
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
`fill_rng_2d`

The operation generations pseudo random numbers using a linear congruential
generator. It provides no guarantees regarding the distribution of the
generated random numbers. Instead of generating the random numbers
sequentially, it instantiates one random number generator per data element
and runs them in parallel. The seed operand and the indices of the data
element seed the random number generation. The min and max operands limit
the range of the generated random numbers.
"""
function fill_rng_2d(
    inputs::Vector{Value},
    outputs::Vector{Value};
    result_tensors::Vector{IR.Type},
    region::Region,
    location=Location(),
)
    _results = IR.Type[result_tensors...,]
    _operands = Value[inputs..., outputs...]
    _owned_regions = Region[region,]
    _successors = Block[]
    _attributes = NamedAttribute[]
    push!(_attributes, operandsegmentsizes([length(inputs), length(outputs)]))

    return IR.create_operation(
        "linalg.fill_rng_2d",
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
function generic(
    inputs::Vector{Value},
    outputs::Vector{Value};
    result_tensors::Vector{IR.Type},
    indexing_maps,
    iterator_types,
    doc=nothing,
    library_call=nothing,
    region::Region,
    location=Location(),
)
    _results = IR.Type[result_tensors...,]
    _operands = Value[inputs..., outputs...]
    _owned_regions = Region[region,]
    _successors = Block[]
    _attributes = NamedAttribute[
        namedattribute("indexing_maps", indexing_maps),
        namedattribute("iterator_types", iterator_types),
    ]
    push!(_attributes, operandsegmentsizes([length(inputs), length(outputs)]))
    !isnothing(doc) && push!(_attributes, namedattribute("doc", doc))
    !isnothing(library_call) &&
        push!(_attributes, namedattribute("library_call", library_call))

    return IR.create_operation(
        "linalg.generic",
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
`matmul`

Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output.
"""
function matmul(
    inputs::Vector{Value},
    outputs::Vector{Value};
    result_tensors::Vector{IR.Type},
    cast=nothing,
    region::Region,
    location=Location(),
)
    _results = IR.Type[result_tensors...,]
    _operands = Value[inputs..., outputs...]
    _owned_regions = Region[region,]
    _successors = Block[]
    _attributes = NamedAttribute[]
    push!(_attributes, operandsegmentsizes([length(inputs), length(outputs)]))
    !isnothing(cast) && push!(_attributes, namedattribute("cast", cast))

    return IR.create_operation(
        "linalg.matmul",
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
`matmul_unsigned`

Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output.
"""
function matmul_unsigned(
    inputs::Vector{Value},
    outputs::Vector{Value};
    result_tensors::Vector{IR.Type},
    region::Region,
    location=Location(),
)
    _results = IR.Type[result_tensors...,]
    _operands = Value[inputs..., outputs...]
    _owned_regions = Region[region,]
    _successors = Block[]
    _attributes = NamedAttribute[]
    push!(_attributes, operandsegmentsizes([length(inputs), length(outputs)]))

    return IR.create_operation(
        "linalg.matmul_unsigned",
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
`matvec`

Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output.
"""
function matvec(
    inputs::Vector{Value},
    outputs::Vector{Value};
    result_tensors::Vector{IR.Type},
    region::Region,
    location=Location(),
)
    _results = IR.Type[result_tensors...,]
    _operands = Value[inputs..., outputs...]
    _owned_regions = Region[region,]
    _successors = Block[]
    _attributes = NamedAttribute[]
    push!(_attributes, operandsegmentsizes([length(inputs), length(outputs)]))

    return IR.create_operation(
        "linalg.matvec",
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
`mmt4d`

Differences from linalg.matmul:
* The right hand side is transposed, whence the \'t\' in \'mmt\'.
* The input and output tensors have a 4D shape instead of a 2D shape. They
  are interpreted as 2D matrices with one level of 2D tile subdivision,
  whence the 2+2=4 dimensions. The inner tile dimensions are identified with
  \'0\' suffixes below, for instance the LHS matrix shape (M, K, M0, K0) reads
  as: MxK tiles, each of shape M0xK0.
"""
function mmt4d(
    inputs::Vector{Value},
    outputs::Vector{Value};
    result_tensors::Vector{IR.Type},
    region::Region,
    location=Location(),
)
    _results = IR.Type[result_tensors...,]
    _operands = Value[inputs..., outputs...]
    _owned_regions = Region[region,]
    _successors = Block[]
    _attributes = NamedAttribute[]
    push!(_attributes, operandsegmentsizes([length(inputs), length(outputs)]))

    return IR.create_operation(
        "linalg.mmt4d",
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
`pooling_nchw_max`

Numeric casting is performed on the input operand, promoting it to the same
data type as the accumulator/output.
"""
function pooling_nchw_max(
    inputs::Vector{Value},
    outputs::Vector{Value};
    result_tensors::Vector{IR.Type},
    strides=nothing,
    dilations=nothing,
    region::Region,
    location=Location(),
)
    _results = IR.Type[result_tensors...,]
    _operands = Value[inputs..., outputs...]
    _owned_regions = Region[region,]
    _successors = Block[]
    _attributes = NamedAttribute[]
    push!(_attributes, operandsegmentsizes([length(inputs), length(outputs)]))
    !isnothing(strides) && push!(_attributes, namedattribute("strides", strides))
    !isnothing(dilations) && push!(_attributes, namedattribute("dilations", dilations))

    return IR.create_operation(
        "linalg.pooling_nchw_max",
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
`pooling_nchw_sum`

Layout:
  * Input: NCHW.
  * Kernel: HW.

Numeric casting is performed on the input operand, promoting it to the same
data type as the accumulator/output.
"""
function pooling_nchw_sum(
    inputs::Vector{Value},
    outputs::Vector{Value};
    result_tensors::Vector{IR.Type},
    strides=nothing,
    dilations=nothing,
    region::Region,
    location=Location(),
)
    _results = IR.Type[result_tensors...,]
    _operands = Value[inputs..., outputs...]
    _owned_regions = Region[region,]
    _successors = Block[]
    _attributes = NamedAttribute[]
    push!(_attributes, operandsegmentsizes([length(inputs), length(outputs)]))
    !isnothing(strides) && push!(_attributes, namedattribute("strides", strides))
    !isnothing(dilations) && push!(_attributes, namedattribute("dilations", dilations))

    return IR.create_operation(
        "linalg.pooling_nchw_sum",
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
`pooling_ndhwc_max`

Numeric casting is performed on the input operand, promoting it to the same
data type as the accumulator/output.
"""
function pooling_ndhwc_max(
    inputs::Vector{Value},
    outputs::Vector{Value};
    result_tensors::Vector{IR.Type},
    strides=nothing,
    dilations=nothing,
    region::Region,
    location=Location(),
)
    _results = IR.Type[result_tensors...,]
    _operands = Value[inputs..., outputs...]
    _owned_regions = Region[region,]
    _successors = Block[]
    _attributes = NamedAttribute[]
    push!(_attributes, operandsegmentsizes([length(inputs), length(outputs)]))
    !isnothing(strides) && push!(_attributes, namedattribute("strides", strides))
    !isnothing(dilations) && push!(_attributes, namedattribute("dilations", dilations))

    return IR.create_operation(
        "linalg.pooling_ndhwc_max",
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
`pooling_ndhwc_min`

Numeric casting is performed on the input operand, promoting it to the same
data type as the accumulator/output.
"""
function pooling_ndhwc_min(
    inputs::Vector{Value},
    outputs::Vector{Value};
    result_tensors::Vector{IR.Type},
    strides=nothing,
    dilations=nothing,
    region::Region,
    location=Location(),
)
    _results = IR.Type[result_tensors...,]
    _operands = Value[inputs..., outputs...]
    _owned_regions = Region[region,]
    _successors = Block[]
    _attributes = NamedAttribute[]
    push!(_attributes, operandsegmentsizes([length(inputs), length(outputs)]))
    !isnothing(strides) && push!(_attributes, namedattribute("strides", strides))
    !isnothing(dilations) && push!(_attributes, namedattribute("dilations", dilations))

    return IR.create_operation(
        "linalg.pooling_ndhwc_min",
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
`pooling_ndhwc_sum`

Numeric casting is performed on the input operand, promoting it to the same
data type as the accumulator/output.
"""
function pooling_ndhwc_sum(
    inputs::Vector{Value},
    outputs::Vector{Value};
    result_tensors::Vector{IR.Type},
    strides=nothing,
    dilations=nothing,
    region::Region,
    location=Location(),
)
    _results = IR.Type[result_tensors...,]
    _operands = Value[inputs..., outputs...]
    _owned_regions = Region[region,]
    _successors = Block[]
    _attributes = NamedAttribute[]
    push!(_attributes, operandsegmentsizes([length(inputs), length(outputs)]))
    !isnothing(strides) && push!(_attributes, namedattribute("strides", strides))
    !isnothing(dilations) && push!(_attributes, namedattribute("dilations", dilations))

    return IR.create_operation(
        "linalg.pooling_ndhwc_sum",
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
`pooling_nhwc_max`

Numeric casting is performed on the input operand, promoting it to the same
data type as the accumulator/output.
"""
function pooling_nhwc_max(
    inputs::Vector{Value},
    outputs::Vector{Value};
    result_tensors::Vector{IR.Type},
    strides=nothing,
    dilations=nothing,
    region::Region,
    location=Location(),
)
    _results = IR.Type[result_tensors...,]
    _operands = Value[inputs..., outputs...]
    _owned_regions = Region[region,]
    _successors = Block[]
    _attributes = NamedAttribute[]
    push!(_attributes, operandsegmentsizes([length(inputs), length(outputs)]))
    !isnothing(strides) && push!(_attributes, namedattribute("strides", strides))
    !isnothing(dilations) && push!(_attributes, namedattribute("dilations", dilations))

    return IR.create_operation(
        "linalg.pooling_nhwc_max",
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
`pooling_nhwc_max_unsigned`

Numeric casting is performed on the input operand, promoting it to the same
data type as the accumulator/output.
"""
function pooling_nhwc_max_unsigned(
    inputs::Vector{Value},
    outputs::Vector{Value};
    result_tensors::Vector{IR.Type},
    strides=nothing,
    dilations=nothing,
    region::Region,
    location=Location(),
)
    _results = IR.Type[result_tensors...,]
    _operands = Value[inputs..., outputs...]
    _owned_regions = Region[region,]
    _successors = Block[]
    _attributes = NamedAttribute[]
    push!(_attributes, operandsegmentsizes([length(inputs), length(outputs)]))
    !isnothing(strides) && push!(_attributes, namedattribute("strides", strides))
    !isnothing(dilations) && push!(_attributes, namedattribute("dilations", dilations))

    return IR.create_operation(
        "linalg.pooling_nhwc_max_unsigned",
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
`pooling_nhwc_min`

Numeric casting is performed on the input operand, promoting it to the same
data type as the accumulator/output.
"""
function pooling_nhwc_min(
    inputs::Vector{Value},
    outputs::Vector{Value};
    result_tensors::Vector{IR.Type},
    strides=nothing,
    dilations=nothing,
    region::Region,
    location=Location(),
)
    _results = IR.Type[result_tensors...,]
    _operands = Value[inputs..., outputs...]
    _owned_regions = Region[region,]
    _successors = Block[]
    _attributes = NamedAttribute[]
    push!(_attributes, operandsegmentsizes([length(inputs), length(outputs)]))
    !isnothing(strides) && push!(_attributes, namedattribute("strides", strides))
    !isnothing(dilations) && push!(_attributes, namedattribute("dilations", dilations))

    return IR.create_operation(
        "linalg.pooling_nhwc_min",
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
`pooling_nhwc_min_unsigned`

Numeric casting is performed on the input operand, promoting it to the same
data type as the accumulator/output.
"""
function pooling_nhwc_min_unsigned(
    inputs::Vector{Value},
    outputs::Vector{Value};
    result_tensors::Vector{IR.Type},
    strides=nothing,
    dilations=nothing,
    region::Region,
    location=Location(),
)
    _results = IR.Type[result_tensors...,]
    _operands = Value[inputs..., outputs...]
    _owned_regions = Region[region,]
    _successors = Block[]
    _attributes = NamedAttribute[]
    push!(_attributes, operandsegmentsizes([length(inputs), length(outputs)]))
    !isnothing(strides) && push!(_attributes, namedattribute("strides", strides))
    !isnothing(dilations) && push!(_attributes, namedattribute("dilations", dilations))

    return IR.create_operation(
        "linalg.pooling_nhwc_min_unsigned",
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
`pooling_nhwc_sum`

Layout:
  * Input: NHWC.
  * Kernel: HW.

Numeric casting is performed on the input operand, promoting it to the same
data type as the accumulator/output.
"""
function pooling_nhwc_sum(
    inputs::Vector{Value},
    outputs::Vector{Value};
    result_tensors::Vector{IR.Type},
    strides=nothing,
    dilations=nothing,
    region::Region,
    location=Location(),
)
    _results = IR.Type[result_tensors...,]
    _operands = Value[inputs..., outputs...]
    _owned_regions = Region[region,]
    _successors = Block[]
    _attributes = NamedAttribute[]
    push!(_attributes, operandsegmentsizes([length(inputs), length(outputs)]))
    !isnothing(strides) && push!(_attributes, namedattribute("strides", strides))
    !isnothing(dilations) && push!(_attributes, namedattribute("dilations", dilations))

    return IR.create_operation(
        "linalg.pooling_nhwc_sum",
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
`quantized_batch_matmul`

Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output. The quantized variant
includes zero-point adjustments for the left and right operands of the
matmul.
"""
function quantized_batch_matmul(
    inputs::Vector{Value},
    outputs::Vector{Value};
    result_tensors::Vector{IR.Type},
    region::Region,
    location=Location(),
)
    _results = IR.Type[result_tensors...,]
    _operands = Value[inputs..., outputs...]
    _owned_regions = Region[region,]
    _successors = Block[]
    _attributes = NamedAttribute[]
    push!(_attributes, operandsegmentsizes([length(inputs), length(outputs)]))

    return IR.create_operation(
        "linalg.quantized_batch_matmul",
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
`quantized_matmul`

Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output. The quantized variant
includes zero-point adjustments for the left and right operands of the
matmul.
"""
function quantized_matmul(
    inputs::Vector{Value},
    outputs::Vector{Value};
    result_tensors::Vector{IR.Type},
    region::Region,
    location=Location(),
)
    _results = IR.Type[result_tensors...,]
    _operands = Value[inputs..., outputs...]
    _owned_regions = Region[region,]
    _successors = Block[]
    _attributes = NamedAttribute[]
    push!(_attributes, operandsegmentsizes([length(inputs), length(outputs)]))

    return IR.create_operation(
        "linalg.quantized_matmul",
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
`vecmat`

Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output.
"""
function vecmat(
    inputs::Vector{Value},
    outputs::Vector{Value};
    result_tensors::Vector{IR.Type},
    region::Region,
    location=Location(),
)
    _results = IR.Type[result_tensors...,]
    _operands = Value[inputs..., outputs...]
    _owned_regions = Region[region,]
    _successors = Block[]
    _attributes = NamedAttribute[]
    push!(_attributes, operandsegmentsizes([length(inputs), length(outputs)]))

    return IR.create_operation(
        "linalg.vecmat",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

end # linalg
