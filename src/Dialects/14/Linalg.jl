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
    results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("dim", dim),]
    !isnothing(result) && push!(results, result)

    return IR.create_operation(
        "linalg.index",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false),
    )
end

"""
`init_tensor`

`linalg.init_tensor` is an operation that materializes a tensor of
a given shape. The shape could be dynamic or static.
"""
function init_tensor(
    sizes::Vector{Value}; result::IR.Type, static_sizes, location=Location()
)
    results = IR.Type[result,]
    operands = Value[sizes...,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("static_sizes", static_sizes),]

    return IR.create_operation(
        "linalg.init_tensor",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=results,
        result_inference=false,
    )
end

"""
`tiled_loop`

This is a loop-like operation with additional properties. The arguments
also include the input and the output tensors or memrefs and the attributes
to specify the iterator types.

Parsing TiledLoopOp will set all elements of the `iterator_types` attribute
to \"parallel\" type, when it is absent from the custom format.

Tensor-based version:

The body region of the loop contains `extract_slice` operations applied to
every tensor argument of TiledLoopOp.

The body region must contain exactly one block that terminates with
`linalg.yield` with the operands resulting from `insert_slice` operations.

# Example

```mlir
%0 = linalg.tiled_loop (%i) = (%c0) to (%c24) step (%c4)
    ins(%lhs, %rhs : tensor<24x64xi8>, tensor<24x64xi8>)
    outs(%out : tensor<24x64xi8>)
    iterators(\"parallel\")
    distribution(\"block_x\") {
  %lhs_sub = tensor.extract_slice %lhs[%i, 0] [%c4, %c64] [1, 1]
      : tensor<24x64xi8> to tensor<?x?xi8>
  %rhs_sub = tensor.extract_slice %rhs[%i, 0] [%c4, %c64] [1, 1]
      : tensor<24x64xi8> to tensor<?x?xi8>
  %out_sub = tensor.extract_slice %out[%i, 0] [%c4, %c64] [1, 1]
      : tensor<24x64xi8> to tensor<?x?xi8>

  %result_sub = linalg.generic ...

  %result = tensor.insert_slice %result_sub into %out[%i, 0][%c4, %c64][1, 1]
    : tensor<?x?xi8> into tensor<24x64xi8>
  linalg.yield %result : tensor<24x64xi8>
}
```

MemRef-based version:

The body region of the loop contains `subview` operations applied to
every memref argument of TiledLoopOp.

The body region must contain exactly one block that terminates with
`linalg.yield` with no operands.

# Example

```mlir
linalg.tiled_loop (%i) = (%c0) to (%c24) step (%c4)
    ins(%lhs, %rhs : memref<24x64xi8>, memref<24x64xi8>)
    outs(%out : memref<24x64xi8>)
    iterators(\"parallel\")
    distribution(\"block_x\") {
  %lhs_sub = subview %lhs[%i, 0] [%c4, %c64] [1, 1]
      : memref<24x64xi8> to memref<?x?xi8>
  %rhs_sub = subview %rhs[%i, 0] [%c4, %c64] [1, 1]
      : memref<24x64xi8> to memref<?x?xi8>
  %out_sub = subview %out[%i, 0] [%c4, %c64] [1, 1]
      : memref<24x64xi8> to memref<?x?xi8>

  %result_sub = linalg.generic ...
  linalg.yield
}
```
"""
function tiled_loop(
    lowerBound::Vector{Value},
    upperBound::Vector{Value},
    step::Vector{Value},
    inputs::Vector{Value},
    outputs::Vector{Value};
    results::Vector{IR.Type},
    iterator_types,
    distribution_types=nothing,
    region::Region,
    location=Location(),
)
    results = IR.Type[results...,]
    operands = Value[lowerBound..., upperBound..., step..., inputs..., outputs...]
    owned_regions = Region[region,]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("iterator_types", iterator_types),]
    push!(
        attributes,
        operandsegmentsizes([
            length(lowerBound),
            length(upperBound),
            length(step),
            length(inputs),
            length(outputs),
        ]),
    )
    !isnothing(distribution_types) &&
        push!(attributes, namedattribute("distribution_types", distribution_types))

    return IR.create_operation(
        "linalg.tiled_loop",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=results,
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
    results = IR.Type[]
    operands = Value[values...,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return IR.create_operation(
        "linalg.yield",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=results,
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
    results = IR.Type[result_tensors...,]
    operands = Value[inputs..., outputs...]
    owned_regions = Region[region,]
    successors = Block[]
    attributes = NamedAttribute[]
    push!(attributes, operandsegmentsizes([length(inputs), length(outputs)]))

    return IR.create_operation(
        "linalg.batch_matmul",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=results,
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
    results = IR.Type[result_tensors...,]
    operands = Value[inputs..., outputs...]
    owned_regions = Region[region,]
    successors = Block[]
    attributes = NamedAttribute[]
    push!(attributes, operandsegmentsizes([length(inputs), length(outputs)]))

    return IR.create_operation(
        "linalg.batch_matvec",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=results,
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
    strides,
    dilations,
    region::Region,
    location=Location(),
)
    results = IR.Type[result_tensors...,]
    operands = Value[inputs..., outputs...]
    owned_regions = Region[region,]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("strides", strides), namedattribute("dilations", dilations)
    ]
    push!(attributes, operandsegmentsizes([length(inputs), length(outputs)]))

    return IR.create_operation(
        "linalg.conv_1d_nwc_wcf",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=results,
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
    results = IR.Type[result_tensors...,]
    operands = Value[inputs..., outputs...]
    owned_regions = Region[region,]
    successors = Block[]
    attributes = NamedAttribute[]
    push!(attributes, operandsegmentsizes([length(inputs), length(outputs)]))

    return IR.create_operation(
        "linalg.conv_1d",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=results,
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
    strides,
    dilations,
    region::Region,
    location=Location(),
)
    results = IR.Type[result_tensors...,]
    operands = Value[inputs..., outputs...]
    owned_regions = Region[region,]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("strides", strides), namedattribute("dilations", dilations)
    ]
    push!(attributes, operandsegmentsizes([length(inputs), length(outputs)]))

    return IR.create_operation(
        "linalg.conv_2d_nchw_fchw",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=results,
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
    strides,
    dilations,
    region::Region,
    location=Location(),
)
    results = IR.Type[result_tensors...,]
    operands = Value[inputs..., outputs...]
    owned_regions = Region[region,]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("strides", strides), namedattribute("dilations", dilations)
    ]
    push!(attributes, operandsegmentsizes([length(inputs), length(outputs)]))

    return IR.create_operation(
        "linalg.conv_2d_nhwc_hwcf",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=results,
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
    strides,
    dilations,
    region::Region,
    location=Location(),
)
    results = IR.Type[result_tensors...,]
    operands = Value[inputs..., outputs...]
    owned_regions = Region[region,]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("strides", strides), namedattribute("dilations", dilations)
    ]
    push!(attributes, operandsegmentsizes([length(inputs), length(outputs)]))

    return IR.create_operation(
        "linalg.conv_2d_nhwc_hwcf_q",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=results,
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
    results = IR.Type[result_tensors...,]
    operands = Value[inputs..., outputs...]
    owned_regions = Region[region,]
    successors = Block[]
    attributes = NamedAttribute[]
    push!(attributes, operandsegmentsizes([length(inputs), length(outputs)]))

    return IR.create_operation(
        "linalg.conv_2d",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=results,
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
    strides,
    dilations,
    region::Region,
    location=Location(),
)
    results = IR.Type[result_tensors...,]
    operands = Value[inputs..., outputs...]
    owned_regions = Region[region,]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("strides", strides), namedattribute("dilations", dilations)
    ]
    push!(attributes, operandsegmentsizes([length(inputs), length(outputs)]))

    return IR.create_operation(
        "linalg.conv_3d_ndhwc_dhwcf",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=results,
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
    results = IR.Type[result_tensors...,]
    operands = Value[inputs..., outputs...]
    owned_regions = Region[region,]
    successors = Block[]
    attributes = NamedAttribute[]
    push!(attributes, operandsegmentsizes([length(inputs), length(outputs)]))

    return IR.create_operation(
        "linalg.conv_3d",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=results,
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
    strides,
    dilations,
    region::Region,
    location=Location(),
)
    results = IR.Type[result_tensors...,]
    operands = Value[inputs..., outputs...]
    owned_regions = Region[region,]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("strides", strides), namedattribute("dilations", dilations)
    ]
    push!(attributes, operandsegmentsizes([length(inputs), length(outputs)]))

    return IR.create_operation(
        "linalg.depthwise_conv_1d_nwc_wc",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=results,
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
    strides,
    dilations,
    region::Region,
    location=Location(),
)
    results = IR.Type[result_tensors...,]
    operands = Value[inputs..., outputs...]
    owned_regions = Region[region,]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("strides", strides), namedattribute("dilations", dilations)
    ]
    push!(attributes, operandsegmentsizes([length(inputs), length(outputs)]))

    return IR.create_operation(
        "linalg.depthwise_conv_2d_nhwc_hwc",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=results,
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
    strides,
    dilations,
    region::Region,
    location=Location(),
)
    results = IR.Type[result_tensors...,]
    operands = Value[inputs..., outputs...]
    owned_regions = Region[region,]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("strides", strides), namedattribute("dilations", dilations)
    ]
    push!(attributes, operandsegmentsizes([length(inputs), length(outputs)]))

    return IR.create_operation(
        "linalg.depthwise_conv_2d_nhwc_hwc_q",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=results,
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
    strides,
    dilations,
    region::Region,
    location=Location(),
)
    results = IR.Type[result_tensors...,]
    operands = Value[inputs..., outputs...]
    owned_regions = Region[region,]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("strides", strides), namedattribute("dilations", dilations)
    ]
    push!(attributes, operandsegmentsizes([length(inputs), length(outputs)]))

    return IR.create_operation(
        "linalg.depthwise_conv_2d_nhwc_hwcm",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=results,
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
    strides,
    dilations,
    region::Region,
    location=Location(),
)
    results = IR.Type[result_tensors...,]
    operands = Value[inputs..., outputs...]
    owned_regions = Region[region,]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("strides", strides), namedattribute("dilations", dilations)
    ]
    push!(attributes, operandsegmentsizes([length(inputs), length(outputs)]))

    return IR.create_operation(
        "linalg.depthwise_conv_2d_nhwc_hwcm_q",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=results,
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
    results = IR.Type[result_tensors...,]
    operands = Value[inputs..., outputs...]
    owned_regions = Region[region,]
    successors = Block[]
    attributes = NamedAttribute[]
    push!(attributes, operandsegmentsizes([length(inputs), length(outputs)]))

    return IR.create_operation(
        "linalg.dot",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=results,
        result_inference=false,
    )
end

"""
`fill`

"""
function fill(
    value::Value,
    output::Value;
    result=nothing::Union{Nothing,IR.Type},
    region::Region,
    location=Location(),
)
    results = IR.Type[]
    operands = Value[value, output]
    owned_regions = Region[region,]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(results, result)

    return IR.create_operation(
        "linalg.fill",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=results,
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
    results = IR.Type[result_tensors...,]
    operands = Value[inputs..., outputs...]
    owned_regions = Region[region,]
    successors = Block[]
    attributes = NamedAttribute[]
    push!(attributes, operandsegmentsizes([length(inputs), length(outputs)]))

    return IR.create_operation(
        "linalg.fill_rng_2d",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=results,
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
    results = IR.Type[result_tensors...,]
    operands = Value[inputs..., outputs...]
    owned_regions = Region[region,]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("indexing_maps", indexing_maps),
        namedattribute("iterator_types", iterator_types),
    ]
    push!(attributes, operandsegmentsizes([length(inputs), length(outputs)]))
    !isnothing(doc) && push!(attributes, namedattribute("doc", doc))
    !isnothing(library_call) &&
        push!(attributes, namedattribute("library_call", library_call))

    return IR.create_operation(
        "linalg.generic",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=results,
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
    region::Region,
    location=Location(),
)
    results = IR.Type[result_tensors...,]
    operands = Value[inputs..., outputs...]
    owned_regions = Region[region,]
    successors = Block[]
    attributes = NamedAttribute[]
    push!(attributes, operandsegmentsizes([length(inputs), length(outputs)]))

    return IR.create_operation(
        "linalg.matmul",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=results,
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
    results = IR.Type[result_tensors...,]
    operands = Value[inputs..., outputs...]
    owned_regions = Region[region,]
    successors = Block[]
    attributes = NamedAttribute[]
    push!(attributes, operandsegmentsizes([length(inputs), length(outputs)]))

    return IR.create_operation(
        "linalg.matmul_unsigned",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=results,
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
    results = IR.Type[result_tensors...,]
    operands = Value[inputs..., outputs...]
    owned_regions = Region[region,]
    successors = Block[]
    attributes = NamedAttribute[]
    push!(attributes, operandsegmentsizes([length(inputs), length(outputs)]))

    return IR.create_operation(
        "linalg.matvec",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=results,
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
    results = IR.Type[result_tensors...,]
    operands = Value[inputs..., outputs...]
    owned_regions = Region[region,]
    successors = Block[]
    attributes = NamedAttribute[]
    push!(attributes, operandsegmentsizes([length(inputs), length(outputs)]))

    return IR.create_operation(
        "linalg.mmt4d",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=results,
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
    strides,
    dilations,
    region::Region,
    location=Location(),
)
    results = IR.Type[result_tensors...,]
    operands = Value[inputs..., outputs...]
    owned_regions = Region[region,]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("strides", strides), namedattribute("dilations", dilations)
    ]
    push!(attributes, operandsegmentsizes([length(inputs), length(outputs)]))

    return IR.create_operation(
        "linalg.pooling_nchw_max",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=results,
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
    strides,
    dilations,
    region::Region,
    location=Location(),
)
    results = IR.Type[result_tensors...,]
    operands = Value[inputs..., outputs...]
    owned_regions = Region[region,]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("strides", strides), namedattribute("dilations", dilations)
    ]
    push!(attributes, operandsegmentsizes([length(inputs), length(outputs)]))

    return IR.create_operation(
        "linalg.pooling_ndhwc_max",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=results,
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
    strides,
    dilations,
    region::Region,
    location=Location(),
)
    results = IR.Type[result_tensors...,]
    operands = Value[inputs..., outputs...]
    owned_regions = Region[region,]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("strides", strides), namedattribute("dilations", dilations)
    ]
    push!(attributes, operandsegmentsizes([length(inputs), length(outputs)]))

    return IR.create_operation(
        "linalg.pooling_ndhwc_min",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=results,
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
    strides,
    dilations,
    region::Region,
    location=Location(),
)
    results = IR.Type[result_tensors...,]
    operands = Value[inputs..., outputs...]
    owned_regions = Region[region,]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("strides", strides), namedattribute("dilations", dilations)
    ]
    push!(attributes, operandsegmentsizes([length(inputs), length(outputs)]))

    return IR.create_operation(
        "linalg.pooling_ndhwc_sum",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=results,
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
    strides,
    dilations,
    region::Region,
    location=Location(),
)
    results = IR.Type[result_tensors...,]
    operands = Value[inputs..., outputs...]
    owned_regions = Region[region,]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("strides", strides), namedattribute("dilations", dilations)
    ]
    push!(attributes, operandsegmentsizes([length(inputs), length(outputs)]))

    return IR.create_operation(
        "linalg.pooling_nhwc_max",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=results,
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
    strides,
    dilations,
    region::Region,
    location=Location(),
)
    results = IR.Type[result_tensors...,]
    operands = Value[inputs..., outputs...]
    owned_regions = Region[region,]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("strides", strides), namedattribute("dilations", dilations)
    ]
    push!(attributes, operandsegmentsizes([length(inputs), length(outputs)]))

    return IR.create_operation(
        "linalg.pooling_nhwc_max_unsigned",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=results,
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
    strides,
    dilations,
    region::Region,
    location=Location(),
)
    results = IR.Type[result_tensors...,]
    operands = Value[inputs..., outputs...]
    owned_regions = Region[region,]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("strides", strides), namedattribute("dilations", dilations)
    ]
    push!(attributes, operandsegmentsizes([length(inputs), length(outputs)]))

    return IR.create_operation(
        "linalg.pooling_nhwc_min",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=results,
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
    strides,
    dilations,
    region::Region,
    location=Location(),
)
    results = IR.Type[result_tensors...,]
    operands = Value[inputs..., outputs...]
    owned_regions = Region[region,]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("strides", strides), namedattribute("dilations", dilations)
    ]
    push!(attributes, operandsegmentsizes([length(inputs), length(outputs)]))

    return IR.create_operation(
        "linalg.pooling_nhwc_min_unsigned",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=results,
        result_inference=false,
    )
end

"""
`pooling_nhwc_sum`

Numeric casting is performed on the input operand, promoting it to the same
data type as the accumulator/output.
"""
function pooling_nhwc_sum(
    inputs::Vector{Value},
    outputs::Vector{Value};
    result_tensors::Vector{IR.Type},
    strides,
    dilations,
    region::Region,
    location=Location(),
)
    results = IR.Type[result_tensors...,]
    operands = Value[inputs..., outputs...]
    owned_regions = Region[region,]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("strides", strides), namedattribute("dilations", dilations)
    ]
    push!(attributes, operandsegmentsizes([length(inputs), length(outputs)]))

    return IR.create_operation(
        "linalg.pooling_nhwc_sum",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=results,
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
    results = IR.Type[result_tensors...,]
    operands = Value[inputs..., outputs...]
    owned_regions = Region[region,]
    successors = Block[]
    attributes = NamedAttribute[]
    push!(attributes, operandsegmentsizes([length(inputs), length(outputs)]))

    return IR.create_operation(
        "linalg.quantized_batch_matmul",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=results,
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
    results = IR.Type[result_tensors...,]
    operands = Value[inputs..., outputs...]
    owned_regions = Region[region,]
    successors = Block[]
    attributes = NamedAttribute[]
    push!(attributes, operandsegmentsizes([length(inputs), length(outputs)]))

    return IR.create_operation(
        "linalg.quantized_matmul",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=results,
        result_inference=false,
    )
end

"""
`soft_plus_2d`

Numeric casting is performed on the input operand, promoting it to the same
data type as the accumulator/output.
"""
function soft_plus_2d(
    inputs::Vector{Value},
    outputs::Vector{Value};
    result_tensors::Vector{IR.Type},
    region::Region,
    location=Location(),
)
    results = IR.Type[result_tensors...,]
    operands = Value[inputs..., outputs...]
    owned_regions = Region[region,]
    successors = Block[]
    attributes = NamedAttribute[]
    push!(attributes, operandsegmentsizes([length(inputs), length(outputs)]))

    return IR.create_operation(
        "linalg.soft_plus_2d",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=results,
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
    results = IR.Type[result_tensors...,]
    operands = Value[inputs..., outputs...]
    owned_regions = Region[region,]
    successors = Block[]
    attributes = NamedAttribute[]
    push!(attributes, operandsegmentsizes([length(inputs), length(outputs)]))

    return IR.create_operation(
        "linalg.vecmat",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=results,
        result_inference=false,
    )
end

end # linalg
