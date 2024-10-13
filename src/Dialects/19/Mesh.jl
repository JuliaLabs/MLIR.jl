module mesh

import ...IR:
    IR, NamedAttribute, Value, Location, Block, Region, Attribute, context, IndexType
import ..Dialects: namedattribute, operandsegmentsizes

"""
`all_gather`

Gathers along the `gather_axis` tensor axis.

# Example
```mlir
mesh.mesh @mesh0(shape = 2x2)
...
%1 = mesh.all_gather %0 on @mesh0 mesh_axes = [1] gather_axis = 1
  : tensor<2x2xi8> -> tensor<2x4xi8>
```
Input:
```
                 +-------+-------+
device (0, 0) -> |  1  2 |  5  6 | <- device (0, 1)
                 |  3  4 |  7  8 |
                 +-------+-------+
device (1, 0) -> |  9 10 | 13 14 | <- device (1, 1)
                 | 11 12 | 15 16 |
                 +-------+-------+
```
Result:
```
gather tensor
axis 1
------------>
+-------------+
|  1  2  5  6 | <- devices (0, 0) and (0, 1)
|  3  4  7  8 |
+-------------+
|  9 10 13 14 | <- devices (1, 0) and (1, 1)
| 11 12 15 16 |
+-------------+
```
"""
function all_gather(
    input::Value; result::IR.Type, mesh, mesh_axes=nothing, gather_axis, location=Location()
)
    _results = IR.Type[result,]
    _operands = Value[input,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[
        namedattribute("mesh", mesh), namedattribute("gather_axis", gather_axis)
    ]
    !isnothing(mesh_axes) && push!(_attributes, namedattribute("mesh_axes", mesh_axes))

    return IR.create_operation(
        "mesh.all_gather",
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
`all_reduce`

The accumulation element type is specified by the result type and
it does not need to match the input element type.
The input element is converted to the result element type before
performing the reduction.

Attributes:
`reduction`: Indicates the reduction method.

# Example
```
%1 = mesh.all_reduce %0 on @mesh0 mesh_axes = [1, 0] reduction = <max>
  : tensor<3x4xf32> -> tensor<3x4xf64>
```
"""
function all_reduce(
    input::Value;
    result::IR.Type,
    mesh,
    mesh_axes=nothing,
    reduction=nothing,
    location=Location(),
)
    _results = IR.Type[result,]
    _operands = Value[input,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[namedattribute("mesh", mesh),]
    !isnothing(mesh_axes) && push!(_attributes, namedattribute("mesh_axes", mesh_axes))
    !isnothing(reduction) && push!(_attributes, namedattribute("reduction", reduction))

    return IR.create_operation(
        "mesh.all_reduce",
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
`all_slice`

Slice along the `slice_axis` tensor axis.
This operation can be thought of as the inverse of all-gather.
Technically, it is not required that all processes have the same input tensor.
Each process will slice a piece of its local tensor based on its in-group device index.
The operation does not communicate data between devices. 

# Example
```mlir
mesh.mesh @mesh0(shape = 2x2)
...
%1 = mesh.all_slice %0 on @mesh0 mesh_axes = [1] slice_axis = 1
  : tensor<2x4xi8> -> tensor<2x2xi8>
```
Input:
```
+-------------+
|  1  2  5  6 | <- devices (0, 0) and (0, 1)
|  3  4  7  8 |
+-------------+
|  9 10 13 14 | <- devices (1, 0) and (1, 1)
| 11 12 15 16 |
+-------------+
```
Result:
```
gather tensor
axis 1
------------>
                 +-------+-------+
device (0, 0) -> |  1  2 |  5  6 | <- device (0, 1)
                 |  3  4 |  7  8 |
                 +-------+-------+
device (1, 0) -> |  9 10 | 13 14 | <- device (1, 1)
                 | 11 12 | 15 16 |
                 +-------+-------+
```
"""
function all_slice(
    input::Value; result::IR.Type, mesh, mesh_axes=nothing, slice_axis, location=Location()
)
    _results = IR.Type[result,]
    _operands = Value[input,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[
        namedattribute("mesh", mesh), namedattribute("slice_axis", slice_axis)
    ]
    !isnothing(mesh_axes) && push!(_attributes, namedattribute("mesh_axes", mesh_axes))

    return IR.create_operation(
        "mesh.all_slice",
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
`all_to_all`

Performs an all-to-all on tensor pieces split along `split_axis`.
The resulting pieces are concatenated along `concat_axis` on ech device.

# Example
```
mesh.mesh @mesh0(shape = 3)
...
%1 = mesh.all_to_all %0 on @mesh0 mesh_axes = [0]
  split_axis = 0 concat_axis = 0
  : tensor<3x2xi8> -> tensor<3x2xi8>
```
Input:
```
 device  device  device
 (0)     (1)     (2)
+-------+-------+-------+  | split and concat along
| 11 12 | 21 22 | 31 32 |  | tensor axis 0
| 13 14 | 23 24 | 33 34 |  ↓
| 15 16 | 25 26 | 35 36 |
+-------+-------+-------+
```
Result:
```
 device  device  device
 (0)     (1)     (2)
+-------+-------+-------+
| 11 12 | 13 14 | 15 16 |
| 21 22 | 23 24 | 25 26 |
| 31 32 | 33 34 | 35 36 |
+-------+-------+-------+
```
"""
function all_to_all(
    input::Value;
    result::IR.Type,
    mesh,
    mesh_axes=nothing,
    split_axis,
    concat_axis,
    location=Location(),
)
    _results = IR.Type[result,]
    _operands = Value[input,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[
        namedattribute("mesh", mesh),
        namedattribute("split_axis", split_axis),
        namedattribute("concat_axis", concat_axis),
    ]
    !isnothing(mesh_axes) && push!(_attributes, namedattribute("mesh_axes", mesh_axes))

    return IR.create_operation(
        "mesh.all_to_all",
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
`broadcast`

Broadcast the tensor on `root` to all devices in each respective group.
The operation broadcasts along mesh axes `mesh_axes`.
The `root` device specifies the in-group multi-index that is broadcast to
all other devices in the group.

# Example
```
mesh.mesh @mesh0(shape = 2x2)

%1 = mesh.broadcast %0 on @mesh0
  mesh_axes = [0]
  root = [0]
  : (tensor<2xi8>) -> tensor<2xi8>
```

Input:
```
                 +-------+-------+                   | broadcast
device (0, 0) -> |  1  2 |  3  4 | <- device (0, 1)  | along axis 0
                 +-------+-------+                   ↓
device (1, 0) -> |       |       | <- device (1, 1) 
                 +-------+-------+
```

Output:
```
                 +-------+-------+
device (0, 0) -> |  1  2 |  3  4 | <- device (0, 1)
                 +-------+-------+
device (1, 0) -> |  1  2 |  3  4 | <- device (1, 1)
                 +-------+-------+
```
"""
function broadcast(
    input::Value,
    root_dynamic::Vector{Value};
    result::IR.Type,
    mesh,
    mesh_axes=nothing,
    root,
    location=Location(),
)
    _results = IR.Type[result,]
    _operands = Value[input, root_dynamic...]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[namedattribute("mesh", mesh), namedattribute("root", root)]
    !isnothing(mesh_axes) && push!(_attributes, namedattribute("mesh_axes", mesh_axes))

    return IR.create_operation(
        "mesh.broadcast",
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
`gather`

Gathers on device `root` along the `gather_axis` tensor axis.
`root` specifies the coordinates of a device along `mesh_axes`.
It uniquely identifies the root device for each device group.
The result tensor on non-root devices is undefined.
Using it will result in undefined behavior.

# Example
```mlir
mesh.mesh @mesh0(shape = 2x2)
...
%1 = mesh.gather %0 on @mesh0 mesh_axes = [1]
  gather_axis = 1 root = [1]
  : (tensor<2x2xi8>) -> tensor<2x4xi8>
```
Input:
```
                  gather tensor
                  axis 1
                  ------------>
                 +-------+-------+
device (0, 0) -> |  1  2 |  5  6 | <- device (0, 1)
                 |  3  4 |  7  8 |
                 +-------+-------+
device (1, 0) -> |  9 10 | 13 14 | <- device (1, 1)
                 | 11 12 | 15 16 |
                 +-------+-------+
```
Result:
```
+-------------+
|  1  2  5  6 | <- devices (0, 1)
|  3  4  7  8 |
+-------------+
|  9 10 13 14 | <- devices (1, 1)
| 11 12 15 16 |
+-------------+
```
Devices `(0, 0)` and `(1, 0)` have undefined result.
"""
function gather(
    input::Value,
    root_dynamic::Vector{Value};
    result::IR.Type,
    mesh,
    mesh_axes=nothing,
    gather_axis,
    root,
    location=Location(),
)
    _results = IR.Type[result,]
    _operands = Value[input, root_dynamic...]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[
        namedattribute("mesh", mesh),
        namedattribute("gather_axis", gather_axis),
        namedattribute("root", root),
    ]
    !isnothing(mesh_axes) && push!(_attributes, namedattribute("mesh_axes", mesh_axes))

    return IR.create_operation(
        "mesh.gather",
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
`mesh_`

The mesh.mesh operation is a symbol operation that identifies a specific
mesh. The operation has three attributes:

1. `sym_name`: This attribute uniquely identifies the name of the mesh.
This name serves as a symbolic reference to the mesh throughout
the MLIR module, allowing for consistent referencing and easier debugging.

2. `shape`: This attribute represents the shape of the device mesh.
It uses the same notation as a tensor shape. Also allowing for dynamic
dimensions.
This flexibility allows for dynamic device assignment or configurations
where the exact number of devices might not be determined during compile
time.
For example `2x?x4`.

# Example
```
// A device mesh with 3 axes, the total device number is 4 * 8 * 12
// The dimension sizes are 4, 8, 12 
mesh.mesh @mesh0(shape = 4x8x12)

// A device mesh with 2 axes, the total device number is unknown
// The first dimension size is 4 and the second is unknown
mesh.mesh @mesh1(shape = 4x?)

// A device mesh with 2 axes, the total device number is unknown
// The first dimension size is unknown and the second is 4
mesh.mesh @mesh2(shape = ?x4)

// A device mesh with 2 axes, the number of devices along both axes
// is unknown
mesh.mesh @mesh3(shape = ?x?)
```
"""
function mesh_(; sym_name, shape, location=Location())
    _results = IR.Type[]
    _operands = Value[]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[
        namedattribute("sym_name", sym_name), namedattribute("shape", shape)
    ]

    return IR.create_operation(
        "mesh.mesh",
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
`mesh_shape`

"""
function mesh_shape(; result::Vector{IR.Type}, mesh, axes=nothing, location=Location())
    _results = IR.Type[result...,]
    _operands = Value[]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[namedattribute("mesh", mesh),]
    !isnothing(axes) && push!(_attributes, namedattribute("axes", axes))

    return IR.create_operation(
        "mesh.mesh_shape",
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
`process_linear_index`

# Example
```
%idx = mesh.process_linear_index on @mesh : index
```
if `@mesh` has shape `(10, 20, 30)`, a device with multi
index `(1, 2, 3)` will have linear index `3 + 30*2 + 20*30*1`.
"""
function process_linear_index(;
    result=nothing::Union{Nothing,IR.Type}, mesh, location=Location()
)
    _results = IR.Type[]
    _operands = Value[]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[namedattribute("mesh", mesh),]
    !isnothing(result) && push!(_results, result)

    return IR.create_operation(
        "mesh.process_linear_index",
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
`process_multi_index`

It is used in the SPMD format of IR.
The `axes` mush be non-negative and less than the total number of mesh axes.
If the axes are empty then get the index along all axes.
"""
function process_multi_index(;
    result::Vector{IR.Type}, mesh, axes=nothing, location=Location()
)
    _results = IR.Type[result...,]
    _operands = Value[]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[namedattribute("mesh", mesh),]
    !isnothing(axes) && push!(_attributes, namedattribute("axes", axes))

    return IR.create_operation(
        "mesh.process_multi_index",
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
`recv`

Receive from a device within a device group.
"""
function recv(
    input::Value,
    source_dynamic::Vector{Value};
    result::IR.Type,
    mesh,
    mesh_axes=nothing,
    source=nothing,
    location=Location(),
)
    _results = IR.Type[result,]
    _operands = Value[input, source_dynamic...]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[namedattribute("mesh", mesh),]
    !isnothing(mesh_axes) && push!(_attributes, namedattribute("mesh_axes", mesh_axes))
    !isnothing(source) && push!(_attributes, namedattribute("source", source))

    return IR.create_operation(
        "mesh.recv",
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
`reduce`

Reduces on device `root` within each device group.
`root` specifies the coordinates of a device along `mesh_axes`.
It uniquely identifies the root device within its device group.
The accumulation element type is specified by the result type and
it does not need to match the input element type.
The input element is converted to the result element type before
performing the reduction.

Attributes:
`reduction`: Indicates the reduction method.

# Example
```
%1 = mesh.reduce %0 on @mesh0 mesh_axes = [1, 0]
  reduction = <max> root = [2, 3]
  : (tensor<3x4xf32>) -> tensor<3x4xf64>
```
"""
function reduce(
    input::Value,
    root_dynamic::Vector{Value};
    result::IR.Type,
    mesh,
    mesh_axes=nothing,
    reduction=nothing,
    root,
    location=Location(),
)
    _results = IR.Type[result,]
    _operands = Value[input, root_dynamic...]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[namedattribute("mesh", mesh), namedattribute("root", root)]
    !isnothing(mesh_axes) && push!(_attributes, namedattribute("mesh_axes", mesh_axes))
    !isnothing(reduction) && push!(_attributes, namedattribute("reduction", reduction))

    return IR.create_operation(
        "mesh.reduce",
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
`reduce_scatter`

After the reduction, the result is scattered within each device group.
The tensor is split along `scatter_axis` and the pieces distributed
across the device group.
# Example
```
mesh.mesh @mesh0(shape = 2x2)
...
%1 = mesh.reduce_scatter %0 on @mesh0 mesh_axes = [1]
  reduction = <max> scatter_axis = 0
  : tensor<3x4xf32> -> tensor<1x4xf64>
```
Input:
```
                          device
                          (0, 1)
                             ↓
                 +-------+-------+  | scatter tensor
device (0, 0) -> |  1  2 |  5  6 |  | axis 0
                 |  3  4 |  7  8 |  ↓
                 +-------+-------+
device (1, 0) -> |  9 10 | 13 14 |
                 | 11 12 | 15 16 |
                 +-------+-------+
                            ↑
                          device
                          (1, 1)
```
Result:
```
+-------+
|  6  8 | <- devices (0, 0)
+-------+
| 10 12 | <- devices (0, 1)
+-------+
| 22 24 | <- devices (1, 0)
+-------+
| 26 28 | <- devices (1, 1)
+-------+
```
"""
function reduce_scatter(
    input::Value;
    result::IR.Type,
    mesh,
    mesh_axes=nothing,
    reduction=nothing,
    scatter_axis,
    location=Location(),
)
    _results = IR.Type[result,]
    _operands = Value[input,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[
        namedattribute("mesh", mesh), namedattribute("scatter_axis", scatter_axis)
    ]
    !isnothing(mesh_axes) && push!(_attributes, namedattribute("mesh_axes", mesh_axes))
    !isnothing(reduction) && push!(_attributes, namedattribute("reduction", reduction))

    return IR.create_operation(
        "mesh.reduce_scatter",
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
`scatter`

For each device group split the input tensor on the `root` device along
axis `scatter_axis` and scatter the parts across the group devices.

# Example
```
mesh.mesh @mesh0(shape = 2x2)
%1 = mesh.scatter %0 on @mesh0 mesh_axes = [0]
  scatter_axis = 0
  root = [1]
  : (tensor<2x2xi8>) -> tensor<1x2xi8>
```

Input:
```
                          device
                          (0, 1)
                             ↓
                 +-------+-------+  | scatter tensor
device (0, 0) -> |       |       |  | axis 0
                 |       |       |  ↓
                 +-------+-------+
device (1, 0) -> |  1  2 |  5  6 |
                 |  3  4 |  7  8 |
                 +-------+-------+
                            ↑
                          device
                          (1, 1)
```

Result:
```
                          device
                          (0, 1)
                             ↓
                 +-------+-------+
device (0, 0) -> |  1  2 |  5  6 |
                 +-------+-------+ 
device (1, 0) -> |  3  4 |  7  8 |
                 +-------+-------+
                            ↑
                          device
                          (1, 1)
```
"""
function scatter(
    input::Value,
    root_dynamic::Vector{Value};
    result::IR.Type,
    mesh,
    mesh_axes=nothing,
    scatter_axis,
    root,
    location=Location(),
)
    _results = IR.Type[result,]
    _operands = Value[input, root_dynamic...]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[
        namedattribute("mesh", mesh),
        namedattribute("scatter_axis", scatter_axis),
        namedattribute("root", root),
    ]
    !isnothing(mesh_axes) && push!(_attributes, namedattribute("mesh_axes", mesh_axes))

    return IR.create_operation(
        "mesh.scatter",
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
`send`

Send from one device to another within a device group.
"""
function send(
    input::Value,
    destination_dynamic::Vector{Value};
    result::IR.Type,
    mesh,
    mesh_axes=nothing,
    destination,
    location=Location(),
)
    _results = IR.Type[result,]
    _operands = Value[input, destination_dynamic...]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[
        namedattribute("mesh", mesh), namedattribute("destination", destination)
    ]
    !isnothing(mesh_axes) && push!(_attributes, namedattribute("mesh_axes", mesh_axes))

    return IR.create_operation(
        "mesh.send",
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
`shard`

The mesh.shard operation is designed to specify and guide the sharding
behavior of a tensor value across a mesh topology. This operation has one
operand and two attributes:

1. `input`: This operand represents the tensor value that needs to be
annotated for sharding.

2. `shard`: This attribute is type of `MeshSharding`, which is the core data
structure to represent distribution of a tensor on a mesh.

3. `annotate_for_users`: A unit attribute addressing the scenario when a
tensor\'s sharding annotation differs based on its context of use (either as
a result or an operand). If specified, the sharding pertains to specific
users of the tensor value, indicating how it should be considered when used
as an operand in subsequent operations. If not, the sharding applies to the
operation that defines the tensor value.

# Example
```
func.func @only_result_annotated(%arg0 : tensor<4x8xf32>) -> () {
  %0 = mesh.shard %arg0 to <@mesh0, [[0]]> : tensor<4x8xf32>
  ...
}

func.func @only_operand_annotated(%arg0 : tensor<4x8xf32>) -> () {
  %0 = mesh.shard %arg0 to <@mesh0, [[0]]> annotate_for_users : tensor<4x8xf32>
  ...
}

// The first mesh.shard op applies to %arg0, the second mesh.shard op
// applies for the operand of op0, the third mesh.shard op applies for the
// operand of op2
func.func @both_result_and_multi_operands_annotated(
    %arg0 : tensor<4x8xf32>) -> () {
  %0 = mesh.shard %arg0 to <@mesh0, [[0]]> : tensor<4x8xf32>
  %1 = mesh.shard %0 to <@mesh0, [[1]]> annotate_for_users : tensor<4x8xf32>
  %2 = mesh.shard %0 to <@mesh0, [[2]]> annotate_for_users : tensor<4x8xf32>
  \"op0\"(%1) : ...
  \"op1\"(%2) : ...
  ...
}
```

The following usages are undefined:
```
func.func @annotate_on_same_result_with_different_sharding(
    %arg0 : tensor<4x8xf32>) -> () {
  %0 = mesh.shard %arg0 to <@mesh0, [[0]]> : tensor<4x8xf32>
  %1 = mesh.shard %0 to <@mesh0, [[1]]> : tensor<4x8xf32>
  ...
}

func.func @annotate_on_same_result_same_value_with_different_sharding(
    %arg0 : tensor<4x8xf32>) -> () {
  %0 = mesh.shard %arg0 to <@mesh0, [[0]]> : tensor<4x8xf32>
  %1 = mesh.shard %arg0 to <@mesh0, [[1]]> : tensor<4x8xf32>
  ...
}

func.func @annotate_on_same_operand_with_different_sharding(
    %arg0 : tensor<4x8xf32>) -> () {
  %0 = mesh.shard %arg0 to <@mesh0, [[0]]> annotate_for_users : tensor<4x8xf32>
  %1 = mesh.shard %0 to <@mesh0, [[1]]> annotate_for_users : tensor<4x8xf32>
  ...
}

func.func @result_annotated_after_operand(
    %arg0 : tensor<4x8xf32>) -> () {
  %0 = mesh.shard %arg0 to <@mesh0, [[0]]> annotate_for_users : tensor<4x8xf32>
  %1 = mesh.shard %0 to <@mesh0, [[1]]> : tensor<4x8xf32>
  ...
}
```
"""
function shard(
    src::Value;
    result=nothing::Union{Nothing,IR.Type},
    shard,
    annotate_for_users=nothing,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[src,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[namedattribute("shard", shard),]
    !isnothing(result) && push!(_results, result)
    !isnothing(annotate_for_users) &&
        push!(_attributes, namedattribute("annotate_for_users", annotate_for_users))

    return IR.create_operation(
        "mesh.shard",
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
`shift`

Within each device group shift along mesh axis `shift_axis` by an offset
`offset`.
The result on devices that do not have a corresponding source is undefined.
`shift_axis` must be one of `mesh_axes`.
If the `rotate` attribute is present,
instead of a shift a rotation is done.

# Example
```
mesh.mesh @mesh0(shape = 2x4)
%1 = mesh.shift on @mesh0 mesh_axes = [1]
  shift_axis = 1 offset = 2 rotate
  : tensor<2xi8> -> tensor<2xi8>
```

Input:
```
mesh axis 1
----------->

+----+----+----+----+
|  1 |  2 |  3 |  4 |
+----+----+----+----+
|  5 |  6 |  7 |  8 |
+----+----+----+----+
```

Result:
```
+----+----+----+----+
|  3 |  4 |  1 |  2 |
+----+----+----+----+
|  7 |  8 |  5 |  6 |
+----+----+----+----+
```
"""
function shift(
    input::Value;
    result::IR.Type,
    mesh,
    mesh_axes=nothing,
    shift_axis,
    offset,
    rotate=nothing,
    location=Location(),
)
    _results = IR.Type[result,]
    _operands = Value[input,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[
        namedattribute("mesh", mesh),
        namedattribute("shift_axis", shift_axis),
        namedattribute("offset", offset),
    ]
    !isnothing(mesh_axes) && push!(_attributes, namedattribute("mesh_axes", mesh_axes))
    !isnothing(rotate) && push!(_attributes, namedattribute("rotate", rotate))

    return IR.create_operation(
        "mesh.shift",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

end # mesh
