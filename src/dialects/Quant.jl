module Quantization

import ...IR: NamedAttribute, MLIRType, Value, Location, Block, Region, Attribute, create_operation, context, IndexType
import ..Dialects: namedattribute, operandsegmentsizes
import ...API


"""
dcast

A DequantizeCast op `dcast` represents the inverse of a `qcast`,
converting back from a quantized to quantizable (expressed) type.

Like `qcast`s, a `dcast` is allowed to have both its operand and result
as non quantized types. This facilitates transformations and marks edges
where the computation must be carried out in the expressed type.

Especially early in transformation, it is common to have `dcast`s on
all operands to ops that must operate with the expressed type (typically
math ops prior to lowering to target-specific, quantized kernels).
  
"""
function dcast(arg::Value; res::MLIRType, location=Location())
    results = MLIRType[res, ]
    operands = Value[arg, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "quant.dcast", location,
        results=results,
        operands=operands,
        owned_regions=owned_regions,
        successors=successors,
        attributes=attributes,
        result_inference=false
    )
end

"""
qcast

A QuantizeCast `qcast` represents a potential type shift from a quantizable
type to a quantized type.

At runtime, a `qcast` will apply the transformation expressed by its
operand and result type. For flexibility during transformation, it is also
possible to have a `qcast` that performs no transformation (both its
operand and result type are quantizable).

A `qcast` will typically originate from either:
  a) An expressed or implied constraint in the source dialect which signals
     that a certain level of quantization is possible or required.
  b) An inference made by a quantization algorithm indicating that a
     quantized representation may be acceptable.

Especially early in transformation, it is common to have pairs of
`qcast` and `dcast` at points where a transition to a quantized type is
required. In addition, it is also common to have an identity `qcast`
(where the operand and result type are not quantized) at all points where
it is legal to use a quantized representation (but is not known to be
acceptable).
  
"""
function qcast(arg::Value; res::MLIRType, location=Location())
    results = MLIRType[res, ]
    operands = Value[arg, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "quant.qcast", location,
        results=results,
        operands=operands,
        owned_regions=owned_regions,
        successors=successors,
        attributes=attributes,
        result_inference=false
    )
end

"""
scast

A StorageCast `scast` represents a cast from or to a type based on the
storage type and a type based on a corresponding quantized type.

This op exists to ensure type coherency for between parts of the computation
which are operating directly on an underlying storage type and those which
operate on quantized values.

Examples from storage to quantized type:
```
i8 -> !quant<\"uniform[i8:f32]{1.0}\">
```
```
tensor<4xi8> -> tensor<4x!quant<\"uniform[i8:f32]{1.0}\">>
```
```
vector<4xi8> -> vector<4x!quant<\"uniform[i8:f32]{1.0}\">>
```
  
"""
function scast(arg::Value; res::MLIRType, location=Location())
    results = MLIRType[res, ]
    operands = Value[arg, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "quant.scast", location,
        results=results,
        operands=operands,
        owned_regions=owned_regions,
        successors=successors,
        attributes=attributes,
        result_inference=false
    )
end

end # Quantization
