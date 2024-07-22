module arm_sve

import ...IR: IR, NamedAttribute, Value, Location, Block, Region, Attribute, context, IndexType
import ..Dialects: namedattribute, operandsegmentsizes


"""
`intr_fadd`

"""
function intr_fadd(operand_0::Value, operand_1::Value, operand_2::Value; res::IR.Type, location=Location())
    results = IR.Type[res, ]
    operands = Value[operand_0, operand_1, operand_2, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "arm_sve.intr.fadd", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`masked_addf`

The `arm_sve.masked.addf` operation takes one scalable vector mask
and two scalable vector operands, and perform floating point addition on active lanes. Inactive lanes will keep the value of
the first operand.
"""
function masked_addf(mask::Value, src1::Value, src2::Value; res::IR.Type, location=Location())
    results = IR.Type[res, ]
    operands = Value[mask, src1, src2, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "arm_sve.masked.addf", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`intr_add`

"""
function intr_add(operand_0::Value, operand_1::Value, operand_2::Value; res::IR.Type, location=Location())
    results = IR.Type[res, ]
    operands = Value[operand_0, operand_1, operand_2, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "arm_sve.intr.add", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`masked_addi`

The `arm_sve.masked.addi` operation takes one scalable vector mask
and two scalable vector operands, and perform integer addition on active lanes. Inactive lanes will keep the value of
the first operand.
"""
function masked_addi(mask::Value, src1::Value, src2::Value; res::IR.Type, location=Location())
    results = IR.Type[res, ]
    operands = Value[mask, src1, src2, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "arm_sve.masked.addi", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`intr_fdiv`

"""
function intr_fdiv(operand_0::Value, operand_1::Value, operand_2::Value; res::IR.Type, location=Location())
    results = IR.Type[res, ]
    operands = Value[operand_0, operand_1, operand_2, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "arm_sve.intr.fdiv", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`masked_divf`

The `arm_sve.masked.divf` operation takes one scalable vector mask
and two scalable vector operands, and perform floating point division on active lanes. Inactive lanes will keep the value of
the first operand.
"""
function masked_divf(mask::Value, src1::Value, src2::Value; res::IR.Type, location=Location())
    results = IR.Type[res, ]
    operands = Value[mask, src1, src2, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "arm_sve.masked.divf", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`intr_fmul`

"""
function intr_fmul(operand_0::Value, operand_1::Value, operand_2::Value; res::IR.Type, location=Location())
    results = IR.Type[res, ]
    operands = Value[operand_0, operand_1, operand_2, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "arm_sve.intr.fmul", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`masked_mulf`

The `arm_sve.masked.mulf` operation takes one scalable vector mask
and two scalable vector operands, and perform floating point multiplication on active lanes. Inactive lanes will keep the value of
the first operand.
"""
function masked_mulf(mask::Value, src1::Value, src2::Value; res::IR.Type, location=Location())
    results = IR.Type[res, ]
    operands = Value[mask, src1, src2, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "arm_sve.masked.mulf", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`intr_mul`

"""
function intr_mul(operand_0::Value, operand_1::Value, operand_2::Value; res::IR.Type, location=Location())
    results = IR.Type[res, ]
    operands = Value[operand_0, operand_1, operand_2, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "arm_sve.intr.mul", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`masked_muli`

The `arm_sve.masked.muli` operation takes one scalable vector mask
and two scalable vector operands, and perform integer multiplication on active lanes. Inactive lanes will keep the value of
the first operand.
"""
function masked_muli(mask::Value, src1::Value, src2::Value; res::IR.Type, location=Location())
    results = IR.Type[res, ]
    operands = Value[mask, src1, src2, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "arm_sve.masked.muli", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`intr_sdiv`

"""
function intr_sdiv(operand_0::Value, operand_1::Value, operand_2::Value; res::IR.Type, location=Location())
    results = IR.Type[res, ]
    operands = Value[operand_0, operand_1, operand_2, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "arm_sve.intr.sdiv", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`masked_divi_signed`

The `arm_sve.masked.divi_signed` operation takes one scalable vector mask
and two scalable vector operands, and perform integer signed division on active lanes. Inactive lanes will keep the value of
the first operand.
"""
function masked_divi_signed(mask::Value, src1::Value, src2::Value; res::IR.Type, location=Location())
    results = IR.Type[res, ]
    operands = Value[mask, src1, src2, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "arm_sve.masked.divi_signed", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`intr_fsub`

"""
function intr_fsub(operand_0::Value, operand_1::Value, operand_2::Value; res::IR.Type, location=Location())
    results = IR.Type[res, ]
    operands = Value[operand_0, operand_1, operand_2, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "arm_sve.intr.fsub", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`masked_subf`

The `arm_sve.masked.subf` operation takes one scalable vector mask
and two scalable vector operands, and perform floating point subtraction on active lanes. Inactive lanes will keep the value of
the first operand.
"""
function masked_subf(mask::Value, src1::Value, src2::Value; res::IR.Type, location=Location())
    results = IR.Type[res, ]
    operands = Value[mask, src1, src2, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "arm_sve.masked.subf", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`intr_sub`

"""
function intr_sub(operand_0::Value, operand_1::Value, operand_2::Value; res::IR.Type, location=Location())
    results = IR.Type[res, ]
    operands = Value[operand_0, operand_1, operand_2, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "arm_sve.intr.sub", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`masked_subi`

The `arm_sve.masked.subi` operation takes one scalable vector mask
and two scalable vector operands, and perform integer subtraction on active lanes. Inactive lanes will keep the value of
the first operand.
"""
function masked_subi(mask::Value, src1::Value, src2::Value; res::IR.Type, location=Location())
    results = IR.Type[res, ]
    operands = Value[mask, src1, src2, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "arm_sve.masked.subi", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`intr_udiv`

"""
function intr_udiv(operand_0::Value, operand_1::Value, operand_2::Value; res::IR.Type, location=Location())
    results = IR.Type[res, ]
    operands = Value[operand_0, operand_1, operand_2, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "arm_sve.intr.udiv", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`masked_divi_unsigned`

The `arm_sve.masked.divi_unsigned` operation takes one scalable vector mask
and two scalable vector operands, and perform integer unsigned division on active lanes. Inactive lanes will keep the value of
the first operand.
"""
function masked_divi_unsigned(mask::Value, src1::Value, src2::Value; res::IR.Type, location=Location())
    results = IR.Type[res, ]
    operands = Value[mask, src1, src2, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "arm_sve.masked.divi_unsigned", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`intr_sdot`

"""
function intr_sdot(operand_0::Value, operand_1::Value, operand_2::Value; res::IR.Type, location=Location())
    results = IR.Type[res, ]
    operands = Value[operand_0, operand_1, operand_2, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "arm_sve.intr.sdot", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`sdot`

SDOT: Signed integer addition of dot product.

This function maps to the SDOT instruction, and it takes signless integer
operands that the operation interprets as signed. It partitions the second
and third vector inputs into groups of four elements. They calculate the dot
product of each group (without loss of precision) and then add each result
to the overlapping element of the first vector input.

Source:
https://developer.arm.com/documentation/100987/0000
"""
function sdot(acc::Value, src1::Value, src2::Value; dst::IR.Type, location=Location())
    results = IR.Type[dst, ]
    operands = Value[acc, src1, src2, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "arm_sve.sdot", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`intr_smmla`

"""
function intr_smmla(operand_0::Value, operand_1::Value, operand_2::Value; res::IR.Type, location=Location())
    results = IR.Type[res, ]
    operands = Value[operand_0, operand_1, operand_2, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "arm_sve.intr.smmla", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`smmla`

SMMLA: Signed integer matrix multiply-accumulate.

This function maps to the SMMLA instruction, and it takes signless integer
operands that the operation interprets as signed. It partitions the inputs
into 128-bit quadwords, with the first input containing a row-by-row 2×2
matrix of 32-bit integers, the second input containing a row-by-row 2×8
matrix of 8-bit integers, and the third input containing a column-by-column
8×2 matrix of 8-bit integers. For each quadword, they multiply the second
input matrix by the third input matrix using natural arithmetic and then add
the result to the first input using modular arithmetic.

Source:
https://developer.arm.com/documentation/100987/0000
"""
function smmla(acc::Value, src1::Value, src2::Value; dst::IR.Type, location=Location())
    results = IR.Type[dst, ]
    operands = Value[acc, src1, src2, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "arm_sve.smmla", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`intr_udot`

"""
function intr_udot(operand_0::Value, operand_1::Value, operand_2::Value; res::IR.Type, location=Location())
    results = IR.Type[res, ]
    operands = Value[operand_0, operand_1, operand_2, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "arm_sve.intr.udot", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`udot`

UDOT: Unsigned integer addition of dot product.

This function maps to the UDOT instruction, and it takes signless integer
operands that the operation interprets as unsigned. It partitions the second
and third vector inputs into groups of four elements. They calculate the dot
product of each group (without loss of precision) and then add each result
to the overlapping element of the first vector input.

Source:
https://developer.arm.com/documentation/100987/0000
"""
function udot(acc::Value, src1::Value, src2::Value; dst::IR.Type, location=Location())
    results = IR.Type[dst, ]
    operands = Value[acc, src1, src2, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "arm_sve.udot", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`intr_ummla`

"""
function intr_ummla(operand_0::Value, operand_1::Value, operand_2::Value; res::IR.Type, location=Location())
    results = IR.Type[res, ]
    operands = Value[operand_0, operand_1, operand_2, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "arm_sve.intr.ummla", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`ummla`

UMMLA: Unsigned integer matrix multiply-accumulate.

This function maps to the UMMLA instruction, and it takes signless integer
operands that the operation interprets as unsigned. It partitions the inputs
into 128-bit quadwords, with the first input containing a row-by-row 2×2
matrix of 32-bit integers, the second input containing a row-by-row 2×8
matrix of 8-bit integers, and the third input containing a column-by-column
8×2 matrix of 8-bit integers. For each quadword, they multiply the second
input matrix by the third input matrix using natural arithmetic and then add
the result to the first input using modular arithmetic.

Source:
https://developer.arm.com/documentation/100987/0000
"""
function ummla(acc::Value, src1::Value, src2::Value; dst::IR.Type, location=Location())
    results = IR.Type[dst, ]
    operands = Value[acc, src1, src2, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "arm_sve.ummla", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

end # arm_sve
