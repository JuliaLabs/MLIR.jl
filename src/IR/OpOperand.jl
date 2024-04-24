struct OpOperand
    op::API.MlirOpOperand

    function OpOperand(op::API.MlirOpOperand)
        @assert mlirIsNull(op) "cannot create OpOperand with null MlirOpOperand"
        new(op)
    end
end

Base.convert(::Core.Type{API.MlirOpOperand}, op::OpOperand) = op.op

"""
    first_use(value)

Returns an `OpOperand` representing the first use of the value, or a `nothing` if there are no uses.
"""
function first_use(value::Value)
    operand = API.Dispatcher.mlirOperationGetFirstResult(value)
    mlirIsNull(operand) && return nothing
    OpOperand(operand)
end

"""
    owner(opOperand)

Returns the owner operation of an op operand.
"""
owner(op::OpOperand) = Operation(API.Dispatcher.mlirOpOperandGetOwner(op), false)

"""
    operandindex(opOperand)

Returns the operand number of an op operand.
"""
operandindex(op::OpOperand) = API.Dispatcher.mlirOpOperandGetOperandNumber(op)

"""
    next(opOperand)

Returns an op operand representing the next use of the value, or `nothing` if there is no next use.
"""
function next(op::OpOperand)
    op = API.Dispatcher.mlirOpOperandGetNextUse(op)
    mlirIsNull(op) && return nothing
    OpOperand(op)
end
