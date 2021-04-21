#####
##### MlirOperation alias and APIs
#####

struct Operation
    ref::MLIR.API.MlirOperation
end

@doc(
"""
const Operation = MLIR.API.MlirOperation
""", Operation)

unwrap(o::Operation) = o.ref

destroy!(op::Operation) = MLIR.API.mlirOperationDestroy(unwrap(op))
is_null(op::Operation) = MLIR.API.mlirOperationIsNull(unwrap(op))
Base.:(==)(op1::Operation, op2::Operation) = MLIR.API.mlirOperationEqual(unwrap(op1), unwrap(op2))
get_block(op::Operation) = MLIR.API.mlirOperationGetBlock(unwrap(op))
get_parent_operation(op::Operation) = MLIR.API.mlirOperationGetParentOperation(unwrap(op))
get_num_regions(op::Operation) = MLIR.API.mlirOperationGetNumRegions(unwrap(op))
get_region(op::Operation, pos::Int) = MLIR.API.mlirOperationGetRegion(unwrap(op), pos)
get_next_in_block(op::Operation) = MLIR.API.mlirOperationGetNextInBlock(unwrap(op))
get_num_operands(op::Operation) = MLIR.API.mlirOperationGetNumOperands(unwrap(op))
get_operand(op::Operation, pos::Int) = MLIR.API.mlirOperationGetOperand(unwrap(op), pos)
get_num_results(op::Operation) = MLIR.API.mlirOperationGetNumResults(unwrap(op))
get_result(op::Operation, pos::Int) = MLIR.API.mlirOperationGetResult(unwrap(op), pos)
verify(op::Operation) = MLIR.API.mlirOperationVerify(unwrap(op))
function dump(op::Operation)
    @assert(verify(op))
    MLIR.API.mlirOperationDump(unwrap(op))
end
