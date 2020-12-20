# ------------ Operation state alias and APIs ------------ #

const OperationState = MLIR.API.MlirOperationState

create_operation_state(name::StringRef, l::Location) = MLIR.API.mlirOperationStateGet(name, l)
create_operation_state(name::String, l::Location) = create_operation_state(StringRef(name), l)
OperationState(name::String, l::Location) = create_operation_state(name, l)

function Base.display(state::OperationState)
    println("  __________________________________\n")
    println("             OperationState\n")
    for f in fieldnames(OperationState)
        println(" $f : $(getfield(state, f))")
    end
    println("  __________________________________\n")
end

@doc(
"""
const OperationState = MLIR.API.MlirOperationState
""", OperationState)

# ------------ Operation and APIs ------------ #

struct Operation
    ref::MLIR.API.MlirOperation
end

unwrap(o::Operation) = o.ref

create_operation(state::OperationState) = Operation(MLIR.API.mlirOperationCreate(Ref(state)))
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
dump(op::Operation) = MLIR.API.mlirOperationDump(unwrap(op))

@inline Operation(state::OperationState) = create_operation(state)
function Operation(name::String, l::Location, 
                   num_results::Int, result_types::Vector{Type},
                   num_operands::Int, operands::Vector{Value},
                   num_regions::Int, regions::Vector,
                   num_successors::Int, successors::Vector,
                   num_attributes::Int, attributes::Vector{NamedAttribute})
    state = OperationState(name, l, num_results, result_types, num_operands, operands, num_regions, regions, num_successors, successors, num_attributes, attributes)
    Operation(state)
end

@doc(
"""
const Operation = MLIR.API.MlirOperation
""", Operation)
