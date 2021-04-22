#####
##### MlirOperationState alias and APIs
#####

const OperationState = MLIR.API.MlirOperationState

@doc(
"""
const OperationState = MLIR.API.MlirOperationState
""", OperationState)

create_operation_state(name::StringRef, l::Location) = MLIR.API.mlirOperationStateGet(name, l)
create_operation_state(name::String, l::Location) = create_operation_state(StringRef(name), l)

# Constructor.
OperationState(name::String, l::Location) = create_operation_state(name, l)

# Builders for state.
push_results!(state::OperationState, result::Type) = MLIR.API.mlirOperationStateAddResults(Ref(state), 1, Ref(result))
push_results!(state::OperationState, results::Vector{Type}) = MLIR.API.mlirOperationStateAddResults(Ref(state), length(results), results)

push_operands!(state::OperationState, operand::Value) = MLIR.API.mlirOperationStateAddOperands(Ref(state), 1, Ref(operand))
push_operands!(state::OperationState, operands::Vector{Value}) = MLIR.API.mlirOperationStateAddOperands(Ref(state), length(operands), operands)

push_regions!(state::OperationState, region::Region) = MLIR.API.mlirOperationStateAddOwnedRegions(Ref(state), 1, Ref(region))
push_regions!(state::OperationState, regions::Vector{Region}) = MLIR.API.mlirOperationStateAddOwnedRegions(Ref(state), length(regions), regions)

push_successors!(state::OperationState, successor::Block) = MLIR.API.mlirOperationStateAddSuccessors(Ref(state), 1, Ref(successor))
push_successors!(state::OperationState, successors::Vector{Block}) = MLIR.API.mlirOperationStateAddSuccessors(Ref(state), length(successors), successors)

push_attributes!(state::OperationState, attr::NamedAttribute) = MLIR.API.mlirOperationStateAddAttributes(Ref(state), 1, Ref(attr))
push_attributes!(state::OperationState, attrs::Vector{NamedAttribute}) = MLIR.API.mlirOperationStateAddAttributes(Ref(state), length(attrs), attrs)

function Base.display(state::OperationState)
    println("  __________________________________\n")
    println("             OperationState\n")
    for f in fieldnames(OperationState)
        println(" $f : $(getfield(state, f))")
    end
    println("  __________________________________\n")
end

# Creating operations from state
create_operation(state::OperationState) = Operation(MLIR.API.mlirOperationCreate(Ref(state)))

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

#####
##### Utilities
#####

function add_entry_block!(state::OperationState, arg_types::Vector{Type})
    r = create_region()
    b = create_block(arg_types)
    push!(r, b)
    push_regions!(state, r)
    return (b, r)
end
