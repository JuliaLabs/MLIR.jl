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
push!(state::OperationState, n::Int, results::Vector{Type}) = MLIR.API.mlirOperationStateAddResults(Ref(state), n, results)
push_results!(state::OperationState, results::Vector{Type}) = push!(state, length(results), results)

push!(state::OperationState, n::Int, operands::Vector{Value}) = MLIR.API.mlirOperationStateAddOperands(Ref(state), n, operands)
push_operands!(state::OperationState, n::Int, operands::Vector{Value}) = push!(state, length(operands), operands)

push!(state::OperationState, n::Int, regions::Vector{Region}) = MLIR.API.mlirOperationStateAddOwnedRegions(Ref(state), n, regions)
push_regions!(state::OperationState, n::Int, regions::Vector{Region}) = push!(state, length(regions), regions)

push!(state::OperationState, n::Int, successors::Vector{Block}) = MLIR.API.mlirOperationStateAddSuccessors(Ref(state), n, successors)
push_successors!(state::OperationState, n::Int, successors::Vector{Block}) = push!(state, length(successors), successors)

push!(state::OperationState, n::Int, attrs::Vector{NamedAttribute}) = MLIR.API.mlirOperationStateAddAttributes(Ref(state), n, attrs)
push_attributes!(state::OperationState, n::Int, attrs::Vector{NamedAttribute}) = push!(state, length(attrs), attrs)

push!(state::OperationState, args::Vector) = push!(state, length(args), args)
push!(state::OperationState, arg::T) where T = push!(state, T[arg])

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
