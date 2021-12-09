# Builders for regions.
push!(r::Region, b::Block) = API.mlirRegionAppendOwnedBlock(r, b)
push_block!(r::Region, b::Block) = push!(r, b)

# Builders for blocks.
push!(b::Block, op::Operation) = API.mlirBlockAppendOwnedOperation(b, unwrap(op))
push_operation!(b::Block, op::Operation) = push!(b, op)

# Builders for state.
push!(state::OperationState, n::Int, results::Vector{Type}) = API.mlirOperationStateAddResults(Ref(state), n, results)
push!(state::OperationState, n::Int, operands::Vector{Value}) = API.mlirOperationStateAddOperands(Ref(state), n, operands)
push!(state::OperationState, n::Int, regions::Vector{Region}) = API.mlirOperationStateAddOwnedRegions(Ref(state), n, regions)
push!(state::OperationState, n::Int, successors::Vector{Block}) = API.mlirOperationStateAddSuccessors(Ref(state), n, successors)
push!(state::OperationState, n::Int, attrs::Vector{NamedAttribute}) = API.mlirOperationStateAddAttributes(Ref(state), n, attrs)
push!(state::OperationState, args::Vector) = push!(state, length(args), args)
push!(state::OperationState, arg::T) where T = push!(state, T[arg])
