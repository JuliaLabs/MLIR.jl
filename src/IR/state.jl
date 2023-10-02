# Global state

# to simplify the API, we maintain a stack of contexts in task local storage
# and pass them implicitly to MLIR API's that require them.

export context, activate, deactivate, context!

using ..IR

_has_context() = haskey(task_local_storage(), :MLIRContext) &&
                 !isempty(task_local_storage(:MLIRContext))

function context(; throw_error::Core.Bool=true)
    if !_has_context()
        throw_error && error("No MLIR context is active")
        return nothing
    end
    last(task_local_storage(:MLIRContext))
end

function activate(ctx::Context)
    stack = get!(task_local_storage(), :MLIRContext) do
        Context[]
    end
    push!(stack, ctx)
    return
end

function deactivate(ctx::Context)
    context() == ctx || error("Deactivating wrong context")
    pop!(task_local_storage(:MLIRContext))
end

function context!(f, ctx::Context)
    activate(ctx)
    try
        f()
    finally
        deactivate(ctx)
    end
end


