export Context, context, context!, activate!, deactivate!, dispose, enable_multithreading!

struct Context
    context::API.MlirContext

    function Context(context)
        @assert !API.mlirContextIsNull(context) "cannot create Context with null MlirContext"
        new(context)
    end
end

function Context()
    context = API.mlirContextCreate()
    context = Context(context)
    activate!(context)
    context
end

function Context(f::Core.Function)
    ctx = Context()
    try
        f(ctx)
    finally
        dispose!(ctx)
    end
end

Base.convert(::Core.Type{API.MlirContext}, c::Context) = c.context

# Global state

# to simplify the API, we maintain a stack of contexts in task local storage
# and pass them implicitly to MLIR API's that require them.
function activate!(ctx::Context)
    stack = get!(task_local_storage(), :MLIRContext) do
        Context[]
    end
    push!(stack, ctx)
    return
end

function deactivate!(ctx::Context)
    context() == ctx || error("Deactivating wrong context")
    pop!(task_local_storage(:MLIRContext))
end

function dispose!(ctx::Context)
    deactivate!(ctx)
    API.mlirContextDestroy(ctx.context)
end

_has_context() = haskey(task_local_storage(), :MLIRContext) &&
                 !isempty(task_local_storage(:MLIRContext))

function context(; throw_error::Core.Bool=true)
    if !_has_context()
        throw_error && error("No MLIR context is active")
        return nothing
    end
    last(task_local_storage(:MLIRContext))
end

function context!(f, ctx::Context)
    activate!(ctx)
    try
        f()
    finally
        deactivate!(ctx)
    end
end

function enable_multithreading!(enable::Bool=true)
    API.mlirContextEnableMultithreading(context(), enable)
    context()
end

Base.:(==)(a::Context, b::Context) = API.mlirContextEqual(a.context, b.context)
