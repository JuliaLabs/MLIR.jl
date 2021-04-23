#####
##### MlirContext alias and APIs
#####

const Context = MLIR.API.MlirContext

create_context() = MLIR.API.mlirContextCreate()
num_loaded_dialects(ctx::Context) = Base.convert(Int64, MLIR.API.mlirContextGetNumLoadedDialects(ctx))
num_registered_dialects(ctx::Context) = Base.convert(Int64, MLIR.API.mlirContextGetNumRegisteredDialects(ctx))
is_null(ctx::Context) = MLIR.API.mlirContextIsNull(ctx)
destroy!(ctx::Context) = MLIR.API.mlirContextDestroy(ctx)
set_allow_unregistered_dialects!(ctx::Context, b::Bool) = MLIR.API.mlirContextSetAllowUnregisteredDialects(ctx, b)
get_allow_unregistered_dialects(ctx::Context) = MLIR.API.mlirContextGetAllowUnregisteredDialects(ctx)
create_unknown_location(ctx::Context) = MLIR.API.mlirLocationUnknownGet(ctx)
register_all_dialects!(ctx::Context) = MLIR.API.MLIR.API.mlirRegisterAllDialects(ctx)

# Constructor.
Context() = create_context()

function Context(func::Function)
    ctx = Context()
    try
        func(ctx)
    finally
        destroy!(ctx)
    end
end

@doc(
"""
const Context = MLIR.API.MlirContext
""", Context)
