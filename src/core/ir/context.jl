# ------------ Context alias and APIs ------------ #

const Context = API.MlirContext

create_context() = API.mlirContextCreate()
num_loaded_dialects(ctx::Context) = Base.convert(Int64, API.mlirContextGetNumLoadedDialects(ctx))
num_registered_dialects(ctx::Context) = Base.convert(Int64, API.mlirContextGetNumRegisteredDialects(ctx))
is_null(ctx::Context) = API.mlirContextIsNull(ctx)
destroy!(ctx::Context) = API.mlirContextDestroy(ctx)
get_allow_unregistered_dialects(ctx::Context) = API.mlirContextGetAllowUnregisteredDialects(ctx)
create_unknown_location(ctx::Context) = API.mlirLocationUnknownGet(ctx)
register_all_dialects!(ctx::Context) = API.mlirRegisterAllDialects(ctx)
register_standard_dialect!(ctx::Context) = API.mlirContextRegisterStandardDialect(ctx)
load_standard_dialect!(ctx::Context) = API.mlirContextLoadStandardDialect(ctx)

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
const Context = API.MlirContext
""", Context)
