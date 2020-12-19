# ------------ Context alias and APIs ------------ #

const Context = MLIR.API.MlirContext

create_context() = MLIR.API.mlirContextCreate()
num_loaded_dialects(ctx::Context) = Base.convert(Int64, MLIR.API.mlirContextGetNumLoadedDialects(ctx))
num_registered_dialects(ctx::Context) = Base.convert(Int64, MLIR.API.mlirContextGetNumRegisteredDialects(ctx))
is_null(ctx::Context) = MLIR.API.mlirContextIsNull(ctx)
destroy!(ctx::Context) = MLIR.API.mlirContextDestroy(ctx)
get_allow_unregistered_dialects(ctx::Context) = MLIR.API.mlirContextGetAllowUnregisteredDialects(ctx)
create_unknown_location(ctx::Context) = MLIR.API.mlirLocationUnknownGet(ctx)

register_all_dialects!(ctx::Context) = MLIR.API.MLIR.API.mlirRegisterAllDialects(ctx)
register_standard_dialect!(ctx::Context) = MLIR.API.MLIR.API.mlirContextRegisterStandardDialect(ctx)
load_standard_dialect!(ctx::Context) = MLIR.API.MLIR.API.mlirContextLoadStandardDialect(ctx)

Context() = create_context()

@doc(
"""
const Context = MLIR.API.MlirContext
""", Context)
