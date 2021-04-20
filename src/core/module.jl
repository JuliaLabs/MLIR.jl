#####
##### MlirModule alias and APIs
#####

const Module = MLIR.API.MlirModule

create_empty_module(l::Location) = MLIR.API.mlirModuleCreateEmpty(l)
create_parse(ctx::Context, m::Module) = MLIR.API.mlirModuleCreateParse(ctx, m)
get_context(m::Module) = MLIR.API.mlirModuleGetContext(m)
get_body(m::Module) = MLIR.API.mlirModuleGetBody(m)
is_null(m::Module) = MLIR.API.mlirModuleIsNull(m)
destroy!(m::Module) = MLIR.API.mlirModuleDestroy(m)
get_operation(m::Module) = Operation(MLIR.API.mlirModuleGetOperation(m))

# Constructor.
Module(l::Location) = create_empty_module(l)

@doc(
"""
const Module = MLIR.API.MlirModule
""", Module)
