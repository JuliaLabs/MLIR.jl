# ------------ Module alias and APIs ------------ #

const Module = API.MlirModule

create_empty_module(l::Location) = API.mlirModuleCreateEmpty(l)
create_parse(ctx::Context, m::Module) = API.mlirModuleCreateParse(ctx, m)
get_context(m::Module) = API.mlirModuleGetContext(m)
get_body(m::Module) = API.mlirModuleGetBody(m)
is_null(m::Module) = API.mlirModuleIsNull(m)
destroy!(m::Module) = API.mlirModuleDestroy(m)
get_operation(m::Module) = Operation(API.mlirModuleGetOperation(m))

# Constructor.
Module(l::Location) = create_empty_module(l)

@doc(
"""
const Module = API.MlirModule
""", Module)
