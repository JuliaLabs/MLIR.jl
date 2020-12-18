const Context = MLIR.API.MlirContext

context_create() = MLIR.API.mlirContextCreate()
num_registered_dialects(ctx::Context) = MLIR.API.mlirContextGetNumRegisteredDialects(ctx)
is_null(ctx::Context) = MLIR.API.mlirContextIsNull(ctx)
context_destroy(ctx::Context) = MLIR.API.mlirContextDestroy(ctx)
get_allow_unregistered_dialects(ctx::Context) = MLIR.API.mlirContextGetAllowUnregisteredDialects(ctx)

struct Dialect
    ref::MLIR.API.MlirDialect
end

struct Type
    ref::MLIR.API.MlirType
end

struct Location
    ref::MLIR.API.MlirLocation
end

struct Attribute
    ref::MLIR.API.MlirAttribute
end

struct OperationState
    ref::MLIR.API.MlirOperationState
end

struct Operation
    ref::MLIR.API.MlirOperation
end

struct Value
    ref::MLIR.API.MlirValue
end

struct Block
    ref::MLIR.API.MlirBlock
end

struct Module
    ref::MLIR.API.MlirModule
end

struct Region
    ref::MLIR.API.MlirRegion
end
