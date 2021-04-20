#####
##### MlirDialect alias and APIs
#####

const Dialect = MLIR.API.MlirDialect

get_context(d::Dialect) = MLIR.API.mlirDialectGetContext(d)
is_null(d::Dialect) = MLIR.API.mlirDialectIsNull(d)
Base.:(==)(d1::Dialect, d2::Dialect) = MLIR.API.mlirDialectEqual(d1, d2)
get_namespace(d::Dialect) = MLIR.API.mlirDialectGetNamespace(d)

@doc(
"""
const Dialect = MLIR.API.MlirDialect
""", Dialect)

