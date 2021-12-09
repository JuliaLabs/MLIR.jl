# ------------ Dialect alias and APIs ------------ #

const Dialect = API.MlirDialect

get_context(d::Dialect) = API.mlirDialectGetContext(d)
is_null(d::Dialect) = API.mlirDialectIsNull(d)
Base.:(==)(d1::Dialect, d2::Dialect) = API.mlirDialectEqual(d1, d2)
get_namespace(d::Dialect) = API.mlirDialectGetNamespace(d)

@doc(
"""
const Dialect = API.MlirDialect
""", Dialect)

