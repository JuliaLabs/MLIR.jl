# ------------ Identifier alias and APIs ------------ #

const Identifier = MLIR.API.MlirIdentifier

get_name(op::Operation) = MLIR.API.mlirOperationGetName(uwnrap(op))
create_identifier(ctx::Context, str::StringRef) = MLIR.API.mlirIdentifierGet(ctx, str)
Base.:(==)(id1::Identifier, id2::Identifier) = MLIR.API.mlirIdentifierEqual(id1, id2)
get_str(id::Identifier) = MLIR.API.mlirIdentifierStr(id)

# Constructor.
Identifier(ctx::Context, str::String) = create_identifier(ctx, StringRef(str))

@doc(
"""
const Identifier = MLIR.API.MlirIdentifier
""", Identifier)
