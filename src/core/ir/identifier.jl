# ------------ Identifier alias and APIs ------------ #

const Identifier = API.MlirIdentifier

get_name(op::Operation) = API.mlirOperationGetName(uwnrap(op))
create_identifier(ctx::Context, str::StringRef) = API.mlirIdentifierGet(ctx, str)
Base.:(==)(id1::Identifier, id2::Identifier) = API.mlirIdentifierEqual(id1, id2)
get_str(id::Identifier) = API.mlirIdentifierStr(id)

# Constructor.
Identifier(ctx::Context, str::String) = create_identifier(ctx, StringRef(str))

@doc(
"""
const Identifier = API.MlirIdentifier
""", Identifier)
