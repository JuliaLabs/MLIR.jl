# ------------ Attribute alias and APIs ------------ #

const Attribute = MLIR.API.MlirAttribute

get_context(attr::Attribute) = MLIR.API.mlirAttributeGetContext(attr)
get_type(attr::Attribute) = MLIR.API.mlirAttributeGetType(attr)
is_null(attr::Attribute) = MLIR.API.mlirAttributeIsNull(attr)
Base.:(==)(attr1::Attribute, attr2::Attribute) = MLIR.API.mlirAttributeEqual(attr1, attr2)
function parse_attribute(ctx::Context, attr::String)
    ref = StringRef(attr)
    MLIR.API.mlirAttributeParseGet(ctx, ref)
end
dump(attr::Attribute) = MLIR.API.mlirAttributeDump(attr)

# Constructor.
Attribute(ctx::Context, attr::String) = parse_attribute(ctx, attr)

@doc(
"""
const Attribute = MLIR.API.MlirAttribute
""", Attribute)

# ------------ Named attribute alias and APIs ------------ #

const NamedAttribute = MLIR.API.MlirNamedAttribute

create_named_attribute(name::StringRef, attr::Attribute) = MLIR.API.mlirNamedAttributeGet(name, attr)

# Constructor.
NamedAttribute(name::String, attr::Attribute) = create_named_attribute(StringRef(name), attr)

@doc(
"""
const NamedAttribute = MLIR.API.MlirNamedAttribute
""", NamedAttribute)
