# ------------ Attribute alias and APIs ------------ #

const Attribute = API.MlirAttribute

get_context(attr::Attribute) = API.mlirAttributeGetContext(attr)
get_type(attr::Attribute) = API.mlirAttributeGetType(attr)
is_null(attr::Attribute) = API.mlirAttributeIsNull(attr)
Base.:(==)(attr1::Attribute, attr2::Attribute) = API.mlirAttributeEqual(attr1, attr2)
function parse_attribute(ctx::Context, attr::String)
    ref = StringRef(attr)
    API.mlirAttributeParseGet(ctx, ref)
end
dump(attr::Attribute) = API.mlirAttributeDump(attr)

# Constructor.
Attribute(ctx::Context, attr::String) = parse_attribute(ctx, attr)

@doc(
"""
const Attribute = API.MlirAttribute
""", Attribute)

# ------------ Named attribute alias and APIs ------------ #

const NamedAttribute = API.MlirNamedAttribute

create_named_attribute(name, attr::Attribute) = API.mlirNamedAttributeGet(name, attr)

# Constructor.
function NamedAttribute(ctx::Context, name::String, attr::Attribute)
    id = MLIR.IR.Identifier(ctx, name) # owned by MLIR
    create_named_attribute(id, attr)
end

@doc(
"""
const NamedAttribute = API.MlirNamedAttribute
""", NamedAttribute)
