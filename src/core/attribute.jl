#####
##### MlirAttribute alias and APIs
#####

const Attribute = MLIR.API.MlirAttribute

get_context(attr::Attribute) = MLIR.API.mlirAttributeGetContext(attr)
get_type(attr::Attribute) = MLIR.API.mlirAttributeGetType(attr)
get_unit_attribute(ctx::Context) = MLIR.API.mlirUnitAttrGet(ctx)
get_type_attribute(type::Type) = MLIR.API.mlirTypeAttrGet(type)
get_string_attribute(ctx::Context, str::String) = MLIR.API.mlirStringAttrGet(ctx, StringRef(str))
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

#####
##### MlirNamedAttribute alias and APIs
#####

const NamedAttribute = MLIR.API.MlirNamedAttribute

create_named_attribute(name, attr::Attribute) = MLIR.API.mlirNamedAttributeGet(name, attr)

# Constructor.
function NamedAttribute(ctx::Context, name::String, attr::Attribute)
    id = MLIR.IR.Identifier(ctx, name) # owned by MLIR
    create_named_attribute(id, attr)
end

@doc(
"""
const NamedAttribute = MLIR.API.MlirNamedAttribute
""", NamedAttribute)
