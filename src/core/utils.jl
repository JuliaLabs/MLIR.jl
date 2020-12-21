# Should not be exposed to the user.
const StringRef = MLIR.API.MlirStringRef
create_string_ref(str::String) = MLIR.API.mlirStringRefCreateFromCString(str)
StringRef(str::String) = create_string_ref(str)

# Unwraps API structs to ptr field.
unwrap(o) = getfield(o, :ptr)
unwrap(ptr::P) where P <: Ptr = ptr
