const StringRef = MLIR.API.MlirStringRef

create_string_ref(cstr::Cstring) = MLIR.API.mlirStringRefCreateFromCString(cstr)
StringRef(str::String) = create_string_ref(Base.unsafe_convert(Cstring, str))
