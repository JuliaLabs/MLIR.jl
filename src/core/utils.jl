const StringRef = MLIR.API.MlirStringRef

create_string_ref(cstr::Cstring, len::Csize_t) = mlirStringRefCreate(cstr, len)
create_string_ref(cstr::Cstring) = mlirStringRefCreateFromCString(cstr)

const LogicalResult = MLIR.API.MlirLogicalResult
