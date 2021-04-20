#####
##### MlirValue alias and APIs
#####

const Value = MLIR.API.MlirValue

get_type(v::Value) = MLIR.API.mlirValueGetType(v)
get_owner_block(v::Value) = MLIR.API.mlirBlockArgumentGetOwner(v)
get_owner_block_arg_index(v::Value) = MLIR.API.mlirBlockArgumentGetArgNumber(v)
get_owner_op(v::Value) = MLIR.API.mlirOpResultGetOwner(v)
get_owner_op_result_index(v::Value) = MLIR.API.mlirOpResultGetResultNumber(v)
is_null(v::Value) = MLIR.API.mlirValueIsNull(v)
Base.:(==)(v1::Value, v2::Value) = MLIR.API.mlirValueEqual(v1, v2)
is_block_args(v::Value) = MLIR.API.mlirValueIsABlockArgument(v)
is_op_result(v::Value) = MLIR.API.mlirValueIsAOpResult(v)
dump(v::Value) = MLIR.API.mlirValueDump(v)
set_type!(v::Value, t::Type) = MLIR.API.mlirBlockArgumentSetType(v, t)

@doc(
"""
const Value = MLIR.API.MlirValue
""", Value)
