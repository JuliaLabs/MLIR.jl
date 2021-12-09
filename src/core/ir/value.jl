# ------------ Value alias and APIs ------------ #

const Value = API.MlirValue

get_type(v::Value) = API.mlirValueGetType(v)
get_owner_block(v::Value) = API.mlirBlockArgumentGetOwner(v)
get_owner_block_arg_index(v::Value) = API.mlirBlockArgumentGetArgNumber(v)
get_owner_op(v::Value) = API.mlirOpResultGetOwner(v)
get_owner_op_result_index(v::Value) = API.mlirOpResultGetResultNumber(v)
is_null(v::Value) = API.mlirValueIsNull(v)
Base.:(==)(v1::Value, v2::Value) = API.mlirValueEqual(v1, v2)
is_block_args(v::Value) = API.mlirValueIsABlockArgument(v)
is_op_result(v::Value) = API.mlirValueIsAOpResult(v)
dump(v::Value) = API.mlirValueDump(v)
set_type!(v::Value, t::Type) = API.mlirBlockArgumentSetType(v, t)

@doc(
"""
const Value = API.MlirValue
""", Value)
