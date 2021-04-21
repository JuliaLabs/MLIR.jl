#####
##### MlirType alias and APIs
#####

const Type = MLIR.API.MlirType

get_context(t::Type) = MLIR.API.mlirTypeGetContext(t)
is_null(t::Type) = MLIR.API.mlirTypeIsNull(t)
Base.:(==)(t1::Type, t2::Type) = MLIR.API.mlirTypeEqual(t1, t2)
Base.display(t::Type) = MLIR.API.mlirTypeDump(t)
parse_type(ctx::Context, t::String) = MLIR.API.mlirTypeParseGet(ctx, StringRef(t))

# Constructor.
Type(ctx::Context, t::String) = parse_type(ctx, t)

@doc(
"""
const Type = MLIR.API.MlirType
""", Type)

#####
##### Builtin MLIR types
#####

is_int(t::Type) = MLIR.API.mlirTypeIsAInteger(t)
get_int_type(ctx::Context, bitwidth::Int64) = MLIR.API.mlirIntegerTypeGet(ctx, bitwidth)
get_sint_type(ctx::Context, bitwidth::Int64) = MLIR.API.mlirIntegerSignedGet(ctx, bitwidth)
get_uint_type(ctx::Context, bitwidth::Int64) = MLIR.API.mlirIntegerUnsignedGet(ctx, bitwidth)
get_bitwidth(t::Type) = MLIR.API.mlirIntegerTypeGetWidth(t)
get_f32_type(ctx::Context) = MLIR.API.mlirF32TypeGet(ctx)
get_f64_type(ctx::Context) = MLIR.API.mlirF64TypeGet(ctx)
