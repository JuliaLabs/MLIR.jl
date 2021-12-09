# ------------ Type alias and APIs ------------ #

const Type = API.MlirType

get_context(t::Type) = API.mlirTypeGetContext(t)
is_null(t::Type) = API.mlirTypeIsNull(t)
Base.:(==)(t1::Type, t2::Type) = API.mlirTypeEqual(t1, t2)
dump(t::Type) = API.mlirTypeDump(t)
parse_type(ctx::Context, t::String) = API.mlirTypeParseGet(ctx, StringRef(t))

# Constructor.
Type(ctx::Context, t::String) = parse_type(ctx, t)

@doc(
"""
const Type = API.MlirType
""", Type)

# ------------ Builtins ------------ #

# Integers.
is_int(t::Type) = API.mlirTypeIsAInteger(t)
get_int_type(ctx::Context, bitwidth::Int64) = API.mlirIntegerTypeGet(ctx, bitwidth)
get_sint_type(ctx::Context, bitwidth::Int64) = API.mlirIntegerSignedGet(ctx, bitwidth)
get_uint_type(ctx::Context, bitwidth::Int64) = API.mlirIntegerUnsignedGet(ctx, bitwidth)
get_bitwidth(t::Type) = API.mlirIntegerTypeGetWidth(t)

# Floating point.
get_f32_type(ctx::Context) = API.mlirF32TypeGet(ctx)
get_f64_type(ctx::Context) = API.mlirF64TypeGet(ctx)
