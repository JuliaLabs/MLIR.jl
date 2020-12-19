# ------------ Location alias and APIs ------------ #

const Location = MLIR.API.MlirLocation

get_context(l::Location) = MLIR.API.mlirLocationGetContext(l)
is_null(l::Location) = MLIR.API.mlirLocationIsNull(l)
Base.:(==)(l1::Location, l2::Location) = MLIR.API.mlirLocationEquation(l1, l2)

Location(ctx::Context) = create_unknown_location(ctx)

@doc(
"""
const Location = MLIR.API.MlirLocation
""", Location)
