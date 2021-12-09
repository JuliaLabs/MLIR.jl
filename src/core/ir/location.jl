# ------------ Location alias and APIs ------------ #

const Location = API.MlirLocation

get_context(l::Location) = API.mlirLocationGetContext(l)
is_null(l::Location) = API.mlirLocationIsNull(l)
Base.:(==)(l1::Location, l2::Location) = API.mlirLocationEquation(l1, l2)

# Constructor.
Location(ctx::Context) = create_unknown_location(ctx)

@doc(
"""
const Location = API.MlirLocation
""", Location)
