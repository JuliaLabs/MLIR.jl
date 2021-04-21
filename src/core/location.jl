#####
##### MlirLocation alias and APIs
#####

const Location = MLIR.API.MlirLocation

get_context(l::Location) = MLIR.API.mlirLocationGetContext(l)
is_null(l::Location) = MLIR.API.mlirLocationIsNull(l)
Base.:(==)(l1::Location, l2::Location) = MLIR.API.mlirLocationEquation(l1, l2)

# Constructors.
Location(ctx::Context) = create_unknown_location(ctx)
Location(ctx::Context, filename::String, line::UInt32, col::UInt32) = MLIR.API.mlirLocationFileLineColGet(ctx, StringRef(filename), line, col)
Location(callee::Location, caller::Location) = MLIR.API.mlirLocationCallSiteGet(callee, caller)

@doc(
"""
const Location = MLIR.API.MlirLocation
""", Location)
