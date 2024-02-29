"""
    LogicalResult

A logical result value, essentially a boolean with named states.
LLVM convention for using boolean values to designate success or failure of an operation is a moving target, so MLIR opted for an explicit class.
Instances of [`LogicalResult`](@ref) must only be inspected using the associated functions.
"""
struct LogicalResult
    result::API.MlirLogicalResult
end

Base.convert(::Core.Type{API.MlirLogicalResult}, result::LogicalResult) = result.result

"""
    success()

Creates a logical result representing a success.
"""
success() = LogicalResult(API.mlirLogicalResultSuccess())

"""
    failure()

Creates a logical result representing a failure.
"""
failure() = LogicalResult(API.mlirLogicalResultFailure())

"""
    issuccess(res)

Checks if the given logical result represents a success.
"""
issuccess(result::LogicalResult) = API.mlirLogicalResultIsSuccess(result)

"""
    isfailure(res)

Checks if the given logical result represents a failure.
"""
isfailure(result::LogicalResult) = API.mlirLogicalResultIsFailure(result)
