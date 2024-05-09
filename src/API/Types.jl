module Types

import ..API

const IS_LIBC_MUSL = occursin("musl", Base.MACHINE)

if Sys.islinux() && Sys.ARCH === :aarch64 && !IS_LIBC_MUSL
    const __off_t = Clong
    const off_t = __off_t
elseif Sys.islinux() && Sys.ARCH === :aarch64 && IS_LIBC_MUSL
    const off_t = Clong
elseif Sys.islinux() && startswith(string(Sys.ARCH), "arm") && !IS_LIBC_MUSL
    const __off_t = Clong
    const off_t = __off_t
elseif Sys.islinux() && startswith(string(Sys.ARCH), "arm") && IS_LIBC_MUSL
    const off_t = Clonglong
elseif Sys.islinux() && Sys.ARCH === :i686 && !IS_LIBC_MUSL
    const __off_t = Clong
    const off_t = __off_t
elseif Sys.islinux() && Sys.ARCH === :i686 && IS_LIBC_MUSL
    const off_t = Clonglong
elseif Sys.iswindows() && Sys.ARCH === :i686
    const off32_t = Clong
    const off_t = off32_t
elseif Sys.islinux() && Sys.ARCH === :powerpc64le
    const __off_t = Clong
    const off_t = __off_t
elseif Sys.isapple()
    const __darwin_off_t = Int64
    const off_t = __darwin_off_t
elseif Sys.islinux() && Sys.ARCH === :x86_64 && !IS_LIBC_MUSL
    const __off_t = Clong
    const off_t = __off_t
elseif Sys.islinux() && Sys.ARCH === :x86_64 && IS_LIBC_MUSL
    const off_t = Clong
elseif Sys.isbsd() && !Sys.isapple()
    const __off_t = Int64
    const off_t = __off_t
elseif Sys.iswindows() && Sys.ARCH === :x86_64
    const off32_t = Clong
    const off_t = off32_t
end

const intptr_t = Clong
export intptr_t

export
    MlirAffineExpr,
    MlirAffineMap,
    MlirAsmState,
    MlirAttribute,
    MlirBlock,
    MlirBytecodeWriterConfig,
    MlirContext,
    MlirDiagnostic,
    MlirDialect,
    MlirDialectHandle,
    MlirDialectRegistry,
    MlirExecutionEngine,
    MlirExternalPass,
    MlirExternalPassCallbacks,
    MlirIdentifier,
    MlirIntegerSet,
    MlirLlvmThreadPool,
    MlirLocation,
    MlirLogicalResult,
    MlirModule,
    MlirNamedAttribute,
    MlirOperation,
    MlirOperationState,
    MlirOpOperand,
    MlirOpPassManager,
    MlirOpPrintingFlags,
    MlirPass,
    MlirPassManager,
    MlirRegion,
    MlirStringRef,
    MlirSymbolTable,
    MlirType,
    MlirTypeID,
    MlirTypeIDAllocator,
    MlirValue

struct MlirTypeID
    ptr::Ptr{Cvoid}
end

struct MlirTypeIDAllocator
    ptr::Ptr{Cvoid}
end

"""
    MlirLlvmThreadPool

Re-export llvm::ThreadPool so as to avoid including the LLVM C API directly.
"""
struct MlirLlvmThreadPool
    ptr::Ptr{Cvoid}
end

"""
    MlirStringRef

A pointer to a sized fragment of a string, not necessarily null-terminated. Does not own the underlying string. This is equivalent to llvm::StringRef.

| Field  | Note                          |
| :----- | :---------------------------- |
| data   | Pointer to the first symbol.  |
| length | Length of the fragment.       |
"""
struct MlirStringRef
    data::Cstring
    length::Csize_t
end

# MlirStringRef is a non-owning reference to a string,
# we thus need to ensure that the Julia string remains alive
# over the use. For that we use the cconvert/unsafe_convert mechanism
# for foreign-calls. The returned value of the cconvert is rooted across
# foreign-call.
Base.cconvert(::Core.Type{MlirStringRef}, s::Union{Symbol,String}) = s
Base.cconvert(::Core.Type{MlirStringRef}, s::AbstractString) = Base.cconvert(MlirStringRef, String(s)::String)

# Directly create `MlirStringRef` instead of adding an extra ccall.
function Base.unsafe_convert(::Core.Type{MlirStringRef}, s::Union{Symbol,String,AbstractVector{UInt8}})
    p = Base.unsafe_convert(Ptr{Cchar}, s)
    return MlirStringRef(p, sizeof(s))
end

Base.String(str::MlirStringRef) = Base.unsafe_string(pointer(str.data), str.length)

"""
    MlirLogicalResult

A logical result value, essentially a boolean with named states. LLVM convention for using boolean values to designate success or failure of an operation is a moving target, so MLIR opted for an explicit class. Instances of [`MlirLogicalResult`](@ref) must only be inspected using the associated functions.
"""
struct MlirLogicalResult
    value::Int8
end

struct MlirContext
    ptr::Ptr{Cvoid}
end

struct MlirDialect
    ptr::Ptr{Cvoid}
end

struct MlirDialectHandle
    ptr::Ptr{Cvoid}
end

struct MlirDialectRegistry
    ptr::Ptr{Cvoid}
end

struct MlirOperation
    ptr::Ptr{Cvoid}
end

# introduced in LLVM 16
struct MlirOpOperand
    ptr::Ptr{Cvoid}
end

struct MlirOpPrintingFlags
    ptr::Ptr{Cvoid}
end

struct MlirBlock
    ptr::Ptr{Cvoid}
end

struct MlirRegion
    ptr::Ptr{Cvoid}
end

struct MlirSymbolTable
    ptr::Ptr{Cvoid}
end

struct MlirAttribute
    ptr::Ptr{Cvoid}
end

struct MlirIdentifier
    ptr::Ptr{Cvoid}
end

Base.String(str::MlirIdentifier) = String(API.mlirIdentifierStr(str))

struct MlirLocation
    ptr::Ptr{Cvoid}
end

struct MlirModule
    ptr::Ptr{Cvoid}
end

struct MlirType
    ptr::Ptr{Cvoid}
end

struct MlirValue
    ptr::Ptr{Cvoid}
end

"""
    MlirNamedAttribute

Named MLIR attribute.

A named attribute is essentially a (name, attribute) pair where the name is a string.
"""
struct MlirNamedAttribute
    name::MlirIdentifier
    attribute::MlirAttribute
end

"""
    MlirOperationState

An auxiliary class for constructing operations.

This class contains all the information necessary to construct the operation. It owns the MlirRegions it has pointers to and does not own anything else. By default, the state can be constructed from a name and location, the latter being also used to access the context, and has no other components. These components can be added progressively until the operation is constructed. Users are not expected to rely on the internals of this class and should use mlirOperationState* functions instead.
"""
struct MlirOperationState
    name::MlirStringRef
    location::MlirLocation
    nResults::intptr_t
    results::Ptr{MlirType}
    nOperands::intptr_t
    operands::Ptr{MlirValue}
    nRegions::intptr_t
    regions::Ptr{MlirRegion}
    nSuccessors::intptr_t
    successors::Ptr{MlirBlock}
    nAttributes::intptr_t
    attributes::Ptr{MlirNamedAttribute}
    enableResultTypeInference::Bool
end

struct MlirAffineExpr
    ptr::Ptr{Cvoid}
end

struct MlirAffineMap
    ptr::Ptr{Cvoid}
end

struct MlirPass
    ptr::Ptr{Cvoid}
end

# introduced in LLVM 15
struct MlirExternalPass
    ptr::Ptr{Cvoid}
end

struct MlirPassManager
    ptr::Ptr{Cvoid}
end

struct MlirOpPassManager
    ptr::Ptr{Cvoid}
end

"""
    MlirExternalPassCallbacks

Structure of external [`MlirPass`](@ref) callbacks. All callbacks are required to be set unless otherwise specified.

| Field      | Note                                                                                                                                                                                              |
| :--------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| construct  | This callback is called from the pass is created. This is analogous to a C++ pass constructor.                                                                                                    |
| destruct   | This callback is called when the pass is destroyed This is analogous to a C++ pass destructor.                                                                                                    |
| initialize | This callback is optional. The callback is called before the pass is run, allowing a chance to initialize any complex state necessary for running the pass. See Pass::initialize(MLIRContext *).  |
| clone      | This callback is called when the pass is cloned. See Pass::clonePass().                                                                                                                           |
| run        | This callback is called when the pass is run. See Pass::runOnOperation().                                                                                                                         |
"""
struct MlirExternalPassCallbacks
    construct::Ptr{Cvoid}
    destruct::Ptr{Cvoid}
    initialize::Ptr{Cvoid}
    clone::Ptr{Cvoid}
    run::Ptr{Cvoid}
end

"""
    MlirDiagnostic

An opaque reference to a diagnostic, always owned by the diagnostics engine (context). Must not be stored outside of the diagnostic handler.
"""
struct MlirDiagnostic
    ptr::Ptr{Cvoid}
end

struct MlirExecutionEngine
    ptr::Ptr{Cvoid}
end

struct MlirIntegerSet
    ptr::Ptr{Cvoid}
end

struct MlirAsmState
    ptr::Ptr{Cvoid}
end

struct MlirBytecodeWriterConfig
    ptr::Ptr{Cvoid}
end

end # module Types
