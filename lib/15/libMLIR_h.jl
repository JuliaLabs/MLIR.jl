using CEnum

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

struct MlirDialectHandle
    ptr::Ptr{Cvoid}
end

struct MlirTypeID
    ptr::Ptr{Cvoid}
end

struct MlirTypeIDAllocator
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

"""
    mlirStringRefCreate(str, length)

Constructs a string reference from the pointer and length. The pointer need not reference to a null-terminated string.
"""
function mlirStringRefCreate(str, length)
    @ccall mlir_c.mlirStringRefCreate(str::Cstring, length::Csize_t)::MlirStringRef
end

"""
    mlirStringRefCreateFromCString(str)

Constructs a string reference from a null-terminated C string. Prefer [`mlirStringRefCreate`](@ref) if the length of the string is known.
"""
function mlirStringRefCreateFromCString(str)
    @ccall mlir_c.mlirStringRefCreateFromCString(str::Cstring)::MlirStringRef
end

"""
    mlirStringRefEqual(string, other)

Returns true if two string references are equal, false otherwise.
"""
function mlirStringRefEqual(string, other)
    @ccall mlir_c.mlirStringRefEqual(string::MlirStringRef, other::MlirStringRef)::Bool
end

# typedef void ( * MlirStringCallback ) ( MlirStringRef , void * )
"""
A callback for returning string references.

This function is called back by the functions that need to return a reference to the portion of the string with the following arguments: - an [`MlirStringRef`](@ref) representing the current portion of the string - a pointer to user data forwarded from the printing call.
"""
const MlirStringCallback = Ptr{Cvoid}

"""
    MlirLogicalResult

A logical result value, essentially a boolean with named states. LLVM convention for using boolean values to designate success or failure of an operation is a moving target, so MLIR opted for an explicit class. Instances of [`MlirLogicalResult`](@ref) must only be inspected using the associated functions.
"""
struct MlirLogicalResult
    value::Int8
end

"""
    mlirLogicalResultIsSuccess(res)

Checks if the given logical result represents a success.
"""
function mlirLogicalResultIsSuccess(res)
    @ccall mlir_c.mlirLogicalResultIsSuccess(res::MlirLogicalResult)::Bool
end

"""
    mlirLogicalResultIsFailure(res)

Checks if the given logical result represents a failure.
"""
function mlirLogicalResultIsFailure(res)
    @ccall mlir_c.mlirLogicalResultIsFailure(res::MlirLogicalResult)::Bool
end

"""
    mlirLogicalResultSuccess()

Creates a logical result representing a success.
"""
function mlirLogicalResultSuccess()
    @ccall mlir_c.mlirLogicalResultSuccess()::MlirLogicalResult
end

"""
    mlirLogicalResultFailure()

Creates a logical result representing a failure.
"""
function mlirLogicalResultFailure()
    @ccall mlir_c.mlirLogicalResultFailure()::MlirLogicalResult
end

"""
    mlirTypeIDCreate(ptr)

`ptr` must be 8 byte aligned and unique to a type valid for the duration of the returned type id's usage
"""
function mlirTypeIDCreate(ptr)
    @ccall mlir_c.mlirTypeIDCreate(ptr::Ptr{Cvoid})::MlirTypeID
end

"""
    mlirTypeIDIsNull(typeID)

Checks whether a type id is null.
"""
function mlirTypeIDIsNull(typeID)
    @ccall mlir_c.mlirTypeIDIsNull(typeID::MlirTypeID)::Bool
end

"""
    mlirTypeIDEqual(typeID1, typeID2)

Checks if two type ids are equal.
"""
function mlirTypeIDEqual(typeID1, typeID2)
    @ccall mlir_c.mlirTypeIDEqual(typeID1::MlirTypeID, typeID2::MlirTypeID)::Bool
end

"""
    mlirTypeIDHashValue(typeID)

Returns the hash value of the type id.
"""
function mlirTypeIDHashValue(typeID)
    @ccall mlir_c.mlirTypeIDHashValue(typeID::MlirTypeID)::Csize_t
end

"""
    mlirTypeIDAllocatorCreate()

Creates a type id allocator for dynamic type id creation
"""
function mlirTypeIDAllocatorCreate()
    @ccall mlir_c.mlirTypeIDAllocatorCreate()::MlirTypeIDAllocator
end

"""
    mlirTypeIDAllocatorDestroy(allocator)

Deallocates the allocator and all allocated type ids
"""
function mlirTypeIDAllocatorDestroy(allocator)
    @ccall mlir_c.mlirTypeIDAllocatorDestroy(allocator::MlirTypeIDAllocator)::Cvoid
end

"""
    mlirTypeIDAllocatorAllocateTypeID(allocator)

Allocates a type id that is valid for the lifetime of the allocator
"""
function mlirTypeIDAllocatorAllocateTypeID(allocator)
    @ccall mlir_c.mlirTypeIDAllocatorAllocateTypeID(allocator::MlirTypeIDAllocator)::MlirTypeID
end

struct MlirContext
    ptr::Ptr{Cvoid}
end

struct MlirDialect
    ptr::Ptr{Cvoid}
end

struct MlirDialectRegistry
    ptr::Ptr{Cvoid}
end

struct MlirOperation
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
    mlirContextCreate()

Creates an MLIR context and transfers its ownership to the caller.
"""
function mlirContextCreate()
    @ccall mlir_c.mlirContextCreate()::MlirContext
end

"""
    mlirContextEqual(ctx1, ctx2)

Checks if two contexts are equal.
"""
function mlirContextEqual(ctx1, ctx2)
    @ccall mlir_c.mlirContextEqual(ctx1::MlirContext, ctx2::MlirContext)::Bool
end

"""
    mlirContextIsNull(context)

Checks whether a context is null.
"""
function mlirContextIsNull(context)
    @ccall mlir_c.mlirContextIsNull(context::MlirContext)::Bool
end

"""
    mlirContextDestroy(context)

Takes an MLIR context owned by the caller and destroys it.
"""
function mlirContextDestroy(context)
    @ccall mlir_c.mlirContextDestroy(context::MlirContext)::Cvoid
end

"""
    mlirContextSetAllowUnregisteredDialects(context, allow)

Sets whether unregistered dialects are allowed in this context.
"""
function mlirContextSetAllowUnregisteredDialects(context, allow)
    @ccall mlir_c.mlirContextSetAllowUnregisteredDialects(context::MlirContext, allow::Bool)::Cvoid
end

"""
    mlirContextGetAllowUnregisteredDialects(context)

Returns whether the context allows unregistered dialects.
"""
function mlirContextGetAllowUnregisteredDialects(context)
    @ccall mlir_c.mlirContextGetAllowUnregisteredDialects(context::MlirContext)::Bool
end

"""
    mlirContextGetNumRegisteredDialects(context)

Returns the number of dialects registered with the given context. A registered dialect will be loaded if needed by the parser.
"""
function mlirContextGetNumRegisteredDialects(context)
    @ccall mlir_c.mlirContextGetNumRegisteredDialects(context::MlirContext)::intptr_t
end

"""
    mlirContextAppendDialectRegistry(ctx, registry)

Append the contents of the given dialect registry to the registry associated with the context.
"""
function mlirContextAppendDialectRegistry(ctx, registry)
    @ccall mlir_c.mlirContextAppendDialectRegistry(ctx::MlirContext, registry::MlirDialectRegistry)::Cvoid
end

"""
    mlirContextGetNumLoadedDialects(context)

Returns the number of dialects loaded by the context.
"""
function mlirContextGetNumLoadedDialects(context)
    @ccall mlir_c.mlirContextGetNumLoadedDialects(context::MlirContext)::intptr_t
end

"""
    mlirContextGetOrLoadDialect(context, name)

Gets the dialect instance owned by the given context using the dialect namespace to identify it, loads (i.e., constructs the instance of) the dialect if necessary. If the dialect is not registered with the context, returns null. Use mlirContextLoad<Name>Dialect to load an unregistered dialect.
"""
function mlirContextGetOrLoadDialect(context, name)
    @ccall mlir_c.mlirContextGetOrLoadDialect(context::MlirContext, name::MlirStringRef)::MlirDialect
end

"""
    mlirContextEnableMultithreading(context, enable)

Set threading mode (must be set to false to mlir-print-ir-after-all).
"""
function mlirContextEnableMultithreading(context, enable)
    @ccall mlir_c.mlirContextEnableMultithreading(context::MlirContext, enable::Bool)::Cvoid
end

"""
    mlirContextLoadAllAvailableDialects(context)

Eagerly loads all available dialects registered with a context, making them available for use for IR construction.
"""
function mlirContextLoadAllAvailableDialects(context)
    @ccall mlir_c.mlirContextLoadAllAvailableDialects(context::MlirContext)::Cvoid
end

"""
    mlirContextIsRegisteredOperation(context, name)

Returns whether the given fully-qualified operation (i.e. 'dialect.operation') is registered with the context. This will return true if the dialect is loaded and the operation is registered within the dialect.
"""
function mlirContextIsRegisteredOperation(context, name)
    @ccall mlir_c.mlirContextIsRegisteredOperation(context::MlirContext, name::MlirStringRef)::Bool
end

"""
    mlirDialectGetContext(dialect)

Returns the context that owns the dialect.
"""
function mlirDialectGetContext(dialect)
    @ccall mlir_c.mlirDialectGetContext(dialect::MlirDialect)::MlirContext
end

"""
    mlirDialectIsNull(dialect)

Checks if the dialect is null.
"""
function mlirDialectIsNull(dialect)
    @ccall mlir_c.mlirDialectIsNull(dialect::MlirDialect)::Bool
end

"""
    mlirDialectEqual(dialect1, dialect2)

Checks if two dialects that belong to the same context are equal. Dialects from different contexts will not compare equal.
"""
function mlirDialectEqual(dialect1, dialect2)
    @ccall mlir_c.mlirDialectEqual(dialect1::MlirDialect, dialect2::MlirDialect)::Bool
end

"""
    mlirDialectGetNamespace(dialect)

Returns the namespace of the given dialect.
"""
function mlirDialectGetNamespace(dialect)
    @ccall mlir_c.mlirDialectGetNamespace(dialect::MlirDialect)::MlirStringRef
end

"""
    mlirDialectHandleGetNamespace(arg1)

Returns the namespace associated with the provided dialect handle.
"""
function mlirDialectHandleGetNamespace(arg1)
    @ccall mlir_c.mlirDialectHandleGetNamespace(arg1::MlirDialectHandle)::MlirStringRef
end

"""
    mlirDialectHandleInsertDialect(arg1, arg2)

Inserts the dialect associated with the provided dialect handle into the provided dialect registry
"""
function mlirDialectHandleInsertDialect(arg1, arg2)
    @ccall mlir_c.mlirDialectHandleInsertDialect(arg1::MlirDialectHandle, arg2::MlirDialectRegistry)::Cvoid
end

"""
    mlirDialectHandleRegisterDialect(arg1, arg2)

Registers the dialect associated with the provided dialect handle.
"""
function mlirDialectHandleRegisterDialect(arg1, arg2)
    @ccall mlir_c.mlirDialectHandleRegisterDialect(arg1::MlirDialectHandle, arg2::MlirContext)::Cvoid
end

"""
    mlirDialectHandleLoadDialect(arg1, arg2)

Loads the dialect associated with the provided dialect handle.
"""
function mlirDialectHandleLoadDialect(arg1, arg2)
    @ccall mlir_c.mlirDialectHandleLoadDialect(arg1::MlirDialectHandle, arg2::MlirContext)::MlirDialect
end

"""
    mlirDialectRegistryCreate()

Creates a dialect registry and transfers its ownership to the caller.
"""
function mlirDialectRegistryCreate()
    @ccall mlir_c.mlirDialectRegistryCreate()::MlirDialectRegistry
end

"""
    mlirDialectRegistryIsNull(registry)

Checks if the dialect registry is null.
"""
function mlirDialectRegistryIsNull(registry)
    @ccall mlir_c.mlirDialectRegistryIsNull(registry::MlirDialectRegistry)::Bool
end

"""
    mlirDialectRegistryDestroy(registry)

Takes a dialect registry owned by the caller and destroys it.
"""
function mlirDialectRegistryDestroy(registry)
    @ccall mlir_c.mlirDialectRegistryDestroy(registry::MlirDialectRegistry)::Cvoid
end

"""
    mlirLocationFileLineColGet(context, filename, line, col)

Creates an File/Line/Column location owned by the given context.
"""
function mlirLocationFileLineColGet(context, filename, line, col)
    @ccall mlir_c.mlirLocationFileLineColGet(context::MlirContext, filename::MlirStringRef, line::Cuint, col::Cuint)::MlirLocation
end

"""
    mlirLocationCallSiteGet(callee, caller)

Creates a call site location with a callee and a caller.
"""
function mlirLocationCallSiteGet(callee, caller)
    @ccall mlir_c.mlirLocationCallSiteGet(callee::MlirLocation, caller::MlirLocation)::MlirLocation
end

"""
    mlirLocationFusedGet(ctx, nLocations, locations, metadata)

Creates a fused location with an array of locations and metadata.
"""
function mlirLocationFusedGet(ctx, nLocations, locations, metadata)
    @ccall mlir_c.mlirLocationFusedGet(ctx::MlirContext, nLocations::intptr_t, locations::Ptr{MlirLocation}, metadata::MlirAttribute)::MlirLocation
end

"""
    mlirLocationNameGet(context, name, childLoc)

Creates a name location owned by the given context. Providing null location for childLoc is allowed and if childLoc is null location, then the behavior is the same as having unknown child location.
"""
function mlirLocationNameGet(context, name, childLoc)
    @ccall mlir_c.mlirLocationNameGet(context::MlirContext, name::MlirStringRef, childLoc::MlirLocation)::MlirLocation
end

"""
    mlirLocationUnknownGet(context)

Creates a location with unknown position owned by the given context.
"""
function mlirLocationUnknownGet(context)
    @ccall mlir_c.mlirLocationUnknownGet(context::MlirContext)::MlirLocation
end

"""
    mlirLocationGetContext(location)

Gets the context that a location was created with.
"""
function mlirLocationGetContext(location)
    @ccall mlir_c.mlirLocationGetContext(location::MlirLocation)::MlirContext
end

"""
    mlirLocationIsNull(location)

Checks if the location is null.
"""
function mlirLocationIsNull(location)
    @ccall mlir_c.mlirLocationIsNull(location::MlirLocation)::Bool
end

"""
    mlirLocationEqual(l1, l2)

Checks if two locations are equal.
"""
function mlirLocationEqual(l1, l2)
    @ccall mlir_c.mlirLocationEqual(l1::MlirLocation, l2::MlirLocation)::Bool
end

"""
    mlirLocationPrint(location, callback, userData)

Prints a location by sending chunks of the string representation and forwarding `userData to `callback`. Note that the callback may be called several times with consecutive chunks of the string.
"""
function mlirLocationPrint(location, callback, userData)
    @ccall mlir_c.mlirLocationPrint(location::MlirLocation, callback::MlirStringCallback, userData::Ptr{Cvoid})::Cvoid
end

"""
    mlirModuleCreateEmpty(location)

Creates a new, empty module and transfers ownership to the caller.
"""
function mlirModuleCreateEmpty(location)
    @ccall mlir_c.mlirModuleCreateEmpty(location::MlirLocation)::MlirModule
end

"""
    mlirModuleCreateParse(context, _module)

Parses a module from the string and transfers ownership to the caller.
"""
function mlirModuleCreateParse(context, _module)
    @ccall mlir_c.mlirModuleCreateParse(context::MlirContext, _module::MlirStringRef)::MlirModule
end

"""
    mlirModuleGetContext(_module)

Gets the context that a module was created with.
"""
function mlirModuleGetContext(_module)
    @ccall mlir_c.mlirModuleGetContext(_module::MlirModule)::MlirContext
end

"""
    mlirModuleGetBody(_module)

Gets the body of the module, i.e. the only block it contains.
"""
function mlirModuleGetBody(_module)
    @ccall mlir_c.mlirModuleGetBody(_module::MlirModule)::MlirBlock
end

"""
    mlirModuleIsNull(_module)

Checks whether a module is null.
"""
function mlirModuleIsNull(_module)
    @ccall mlir_c.mlirModuleIsNull(_module::MlirModule)::Bool
end

"""
    mlirModuleDestroy(_module)

Takes a module owned by the caller and deletes it.
"""
function mlirModuleDestroy(_module)
    @ccall mlir_c.mlirModuleDestroy(_module::MlirModule)::Cvoid
end

"""
    mlirModuleGetOperation(_module)

Views the module as a generic operation.
"""
function mlirModuleGetOperation(_module)
    @ccall mlir_c.mlirModuleGetOperation(_module::MlirModule)::MlirOperation
end

"""
    mlirModuleFromOperation(op)

Views the generic operation as a module. The returned module is null when the input operation was not a ModuleOp.
"""
function mlirModuleFromOperation(op)
    @ccall mlir_c.mlirModuleFromOperation(op::MlirOperation)::MlirModule
end

"""
    MlirOperationState

An auxiliary class for constructing operations.

This class contains all the information necessary to construct the operation. It owns the MlirRegions it has pointers to and does not own anything else. By default, the state can be constructed from a name and location, the latter being also used to access the context, and has no other components. These components can be added progressively until the operation is constructed. Users are not expected to rely on the internals of this class and should use mlirOperationState* functions instead.
"""
mutable struct MlirOperationState
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

"""
    mlirOperationStateGet(name, loc)

Constructs an operation state from a name and a location.
"""
function mlirOperationStateGet(name, loc)
    @ccall mlir_c.mlirOperationStateGet(name::MlirStringRef, loc::MlirLocation)::MlirOperationState
end

"""
    mlirOperationStateAddResults(state, n, results)

Adds a list of components to the operation state.
"""
function mlirOperationStateAddResults(state, n, results)
    @ccall mlir_c.mlirOperationStateAddResults(state::Ptr{MlirOperationState}, n::intptr_t, results::Ptr{MlirType})::Cvoid
end

function mlirOperationStateAddOperands(state, n, operands)
    @ccall mlir_c.mlirOperationStateAddOperands(state::Ptr{MlirOperationState}, n::intptr_t, operands::Ptr{MlirValue})::Cvoid
end

function mlirOperationStateAddOwnedRegions(state, n, regions)
    @ccall mlir_c.mlirOperationStateAddOwnedRegions(state::Ptr{MlirOperationState}, n::intptr_t, regions::Ptr{MlirRegion})::Cvoid
end

function mlirOperationStateAddSuccessors(state, n, successors)
    @ccall mlir_c.mlirOperationStateAddSuccessors(state::Ptr{MlirOperationState}, n::intptr_t, successors::Ptr{MlirBlock})::Cvoid
end

function mlirOperationStateAddAttributes(state, n, attributes)
    @ccall mlir_c.mlirOperationStateAddAttributes(state::Ptr{MlirOperationState}, n::intptr_t, attributes::Ptr{MlirNamedAttribute})::Cvoid
end

"""
    mlirOperationStateEnableResultTypeInference(state)

Enables result type inference for the operation under construction. If enabled, then the caller must not have called [`mlirOperationStateAddResults`](@ref)(). Note that if enabled, the [`mlirOperationCreate`](@ref)() call is failable: it will return a null operation on inference failure and will emit diagnostics.
"""
function mlirOperationStateEnableResultTypeInference(state)
    @ccall mlir_c.mlirOperationStateEnableResultTypeInference(state::Ptr{MlirOperationState})::Cvoid
end

"""
    mlirOpPrintingFlagsCreate()

Creates new printing flags with defaults, intended for customization. Must be freed with a call to [`mlirOpPrintingFlagsDestroy`](@ref)().
"""
function mlirOpPrintingFlagsCreate()
    @ccall mlir_c.mlirOpPrintingFlagsCreate()::MlirOpPrintingFlags
end

"""
    mlirOpPrintingFlagsDestroy(flags)

Destroys printing flags created with [`mlirOpPrintingFlagsCreate`](@ref).
"""
function mlirOpPrintingFlagsDestroy(flags)
    @ccall mlir_c.mlirOpPrintingFlagsDestroy(flags::MlirOpPrintingFlags)::Cvoid
end

"""
    mlirOpPrintingFlagsElideLargeElementsAttrs(flags, largeElementLimit)

Enables the elision of large elements attributes by printing a lexically valid but otherwise meaningless form instead of the element data. The `largeElementLimit` is used to configure what is considered to be a "large" ElementsAttr by providing an upper limit to the number of elements.
"""
function mlirOpPrintingFlagsElideLargeElementsAttrs(flags, largeElementLimit)
    @ccall mlir_c.mlirOpPrintingFlagsElideLargeElementsAttrs(flags::MlirOpPrintingFlags, largeElementLimit::intptr_t)::Cvoid
end

"""
    mlirOpPrintingFlagsEnableDebugInfo(flags, prettyForm)

Enable printing of debug information. If 'prettyForm' is set to true, debug information is printed in a more readable 'pretty' form. Note: The IR generated with 'prettyForm' is not parsable.
"""
function mlirOpPrintingFlagsEnableDebugInfo(flags, prettyForm)
    @ccall mlir_c.mlirOpPrintingFlagsEnableDebugInfo(flags::MlirOpPrintingFlags, prettyForm::Bool)::Cvoid
end

"""
    mlirOpPrintingFlagsPrintGenericOpForm(flags)

Always print operations in the generic form.
"""
function mlirOpPrintingFlagsPrintGenericOpForm(flags)
    @ccall mlir_c.mlirOpPrintingFlagsPrintGenericOpForm(flags::MlirOpPrintingFlags)::Cvoid
end

"""
    mlirOpPrintingFlagsUseLocalScope(flags)

Use local scope when printing the operation. This allows for using the printer in a more localized and thread-safe setting, but may not necessarily be identical to what the IR will look like when dumping the full module.
"""
function mlirOpPrintingFlagsUseLocalScope(flags)
    @ccall mlir_c.mlirOpPrintingFlagsUseLocalScope(flags::MlirOpPrintingFlags)::Cvoid
end

"""
    mlirOperationCreate(state)

Creates an operation and transfers ownership to the caller. Note that caller owned child objects are transferred in this call and must not be further used. Particularly, this applies to any regions added to the state (the implementation may invalidate any such pointers).

This call can fail under the following conditions, in which case, it will return a null operation and emit diagnostics: - Result type inference is enabled and cannot be performed.
"""
function mlirOperationCreate(state)
    @ccall mlir_c.mlirOperationCreate(state::Ptr{MlirOperationState})::MlirOperation
end

"""
    mlirOperationClone(op)

Creates a deep copy of an operation. The operation is not inserted and ownership is transferred to the caller.
"""
function mlirOperationClone(op)
    @ccall mlir_c.mlirOperationClone(op::MlirOperation)::MlirOperation
end

"""
    mlirOperationDestroy(op)

Takes an operation owned by the caller and destroys it.
"""
function mlirOperationDestroy(op)
    @ccall mlir_c.mlirOperationDestroy(op::MlirOperation)::Cvoid
end

"""
    mlirOperationRemoveFromParent(op)

Removes the given operation from its parent block. The operation is not destroyed. The ownership of the operation is transferred to the caller.
"""
function mlirOperationRemoveFromParent(op)
    @ccall mlir_c.mlirOperationRemoveFromParent(op::MlirOperation)::Cvoid
end

"""
    mlirOperationIsNull(op)

Checks whether the underlying operation is null.
"""
function mlirOperationIsNull(op)
    @ccall mlir_c.mlirOperationIsNull(op::MlirOperation)::Bool
end

"""
    mlirOperationEqual(op, other)

Checks whether two operation handles point to the same operation. This does not perform deep comparison.
"""
function mlirOperationEqual(op, other)
    @ccall mlir_c.mlirOperationEqual(op::MlirOperation, other::MlirOperation)::Bool
end

"""
    mlirOperationGetContext(op)

Gets the context this operation is associated with
"""
function mlirOperationGetContext(op)
    @ccall mlir_c.mlirOperationGetContext(op::MlirOperation)::MlirContext
end

"""
    mlirOperationGetLocation(op)

Gets the location of the operation.
"""
function mlirOperationGetLocation(op)
    @ccall mlir_c.mlirOperationGetLocation(op::MlirOperation)::MlirLocation
end

"""
    mlirOperationGetTypeID(op)

Gets the type id of the operation. Returns null if the operation does not have a registered operation description.
"""
function mlirOperationGetTypeID(op)
    @ccall mlir_c.mlirOperationGetTypeID(op::MlirOperation)::MlirTypeID
end

"""
    mlirOperationGetName(op)

Gets the name of the operation as an identifier.
"""
function mlirOperationGetName(op)
    @ccall mlir_c.mlirOperationGetName(op::MlirOperation)::MlirIdentifier
end

"""
    mlirOperationGetBlock(op)

Gets the block that owns this operation, returning null if the operation is not owned.
"""
function mlirOperationGetBlock(op)
    @ccall mlir_c.mlirOperationGetBlock(op::MlirOperation)::MlirBlock
end

"""
    mlirOperationGetParentOperation(op)

Gets the operation that owns this operation, returning null if the operation is not owned.
"""
function mlirOperationGetParentOperation(op)
    @ccall mlir_c.mlirOperationGetParentOperation(op::MlirOperation)::MlirOperation
end

"""
    mlirOperationGetNumRegions(op)

Returns the number of regions attached to the given operation.
"""
function mlirOperationGetNumRegions(op)
    @ccall mlir_c.mlirOperationGetNumRegions(op::MlirOperation)::intptr_t
end

"""
    mlirOperationGetRegion(op, pos)

Returns `pos`-th region attached to the operation.
"""
function mlirOperationGetRegion(op, pos)
    @ccall mlir_c.mlirOperationGetRegion(op::MlirOperation, pos::intptr_t)::MlirRegion
end

"""
    mlirOperationGetNextInBlock(op)

Returns an operation immediately following the given operation it its enclosing block.
"""
function mlirOperationGetNextInBlock(op)
    @ccall mlir_c.mlirOperationGetNextInBlock(op::MlirOperation)::MlirOperation
end

"""
    mlirOperationGetNumOperands(op)

Returns the number of operands of the operation.
"""
function mlirOperationGetNumOperands(op)
    @ccall mlir_c.mlirOperationGetNumOperands(op::MlirOperation)::intptr_t
end

"""
    mlirOperationGetOperand(op, pos)

Returns `pos`-th operand of the operation.
"""
function mlirOperationGetOperand(op, pos)
    @ccall mlir_c.mlirOperationGetOperand(op::MlirOperation, pos::intptr_t)::MlirValue
end

"""
    mlirOperationSetOperand(op, pos, newValue)

Sets the `pos`-th operand of the operation.
"""
function mlirOperationSetOperand(op, pos, newValue)
    @ccall mlir_c.mlirOperationSetOperand(op::MlirOperation, pos::intptr_t, newValue::MlirValue)::Cvoid
end

"""
    mlirOperationGetNumResults(op)

Returns the number of results of the operation.
"""
function mlirOperationGetNumResults(op)
    @ccall mlir_c.mlirOperationGetNumResults(op::MlirOperation)::intptr_t
end

"""
    mlirOperationGetResult(op, pos)

Returns `pos`-th result of the operation.
"""
function mlirOperationGetResult(op, pos)
    @ccall mlir_c.mlirOperationGetResult(op::MlirOperation, pos::intptr_t)::MlirValue
end

"""
    mlirOperationGetNumSuccessors(op)

Returns the number of successor blocks of the operation.
"""
function mlirOperationGetNumSuccessors(op)
    @ccall mlir_c.mlirOperationGetNumSuccessors(op::MlirOperation)::intptr_t
end

"""
    mlirOperationGetSuccessor(op, pos)

Returns `pos`-th successor of the operation.
"""
function mlirOperationGetSuccessor(op, pos)
    @ccall mlir_c.mlirOperationGetSuccessor(op::MlirOperation, pos::intptr_t)::MlirBlock
end

"""
    mlirOperationGetNumAttributes(op)

Returns the number of attributes attached to the operation.
"""
function mlirOperationGetNumAttributes(op)
    @ccall mlir_c.mlirOperationGetNumAttributes(op::MlirOperation)::intptr_t
end

"""
    mlirOperationGetAttribute(op, pos)

Return `pos`-th attribute of the operation.
"""
function mlirOperationGetAttribute(op, pos)
    @ccall mlir_c.mlirOperationGetAttribute(op::MlirOperation, pos::intptr_t)::MlirNamedAttribute
end

"""
    mlirOperationGetAttributeByName(op, name)

Returns an attribute attached to the operation given its name.
"""
function mlirOperationGetAttributeByName(op, name)
    @ccall mlir_c.mlirOperationGetAttributeByName(op::MlirOperation, name::MlirStringRef)::MlirAttribute
end

"""
    mlirOperationSetAttributeByName(op, name, attr)

Sets an attribute by name, replacing the existing if it exists or adding a new one otherwise.
"""
function mlirOperationSetAttributeByName(op, name, attr)
    @ccall mlir_c.mlirOperationSetAttributeByName(op::MlirOperation, name::MlirStringRef, attr::MlirAttribute)::Cvoid
end

"""
    mlirOperationRemoveAttributeByName(op, name)

Removes an attribute by name. Returns false if the attribute was not found and true if removed.
"""
function mlirOperationRemoveAttributeByName(op, name)
    @ccall mlir_c.mlirOperationRemoveAttributeByName(op::MlirOperation, name::MlirStringRef)::Bool
end

"""
    mlirOperationPrint(op, callback, userData)

Prints an operation by sending chunks of the string representation and forwarding `userData to `callback`. Note that the callback may be called several times with consecutive chunks of the string.
"""
function mlirOperationPrint(op, callback, userData)
    @ccall mlir_c.mlirOperationPrint(op::MlirOperation, callback::MlirStringCallback, userData::Ptr{Cvoid})::Cvoid
end

"""
    mlirOperationPrintWithFlags(op, flags, callback, userData)

Same as [`mlirOperationPrint`](@ref) but accepts flags controlling the printing behavior.
"""
function mlirOperationPrintWithFlags(op, flags, callback, userData)
    @ccall mlir_c.mlirOperationPrintWithFlags(op::MlirOperation, flags::MlirOpPrintingFlags, callback::MlirStringCallback, userData::Ptr{Cvoid})::Cvoid
end

"""
    mlirOperationDump(op)

Prints an operation to stderr.
"""
function mlirOperationDump(op)
    @ccall mlir_c.mlirOperationDump(op::MlirOperation)::Cvoid
end

"""
    mlirOperationVerify(op)

Verify the operation and return true if it passes, false if it fails.
"""
function mlirOperationVerify(op)
    @ccall mlir_c.mlirOperationVerify(op::MlirOperation)::Bool
end

"""
    mlirOperationMoveAfter(op, other)

Moves the given operation immediately after the other operation in its parent block. The given operation may be owned by the caller or by its current block. The other operation must belong to a block. In any case, the ownership is transferred to the block of the other operation.
"""
function mlirOperationMoveAfter(op, other)
    @ccall mlir_c.mlirOperationMoveAfter(op::MlirOperation, other::MlirOperation)::Cvoid
end

"""
    mlirOperationMoveBefore(op, other)

Moves the given operation immediately before the other operation in its parent block. The given operation may be owner by the caller or by its current block. The other operation must belong to a block. In any case, the ownership is transferred to the block of the other operation.
"""
function mlirOperationMoveBefore(op, other)
    @ccall mlir_c.mlirOperationMoveBefore(op::MlirOperation, other::MlirOperation)::Cvoid
end

"""
    mlirRegionCreate()

Creates a new empty region and transfers ownership to the caller.
"""
function mlirRegionCreate()
    @ccall mlir_c.mlirRegionCreate()::MlirRegion
end

"""
    mlirRegionDestroy(region)

Takes a region owned by the caller and destroys it.
"""
function mlirRegionDestroy(region)
    @ccall mlir_c.mlirRegionDestroy(region::MlirRegion)::Cvoid
end

"""
    mlirRegionIsNull(region)

Checks whether a region is null.
"""
function mlirRegionIsNull(region)
    @ccall mlir_c.mlirRegionIsNull(region::MlirRegion)::Bool
end

"""
    mlirRegionEqual(region, other)

Checks whether two region handles point to the same region. This does not perform deep comparison.
"""
function mlirRegionEqual(region, other)
    @ccall mlir_c.mlirRegionEqual(region::MlirRegion, other::MlirRegion)::Bool
end

"""
    mlirRegionGetFirstBlock(region)

Gets the first block in the region.
"""
function mlirRegionGetFirstBlock(region)
    @ccall mlir_c.mlirRegionGetFirstBlock(region::MlirRegion)::MlirBlock
end

"""
    mlirRegionAppendOwnedBlock(region, block)

Takes a block owned by the caller and appends it to the given region.
"""
function mlirRegionAppendOwnedBlock(region, block)
    @ccall mlir_c.mlirRegionAppendOwnedBlock(region::MlirRegion, block::MlirBlock)::Cvoid
end

"""
    mlirRegionInsertOwnedBlock(region, pos, block)

Takes a block owned by the caller and inserts it at `pos` to the given region. This is an expensive operation that linearly scans the region, prefer insertAfter/Before instead.
"""
function mlirRegionInsertOwnedBlock(region, pos, block)
    @ccall mlir_c.mlirRegionInsertOwnedBlock(region::MlirRegion, pos::intptr_t, block::MlirBlock)::Cvoid
end

"""
    mlirRegionInsertOwnedBlockAfter(region, reference, block)

Takes a block owned by the caller and inserts it after the (non-owned) reference block in the given region. The reference block must belong to the region. If the reference block is null, prepends the block to the region.
"""
function mlirRegionInsertOwnedBlockAfter(region, reference, block)
    @ccall mlir_c.mlirRegionInsertOwnedBlockAfter(region::MlirRegion, reference::MlirBlock, block::MlirBlock)::Cvoid
end

"""
    mlirRegionInsertOwnedBlockBefore(region, reference, block)

Takes a block owned by the caller and inserts it before the (non-owned) reference block in the given region. The reference block must belong to the region. If the reference block is null, appends the block to the region.
"""
function mlirRegionInsertOwnedBlockBefore(region, reference, block)
    @ccall mlir_c.mlirRegionInsertOwnedBlockBefore(region::MlirRegion, reference::MlirBlock, block::MlirBlock)::Cvoid
end

"""
    mlirOperationGetFirstRegion(op)

Returns first region attached to the operation.
"""
function mlirOperationGetFirstRegion(op)
    @ccall mlir_c.mlirOperationGetFirstRegion(op::MlirOperation)::MlirRegion
end

"""
    mlirRegionGetNextInOperation(region)

Returns the region immediately following the given region in its parent operation.
"""
function mlirRegionGetNextInOperation(region)
    @ccall mlir_c.mlirRegionGetNextInOperation(region::MlirRegion)::MlirRegion
end

"""
    mlirBlockCreate(nArgs, args, locs)

Creates a new empty block with the given argument types and transfers ownership to the caller.
"""
function mlirBlockCreate(nArgs, args, locs)
    @ccall mlir_c.mlirBlockCreate(nArgs::intptr_t, args::Ptr{MlirType}, locs::Ptr{MlirLocation})::MlirBlock
end

"""
    mlirBlockDestroy(block)

Takes a block owned by the caller and destroys it.
"""
function mlirBlockDestroy(block)
    @ccall mlir_c.mlirBlockDestroy(block::MlirBlock)::Cvoid
end

"""
    mlirBlockDetach(block)

Detach a block from the owning region and assume ownership.
"""
function mlirBlockDetach(block)
    @ccall mlir_c.mlirBlockDetach(block::MlirBlock)::Cvoid
end

"""
    mlirBlockIsNull(block)

Checks whether a block is null.
"""
function mlirBlockIsNull(block)
    @ccall mlir_c.mlirBlockIsNull(block::MlirBlock)::Bool
end

"""
    mlirBlockEqual(block, other)

Checks whether two blocks handles point to the same block. This does not perform deep comparison.
"""
function mlirBlockEqual(block, other)
    @ccall mlir_c.mlirBlockEqual(block::MlirBlock, other::MlirBlock)::Bool
end

"""
    mlirBlockGetParentOperation(arg1)

Returns the closest surrounding operation that contains this block.
"""
function mlirBlockGetParentOperation(arg1)
    @ccall mlir_c.mlirBlockGetParentOperation(arg1::MlirBlock)::MlirOperation
end

"""
    mlirBlockGetParentRegion(block)

Returns the region that contains this block.
"""
function mlirBlockGetParentRegion(block)
    @ccall mlir_c.mlirBlockGetParentRegion(block::MlirBlock)::MlirRegion
end

"""
    mlirBlockGetNextInRegion(block)

Returns the block immediately following the given block in its parent region.
"""
function mlirBlockGetNextInRegion(block)
    @ccall mlir_c.mlirBlockGetNextInRegion(block::MlirBlock)::MlirBlock
end

"""
    mlirBlockGetFirstOperation(block)

Returns the first operation in the block.
"""
function mlirBlockGetFirstOperation(block)
    @ccall mlir_c.mlirBlockGetFirstOperation(block::MlirBlock)::MlirOperation
end

"""
    mlirBlockGetTerminator(block)

Returns the terminator operation in the block or null if no terminator.
"""
function mlirBlockGetTerminator(block)
    @ccall mlir_c.mlirBlockGetTerminator(block::MlirBlock)::MlirOperation
end

"""
    mlirBlockAppendOwnedOperation(block, operation)

Takes an operation owned by the caller and appends it to the block.
"""
function mlirBlockAppendOwnedOperation(block, operation)
    @ccall mlir_c.mlirBlockAppendOwnedOperation(block::MlirBlock, operation::MlirOperation)::Cvoid
end

"""
    mlirBlockInsertOwnedOperation(block, pos, operation)

Takes an operation owned by the caller and inserts it as `pos` to the block. This is an expensive operation that scans the block linearly, prefer insertBefore/After instead.
"""
function mlirBlockInsertOwnedOperation(block, pos, operation)
    @ccall mlir_c.mlirBlockInsertOwnedOperation(block::MlirBlock, pos::intptr_t, operation::MlirOperation)::Cvoid
end

"""
    mlirBlockInsertOwnedOperationAfter(block, reference, operation)

Takes an operation owned by the caller and inserts it after the (non-owned) reference operation in the given block. If the reference is null, prepends the operation. Otherwise, the reference must belong to the block.
"""
function mlirBlockInsertOwnedOperationAfter(block, reference, operation)
    @ccall mlir_c.mlirBlockInsertOwnedOperationAfter(block::MlirBlock, reference::MlirOperation, operation::MlirOperation)::Cvoid
end

"""
    mlirBlockInsertOwnedOperationBefore(block, reference, operation)

Takes an operation owned by the caller and inserts it before the (non-owned) reference operation in the given block. If the reference is null, appends the operation. Otherwise, the reference must belong to the block.
"""
function mlirBlockInsertOwnedOperationBefore(block, reference, operation)
    @ccall mlir_c.mlirBlockInsertOwnedOperationBefore(block::MlirBlock, reference::MlirOperation, operation::MlirOperation)::Cvoid
end

"""
    mlirBlockGetNumArguments(block)

Returns the number of arguments of the block.
"""
function mlirBlockGetNumArguments(block)
    @ccall mlir_c.mlirBlockGetNumArguments(block::MlirBlock)::intptr_t
end

"""
    mlirBlockAddArgument(block, type, loc)

Appends an argument of the specified type to the block. Returns the newly added argument.
"""
function mlirBlockAddArgument(block, type, loc)
    @ccall mlir_c.mlirBlockAddArgument(block::MlirBlock, type::MlirType, loc::MlirLocation)::MlirValue
end

"""
    mlirBlockGetArgument(block, pos)

Returns `pos`-th argument of the block.
"""
function mlirBlockGetArgument(block, pos)
    @ccall mlir_c.mlirBlockGetArgument(block::MlirBlock, pos::intptr_t)::MlirValue
end

"""
    mlirBlockPrint(block, callback, userData)

Prints a block by sending chunks of the string representation and forwarding `userData to `callback`. Note that the callback may be called several times with consecutive chunks of the string.
"""
function mlirBlockPrint(block, callback, userData)
    @ccall mlir_c.mlirBlockPrint(block::MlirBlock, callback::MlirStringCallback, userData::Ptr{Cvoid})::Cvoid
end

"""
    mlirValueIsNull(value)

Returns whether the value is null.
"""
function mlirValueIsNull(value)
    @ccall mlir_c.mlirValueIsNull(value::MlirValue)::Bool
end

"""
    mlirValueEqual(value1, value2)

Returns 1 if two values are equal, 0 otherwise.
"""
function mlirValueEqual(value1, value2)
    @ccall mlir_c.mlirValueEqual(value1::MlirValue, value2::MlirValue)::Bool
end

"""
    mlirValueIsABlockArgument(value)

Returns 1 if the value is a block argument, 0 otherwise.
"""
function mlirValueIsABlockArgument(value)
    @ccall mlir_c.mlirValueIsABlockArgument(value::MlirValue)::Bool
end

"""
    mlirValueIsAOpResult(value)

Returns 1 if the value is an operation result, 0 otherwise.
"""
function mlirValueIsAOpResult(value)
    @ccall mlir_c.mlirValueIsAOpResult(value::MlirValue)::Bool
end

"""
    mlirBlockArgumentGetOwner(value)

Returns the block in which this value is defined as an argument. Asserts if the value is not a block argument.
"""
function mlirBlockArgumentGetOwner(value)
    @ccall mlir_c.mlirBlockArgumentGetOwner(value::MlirValue)::MlirBlock
end

"""
    mlirBlockArgumentGetArgNumber(value)

Returns the position of the value in the argument list of its block.
"""
function mlirBlockArgumentGetArgNumber(value)
    @ccall mlir_c.mlirBlockArgumentGetArgNumber(value::MlirValue)::intptr_t
end

"""
    mlirBlockArgumentSetType(value, type)

Sets the type of the block argument to the given type.
"""
function mlirBlockArgumentSetType(value, type)
    @ccall mlir_c.mlirBlockArgumentSetType(value::MlirValue, type::MlirType)::Cvoid
end

"""
    mlirOpResultGetOwner(value)

Returns an operation that produced this value as its result. Asserts if the value is not an op result.
"""
function mlirOpResultGetOwner(value)
    @ccall mlir_c.mlirOpResultGetOwner(value::MlirValue)::MlirOperation
end

"""
    mlirOpResultGetResultNumber(value)

Returns the position of the value in the list of results of the operation that produced it.
"""
function mlirOpResultGetResultNumber(value)
    @ccall mlir_c.mlirOpResultGetResultNumber(value::MlirValue)::intptr_t
end

"""
    mlirValueGetType(value)

Returns the type of the value.
"""
function mlirValueGetType(value)
    @ccall mlir_c.mlirValueGetType(value::MlirValue)::MlirType
end

"""
    mlirValueDump(value)

Prints the value to the standard error stream.
"""
function mlirValueDump(value)
    @ccall mlir_c.mlirValueDump(value::MlirValue)::Cvoid
end

"""
    mlirValuePrint(value, callback, userData)

Prints a value by sending chunks of the string representation and forwarding `userData to `callback`. Note that the callback may be called several times with consecutive chunks of the string.
"""
function mlirValuePrint(value, callback, userData)
    @ccall mlir_c.mlirValuePrint(value::MlirValue, callback::MlirStringCallback, userData::Ptr{Cvoid})::Cvoid
end

"""
    mlirTypeParseGet(context, type)

Parses a type. The type is owned by the context.
"""
function mlirTypeParseGet(context, type)
    @ccall mlir_c.mlirTypeParseGet(context::MlirContext, type::MlirStringRef)::MlirType
end

"""
    mlirTypeGetContext(type)

Gets the context that a type was created with.
"""
function mlirTypeGetContext(type)
    @ccall mlir_c.mlirTypeGetContext(type::MlirType)::MlirContext
end

"""
    mlirTypeGetTypeID(type)

Gets the type ID of the type.
"""
function mlirTypeGetTypeID(type)
    @ccall mlir_c.mlirTypeGetTypeID(type::MlirType)::MlirTypeID
end

"""
    mlirTypeIsNull(type)

Checks whether a type is null.
"""
function mlirTypeIsNull(type)
    @ccall mlir_c.mlirTypeIsNull(type::MlirType)::Bool
end

"""
    mlirTypeEqual(t1, t2)

Checks if two types are equal.
"""
function mlirTypeEqual(t1, t2)
    @ccall mlir_c.mlirTypeEqual(t1::MlirType, t2::MlirType)::Bool
end

"""
    mlirTypePrint(type, callback, userData)

Prints a location by sending chunks of the string representation and forwarding `userData to `callback`. Note that the callback may be called several times with consecutive chunks of the string.
"""
function mlirTypePrint(type, callback, userData)
    @ccall mlir_c.mlirTypePrint(type::MlirType, callback::MlirStringCallback, userData::Ptr{Cvoid})::Cvoid
end

"""
    mlirTypeDump(type)

Prints the type to the standard error stream.
"""
function mlirTypeDump(type)
    @ccall mlir_c.mlirTypeDump(type::MlirType)::Cvoid
end

"""
    mlirAttributeParseGet(context, attr)

Parses an attribute. The attribute is owned by the context.
"""
function mlirAttributeParseGet(context, attr)
    @ccall mlir_c.mlirAttributeParseGet(context::MlirContext, attr::MlirStringRef)::MlirAttribute
end

"""
    mlirAttributeGetContext(attribute)

Gets the context that an attribute was created with.
"""
function mlirAttributeGetContext(attribute)
    @ccall mlir_c.mlirAttributeGetContext(attribute::MlirAttribute)::MlirContext
end

"""
    mlirAttributeGetType(attribute)

Gets the type of this attribute.
"""
function mlirAttributeGetType(attribute)
    @ccall mlir_c.mlirAttributeGetType(attribute::MlirAttribute)::MlirType
end

"""
    mlirAttributeGetTypeID(attribute)

Gets the type id of the attribute.
"""
function mlirAttributeGetTypeID(attribute)
    @ccall mlir_c.mlirAttributeGetTypeID(attribute::MlirAttribute)::MlirTypeID
end

"""
    mlirAttributeIsNull(attr)

Checks whether an attribute is null.
"""
function mlirAttributeIsNull(attr)
    @ccall mlir_c.mlirAttributeIsNull(attr::MlirAttribute)::Bool
end

"""
    mlirAttributeEqual(a1, a2)

Checks if two attributes are equal.
"""
function mlirAttributeEqual(a1, a2)
    @ccall mlir_c.mlirAttributeEqual(a1::MlirAttribute, a2::MlirAttribute)::Bool
end

"""
    mlirAttributePrint(attr, callback, userData)

Prints an attribute by sending chunks of the string representation and forwarding `userData to `callback`. Note that the callback may be called several times with consecutive chunks of the string.
"""
function mlirAttributePrint(attr, callback, userData)
    @ccall mlir_c.mlirAttributePrint(attr::MlirAttribute, callback::MlirStringCallback, userData::Ptr{Cvoid})::Cvoid
end

"""
    mlirAttributeDump(attr)

Prints the attribute to the standard error stream.
"""
function mlirAttributeDump(attr)
    @ccall mlir_c.mlirAttributeDump(attr::MlirAttribute)::Cvoid
end

"""
    mlirNamedAttributeGet(name, attr)

Associates an attribute with the name. Takes ownership of neither.
"""
function mlirNamedAttributeGet(name, attr)
    @ccall mlir_c.mlirNamedAttributeGet(name::MlirIdentifier, attr::MlirAttribute)::MlirNamedAttribute
end

"""
    mlirIdentifierGet(context, str)

Gets an identifier with the given string value.
"""
function mlirIdentifierGet(context, str)
    @ccall mlir_c.mlirIdentifierGet(context::MlirContext, str::MlirStringRef)::MlirIdentifier
end

"""
    mlirIdentifierGetContext(arg1)

Returns the context associated with this identifier
"""
function mlirIdentifierGetContext(arg1)
    @ccall mlir_c.mlirIdentifierGetContext(arg1::MlirIdentifier)::MlirContext
end

"""
    mlirIdentifierEqual(ident, other)

Checks whether two identifiers are the same.
"""
function mlirIdentifierEqual(ident, other)
    @ccall mlir_c.mlirIdentifierEqual(ident::MlirIdentifier, other::MlirIdentifier)::Bool
end

"""
    mlirIdentifierStr(ident)

Gets the string value of the identifier.
"""
function mlirIdentifierStr(ident)
    @ccall mlir_c.mlirIdentifierStr(ident::MlirIdentifier)::MlirStringRef
end

"""
    mlirSymbolTableGetSymbolAttributeName()

Returns the name of the attribute used to store symbol names compatible with symbol tables.
"""
function mlirSymbolTableGetSymbolAttributeName()
    @ccall mlir_c.mlirSymbolTableGetSymbolAttributeName()::MlirStringRef
end

"""
    mlirSymbolTableGetVisibilityAttributeName()

Returns the name of the attribute used to store symbol visibility.
"""
function mlirSymbolTableGetVisibilityAttributeName()
    @ccall mlir_c.mlirSymbolTableGetVisibilityAttributeName()::MlirStringRef
end

"""
    mlirSymbolTableCreate(operation)

Creates a symbol table for the given operation. If the operation does not have the SymbolTable trait, returns a null symbol table.
"""
function mlirSymbolTableCreate(operation)
    @ccall mlir_c.mlirSymbolTableCreate(operation::MlirOperation)::MlirSymbolTable
end

"""
    mlirSymbolTableIsNull(symbolTable)

Returns true if the symbol table is null.
"""
function mlirSymbolTableIsNull(symbolTable)
    @ccall mlir_c.mlirSymbolTableIsNull(symbolTable::MlirSymbolTable)::Bool
end

"""
    mlirSymbolTableDestroy(symbolTable)

Destroys the symbol table created with [`mlirSymbolTableCreate`](@ref). This does not affect the operations in the table.
"""
function mlirSymbolTableDestroy(symbolTable)
    @ccall mlir_c.mlirSymbolTableDestroy(symbolTable::MlirSymbolTable)::Cvoid
end

"""
    mlirSymbolTableLookup(symbolTable, name)

Looks up a symbol with the given name in the given symbol table and returns the operation that corresponds to the symbol. If the symbol cannot be found, returns a null operation.
"""
function mlirSymbolTableLookup(symbolTable, name)
    @ccall mlir_c.mlirSymbolTableLookup(symbolTable::MlirSymbolTable, name::MlirStringRef)::MlirOperation
end

"""
    mlirSymbolTableInsert(symbolTable, operation)

Inserts the given operation into the given symbol table. The operation must have the symbol trait. If the symbol table already has a symbol with the same name, renames the symbol being inserted to ensure name uniqueness. Note that this does not move the operation itself into the block of the symbol table operation, this should be done separately. Returns the name of the symbol after insertion.
"""
function mlirSymbolTableInsert(symbolTable, operation)
    @ccall mlir_c.mlirSymbolTableInsert(symbolTable::MlirSymbolTable, operation::MlirOperation)::MlirAttribute
end

"""
    mlirSymbolTableErase(symbolTable, operation)

Removes the given operation from the symbol table and erases it.
"""
function mlirSymbolTableErase(symbolTable, operation)
    @ccall mlir_c.mlirSymbolTableErase(symbolTable::MlirSymbolTable, operation::MlirOperation)::Cvoid
end

"""
    mlirSymbolTableReplaceAllSymbolUses(oldSymbol, newSymbol, from)

Attempt to replace all uses that are nested within the given operation of the given symbol 'oldSymbol' with the provided 'newSymbol'. This does not traverse into nested symbol tables. Will fail atomically if there are any unknown operations that may be potential symbol tables.
"""
function mlirSymbolTableReplaceAllSymbolUses(oldSymbol, newSymbol, from)
    @ccall mlir_c.mlirSymbolTableReplaceAllSymbolUses(oldSymbol::MlirStringRef, newSymbol::MlirStringRef, from::MlirOperation)::MlirLogicalResult
end

"""
    mlirSymbolTableWalkSymbolTables(from, allSymUsesVisible, callback, userData)

Walks all symbol table operations nested within, and including, `op`. For each symbol table operation, the provided callback is invoked with the op and a boolean signifying if the symbols within that symbol table can be treated as if all uses within the IR are visible to the caller. `allSymUsesVisible` identifies whether all of the symbol uses of symbols within `op` are visible.
"""
function mlirSymbolTableWalkSymbolTables(from, allSymUsesVisible, callback, userData)
    @ccall mlir_c.mlirSymbolTableWalkSymbolTables(from::MlirOperation, allSymUsesVisible::Bool, callback::Ptr{Cvoid}, userData::Ptr{Cvoid})::Cvoid
end

struct MlirAffineExpr
    ptr::Ptr{Cvoid}
end

"""
    mlirAffineExprGetContext(affineExpr)

Gets the context that owns the affine expression.
"""
function mlirAffineExprGetContext(affineExpr)
    @ccall mlir_c.mlirAffineExprGetContext(affineExpr::MlirAffineExpr)::MlirContext
end

"""
    mlirAffineExprEqual(lhs, rhs)

Returns `true` if the two affine expressions are equal.
"""
function mlirAffineExprEqual(lhs, rhs)
    @ccall mlir_c.mlirAffineExprEqual(lhs::MlirAffineExpr, rhs::MlirAffineExpr)::Bool
end

"""
    mlirAffineExprIsNull(affineExpr)

Returns `true` if the given affine expression is a null expression. Note constant zero is not a null expression.
"""
function mlirAffineExprIsNull(affineExpr)
    @ccall mlir_c.mlirAffineExprIsNull(affineExpr::MlirAffineExpr)::Bool
end

"""
    mlirAffineExprPrint(affineExpr, callback, userData)

Prints an affine expression by sending chunks of the string representation and forwarding `userData to `callback`. Note that the callback may be called several times with consecutive chunks of the string.
"""
function mlirAffineExprPrint(affineExpr, callback, userData)
    @ccall mlir_c.mlirAffineExprPrint(affineExpr::MlirAffineExpr, callback::MlirStringCallback, userData::Ptr{Cvoid})::Cvoid
end

"""
    mlirAffineExprDump(affineExpr)

Prints the affine expression to the standard error stream.
"""
function mlirAffineExprDump(affineExpr)
    @ccall mlir_c.mlirAffineExprDump(affineExpr::MlirAffineExpr)::Cvoid
end

"""
    mlirAffineExprIsSymbolicOrConstant(affineExpr)

Checks whether the given affine expression is made out of only symbols and constants.
"""
function mlirAffineExprIsSymbolicOrConstant(affineExpr)
    @ccall mlir_c.mlirAffineExprIsSymbolicOrConstant(affineExpr::MlirAffineExpr)::Bool
end

"""
    mlirAffineExprIsPureAffine(affineExpr)

Checks whether the given affine expression is a pure affine expression, i.e. mul, floordiv, ceildic, and mod is only allowed w.r.t constants.
"""
function mlirAffineExprIsPureAffine(affineExpr)
    @ccall mlir_c.mlirAffineExprIsPureAffine(affineExpr::MlirAffineExpr)::Bool
end

"""
    mlirAffineExprGetLargestKnownDivisor(affineExpr)

Returns the greatest known integral divisor of this affine expression. The result is always positive.
"""
function mlirAffineExprGetLargestKnownDivisor(affineExpr)
    @ccall mlir_c.mlirAffineExprGetLargestKnownDivisor(affineExpr::MlirAffineExpr)::Int64
end

"""
    mlirAffineExprIsMultipleOf(affineExpr, factor)

Checks whether the given affine expression is a multiple of 'factor'.
"""
function mlirAffineExprIsMultipleOf(affineExpr, factor)
    @ccall mlir_c.mlirAffineExprIsMultipleOf(affineExpr::MlirAffineExpr, factor::Int64)::Bool
end

"""
    mlirAffineExprIsFunctionOfDim(affineExpr, position)

Checks whether the given affine expression involves AffineDimExpr 'position'.
"""
function mlirAffineExprIsFunctionOfDim(affineExpr, position)
    @ccall mlir_c.mlirAffineExprIsFunctionOfDim(affineExpr::MlirAffineExpr, position::intptr_t)::Bool
end

struct MlirAffineMap
    ptr::Ptr{Cvoid}
end

"""
    mlirAffineExprCompose(affineExpr, affineMap)

Composes the given map with the given expression.
"""
function mlirAffineExprCompose(affineExpr, affineMap)
    @ccall mlir_c.mlirAffineExprCompose(affineExpr::MlirAffineExpr, affineMap::MlirAffineMap)::MlirAffineExpr
end

"""
    mlirAffineExprIsADim(affineExpr)

Checks whether the given affine expression is a dimension expression.
"""
function mlirAffineExprIsADim(affineExpr)
    @ccall mlir_c.mlirAffineExprIsADim(affineExpr::MlirAffineExpr)::Bool
end

"""
    mlirAffineDimExprGet(ctx, position)

Creates an affine dimension expression with 'position' in the context.
"""
function mlirAffineDimExprGet(ctx, position)
    @ccall mlir_c.mlirAffineDimExprGet(ctx::MlirContext, position::intptr_t)::MlirAffineExpr
end

"""
    mlirAffineDimExprGetPosition(affineExpr)

Returns the position of the given affine dimension expression.
"""
function mlirAffineDimExprGetPosition(affineExpr)
    @ccall mlir_c.mlirAffineDimExprGetPosition(affineExpr::MlirAffineExpr)::intptr_t
end

"""
    mlirAffineExprIsASymbol(affineExpr)

Checks whether the given affine expression is a symbol expression.
"""
function mlirAffineExprIsASymbol(affineExpr)
    @ccall mlir_c.mlirAffineExprIsASymbol(affineExpr::MlirAffineExpr)::Bool
end

"""
    mlirAffineSymbolExprGet(ctx, position)

Creates an affine symbol expression with 'position' in the context.
"""
function mlirAffineSymbolExprGet(ctx, position)
    @ccall mlir_c.mlirAffineSymbolExprGet(ctx::MlirContext, position::intptr_t)::MlirAffineExpr
end

"""
    mlirAffineSymbolExprGetPosition(affineExpr)

Returns the position of the given affine symbol expression.
"""
function mlirAffineSymbolExprGetPosition(affineExpr)
    @ccall mlir_c.mlirAffineSymbolExprGetPosition(affineExpr::MlirAffineExpr)::intptr_t
end

"""
    mlirAffineExprIsAConstant(affineExpr)

Checks whether the given affine expression is a constant expression.
"""
function mlirAffineExprIsAConstant(affineExpr)
    @ccall mlir_c.mlirAffineExprIsAConstant(affineExpr::MlirAffineExpr)::Bool
end

"""
    mlirAffineConstantExprGet(ctx, constant)

Creates an affine constant expression with 'constant' in the context.
"""
function mlirAffineConstantExprGet(ctx, constant)
    @ccall mlir_c.mlirAffineConstantExprGet(ctx::MlirContext, constant::Int64)::MlirAffineExpr
end

"""
    mlirAffineConstantExprGetValue(affineExpr)

Returns the value of the given affine constant expression.
"""
function mlirAffineConstantExprGetValue(affineExpr)
    @ccall mlir_c.mlirAffineConstantExprGetValue(affineExpr::MlirAffineExpr)::Int64
end

"""
    mlirAffineExprIsAAdd(affineExpr)

Checks whether the given affine expression is an add expression.
"""
function mlirAffineExprIsAAdd(affineExpr)
    @ccall mlir_c.mlirAffineExprIsAAdd(affineExpr::MlirAffineExpr)::Bool
end

"""
    mlirAffineAddExprGet(lhs, rhs)

Creates an affine add expression with 'lhs' and 'rhs'.
"""
function mlirAffineAddExprGet(lhs, rhs)
    @ccall mlir_c.mlirAffineAddExprGet(lhs::MlirAffineExpr, rhs::MlirAffineExpr)::MlirAffineExpr
end

"""
    mlirAffineExprIsAMul(affineExpr)

Checks whether the given affine expression is an mul expression.
"""
function mlirAffineExprIsAMul(affineExpr)
    @ccall mlir_c.mlirAffineExprIsAMul(affineExpr::MlirAffineExpr)::Bool
end

"""
    mlirAffineMulExprGet(lhs, rhs)

Creates an affine mul expression with 'lhs' and 'rhs'.
"""
function mlirAffineMulExprGet(lhs, rhs)
    @ccall mlir_c.mlirAffineMulExprGet(lhs::MlirAffineExpr, rhs::MlirAffineExpr)::MlirAffineExpr
end

"""
    mlirAffineExprIsAMod(affineExpr)

Checks whether the given affine expression is an mod expression.
"""
function mlirAffineExprIsAMod(affineExpr)
    @ccall mlir_c.mlirAffineExprIsAMod(affineExpr::MlirAffineExpr)::Bool
end

"""
    mlirAffineModExprGet(lhs, rhs)

Creates an affine mod expression with 'lhs' and 'rhs'.
"""
function mlirAffineModExprGet(lhs, rhs)
    @ccall mlir_c.mlirAffineModExprGet(lhs::MlirAffineExpr, rhs::MlirAffineExpr)::MlirAffineExpr
end

"""
    mlirAffineExprIsAFloorDiv(affineExpr)

Checks whether the given affine expression is an floordiv expression.
"""
function mlirAffineExprIsAFloorDiv(affineExpr)
    @ccall mlir_c.mlirAffineExprIsAFloorDiv(affineExpr::MlirAffineExpr)::Bool
end

"""
    mlirAffineFloorDivExprGet(lhs, rhs)

Creates an affine floordiv expression with 'lhs' and 'rhs'.
"""
function mlirAffineFloorDivExprGet(lhs, rhs)
    @ccall mlir_c.mlirAffineFloorDivExprGet(lhs::MlirAffineExpr, rhs::MlirAffineExpr)::MlirAffineExpr
end

"""
    mlirAffineExprIsACeilDiv(affineExpr)

Checks whether the given affine expression is an ceildiv expression.
"""
function mlirAffineExprIsACeilDiv(affineExpr)
    @ccall mlir_c.mlirAffineExprIsACeilDiv(affineExpr::MlirAffineExpr)::Bool
end

"""
    mlirAffineCeilDivExprGet(lhs, rhs)

Creates an affine ceildiv expression with 'lhs' and 'rhs'.
"""
function mlirAffineCeilDivExprGet(lhs, rhs)
    @ccall mlir_c.mlirAffineCeilDivExprGet(lhs::MlirAffineExpr, rhs::MlirAffineExpr)::MlirAffineExpr
end

"""
    mlirAffineExprIsABinary(affineExpr)

Checks whether the given affine expression is binary.
"""
function mlirAffineExprIsABinary(affineExpr)
    @ccall mlir_c.mlirAffineExprIsABinary(affineExpr::MlirAffineExpr)::Bool
end

"""
    mlirAffineBinaryOpExprGetLHS(affineExpr)

Returns the left hand side affine expression of the given affine binary operation expression.
"""
function mlirAffineBinaryOpExprGetLHS(affineExpr)
    @ccall mlir_c.mlirAffineBinaryOpExprGetLHS(affineExpr::MlirAffineExpr)::MlirAffineExpr
end

"""
    mlirAffineBinaryOpExprGetRHS(affineExpr)

Returns the right hand side affine expression of the given affine binary operation expression.
"""
function mlirAffineBinaryOpExprGetRHS(affineExpr)
    @ccall mlir_c.mlirAffineBinaryOpExprGetRHS(affineExpr::MlirAffineExpr)::MlirAffineExpr
end

"""
    mlirAffineMapGetContext(affineMap)

Gets the context that the given affine map was created with
"""
function mlirAffineMapGetContext(affineMap)
    @ccall mlir_c.mlirAffineMapGetContext(affineMap::MlirAffineMap)::MlirContext
end

"""
    mlirAffineMapIsNull(affineMap)

Checks whether an affine map is null.
"""
function mlirAffineMapIsNull(affineMap)
    @ccall mlir_c.mlirAffineMapIsNull(affineMap::MlirAffineMap)::Bool
end

"""
    mlirAffineMapEqual(a1, a2)

Checks if two affine maps are equal.
"""
function mlirAffineMapEqual(a1, a2)
    @ccall mlir_c.mlirAffineMapEqual(a1::MlirAffineMap, a2::MlirAffineMap)::Bool
end

"""
    mlirAffineMapPrint(affineMap, callback, userData)

Prints an affine map by sending chunks of the string representation and forwarding `userData to `callback`. Note that the callback may be called several times with consecutive chunks of the string.
"""
function mlirAffineMapPrint(affineMap, callback, userData)
    @ccall mlir_c.mlirAffineMapPrint(affineMap::MlirAffineMap, callback::MlirStringCallback, userData::Ptr{Cvoid})::Cvoid
end

"""
    mlirAffineMapDump(affineMap)

Prints the affine map to the standard error stream.
"""
function mlirAffineMapDump(affineMap)
    @ccall mlir_c.mlirAffineMapDump(affineMap::MlirAffineMap)::Cvoid
end

"""
    mlirAffineMapEmptyGet(ctx)

Creates a zero result affine map with no dimensions or symbols in the context. The affine map is owned by the context.
"""
function mlirAffineMapEmptyGet(ctx)
    @ccall mlir_c.mlirAffineMapEmptyGet(ctx::MlirContext)::MlirAffineMap
end

"""
    mlirAffineMapZeroResultGet(ctx, dimCount, symbolCount)

Creates a zero result affine map of the given dimensions and symbols in the context. The affine map is owned by the context.
"""
function mlirAffineMapZeroResultGet(ctx, dimCount, symbolCount)
    @ccall mlir_c.mlirAffineMapZeroResultGet(ctx::MlirContext, dimCount::intptr_t, symbolCount::intptr_t)::MlirAffineMap
end

"""
    mlirAffineMapGet(ctx, dimCount, symbolCount, nAffineExprs, affineExprs)

Creates an affine map with results defined by the given list of affine expressions. The map resulting map also has the requested number of input dimensions and symbols, regardless of them being used in the results.
"""
function mlirAffineMapGet(ctx, dimCount, symbolCount, nAffineExprs, affineExprs)
    @ccall mlir_c.mlirAffineMapGet(ctx::MlirContext, dimCount::intptr_t, symbolCount::intptr_t, nAffineExprs::intptr_t, affineExprs::Ptr{MlirAffineExpr})::MlirAffineMap
end

"""
    mlirAffineMapConstantGet(ctx, val)

Creates a single constant result affine map in the context. The affine map is owned by the context.
"""
function mlirAffineMapConstantGet(ctx, val)
    @ccall mlir_c.mlirAffineMapConstantGet(ctx::MlirContext, val::Int64)::MlirAffineMap
end

"""
    mlirAffineMapMultiDimIdentityGet(ctx, numDims)

Creates an affine map with 'numDims' identity in the context. The affine map is owned by the context.
"""
function mlirAffineMapMultiDimIdentityGet(ctx, numDims)
    @ccall mlir_c.mlirAffineMapMultiDimIdentityGet(ctx::MlirContext, numDims::intptr_t)::MlirAffineMap
end

"""
    mlirAffineMapMinorIdentityGet(ctx, dims, results)

Creates an identity affine map on the most minor dimensions in the context. The affine map is owned by the context. The function asserts that the number of dimensions is greater or equal to the number of results.
"""
function mlirAffineMapMinorIdentityGet(ctx, dims, results)
    @ccall mlir_c.mlirAffineMapMinorIdentityGet(ctx::MlirContext, dims::intptr_t, results::intptr_t)::MlirAffineMap
end

"""
    mlirAffineMapPermutationGet(ctx, size, permutation)

Creates an affine map with a permutation expression and its size in the context. The permutation expression is a non-empty vector of integers. The elements of the permutation vector must be continuous from 0 and cannot be repeated (i.e. `[1,2,0]` is a valid permutation. `[2,0]` or `[1,1,2]` is an invalid invalid permutation.) The affine map is owned by the context.
"""
function mlirAffineMapPermutationGet(ctx, size, permutation)
    @ccall mlir_c.mlirAffineMapPermutationGet(ctx::MlirContext, size::intptr_t, permutation::Ptr{Cuint})::MlirAffineMap
end

"""
    mlirAffineMapIsIdentity(affineMap)

Checks whether the given affine map is an identity affine map. The function asserts that the number of dimensions is greater or equal to the number of results.
"""
function mlirAffineMapIsIdentity(affineMap)
    @ccall mlir_c.mlirAffineMapIsIdentity(affineMap::MlirAffineMap)::Bool
end

"""
    mlirAffineMapIsMinorIdentity(affineMap)

Checks whether the given affine map is a minor identity affine map.
"""
function mlirAffineMapIsMinorIdentity(affineMap)
    @ccall mlir_c.mlirAffineMapIsMinorIdentity(affineMap::MlirAffineMap)::Bool
end

"""
    mlirAffineMapIsEmpty(affineMap)

Checks whether the given affine map is an empty affine map.
"""
function mlirAffineMapIsEmpty(affineMap)
    @ccall mlir_c.mlirAffineMapIsEmpty(affineMap::MlirAffineMap)::Bool
end

"""
    mlirAffineMapIsSingleConstant(affineMap)

Checks whether the given affine map is a single result constant affine map.
"""
function mlirAffineMapIsSingleConstant(affineMap)
    @ccall mlir_c.mlirAffineMapIsSingleConstant(affineMap::MlirAffineMap)::Bool
end

"""
    mlirAffineMapGetSingleConstantResult(affineMap)

Returns the constant result of the given affine map. The function asserts that the map has a single constant result.
"""
function mlirAffineMapGetSingleConstantResult(affineMap)
    @ccall mlir_c.mlirAffineMapGetSingleConstantResult(affineMap::MlirAffineMap)::Int64
end

"""
    mlirAffineMapGetNumDims(affineMap)

Returns the number of dimensions of the given affine map.
"""
function mlirAffineMapGetNumDims(affineMap)
    @ccall mlir_c.mlirAffineMapGetNumDims(affineMap::MlirAffineMap)::intptr_t
end

"""
    mlirAffineMapGetNumSymbols(affineMap)

Returns the number of symbols of the given affine map.
"""
function mlirAffineMapGetNumSymbols(affineMap)
    @ccall mlir_c.mlirAffineMapGetNumSymbols(affineMap::MlirAffineMap)::intptr_t
end

"""
    mlirAffineMapGetNumResults(affineMap)

Returns the number of results of the given affine map.
"""
function mlirAffineMapGetNumResults(affineMap)
    @ccall mlir_c.mlirAffineMapGetNumResults(affineMap::MlirAffineMap)::intptr_t
end

"""
    mlirAffineMapGetResult(affineMap, pos)

Returns the result at the given position.
"""
function mlirAffineMapGetResult(affineMap, pos)
    @ccall mlir_c.mlirAffineMapGetResult(affineMap::MlirAffineMap, pos::intptr_t)::MlirAffineExpr
end

"""
    mlirAffineMapGetNumInputs(affineMap)

Returns the number of inputs (dimensions + symbols) of the given affine map.
"""
function mlirAffineMapGetNumInputs(affineMap)
    @ccall mlir_c.mlirAffineMapGetNumInputs(affineMap::MlirAffineMap)::intptr_t
end

"""
    mlirAffineMapIsProjectedPermutation(affineMap)

Checks whether the given affine map represents a subset of a symbol-less permutation map.
"""
function mlirAffineMapIsProjectedPermutation(affineMap)
    @ccall mlir_c.mlirAffineMapIsProjectedPermutation(affineMap::MlirAffineMap)::Bool
end

"""
    mlirAffineMapIsPermutation(affineMap)

Checks whether the given affine map represents a symbol-less permutation map.
"""
function mlirAffineMapIsPermutation(affineMap)
    @ccall mlir_c.mlirAffineMapIsPermutation(affineMap::MlirAffineMap)::Bool
end

"""
    mlirAffineMapGetSubMap(affineMap, size, resultPos)

Returns the affine map consisting of the `resultPos` subset.
"""
function mlirAffineMapGetSubMap(affineMap, size, resultPos)
    @ccall mlir_c.mlirAffineMapGetSubMap(affineMap::MlirAffineMap, size::intptr_t, resultPos::Ptr{intptr_t})::MlirAffineMap
end

"""
    mlirAffineMapGetMajorSubMap(affineMap, numResults)

Returns the affine map consisting of the most major `numResults` results. Returns the null AffineMap if the `numResults` is equal to zero. Returns the `affineMap` if `numResults` is greater or equals to number of results of the given affine map.
"""
function mlirAffineMapGetMajorSubMap(affineMap, numResults)
    @ccall mlir_c.mlirAffineMapGetMajorSubMap(affineMap::MlirAffineMap, numResults::intptr_t)::MlirAffineMap
end

"""
    mlirAffineMapGetMinorSubMap(affineMap, numResults)

Returns the affine map consisting of the most minor `numResults` results. Returns the null AffineMap if the `numResults` is equal to zero. Returns the `affineMap` if `numResults` is greater or equals to number of results of the given affine map.
"""
function mlirAffineMapGetMinorSubMap(affineMap, numResults)
    @ccall mlir_c.mlirAffineMapGetMinorSubMap(affineMap::MlirAffineMap, numResults::intptr_t)::MlirAffineMap
end

"""
    mlirAffineMapReplace(affineMap, expression, replacement, numResultDims, numResultSyms)

Apply AffineExpr::replace(`map`) to each of the results and return a new new AffineMap with the new results and the specified number of dims and symbols.
"""
function mlirAffineMapReplace(affineMap, expression, replacement, numResultDims, numResultSyms)
    @ccall mlir_c.mlirAffineMapReplace(affineMap::MlirAffineMap, expression::MlirAffineExpr, replacement::MlirAffineExpr, numResultDims::intptr_t, numResultSyms::intptr_t)::MlirAffineMap
end

"""
    mlirAffineMapCompressUnusedSymbols(affineMaps, size, result, populateResult)

Returns the simplified affine map resulting from dropping the symbols that do not appear in any of the individual maps in `affineMaps`. Asserts that all maps in `affineMaps` are normalized to the same number of dims and symbols. Takes a callback `populateResult` to fill the `res` container with value `m` at entry `idx`. This allows returning without worrying about ownership considerations.
"""
function mlirAffineMapCompressUnusedSymbols(affineMaps, size, result, populateResult)
    @ccall mlir_c.mlirAffineMapCompressUnusedSymbols(affineMaps::Ptr{MlirAffineMap}, size::intptr_t, result::Ptr{Cvoid}, populateResult::Ptr{Cvoid})::Cvoid
end

"""
    mlirAttributeGetNull()

Returns an empty attribute.
"""
function mlirAttributeGetNull()
    @ccall mlir_c.mlirAttributeGetNull()::MlirAttribute
end

"""
    mlirAttributeIsAAffineMap(attr)

Checks whether the given attribute is an affine map attribute.
"""
function mlirAttributeIsAAffineMap(attr)
    @ccall mlir_c.mlirAttributeIsAAffineMap(attr::MlirAttribute)::Bool
end

"""
    mlirAffineMapAttrGet(map)

Creates an affine map attribute wrapping the given map. The attribute belongs to the same context as the affine map.
"""
function mlirAffineMapAttrGet(map)
    @ccall mlir_c.mlirAffineMapAttrGet(map::MlirAffineMap)::MlirAttribute
end

"""
    mlirAffineMapAttrGetValue(attr)

Returns the affine map wrapped in the given affine map attribute.
"""
function mlirAffineMapAttrGetValue(attr)
    @ccall mlir_c.mlirAffineMapAttrGetValue(attr::MlirAttribute)::MlirAffineMap
end

"""
    mlirAttributeIsAArray(attr)

Checks whether the given attribute is an array attribute.
"""
function mlirAttributeIsAArray(attr)
    @ccall mlir_c.mlirAttributeIsAArray(attr::MlirAttribute)::Bool
end

"""
    mlirArrayAttrGet(ctx, numElements, elements)

Creates an array element containing the given list of elements in the given context.
"""
function mlirArrayAttrGet(ctx, numElements, elements)
    @ccall mlir_c.mlirArrayAttrGet(ctx::MlirContext, numElements::intptr_t, elements::Ptr{MlirAttribute})::MlirAttribute
end

"""
    mlirArrayAttrGetNumElements(attr)

Returns the number of elements stored in the given array attribute.
"""
function mlirArrayAttrGetNumElements(attr)
    @ccall mlir_c.mlirArrayAttrGetNumElements(attr::MlirAttribute)::intptr_t
end

"""
    mlirArrayAttrGetElement(attr, pos)

Returns pos-th element stored in the given array attribute.
"""
function mlirArrayAttrGetElement(attr, pos)
    @ccall mlir_c.mlirArrayAttrGetElement(attr::MlirAttribute, pos::intptr_t)::MlirAttribute
end

"""
    mlirAttributeIsADictionary(attr)

Checks whether the given attribute is a dictionary attribute.
"""
function mlirAttributeIsADictionary(attr)
    @ccall mlir_c.mlirAttributeIsADictionary(attr::MlirAttribute)::Bool
end

"""
    mlirDictionaryAttrGet(ctx, numElements, elements)

Creates a dictionary attribute containing the given list of elements in the provided context.
"""
function mlirDictionaryAttrGet(ctx, numElements, elements)
    @ccall mlir_c.mlirDictionaryAttrGet(ctx::MlirContext, numElements::intptr_t, elements::Ptr{MlirNamedAttribute})::MlirAttribute
end

"""
    mlirDictionaryAttrGetNumElements(attr)

Returns the number of attributes contained in a dictionary attribute.
"""
function mlirDictionaryAttrGetNumElements(attr)
    @ccall mlir_c.mlirDictionaryAttrGetNumElements(attr::MlirAttribute)::intptr_t
end

"""
    mlirDictionaryAttrGetElement(attr, pos)

Returns pos-th element of the given dictionary attribute.
"""
function mlirDictionaryAttrGetElement(attr, pos)
    @ccall mlir_c.mlirDictionaryAttrGetElement(attr::MlirAttribute, pos::intptr_t)::MlirNamedAttribute
end

"""
    mlirDictionaryAttrGetElementByName(attr, name)

Returns the dictionary attribute element with the given name or NULL if the given name does not exist in the dictionary.
"""
function mlirDictionaryAttrGetElementByName(attr, name)
    @ccall mlir_c.mlirDictionaryAttrGetElementByName(attr::MlirAttribute, name::MlirStringRef)::MlirAttribute
end

"""
    mlirAttributeIsAFloat(attr)

Checks whether the given attribute is a floating point attribute.
"""
function mlirAttributeIsAFloat(attr)
    @ccall mlir_c.mlirAttributeIsAFloat(attr::MlirAttribute)::Bool
end

"""
    mlirFloatAttrDoubleGet(ctx, type, value)

Creates a floating point attribute in the given context with the given double value and double-precision FP semantics.
"""
function mlirFloatAttrDoubleGet(ctx, type, value)
    @ccall mlir_c.mlirFloatAttrDoubleGet(ctx::MlirContext, type::MlirType, value::Cdouble)::MlirAttribute
end

"""
    mlirFloatAttrDoubleGetChecked(loc, type, value)

Same as "[`mlirFloatAttrDoubleGet`](@ref)", but if the type is not valid for a construction of a FloatAttr, returns a null [`MlirAttribute`](@ref).
"""
function mlirFloatAttrDoubleGetChecked(loc, type, value)
    @ccall mlir_c.mlirFloatAttrDoubleGetChecked(loc::MlirLocation, type::MlirType, value::Cdouble)::MlirAttribute
end

"""
    mlirFloatAttrGetValueDouble(attr)

Returns the value stored in the given floating point attribute, interpreting the value as double.
"""
function mlirFloatAttrGetValueDouble(attr)
    @ccall mlir_c.mlirFloatAttrGetValueDouble(attr::MlirAttribute)::Cdouble
end

"""
    mlirAttributeIsAInteger(attr)

Checks whether the given attribute is an integer attribute.
"""
function mlirAttributeIsAInteger(attr)
    @ccall mlir_c.mlirAttributeIsAInteger(attr::MlirAttribute)::Bool
end

"""
    mlirIntegerAttrGet(type, value)

Creates an integer attribute of the given type with the given integer value.
"""
function mlirIntegerAttrGet(type, value)
    @ccall mlir_c.mlirIntegerAttrGet(type::MlirType, value::Int64)::MlirAttribute
end

"""
    mlirIntegerAttrGetValueInt(attr)

Returns the value stored in the given integer attribute, assuming the value is of signless type and fits into a signed 64-bit integer.
"""
function mlirIntegerAttrGetValueInt(attr)
    @ccall mlir_c.mlirIntegerAttrGetValueInt(attr::MlirAttribute)::Int64
end

"""
    mlirIntegerAttrGetValueSInt(attr)

Returns the value stored in the given integer attribute, assuming the value is of signed type and fits into a signed 64-bit integer.
"""
function mlirIntegerAttrGetValueSInt(attr)
    @ccall mlir_c.mlirIntegerAttrGetValueSInt(attr::MlirAttribute)::Int64
end

"""
    mlirIntegerAttrGetValueUInt(attr)

Returns the value stored in the given integer attribute, assuming the value is of unsigned type and fits into an unsigned 64-bit integer.
"""
function mlirIntegerAttrGetValueUInt(attr)
    @ccall mlir_c.mlirIntegerAttrGetValueUInt(attr::MlirAttribute)::UInt64
end

"""
    mlirAttributeIsABool(attr)

Checks whether the given attribute is a bool attribute.
"""
function mlirAttributeIsABool(attr)
    @ccall mlir_c.mlirAttributeIsABool(attr::MlirAttribute)::Bool
end

"""
    mlirBoolAttrGet(ctx, value)

Creates a bool attribute in the given context with the given value.
"""
function mlirBoolAttrGet(ctx, value)
    @ccall mlir_c.mlirBoolAttrGet(ctx::MlirContext, value::Cint)::MlirAttribute
end

"""
    mlirBoolAttrGetValue(attr)

Returns the value stored in the given bool attribute.
"""
function mlirBoolAttrGetValue(attr)
    @ccall mlir_c.mlirBoolAttrGetValue(attr::MlirAttribute)::Bool
end

"""
    mlirAttributeIsAIntegerSet(attr)

Checks whether the given attribute is an integer set attribute.
"""
function mlirAttributeIsAIntegerSet(attr)
    @ccall mlir_c.mlirAttributeIsAIntegerSet(attr::MlirAttribute)::Bool
end

"""
    mlirAttributeIsAOpaque(attr)

Checks whether the given attribute is an opaque attribute.
"""
function mlirAttributeIsAOpaque(attr)
    @ccall mlir_c.mlirAttributeIsAOpaque(attr::MlirAttribute)::Bool
end

"""
    mlirOpaqueAttrGet(ctx, dialectNamespace, dataLength, data, type)

Creates an opaque attribute in the given context associated with the dialect identified by its namespace. The attribute contains opaque byte data of the specified length (data need not be null-terminated).
"""
function mlirOpaqueAttrGet(ctx, dialectNamespace, dataLength, data, type)
    @ccall mlir_c.mlirOpaqueAttrGet(ctx::MlirContext, dialectNamespace::MlirStringRef, dataLength::intptr_t, data::Cstring, type::MlirType)::MlirAttribute
end

"""
    mlirOpaqueAttrGetDialectNamespace(attr)

Returns the namespace of the dialect with which the given opaque attribute is associated. The namespace string is owned by the context.
"""
function mlirOpaqueAttrGetDialectNamespace(attr)
    @ccall mlir_c.mlirOpaqueAttrGetDialectNamespace(attr::MlirAttribute)::MlirStringRef
end

"""
    mlirOpaqueAttrGetData(attr)

Returns the raw data as a string reference. The data remains live as long as the context in which the attribute lives.
"""
function mlirOpaqueAttrGetData(attr)
    @ccall mlir_c.mlirOpaqueAttrGetData(attr::MlirAttribute)::MlirStringRef
end

"""
    mlirAttributeIsAString(attr)

Checks whether the given attribute is a string attribute.
"""
function mlirAttributeIsAString(attr)
    @ccall mlir_c.mlirAttributeIsAString(attr::MlirAttribute)::Bool
end

"""
    mlirStringAttrGet(ctx, str)

Creates a string attribute in the given context containing the given string.
"""
function mlirStringAttrGet(ctx, str)
    @ccall mlir_c.mlirStringAttrGet(ctx::MlirContext, str::MlirStringRef)::MlirAttribute
end

"""
    mlirStringAttrTypedGet(type, str)

Creates a string attribute in the given context containing the given string. Additionally, the attribute has the given type.
"""
function mlirStringAttrTypedGet(type, str)
    @ccall mlir_c.mlirStringAttrTypedGet(type::MlirType, str::MlirStringRef)::MlirAttribute
end

"""
    mlirStringAttrGetValue(attr)

Returns the attribute values as a string reference. The data remains live as long as the context in which the attribute lives.
"""
function mlirStringAttrGetValue(attr)
    @ccall mlir_c.mlirStringAttrGetValue(attr::MlirAttribute)::MlirStringRef
end

"""
    mlirAttributeIsASymbolRef(attr)

Checks whether the given attribute is a symbol reference attribute.
"""
function mlirAttributeIsASymbolRef(attr)
    @ccall mlir_c.mlirAttributeIsASymbolRef(attr::MlirAttribute)::Bool
end

"""
    mlirSymbolRefAttrGet(ctx, symbol, numReferences, references)

Creates a symbol reference attribute in the given context referencing a symbol identified by the given string inside a list of nested references. Each of the references in the list must not be nested.
"""
function mlirSymbolRefAttrGet(ctx, symbol, numReferences, references)
    @ccall mlir_c.mlirSymbolRefAttrGet(ctx::MlirContext, symbol::MlirStringRef, numReferences::intptr_t, references::Ptr{MlirAttribute})::MlirAttribute
end

"""
    mlirSymbolRefAttrGetRootReference(attr)

Returns the string reference to the root referenced symbol. The data remains live as long as the context in which the attribute lives.
"""
function mlirSymbolRefAttrGetRootReference(attr)
    @ccall mlir_c.mlirSymbolRefAttrGetRootReference(attr::MlirAttribute)::MlirStringRef
end

"""
    mlirSymbolRefAttrGetLeafReference(attr)

Returns the string reference to the leaf referenced symbol. The data remains live as long as the context in which the attribute lives.
"""
function mlirSymbolRefAttrGetLeafReference(attr)
    @ccall mlir_c.mlirSymbolRefAttrGetLeafReference(attr::MlirAttribute)::MlirStringRef
end

"""
    mlirSymbolRefAttrGetNumNestedReferences(attr)

Returns the number of references nested in the given symbol reference attribute.
"""
function mlirSymbolRefAttrGetNumNestedReferences(attr)
    @ccall mlir_c.mlirSymbolRefAttrGetNumNestedReferences(attr::MlirAttribute)::intptr_t
end

"""
    mlirSymbolRefAttrGetNestedReference(attr, pos)

Returns pos-th reference nested in the given symbol reference attribute.
"""
function mlirSymbolRefAttrGetNestedReference(attr, pos)
    @ccall mlir_c.mlirSymbolRefAttrGetNestedReference(attr::MlirAttribute, pos::intptr_t)::MlirAttribute
end

"""
    mlirAttributeIsAFlatSymbolRef(attr)

Checks whether the given attribute is a flat symbol reference attribute.
"""
function mlirAttributeIsAFlatSymbolRef(attr)
    @ccall mlir_c.mlirAttributeIsAFlatSymbolRef(attr::MlirAttribute)::Bool
end

"""
    mlirFlatSymbolRefAttrGet(ctx, symbol)

Creates a flat symbol reference attribute in the given context referencing a symbol identified by the given string.
"""
function mlirFlatSymbolRefAttrGet(ctx, symbol)
    @ccall mlir_c.mlirFlatSymbolRefAttrGet(ctx::MlirContext, symbol::MlirStringRef)::MlirAttribute
end

"""
    mlirFlatSymbolRefAttrGetValue(attr)

Returns the referenced symbol as a string reference. The data remains live as long as the context in which the attribute lives.
"""
function mlirFlatSymbolRefAttrGetValue(attr)
    @ccall mlir_c.mlirFlatSymbolRefAttrGetValue(attr::MlirAttribute)::MlirStringRef
end

"""
    mlirAttributeIsAType(attr)

Checks whether the given attribute is a type attribute.
"""
function mlirAttributeIsAType(attr)
    @ccall mlir_c.mlirAttributeIsAType(attr::MlirAttribute)::Bool
end

"""
    mlirTypeAttrGet(type)

Creates a type attribute wrapping the given type in the same context as the type.
"""
function mlirTypeAttrGet(type)
    @ccall mlir_c.mlirTypeAttrGet(type::MlirType)::MlirAttribute
end

"""
    mlirTypeAttrGetValue(attr)

Returns the type stored in the given type attribute.
"""
function mlirTypeAttrGetValue(attr)
    @ccall mlir_c.mlirTypeAttrGetValue(attr::MlirAttribute)::MlirType
end

"""
    mlirAttributeIsAUnit(attr)

Checks whether the given attribute is a unit attribute.
"""
function mlirAttributeIsAUnit(attr)
    @ccall mlir_c.mlirAttributeIsAUnit(attr::MlirAttribute)::Bool
end

"""
    mlirUnitAttrGet(ctx)

Creates a unit attribute in the given context.
"""
function mlirUnitAttrGet(ctx)
    @ccall mlir_c.mlirUnitAttrGet(ctx::MlirContext)::MlirAttribute
end

"""
    mlirAttributeIsAElements(attr)

Checks whether the given attribute is an elements attribute.
"""
function mlirAttributeIsAElements(attr)
    @ccall mlir_c.mlirAttributeIsAElements(attr::MlirAttribute)::Bool
end

"""
    mlirElementsAttrGetValue(attr, rank, idxs)

Returns the element at the given rank-dimensional index.
"""
function mlirElementsAttrGetValue(attr, rank, idxs)
    @ccall mlir_c.mlirElementsAttrGetValue(attr::MlirAttribute, rank::intptr_t, idxs::Ptr{UInt64})::MlirAttribute
end

"""
    mlirElementsAttrIsValidIndex(attr, rank, idxs)

Checks whether the given rank-dimensional index is valid in the given elements attribute.
"""
function mlirElementsAttrIsValidIndex(attr, rank, idxs)
    @ccall mlir_c.mlirElementsAttrIsValidIndex(attr::MlirAttribute, rank::intptr_t, idxs::Ptr{UInt64})::Bool
end

"""
    mlirElementsAttrGetNumElements(attr)

Gets the total number of elements in the given elements attribute. In order to iterate over the attribute, obtain its type, which must be a statically shaped type and use its sizes to build a multi-dimensional index.
"""
function mlirElementsAttrGetNumElements(attr)
    @ccall mlir_c.mlirElementsAttrGetNumElements(attr::MlirAttribute)::Int64
end

"""
    mlirAttributeIsADenseElements(attr)

Checks whether the given attribute is a dense elements attribute.
"""
function mlirAttributeIsADenseElements(attr)
    @ccall mlir_c.mlirAttributeIsADenseElements(attr::MlirAttribute)::Bool
end

function mlirAttributeIsADenseIntElements(attr)
    @ccall mlir_c.mlirAttributeIsADenseIntElements(attr::MlirAttribute)::Bool
end

function mlirAttributeIsADenseFPElements(attr)
    @ccall mlir_c.mlirAttributeIsADenseFPElements(attr::MlirAttribute)::Bool
end

"""
    mlirDenseElementsAttrGet(shapedType, numElements, elements)

Creates a dense elements attribute with the given Shaped type and elements in the same context as the type.
"""
function mlirDenseElementsAttrGet(shapedType, numElements, elements)
    @ccall mlir_c.mlirDenseElementsAttrGet(shapedType::MlirType, numElements::intptr_t, elements::Ptr{MlirAttribute})::MlirAttribute
end

"""
    mlirDenseElementsAttrRawBufferGet(shapedType, rawBufferSize, rawBuffer)

Creates a dense elements attribute with the given Shaped type and elements populated from a packed, row-major opaque buffer of contents.

The format of the raw buffer is a densely packed array of values that can be bitcast to the storage format of the element type specified. Types that are not byte aligned will be: - For bitwidth > 1: Rounded up to the next byte. - For bitwidth = 1: Packed into 8bit bytes with bits corresponding to the linear order of the shape type from MSB to LSB, padded to on the right.

A raw buffer of a single element (or for 1-bit, a byte of value 0 or 255) will be interpreted as a splat. User code should be prepared for additional, conformant patterns to be identified as splats in the future.
"""
function mlirDenseElementsAttrRawBufferGet(shapedType, rawBufferSize, rawBuffer)
    @ccall mlir_c.mlirDenseElementsAttrRawBufferGet(shapedType::MlirType, rawBufferSize::Csize_t, rawBuffer::Ptr{Cvoid})::MlirAttribute
end

"""
    mlirDenseElementsAttrSplatGet(shapedType, element)

Creates a dense elements attribute with the given Shaped type containing a single replicated element (splat).
"""
function mlirDenseElementsAttrSplatGet(shapedType, element)
    @ccall mlir_c.mlirDenseElementsAttrSplatGet(shapedType::MlirType, element::MlirAttribute)::MlirAttribute
end

function mlirDenseElementsAttrBoolSplatGet(shapedType, element)
    @ccall mlir_c.mlirDenseElementsAttrBoolSplatGet(shapedType::MlirType, element::Bool)::MlirAttribute
end

function mlirDenseElementsAttrUInt8SplatGet(shapedType, element)
    @ccall mlir_c.mlirDenseElementsAttrUInt8SplatGet(shapedType::MlirType, element::UInt8)::MlirAttribute
end

function mlirDenseElementsAttrInt8SplatGet(shapedType, element)
    @ccall mlir_c.mlirDenseElementsAttrInt8SplatGet(shapedType::MlirType, element::Int8)::MlirAttribute
end

function mlirDenseElementsAttrUInt32SplatGet(shapedType, element)
    @ccall mlir_c.mlirDenseElementsAttrUInt32SplatGet(shapedType::MlirType, element::UInt32)::MlirAttribute
end

function mlirDenseElementsAttrInt32SplatGet(shapedType, element)
    @ccall mlir_c.mlirDenseElementsAttrInt32SplatGet(shapedType::MlirType, element::Int32)::MlirAttribute
end

function mlirDenseElementsAttrUInt64SplatGet(shapedType, element)
    @ccall mlir_c.mlirDenseElementsAttrUInt64SplatGet(shapedType::MlirType, element::UInt64)::MlirAttribute
end

function mlirDenseElementsAttrInt64SplatGet(shapedType, element)
    @ccall mlir_c.mlirDenseElementsAttrInt64SplatGet(shapedType::MlirType, element::Int64)::MlirAttribute
end

function mlirDenseElementsAttrFloatSplatGet(shapedType, element)
    @ccall mlir_c.mlirDenseElementsAttrFloatSplatGet(shapedType::MlirType, element::Cfloat)::MlirAttribute
end

function mlirDenseElementsAttrDoubleSplatGet(shapedType, element)
    @ccall mlir_c.mlirDenseElementsAttrDoubleSplatGet(shapedType::MlirType, element::Cdouble)::MlirAttribute
end

"""
    mlirDenseElementsAttrBoolGet(shapedType, numElements, elements)

Creates a dense elements attribute with the given shaped type from elements of a specific type. Expects the element type of the shaped type to match the data element type.
"""
function mlirDenseElementsAttrBoolGet(shapedType, numElements, elements)
    @ccall mlir_c.mlirDenseElementsAttrBoolGet(shapedType::MlirType, numElements::intptr_t, elements::Ptr{Cint})::MlirAttribute
end

function mlirDenseElementsAttrUInt8Get(shapedType, numElements, elements)
    @ccall mlir_c.mlirDenseElementsAttrUInt8Get(shapedType::MlirType, numElements::intptr_t, elements::Ptr{UInt8})::MlirAttribute
end

function mlirDenseElementsAttrInt8Get(shapedType, numElements, elements)
    @ccall mlir_c.mlirDenseElementsAttrInt8Get(shapedType::MlirType, numElements::intptr_t, elements::Ptr{Int8})::MlirAttribute
end

function mlirDenseElementsAttrUInt16Get(shapedType, numElements, elements)
    @ccall mlir_c.mlirDenseElementsAttrUInt16Get(shapedType::MlirType, numElements::intptr_t, elements::Ptr{UInt16})::MlirAttribute
end

function mlirDenseElementsAttrInt16Get(shapedType, numElements, elements)
    @ccall mlir_c.mlirDenseElementsAttrInt16Get(shapedType::MlirType, numElements::intptr_t, elements::Ptr{Int16})::MlirAttribute
end

function mlirDenseElementsAttrUInt32Get(shapedType, numElements, elements)
    @ccall mlir_c.mlirDenseElementsAttrUInt32Get(shapedType::MlirType, numElements::intptr_t, elements::Ptr{UInt32})::MlirAttribute
end

function mlirDenseElementsAttrInt32Get(shapedType, numElements, elements)
    @ccall mlir_c.mlirDenseElementsAttrInt32Get(shapedType::MlirType, numElements::intptr_t, elements::Ptr{Int32})::MlirAttribute
end

function mlirDenseElementsAttrUInt64Get(shapedType, numElements, elements)
    @ccall mlir_c.mlirDenseElementsAttrUInt64Get(shapedType::MlirType, numElements::intptr_t, elements::Ptr{UInt64})::MlirAttribute
end

function mlirDenseElementsAttrInt64Get(shapedType, numElements, elements)
    @ccall mlir_c.mlirDenseElementsAttrInt64Get(shapedType::MlirType, numElements::intptr_t, elements::Ptr{Int64})::MlirAttribute
end

function mlirDenseElementsAttrFloatGet(shapedType, numElements, elements)
    @ccall mlir_c.mlirDenseElementsAttrFloatGet(shapedType::MlirType, numElements::intptr_t, elements::Ptr{Cfloat})::MlirAttribute
end

function mlirDenseElementsAttrDoubleGet(shapedType, numElements, elements)
    @ccall mlir_c.mlirDenseElementsAttrDoubleGet(shapedType::MlirType, numElements::intptr_t, elements::Ptr{Cdouble})::MlirAttribute
end

function mlirDenseElementsAttrBFloat16Get(shapedType, numElements, elements)
    @ccall mlir_c.mlirDenseElementsAttrBFloat16Get(shapedType::MlirType, numElements::intptr_t, elements::Ptr{UInt16})::MlirAttribute
end

function mlirDenseElementsAttrFloat16Get(shapedType, numElements, elements)
    @ccall mlir_c.mlirDenseElementsAttrFloat16Get(shapedType::MlirType, numElements::intptr_t, elements::Ptr{UInt16})::MlirAttribute
end

"""
    mlirDenseElementsAttrStringGet(shapedType, numElements, strs)

Creates a dense elements attribute with the given shaped type from string elements.
"""
function mlirDenseElementsAttrStringGet(shapedType, numElements, strs)
    @ccall mlir_c.mlirDenseElementsAttrStringGet(shapedType::MlirType, numElements::intptr_t, strs::Ptr{MlirStringRef})::MlirAttribute
end

"""
    mlirDenseElementsAttrReshapeGet(attr, shapedType)

Creates a dense elements attribute that has the same data as the given dense elements attribute and a different shaped type. The new type must have the same total number of elements.
"""
function mlirDenseElementsAttrReshapeGet(attr, shapedType)
    @ccall mlir_c.mlirDenseElementsAttrReshapeGet(attr::MlirAttribute, shapedType::MlirType)::MlirAttribute
end

"""
    mlirDenseElementsAttrIsSplat(attr)

Checks whether the given dense elements attribute contains a single replicated value (splat).
"""
function mlirDenseElementsAttrIsSplat(attr)
    @ccall mlir_c.mlirDenseElementsAttrIsSplat(attr::MlirAttribute)::Bool
end

"""
    mlirDenseElementsAttrGetSplatValue(attr)

Returns the single replicated value (splat) of a specific type contained by the given dense elements attribute.
"""
function mlirDenseElementsAttrGetSplatValue(attr)
    @ccall mlir_c.mlirDenseElementsAttrGetSplatValue(attr::MlirAttribute)::MlirAttribute
end

function mlirDenseElementsAttrGetBoolSplatValue(attr)
    @ccall mlir_c.mlirDenseElementsAttrGetBoolSplatValue(attr::MlirAttribute)::Cint
end

function mlirDenseElementsAttrGetInt8SplatValue(attr)
    @ccall mlir_c.mlirDenseElementsAttrGetInt8SplatValue(attr::MlirAttribute)::Int8
end

function mlirDenseElementsAttrGetUInt8SplatValue(attr)
    @ccall mlir_c.mlirDenseElementsAttrGetUInt8SplatValue(attr::MlirAttribute)::UInt8
end

function mlirDenseElementsAttrGetInt32SplatValue(attr)
    @ccall mlir_c.mlirDenseElementsAttrGetInt32SplatValue(attr::MlirAttribute)::Int32
end

function mlirDenseElementsAttrGetUInt32SplatValue(attr)
    @ccall mlir_c.mlirDenseElementsAttrGetUInt32SplatValue(attr::MlirAttribute)::UInt32
end

function mlirDenseElementsAttrGetInt64SplatValue(attr)
    @ccall mlir_c.mlirDenseElementsAttrGetInt64SplatValue(attr::MlirAttribute)::Int64
end

function mlirDenseElementsAttrGetUInt64SplatValue(attr)
    @ccall mlir_c.mlirDenseElementsAttrGetUInt64SplatValue(attr::MlirAttribute)::UInt64
end

function mlirDenseElementsAttrGetFloatSplatValue(attr)
    @ccall mlir_c.mlirDenseElementsAttrGetFloatSplatValue(attr::MlirAttribute)::Cfloat
end

function mlirDenseElementsAttrGetDoubleSplatValue(attr)
    @ccall mlir_c.mlirDenseElementsAttrGetDoubleSplatValue(attr::MlirAttribute)::Cdouble
end

function mlirDenseElementsAttrGetStringSplatValue(attr)
    @ccall mlir_c.mlirDenseElementsAttrGetStringSplatValue(attr::MlirAttribute)::MlirStringRef
end

"""
    mlirDenseElementsAttrGetBoolValue(attr, pos)

Returns the pos-th value (flat contiguous indexing) of a specific type contained by the given dense elements attribute.
"""
function mlirDenseElementsAttrGetBoolValue(attr, pos)
    @ccall mlir_c.mlirDenseElementsAttrGetBoolValue(attr::MlirAttribute, pos::intptr_t)::Bool
end

function mlirDenseElementsAttrGetInt8Value(attr, pos)
    @ccall mlir_c.mlirDenseElementsAttrGetInt8Value(attr::MlirAttribute, pos::intptr_t)::Int8
end

function mlirDenseElementsAttrGetUInt8Value(attr, pos)
    @ccall mlir_c.mlirDenseElementsAttrGetUInt8Value(attr::MlirAttribute, pos::intptr_t)::UInt8
end

function mlirDenseElementsAttrGetInt16Value(attr, pos)
    @ccall mlir_c.mlirDenseElementsAttrGetInt16Value(attr::MlirAttribute, pos::intptr_t)::Int16
end

function mlirDenseElementsAttrGetUInt16Value(attr, pos)
    @ccall mlir_c.mlirDenseElementsAttrGetUInt16Value(attr::MlirAttribute, pos::intptr_t)::UInt16
end

function mlirDenseElementsAttrGetInt32Value(attr, pos)
    @ccall mlir_c.mlirDenseElementsAttrGetInt32Value(attr::MlirAttribute, pos::intptr_t)::Int32
end

function mlirDenseElementsAttrGetUInt32Value(attr, pos)
    @ccall mlir_c.mlirDenseElementsAttrGetUInt32Value(attr::MlirAttribute, pos::intptr_t)::UInt32
end

function mlirDenseElementsAttrGetInt64Value(attr, pos)
    @ccall mlir_c.mlirDenseElementsAttrGetInt64Value(attr::MlirAttribute, pos::intptr_t)::Int64
end

function mlirDenseElementsAttrGetUInt64Value(attr, pos)
    @ccall mlir_c.mlirDenseElementsAttrGetUInt64Value(attr::MlirAttribute, pos::intptr_t)::UInt64
end

function mlirDenseElementsAttrGetFloatValue(attr, pos)
    @ccall mlir_c.mlirDenseElementsAttrGetFloatValue(attr::MlirAttribute, pos::intptr_t)::Cfloat
end

function mlirDenseElementsAttrGetDoubleValue(attr, pos)
    @ccall mlir_c.mlirDenseElementsAttrGetDoubleValue(attr::MlirAttribute, pos::intptr_t)::Cdouble
end

function mlirDenseElementsAttrGetStringValue(attr, pos)
    @ccall mlir_c.mlirDenseElementsAttrGetStringValue(attr::MlirAttribute, pos::intptr_t)::MlirStringRef
end

"""
    mlirDenseElementsAttrGetRawData(attr)

Returns the raw data of the given dense elements attribute.
"""
function mlirDenseElementsAttrGetRawData(attr)
    @ccall mlir_c.mlirDenseElementsAttrGetRawData(attr::MlirAttribute)::Ptr{Cvoid}
end

"""
    mlirAttributeIsAOpaqueElements(attr)

Checks whether the given attribute is an opaque elements attribute.
"""
function mlirAttributeIsAOpaqueElements(attr)
    @ccall mlir_c.mlirAttributeIsAOpaqueElements(attr::MlirAttribute)::Bool
end

"""
    mlirAttributeIsASparseElements(attr)

Checks whether the given attribute is a sparse elements attribute.
"""
function mlirAttributeIsASparseElements(attr)
    @ccall mlir_c.mlirAttributeIsASparseElements(attr::MlirAttribute)::Bool
end

"""
    mlirSparseElementsAttribute(shapedType, denseIndices, denseValues)

Creates a sparse elements attribute of the given shape from a list of indices and a list of associated values. Both lists are expected to be dense elements attributes with the same number of elements. The list of indices is expected to contain 64-bit integers. The attribute is created in the same context as the type.
"""
function mlirSparseElementsAttribute(shapedType, denseIndices, denseValues)
    @ccall mlir_c.mlirSparseElementsAttribute(shapedType::MlirType, denseIndices::MlirAttribute, denseValues::MlirAttribute)::MlirAttribute
end

"""
    mlirSparseElementsAttrGetIndices(attr)

Returns the dense elements attribute containing 64-bit integer indices of non-null elements in the given sparse elements attribute.
"""
function mlirSparseElementsAttrGetIndices(attr)
    @ccall mlir_c.mlirSparseElementsAttrGetIndices(attr::MlirAttribute)::MlirAttribute
end

"""
    mlirSparseElementsAttrGetValues(attr)

Returns the dense elements attribute containing the non-null elements in the given sparse elements attribute.
"""
function mlirSparseElementsAttrGetValues(attr)
    @ccall mlir_c.mlirSparseElementsAttrGetValues(attr::MlirAttribute)::MlirAttribute
end

"""
    mlirTypeIsAInteger(type)

Checks whether the given type is an integer type.
"""
function mlirTypeIsAInteger(type)
    @ccall mlir_c.mlirTypeIsAInteger(type::MlirType)::Bool
end

"""
    mlirIntegerTypeGet(ctx, bitwidth)

Creates a signless integer type of the given bitwidth in the context. The type is owned by the context.
"""
function mlirIntegerTypeGet(ctx, bitwidth)
    @ccall mlir_c.mlirIntegerTypeGet(ctx::MlirContext, bitwidth::Cuint)::MlirType
end

"""
    mlirIntegerTypeSignedGet(ctx, bitwidth)

Creates a signed integer type of the given bitwidth in the context. The type is owned by the context.
"""
function mlirIntegerTypeSignedGet(ctx, bitwidth)
    @ccall mlir_c.mlirIntegerTypeSignedGet(ctx::MlirContext, bitwidth::Cuint)::MlirType
end

"""
    mlirIntegerTypeUnsignedGet(ctx, bitwidth)

Creates an unsigned integer type of the given bitwidth in the context. The type is owned by the context.
"""
function mlirIntegerTypeUnsignedGet(ctx, bitwidth)
    @ccall mlir_c.mlirIntegerTypeUnsignedGet(ctx::MlirContext, bitwidth::Cuint)::MlirType
end

"""
    mlirIntegerTypeGetWidth(type)

Returns the bitwidth of an integer type.
"""
function mlirIntegerTypeGetWidth(type)
    @ccall mlir_c.mlirIntegerTypeGetWidth(type::MlirType)::Cuint
end

"""
    mlirIntegerTypeIsSignless(type)

Checks whether the given integer type is signless.
"""
function mlirIntegerTypeIsSignless(type)
    @ccall mlir_c.mlirIntegerTypeIsSignless(type::MlirType)::Bool
end

"""
    mlirIntegerTypeIsSigned(type)

Checks whether the given integer type is signed.
"""
function mlirIntegerTypeIsSigned(type)
    @ccall mlir_c.mlirIntegerTypeIsSigned(type::MlirType)::Bool
end

"""
    mlirIntegerTypeIsUnsigned(type)

Checks whether the given integer type is unsigned.
"""
function mlirIntegerTypeIsUnsigned(type)
    @ccall mlir_c.mlirIntegerTypeIsUnsigned(type::MlirType)::Bool
end

"""
    mlirTypeIsAIndex(type)

Checks whether the given type is an index type.
"""
function mlirTypeIsAIndex(type)
    @ccall mlir_c.mlirTypeIsAIndex(type::MlirType)::Bool
end

"""
    mlirIndexTypeGet(ctx)

Creates an index type in the given context. The type is owned by the context.
"""
function mlirIndexTypeGet(ctx)
    @ccall mlir_c.mlirIndexTypeGet(ctx::MlirContext)::MlirType
end

"""
    mlirTypeIsABF16(type)

Checks whether the given type is a bf16 type.
"""
function mlirTypeIsABF16(type)
    @ccall mlir_c.mlirTypeIsABF16(type::MlirType)::Bool
end

"""
    mlirBF16TypeGet(ctx)

Creates a bf16 type in the given context. The type is owned by the context.
"""
function mlirBF16TypeGet(ctx)
    @ccall mlir_c.mlirBF16TypeGet(ctx::MlirContext)::MlirType
end

"""
    mlirTypeIsAF16(type)

Checks whether the given type is an f16 type.
"""
function mlirTypeIsAF16(type)
    @ccall mlir_c.mlirTypeIsAF16(type::MlirType)::Bool
end

"""
    mlirF16TypeGet(ctx)

Creates an f16 type in the given context. The type is owned by the context.
"""
function mlirF16TypeGet(ctx)
    @ccall mlir_c.mlirF16TypeGet(ctx::MlirContext)::MlirType
end

"""
    mlirTypeIsAF32(type)

Checks whether the given type is an f32 type.
"""
function mlirTypeIsAF32(type)
    @ccall mlir_c.mlirTypeIsAF32(type::MlirType)::Bool
end

"""
    mlirF32TypeGet(ctx)

Creates an f32 type in the given context. The type is owned by the context.
"""
function mlirF32TypeGet(ctx)
    @ccall mlir_c.mlirF32TypeGet(ctx::MlirContext)::MlirType
end

"""
    mlirTypeIsAF64(type)

Checks whether the given type is an f64 type.
"""
function mlirTypeIsAF64(type)
    @ccall mlir_c.mlirTypeIsAF64(type::MlirType)::Bool
end

"""
    mlirF64TypeGet(ctx)

Creates a f64 type in the given context. The type is owned by the context.
"""
function mlirF64TypeGet(ctx)
    @ccall mlir_c.mlirF64TypeGet(ctx::MlirContext)::MlirType
end

"""
    mlirTypeIsANone(type)

Checks whether the given type is a None type.
"""
function mlirTypeIsANone(type)
    @ccall mlir_c.mlirTypeIsANone(type::MlirType)::Bool
end

"""
    mlirNoneTypeGet(ctx)

Creates a None type in the given context. The type is owned by the context.
"""
function mlirNoneTypeGet(ctx)
    @ccall mlir_c.mlirNoneTypeGet(ctx::MlirContext)::MlirType
end

"""
    mlirTypeIsAComplex(type)

Checks whether the given type is a Complex type.
"""
function mlirTypeIsAComplex(type)
    @ccall mlir_c.mlirTypeIsAComplex(type::MlirType)::Bool
end

"""
    mlirComplexTypeGet(elementType)

Creates a complex type with the given element type in the same context as the element type. The type is owned by the context.
"""
function mlirComplexTypeGet(elementType)
    @ccall mlir_c.mlirComplexTypeGet(elementType::MlirType)::MlirType
end

"""
    mlirComplexTypeGetElementType(type)

Returns the element type of the given complex type.
"""
function mlirComplexTypeGetElementType(type)
    @ccall mlir_c.mlirComplexTypeGetElementType(type::MlirType)::MlirType
end

"""
    mlirTypeIsAShaped(type)

Checks whether the given type is a Shaped type.
"""
function mlirTypeIsAShaped(type)
    @ccall mlir_c.mlirTypeIsAShaped(type::MlirType)::Bool
end

"""
    mlirShapedTypeGetElementType(type)

Returns the element type of the shaped type.
"""
function mlirShapedTypeGetElementType(type)
    @ccall mlir_c.mlirShapedTypeGetElementType(type::MlirType)::MlirType
end

"""
    mlirShapedTypeHasRank(type)

Checks whether the given shaped type is ranked.
"""
function mlirShapedTypeHasRank(type)
    @ccall mlir_c.mlirShapedTypeHasRank(type::MlirType)::Bool
end

"""
    mlirShapedTypeGetRank(type)

Returns the rank of the given ranked shaped type.
"""
function mlirShapedTypeGetRank(type)
    @ccall mlir_c.mlirShapedTypeGetRank(type::MlirType)::Int64
end

"""
    mlirShapedTypeHasStaticShape(type)

Checks whether the given shaped type has a static shape.
"""
function mlirShapedTypeHasStaticShape(type)
    @ccall mlir_c.mlirShapedTypeHasStaticShape(type::MlirType)::Bool
end

"""
    mlirShapedTypeIsDynamicDim(type, dim)

Checks wither the dim-th dimension of the given shaped type is dynamic.
"""
function mlirShapedTypeIsDynamicDim(type, dim)
    @ccall mlir_c.mlirShapedTypeIsDynamicDim(type::MlirType, dim::intptr_t)::Bool
end

"""
    mlirShapedTypeGetDimSize(type, dim)

Returns the dim-th dimension of the given ranked shaped type.
"""
function mlirShapedTypeGetDimSize(type, dim)
    @ccall mlir_c.mlirShapedTypeGetDimSize(type::MlirType, dim::intptr_t)::Int64
end

"""
    mlirShapedTypeIsDynamicSize(size)

Checks whether the given value is used as a placeholder for dynamic sizes in shaped types.
"""
function mlirShapedTypeIsDynamicSize(size)
    @ccall mlir_c.mlirShapedTypeIsDynamicSize(size::Int64)::Bool
end

"""
    mlirShapedTypeGetDynamicSize()

Returns the value indicating a dynamic size in a shaped type. Prefer [`mlirShapedTypeIsDynamicSize`](@ref) to direct comparisons with this value.
"""
function mlirShapedTypeGetDynamicSize()
    @ccall mlir_c.mlirShapedTypeGetDynamicSize()::Int64
end

"""
    mlirShapedTypeIsDynamicStrideOrOffset(val)

Checks whether the given value is used as a placeholder for dynamic strides and offsets in shaped types.
"""
function mlirShapedTypeIsDynamicStrideOrOffset(val)
    @ccall mlir_c.mlirShapedTypeIsDynamicStrideOrOffset(val::Int64)::Bool
end

"""
    mlirShapedTypeGetDynamicStrideOrOffset()

Returns the value indicating a dynamic stride or offset in a shaped type. Prefer [`mlirShapedTypeGetDynamicStrideOrOffset`](@ref) to direct comparisons with this value.
"""
function mlirShapedTypeGetDynamicStrideOrOffset()
    @ccall mlir_c.mlirShapedTypeGetDynamicStrideOrOffset()::Int64
end

"""
    mlirTypeIsAVector(type)

Checks whether the given type is a Vector type.
"""
function mlirTypeIsAVector(type)
    @ccall mlir_c.mlirTypeIsAVector(type::MlirType)::Bool
end

"""
    mlirVectorTypeGet(rank, shape, elementType)

Creates a vector type of the shape identified by its rank and dimensions, with the given element type in the same context as the element type. The type is owned by the context.
"""
function mlirVectorTypeGet(rank, shape, elementType)
    @ccall mlir_c.mlirVectorTypeGet(rank::intptr_t, shape::Ptr{Int64}, elementType::MlirType)::MlirType
end

"""
    mlirVectorTypeGetChecked(loc, rank, shape, elementType)

Same as "[`mlirVectorTypeGet`](@ref)" but returns a nullptr wrapping [`MlirType`](@ref) on illegal arguments, emitting appropriate diagnostics.
"""
function mlirVectorTypeGetChecked(loc, rank, shape, elementType)
    @ccall mlir_c.mlirVectorTypeGetChecked(loc::MlirLocation, rank::intptr_t, shape::Ptr{Int64}, elementType::MlirType)::MlirType
end

"""
    mlirTypeIsATensor(type)

Checks whether the given type is a Tensor type.
"""
function mlirTypeIsATensor(type)
    @ccall mlir_c.mlirTypeIsATensor(type::MlirType)::Bool
end

"""
    mlirTypeIsARankedTensor(type)

Checks whether the given type is a ranked tensor type.
"""
function mlirTypeIsARankedTensor(type)
    @ccall mlir_c.mlirTypeIsARankedTensor(type::MlirType)::Bool
end

"""
    mlirTypeIsAUnrankedTensor(type)

Checks whether the given type is an unranked tensor type.
"""
function mlirTypeIsAUnrankedTensor(type)
    @ccall mlir_c.mlirTypeIsAUnrankedTensor(type::MlirType)::Bool
end

"""
    mlirRankedTensorTypeGet(rank, shape, elementType, encoding)

Creates a tensor type of a fixed rank with the given shape, element type, and optional encoding in the same context as the element type. The type is owned by the context. Tensor types without any specific encoding field should assign [`mlirAttributeGetNull`](@ref)() to this parameter.
"""
function mlirRankedTensorTypeGet(rank, shape, elementType, encoding)
    @ccall mlir_c.mlirRankedTensorTypeGet(rank::intptr_t, shape::Ptr{Int64}, elementType::MlirType, encoding::MlirAttribute)::MlirType
end

"""
    mlirRankedTensorTypeGetChecked(loc, rank, shape, elementType, encoding)

Same as "[`mlirRankedTensorTypeGet`](@ref)" but returns a nullptr wrapping [`MlirType`](@ref) on illegal arguments, emitting appropriate diagnostics.
"""
function mlirRankedTensorTypeGetChecked(loc, rank, shape, elementType, encoding)
    @ccall mlir_c.mlirRankedTensorTypeGetChecked(loc::MlirLocation, rank::intptr_t, shape::Ptr{Int64}, elementType::MlirType, encoding::MlirAttribute)::MlirType
end

"""
    mlirRankedTensorTypeGetEncoding(type)

Gets the 'encoding' attribute from the ranked tensor type, returning a null attribute if none.
"""
function mlirRankedTensorTypeGetEncoding(type)
    @ccall mlir_c.mlirRankedTensorTypeGetEncoding(type::MlirType)::MlirAttribute
end

"""
    mlirUnrankedTensorTypeGet(elementType)

Creates an unranked tensor type with the given element type in the same context as the element type. The type is owned by the context.
"""
function mlirUnrankedTensorTypeGet(elementType)
    @ccall mlir_c.mlirUnrankedTensorTypeGet(elementType::MlirType)::MlirType
end

"""
    mlirUnrankedTensorTypeGetChecked(loc, elementType)

Same as "[`mlirUnrankedTensorTypeGet`](@ref)" but returns a nullptr wrapping [`MlirType`](@ref) on illegal arguments, emitting appropriate diagnostics.
"""
function mlirUnrankedTensorTypeGetChecked(loc, elementType)
    @ccall mlir_c.mlirUnrankedTensorTypeGetChecked(loc::MlirLocation, elementType::MlirType)::MlirType
end

"""
    mlirTypeIsAMemRef(type)

Checks whether the given type is a MemRef type.
"""
function mlirTypeIsAMemRef(type)
    @ccall mlir_c.mlirTypeIsAMemRef(type::MlirType)::Bool
end

"""
    mlirTypeIsAUnrankedMemRef(type)

Checks whether the given type is an UnrankedMemRef type.
"""
function mlirTypeIsAUnrankedMemRef(type)
    @ccall mlir_c.mlirTypeIsAUnrankedMemRef(type::MlirType)::Bool
end

"""
    mlirMemRefTypeGet(elementType, rank, shape, layout, memorySpace)

Creates a MemRef type with the given rank and shape, a potentially empty list of affine layout maps, the given memory space and element type, in the same context as element type. The type is owned by the context.
"""
function mlirMemRefTypeGet(elementType, rank, shape, layout, memorySpace)
    @ccall mlir_c.mlirMemRefTypeGet(elementType::MlirType, rank::intptr_t, shape::Ptr{Int64}, layout::MlirAttribute, memorySpace::MlirAttribute)::MlirType
end

"""
    mlirMemRefTypeGetChecked(loc, elementType, rank, shape, layout, memorySpace)

Same as "[`mlirMemRefTypeGet`](@ref)" but returns a nullptr-wrapping [`MlirType`](@ref) o illegal arguments, emitting appropriate diagnostics.
"""
function mlirMemRefTypeGetChecked(loc, elementType, rank, shape, layout, memorySpace)
    @ccall mlir_c.mlirMemRefTypeGetChecked(loc::MlirLocation, elementType::MlirType, rank::intptr_t, shape::Ptr{Int64}, layout::MlirAttribute, memorySpace::MlirAttribute)::MlirType
end

"""
    mlirMemRefTypeContiguousGet(elementType, rank, shape, memorySpace)

Creates a MemRef type with the given rank, shape, memory space and element type in the same context as the element type. The type has no affine maps, i.e. represents a default row-major contiguous memref. The type is owned by the context.
"""
function mlirMemRefTypeContiguousGet(elementType, rank, shape, memorySpace)
    @ccall mlir_c.mlirMemRefTypeContiguousGet(elementType::MlirType, rank::intptr_t, shape::Ptr{Int64}, memorySpace::MlirAttribute)::MlirType
end

"""
    mlirMemRefTypeContiguousGetChecked(loc, elementType, rank, shape, memorySpace)

Same as "[`mlirMemRefTypeContiguousGet`](@ref)" but returns a nullptr wrapping [`MlirType`](@ref) on illegal arguments, emitting appropriate diagnostics.
"""
function mlirMemRefTypeContiguousGetChecked(loc, elementType, rank, shape, memorySpace)
    @ccall mlir_c.mlirMemRefTypeContiguousGetChecked(loc::MlirLocation, elementType::MlirType, rank::intptr_t, shape::Ptr{Int64}, memorySpace::MlirAttribute)::MlirType
end

"""
    mlirUnrankedMemRefTypeGet(elementType, memorySpace)

Creates an Unranked MemRef type with the given element type and in the given memory space. The type is owned by the context of element type.
"""
function mlirUnrankedMemRefTypeGet(elementType, memorySpace)
    @ccall mlir_c.mlirUnrankedMemRefTypeGet(elementType::MlirType, memorySpace::MlirAttribute)::MlirType
end

"""
    mlirUnrankedMemRefTypeGetChecked(loc, elementType, memorySpace)

Same as "[`mlirUnrankedMemRefTypeGet`](@ref)" but returns a nullptr wrapping [`MlirType`](@ref) on illegal arguments, emitting appropriate diagnostics.
"""
function mlirUnrankedMemRefTypeGetChecked(loc, elementType, memorySpace)
    @ccall mlir_c.mlirUnrankedMemRefTypeGetChecked(loc::MlirLocation, elementType::MlirType, memorySpace::MlirAttribute)::MlirType
end

"""
    mlirMemRefTypeGetLayout(type)

Returns the layout of the given MemRef type.
"""
function mlirMemRefTypeGetLayout(type)
    @ccall mlir_c.mlirMemRefTypeGetLayout(type::MlirType)::MlirAttribute
end

"""
    mlirMemRefTypeGetAffineMap(type)

Returns the affine map of the given MemRef type.
"""
function mlirMemRefTypeGetAffineMap(type)
    @ccall mlir_c.mlirMemRefTypeGetAffineMap(type::MlirType)::MlirAffineMap
end

"""
    mlirMemRefTypeGetMemorySpace(type)

Returns the memory space of the given MemRef type.
"""
function mlirMemRefTypeGetMemorySpace(type)
    @ccall mlir_c.mlirMemRefTypeGetMemorySpace(type::MlirType)::MlirAttribute
end

"""
    mlirUnrankedMemrefGetMemorySpace(type)

Returns the memory spcae of the given Unranked MemRef type.
"""
function mlirUnrankedMemrefGetMemorySpace(type)
    @ccall mlir_c.mlirUnrankedMemrefGetMemorySpace(type::MlirType)::MlirAttribute
end

"""
    mlirTypeIsATuple(type)

Checks whether the given type is a tuple type.
"""
function mlirTypeIsATuple(type)
    @ccall mlir_c.mlirTypeIsATuple(type::MlirType)::Bool
end

"""
    mlirTupleTypeGet(ctx, numElements, elements)

Creates a tuple type that consists of the given list of elemental types. The type is owned by the context.
"""
function mlirTupleTypeGet(ctx, numElements, elements)
    @ccall mlir_c.mlirTupleTypeGet(ctx::MlirContext, numElements::intptr_t, elements::Ptr{MlirType})::MlirType
end

"""
    mlirTupleTypeGetNumTypes(type)

Returns the number of types contained in a tuple.
"""
function mlirTupleTypeGetNumTypes(type)
    @ccall mlir_c.mlirTupleTypeGetNumTypes(type::MlirType)::intptr_t
end

"""
    mlirTupleTypeGetType(type, pos)

Returns the pos-th type in the tuple type.
"""
function mlirTupleTypeGetType(type, pos)
    @ccall mlir_c.mlirTupleTypeGetType(type::MlirType, pos::intptr_t)::MlirType
end

"""
    mlirTypeIsAFunction(type)

Checks whether the given type is a function type.
"""
function mlirTypeIsAFunction(type)
    @ccall mlir_c.mlirTypeIsAFunction(type::MlirType)::Bool
end

"""
    mlirFunctionTypeGet(ctx, numInputs, inputs, numResults, results)

Creates a function type, mapping a list of input types to result types.
"""
function mlirFunctionTypeGet(ctx, numInputs, inputs, numResults, results)
    @ccall mlir_c.mlirFunctionTypeGet(ctx::MlirContext, numInputs::intptr_t, inputs::Ptr{MlirType}, numResults::intptr_t, results::Ptr{MlirType})::MlirType
end

"""
    mlirFunctionTypeGetNumInputs(type)

Returns the number of input types.
"""
function mlirFunctionTypeGetNumInputs(type)
    @ccall mlir_c.mlirFunctionTypeGetNumInputs(type::MlirType)::intptr_t
end

"""
    mlirFunctionTypeGetNumResults(type)

Returns the number of result types.
"""
function mlirFunctionTypeGetNumResults(type)
    @ccall mlir_c.mlirFunctionTypeGetNumResults(type::MlirType)::intptr_t
end

"""
    mlirFunctionTypeGetInput(type, pos)

Returns the pos-th input type.
"""
function mlirFunctionTypeGetInput(type, pos)
    @ccall mlir_c.mlirFunctionTypeGetInput(type::MlirType, pos::intptr_t)::MlirType
end

"""
    mlirFunctionTypeGetResult(type, pos)

Returns the pos-th result type.
"""
function mlirFunctionTypeGetResult(type, pos)
    @ccall mlir_c.mlirFunctionTypeGetResult(type::MlirType, pos::intptr_t)::MlirType
end

"""
    mlirTypeIsAOpaque(type)

Checks whether the given type is an opaque type.
"""
function mlirTypeIsAOpaque(type)
    @ccall mlir_c.mlirTypeIsAOpaque(type::MlirType)::Bool
end

"""
    mlirOpaqueTypeGet(ctx, dialectNamespace, typeData)

Creates an opaque type in the given context associated with the dialect identified by its namespace. The type contains opaque byte data of the specified length (data need not be null-terminated).
"""
function mlirOpaqueTypeGet(ctx, dialectNamespace, typeData)
    @ccall mlir_c.mlirOpaqueTypeGet(ctx::MlirContext, dialectNamespace::MlirStringRef, typeData::MlirStringRef)::MlirType
end

"""
    mlirOpaqueTypeGetDialectNamespace(type)

Returns the namespace of the dialect with which the given opaque type is associated. The namespace string is owned by the context.
"""
function mlirOpaqueTypeGetDialectNamespace(type)
    @ccall mlir_c.mlirOpaqueTypeGetDialectNamespace(type::MlirType)::MlirStringRef
end

"""
    mlirOpaqueTypeGetData(type)

Returns the raw data as a string reference. The data remains live as long as the context in which the type lives.
"""
function mlirOpaqueTypeGetData(type)
    @ccall mlir_c.mlirOpaqueTypeGetData(type::MlirType)::MlirStringRef
end

struct MlirPass
    ptr::Ptr{Cvoid}
end

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
    mlirPassManagerCreate(ctx)

Create a new top-level PassManager.
"""
function mlirPassManagerCreate(ctx)
    @ccall mlir_c.mlirPassManagerCreate(ctx::MlirContext)::MlirPassManager
end

"""
    mlirPassManagerDestroy(passManager)

Destroy the provided PassManager.
"""
function mlirPassManagerDestroy(passManager)
    @ccall mlir_c.mlirPassManagerDestroy(passManager::MlirPassManager)::Cvoid
end

"""
    mlirPassManagerIsNull(passManager)

Checks if a PassManager is null.
"""
function mlirPassManagerIsNull(passManager)
    @ccall mlir_c.mlirPassManagerIsNull(passManager::MlirPassManager)::Bool
end

"""
    mlirPassManagerGetAsOpPassManager(passManager)

Cast a top-level PassManager to a generic OpPassManager.
"""
function mlirPassManagerGetAsOpPassManager(passManager)
    @ccall mlir_c.mlirPassManagerGetAsOpPassManager(passManager::MlirPassManager)::MlirOpPassManager
end

"""
    mlirPassManagerRun(passManager, _module)

Run the provided `passManager` on the given `module`.
"""
function mlirPassManagerRun(passManager, _module)
    @ccall mlir_c.mlirPassManagerRun(passManager::MlirPassManager, _module::MlirModule)::MlirLogicalResult
end

"""
    mlirPassManagerEnableIRPrinting(passManager)

Enable mlir-print-ir-after-all.
"""
function mlirPassManagerEnableIRPrinting(passManager)
    @ccall mlir_c.mlirPassManagerEnableIRPrinting(passManager::MlirPassManager)::Cvoid
end

"""
    mlirPassManagerEnableVerifier(passManager, enable)

Enable / disable verify-each.
"""
function mlirPassManagerEnableVerifier(passManager, enable)
    @ccall mlir_c.mlirPassManagerEnableVerifier(passManager::MlirPassManager, enable::Bool)::Cvoid
end

"""
    mlirPassManagerGetNestedUnder(passManager, operationName)

Nest an OpPassManager under the top-level PassManager, the nested passmanager will only run on operations matching the provided name. The returned OpPassManager will be destroyed when the parent is destroyed. To further nest more OpPassManager under the newly returned one, see `mlirOpPassManagerNest` below.
"""
function mlirPassManagerGetNestedUnder(passManager, operationName)
    @ccall mlir_c.mlirPassManagerGetNestedUnder(passManager::MlirPassManager, operationName::MlirStringRef)::MlirOpPassManager
end

"""
    mlirOpPassManagerGetNestedUnder(passManager, operationName)

Nest an OpPassManager under the provided OpPassManager, the nested passmanager will only run on operations matching the provided name. The returned OpPassManager will be destroyed when the parent is destroyed.
"""
function mlirOpPassManagerGetNestedUnder(passManager, operationName)
    @ccall mlir_c.mlirOpPassManagerGetNestedUnder(passManager::MlirOpPassManager, operationName::MlirStringRef)::MlirOpPassManager
end

"""
    mlirPassManagerAddOwnedPass(passManager, pass)

Add a pass and transfer ownership to the provided top-level mlirPassManager. If the pass is not a generic operation pass or a ModulePass, a new OpPassManager is implicitly nested under the provided PassManager.
"""
function mlirPassManagerAddOwnedPass(passManager, pass)
    @ccall mlir_c.mlirPassManagerAddOwnedPass(passManager::MlirPassManager, pass::MlirPass)::Cvoid
end

"""
    mlirOpPassManagerAddOwnedPass(passManager, pass)

Add a pass and transfer ownership to the provided mlirOpPassManager. If the pass is not a generic operation pass or matching the type of the provided PassManager, a new OpPassManager is implicitly nested under the provided PassManager.
"""
function mlirOpPassManagerAddOwnedPass(passManager, pass)
    @ccall mlir_c.mlirOpPassManagerAddOwnedPass(passManager::MlirOpPassManager, pass::MlirPass)::Cvoid
end

"""
    mlirPrintPassPipeline(passManager, callback, userData)

Print a textual MLIR pass pipeline by sending chunks of the string representation and forwarding `userData to `callback`. Note that the callback may be called several times with consecutive chunks of the string.
"""
function mlirPrintPassPipeline(passManager, callback, userData)
    @ccall mlir_c.mlirPrintPassPipeline(passManager::MlirOpPassManager, callback::MlirStringCallback, userData::Ptr{Cvoid})::Cvoid
end

"""
    mlirParsePassPipeline(passManager, pipeline)

Parse a textual MLIR pass pipeline and add it to the provided OpPassManager.
"""
function mlirParsePassPipeline(passManager, pipeline)
    @ccall mlir_c.mlirParsePassPipeline(passManager::MlirOpPassManager, pipeline::MlirStringRef)::MlirLogicalResult
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
    mlirCreateExternalPass(passID, name, argument, description, opName, nDependentDialects, dependentDialects, callbacks, userData)

Creates an external [`MlirPass`](@ref) that calls the supplied `callbacks` using the supplied `userData`. If `opName` is empty, the pass is a generic operation pass. Otherwise it is an operation pass specific to the specified pass name.
"""
function mlirCreateExternalPass(passID, name, argument, description, opName, nDependentDialects, dependentDialects, callbacks, userData)
    @ccall mlir_c.mlirCreateExternalPass(passID::MlirTypeID, name::MlirStringRef, argument::MlirStringRef, description::MlirStringRef, opName::MlirStringRef, nDependentDialects::intptr_t, dependentDialects::Ptr{MlirDialectHandle}, callbacks::MlirExternalPassCallbacks, userData::Ptr{Cvoid})::MlirPass
end

"""
    mlirExternalPassSignalFailure(pass)

This signals that the pass has failed. This is only valid to call during the `run` callback of [`MlirExternalPassCallbacks`](@ref). See Pass::signalPassFailure().
"""
function mlirExternalPassSignalFailure(pass)
    @ccall mlir_c.mlirExternalPassSignalFailure(pass::MlirExternalPass)::Cvoid
end

function mlirRegisterConversionPasses()
    @ccall mlir_c.mlirRegisterConversionPasses()::Cvoid
end

function mlirCreateConversionConvertAMDGPUToROCDL()
    @ccall mlir_c.mlirCreateConversionConvertAMDGPUToROCDL()::MlirPass
end

function mlirRegisterConversionConvertAMDGPUToROCDL()
    @ccall mlir_c.mlirRegisterConversionConvertAMDGPUToROCDL()::Cvoid
end

function mlirCreateConversionConvertAffineForToGPU()
    @ccall mlir_c.mlirCreateConversionConvertAffineForToGPU()::MlirPass
end

function mlirRegisterConversionConvertAffineForToGPU()
    @ccall mlir_c.mlirRegisterConversionConvertAffineForToGPU()::Cvoid
end

function mlirCreateConversionConvertAffineToStandard()
    @ccall mlir_c.mlirCreateConversionConvertAffineToStandard()::MlirPass
end

function mlirRegisterConversionConvertAffineToStandard()
    @ccall mlir_c.mlirRegisterConversionConvertAffineToStandard()::Cvoid
end

function mlirCreateConversionConvertArithmeticToLLVM()
    @ccall mlir_c.mlirCreateConversionConvertArithmeticToLLVM()::MlirPass
end

function mlirRegisterConversionConvertArithmeticToLLVM()
    @ccall mlir_c.mlirRegisterConversionConvertArithmeticToLLVM()::Cvoid
end

function mlirCreateConversionConvertArithmeticToSPIRV()
    @ccall mlir_c.mlirCreateConversionConvertArithmeticToSPIRV()::MlirPass
end

function mlirRegisterConversionConvertArithmeticToSPIRV()
    @ccall mlir_c.mlirRegisterConversionConvertArithmeticToSPIRV()::Cvoid
end

function mlirCreateConversionConvertArmNeon2dToIntr()
    @ccall mlir_c.mlirCreateConversionConvertArmNeon2dToIntr()::MlirPass
end

function mlirRegisterConversionConvertArmNeon2dToIntr()
    @ccall mlir_c.mlirRegisterConversionConvertArmNeon2dToIntr()::Cvoid
end

function mlirCreateConversionConvertAsyncToLLVM()
    @ccall mlir_c.mlirCreateConversionConvertAsyncToLLVM()::MlirPass
end

function mlirRegisterConversionConvertAsyncToLLVM()
    @ccall mlir_c.mlirRegisterConversionConvertAsyncToLLVM()::Cvoid
end

function mlirCreateConversionConvertBufferizationToMemRef()
    @ccall mlir_c.mlirCreateConversionConvertBufferizationToMemRef()::MlirPass
end

function mlirRegisterConversionConvertBufferizationToMemRef()
    @ccall mlir_c.mlirRegisterConversionConvertBufferizationToMemRef()::Cvoid
end

function mlirCreateConversionConvertComplexToLLVM()
    @ccall mlir_c.mlirCreateConversionConvertComplexToLLVM()::MlirPass
end

function mlirRegisterConversionConvertComplexToLLVM()
    @ccall mlir_c.mlirRegisterConversionConvertComplexToLLVM()::Cvoid
end

function mlirCreateConversionConvertComplexToLibm()
    @ccall mlir_c.mlirCreateConversionConvertComplexToLibm()::MlirPass
end

function mlirRegisterConversionConvertComplexToLibm()
    @ccall mlir_c.mlirRegisterConversionConvertComplexToLibm()::Cvoid
end

function mlirCreateConversionConvertComplexToStandard()
    @ccall mlir_c.mlirCreateConversionConvertComplexToStandard()::MlirPass
end

function mlirRegisterConversionConvertComplexToStandard()
    @ccall mlir_c.mlirRegisterConversionConvertComplexToStandard()::Cvoid
end

function mlirCreateConversionConvertControlFlowToLLVM()
    @ccall mlir_c.mlirCreateConversionConvertControlFlowToLLVM()::MlirPass
end

function mlirRegisterConversionConvertControlFlowToLLVM()
    @ccall mlir_c.mlirRegisterConversionConvertControlFlowToLLVM()::Cvoid
end

function mlirCreateConversionConvertControlFlowToSPIRV()
    @ccall mlir_c.mlirCreateConversionConvertControlFlowToSPIRV()::MlirPass
end

function mlirRegisterConversionConvertControlFlowToSPIRV()
    @ccall mlir_c.mlirRegisterConversionConvertControlFlowToSPIRV()::Cvoid
end

function mlirCreateConversionConvertFuncToLLVM()
    @ccall mlir_c.mlirCreateConversionConvertFuncToLLVM()::MlirPass
end

function mlirRegisterConversionConvertFuncToLLVM()
    @ccall mlir_c.mlirRegisterConversionConvertFuncToLLVM()::Cvoid
end

function mlirCreateConversionConvertFuncToSPIRV()
    @ccall mlir_c.mlirCreateConversionConvertFuncToSPIRV()::MlirPass
end

function mlirRegisterConversionConvertFuncToSPIRV()
    @ccall mlir_c.mlirRegisterConversionConvertFuncToSPIRV()::Cvoid
end

function mlirCreateConversionConvertGPUToSPIRV()
    @ccall mlir_c.mlirCreateConversionConvertGPUToSPIRV()::MlirPass
end

function mlirRegisterConversionConvertGPUToSPIRV()
    @ccall mlir_c.mlirRegisterConversionConvertGPUToSPIRV()::Cvoid
end

function mlirCreateConversionConvertGpuLaunchFuncToVulkanLaunchFunc()
    @ccall mlir_c.mlirCreateConversionConvertGpuLaunchFuncToVulkanLaunchFunc()::MlirPass
end

function mlirRegisterConversionConvertGpuLaunchFuncToVulkanLaunchFunc()
    @ccall mlir_c.mlirRegisterConversionConvertGpuLaunchFuncToVulkanLaunchFunc()::Cvoid
end

function mlirCreateConversionConvertGpuOpsToNVVMOps()
    @ccall mlir_c.mlirCreateConversionConvertGpuOpsToNVVMOps()::MlirPass
end

function mlirRegisterConversionConvertGpuOpsToNVVMOps()
    @ccall mlir_c.mlirRegisterConversionConvertGpuOpsToNVVMOps()::Cvoid
end

function mlirCreateConversionConvertGpuOpsToROCDLOps()
    @ccall mlir_c.mlirCreateConversionConvertGpuOpsToROCDLOps()::MlirPass
end

function mlirRegisterConversionConvertGpuOpsToROCDLOps()
    @ccall mlir_c.mlirRegisterConversionConvertGpuOpsToROCDLOps()::Cvoid
end

function mlirCreateConversionConvertLinalgToLLVM()
    @ccall mlir_c.mlirCreateConversionConvertLinalgToLLVM()::MlirPass
end

function mlirRegisterConversionConvertLinalgToLLVM()
    @ccall mlir_c.mlirRegisterConversionConvertLinalgToLLVM()::Cvoid
end

function mlirCreateConversionConvertLinalgToSPIRV()
    @ccall mlir_c.mlirCreateConversionConvertLinalgToSPIRV()::MlirPass
end

function mlirRegisterConversionConvertLinalgToSPIRV()
    @ccall mlir_c.mlirRegisterConversionConvertLinalgToSPIRV()::Cvoid
end

function mlirCreateConversionConvertLinalgToStandard()
    @ccall mlir_c.mlirCreateConversionConvertLinalgToStandard()::MlirPass
end

function mlirRegisterConversionConvertLinalgToStandard()
    @ccall mlir_c.mlirRegisterConversionConvertLinalgToStandard()::Cvoid
end

function mlirCreateConversionConvertMathToLLVM()
    @ccall mlir_c.mlirCreateConversionConvertMathToLLVM()::MlirPass
end

function mlirRegisterConversionConvertMathToLLVM()
    @ccall mlir_c.mlirRegisterConversionConvertMathToLLVM()::Cvoid
end

function mlirCreateConversionConvertMathToLibm()
    @ccall mlir_c.mlirCreateConversionConvertMathToLibm()::MlirPass
end

function mlirRegisterConversionConvertMathToLibm()
    @ccall mlir_c.mlirRegisterConversionConvertMathToLibm()::Cvoid
end

function mlirCreateConversionConvertMathToSPIRV()
    @ccall mlir_c.mlirCreateConversionConvertMathToSPIRV()::MlirPass
end

function mlirRegisterConversionConvertMathToSPIRV()
    @ccall mlir_c.mlirRegisterConversionConvertMathToSPIRV()::Cvoid
end

function mlirCreateConversionConvertMemRefToLLVM()
    @ccall mlir_c.mlirCreateConversionConvertMemRefToLLVM()::MlirPass
end

function mlirRegisterConversionConvertMemRefToLLVM()
    @ccall mlir_c.mlirRegisterConversionConvertMemRefToLLVM()::Cvoid
end

function mlirCreateConversionConvertMemRefToSPIRV()
    @ccall mlir_c.mlirCreateConversionConvertMemRefToSPIRV()::MlirPass
end

function mlirRegisterConversionConvertMemRefToSPIRV()
    @ccall mlir_c.mlirRegisterConversionConvertMemRefToSPIRV()::Cvoid
end

function mlirCreateConversionConvertNVGPUToNVVM()
    @ccall mlir_c.mlirCreateConversionConvertNVGPUToNVVM()::MlirPass
end

function mlirRegisterConversionConvertNVGPUToNVVM()
    @ccall mlir_c.mlirRegisterConversionConvertNVGPUToNVVM()::Cvoid
end

function mlirCreateConversionConvertOpenACCToLLVM()
    @ccall mlir_c.mlirCreateConversionConvertOpenACCToLLVM()::MlirPass
end

function mlirRegisterConversionConvertOpenACCToLLVM()
    @ccall mlir_c.mlirRegisterConversionConvertOpenACCToLLVM()::Cvoid
end

function mlirCreateConversionConvertOpenACCToSCF()
    @ccall mlir_c.mlirCreateConversionConvertOpenACCToSCF()::MlirPass
end

function mlirRegisterConversionConvertOpenACCToSCF()
    @ccall mlir_c.mlirRegisterConversionConvertOpenACCToSCF()::Cvoid
end

function mlirCreateConversionConvertOpenMPToLLVM()
    @ccall mlir_c.mlirCreateConversionConvertOpenMPToLLVM()::MlirPass
end

function mlirRegisterConversionConvertOpenMPToLLVM()
    @ccall mlir_c.mlirRegisterConversionConvertOpenMPToLLVM()::Cvoid
end

function mlirCreateConversionConvertPDLToPDLInterp()
    @ccall mlir_c.mlirCreateConversionConvertPDLToPDLInterp()::MlirPass
end

function mlirRegisterConversionConvertPDLToPDLInterp()
    @ccall mlir_c.mlirRegisterConversionConvertPDLToPDLInterp()::Cvoid
end

function mlirCreateConversionConvertParallelLoopToGpu()
    @ccall mlir_c.mlirCreateConversionConvertParallelLoopToGpu()::MlirPass
end

function mlirRegisterConversionConvertParallelLoopToGpu()
    @ccall mlir_c.mlirRegisterConversionConvertParallelLoopToGpu()::Cvoid
end

function mlirCreateConversionConvertSCFToOpenMP()
    @ccall mlir_c.mlirCreateConversionConvertSCFToOpenMP()::MlirPass
end

function mlirRegisterConversionConvertSCFToOpenMP()
    @ccall mlir_c.mlirRegisterConversionConvertSCFToOpenMP()::Cvoid
end

function mlirCreateConversionConvertSPIRVToLLVM()
    @ccall mlir_c.mlirCreateConversionConvertSPIRVToLLVM()::MlirPass
end

function mlirRegisterConversionConvertSPIRVToLLVM()
    @ccall mlir_c.mlirRegisterConversionConvertSPIRVToLLVM()::Cvoid
end

function mlirCreateConversionConvertShapeConstraints()
    @ccall mlir_c.mlirCreateConversionConvertShapeConstraints()::MlirPass
end

function mlirRegisterConversionConvertShapeConstraints()
    @ccall mlir_c.mlirRegisterConversionConvertShapeConstraints()::Cvoid
end

function mlirCreateConversionConvertShapeToStandard()
    @ccall mlir_c.mlirCreateConversionConvertShapeToStandard()::MlirPass
end

function mlirRegisterConversionConvertShapeToStandard()
    @ccall mlir_c.mlirRegisterConversionConvertShapeToStandard()::Cvoid
end

function mlirCreateConversionConvertTensorToLinalg()
    @ccall mlir_c.mlirCreateConversionConvertTensorToLinalg()::MlirPass
end

function mlirRegisterConversionConvertTensorToLinalg()
    @ccall mlir_c.mlirRegisterConversionConvertTensorToLinalg()::Cvoid
end

function mlirCreateConversionConvertTensorToSPIRV()
    @ccall mlir_c.mlirCreateConversionConvertTensorToSPIRV()::MlirPass
end

function mlirRegisterConversionConvertTensorToSPIRV()
    @ccall mlir_c.mlirRegisterConversionConvertTensorToSPIRV()::Cvoid
end

function mlirCreateConversionConvertVectorToGPU()
    @ccall mlir_c.mlirCreateConversionConvertVectorToGPU()::MlirPass
end

function mlirRegisterConversionConvertVectorToGPU()
    @ccall mlir_c.mlirRegisterConversionConvertVectorToGPU()::Cvoid
end

function mlirCreateConversionConvertVectorToLLVM()
    @ccall mlir_c.mlirCreateConversionConvertVectorToLLVM()::MlirPass
end

function mlirRegisterConversionConvertVectorToLLVM()
    @ccall mlir_c.mlirRegisterConversionConvertVectorToLLVM()::Cvoid
end

function mlirCreateConversionConvertVectorToSCF()
    @ccall mlir_c.mlirCreateConversionConvertVectorToSCF()::MlirPass
end

function mlirRegisterConversionConvertVectorToSCF()
    @ccall mlir_c.mlirRegisterConversionConvertVectorToSCF()::Cvoid
end

function mlirCreateConversionConvertVectorToSPIRV()
    @ccall mlir_c.mlirCreateConversionConvertVectorToSPIRV()::MlirPass
end

function mlirRegisterConversionConvertVectorToSPIRV()
    @ccall mlir_c.mlirRegisterConversionConvertVectorToSPIRV()::Cvoid
end

function mlirCreateConversionConvertVulkanLaunchFuncToVulkanCalls()
    @ccall mlir_c.mlirCreateConversionConvertVulkanLaunchFuncToVulkanCalls()::MlirPass
end

function mlirRegisterConversionConvertVulkanLaunchFuncToVulkanCalls()
    @ccall mlir_c.mlirRegisterConversionConvertVulkanLaunchFuncToVulkanCalls()::Cvoid
end

function mlirCreateConversionGpuToLLVMConversionPass()
    @ccall mlir_c.mlirCreateConversionGpuToLLVMConversionPass()::MlirPass
end

function mlirRegisterConversionGpuToLLVMConversionPass()
    @ccall mlir_c.mlirRegisterConversionGpuToLLVMConversionPass()::Cvoid
end

function mlirCreateConversionLowerHostCodeToLLVM()
    @ccall mlir_c.mlirCreateConversionLowerHostCodeToLLVM()::MlirPass
end

function mlirRegisterConversionLowerHostCodeToLLVM()
    @ccall mlir_c.mlirRegisterConversionLowerHostCodeToLLVM()::Cvoid
end

function mlirCreateConversionReconcileUnrealizedCasts()
    @ccall mlir_c.mlirCreateConversionReconcileUnrealizedCasts()::MlirPass
end

function mlirRegisterConversionReconcileUnrealizedCasts()
    @ccall mlir_c.mlirRegisterConversionReconcileUnrealizedCasts()::Cvoid
end

function mlirCreateConversionSCFToControlFlow()
    @ccall mlir_c.mlirCreateConversionSCFToControlFlow()::MlirPass
end

function mlirRegisterConversionSCFToControlFlow()
    @ccall mlir_c.mlirRegisterConversionSCFToControlFlow()::Cvoid
end

function mlirCreateConversionSCFToSPIRV()
    @ccall mlir_c.mlirCreateConversionSCFToSPIRV()::MlirPass
end

function mlirRegisterConversionSCFToSPIRV()
    @ccall mlir_c.mlirRegisterConversionSCFToSPIRV()::Cvoid
end

function mlirCreateConversionTosaToArith()
    @ccall mlir_c.mlirCreateConversionTosaToArith()::MlirPass
end

function mlirRegisterConversionTosaToArith()
    @ccall mlir_c.mlirRegisterConversionTosaToArith()::Cvoid
end

function mlirCreateConversionTosaToLinalg()
    @ccall mlir_c.mlirCreateConversionTosaToLinalg()::MlirPass
end

function mlirRegisterConversionTosaToLinalg()
    @ccall mlir_c.mlirRegisterConversionTosaToLinalg()::Cvoid
end

function mlirCreateConversionTosaToLinalgNamed()
    @ccall mlir_c.mlirCreateConversionTosaToLinalgNamed()::MlirPass
end

function mlirRegisterConversionTosaToLinalgNamed()
    @ccall mlir_c.mlirRegisterConversionTosaToLinalgNamed()::Cvoid
end

function mlirCreateConversionTosaToSCF()
    @ccall mlir_c.mlirCreateConversionTosaToSCF()::MlirPass
end

function mlirRegisterConversionTosaToSCF()
    @ccall mlir_c.mlirRegisterConversionTosaToSCF()::Cvoid
end

function mlirCreateConversionTosaToTensor()
    @ccall mlir_c.mlirCreateConversionTosaToTensor()::MlirPass
end

function mlirRegisterConversionTosaToTensor()
    @ccall mlir_c.mlirRegisterConversionTosaToTensor()::Cvoid
end

"""
    mlirEnableGlobalDebug(enable)

Sets the global debugging flag.
"""
function mlirEnableGlobalDebug(enable)
    @ccall mlir_c.mlirEnableGlobalDebug(enable::Bool)::Cvoid
end

"""
    mlirIsGlobalDebugEnabled()

Retuns `true` if the global debugging flag is set, false otherwise.
"""
function mlirIsGlobalDebugEnabled()
    @ccall mlir_c.mlirIsGlobalDebugEnabled()::Bool
end

"""
    MlirDiagnostic

An opaque reference to a diagnostic, always owned by the diagnostics engine (context). Must not be stored outside of the diagnostic handler.
"""
struct MlirDiagnostic
    ptr::Ptr{Cvoid}
end

"""
    MlirDiagnosticSeverity

Severity of a diagnostic.
"""
@cenum MlirDiagnosticSeverity::UInt32 begin
    MlirDiagnosticError = 0x0000000000000000
    MlirDiagnosticWarning = 0x0000000000000001
    MlirDiagnosticNote = 0x0000000000000002
    MlirDiagnosticRemark = 0x0000000000000003
end

"""
Opaque identifier of a diagnostic handler, useful to detach a handler.
"""
const MlirDiagnosticHandlerID = UInt64

# typedef MlirLogicalResult ( * MlirDiagnosticHandler ) ( MlirDiagnostic , void * userData )
"""
Diagnostic handler type. Accepts a reference to a diagnostic, which is only guaranteed to be live during the call. The handler is passed the `userData` that was provided when the handler was attached to a context. If the handler processed the diagnostic completely, it is expected to return success. Otherwise, it is expected to return failure to indicate that other handlers should attempt to process the diagnostic.
"""
const MlirDiagnosticHandler = Ptr{Cvoid}

"""
    mlirDiagnosticPrint(diagnostic, callback, userData)

Prints a diagnostic using the provided callback.
"""
function mlirDiagnosticPrint(diagnostic, callback, userData)
    @ccall mlir_c.mlirDiagnosticPrint(diagnostic::MlirDiagnostic, callback::MlirStringCallback, userData::Ptr{Cvoid})::Cvoid
end

"""
    mlirDiagnosticGetLocation(diagnostic)

Returns the location at which the diagnostic is reported.
"""
function mlirDiagnosticGetLocation(diagnostic)
    @ccall mlir_c.mlirDiagnosticGetLocation(diagnostic::MlirDiagnostic)::MlirLocation
end

"""
    mlirDiagnosticGetSeverity(diagnostic)

Returns the severity of the diagnostic.
"""
function mlirDiagnosticGetSeverity(diagnostic)
    @ccall mlir_c.mlirDiagnosticGetSeverity(diagnostic::MlirDiagnostic)::MlirDiagnosticSeverity
end

"""
    mlirDiagnosticGetNumNotes(diagnostic)

Returns the number of notes attached to the diagnostic.
"""
function mlirDiagnosticGetNumNotes(diagnostic)
    @ccall mlir_c.mlirDiagnosticGetNumNotes(diagnostic::MlirDiagnostic)::intptr_t
end

"""
    mlirDiagnosticGetNote(diagnostic, pos)

Returns `pos`-th note attached to the diagnostic. Expects `pos` to be a valid zero-based index into the list of notes.
"""
function mlirDiagnosticGetNote(diagnostic, pos)
    @ccall mlir_c.mlirDiagnosticGetNote(diagnostic::MlirDiagnostic, pos::intptr_t)::MlirDiagnostic
end

"""
    mlirContextAttachDiagnosticHandler(context, handler, userData, deleteUserData)

Attaches the diagnostic handler to the context. Handlers are invoked in the reverse order of attachment until one of them processes the diagnostic completely. When a handler is invoked it is passed the `userData` that was provided when it was attached. If non-NULL, `deleteUserData` is called once the system no longer needs to call the handler (for instance after the handler is detached or the context is destroyed). Returns an identifier that can be used to detach the handler.
"""
function mlirContextAttachDiagnosticHandler(context, handler, userData, deleteUserData)
    @ccall mlir_c.mlirContextAttachDiagnosticHandler(context::MlirContext, handler::MlirDiagnosticHandler, userData::Ptr{Cvoid}, deleteUserData::Ptr{Cvoid})::MlirDiagnosticHandlerID
end

"""
    mlirContextDetachDiagnosticHandler(context, id)

Detaches an attached diagnostic handler from the context given its identifier.
"""
function mlirContextDetachDiagnosticHandler(context, id)
    @ccall mlir_c.mlirContextDetachDiagnosticHandler(context::MlirContext, id::MlirDiagnosticHandlerID)::Cvoid
end

"""
    mlirEmitError(location, message)

Emits an error at the given location through the diagnostics engine. Used for testing purposes.
"""
function mlirEmitError(location, message)
    @ccall mlir_c.mlirEmitError(location::MlirLocation, message::Cstring)::Cvoid
end

function mlirGetDialectHandle__async__()
    @ccall mlir_c.mlirGetDialectHandle__async__()::MlirDialectHandle
end

function mlirRegisterAsyncPasses()
    @ccall mlir_c.mlirRegisterAsyncPasses()::Cvoid
end

function mlirCreateAsyncAsyncParallelFor()
    @ccall mlir_c.mlirCreateAsyncAsyncParallelFor()::MlirPass
end

function mlirRegisterAsyncAsyncParallelFor()
    @ccall mlir_c.mlirRegisterAsyncAsyncParallelFor()::Cvoid
end

function mlirCreateAsyncAsyncRuntimePolicyBasedRefCounting()
    @ccall mlir_c.mlirCreateAsyncAsyncRuntimePolicyBasedRefCounting()::MlirPass
end

function mlirRegisterAsyncAsyncRuntimePolicyBasedRefCounting()
    @ccall mlir_c.mlirRegisterAsyncAsyncRuntimePolicyBasedRefCounting()::Cvoid
end

function mlirCreateAsyncAsyncRuntimeRefCounting()
    @ccall mlir_c.mlirCreateAsyncAsyncRuntimeRefCounting()::MlirPass
end

function mlirRegisterAsyncAsyncRuntimeRefCounting()
    @ccall mlir_c.mlirRegisterAsyncAsyncRuntimeRefCounting()::Cvoid
end

function mlirCreateAsyncAsyncRuntimeRefCountingOpt()
    @ccall mlir_c.mlirCreateAsyncAsyncRuntimeRefCountingOpt()::MlirPass
end

function mlirRegisterAsyncAsyncRuntimeRefCountingOpt()
    @ccall mlir_c.mlirRegisterAsyncAsyncRuntimeRefCountingOpt()::Cvoid
end

function mlirCreateAsyncAsyncToAsyncRuntime()
    @ccall mlir_c.mlirCreateAsyncAsyncToAsyncRuntime()::MlirPass
end

function mlirRegisterAsyncAsyncToAsyncRuntime()
    @ccall mlir_c.mlirRegisterAsyncAsyncToAsyncRuntime()::Cvoid
end

function mlirGetDialectHandle__cf__()
    @ccall mlir_c.mlirGetDialectHandle__cf__()::MlirDialectHandle
end

function mlirGetDialectHandle__func__()
    @ccall mlir_c.mlirGetDialectHandle__func__()::MlirDialectHandle
end

function mlirGetDialectHandle__gpu__()
    @ccall mlir_c.mlirGetDialectHandle__gpu__()::MlirDialectHandle
end

function mlirRegisterGPUPasses()
    @ccall mlir_c.mlirRegisterGPUPasses()::Cvoid
end

function mlirCreateGPUGpuAsyncRegionPass()
    @ccall mlir_c.mlirCreateGPUGpuAsyncRegionPass()::MlirPass
end

function mlirRegisterGPUGpuAsyncRegionPass()
    @ccall mlir_c.mlirRegisterGPUGpuAsyncRegionPass()::Cvoid
end

function mlirCreateGPUGpuKernelOutlining()
    @ccall mlir_c.mlirCreateGPUGpuKernelOutlining()::MlirPass
end

function mlirRegisterGPUGpuKernelOutlining()
    @ccall mlir_c.mlirRegisterGPUGpuKernelOutlining()::Cvoid
end

function mlirCreateGPUGpuLaunchSinkIndexComputations()
    @ccall mlir_c.mlirCreateGPUGpuLaunchSinkIndexComputations()::MlirPass
end

function mlirRegisterGPUGpuLaunchSinkIndexComputations()
    @ccall mlir_c.mlirRegisterGPUGpuLaunchSinkIndexComputations()::Cvoid
end

function mlirCreateGPUGpuMapParallelLoopsPass()
    @ccall mlir_c.mlirCreateGPUGpuMapParallelLoopsPass()::MlirPass
end

function mlirRegisterGPUGpuMapParallelLoopsPass()
    @ccall mlir_c.mlirRegisterGPUGpuMapParallelLoopsPass()::Cvoid
end

function mlirGetDialectHandle__llvm__()
    @ccall mlir_c.mlirGetDialectHandle__llvm__()::MlirDialectHandle
end

"""
    mlirLLVMPointerTypeGet(pointee, addressSpace)

Creates an llvm.ptr type.
"""
function mlirLLVMPointerTypeGet(pointee, addressSpace)
    @ccall mlir_c.mlirLLVMPointerTypeGet(pointee::MlirType, addressSpace::Cuint)::MlirType
end

"""
    mlirLLVMVoidTypeGet(ctx)

Creates an llmv.void type.
"""
function mlirLLVMVoidTypeGet(ctx)
    @ccall mlir_c.mlirLLVMVoidTypeGet(ctx::MlirContext)::MlirType
end

"""
    mlirLLVMArrayTypeGet(elementType, numElements)

Creates an llvm.array type.
"""
function mlirLLVMArrayTypeGet(elementType, numElements)
    @ccall mlir_c.mlirLLVMArrayTypeGet(elementType::MlirType, numElements::Cuint)::MlirType
end

"""
    mlirLLVMFunctionTypeGet(resultType, nArgumentTypes, argumentTypes, isVarArg)

Creates an llvm.func type.
"""
function mlirLLVMFunctionTypeGet(resultType, nArgumentTypes, argumentTypes, isVarArg)
    @ccall mlir_c.mlirLLVMFunctionTypeGet(resultType::MlirType, nArgumentTypes::intptr_t, argumentTypes::Ptr{MlirType}, isVarArg::Bool)::MlirType
end

"""
    mlirLLVMStructTypeLiteralGet(ctx, nFieldTypes, fieldTypes, isPacked)

Creates an LLVM literal (unnamed) struct type.
"""
function mlirLLVMStructTypeLiteralGet(ctx, nFieldTypes, fieldTypes, isPacked)
    @ccall mlir_c.mlirLLVMStructTypeLiteralGet(ctx::MlirContext, nFieldTypes::intptr_t, fieldTypes::Ptr{MlirType}, isPacked::Bool)::MlirType
end

"""
    mlirLinalgFillBuiltinNamedOpRegion(mlirOp)

Apply the special region builder for the builtin named Linalg op. Assert that `mlirOp` is a builtin named Linalg op.
"""
function mlirLinalgFillBuiltinNamedOpRegion(mlirOp)
    @ccall mlir_c.mlirLinalgFillBuiltinNamedOpRegion(mlirOp::MlirOperation)::Cvoid
end

function mlirGetDialectHandle__linalg__()
    @ccall mlir_c.mlirGetDialectHandle__linalg__()::MlirDialectHandle
end

function mlirRegisterLinalgPasses()
    @ccall mlir_c.mlirRegisterLinalgPasses()::Cvoid
end

function mlirCreateLinalgConvertElementwiseToLinalg()
    @ccall mlir_c.mlirCreateLinalgConvertElementwiseToLinalg()::MlirPass
end

function mlirRegisterLinalgConvertElementwiseToLinalg()
    @ccall mlir_c.mlirRegisterLinalgConvertElementwiseToLinalg()::Cvoid
end

function mlirCreateLinalgLinalgBufferize()
    @ccall mlir_c.mlirCreateLinalgLinalgBufferize()::MlirPass
end

function mlirRegisterLinalgLinalgBufferize()
    @ccall mlir_c.mlirRegisterLinalgLinalgBufferize()::Cvoid
end

function mlirCreateLinalgLinalgDetensorize()
    @ccall mlir_c.mlirCreateLinalgLinalgDetensorize()::MlirPass
end

function mlirRegisterLinalgLinalgDetensorize()
    @ccall mlir_c.mlirRegisterLinalgLinalgDetensorize()::Cvoid
end

function mlirCreateLinalgLinalgElementwiseOpFusion()
    @ccall mlir_c.mlirCreateLinalgLinalgElementwiseOpFusion()::MlirPass
end

function mlirRegisterLinalgLinalgElementwiseOpFusion()
    @ccall mlir_c.mlirRegisterLinalgLinalgElementwiseOpFusion()::Cvoid
end

function mlirCreateLinalgLinalgFoldUnitExtentDims()
    @ccall mlir_c.mlirCreateLinalgLinalgFoldUnitExtentDims()::MlirPass
end

function mlirRegisterLinalgLinalgFoldUnitExtentDims()
    @ccall mlir_c.mlirRegisterLinalgLinalgFoldUnitExtentDims()::Cvoid
end

function mlirCreateLinalgLinalgGeneralization()
    @ccall mlir_c.mlirCreateLinalgLinalgGeneralization()::MlirPass
end

function mlirRegisterLinalgLinalgGeneralization()
    @ccall mlir_c.mlirRegisterLinalgLinalgGeneralization()::Cvoid
end

function mlirCreateLinalgLinalgInitTensorToAllocTensor()
    @ccall mlir_c.mlirCreateLinalgLinalgInitTensorToAllocTensor()::MlirPass
end

function mlirRegisterLinalgLinalgInitTensorToAllocTensor()
    @ccall mlir_c.mlirRegisterLinalgLinalgInitTensorToAllocTensor()::Cvoid
end

function mlirCreateLinalgLinalgInlineScalarOperands()
    @ccall mlir_c.mlirCreateLinalgLinalgInlineScalarOperands()::MlirPass
end

function mlirRegisterLinalgLinalgInlineScalarOperands()
    @ccall mlir_c.mlirRegisterLinalgLinalgInlineScalarOperands()::Cvoid
end

function mlirCreateLinalgLinalgLowerToAffineLoops()
    @ccall mlir_c.mlirCreateLinalgLinalgLowerToAffineLoops()::MlirPass
end

function mlirRegisterLinalgLinalgLowerToAffineLoops()
    @ccall mlir_c.mlirRegisterLinalgLinalgLowerToAffineLoops()::Cvoid
end

function mlirCreateLinalgLinalgLowerToLoops()
    @ccall mlir_c.mlirCreateLinalgLinalgLowerToLoops()::MlirPass
end

function mlirRegisterLinalgLinalgLowerToLoops()
    @ccall mlir_c.mlirRegisterLinalgLinalgLowerToLoops()::Cvoid
end

function mlirCreateLinalgLinalgLowerToParallelLoops()
    @ccall mlir_c.mlirCreateLinalgLinalgLowerToParallelLoops()::MlirPass
end

function mlirRegisterLinalgLinalgLowerToParallelLoops()
    @ccall mlir_c.mlirRegisterLinalgLinalgLowerToParallelLoops()::Cvoid
end

function mlirCreateLinalgLinalgNamedOpConversion()
    @ccall mlir_c.mlirCreateLinalgLinalgNamedOpConversion()::MlirPass
end

function mlirRegisterLinalgLinalgNamedOpConversion()
    @ccall mlir_c.mlirRegisterLinalgLinalgNamedOpConversion()::Cvoid
end

function mlirCreateLinalgLinalgStrategyDecomposePass()
    @ccall mlir_c.mlirCreateLinalgLinalgStrategyDecomposePass()::MlirPass
end

function mlirRegisterLinalgLinalgStrategyDecomposePass()
    @ccall mlir_c.mlirRegisterLinalgLinalgStrategyDecomposePass()::Cvoid
end

function mlirCreateLinalgLinalgStrategyEnablePass()
    @ccall mlir_c.mlirCreateLinalgLinalgStrategyEnablePass()::MlirPass
end

function mlirRegisterLinalgLinalgStrategyEnablePass()
    @ccall mlir_c.mlirRegisterLinalgLinalgStrategyEnablePass()::Cvoid
end

function mlirCreateLinalgLinalgStrategyGeneralizePass()
    @ccall mlir_c.mlirCreateLinalgLinalgStrategyGeneralizePass()::MlirPass
end

function mlirRegisterLinalgLinalgStrategyGeneralizePass()
    @ccall mlir_c.mlirRegisterLinalgLinalgStrategyGeneralizePass()::Cvoid
end

function mlirCreateLinalgLinalgStrategyInterchangePass()
    @ccall mlir_c.mlirCreateLinalgLinalgStrategyInterchangePass()::MlirPass
end

function mlirRegisterLinalgLinalgStrategyInterchangePass()
    @ccall mlir_c.mlirRegisterLinalgLinalgStrategyInterchangePass()::Cvoid
end

function mlirCreateLinalgLinalgStrategyLowerVectorsPass()
    @ccall mlir_c.mlirCreateLinalgLinalgStrategyLowerVectorsPass()::MlirPass
end

function mlirRegisterLinalgLinalgStrategyLowerVectorsPass()
    @ccall mlir_c.mlirRegisterLinalgLinalgStrategyLowerVectorsPass()::Cvoid
end

function mlirCreateLinalgLinalgStrategyPadPass()
    @ccall mlir_c.mlirCreateLinalgLinalgStrategyPadPass()::MlirPass
end

function mlirRegisterLinalgLinalgStrategyPadPass()
    @ccall mlir_c.mlirRegisterLinalgLinalgStrategyPadPass()::Cvoid
end

function mlirCreateLinalgLinalgStrategyPeelPass()
    @ccall mlir_c.mlirCreateLinalgLinalgStrategyPeelPass()::MlirPass
end

function mlirRegisterLinalgLinalgStrategyPeelPass()
    @ccall mlir_c.mlirRegisterLinalgLinalgStrategyPeelPass()::Cvoid
end

function mlirCreateLinalgLinalgStrategyRemoveMarkersPass()
    @ccall mlir_c.mlirCreateLinalgLinalgStrategyRemoveMarkersPass()::MlirPass
end

function mlirRegisterLinalgLinalgStrategyRemoveMarkersPass()
    @ccall mlir_c.mlirRegisterLinalgLinalgStrategyRemoveMarkersPass()::Cvoid
end

function mlirCreateLinalgLinalgStrategyTileAndFusePass()
    @ccall mlir_c.mlirCreateLinalgLinalgStrategyTileAndFusePass()::MlirPass
end

function mlirRegisterLinalgLinalgStrategyTileAndFusePass()
    @ccall mlir_c.mlirRegisterLinalgLinalgStrategyTileAndFusePass()::Cvoid
end

function mlirCreateLinalgLinalgStrategyTilePass()
    @ccall mlir_c.mlirCreateLinalgLinalgStrategyTilePass()::MlirPass
end

function mlirRegisterLinalgLinalgStrategyTilePass()
    @ccall mlir_c.mlirRegisterLinalgLinalgStrategyTilePass()::Cvoid
end

function mlirCreateLinalgLinalgStrategyVectorizePass()
    @ccall mlir_c.mlirCreateLinalgLinalgStrategyVectorizePass()::MlirPass
end

function mlirRegisterLinalgLinalgStrategyVectorizePass()
    @ccall mlir_c.mlirRegisterLinalgLinalgStrategyVectorizePass()::Cvoid
end

function mlirCreateLinalgLinalgTiling()
    @ccall mlir_c.mlirCreateLinalgLinalgTiling()::MlirPass
end

function mlirRegisterLinalgLinalgTiling()
    @ccall mlir_c.mlirRegisterLinalgLinalgTiling()::Cvoid
end

function mlirGetDialectHandle__pdl__()
    @ccall mlir_c.mlirGetDialectHandle__pdl__()::MlirDialectHandle
end

function mlirTypeIsAPDLType(type)
    @ccall mlir_c.mlirTypeIsAPDLType(type::MlirType)::Bool
end

function mlirTypeIsAPDLAttributeType(type)
    @ccall mlir_c.mlirTypeIsAPDLAttributeType(type::MlirType)::Bool
end

function mlirPDLAttributeTypeGet(ctx)
    @ccall mlir_c.mlirPDLAttributeTypeGet(ctx::MlirContext)::MlirType
end

function mlirTypeIsAPDLOperationType(type)
    @ccall mlir_c.mlirTypeIsAPDLOperationType(type::MlirType)::Bool
end

function mlirPDLOperationTypeGet(ctx)
    @ccall mlir_c.mlirPDLOperationTypeGet(ctx::MlirContext)::MlirType
end

function mlirTypeIsAPDLRangeType(type)
    @ccall mlir_c.mlirTypeIsAPDLRangeType(type::MlirType)::Bool
end

function mlirPDLRangeTypeGet(elementType)
    @ccall mlir_c.mlirPDLRangeTypeGet(elementType::MlirType)::MlirType
end

function mlirPDLRangeTypeGetElementType(type)
    @ccall mlir_c.mlirPDLRangeTypeGetElementType(type::MlirType)::MlirType
end

function mlirTypeIsAPDLTypeType(type)
    @ccall mlir_c.mlirTypeIsAPDLTypeType(type::MlirType)::Bool
end

function mlirPDLTypeTypeGet(ctx)
    @ccall mlir_c.mlirPDLTypeTypeGet(ctx::MlirContext)::MlirType
end

function mlirTypeIsAPDLValueType(type)
    @ccall mlir_c.mlirTypeIsAPDLValueType(type::MlirType)::Bool
end

function mlirPDLValueTypeGet(ctx)
    @ccall mlir_c.mlirPDLValueTypeGet(ctx::MlirContext)::MlirType
end

function mlirGetDialectHandle__quant__()
    @ccall mlir_c.mlirGetDialectHandle__quant__()::MlirDialectHandle
end

"""
    mlirTypeIsAQuantizedType(type)

Returns `true` if the given type is a quantization dialect type.
"""
function mlirTypeIsAQuantizedType(type)
    @ccall mlir_c.mlirTypeIsAQuantizedType(type::MlirType)::Bool
end

"""
    mlirQuantizedTypeGetSignedFlag()

Returns the bit flag used to indicate signedness of a quantized type.
"""
function mlirQuantizedTypeGetSignedFlag()
    @ccall mlir_c.mlirQuantizedTypeGetSignedFlag()::Cuint
end

"""
    mlirQuantizedTypeGetDefaultMinimumForInteger(isSigned, integralWidth)

Returns the minimum possible value stored by a quantized type.
"""
function mlirQuantizedTypeGetDefaultMinimumForInteger(isSigned, integralWidth)
    @ccall mlir_c.mlirQuantizedTypeGetDefaultMinimumForInteger(isSigned::Bool, integralWidth::Cuint)::Int64
end

"""
    mlirQuantizedTypeGetDefaultMaximumForInteger(isSigned, integralWidth)

Returns the maximum possible value stored by a quantized type.
"""
function mlirQuantizedTypeGetDefaultMaximumForInteger(isSigned, integralWidth)
    @ccall mlir_c.mlirQuantizedTypeGetDefaultMaximumForInteger(isSigned::Bool, integralWidth::Cuint)::Int64
end

"""
    mlirQuantizedTypeGetExpressedType(type)

Gets the original type approximated by the given quantized type.
"""
function mlirQuantizedTypeGetExpressedType(type)
    @ccall mlir_c.mlirQuantizedTypeGetExpressedType(type::MlirType)::MlirType
end

"""
    mlirQuantizedTypeGetFlags(type)

Gets the flags associated with the given quantized type.
"""
function mlirQuantizedTypeGetFlags(type)
    @ccall mlir_c.mlirQuantizedTypeGetFlags(type::MlirType)::Cuint
end

"""
    mlirQuantizedTypeIsSigned(type)

Returns `true` if the given type is signed, `false` otherwise.
"""
function mlirQuantizedTypeIsSigned(type)
    @ccall mlir_c.mlirQuantizedTypeIsSigned(type::MlirType)::Bool
end

"""
    mlirQuantizedTypeGetStorageType(type)

Returns the underlying type used to store the values.
"""
function mlirQuantizedTypeGetStorageType(type)
    @ccall mlir_c.mlirQuantizedTypeGetStorageType(type::MlirType)::MlirType
end

"""
    mlirQuantizedTypeGetStorageTypeMin(type)

Returns the minimum value that the storage type of the given quantized type can take.
"""
function mlirQuantizedTypeGetStorageTypeMin(type)
    @ccall mlir_c.mlirQuantizedTypeGetStorageTypeMin(type::MlirType)::Int64
end

"""
    mlirQuantizedTypeGetStorageTypeMax(type)

Returns the maximum value that the storage type of the given quantized type can take.
"""
function mlirQuantizedTypeGetStorageTypeMax(type)
    @ccall mlir_c.mlirQuantizedTypeGetStorageTypeMax(type::MlirType)::Int64
end

"""
    mlirQuantizedTypeGetStorageTypeIntegralWidth(type)

Returns the integral bitwidth that the storage type of the given quantized type can represent exactly.
"""
function mlirQuantizedTypeGetStorageTypeIntegralWidth(type)
    @ccall mlir_c.mlirQuantizedTypeGetStorageTypeIntegralWidth(type::MlirType)::Cuint
end

"""
    mlirQuantizedTypeIsCompatibleExpressedType(type, candidate)

Returns `true` if the `candidate` type is compatible with the given quantized `type`.
"""
function mlirQuantizedTypeIsCompatibleExpressedType(type, candidate)
    @ccall mlir_c.mlirQuantizedTypeIsCompatibleExpressedType(type::MlirType, candidate::MlirType)::Bool
end

"""
    mlirQuantizedTypeGetQuantizedElementType(type)

Returns the element type of the given quantized type as another quantized type.
"""
function mlirQuantizedTypeGetQuantizedElementType(type)
    @ccall mlir_c.mlirQuantizedTypeGetQuantizedElementType(type::MlirType)::MlirType
end

"""
    mlirQuantizedTypeCastFromStorageType(type, candidate)

Casts from a type based on the storage type of the given type to a corresponding type based on the given type. Returns a null type if the cast is not valid.
"""
function mlirQuantizedTypeCastFromStorageType(type, candidate)
    @ccall mlir_c.mlirQuantizedTypeCastFromStorageType(type::MlirType, candidate::MlirType)::MlirType
end

"""
    mlirQuantizedTypeCastToStorageType(type)

Casts from a type based on a quantized type to a corresponding typed based on the storage type. Returns a null type if the cast is not valid.
"""
function mlirQuantizedTypeCastToStorageType(type)
    @ccall mlir_c.mlirQuantizedTypeCastToStorageType(type::MlirType)::MlirType
end

"""
    mlirQuantizedTypeCastFromExpressedType(type, candidate)

Casts from a type based on the expressed type of the given type to a corresponding type based on the given type. Returns a null type if the cast is not valid.
"""
function mlirQuantizedTypeCastFromExpressedType(type, candidate)
    @ccall mlir_c.mlirQuantizedTypeCastFromExpressedType(type::MlirType, candidate::MlirType)::MlirType
end

"""
    mlirQuantizedTypeCastToExpressedType(type)

Casts from a type based on a quantized type to a corresponding typed based on the expressed type. Returns a null type if the cast is not valid.
"""
function mlirQuantizedTypeCastToExpressedType(type)
    @ccall mlir_c.mlirQuantizedTypeCastToExpressedType(type::MlirType)::MlirType
end

"""
    mlirQuantizedTypeCastExpressedToStorageType(type, candidate)

Casts from a type based on the expressed type of the given quantized type to equivalent type based on storage type of the same quantized type.
"""
function mlirQuantizedTypeCastExpressedToStorageType(type, candidate)
    @ccall mlir_c.mlirQuantizedTypeCastExpressedToStorageType(type::MlirType, candidate::MlirType)::MlirType
end

"""
    mlirTypeIsAAnyQuantizedType(type)

Returns `true` if the given type is an AnyQuantizedType.
"""
function mlirTypeIsAAnyQuantizedType(type)
    @ccall mlir_c.mlirTypeIsAAnyQuantizedType(type::MlirType)::Bool
end

"""
    mlirAnyQuantizedTypeGet(flags, storageType, expressedType, storageTypeMin, storageTypeMax)

Creates an instance of AnyQuantizedType with the given parameters in the same context as `storageType` and returns it. The instance is owned by the context.
"""
function mlirAnyQuantizedTypeGet(flags, storageType, expressedType, storageTypeMin, storageTypeMax)
    @ccall mlir_c.mlirAnyQuantizedTypeGet(flags::Cuint, storageType::MlirType, expressedType::MlirType, storageTypeMin::Int64, storageTypeMax::Int64)::MlirType
end

"""
    mlirTypeIsAUniformQuantizedType(type)

Returns `true` if the given type is a UniformQuantizedType.
"""
function mlirTypeIsAUniformQuantizedType(type)
    @ccall mlir_c.mlirTypeIsAUniformQuantizedType(type::MlirType)::Bool
end

"""
    mlirUniformQuantizedTypeGet(flags, storageType, expressedType, scale, zeroPoint, storageTypeMin, storageTypeMax)

Creates an instance of UniformQuantizedType with the given parameters in the same context as `storageType` and returns it. The instance is owned by the context.
"""
function mlirUniformQuantizedTypeGet(flags, storageType, expressedType, scale, zeroPoint, storageTypeMin, storageTypeMax)
    @ccall mlir_c.mlirUniformQuantizedTypeGet(flags::Cuint, storageType::MlirType, expressedType::MlirType, scale::Cdouble, zeroPoint::Int64, storageTypeMin::Int64, storageTypeMax::Int64)::MlirType
end

"""
    mlirUniformQuantizedTypeGetScale(type)

Returns the scale of the given uniform quantized type.
"""
function mlirUniformQuantizedTypeGetScale(type)
    @ccall mlir_c.mlirUniformQuantizedTypeGetScale(type::MlirType)::Cdouble
end

"""
    mlirUniformQuantizedTypeGetZeroPoint(type)

Returns the zero point of the given uniform quantized type.
"""
function mlirUniformQuantizedTypeGetZeroPoint(type)
    @ccall mlir_c.mlirUniformQuantizedTypeGetZeroPoint(type::MlirType)::Int64
end

"""
    mlirUniformQuantizedTypeIsFixedPoint(type)

Returns `true` if the given uniform quantized type is fixed-point.
"""
function mlirUniformQuantizedTypeIsFixedPoint(type)
    @ccall mlir_c.mlirUniformQuantizedTypeIsFixedPoint(type::MlirType)::Bool
end

"""
    mlirTypeIsAUniformQuantizedPerAxisType(type)

Returns `true` if the given type is a UniformQuantizedPerAxisType.
"""
function mlirTypeIsAUniformQuantizedPerAxisType(type)
    @ccall mlir_c.mlirTypeIsAUniformQuantizedPerAxisType(type::MlirType)::Bool
end

"""
    mlirUniformQuantizedPerAxisTypeGet(flags, storageType, expressedType, nDims, scales, zeroPoints, quantizedDimension, storageTypeMin, storageTypeMax)

Creates an instance of UniformQuantizedPerAxisType with the given parameters in the same context as `storageType` and returns it. `scales` and `zeroPoints` point to `nDims` number of elements. The instance is owned by the context.
"""
function mlirUniformQuantizedPerAxisTypeGet(flags, storageType, expressedType, nDims, scales, zeroPoints, quantizedDimension, storageTypeMin, storageTypeMax)
    @ccall mlir_c.mlirUniformQuantizedPerAxisTypeGet(flags::Cuint, storageType::MlirType, expressedType::MlirType, nDims::intptr_t, scales::Ptr{Cdouble}, zeroPoints::Ptr{Int64}, quantizedDimension::Int32, storageTypeMin::Int64, storageTypeMax::Int64)::MlirType
end

"""
    mlirUniformQuantizedPerAxisTypeGetNumDims(type)

Returns the number of axes in the given quantized per-axis type.
"""
function mlirUniformQuantizedPerAxisTypeGetNumDims(type)
    @ccall mlir_c.mlirUniformQuantizedPerAxisTypeGetNumDims(type::MlirType)::intptr_t
end

"""
    mlirUniformQuantizedPerAxisTypeGetScale(type, pos)

Returns `pos`-th scale of the given quantized per-axis type.
"""
function mlirUniformQuantizedPerAxisTypeGetScale(type, pos)
    @ccall mlir_c.mlirUniformQuantizedPerAxisTypeGetScale(type::MlirType, pos::intptr_t)::Cdouble
end

"""
    mlirUniformQuantizedPerAxisTypeGetZeroPoint(type, pos)

Returns `pos`-th zero point of the given quantized per-axis type.
"""
function mlirUniformQuantizedPerAxisTypeGetZeroPoint(type, pos)
    @ccall mlir_c.mlirUniformQuantizedPerAxisTypeGetZeroPoint(type::MlirType, pos::intptr_t)::Int64
end

"""
    mlirUniformQuantizedPerAxisTypeGetQuantizedDimension(type)

Returns the index of the quantized dimension in the given quantized per-axis type.
"""
function mlirUniformQuantizedPerAxisTypeGetQuantizedDimension(type)
    @ccall mlir_c.mlirUniformQuantizedPerAxisTypeGetQuantizedDimension(type::MlirType)::Int32
end

"""
    mlirUniformQuantizedPerAxisTypeIsFixedPoint(type)

Returns `true` if the given uniform quantized per-axis type is fixed-point.
"""
function mlirUniformQuantizedPerAxisTypeIsFixedPoint(type)
    @ccall mlir_c.mlirUniformQuantizedPerAxisTypeIsFixedPoint(type::MlirType)::Bool
end

"""
    mlirTypeIsACalibratedQuantizedType(type)

Returns `true` if the given type is a CalibratedQuantizedType.
"""
function mlirTypeIsACalibratedQuantizedType(type)
    @ccall mlir_c.mlirTypeIsACalibratedQuantizedType(type::MlirType)::Bool
end

"""
    mlirCalibratedQuantizedTypeGet(expressedType, min, max)

Creates an instance of CalibratedQuantizedType with the given parameters in the same context as `expressedType` and returns it. The instance is owned by the context.
"""
function mlirCalibratedQuantizedTypeGet(expressedType, min, max)
    @ccall mlir_c.mlirCalibratedQuantizedTypeGet(expressedType::MlirType, min::Cdouble, max::Cdouble)::MlirType
end

"""
    mlirCalibratedQuantizedTypeGetMin(type)

Returns the min value of the given calibrated quantized type.
"""
function mlirCalibratedQuantizedTypeGetMin(type)
    @ccall mlir_c.mlirCalibratedQuantizedTypeGetMin(type::MlirType)::Cdouble
end

"""
    mlirCalibratedQuantizedTypeGetMax(type)

Returns the max value of the given calibrated quantized type.
"""
function mlirCalibratedQuantizedTypeGetMax(type)
    @ccall mlir_c.mlirCalibratedQuantizedTypeGetMax(type::MlirType)::Cdouble
end

function mlirGetDialectHandle__scf__()
    @ccall mlir_c.mlirGetDialectHandle__scf__()::MlirDialectHandle
end

function mlirGetDialectHandle__shape__()
    @ccall mlir_c.mlirGetDialectHandle__shape__()::MlirDialectHandle
end

function mlirGetDialectHandle__sparse_tensor__()
    @ccall mlir_c.mlirGetDialectHandle__sparse_tensor__()::MlirDialectHandle
end

"""
    MlirSparseTensorDimLevelType

Dimension level types that define sparse tensors: - MLIR\\_SPARSE\\_TENSOR\\_DIM\\_LEVEL\\_DENSE - dimension is dense, every entry is stored - MLIR\\_SPARSE\\_TENSOR\\_DIM\\_LEVEL\\_COMPRESSED - dimension is sparse, only nonzeros are stored. - MLIR\\_SPARSE\\_TENSOR\\_DIM\\_LEVEL\\_SINGLETON - dimension contains single coordinate, no siblings.

These correspond to SparseTensorEncodingAttr::DimLevelType in the C++ API. If updating, keep them in sync and update the static\\_assert in the impl file.
"""
@cenum MlirSparseTensorDimLevelType::UInt32 begin
    MLIR_SPARSE_TENSOR_DIM_LEVEL_DENSE = 0x0000000000000000
    MLIR_SPARSE_TENSOR_DIM_LEVEL_COMPRESSED = 0x0000000000000001
    MLIR_SPARSE_TENSOR_DIM_LEVEL_SINGLETON = 0x0000000000000002
end

"""
    mlirAttributeIsASparseTensorEncodingAttr(attr)

Checks whether the given attribute is a sparse\\_tensor.encoding attribute.
"""
function mlirAttributeIsASparseTensorEncodingAttr(attr)
    @ccall mlir_c.mlirAttributeIsASparseTensorEncodingAttr(attr::MlirAttribute)::Bool
end

"""
    mlirSparseTensorEncodingAttrGet(ctx, numDimLevelTypes, dimLevelTypes, dimOrdering, pointerBitWidth, indexBitWidth)

Creates a sparse\\_tensor.encoding attribute with the given parameters.
"""
function mlirSparseTensorEncodingAttrGet(ctx, numDimLevelTypes, dimLevelTypes, dimOrdering, pointerBitWidth, indexBitWidth)
    @ccall mlir_c.mlirSparseTensorEncodingAttrGet(ctx::MlirContext, numDimLevelTypes::intptr_t, dimLevelTypes::Ptr{MlirSparseTensorDimLevelType}, dimOrdering::MlirAffineMap, pointerBitWidth::Cint, indexBitWidth::Cint)::MlirAttribute
end

"""
    mlirSparseTensorEncodingGetNumDimLevelTypes(attr)

Returns the number of dim level types in a sparse\\_tensor.encoding attribute.
"""
function mlirSparseTensorEncodingGetNumDimLevelTypes(attr)
    @ccall mlir_c.mlirSparseTensorEncodingGetNumDimLevelTypes(attr::MlirAttribute)::intptr_t
end

"""
    mlirSparseTensorEncodingAttrGetDimLevelType(attr, pos)

Returns a specified dim level type in a sparse\\_tensor.encoding attribute.
"""
function mlirSparseTensorEncodingAttrGetDimLevelType(attr, pos)
    @ccall mlir_c.mlirSparseTensorEncodingAttrGetDimLevelType(attr::MlirAttribute, pos::intptr_t)::MlirSparseTensorDimLevelType
end

"""
    mlirSparseTensorEncodingAttrGetDimOrdering(attr)

Returns the dimension ordering in a sparse\\_tensor.encoding attribute.
"""
function mlirSparseTensorEncodingAttrGetDimOrdering(attr)
    @ccall mlir_c.mlirSparseTensorEncodingAttrGetDimOrdering(attr::MlirAttribute)::MlirAffineMap
end

"""
    mlirSparseTensorEncodingAttrGetPointerBitWidth(attr)

Returns the pointer bit width in a sparse\\_tensor.encoding attribute.
"""
function mlirSparseTensorEncodingAttrGetPointerBitWidth(attr)
    @ccall mlir_c.mlirSparseTensorEncodingAttrGetPointerBitWidth(attr::MlirAttribute)::Cint
end

"""
    mlirSparseTensorEncodingAttrGetIndexBitWidth(attr)

Returns the index bit width in a sparse\\_tensor.encoding attribute.
"""
function mlirSparseTensorEncodingAttrGetIndexBitWidth(attr)
    @ccall mlir_c.mlirSparseTensorEncodingAttrGetIndexBitWidth(attr::MlirAttribute)::Cint
end

function mlirRegisterSparseTensorPasses()
    @ccall mlir_c.mlirRegisterSparseTensorPasses()::Cvoid
end

function mlirCreateSparseTensorSparseTensorConversion()
    @ccall mlir_c.mlirCreateSparseTensorSparseTensorConversion()::MlirPass
end

function mlirRegisterSparseTensorSparseTensorConversion()
    @ccall mlir_c.mlirRegisterSparseTensorSparseTensorConversion()::Cvoid
end

function mlirCreateSparseTensorSparsification()
    @ccall mlir_c.mlirCreateSparseTensorSparsification()::MlirPass
end

function mlirRegisterSparseTensorSparsification()
    @ccall mlir_c.mlirRegisterSparseTensorSparsification()::Cvoid
end

function mlirGetDialectHandle__tensor__()
    @ccall mlir_c.mlirGetDialectHandle__tensor__()::MlirDialectHandle
end

struct MlirExecutionEngine
    ptr::Ptr{Cvoid}
end

"""
    mlirExecutionEngineCreate(op, optLevel, numPaths, sharedLibPaths)

Creates an ExecutionEngine for the provided ModuleOp. The ModuleOp is expected to be "translatable" to LLVM IR (only contains operations in dialects that implement the `LLVMTranslationDialectInterface`). The module ownership stays with the client and can be destroyed as soon as the call returns. `optLevel` is the optimization level to be used for transformation and code generation. LLVM passes at `optLevel` are run before code generation. The number and array of paths corresponding to shared libraries that will be loaded are specified via `numPaths` and `sharedLibPaths` respectively. TODO: figure out other options.
"""
function mlirExecutionEngineCreate(op, optLevel, numPaths, sharedLibPaths)
    @ccall mlir_c.mlirExecutionEngineCreate(op::MlirModule, optLevel::Cint, numPaths::Cint, sharedLibPaths::Ptr{MlirStringRef})::MlirExecutionEngine
end

"""
    mlirExecutionEngineDestroy(jit)

Destroy an ExecutionEngine instance.
"""
function mlirExecutionEngineDestroy(jit)
    @ccall mlir_c.mlirExecutionEngineDestroy(jit::MlirExecutionEngine)::Cvoid
end

"""
    mlirExecutionEngineIsNull(jit)

Checks whether an execution engine is null.
"""
function mlirExecutionEngineIsNull(jit)
    @ccall mlir_c.mlirExecutionEngineIsNull(jit::MlirExecutionEngine)::Bool
end

"""
    mlirExecutionEngineInvokePacked(jit, name, arguments)

Invoke a native function in the execution engine by name with the arguments and result of the invoked function passed as an array of pointers. The function must have been tagged with the `llvm.emit\\_c\\_interface` attribute. Returns a failure if the execution fails for any reason (the function name can't be resolved for instance).
"""
function mlirExecutionEngineInvokePacked(jit, name, arguments)
    @ccall mlir_c.mlirExecutionEngineInvokePacked(jit::MlirExecutionEngine, name::MlirStringRef, arguments::Ptr{Ptr{Cvoid}})::MlirLogicalResult
end

"""
    mlirExecutionEngineLookupPacked(jit, name)

Lookup the wrapper of the native function in the execution engine with the given name, returns nullptr if the function can't be looked-up.
"""
function mlirExecutionEngineLookupPacked(jit, name)
    @ccall mlir_c.mlirExecutionEngineLookupPacked(jit::MlirExecutionEngine, name::MlirStringRef)::Ptr{Cvoid}
end

"""
    mlirExecutionEngineLookup(jit, name)

Lookup a native function in the execution engine by name, returns nullptr if the name can't be looked-up.
"""
function mlirExecutionEngineLookup(jit, name)
    @ccall mlir_c.mlirExecutionEngineLookup(jit::MlirExecutionEngine, name::MlirStringRef)::Ptr{Cvoid}
end

"""
    mlirExecutionEngineRegisterSymbol(jit, name, sym)

Register a symbol with the jit: this symbol will be accessible to the jitted code.
"""
function mlirExecutionEngineRegisterSymbol(jit, name, sym)
    @ccall mlir_c.mlirExecutionEngineRegisterSymbol(jit::MlirExecutionEngine, name::MlirStringRef, sym::Ptr{Cvoid})::Cvoid
end

"""
    mlirExecutionEngineDumpToObjectFile(jit, fileName)

Dump as an object in `fileName`.
"""
function mlirExecutionEngineDumpToObjectFile(jit, fileName)
    @ccall mlir_c.mlirExecutionEngineDumpToObjectFile(jit::MlirExecutionEngine, fileName::MlirStringRef)::Cvoid
end

struct MlirIntegerSet
    ptr::Ptr{Cvoid}
end

"""
    mlirIntegerSetGetContext(set)

Gets the context in which the given integer set lives.
"""
function mlirIntegerSetGetContext(set)
    @ccall mlir_c.mlirIntegerSetGetContext(set::MlirIntegerSet)::MlirContext
end

"""
    mlirIntegerSetIsNull(set)

Checks whether an integer set is a null object.
"""
function mlirIntegerSetIsNull(set)
    @ccall mlir_c.mlirIntegerSetIsNull(set::MlirIntegerSet)::Bool
end

"""
    mlirIntegerSetEqual(s1, s2)

Checks if two integer set objects are equal. This is a "shallow" comparison of two objects. Only the sets with some small number of constraints are uniqued and compare equal here. Set objects that represent the same integer set with different constraints may be considered non-equal by this check. Set difference followed by an (expensive) emptiness check should be used to check equivalence of the underlying integer sets.
"""
function mlirIntegerSetEqual(s1, s2)
    @ccall mlir_c.mlirIntegerSetEqual(s1::MlirIntegerSet, s2::MlirIntegerSet)::Bool
end

"""
    mlirIntegerSetPrint(set, callback, userData)

Prints an integer set by sending chunks of the string representation and forwarding `userData to `callback`. Note that the callback may be called several times with consecutive chunks of the string.
"""
function mlirIntegerSetPrint(set, callback, userData)
    @ccall mlir_c.mlirIntegerSetPrint(set::MlirIntegerSet, callback::MlirStringCallback, userData::Ptr{Cvoid})::Cvoid
end

"""
    mlirIntegerSetDump(set)

Prints an integer set to the standard error stream.
"""
function mlirIntegerSetDump(set)
    @ccall mlir_c.mlirIntegerSetDump(set::MlirIntegerSet)::Cvoid
end

"""
    mlirIntegerSetEmptyGet(context, numDims, numSymbols)

Gets or creates a new canonically empty integer set with the give number of dimensions and symbols in the given context.
"""
function mlirIntegerSetEmptyGet(context, numDims, numSymbols)
    @ccall mlir_c.mlirIntegerSetEmptyGet(context::MlirContext, numDims::intptr_t, numSymbols::intptr_t)::MlirIntegerSet
end

"""
    mlirIntegerSetGet(context, numDims, numSymbols, numConstraints, constraints, eqFlags)

Gets or creates a new integer set in the given context. The set is defined by a list of affine constraints, with the given number of input dimensions and symbols, which are treated as either equalities (eqFlags is 1) or inequalities (eqFlags is 0). Both `constraints` and `eqFlags` are expected to point to at least `numConstraint` consecutive values.
"""
function mlirIntegerSetGet(context, numDims, numSymbols, numConstraints, constraints, eqFlags)
    @ccall mlir_c.mlirIntegerSetGet(context::MlirContext, numDims::intptr_t, numSymbols::intptr_t, numConstraints::intptr_t, constraints::Ptr{MlirAffineExpr}, eqFlags::Ptr{Bool})::MlirIntegerSet
end

"""
    mlirIntegerSetReplaceGet(set, dimReplacements, symbolReplacements, numResultDims, numResultSymbols)

Gets or creates a new integer set in which the values and dimensions of the given set are replaced with the given affine expressions. `dimReplacements` and `symbolReplacements` are expected to point to at least as many consecutive expressions as the given set has dimensions and symbols, respectively. The new set will have `numResultDims` and `numResultSymbols` dimensions and symbols, respectively.
"""
function mlirIntegerSetReplaceGet(set, dimReplacements, symbolReplacements, numResultDims, numResultSymbols)
    @ccall mlir_c.mlirIntegerSetReplaceGet(set::MlirIntegerSet, dimReplacements::Ptr{MlirAffineExpr}, symbolReplacements::Ptr{MlirAffineExpr}, numResultDims::intptr_t, numResultSymbols::intptr_t)::MlirIntegerSet
end

"""
    mlirIntegerSetIsCanonicalEmpty(set)

Checks whether the given set is a canonical empty set, e.g., the set returned by [`mlirIntegerSetEmptyGet`](@ref).
"""
function mlirIntegerSetIsCanonicalEmpty(set)
    @ccall mlir_c.mlirIntegerSetIsCanonicalEmpty(set::MlirIntegerSet)::Bool
end

"""
    mlirIntegerSetGetNumDims(set)

Returns the number of dimensions in the given set.
"""
function mlirIntegerSetGetNumDims(set)
    @ccall mlir_c.mlirIntegerSetGetNumDims(set::MlirIntegerSet)::intptr_t
end

"""
    mlirIntegerSetGetNumSymbols(set)

Returns the number of symbols in the given set.
"""
function mlirIntegerSetGetNumSymbols(set)
    @ccall mlir_c.mlirIntegerSetGetNumSymbols(set::MlirIntegerSet)::intptr_t
end

"""
    mlirIntegerSetGetNumInputs(set)

Returns the number of inputs (dimensions + symbols) in the given set.
"""
function mlirIntegerSetGetNumInputs(set)
    @ccall mlir_c.mlirIntegerSetGetNumInputs(set::MlirIntegerSet)::intptr_t
end

"""
    mlirIntegerSetGetNumConstraints(set)

Returns the number of constraints (equalities + inequalities) in the given set.
"""
function mlirIntegerSetGetNumConstraints(set)
    @ccall mlir_c.mlirIntegerSetGetNumConstraints(set::MlirIntegerSet)::intptr_t
end

"""
    mlirIntegerSetGetNumEqualities(set)

Returns the number of equalities in the given set.
"""
function mlirIntegerSetGetNumEqualities(set)
    @ccall mlir_c.mlirIntegerSetGetNumEqualities(set::MlirIntegerSet)::intptr_t
end

"""
    mlirIntegerSetGetNumInequalities(set)

Returns the number of inequalities in the given set.
"""
function mlirIntegerSetGetNumInequalities(set)
    @ccall mlir_c.mlirIntegerSetGetNumInequalities(set::MlirIntegerSet)::intptr_t
end

"""
    mlirIntegerSetGetConstraint(set, pos)

Returns `pos`-th constraint of the set.
"""
function mlirIntegerSetGetConstraint(set, pos)
    @ccall mlir_c.mlirIntegerSetGetConstraint(set::MlirIntegerSet, pos::intptr_t)::MlirAffineExpr
end

"""
    mlirIntegerSetIsConstraintEq(set, pos)

Returns `true` of the `pos`-th constraint of the set is an equality constraint, `false` otherwise.
"""
function mlirIntegerSetIsConstraintEq(set, pos)
    @ccall mlir_c.mlirIntegerSetIsConstraintEq(set::MlirIntegerSet, pos::intptr_t)::Bool
end

"""
    mlirOperationImplementsInterface(operation, interfaceTypeID)

Returns `true` if the given operation implements an interface identified by its TypeID.
"""
function mlirOperationImplementsInterface(operation, interfaceTypeID)
    @ccall mlir_c.mlirOperationImplementsInterface(operation::MlirOperation, interfaceTypeID::MlirTypeID)::Bool
end

"""
    mlirOperationImplementsInterfaceStatic(operationName, context, interfaceTypeID)

Returns `true` if the operation identified by its canonical string name implements the interface identified by its TypeID in the given context. Note that interfaces may be attached to operations in some contexts and not others.
"""
function mlirOperationImplementsInterfaceStatic(operationName, context, interfaceTypeID)
    @ccall mlir_c.mlirOperationImplementsInterfaceStatic(operationName::MlirStringRef, context::MlirContext, interfaceTypeID::MlirTypeID)::Bool
end

"""
    mlirInferTypeOpInterfaceTypeID()

Returns the interface TypeID of the InferTypeOpInterface.
"""
function mlirInferTypeOpInterfaceTypeID()
    @ccall mlir_c.mlirInferTypeOpInterfaceTypeID()::MlirTypeID
end

# typedef void ( * MlirTypesCallback ) ( intptr_t , MlirType * , void * )
"""
These callbacks are used to return multiple types from functions while transferring ownership to the caller. The first argument is the number of consecutive elements pointed to by the second argument. The third argument is an opaque pointer forwarded to the callback by the caller.
"""
const MlirTypesCallback = Ptr{Cvoid}

"""
    mlirInferTypeOpInterfaceInferReturnTypes(opName, context, location, nOperands, operands, attributes, nRegions, regions, callback, userData)

Infers the return types of the operation identified by its canonical given the arguments that will be supplied to its generic builder. Calls `callback` with the types of inferred arguments, potentially several times, on success. Returns failure otherwise.
"""
function mlirInferTypeOpInterfaceInferReturnTypes(opName, context, location, nOperands, operands, attributes, nRegions, regions, callback, userData)
    @ccall mlir_c.mlirInferTypeOpInterfaceInferReturnTypes(opName::MlirStringRef, context::MlirContext, location::MlirLocation, nOperands::intptr_t, operands::Ptr{MlirValue}, attributes::MlirAttribute, nRegions::intptr_t, regions::Ptr{MlirRegion}, callback::MlirTypesCallback, userData::Ptr{Cvoid})::MlirLogicalResult
end

"""
    mlirRegisterAllDialects(registry)

Appends all upstream dialects and extensions to the dialect registry.
"""
function mlirRegisterAllDialects(registry)
    @ccall mlir_c.mlirRegisterAllDialects(registry::MlirDialectRegistry)::Cvoid
end

"""
    mlirRegisterAllLLVMTranslations(context)

Register all translations to LLVM IR for dialects that can support it.
"""
function mlirRegisterAllLLVMTranslations(context)
    @ccall mlir_c.mlirRegisterAllLLVMTranslations(context::MlirContext)::Cvoid
end

"""
    mlirRegisterAllPasses()

Register all compiler passes of MLIR.
"""
function mlirRegisterAllPasses()
    @ccall mlir_c.mlirRegisterAllPasses()::Cvoid
end

function mlirRegisterTransformsPasses()
    @ccall mlir_c.mlirRegisterTransformsPasses()::Cvoid
end

function mlirCreateTransformsCSE()
    @ccall mlir_c.mlirCreateTransformsCSE()::MlirPass
end

function mlirRegisterTransformsCSE()
    @ccall mlir_c.mlirRegisterTransformsCSE()::Cvoid
end

function mlirCreateTransformsCanonicalizer()
    @ccall mlir_c.mlirCreateTransformsCanonicalizer()::MlirPass
end

function mlirRegisterTransformsCanonicalizer()
    @ccall mlir_c.mlirRegisterTransformsCanonicalizer()::Cvoid
end

function mlirCreateTransformsControlFlowSink()
    @ccall mlir_c.mlirCreateTransformsControlFlowSink()::MlirPass
end

function mlirRegisterTransformsControlFlowSink()
    @ccall mlir_c.mlirRegisterTransformsControlFlowSink()::Cvoid
end

function mlirCreateTransformsInliner()
    @ccall mlir_c.mlirCreateTransformsInliner()::MlirPass
end

function mlirRegisterTransformsInliner()
    @ccall mlir_c.mlirRegisterTransformsInliner()::Cvoid
end

function mlirCreateTransformsLocationSnapshot()
    @ccall mlir_c.mlirCreateTransformsLocationSnapshot()::MlirPass
end

function mlirRegisterTransformsLocationSnapshot()
    @ccall mlir_c.mlirRegisterTransformsLocationSnapshot()::Cvoid
end

function mlirCreateTransformsLoopInvariantCodeMotion()
    @ccall mlir_c.mlirCreateTransformsLoopInvariantCodeMotion()::MlirPass
end

function mlirRegisterTransformsLoopInvariantCodeMotion()
    @ccall mlir_c.mlirRegisterTransformsLoopInvariantCodeMotion()::Cvoid
end

function mlirCreateTransformsPrintOpStats()
    @ccall mlir_c.mlirCreateTransformsPrintOpStats()::MlirPass
end

function mlirRegisterTransformsPrintOpStats()
    @ccall mlir_c.mlirRegisterTransformsPrintOpStats()::Cvoid
end

function mlirCreateTransformsSCCP()
    @ccall mlir_c.mlirCreateTransformsSCCP()::MlirPass
end

function mlirRegisterTransformsSCCP()
    @ccall mlir_c.mlirRegisterTransformsSCCP()::Cvoid
end

function mlirCreateTransformsStripDebugInfo()
    @ccall mlir_c.mlirCreateTransformsStripDebugInfo()::MlirPass
end

function mlirRegisterTransformsStripDebugInfo()
    @ccall mlir_c.mlirRegisterTransformsStripDebugInfo()::Cvoid
end

function mlirCreateTransformsSymbolDCE()
    @ccall mlir_c.mlirCreateTransformsSymbolDCE()::MlirPass
end

function mlirRegisterTransformsSymbolDCE()
    @ccall mlir_c.mlirRegisterTransformsSymbolDCE()::Cvoid
end

function mlirCreateTransformsSymbolPrivatize()
    @ccall mlir_c.mlirCreateTransformsSymbolPrivatize()::MlirPass
end

function mlirRegisterTransformsSymbolPrivatize()
    @ccall mlir_c.mlirRegisterTransformsSymbolPrivatize()::Cvoid
end

function mlirCreateTransformsTopologicalSort()
    @ccall mlir_c.mlirCreateTransformsTopologicalSort()::MlirPass
end

function mlirRegisterTransformsTopologicalSort()
    @ccall mlir_c.mlirRegisterTransformsTopologicalSort()::Cvoid
end

function mlirCreateTransformsViewOpGraph()
    @ccall mlir_c.mlirCreateTransformsViewOpGraph()::MlirPass
end

function mlirRegisterTransformsViewOpGraph()
    @ccall mlir_c.mlirRegisterTransformsViewOpGraph()::Cvoid
end

