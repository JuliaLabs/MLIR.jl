using CEnum

const intptr_t = Clong

function mlirStringRefCreate(str, length)
    @ccall (MLIR_C_PATH[]).mlirStringRefCreate(str::Cstring, length::Cint)::MlirStringRef
end

"""
    mlirStringRefCreateFromCString(str)

Constructs a string reference from a null-terminated C string. Prefer [`mlirStringRefCreate`](@ref) if the length of the string is known.
"""
function mlirStringRefCreateFromCString(str)
    @ccall (MLIR_C_PATH[]).mlirStringRefCreateFromCString(str::Cstring)::MlirStringRef
end

"""
    mlirStringRefEqual(string, other)

Returns true if two string references are equal, false otherwise.
"""
function mlirStringRefEqual(string, other)
    @ccall (MLIR_C_PATH[]).mlirStringRefEqual(string::MlirStringRef, other::MlirStringRef)::Bool
end

# typedef void ( * MlirStringCallback ) ( MlirStringRef , void * )
"""
A callback for returning string references.

This function is called back by the functions that need to return a reference to the portion of the string with the following arguments: - an [`MlirStringRef`](@ref) representing the current portion of the string - a pointer to user data forwarded from the printing call.
"""
const MlirStringCallback = Ptr{Cvoid}

"""
    mlirLogicalResultIsSuccess(res)

Checks if the given logical result represents a success.
"""
function mlirLogicalResultIsSuccess(res)
    @ccall (MLIR_C_PATH[]).mlirLogicalResultIsSuccess(res::MlirLogicalResult)::Bool
end

"""
    mlirLogicalResultIsFailure(res)

Checks if the given logical result represents a failure.
"""
function mlirLogicalResultIsFailure(res)
    @ccall (MLIR_C_PATH[]).mlirLogicalResultIsFailure(res::MlirLogicalResult)::Bool
end

"""
    mlirLogicalResultSuccess()

Creates a logical result representing a success.
"""
function mlirLogicalResultSuccess()
    @ccall (MLIR_C_PATH[]).mlirLogicalResultSuccess()::MlirLogicalResult
end

"""
    mlirLogicalResultFailure()

Creates a logical result representing a failure.
"""
function mlirLogicalResultFailure()
    @ccall (MLIR_C_PATH[]).mlirLogicalResultFailure()::MlirLogicalResult
end

"""
    mlirLlvmThreadPoolCreate()

Create an LLVM thread pool. This is reexported here to avoid directly pulling in the LLVM headers directly.
"""
function mlirLlvmThreadPoolCreate()
    @ccall (MLIR_C_PATH[]).mlirLlvmThreadPoolCreate()::MlirLlvmThreadPool
end

"""
    mlirLlvmThreadPoolDestroy(pool)

Destroy an LLVM thread pool.
"""
function mlirLlvmThreadPoolDestroy(pool)
    @ccall (MLIR_C_PATH[]).mlirLlvmThreadPoolDestroy(pool::MlirLlvmThreadPool)::Cvoid
end

"""
    mlirTypeIDCreate(ptr)

`ptr` must be 8 byte aligned and unique to a type valid for the duration of the returned type id's usage
"""
function mlirTypeIDCreate(ptr)
    @ccall (MLIR_C_PATH[]).mlirTypeIDCreate(ptr::Ptr{Cvoid})::MlirTypeID
end

"""
    mlirTypeIDIsNull(typeID)

Checks whether a type id is null.
"""
function mlirTypeIDIsNull(typeID)
    @ccall (MLIR_C_PATH[]).mlirTypeIDIsNull(typeID::MlirTypeID)::Bool
end

"""
    mlirTypeIDEqual(typeID1, typeID2)

Checks if two type ids are equal.
"""
function mlirTypeIDEqual(typeID1, typeID2)
    @ccall (MLIR_C_PATH[]).mlirTypeIDEqual(typeID1::MlirTypeID, typeID2::MlirTypeID)::Bool
end

function mlirTypeIDHashValue(typeID)
    @ccall (MLIR_C_PATH[]).mlirTypeIDHashValue(typeID::MlirTypeID)::Cint
end

"""
    mlirTypeIDAllocatorCreate()

Creates a type id allocator for dynamic type id creation
"""
function mlirTypeIDAllocatorCreate()
    @ccall (MLIR_C_PATH[]).mlirTypeIDAllocatorCreate()::MlirTypeIDAllocator
end

"""
    mlirTypeIDAllocatorDestroy(allocator)

Deallocates the allocator and all allocated type ids
"""
function mlirTypeIDAllocatorDestroy(allocator)
    @ccall (MLIR_C_PATH[]).mlirTypeIDAllocatorDestroy(allocator::MlirTypeIDAllocator)::Cvoid
end

"""
    mlirTypeIDAllocatorAllocateTypeID(allocator)

Allocates a type id that is valid for the lifetime of the allocator
"""
function mlirTypeIDAllocatorAllocateTypeID(allocator)
    @ccall (MLIR_C_PATH[]).mlirTypeIDAllocatorAllocateTypeID(allocator::MlirTypeIDAllocator)::MlirTypeID
end

"""
    mlirContextCreate()

Creates an MLIR context and transfers its ownership to the caller. This sets the default multithreading option (enabled).
"""
function mlirContextCreate()
    @ccall (MLIR_C_PATH[]).mlirContextCreate()::MlirContext
end

"""
    mlirContextCreateWithThreading(threadingEnabled)

Creates an MLIR context with an explicit setting of the multithreading setting and transfers its ownership to the caller.
"""
function mlirContextCreateWithThreading(threadingEnabled)
    @ccall (MLIR_C_PATH[]).mlirContextCreateWithThreading(threadingEnabled::Bool)::MlirContext
end

"""
    mlirContextCreateWithRegistry(registry, threadingEnabled)

Creates an MLIR context, setting the multithreading setting explicitly and pre-loading the dialects from the provided DialectRegistry.
"""
function mlirContextCreateWithRegistry(registry, threadingEnabled)
    @ccall (MLIR_C_PATH[]).mlirContextCreateWithRegistry(registry::MlirDialectRegistry, threadingEnabled::Bool)::MlirContext
end

"""
    mlirContextEqual(ctx1, ctx2)

Checks if two contexts are equal.
"""
function mlirContextEqual(ctx1, ctx2)
    @ccall (MLIR_C_PATH[]).mlirContextEqual(ctx1::MlirContext, ctx2::MlirContext)::Bool
end

"""
    mlirContextIsNull(context)

Checks whether a context is null.
"""
function mlirContextIsNull(context)
    @ccall (MLIR_C_PATH[]).mlirContextIsNull(context::MlirContext)::Bool
end

"""
    mlirContextDestroy(context)

Takes an MLIR context owned by the caller and destroys it.
"""
function mlirContextDestroy(context)
    @ccall (MLIR_C_PATH[]).mlirContextDestroy(context::MlirContext)::Cvoid
end

"""
    mlirContextSetAllowUnregisteredDialects(context, allow)

Sets whether unregistered dialects are allowed in this context.
"""
function mlirContextSetAllowUnregisteredDialects(context, allow)
    @ccall (MLIR_C_PATH[]).mlirContextSetAllowUnregisteredDialects(context::MlirContext, allow::Bool)::Cvoid
end

"""
    mlirContextGetAllowUnregisteredDialects(context)

Returns whether the context allows unregistered dialects.
"""
function mlirContextGetAllowUnregisteredDialects(context)
    @ccall (MLIR_C_PATH[]).mlirContextGetAllowUnregisteredDialects(context::MlirContext)::Bool
end

"""
    mlirContextGetNumRegisteredDialects(context)

Returns the number of dialects registered with the given context. A registered dialect will be loaded if needed by the parser.
"""
function mlirContextGetNumRegisteredDialects(context)
    @ccall (MLIR_C_PATH[]).mlirContextGetNumRegisteredDialects(context::MlirContext)::intptr_t
end

"""
    mlirContextAppendDialectRegistry(ctx, registry)

Append the contents of the given dialect registry to the registry associated with the context.
"""
function mlirContextAppendDialectRegistry(ctx, registry)
    @ccall (MLIR_C_PATH[]).mlirContextAppendDialectRegistry(ctx::MlirContext, registry::MlirDialectRegistry)::Cvoid
end

"""
    mlirContextGetNumLoadedDialects(context)

Returns the number of dialects loaded by the context.
"""
function mlirContextGetNumLoadedDialects(context)
    @ccall (MLIR_C_PATH[]).mlirContextGetNumLoadedDialects(context::MlirContext)::intptr_t
end

"""
    mlirContextGetOrLoadDialect(context, name)

Gets the dialect instance owned by the given context using the dialect namespace to identify it, loads (i.e., constructs the instance of) the dialect if necessary. If the dialect is not registered with the context, returns null. Use mlirContextLoad<Name>Dialect to load an unregistered dialect.
"""
function mlirContextGetOrLoadDialect(context, name)
    @ccall (MLIR_C_PATH[]).mlirContextGetOrLoadDialect(context::MlirContext, name::MlirStringRef)::MlirDialect
end

"""
    mlirContextEnableMultithreading(context, enable)

Set threading mode (must be set to false to mlir-print-ir-after-all).
"""
function mlirContextEnableMultithreading(context, enable)
    @ccall (MLIR_C_PATH[]).mlirContextEnableMultithreading(context::MlirContext, enable::Bool)::Cvoid
end

"""
    mlirContextLoadAllAvailableDialects(context)

Eagerly loads all available dialects registered with a context, making them available for use for IR construction.
"""
function mlirContextLoadAllAvailableDialects(context)
    @ccall (MLIR_C_PATH[]).mlirContextLoadAllAvailableDialects(context::MlirContext)::Cvoid
end

"""
    mlirContextIsRegisteredOperation(context, name)

Returns whether the given fully-qualified operation (i.e. 'dialect.operation') is registered with the context. This will return true if the dialect is loaded and the operation is registered within the dialect.
"""
function mlirContextIsRegisteredOperation(context, name)
    @ccall (MLIR_C_PATH[]).mlirContextIsRegisteredOperation(context::MlirContext, name::MlirStringRef)::Bool
end

"""
    mlirContextSetThreadPool(context, threadPool)

Sets the thread pool of the context explicitly, enabling multithreading in the process. This API should be used to avoid re-creating thread pools in long-running applications that perform multiple compilations, see the C++ documentation for MLIRContext for details.
"""
function mlirContextSetThreadPool(context, threadPool)
    @ccall (MLIR_C_PATH[]).mlirContextSetThreadPool(context::MlirContext, threadPool::MlirLlvmThreadPool)::Cvoid
end

"""
    mlirDialectGetContext(dialect)

Returns the context that owns the dialect.
"""
function mlirDialectGetContext(dialect)
    @ccall (MLIR_C_PATH[]).mlirDialectGetContext(dialect::MlirDialect)::MlirContext
end

"""
    mlirDialectIsNull(dialect)

Checks if the dialect is null.
"""
function mlirDialectIsNull(dialect)
    @ccall (MLIR_C_PATH[]).mlirDialectIsNull(dialect::MlirDialect)::Bool
end

"""
    mlirDialectEqual(dialect1, dialect2)

Checks if two dialects that belong to the same context are equal. Dialects from different contexts will not compare equal.
"""
function mlirDialectEqual(dialect1, dialect2)
    @ccall (MLIR_C_PATH[]).mlirDialectEqual(dialect1::MlirDialect, dialect2::MlirDialect)::Bool
end

"""
    mlirDialectGetNamespace(dialect)

Returns the namespace of the given dialect.
"""
function mlirDialectGetNamespace(dialect)
    @ccall (MLIR_C_PATH[]).mlirDialectGetNamespace(dialect::MlirDialect)::MlirStringRef
end

"""
    mlirDialectHandleGetNamespace(arg1)

Returns the namespace associated with the provided dialect handle.
"""
function mlirDialectHandleGetNamespace(arg1)
    @ccall (MLIR_C_PATH[]).mlirDialectHandleGetNamespace(arg1::MlirDialectHandle)::MlirStringRef
end

"""
    mlirDialectHandleInsertDialect(arg1, arg2)

Inserts the dialect associated with the provided dialect handle into the provided dialect registry
"""
function mlirDialectHandleInsertDialect(arg1, arg2)
    @ccall (MLIR_C_PATH[]).mlirDialectHandleInsertDialect(arg1::MlirDialectHandle, arg2::MlirDialectRegistry)::Cvoid
end

"""
    mlirDialectHandleRegisterDialect(arg1, arg2)

Registers the dialect associated with the provided dialect handle.
"""
function mlirDialectHandleRegisterDialect(arg1, arg2)
    @ccall (MLIR_C_PATH[]).mlirDialectHandleRegisterDialect(arg1::MlirDialectHandle, arg2::MlirContext)::Cvoid
end

"""
    mlirDialectHandleLoadDialect(arg1, arg2)

Loads the dialect associated with the provided dialect handle.
"""
function mlirDialectHandleLoadDialect(arg1, arg2)
    @ccall (MLIR_C_PATH[]).mlirDialectHandleLoadDialect(arg1::MlirDialectHandle, arg2::MlirContext)::MlirDialect
end

"""
    mlirDialectRegistryCreate()

Creates a dialect registry and transfers its ownership to the caller.
"""
function mlirDialectRegistryCreate()
    @ccall (MLIR_C_PATH[]).mlirDialectRegistryCreate()::MlirDialectRegistry
end

"""
    mlirDialectRegistryIsNull(registry)

Checks if the dialect registry is null.
"""
function mlirDialectRegistryIsNull(registry)
    @ccall (MLIR_C_PATH[]).mlirDialectRegistryIsNull(registry::MlirDialectRegistry)::Bool
end

"""
    mlirDialectRegistryDestroy(registry)

Takes a dialect registry owned by the caller and destroys it.
"""
function mlirDialectRegistryDestroy(registry)
    @ccall (MLIR_C_PATH[]).mlirDialectRegistryDestroy(registry::MlirDialectRegistry)::Cvoid
end

"""
    mlirLocationGetAttribute(location)

Returns the underlying location attribute of this location.
"""
function mlirLocationGetAttribute(location)
    @ccall (MLIR_C_PATH[]).mlirLocationGetAttribute(location::MlirLocation)::MlirAttribute
end

"""
    mlirLocationFromAttribute(attribute)

Creates a location from a location attribute.
"""
function mlirLocationFromAttribute(attribute)
    @ccall (MLIR_C_PATH[]).mlirLocationFromAttribute(attribute::MlirAttribute)::MlirLocation
end

"""
    mlirLocationFileLineColGet(context, filename, line, col)

Creates an File/Line/Column location owned by the given context.
"""
function mlirLocationFileLineColGet(context, filename, line, col)
    @ccall (MLIR_C_PATH[]).mlirLocationFileLineColGet(context::MlirContext, filename::MlirStringRef, line::Cuint, col::Cuint)::MlirLocation
end

"""
    mlirLocationCallSiteGet(callee, caller)

Creates a call site location with a callee and a caller.
"""
function mlirLocationCallSiteGet(callee, caller)
    @ccall (MLIR_C_PATH[]).mlirLocationCallSiteGet(callee::MlirLocation, caller::MlirLocation)::MlirLocation
end

"""
    mlirLocationFusedGet(ctx, nLocations, locations, metadata)

Creates a fused location with an array of locations and metadata.
"""
function mlirLocationFusedGet(ctx, nLocations, locations, metadata)
    @ccall (MLIR_C_PATH[]).mlirLocationFusedGet(ctx::MlirContext, nLocations::intptr_t, locations::Ptr{MlirLocation}, metadata::MlirAttribute)::MlirLocation
end

"""
    mlirLocationNameGet(context, name, childLoc)

Creates a name location owned by the given context. Providing null location for childLoc is allowed and if childLoc is null location, then the behavior is the same as having unknown child location.
"""
function mlirLocationNameGet(context, name, childLoc)
    @ccall (MLIR_C_PATH[]).mlirLocationNameGet(context::MlirContext, name::MlirStringRef, childLoc::MlirLocation)::MlirLocation
end

"""
    mlirLocationUnknownGet(context)

Creates a location with unknown position owned by the given context.
"""
function mlirLocationUnknownGet(context)
    @ccall (MLIR_C_PATH[]).mlirLocationUnknownGet(context::MlirContext)::MlirLocation
end

"""
    mlirLocationGetContext(location)

Gets the context that a location was created with.
"""
function mlirLocationGetContext(location)
    @ccall (MLIR_C_PATH[]).mlirLocationGetContext(location::MlirLocation)::MlirContext
end

"""
    mlirLocationIsNull(location)

Checks if the location is null.
"""
function mlirLocationIsNull(location)
    @ccall (MLIR_C_PATH[]).mlirLocationIsNull(location::MlirLocation)::Bool
end

"""
    mlirLocationEqual(l1, l2)

Checks if two locations are equal.
"""
function mlirLocationEqual(l1, l2)
    @ccall (MLIR_C_PATH[]).mlirLocationEqual(l1::MlirLocation, l2::MlirLocation)::Bool
end

"""
    mlirLocationPrint(location, callback, userData)

Prints a location by sending chunks of the string representation and forwarding `userData to `callback`. Note that the callback may be called several times with consecutive chunks of the string.
"""
function mlirLocationPrint(location, callback, userData)
    @ccall (MLIR_C_PATH[]).mlirLocationPrint(location::MlirLocation, callback::MlirStringCallback, userData::Ptr{Cvoid})::Cvoid
end

"""
    mlirModuleCreateEmpty(location)

Creates a new, empty module and transfers ownership to the caller.
"""
function mlirModuleCreateEmpty(location)
    @ccall (MLIR_C_PATH[]).mlirModuleCreateEmpty(location::MlirLocation)::MlirModule
end

"""
    mlirModuleCreateParse(context, _module)

Parses a module from the string and transfers ownership to the caller.
"""
function mlirModuleCreateParse(context, _module)
    @ccall (MLIR_C_PATH[]).mlirModuleCreateParse(context::MlirContext, _module::MlirStringRef)::MlirModule
end

"""
    mlirModuleGetContext(_module)

Gets the context that a module was created with.
"""
function mlirModuleGetContext(_module)
    @ccall (MLIR_C_PATH[]).mlirModuleGetContext(_module::MlirModule)::MlirContext
end

"""
    mlirModuleGetBody(_module)

Gets the body of the module, i.e. the only block it contains.
"""
function mlirModuleGetBody(_module)
    @ccall (MLIR_C_PATH[]).mlirModuleGetBody(_module::MlirModule)::MlirBlock
end

"""
    mlirModuleIsNull(_module)

Checks whether a module is null.
"""
function mlirModuleIsNull(_module)
    @ccall (MLIR_C_PATH[]).mlirModuleIsNull(_module::MlirModule)::Bool
end

"""
    mlirModuleDestroy(_module)

Takes a module owned by the caller and deletes it.
"""
function mlirModuleDestroy(_module)
    @ccall (MLIR_C_PATH[]).mlirModuleDestroy(_module::MlirModule)::Cvoid
end

"""
    mlirModuleGetOperation(_module)

Views the module as a generic operation.
"""
function mlirModuleGetOperation(_module)
    @ccall (MLIR_C_PATH[]).mlirModuleGetOperation(_module::MlirModule)::MlirOperation
end

"""
    mlirModuleFromOperation(op)

Views the generic operation as a module. The returned module is null when the input operation was not a ModuleOp.
"""
function mlirModuleFromOperation(op)
    @ccall (MLIR_C_PATH[]).mlirModuleFromOperation(op::MlirOperation)::MlirModule
end

"""
    mlirOperationStateGet(name, loc)

Constructs an operation state from a name and a location.
"""
function mlirOperationStateGet(name, loc)
    @ccall (MLIR_C_PATH[]).mlirOperationStateGet(name::MlirStringRef, loc::MlirLocation)::MlirOperationState
end

"""
    mlirOperationStateAddResults(state, n, results)

Adds a list of components to the operation state.
"""
function mlirOperationStateAddResults(state, n, results)
    @ccall (MLIR_C_PATH[]).mlirOperationStateAddResults(state::Ptr{MlirOperationState}, n::intptr_t, results::Ptr{MlirType})::Cvoid
end

function mlirOperationStateAddOperands(state, n, operands)
    @ccall (MLIR_C_PATH[]).mlirOperationStateAddOperands(state::Ptr{MlirOperationState}, n::intptr_t, operands::Ptr{MlirValue})::Cvoid
end

function mlirOperationStateAddOwnedRegions(state, n, regions)
    @ccall (MLIR_C_PATH[]).mlirOperationStateAddOwnedRegions(state::Ptr{MlirOperationState}, n::intptr_t, regions::Ptr{MlirRegion})::Cvoid
end

function mlirOperationStateAddSuccessors(state, n, successors)
    @ccall (MLIR_C_PATH[]).mlirOperationStateAddSuccessors(state::Ptr{MlirOperationState}, n::intptr_t, successors::Ptr{MlirBlock})::Cvoid
end

function mlirOperationStateAddAttributes(state, n, attributes)
    @ccall (MLIR_C_PATH[]).mlirOperationStateAddAttributes(state::Ptr{MlirOperationState}, n::intptr_t, attributes::Ptr{MlirNamedAttribute})::Cvoid
end

"""
    mlirOperationStateEnableResultTypeInference(state)

Enables result type inference for the operation under construction. If enabled, then the caller must not have called [`mlirOperationStateAddResults`](@ref)(). Note that if enabled, the [`mlirOperationCreate`](@ref)() call is failable: it will return a null operation on inference failure and will emit diagnostics.
"""
function mlirOperationStateEnableResultTypeInference(state)
    @ccall (MLIR_C_PATH[]).mlirOperationStateEnableResultTypeInference(state::Ptr{MlirOperationState})::Cvoid
end

"""
    mlirOpPrintingFlagsCreate()

Creates new printing flags with defaults, intended for customization. Must be freed with a call to [`mlirOpPrintingFlagsDestroy`](@ref)().
"""
function mlirOpPrintingFlagsCreate()
    @ccall (MLIR_C_PATH[]).mlirOpPrintingFlagsCreate()::MlirOpPrintingFlags
end

"""
    mlirOpPrintingFlagsDestroy(flags)

Destroys printing flags created with [`mlirOpPrintingFlagsCreate`](@ref).
"""
function mlirOpPrintingFlagsDestroy(flags)
    @ccall (MLIR_C_PATH[]).mlirOpPrintingFlagsDestroy(flags::MlirOpPrintingFlags)::Cvoid
end

"""
    mlirOpPrintingFlagsElideLargeElementsAttrs(flags, largeElementLimit)

Enables the elision of large elements attributes by printing a lexically valid but otherwise meaningless form instead of the element data. The `largeElementLimit` is used to configure what is considered to be a "large" ElementsAttr by providing an upper limit to the number of elements.
"""
function mlirOpPrintingFlagsElideLargeElementsAttrs(flags, largeElementLimit)
    @ccall (MLIR_C_PATH[]).mlirOpPrintingFlagsElideLargeElementsAttrs(flags::MlirOpPrintingFlags, largeElementLimit::intptr_t)::Cvoid
end

"""
    mlirOpPrintingFlagsEnableDebugInfo(flags, enable, prettyForm)

Enable or disable printing of debug information (based on `enable`). If 'prettyForm' is set to true, debug information is printed in a more readable 'pretty' form. Note: The IR generated with 'prettyForm' is not parsable.
"""
function mlirOpPrintingFlagsEnableDebugInfo(flags, enable, prettyForm)
    @ccall (MLIR_C_PATH[]).mlirOpPrintingFlagsEnableDebugInfo(flags::MlirOpPrintingFlags, enable::Bool, prettyForm::Bool)::Cvoid
end

"""
    mlirOpPrintingFlagsPrintGenericOpForm(flags)

Always print operations in the generic form.
"""
function mlirOpPrintingFlagsPrintGenericOpForm(flags)
    @ccall (MLIR_C_PATH[]).mlirOpPrintingFlagsPrintGenericOpForm(flags::MlirOpPrintingFlags)::Cvoid
end

"""
    mlirOpPrintingFlagsUseLocalScope(flags)

Use local scope when printing the operation. This allows for using the printer in a more localized and thread-safe setting, but may not necessarily be identical to what the IR will look like when dumping the full module.
"""
function mlirOpPrintingFlagsUseLocalScope(flags)
    @ccall (MLIR_C_PATH[]).mlirOpPrintingFlagsUseLocalScope(flags::MlirOpPrintingFlags)::Cvoid
end

"""
    mlirOpPrintingFlagsAssumeVerified(flags)

Do not verify the operation when using custom operation printers.
"""
function mlirOpPrintingFlagsAssumeVerified(flags)
    @ccall (MLIR_C_PATH[]).mlirOpPrintingFlagsAssumeVerified(flags::MlirOpPrintingFlags)::Cvoid
end

"""
    mlirBytecodeWriterConfigCreate()

Creates new printing flags with defaults, intended for customization. Must be freed with a call to [`mlirBytecodeWriterConfigDestroy`](@ref)().
"""
function mlirBytecodeWriterConfigCreate()
    @ccall (MLIR_C_PATH[]).mlirBytecodeWriterConfigCreate()::MlirBytecodeWriterConfig
end

"""
    mlirBytecodeWriterConfigDestroy(config)

Destroys printing flags created with [`mlirBytecodeWriterConfigCreate`](@ref).
"""
function mlirBytecodeWriterConfigDestroy(config)
    @ccall (MLIR_C_PATH[]).mlirBytecodeWriterConfigDestroy(config::MlirBytecodeWriterConfig)::Cvoid
end

"""
    mlirBytecodeWriterConfigDesiredEmitVersion(flags, version)

Sets the version to emit in the writer config.
"""
function mlirBytecodeWriterConfigDesiredEmitVersion(flags, version)
    @ccall (MLIR_C_PATH[]).mlirBytecodeWriterConfigDesiredEmitVersion(flags::MlirBytecodeWriterConfig, version::Int64)::Cvoid
end

"""
    mlirOperationCreate(state)

Creates an operation and transfers ownership to the caller. Note that caller owned child objects are transferred in this call and must not be further used. Particularly, this applies to any regions added to the state (the implementation may invalidate any such pointers).

This call can fail under the following conditions, in which case, it will return a null operation and emit diagnostics: - Result type inference is enabled and cannot be performed.
"""
function mlirOperationCreate(state)
    @ccall (MLIR_C_PATH[]).mlirOperationCreate(state::Ptr{MlirOperationState})::MlirOperation
end

"""
    mlirOperationCreateParse(context, sourceStr, sourceName)

Parses an operation, giving ownership to the caller. If parsing fails a null operation will be returned, and an error diagnostic emitted.

`sourceStr` may be either the text assembly format, or binary bytecode format. `sourceName` is used as the file name of the source; any IR without locations will get a `FileLineColLoc` location with `sourceName` as the file name.
"""
function mlirOperationCreateParse(context, sourceStr, sourceName)
    @ccall (MLIR_C_PATH[]).mlirOperationCreateParse(context::MlirContext, sourceStr::MlirStringRef, sourceName::MlirStringRef)::MlirOperation
end

"""
    mlirOperationClone(op)

Creates a deep copy of an operation. The operation is not inserted and ownership is transferred to the caller.
"""
function mlirOperationClone(op)
    @ccall (MLIR_C_PATH[]).mlirOperationClone(op::MlirOperation)::MlirOperation
end

"""
    mlirOperationDestroy(op)

Takes an operation owned by the caller and destroys it.
"""
function mlirOperationDestroy(op)
    @ccall (MLIR_C_PATH[]).mlirOperationDestroy(op::MlirOperation)::Cvoid
end

"""
    mlirOperationRemoveFromParent(op)

Removes the given operation from its parent block. The operation is not destroyed. The ownership of the operation is transferred to the caller.
"""
function mlirOperationRemoveFromParent(op)
    @ccall (MLIR_C_PATH[]).mlirOperationRemoveFromParent(op::MlirOperation)::Cvoid
end

"""
    mlirOperationIsNull(op)

Checks whether the underlying operation is null.
"""
function mlirOperationIsNull(op)
    @ccall (MLIR_C_PATH[]).mlirOperationIsNull(op::MlirOperation)::Bool
end

"""
    mlirOperationEqual(op, other)

Checks whether two operation handles point to the same operation. This does not perform deep comparison.
"""
function mlirOperationEqual(op, other)
    @ccall (MLIR_C_PATH[]).mlirOperationEqual(op::MlirOperation, other::MlirOperation)::Bool
end

"""
    mlirOperationGetContext(op)

Gets the context this operation is associated with
"""
function mlirOperationGetContext(op)
    @ccall (MLIR_C_PATH[]).mlirOperationGetContext(op::MlirOperation)::MlirContext
end

"""
    mlirOperationGetLocation(op)

Gets the location of the operation.
"""
function mlirOperationGetLocation(op)
    @ccall (MLIR_C_PATH[]).mlirOperationGetLocation(op::MlirOperation)::MlirLocation
end

"""
    mlirOperationGetTypeID(op)

Gets the type id of the operation. Returns null if the operation does not have a registered operation description.
"""
function mlirOperationGetTypeID(op)
    @ccall (MLIR_C_PATH[]).mlirOperationGetTypeID(op::MlirOperation)::MlirTypeID
end

"""
    mlirOperationGetName(op)

Gets the name of the operation as an identifier.
"""
function mlirOperationGetName(op)
    @ccall (MLIR_C_PATH[]).mlirOperationGetName(op::MlirOperation)::MlirIdentifier
end

"""
    mlirOperationGetBlock(op)

Gets the block that owns this operation, returning null if the operation is not owned.
"""
function mlirOperationGetBlock(op)
    @ccall (MLIR_C_PATH[]).mlirOperationGetBlock(op::MlirOperation)::MlirBlock
end

"""
    mlirOperationGetParentOperation(op)

Gets the operation that owns this operation, returning null if the operation is not owned.
"""
function mlirOperationGetParentOperation(op)
    @ccall (MLIR_C_PATH[]).mlirOperationGetParentOperation(op::MlirOperation)::MlirOperation
end

"""
    mlirOperationGetNumRegions(op)

Returns the number of regions attached to the given operation.
"""
function mlirOperationGetNumRegions(op)
    @ccall (MLIR_C_PATH[]).mlirOperationGetNumRegions(op::MlirOperation)::intptr_t
end

"""
    mlirOperationGetRegion(op, pos)

Returns `pos`-th region attached to the operation.
"""
function mlirOperationGetRegion(op, pos)
    @ccall (MLIR_C_PATH[]).mlirOperationGetRegion(op::MlirOperation, pos::intptr_t)::MlirRegion
end

"""
    mlirOperationGetNextInBlock(op)

Returns an operation immediately following the given operation it its enclosing block.
"""
function mlirOperationGetNextInBlock(op)
    @ccall (MLIR_C_PATH[]).mlirOperationGetNextInBlock(op::MlirOperation)::MlirOperation
end

"""
    mlirOperationGetNumOperands(op)

Returns the number of operands of the operation.
"""
function mlirOperationGetNumOperands(op)
    @ccall (MLIR_C_PATH[]).mlirOperationGetNumOperands(op::MlirOperation)::intptr_t
end

"""
    mlirOperationGetOperand(op, pos)

Returns `pos`-th operand of the operation.
"""
function mlirOperationGetOperand(op, pos)
    @ccall (MLIR_C_PATH[]).mlirOperationGetOperand(op::MlirOperation, pos::intptr_t)::MlirValue
end

"""
    mlirOperationSetOperand(op, pos, newValue)

Sets the `pos`-th operand of the operation.
"""
function mlirOperationSetOperand(op, pos, newValue)
    @ccall (MLIR_C_PATH[]).mlirOperationSetOperand(op::MlirOperation, pos::intptr_t, newValue::MlirValue)::Cvoid
end

"""
    mlirOperationSetOperands(op, nOperands, operands)

Replaces the operands of the operation.
"""
function mlirOperationSetOperands(op, nOperands, operands)
    @ccall (MLIR_C_PATH[]).mlirOperationSetOperands(op::MlirOperation, nOperands::intptr_t, operands::Ptr{MlirValue})::Cvoid
end

"""
    mlirOperationGetNumResults(op)

Returns the number of results of the operation.
"""
function mlirOperationGetNumResults(op)
    @ccall (MLIR_C_PATH[]).mlirOperationGetNumResults(op::MlirOperation)::intptr_t
end

"""
    mlirOperationGetResult(op, pos)

Returns `pos`-th result of the operation.
"""
function mlirOperationGetResult(op, pos)
    @ccall (MLIR_C_PATH[]).mlirOperationGetResult(op::MlirOperation, pos::intptr_t)::MlirValue
end

"""
    mlirOperationGetNumSuccessors(op)

Returns the number of successor blocks of the operation.
"""
function mlirOperationGetNumSuccessors(op)
    @ccall (MLIR_C_PATH[]).mlirOperationGetNumSuccessors(op::MlirOperation)::intptr_t
end

"""
    mlirOperationGetSuccessor(op, pos)

Returns `pos`-th successor of the operation.
"""
function mlirOperationGetSuccessor(op, pos)
    @ccall (MLIR_C_PATH[]).mlirOperationGetSuccessor(op::MlirOperation, pos::intptr_t)::MlirBlock
end

"""
    mlirOperationGetNumAttributes(op)

Returns the number of attributes attached to the operation.
"""
function mlirOperationGetNumAttributes(op)
    @ccall (MLIR_C_PATH[]).mlirOperationGetNumAttributes(op::MlirOperation)::intptr_t
end

"""
    mlirOperationGetAttribute(op, pos)

Return `pos`-th attribute of the operation.
"""
function mlirOperationGetAttribute(op, pos)
    @ccall (MLIR_C_PATH[]).mlirOperationGetAttribute(op::MlirOperation, pos::intptr_t)::MlirNamedAttribute
end

"""
    mlirOperationGetAttributeByName(op, name)

Returns an attribute attached to the operation given its name.
"""
function mlirOperationGetAttributeByName(op, name)
    @ccall (MLIR_C_PATH[]).mlirOperationGetAttributeByName(op::MlirOperation, name::MlirStringRef)::MlirAttribute
end

"""
    mlirOperationSetAttributeByName(op, name, attr)

Sets an attribute by name, replacing the existing if it exists or adding a new one otherwise.
"""
function mlirOperationSetAttributeByName(op, name, attr)
    @ccall (MLIR_C_PATH[]).mlirOperationSetAttributeByName(op::MlirOperation, name::MlirStringRef, attr::MlirAttribute)::Cvoid
end

"""
    mlirOperationRemoveAttributeByName(op, name)

Removes an attribute by name. Returns false if the attribute was not found and true if removed.
"""
function mlirOperationRemoveAttributeByName(op, name)
    @ccall (MLIR_C_PATH[]).mlirOperationRemoveAttributeByName(op::MlirOperation, name::MlirStringRef)::Bool
end

"""
    mlirOperationPrint(op, callback, userData)

Prints an operation by sending chunks of the string representation and forwarding `userData to `callback`. Note that the callback may be called several times with consecutive chunks of the string.
"""
function mlirOperationPrint(op, callback, userData)
    @ccall (MLIR_C_PATH[]).mlirOperationPrint(op::MlirOperation, callback::MlirStringCallback, userData::Ptr{Cvoid})::Cvoid
end

"""
    mlirOperationPrintWithFlags(op, flags, callback, userData)

Same as [`mlirOperationPrint`](@ref) but accepts flags controlling the printing behavior.
"""
function mlirOperationPrintWithFlags(op, flags, callback, userData)
    @ccall (MLIR_C_PATH[]).mlirOperationPrintWithFlags(op::MlirOperation, flags::MlirOpPrintingFlags, callback::MlirStringCallback, userData::Ptr{Cvoid})::Cvoid
end

"""
    mlirOperationWriteBytecode(op, callback, userData)

Same as [`mlirOperationPrint`](@ref) but writing the bytecode format.
"""
function mlirOperationWriteBytecode(op, callback, userData)
    @ccall (MLIR_C_PATH[]).mlirOperationWriteBytecode(op::MlirOperation, callback::MlirStringCallback, userData::Ptr{Cvoid})::Cvoid
end

"""
    mlirOperationWriteBytecodeWithConfig(op, config, callback, userData)

Same as [`mlirOperationWriteBytecode`](@ref) but with writer config and returns failure only if desired bytecode could not be honored.
"""
function mlirOperationWriteBytecodeWithConfig(op, config, callback, userData)
    @ccall (MLIR_C_PATH[]).mlirOperationWriteBytecodeWithConfig(op::MlirOperation, config::MlirBytecodeWriterConfig, callback::MlirStringCallback, userData::Ptr{Cvoid})::MlirLogicalResult
end

"""
    mlirOperationDump(op)

Prints an operation to stderr.
"""
function mlirOperationDump(op)
    @ccall (MLIR_C_PATH[]).mlirOperationDump(op::MlirOperation)::Cvoid
end

"""
    mlirOperationVerify(op)

Verify the operation and return true if it passes, false if it fails.
"""
function mlirOperationVerify(op)
    @ccall (MLIR_C_PATH[]).mlirOperationVerify(op::MlirOperation)::Bool
end

"""
    mlirOperationMoveAfter(op, other)

Moves the given operation immediately after the other operation in its parent block. The given operation may be owned by the caller or by its current block. The other operation must belong to a block. In any case, the ownership is transferred to the block of the other operation.
"""
function mlirOperationMoveAfter(op, other)
    @ccall (MLIR_C_PATH[]).mlirOperationMoveAfter(op::MlirOperation, other::MlirOperation)::Cvoid
end

"""
    mlirOperationMoveBefore(op, other)

Moves the given operation immediately before the other operation in its parent block. The given operation may be owner by the caller or by its current block. The other operation must belong to a block. In any case, the ownership is transferred to the block of the other operation.
"""
function mlirOperationMoveBefore(op, other)
    @ccall (MLIR_C_PATH[]).mlirOperationMoveBefore(op::MlirOperation, other::MlirOperation)::Cvoid
end

"""
    mlirRegionCreate()

Creates a new empty region and transfers ownership to the caller.
"""
function mlirRegionCreate()
    @ccall (MLIR_C_PATH[]).mlirRegionCreate()::MlirRegion
end

"""
    mlirRegionDestroy(region)

Takes a region owned by the caller and destroys it.
"""
function mlirRegionDestroy(region)
    @ccall (MLIR_C_PATH[]).mlirRegionDestroy(region::MlirRegion)::Cvoid
end

"""
    mlirRegionIsNull(region)

Checks whether a region is null.
"""
function mlirRegionIsNull(region)
    @ccall (MLIR_C_PATH[]).mlirRegionIsNull(region::MlirRegion)::Bool
end

"""
    mlirRegionEqual(region, other)

Checks whether two region handles point to the same region. This does not perform deep comparison.
"""
function mlirRegionEqual(region, other)
    @ccall (MLIR_C_PATH[]).mlirRegionEqual(region::MlirRegion, other::MlirRegion)::Bool
end

"""
    mlirRegionGetFirstBlock(region)

Gets the first block in the region.
"""
function mlirRegionGetFirstBlock(region)
    @ccall (MLIR_C_PATH[]).mlirRegionGetFirstBlock(region::MlirRegion)::MlirBlock
end

"""
    mlirRegionAppendOwnedBlock(region, block)

Takes a block owned by the caller and appends it to the given region.
"""
function mlirRegionAppendOwnedBlock(region, block)
    @ccall (MLIR_C_PATH[]).mlirRegionAppendOwnedBlock(region::MlirRegion, block::MlirBlock)::Cvoid
end

"""
    mlirRegionInsertOwnedBlock(region, pos, block)

Takes a block owned by the caller and inserts it at `pos` to the given region. This is an expensive operation that linearly scans the region, prefer insertAfter/Before instead.
"""
function mlirRegionInsertOwnedBlock(region, pos, block)
    @ccall (MLIR_C_PATH[]).mlirRegionInsertOwnedBlock(region::MlirRegion, pos::intptr_t, block::MlirBlock)::Cvoid
end

"""
    mlirRegionInsertOwnedBlockAfter(region, reference, block)

Takes a block owned by the caller and inserts it after the (non-owned) reference block in the given region. The reference block must belong to the region. If the reference block is null, prepends the block to the region.
"""
function mlirRegionInsertOwnedBlockAfter(region, reference, block)
    @ccall (MLIR_C_PATH[]).mlirRegionInsertOwnedBlockAfter(region::MlirRegion, reference::MlirBlock, block::MlirBlock)::Cvoid
end

"""
    mlirRegionInsertOwnedBlockBefore(region, reference, block)

Takes a block owned by the caller and inserts it before the (non-owned) reference block in the given region. The reference block must belong to the region. If the reference block is null, appends the block to the region.
"""
function mlirRegionInsertOwnedBlockBefore(region, reference, block)
    @ccall (MLIR_C_PATH[]).mlirRegionInsertOwnedBlockBefore(region::MlirRegion, reference::MlirBlock, block::MlirBlock)::Cvoid
end

"""
    mlirOperationGetFirstRegion(op)

Returns first region attached to the operation.
"""
function mlirOperationGetFirstRegion(op)
    @ccall (MLIR_C_PATH[]).mlirOperationGetFirstRegion(op::MlirOperation)::MlirRegion
end

"""
    mlirRegionGetNextInOperation(region)

Returns the region immediately following the given region in its parent operation.
"""
function mlirRegionGetNextInOperation(region)
    @ccall (MLIR_C_PATH[]).mlirRegionGetNextInOperation(region::MlirRegion)::MlirRegion
end

"""
    mlirRegionTakeBody(target, source)

Moves the entire content of the source region to the target region.
"""
function mlirRegionTakeBody(target, source)
    @ccall (MLIR_C_PATH[]).mlirRegionTakeBody(target::MlirRegion, source::MlirRegion)::Cvoid
end

"""
    mlirBlockCreate(nArgs, args, locs)

Creates a new empty block with the given argument types and transfers ownership to the caller.
"""
function mlirBlockCreate(nArgs, args, locs)
    @ccall (MLIR_C_PATH[]).mlirBlockCreate(nArgs::intptr_t, args::Ptr{MlirType}, locs::Ptr{MlirLocation})::MlirBlock
end

"""
    mlirBlockDestroy(block)

Takes a block owned by the caller and destroys it.
"""
function mlirBlockDestroy(block)
    @ccall (MLIR_C_PATH[]).mlirBlockDestroy(block::MlirBlock)::Cvoid
end

"""
    mlirBlockDetach(block)

Detach a block from the owning region and assume ownership.
"""
function mlirBlockDetach(block)
    @ccall (MLIR_C_PATH[]).mlirBlockDetach(block::MlirBlock)::Cvoid
end

"""
    mlirBlockIsNull(block)

Checks whether a block is null.
"""
function mlirBlockIsNull(block)
    @ccall (MLIR_C_PATH[]).mlirBlockIsNull(block::MlirBlock)::Bool
end

"""
    mlirBlockEqual(block, other)

Checks whether two blocks handles point to the same block. This does not perform deep comparison.
"""
function mlirBlockEqual(block, other)
    @ccall (MLIR_C_PATH[]).mlirBlockEqual(block::MlirBlock, other::MlirBlock)::Bool
end

"""
    mlirBlockGetParentOperation(arg1)

Returns the closest surrounding operation that contains this block.
"""
function mlirBlockGetParentOperation(arg1)
    @ccall (MLIR_C_PATH[]).mlirBlockGetParentOperation(arg1::MlirBlock)::MlirOperation
end

"""
    mlirBlockGetParentRegion(block)

Returns the region that contains this block.
"""
function mlirBlockGetParentRegion(block)
    @ccall (MLIR_C_PATH[]).mlirBlockGetParentRegion(block::MlirBlock)::MlirRegion
end

"""
    mlirBlockGetNextInRegion(block)

Returns the block immediately following the given block in its parent region.
"""
function mlirBlockGetNextInRegion(block)
    @ccall (MLIR_C_PATH[]).mlirBlockGetNextInRegion(block::MlirBlock)::MlirBlock
end

"""
    mlirBlockGetFirstOperation(block)

Returns the first operation in the block.
"""
function mlirBlockGetFirstOperation(block)
    @ccall (MLIR_C_PATH[]).mlirBlockGetFirstOperation(block::MlirBlock)::MlirOperation
end

"""
    mlirBlockGetTerminator(block)

Returns the terminator operation in the block or null if no terminator.
"""
function mlirBlockGetTerminator(block)
    @ccall (MLIR_C_PATH[]).mlirBlockGetTerminator(block::MlirBlock)::MlirOperation
end

"""
    mlirBlockAppendOwnedOperation(block, operation)

Takes an operation owned by the caller and appends it to the block.
"""
function mlirBlockAppendOwnedOperation(block, operation)
    @ccall (MLIR_C_PATH[]).mlirBlockAppendOwnedOperation(block::MlirBlock, operation::MlirOperation)::Cvoid
end

"""
    mlirBlockInsertOwnedOperation(block, pos, operation)

Takes an operation owned by the caller and inserts it as `pos` to the block. This is an expensive operation that scans the block linearly, prefer insertBefore/After instead.
"""
function mlirBlockInsertOwnedOperation(block, pos, operation)
    @ccall (MLIR_C_PATH[]).mlirBlockInsertOwnedOperation(block::MlirBlock, pos::intptr_t, operation::MlirOperation)::Cvoid
end

"""
    mlirBlockInsertOwnedOperationAfter(block, reference, operation)

Takes an operation owned by the caller and inserts it after the (non-owned) reference operation in the given block. If the reference is null, prepends the operation. Otherwise, the reference must belong to the block.
"""
function mlirBlockInsertOwnedOperationAfter(block, reference, operation)
    @ccall (MLIR_C_PATH[]).mlirBlockInsertOwnedOperationAfter(block::MlirBlock, reference::MlirOperation, operation::MlirOperation)::Cvoid
end

"""
    mlirBlockInsertOwnedOperationBefore(block, reference, operation)

Takes an operation owned by the caller and inserts it before the (non-owned) reference operation in the given block. If the reference is null, appends the operation. Otherwise, the reference must belong to the block.
"""
function mlirBlockInsertOwnedOperationBefore(block, reference, operation)
    @ccall (MLIR_C_PATH[]).mlirBlockInsertOwnedOperationBefore(block::MlirBlock, reference::MlirOperation, operation::MlirOperation)::Cvoid
end

"""
    mlirBlockGetNumArguments(block)

Returns the number of arguments of the block.
"""
function mlirBlockGetNumArguments(block)
    @ccall (MLIR_C_PATH[]).mlirBlockGetNumArguments(block::MlirBlock)::intptr_t
end

"""
    mlirBlockAddArgument(block, type, loc)

Appends an argument of the specified type to the block. Returns the newly added argument.
"""
function mlirBlockAddArgument(block, type, loc)
    @ccall (MLIR_C_PATH[]).mlirBlockAddArgument(block::MlirBlock, type::MlirType, loc::MlirLocation)::MlirValue
end

"""
    mlirBlockInsertArgument(block, pos, type, loc)

Inserts an argument of the specified type at a specified index to the block. Returns the newly added argument.
"""
function mlirBlockInsertArgument(block, pos, type, loc)
    @ccall (MLIR_C_PATH[]).mlirBlockInsertArgument(block::MlirBlock, pos::intptr_t, type::MlirType, loc::MlirLocation)::MlirValue
end

"""
    mlirBlockGetArgument(block, pos)

Returns `pos`-th argument of the block.
"""
function mlirBlockGetArgument(block, pos)
    @ccall (MLIR_C_PATH[]).mlirBlockGetArgument(block::MlirBlock, pos::intptr_t)::MlirValue
end

"""
    mlirBlockPrint(block, callback, userData)

Prints a block by sending chunks of the string representation and forwarding `userData to `callback`. Note that the callback may be called several times with consecutive chunks of the string.
"""
function mlirBlockPrint(block, callback, userData)
    @ccall (MLIR_C_PATH[]).mlirBlockPrint(block::MlirBlock, callback::MlirStringCallback, userData::Ptr{Cvoid})::Cvoid
end

"""
    mlirValueIsNull(value)

Returns whether the value is null.
"""
function mlirValueIsNull(value)
    @ccall (MLIR_C_PATH[]).mlirValueIsNull(value::MlirValue)::Bool
end

"""
    mlirValueEqual(value1, value2)

Returns 1 if two values are equal, 0 otherwise.
"""
function mlirValueEqual(value1, value2)
    @ccall (MLIR_C_PATH[]).mlirValueEqual(value1::MlirValue, value2::MlirValue)::Bool
end

"""
    mlirValueIsABlockArgument(value)

Returns 1 if the value is a block argument, 0 otherwise.
"""
function mlirValueIsABlockArgument(value)
    @ccall (MLIR_C_PATH[]).mlirValueIsABlockArgument(value::MlirValue)::Bool
end

"""
    mlirValueIsAOpResult(value)

Returns 1 if the value is an operation result, 0 otherwise.
"""
function mlirValueIsAOpResult(value)
    @ccall (MLIR_C_PATH[]).mlirValueIsAOpResult(value::MlirValue)::Bool
end

"""
    mlirBlockArgumentGetOwner(value)

Returns the block in which this value is defined as an argument. Asserts if the value is not a block argument.
"""
function mlirBlockArgumentGetOwner(value)
    @ccall (MLIR_C_PATH[]).mlirBlockArgumentGetOwner(value::MlirValue)::MlirBlock
end

"""
    mlirBlockArgumentGetArgNumber(value)

Returns the position of the value in the argument list of its block.
"""
function mlirBlockArgumentGetArgNumber(value)
    @ccall (MLIR_C_PATH[]).mlirBlockArgumentGetArgNumber(value::MlirValue)::intptr_t
end

"""
    mlirBlockArgumentSetType(value, type)

Sets the type of the block argument to the given type.
"""
function mlirBlockArgumentSetType(value, type)
    @ccall (MLIR_C_PATH[]).mlirBlockArgumentSetType(value::MlirValue, type::MlirType)::Cvoid
end

"""
    mlirOpResultGetOwner(value)

Returns an operation that produced this value as its result. Asserts if the value is not an op result.
"""
function mlirOpResultGetOwner(value)
    @ccall (MLIR_C_PATH[]).mlirOpResultGetOwner(value::MlirValue)::MlirOperation
end

"""
    mlirOpResultGetResultNumber(value)

Returns the position of the value in the list of results of the operation that produced it.
"""
function mlirOpResultGetResultNumber(value)
    @ccall (MLIR_C_PATH[]).mlirOpResultGetResultNumber(value::MlirValue)::intptr_t
end

"""
    mlirValueGetType(value)

Returns the type of the value.
"""
function mlirValueGetType(value)
    @ccall (MLIR_C_PATH[]).mlirValueGetType(value::MlirValue)::MlirType
end

"""
    mlirValueDump(value)

Prints the value to the standard error stream.
"""
function mlirValueDump(value)
    @ccall (MLIR_C_PATH[]).mlirValueDump(value::MlirValue)::Cvoid
end

"""
    mlirValuePrint(value, callback, userData)

Prints a value by sending chunks of the string representation and forwarding `userData to `callback`. Note that the callback may be called several times with consecutive chunks of the string.
"""
function mlirValuePrint(value, callback, userData)
    @ccall (MLIR_C_PATH[]).mlirValuePrint(value::MlirValue, callback::MlirStringCallback, userData::Ptr{Cvoid})::Cvoid
end

"""
    mlirValuePrintAsOperand(value, flags, callback, userData)

Prints a value as an operand (i.e., the ValueID).
"""
function mlirValuePrintAsOperand(value, flags, callback, userData)
    @ccall (MLIR_C_PATH[]).mlirValuePrintAsOperand(value::MlirValue, flags::MlirOpPrintingFlags, callback::MlirStringCallback, userData::Ptr{Cvoid})::Cvoid
end

"""
    mlirValueGetFirstUse(value)

Returns an op operand representing the first use of the value, or a null op operand if there are no uses.
"""
function mlirValueGetFirstUse(value)
    @ccall (MLIR_C_PATH[]).mlirValueGetFirstUse(value::MlirValue)::MlirOpOperand
end

"""
    mlirValueReplaceAllUsesOfWith(of, with)

Replace all uses of 'of' value with the 'with' value, updating anything in the IR that uses 'of' to use the other value instead. When this returns there are zero uses of 'of'.
"""
function mlirValueReplaceAllUsesOfWith(of, with)
    @ccall (MLIR_C_PATH[]).mlirValueReplaceAllUsesOfWith(of::MlirValue, with::MlirValue)::Cvoid
end

"""
    mlirOpOperandIsNull(opOperand)

Returns whether the op operand is null.
"""
function mlirOpOperandIsNull(opOperand)
    @ccall (MLIR_C_PATH[]).mlirOpOperandIsNull(opOperand::MlirOpOperand)::Bool
end

"""
    mlirOpOperandGetOwner(opOperand)

Returns the owner operation of an op operand.
"""
function mlirOpOperandGetOwner(opOperand)
    @ccall (MLIR_C_PATH[]).mlirOpOperandGetOwner(opOperand::MlirOpOperand)::MlirOperation
end

"""
    mlirOpOperandGetOperandNumber(opOperand)

Returns the operand number of an op operand.
"""
function mlirOpOperandGetOperandNumber(opOperand)
    @ccall (MLIR_C_PATH[]).mlirOpOperandGetOperandNumber(opOperand::MlirOpOperand)::Cuint
end

"""
    mlirOpOperandGetNextUse(opOperand)

Returns an op operand representing the next use of the value, or a null op operand if there is no next use.
"""
function mlirOpOperandGetNextUse(opOperand)
    @ccall (MLIR_C_PATH[]).mlirOpOperandGetNextUse(opOperand::MlirOpOperand)::MlirOpOperand
end

"""
    mlirTypeParseGet(context, type)

Parses a type. The type is owned by the context.
"""
function mlirTypeParseGet(context, type)
    @ccall (MLIR_C_PATH[]).mlirTypeParseGet(context::MlirContext, type::MlirStringRef)::MlirType
end

"""
    mlirTypeGetContext(type)

Gets the context that a type was created with.
"""
function mlirTypeGetContext(type)
    @ccall (MLIR_C_PATH[]).mlirTypeGetContext(type::MlirType)::MlirContext
end

"""
    mlirTypeGetTypeID(type)

Gets the type ID of the type.
"""
function mlirTypeGetTypeID(type)
    @ccall (MLIR_C_PATH[]).mlirTypeGetTypeID(type::MlirType)::MlirTypeID
end

"""
    mlirTypeGetDialect(type)

Gets the dialect a type belongs to.
"""
function mlirTypeGetDialect(type)
    @ccall (MLIR_C_PATH[]).mlirTypeGetDialect(type::MlirType)::MlirDialect
end

"""
    mlirTypeIsNull(type)

Checks whether a type is null.
"""
function mlirTypeIsNull(type)
    @ccall (MLIR_C_PATH[]).mlirTypeIsNull(type::MlirType)::Bool
end

"""
    mlirTypeEqual(t1, t2)

Checks if two types are equal.
"""
function mlirTypeEqual(t1, t2)
    @ccall (MLIR_C_PATH[]).mlirTypeEqual(t1::MlirType, t2::MlirType)::Bool
end

"""
    mlirTypePrint(type, callback, userData)

Prints a location by sending chunks of the string representation and forwarding `userData to `callback`. Note that the callback may be called several times with consecutive chunks of the string.
"""
function mlirTypePrint(type, callback, userData)
    @ccall (MLIR_C_PATH[]).mlirTypePrint(type::MlirType, callback::MlirStringCallback, userData::Ptr{Cvoid})::Cvoid
end

"""
    mlirTypeDump(type)

Prints the type to the standard error stream.
"""
function mlirTypeDump(type)
    @ccall (MLIR_C_PATH[]).mlirTypeDump(type::MlirType)::Cvoid
end

"""
    mlirAttributeParseGet(context, attr)

Parses an attribute. The attribute is owned by the context.
"""
function mlirAttributeParseGet(context, attr)
    @ccall (MLIR_C_PATH[]).mlirAttributeParseGet(context::MlirContext, attr::MlirStringRef)::MlirAttribute
end

"""
    mlirAttributeGetContext(attribute)

Gets the context that an attribute was created with.
"""
function mlirAttributeGetContext(attribute)
    @ccall (MLIR_C_PATH[]).mlirAttributeGetContext(attribute::MlirAttribute)::MlirContext
end

"""
    mlirAttributeGetType(attribute)

Gets the type of this attribute.
"""
function mlirAttributeGetType(attribute)
    @ccall (MLIR_C_PATH[]).mlirAttributeGetType(attribute::MlirAttribute)::MlirType
end

"""
    mlirAttributeGetTypeID(attribute)

Gets the type id of the attribute.
"""
function mlirAttributeGetTypeID(attribute)
    @ccall (MLIR_C_PATH[]).mlirAttributeGetTypeID(attribute::MlirAttribute)::MlirTypeID
end

"""
    mlirAttributeGetDialect(attribute)

Gets the dialect of the attribute.
"""
function mlirAttributeGetDialect(attribute)
    @ccall (MLIR_C_PATH[]).mlirAttributeGetDialect(attribute::MlirAttribute)::MlirDialect
end

"""
    mlirAttributeIsNull(attr)

Checks whether an attribute is null.
"""
function mlirAttributeIsNull(attr)
    @ccall (MLIR_C_PATH[]).mlirAttributeIsNull(attr::MlirAttribute)::Bool
end

"""
    mlirAttributeEqual(a1, a2)

Checks if two attributes are equal.
"""
function mlirAttributeEqual(a1, a2)
    @ccall (MLIR_C_PATH[]).mlirAttributeEqual(a1::MlirAttribute, a2::MlirAttribute)::Bool
end

"""
    mlirAttributePrint(attr, callback, userData)

Prints an attribute by sending chunks of the string representation and forwarding `userData to `callback`. Note that the callback may be called several times with consecutive chunks of the string.
"""
function mlirAttributePrint(attr, callback, userData)
    @ccall (MLIR_C_PATH[]).mlirAttributePrint(attr::MlirAttribute, callback::MlirStringCallback, userData::Ptr{Cvoid})::Cvoid
end

"""
    mlirAttributeDump(attr)

Prints the attribute to the standard error stream.
"""
function mlirAttributeDump(attr)
    @ccall (MLIR_C_PATH[]).mlirAttributeDump(attr::MlirAttribute)::Cvoid
end

"""
    mlirNamedAttributeGet(name, attr)

Associates an attribute with the name. Takes ownership of neither.
"""
function mlirNamedAttributeGet(name, attr)
    @ccall (MLIR_C_PATH[]).mlirNamedAttributeGet(name::MlirIdentifier, attr::MlirAttribute)::MlirNamedAttribute
end

"""
    mlirIdentifierGet(context, str)

Gets an identifier with the given string value.
"""
function mlirIdentifierGet(context, str)
    @ccall (MLIR_C_PATH[]).mlirIdentifierGet(context::MlirContext, str::MlirStringRef)::MlirIdentifier
end

"""
    mlirIdentifierGetContext(arg1)

Returns the context associated with this identifier
"""
function mlirIdentifierGetContext(arg1)
    @ccall (MLIR_C_PATH[]).mlirIdentifierGetContext(arg1::MlirIdentifier)::MlirContext
end

"""
    mlirIdentifierEqual(ident, other)

Checks whether two identifiers are the same.
"""
function mlirIdentifierEqual(ident, other)
    @ccall (MLIR_C_PATH[]).mlirIdentifierEqual(ident::MlirIdentifier, other::MlirIdentifier)::Bool
end

"""
    mlirIdentifierStr(ident)

Gets the string value of the identifier.
"""
function mlirIdentifierStr(ident)
    @ccall (MLIR_C_PATH[]).mlirIdentifierStr(ident::MlirIdentifier)::MlirStringRef
end

"""
    mlirSymbolTableGetSymbolAttributeName()

Returns the name of the attribute used to store symbol names compatible with symbol tables.
"""
function mlirSymbolTableGetSymbolAttributeName()
    @ccall (MLIR_C_PATH[]).mlirSymbolTableGetSymbolAttributeName()::MlirStringRef
end

"""
    mlirSymbolTableGetVisibilityAttributeName()

Returns the name of the attribute used to store symbol visibility.
"""
function mlirSymbolTableGetVisibilityAttributeName()
    @ccall (MLIR_C_PATH[]).mlirSymbolTableGetVisibilityAttributeName()::MlirStringRef
end

"""
    mlirSymbolTableCreate(operation)

Creates a symbol table for the given operation. If the operation does not have the SymbolTable trait, returns a null symbol table.
"""
function mlirSymbolTableCreate(operation)
    @ccall (MLIR_C_PATH[]).mlirSymbolTableCreate(operation::MlirOperation)::MlirSymbolTable
end

"""
    mlirSymbolTableIsNull(symbolTable)

Returns true if the symbol table is null.
"""
function mlirSymbolTableIsNull(symbolTable)
    @ccall (MLIR_C_PATH[]).mlirSymbolTableIsNull(symbolTable::MlirSymbolTable)::Bool
end

"""
    mlirSymbolTableDestroy(symbolTable)

Destroys the symbol table created with [`mlirSymbolTableCreate`](@ref). This does not affect the operations in the table.
"""
function mlirSymbolTableDestroy(symbolTable)
    @ccall (MLIR_C_PATH[]).mlirSymbolTableDestroy(symbolTable::MlirSymbolTable)::Cvoid
end

"""
    mlirSymbolTableLookup(symbolTable, name)

Looks up a symbol with the given name in the given symbol table and returns the operation that corresponds to the symbol. If the symbol cannot be found, returns a null operation.
"""
function mlirSymbolTableLookup(symbolTable, name)
    @ccall (MLIR_C_PATH[]).mlirSymbolTableLookup(symbolTable::MlirSymbolTable, name::MlirStringRef)::MlirOperation
end

"""
    mlirSymbolTableInsert(symbolTable, operation)

Inserts the given operation into the given symbol table. The operation must have the symbol trait. If the symbol table already has a symbol with the same name, renames the symbol being inserted to ensure name uniqueness. Note that this does not move the operation itself into the block of the symbol table operation, this should be done separately. Returns the name of the symbol after insertion.
"""
function mlirSymbolTableInsert(symbolTable, operation)
    @ccall (MLIR_C_PATH[]).mlirSymbolTableInsert(symbolTable::MlirSymbolTable, operation::MlirOperation)::MlirAttribute
end

"""
    mlirSymbolTableErase(symbolTable, operation)

Removes the given operation from the symbol table and erases it.
"""
function mlirSymbolTableErase(symbolTable, operation)
    @ccall (MLIR_C_PATH[]).mlirSymbolTableErase(symbolTable::MlirSymbolTable, operation::MlirOperation)::Cvoid
end

"""
    mlirSymbolTableReplaceAllSymbolUses(oldSymbol, newSymbol, from)

Attempt to replace all uses that are nested within the given operation of the given symbol 'oldSymbol' with the provided 'newSymbol'. This does not traverse into nested symbol tables. Will fail atomically if there are any unknown operations that may be potential symbol tables.
"""
function mlirSymbolTableReplaceAllSymbolUses(oldSymbol, newSymbol, from)
    @ccall (MLIR_C_PATH[]).mlirSymbolTableReplaceAllSymbolUses(oldSymbol::MlirStringRef, newSymbol::MlirStringRef, from::MlirOperation)::MlirLogicalResult
end

"""
    mlirSymbolTableWalkSymbolTables(from, allSymUsesVisible, callback, userData)

Walks all symbol table operations nested within, and including, `op`. For each symbol table operation, the provided callback is invoked with the op and a boolean signifying if the symbols within that symbol table can be treated as if all uses within the IR are visible to the caller. `allSymUsesVisible` identifies whether all of the symbol uses of symbols within `op` are visible.
"""
function mlirSymbolTableWalkSymbolTables(from, allSymUsesVisible, callback, userData)
    @ccall (MLIR_C_PATH[]).mlirSymbolTableWalkSymbolTables(from::MlirOperation, allSymUsesVisible::Bool, callback::Ptr{Cvoid}, userData::Ptr{Cvoid})::Cvoid
end

"""
    mlirAffineExprGetContext(affineExpr)

Gets the context that owns the affine expression.
"""
function mlirAffineExprGetContext(affineExpr)
    @ccall (MLIR_C_PATH[]).mlirAffineExprGetContext(affineExpr::MlirAffineExpr)::MlirContext
end

"""
    mlirAffineExprEqual(lhs, rhs)

Returns `true` if the two affine expressions are equal.
"""
function mlirAffineExprEqual(lhs, rhs)
    @ccall (MLIR_C_PATH[]).mlirAffineExprEqual(lhs::MlirAffineExpr, rhs::MlirAffineExpr)::Bool
end

"""
    mlirAffineExprIsNull(affineExpr)

Returns `true` if the given affine expression is a null expression. Note constant zero is not a null expression.
"""
function mlirAffineExprIsNull(affineExpr)
    @ccall (MLIR_C_PATH[]).mlirAffineExprIsNull(affineExpr::MlirAffineExpr)::Bool
end

"""
    mlirAffineExprPrint(affineExpr, callback, userData)

Prints an affine expression by sending chunks of the string representation and forwarding `userData to `callback`. Note that the callback may be called several times with consecutive chunks of the string.
"""
function mlirAffineExprPrint(affineExpr, callback, userData)
    @ccall (MLIR_C_PATH[]).mlirAffineExprPrint(affineExpr::MlirAffineExpr, callback::MlirStringCallback, userData::Ptr{Cvoid})::Cvoid
end

"""
    mlirAffineExprDump(affineExpr)

Prints the affine expression to the standard error stream.
"""
function mlirAffineExprDump(affineExpr)
    @ccall (MLIR_C_PATH[]).mlirAffineExprDump(affineExpr::MlirAffineExpr)::Cvoid
end

"""
    mlirAffineExprIsSymbolicOrConstant(affineExpr)

Checks whether the given affine expression is made out of only symbols and constants.
"""
function mlirAffineExprIsSymbolicOrConstant(affineExpr)
    @ccall (MLIR_C_PATH[]).mlirAffineExprIsSymbolicOrConstant(affineExpr::MlirAffineExpr)::Bool
end

"""
    mlirAffineExprIsPureAffine(affineExpr)

Checks whether the given affine expression is a pure affine expression, i.e. mul, floordiv, ceildic, and mod is only allowed w.r.t constants.
"""
function mlirAffineExprIsPureAffine(affineExpr)
    @ccall (MLIR_C_PATH[]).mlirAffineExprIsPureAffine(affineExpr::MlirAffineExpr)::Bool
end

"""
    mlirAffineExprGetLargestKnownDivisor(affineExpr)

Returns the greatest known integral divisor of this affine expression. The result is always positive.
"""
function mlirAffineExprGetLargestKnownDivisor(affineExpr)
    @ccall (MLIR_C_PATH[]).mlirAffineExprGetLargestKnownDivisor(affineExpr::MlirAffineExpr)::Int64
end

"""
    mlirAffineExprIsMultipleOf(affineExpr, factor)

Checks whether the given affine expression is a multiple of 'factor'.
"""
function mlirAffineExprIsMultipleOf(affineExpr, factor)
    @ccall (MLIR_C_PATH[]).mlirAffineExprIsMultipleOf(affineExpr::MlirAffineExpr, factor::Int64)::Bool
end

"""
    mlirAffineExprIsFunctionOfDim(affineExpr, position)

Checks whether the given affine expression involves AffineDimExpr 'position'.
"""
function mlirAffineExprIsFunctionOfDim(affineExpr, position)
    @ccall (MLIR_C_PATH[]).mlirAffineExprIsFunctionOfDim(affineExpr::MlirAffineExpr, position::intptr_t)::Bool
end

"""
    mlirAffineExprCompose(affineExpr, affineMap)

Composes the given map with the given expression.
"""
function mlirAffineExprCompose(affineExpr, affineMap)
    @ccall (MLIR_C_PATH[]).mlirAffineExprCompose(affineExpr::MlirAffineExpr, affineMap::MlirAffineMap)::MlirAffineExpr
end

"""
    mlirAffineExprIsADim(affineExpr)

Checks whether the given affine expression is a dimension expression.
"""
function mlirAffineExprIsADim(affineExpr)
    @ccall (MLIR_C_PATH[]).mlirAffineExprIsADim(affineExpr::MlirAffineExpr)::Bool
end

"""
    mlirAffineDimExprGet(ctx, position)

Creates an affine dimension expression with 'position' in the context.
"""
function mlirAffineDimExprGet(ctx, position)
    @ccall (MLIR_C_PATH[]).mlirAffineDimExprGet(ctx::MlirContext, position::intptr_t)::MlirAffineExpr
end

"""
    mlirAffineDimExprGetPosition(affineExpr)

Returns the position of the given affine dimension expression.
"""
function mlirAffineDimExprGetPosition(affineExpr)
    @ccall (MLIR_C_PATH[]).mlirAffineDimExprGetPosition(affineExpr::MlirAffineExpr)::intptr_t
end

"""
    mlirAffineExprIsASymbol(affineExpr)

Checks whether the given affine expression is a symbol expression.
"""
function mlirAffineExprIsASymbol(affineExpr)
    @ccall (MLIR_C_PATH[]).mlirAffineExprIsASymbol(affineExpr::MlirAffineExpr)::Bool
end

"""
    mlirAffineSymbolExprGet(ctx, position)

Creates an affine symbol expression with 'position' in the context.
"""
function mlirAffineSymbolExprGet(ctx, position)
    @ccall (MLIR_C_PATH[]).mlirAffineSymbolExprGet(ctx::MlirContext, position::intptr_t)::MlirAffineExpr
end

"""
    mlirAffineSymbolExprGetPosition(affineExpr)

Returns the position of the given affine symbol expression.
"""
function mlirAffineSymbolExprGetPosition(affineExpr)
    @ccall (MLIR_C_PATH[]).mlirAffineSymbolExprGetPosition(affineExpr::MlirAffineExpr)::intptr_t
end

"""
    mlirAffineExprIsAConstant(affineExpr)

Checks whether the given affine expression is a constant expression.
"""
function mlirAffineExprIsAConstant(affineExpr)
    @ccall (MLIR_C_PATH[]).mlirAffineExprIsAConstant(affineExpr::MlirAffineExpr)::Bool
end

"""
    mlirAffineConstantExprGet(ctx, constant)

Creates an affine constant expression with 'constant' in the context.
"""
function mlirAffineConstantExprGet(ctx, constant)
    @ccall (MLIR_C_PATH[]).mlirAffineConstantExprGet(ctx::MlirContext, constant::Int64)::MlirAffineExpr
end

"""
    mlirAffineConstantExprGetValue(affineExpr)

Returns the value of the given affine constant expression.
"""
function mlirAffineConstantExprGetValue(affineExpr)
    @ccall (MLIR_C_PATH[]).mlirAffineConstantExprGetValue(affineExpr::MlirAffineExpr)::Int64
end

"""
    mlirAffineExprIsAAdd(affineExpr)

Checks whether the given affine expression is an add expression.
"""
function mlirAffineExprIsAAdd(affineExpr)
    @ccall (MLIR_C_PATH[]).mlirAffineExprIsAAdd(affineExpr::MlirAffineExpr)::Bool
end

"""
    mlirAffineAddExprGet(lhs, rhs)

Creates an affine add expression with 'lhs' and 'rhs'.
"""
function mlirAffineAddExprGet(lhs, rhs)
    @ccall (MLIR_C_PATH[]).mlirAffineAddExprGet(lhs::MlirAffineExpr, rhs::MlirAffineExpr)::MlirAffineExpr
end

"""
    mlirAffineExprIsAMul(affineExpr)

Checks whether the given affine expression is an mul expression.
"""
function mlirAffineExprIsAMul(affineExpr)
    @ccall (MLIR_C_PATH[]).mlirAffineExprIsAMul(affineExpr::MlirAffineExpr)::Bool
end

"""
    mlirAffineMulExprGet(lhs, rhs)

Creates an affine mul expression with 'lhs' and 'rhs'.
"""
function mlirAffineMulExprGet(lhs, rhs)
    @ccall (MLIR_C_PATH[]).mlirAffineMulExprGet(lhs::MlirAffineExpr, rhs::MlirAffineExpr)::MlirAffineExpr
end

"""
    mlirAffineExprIsAMod(affineExpr)

Checks whether the given affine expression is an mod expression.
"""
function mlirAffineExprIsAMod(affineExpr)
    @ccall (MLIR_C_PATH[]).mlirAffineExprIsAMod(affineExpr::MlirAffineExpr)::Bool
end

"""
    mlirAffineModExprGet(lhs, rhs)

Creates an affine mod expression with 'lhs' and 'rhs'.
"""
function mlirAffineModExprGet(lhs, rhs)
    @ccall (MLIR_C_PATH[]).mlirAffineModExprGet(lhs::MlirAffineExpr, rhs::MlirAffineExpr)::MlirAffineExpr
end

"""
    mlirAffineExprIsAFloorDiv(affineExpr)

Checks whether the given affine expression is an floordiv expression.
"""
function mlirAffineExprIsAFloorDiv(affineExpr)
    @ccall (MLIR_C_PATH[]).mlirAffineExprIsAFloorDiv(affineExpr::MlirAffineExpr)::Bool
end

"""
    mlirAffineFloorDivExprGet(lhs, rhs)

Creates an affine floordiv expression with 'lhs' and 'rhs'.
"""
function mlirAffineFloorDivExprGet(lhs, rhs)
    @ccall (MLIR_C_PATH[]).mlirAffineFloorDivExprGet(lhs::MlirAffineExpr, rhs::MlirAffineExpr)::MlirAffineExpr
end

"""
    mlirAffineExprIsACeilDiv(affineExpr)

Checks whether the given affine expression is an ceildiv expression.
"""
function mlirAffineExprIsACeilDiv(affineExpr)
    @ccall (MLIR_C_PATH[]).mlirAffineExprIsACeilDiv(affineExpr::MlirAffineExpr)::Bool
end

"""
    mlirAffineCeilDivExprGet(lhs, rhs)

Creates an affine ceildiv expression with 'lhs' and 'rhs'.
"""
function mlirAffineCeilDivExprGet(lhs, rhs)
    @ccall (MLIR_C_PATH[]).mlirAffineCeilDivExprGet(lhs::MlirAffineExpr, rhs::MlirAffineExpr)::MlirAffineExpr
end

"""
    mlirAffineExprIsABinary(affineExpr)

Checks whether the given affine expression is binary.
"""
function mlirAffineExprIsABinary(affineExpr)
    @ccall (MLIR_C_PATH[]).mlirAffineExprIsABinary(affineExpr::MlirAffineExpr)::Bool
end

"""
    mlirAffineBinaryOpExprGetLHS(affineExpr)

Returns the left hand side affine expression of the given affine binary operation expression.
"""
function mlirAffineBinaryOpExprGetLHS(affineExpr)
    @ccall (MLIR_C_PATH[]).mlirAffineBinaryOpExprGetLHS(affineExpr::MlirAffineExpr)::MlirAffineExpr
end

"""
    mlirAffineBinaryOpExprGetRHS(affineExpr)

Returns the right hand side affine expression of the given affine binary operation expression.
"""
function mlirAffineBinaryOpExprGetRHS(affineExpr)
    @ccall (MLIR_C_PATH[]).mlirAffineBinaryOpExprGetRHS(affineExpr::MlirAffineExpr)::MlirAffineExpr
end

"""
    mlirAffineMapGetContext(affineMap)

Gets the context that the given affine map was created with
"""
function mlirAffineMapGetContext(affineMap)
    @ccall (MLIR_C_PATH[]).mlirAffineMapGetContext(affineMap::MlirAffineMap)::MlirContext
end

"""
    mlirAffineMapIsNull(affineMap)

Checks whether an affine map is null.
"""
function mlirAffineMapIsNull(affineMap)
    @ccall (MLIR_C_PATH[]).mlirAffineMapIsNull(affineMap::MlirAffineMap)::Bool
end

"""
    mlirAffineMapEqual(a1, a2)

Checks if two affine maps are equal.
"""
function mlirAffineMapEqual(a1, a2)
    @ccall (MLIR_C_PATH[]).mlirAffineMapEqual(a1::MlirAffineMap, a2::MlirAffineMap)::Bool
end

"""
    mlirAffineMapPrint(affineMap, callback, userData)

Prints an affine map by sending chunks of the string representation and forwarding `userData to `callback`. Note that the callback may be called several times with consecutive chunks of the string.
"""
function mlirAffineMapPrint(affineMap, callback, userData)
    @ccall (MLIR_C_PATH[]).mlirAffineMapPrint(affineMap::MlirAffineMap, callback::MlirStringCallback, userData::Ptr{Cvoid})::Cvoid
end

"""
    mlirAffineMapDump(affineMap)

Prints the affine map to the standard error stream.
"""
function mlirAffineMapDump(affineMap)
    @ccall (MLIR_C_PATH[]).mlirAffineMapDump(affineMap::MlirAffineMap)::Cvoid
end

"""
    mlirAffineMapEmptyGet(ctx)

Creates a zero result affine map with no dimensions or symbols in the context. The affine map is owned by the context.
"""
function mlirAffineMapEmptyGet(ctx)
    @ccall (MLIR_C_PATH[]).mlirAffineMapEmptyGet(ctx::MlirContext)::MlirAffineMap
end

"""
    mlirAffineMapZeroResultGet(ctx, dimCount, symbolCount)

Creates a zero result affine map of the given dimensions and symbols in the context. The affine map is owned by the context.
"""
function mlirAffineMapZeroResultGet(ctx, dimCount, symbolCount)
    @ccall (MLIR_C_PATH[]).mlirAffineMapZeroResultGet(ctx::MlirContext, dimCount::intptr_t, symbolCount::intptr_t)::MlirAffineMap
end

"""
    mlirAffineMapGet(ctx, dimCount, symbolCount, nAffineExprs, affineExprs)

Creates an affine map with results defined by the given list of affine expressions. The map resulting map also has the requested number of input dimensions and symbols, regardless of them being used in the results.
"""
function mlirAffineMapGet(ctx, dimCount, symbolCount, nAffineExprs, affineExprs)
    @ccall (MLIR_C_PATH[]).mlirAffineMapGet(ctx::MlirContext, dimCount::intptr_t, symbolCount::intptr_t, nAffineExprs::intptr_t, affineExprs::Ptr{MlirAffineExpr})::MlirAffineMap
end

"""
    mlirAffineMapConstantGet(ctx, val)

Creates a single constant result affine map in the context. The affine map is owned by the context.
"""
function mlirAffineMapConstantGet(ctx, val)
    @ccall (MLIR_C_PATH[]).mlirAffineMapConstantGet(ctx::MlirContext, val::Int64)::MlirAffineMap
end

"""
    mlirAffineMapMultiDimIdentityGet(ctx, numDims)

Creates an affine map with 'numDims' identity in the context. The affine map is owned by the context.
"""
function mlirAffineMapMultiDimIdentityGet(ctx, numDims)
    @ccall (MLIR_C_PATH[]).mlirAffineMapMultiDimIdentityGet(ctx::MlirContext, numDims::intptr_t)::MlirAffineMap
end

"""
    mlirAffineMapMinorIdentityGet(ctx, dims, results)

Creates an identity affine map on the most minor dimensions in the context. The affine map is owned by the context. The function asserts that the number of dimensions is greater or equal to the number of results.
"""
function mlirAffineMapMinorIdentityGet(ctx, dims, results)
    @ccall (MLIR_C_PATH[]).mlirAffineMapMinorIdentityGet(ctx::MlirContext, dims::intptr_t, results::intptr_t)::MlirAffineMap
end

"""
    mlirAffineMapPermutationGet(ctx, size, permutation)

Creates an affine map with a permutation expression and its size in the context. The permutation expression is a non-empty vector of integers. The elements of the permutation vector must be continuous from 0 and cannot be repeated (i.e. `[1,2,0]` is a valid permutation. `[2,0]` or `[1,1,2]` is an invalid invalid permutation.) The affine map is owned by the context.
"""
function mlirAffineMapPermutationGet(ctx, size, permutation)
    @ccall (MLIR_C_PATH[]).mlirAffineMapPermutationGet(ctx::MlirContext, size::intptr_t, permutation::Ptr{Cuint})::MlirAffineMap
end

"""
    mlirAffineMapIsIdentity(affineMap)

Checks whether the given affine map is an identity affine map. The function asserts that the number of dimensions is greater or equal to the number of results.
"""
function mlirAffineMapIsIdentity(affineMap)
    @ccall (MLIR_C_PATH[]).mlirAffineMapIsIdentity(affineMap::MlirAffineMap)::Bool
end

"""
    mlirAffineMapIsMinorIdentity(affineMap)

Checks whether the given affine map is a minor identity affine map.
"""
function mlirAffineMapIsMinorIdentity(affineMap)
    @ccall (MLIR_C_PATH[]).mlirAffineMapIsMinorIdentity(affineMap::MlirAffineMap)::Bool
end

"""
    mlirAffineMapIsEmpty(affineMap)

Checks whether the given affine map is an empty affine map.
"""
function mlirAffineMapIsEmpty(affineMap)
    @ccall (MLIR_C_PATH[]).mlirAffineMapIsEmpty(affineMap::MlirAffineMap)::Bool
end

"""
    mlirAffineMapIsSingleConstant(affineMap)

Checks whether the given affine map is a single result constant affine map.
"""
function mlirAffineMapIsSingleConstant(affineMap)
    @ccall (MLIR_C_PATH[]).mlirAffineMapIsSingleConstant(affineMap::MlirAffineMap)::Bool
end

"""
    mlirAffineMapGetSingleConstantResult(affineMap)

Returns the constant result of the given affine map. The function asserts that the map has a single constant result.
"""
function mlirAffineMapGetSingleConstantResult(affineMap)
    @ccall (MLIR_C_PATH[]).mlirAffineMapGetSingleConstantResult(affineMap::MlirAffineMap)::Int64
end

"""
    mlirAffineMapGetNumDims(affineMap)

Returns the number of dimensions of the given affine map.
"""
function mlirAffineMapGetNumDims(affineMap)
    @ccall (MLIR_C_PATH[]).mlirAffineMapGetNumDims(affineMap::MlirAffineMap)::intptr_t
end

"""
    mlirAffineMapGetNumSymbols(affineMap)

Returns the number of symbols of the given affine map.
"""
function mlirAffineMapGetNumSymbols(affineMap)
    @ccall (MLIR_C_PATH[]).mlirAffineMapGetNumSymbols(affineMap::MlirAffineMap)::intptr_t
end

"""
    mlirAffineMapGetNumResults(affineMap)

Returns the number of results of the given affine map.
"""
function mlirAffineMapGetNumResults(affineMap)
    @ccall (MLIR_C_PATH[]).mlirAffineMapGetNumResults(affineMap::MlirAffineMap)::intptr_t
end

"""
    mlirAffineMapGetResult(affineMap, pos)

Returns the result at the given position.
"""
function mlirAffineMapGetResult(affineMap, pos)
    @ccall (MLIR_C_PATH[]).mlirAffineMapGetResult(affineMap::MlirAffineMap, pos::intptr_t)::MlirAffineExpr
end

"""
    mlirAffineMapGetNumInputs(affineMap)

Returns the number of inputs (dimensions + symbols) of the given affine map.
"""
function mlirAffineMapGetNumInputs(affineMap)
    @ccall (MLIR_C_PATH[]).mlirAffineMapGetNumInputs(affineMap::MlirAffineMap)::intptr_t
end

"""
    mlirAffineMapIsProjectedPermutation(affineMap)

Checks whether the given affine map represents a subset of a symbol-less permutation map.
"""
function mlirAffineMapIsProjectedPermutation(affineMap)
    @ccall (MLIR_C_PATH[]).mlirAffineMapIsProjectedPermutation(affineMap::MlirAffineMap)::Bool
end

"""
    mlirAffineMapIsPermutation(affineMap)

Checks whether the given affine map represents a symbol-less permutation map.
"""
function mlirAffineMapIsPermutation(affineMap)
    @ccall (MLIR_C_PATH[]).mlirAffineMapIsPermutation(affineMap::MlirAffineMap)::Bool
end

"""
    mlirAffineMapGetSubMap(affineMap, size, resultPos)

Returns the affine map consisting of the `resultPos` subset.
"""
function mlirAffineMapGetSubMap(affineMap, size, resultPos)
    @ccall (MLIR_C_PATH[]).mlirAffineMapGetSubMap(affineMap::MlirAffineMap, size::intptr_t, resultPos::Ptr{intptr_t})::MlirAffineMap
end

"""
    mlirAffineMapGetMajorSubMap(affineMap, numResults)

Returns the affine map consisting of the most major `numResults` results. Returns the null AffineMap if the `numResults` is equal to zero. Returns the `affineMap` if `numResults` is greater or equals to number of results of the given affine map.
"""
function mlirAffineMapGetMajorSubMap(affineMap, numResults)
    @ccall (MLIR_C_PATH[]).mlirAffineMapGetMajorSubMap(affineMap::MlirAffineMap, numResults::intptr_t)::MlirAffineMap
end

"""
    mlirAffineMapGetMinorSubMap(affineMap, numResults)

Returns the affine map consisting of the most minor `numResults` results. Returns the null AffineMap if the `numResults` is equal to zero. Returns the `affineMap` if `numResults` is greater or equals to number of results of the given affine map.
"""
function mlirAffineMapGetMinorSubMap(affineMap, numResults)
    @ccall (MLIR_C_PATH[]).mlirAffineMapGetMinorSubMap(affineMap::MlirAffineMap, numResults::intptr_t)::MlirAffineMap
end

"""
    mlirAffineMapReplace(affineMap, expression, replacement, numResultDims, numResultSyms)

Apply AffineExpr::replace(`map`) to each of the results and return a new new AffineMap with the new results and the specified number of dims and symbols.
"""
function mlirAffineMapReplace(affineMap, expression, replacement, numResultDims, numResultSyms)
    @ccall (MLIR_C_PATH[]).mlirAffineMapReplace(affineMap::MlirAffineMap, expression::MlirAffineExpr, replacement::MlirAffineExpr, numResultDims::intptr_t, numResultSyms::intptr_t)::MlirAffineMap
end

"""
    mlirAffineMapCompressUnusedSymbols(affineMaps, size, result, populateResult)

Returns the simplified affine map resulting from dropping the symbols that do not appear in any of the individual maps in `affineMaps`. Asserts that all maps in `affineMaps` are normalized to the same number of dims and symbols. Takes a callback `populateResult` to fill the `res` container with value `m` at entry `idx`. This allows returning without worrying about ownership considerations.
"""
function mlirAffineMapCompressUnusedSymbols(affineMaps, size, result, populateResult)
    @ccall (MLIR_C_PATH[]).mlirAffineMapCompressUnusedSymbols(affineMaps::Ptr{MlirAffineMap}, size::intptr_t, result::Ptr{Cvoid}, populateResult::Ptr{Cvoid})::Cvoid
end

"""
    mlirAttributeGetNull()

Returns an empty attribute.
"""
function mlirAttributeGetNull()
    @ccall (MLIR_C_PATH[]).mlirAttributeGetNull()::MlirAttribute
end

function mlirAttributeIsALocation(attr)
    @ccall (MLIR_C_PATH[]).mlirAttributeIsALocation(attr::MlirAttribute)::Bool
end

"""
    mlirAttributeIsAAffineMap(attr)

Checks whether the given attribute is an affine map attribute.
"""
function mlirAttributeIsAAffineMap(attr)
    @ccall (MLIR_C_PATH[]).mlirAttributeIsAAffineMap(attr::MlirAttribute)::Bool
end

"""
    mlirAffineMapAttrGet(map)

Creates an affine map attribute wrapping the given map. The attribute belongs to the same context as the affine map.
"""
function mlirAffineMapAttrGet(map)
    @ccall (MLIR_C_PATH[]).mlirAffineMapAttrGet(map::MlirAffineMap)::MlirAttribute
end

"""
    mlirAffineMapAttrGetValue(attr)

Returns the affine map wrapped in the given affine map attribute.
"""
function mlirAffineMapAttrGetValue(attr)
    @ccall (MLIR_C_PATH[]).mlirAffineMapAttrGetValue(attr::MlirAttribute)::MlirAffineMap
end

"""
    mlirAffineMapAttrGetTypeID()

Returns the typeID of an AffineMap attribute.
"""
function mlirAffineMapAttrGetTypeID()
    @ccall (MLIR_C_PATH[]).mlirAffineMapAttrGetTypeID()::MlirTypeID
end

"""
    mlirAttributeIsAArray(attr)

Checks whether the given attribute is an array attribute.
"""
function mlirAttributeIsAArray(attr)
    @ccall (MLIR_C_PATH[]).mlirAttributeIsAArray(attr::MlirAttribute)::Bool
end

"""
    mlirArrayAttrGet(ctx, numElements, elements)

Creates an array element containing the given list of elements in the given context.
"""
function mlirArrayAttrGet(ctx, numElements, elements)
    @ccall (MLIR_C_PATH[]).mlirArrayAttrGet(ctx::MlirContext, numElements::intptr_t, elements::Ptr{MlirAttribute})::MlirAttribute
end

"""
    mlirArrayAttrGetNumElements(attr)

Returns the number of elements stored in the given array attribute.
"""
function mlirArrayAttrGetNumElements(attr)
    @ccall (MLIR_C_PATH[]).mlirArrayAttrGetNumElements(attr::MlirAttribute)::intptr_t
end

"""
    mlirArrayAttrGetElement(attr, pos)

Returns pos-th element stored in the given array attribute.
"""
function mlirArrayAttrGetElement(attr, pos)
    @ccall (MLIR_C_PATH[]).mlirArrayAttrGetElement(attr::MlirAttribute, pos::intptr_t)::MlirAttribute
end

"""
    mlirArrayAttrGetTypeID()

Returns the typeID of an Array attribute.
"""
function mlirArrayAttrGetTypeID()
    @ccall (MLIR_C_PATH[]).mlirArrayAttrGetTypeID()::MlirTypeID
end

"""
    mlirAttributeIsADictionary(attr)

Checks whether the given attribute is a dictionary attribute.
"""
function mlirAttributeIsADictionary(attr)
    @ccall (MLIR_C_PATH[]).mlirAttributeIsADictionary(attr::MlirAttribute)::Bool
end

"""
    mlirDictionaryAttrGet(ctx, numElements, elements)

Creates a dictionary attribute containing the given list of elements in the provided context.
"""
function mlirDictionaryAttrGet(ctx, numElements, elements)
    @ccall (MLIR_C_PATH[]).mlirDictionaryAttrGet(ctx::MlirContext, numElements::intptr_t, elements::Ptr{MlirNamedAttribute})::MlirAttribute
end

"""
    mlirDictionaryAttrGetNumElements(attr)

Returns the number of attributes contained in a dictionary attribute.
"""
function mlirDictionaryAttrGetNumElements(attr)
    @ccall (MLIR_C_PATH[]).mlirDictionaryAttrGetNumElements(attr::MlirAttribute)::intptr_t
end

"""
    mlirDictionaryAttrGetElement(attr, pos)

Returns pos-th element of the given dictionary attribute.
"""
function mlirDictionaryAttrGetElement(attr, pos)
    @ccall (MLIR_C_PATH[]).mlirDictionaryAttrGetElement(attr::MlirAttribute, pos::intptr_t)::MlirNamedAttribute
end

"""
    mlirDictionaryAttrGetElementByName(attr, name)

Returns the dictionary attribute element with the given name or NULL if the given name does not exist in the dictionary.
"""
function mlirDictionaryAttrGetElementByName(attr, name)
    @ccall (MLIR_C_PATH[]).mlirDictionaryAttrGetElementByName(attr::MlirAttribute, name::MlirStringRef)::MlirAttribute
end

"""
    mlirDictionaryAttrGetTypeID()

Returns the typeID of a Dictionary attribute.
"""
function mlirDictionaryAttrGetTypeID()
    @ccall (MLIR_C_PATH[]).mlirDictionaryAttrGetTypeID()::MlirTypeID
end

"""
    mlirAttributeIsAFloat(attr)

Checks whether the given attribute is a floating point attribute.
"""
function mlirAttributeIsAFloat(attr)
    @ccall (MLIR_C_PATH[]).mlirAttributeIsAFloat(attr::MlirAttribute)::Bool
end

"""
    mlirFloatAttrDoubleGet(ctx, type, value)

Creates a floating point attribute in the given context with the given double value and double-precision FP semantics.
"""
function mlirFloatAttrDoubleGet(ctx, type, value)
    @ccall (MLIR_C_PATH[]).mlirFloatAttrDoubleGet(ctx::MlirContext, type::MlirType, value::Cdouble)::MlirAttribute
end

"""
    mlirFloatAttrDoubleGetChecked(loc, type, value)

Same as "[`mlirFloatAttrDoubleGet`](@ref)", but if the type is not valid for a construction of a FloatAttr, returns a null [`MlirAttribute`](@ref).
"""
function mlirFloatAttrDoubleGetChecked(loc, type, value)
    @ccall (MLIR_C_PATH[]).mlirFloatAttrDoubleGetChecked(loc::MlirLocation, type::MlirType, value::Cdouble)::MlirAttribute
end

"""
    mlirFloatAttrGetValueDouble(attr)

Returns the value stored in the given floating point attribute, interpreting the value as double.
"""
function mlirFloatAttrGetValueDouble(attr)
    @ccall (MLIR_C_PATH[]).mlirFloatAttrGetValueDouble(attr::MlirAttribute)::Cdouble
end

"""
    mlirFloatAttrGetTypeID()

Returns the typeID of a Float attribute.
"""
function mlirFloatAttrGetTypeID()
    @ccall (MLIR_C_PATH[]).mlirFloatAttrGetTypeID()::MlirTypeID
end

"""
    mlirAttributeIsAInteger(attr)

Checks whether the given attribute is an integer attribute.
"""
function mlirAttributeIsAInteger(attr)
    @ccall (MLIR_C_PATH[]).mlirAttributeIsAInteger(attr::MlirAttribute)::Bool
end

"""
    mlirIntegerAttrGet(type, value)

Creates an integer attribute of the given type with the given integer value.
"""
function mlirIntegerAttrGet(type, value)
    @ccall (MLIR_C_PATH[]).mlirIntegerAttrGet(type::MlirType, value::Int64)::MlirAttribute
end

"""
    mlirIntegerAttrGetValueInt(attr)

Returns the value stored in the given integer attribute, assuming the value is of signless type and fits into a signed 64-bit integer.
"""
function mlirIntegerAttrGetValueInt(attr)
    @ccall (MLIR_C_PATH[]).mlirIntegerAttrGetValueInt(attr::MlirAttribute)::Int64
end

"""
    mlirIntegerAttrGetValueSInt(attr)

Returns the value stored in the given integer attribute, assuming the value is of signed type and fits into a signed 64-bit integer.
"""
function mlirIntegerAttrGetValueSInt(attr)
    @ccall (MLIR_C_PATH[]).mlirIntegerAttrGetValueSInt(attr::MlirAttribute)::Int64
end

"""
    mlirIntegerAttrGetValueUInt(attr)

Returns the value stored in the given integer attribute, assuming the value is of unsigned type and fits into an unsigned 64-bit integer.
"""
function mlirIntegerAttrGetValueUInt(attr)
    @ccall (MLIR_C_PATH[]).mlirIntegerAttrGetValueUInt(attr::MlirAttribute)::UInt64
end

"""
    mlirIntegerAttrGetTypeID()

Returns the typeID of an Integer attribute.
"""
function mlirIntegerAttrGetTypeID()
    @ccall (MLIR_C_PATH[]).mlirIntegerAttrGetTypeID()::MlirTypeID
end

"""
    mlirAttributeIsABool(attr)

Checks whether the given attribute is a bool attribute.
"""
function mlirAttributeIsABool(attr)
    @ccall (MLIR_C_PATH[]).mlirAttributeIsABool(attr::MlirAttribute)::Bool
end

"""
    mlirBoolAttrGet(ctx, value)

Creates a bool attribute in the given context with the given value.
"""
function mlirBoolAttrGet(ctx, value)
    @ccall (MLIR_C_PATH[]).mlirBoolAttrGet(ctx::MlirContext, value::Cint)::MlirAttribute
end

"""
    mlirBoolAttrGetValue(attr)

Returns the value stored in the given bool attribute.
"""
function mlirBoolAttrGetValue(attr)
    @ccall (MLIR_C_PATH[]).mlirBoolAttrGetValue(attr::MlirAttribute)::Bool
end

"""
    mlirAttributeIsAIntegerSet(attr)

Checks whether the given attribute is an integer set attribute.
"""
function mlirAttributeIsAIntegerSet(attr)
    @ccall (MLIR_C_PATH[]).mlirAttributeIsAIntegerSet(attr::MlirAttribute)::Bool
end

"""
    mlirIntegerSetAttrGetTypeID()

Returns the typeID of an IntegerSet attribute.
"""
function mlirIntegerSetAttrGetTypeID()
    @ccall (MLIR_C_PATH[]).mlirIntegerSetAttrGetTypeID()::MlirTypeID
end

"""
    mlirAttributeIsAOpaque(attr)

Checks whether the given attribute is an opaque attribute.
"""
function mlirAttributeIsAOpaque(attr)
    @ccall (MLIR_C_PATH[]).mlirAttributeIsAOpaque(attr::MlirAttribute)::Bool
end

"""
    mlirOpaqueAttrGet(ctx, dialectNamespace, dataLength, data, type)

Creates an opaque attribute in the given context associated with the dialect identified by its namespace. The attribute contains opaque byte data of the specified length (data need not be null-terminated).
"""
function mlirOpaqueAttrGet(ctx, dialectNamespace, dataLength, data, type)
    @ccall (MLIR_C_PATH[]).mlirOpaqueAttrGet(ctx::MlirContext, dialectNamespace::MlirStringRef, dataLength::intptr_t, data::Cstring, type::MlirType)::MlirAttribute
end

"""
    mlirOpaqueAttrGetDialectNamespace(attr)

Returns the namespace of the dialect with which the given opaque attribute is associated. The namespace string is owned by the context.
"""
function mlirOpaqueAttrGetDialectNamespace(attr)
    @ccall (MLIR_C_PATH[]).mlirOpaqueAttrGetDialectNamespace(attr::MlirAttribute)::MlirStringRef
end

"""
    mlirOpaqueAttrGetData(attr)

Returns the raw data as a string reference. The data remains live as long as the context in which the attribute lives.
"""
function mlirOpaqueAttrGetData(attr)
    @ccall (MLIR_C_PATH[]).mlirOpaqueAttrGetData(attr::MlirAttribute)::MlirStringRef
end

"""
    mlirOpaqueAttrGetTypeID()

Returns the typeID of an Opaque attribute.
"""
function mlirOpaqueAttrGetTypeID()
    @ccall (MLIR_C_PATH[]).mlirOpaqueAttrGetTypeID()::MlirTypeID
end

"""
    mlirAttributeIsAString(attr)

Checks whether the given attribute is a string attribute.
"""
function mlirAttributeIsAString(attr)
    @ccall (MLIR_C_PATH[]).mlirAttributeIsAString(attr::MlirAttribute)::Bool
end

"""
    mlirStringAttrGet(ctx, str)

Creates a string attribute in the given context containing the given string.
"""
function mlirStringAttrGet(ctx, str)
    @ccall (MLIR_C_PATH[]).mlirStringAttrGet(ctx::MlirContext, str::MlirStringRef)::MlirAttribute
end

"""
    mlirStringAttrTypedGet(type, str)

Creates a string attribute in the given context containing the given string. Additionally, the attribute has the given type.
"""
function mlirStringAttrTypedGet(type, str)
    @ccall (MLIR_C_PATH[]).mlirStringAttrTypedGet(type::MlirType, str::MlirStringRef)::MlirAttribute
end

"""
    mlirStringAttrGetValue(attr)

Returns the attribute values as a string reference. The data remains live as long as the context in which the attribute lives.
"""
function mlirStringAttrGetValue(attr)
    @ccall (MLIR_C_PATH[]).mlirStringAttrGetValue(attr::MlirAttribute)::MlirStringRef
end

"""
    mlirStringAttrGetTypeID()

Returns the typeID of a String attribute.
"""
function mlirStringAttrGetTypeID()
    @ccall (MLIR_C_PATH[]).mlirStringAttrGetTypeID()::MlirTypeID
end

"""
    mlirAttributeIsASymbolRef(attr)

Checks whether the given attribute is a symbol reference attribute.
"""
function mlirAttributeIsASymbolRef(attr)
    @ccall (MLIR_C_PATH[]).mlirAttributeIsASymbolRef(attr::MlirAttribute)::Bool
end

"""
    mlirSymbolRefAttrGet(ctx, symbol, numReferences, references)

Creates a symbol reference attribute in the given context referencing a symbol identified by the given string inside a list of nested references. Each of the references in the list must not be nested.
"""
function mlirSymbolRefAttrGet(ctx, symbol, numReferences, references)
    @ccall (MLIR_C_PATH[]).mlirSymbolRefAttrGet(ctx::MlirContext, symbol::MlirStringRef, numReferences::intptr_t, references::Ptr{MlirAttribute})::MlirAttribute
end

"""
    mlirSymbolRefAttrGetRootReference(attr)

Returns the string reference to the root referenced symbol. The data remains live as long as the context in which the attribute lives.
"""
function mlirSymbolRefAttrGetRootReference(attr)
    @ccall (MLIR_C_PATH[]).mlirSymbolRefAttrGetRootReference(attr::MlirAttribute)::MlirStringRef
end

"""
    mlirSymbolRefAttrGetLeafReference(attr)

Returns the string reference to the leaf referenced symbol. The data remains live as long as the context in which the attribute lives.
"""
function mlirSymbolRefAttrGetLeafReference(attr)
    @ccall (MLIR_C_PATH[]).mlirSymbolRefAttrGetLeafReference(attr::MlirAttribute)::MlirStringRef
end

"""
    mlirSymbolRefAttrGetNumNestedReferences(attr)

Returns the number of references nested in the given symbol reference attribute.
"""
function mlirSymbolRefAttrGetNumNestedReferences(attr)
    @ccall (MLIR_C_PATH[]).mlirSymbolRefAttrGetNumNestedReferences(attr::MlirAttribute)::intptr_t
end

"""
    mlirSymbolRefAttrGetNestedReference(attr, pos)

Returns pos-th reference nested in the given symbol reference attribute.
"""
function mlirSymbolRefAttrGetNestedReference(attr, pos)
    @ccall (MLIR_C_PATH[]).mlirSymbolRefAttrGetNestedReference(attr::MlirAttribute, pos::intptr_t)::MlirAttribute
end

"""
    mlirSymbolRefAttrGetTypeID()

Returns the typeID of an SymbolRef attribute.
"""
function mlirSymbolRefAttrGetTypeID()
    @ccall (MLIR_C_PATH[]).mlirSymbolRefAttrGetTypeID()::MlirTypeID
end

"""
    mlirAttributeIsAFlatSymbolRef(attr)

Checks whether the given attribute is a flat symbol reference attribute.
"""
function mlirAttributeIsAFlatSymbolRef(attr)
    @ccall (MLIR_C_PATH[]).mlirAttributeIsAFlatSymbolRef(attr::MlirAttribute)::Bool
end

"""
    mlirFlatSymbolRefAttrGet(ctx, symbol)

Creates a flat symbol reference attribute in the given context referencing a symbol identified by the given string.
"""
function mlirFlatSymbolRefAttrGet(ctx, symbol)
    @ccall (MLIR_C_PATH[]).mlirFlatSymbolRefAttrGet(ctx::MlirContext, symbol::MlirStringRef)::MlirAttribute
end

"""
    mlirFlatSymbolRefAttrGetValue(attr)

Returns the referenced symbol as a string reference. The data remains live as long as the context in which the attribute lives.
"""
function mlirFlatSymbolRefAttrGetValue(attr)
    @ccall (MLIR_C_PATH[]).mlirFlatSymbolRefAttrGetValue(attr::MlirAttribute)::MlirStringRef
end

"""
    mlirAttributeIsAType(attr)

Checks whether the given attribute is a type attribute.
"""
function mlirAttributeIsAType(attr)
    @ccall (MLIR_C_PATH[]).mlirAttributeIsAType(attr::MlirAttribute)::Bool
end

"""
    mlirTypeAttrGet(type)

Creates a type attribute wrapping the given type in the same context as the type.
"""
function mlirTypeAttrGet(type)
    @ccall (MLIR_C_PATH[]).mlirTypeAttrGet(type::MlirType)::MlirAttribute
end

"""
    mlirTypeAttrGetValue(attr)

Returns the type stored in the given type attribute.
"""
function mlirTypeAttrGetValue(attr)
    @ccall (MLIR_C_PATH[]).mlirTypeAttrGetValue(attr::MlirAttribute)::MlirType
end

"""
    mlirTypeAttrGetTypeID()

Returns the typeID of a Type attribute.
"""
function mlirTypeAttrGetTypeID()
    @ccall (MLIR_C_PATH[]).mlirTypeAttrGetTypeID()::MlirTypeID
end

"""
    mlirAttributeIsAUnit(attr)

Checks whether the given attribute is a unit attribute.
"""
function mlirAttributeIsAUnit(attr)
    @ccall (MLIR_C_PATH[]).mlirAttributeIsAUnit(attr::MlirAttribute)::Bool
end

"""
    mlirUnitAttrGet(ctx)

Creates a unit attribute in the given context.
"""
function mlirUnitAttrGet(ctx)
    @ccall (MLIR_C_PATH[]).mlirUnitAttrGet(ctx::MlirContext)::MlirAttribute
end

"""
    mlirUnitAttrGetTypeID()

Returns the typeID of a Unit attribute.
"""
function mlirUnitAttrGetTypeID()
    @ccall (MLIR_C_PATH[]).mlirUnitAttrGetTypeID()::MlirTypeID
end

"""
    mlirAttributeIsAElements(attr)

Checks whether the given attribute is an elements attribute.
"""
function mlirAttributeIsAElements(attr)
    @ccall (MLIR_C_PATH[]).mlirAttributeIsAElements(attr::MlirAttribute)::Bool
end

"""
    mlirElementsAttrGetValue(attr, rank, idxs)

Returns the element at the given rank-dimensional index.
"""
function mlirElementsAttrGetValue(attr, rank, idxs)
    @ccall (MLIR_C_PATH[]).mlirElementsAttrGetValue(attr::MlirAttribute, rank::intptr_t, idxs::Ptr{UInt64})::MlirAttribute
end

"""
    mlirElementsAttrIsValidIndex(attr, rank, idxs)

Checks whether the given rank-dimensional index is valid in the given elements attribute.
"""
function mlirElementsAttrIsValidIndex(attr, rank, idxs)
    @ccall (MLIR_C_PATH[]).mlirElementsAttrIsValidIndex(attr::MlirAttribute, rank::intptr_t, idxs::Ptr{UInt64})::Bool
end

"""
    mlirElementsAttrGetNumElements(attr)

Gets the total number of elements in the given elements attribute. In order to iterate over the attribute, obtain its type, which must be a statically shaped type and use its sizes to build a multi-dimensional index.
"""
function mlirElementsAttrGetNumElements(attr)
    @ccall (MLIR_C_PATH[]).mlirElementsAttrGetNumElements(attr::MlirAttribute)::Int64
end

function mlirDenseArrayAttrGetTypeID()
    @ccall (MLIR_C_PATH[]).mlirDenseArrayAttrGetTypeID()::MlirTypeID
end

"""
    mlirAttributeIsADenseBoolArray(attr)

Checks whether the given attribute is a dense array attribute.
"""
function mlirAttributeIsADenseBoolArray(attr)
    @ccall (MLIR_C_PATH[]).mlirAttributeIsADenseBoolArray(attr::MlirAttribute)::Bool
end

function mlirAttributeIsADenseI8Array(attr)
    @ccall (MLIR_C_PATH[]).mlirAttributeIsADenseI8Array(attr::MlirAttribute)::Bool
end

function mlirAttributeIsADenseI16Array(attr)
    @ccall (MLIR_C_PATH[]).mlirAttributeIsADenseI16Array(attr::MlirAttribute)::Bool
end

function mlirAttributeIsADenseI32Array(attr)
    @ccall (MLIR_C_PATH[]).mlirAttributeIsADenseI32Array(attr::MlirAttribute)::Bool
end

function mlirAttributeIsADenseI64Array(attr)
    @ccall (MLIR_C_PATH[]).mlirAttributeIsADenseI64Array(attr::MlirAttribute)::Bool
end

function mlirAttributeIsADenseF32Array(attr)
    @ccall (MLIR_C_PATH[]).mlirAttributeIsADenseF32Array(attr::MlirAttribute)::Bool
end

function mlirAttributeIsADenseF64Array(attr)
    @ccall (MLIR_C_PATH[]).mlirAttributeIsADenseF64Array(attr::MlirAttribute)::Bool
end

"""
    mlirDenseBoolArrayGet(ctx, size, values)

Create a dense array attribute with the given elements.
"""
function mlirDenseBoolArrayGet(ctx, size, values)
    @ccall (MLIR_C_PATH[]).mlirDenseBoolArrayGet(ctx::MlirContext, size::intptr_t, values::Ptr{Cint})::MlirAttribute
end

function mlirDenseI8ArrayGet(ctx, size, values)
    @ccall (MLIR_C_PATH[]).mlirDenseI8ArrayGet(ctx::MlirContext, size::intptr_t, values::Ptr{Int8})::MlirAttribute
end

function mlirDenseI16ArrayGet(ctx, size, values)
    @ccall (MLIR_C_PATH[]).mlirDenseI16ArrayGet(ctx::MlirContext, size::intptr_t, values::Ptr{Int16})::MlirAttribute
end

function mlirDenseI32ArrayGet(ctx, size, values)
    @ccall (MLIR_C_PATH[]).mlirDenseI32ArrayGet(ctx::MlirContext, size::intptr_t, values::Ptr{Int32})::MlirAttribute
end

function mlirDenseI64ArrayGet(ctx, size, values)
    @ccall (MLIR_C_PATH[]).mlirDenseI64ArrayGet(ctx::MlirContext, size::intptr_t, values::Ptr{Int64})::MlirAttribute
end

function mlirDenseF32ArrayGet(ctx, size, values)
    @ccall (MLIR_C_PATH[]).mlirDenseF32ArrayGet(ctx::MlirContext, size::intptr_t, values::Ptr{Cfloat})::MlirAttribute
end

function mlirDenseF64ArrayGet(ctx, size, values)
    @ccall (MLIR_C_PATH[]).mlirDenseF64ArrayGet(ctx::MlirContext, size::intptr_t, values::Ptr{Cdouble})::MlirAttribute
end

"""
    mlirDenseArrayGetNumElements(attr)

Get the size of a dense array.
"""
function mlirDenseArrayGetNumElements(attr)
    @ccall (MLIR_C_PATH[]).mlirDenseArrayGetNumElements(attr::MlirAttribute)::intptr_t
end

"""
    mlirDenseBoolArrayGetElement(attr, pos)

Get an element of a dense array.
"""
function mlirDenseBoolArrayGetElement(attr, pos)
    @ccall (MLIR_C_PATH[]).mlirDenseBoolArrayGetElement(attr::MlirAttribute, pos::intptr_t)::Bool
end

function mlirDenseI8ArrayGetElement(attr, pos)
    @ccall (MLIR_C_PATH[]).mlirDenseI8ArrayGetElement(attr::MlirAttribute, pos::intptr_t)::Int8
end

function mlirDenseI16ArrayGetElement(attr, pos)
    @ccall (MLIR_C_PATH[]).mlirDenseI16ArrayGetElement(attr::MlirAttribute, pos::intptr_t)::Int16
end

function mlirDenseI32ArrayGetElement(attr, pos)
    @ccall (MLIR_C_PATH[]).mlirDenseI32ArrayGetElement(attr::MlirAttribute, pos::intptr_t)::Int32
end

function mlirDenseI64ArrayGetElement(attr, pos)
    @ccall (MLIR_C_PATH[]).mlirDenseI64ArrayGetElement(attr::MlirAttribute, pos::intptr_t)::Int64
end

function mlirDenseF32ArrayGetElement(attr, pos)
    @ccall (MLIR_C_PATH[]).mlirDenseF32ArrayGetElement(attr::MlirAttribute, pos::intptr_t)::Cfloat
end

function mlirDenseF64ArrayGetElement(attr, pos)
    @ccall (MLIR_C_PATH[]).mlirDenseF64ArrayGetElement(attr::MlirAttribute, pos::intptr_t)::Cdouble
end

"""
    mlirAttributeIsADenseElements(attr)

Checks whether the given attribute is a dense elements attribute.
"""
function mlirAttributeIsADenseElements(attr)
    @ccall (MLIR_C_PATH[]).mlirAttributeIsADenseElements(attr::MlirAttribute)::Bool
end

function mlirAttributeIsADenseIntElements(attr)
    @ccall (MLIR_C_PATH[]).mlirAttributeIsADenseIntElements(attr::MlirAttribute)::Bool
end

function mlirAttributeIsADenseFPElements(attr)
    @ccall (MLIR_C_PATH[]).mlirAttributeIsADenseFPElements(attr::MlirAttribute)::Bool
end

"""
    mlirDenseIntOrFPElementsAttrGetTypeID()

Returns the typeID of an DenseIntOrFPElements attribute.
"""
function mlirDenseIntOrFPElementsAttrGetTypeID()
    @ccall (MLIR_C_PATH[]).mlirDenseIntOrFPElementsAttrGetTypeID()::MlirTypeID
end

"""
    mlirDenseElementsAttrGet(shapedType, numElements, elements)

Creates a dense elements attribute with the given Shaped type and elements in the same context as the type.
"""
function mlirDenseElementsAttrGet(shapedType, numElements, elements)
    @ccall (MLIR_C_PATH[]).mlirDenseElementsAttrGet(shapedType::MlirType, numElements::intptr_t, elements::Ptr{MlirAttribute})::MlirAttribute
end

function mlirDenseElementsAttrRawBufferGet(shapedType, rawBufferSize, rawBuffer)
    @ccall (MLIR_C_PATH[]).mlirDenseElementsAttrRawBufferGet(shapedType::MlirType, rawBufferSize::Cint, rawBuffer::Ptr{Cvoid})::MlirAttribute
end

"""
    mlirDenseElementsAttrSplatGet(shapedType, element)

Creates a dense elements attribute with the given Shaped type containing a single replicated element (splat).
"""
function mlirDenseElementsAttrSplatGet(shapedType, element)
    @ccall (MLIR_C_PATH[]).mlirDenseElementsAttrSplatGet(shapedType::MlirType, element::MlirAttribute)::MlirAttribute
end

function mlirDenseElementsAttrBoolSplatGet(shapedType, element)
    @ccall (MLIR_C_PATH[]).mlirDenseElementsAttrBoolSplatGet(shapedType::MlirType, element::Bool)::MlirAttribute
end

function mlirDenseElementsAttrUInt8SplatGet(shapedType, element)
    @ccall (MLIR_C_PATH[]).mlirDenseElementsAttrUInt8SplatGet(shapedType::MlirType, element::UInt8)::MlirAttribute
end

function mlirDenseElementsAttrInt8SplatGet(shapedType, element)
    @ccall (MLIR_C_PATH[]).mlirDenseElementsAttrInt8SplatGet(shapedType::MlirType, element::Int8)::MlirAttribute
end

function mlirDenseElementsAttrUInt32SplatGet(shapedType, element)
    @ccall (MLIR_C_PATH[]).mlirDenseElementsAttrUInt32SplatGet(shapedType::MlirType, element::UInt32)::MlirAttribute
end

function mlirDenseElementsAttrInt32SplatGet(shapedType, element)
    @ccall (MLIR_C_PATH[]).mlirDenseElementsAttrInt32SplatGet(shapedType::MlirType, element::Int32)::MlirAttribute
end

function mlirDenseElementsAttrUInt64SplatGet(shapedType, element)
    @ccall (MLIR_C_PATH[]).mlirDenseElementsAttrUInt64SplatGet(shapedType::MlirType, element::UInt64)::MlirAttribute
end

function mlirDenseElementsAttrInt64SplatGet(shapedType, element)
    @ccall (MLIR_C_PATH[]).mlirDenseElementsAttrInt64SplatGet(shapedType::MlirType, element::Int64)::MlirAttribute
end

function mlirDenseElementsAttrFloatSplatGet(shapedType, element)
    @ccall (MLIR_C_PATH[]).mlirDenseElementsAttrFloatSplatGet(shapedType::MlirType, element::Cfloat)::MlirAttribute
end

function mlirDenseElementsAttrDoubleSplatGet(shapedType, element)
    @ccall (MLIR_C_PATH[]).mlirDenseElementsAttrDoubleSplatGet(shapedType::MlirType, element::Cdouble)::MlirAttribute
end

"""
    mlirDenseElementsAttrBoolGet(shapedType, numElements, elements)

Creates a dense elements attribute with the given shaped type from elements of a specific type. Expects the element type of the shaped type to match the data element type.
"""
function mlirDenseElementsAttrBoolGet(shapedType, numElements, elements)
    @ccall (MLIR_C_PATH[]).mlirDenseElementsAttrBoolGet(shapedType::MlirType, numElements::intptr_t, elements::Ptr{Cint})::MlirAttribute
end

function mlirDenseElementsAttrUInt8Get(shapedType, numElements, elements)
    @ccall (MLIR_C_PATH[]).mlirDenseElementsAttrUInt8Get(shapedType::MlirType, numElements::intptr_t, elements::Ptr{UInt8})::MlirAttribute
end

function mlirDenseElementsAttrInt8Get(shapedType, numElements, elements)
    @ccall (MLIR_C_PATH[]).mlirDenseElementsAttrInt8Get(shapedType::MlirType, numElements::intptr_t, elements::Ptr{Int8})::MlirAttribute
end

function mlirDenseElementsAttrUInt16Get(shapedType, numElements, elements)
    @ccall (MLIR_C_PATH[]).mlirDenseElementsAttrUInt16Get(shapedType::MlirType, numElements::intptr_t, elements::Ptr{UInt16})::MlirAttribute
end

function mlirDenseElementsAttrInt16Get(shapedType, numElements, elements)
    @ccall (MLIR_C_PATH[]).mlirDenseElementsAttrInt16Get(shapedType::MlirType, numElements::intptr_t, elements::Ptr{Int16})::MlirAttribute
end

function mlirDenseElementsAttrUInt32Get(shapedType, numElements, elements)
    @ccall (MLIR_C_PATH[]).mlirDenseElementsAttrUInt32Get(shapedType::MlirType, numElements::intptr_t, elements::Ptr{UInt32})::MlirAttribute
end

function mlirDenseElementsAttrInt32Get(shapedType, numElements, elements)
    @ccall (MLIR_C_PATH[]).mlirDenseElementsAttrInt32Get(shapedType::MlirType, numElements::intptr_t, elements::Ptr{Int32})::MlirAttribute
end

function mlirDenseElementsAttrUInt64Get(shapedType, numElements, elements)
    @ccall (MLIR_C_PATH[]).mlirDenseElementsAttrUInt64Get(shapedType::MlirType, numElements::intptr_t, elements::Ptr{UInt64})::MlirAttribute
end

function mlirDenseElementsAttrInt64Get(shapedType, numElements, elements)
    @ccall (MLIR_C_PATH[]).mlirDenseElementsAttrInt64Get(shapedType::MlirType, numElements::intptr_t, elements::Ptr{Int64})::MlirAttribute
end

function mlirDenseElementsAttrFloatGet(shapedType, numElements, elements)
    @ccall (MLIR_C_PATH[]).mlirDenseElementsAttrFloatGet(shapedType::MlirType, numElements::intptr_t, elements::Ptr{Cfloat})::MlirAttribute
end

function mlirDenseElementsAttrDoubleGet(shapedType, numElements, elements)
    @ccall (MLIR_C_PATH[]).mlirDenseElementsAttrDoubleGet(shapedType::MlirType, numElements::intptr_t, elements::Ptr{Cdouble})::MlirAttribute
end

function mlirDenseElementsAttrBFloat16Get(shapedType, numElements, elements)
    @ccall (MLIR_C_PATH[]).mlirDenseElementsAttrBFloat16Get(shapedType::MlirType, numElements::intptr_t, elements::Ptr{UInt16})::MlirAttribute
end

function mlirDenseElementsAttrFloat16Get(shapedType, numElements, elements)
    @ccall (MLIR_C_PATH[]).mlirDenseElementsAttrFloat16Get(shapedType::MlirType, numElements::intptr_t, elements::Ptr{UInt16})::MlirAttribute
end

"""
    mlirDenseElementsAttrStringGet(shapedType, numElements, strs)

Creates a dense elements attribute with the given shaped type from string elements.
"""
function mlirDenseElementsAttrStringGet(shapedType, numElements, strs)
    @ccall (MLIR_C_PATH[]).mlirDenseElementsAttrStringGet(shapedType::MlirType, numElements::intptr_t, strs::Ptr{MlirStringRef})::MlirAttribute
end

"""
    mlirDenseElementsAttrReshapeGet(attr, shapedType)

Creates a dense elements attribute that has the same data as the given dense elements attribute and a different shaped type. The new type must have the same total number of elements.
"""
function mlirDenseElementsAttrReshapeGet(attr, shapedType)
    @ccall (MLIR_C_PATH[]).mlirDenseElementsAttrReshapeGet(attr::MlirAttribute, shapedType::MlirType)::MlirAttribute
end

"""
    mlirDenseElementsAttrIsSplat(attr)

Checks whether the given dense elements attribute contains a single replicated value (splat).
"""
function mlirDenseElementsAttrIsSplat(attr)
    @ccall (MLIR_C_PATH[]).mlirDenseElementsAttrIsSplat(attr::MlirAttribute)::Bool
end

"""
    mlirDenseElementsAttrGetSplatValue(attr)

Returns the single replicated value (splat) of a specific type contained by the given dense elements attribute.
"""
function mlirDenseElementsAttrGetSplatValue(attr)
    @ccall (MLIR_C_PATH[]).mlirDenseElementsAttrGetSplatValue(attr::MlirAttribute)::MlirAttribute
end

function mlirDenseElementsAttrGetBoolSplatValue(attr)
    @ccall (MLIR_C_PATH[]).mlirDenseElementsAttrGetBoolSplatValue(attr::MlirAttribute)::Cint
end

function mlirDenseElementsAttrGetInt8SplatValue(attr)
    @ccall (MLIR_C_PATH[]).mlirDenseElementsAttrGetInt8SplatValue(attr::MlirAttribute)::Int8
end

function mlirDenseElementsAttrGetUInt8SplatValue(attr)
    @ccall (MLIR_C_PATH[]).mlirDenseElementsAttrGetUInt8SplatValue(attr::MlirAttribute)::UInt8
end

function mlirDenseElementsAttrGetInt32SplatValue(attr)
    @ccall (MLIR_C_PATH[]).mlirDenseElementsAttrGetInt32SplatValue(attr::MlirAttribute)::Int32
end

function mlirDenseElementsAttrGetUInt32SplatValue(attr)
    @ccall (MLIR_C_PATH[]).mlirDenseElementsAttrGetUInt32SplatValue(attr::MlirAttribute)::UInt32
end

function mlirDenseElementsAttrGetInt64SplatValue(attr)
    @ccall (MLIR_C_PATH[]).mlirDenseElementsAttrGetInt64SplatValue(attr::MlirAttribute)::Int64
end

function mlirDenseElementsAttrGetUInt64SplatValue(attr)
    @ccall (MLIR_C_PATH[]).mlirDenseElementsAttrGetUInt64SplatValue(attr::MlirAttribute)::UInt64
end

function mlirDenseElementsAttrGetFloatSplatValue(attr)
    @ccall (MLIR_C_PATH[]).mlirDenseElementsAttrGetFloatSplatValue(attr::MlirAttribute)::Cfloat
end

function mlirDenseElementsAttrGetDoubleSplatValue(attr)
    @ccall (MLIR_C_PATH[]).mlirDenseElementsAttrGetDoubleSplatValue(attr::MlirAttribute)::Cdouble
end

function mlirDenseElementsAttrGetStringSplatValue(attr)
    @ccall (MLIR_C_PATH[]).mlirDenseElementsAttrGetStringSplatValue(attr::MlirAttribute)::MlirStringRef
end

"""
    mlirDenseElementsAttrGetBoolValue(attr, pos)

Returns the pos-th value (flat contiguous indexing) of a specific type contained by the given dense elements attribute.
"""
function mlirDenseElementsAttrGetBoolValue(attr, pos)
    @ccall (MLIR_C_PATH[]).mlirDenseElementsAttrGetBoolValue(attr::MlirAttribute, pos::intptr_t)::Bool
end

function mlirDenseElementsAttrGetInt8Value(attr, pos)
    @ccall (MLIR_C_PATH[]).mlirDenseElementsAttrGetInt8Value(attr::MlirAttribute, pos::intptr_t)::Int8
end

function mlirDenseElementsAttrGetUInt8Value(attr, pos)
    @ccall (MLIR_C_PATH[]).mlirDenseElementsAttrGetUInt8Value(attr::MlirAttribute, pos::intptr_t)::UInt8
end

function mlirDenseElementsAttrGetInt16Value(attr, pos)
    @ccall (MLIR_C_PATH[]).mlirDenseElementsAttrGetInt16Value(attr::MlirAttribute, pos::intptr_t)::Int16
end

function mlirDenseElementsAttrGetUInt16Value(attr, pos)
    @ccall (MLIR_C_PATH[]).mlirDenseElementsAttrGetUInt16Value(attr::MlirAttribute, pos::intptr_t)::UInt16
end

function mlirDenseElementsAttrGetInt32Value(attr, pos)
    @ccall (MLIR_C_PATH[]).mlirDenseElementsAttrGetInt32Value(attr::MlirAttribute, pos::intptr_t)::Int32
end

function mlirDenseElementsAttrGetUInt32Value(attr, pos)
    @ccall (MLIR_C_PATH[]).mlirDenseElementsAttrGetUInt32Value(attr::MlirAttribute, pos::intptr_t)::UInt32
end

function mlirDenseElementsAttrGetInt64Value(attr, pos)
    @ccall (MLIR_C_PATH[]).mlirDenseElementsAttrGetInt64Value(attr::MlirAttribute, pos::intptr_t)::Int64
end

function mlirDenseElementsAttrGetUInt64Value(attr, pos)
    @ccall (MLIR_C_PATH[]).mlirDenseElementsAttrGetUInt64Value(attr::MlirAttribute, pos::intptr_t)::UInt64
end

function mlirDenseElementsAttrGetFloatValue(attr, pos)
    @ccall (MLIR_C_PATH[]).mlirDenseElementsAttrGetFloatValue(attr::MlirAttribute, pos::intptr_t)::Cfloat
end

function mlirDenseElementsAttrGetDoubleValue(attr, pos)
    @ccall (MLIR_C_PATH[]).mlirDenseElementsAttrGetDoubleValue(attr::MlirAttribute, pos::intptr_t)::Cdouble
end

function mlirDenseElementsAttrGetStringValue(attr, pos)
    @ccall (MLIR_C_PATH[]).mlirDenseElementsAttrGetStringValue(attr::MlirAttribute, pos::intptr_t)::MlirStringRef
end

"""
    mlirDenseElementsAttrGetRawData(attr)

Returns the raw data of the given dense elements attribute.
"""
function mlirDenseElementsAttrGetRawData(attr)
    @ccall (MLIR_C_PATH[]).mlirDenseElementsAttrGetRawData(attr::MlirAttribute)::Ptr{Cvoid}
end

function mlirUnmanagedDenseBoolResourceElementsAttrGet(shapedType, name, numElements, elements)
    @ccall (MLIR_C_PATH[]).mlirUnmanagedDenseBoolResourceElementsAttrGet(shapedType::MlirType, name::MlirStringRef, numElements::intptr_t, elements::Ptr{Cint})::MlirAttribute
end

function mlirUnmanagedDenseUInt8ResourceElementsAttrGet(shapedType, name, numElements, elements)
    @ccall (MLIR_C_PATH[]).mlirUnmanagedDenseUInt8ResourceElementsAttrGet(shapedType::MlirType, name::MlirStringRef, numElements::intptr_t, elements::Ptr{UInt8})::MlirAttribute
end

function mlirUnmanagedDenseInt8ResourceElementsAttrGet(shapedType, name, numElements, elements)
    @ccall (MLIR_C_PATH[]).mlirUnmanagedDenseInt8ResourceElementsAttrGet(shapedType::MlirType, name::MlirStringRef, numElements::intptr_t, elements::Ptr{Int8})::MlirAttribute
end

function mlirUnmanagedDenseUInt16ResourceElementsAttrGet(shapedType, name, numElements, elements)
    @ccall (MLIR_C_PATH[]).mlirUnmanagedDenseUInt16ResourceElementsAttrGet(shapedType::MlirType, name::MlirStringRef, numElements::intptr_t, elements::Ptr{UInt16})::MlirAttribute
end

function mlirUnmanagedDenseInt16ResourceElementsAttrGet(shapedType, name, numElements, elements)
    @ccall (MLIR_C_PATH[]).mlirUnmanagedDenseInt16ResourceElementsAttrGet(shapedType::MlirType, name::MlirStringRef, numElements::intptr_t, elements::Ptr{Int16})::MlirAttribute
end

function mlirUnmanagedDenseUInt32ResourceElementsAttrGet(shapedType, name, numElements, elements)
    @ccall (MLIR_C_PATH[]).mlirUnmanagedDenseUInt32ResourceElementsAttrGet(shapedType::MlirType, name::MlirStringRef, numElements::intptr_t, elements::Ptr{UInt32})::MlirAttribute
end

function mlirUnmanagedDenseInt32ResourceElementsAttrGet(shapedType, name, numElements, elements)
    @ccall (MLIR_C_PATH[]).mlirUnmanagedDenseInt32ResourceElementsAttrGet(shapedType::MlirType, name::MlirStringRef, numElements::intptr_t, elements::Ptr{Int32})::MlirAttribute
end

function mlirUnmanagedDenseUInt64ResourceElementsAttrGet(shapedType, name, numElements, elements)
    @ccall (MLIR_C_PATH[]).mlirUnmanagedDenseUInt64ResourceElementsAttrGet(shapedType::MlirType, name::MlirStringRef, numElements::intptr_t, elements::Ptr{UInt64})::MlirAttribute
end

function mlirUnmanagedDenseInt64ResourceElementsAttrGet(shapedType, name, numElements, elements)
    @ccall (MLIR_C_PATH[]).mlirUnmanagedDenseInt64ResourceElementsAttrGet(shapedType::MlirType, name::MlirStringRef, numElements::intptr_t, elements::Ptr{Int64})::MlirAttribute
end

function mlirUnmanagedDenseFloatResourceElementsAttrGet(shapedType, name, numElements, elements)
    @ccall (MLIR_C_PATH[]).mlirUnmanagedDenseFloatResourceElementsAttrGet(shapedType::MlirType, name::MlirStringRef, numElements::intptr_t, elements::Ptr{Cfloat})::MlirAttribute
end

function mlirUnmanagedDenseDoubleResourceElementsAttrGet(shapedType, name, numElements, elements)
    @ccall (MLIR_C_PATH[]).mlirUnmanagedDenseDoubleResourceElementsAttrGet(shapedType::MlirType, name::MlirStringRef, numElements::intptr_t, elements::Ptr{Cdouble})::MlirAttribute
end

"""
    mlirDenseBoolResourceElementsAttrGetValue(attr, pos)

Returns the pos-th value (flat contiguous indexing) of a specific type contained by the given dense resource elements attribute.
"""
function mlirDenseBoolResourceElementsAttrGetValue(attr, pos)
    @ccall (MLIR_C_PATH[]).mlirDenseBoolResourceElementsAttrGetValue(attr::MlirAttribute, pos::intptr_t)::Bool
end

function mlirDenseInt8ResourceElementsAttrGetValue(attr, pos)
    @ccall (MLIR_C_PATH[]).mlirDenseInt8ResourceElementsAttrGetValue(attr::MlirAttribute, pos::intptr_t)::Int8
end

function mlirDenseUInt8ResourceElementsAttrGetValue(attr, pos)
    @ccall (MLIR_C_PATH[]).mlirDenseUInt8ResourceElementsAttrGetValue(attr::MlirAttribute, pos::intptr_t)::UInt8
end

function mlirDenseInt16ResourceElementsAttrGetValue(attr, pos)
    @ccall (MLIR_C_PATH[]).mlirDenseInt16ResourceElementsAttrGetValue(attr::MlirAttribute, pos::intptr_t)::Int16
end

function mlirDenseUInt16ResourceElementsAttrGetValue(attr, pos)
    @ccall (MLIR_C_PATH[]).mlirDenseUInt16ResourceElementsAttrGetValue(attr::MlirAttribute, pos::intptr_t)::UInt16
end

function mlirDenseInt32ResourceElementsAttrGetValue(attr, pos)
    @ccall (MLIR_C_PATH[]).mlirDenseInt32ResourceElementsAttrGetValue(attr::MlirAttribute, pos::intptr_t)::Int32
end

function mlirDenseUInt32ResourceElementsAttrGetValue(attr, pos)
    @ccall (MLIR_C_PATH[]).mlirDenseUInt32ResourceElementsAttrGetValue(attr::MlirAttribute, pos::intptr_t)::UInt32
end

function mlirDenseInt64ResourceElementsAttrGetValue(attr, pos)
    @ccall (MLIR_C_PATH[]).mlirDenseInt64ResourceElementsAttrGetValue(attr::MlirAttribute, pos::intptr_t)::Int64
end

function mlirDenseUInt64ResourceElementsAttrGetValue(attr, pos)
    @ccall (MLIR_C_PATH[]).mlirDenseUInt64ResourceElementsAttrGetValue(attr::MlirAttribute, pos::intptr_t)::UInt64
end

function mlirDenseFloatResourceElementsAttrGetValue(attr, pos)
    @ccall (MLIR_C_PATH[]).mlirDenseFloatResourceElementsAttrGetValue(attr::MlirAttribute, pos::intptr_t)::Cfloat
end

function mlirDenseDoubleResourceElementsAttrGetValue(attr, pos)
    @ccall (MLIR_C_PATH[]).mlirDenseDoubleResourceElementsAttrGetValue(attr::MlirAttribute, pos::intptr_t)::Cdouble
end

"""
    mlirAttributeIsASparseElements(attr)

Checks whether the given attribute is a sparse elements attribute.
"""
function mlirAttributeIsASparseElements(attr)
    @ccall (MLIR_C_PATH[]).mlirAttributeIsASparseElements(attr::MlirAttribute)::Bool
end

"""
    mlirSparseElementsAttribute(shapedType, denseIndices, denseValues)

Creates a sparse elements attribute of the given shape from a list of indices and a list of associated values. Both lists are expected to be dense elements attributes with the same number of elements. The list of indices is expected to contain 64-bit integers. The attribute is created in the same context as the type.
"""
function mlirSparseElementsAttribute(shapedType, denseIndices, denseValues)
    @ccall (MLIR_C_PATH[]).mlirSparseElementsAttribute(shapedType::MlirType, denseIndices::MlirAttribute, denseValues::MlirAttribute)::MlirAttribute
end

"""
    mlirSparseElementsAttrGetIndices(attr)

Returns the dense elements attribute containing 64-bit integer indices of non-null elements in the given sparse elements attribute.
"""
function mlirSparseElementsAttrGetIndices(attr)
    @ccall (MLIR_C_PATH[]).mlirSparseElementsAttrGetIndices(attr::MlirAttribute)::MlirAttribute
end

"""
    mlirSparseElementsAttrGetValues(attr)

Returns the dense elements attribute containing the non-null elements in the given sparse elements attribute.
"""
function mlirSparseElementsAttrGetValues(attr)
    @ccall (MLIR_C_PATH[]).mlirSparseElementsAttrGetValues(attr::MlirAttribute)::MlirAttribute
end

"""
    mlirSparseElementsAttrGetTypeID()

Returns the typeID of a SparseElements attribute.
"""
function mlirSparseElementsAttrGetTypeID()
    @ccall (MLIR_C_PATH[]).mlirSparseElementsAttrGetTypeID()::MlirTypeID
end

function mlirAttributeIsAStridedLayout(attr)
    @ccall (MLIR_C_PATH[]).mlirAttributeIsAStridedLayout(attr::MlirAttribute)::Bool
end

function mlirStridedLayoutAttrGet(ctx, offset, numStrides, strides)
    @ccall (MLIR_C_PATH[]).mlirStridedLayoutAttrGet(ctx::MlirContext, offset::Int64, numStrides::intptr_t, strides::Ptr{Int64})::MlirAttribute
end

function mlirStridedLayoutAttrGetOffset(attr)
    @ccall (MLIR_C_PATH[]).mlirStridedLayoutAttrGetOffset(attr::MlirAttribute)::Int64
end

function mlirStridedLayoutAttrGetNumStrides(attr)
    @ccall (MLIR_C_PATH[]).mlirStridedLayoutAttrGetNumStrides(attr::MlirAttribute)::intptr_t
end

function mlirStridedLayoutAttrGetStride(attr, pos)
    @ccall (MLIR_C_PATH[]).mlirStridedLayoutAttrGetStride(attr::MlirAttribute, pos::intptr_t)::Int64
end

"""
    mlirStridedLayoutAttrGetTypeID()

Returns the typeID of a StridedLayout attribute.
"""
function mlirStridedLayoutAttrGetTypeID()
    @ccall (MLIR_C_PATH[]).mlirStridedLayoutAttrGetTypeID()::MlirTypeID
end

"""
    mlirIntegerTypeGetTypeID()

Returns the typeID of an Integer type.
"""
function mlirIntegerTypeGetTypeID()
    @ccall (MLIR_C_PATH[]).mlirIntegerTypeGetTypeID()::MlirTypeID
end

"""
    mlirTypeIsAInteger(type)

Checks whether the given type is an integer type.
"""
function mlirTypeIsAInteger(type)
    @ccall (MLIR_C_PATH[]).mlirTypeIsAInteger(type::MlirType)::Bool
end

"""
    mlirIntegerTypeGet(ctx, bitwidth)

Creates a signless integer type of the given bitwidth in the context. The type is owned by the context.
"""
function mlirIntegerTypeGet(ctx, bitwidth)
    @ccall (MLIR_C_PATH[]).mlirIntegerTypeGet(ctx::MlirContext, bitwidth::Cuint)::MlirType
end

"""
    mlirIntegerTypeSignedGet(ctx, bitwidth)

Creates a signed integer type of the given bitwidth in the context. The type is owned by the context.
"""
function mlirIntegerTypeSignedGet(ctx, bitwidth)
    @ccall (MLIR_C_PATH[]).mlirIntegerTypeSignedGet(ctx::MlirContext, bitwidth::Cuint)::MlirType
end

"""
    mlirIntegerTypeUnsignedGet(ctx, bitwidth)

Creates an unsigned integer type of the given bitwidth in the context. The type is owned by the context.
"""
function mlirIntegerTypeUnsignedGet(ctx, bitwidth)
    @ccall (MLIR_C_PATH[]).mlirIntegerTypeUnsignedGet(ctx::MlirContext, bitwidth::Cuint)::MlirType
end

"""
    mlirIntegerTypeGetWidth(type)

Returns the bitwidth of an integer type.
"""
function mlirIntegerTypeGetWidth(type)
    @ccall (MLIR_C_PATH[]).mlirIntegerTypeGetWidth(type::MlirType)::Cuint
end

"""
    mlirIntegerTypeIsSignless(type)

Checks whether the given integer type is signless.
"""
function mlirIntegerTypeIsSignless(type)
    @ccall (MLIR_C_PATH[]).mlirIntegerTypeIsSignless(type::MlirType)::Bool
end

"""
    mlirIntegerTypeIsSigned(type)

Checks whether the given integer type is signed.
"""
function mlirIntegerTypeIsSigned(type)
    @ccall (MLIR_C_PATH[]).mlirIntegerTypeIsSigned(type::MlirType)::Bool
end

"""
    mlirIntegerTypeIsUnsigned(type)

Checks whether the given integer type is unsigned.
"""
function mlirIntegerTypeIsUnsigned(type)
    @ccall (MLIR_C_PATH[]).mlirIntegerTypeIsUnsigned(type::MlirType)::Bool
end

"""
    mlirIndexTypeGetTypeID()

Returns the typeID of an Index type.
"""
function mlirIndexTypeGetTypeID()
    @ccall (MLIR_C_PATH[]).mlirIndexTypeGetTypeID()::MlirTypeID
end

"""
    mlirTypeIsAIndex(type)

Checks whether the given type is an index type.
"""
function mlirTypeIsAIndex(type)
    @ccall (MLIR_C_PATH[]).mlirTypeIsAIndex(type::MlirType)::Bool
end

"""
    mlirIndexTypeGet(ctx)

Creates an index type in the given context. The type is owned by the context.
"""
function mlirIndexTypeGet(ctx)
    @ccall (MLIR_C_PATH[]).mlirIndexTypeGet(ctx::MlirContext)::MlirType
end

"""
    mlirFloat8E5M2TypeGetTypeID()

Returns the typeID of an Float8E5M2 type.
"""
function mlirFloat8E5M2TypeGetTypeID()
    @ccall (MLIR_C_PATH[]).mlirFloat8E5M2TypeGetTypeID()::MlirTypeID
end

"""
    mlirTypeIsAFloat8E5M2(type)

Checks whether the given type is an f8E5M2 type.
"""
function mlirTypeIsAFloat8E5M2(type)
    @ccall (MLIR_C_PATH[]).mlirTypeIsAFloat8E5M2(type::MlirType)::Bool
end

"""
    mlirFloat8E5M2TypeGet(ctx)

Creates an f8E5M2 type in the given context. The type is owned by the context.
"""
function mlirFloat8E5M2TypeGet(ctx)
    @ccall (MLIR_C_PATH[]).mlirFloat8E5M2TypeGet(ctx::MlirContext)::MlirType
end

"""
    mlirFloat8E4M3FNTypeGetTypeID()

Returns the typeID of an Float8E4M3FN type.
"""
function mlirFloat8E4M3FNTypeGetTypeID()
    @ccall (MLIR_C_PATH[]).mlirFloat8E4M3FNTypeGetTypeID()::MlirTypeID
end

"""
    mlirTypeIsAFloat8E4M3FN(type)

Checks whether the given type is an f8E4M3FN type.
"""
function mlirTypeIsAFloat8E4M3FN(type)
    @ccall (MLIR_C_PATH[]).mlirTypeIsAFloat8E4M3FN(type::MlirType)::Bool
end

"""
    mlirFloat8E4M3FNTypeGet(ctx)

Creates an f8E4M3FN type in the given context. The type is owned by the context.
"""
function mlirFloat8E4M3FNTypeGet(ctx)
    @ccall (MLIR_C_PATH[]).mlirFloat8E4M3FNTypeGet(ctx::MlirContext)::MlirType
end

"""
    mlirFloat8E5M2FNUZTypeGetTypeID()

Returns the typeID of an Float8E5M2FNUZ type.
"""
function mlirFloat8E5M2FNUZTypeGetTypeID()
    @ccall (MLIR_C_PATH[]).mlirFloat8E5M2FNUZTypeGetTypeID()::MlirTypeID
end

"""
    mlirTypeIsAFloat8E5M2FNUZ(type)

Checks whether the given type is an f8E5M2FNUZ type.
"""
function mlirTypeIsAFloat8E5M2FNUZ(type)
    @ccall (MLIR_C_PATH[]).mlirTypeIsAFloat8E5M2FNUZ(type::MlirType)::Bool
end

"""
    mlirFloat8E5M2FNUZTypeGet(ctx)

Creates an f8E5M2FNUZ type in the given context. The type is owned by the context.
"""
function mlirFloat8E5M2FNUZTypeGet(ctx)
    @ccall (MLIR_C_PATH[]).mlirFloat8E5M2FNUZTypeGet(ctx::MlirContext)::MlirType
end

"""
    mlirFloat8E4M3FNUZTypeGetTypeID()

Returns the typeID of an Float8E4M3FNUZ type.
"""
function mlirFloat8E4M3FNUZTypeGetTypeID()
    @ccall (MLIR_C_PATH[]).mlirFloat8E4M3FNUZTypeGetTypeID()::MlirTypeID
end

"""
    mlirTypeIsAFloat8E4M3FNUZ(type)

Checks whether the given type is an f8E4M3FNUZ type.
"""
function mlirTypeIsAFloat8E4M3FNUZ(type)
    @ccall (MLIR_C_PATH[]).mlirTypeIsAFloat8E4M3FNUZ(type::MlirType)::Bool
end

"""
    mlirFloat8E4M3FNUZTypeGet(ctx)

Creates an f8E4M3FNUZ type in the given context. The type is owned by the context.
"""
function mlirFloat8E4M3FNUZTypeGet(ctx)
    @ccall (MLIR_C_PATH[]).mlirFloat8E4M3FNUZTypeGet(ctx::MlirContext)::MlirType
end

"""
    mlirFloat8E4M3B11FNUZTypeGetTypeID()

Returns the typeID of an Float8E4M3B11FNUZ type.
"""
function mlirFloat8E4M3B11FNUZTypeGetTypeID()
    @ccall (MLIR_C_PATH[]).mlirFloat8E4M3B11FNUZTypeGetTypeID()::MlirTypeID
end

"""
    mlirTypeIsAFloat8E4M3B11FNUZ(type)

Checks whether the given type is an f8E4M3B11FNUZ type.
"""
function mlirTypeIsAFloat8E4M3B11FNUZ(type)
    @ccall (MLIR_C_PATH[]).mlirTypeIsAFloat8E4M3B11FNUZ(type::MlirType)::Bool
end

"""
    mlirFloat8E4M3B11FNUZTypeGet(ctx)

Creates an f8E4M3B11FNUZ type in the given context. The type is owned by the context.
"""
function mlirFloat8E4M3B11FNUZTypeGet(ctx)
    @ccall (MLIR_C_PATH[]).mlirFloat8E4M3B11FNUZTypeGet(ctx::MlirContext)::MlirType
end

"""
    mlirBFloat16TypeGetTypeID()

Returns the typeID of an BFloat16 type.
"""
function mlirBFloat16TypeGetTypeID()
    @ccall (MLIR_C_PATH[]).mlirBFloat16TypeGetTypeID()::MlirTypeID
end

"""
    mlirTypeIsABF16(type)

Checks whether the given type is a bf16 type.
"""
function mlirTypeIsABF16(type)
    @ccall (MLIR_C_PATH[]).mlirTypeIsABF16(type::MlirType)::Bool
end

"""
    mlirBF16TypeGet(ctx)

Creates a bf16 type in the given context. The type is owned by the context.
"""
function mlirBF16TypeGet(ctx)
    @ccall (MLIR_C_PATH[]).mlirBF16TypeGet(ctx::MlirContext)::MlirType
end

"""
    mlirFloat16TypeGetTypeID()

Returns the typeID of an Float16 type.
"""
function mlirFloat16TypeGetTypeID()
    @ccall (MLIR_C_PATH[]).mlirFloat16TypeGetTypeID()::MlirTypeID
end

"""
    mlirTypeIsAF16(type)

Checks whether the given type is an f16 type.
"""
function mlirTypeIsAF16(type)
    @ccall (MLIR_C_PATH[]).mlirTypeIsAF16(type::MlirType)::Bool
end

"""
    mlirF16TypeGet(ctx)

Creates an f16 type in the given context. The type is owned by the context.
"""
function mlirF16TypeGet(ctx)
    @ccall (MLIR_C_PATH[]).mlirF16TypeGet(ctx::MlirContext)::MlirType
end

"""
    mlirFloat32TypeGetTypeID()

Returns the typeID of an Float32 type.
"""
function mlirFloat32TypeGetTypeID()
    @ccall (MLIR_C_PATH[]).mlirFloat32TypeGetTypeID()::MlirTypeID
end

"""
    mlirTypeIsAF32(type)

Checks whether the given type is an f32 type.
"""
function mlirTypeIsAF32(type)
    @ccall (MLIR_C_PATH[]).mlirTypeIsAF32(type::MlirType)::Bool
end

"""
    mlirF32TypeGet(ctx)

Creates an f32 type in the given context. The type is owned by the context.
"""
function mlirF32TypeGet(ctx)
    @ccall (MLIR_C_PATH[]).mlirF32TypeGet(ctx::MlirContext)::MlirType
end

"""
    mlirFloat64TypeGetTypeID()

Returns the typeID of an Float64 type.
"""
function mlirFloat64TypeGetTypeID()
    @ccall (MLIR_C_PATH[]).mlirFloat64TypeGetTypeID()::MlirTypeID
end

"""
    mlirTypeIsAF64(type)

Checks whether the given type is an f64 type.
"""
function mlirTypeIsAF64(type)
    @ccall (MLIR_C_PATH[]).mlirTypeIsAF64(type::MlirType)::Bool
end

"""
    mlirF64TypeGet(ctx)

Creates a f64 type in the given context. The type is owned by the context.
"""
function mlirF64TypeGet(ctx)
    @ccall (MLIR_C_PATH[]).mlirF64TypeGet(ctx::MlirContext)::MlirType
end

"""
    mlirFloatTF32TypeGetTypeID()

Returns the typeID of a TF32 type.
"""
function mlirFloatTF32TypeGetTypeID()
    @ccall (MLIR_C_PATH[]).mlirFloatTF32TypeGetTypeID()::MlirTypeID
end

"""
    mlirTypeIsATF32(type)

Checks whether the given type is an TF32 type.
"""
function mlirTypeIsATF32(type)
    @ccall (MLIR_C_PATH[]).mlirTypeIsATF32(type::MlirType)::Bool
end

"""
    mlirTF32TypeGet(ctx)

Creates a TF32 type in the given context. The type is owned by the context.
"""
function mlirTF32TypeGet(ctx)
    @ccall (MLIR_C_PATH[]).mlirTF32TypeGet(ctx::MlirContext)::MlirType
end

"""
    mlirNoneTypeGetTypeID()

Returns the typeID of an None type.
"""
function mlirNoneTypeGetTypeID()
    @ccall (MLIR_C_PATH[]).mlirNoneTypeGetTypeID()::MlirTypeID
end

"""
    mlirTypeIsANone(type)

Checks whether the given type is a None type.
"""
function mlirTypeIsANone(type)
    @ccall (MLIR_C_PATH[]).mlirTypeIsANone(type::MlirType)::Bool
end

"""
    mlirNoneTypeGet(ctx)

Creates a None type in the given context. The type is owned by the context.
"""
function mlirNoneTypeGet(ctx)
    @ccall (MLIR_C_PATH[]).mlirNoneTypeGet(ctx::MlirContext)::MlirType
end

"""
    mlirComplexTypeGetTypeID()

Returns the typeID of an Complex type.
"""
function mlirComplexTypeGetTypeID()
    @ccall (MLIR_C_PATH[]).mlirComplexTypeGetTypeID()::MlirTypeID
end

"""
    mlirTypeIsAComplex(type)

Checks whether the given type is a Complex type.
"""
function mlirTypeIsAComplex(type)
    @ccall (MLIR_C_PATH[]).mlirTypeIsAComplex(type::MlirType)::Bool
end

"""
    mlirComplexTypeGet(elementType)

Creates a complex type with the given element type in the same context as the element type. The type is owned by the context.
"""
function mlirComplexTypeGet(elementType)
    @ccall (MLIR_C_PATH[]).mlirComplexTypeGet(elementType::MlirType)::MlirType
end

"""
    mlirComplexTypeGetElementType(type)

Returns the element type of the given complex type.
"""
function mlirComplexTypeGetElementType(type)
    @ccall (MLIR_C_PATH[]).mlirComplexTypeGetElementType(type::MlirType)::MlirType
end

"""
    mlirTypeIsAShaped(type)

Checks whether the given type is a Shaped type.
"""
function mlirTypeIsAShaped(type)
    @ccall (MLIR_C_PATH[]).mlirTypeIsAShaped(type::MlirType)::Bool
end

"""
    mlirShapedTypeGetElementType(type)

Returns the element type of the shaped type.
"""
function mlirShapedTypeGetElementType(type)
    @ccall (MLIR_C_PATH[]).mlirShapedTypeGetElementType(type::MlirType)::MlirType
end

"""
    mlirShapedTypeHasRank(type)

Checks whether the given shaped type is ranked.
"""
function mlirShapedTypeHasRank(type)
    @ccall (MLIR_C_PATH[]).mlirShapedTypeHasRank(type::MlirType)::Bool
end

"""
    mlirShapedTypeGetRank(type)

Returns the rank of the given ranked shaped type.
"""
function mlirShapedTypeGetRank(type)
    @ccall (MLIR_C_PATH[]).mlirShapedTypeGetRank(type::MlirType)::Int64
end

"""
    mlirShapedTypeHasStaticShape(type)

Checks whether the given shaped type has a static shape.
"""
function mlirShapedTypeHasStaticShape(type)
    @ccall (MLIR_C_PATH[]).mlirShapedTypeHasStaticShape(type::MlirType)::Bool
end

"""
    mlirShapedTypeIsDynamicDim(type, dim)

Checks wither the dim-th dimension of the given shaped type is dynamic.
"""
function mlirShapedTypeIsDynamicDim(type, dim)
    @ccall (MLIR_C_PATH[]).mlirShapedTypeIsDynamicDim(type::MlirType, dim::intptr_t)::Bool
end

"""
    mlirShapedTypeGetDimSize(type, dim)

Returns the dim-th dimension of the given ranked shaped type.
"""
function mlirShapedTypeGetDimSize(type, dim)
    @ccall (MLIR_C_PATH[]).mlirShapedTypeGetDimSize(type::MlirType, dim::intptr_t)::Int64
end

"""
    mlirShapedTypeIsDynamicSize(size)

Checks whether the given value is used as a placeholder for dynamic sizes in shaped types.
"""
function mlirShapedTypeIsDynamicSize(size)
    @ccall (MLIR_C_PATH[]).mlirShapedTypeIsDynamicSize(size::Int64)::Bool
end

"""
    mlirShapedTypeGetDynamicSize()

Returns the value indicating a dynamic size in a shaped type. Prefer [`mlirShapedTypeIsDynamicSize`](@ref) to direct comparisons with this value.
"""
function mlirShapedTypeGetDynamicSize()
    @ccall (MLIR_C_PATH[]).mlirShapedTypeGetDynamicSize()::Int64
end

"""
    mlirShapedTypeIsDynamicStrideOrOffset(val)

Checks whether the given value is used as a placeholder for dynamic strides and offsets in shaped types.
"""
function mlirShapedTypeIsDynamicStrideOrOffset(val)
    @ccall (MLIR_C_PATH[]).mlirShapedTypeIsDynamicStrideOrOffset(val::Int64)::Bool
end

"""
    mlirShapedTypeGetDynamicStrideOrOffset()

Returns the value indicating a dynamic stride or offset in a shaped type. Prefer [`mlirShapedTypeGetDynamicStrideOrOffset`](@ref) to direct comparisons with this value.
"""
function mlirShapedTypeGetDynamicStrideOrOffset()
    @ccall (MLIR_C_PATH[]).mlirShapedTypeGetDynamicStrideOrOffset()::Int64
end

"""
    mlirVectorTypeGetTypeID()

Returns the typeID of an Vector type.
"""
function mlirVectorTypeGetTypeID()
    @ccall (MLIR_C_PATH[]).mlirVectorTypeGetTypeID()::MlirTypeID
end

"""
    mlirTypeIsAVector(type)

Checks whether the given type is a Vector type.
"""
function mlirTypeIsAVector(type)
    @ccall (MLIR_C_PATH[]).mlirTypeIsAVector(type::MlirType)::Bool
end

"""
    mlirVectorTypeGet(rank, shape, elementType)

Creates a vector type of the shape identified by its rank and dimensions, with the given element type in the same context as the element type. The type is owned by the context.
"""
function mlirVectorTypeGet(rank, shape, elementType)
    @ccall (MLIR_C_PATH[]).mlirVectorTypeGet(rank::intptr_t, shape::Ptr{Int64}, elementType::MlirType)::MlirType
end

"""
    mlirVectorTypeGetChecked(loc, rank, shape, elementType)

Same as "[`mlirVectorTypeGet`](@ref)" but returns a nullptr wrapping [`MlirType`](@ref) on illegal arguments, emitting appropriate diagnostics.
"""
function mlirVectorTypeGetChecked(loc, rank, shape, elementType)
    @ccall (MLIR_C_PATH[]).mlirVectorTypeGetChecked(loc::MlirLocation, rank::intptr_t, shape::Ptr{Int64}, elementType::MlirType)::MlirType
end

"""
    mlirTypeIsATensor(type)

Checks whether the given type is a Tensor type.
"""
function mlirTypeIsATensor(type)
    @ccall (MLIR_C_PATH[]).mlirTypeIsATensor(type::MlirType)::Bool
end

"""
    mlirRankedTensorTypeGetTypeID()

Returns the typeID of an RankedTensor type.
"""
function mlirRankedTensorTypeGetTypeID()
    @ccall (MLIR_C_PATH[]).mlirRankedTensorTypeGetTypeID()::MlirTypeID
end

"""
    mlirTypeIsARankedTensor(type)

Checks whether the given type is a ranked tensor type.
"""
function mlirTypeIsARankedTensor(type)
    @ccall (MLIR_C_PATH[]).mlirTypeIsARankedTensor(type::MlirType)::Bool
end

"""
    mlirUnrankedTensorTypeGetTypeID()

Returns the typeID of an UnrankedTensor type.
"""
function mlirUnrankedTensorTypeGetTypeID()
    @ccall (MLIR_C_PATH[]).mlirUnrankedTensorTypeGetTypeID()::MlirTypeID
end

"""
    mlirTypeIsAUnrankedTensor(type)

Checks whether the given type is an unranked tensor type.
"""
function mlirTypeIsAUnrankedTensor(type)
    @ccall (MLIR_C_PATH[]).mlirTypeIsAUnrankedTensor(type::MlirType)::Bool
end

"""
    mlirRankedTensorTypeGet(rank, shape, elementType, encoding)

Creates a tensor type of a fixed rank with the given shape, element type, and optional encoding in the same context as the element type. The type is owned by the context. Tensor types without any specific encoding field should assign [`mlirAttributeGetNull`](@ref)() to this parameter.
"""
function mlirRankedTensorTypeGet(rank, shape, elementType, encoding)
    @ccall (MLIR_C_PATH[]).mlirRankedTensorTypeGet(rank::intptr_t, shape::Ptr{Int64}, elementType::MlirType, encoding::MlirAttribute)::MlirType
end

"""
    mlirRankedTensorTypeGetChecked(loc, rank, shape, elementType, encoding)

Same as "[`mlirRankedTensorTypeGet`](@ref)" but returns a nullptr wrapping [`MlirType`](@ref) on illegal arguments, emitting appropriate diagnostics.
"""
function mlirRankedTensorTypeGetChecked(loc, rank, shape, elementType, encoding)
    @ccall (MLIR_C_PATH[]).mlirRankedTensorTypeGetChecked(loc::MlirLocation, rank::intptr_t, shape::Ptr{Int64}, elementType::MlirType, encoding::MlirAttribute)::MlirType
end

"""
    mlirRankedTensorTypeGetEncoding(type)

Gets the 'encoding' attribute from the ranked tensor type, returning a null attribute if none.
"""
function mlirRankedTensorTypeGetEncoding(type)
    @ccall (MLIR_C_PATH[]).mlirRankedTensorTypeGetEncoding(type::MlirType)::MlirAttribute
end

"""
    mlirUnrankedTensorTypeGet(elementType)

Creates an unranked tensor type with the given element type in the same context as the element type. The type is owned by the context.
"""
function mlirUnrankedTensorTypeGet(elementType)
    @ccall (MLIR_C_PATH[]).mlirUnrankedTensorTypeGet(elementType::MlirType)::MlirType
end

"""
    mlirUnrankedTensorTypeGetChecked(loc, elementType)

Same as "[`mlirUnrankedTensorTypeGet`](@ref)" but returns a nullptr wrapping [`MlirType`](@ref) on illegal arguments, emitting appropriate diagnostics.
"""
function mlirUnrankedTensorTypeGetChecked(loc, elementType)
    @ccall (MLIR_C_PATH[]).mlirUnrankedTensorTypeGetChecked(loc::MlirLocation, elementType::MlirType)::MlirType
end

"""
    mlirMemRefTypeGetTypeID()

Returns the typeID of an MemRef type.
"""
function mlirMemRefTypeGetTypeID()
    @ccall (MLIR_C_PATH[]).mlirMemRefTypeGetTypeID()::MlirTypeID
end

"""
    mlirTypeIsAMemRef(type)

Checks whether the given type is a MemRef type.
"""
function mlirTypeIsAMemRef(type)
    @ccall (MLIR_C_PATH[]).mlirTypeIsAMemRef(type::MlirType)::Bool
end

"""
    mlirUnrankedMemRefTypeGetTypeID()

Returns the typeID of an UnrankedMemRef type.
"""
function mlirUnrankedMemRefTypeGetTypeID()
    @ccall (MLIR_C_PATH[]).mlirUnrankedMemRefTypeGetTypeID()::MlirTypeID
end

"""
    mlirTypeIsAUnrankedMemRef(type)

Checks whether the given type is an UnrankedMemRef type.
"""
function mlirTypeIsAUnrankedMemRef(type)
    @ccall (MLIR_C_PATH[]).mlirTypeIsAUnrankedMemRef(type::MlirType)::Bool
end

"""
    mlirMemRefTypeGet(elementType, rank, shape, layout, memorySpace)

Creates a MemRef type with the given rank and shape, a potentially empty list of affine layout maps, the given memory space and element type, in the same context as element type. The type is owned by the context.
"""
function mlirMemRefTypeGet(elementType, rank, shape, layout, memorySpace)
    @ccall (MLIR_C_PATH[]).mlirMemRefTypeGet(elementType::MlirType, rank::intptr_t, shape::Ptr{Int64}, layout::MlirAttribute, memorySpace::MlirAttribute)::MlirType
end

"""
    mlirMemRefTypeGetChecked(loc, elementType, rank, shape, layout, memorySpace)

Same as "[`mlirMemRefTypeGet`](@ref)" but returns a nullptr-wrapping [`MlirType`](@ref) o illegal arguments, emitting appropriate diagnostics.
"""
function mlirMemRefTypeGetChecked(loc, elementType, rank, shape, layout, memorySpace)
    @ccall (MLIR_C_PATH[]).mlirMemRefTypeGetChecked(loc::MlirLocation, elementType::MlirType, rank::intptr_t, shape::Ptr{Int64}, layout::MlirAttribute, memorySpace::MlirAttribute)::MlirType
end

"""
    mlirMemRefTypeContiguousGet(elementType, rank, shape, memorySpace)

Creates a MemRef type with the given rank, shape, memory space and element type in the same context as the element type. The type has no affine maps, i.e. represents a default row-major contiguous memref. The type is owned by the context.
"""
function mlirMemRefTypeContiguousGet(elementType, rank, shape, memorySpace)
    @ccall (MLIR_C_PATH[]).mlirMemRefTypeContiguousGet(elementType::MlirType, rank::intptr_t, shape::Ptr{Int64}, memorySpace::MlirAttribute)::MlirType
end

"""
    mlirMemRefTypeContiguousGetChecked(loc, elementType, rank, shape, memorySpace)

Same as "[`mlirMemRefTypeContiguousGet`](@ref)" but returns a nullptr wrapping [`MlirType`](@ref) on illegal arguments, emitting appropriate diagnostics.
"""
function mlirMemRefTypeContiguousGetChecked(loc, elementType, rank, shape, memorySpace)
    @ccall (MLIR_C_PATH[]).mlirMemRefTypeContiguousGetChecked(loc::MlirLocation, elementType::MlirType, rank::intptr_t, shape::Ptr{Int64}, memorySpace::MlirAttribute)::MlirType
end

"""
    mlirUnrankedMemRefTypeGet(elementType, memorySpace)

Creates an Unranked MemRef type with the given element type and in the given memory space. The type is owned by the context of element type.
"""
function mlirUnrankedMemRefTypeGet(elementType, memorySpace)
    @ccall (MLIR_C_PATH[]).mlirUnrankedMemRefTypeGet(elementType::MlirType, memorySpace::MlirAttribute)::MlirType
end

"""
    mlirUnrankedMemRefTypeGetChecked(loc, elementType, memorySpace)

Same as "[`mlirUnrankedMemRefTypeGet`](@ref)" but returns a nullptr wrapping [`MlirType`](@ref) on illegal arguments, emitting appropriate diagnostics.
"""
function mlirUnrankedMemRefTypeGetChecked(loc, elementType, memorySpace)
    @ccall (MLIR_C_PATH[]).mlirUnrankedMemRefTypeGetChecked(loc::MlirLocation, elementType::MlirType, memorySpace::MlirAttribute)::MlirType
end

"""
    mlirMemRefTypeGetLayout(type)

Returns the layout of the given MemRef type.
"""
function mlirMemRefTypeGetLayout(type)
    @ccall (MLIR_C_PATH[]).mlirMemRefTypeGetLayout(type::MlirType)::MlirAttribute
end

"""
    mlirMemRefTypeGetAffineMap(type)

Returns the affine map of the given MemRef type.
"""
function mlirMemRefTypeGetAffineMap(type)
    @ccall (MLIR_C_PATH[]).mlirMemRefTypeGetAffineMap(type::MlirType)::MlirAffineMap
end

"""
    mlirMemRefTypeGetMemorySpace(type)

Returns the memory space of the given MemRef type.
"""
function mlirMemRefTypeGetMemorySpace(type)
    @ccall (MLIR_C_PATH[]).mlirMemRefTypeGetMemorySpace(type::MlirType)::MlirAttribute
end

"""
    mlirUnrankedMemrefGetMemorySpace(type)

Returns the memory spcae of the given Unranked MemRef type.
"""
function mlirUnrankedMemrefGetMemorySpace(type)
    @ccall (MLIR_C_PATH[]).mlirUnrankedMemrefGetMemorySpace(type::MlirType)::MlirAttribute
end

"""
    mlirTupleTypeGetTypeID()

Returns the typeID of an Tuple type.
"""
function mlirTupleTypeGetTypeID()
    @ccall (MLIR_C_PATH[]).mlirTupleTypeGetTypeID()::MlirTypeID
end

"""
    mlirTypeIsATuple(type)

Checks whether the given type is a tuple type.
"""
function mlirTypeIsATuple(type)
    @ccall (MLIR_C_PATH[]).mlirTypeIsATuple(type::MlirType)::Bool
end

"""
    mlirTupleTypeGet(ctx, numElements, elements)

Creates a tuple type that consists of the given list of elemental types. The type is owned by the context.
"""
function mlirTupleTypeGet(ctx, numElements, elements)
    @ccall (MLIR_C_PATH[]).mlirTupleTypeGet(ctx::MlirContext, numElements::intptr_t, elements::Ptr{MlirType})::MlirType
end

"""
    mlirTupleTypeGetNumTypes(type)

Returns the number of types contained in a tuple.
"""
function mlirTupleTypeGetNumTypes(type)
    @ccall (MLIR_C_PATH[]).mlirTupleTypeGetNumTypes(type::MlirType)::intptr_t
end

"""
    mlirTupleTypeGetType(type, pos)

Returns the pos-th type in the tuple type.
"""
function mlirTupleTypeGetType(type, pos)
    @ccall (MLIR_C_PATH[]).mlirTupleTypeGetType(type::MlirType, pos::intptr_t)::MlirType
end

"""
    mlirFunctionTypeGetTypeID()

Returns the typeID of an Function type.
"""
function mlirFunctionTypeGetTypeID()
    @ccall (MLIR_C_PATH[]).mlirFunctionTypeGetTypeID()::MlirTypeID
end

"""
    mlirTypeIsAFunction(type)

Checks whether the given type is a function type.
"""
function mlirTypeIsAFunction(type)
    @ccall (MLIR_C_PATH[]).mlirTypeIsAFunction(type::MlirType)::Bool
end

"""
    mlirFunctionTypeGet(ctx, numInputs, inputs, numResults, results)

Creates a function type, mapping a list of input types to result types.
"""
function mlirFunctionTypeGet(ctx, numInputs, inputs, numResults, results)
    @ccall (MLIR_C_PATH[]).mlirFunctionTypeGet(ctx::MlirContext, numInputs::intptr_t, inputs::Ptr{MlirType}, numResults::intptr_t, results::Ptr{MlirType})::MlirType
end

"""
    mlirFunctionTypeGetNumInputs(type)

Returns the number of input types.
"""
function mlirFunctionTypeGetNumInputs(type)
    @ccall (MLIR_C_PATH[]).mlirFunctionTypeGetNumInputs(type::MlirType)::intptr_t
end

"""
    mlirFunctionTypeGetNumResults(type)

Returns the number of result types.
"""
function mlirFunctionTypeGetNumResults(type)
    @ccall (MLIR_C_PATH[]).mlirFunctionTypeGetNumResults(type::MlirType)::intptr_t
end

"""
    mlirFunctionTypeGetInput(type, pos)

Returns the pos-th input type.
"""
function mlirFunctionTypeGetInput(type, pos)
    @ccall (MLIR_C_PATH[]).mlirFunctionTypeGetInput(type::MlirType, pos::intptr_t)::MlirType
end

"""
    mlirFunctionTypeGetResult(type, pos)

Returns the pos-th result type.
"""
function mlirFunctionTypeGetResult(type, pos)
    @ccall (MLIR_C_PATH[]).mlirFunctionTypeGetResult(type::MlirType, pos::intptr_t)::MlirType
end

"""
    mlirOpaqueTypeGetTypeID()

Returns the typeID of an Opaque type.
"""
function mlirOpaqueTypeGetTypeID()
    @ccall (MLIR_C_PATH[]).mlirOpaqueTypeGetTypeID()::MlirTypeID
end

"""
    mlirTypeIsAOpaque(type)

Checks whether the given type is an opaque type.
"""
function mlirTypeIsAOpaque(type)
    @ccall (MLIR_C_PATH[]).mlirTypeIsAOpaque(type::MlirType)::Bool
end

"""
    mlirOpaqueTypeGet(ctx, dialectNamespace, typeData)

Creates an opaque type in the given context associated with the dialect identified by its namespace. The type contains opaque byte data of the specified length (data need not be null-terminated).
"""
function mlirOpaqueTypeGet(ctx, dialectNamespace, typeData)
    @ccall (MLIR_C_PATH[]).mlirOpaqueTypeGet(ctx::MlirContext, dialectNamespace::MlirStringRef, typeData::MlirStringRef)::MlirType
end

"""
    mlirOpaqueTypeGetDialectNamespace(type)

Returns the namespace of the dialect with which the given opaque type is associated. The namespace string is owned by the context.
"""
function mlirOpaqueTypeGetDialectNamespace(type)
    @ccall (MLIR_C_PATH[]).mlirOpaqueTypeGetDialectNamespace(type::MlirType)::MlirStringRef
end

"""
    mlirOpaqueTypeGetData(type)

Returns the raw data as a string reference. The data remains live as long as the context in which the type lives.
"""
function mlirOpaqueTypeGetData(type)
    @ccall (MLIR_C_PATH[]).mlirOpaqueTypeGetData(type::MlirType)::MlirStringRef
end

"""
    mlirPassManagerCreate(ctx)

Create a new top-level PassManager with the default anchor.
"""
function mlirPassManagerCreate(ctx)
    @ccall (MLIR_C_PATH[]).mlirPassManagerCreate(ctx::MlirContext)::MlirPassManager
end

"""
    mlirPassManagerCreateOnOperation(ctx, anchorOp)

Create a new top-level PassManager anchored on `anchorOp`.
"""
function mlirPassManagerCreateOnOperation(ctx, anchorOp)
    @ccall (MLIR_C_PATH[]).mlirPassManagerCreateOnOperation(ctx::MlirContext, anchorOp::MlirStringRef)::MlirPassManager
end

"""
    mlirPassManagerDestroy(passManager)

Destroy the provided PassManager.
"""
function mlirPassManagerDestroy(passManager)
    @ccall (MLIR_C_PATH[]).mlirPassManagerDestroy(passManager::MlirPassManager)::Cvoid
end

"""
    mlirPassManagerIsNull(passManager)

Checks if a PassManager is null.
"""
function mlirPassManagerIsNull(passManager)
    @ccall (MLIR_C_PATH[]).mlirPassManagerIsNull(passManager::MlirPassManager)::Bool
end

"""
    mlirPassManagerGetAsOpPassManager(passManager)

Cast a top-level PassManager to a generic OpPassManager.
"""
function mlirPassManagerGetAsOpPassManager(passManager)
    @ccall (MLIR_C_PATH[]).mlirPassManagerGetAsOpPassManager(passManager::MlirPassManager)::MlirOpPassManager
end

"""
    mlirPassManagerRunOnOp(passManager, op)

Run the provided `passManager` on the given `op`.
"""
function mlirPassManagerRunOnOp(passManager, op)
    @ccall (MLIR_C_PATH[]).mlirPassManagerRunOnOp(passManager::MlirPassManager, op::MlirOperation)::MlirLogicalResult
end

"""
    mlirPassManagerEnableIRPrinting(passManager)

Enable mlir-print-ir-after-all.
"""
function mlirPassManagerEnableIRPrinting(passManager)
    @ccall (MLIR_C_PATH[]).mlirPassManagerEnableIRPrinting(passManager::MlirPassManager)::Cvoid
end

"""
    mlirPassManagerEnableVerifier(passManager, enable)

Enable / disable verify-each.
"""
function mlirPassManagerEnableVerifier(passManager, enable)
    @ccall (MLIR_C_PATH[]).mlirPassManagerEnableVerifier(passManager::MlirPassManager, enable::Bool)::Cvoid
end

"""
    mlirPassManagerGetNestedUnder(passManager, operationName)

Nest an OpPassManager under the top-level PassManager, the nested passmanager will only run on operations matching the provided name. The returned OpPassManager will be destroyed when the parent is destroyed. To further nest more OpPassManager under the newly returned one, see `mlirOpPassManagerNest` below.
"""
function mlirPassManagerGetNestedUnder(passManager, operationName)
    @ccall (MLIR_C_PATH[]).mlirPassManagerGetNestedUnder(passManager::MlirPassManager, operationName::MlirStringRef)::MlirOpPassManager
end

"""
    mlirOpPassManagerGetNestedUnder(passManager, operationName)

Nest an OpPassManager under the provided OpPassManager, the nested passmanager will only run on operations matching the provided name. The returned OpPassManager will be destroyed when the parent is destroyed.
"""
function mlirOpPassManagerGetNestedUnder(passManager, operationName)
    @ccall (MLIR_C_PATH[]).mlirOpPassManagerGetNestedUnder(passManager::MlirOpPassManager, operationName::MlirStringRef)::MlirOpPassManager
end

"""
    mlirPassManagerAddOwnedPass(passManager, pass)

Add a pass and transfer ownership to the provided top-level mlirPassManager. If the pass is not a generic operation pass or a ModulePass, a new OpPassManager is implicitly nested under the provided PassManager.
"""
function mlirPassManagerAddOwnedPass(passManager, pass)
    @ccall (MLIR_C_PATH[]).mlirPassManagerAddOwnedPass(passManager::MlirPassManager, pass::MlirPass)::Cvoid
end

"""
    mlirOpPassManagerAddOwnedPass(passManager, pass)

Add a pass and transfer ownership to the provided mlirOpPassManager. If the pass is not a generic operation pass or matching the type of the provided PassManager, a new OpPassManager is implicitly nested under the provided PassManager.
"""
function mlirOpPassManagerAddOwnedPass(passManager, pass)
    @ccall (MLIR_C_PATH[]).mlirOpPassManagerAddOwnedPass(passManager::MlirOpPassManager, pass::MlirPass)::Cvoid
end

"""
    mlirOpPassManagerAddPipeline(passManager, pipelineElements, callback, userData)

Parse a sequence of textual MLIR pass pipeline elements and add them to the provided OpPassManager. If parsing fails an error message is reported using the provided callback.
"""
function mlirOpPassManagerAddPipeline(passManager, pipelineElements, callback, userData)
    @ccall (MLIR_C_PATH[]).mlirOpPassManagerAddPipeline(passManager::MlirOpPassManager, pipelineElements::MlirStringRef, callback::MlirStringCallback, userData::Ptr{Cvoid})::MlirLogicalResult
end

"""
    mlirPrintPassPipeline(passManager, callback, userData)

Print a textual MLIR pass pipeline by sending chunks of the string representation and forwarding `userData to `callback`. Note that the callback may be called several times with consecutive chunks of the string.
"""
function mlirPrintPassPipeline(passManager, callback, userData)
    @ccall (MLIR_C_PATH[]).mlirPrintPassPipeline(passManager::MlirOpPassManager, callback::MlirStringCallback, userData::Ptr{Cvoid})::Cvoid
end

"""
    mlirParsePassPipeline(passManager, pipeline, callback, userData)

Parse a textual MLIR pass pipeline and assign it to the provided OpPassManager. If parsing fails an error message is reported using the provided callback.
"""
function mlirParsePassPipeline(passManager, pipeline, callback, userData)
    @ccall (MLIR_C_PATH[]).mlirParsePassPipeline(passManager::MlirOpPassManager, pipeline::MlirStringRef, callback::MlirStringCallback, userData::Ptr{Cvoid})::MlirLogicalResult
end

"""
    mlirCreateExternalPass(passID, name, argument, description, opName, nDependentDialects, dependentDialects, callbacks, userData)

Creates an external [`MlirPass`](@ref) that calls the supplied `callbacks` using the supplied `userData`. If `opName` is empty, the pass is a generic operation pass. Otherwise it is an operation pass specific to the specified pass name.
"""
function mlirCreateExternalPass(passID, name, argument, description, opName, nDependentDialects, dependentDialects, callbacks, userData)
    @ccall (MLIR_C_PATH[]).mlirCreateExternalPass(passID::MlirTypeID, name::MlirStringRef, argument::MlirStringRef, description::MlirStringRef, opName::MlirStringRef, nDependentDialects::intptr_t, dependentDialects::Ptr{MlirDialectHandle}, callbacks::MlirExternalPassCallbacks, userData::Ptr{Cvoid})::MlirPass
end

"""
    mlirExternalPassSignalFailure(pass)

This signals that the pass has failed. This is only valid to call during the `run` callback of [`MlirExternalPassCallbacks`](@ref). See Pass::signalPassFailure().
"""
function mlirExternalPassSignalFailure(pass)
    @ccall (MLIR_C_PATH[]).mlirExternalPassSignalFailure(pass::MlirExternalPass)::Cvoid
end

function mlirRegisterConversionPasses()
    @ccall (MLIR_C_PATH[]).mlirRegisterConversionPasses()::Cvoid
end

function mlirCreateConversionArithToLLVMConversionPass()
    @ccall (MLIR_C_PATH[]).mlirCreateConversionArithToLLVMConversionPass()::MlirPass
end

function mlirRegisterConversionArithToLLVMConversionPass()
    @ccall (MLIR_C_PATH[]).mlirRegisterConversionArithToLLVMConversionPass()::Cvoid
end

function mlirCreateConversionConvertAMDGPUToROCDL()
    @ccall (MLIR_C_PATH[]).mlirCreateConversionConvertAMDGPUToROCDL()::MlirPass
end

function mlirRegisterConversionConvertAMDGPUToROCDL()
    @ccall (MLIR_C_PATH[]).mlirRegisterConversionConvertAMDGPUToROCDL()::Cvoid
end

function mlirCreateConversionConvertAffineForToGPU()
    @ccall (MLIR_C_PATH[]).mlirCreateConversionConvertAffineForToGPU()::MlirPass
end

function mlirRegisterConversionConvertAffineForToGPU()
    @ccall (MLIR_C_PATH[]).mlirRegisterConversionConvertAffineForToGPU()::Cvoid
end

function mlirCreateConversionConvertAffineToStandard()
    @ccall (MLIR_C_PATH[]).mlirCreateConversionConvertAffineToStandard()::MlirPass
end

function mlirRegisterConversionConvertAffineToStandard()
    @ccall (MLIR_C_PATH[]).mlirRegisterConversionConvertAffineToStandard()::Cvoid
end

function mlirCreateConversionConvertArithToSPIRV()
    @ccall (MLIR_C_PATH[]).mlirCreateConversionConvertArithToSPIRV()::MlirPass
end

function mlirRegisterConversionConvertArithToSPIRV()
    @ccall (MLIR_C_PATH[]).mlirRegisterConversionConvertArithToSPIRV()::Cvoid
end

function mlirCreateConversionConvertArmNeon2dToIntr()
    @ccall (MLIR_C_PATH[]).mlirCreateConversionConvertArmNeon2dToIntr()::MlirPass
end

function mlirRegisterConversionConvertArmNeon2dToIntr()
    @ccall (MLIR_C_PATH[]).mlirRegisterConversionConvertArmNeon2dToIntr()::Cvoid
end

function mlirCreateConversionConvertAsyncToLLVMPass()
    @ccall (MLIR_C_PATH[]).mlirCreateConversionConvertAsyncToLLVMPass()::MlirPass
end

function mlirRegisterConversionConvertAsyncToLLVMPass()
    @ccall (MLIR_C_PATH[]).mlirRegisterConversionConvertAsyncToLLVMPass()::Cvoid
end

function mlirCreateConversionConvertBufferizationToMemRef()
    @ccall (MLIR_C_PATH[]).mlirCreateConversionConvertBufferizationToMemRef()::MlirPass
end

function mlirRegisterConversionConvertBufferizationToMemRef()
    @ccall (MLIR_C_PATH[]).mlirRegisterConversionConvertBufferizationToMemRef()::Cvoid
end

function mlirCreateConversionConvertComplexToLLVMPass()
    @ccall (MLIR_C_PATH[]).mlirCreateConversionConvertComplexToLLVMPass()::MlirPass
end

function mlirRegisterConversionConvertComplexToLLVMPass()
    @ccall (MLIR_C_PATH[]).mlirRegisterConversionConvertComplexToLLVMPass()::Cvoid
end

function mlirCreateConversionConvertComplexToLibm()
    @ccall (MLIR_C_PATH[]).mlirCreateConversionConvertComplexToLibm()::MlirPass
end

function mlirRegisterConversionConvertComplexToLibm()
    @ccall (MLIR_C_PATH[]).mlirRegisterConversionConvertComplexToLibm()::Cvoid
end

function mlirCreateConversionConvertComplexToSPIRVPass()
    @ccall (MLIR_C_PATH[]).mlirCreateConversionConvertComplexToSPIRVPass()::MlirPass
end

function mlirRegisterConversionConvertComplexToSPIRVPass()
    @ccall (MLIR_C_PATH[]).mlirRegisterConversionConvertComplexToSPIRVPass()::Cvoid
end

function mlirCreateConversionConvertComplexToStandard()
    @ccall (MLIR_C_PATH[]).mlirCreateConversionConvertComplexToStandard()::MlirPass
end

function mlirRegisterConversionConvertComplexToStandard()
    @ccall (MLIR_C_PATH[]).mlirRegisterConversionConvertComplexToStandard()::Cvoid
end

function mlirCreateConversionConvertControlFlowToLLVMPass()
    @ccall (MLIR_C_PATH[]).mlirCreateConversionConvertControlFlowToLLVMPass()::MlirPass
end

function mlirRegisterConversionConvertControlFlowToLLVMPass()
    @ccall (MLIR_C_PATH[]).mlirRegisterConversionConvertControlFlowToLLVMPass()::Cvoid
end

function mlirCreateConversionConvertControlFlowToSPIRV()
    @ccall (MLIR_C_PATH[]).mlirCreateConversionConvertControlFlowToSPIRV()::MlirPass
end

function mlirRegisterConversionConvertControlFlowToSPIRV()
    @ccall (MLIR_C_PATH[]).mlirRegisterConversionConvertControlFlowToSPIRV()::Cvoid
end

function mlirCreateConversionConvertFuncToLLVMPass()
    @ccall (MLIR_C_PATH[]).mlirCreateConversionConvertFuncToLLVMPass()::MlirPass
end

function mlirRegisterConversionConvertFuncToLLVMPass()
    @ccall (MLIR_C_PATH[]).mlirRegisterConversionConvertFuncToLLVMPass()::Cvoid
end

function mlirCreateConversionConvertFuncToSPIRV()
    @ccall (MLIR_C_PATH[]).mlirCreateConversionConvertFuncToSPIRV()::MlirPass
end

function mlirRegisterConversionConvertFuncToSPIRV()
    @ccall (MLIR_C_PATH[]).mlirRegisterConversionConvertFuncToSPIRV()::Cvoid
end

function mlirCreateConversionConvertGPUToSPIRV()
    @ccall (MLIR_C_PATH[]).mlirCreateConversionConvertGPUToSPIRV()::MlirPass
end

function mlirRegisterConversionConvertGPUToSPIRV()
    @ccall (MLIR_C_PATH[]).mlirRegisterConversionConvertGPUToSPIRV()::Cvoid
end

function mlirCreateConversionConvertGpuLaunchFuncToVulkanLaunchFunc()
    @ccall (MLIR_C_PATH[]).mlirCreateConversionConvertGpuLaunchFuncToVulkanLaunchFunc()::MlirPass
end

function mlirRegisterConversionConvertGpuLaunchFuncToVulkanLaunchFunc()
    @ccall (MLIR_C_PATH[]).mlirRegisterConversionConvertGpuLaunchFuncToVulkanLaunchFunc()::Cvoid
end

function mlirCreateConversionConvertGpuOpsToNVVMOps()
    @ccall (MLIR_C_PATH[]).mlirCreateConversionConvertGpuOpsToNVVMOps()::MlirPass
end

function mlirRegisterConversionConvertGpuOpsToNVVMOps()
    @ccall (MLIR_C_PATH[]).mlirRegisterConversionConvertGpuOpsToNVVMOps()::Cvoid
end

function mlirCreateConversionConvertGpuOpsToROCDLOps()
    @ccall (MLIR_C_PATH[]).mlirCreateConversionConvertGpuOpsToROCDLOps()::MlirPass
end

function mlirRegisterConversionConvertGpuOpsToROCDLOps()
    @ccall (MLIR_C_PATH[]).mlirRegisterConversionConvertGpuOpsToROCDLOps()::Cvoid
end

function mlirCreateConversionConvertIndexToLLVMPass()
    @ccall (MLIR_C_PATH[]).mlirCreateConversionConvertIndexToLLVMPass()::MlirPass
end

function mlirRegisterConversionConvertIndexToLLVMPass()
    @ccall (MLIR_C_PATH[]).mlirRegisterConversionConvertIndexToLLVMPass()::Cvoid
end

function mlirCreateConversionConvertLinalgToLLVMPass()
    @ccall (MLIR_C_PATH[]).mlirCreateConversionConvertLinalgToLLVMPass()::MlirPass
end

function mlirRegisterConversionConvertLinalgToLLVMPass()
    @ccall (MLIR_C_PATH[]).mlirRegisterConversionConvertLinalgToLLVMPass()::Cvoid
end

function mlirCreateConversionConvertLinalgToStandard()
    @ccall (MLIR_C_PATH[]).mlirCreateConversionConvertLinalgToStandard()::MlirPass
end

function mlirRegisterConversionConvertLinalgToStandard()
    @ccall (MLIR_C_PATH[]).mlirRegisterConversionConvertLinalgToStandard()::Cvoid
end

function mlirCreateConversionConvertMathToFuncs()
    @ccall (MLIR_C_PATH[]).mlirCreateConversionConvertMathToFuncs()::MlirPass
end

function mlirRegisterConversionConvertMathToFuncs()
    @ccall (MLIR_C_PATH[]).mlirRegisterConversionConvertMathToFuncs()::Cvoid
end

function mlirCreateConversionConvertMathToLLVMPass()
    @ccall (MLIR_C_PATH[]).mlirCreateConversionConvertMathToLLVMPass()::MlirPass
end

function mlirRegisterConversionConvertMathToLLVMPass()
    @ccall (MLIR_C_PATH[]).mlirRegisterConversionConvertMathToLLVMPass()::Cvoid
end

function mlirCreateConversionConvertMathToLibm()
    @ccall (MLIR_C_PATH[]).mlirCreateConversionConvertMathToLibm()::MlirPass
end

function mlirRegisterConversionConvertMathToLibm()
    @ccall (MLIR_C_PATH[]).mlirRegisterConversionConvertMathToLibm()::Cvoid
end

function mlirCreateConversionConvertMathToSPIRV()
    @ccall (MLIR_C_PATH[]).mlirCreateConversionConvertMathToSPIRV()::MlirPass
end

function mlirRegisterConversionConvertMathToSPIRV()
    @ccall (MLIR_C_PATH[]).mlirRegisterConversionConvertMathToSPIRV()::Cvoid
end

function mlirCreateConversionConvertMemRefToSPIRV()
    @ccall (MLIR_C_PATH[]).mlirCreateConversionConvertMemRefToSPIRV()::MlirPass
end

function mlirRegisterConversionConvertMemRefToSPIRV()
    @ccall (MLIR_C_PATH[]).mlirRegisterConversionConvertMemRefToSPIRV()::Cvoid
end

function mlirCreateConversionConvertNVGPUToNVVMPass()
    @ccall (MLIR_C_PATH[]).mlirCreateConversionConvertNVGPUToNVVMPass()::MlirPass
end

function mlirRegisterConversionConvertNVGPUToNVVMPass()
    @ccall (MLIR_C_PATH[]).mlirRegisterConversionConvertNVGPUToNVVMPass()::Cvoid
end

function mlirCreateConversionConvertNVVMToLLVMPass()
    @ccall (MLIR_C_PATH[]).mlirCreateConversionConvertNVVMToLLVMPass()::MlirPass
end

function mlirRegisterConversionConvertNVVMToLLVMPass()
    @ccall (MLIR_C_PATH[]).mlirRegisterConversionConvertNVVMToLLVMPass()::Cvoid
end

function mlirCreateConversionConvertOpenACCToSCF()
    @ccall (MLIR_C_PATH[]).mlirCreateConversionConvertOpenACCToSCF()::MlirPass
end

function mlirRegisterConversionConvertOpenACCToSCF()
    @ccall (MLIR_C_PATH[]).mlirRegisterConversionConvertOpenACCToSCF()::Cvoid
end

function mlirCreateConversionConvertOpenMPToLLVMPass()
    @ccall (MLIR_C_PATH[]).mlirCreateConversionConvertOpenMPToLLVMPass()::MlirPass
end

function mlirRegisterConversionConvertOpenMPToLLVMPass()
    @ccall (MLIR_C_PATH[]).mlirRegisterConversionConvertOpenMPToLLVMPass()::Cvoid
end

function mlirCreateConversionConvertPDLToPDLInterp()
    @ccall (MLIR_C_PATH[]).mlirCreateConversionConvertPDLToPDLInterp()::MlirPass
end

function mlirRegisterConversionConvertPDLToPDLInterp()
    @ccall (MLIR_C_PATH[]).mlirRegisterConversionConvertPDLToPDLInterp()::Cvoid
end

function mlirCreateConversionConvertParallelLoopToGpu()
    @ccall (MLIR_C_PATH[]).mlirCreateConversionConvertParallelLoopToGpu()::MlirPass
end

function mlirRegisterConversionConvertParallelLoopToGpu()
    @ccall (MLIR_C_PATH[]).mlirRegisterConversionConvertParallelLoopToGpu()::Cvoid
end

function mlirCreateConversionConvertSCFToOpenMPPass()
    @ccall (MLIR_C_PATH[]).mlirCreateConversionConvertSCFToOpenMPPass()::MlirPass
end

function mlirRegisterConversionConvertSCFToOpenMPPass()
    @ccall (MLIR_C_PATH[]).mlirRegisterConversionConvertSCFToOpenMPPass()::Cvoid
end

function mlirCreateConversionConvertSPIRVToLLVMPass()
    @ccall (MLIR_C_PATH[]).mlirCreateConversionConvertSPIRVToLLVMPass()::MlirPass
end

function mlirRegisterConversionConvertSPIRVToLLVMPass()
    @ccall (MLIR_C_PATH[]).mlirRegisterConversionConvertSPIRVToLLVMPass()::Cvoid
end

function mlirCreateConversionConvertShapeConstraints()
    @ccall (MLIR_C_PATH[]).mlirCreateConversionConvertShapeConstraints()::MlirPass
end

function mlirRegisterConversionConvertShapeConstraints()
    @ccall (MLIR_C_PATH[]).mlirRegisterConversionConvertShapeConstraints()::Cvoid
end

function mlirCreateConversionConvertShapeToStandard()
    @ccall (MLIR_C_PATH[]).mlirCreateConversionConvertShapeToStandard()::MlirPass
end

function mlirRegisterConversionConvertShapeToStandard()
    @ccall (MLIR_C_PATH[]).mlirRegisterConversionConvertShapeToStandard()::Cvoid
end

function mlirCreateConversionConvertTensorToLinalg()
    @ccall (MLIR_C_PATH[]).mlirCreateConversionConvertTensorToLinalg()::MlirPass
end

function mlirRegisterConversionConvertTensorToLinalg()
    @ccall (MLIR_C_PATH[]).mlirRegisterConversionConvertTensorToLinalg()::Cvoid
end

function mlirCreateConversionConvertTensorToSPIRV()
    @ccall (MLIR_C_PATH[]).mlirCreateConversionConvertTensorToSPIRV()::MlirPass
end

function mlirRegisterConversionConvertTensorToSPIRV()
    @ccall (MLIR_C_PATH[]).mlirRegisterConversionConvertTensorToSPIRV()::Cvoid
end

function mlirCreateConversionConvertVectorToArmSME()
    @ccall (MLIR_C_PATH[]).mlirCreateConversionConvertVectorToArmSME()::MlirPass
end

function mlirRegisterConversionConvertVectorToArmSME()
    @ccall (MLIR_C_PATH[]).mlirRegisterConversionConvertVectorToArmSME()::Cvoid
end

function mlirCreateConversionConvertVectorToGPU()
    @ccall (MLIR_C_PATH[]).mlirCreateConversionConvertVectorToGPU()::MlirPass
end

function mlirRegisterConversionConvertVectorToGPU()
    @ccall (MLIR_C_PATH[]).mlirRegisterConversionConvertVectorToGPU()::Cvoid
end

function mlirCreateConversionConvertVectorToLLVMPass()
    @ccall (MLIR_C_PATH[]).mlirCreateConversionConvertVectorToLLVMPass()::MlirPass
end

function mlirRegisterConversionConvertVectorToLLVMPass()
    @ccall (MLIR_C_PATH[]).mlirRegisterConversionConvertVectorToLLVMPass()::Cvoid
end

function mlirCreateConversionConvertVectorToSCF()
    @ccall (MLIR_C_PATH[]).mlirCreateConversionConvertVectorToSCF()::MlirPass
end

function mlirRegisterConversionConvertVectorToSCF()
    @ccall (MLIR_C_PATH[]).mlirRegisterConversionConvertVectorToSCF()::Cvoid
end

function mlirCreateConversionConvertVectorToSPIRV()
    @ccall (MLIR_C_PATH[]).mlirCreateConversionConvertVectorToSPIRV()::MlirPass
end

function mlirRegisterConversionConvertVectorToSPIRV()
    @ccall (MLIR_C_PATH[]).mlirRegisterConversionConvertVectorToSPIRV()::Cvoid
end

function mlirCreateConversionConvertVulkanLaunchFuncToVulkanCallsPass()
    @ccall (MLIR_C_PATH[]).mlirCreateConversionConvertVulkanLaunchFuncToVulkanCallsPass()::MlirPass
end

function mlirRegisterConversionConvertVulkanLaunchFuncToVulkanCallsPass()
    @ccall (MLIR_C_PATH[]).mlirRegisterConversionConvertVulkanLaunchFuncToVulkanCallsPass()::Cvoid
end

function mlirCreateConversionFinalizeMemRefToLLVMConversionPass()
    @ccall (MLIR_C_PATH[]).mlirCreateConversionFinalizeMemRefToLLVMConversionPass()::MlirPass
end

function mlirRegisterConversionFinalizeMemRefToLLVMConversionPass()
    @ccall (MLIR_C_PATH[]).mlirRegisterConversionFinalizeMemRefToLLVMConversionPass()::Cvoid
end

function mlirCreateConversionGpuToLLVMConversionPass()
    @ccall (MLIR_C_PATH[]).mlirCreateConversionGpuToLLVMConversionPass()::MlirPass
end

function mlirRegisterConversionGpuToLLVMConversionPass()
    @ccall (MLIR_C_PATH[]).mlirRegisterConversionGpuToLLVMConversionPass()::Cvoid
end

function mlirCreateConversionLowerHostCodeToLLVMPass()
    @ccall (MLIR_C_PATH[]).mlirCreateConversionLowerHostCodeToLLVMPass()::MlirPass
end

function mlirRegisterConversionLowerHostCodeToLLVMPass()
    @ccall (MLIR_C_PATH[]).mlirRegisterConversionLowerHostCodeToLLVMPass()::Cvoid
end

function mlirCreateConversionMapMemRefStorageClass()
    @ccall (MLIR_C_PATH[]).mlirCreateConversionMapMemRefStorageClass()::MlirPass
end

function mlirRegisterConversionMapMemRefStorageClass()
    @ccall (MLIR_C_PATH[]).mlirRegisterConversionMapMemRefStorageClass()::Cvoid
end

function mlirCreateConversionReconcileUnrealizedCasts()
    @ccall (MLIR_C_PATH[]).mlirCreateConversionReconcileUnrealizedCasts()::MlirPass
end

function mlirRegisterConversionReconcileUnrealizedCasts()
    @ccall (MLIR_C_PATH[]).mlirRegisterConversionReconcileUnrealizedCasts()::Cvoid
end

function mlirCreateConversionSCFToControlFlow()
    @ccall (MLIR_C_PATH[]).mlirCreateConversionSCFToControlFlow()::MlirPass
end

function mlirRegisterConversionSCFToControlFlow()
    @ccall (MLIR_C_PATH[]).mlirRegisterConversionSCFToControlFlow()::Cvoid
end

function mlirCreateConversionSCFToSPIRV()
    @ccall (MLIR_C_PATH[]).mlirCreateConversionSCFToSPIRV()::MlirPass
end

function mlirRegisterConversionSCFToSPIRV()
    @ccall (MLIR_C_PATH[]).mlirRegisterConversionSCFToSPIRV()::Cvoid
end

function mlirCreateConversionTosaToArith()
    @ccall (MLIR_C_PATH[]).mlirCreateConversionTosaToArith()::MlirPass
end

function mlirRegisterConversionTosaToArith()
    @ccall (MLIR_C_PATH[]).mlirRegisterConversionTosaToArith()::Cvoid
end

function mlirCreateConversionTosaToLinalg()
    @ccall (MLIR_C_PATH[]).mlirCreateConversionTosaToLinalg()::MlirPass
end

function mlirRegisterConversionTosaToLinalg()
    @ccall (MLIR_C_PATH[]).mlirRegisterConversionTosaToLinalg()::Cvoid
end

function mlirCreateConversionTosaToLinalgNamed()
    @ccall (MLIR_C_PATH[]).mlirCreateConversionTosaToLinalgNamed()::MlirPass
end

function mlirRegisterConversionTosaToLinalgNamed()
    @ccall (MLIR_C_PATH[]).mlirRegisterConversionTosaToLinalgNamed()::Cvoid
end

function mlirCreateConversionTosaToSCF()
    @ccall (MLIR_C_PATH[]).mlirCreateConversionTosaToSCF()::MlirPass
end

function mlirRegisterConversionTosaToSCF()
    @ccall (MLIR_C_PATH[]).mlirRegisterConversionTosaToSCF()::Cvoid
end

function mlirCreateConversionTosaToTensor()
    @ccall (MLIR_C_PATH[]).mlirCreateConversionTosaToTensor()::MlirPass
end

function mlirRegisterConversionTosaToTensor()
    @ccall (MLIR_C_PATH[]).mlirRegisterConversionTosaToTensor()::Cvoid
end

function mlirCreateConversionUBToLLVMConversionPass()
    @ccall (MLIR_C_PATH[]).mlirCreateConversionUBToLLVMConversionPass()::MlirPass
end

function mlirRegisterConversionUBToLLVMConversionPass()
    @ccall (MLIR_C_PATH[]).mlirRegisterConversionUBToLLVMConversionPass()::Cvoid
end

function mlirCreateConversionUBToSPIRVConversionPass()
    @ccall (MLIR_C_PATH[]).mlirCreateConversionUBToSPIRVConversionPass()::MlirPass
end

function mlirRegisterConversionUBToSPIRVConversionPass()
    @ccall (MLIR_C_PATH[]).mlirRegisterConversionUBToSPIRVConversionPass()::Cvoid
end

"""
    mlirEnableGlobalDebug(enable)

Sets the global debugging flag.
"""
function mlirEnableGlobalDebug(enable)
    @ccall (MLIR_C_PATH[]).mlirEnableGlobalDebug(enable::Bool)::Cvoid
end

"""
    mlirIsGlobalDebugEnabled()

Retuns `true` if the global debugging flag is set, false otherwise.
"""
function mlirIsGlobalDebugEnabled()
    @ccall (MLIR_C_PATH[]).mlirIsGlobalDebugEnabled()::Bool
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
    @ccall (MLIR_C_PATH[]).mlirDiagnosticPrint(diagnostic::MlirDiagnostic, callback::MlirStringCallback, userData::Ptr{Cvoid})::Cvoid
end

"""
    mlirDiagnosticGetLocation(diagnostic)

Returns the location at which the diagnostic is reported.
"""
function mlirDiagnosticGetLocation(diagnostic)
    @ccall (MLIR_C_PATH[]).mlirDiagnosticGetLocation(diagnostic::MlirDiagnostic)::MlirLocation
end

"""
    mlirDiagnosticGetSeverity(diagnostic)

Returns the severity of the diagnostic.
"""
function mlirDiagnosticGetSeverity(diagnostic)
    @ccall (MLIR_C_PATH[]).mlirDiagnosticGetSeverity(diagnostic::MlirDiagnostic)::MlirDiagnosticSeverity
end

"""
    mlirDiagnosticGetNumNotes(diagnostic)

Returns the number of notes attached to the diagnostic.
"""
function mlirDiagnosticGetNumNotes(diagnostic)
    @ccall (MLIR_C_PATH[]).mlirDiagnosticGetNumNotes(diagnostic::MlirDiagnostic)::intptr_t
end

"""
    mlirDiagnosticGetNote(diagnostic, pos)

Returns `pos`-th note attached to the diagnostic. Expects `pos` to be a valid zero-based index into the list of notes.
"""
function mlirDiagnosticGetNote(diagnostic, pos)
    @ccall (MLIR_C_PATH[]).mlirDiagnosticGetNote(diagnostic::MlirDiagnostic, pos::intptr_t)::MlirDiagnostic
end

"""
    mlirContextAttachDiagnosticHandler(context, handler, userData, deleteUserData)

Attaches the diagnostic handler to the context. Handlers are invoked in the reverse order of attachment until one of them processes the diagnostic completely. When a handler is invoked it is passed the `userData` that was provided when it was attached. If non-NULL, `deleteUserData` is called once the system no longer needs to call the handler (for instance after the handler is detached or the context is destroyed). Returns an identifier that can be used to detach the handler.
"""
function mlirContextAttachDiagnosticHandler(context, handler, userData, deleteUserData)
    @ccall (MLIR_C_PATH[]).mlirContextAttachDiagnosticHandler(context::MlirContext, handler::MlirDiagnosticHandler, userData::Ptr{Cvoid}, deleteUserData::Ptr{Cvoid})::MlirDiagnosticHandlerID
end

"""
    mlirContextDetachDiagnosticHandler(context, id)

Detaches an attached diagnostic handler from the context given its identifier.
"""
function mlirContextDetachDiagnosticHandler(context, id)
    @ccall (MLIR_C_PATH[]).mlirContextDetachDiagnosticHandler(context::MlirContext, id::MlirDiagnosticHandlerID)::Cvoid
end

"""
    mlirEmitError(location, message)

Emits an error at the given location through the diagnostics engine. Used for testing purposes.
"""
function mlirEmitError(location, message)
    @ccall (MLIR_C_PATH[]).mlirEmitError(location::MlirLocation, message::Cstring)::Cvoid
end

function mlirGetDialectHandle__arith__()
    @ccall (MLIR_C_PATH[]).mlirGetDialectHandle__arith__()::MlirDialectHandle
end

function mlirGetDialectHandle__async__()
    @ccall (MLIR_C_PATH[]).mlirGetDialectHandle__async__()::MlirDialectHandle
end

function mlirRegisterAsyncPasses()
    @ccall (MLIR_C_PATH[]).mlirRegisterAsyncPasses()::Cvoid
end

function mlirCreateAsyncAsyncFuncToAsyncRuntime()
    @ccall (MLIR_C_PATH[]).mlirCreateAsyncAsyncFuncToAsyncRuntime()::MlirPass
end

function mlirRegisterAsyncAsyncFuncToAsyncRuntime()
    @ccall (MLIR_C_PATH[]).mlirRegisterAsyncAsyncFuncToAsyncRuntime()::Cvoid
end

function mlirCreateAsyncAsyncParallelFor()
    @ccall (MLIR_C_PATH[]).mlirCreateAsyncAsyncParallelFor()::MlirPass
end

function mlirRegisterAsyncAsyncParallelFor()
    @ccall (MLIR_C_PATH[]).mlirRegisterAsyncAsyncParallelFor()::Cvoid
end

function mlirCreateAsyncAsyncRuntimePolicyBasedRefCounting()
    @ccall (MLIR_C_PATH[]).mlirCreateAsyncAsyncRuntimePolicyBasedRefCounting()::MlirPass
end

function mlirRegisterAsyncAsyncRuntimePolicyBasedRefCounting()
    @ccall (MLIR_C_PATH[]).mlirRegisterAsyncAsyncRuntimePolicyBasedRefCounting()::Cvoid
end

function mlirCreateAsyncAsyncRuntimeRefCounting()
    @ccall (MLIR_C_PATH[]).mlirCreateAsyncAsyncRuntimeRefCounting()::MlirPass
end

function mlirRegisterAsyncAsyncRuntimeRefCounting()
    @ccall (MLIR_C_PATH[]).mlirRegisterAsyncAsyncRuntimeRefCounting()::Cvoid
end

function mlirCreateAsyncAsyncRuntimeRefCountingOpt()
    @ccall (MLIR_C_PATH[]).mlirCreateAsyncAsyncRuntimeRefCountingOpt()::MlirPass
end

function mlirRegisterAsyncAsyncRuntimeRefCountingOpt()
    @ccall (MLIR_C_PATH[]).mlirRegisterAsyncAsyncRuntimeRefCountingOpt()::Cvoid
end

function mlirCreateAsyncAsyncToAsyncRuntime()
    @ccall (MLIR_C_PATH[]).mlirCreateAsyncAsyncToAsyncRuntime()::MlirPass
end

function mlirRegisterAsyncAsyncToAsyncRuntime()
    @ccall (MLIR_C_PATH[]).mlirRegisterAsyncAsyncToAsyncRuntime()::Cvoid
end

function mlirGetDialectHandle__cf__()
    @ccall (MLIR_C_PATH[]).mlirGetDialectHandle__cf__()::MlirDialectHandle
end

function mlirGetDialectHandle__func__()
    @ccall (MLIR_C_PATH[]).mlirGetDialectHandle__func__()::MlirDialectHandle
end

"""
    mlirFuncSetArgAttr(op, pos, name, attr)

Sets the argument attribute 'name' of an argument at index 'pos'. Asserts that the operation is a FuncOp.
"""
function mlirFuncSetArgAttr(op, pos, name, attr)
    @ccall (MLIR_C_PATH[]).mlirFuncSetArgAttr(op::MlirOperation, pos::intptr_t, name::MlirStringRef, attr::MlirAttribute)::Cvoid
end

function mlirGetDialectHandle__gpu__()
    @ccall (MLIR_C_PATH[]).mlirGetDialectHandle__gpu__()::MlirDialectHandle
end

function mlirRegisterGPUPasses()
    @ccall (MLIR_C_PATH[]).mlirRegisterGPUPasses()::Cvoid
end

function mlirCreateGPUGpuAsyncRegionPass()
    @ccall (MLIR_C_PATH[]).mlirCreateGPUGpuAsyncRegionPass()::MlirPass
end

function mlirRegisterGPUGpuAsyncRegionPass()
    @ccall (MLIR_C_PATH[]).mlirRegisterGPUGpuAsyncRegionPass()::Cvoid
end

function mlirCreateGPUGpuKernelOutlining()
    @ccall (MLIR_C_PATH[]).mlirCreateGPUGpuKernelOutlining()::MlirPass
end

function mlirRegisterGPUGpuKernelOutlining()
    @ccall (MLIR_C_PATH[]).mlirRegisterGPUGpuKernelOutlining()::Cvoid
end

function mlirCreateGPUGpuLaunchSinkIndexComputations()
    @ccall (MLIR_C_PATH[]).mlirCreateGPUGpuLaunchSinkIndexComputations()::MlirPass
end

function mlirRegisterGPUGpuLaunchSinkIndexComputations()
    @ccall (MLIR_C_PATH[]).mlirRegisterGPUGpuLaunchSinkIndexComputations()::Cvoid
end

function mlirCreateGPUGpuMapParallelLoopsPass()
    @ccall (MLIR_C_PATH[]).mlirCreateGPUGpuMapParallelLoopsPass()::MlirPass
end

function mlirRegisterGPUGpuMapParallelLoopsPass()
    @ccall (MLIR_C_PATH[]).mlirRegisterGPUGpuMapParallelLoopsPass()::Cvoid
end

function mlirGetDialectHandle__llvm__()
    @ccall (MLIR_C_PATH[]).mlirGetDialectHandle__llvm__()::MlirDialectHandle
end

"""
    mlirLLVMPointerTypeGet(pointee, addressSpace)

Creates an llvm.ptr type.
"""
function mlirLLVMPointerTypeGet(pointee, addressSpace)
    @ccall (MLIR_C_PATH[]).mlirLLVMPointerTypeGet(pointee::MlirType, addressSpace::Cuint)::MlirType
end

"""
    mlirLLVMVoidTypeGet(ctx)

Creates an llmv.void type.
"""
function mlirLLVMVoidTypeGet(ctx)
    @ccall (MLIR_C_PATH[]).mlirLLVMVoidTypeGet(ctx::MlirContext)::MlirType
end

"""
    mlirLLVMArrayTypeGet(elementType, numElements)

Creates an llvm.array type.
"""
function mlirLLVMArrayTypeGet(elementType, numElements)
    @ccall (MLIR_C_PATH[]).mlirLLVMArrayTypeGet(elementType::MlirType, numElements::Cuint)::MlirType
end

"""
    mlirLLVMFunctionTypeGet(resultType, nArgumentTypes, argumentTypes, isVarArg)

Creates an llvm.func type.
"""
function mlirLLVMFunctionTypeGet(resultType, nArgumentTypes, argumentTypes, isVarArg)
    @ccall (MLIR_C_PATH[]).mlirLLVMFunctionTypeGet(resultType::MlirType, nArgumentTypes::intptr_t, argumentTypes::Ptr{MlirType}, isVarArg::Bool)::MlirType
end

"""
    mlirLLVMStructTypeLiteralGet(ctx, nFieldTypes, fieldTypes, isPacked)

Creates an LLVM literal (unnamed) struct type.
"""
function mlirLLVMStructTypeLiteralGet(ctx, nFieldTypes, fieldTypes, isPacked)
    @ccall (MLIR_C_PATH[]).mlirLLVMStructTypeLiteralGet(ctx::MlirContext, nFieldTypes::intptr_t, fieldTypes::Ptr{MlirType}, isPacked::Bool)::MlirType
end

"""
    mlirLinalgFillBuiltinNamedOpRegion(mlirOp)

Apply the special region builder for the builtin named Linalg op. Assert that `mlirOp` is a builtin named Linalg op.
"""
function mlirLinalgFillBuiltinNamedOpRegion(mlirOp)
    @ccall (MLIR_C_PATH[]).mlirLinalgFillBuiltinNamedOpRegion(mlirOp::MlirOperation)::Cvoid
end

function mlirGetDialectHandle__linalg__()
    @ccall (MLIR_C_PATH[]).mlirGetDialectHandle__linalg__()::MlirDialectHandle
end

function mlirRegisterLinalgPasses()
    @ccall (MLIR_C_PATH[]).mlirRegisterLinalgPasses()::Cvoid
end

function mlirCreateLinalgConvertElementwiseToLinalg()
    @ccall (MLIR_C_PATH[]).mlirCreateLinalgConvertElementwiseToLinalg()::MlirPass
end

function mlirRegisterLinalgConvertElementwiseToLinalg()
    @ccall (MLIR_C_PATH[]).mlirRegisterLinalgConvertElementwiseToLinalg()::Cvoid
end

function mlirCreateLinalgLinalgBufferize()
    @ccall (MLIR_C_PATH[]).mlirCreateLinalgLinalgBufferize()::MlirPass
end

function mlirRegisterLinalgLinalgBufferize()
    @ccall (MLIR_C_PATH[]).mlirRegisterLinalgLinalgBufferize()::Cvoid
end

function mlirCreateLinalgLinalgDetensorize()
    @ccall (MLIR_C_PATH[]).mlirCreateLinalgLinalgDetensorize()::MlirPass
end

function mlirRegisterLinalgLinalgDetensorize()
    @ccall (MLIR_C_PATH[]).mlirRegisterLinalgLinalgDetensorize()::Cvoid
end

function mlirCreateLinalgLinalgElementwiseOpFusion()
    @ccall (MLIR_C_PATH[]).mlirCreateLinalgLinalgElementwiseOpFusion()::MlirPass
end

function mlirRegisterLinalgLinalgElementwiseOpFusion()
    @ccall (MLIR_C_PATH[]).mlirRegisterLinalgLinalgElementwiseOpFusion()::Cvoid
end

function mlirCreateLinalgLinalgFoldUnitExtentDims()
    @ccall (MLIR_C_PATH[]).mlirCreateLinalgLinalgFoldUnitExtentDims()::MlirPass
end

function mlirRegisterLinalgLinalgFoldUnitExtentDims()
    @ccall (MLIR_C_PATH[]).mlirRegisterLinalgLinalgFoldUnitExtentDims()::Cvoid
end

function mlirCreateLinalgLinalgGeneralization()
    @ccall (MLIR_C_PATH[]).mlirCreateLinalgLinalgGeneralization()::MlirPass
end

function mlirRegisterLinalgLinalgGeneralization()
    @ccall (MLIR_C_PATH[]).mlirRegisterLinalgLinalgGeneralization()::Cvoid
end

function mlirCreateLinalgLinalgInlineScalarOperands()
    @ccall (MLIR_C_PATH[]).mlirCreateLinalgLinalgInlineScalarOperands()::MlirPass
end

function mlirRegisterLinalgLinalgInlineScalarOperands()
    @ccall (MLIR_C_PATH[]).mlirRegisterLinalgLinalgInlineScalarOperands()::Cvoid
end

function mlirCreateLinalgLinalgLowerToAffineLoops()
    @ccall (MLIR_C_PATH[]).mlirCreateLinalgLinalgLowerToAffineLoops()::MlirPass
end

function mlirRegisterLinalgLinalgLowerToAffineLoops()
    @ccall (MLIR_C_PATH[]).mlirRegisterLinalgLinalgLowerToAffineLoops()::Cvoid
end

function mlirCreateLinalgLinalgLowerToLoops()
    @ccall (MLIR_C_PATH[]).mlirCreateLinalgLinalgLowerToLoops()::MlirPass
end

function mlirRegisterLinalgLinalgLowerToLoops()
    @ccall (MLIR_C_PATH[]).mlirRegisterLinalgLinalgLowerToLoops()::Cvoid
end

function mlirCreateLinalgLinalgLowerToParallelLoops()
    @ccall (MLIR_C_PATH[]).mlirCreateLinalgLinalgLowerToParallelLoops()::MlirPass
end

function mlirRegisterLinalgLinalgLowerToParallelLoops()
    @ccall (MLIR_C_PATH[]).mlirRegisterLinalgLinalgLowerToParallelLoops()::Cvoid
end

function mlirCreateLinalgLinalgNamedOpConversion()
    @ccall (MLIR_C_PATH[]).mlirCreateLinalgLinalgNamedOpConversion()::MlirPass
end

function mlirRegisterLinalgLinalgNamedOpConversion()
    @ccall (MLIR_C_PATH[]).mlirRegisterLinalgLinalgNamedOpConversion()::Cvoid
end

function mlirGetDialectHandle__ml_program__()
    @ccall (MLIR_C_PATH[]).mlirGetDialectHandle__ml_program__()::MlirDialectHandle
end

function mlirGetDialectHandle__math__()
    @ccall (MLIR_C_PATH[]).mlirGetDialectHandle__math__()::MlirDialectHandle
end

function mlirGetDialectHandle__memref__()
    @ccall (MLIR_C_PATH[]).mlirGetDialectHandle__memref__()::MlirDialectHandle
end

function mlirGetDialectHandle__pdl__()
    @ccall (MLIR_C_PATH[]).mlirGetDialectHandle__pdl__()::MlirDialectHandle
end

function mlirTypeIsAPDLType(type)
    @ccall (MLIR_C_PATH[]).mlirTypeIsAPDLType(type::MlirType)::Bool
end

function mlirTypeIsAPDLAttributeType(type)
    @ccall (MLIR_C_PATH[]).mlirTypeIsAPDLAttributeType(type::MlirType)::Bool
end

function mlirPDLAttributeTypeGet(ctx)
    @ccall (MLIR_C_PATH[]).mlirPDLAttributeTypeGet(ctx::MlirContext)::MlirType
end

function mlirTypeIsAPDLOperationType(type)
    @ccall (MLIR_C_PATH[]).mlirTypeIsAPDLOperationType(type::MlirType)::Bool
end

function mlirPDLOperationTypeGet(ctx)
    @ccall (MLIR_C_PATH[]).mlirPDLOperationTypeGet(ctx::MlirContext)::MlirType
end

function mlirTypeIsAPDLRangeType(type)
    @ccall (MLIR_C_PATH[]).mlirTypeIsAPDLRangeType(type::MlirType)::Bool
end

function mlirPDLRangeTypeGet(elementType)
    @ccall (MLIR_C_PATH[]).mlirPDLRangeTypeGet(elementType::MlirType)::MlirType
end

function mlirPDLRangeTypeGetElementType(type)
    @ccall (MLIR_C_PATH[]).mlirPDLRangeTypeGetElementType(type::MlirType)::MlirType
end

function mlirTypeIsAPDLTypeType(type)
    @ccall (MLIR_C_PATH[]).mlirTypeIsAPDLTypeType(type::MlirType)::Bool
end

function mlirPDLTypeTypeGet(ctx)
    @ccall (MLIR_C_PATH[]).mlirPDLTypeTypeGet(ctx::MlirContext)::MlirType
end

function mlirTypeIsAPDLValueType(type)
    @ccall (MLIR_C_PATH[]).mlirTypeIsAPDLValueType(type::MlirType)::Bool
end

function mlirPDLValueTypeGet(ctx)
    @ccall (MLIR_C_PATH[]).mlirPDLValueTypeGet(ctx::MlirContext)::MlirType
end

function mlirGetDialectHandle__quant__()
    @ccall (MLIR_C_PATH[]).mlirGetDialectHandle__quant__()::MlirDialectHandle
end

"""
    mlirTypeIsAQuantizedType(type)

Returns `true` if the given type is a quantization dialect type.
"""
function mlirTypeIsAQuantizedType(type)
    @ccall (MLIR_C_PATH[]).mlirTypeIsAQuantizedType(type::MlirType)::Bool
end

"""
    mlirQuantizedTypeGetSignedFlag()

Returns the bit flag used to indicate signedness of a quantized type.
"""
function mlirQuantizedTypeGetSignedFlag()
    @ccall (MLIR_C_PATH[]).mlirQuantizedTypeGetSignedFlag()::Cuint
end

"""
    mlirQuantizedTypeGetDefaultMinimumForInteger(isSigned, integralWidth)

Returns the minimum possible value stored by a quantized type.
"""
function mlirQuantizedTypeGetDefaultMinimumForInteger(isSigned, integralWidth)
    @ccall (MLIR_C_PATH[]).mlirQuantizedTypeGetDefaultMinimumForInteger(isSigned::Bool, integralWidth::Cuint)::Int64
end

"""
    mlirQuantizedTypeGetDefaultMaximumForInteger(isSigned, integralWidth)

Returns the maximum possible value stored by a quantized type.
"""
function mlirQuantizedTypeGetDefaultMaximumForInteger(isSigned, integralWidth)
    @ccall (MLIR_C_PATH[]).mlirQuantizedTypeGetDefaultMaximumForInteger(isSigned::Bool, integralWidth::Cuint)::Int64
end

"""
    mlirQuantizedTypeGetExpressedType(type)

Gets the original type approximated by the given quantized type.
"""
function mlirQuantizedTypeGetExpressedType(type)
    @ccall (MLIR_C_PATH[]).mlirQuantizedTypeGetExpressedType(type::MlirType)::MlirType
end

"""
    mlirQuantizedTypeGetFlags(type)

Gets the flags associated with the given quantized type.
"""
function mlirQuantizedTypeGetFlags(type)
    @ccall (MLIR_C_PATH[]).mlirQuantizedTypeGetFlags(type::MlirType)::Cuint
end

"""
    mlirQuantizedTypeIsSigned(type)

Returns `true` if the given type is signed, `false` otherwise.
"""
function mlirQuantizedTypeIsSigned(type)
    @ccall (MLIR_C_PATH[]).mlirQuantizedTypeIsSigned(type::MlirType)::Bool
end

"""
    mlirQuantizedTypeGetStorageType(type)

Returns the underlying type used to store the values.
"""
function mlirQuantizedTypeGetStorageType(type)
    @ccall (MLIR_C_PATH[]).mlirQuantizedTypeGetStorageType(type::MlirType)::MlirType
end

"""
    mlirQuantizedTypeGetStorageTypeMin(type)

Returns the minimum value that the storage type of the given quantized type can take.
"""
function mlirQuantizedTypeGetStorageTypeMin(type)
    @ccall (MLIR_C_PATH[]).mlirQuantizedTypeGetStorageTypeMin(type::MlirType)::Int64
end

"""
    mlirQuantizedTypeGetStorageTypeMax(type)

Returns the maximum value that the storage type of the given quantized type can take.
"""
function mlirQuantizedTypeGetStorageTypeMax(type)
    @ccall (MLIR_C_PATH[]).mlirQuantizedTypeGetStorageTypeMax(type::MlirType)::Int64
end

"""
    mlirQuantizedTypeGetStorageTypeIntegralWidth(type)

Returns the integral bitwidth that the storage type of the given quantized type can represent exactly.
"""
function mlirQuantizedTypeGetStorageTypeIntegralWidth(type)
    @ccall (MLIR_C_PATH[]).mlirQuantizedTypeGetStorageTypeIntegralWidth(type::MlirType)::Cuint
end

"""
    mlirQuantizedTypeIsCompatibleExpressedType(type, candidate)

Returns `true` if the `candidate` type is compatible with the given quantized `type`.
"""
function mlirQuantizedTypeIsCompatibleExpressedType(type, candidate)
    @ccall (MLIR_C_PATH[]).mlirQuantizedTypeIsCompatibleExpressedType(type::MlirType, candidate::MlirType)::Bool
end

"""
    mlirQuantizedTypeGetQuantizedElementType(type)

Returns the element type of the given quantized type as another quantized type.
"""
function mlirQuantizedTypeGetQuantizedElementType(type)
    @ccall (MLIR_C_PATH[]).mlirQuantizedTypeGetQuantizedElementType(type::MlirType)::MlirType
end

"""
    mlirQuantizedTypeCastFromStorageType(type, candidate)

Casts from a type based on the storage type of the given type to a corresponding type based on the given type. Returns a null type if the cast is not valid.
"""
function mlirQuantizedTypeCastFromStorageType(type, candidate)
    @ccall (MLIR_C_PATH[]).mlirQuantizedTypeCastFromStorageType(type::MlirType, candidate::MlirType)::MlirType
end

"""
    mlirQuantizedTypeCastToStorageType(type)

Casts from a type based on a quantized type to a corresponding typed based on the storage type. Returns a null type if the cast is not valid.
"""
function mlirQuantizedTypeCastToStorageType(type)
    @ccall (MLIR_C_PATH[]).mlirQuantizedTypeCastToStorageType(type::MlirType)::MlirType
end

"""
    mlirQuantizedTypeCastFromExpressedType(type, candidate)

Casts from a type based on the expressed type of the given type to a corresponding type based on the given type. Returns a null type if the cast is not valid.
"""
function mlirQuantizedTypeCastFromExpressedType(type, candidate)
    @ccall (MLIR_C_PATH[]).mlirQuantizedTypeCastFromExpressedType(type::MlirType, candidate::MlirType)::MlirType
end

"""
    mlirQuantizedTypeCastToExpressedType(type)

Casts from a type based on a quantized type to a corresponding typed based on the expressed type. Returns a null type if the cast is not valid.
"""
function mlirQuantizedTypeCastToExpressedType(type)
    @ccall (MLIR_C_PATH[]).mlirQuantizedTypeCastToExpressedType(type::MlirType)::MlirType
end

"""
    mlirQuantizedTypeCastExpressedToStorageType(type, candidate)

Casts from a type based on the expressed type of the given quantized type to equivalent type based on storage type of the same quantized type.
"""
function mlirQuantizedTypeCastExpressedToStorageType(type, candidate)
    @ccall (MLIR_C_PATH[]).mlirQuantizedTypeCastExpressedToStorageType(type::MlirType, candidate::MlirType)::MlirType
end

"""
    mlirTypeIsAAnyQuantizedType(type)

Returns `true` if the given type is an AnyQuantizedType.
"""
function mlirTypeIsAAnyQuantizedType(type)
    @ccall (MLIR_C_PATH[]).mlirTypeIsAAnyQuantizedType(type::MlirType)::Bool
end

"""
    mlirAnyQuantizedTypeGet(flags, storageType, expressedType, storageTypeMin, storageTypeMax)

Creates an instance of AnyQuantizedType with the given parameters in the same context as `storageType` and returns it. The instance is owned by the context.
"""
function mlirAnyQuantizedTypeGet(flags, storageType, expressedType, storageTypeMin, storageTypeMax)
    @ccall (MLIR_C_PATH[]).mlirAnyQuantizedTypeGet(flags::Cuint, storageType::MlirType, expressedType::MlirType, storageTypeMin::Int64, storageTypeMax::Int64)::MlirType
end

"""
    mlirTypeIsAUniformQuantizedType(type)

Returns `true` if the given type is a UniformQuantizedType.
"""
function mlirTypeIsAUniformQuantizedType(type)
    @ccall (MLIR_C_PATH[]).mlirTypeIsAUniformQuantizedType(type::MlirType)::Bool
end

"""
    mlirUniformQuantizedTypeGet(flags, storageType, expressedType, scale, zeroPoint, storageTypeMin, storageTypeMax)

Creates an instance of UniformQuantizedType with the given parameters in the same context as `storageType` and returns it. The instance is owned by the context.
"""
function mlirUniformQuantizedTypeGet(flags, storageType, expressedType, scale, zeroPoint, storageTypeMin, storageTypeMax)
    @ccall (MLIR_C_PATH[]).mlirUniformQuantizedTypeGet(flags::Cuint, storageType::MlirType, expressedType::MlirType, scale::Cdouble, zeroPoint::Int64, storageTypeMin::Int64, storageTypeMax::Int64)::MlirType
end

"""
    mlirUniformQuantizedTypeGetScale(type)

Returns the scale of the given uniform quantized type.
"""
function mlirUniformQuantizedTypeGetScale(type)
    @ccall (MLIR_C_PATH[]).mlirUniformQuantizedTypeGetScale(type::MlirType)::Cdouble
end

"""
    mlirUniformQuantizedTypeGetZeroPoint(type)

Returns the zero point of the given uniform quantized type.
"""
function mlirUniformQuantizedTypeGetZeroPoint(type)
    @ccall (MLIR_C_PATH[]).mlirUniformQuantizedTypeGetZeroPoint(type::MlirType)::Int64
end

"""
    mlirUniformQuantizedTypeIsFixedPoint(type)

Returns `true` if the given uniform quantized type is fixed-point.
"""
function mlirUniformQuantizedTypeIsFixedPoint(type)
    @ccall (MLIR_C_PATH[]).mlirUniformQuantizedTypeIsFixedPoint(type::MlirType)::Bool
end

"""
    mlirTypeIsAUniformQuantizedPerAxisType(type)

Returns `true` if the given type is a UniformQuantizedPerAxisType.
"""
function mlirTypeIsAUniformQuantizedPerAxisType(type)
    @ccall (MLIR_C_PATH[]).mlirTypeIsAUniformQuantizedPerAxisType(type::MlirType)::Bool
end

"""
    mlirUniformQuantizedPerAxisTypeGet(flags, storageType, expressedType, nDims, scales, zeroPoints, quantizedDimension, storageTypeMin, storageTypeMax)

Creates an instance of UniformQuantizedPerAxisType with the given parameters in the same context as `storageType` and returns it. `scales` and `zeroPoints` point to `nDims` number of elements. The instance is owned by the context.
"""
function mlirUniformQuantizedPerAxisTypeGet(flags, storageType, expressedType, nDims, scales, zeroPoints, quantizedDimension, storageTypeMin, storageTypeMax)
    @ccall (MLIR_C_PATH[]).mlirUniformQuantizedPerAxisTypeGet(flags::Cuint, storageType::MlirType, expressedType::MlirType, nDims::intptr_t, scales::Ptr{Cdouble}, zeroPoints::Ptr{Int64}, quantizedDimension::Int32, storageTypeMin::Int64, storageTypeMax::Int64)::MlirType
end

"""
    mlirUniformQuantizedPerAxisTypeGetNumDims(type)

Returns the number of axes in the given quantized per-axis type.
"""
function mlirUniformQuantizedPerAxisTypeGetNumDims(type)
    @ccall (MLIR_C_PATH[]).mlirUniformQuantizedPerAxisTypeGetNumDims(type::MlirType)::intptr_t
end

"""
    mlirUniformQuantizedPerAxisTypeGetScale(type, pos)

Returns `pos`-th scale of the given quantized per-axis type.
"""
function mlirUniformQuantizedPerAxisTypeGetScale(type, pos)
    @ccall (MLIR_C_PATH[]).mlirUniformQuantizedPerAxisTypeGetScale(type::MlirType, pos::intptr_t)::Cdouble
end

"""
    mlirUniformQuantizedPerAxisTypeGetZeroPoint(type, pos)

Returns `pos`-th zero point of the given quantized per-axis type.
"""
function mlirUniformQuantizedPerAxisTypeGetZeroPoint(type, pos)
    @ccall (MLIR_C_PATH[]).mlirUniformQuantizedPerAxisTypeGetZeroPoint(type::MlirType, pos::intptr_t)::Int64
end

"""
    mlirUniformQuantizedPerAxisTypeGetQuantizedDimension(type)

Returns the index of the quantized dimension in the given quantized per-axis type.
"""
function mlirUniformQuantizedPerAxisTypeGetQuantizedDimension(type)
    @ccall (MLIR_C_PATH[]).mlirUniformQuantizedPerAxisTypeGetQuantizedDimension(type::MlirType)::Int32
end

"""
    mlirUniformQuantizedPerAxisTypeIsFixedPoint(type)

Returns `true` if the given uniform quantized per-axis type is fixed-point.
"""
function mlirUniformQuantizedPerAxisTypeIsFixedPoint(type)
    @ccall (MLIR_C_PATH[]).mlirUniformQuantizedPerAxisTypeIsFixedPoint(type::MlirType)::Bool
end

"""
    mlirTypeIsACalibratedQuantizedType(type)

Returns `true` if the given type is a CalibratedQuantizedType.
"""
function mlirTypeIsACalibratedQuantizedType(type)
    @ccall (MLIR_C_PATH[]).mlirTypeIsACalibratedQuantizedType(type::MlirType)::Bool
end

"""
    mlirCalibratedQuantizedTypeGet(expressedType, min, max)

Creates an instance of CalibratedQuantizedType with the given parameters in the same context as `expressedType` and returns it. The instance is owned by the context.
"""
function mlirCalibratedQuantizedTypeGet(expressedType, min, max)
    @ccall (MLIR_C_PATH[]).mlirCalibratedQuantizedTypeGet(expressedType::MlirType, min::Cdouble, max::Cdouble)::MlirType
end

"""
    mlirCalibratedQuantizedTypeGetMin(type)

Returns the min value of the given calibrated quantized type.
"""
function mlirCalibratedQuantizedTypeGetMin(type)
    @ccall (MLIR_C_PATH[]).mlirCalibratedQuantizedTypeGetMin(type::MlirType)::Cdouble
end

"""
    mlirCalibratedQuantizedTypeGetMax(type)

Returns the max value of the given calibrated quantized type.
"""
function mlirCalibratedQuantizedTypeGetMax(type)
    @ccall (MLIR_C_PATH[]).mlirCalibratedQuantizedTypeGetMax(type::MlirType)::Cdouble
end

function mlirGetDialectHandle__scf__()
    @ccall (MLIR_C_PATH[]).mlirGetDialectHandle__scf__()::MlirDialectHandle
end

function mlirGetDialectHandle__shape__()
    @ccall (MLIR_C_PATH[]).mlirGetDialectHandle__shape__()::MlirDialectHandle
end

function mlirGetDialectHandle__sparse_tensor__()
    @ccall (MLIR_C_PATH[]).mlirGetDialectHandle__sparse_tensor__()::MlirDialectHandle
end

"""
    MlirSparseTensorDimLevelType

Dimension level types (and properties) that define sparse tensors. See the documentation in SparseTensorAttrDefs.td for their meaning.

These correspond to SparseTensorEncodingAttr::DimLevelType in the C++ API. If updating, keep them in sync and update the static\\_assert in the impl file.
"""
@cenum MlirSparseTensorDimLevelType::UInt32 begin
    MLIR_SPARSE_TENSOR_DIM_LEVEL_DENSE = 0x0000000000000004
    MLIR_SPARSE_TENSOR_DIM_LEVEL_COMPRESSED = 0x0000000000000008
    MLIR_SPARSE_TENSOR_DIM_LEVEL_COMPRESSED_NU = 0x0000000000000009
    MLIR_SPARSE_TENSOR_DIM_LEVEL_COMPRESSED_NO = 0x000000000000000a
    MLIR_SPARSE_TENSOR_DIM_LEVEL_COMPRESSED_NU_NO = 0x000000000000000b
    MLIR_SPARSE_TENSOR_DIM_LEVEL_SINGLETON = 0x0000000000000010
    MLIR_SPARSE_TENSOR_DIM_LEVEL_SINGLETON_NU = 0x0000000000000011
    MLIR_SPARSE_TENSOR_DIM_LEVEL_SINGLETON_NO = 0x0000000000000012
    MLIR_SPARSE_TENSOR_DIM_LEVEL_SINGLETON_NU_NO = 0x0000000000000013
    MLIR_SPARSE_TENSOR_DIM_LEVEL_COMPRESSED_WITH_HI = 0x0000000000000020
    MLIR_SPARSE_TENSOR_DIM_LEVEL_COMPRESSED_WITH_HI_NU = 0x0000000000000021
    MLIR_SPARSE_TENSOR_DIM_LEVEL_COMPRESSED_WITH_HI_NO = 0x0000000000000022
    MLIR_SPARSE_TENSOR_DIM_LEVEL_COMPRESSED_WITH_HI_NU_NO = 0x0000000000000023
    MLIR_SPARSE_TENSOR_DIM_LEVEL_TWO_OUT_OF_FOUR = 0x0000000000000040
end

"""
    mlirAttributeIsASparseTensorEncodingAttr(attr)

Checks whether the given attribute is a `sparse\\_tensor.encoding` attribute.
"""
function mlirAttributeIsASparseTensorEncodingAttr(attr)
    @ccall (MLIR_C_PATH[]).mlirAttributeIsASparseTensorEncodingAttr(attr::MlirAttribute)::Bool
end

"""
    mlirSparseTensorEncodingAttrGet(ctx, lvlRank, lvlTypes, dimToLvl, posWidth, crdWidth)

Creates a `sparse\\_tensor.encoding` attribute with the given parameters.
"""
function mlirSparseTensorEncodingAttrGet(ctx, lvlRank, lvlTypes, dimToLvl, posWidth, crdWidth)
    @ccall (MLIR_C_PATH[]).mlirSparseTensorEncodingAttrGet(ctx::MlirContext, lvlRank::intptr_t, lvlTypes::Ptr{MlirSparseTensorDimLevelType}, dimToLvl::MlirAffineMap, posWidth::Cint, crdWidth::Cint)::MlirAttribute
end

"""
    mlirSparseTensorEncodingGetLvlRank(attr)

Returns the level-rank of the `sparse\\_tensor.encoding` attribute.
"""
function mlirSparseTensorEncodingGetLvlRank(attr)
    @ccall (MLIR_C_PATH[]).mlirSparseTensorEncodingGetLvlRank(attr::MlirAttribute)::intptr_t
end

"""
    mlirSparseTensorEncodingAttrGetLvlType(attr, lvl)

Returns a specified level-type of the `sparse\\_tensor.encoding` attribute.
"""
function mlirSparseTensorEncodingAttrGetLvlType(attr, lvl)
    @ccall (MLIR_C_PATH[]).mlirSparseTensorEncodingAttrGetLvlType(attr::MlirAttribute, lvl::intptr_t)::MlirSparseTensorDimLevelType
end

"""
    mlirSparseTensorEncodingAttrGetDimToLvl(attr)

Returns the dimension-to-level mapping of the `sparse\\_tensor.encoding` attribute.
"""
function mlirSparseTensorEncodingAttrGetDimToLvl(attr)
    @ccall (MLIR_C_PATH[]).mlirSparseTensorEncodingAttrGetDimToLvl(attr::MlirAttribute)::MlirAffineMap
end

"""
    mlirSparseTensorEncodingAttrGetPosWidth(attr)

Returns the position bitwidth of the `sparse\\_tensor.encoding` attribute.
"""
function mlirSparseTensorEncodingAttrGetPosWidth(attr)
    @ccall (MLIR_C_PATH[]).mlirSparseTensorEncodingAttrGetPosWidth(attr::MlirAttribute)::Cint
end

"""
    mlirSparseTensorEncodingAttrGetCrdWidth(attr)

Returns the coordinate bitwidth of the `sparse\\_tensor.encoding` attribute.
"""
function mlirSparseTensorEncodingAttrGetCrdWidth(attr)
    @ccall (MLIR_C_PATH[]).mlirSparseTensorEncodingAttrGetCrdWidth(attr::MlirAttribute)::Cint
end

function mlirRegisterSparseTensorPasses()
    @ccall (MLIR_C_PATH[]).mlirRegisterSparseTensorPasses()::Cvoid
end

function mlirCreateSparseTensorPostSparsificationRewrite()
    @ccall (MLIR_C_PATH[]).mlirCreateSparseTensorPostSparsificationRewrite()::MlirPass
end

function mlirRegisterSparseTensorPostSparsificationRewrite()
    @ccall (MLIR_C_PATH[]).mlirRegisterSparseTensorPostSparsificationRewrite()::Cvoid
end

function mlirCreateSparseTensorPreSparsificationRewrite()
    @ccall (MLIR_C_PATH[]).mlirCreateSparseTensorPreSparsificationRewrite()::MlirPass
end

function mlirRegisterSparseTensorPreSparsificationRewrite()
    @ccall (MLIR_C_PATH[]).mlirRegisterSparseTensorPreSparsificationRewrite()::Cvoid
end

function mlirCreateSparseTensorSparseBufferRewrite()
    @ccall (MLIR_C_PATH[]).mlirCreateSparseTensorSparseBufferRewrite()::MlirPass
end

function mlirRegisterSparseTensorSparseBufferRewrite()
    @ccall (MLIR_C_PATH[]).mlirRegisterSparseTensorSparseBufferRewrite()::Cvoid
end

function mlirCreateSparseTensorSparseGPUCodegen()
    @ccall (MLIR_C_PATH[]).mlirCreateSparseTensorSparseGPUCodegen()::MlirPass
end

function mlirRegisterSparseTensorSparseGPUCodegen()
    @ccall (MLIR_C_PATH[]).mlirRegisterSparseTensorSparseGPUCodegen()::Cvoid
end

function mlirCreateSparseTensorSparseTensorCodegen()
    @ccall (MLIR_C_PATH[]).mlirCreateSparseTensorSparseTensorCodegen()::MlirPass
end

function mlirRegisterSparseTensorSparseTensorCodegen()
    @ccall (MLIR_C_PATH[]).mlirRegisterSparseTensorSparseTensorCodegen()::Cvoid
end

function mlirCreateSparseTensorSparseTensorConversionPass()
    @ccall (MLIR_C_PATH[]).mlirCreateSparseTensorSparseTensorConversionPass()::MlirPass
end

function mlirRegisterSparseTensorSparseTensorConversionPass()
    @ccall (MLIR_C_PATH[]).mlirRegisterSparseTensorSparseTensorConversionPass()::Cvoid
end

function mlirCreateSparseTensorSparseVectorization()
    @ccall (MLIR_C_PATH[]).mlirCreateSparseTensorSparseVectorization()::MlirPass
end

function mlirRegisterSparseTensorSparseVectorization()
    @ccall (MLIR_C_PATH[]).mlirRegisterSparseTensorSparseVectorization()::Cvoid
end

function mlirCreateSparseTensorSparsificationPass()
    @ccall (MLIR_C_PATH[]).mlirCreateSparseTensorSparsificationPass()::MlirPass
end

function mlirRegisterSparseTensorSparsificationPass()
    @ccall (MLIR_C_PATH[]).mlirRegisterSparseTensorSparsificationPass()::Cvoid
end

function mlirCreateSparseTensorStorageSpecifierToLLVM()
    @ccall (MLIR_C_PATH[]).mlirCreateSparseTensorStorageSpecifierToLLVM()::MlirPass
end

function mlirRegisterSparseTensorStorageSpecifierToLLVM()
    @ccall (MLIR_C_PATH[]).mlirRegisterSparseTensorStorageSpecifierToLLVM()::Cvoid
end

function mlirGetDialectHandle__tensor__()
    @ccall (MLIR_C_PATH[]).mlirGetDialectHandle__tensor__()::MlirDialectHandle
end

function mlirGetDialectHandle__transform__()
    @ccall (MLIR_C_PATH[]).mlirGetDialectHandle__transform__()::MlirDialectHandle
end

function mlirTypeIsATransformAnyOpType(type)
    @ccall (MLIR_C_PATH[]).mlirTypeIsATransformAnyOpType(type::MlirType)::Bool
end

function mlirTransformAnyOpTypeGet(ctx)
    @ccall (MLIR_C_PATH[]).mlirTransformAnyOpTypeGet(ctx::MlirContext)::MlirType
end

function mlirTypeIsATransformOperationType(type)
    @ccall (MLIR_C_PATH[]).mlirTypeIsATransformOperationType(type::MlirType)::Bool
end

function mlirTransformOperationTypeGetTypeID()
    @ccall (MLIR_C_PATH[]).mlirTransformOperationTypeGetTypeID()::MlirTypeID
end

function mlirTransformOperationTypeGet(ctx, operationName)
    @ccall (MLIR_C_PATH[]).mlirTransformOperationTypeGet(ctx::MlirContext, operationName::MlirStringRef)::MlirType
end

function mlirTransformOperationTypeGetOperationName(type)
    @ccall (MLIR_C_PATH[]).mlirTransformOperationTypeGetOperationName(type::MlirType)::MlirStringRef
end

function mlirGetDialectHandle__vector__()
    @ccall (MLIR_C_PATH[]).mlirGetDialectHandle__vector__()::MlirDialectHandle
end

"""
    mlirExecutionEngineCreate(op, optLevel, numPaths, sharedLibPaths, enableObjectDump)

Creates an ExecutionEngine for the provided ModuleOp. The ModuleOp is expected to be "translatable" to LLVM IR (only contains operations in dialects that implement the `LLVMTranslationDialectInterface`). The module ownership stays with the client and can be destroyed as soon as the call returns. `optLevel` is the optimization level to be used for transformation and code generation. LLVM passes at `optLevel` are run before code generation. The number and array of paths corresponding to shared libraries that will be loaded are specified via `numPaths` and `sharedLibPaths` respectively. TODO: figure out other options.
"""
function mlirExecutionEngineCreate(op, optLevel, numPaths, sharedLibPaths, enableObjectDump)
    @ccall (MLIR_C_PATH[]).mlirExecutionEngineCreate(op::MlirModule, optLevel::Cint, numPaths::Cint, sharedLibPaths::Ptr{MlirStringRef}, enableObjectDump::Bool)::MlirExecutionEngine
end

"""
    mlirExecutionEngineDestroy(jit)

Destroy an ExecutionEngine instance.
"""
function mlirExecutionEngineDestroy(jit)
    @ccall (MLIR_C_PATH[]).mlirExecutionEngineDestroy(jit::MlirExecutionEngine)::Cvoid
end

"""
    mlirExecutionEngineIsNull(jit)

Checks whether an execution engine is null.
"""
function mlirExecutionEngineIsNull(jit)
    @ccall (MLIR_C_PATH[]).mlirExecutionEngineIsNull(jit::MlirExecutionEngine)::Bool
end

"""
    mlirExecutionEngineInvokePacked(jit, name, arguments)

Invoke a native function in the execution engine by name with the arguments and result of the invoked function passed as an array of pointers. The function must have been tagged with the `llvm.emit\\_c\\_interface` attribute. Returns a failure if the execution fails for any reason (the function name can't be resolved for instance).
"""
function mlirExecutionEngineInvokePacked(jit, name, arguments)
    @ccall (MLIR_C_PATH[]).mlirExecutionEngineInvokePacked(jit::MlirExecutionEngine, name::MlirStringRef, arguments::Ptr{Ptr{Cvoid}})::MlirLogicalResult
end

"""
    mlirExecutionEngineLookupPacked(jit, name)

Lookup the wrapper of the native function in the execution engine with the given name, returns nullptr if the function can't be looked-up.
"""
function mlirExecutionEngineLookupPacked(jit, name)
    @ccall (MLIR_C_PATH[]).mlirExecutionEngineLookupPacked(jit::MlirExecutionEngine, name::MlirStringRef)::Ptr{Cvoid}
end

"""
    mlirExecutionEngineLookup(jit, name)

Lookup a native function in the execution engine by name, returns nullptr if the name can't be looked-up.
"""
function mlirExecutionEngineLookup(jit, name)
    @ccall (MLIR_C_PATH[]).mlirExecutionEngineLookup(jit::MlirExecutionEngine, name::MlirStringRef)::Ptr{Cvoid}
end

"""
    mlirExecutionEngineRegisterSymbol(jit, name, sym)

Register a symbol with the jit: this symbol will be accessible to the jitted code.
"""
function mlirExecutionEngineRegisterSymbol(jit, name, sym)
    @ccall (MLIR_C_PATH[]).mlirExecutionEngineRegisterSymbol(jit::MlirExecutionEngine, name::MlirStringRef, sym::Ptr{Cvoid})::Cvoid
end

"""
    mlirExecutionEngineDumpToObjectFile(jit, fileName)

Dump as an object in `fileName`.
"""
function mlirExecutionEngineDumpToObjectFile(jit, fileName)
    @ccall (MLIR_C_PATH[]).mlirExecutionEngineDumpToObjectFile(jit::MlirExecutionEngine, fileName::MlirStringRef)::Cvoid
end

"""
    mlirIntegerSetGetContext(set)

Gets the context in which the given integer set lives.
"""
function mlirIntegerSetGetContext(set)
    @ccall (MLIR_C_PATH[]).mlirIntegerSetGetContext(set::MlirIntegerSet)::MlirContext
end

"""
    mlirIntegerSetIsNull(set)

Checks whether an integer set is a null object.
"""
function mlirIntegerSetIsNull(set)
    @ccall (MLIR_C_PATH[]).mlirIntegerSetIsNull(set::MlirIntegerSet)::Bool
end

"""
    mlirIntegerSetEqual(s1, s2)

Checks if two integer set objects are equal. This is a "shallow" comparison of two objects. Only the sets with some small number of constraints are uniqued and compare equal here. Set objects that represent the same integer set with different constraints may be considered non-equal by this check. Set difference followed by an (expensive) emptiness check should be used to check equivalence of the underlying integer sets.
"""
function mlirIntegerSetEqual(s1, s2)
    @ccall (MLIR_C_PATH[]).mlirIntegerSetEqual(s1::MlirIntegerSet, s2::MlirIntegerSet)::Bool
end

"""
    mlirIntegerSetPrint(set, callback, userData)

Prints an integer set by sending chunks of the string representation and forwarding `userData to `callback`. Note that the callback may be called several times with consecutive chunks of the string.
"""
function mlirIntegerSetPrint(set, callback, userData)
    @ccall (MLIR_C_PATH[]).mlirIntegerSetPrint(set::MlirIntegerSet, callback::MlirStringCallback, userData::Ptr{Cvoid})::Cvoid
end

"""
    mlirIntegerSetDump(set)

Prints an integer set to the standard error stream.
"""
function mlirIntegerSetDump(set)
    @ccall (MLIR_C_PATH[]).mlirIntegerSetDump(set::MlirIntegerSet)::Cvoid
end

"""
    mlirIntegerSetEmptyGet(context, numDims, numSymbols)

Gets or creates a new canonically empty integer set with the give number of dimensions and symbols in the given context.
"""
function mlirIntegerSetEmptyGet(context, numDims, numSymbols)
    @ccall (MLIR_C_PATH[]).mlirIntegerSetEmptyGet(context::MlirContext, numDims::intptr_t, numSymbols::intptr_t)::MlirIntegerSet
end

"""
    mlirIntegerSetGet(context, numDims, numSymbols, numConstraints, constraints, eqFlags)

Gets or creates a new integer set in the given context. The set is defined by a list of affine constraints, with the given number of input dimensions and symbols, which are treated as either equalities (eqFlags is 1) or inequalities (eqFlags is 0). Both `constraints` and `eqFlags` are expected to point to at least `numConstraint` consecutive values.
"""
function mlirIntegerSetGet(context, numDims, numSymbols, numConstraints, constraints, eqFlags)
    @ccall (MLIR_C_PATH[]).mlirIntegerSetGet(context::MlirContext, numDims::intptr_t, numSymbols::intptr_t, numConstraints::intptr_t, constraints::Ptr{MlirAffineExpr}, eqFlags::Ptr{Bool})::MlirIntegerSet
end

"""
    mlirIntegerSetReplaceGet(set, dimReplacements, symbolReplacements, numResultDims, numResultSymbols)

Gets or creates a new integer set in which the values and dimensions of the given set are replaced with the given affine expressions. `dimReplacements` and `symbolReplacements` are expected to point to at least as many consecutive expressions as the given set has dimensions and symbols, respectively. The new set will have `numResultDims` and `numResultSymbols` dimensions and symbols, respectively.
"""
function mlirIntegerSetReplaceGet(set, dimReplacements, symbolReplacements, numResultDims, numResultSymbols)
    @ccall (MLIR_C_PATH[]).mlirIntegerSetReplaceGet(set::MlirIntegerSet, dimReplacements::Ptr{MlirAffineExpr}, symbolReplacements::Ptr{MlirAffineExpr}, numResultDims::intptr_t, numResultSymbols::intptr_t)::MlirIntegerSet
end

"""
    mlirIntegerSetIsCanonicalEmpty(set)

Checks whether the given set is a canonical empty set, e.g., the set returned by [`mlirIntegerSetEmptyGet`](@ref).
"""
function mlirIntegerSetIsCanonicalEmpty(set)
    @ccall (MLIR_C_PATH[]).mlirIntegerSetIsCanonicalEmpty(set::MlirIntegerSet)::Bool
end

"""
    mlirIntegerSetGetNumDims(set)

Returns the number of dimensions in the given set.
"""
function mlirIntegerSetGetNumDims(set)
    @ccall (MLIR_C_PATH[]).mlirIntegerSetGetNumDims(set::MlirIntegerSet)::intptr_t
end

"""
    mlirIntegerSetGetNumSymbols(set)

Returns the number of symbols in the given set.
"""
function mlirIntegerSetGetNumSymbols(set)
    @ccall (MLIR_C_PATH[]).mlirIntegerSetGetNumSymbols(set::MlirIntegerSet)::intptr_t
end

"""
    mlirIntegerSetGetNumInputs(set)

Returns the number of inputs (dimensions + symbols) in the given set.
"""
function mlirIntegerSetGetNumInputs(set)
    @ccall (MLIR_C_PATH[]).mlirIntegerSetGetNumInputs(set::MlirIntegerSet)::intptr_t
end

"""
    mlirIntegerSetGetNumConstraints(set)

Returns the number of constraints (equalities + inequalities) in the given set.
"""
function mlirIntegerSetGetNumConstraints(set)
    @ccall (MLIR_C_PATH[]).mlirIntegerSetGetNumConstraints(set::MlirIntegerSet)::intptr_t
end

"""
    mlirIntegerSetGetNumEqualities(set)

Returns the number of equalities in the given set.
"""
function mlirIntegerSetGetNumEqualities(set)
    @ccall (MLIR_C_PATH[]).mlirIntegerSetGetNumEqualities(set::MlirIntegerSet)::intptr_t
end

"""
    mlirIntegerSetGetNumInequalities(set)

Returns the number of inequalities in the given set.
"""
function mlirIntegerSetGetNumInequalities(set)
    @ccall (MLIR_C_PATH[]).mlirIntegerSetGetNumInequalities(set::MlirIntegerSet)::intptr_t
end

"""
    mlirIntegerSetGetConstraint(set, pos)

Returns `pos`-th constraint of the set.
"""
function mlirIntegerSetGetConstraint(set, pos)
    @ccall (MLIR_C_PATH[]).mlirIntegerSetGetConstraint(set::MlirIntegerSet, pos::intptr_t)::MlirAffineExpr
end

"""
    mlirIntegerSetIsConstraintEq(set, pos)

Returns `true` of the `pos`-th constraint of the set is an equality constraint, `false` otherwise.
"""
function mlirIntegerSetIsConstraintEq(set, pos)
    @ccall (MLIR_C_PATH[]).mlirIntegerSetIsConstraintEq(set::MlirIntegerSet, pos::intptr_t)::Bool
end

"""
    mlirOperationImplementsInterface(operation, interfaceTypeID)

Returns `true` if the given operation implements an interface identified by its TypeID.
"""
function mlirOperationImplementsInterface(operation, interfaceTypeID)
    @ccall (MLIR_C_PATH[]).mlirOperationImplementsInterface(operation::MlirOperation, interfaceTypeID::MlirTypeID)::Bool
end

"""
    mlirOperationImplementsInterfaceStatic(operationName, context, interfaceTypeID)

Returns `true` if the operation identified by its canonical string name implements the interface identified by its TypeID in the given context. Note that interfaces may be attached to operations in some contexts and not others.
"""
function mlirOperationImplementsInterfaceStatic(operationName, context, interfaceTypeID)
    @ccall (MLIR_C_PATH[]).mlirOperationImplementsInterfaceStatic(operationName::MlirStringRef, context::MlirContext, interfaceTypeID::MlirTypeID)::Bool
end

"""
    mlirInferTypeOpInterfaceTypeID()

Returns the interface TypeID of the InferTypeOpInterface.
"""
function mlirInferTypeOpInterfaceTypeID()
    @ccall (MLIR_C_PATH[]).mlirInferTypeOpInterfaceTypeID()::MlirTypeID
end

# typedef void ( * MlirTypesCallback ) ( intptr_t , MlirType * , void * )
"""
These callbacks are used to return multiple types from functions while transferring ownership to the caller. The first argument is the number of consecutive elements pointed to by the second argument. The third argument is an opaque pointer forwarded to the callback by the caller.
"""
const MlirTypesCallback = Ptr{Cvoid}

"""
    mlirInferTypeOpInterfaceInferReturnTypes(opName, context, location, nOperands, operands, attributes, properties, nRegions, regions, callback, userData)

Infers the return types of the operation identified by its canonical given the arguments that will be supplied to its generic builder. Calls `callback` with the types of inferred arguments, potentially several times, on success. Returns failure otherwise.
"""
function mlirInferTypeOpInterfaceInferReturnTypes(opName, context, location, nOperands, operands, attributes, properties, nRegions, regions, callback, userData)
    @ccall (MLIR_C_PATH[]).mlirInferTypeOpInterfaceInferReturnTypes(opName::MlirStringRef, context::MlirContext, location::MlirLocation, nOperands::intptr_t, operands::Ptr{MlirValue}, attributes::MlirAttribute, properties::Ptr{Cvoid}, nRegions::intptr_t, regions::Ptr{MlirRegion}, callback::MlirTypesCallback, userData::Ptr{Cvoid})::MlirLogicalResult
end

"""
    mlirInferShapedTypeOpInterfaceTypeID()

Returns the interface TypeID of the InferShapedTypeOpInterface.
"""
function mlirInferShapedTypeOpInterfaceTypeID()
    @ccall (MLIR_C_PATH[]).mlirInferShapedTypeOpInterfaceTypeID()::MlirTypeID
end

# typedef void ( * MlirShapedTypeComponentsCallback ) ( bool , intptr_t , const int64_t * , MlirType , MlirAttribute , void * )
"""
These callbacks are used to return multiple shaped type components from functions while transferring ownership to the caller. The first argument is the has rank boolean followed by the the rank and a pointer to the shape (if applicable). The next argument is the element type, then the attribute. The last argument is an opaque pointer forwarded to the callback by the caller. This callback will be called potentially multiple times for each shaped type components.
"""
const MlirShapedTypeComponentsCallback = Ptr{Cvoid}

"""
    mlirInferShapedTypeOpInterfaceInferReturnTypes(opName, context, location, nOperands, operands, attributes, properties, nRegions, regions, callback, userData)

Infers the return shaped type components of the operation. Calls `callback` with the types of inferred arguments on success. Returns failure otherwise.
"""
function mlirInferShapedTypeOpInterfaceInferReturnTypes(opName, context, location, nOperands, operands, attributes, properties, nRegions, regions, callback, userData)
    @ccall (MLIR_C_PATH[]).mlirInferShapedTypeOpInterfaceInferReturnTypes(opName::MlirStringRef, context::MlirContext, location::MlirLocation, nOperands::intptr_t, operands::Ptr{MlirValue}, attributes::MlirAttribute, properties::Ptr{Cvoid}, nRegions::intptr_t, regions::Ptr{MlirRegion}, callback::MlirShapedTypeComponentsCallback, userData::Ptr{Cvoid})::MlirLogicalResult
end

"""
    mlirRegisterAllDialects(registry)

Appends all upstream dialects and extensions to the dialect registry.
"""
function mlirRegisterAllDialects(registry)
    @ccall (MLIR_C_PATH[]).mlirRegisterAllDialects(registry::MlirDialectRegistry)::Cvoid
end

"""
    mlirRegisterAllLLVMTranslations(context)

Register all translations to LLVM IR for dialects that can support it.
"""
function mlirRegisterAllLLVMTranslations(context)
    @ccall (MLIR_C_PATH[]).mlirRegisterAllLLVMTranslations(context::MlirContext)::Cvoid
end

"""
    mlirRegisterAllPasses()

Register all compiler passes of MLIR.
"""
function mlirRegisterAllPasses()
    @ccall (MLIR_C_PATH[]).mlirRegisterAllPasses()::Cvoid
end

function mlirRegisterTransformsPasses()
    @ccall (MLIR_C_PATH[]).mlirRegisterTransformsPasses()::Cvoid
end

function mlirCreateTransformsCSE()
    @ccall (MLIR_C_PATH[]).mlirCreateTransformsCSE()::MlirPass
end

function mlirRegisterTransformsCSE()
    @ccall (MLIR_C_PATH[]).mlirRegisterTransformsCSE()::Cvoid
end

function mlirCreateTransformsCanonicalizer()
    @ccall (MLIR_C_PATH[]).mlirCreateTransformsCanonicalizer()::MlirPass
end

function mlirRegisterTransformsCanonicalizer()
    @ccall (MLIR_C_PATH[]).mlirRegisterTransformsCanonicalizer()::Cvoid
end

function mlirCreateTransformsControlFlowSink()
    @ccall (MLIR_C_PATH[]).mlirCreateTransformsControlFlowSink()::MlirPass
end

function mlirRegisterTransformsControlFlowSink()
    @ccall (MLIR_C_PATH[]).mlirRegisterTransformsControlFlowSink()::Cvoid
end

function mlirCreateTransformsGenerateRuntimeVerification()
    @ccall (MLIR_C_PATH[]).mlirCreateTransformsGenerateRuntimeVerification()::MlirPass
end

function mlirRegisterTransformsGenerateRuntimeVerification()
    @ccall (MLIR_C_PATH[]).mlirRegisterTransformsGenerateRuntimeVerification()::Cvoid
end

function mlirCreateTransformsInliner()
    @ccall (MLIR_C_PATH[]).mlirCreateTransformsInliner()::MlirPass
end

function mlirRegisterTransformsInliner()
    @ccall (MLIR_C_PATH[]).mlirRegisterTransformsInliner()::Cvoid
end

function mlirCreateTransformsLocationSnapshot()
    @ccall (MLIR_C_PATH[]).mlirCreateTransformsLocationSnapshot()::MlirPass
end

function mlirRegisterTransformsLocationSnapshot()
    @ccall (MLIR_C_PATH[]).mlirRegisterTransformsLocationSnapshot()::Cvoid
end

function mlirCreateTransformsLoopInvariantCodeMotion()
    @ccall (MLIR_C_PATH[]).mlirCreateTransformsLoopInvariantCodeMotion()::MlirPass
end

function mlirRegisterTransformsLoopInvariantCodeMotion()
    @ccall (MLIR_C_PATH[]).mlirRegisterTransformsLoopInvariantCodeMotion()::Cvoid
end

function mlirCreateTransformsMem2Reg()
    @ccall (MLIR_C_PATH[]).mlirCreateTransformsMem2Reg()::MlirPass
end

function mlirRegisterTransformsMem2Reg()
    @ccall (MLIR_C_PATH[]).mlirRegisterTransformsMem2Reg()::Cvoid
end

function mlirCreateTransformsPrintIRPass()
    @ccall (MLIR_C_PATH[]).mlirCreateTransformsPrintIRPass()::MlirPass
end

function mlirRegisterTransformsPrintIRPass()
    @ccall (MLIR_C_PATH[]).mlirRegisterTransformsPrintIRPass()::Cvoid
end

function mlirCreateTransformsPrintOpStats()
    @ccall (MLIR_C_PATH[]).mlirCreateTransformsPrintOpStats()::MlirPass
end

function mlirRegisterTransformsPrintOpStats()
    @ccall (MLIR_C_PATH[]).mlirRegisterTransformsPrintOpStats()::Cvoid
end

function mlirCreateTransformsSCCP()
    @ccall (MLIR_C_PATH[]).mlirCreateTransformsSCCP()::MlirPass
end

function mlirRegisterTransformsSCCP()
    @ccall (MLIR_C_PATH[]).mlirRegisterTransformsSCCP()::Cvoid
end

function mlirCreateTransformsSROA()
    @ccall (MLIR_C_PATH[]).mlirCreateTransformsSROA()::MlirPass
end

function mlirRegisterTransformsSROA()
    @ccall (MLIR_C_PATH[]).mlirRegisterTransformsSROA()::Cvoid
end

function mlirCreateTransformsStripDebugInfo()
    @ccall (MLIR_C_PATH[]).mlirCreateTransformsStripDebugInfo()::MlirPass
end

function mlirRegisterTransformsStripDebugInfo()
    @ccall (MLIR_C_PATH[]).mlirRegisterTransformsStripDebugInfo()::Cvoid
end

function mlirCreateTransformsSymbolDCE()
    @ccall (MLIR_C_PATH[]).mlirCreateTransformsSymbolDCE()::MlirPass
end

function mlirRegisterTransformsSymbolDCE()
    @ccall (MLIR_C_PATH[]).mlirRegisterTransformsSymbolDCE()::Cvoid
end

function mlirCreateTransformsSymbolPrivatize()
    @ccall (MLIR_C_PATH[]).mlirCreateTransformsSymbolPrivatize()::MlirPass
end

function mlirRegisterTransformsSymbolPrivatize()
    @ccall (MLIR_C_PATH[]).mlirRegisterTransformsSymbolPrivatize()::Cvoid
end

function mlirCreateTransformsTopologicalSort()
    @ccall (MLIR_C_PATH[]).mlirCreateTransformsTopologicalSort()::MlirPass
end

function mlirRegisterTransformsTopologicalSort()
    @ccall (MLIR_C_PATH[]).mlirRegisterTransformsTopologicalSort()::Cvoid
end

function mlirCreateTransformsViewOpGraph()
    @ccall (MLIR_C_PATH[]).mlirCreateTransformsViewOpGraph()::MlirPass
end

function mlirRegisterTransformsViewOpGraph()
    @ccall (MLIR_C_PATH[]).mlirRegisterTransformsViewOpGraph()::Cvoid
end

