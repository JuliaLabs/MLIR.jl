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

struct MlirStringRef
    data::Cstring
    length::Csize_t
end

function mlirStringRefCreate(str, length)
    ccall((:mlirStringRefCreate, mlir_c), MlirStringRef, (Cstring, Csize_t), str, length)
end

function mlirStringRefCreateFromCString(str)
    ccall((:mlirStringRefCreateFromCString, mlir_c), MlirStringRef, (Cstring,), str)
end

function mlirStringRefEqual(string, other)
    ccall((:mlirStringRefEqual, mlir_c), Bool, (MlirStringRef, MlirStringRef), string, other)
end

# typedef void ( * MlirStringCallback ) ( MlirStringRef , void * )
const MlirStringCallback = Ptr{Cvoid}

struct MlirLogicalResult
    value::Int8
end

function mlirLogicalResultIsSuccess(res)
    ccall((:mlirLogicalResultIsSuccess, mlir_c), Bool, (MlirLogicalResult,), res)
end

function mlirLogicalResultIsFailure(res)
    ccall((:mlirLogicalResultIsFailure, mlir_c), Bool, (MlirLogicalResult,), res)
end

# no prototype is found for this function at Support.h:130:33, please use with caution
function mlirLogicalResultSuccess()
    ccall((:mlirLogicalResultSuccess, mlir_c), MlirLogicalResult, ())
end

# no prototype is found for this function at Support.h:136:33, please use with caution
function mlirLogicalResultFailure()
    ccall((:mlirLogicalResultFailure, mlir_c), MlirLogicalResult, ())
end

function mlirTypeIDCreate(ptr)
    ccall((:mlirTypeIDCreate, mlir_c), MlirTypeID, (Ptr{Cvoid},), ptr)
end

function mlirTypeIDIsNull(typeID)
    ccall((:mlirTypeIDIsNull, mlir_c), Bool, (MlirTypeID,), typeID)
end

function mlirTypeIDEqual(typeID1, typeID2)
    ccall((:mlirTypeIDEqual, mlir_c), Bool, (MlirTypeID, MlirTypeID), typeID1, typeID2)
end

function mlirTypeIDHashValue(typeID)
    ccall((:mlirTypeIDHashValue, mlir_c), Csize_t, (MlirTypeID,), typeID)
end

# no prototype is found for this function at Support.h:163:40, please use with caution
function mlirTypeIDAllocatorCreate()
    ccall((:mlirTypeIDAllocatorCreate, mlir_c), MlirTypeIDAllocator, ())
end

function mlirTypeIDAllocatorDestroy(allocator)
    ccall((:mlirTypeIDAllocatorDestroy, mlir_c), Cvoid, (MlirTypeIDAllocator,), allocator)
end

function mlirTypeIDAllocatorAllocateTypeID(allocator)
    ccall((:mlirTypeIDAllocatorAllocateTypeID, mlir_c), MlirTypeID, (MlirTypeIDAllocator,), allocator)
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

struct MlirNamedAttribute
    name::MlirIdentifier
    attribute::MlirAttribute
end

# no prototype is found for this function at IR.h:85:32, please use with caution
function mlirContextCreate()
    ccall((:mlirContextCreate, mlir_c), MlirContext, ())
end

function mlirContextEqual(ctx1, ctx2)
    ccall((:mlirContextEqual, mlir_c), Bool, (MlirContext, MlirContext), ctx1, ctx2)
end

function mlirContextIsNull(context)
    ccall((:mlirContextIsNull, mlir_c), Bool, (MlirContext,), context)
end

function mlirContextDestroy(context)
    ccall((:mlirContextDestroy, mlir_c), Cvoid, (MlirContext,), context)
end

function mlirContextSetAllowUnregisteredDialects(context, allow)
    ccall((:mlirContextSetAllowUnregisteredDialects, mlir_c), Cvoid, (MlirContext, Bool), context, allow)
end

function mlirContextGetAllowUnregisteredDialects(context)
    ccall((:mlirContextGetAllowUnregisteredDialects, mlir_c), Bool, (MlirContext,), context)
end

function mlirContextGetNumRegisteredDialects(context)
    ccall((:mlirContextGetNumRegisteredDialects, mlir_c), intptr_t, (MlirContext,), context)
end

function mlirContextAppendDialectRegistry(ctx, registry)
    ccall((:mlirContextAppendDialectRegistry, mlir_c), Cvoid, (MlirContext, MlirDialectRegistry), ctx, registry)
end

function mlirContextGetNumLoadedDialects(context)
    ccall((:mlirContextGetNumLoadedDialects, mlir_c), intptr_t, (MlirContext,), context)
end

function mlirContextGetOrLoadDialect(context, name)
    ccall((:mlirContextGetOrLoadDialect, mlir_c), MlirDialect, (MlirContext, MlirStringRef), context, name)
end

function mlirContextEnableMultithreading(context, enable)
    ccall((:mlirContextEnableMultithreading, mlir_c), Cvoid, (MlirContext, Bool), context, enable)
end

function mlirContextLoadAllAvailableDialects(context)
    ccall((:mlirContextLoadAllAvailableDialects, mlir_c), Cvoid, (MlirContext,), context)
end

function mlirContextIsRegisteredOperation(context, name)
    ccall((:mlirContextIsRegisteredOperation, mlir_c), Bool, (MlirContext, MlirStringRef), context, name)
end

function mlirDialectGetContext(dialect)
    ccall((:mlirDialectGetContext, mlir_c), MlirContext, (MlirDialect,), dialect)
end

function mlirDialectIsNull(dialect)
    ccall((:mlirDialectIsNull, mlir_c), Bool, (MlirDialect,), dialect)
end

function mlirDialectEqual(dialect1, dialect2)
    ccall((:mlirDialectEqual, mlir_c), Bool, (MlirDialect, MlirDialect), dialect1, dialect2)
end

function mlirDialectGetNamespace(dialect)
    ccall((:mlirDialectGetNamespace, mlir_c), MlirStringRef, (MlirDialect,), dialect)
end

function mlirDialectHandleGetNamespace(arg1)
    ccall((:mlirDialectHandleGetNamespace, mlir_c), MlirStringRef, (MlirDialectHandle,), arg1)
end

function mlirDialectHandleInsertDialect(arg1, arg2)
    ccall((:mlirDialectHandleInsertDialect, mlir_c), Cvoid, (MlirDialectHandle, MlirDialectRegistry), arg1, arg2)
end

function mlirDialectHandleRegisterDialect(arg1, arg2)
    ccall((:mlirDialectHandleRegisterDialect, mlir_c), Cvoid, (MlirDialectHandle, MlirContext), arg1, arg2)
end

function mlirDialectHandleLoadDialect(arg1, arg2)
    ccall((:mlirDialectHandleLoadDialect, mlir_c), MlirDialect, (MlirDialectHandle, MlirContext), arg1, arg2)
end

# no prototype is found for this function at IR.h:211:40, please use with caution
function mlirDialectRegistryCreate()
    ccall((:mlirDialectRegistryCreate, mlir_c), MlirDialectRegistry, ())
end

function mlirDialectRegistryIsNull(registry)
    ccall((:mlirDialectRegistryIsNull, mlir_c), Bool, (MlirDialectRegistry,), registry)
end

function mlirDialectRegistryDestroy(registry)
    ccall((:mlirDialectRegistryDestroy, mlir_c), Cvoid, (MlirDialectRegistry,), registry)
end

function mlirLocationFileLineColGet(context, filename, line, col)
    ccall((:mlirLocationFileLineColGet, mlir_c), MlirLocation, (MlirContext, MlirStringRef, Cuint, Cuint), context, filename, line, col)
end

function mlirLocationCallSiteGet(callee, caller)
    ccall((:mlirLocationCallSiteGet, mlir_c), MlirLocation, (MlirLocation, MlirLocation), callee, caller)
end

function mlirLocationFusedGet(ctx, nLocations, locations, metadata)
    ccall((:mlirLocationFusedGet, mlir_c), MlirLocation, (MlirContext, intptr_t, Ptr{MlirLocation}, MlirAttribute), ctx, nLocations, locations, metadata)
end

function mlirLocationNameGet(context, name, childLoc)
    ccall((:mlirLocationNameGet, mlir_c), MlirLocation, (MlirContext, MlirStringRef, MlirLocation), context, name, childLoc)
end

function mlirLocationUnknownGet(context)
    ccall((:mlirLocationUnknownGet, mlir_c), MlirLocation, (MlirContext,), context)
end

function mlirLocationGetContext(location)
    ccall((:mlirLocationGetContext, mlir_c), MlirContext, (MlirLocation,), location)
end

function mlirLocationIsNull(location)
    ccall((:mlirLocationIsNull, mlir_c), Bool, (MlirLocation,), location)
end

function mlirLocationEqual(l1, l2)
    ccall((:mlirLocationEqual, mlir_c), Bool, (MlirLocation, MlirLocation), l1, l2)
end

function mlirLocationPrint(location, callback, userData)
    ccall((:mlirLocationPrint, mlir_c), Cvoid, (MlirLocation, MlirStringCallback, Ptr{Cvoid}), location, callback, userData)
end

function mlirModuleCreateEmpty(location)
    ccall((:mlirModuleCreateEmpty, mlir_c), MlirModule, (MlirLocation,), location)
end

function mlirModuleCreateParse(context, _module)
    ccall((:mlirModuleCreateParse, mlir_c), MlirModule, (MlirContext, MlirStringRef), context, _module)
end

function mlirModuleGetContext(_module)
    ccall((:mlirModuleGetContext, mlir_c), MlirContext, (MlirModule,), _module)
end

function mlirModuleGetBody(_module)
    ccall((:mlirModuleGetBody, mlir_c), MlirBlock, (MlirModule,), _module)
end

function mlirModuleIsNull(_module)
    ccall((:mlirModuleIsNull, mlir_c), Bool, (MlirModule,), _module)
end

function mlirModuleDestroy(_module)
    ccall((:mlirModuleDestroy, mlir_c), Cvoid, (MlirModule,), _module)
end

function mlirModuleGetOperation(_module)
    ccall((:mlirModuleGetOperation, mlir_c), MlirOperation, (MlirModule,), _module)
end

function mlirModuleFromOperation(op)
    ccall((:mlirModuleFromOperation, mlir_c), MlirModule, (MlirOperation,), op)
end

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

function mlirOperationStateGet(name, loc)
    ccall((:mlirOperationStateGet, mlir_c), MlirOperationState, (MlirStringRef, MlirLocation), name, loc)
end

function mlirOperationStateAddResults(state, n, results)
    ccall((:mlirOperationStateAddResults, mlir_c), Cvoid, (Ptr{MlirOperationState}, intptr_t, Ptr{MlirType}), state, n, results)
end

function mlirOperationStateAddOperands(state, n, operands)
    ccall((:mlirOperationStateAddOperands, mlir_c), Cvoid, (Ptr{MlirOperationState}, intptr_t, Ptr{MlirValue}), state, n, operands)
end

function mlirOperationStateAddOwnedRegions(state, n, regions)
    ccall((:mlirOperationStateAddOwnedRegions, mlir_c), Cvoid, (Ptr{MlirOperationState}, intptr_t, Ptr{MlirRegion}), state, n, regions)
end

function mlirOperationStateAddSuccessors(state, n, successors)
    ccall((:mlirOperationStateAddSuccessors, mlir_c), Cvoid, (Ptr{MlirOperationState}, intptr_t, Ptr{MlirBlock}), state, n, successors)
end

function mlirOperationStateAddAttributes(state, n, attributes)
    ccall((:mlirOperationStateAddAttributes, mlir_c), Cvoid, (Ptr{MlirOperationState}, intptr_t, Ptr{MlirNamedAttribute}), state, n, attributes)
end

function mlirOperationStateEnableResultTypeInference(state)
    ccall((:mlirOperationStateEnableResultTypeInference, mlir_c), Cvoid, (Ptr{MlirOperationState},), state)
end

# no prototype is found for this function at IR.h:366:40, please use with caution
function mlirOpPrintingFlagsCreate()
    ccall((:mlirOpPrintingFlagsCreate, mlir_c), MlirOpPrintingFlags, ())
end

function mlirOpPrintingFlagsDestroy(flags)
    ccall((:mlirOpPrintingFlagsDestroy, mlir_c), Cvoid, (MlirOpPrintingFlags,), flags)
end

function mlirOpPrintingFlagsElideLargeElementsAttrs(flags, largeElementLimit)
    ccall((:mlirOpPrintingFlagsElideLargeElementsAttrs, mlir_c), Cvoid, (MlirOpPrintingFlags, intptr_t), flags, largeElementLimit)
end

function mlirOpPrintingFlagsEnableDebugInfo(flags, prettyForm)
    ccall((:mlirOpPrintingFlagsEnableDebugInfo, mlir_c), Cvoid, (MlirOpPrintingFlags, Bool), flags, prettyForm)
end

function mlirOpPrintingFlagsPrintGenericOpForm(flags)
    ccall((:mlirOpPrintingFlagsPrintGenericOpForm, mlir_c), Cvoid, (MlirOpPrintingFlags,), flags)
end

function mlirOpPrintingFlagsUseLocalScope(flags)
    ccall((:mlirOpPrintingFlagsUseLocalScope, mlir_c), Cvoid, (MlirOpPrintingFlags,), flags)
end

function mlirOperationCreate(state)
    ccall((:mlirOperationCreate, mlir_c), MlirOperation, (Ptr{MlirOperationState},), state)
end

function mlirOperationClone(op)
    ccall((:mlirOperationClone, mlir_c), MlirOperation, (MlirOperation,), op)
end

function mlirOperationDestroy(op)
    ccall((:mlirOperationDestroy, mlir_c), Cvoid, (MlirOperation,), op)
end

function mlirOperationRemoveFromParent(op)
    ccall((:mlirOperationRemoveFromParent, mlir_c), Cvoid, (MlirOperation,), op)
end

function mlirOperationIsNull(op)
    ccall((:mlirOperationIsNull, mlir_c), Bool, (MlirOperation,), op)
end

function mlirOperationEqual(op, other)
    ccall((:mlirOperationEqual, mlir_c), Bool, (MlirOperation, MlirOperation), op, other)
end

function mlirOperationGetContext(op)
    ccall((:mlirOperationGetContext, mlir_c), MlirContext, (MlirOperation,), op)
end

function mlirOperationGetLocation(op)
    ccall((:mlirOperationGetLocation, mlir_c), MlirLocation, (MlirOperation,), op)
end

function mlirOperationGetTypeID(op)
    ccall((:mlirOperationGetTypeID, mlir_c), MlirTypeID, (MlirOperation,), op)
end

function mlirOperationGetName(op)
    ccall((:mlirOperationGetName, mlir_c), MlirIdentifier, (MlirOperation,), op)
end

function mlirOperationGetBlock(op)
    ccall((:mlirOperationGetBlock, mlir_c), MlirBlock, (MlirOperation,), op)
end

function mlirOperationGetParentOperation(op)
    ccall((:mlirOperationGetParentOperation, mlir_c), MlirOperation, (MlirOperation,), op)
end

function mlirOperationGetNumRegions(op)
    ccall((:mlirOperationGetNumRegions, mlir_c), intptr_t, (MlirOperation,), op)
end

function mlirOperationGetRegion(op, pos)
    ccall((:mlirOperationGetRegion, mlir_c), MlirRegion, (MlirOperation, intptr_t), op, pos)
end

function mlirOperationGetNextInBlock(op)
    ccall((:mlirOperationGetNextInBlock, mlir_c), MlirOperation, (MlirOperation,), op)
end

function mlirOperationGetNumOperands(op)
    ccall((:mlirOperationGetNumOperands, mlir_c), intptr_t, (MlirOperation,), op)
end

function mlirOperationGetOperand(op, pos)
    ccall((:mlirOperationGetOperand, mlir_c), MlirValue, (MlirOperation, intptr_t), op, pos)
end

function mlirOperationSetOperand(op, pos, newValue)
    ccall((:mlirOperationSetOperand, mlir_c), Cvoid, (MlirOperation, intptr_t, MlirValue), op, pos, newValue)
end

function mlirOperationGetNumResults(op)
    ccall((:mlirOperationGetNumResults, mlir_c), intptr_t, (MlirOperation,), op)
end

function mlirOperationGetResult(op, pos)
    ccall((:mlirOperationGetResult, mlir_c), MlirValue, (MlirOperation, intptr_t), op, pos)
end

function mlirOperationGetNumSuccessors(op)
    ccall((:mlirOperationGetNumSuccessors, mlir_c), intptr_t, (MlirOperation,), op)
end

function mlirOperationGetSuccessor(op, pos)
    ccall((:mlirOperationGetSuccessor, mlir_c), MlirBlock, (MlirOperation, intptr_t), op, pos)
end

function mlirOperationGetNumAttributes(op)
    ccall((:mlirOperationGetNumAttributes, mlir_c), intptr_t, (MlirOperation,), op)
end

function mlirOperationGetAttribute(op, pos)
    ccall((:mlirOperationGetAttribute, mlir_c), MlirNamedAttribute, (MlirOperation, intptr_t), op, pos)
end

function mlirOperationGetAttributeByName(op, name)
    ccall((:mlirOperationGetAttributeByName, mlir_c), MlirAttribute, (MlirOperation, MlirStringRef), op, name)
end

function mlirOperationSetAttributeByName(op, name, attr)
    ccall((:mlirOperationSetAttributeByName, mlir_c), Cvoid, (MlirOperation, MlirStringRef, MlirAttribute), op, name, attr)
end

function mlirOperationRemoveAttributeByName(op, name)
    ccall((:mlirOperationRemoveAttributeByName, mlir_c), Bool, (MlirOperation, MlirStringRef), op, name)
end

function mlirOperationPrint(op, callback, userData)
    ccall((:mlirOperationPrint, mlir_c), Cvoid, (MlirOperation, MlirStringCallback, Ptr{Cvoid}), op, callback, userData)
end

function mlirOperationPrintWithFlags(op, flags, callback, userData)
    ccall((:mlirOperationPrintWithFlags, mlir_c), Cvoid, (MlirOperation, MlirOpPrintingFlags, MlirStringCallback, Ptr{Cvoid}), op, flags, callback, userData)
end

function mlirOperationDump(op)
    ccall((:mlirOperationDump, mlir_c), Cvoid, (MlirOperation,), op)
end

function mlirOperationVerify(op)
    ccall((:mlirOperationVerify, mlir_c), Bool, (MlirOperation,), op)
end

function mlirOperationMoveAfter(op, other)
    ccall((:mlirOperationMoveAfter, mlir_c), Cvoid, (MlirOperation, MlirOperation), op, other)
end

function mlirOperationMoveBefore(op, other)
    ccall((:mlirOperationMoveBefore, mlir_c), Cvoid, (MlirOperation, MlirOperation), op, other)
end

# no prototype is found for this function at IR.h:548:31, please use with caution
function mlirRegionCreate()
    ccall((:mlirRegionCreate, mlir_c), MlirRegion, ())
end

function mlirRegionDestroy(region)
    ccall((:mlirRegionDestroy, mlir_c), Cvoid, (MlirRegion,), region)
end

function mlirRegionIsNull(region)
    ccall((:mlirRegionIsNull, mlir_c), Bool, (MlirRegion,), region)
end

function mlirRegionEqual(region, other)
    ccall((:mlirRegionEqual, mlir_c), Bool, (MlirRegion, MlirRegion), region, other)
end

function mlirRegionGetFirstBlock(region)
    ccall((:mlirRegionGetFirstBlock, mlir_c), MlirBlock, (MlirRegion,), region)
end

function mlirRegionAppendOwnedBlock(region, block)
    ccall((:mlirRegionAppendOwnedBlock, mlir_c), Cvoid, (MlirRegion, MlirBlock), region, block)
end

function mlirRegionInsertOwnedBlock(region, pos, block)
    ccall((:mlirRegionInsertOwnedBlock, mlir_c), Cvoid, (MlirRegion, intptr_t, MlirBlock), region, pos, block)
end

function mlirRegionInsertOwnedBlockAfter(region, reference, block)
    ccall((:mlirRegionInsertOwnedBlockAfter, mlir_c), Cvoid, (MlirRegion, MlirBlock, MlirBlock), region, reference, block)
end

function mlirRegionInsertOwnedBlockBefore(region, reference, block)
    ccall((:mlirRegionInsertOwnedBlockBefore, mlir_c), Cvoid, (MlirRegion, MlirBlock, MlirBlock), region, reference, block)
end

function mlirOperationGetFirstRegion(op)
    ccall((:mlirOperationGetFirstRegion, mlir_c), MlirRegion, (MlirOperation,), op)
end

function mlirRegionGetNextInOperation(region)
    ccall((:mlirRegionGetNextInOperation, mlir_c), MlirRegion, (MlirRegion,), region)
end

function mlirBlockCreate(nArgs, args, locs)
    ccall((:mlirBlockCreate, mlir_c), MlirBlock, (intptr_t, Ptr{MlirType}, Ptr{MlirLocation}), nArgs, args, locs)
end

function mlirBlockDestroy(block)
    ccall((:mlirBlockDestroy, mlir_c), Cvoid, (MlirBlock,), block)
end

function mlirBlockDetach(block)
    ccall((:mlirBlockDetach, mlir_c), Cvoid, (MlirBlock,), block)
end

function mlirBlockIsNull(block)
    ccall((:mlirBlockIsNull, mlir_c), Bool, (MlirBlock,), block)
end

function mlirBlockEqual(block, other)
    ccall((:mlirBlockEqual, mlir_c), Bool, (MlirBlock, MlirBlock), block, other)
end

function mlirBlockGetParentOperation(arg1)
    ccall((:mlirBlockGetParentOperation, mlir_c), MlirOperation, (MlirBlock,), arg1)
end

function mlirBlockGetParentRegion(block)
    ccall((:mlirBlockGetParentRegion, mlir_c), MlirRegion, (MlirBlock,), block)
end

function mlirBlockGetNextInRegion(block)
    ccall((:mlirBlockGetNextInRegion, mlir_c), MlirBlock, (MlirBlock,), block)
end

function mlirBlockGetFirstOperation(block)
    ccall((:mlirBlockGetFirstOperation, mlir_c), MlirOperation, (MlirBlock,), block)
end

function mlirBlockGetTerminator(block)
    ccall((:mlirBlockGetTerminator, mlir_c), MlirOperation, (MlirBlock,), block)
end

function mlirBlockAppendOwnedOperation(block, operation)
    ccall((:mlirBlockAppendOwnedOperation, mlir_c), Cvoid, (MlirBlock, MlirOperation), block, operation)
end

function mlirBlockInsertOwnedOperation(block, pos, operation)
    ccall((:mlirBlockInsertOwnedOperation, mlir_c), Cvoid, (MlirBlock, intptr_t, MlirOperation), block, pos, operation)
end

function mlirBlockInsertOwnedOperationAfter(block, reference, operation)
    ccall((:mlirBlockInsertOwnedOperationAfter, mlir_c), Cvoid, (MlirBlock, MlirOperation, MlirOperation), block, reference, operation)
end

function mlirBlockInsertOwnedOperationBefore(block, reference, operation)
    ccall((:mlirBlockInsertOwnedOperationBefore, mlir_c), Cvoid, (MlirBlock, MlirOperation, MlirOperation), block, reference, operation)
end

function mlirBlockGetNumArguments(block)
    ccall((:mlirBlockGetNumArguments, mlir_c), intptr_t, (MlirBlock,), block)
end

function mlirBlockAddArgument(block, type, loc)
    ccall((:mlirBlockAddArgument, mlir_c), MlirValue, (MlirBlock, MlirType, MlirLocation), block, type, loc)
end

function mlirBlockGetArgument(block, pos)
    ccall((:mlirBlockGetArgument, mlir_c), MlirValue, (MlirBlock, intptr_t), block, pos)
end

function mlirBlockPrint(block, callback, userData)
    ccall((:mlirBlockPrint, mlir_c), Cvoid, (MlirBlock, MlirStringCallback, Ptr{Cvoid}), block, callback, userData)
end

function mlirValueIsNull(value)
    ccall((:mlirValueIsNull, mlir_c), Bool, (MlirValue,), value)
end

function mlirValueEqual(value1, value2)
    ccall((:mlirValueEqual, mlir_c), Bool, (MlirValue, MlirValue), value1, value2)
end

function mlirValueIsABlockArgument(value)
    ccall((:mlirValueIsABlockArgument, mlir_c), Bool, (MlirValue,), value)
end

function mlirValueIsAOpResult(value)
    ccall((:mlirValueIsAOpResult, mlir_c), Bool, (MlirValue,), value)
end

function mlirBlockArgumentGetOwner(value)
    ccall((:mlirBlockArgumentGetOwner, mlir_c), MlirBlock, (MlirValue,), value)
end

function mlirBlockArgumentGetArgNumber(value)
    ccall((:mlirBlockArgumentGetArgNumber, mlir_c), intptr_t, (MlirValue,), value)
end

function mlirBlockArgumentSetType(value, type)
    ccall((:mlirBlockArgumentSetType, mlir_c), Cvoid, (MlirValue, MlirType), value, type)
end

function mlirOpResultGetOwner(value)
    ccall((:mlirOpResultGetOwner, mlir_c), MlirOperation, (MlirValue,), value)
end

function mlirOpResultGetResultNumber(value)
    ccall((:mlirOpResultGetResultNumber, mlir_c), intptr_t, (MlirValue,), value)
end

function mlirValueGetType(value)
    ccall((:mlirValueGetType, mlir_c), MlirType, (MlirValue,), value)
end

function mlirValueDump(value)
    ccall((:mlirValueDump, mlir_c), Cvoid, (MlirValue,), value)
end

function mlirValuePrint(value, callback, userData)
    ccall((:mlirValuePrint, mlir_c), Cvoid, (MlirValue, MlirStringCallback, Ptr{Cvoid}), value, callback, userData)
end

function mlirTypeParseGet(context, type)
    ccall((:mlirTypeParseGet, mlir_c), MlirType, (MlirContext, MlirStringRef), context, type)
end

function mlirTypeGetContext(type)
    ccall((:mlirTypeGetContext, mlir_c), MlirContext, (MlirType,), type)
end

function mlirTypeGetTypeID(type)
    ccall((:mlirTypeGetTypeID, mlir_c), MlirTypeID, (MlirType,), type)
end

function mlirTypeIsNull(type)
    ccall((:mlirTypeIsNull, mlir_c), Bool, (MlirType,), type)
end

function mlirTypeEqual(t1, t2)
    ccall((:mlirTypeEqual, mlir_c), Bool, (MlirType, MlirType), t1, t2)
end

function mlirTypePrint(type, callback, userData)
    ccall((:mlirTypePrint, mlir_c), Cvoid, (MlirType, MlirStringCallback, Ptr{Cvoid}), type, callback, userData)
end

function mlirTypeDump(type)
    ccall((:mlirTypeDump, mlir_c), Cvoid, (MlirType,), type)
end

function mlirAttributeParseGet(context, attr)
    ccall((:mlirAttributeParseGet, mlir_c), MlirAttribute, (MlirContext, MlirStringRef), context, attr)
end

function mlirAttributeGetContext(attribute)
    ccall((:mlirAttributeGetContext, mlir_c), MlirContext, (MlirAttribute,), attribute)
end

function mlirAttributeGetType(attribute)
    ccall((:mlirAttributeGetType, mlir_c), MlirType, (MlirAttribute,), attribute)
end

function mlirAttributeGetTypeID(attribute)
    ccall((:mlirAttributeGetTypeID, mlir_c), MlirTypeID, (MlirAttribute,), attribute)
end

function mlirAttributeIsNull(attr)
    ccall((:mlirAttributeIsNull, mlir_c), Bool, (MlirAttribute,), attr)
end

function mlirAttributeEqual(a1, a2)
    ccall((:mlirAttributeEqual, mlir_c), Bool, (MlirAttribute, MlirAttribute), a1, a2)
end

function mlirAttributePrint(attr, callback, userData)
    ccall((:mlirAttributePrint, mlir_c), Cvoid, (MlirAttribute, MlirStringCallback, Ptr{Cvoid}), attr, callback, userData)
end

function mlirAttributeDump(attr)
    ccall((:mlirAttributeDump, mlir_c), Cvoid, (MlirAttribute,), attr)
end

function mlirNamedAttributeGet(name, attr)
    ccall((:mlirNamedAttributeGet, mlir_c), MlirNamedAttribute, (MlirIdentifier, MlirAttribute), name, attr)
end

function mlirIdentifierGet(context, str)
    ccall((:mlirIdentifierGet, mlir_c), MlirIdentifier, (MlirContext, MlirStringRef), context, str)
end

function mlirIdentifierGetContext(arg1)
    ccall((:mlirIdentifierGetContext, mlir_c), MlirContext, (MlirIdentifier,), arg1)
end

function mlirIdentifierEqual(ident, other)
    ccall((:mlirIdentifierEqual, mlir_c), Bool, (MlirIdentifier, MlirIdentifier), ident, other)
end

function mlirIdentifierStr(ident)
    ccall((:mlirIdentifierStr, mlir_c), MlirStringRef, (MlirIdentifier,), ident)
end

# no prototype is found for this function at IR.h:814:34, please use with caution
function mlirSymbolTableGetSymbolAttributeName()
    ccall((:mlirSymbolTableGetSymbolAttributeName, mlir_c), MlirStringRef, ())
end

# no prototype is found for this function at IR.h:817:34, please use with caution
function mlirSymbolTableGetVisibilityAttributeName()
    ccall((:mlirSymbolTableGetVisibilityAttributeName, mlir_c), MlirStringRef, ())
end

function mlirSymbolTableCreate(operation)
    ccall((:mlirSymbolTableCreate, mlir_c), MlirSymbolTable, (MlirOperation,), operation)
end

function mlirSymbolTableIsNull(symbolTable)
    ccall((:mlirSymbolTableIsNull, mlir_c), Bool, (MlirSymbolTable,), symbolTable)
end

function mlirSymbolTableDestroy(symbolTable)
    ccall((:mlirSymbolTableDestroy, mlir_c), Cvoid, (MlirSymbolTable,), symbolTable)
end

function mlirSymbolTableLookup(symbolTable, name)
    ccall((:mlirSymbolTableLookup, mlir_c), MlirOperation, (MlirSymbolTable, MlirStringRef), symbolTable, name)
end

function mlirSymbolTableInsert(symbolTable, operation)
    ccall((:mlirSymbolTableInsert, mlir_c), MlirAttribute, (MlirSymbolTable, MlirOperation), symbolTable, operation)
end

function mlirSymbolTableErase(symbolTable, operation)
    ccall((:mlirSymbolTableErase, mlir_c), Cvoid, (MlirSymbolTable, MlirOperation), symbolTable, operation)
end

function mlirSymbolTableReplaceAllSymbolUses(oldSymbol, newSymbol, from)
    ccall((:mlirSymbolTableReplaceAllSymbolUses, mlir_c), MlirLogicalResult, (MlirStringRef, MlirStringRef, MlirOperation), oldSymbol, newSymbol, from)
end

function mlirSymbolTableWalkSymbolTables(from, allSymUsesVisible, callback, userData)
    ccall((:mlirSymbolTableWalkSymbolTables, mlir_c), Cvoid, (MlirOperation, Bool, Ptr{Cvoid}, Ptr{Cvoid}), from, allSymUsesVisible, callback, userData)
end

struct MlirAffineExpr
    ptr::Ptr{Cvoid}
end

function mlirAffineExprGetContext(affineExpr)
    ccall((:mlirAffineExprGetContext, mlir_c), MlirContext, (MlirAffineExpr,), affineExpr)
end

function mlirAffineExprEqual(lhs, rhs)
    ccall((:mlirAffineExprEqual, mlir_c), Bool, (MlirAffineExpr, MlirAffineExpr), lhs, rhs)
end

function mlirAffineExprIsNull(affineExpr)
    ccall((:mlirAffineExprIsNull, mlir_c), Bool, (MlirAffineExpr,), affineExpr)
end

function mlirAffineExprPrint(affineExpr, callback, userData)
    ccall((:mlirAffineExprPrint, mlir_c), Cvoid, (MlirAffineExpr, MlirStringCallback, Ptr{Cvoid}), affineExpr, callback, userData)
end

function mlirAffineExprDump(affineExpr)
    ccall((:mlirAffineExprDump, mlir_c), Cvoid, (MlirAffineExpr,), affineExpr)
end

function mlirAffineExprIsSymbolicOrConstant(affineExpr)
    ccall((:mlirAffineExprIsSymbolicOrConstant, mlir_c), Bool, (MlirAffineExpr,), affineExpr)
end

function mlirAffineExprIsPureAffine(affineExpr)
    ccall((:mlirAffineExprIsPureAffine, mlir_c), Bool, (MlirAffineExpr,), affineExpr)
end

function mlirAffineExprGetLargestKnownDivisor(affineExpr)
    ccall((:mlirAffineExprGetLargestKnownDivisor, mlir_c), Int64, (MlirAffineExpr,), affineExpr)
end

function mlirAffineExprIsMultipleOf(affineExpr, factor)
    ccall((:mlirAffineExprIsMultipleOf, mlir_c), Bool, (MlirAffineExpr, Int64), affineExpr, factor)
end

function mlirAffineExprIsFunctionOfDim(affineExpr, position)
    ccall((:mlirAffineExprIsFunctionOfDim, mlir_c), Bool, (MlirAffineExpr, intptr_t), affineExpr, position)
end

struct MlirAffineMap
    ptr::Ptr{Cvoid}
end

function mlirAffineExprCompose(affineExpr, affineMap)
    ccall((:mlirAffineExprCompose, mlir_c), MlirAffineExpr, (MlirAffineExpr, MlirAffineMap), affineExpr, affineMap)
end

function mlirAffineExprIsADim(affineExpr)
    ccall((:mlirAffineExprIsADim, mlir_c), Bool, (MlirAffineExpr,), affineExpr)
end

function mlirAffineDimExprGet(ctx, position)
    ccall((:mlirAffineDimExprGet, mlir_c), MlirAffineExpr, (MlirContext, intptr_t), ctx, position)
end

function mlirAffineDimExprGetPosition(affineExpr)
    ccall((:mlirAffineDimExprGetPosition, mlir_c), intptr_t, (MlirAffineExpr,), affineExpr)
end

function mlirAffineExprIsASymbol(affineExpr)
    ccall((:mlirAffineExprIsASymbol, mlir_c), Bool, (MlirAffineExpr,), affineExpr)
end

function mlirAffineSymbolExprGet(ctx, position)
    ccall((:mlirAffineSymbolExprGet, mlir_c), MlirAffineExpr, (MlirContext, intptr_t), ctx, position)
end

function mlirAffineSymbolExprGetPosition(affineExpr)
    ccall((:mlirAffineSymbolExprGetPosition, mlir_c), intptr_t, (MlirAffineExpr,), affineExpr)
end

function mlirAffineExprIsAConstant(affineExpr)
    ccall((:mlirAffineExprIsAConstant, mlir_c), Bool, (MlirAffineExpr,), affineExpr)
end

function mlirAffineConstantExprGet(ctx, constant)
    ccall((:mlirAffineConstantExprGet, mlir_c), MlirAffineExpr, (MlirContext, Int64), ctx, constant)
end

function mlirAffineConstantExprGetValue(affineExpr)
    ccall((:mlirAffineConstantExprGetValue, mlir_c), Int64, (MlirAffineExpr,), affineExpr)
end

function mlirAffineExprIsAAdd(affineExpr)
    ccall((:mlirAffineExprIsAAdd, mlir_c), Bool, (MlirAffineExpr,), affineExpr)
end

function mlirAffineAddExprGet(lhs, rhs)
    ccall((:mlirAffineAddExprGet, mlir_c), MlirAffineExpr, (MlirAffineExpr, MlirAffineExpr), lhs, rhs)
end

function mlirAffineExprIsAMul(affineExpr)
    ccall((:mlirAffineExprIsAMul, mlir_c), Bool, (MlirAffineExpr,), affineExpr)
end

function mlirAffineMulExprGet(lhs, rhs)
    ccall((:mlirAffineMulExprGet, mlir_c), MlirAffineExpr, (MlirAffineExpr, MlirAffineExpr), lhs, rhs)
end

function mlirAffineExprIsAMod(affineExpr)
    ccall((:mlirAffineExprIsAMod, mlir_c), Bool, (MlirAffineExpr,), affineExpr)
end

function mlirAffineModExprGet(lhs, rhs)
    ccall((:mlirAffineModExprGet, mlir_c), MlirAffineExpr, (MlirAffineExpr, MlirAffineExpr), lhs, rhs)
end

function mlirAffineExprIsAFloorDiv(affineExpr)
    ccall((:mlirAffineExprIsAFloorDiv, mlir_c), Bool, (MlirAffineExpr,), affineExpr)
end

function mlirAffineFloorDivExprGet(lhs, rhs)
    ccall((:mlirAffineFloorDivExprGet, mlir_c), MlirAffineExpr, (MlirAffineExpr, MlirAffineExpr), lhs, rhs)
end

function mlirAffineExprIsACeilDiv(affineExpr)
    ccall((:mlirAffineExprIsACeilDiv, mlir_c), Bool, (MlirAffineExpr,), affineExpr)
end

function mlirAffineCeilDivExprGet(lhs, rhs)
    ccall((:mlirAffineCeilDivExprGet, mlir_c), MlirAffineExpr, (MlirAffineExpr, MlirAffineExpr), lhs, rhs)
end

function mlirAffineExprIsABinary(affineExpr)
    ccall((:mlirAffineExprIsABinary, mlir_c), Bool, (MlirAffineExpr,), affineExpr)
end

function mlirAffineBinaryOpExprGetLHS(affineExpr)
    ccall((:mlirAffineBinaryOpExprGetLHS, mlir_c), MlirAffineExpr, (MlirAffineExpr,), affineExpr)
end

function mlirAffineBinaryOpExprGetRHS(affineExpr)
    ccall((:mlirAffineBinaryOpExprGetRHS, mlir_c), MlirAffineExpr, (MlirAffineExpr,), affineExpr)
end

function mlirAffineMapGetContext(affineMap)
    ccall((:mlirAffineMapGetContext, mlir_c), MlirContext, (MlirAffineMap,), affineMap)
end

function mlirAffineMapIsNull(affineMap)
    ccall((:mlirAffineMapIsNull, mlir_c), Bool, (MlirAffineMap,), affineMap)
end

function mlirAffineMapEqual(a1, a2)
    ccall((:mlirAffineMapEqual, mlir_c), Bool, (MlirAffineMap, MlirAffineMap), a1, a2)
end

function mlirAffineMapPrint(affineMap, callback, userData)
    ccall((:mlirAffineMapPrint, mlir_c), Cvoid, (MlirAffineMap, MlirStringCallback, Ptr{Cvoid}), affineMap, callback, userData)
end

function mlirAffineMapDump(affineMap)
    ccall((:mlirAffineMapDump, mlir_c), Cvoid, (MlirAffineMap,), affineMap)
end

function mlirAffineMapEmptyGet(ctx)
    ccall((:mlirAffineMapEmptyGet, mlir_c), MlirAffineMap, (MlirContext,), ctx)
end

function mlirAffineMapZeroResultGet(ctx, dimCount, symbolCount)
    ccall((:mlirAffineMapZeroResultGet, mlir_c), MlirAffineMap, (MlirContext, intptr_t, intptr_t), ctx, dimCount, symbolCount)
end

function mlirAffineMapGet(ctx, dimCount, symbolCount, nAffineExprs, affineExprs)
    ccall((:mlirAffineMapGet, mlir_c), MlirAffineMap, (MlirContext, intptr_t, intptr_t, intptr_t, Ptr{MlirAffineExpr}), ctx, dimCount, symbolCount, nAffineExprs, affineExprs)
end

function mlirAffineMapConstantGet(ctx, val)
    ccall((:mlirAffineMapConstantGet, mlir_c), MlirAffineMap, (MlirContext, Int64), ctx, val)
end

function mlirAffineMapMultiDimIdentityGet(ctx, numDims)
    ccall((:mlirAffineMapMultiDimIdentityGet, mlir_c), MlirAffineMap, (MlirContext, intptr_t), ctx, numDims)
end

function mlirAffineMapMinorIdentityGet(ctx, dims, results)
    ccall((:mlirAffineMapMinorIdentityGet, mlir_c), MlirAffineMap, (MlirContext, intptr_t, intptr_t), ctx, dims, results)
end

function mlirAffineMapPermutationGet(ctx, size, permutation)
    ccall((:mlirAffineMapPermutationGet, mlir_c), MlirAffineMap, (MlirContext, intptr_t, Ptr{Cuint}), ctx, size, permutation)
end

function mlirAffineMapIsIdentity(affineMap)
    ccall((:mlirAffineMapIsIdentity, mlir_c), Bool, (MlirAffineMap,), affineMap)
end

function mlirAffineMapIsMinorIdentity(affineMap)
    ccall((:mlirAffineMapIsMinorIdentity, mlir_c), Bool, (MlirAffineMap,), affineMap)
end

function mlirAffineMapIsEmpty(affineMap)
    ccall((:mlirAffineMapIsEmpty, mlir_c), Bool, (MlirAffineMap,), affineMap)
end

function mlirAffineMapIsSingleConstant(affineMap)
    ccall((:mlirAffineMapIsSingleConstant, mlir_c), Bool, (MlirAffineMap,), affineMap)
end

function mlirAffineMapGetSingleConstantResult(affineMap)
    ccall((:mlirAffineMapGetSingleConstantResult, mlir_c), Int64, (MlirAffineMap,), affineMap)
end

function mlirAffineMapGetNumDims(affineMap)
    ccall((:mlirAffineMapGetNumDims, mlir_c), intptr_t, (MlirAffineMap,), affineMap)
end

function mlirAffineMapGetNumSymbols(affineMap)
    ccall((:mlirAffineMapGetNumSymbols, mlir_c), intptr_t, (MlirAffineMap,), affineMap)
end

function mlirAffineMapGetNumResults(affineMap)
    ccall((:mlirAffineMapGetNumResults, mlir_c), intptr_t, (MlirAffineMap,), affineMap)
end

function mlirAffineMapGetResult(affineMap, pos)
    ccall((:mlirAffineMapGetResult, mlir_c), MlirAffineExpr, (MlirAffineMap, intptr_t), affineMap, pos)
end

function mlirAffineMapGetNumInputs(affineMap)
    ccall((:mlirAffineMapGetNumInputs, mlir_c), intptr_t, (MlirAffineMap,), affineMap)
end

function mlirAffineMapIsProjectedPermutation(affineMap)
    ccall((:mlirAffineMapIsProjectedPermutation, mlir_c), Bool, (MlirAffineMap,), affineMap)
end

function mlirAffineMapIsPermutation(affineMap)
    ccall((:mlirAffineMapIsPermutation, mlir_c), Bool, (MlirAffineMap,), affineMap)
end

function mlirAffineMapGetSubMap(affineMap, size, resultPos)
    ccall((:mlirAffineMapGetSubMap, mlir_c), MlirAffineMap, (MlirAffineMap, intptr_t, Ptr{intptr_t}), affineMap, size, resultPos)
end

function mlirAffineMapGetMajorSubMap(affineMap, numResults)
    ccall((:mlirAffineMapGetMajorSubMap, mlir_c), MlirAffineMap, (MlirAffineMap, intptr_t), affineMap, numResults)
end

function mlirAffineMapGetMinorSubMap(affineMap, numResults)
    ccall((:mlirAffineMapGetMinorSubMap, mlir_c), MlirAffineMap, (MlirAffineMap, intptr_t), affineMap, numResults)
end

function mlirAffineMapReplace(affineMap, expression, replacement, numResultDims, numResultSyms)
    ccall((:mlirAffineMapReplace, mlir_c), MlirAffineMap, (MlirAffineMap, MlirAffineExpr, MlirAffineExpr, intptr_t, intptr_t), affineMap, expression, replacement, numResultDims, numResultSyms)
end

function mlirAffineMapCompressUnusedSymbols(affineMaps, size, result, populateResult)
    ccall((:mlirAffineMapCompressUnusedSymbols, mlir_c), Cvoid, (Ptr{MlirAffineMap}, intptr_t, Ptr{Cvoid}, Ptr{Cvoid}), affineMaps, size, result, populateResult)
end

# no prototype is found for this function at BuiltinAttributes.h:26:34, please use with caution
function mlirAttributeGetNull()
    ccall((:mlirAttributeGetNull, mlir_c), MlirAttribute, ())
end

function mlirAttributeIsAAffineMap(attr)
    ccall((:mlirAttributeIsAAffineMap, mlir_c), Bool, (MlirAttribute,), attr)
end

function mlirAffineMapAttrGet(map)
    ccall((:mlirAffineMapAttrGet, mlir_c), MlirAttribute, (MlirAffineMap,), map)
end

function mlirAffineMapAttrGetValue(attr)
    ccall((:mlirAffineMapAttrGetValue, mlir_c), MlirAffineMap, (MlirAttribute,), attr)
end

function mlirAttributeIsAArray(attr)
    ccall((:mlirAttributeIsAArray, mlir_c), Bool, (MlirAttribute,), attr)
end

function mlirArrayAttrGet(ctx, numElements, elements)
    ccall((:mlirArrayAttrGet, mlir_c), MlirAttribute, (MlirContext, intptr_t, Ptr{MlirAttribute}), ctx, numElements, elements)
end

function mlirArrayAttrGetNumElements(attr)
    ccall((:mlirArrayAttrGetNumElements, mlir_c), intptr_t, (MlirAttribute,), attr)
end

function mlirArrayAttrGetElement(attr, pos)
    ccall((:mlirArrayAttrGetElement, mlir_c), MlirAttribute, (MlirAttribute, intptr_t), attr, pos)
end

function mlirAttributeIsADictionary(attr)
    ccall((:mlirAttributeIsADictionary, mlir_c), Bool, (MlirAttribute,), attr)
end

function mlirDictionaryAttrGet(ctx, numElements, elements)
    ccall((:mlirDictionaryAttrGet, mlir_c), MlirAttribute, (MlirContext, intptr_t, Ptr{MlirNamedAttribute}), ctx, numElements, elements)
end

function mlirDictionaryAttrGetNumElements(attr)
    ccall((:mlirDictionaryAttrGetNumElements, mlir_c), intptr_t, (MlirAttribute,), attr)
end

function mlirDictionaryAttrGetElement(attr, pos)
    ccall((:mlirDictionaryAttrGetElement, mlir_c), MlirNamedAttribute, (MlirAttribute, intptr_t), attr, pos)
end

function mlirDictionaryAttrGetElementByName(attr, name)
    ccall((:mlirDictionaryAttrGetElementByName, mlir_c), MlirAttribute, (MlirAttribute, MlirStringRef), attr, name)
end

function mlirAttributeIsAFloat(attr)
    ccall((:mlirAttributeIsAFloat, mlir_c), Bool, (MlirAttribute,), attr)
end

function mlirFloatAttrDoubleGet(ctx, type, value)
    ccall((:mlirFloatAttrDoubleGet, mlir_c), MlirAttribute, (MlirContext, MlirType, Cdouble), ctx, type, value)
end

function mlirFloatAttrDoubleGetChecked(loc, type, value)
    ccall((:mlirFloatAttrDoubleGetChecked, mlir_c), MlirAttribute, (MlirLocation, MlirType, Cdouble), loc, type, value)
end

function mlirFloatAttrGetValueDouble(attr)
    ccall((:mlirFloatAttrGetValueDouble, mlir_c), Cdouble, (MlirAttribute,), attr)
end

function mlirAttributeIsAInteger(attr)
    ccall((:mlirAttributeIsAInteger, mlir_c), Bool, (MlirAttribute,), attr)
end

function mlirIntegerAttrGet(type, value)
    ccall((:mlirIntegerAttrGet, mlir_c), MlirAttribute, (MlirType, Int64), type, value)
end

function mlirIntegerAttrGetValueInt(attr)
    ccall((:mlirIntegerAttrGetValueInt, mlir_c), Int64, (MlirAttribute,), attr)
end

function mlirIntegerAttrGetValueSInt(attr)
    ccall((:mlirIntegerAttrGetValueSInt, mlir_c), Int64, (MlirAttribute,), attr)
end

function mlirIntegerAttrGetValueUInt(attr)
    ccall((:mlirIntegerAttrGetValueUInt, mlir_c), UInt64, (MlirAttribute,), attr)
end

function mlirAttributeIsABool(attr)
    ccall((:mlirAttributeIsABool, mlir_c), Bool, (MlirAttribute,), attr)
end

function mlirBoolAttrGet(ctx, value)
    ccall((:mlirBoolAttrGet, mlir_c), MlirAttribute, (MlirContext, Cint), ctx, value)
end

function mlirBoolAttrGetValue(attr)
    ccall((:mlirBoolAttrGetValue, mlir_c), Bool, (MlirAttribute,), attr)
end

function mlirAttributeIsAIntegerSet(attr)
    ccall((:mlirAttributeIsAIntegerSet, mlir_c), Bool, (MlirAttribute,), attr)
end

function mlirAttributeIsAOpaque(attr)
    ccall((:mlirAttributeIsAOpaque, mlir_c), Bool, (MlirAttribute,), attr)
end

function mlirOpaqueAttrGet(ctx, dialectNamespace, dataLength, data, type)
    ccall((:mlirOpaqueAttrGet, mlir_c), MlirAttribute, (MlirContext, MlirStringRef, intptr_t, Cstring, MlirType), ctx, dialectNamespace, dataLength, data, type)
end

function mlirOpaqueAttrGetDialectNamespace(attr)
    ccall((:mlirOpaqueAttrGetDialectNamespace, mlir_c), MlirStringRef, (MlirAttribute,), attr)
end

function mlirOpaqueAttrGetData(attr)
    ccall((:mlirOpaqueAttrGetData, mlir_c), MlirStringRef, (MlirAttribute,), attr)
end

function mlirAttributeIsAString(attr)
    ccall((:mlirAttributeIsAString, mlir_c), Bool, (MlirAttribute,), attr)
end

function mlirStringAttrGet(ctx, str)
    ccall((:mlirStringAttrGet, mlir_c), MlirAttribute, (MlirContext, MlirStringRef), ctx, str)
end

function mlirStringAttrTypedGet(type, str)
    ccall((:mlirStringAttrTypedGet, mlir_c), MlirAttribute, (MlirType, MlirStringRef), type, str)
end

function mlirStringAttrGetValue(attr)
    ccall((:mlirStringAttrGetValue, mlir_c), MlirStringRef, (MlirAttribute,), attr)
end

function mlirAttributeIsASymbolRef(attr)
    ccall((:mlirAttributeIsASymbolRef, mlir_c), Bool, (MlirAttribute,), attr)
end

function mlirSymbolRefAttrGet(ctx, symbol, numReferences, references)
    ccall((:mlirSymbolRefAttrGet, mlir_c), MlirAttribute, (MlirContext, MlirStringRef, intptr_t, Ptr{MlirAttribute}), ctx, symbol, numReferences, references)
end

function mlirSymbolRefAttrGetRootReference(attr)
    ccall((:mlirSymbolRefAttrGetRootReference, mlir_c), MlirStringRef, (MlirAttribute,), attr)
end

function mlirSymbolRefAttrGetLeafReference(attr)
    ccall((:mlirSymbolRefAttrGetLeafReference, mlir_c), MlirStringRef, (MlirAttribute,), attr)
end

function mlirSymbolRefAttrGetNumNestedReferences(attr)
    ccall((:mlirSymbolRefAttrGetNumNestedReferences, mlir_c), intptr_t, (MlirAttribute,), attr)
end

function mlirSymbolRefAttrGetNestedReference(attr, pos)
    ccall((:mlirSymbolRefAttrGetNestedReference, mlir_c), MlirAttribute, (MlirAttribute, intptr_t), attr, pos)
end

function mlirAttributeIsAFlatSymbolRef(attr)
    ccall((:mlirAttributeIsAFlatSymbolRef, mlir_c), Bool, (MlirAttribute,), attr)
end

function mlirFlatSymbolRefAttrGet(ctx, symbol)
    ccall((:mlirFlatSymbolRefAttrGet, mlir_c), MlirAttribute, (MlirContext, MlirStringRef), ctx, symbol)
end

function mlirFlatSymbolRefAttrGetValue(attr)
    ccall((:mlirFlatSymbolRefAttrGetValue, mlir_c), MlirStringRef, (MlirAttribute,), attr)
end

function mlirAttributeIsAType(attr)
    ccall((:mlirAttributeIsAType, mlir_c), Bool, (MlirAttribute,), attr)
end

function mlirTypeAttrGet(type)
    ccall((:mlirTypeAttrGet, mlir_c), MlirAttribute, (MlirType,), type)
end

function mlirTypeAttrGetValue(attr)
    ccall((:mlirTypeAttrGetValue, mlir_c), MlirType, (MlirAttribute,), attr)
end

function mlirAttributeIsAUnit(attr)
    ccall((:mlirAttributeIsAUnit, mlir_c), Bool, (MlirAttribute,), attr)
end

function mlirUnitAttrGet(ctx)
    ccall((:mlirUnitAttrGet, mlir_c), MlirAttribute, (MlirContext,), ctx)
end

function mlirAttributeIsAElements(attr)
    ccall((:mlirAttributeIsAElements, mlir_c), Bool, (MlirAttribute,), attr)
end

function mlirElementsAttrGetValue(attr, rank, idxs)
    ccall((:mlirElementsAttrGetValue, mlir_c), MlirAttribute, (MlirAttribute, intptr_t, Ptr{UInt64}), attr, rank, idxs)
end

function mlirElementsAttrIsValidIndex(attr, rank, idxs)
    ccall((:mlirElementsAttrIsValidIndex, mlir_c), Bool, (MlirAttribute, intptr_t, Ptr{UInt64}), attr, rank, idxs)
end

function mlirElementsAttrGetNumElements(attr)
    ccall((:mlirElementsAttrGetNumElements, mlir_c), Int64, (MlirAttribute,), attr)
end

function mlirAttributeIsADenseElements(attr)
    ccall((:mlirAttributeIsADenseElements, mlir_c), Bool, (MlirAttribute,), attr)
end

function mlirAttributeIsADenseIntElements(attr)
    ccall((:mlirAttributeIsADenseIntElements, mlir_c), Bool, (MlirAttribute,), attr)
end

function mlirAttributeIsADenseFPElements(attr)
    ccall((:mlirAttributeIsADenseFPElements, mlir_c), Bool, (MlirAttribute,), attr)
end

function mlirDenseElementsAttrGet(shapedType, numElements, elements)
    ccall((:mlirDenseElementsAttrGet, mlir_c), MlirAttribute, (MlirType, intptr_t, Ptr{MlirAttribute}), shapedType, numElements, elements)
end

function mlirDenseElementsAttrRawBufferGet(shapedType, rawBufferSize, rawBuffer)
    ccall((:mlirDenseElementsAttrRawBufferGet, mlir_c), MlirAttribute, (MlirType, Csize_t, Ptr{Cvoid}), shapedType, rawBufferSize, rawBuffer)
end

function mlirDenseElementsAttrSplatGet(shapedType, element)
    ccall((:mlirDenseElementsAttrSplatGet, mlir_c), MlirAttribute, (MlirType, MlirAttribute), shapedType, element)
end

function mlirDenseElementsAttrBoolSplatGet(shapedType, element)
    ccall((:mlirDenseElementsAttrBoolSplatGet, mlir_c), MlirAttribute, (MlirType, Bool), shapedType, element)
end

function mlirDenseElementsAttrUInt8SplatGet(shapedType, element)
    ccall((:mlirDenseElementsAttrUInt8SplatGet, mlir_c), MlirAttribute, (MlirType, UInt8), shapedType, element)
end

function mlirDenseElementsAttrInt8SplatGet(shapedType, element)
    ccall((:mlirDenseElementsAttrInt8SplatGet, mlir_c), MlirAttribute, (MlirType, Int8), shapedType, element)
end

function mlirDenseElementsAttrUInt32SplatGet(shapedType, element)
    ccall((:mlirDenseElementsAttrUInt32SplatGet, mlir_c), MlirAttribute, (MlirType, UInt32), shapedType, element)
end

function mlirDenseElementsAttrInt32SplatGet(shapedType, element)
    ccall((:mlirDenseElementsAttrInt32SplatGet, mlir_c), MlirAttribute, (MlirType, Int32), shapedType, element)
end

function mlirDenseElementsAttrUInt64SplatGet(shapedType, element)
    ccall((:mlirDenseElementsAttrUInt64SplatGet, mlir_c), MlirAttribute, (MlirType, UInt64), shapedType, element)
end

function mlirDenseElementsAttrInt64SplatGet(shapedType, element)
    ccall((:mlirDenseElementsAttrInt64SplatGet, mlir_c), MlirAttribute, (MlirType, Int64), shapedType, element)
end

function mlirDenseElementsAttrFloatSplatGet(shapedType, element)
    ccall((:mlirDenseElementsAttrFloatSplatGet, mlir_c), MlirAttribute, (MlirType, Cfloat), shapedType, element)
end

function mlirDenseElementsAttrDoubleSplatGet(shapedType, element)
    ccall((:mlirDenseElementsAttrDoubleSplatGet, mlir_c), MlirAttribute, (MlirType, Cdouble), shapedType, element)
end

function mlirDenseElementsAttrBoolGet(shapedType, numElements, elements)
    ccall((:mlirDenseElementsAttrBoolGet, mlir_c), MlirAttribute, (MlirType, intptr_t, Ptr{Cint}), shapedType, numElements, elements)
end

function mlirDenseElementsAttrUInt8Get(shapedType, numElements, elements)
    ccall((:mlirDenseElementsAttrUInt8Get, mlir_c), MlirAttribute, (MlirType, intptr_t, Ptr{UInt8}), shapedType, numElements, elements)
end

function mlirDenseElementsAttrInt8Get(shapedType, numElements, elements)
    ccall((:mlirDenseElementsAttrInt8Get, mlir_c), MlirAttribute, (MlirType, intptr_t, Ptr{Int8}), shapedType, numElements, elements)
end

function mlirDenseElementsAttrUInt16Get(shapedType, numElements, elements)
    ccall((:mlirDenseElementsAttrUInt16Get, mlir_c), MlirAttribute, (MlirType, intptr_t, Ptr{UInt16}), shapedType, numElements, elements)
end

function mlirDenseElementsAttrInt16Get(shapedType, numElements, elements)
    ccall((:mlirDenseElementsAttrInt16Get, mlir_c), MlirAttribute, (MlirType, intptr_t, Ptr{Int16}), shapedType, numElements, elements)
end

function mlirDenseElementsAttrUInt32Get(shapedType, numElements, elements)
    ccall((:mlirDenseElementsAttrUInt32Get, mlir_c), MlirAttribute, (MlirType, intptr_t, Ptr{UInt32}), shapedType, numElements, elements)
end

function mlirDenseElementsAttrInt32Get(shapedType, numElements, elements)
    ccall((:mlirDenseElementsAttrInt32Get, mlir_c), MlirAttribute, (MlirType, intptr_t, Ptr{Int32}), shapedType, numElements, elements)
end

function mlirDenseElementsAttrUInt64Get(shapedType, numElements, elements)
    ccall((:mlirDenseElementsAttrUInt64Get, mlir_c), MlirAttribute, (MlirType, intptr_t, Ptr{UInt64}), shapedType, numElements, elements)
end

function mlirDenseElementsAttrInt64Get(shapedType, numElements, elements)
    ccall((:mlirDenseElementsAttrInt64Get, mlir_c), MlirAttribute, (MlirType, intptr_t, Ptr{Int64}), shapedType, numElements, elements)
end

function mlirDenseElementsAttrFloatGet(shapedType, numElements, elements)
    ccall((:mlirDenseElementsAttrFloatGet, mlir_c), MlirAttribute, (MlirType, intptr_t, Ptr{Cfloat}), shapedType, numElements, elements)
end

function mlirDenseElementsAttrDoubleGet(shapedType, numElements, elements)
    ccall((:mlirDenseElementsAttrDoubleGet, mlir_c), MlirAttribute, (MlirType, intptr_t, Ptr{Cdouble}), shapedType, numElements, elements)
end

function mlirDenseElementsAttrBFloat16Get(shapedType, numElements, elements)
    ccall((:mlirDenseElementsAttrBFloat16Get, mlir_c), MlirAttribute, (MlirType, intptr_t, Ptr{UInt16}), shapedType, numElements, elements)
end

function mlirDenseElementsAttrFloat16Get(shapedType, numElements, elements)
    ccall((:mlirDenseElementsAttrFloat16Get, mlir_c), MlirAttribute, (MlirType, intptr_t, Ptr{UInt16}), shapedType, numElements, elements)
end

function mlirDenseElementsAttrStringGet(shapedType, numElements, strs)
    ccall((:mlirDenseElementsAttrStringGet, mlir_c), MlirAttribute, (MlirType, intptr_t, Ptr{MlirStringRef}), shapedType, numElements, strs)
end

function mlirDenseElementsAttrReshapeGet(attr, shapedType)
    ccall((:mlirDenseElementsAttrReshapeGet, mlir_c), MlirAttribute, (MlirAttribute, MlirType), attr, shapedType)
end

function mlirDenseElementsAttrIsSplat(attr)
    ccall((:mlirDenseElementsAttrIsSplat, mlir_c), Bool, (MlirAttribute,), attr)
end

function mlirDenseElementsAttrGetSplatValue(attr)
    ccall((:mlirDenseElementsAttrGetSplatValue, mlir_c), MlirAttribute, (MlirAttribute,), attr)
end

function mlirDenseElementsAttrGetBoolSplatValue(attr)
    ccall((:mlirDenseElementsAttrGetBoolSplatValue, mlir_c), Cint, (MlirAttribute,), attr)
end

function mlirDenseElementsAttrGetInt8SplatValue(attr)
    ccall((:mlirDenseElementsAttrGetInt8SplatValue, mlir_c), Int8, (MlirAttribute,), attr)
end

function mlirDenseElementsAttrGetUInt8SplatValue(attr)
    ccall((:mlirDenseElementsAttrGetUInt8SplatValue, mlir_c), UInt8, (MlirAttribute,), attr)
end

function mlirDenseElementsAttrGetInt32SplatValue(attr)
    ccall((:mlirDenseElementsAttrGetInt32SplatValue, mlir_c), Int32, (MlirAttribute,), attr)
end

function mlirDenseElementsAttrGetUInt32SplatValue(attr)
    ccall((:mlirDenseElementsAttrGetUInt32SplatValue, mlir_c), UInt32, (MlirAttribute,), attr)
end

function mlirDenseElementsAttrGetInt64SplatValue(attr)
    ccall((:mlirDenseElementsAttrGetInt64SplatValue, mlir_c), Int64, (MlirAttribute,), attr)
end

function mlirDenseElementsAttrGetUInt64SplatValue(attr)
    ccall((:mlirDenseElementsAttrGetUInt64SplatValue, mlir_c), UInt64, (MlirAttribute,), attr)
end

function mlirDenseElementsAttrGetFloatSplatValue(attr)
    ccall((:mlirDenseElementsAttrGetFloatSplatValue, mlir_c), Cfloat, (MlirAttribute,), attr)
end

function mlirDenseElementsAttrGetDoubleSplatValue(attr)
    ccall((:mlirDenseElementsAttrGetDoubleSplatValue, mlir_c), Cdouble, (MlirAttribute,), attr)
end

function mlirDenseElementsAttrGetStringSplatValue(attr)
    ccall((:mlirDenseElementsAttrGetStringSplatValue, mlir_c), MlirStringRef, (MlirAttribute,), attr)
end

function mlirDenseElementsAttrGetBoolValue(attr, pos)
    ccall((:mlirDenseElementsAttrGetBoolValue, mlir_c), Bool, (MlirAttribute, intptr_t), attr, pos)
end

function mlirDenseElementsAttrGetInt8Value(attr, pos)
    ccall((:mlirDenseElementsAttrGetInt8Value, mlir_c), Int8, (MlirAttribute, intptr_t), attr, pos)
end

function mlirDenseElementsAttrGetUInt8Value(attr, pos)
    ccall((:mlirDenseElementsAttrGetUInt8Value, mlir_c), UInt8, (MlirAttribute, intptr_t), attr, pos)
end

function mlirDenseElementsAttrGetInt16Value(attr, pos)
    ccall((:mlirDenseElementsAttrGetInt16Value, mlir_c), Int16, (MlirAttribute, intptr_t), attr, pos)
end

function mlirDenseElementsAttrGetUInt16Value(attr, pos)
    ccall((:mlirDenseElementsAttrGetUInt16Value, mlir_c), UInt16, (MlirAttribute, intptr_t), attr, pos)
end

function mlirDenseElementsAttrGetInt32Value(attr, pos)
    ccall((:mlirDenseElementsAttrGetInt32Value, mlir_c), Int32, (MlirAttribute, intptr_t), attr, pos)
end

function mlirDenseElementsAttrGetUInt32Value(attr, pos)
    ccall((:mlirDenseElementsAttrGetUInt32Value, mlir_c), UInt32, (MlirAttribute, intptr_t), attr, pos)
end

function mlirDenseElementsAttrGetInt64Value(attr, pos)
    ccall((:mlirDenseElementsAttrGetInt64Value, mlir_c), Int64, (MlirAttribute, intptr_t), attr, pos)
end

function mlirDenseElementsAttrGetUInt64Value(attr, pos)
    ccall((:mlirDenseElementsAttrGetUInt64Value, mlir_c), UInt64, (MlirAttribute, intptr_t), attr, pos)
end

function mlirDenseElementsAttrGetFloatValue(attr, pos)
    ccall((:mlirDenseElementsAttrGetFloatValue, mlir_c), Cfloat, (MlirAttribute, intptr_t), attr, pos)
end

function mlirDenseElementsAttrGetDoubleValue(attr, pos)
    ccall((:mlirDenseElementsAttrGetDoubleValue, mlir_c), Cdouble, (MlirAttribute, intptr_t), attr, pos)
end

function mlirDenseElementsAttrGetStringValue(attr, pos)
    ccall((:mlirDenseElementsAttrGetStringValue, mlir_c), MlirStringRef, (MlirAttribute, intptr_t), attr, pos)
end

function mlirDenseElementsAttrGetRawData(attr)
    ccall((:mlirDenseElementsAttrGetRawData, mlir_c), Ptr{Cvoid}, (MlirAttribute,), attr)
end

function mlirAttributeIsAOpaqueElements(attr)
    ccall((:mlirAttributeIsAOpaqueElements, mlir_c), Bool, (MlirAttribute,), attr)
end

function mlirAttributeIsASparseElements(attr)
    ccall((:mlirAttributeIsASparseElements, mlir_c), Bool, (MlirAttribute,), attr)
end

function mlirSparseElementsAttribute(shapedType, denseIndices, denseValues)
    ccall((:mlirSparseElementsAttribute, mlir_c), MlirAttribute, (MlirType, MlirAttribute, MlirAttribute), shapedType, denseIndices, denseValues)
end

function mlirSparseElementsAttrGetIndices(attr)
    ccall((:mlirSparseElementsAttrGetIndices, mlir_c), MlirAttribute, (MlirAttribute,), attr)
end

function mlirSparseElementsAttrGetValues(attr)
    ccall((:mlirSparseElementsAttrGetValues, mlir_c), MlirAttribute, (MlirAttribute,), attr)
end

function mlirTypeIsAInteger(type)
    ccall((:mlirTypeIsAInteger, mlir_c), Bool, (MlirType,), type)
end

function mlirIntegerTypeGet(ctx, bitwidth)
    ccall((:mlirIntegerTypeGet, mlir_c), MlirType, (MlirContext, Cuint), ctx, bitwidth)
end

function mlirIntegerTypeSignedGet(ctx, bitwidth)
    ccall((:mlirIntegerTypeSignedGet, mlir_c), MlirType, (MlirContext, Cuint), ctx, bitwidth)
end

function mlirIntegerTypeUnsignedGet(ctx, bitwidth)
    ccall((:mlirIntegerTypeUnsignedGet, mlir_c), MlirType, (MlirContext, Cuint), ctx, bitwidth)
end

function mlirIntegerTypeGetWidth(type)
    ccall((:mlirIntegerTypeGetWidth, mlir_c), Cuint, (MlirType,), type)
end

function mlirIntegerTypeIsSignless(type)
    ccall((:mlirIntegerTypeIsSignless, mlir_c), Bool, (MlirType,), type)
end

function mlirIntegerTypeIsSigned(type)
    ccall((:mlirIntegerTypeIsSigned, mlir_c), Bool, (MlirType,), type)
end

function mlirIntegerTypeIsUnsigned(type)
    ccall((:mlirIntegerTypeIsUnsigned, mlir_c), Bool, (MlirType,), type)
end

function mlirTypeIsAIndex(type)
    ccall((:mlirTypeIsAIndex, mlir_c), Bool, (MlirType,), type)
end

function mlirIndexTypeGet(ctx)
    ccall((:mlirIndexTypeGet, mlir_c), MlirType, (MlirContext,), ctx)
end

function mlirTypeIsABF16(type)
    ccall((:mlirTypeIsABF16, mlir_c), Bool, (MlirType,), type)
end

function mlirBF16TypeGet(ctx)
    ccall((:mlirBF16TypeGet, mlir_c), MlirType, (MlirContext,), ctx)
end

function mlirTypeIsAF16(type)
    ccall((:mlirTypeIsAF16, mlir_c), Bool, (MlirType,), type)
end

function mlirF16TypeGet(ctx)
    ccall((:mlirF16TypeGet, mlir_c), MlirType, (MlirContext,), ctx)
end

function mlirTypeIsAF32(type)
    ccall((:mlirTypeIsAF32, mlir_c), Bool, (MlirType,), type)
end

function mlirF32TypeGet(ctx)
    ccall((:mlirF32TypeGet, mlir_c), MlirType, (MlirContext,), ctx)
end

function mlirTypeIsAF64(type)
    ccall((:mlirTypeIsAF64, mlir_c), Bool, (MlirType,), type)
end

function mlirF64TypeGet(ctx)
    ccall((:mlirF64TypeGet, mlir_c), MlirType, (MlirContext,), ctx)
end

function mlirTypeIsANone(type)
    ccall((:mlirTypeIsANone, mlir_c), Bool, (MlirType,), type)
end

function mlirNoneTypeGet(ctx)
    ccall((:mlirNoneTypeGet, mlir_c), MlirType, (MlirContext,), ctx)
end

function mlirTypeIsAComplex(type)
    ccall((:mlirTypeIsAComplex, mlir_c), Bool, (MlirType,), type)
end

function mlirComplexTypeGet(elementType)
    ccall((:mlirComplexTypeGet, mlir_c), MlirType, (MlirType,), elementType)
end

function mlirComplexTypeGetElementType(type)
    ccall((:mlirComplexTypeGetElementType, mlir_c), MlirType, (MlirType,), type)
end

function mlirTypeIsAShaped(type)
    ccall((:mlirTypeIsAShaped, mlir_c), Bool, (MlirType,), type)
end

function mlirShapedTypeGetElementType(type)
    ccall((:mlirShapedTypeGetElementType, mlir_c), MlirType, (MlirType,), type)
end

function mlirShapedTypeHasRank(type)
    ccall((:mlirShapedTypeHasRank, mlir_c), Bool, (MlirType,), type)
end

function mlirShapedTypeGetRank(type)
    ccall((:mlirShapedTypeGetRank, mlir_c), Int64, (MlirType,), type)
end

function mlirShapedTypeHasStaticShape(type)
    ccall((:mlirShapedTypeHasStaticShape, mlir_c), Bool, (MlirType,), type)
end

function mlirShapedTypeIsDynamicDim(type, dim)
    ccall((:mlirShapedTypeIsDynamicDim, mlir_c), Bool, (MlirType, intptr_t), type, dim)
end

function mlirShapedTypeGetDimSize(type, dim)
    ccall((:mlirShapedTypeGetDimSize, mlir_c), Int64, (MlirType, intptr_t), type, dim)
end

function mlirShapedTypeIsDynamicSize(size)
    ccall((:mlirShapedTypeIsDynamicSize, mlir_c), Bool, (Int64,), size)
end

# no prototype is found for this function at BuiltinTypes.h:155:28, please use with caution
function mlirShapedTypeGetDynamicSize()
    ccall((:mlirShapedTypeGetDynamicSize, mlir_c), Int64, ())
end

function mlirShapedTypeIsDynamicStrideOrOffset(val)
    ccall((:mlirShapedTypeIsDynamicStrideOrOffset, mlir_c), Bool, (Int64,), val)
end

# no prototype is found for this function at BuiltinTypes.h:164:28, please use with caution
function mlirShapedTypeGetDynamicStrideOrOffset()
    ccall((:mlirShapedTypeGetDynamicStrideOrOffset, mlir_c), Int64, ())
end

function mlirTypeIsAVector(type)
    ccall((:mlirTypeIsAVector, mlir_c), Bool, (MlirType,), type)
end

function mlirVectorTypeGet(rank, shape, elementType)
    ccall((:mlirVectorTypeGet, mlir_c), MlirType, (intptr_t, Ptr{Int64}, MlirType), rank, shape, elementType)
end

function mlirVectorTypeGetChecked(loc, rank, shape, elementType)
    ccall((:mlirVectorTypeGetChecked, mlir_c), MlirType, (MlirLocation, intptr_t, Ptr{Int64}, MlirType), loc, rank, shape, elementType)
end

function mlirTypeIsATensor(type)
    ccall((:mlirTypeIsATensor, mlir_c), Bool, (MlirType,), type)
end

function mlirTypeIsARankedTensor(type)
    ccall((:mlirTypeIsARankedTensor, mlir_c), Bool, (MlirType,), type)
end

function mlirTypeIsAUnrankedTensor(type)
    ccall((:mlirTypeIsAUnrankedTensor, mlir_c), Bool, (MlirType,), type)
end

function mlirRankedTensorTypeGet(rank, shape, elementType, encoding)
    ccall((:mlirRankedTensorTypeGet, mlir_c), MlirType, (intptr_t, Ptr{Int64}, MlirType, MlirAttribute), rank, shape, elementType, encoding)
end

function mlirRankedTensorTypeGetChecked(loc, rank, shape, elementType, encoding)
    ccall((:mlirRankedTensorTypeGetChecked, mlir_c), MlirType, (MlirLocation, intptr_t, Ptr{Int64}, MlirType, MlirAttribute), loc, rank, shape, elementType, encoding)
end

function mlirRankedTensorTypeGetEncoding(type)
    ccall((:mlirRankedTensorTypeGetEncoding, mlir_c), MlirAttribute, (MlirType,), type)
end

function mlirUnrankedTensorTypeGet(elementType)
    ccall((:mlirUnrankedTensorTypeGet, mlir_c), MlirType, (MlirType,), elementType)
end

function mlirUnrankedTensorTypeGetChecked(loc, elementType)
    ccall((:mlirUnrankedTensorTypeGetChecked, mlir_c), MlirType, (MlirLocation, MlirType), loc, elementType)
end

function mlirTypeIsAMemRef(type)
    ccall((:mlirTypeIsAMemRef, mlir_c), Bool, (MlirType,), type)
end

function mlirTypeIsAUnrankedMemRef(type)
    ccall((:mlirTypeIsAUnrankedMemRef, mlir_c), Bool, (MlirType,), type)
end

function mlirMemRefTypeGet(elementType, rank, shape, layout, memorySpace)
    ccall((:mlirMemRefTypeGet, mlir_c), MlirType, (MlirType, intptr_t, Ptr{Int64}, MlirAttribute, MlirAttribute), elementType, rank, shape, layout, memorySpace)
end

function mlirMemRefTypeGetChecked(loc, elementType, rank, shape, layout, memorySpace)
    ccall((:mlirMemRefTypeGetChecked, mlir_c), MlirType, (MlirLocation, MlirType, intptr_t, Ptr{Int64}, MlirAttribute, MlirAttribute), loc, elementType, rank, shape, layout, memorySpace)
end

function mlirMemRefTypeContiguousGet(elementType, rank, shape, memorySpace)
    ccall((:mlirMemRefTypeContiguousGet, mlir_c), MlirType, (MlirType, intptr_t, Ptr{Int64}, MlirAttribute), elementType, rank, shape, memorySpace)
end

function mlirMemRefTypeContiguousGetChecked(loc, elementType, rank, shape, memorySpace)
    ccall((:mlirMemRefTypeContiguousGetChecked, mlir_c), MlirType, (MlirLocation, MlirType, intptr_t, Ptr{Int64}, MlirAttribute), loc, elementType, rank, shape, memorySpace)
end

function mlirUnrankedMemRefTypeGet(elementType, memorySpace)
    ccall((:mlirUnrankedMemRefTypeGet, mlir_c), MlirType, (MlirType, MlirAttribute), elementType, memorySpace)
end

function mlirUnrankedMemRefTypeGetChecked(loc, elementType, memorySpace)
    ccall((:mlirUnrankedMemRefTypeGetChecked, mlir_c), MlirType, (MlirLocation, MlirType, MlirAttribute), loc, elementType, memorySpace)
end

function mlirMemRefTypeGetLayout(type)
    ccall((:mlirMemRefTypeGetLayout, mlir_c), MlirAttribute, (MlirType,), type)
end

function mlirMemRefTypeGetAffineMap(type)
    ccall((:mlirMemRefTypeGetAffineMap, mlir_c), MlirAffineMap, (MlirType,), type)
end

function mlirMemRefTypeGetMemorySpace(type)
    ccall((:mlirMemRefTypeGetMemorySpace, mlir_c), MlirAttribute, (MlirType,), type)
end

function mlirUnrankedMemrefGetMemorySpace(type)
    ccall((:mlirUnrankedMemrefGetMemorySpace, mlir_c), MlirAttribute, (MlirType,), type)
end

function mlirTypeIsATuple(type)
    ccall((:mlirTypeIsATuple, mlir_c), Bool, (MlirType,), type)
end

function mlirTupleTypeGet(ctx, numElements, elements)
    ccall((:mlirTupleTypeGet, mlir_c), MlirType, (MlirContext, intptr_t, Ptr{MlirType}), ctx, numElements, elements)
end

function mlirTupleTypeGetNumTypes(type)
    ccall((:mlirTupleTypeGetNumTypes, mlir_c), intptr_t, (MlirType,), type)
end

function mlirTupleTypeGetType(type, pos)
    ccall((:mlirTupleTypeGetType, mlir_c), MlirType, (MlirType, intptr_t), type, pos)
end

function mlirTypeIsAFunction(type)
    ccall((:mlirTypeIsAFunction, mlir_c), Bool, (MlirType,), type)
end

function mlirFunctionTypeGet(ctx, numInputs, inputs, numResults, results)
    ccall((:mlirFunctionTypeGet, mlir_c), MlirType, (MlirContext, intptr_t, Ptr{MlirType}, intptr_t, Ptr{MlirType}), ctx, numInputs, inputs, numResults, results)
end

function mlirFunctionTypeGetNumInputs(type)
    ccall((:mlirFunctionTypeGetNumInputs, mlir_c), intptr_t, (MlirType,), type)
end

function mlirFunctionTypeGetNumResults(type)
    ccall((:mlirFunctionTypeGetNumResults, mlir_c), intptr_t, (MlirType,), type)
end

function mlirFunctionTypeGetInput(type, pos)
    ccall((:mlirFunctionTypeGetInput, mlir_c), MlirType, (MlirType, intptr_t), type, pos)
end

function mlirFunctionTypeGetResult(type, pos)
    ccall((:mlirFunctionTypeGetResult, mlir_c), MlirType, (MlirType, intptr_t), type, pos)
end

function mlirTypeIsAOpaque(type)
    ccall((:mlirTypeIsAOpaque, mlir_c), Bool, (MlirType,), type)
end

function mlirOpaqueTypeGet(ctx, dialectNamespace, typeData)
    ccall((:mlirOpaqueTypeGet, mlir_c), MlirType, (MlirContext, MlirStringRef, MlirStringRef), ctx, dialectNamespace, typeData)
end

function mlirOpaqueTypeGetDialectNamespace(type)
    ccall((:mlirOpaqueTypeGetDialectNamespace, mlir_c), MlirStringRef, (MlirType,), type)
end

function mlirOpaqueTypeGetData(type)
    ccall((:mlirOpaqueTypeGetData, mlir_c), MlirStringRef, (MlirType,), type)
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

function mlirPassManagerCreate(ctx)
    ccall((:mlirPassManagerCreate, mlir_c), MlirPassManager, (MlirContext,), ctx)
end

function mlirPassManagerDestroy(passManager)
    ccall((:mlirPassManagerDestroy, mlir_c), Cvoid, (MlirPassManager,), passManager)
end

function mlirPassManagerIsNull(passManager)
    ccall((:mlirPassManagerIsNull, mlir_c), Bool, (MlirPassManager,), passManager)
end

function mlirPassManagerGetAsOpPassManager(passManager)
    ccall((:mlirPassManagerGetAsOpPassManager, mlir_c), MlirOpPassManager, (MlirPassManager,), passManager)
end

function mlirPassManagerRun(passManager, _module)
    ccall((:mlirPassManagerRun, mlir_c), MlirLogicalResult, (MlirPassManager, MlirModule), passManager, _module)
end

function mlirPassManagerEnableIRPrinting(passManager)
    ccall((:mlirPassManagerEnableIRPrinting, mlir_c), Cvoid, (MlirPassManager,), passManager)
end

function mlirPassManagerEnableVerifier(passManager, enable)
    ccall((:mlirPassManagerEnableVerifier, mlir_c), Cvoid, (MlirPassManager, Bool), passManager, enable)
end

function mlirPassManagerGetNestedUnder(passManager, operationName)
    ccall((:mlirPassManagerGetNestedUnder, mlir_c), MlirOpPassManager, (MlirPassManager, MlirStringRef), passManager, operationName)
end

function mlirOpPassManagerGetNestedUnder(passManager, operationName)
    ccall((:mlirOpPassManagerGetNestedUnder, mlir_c), MlirOpPassManager, (MlirOpPassManager, MlirStringRef), passManager, operationName)
end

function mlirPassManagerAddOwnedPass(passManager, pass)
    ccall((:mlirPassManagerAddOwnedPass, mlir_c), Cvoid, (MlirPassManager, MlirPass), passManager, pass)
end

function mlirOpPassManagerAddOwnedPass(passManager, pass)
    ccall((:mlirOpPassManagerAddOwnedPass, mlir_c), Cvoid, (MlirOpPassManager, MlirPass), passManager, pass)
end

function mlirPrintPassPipeline(passManager, callback, userData)
    ccall((:mlirPrintPassPipeline, mlir_c), Cvoid, (MlirOpPassManager, MlirStringCallback, Ptr{Cvoid}), passManager, callback, userData)
end

function mlirParsePassPipeline(passManager, pipeline)
    ccall((:mlirParsePassPipeline, mlir_c), MlirLogicalResult, (MlirOpPassManager, MlirStringRef), passManager, pipeline)
end

struct MlirExternalPassCallbacks
    construct::Ptr{Cvoid}
    destruct::Ptr{Cvoid}
    initialize::Ptr{Cvoid}
    clone::Ptr{Cvoid}
    run::Ptr{Cvoid}
end

function mlirCreateExternalPass(passID, name, argument, description, opName, nDependentDialects, dependentDialects, callbacks, userData)
    ccall((:mlirCreateExternalPass, mlir_c), MlirPass, (MlirTypeID, MlirStringRef, MlirStringRef, MlirStringRef, MlirStringRef, intptr_t, Ptr{MlirDialectHandle}, MlirExternalPassCallbacks, Ptr{Cvoid}), passID, name, argument, description, opName, nDependentDialects, dependentDialects, callbacks, userData)
end

function mlirExternalPassSignalFailure(pass)
    ccall((:mlirExternalPassSignalFailure, mlir_c), Cvoid, (MlirExternalPass,), pass)
end

# no prototype is found for this function at Passes.capi.h.inc:11:25, please use with caution
function mlirRegisterConversionPasses()
    ccall((:mlirRegisterConversionPasses, mlir_c), Cvoid, ())
end

# no prototype is found for this function at Passes.capi.h.inc:15:29, please use with caution
function mlirCreateConversionConvertAMDGPUToROCDL()
    ccall((:mlirCreateConversionConvertAMDGPUToROCDL, mlir_c), MlirPass, ())
end

# no prototype is found for this function at Passes.capi.h.inc:16:25, please use with caution
function mlirRegisterConversionConvertAMDGPUToROCDL()
    ccall((:mlirRegisterConversionConvertAMDGPUToROCDL, mlir_c), Cvoid, ())
end

# no prototype is found for this function at Passes.capi.h.inc:20:29, please use with caution
function mlirCreateConversionConvertAffineForToGPU()
    ccall((:mlirCreateConversionConvertAffineForToGPU, mlir_c), MlirPass, ())
end

# no prototype is found for this function at Passes.capi.h.inc:21:25, please use with caution
function mlirRegisterConversionConvertAffineForToGPU()
    ccall((:mlirRegisterConversionConvertAffineForToGPU, mlir_c), Cvoid, ())
end

# no prototype is found for this function at Passes.capi.h.inc:25:29, please use with caution
function mlirCreateConversionConvertAffineToStandard()
    ccall((:mlirCreateConversionConvertAffineToStandard, mlir_c), MlirPass, ())
end

# no prototype is found for this function at Passes.capi.h.inc:26:25, please use with caution
function mlirRegisterConversionConvertAffineToStandard()
    ccall((:mlirRegisterConversionConvertAffineToStandard, mlir_c), Cvoid, ())
end

# no prototype is found for this function at Passes.capi.h.inc:30:29, please use with caution
function mlirCreateConversionConvertArithmeticToLLVM()
    ccall((:mlirCreateConversionConvertArithmeticToLLVM, mlir_c), MlirPass, ())
end

# no prototype is found for this function at Passes.capi.h.inc:31:25, please use with caution
function mlirRegisterConversionConvertArithmeticToLLVM()
    ccall((:mlirRegisterConversionConvertArithmeticToLLVM, mlir_c), Cvoid, ())
end

# no prototype is found for this function at Passes.capi.h.inc:35:29, please use with caution
function mlirCreateConversionConvertArithmeticToSPIRV()
    ccall((:mlirCreateConversionConvertArithmeticToSPIRV, mlir_c), MlirPass, ())
end

# no prototype is found for this function at Passes.capi.h.inc:36:25, please use with caution
function mlirRegisterConversionConvertArithmeticToSPIRV()
    ccall((:mlirRegisterConversionConvertArithmeticToSPIRV, mlir_c), Cvoid, ())
end

# no prototype is found for this function at Passes.capi.h.inc:40:29, please use with caution
function mlirCreateConversionConvertArmNeon2dToIntr()
    ccall((:mlirCreateConversionConvertArmNeon2dToIntr, mlir_c), MlirPass, ())
end

# no prototype is found for this function at Passes.capi.h.inc:41:25, please use with caution
function mlirRegisterConversionConvertArmNeon2dToIntr()
    ccall((:mlirRegisterConversionConvertArmNeon2dToIntr, mlir_c), Cvoid, ())
end

# no prototype is found for this function at Passes.capi.h.inc:45:29, please use with caution
function mlirCreateConversionConvertAsyncToLLVM()
    ccall((:mlirCreateConversionConvertAsyncToLLVM, mlir_c), MlirPass, ())
end

# no prototype is found for this function at Passes.capi.h.inc:46:25, please use with caution
function mlirRegisterConversionConvertAsyncToLLVM()
    ccall((:mlirRegisterConversionConvertAsyncToLLVM, mlir_c), Cvoid, ())
end

# no prototype is found for this function at Passes.capi.h.inc:50:29, please use with caution
function mlirCreateConversionConvertBufferizationToMemRef()
    ccall((:mlirCreateConversionConvertBufferizationToMemRef, mlir_c), MlirPass, ())
end

# no prototype is found for this function at Passes.capi.h.inc:51:25, please use with caution
function mlirRegisterConversionConvertBufferizationToMemRef()
    ccall((:mlirRegisterConversionConvertBufferizationToMemRef, mlir_c), Cvoid, ())
end

# no prototype is found for this function at Passes.capi.h.inc:55:29, please use with caution
function mlirCreateConversionConvertComplexToLLVM()
    ccall((:mlirCreateConversionConvertComplexToLLVM, mlir_c), MlirPass, ())
end

# no prototype is found for this function at Passes.capi.h.inc:56:25, please use with caution
function mlirRegisterConversionConvertComplexToLLVM()
    ccall((:mlirRegisterConversionConvertComplexToLLVM, mlir_c), Cvoid, ())
end

# no prototype is found for this function at Passes.capi.h.inc:60:29, please use with caution
function mlirCreateConversionConvertComplexToLibm()
    ccall((:mlirCreateConversionConvertComplexToLibm, mlir_c), MlirPass, ())
end

# no prototype is found for this function at Passes.capi.h.inc:61:25, please use with caution
function mlirRegisterConversionConvertComplexToLibm()
    ccall((:mlirRegisterConversionConvertComplexToLibm, mlir_c), Cvoid, ())
end

# no prototype is found for this function at Passes.capi.h.inc:65:29, please use with caution
function mlirCreateConversionConvertComplexToStandard()
    ccall((:mlirCreateConversionConvertComplexToStandard, mlir_c), MlirPass, ())
end

# no prototype is found for this function at Passes.capi.h.inc:66:25, please use with caution
function mlirRegisterConversionConvertComplexToStandard()
    ccall((:mlirRegisterConversionConvertComplexToStandard, mlir_c), Cvoid, ())
end

# no prototype is found for this function at Passes.capi.h.inc:70:29, please use with caution
function mlirCreateConversionConvertControlFlowToLLVM()
    ccall((:mlirCreateConversionConvertControlFlowToLLVM, mlir_c), MlirPass, ())
end

# no prototype is found for this function at Passes.capi.h.inc:71:25, please use with caution
function mlirRegisterConversionConvertControlFlowToLLVM()
    ccall((:mlirRegisterConversionConvertControlFlowToLLVM, mlir_c), Cvoid, ())
end

# no prototype is found for this function at Passes.capi.h.inc:75:29, please use with caution
function mlirCreateConversionConvertControlFlowToSPIRV()
    ccall((:mlirCreateConversionConvertControlFlowToSPIRV, mlir_c), MlirPass, ())
end

# no prototype is found for this function at Passes.capi.h.inc:76:25, please use with caution
function mlirRegisterConversionConvertControlFlowToSPIRV()
    ccall((:mlirRegisterConversionConvertControlFlowToSPIRV, mlir_c), Cvoid, ())
end

# no prototype is found for this function at Passes.capi.h.inc:80:29, please use with caution
function mlirCreateConversionConvertFuncToLLVM()
    ccall((:mlirCreateConversionConvertFuncToLLVM, mlir_c), MlirPass, ())
end

# no prototype is found for this function at Passes.capi.h.inc:81:25, please use with caution
function mlirRegisterConversionConvertFuncToLLVM()
    ccall((:mlirRegisterConversionConvertFuncToLLVM, mlir_c), Cvoid, ())
end

# no prototype is found for this function at Passes.capi.h.inc:85:29, please use with caution
function mlirCreateConversionConvertFuncToSPIRV()
    ccall((:mlirCreateConversionConvertFuncToSPIRV, mlir_c), MlirPass, ())
end

# no prototype is found for this function at Passes.capi.h.inc:86:25, please use with caution
function mlirRegisterConversionConvertFuncToSPIRV()
    ccall((:mlirRegisterConversionConvertFuncToSPIRV, mlir_c), Cvoid, ())
end

# no prototype is found for this function at Passes.capi.h.inc:90:29, please use with caution
function mlirCreateConversionConvertGPUToSPIRV()
    ccall((:mlirCreateConversionConvertGPUToSPIRV, mlir_c), MlirPass, ())
end

# no prototype is found for this function at Passes.capi.h.inc:91:25, please use with caution
function mlirRegisterConversionConvertGPUToSPIRV()
    ccall((:mlirRegisterConversionConvertGPUToSPIRV, mlir_c), Cvoid, ())
end

# no prototype is found for this function at Passes.capi.h.inc:95:29, please use with caution
function mlirCreateConversionConvertGpuLaunchFuncToVulkanLaunchFunc()
    ccall((:mlirCreateConversionConvertGpuLaunchFuncToVulkanLaunchFunc, mlir_c), MlirPass, ())
end

# no prototype is found for this function at Passes.capi.h.inc:96:25, please use with caution
function mlirRegisterConversionConvertGpuLaunchFuncToVulkanLaunchFunc()
    ccall((:mlirRegisterConversionConvertGpuLaunchFuncToVulkanLaunchFunc, mlir_c), Cvoid, ())
end

# no prototype is found for this function at Passes.capi.h.inc:100:29, please use with caution
function mlirCreateConversionConvertGpuOpsToNVVMOps()
    ccall((:mlirCreateConversionConvertGpuOpsToNVVMOps, mlir_c), MlirPass, ())
end

# no prototype is found for this function at Passes.capi.h.inc:101:25, please use with caution
function mlirRegisterConversionConvertGpuOpsToNVVMOps()
    ccall((:mlirRegisterConversionConvertGpuOpsToNVVMOps, mlir_c), Cvoid, ())
end

# no prototype is found for this function at Passes.capi.h.inc:105:29, please use with caution
function mlirCreateConversionConvertGpuOpsToROCDLOps()
    ccall((:mlirCreateConversionConvertGpuOpsToROCDLOps, mlir_c), MlirPass, ())
end

# no prototype is found for this function at Passes.capi.h.inc:106:25, please use with caution
function mlirRegisterConversionConvertGpuOpsToROCDLOps()
    ccall((:mlirRegisterConversionConvertGpuOpsToROCDLOps, mlir_c), Cvoid, ())
end

# no prototype is found for this function at Passes.capi.h.inc:110:29, please use with caution
function mlirCreateConversionConvertLinalgToLLVM()
    ccall((:mlirCreateConversionConvertLinalgToLLVM, mlir_c), MlirPass, ())
end

# no prototype is found for this function at Passes.capi.h.inc:111:25, please use with caution
function mlirRegisterConversionConvertLinalgToLLVM()
    ccall((:mlirRegisterConversionConvertLinalgToLLVM, mlir_c), Cvoid, ())
end

# no prototype is found for this function at Passes.capi.h.inc:115:29, please use with caution
function mlirCreateConversionConvertLinalgToSPIRV()
    ccall((:mlirCreateConversionConvertLinalgToSPIRV, mlir_c), MlirPass, ())
end

# no prototype is found for this function at Passes.capi.h.inc:116:25, please use with caution
function mlirRegisterConversionConvertLinalgToSPIRV()
    ccall((:mlirRegisterConversionConvertLinalgToSPIRV, mlir_c), Cvoid, ())
end

# no prototype is found for this function at Passes.capi.h.inc:120:29, please use with caution
function mlirCreateConversionConvertLinalgToStandard()
    ccall((:mlirCreateConversionConvertLinalgToStandard, mlir_c), MlirPass, ())
end

# no prototype is found for this function at Passes.capi.h.inc:121:25, please use with caution
function mlirRegisterConversionConvertLinalgToStandard()
    ccall((:mlirRegisterConversionConvertLinalgToStandard, mlir_c), Cvoid, ())
end

# no prototype is found for this function at Passes.capi.h.inc:125:29, please use with caution
function mlirCreateConversionConvertMathToLLVM()
    ccall((:mlirCreateConversionConvertMathToLLVM, mlir_c), MlirPass, ())
end

# no prototype is found for this function at Passes.capi.h.inc:126:25, please use with caution
function mlirRegisterConversionConvertMathToLLVM()
    ccall((:mlirRegisterConversionConvertMathToLLVM, mlir_c), Cvoid, ())
end

# no prototype is found for this function at Passes.capi.h.inc:130:29, please use with caution
function mlirCreateConversionConvertMathToLibm()
    ccall((:mlirCreateConversionConvertMathToLibm, mlir_c), MlirPass, ())
end

# no prototype is found for this function at Passes.capi.h.inc:131:25, please use with caution
function mlirRegisterConversionConvertMathToLibm()
    ccall((:mlirRegisterConversionConvertMathToLibm, mlir_c), Cvoid, ())
end

# no prototype is found for this function at Passes.capi.h.inc:135:29, please use with caution
function mlirCreateConversionConvertMathToSPIRV()
    ccall((:mlirCreateConversionConvertMathToSPIRV, mlir_c), MlirPass, ())
end

# no prototype is found for this function at Passes.capi.h.inc:136:25, please use with caution
function mlirRegisterConversionConvertMathToSPIRV()
    ccall((:mlirRegisterConversionConvertMathToSPIRV, mlir_c), Cvoid, ())
end

# no prototype is found for this function at Passes.capi.h.inc:140:29, please use with caution
function mlirCreateConversionConvertMemRefToLLVM()
    ccall((:mlirCreateConversionConvertMemRefToLLVM, mlir_c), MlirPass, ())
end

# no prototype is found for this function at Passes.capi.h.inc:141:25, please use with caution
function mlirRegisterConversionConvertMemRefToLLVM()
    ccall((:mlirRegisterConversionConvertMemRefToLLVM, mlir_c), Cvoid, ())
end

# no prototype is found for this function at Passes.capi.h.inc:145:29, please use with caution
function mlirCreateConversionConvertMemRefToSPIRV()
    ccall((:mlirCreateConversionConvertMemRefToSPIRV, mlir_c), MlirPass, ())
end

# no prototype is found for this function at Passes.capi.h.inc:146:25, please use with caution
function mlirRegisterConversionConvertMemRefToSPIRV()
    ccall((:mlirRegisterConversionConvertMemRefToSPIRV, mlir_c), Cvoid, ())
end

# no prototype is found for this function at Passes.capi.h.inc:150:29, please use with caution
function mlirCreateConversionConvertNVGPUToNVVM()
    ccall((:mlirCreateConversionConvertNVGPUToNVVM, mlir_c), MlirPass, ())
end

# no prototype is found for this function at Passes.capi.h.inc:151:25, please use with caution
function mlirRegisterConversionConvertNVGPUToNVVM()
    ccall((:mlirRegisterConversionConvertNVGPUToNVVM, mlir_c), Cvoid, ())
end

# no prototype is found for this function at Passes.capi.h.inc:155:29, please use with caution
function mlirCreateConversionConvertOpenACCToLLVM()
    ccall((:mlirCreateConversionConvertOpenACCToLLVM, mlir_c), MlirPass, ())
end

# no prototype is found for this function at Passes.capi.h.inc:156:25, please use with caution
function mlirRegisterConversionConvertOpenACCToLLVM()
    ccall((:mlirRegisterConversionConvertOpenACCToLLVM, mlir_c), Cvoid, ())
end

# no prototype is found for this function at Passes.capi.h.inc:160:29, please use with caution
function mlirCreateConversionConvertOpenACCToSCF()
    ccall((:mlirCreateConversionConvertOpenACCToSCF, mlir_c), MlirPass, ())
end

# no prototype is found for this function at Passes.capi.h.inc:161:25, please use with caution
function mlirRegisterConversionConvertOpenACCToSCF()
    ccall((:mlirRegisterConversionConvertOpenACCToSCF, mlir_c), Cvoid, ())
end

# no prototype is found for this function at Passes.capi.h.inc:165:29, please use with caution
function mlirCreateConversionConvertOpenMPToLLVM()
    ccall((:mlirCreateConversionConvertOpenMPToLLVM, mlir_c), MlirPass, ())
end

# no prototype is found for this function at Passes.capi.h.inc:166:25, please use with caution
function mlirRegisterConversionConvertOpenMPToLLVM()
    ccall((:mlirRegisterConversionConvertOpenMPToLLVM, mlir_c), Cvoid, ())
end

# no prototype is found for this function at Passes.capi.h.inc:170:29, please use with caution
function mlirCreateConversionConvertPDLToPDLInterp()
    ccall((:mlirCreateConversionConvertPDLToPDLInterp, mlir_c), MlirPass, ())
end

# no prototype is found for this function at Passes.capi.h.inc:171:25, please use with caution
function mlirRegisterConversionConvertPDLToPDLInterp()
    ccall((:mlirRegisterConversionConvertPDLToPDLInterp, mlir_c), Cvoid, ())
end

# no prototype is found for this function at Passes.capi.h.inc:175:29, please use with caution
function mlirCreateConversionConvertParallelLoopToGpu()
    ccall((:mlirCreateConversionConvertParallelLoopToGpu, mlir_c), MlirPass, ())
end

# no prototype is found for this function at Passes.capi.h.inc:176:25, please use with caution
function mlirRegisterConversionConvertParallelLoopToGpu()
    ccall((:mlirRegisterConversionConvertParallelLoopToGpu, mlir_c), Cvoid, ())
end

# no prototype is found for this function at Passes.capi.h.inc:180:29, please use with caution
function mlirCreateConversionConvertSCFToOpenMP()
    ccall((:mlirCreateConversionConvertSCFToOpenMP, mlir_c), MlirPass, ())
end

# no prototype is found for this function at Passes.capi.h.inc:181:25, please use with caution
function mlirRegisterConversionConvertSCFToOpenMP()
    ccall((:mlirRegisterConversionConvertSCFToOpenMP, mlir_c), Cvoid, ())
end

# no prototype is found for this function at Passes.capi.h.inc:185:29, please use with caution
function mlirCreateConversionConvertSPIRVToLLVM()
    ccall((:mlirCreateConversionConvertSPIRVToLLVM, mlir_c), MlirPass, ())
end

# no prototype is found for this function at Passes.capi.h.inc:186:25, please use with caution
function mlirRegisterConversionConvertSPIRVToLLVM()
    ccall((:mlirRegisterConversionConvertSPIRVToLLVM, mlir_c), Cvoid, ())
end

# no prototype is found for this function at Passes.capi.h.inc:190:29, please use with caution
function mlirCreateConversionConvertShapeConstraints()
    ccall((:mlirCreateConversionConvertShapeConstraints, mlir_c), MlirPass, ())
end

# no prototype is found for this function at Passes.capi.h.inc:191:25, please use with caution
function mlirRegisterConversionConvertShapeConstraints()
    ccall((:mlirRegisterConversionConvertShapeConstraints, mlir_c), Cvoid, ())
end

# no prototype is found for this function at Passes.capi.h.inc:195:29, please use with caution
function mlirCreateConversionConvertShapeToStandard()
    ccall((:mlirCreateConversionConvertShapeToStandard, mlir_c), MlirPass, ())
end

# no prototype is found for this function at Passes.capi.h.inc:196:25, please use with caution
function mlirRegisterConversionConvertShapeToStandard()
    ccall((:mlirRegisterConversionConvertShapeToStandard, mlir_c), Cvoid, ())
end

# no prototype is found for this function at Passes.capi.h.inc:200:29, please use with caution
function mlirCreateConversionConvertTensorToLinalg()
    ccall((:mlirCreateConversionConvertTensorToLinalg, mlir_c), MlirPass, ())
end

# no prototype is found for this function at Passes.capi.h.inc:201:25, please use with caution
function mlirRegisterConversionConvertTensorToLinalg()
    ccall((:mlirRegisterConversionConvertTensorToLinalg, mlir_c), Cvoid, ())
end

# no prototype is found for this function at Passes.capi.h.inc:205:29, please use with caution
function mlirCreateConversionConvertTensorToSPIRV()
    ccall((:mlirCreateConversionConvertTensorToSPIRV, mlir_c), MlirPass, ())
end

# no prototype is found for this function at Passes.capi.h.inc:206:25, please use with caution
function mlirRegisterConversionConvertTensorToSPIRV()
    ccall((:mlirRegisterConversionConvertTensorToSPIRV, mlir_c), Cvoid, ())
end

# no prototype is found for this function at Passes.capi.h.inc:210:29, please use with caution
function mlirCreateConversionConvertVectorToGPU()
    ccall((:mlirCreateConversionConvertVectorToGPU, mlir_c), MlirPass, ())
end

# no prototype is found for this function at Passes.capi.h.inc:211:25, please use with caution
function mlirRegisterConversionConvertVectorToGPU()
    ccall((:mlirRegisterConversionConvertVectorToGPU, mlir_c), Cvoid, ())
end

# no prototype is found for this function at Passes.capi.h.inc:215:29, please use with caution
function mlirCreateConversionConvertVectorToLLVM()
    ccall((:mlirCreateConversionConvertVectorToLLVM, mlir_c), MlirPass, ())
end

# no prototype is found for this function at Passes.capi.h.inc:216:25, please use with caution
function mlirRegisterConversionConvertVectorToLLVM()
    ccall((:mlirRegisterConversionConvertVectorToLLVM, mlir_c), Cvoid, ())
end

# no prototype is found for this function at Passes.capi.h.inc:220:29, please use with caution
function mlirCreateConversionConvertVectorToSCF()
    ccall((:mlirCreateConversionConvertVectorToSCF, mlir_c), MlirPass, ())
end

# no prototype is found for this function at Passes.capi.h.inc:221:25, please use with caution
function mlirRegisterConversionConvertVectorToSCF()
    ccall((:mlirRegisterConversionConvertVectorToSCF, mlir_c), Cvoid, ())
end

# no prototype is found for this function at Passes.capi.h.inc:225:29, please use with caution
function mlirCreateConversionConvertVectorToSPIRV()
    ccall((:mlirCreateConversionConvertVectorToSPIRV, mlir_c), MlirPass, ())
end

# no prototype is found for this function at Passes.capi.h.inc:226:25, please use with caution
function mlirRegisterConversionConvertVectorToSPIRV()
    ccall((:mlirRegisterConversionConvertVectorToSPIRV, mlir_c), Cvoid, ())
end

# no prototype is found for this function at Passes.capi.h.inc:230:29, please use with caution
function mlirCreateConversionConvertVulkanLaunchFuncToVulkanCalls()
    ccall((:mlirCreateConversionConvertVulkanLaunchFuncToVulkanCalls, mlir_c), MlirPass, ())
end

# no prototype is found for this function at Passes.capi.h.inc:231:25, please use with caution
function mlirRegisterConversionConvertVulkanLaunchFuncToVulkanCalls()
    ccall((:mlirRegisterConversionConvertVulkanLaunchFuncToVulkanCalls, mlir_c), Cvoid, ())
end

# no prototype is found for this function at Passes.capi.h.inc:235:29, please use with caution
function mlirCreateConversionGpuToLLVMConversionPass()
    ccall((:mlirCreateConversionGpuToLLVMConversionPass, mlir_c), MlirPass, ())
end

# no prototype is found for this function at Passes.capi.h.inc:236:25, please use with caution
function mlirRegisterConversionGpuToLLVMConversionPass()
    ccall((:mlirRegisterConversionGpuToLLVMConversionPass, mlir_c), Cvoid, ())
end

# no prototype is found for this function at Passes.capi.h.inc:240:29, please use with caution
function mlirCreateConversionLowerHostCodeToLLVM()
    ccall((:mlirCreateConversionLowerHostCodeToLLVM, mlir_c), MlirPass, ())
end

# no prototype is found for this function at Passes.capi.h.inc:241:25, please use with caution
function mlirRegisterConversionLowerHostCodeToLLVM()
    ccall((:mlirRegisterConversionLowerHostCodeToLLVM, mlir_c), Cvoid, ())
end

# no prototype is found for this function at Passes.capi.h.inc:245:29, please use with caution
function mlirCreateConversionReconcileUnrealizedCasts()
    ccall((:mlirCreateConversionReconcileUnrealizedCasts, mlir_c), MlirPass, ())
end

# no prototype is found for this function at Passes.capi.h.inc:246:25, please use with caution
function mlirRegisterConversionReconcileUnrealizedCasts()
    ccall((:mlirRegisterConversionReconcileUnrealizedCasts, mlir_c), Cvoid, ())
end

# no prototype is found for this function at Passes.capi.h.inc:250:29, please use with caution
function mlirCreateConversionSCFToControlFlow()
    ccall((:mlirCreateConversionSCFToControlFlow, mlir_c), MlirPass, ())
end

# no prototype is found for this function at Passes.capi.h.inc:251:25, please use with caution
function mlirRegisterConversionSCFToControlFlow()
    ccall((:mlirRegisterConversionSCFToControlFlow, mlir_c), Cvoid, ())
end

# no prototype is found for this function at Passes.capi.h.inc:255:29, please use with caution
function mlirCreateConversionSCFToSPIRV()
    ccall((:mlirCreateConversionSCFToSPIRV, mlir_c), MlirPass, ())
end

# no prototype is found for this function at Passes.capi.h.inc:256:25, please use with caution
function mlirRegisterConversionSCFToSPIRV()
    ccall((:mlirRegisterConversionSCFToSPIRV, mlir_c), Cvoid, ())
end

# no prototype is found for this function at Passes.capi.h.inc:260:29, please use with caution
function mlirCreateConversionTosaToArith()
    ccall((:mlirCreateConversionTosaToArith, mlir_c), MlirPass, ())
end

# no prototype is found for this function at Passes.capi.h.inc:261:25, please use with caution
function mlirRegisterConversionTosaToArith()
    ccall((:mlirRegisterConversionTosaToArith, mlir_c), Cvoid, ())
end

# no prototype is found for this function at Passes.capi.h.inc:265:29, please use with caution
function mlirCreateConversionTosaToLinalg()
    ccall((:mlirCreateConversionTosaToLinalg, mlir_c), MlirPass, ())
end

# no prototype is found for this function at Passes.capi.h.inc:266:25, please use with caution
function mlirRegisterConversionTosaToLinalg()
    ccall((:mlirRegisterConversionTosaToLinalg, mlir_c), Cvoid, ())
end

# no prototype is found for this function at Passes.capi.h.inc:270:29, please use with caution
function mlirCreateConversionTosaToLinalgNamed()
    ccall((:mlirCreateConversionTosaToLinalgNamed, mlir_c), MlirPass, ())
end

# no prototype is found for this function at Passes.capi.h.inc:271:25, please use with caution
function mlirRegisterConversionTosaToLinalgNamed()
    ccall((:mlirRegisterConversionTosaToLinalgNamed, mlir_c), Cvoid, ())
end

# no prototype is found for this function at Passes.capi.h.inc:275:29, please use with caution
function mlirCreateConversionTosaToSCF()
    ccall((:mlirCreateConversionTosaToSCF, mlir_c), MlirPass, ())
end

# no prototype is found for this function at Passes.capi.h.inc:276:25, please use with caution
function mlirRegisterConversionTosaToSCF()
    ccall((:mlirRegisterConversionTosaToSCF, mlir_c), Cvoid, ())
end

# no prototype is found for this function at Passes.capi.h.inc:280:29, please use with caution
function mlirCreateConversionTosaToTensor()
    ccall((:mlirCreateConversionTosaToTensor, mlir_c), MlirPass, ())
end

# no prototype is found for this function at Passes.capi.h.inc:281:25, please use with caution
function mlirRegisterConversionTosaToTensor()
    ccall((:mlirRegisterConversionTosaToTensor, mlir_c), Cvoid, ())
end

function mlirEnableGlobalDebug(enable)
    ccall((:mlirEnableGlobalDebug, mlir_c), Cvoid, (Bool,), enable)
end

# no prototype is found for this function at Debug.h:22:25, please use with caution
function mlirIsGlobalDebugEnabled()
    ccall((:mlirIsGlobalDebugEnabled, mlir_c), Bool, ())
end

struct MlirDiagnostic
    ptr::Ptr{Cvoid}
end

@cenum MlirDiagnosticSeverity::UInt32 begin
    MlirDiagnosticError = 0
    MlirDiagnosticWarning = 1
    MlirDiagnosticNote = 2
    MlirDiagnosticRemark = 3
end

const MlirDiagnosticHandlerID = UInt64

# typedef MlirLogicalResult ( * MlirDiagnosticHandler ) ( MlirDiagnostic , void * userData )
const MlirDiagnosticHandler = Ptr{Cvoid}

function mlirDiagnosticPrint(diagnostic, callback, userData)
    ccall((:mlirDiagnosticPrint, mlir_c), Cvoid, (MlirDiagnostic, MlirStringCallback, Ptr{Cvoid}), diagnostic, callback, userData)
end

function mlirDiagnosticGetLocation(diagnostic)
    ccall((:mlirDiagnosticGetLocation, mlir_c), MlirLocation, (MlirDiagnostic,), diagnostic)
end

function mlirDiagnosticGetSeverity(diagnostic)
    ccall((:mlirDiagnosticGetSeverity, mlir_c), MlirDiagnosticSeverity, (MlirDiagnostic,), diagnostic)
end

function mlirDiagnosticGetNumNotes(diagnostic)
    ccall((:mlirDiagnosticGetNumNotes, mlir_c), intptr_t, (MlirDiagnostic,), diagnostic)
end

function mlirDiagnosticGetNote(diagnostic, pos)
    ccall((:mlirDiagnosticGetNote, mlir_c), MlirDiagnostic, (MlirDiagnostic, intptr_t), diagnostic, pos)
end

function mlirContextAttachDiagnosticHandler(context, handler, userData, deleteUserData)
    ccall((:mlirContextAttachDiagnosticHandler, mlir_c), MlirDiagnosticHandlerID, (MlirContext, MlirDiagnosticHandler, Ptr{Cvoid}, Ptr{Cvoid}), context, handler, userData, deleteUserData)
end

function mlirContextDetachDiagnosticHandler(context, id)
    ccall((:mlirContextDetachDiagnosticHandler, mlir_c), Cvoid, (MlirContext, MlirDiagnosticHandlerID), context, id)
end

function mlirEmitError(location, message)
    ccall((:mlirEmitError, mlir_c), Cvoid, (MlirLocation, Cstring), location, message)
end

# no prototype is found for this function at Async.h:20:1, please use with caution
function mlirGetDialectHandle__async__()
    ccall((:mlirGetDialectHandle__async__, mlir_c), MlirDialectHandle, ())
end

# no prototype is found for this function at Passes.capi.h.inc:11:25, please use with caution
function mlirRegisterAsyncPasses()
    ccall((:mlirRegisterAsyncPasses, mlir_c), Cvoid, ())
end

# no prototype is found for this function at Passes.capi.h.inc:15:29, please use with caution
function mlirCreateAsyncAsyncParallelFor()
    ccall((:mlirCreateAsyncAsyncParallelFor, mlir_c), MlirPass, ())
end

# no prototype is found for this function at Passes.capi.h.inc:16:25, please use with caution
function mlirRegisterAsyncAsyncParallelFor()
    ccall((:mlirRegisterAsyncAsyncParallelFor, mlir_c), Cvoid, ())
end

# no prototype is found for this function at Passes.capi.h.inc:20:29, please use with caution
function mlirCreateAsyncAsyncRuntimePolicyBasedRefCounting()
    ccall((:mlirCreateAsyncAsyncRuntimePolicyBasedRefCounting, mlir_c), MlirPass, ())
end

# no prototype is found for this function at Passes.capi.h.inc:21:25, please use with caution
function mlirRegisterAsyncAsyncRuntimePolicyBasedRefCounting()
    ccall((:mlirRegisterAsyncAsyncRuntimePolicyBasedRefCounting, mlir_c), Cvoid, ())
end

# no prototype is found for this function at Passes.capi.h.inc:25:29, please use with caution
function mlirCreateAsyncAsyncRuntimeRefCounting()
    ccall((:mlirCreateAsyncAsyncRuntimeRefCounting, mlir_c), MlirPass, ())
end

# no prototype is found for this function at Passes.capi.h.inc:26:25, please use with caution
function mlirRegisterAsyncAsyncRuntimeRefCounting()
    ccall((:mlirRegisterAsyncAsyncRuntimeRefCounting, mlir_c), Cvoid, ())
end

# no prototype is found for this function at Passes.capi.h.inc:30:29, please use with caution
function mlirCreateAsyncAsyncRuntimeRefCountingOpt()
    ccall((:mlirCreateAsyncAsyncRuntimeRefCountingOpt, mlir_c), MlirPass, ())
end

# no prototype is found for this function at Passes.capi.h.inc:31:25, please use with caution
function mlirRegisterAsyncAsyncRuntimeRefCountingOpt()
    ccall((:mlirRegisterAsyncAsyncRuntimeRefCountingOpt, mlir_c), Cvoid, ())
end

# no prototype is found for this function at Passes.capi.h.inc:35:29, please use with caution
function mlirCreateAsyncAsyncToAsyncRuntime()
    ccall((:mlirCreateAsyncAsyncToAsyncRuntime, mlir_c), MlirPass, ())
end

# no prototype is found for this function at Passes.capi.h.inc:36:25, please use with caution
function mlirRegisterAsyncAsyncToAsyncRuntime()
    ccall((:mlirRegisterAsyncAsyncToAsyncRuntime, mlir_c), Cvoid, ())
end

# no prototype is found for this function at ControlFlow.h:19:1, please use with caution
function mlirGetDialectHandle__cf__()
    ccall((:mlirGetDialectHandle__cf__, mlir_c), MlirDialectHandle, ())
end

# no prototype is found for this function at Func.h:27:1, please use with caution
function mlirGetDialectHandle__func__()
    ccall((:mlirGetDialectHandle__func__, mlir_c), MlirDialectHandle, ())
end

# no prototype is found for this function at GPU.h:20:1, please use with caution
function mlirGetDialectHandle__gpu__()
    ccall((:mlirGetDialectHandle__gpu__, mlir_c), MlirDialectHandle, ())
end

# no prototype is found for this function at Passes.capi.h.inc:11:25, please use with caution
function mlirRegisterGPUPasses()
    ccall((:mlirRegisterGPUPasses, mlir_c), Cvoid, ())
end

# no prototype is found for this function at Passes.capi.h.inc:15:29, please use with caution
function mlirCreateGPUGpuAsyncRegionPass()
    ccall((:mlirCreateGPUGpuAsyncRegionPass, mlir_c), MlirPass, ())
end

# no prototype is found for this function at Passes.capi.h.inc:16:25, please use with caution
function mlirRegisterGPUGpuAsyncRegionPass()
    ccall((:mlirRegisterGPUGpuAsyncRegionPass, mlir_c), Cvoid, ())
end

# no prototype is found for this function at Passes.capi.h.inc:20:29, please use with caution
function mlirCreateGPUGpuKernelOutlining()
    ccall((:mlirCreateGPUGpuKernelOutlining, mlir_c), MlirPass, ())
end

# no prototype is found for this function at Passes.capi.h.inc:21:25, please use with caution
function mlirRegisterGPUGpuKernelOutlining()
    ccall((:mlirRegisterGPUGpuKernelOutlining, mlir_c), Cvoid, ())
end

# no prototype is found for this function at Passes.capi.h.inc:25:29, please use with caution
function mlirCreateGPUGpuLaunchSinkIndexComputations()
    ccall((:mlirCreateGPUGpuLaunchSinkIndexComputations, mlir_c), MlirPass, ())
end

# no prototype is found for this function at Passes.capi.h.inc:26:25, please use with caution
function mlirRegisterGPUGpuLaunchSinkIndexComputations()
    ccall((:mlirRegisterGPUGpuLaunchSinkIndexComputations, mlir_c), Cvoid, ())
end

# no prototype is found for this function at Passes.capi.h.inc:30:29, please use with caution
function mlirCreateGPUGpuMapParallelLoopsPass()
    ccall((:mlirCreateGPUGpuMapParallelLoopsPass, mlir_c), MlirPass, ())
end

# no prototype is found for this function at Passes.capi.h.inc:31:25, please use with caution
function mlirRegisterGPUGpuMapParallelLoopsPass()
    ccall((:mlirRegisterGPUGpuMapParallelLoopsPass, mlir_c), Cvoid, ())
end

# no prototype is found for this function at LLVM.h:19:1, please use with caution
function mlirGetDialectHandle__llvm__()
    ccall((:mlirGetDialectHandle__llvm__, mlir_c), MlirDialectHandle, ())
end

function mlirLLVMPointerTypeGet(pointee, addressSpace)
    ccall((:mlirLLVMPointerTypeGet, mlir_c), MlirType, (MlirType, Cuint), pointee, addressSpace)
end

function mlirLLVMVoidTypeGet(ctx)
    ccall((:mlirLLVMVoidTypeGet, mlir_c), MlirType, (MlirContext,), ctx)
end

function mlirLLVMArrayTypeGet(elementType, numElements)
    ccall((:mlirLLVMArrayTypeGet, mlir_c), MlirType, (MlirType, Cuint), elementType, numElements)
end

function mlirLLVMFunctionTypeGet(resultType, nArgumentTypes, argumentTypes, isVarArg)
    ccall((:mlirLLVMFunctionTypeGet, mlir_c), MlirType, (MlirType, intptr_t, Ptr{MlirType}, Bool), resultType, nArgumentTypes, argumentTypes, isVarArg)
end

function mlirLLVMStructTypeLiteralGet(ctx, nFieldTypes, fieldTypes, isPacked)
    ccall((:mlirLLVMStructTypeLiteralGet, mlir_c), MlirType, (MlirContext, intptr_t, Ptr{MlirType}, Bool), ctx, nFieldTypes, fieldTypes, isPacked)
end

function mlirLinalgFillBuiltinNamedOpRegion(mlirOp)
    ccall((:mlirLinalgFillBuiltinNamedOpRegion, mlir_c), Cvoid, (MlirOperation,), mlirOp)
end

# no prototype is found for this function at Linalg.h:25:1, please use with caution
function mlirGetDialectHandle__linalg__()
    ccall((:mlirGetDialectHandle__linalg__, mlir_c), MlirDialectHandle, ())
end

# no prototype is found for this function at Passes.capi.h.inc:11:25, please use with caution
function mlirRegisterLinalgPasses()
    ccall((:mlirRegisterLinalgPasses, mlir_c), Cvoid, ())
end

# no prototype is found for this function at Passes.capi.h.inc:15:29, please use with caution
function mlirCreateLinalgConvertElementwiseToLinalg()
    ccall((:mlirCreateLinalgConvertElementwiseToLinalg, mlir_c), MlirPass, ())
end

# no prototype is found for this function at Passes.capi.h.inc:16:25, please use with caution
function mlirRegisterLinalgConvertElementwiseToLinalg()
    ccall((:mlirRegisterLinalgConvertElementwiseToLinalg, mlir_c), Cvoid, ())
end

# no prototype is found for this function at Passes.capi.h.inc:20:29, please use with caution
function mlirCreateLinalgLinalgBufferize()
    ccall((:mlirCreateLinalgLinalgBufferize, mlir_c), MlirPass, ())
end

# no prototype is found for this function at Passes.capi.h.inc:21:25, please use with caution
function mlirRegisterLinalgLinalgBufferize()
    ccall((:mlirRegisterLinalgLinalgBufferize, mlir_c), Cvoid, ())
end

# no prototype is found for this function at Passes.capi.h.inc:25:29, please use with caution
function mlirCreateLinalgLinalgDetensorize()
    ccall((:mlirCreateLinalgLinalgDetensorize, mlir_c), MlirPass, ())
end

# no prototype is found for this function at Passes.capi.h.inc:26:25, please use with caution
function mlirRegisterLinalgLinalgDetensorize()
    ccall((:mlirRegisterLinalgLinalgDetensorize, mlir_c), Cvoid, ())
end

# no prototype is found for this function at Passes.capi.h.inc:30:29, please use with caution
function mlirCreateLinalgLinalgElementwiseOpFusion()
    ccall((:mlirCreateLinalgLinalgElementwiseOpFusion, mlir_c), MlirPass, ())
end

# no prototype is found for this function at Passes.capi.h.inc:31:25, please use with caution
function mlirRegisterLinalgLinalgElementwiseOpFusion()
    ccall((:mlirRegisterLinalgLinalgElementwiseOpFusion, mlir_c), Cvoid, ())
end

# no prototype is found for this function at Passes.capi.h.inc:35:29, please use with caution
function mlirCreateLinalgLinalgFoldUnitExtentDims()
    ccall((:mlirCreateLinalgLinalgFoldUnitExtentDims, mlir_c), MlirPass, ())
end

# no prototype is found for this function at Passes.capi.h.inc:36:25, please use with caution
function mlirRegisterLinalgLinalgFoldUnitExtentDims()
    ccall((:mlirRegisterLinalgLinalgFoldUnitExtentDims, mlir_c), Cvoid, ())
end

# no prototype is found for this function at Passes.capi.h.inc:40:29, please use with caution
function mlirCreateLinalgLinalgGeneralization()
    ccall((:mlirCreateLinalgLinalgGeneralization, mlir_c), MlirPass, ())
end

# no prototype is found for this function at Passes.capi.h.inc:41:25, please use with caution
function mlirRegisterLinalgLinalgGeneralization()
    ccall((:mlirRegisterLinalgLinalgGeneralization, mlir_c), Cvoid, ())
end

# no prototype is found for this function at Passes.capi.h.inc:45:29, please use with caution
function mlirCreateLinalgLinalgInitTensorToAllocTensor()
    ccall((:mlirCreateLinalgLinalgInitTensorToAllocTensor, mlir_c), MlirPass, ())
end

# no prototype is found for this function at Passes.capi.h.inc:46:25, please use with caution
function mlirRegisterLinalgLinalgInitTensorToAllocTensor()
    ccall((:mlirRegisterLinalgLinalgInitTensorToAllocTensor, mlir_c), Cvoid, ())
end

# no prototype is found for this function at Passes.capi.h.inc:50:29, please use with caution
function mlirCreateLinalgLinalgInlineScalarOperands()
    ccall((:mlirCreateLinalgLinalgInlineScalarOperands, mlir_c), MlirPass, ())
end

# no prototype is found for this function at Passes.capi.h.inc:51:25, please use with caution
function mlirRegisterLinalgLinalgInlineScalarOperands()
    ccall((:mlirRegisterLinalgLinalgInlineScalarOperands, mlir_c), Cvoid, ())
end

# no prototype is found for this function at Passes.capi.h.inc:55:29, please use with caution
function mlirCreateLinalgLinalgLowerToAffineLoops()
    ccall((:mlirCreateLinalgLinalgLowerToAffineLoops, mlir_c), MlirPass, ())
end

# no prototype is found for this function at Passes.capi.h.inc:56:25, please use with caution
function mlirRegisterLinalgLinalgLowerToAffineLoops()
    ccall((:mlirRegisterLinalgLinalgLowerToAffineLoops, mlir_c), Cvoid, ())
end

# no prototype is found for this function at Passes.capi.h.inc:60:29, please use with caution
function mlirCreateLinalgLinalgLowerToLoops()
    ccall((:mlirCreateLinalgLinalgLowerToLoops, mlir_c), MlirPass, ())
end

# no prototype is found for this function at Passes.capi.h.inc:61:25, please use with caution
function mlirRegisterLinalgLinalgLowerToLoops()
    ccall((:mlirRegisterLinalgLinalgLowerToLoops, mlir_c), Cvoid, ())
end

# no prototype is found for this function at Passes.capi.h.inc:65:29, please use with caution
function mlirCreateLinalgLinalgLowerToParallelLoops()
    ccall((:mlirCreateLinalgLinalgLowerToParallelLoops, mlir_c), MlirPass, ())
end

# no prototype is found for this function at Passes.capi.h.inc:66:25, please use with caution
function mlirRegisterLinalgLinalgLowerToParallelLoops()
    ccall((:mlirRegisterLinalgLinalgLowerToParallelLoops, mlir_c), Cvoid, ())
end

# no prototype is found for this function at Passes.capi.h.inc:70:29, please use with caution
function mlirCreateLinalgLinalgNamedOpConversion()
    ccall((:mlirCreateLinalgLinalgNamedOpConversion, mlir_c), MlirPass, ())
end

# no prototype is found for this function at Passes.capi.h.inc:71:25, please use with caution
function mlirRegisterLinalgLinalgNamedOpConversion()
    ccall((:mlirRegisterLinalgLinalgNamedOpConversion, mlir_c), Cvoid, ())
end

# no prototype is found for this function at Passes.capi.h.inc:75:29, please use with caution
function mlirCreateLinalgLinalgStrategyDecomposePass()
    ccall((:mlirCreateLinalgLinalgStrategyDecomposePass, mlir_c), MlirPass, ())
end

# no prototype is found for this function at Passes.capi.h.inc:76:25, please use with caution
function mlirRegisterLinalgLinalgStrategyDecomposePass()
    ccall((:mlirRegisterLinalgLinalgStrategyDecomposePass, mlir_c), Cvoid, ())
end

# no prototype is found for this function at Passes.capi.h.inc:80:29, please use with caution
function mlirCreateLinalgLinalgStrategyEnablePass()
    ccall((:mlirCreateLinalgLinalgStrategyEnablePass, mlir_c), MlirPass, ())
end

# no prototype is found for this function at Passes.capi.h.inc:81:25, please use with caution
function mlirRegisterLinalgLinalgStrategyEnablePass()
    ccall((:mlirRegisterLinalgLinalgStrategyEnablePass, mlir_c), Cvoid, ())
end

# no prototype is found for this function at Passes.capi.h.inc:85:29, please use with caution
function mlirCreateLinalgLinalgStrategyGeneralizePass()
    ccall((:mlirCreateLinalgLinalgStrategyGeneralizePass, mlir_c), MlirPass, ())
end

# no prototype is found for this function at Passes.capi.h.inc:86:25, please use with caution
function mlirRegisterLinalgLinalgStrategyGeneralizePass()
    ccall((:mlirRegisterLinalgLinalgStrategyGeneralizePass, mlir_c), Cvoid, ())
end

# no prototype is found for this function at Passes.capi.h.inc:90:29, please use with caution
function mlirCreateLinalgLinalgStrategyInterchangePass()
    ccall((:mlirCreateLinalgLinalgStrategyInterchangePass, mlir_c), MlirPass, ())
end

# no prototype is found for this function at Passes.capi.h.inc:91:25, please use with caution
function mlirRegisterLinalgLinalgStrategyInterchangePass()
    ccall((:mlirRegisterLinalgLinalgStrategyInterchangePass, mlir_c), Cvoid, ())
end

# no prototype is found for this function at Passes.capi.h.inc:95:29, please use with caution
function mlirCreateLinalgLinalgStrategyLowerVectorsPass()
    ccall((:mlirCreateLinalgLinalgStrategyLowerVectorsPass, mlir_c), MlirPass, ())
end

# no prototype is found for this function at Passes.capi.h.inc:96:25, please use with caution
function mlirRegisterLinalgLinalgStrategyLowerVectorsPass()
    ccall((:mlirRegisterLinalgLinalgStrategyLowerVectorsPass, mlir_c), Cvoid, ())
end

# no prototype is found for this function at Passes.capi.h.inc:100:29, please use with caution
function mlirCreateLinalgLinalgStrategyPadPass()
    ccall((:mlirCreateLinalgLinalgStrategyPadPass, mlir_c), MlirPass, ())
end

# no prototype is found for this function at Passes.capi.h.inc:101:25, please use with caution
function mlirRegisterLinalgLinalgStrategyPadPass()
    ccall((:mlirRegisterLinalgLinalgStrategyPadPass, mlir_c), Cvoid, ())
end

# no prototype is found for this function at Passes.capi.h.inc:105:29, please use with caution
function mlirCreateLinalgLinalgStrategyPeelPass()
    ccall((:mlirCreateLinalgLinalgStrategyPeelPass, mlir_c), MlirPass, ())
end

# no prototype is found for this function at Passes.capi.h.inc:106:25, please use with caution
function mlirRegisterLinalgLinalgStrategyPeelPass()
    ccall((:mlirRegisterLinalgLinalgStrategyPeelPass, mlir_c), Cvoid, ())
end

# no prototype is found for this function at Passes.capi.h.inc:110:29, please use with caution
function mlirCreateLinalgLinalgStrategyRemoveMarkersPass()
    ccall((:mlirCreateLinalgLinalgStrategyRemoveMarkersPass, mlir_c), MlirPass, ())
end

# no prototype is found for this function at Passes.capi.h.inc:111:25, please use with caution
function mlirRegisterLinalgLinalgStrategyRemoveMarkersPass()
    ccall((:mlirRegisterLinalgLinalgStrategyRemoveMarkersPass, mlir_c), Cvoid, ())
end

# no prototype is found for this function at Passes.capi.h.inc:115:29, please use with caution
function mlirCreateLinalgLinalgStrategyTileAndFusePass()
    ccall((:mlirCreateLinalgLinalgStrategyTileAndFusePass, mlir_c), MlirPass, ())
end

# no prototype is found for this function at Passes.capi.h.inc:116:25, please use with caution
function mlirRegisterLinalgLinalgStrategyTileAndFusePass()
    ccall((:mlirRegisterLinalgLinalgStrategyTileAndFusePass, mlir_c), Cvoid, ())
end

# no prototype is found for this function at Passes.capi.h.inc:120:29, please use with caution
function mlirCreateLinalgLinalgStrategyTilePass()
    ccall((:mlirCreateLinalgLinalgStrategyTilePass, mlir_c), MlirPass, ())
end

# no prototype is found for this function at Passes.capi.h.inc:121:25, please use with caution
function mlirRegisterLinalgLinalgStrategyTilePass()
    ccall((:mlirRegisterLinalgLinalgStrategyTilePass, mlir_c), Cvoid, ())
end

# no prototype is found for this function at Passes.capi.h.inc:125:29, please use with caution
function mlirCreateLinalgLinalgStrategyVectorizePass()
    ccall((:mlirCreateLinalgLinalgStrategyVectorizePass, mlir_c), MlirPass, ())
end

# no prototype is found for this function at Passes.capi.h.inc:126:25, please use with caution
function mlirRegisterLinalgLinalgStrategyVectorizePass()
    ccall((:mlirRegisterLinalgLinalgStrategyVectorizePass, mlir_c), Cvoid, ())
end

# no prototype is found for this function at Passes.capi.h.inc:130:29, please use with caution
function mlirCreateLinalgLinalgTiling()
    ccall((:mlirCreateLinalgLinalgTiling, mlir_c), MlirPass, ())
end

# no prototype is found for this function at Passes.capi.h.inc:131:25, please use with caution
function mlirRegisterLinalgLinalgTiling()
    ccall((:mlirRegisterLinalgLinalgTiling, mlir_c), Cvoid, ())
end

# no prototype is found for this function at PDL.h:19:1, please use with caution
function mlirGetDialectHandle__pdl__()
    ccall((:mlirGetDialectHandle__pdl__, mlir_c), MlirDialectHandle, ())
end

function mlirTypeIsAPDLType(type)
    ccall((:mlirTypeIsAPDLType, mlir_c), Bool, (MlirType,), type)
end

function mlirTypeIsAPDLAttributeType(type)
    ccall((:mlirTypeIsAPDLAttributeType, mlir_c), Bool, (MlirType,), type)
end

function mlirPDLAttributeTypeGet(ctx)
    ccall((:mlirPDLAttributeTypeGet, mlir_c), MlirType, (MlirContext,), ctx)
end

function mlirTypeIsAPDLOperationType(type)
    ccall((:mlirTypeIsAPDLOperationType, mlir_c), Bool, (MlirType,), type)
end

function mlirPDLOperationTypeGet(ctx)
    ccall((:mlirPDLOperationTypeGet, mlir_c), MlirType, (MlirContext,), ctx)
end

function mlirTypeIsAPDLRangeType(type)
    ccall((:mlirTypeIsAPDLRangeType, mlir_c), Bool, (MlirType,), type)
end

function mlirPDLRangeTypeGet(elementType)
    ccall((:mlirPDLRangeTypeGet, mlir_c), MlirType, (MlirType,), elementType)
end

function mlirPDLRangeTypeGetElementType(type)
    ccall((:mlirPDLRangeTypeGetElementType, mlir_c), MlirType, (MlirType,), type)
end

function mlirTypeIsAPDLTypeType(type)
    ccall((:mlirTypeIsAPDLTypeType, mlir_c), Bool, (MlirType,), type)
end

function mlirPDLTypeTypeGet(ctx)
    ccall((:mlirPDLTypeTypeGet, mlir_c), MlirType, (MlirContext,), ctx)
end

function mlirTypeIsAPDLValueType(type)
    ccall((:mlirTypeIsAPDLValueType, mlir_c), Bool, (MlirType,), type)
end

function mlirPDLValueTypeGet(ctx)
    ccall((:mlirPDLValueTypeGet, mlir_c), MlirType, (MlirContext,), ctx)
end

# no prototype is found for this function at Quant.h:19:1, please use with caution
function mlirGetDialectHandle__quant__()
    ccall((:mlirGetDialectHandle__quant__, mlir_c), MlirDialectHandle, ())
end

function mlirTypeIsAQuantizedType(type)
    ccall((:mlirTypeIsAQuantizedType, mlir_c), Bool, (MlirType,), type)
end

# no prototype is found for this function at Quant.h:29:29, please use with caution
function mlirQuantizedTypeGetSignedFlag()
    ccall((:mlirQuantizedTypeGetSignedFlag, mlir_c), Cuint, ())
end

function mlirQuantizedTypeGetDefaultMinimumForInteger(isSigned, integralWidth)
    ccall((:mlirQuantizedTypeGetDefaultMinimumForInteger, mlir_c), Int64, (Bool, Cuint), isSigned, integralWidth)
end

function mlirQuantizedTypeGetDefaultMaximumForInteger(isSigned, integralWidth)
    ccall((:mlirQuantizedTypeGetDefaultMaximumForInteger, mlir_c), Int64, (Bool, Cuint), isSigned, integralWidth)
end

function mlirQuantizedTypeGetExpressedType(type)
    ccall((:mlirQuantizedTypeGetExpressedType, mlir_c), MlirType, (MlirType,), type)
end

function mlirQuantizedTypeGetFlags(type)
    ccall((:mlirQuantizedTypeGetFlags, mlir_c), Cuint, (MlirType,), type)
end

function mlirQuantizedTypeIsSigned(type)
    ccall((:mlirQuantizedTypeIsSigned, mlir_c), Bool, (MlirType,), type)
end

function mlirQuantizedTypeGetStorageType(type)
    ccall((:mlirQuantizedTypeGetStorageType, mlir_c), MlirType, (MlirType,), type)
end

function mlirQuantizedTypeGetStorageTypeMin(type)
    ccall((:mlirQuantizedTypeGetStorageTypeMin, mlir_c), Int64, (MlirType,), type)
end

function mlirQuantizedTypeGetStorageTypeMax(type)
    ccall((:mlirQuantizedTypeGetStorageTypeMax, mlir_c), Int64, (MlirType,), type)
end

function mlirQuantizedTypeGetStorageTypeIntegralWidth(type)
    ccall((:mlirQuantizedTypeGetStorageTypeIntegralWidth, mlir_c), Cuint, (MlirType,), type)
end

function mlirQuantizedTypeIsCompatibleExpressedType(type, candidate)
    ccall((:mlirQuantizedTypeIsCompatibleExpressedType, mlir_c), Bool, (MlirType, MlirType), type, candidate)
end

function mlirQuantizedTypeGetQuantizedElementType(type)
    ccall((:mlirQuantizedTypeGetQuantizedElementType, mlir_c), MlirType, (MlirType,), type)
end

function mlirQuantizedTypeCastFromStorageType(type, candidate)
    ccall((:mlirQuantizedTypeCastFromStorageType, mlir_c), MlirType, (MlirType, MlirType), type, candidate)
end

function mlirQuantizedTypeCastToStorageType(type)
    ccall((:mlirQuantizedTypeCastToStorageType, mlir_c), MlirType, (MlirType,), type)
end

function mlirQuantizedTypeCastFromExpressedType(type, candidate)
    ccall((:mlirQuantizedTypeCastFromExpressedType, mlir_c), MlirType, (MlirType, MlirType), type, candidate)
end

function mlirQuantizedTypeCastToExpressedType(type)
    ccall((:mlirQuantizedTypeCastToExpressedType, mlir_c), MlirType, (MlirType,), type)
end

function mlirQuantizedTypeCastExpressedToStorageType(type, candidate)
    ccall((:mlirQuantizedTypeCastExpressedToStorageType, mlir_c), MlirType, (MlirType, MlirType), type, candidate)
end

function mlirTypeIsAAnyQuantizedType(type)
    ccall((:mlirTypeIsAAnyQuantizedType, mlir_c), Bool, (MlirType,), type)
end

function mlirAnyQuantizedTypeGet(flags, storageType, expressedType, storageTypeMin, storageTypeMax)
    ccall((:mlirAnyQuantizedTypeGet, mlir_c), MlirType, (Cuint, MlirType, MlirType, Int64, Int64), flags, storageType, expressedType, storageTypeMin, storageTypeMax)
end

function mlirTypeIsAUniformQuantizedType(type)
    ccall((:mlirTypeIsAUniformQuantizedType, mlir_c), Bool, (MlirType,), type)
end

function mlirUniformQuantizedTypeGet(flags, storageType, expressedType, scale, zeroPoint, storageTypeMin, storageTypeMax)
    ccall((:mlirUniformQuantizedTypeGet, mlir_c), MlirType, (Cuint, MlirType, MlirType, Cdouble, Int64, Int64, Int64), flags, storageType, expressedType, scale, zeroPoint, storageTypeMin, storageTypeMax)
end

function mlirUniformQuantizedTypeGetScale(type)
    ccall((:mlirUniformQuantizedTypeGetScale, mlir_c), Cdouble, (MlirType,), type)
end

function mlirUniformQuantizedTypeGetZeroPoint(type)
    ccall((:mlirUniformQuantizedTypeGetZeroPoint, mlir_c), Int64, (MlirType,), type)
end

function mlirUniformQuantizedTypeIsFixedPoint(type)
    ccall((:mlirUniformQuantizedTypeIsFixedPoint, mlir_c), Bool, (MlirType,), type)
end

function mlirTypeIsAUniformQuantizedPerAxisType(type)
    ccall((:mlirTypeIsAUniformQuantizedPerAxisType, mlir_c), Bool, (MlirType,), type)
end

function mlirUniformQuantizedPerAxisTypeGet(flags, storageType, expressedType, nDims, scales, zeroPoints, quantizedDimension, storageTypeMin, storageTypeMax)
    ccall((:mlirUniformQuantizedPerAxisTypeGet, mlir_c), MlirType, (Cuint, MlirType, MlirType, intptr_t, Ptr{Cdouble}, Ptr{Int64}, Int32, Int64, Int64), flags, storageType, expressedType, nDims, scales, zeroPoints, quantizedDimension, storageTypeMin, storageTypeMax)
end

function mlirUniformQuantizedPerAxisTypeGetNumDims(type)
    ccall((:mlirUniformQuantizedPerAxisTypeGetNumDims, mlir_c), intptr_t, (MlirType,), type)
end

function mlirUniformQuantizedPerAxisTypeGetScale(type, pos)
    ccall((:mlirUniformQuantizedPerAxisTypeGetScale, mlir_c), Cdouble, (MlirType, intptr_t), type, pos)
end

function mlirUniformQuantizedPerAxisTypeGetZeroPoint(type, pos)
    ccall((:mlirUniformQuantizedPerAxisTypeGetZeroPoint, mlir_c), Int64, (MlirType, intptr_t), type, pos)
end

function mlirUniformQuantizedPerAxisTypeGetQuantizedDimension(type)
    ccall((:mlirUniformQuantizedPerAxisTypeGetQuantizedDimension, mlir_c), Int32, (MlirType,), type)
end

function mlirUniformQuantizedPerAxisTypeIsFixedPoint(type)
    ccall((:mlirUniformQuantizedPerAxisTypeIsFixedPoint, mlir_c), Bool, (MlirType,), type)
end

function mlirTypeIsACalibratedQuantizedType(type)
    ccall((:mlirTypeIsACalibratedQuantizedType, mlir_c), Bool, (MlirType,), type)
end

function mlirCalibratedQuantizedTypeGet(expressedType, min, max)
    ccall((:mlirCalibratedQuantizedTypeGet, mlir_c), MlirType, (MlirType, Cdouble, Cdouble), expressedType, min, max)
end

function mlirCalibratedQuantizedTypeGetMin(type)
    ccall((:mlirCalibratedQuantizedTypeGetMin, mlir_c), Cdouble, (MlirType,), type)
end

function mlirCalibratedQuantizedTypeGetMax(type)
    ccall((:mlirCalibratedQuantizedTypeGetMax, mlir_c), Cdouble, (MlirType,), type)
end

# no prototype is found for this function at SCF.h:19:1, please use with caution
function mlirGetDialectHandle__scf__()
    ccall((:mlirGetDialectHandle__scf__, mlir_c), MlirDialectHandle, ())
end

# no prototype is found for this function at Shape.h:19:1, please use with caution
function mlirGetDialectHandle__shape__()
    ccall((:mlirGetDialectHandle__shape__, mlir_c), MlirDialectHandle, ())
end

# no prototype is found for this function at SparseTensor.h:20:1, please use with caution
function mlirGetDialectHandle__sparse_tensor__()
    ccall((:mlirGetDialectHandle__sparse_tensor__, mlir_c), MlirDialectHandle, ())
end

@cenum MlirSparseTensorDimLevelType::UInt32 begin
    MLIR_SPARSE_TENSOR_DIM_LEVEL_DENSE = 0
    MLIR_SPARSE_TENSOR_DIM_LEVEL_COMPRESSED = 1
    MLIR_SPARSE_TENSOR_DIM_LEVEL_SINGLETON = 2
end

function mlirAttributeIsASparseTensorEncodingAttr(attr)
    ccall((:mlirAttributeIsASparseTensorEncodingAttr, mlir_c), Bool, (MlirAttribute,), attr)
end

function mlirSparseTensorEncodingAttrGet(ctx, numDimLevelTypes, dimLevelTypes, dimOrdering, pointerBitWidth, indexBitWidth)
    ccall((:mlirSparseTensorEncodingAttrGet, mlir_c), MlirAttribute, (MlirContext, intptr_t, Ptr{MlirSparseTensorDimLevelType}, MlirAffineMap, Cint, Cint), ctx, numDimLevelTypes, dimLevelTypes, dimOrdering, pointerBitWidth, indexBitWidth)
end

function mlirSparseTensorEncodingGetNumDimLevelTypes(attr)
    ccall((:mlirSparseTensorEncodingGetNumDimLevelTypes, mlir_c), intptr_t, (MlirAttribute,), attr)
end

function mlirSparseTensorEncodingAttrGetDimLevelType(attr, pos)
    ccall((:mlirSparseTensorEncodingAttrGetDimLevelType, mlir_c), MlirSparseTensorDimLevelType, (MlirAttribute, intptr_t), attr, pos)
end

function mlirSparseTensorEncodingAttrGetDimOrdering(attr)
    ccall((:mlirSparseTensorEncodingAttrGetDimOrdering, mlir_c), MlirAffineMap, (MlirAttribute,), attr)
end

function mlirSparseTensorEncodingAttrGetPointerBitWidth(attr)
    ccall((:mlirSparseTensorEncodingAttrGetPointerBitWidth, mlir_c), Cint, (MlirAttribute,), attr)
end

function mlirSparseTensorEncodingAttrGetIndexBitWidth(attr)
    ccall((:mlirSparseTensorEncodingAttrGetIndexBitWidth, mlir_c), Cint, (MlirAttribute,), attr)
end

# no prototype is found for this function at Passes.capi.h.inc:11:25, please use with caution
function mlirRegisterSparseTensorPasses()
    ccall((:mlirRegisterSparseTensorPasses, mlir_c), Cvoid, ())
end

# no prototype is found for this function at Passes.capi.h.inc:15:29, please use with caution
function mlirCreateSparseTensorSparseTensorConversion()
    ccall((:mlirCreateSparseTensorSparseTensorConversion, mlir_c), MlirPass, ())
end

# no prototype is found for this function at Passes.capi.h.inc:16:25, please use with caution
function mlirRegisterSparseTensorSparseTensorConversion()
    ccall((:mlirRegisterSparseTensorSparseTensorConversion, mlir_c), Cvoid, ())
end

# no prototype is found for this function at Passes.capi.h.inc:20:29, please use with caution
function mlirCreateSparseTensorSparsification()
    ccall((:mlirCreateSparseTensorSparsification, mlir_c), MlirPass, ())
end

# no prototype is found for this function at Passes.capi.h.inc:21:25, please use with caution
function mlirRegisterSparseTensorSparsification()
    ccall((:mlirRegisterSparseTensorSparsification, mlir_c), Cvoid, ())
end

# no prototype is found for this function at Tensor.h:19:1, please use with caution
function mlirGetDialectHandle__tensor__()
    ccall((:mlirGetDialectHandle__tensor__, mlir_c), MlirDialectHandle, ())
end

function mlirOperationImplementsInterface(operation, interfaceTypeID)
    ccall((:mlirOperationImplementsInterface, mlir_c), Bool, (MlirOperation, MlirTypeID), operation, interfaceTypeID)
end

function mlirOperationImplementsInterfaceStatic(operationName, context, interfaceTypeID)
    ccall((:mlirOperationImplementsInterfaceStatic, mlir_c), Bool, (MlirStringRef, MlirContext, MlirTypeID), operationName, context, interfaceTypeID)
end

# no prototype is found for this function at Interfaces.h:45:31, please use with caution
function mlirInferTypeOpInterfaceTypeID()
    ccall((:mlirInferTypeOpInterfaceTypeID, mlir_c), MlirTypeID, ())
end

# typedef void ( * MlirTypesCallback ) ( intptr_t , MlirType * , void * )
const MlirTypesCallback = Ptr{Cvoid}

function mlirInferTypeOpInterfaceInferReturnTypes(opName, context, location, nOperands, operands, attributes, nRegions, regions, callback, userData)
    ccall((:mlirInferTypeOpInterfaceInferReturnTypes, mlir_c), MlirLogicalResult, (MlirStringRef, MlirContext, MlirLocation, intptr_t, Ptr{MlirValue}, MlirAttribute, intptr_t, Ptr{MlirRegion}, MlirTypesCallback, Ptr{Cvoid}), opName, context, location, nOperands, operands, attributes, nRegions, regions, callback, userData)
end

function mlirRegisterAllDialects(registry)
    ccall((:mlirRegisterAllDialects, mlir_c), Cvoid, (MlirDialectRegistry,), registry)
end

function mlirRegisterAllLLVMTranslations(context)
    ccall((:mlirRegisterAllLLVMTranslations, mlir_c), Cvoid, (MlirContext,), context)
end

# no prototype is found for this function at RegisterEverything.h:32:25, please use with caution
function mlirRegisterAllPasses()
    ccall((:mlirRegisterAllPasses, mlir_c), Cvoid, ())
end

# no prototype is found for this function at Transforms.capi.h.inc:11:25, please use with caution
function mlirRegisterTransformsPasses()
    ccall((:mlirRegisterTransformsPasses, mlir_c), Cvoid, ())
end

# no prototype is found for this function at Transforms.capi.h.inc:15:29, please use with caution
function mlirCreateTransformsCSE()
    ccall((:mlirCreateTransformsCSE, mlir_c), MlirPass, ())
end

# no prototype is found for this function at Transforms.capi.h.inc:16:25, please use with caution
function mlirRegisterTransformsCSE()
    ccall((:mlirRegisterTransformsCSE, mlir_c), Cvoid, ())
end

# no prototype is found for this function at Transforms.capi.h.inc:20:29, please use with caution
function mlirCreateTransformsCanonicalizer()
    ccall((:mlirCreateTransformsCanonicalizer, mlir_c), MlirPass, ())
end

# no prototype is found for this function at Transforms.capi.h.inc:21:25, please use with caution
function mlirRegisterTransformsCanonicalizer()
    ccall((:mlirRegisterTransformsCanonicalizer, mlir_c), Cvoid, ())
end

# no prototype is found for this function at Transforms.capi.h.inc:25:29, please use with caution
function mlirCreateTransformsControlFlowSink()
    ccall((:mlirCreateTransformsControlFlowSink, mlir_c), MlirPass, ())
end

# no prototype is found for this function at Transforms.capi.h.inc:26:25, please use with caution
function mlirRegisterTransformsControlFlowSink()
    ccall((:mlirRegisterTransformsControlFlowSink, mlir_c), Cvoid, ())
end

# no prototype is found for this function at Transforms.capi.h.inc:30:29, please use with caution
function mlirCreateTransformsInliner()
    ccall((:mlirCreateTransformsInliner, mlir_c), MlirPass, ())
end

# no prototype is found for this function at Transforms.capi.h.inc:31:25, please use with caution
function mlirRegisterTransformsInliner()
    ccall((:mlirRegisterTransformsInliner, mlir_c), Cvoid, ())
end

# no prototype is found for this function at Transforms.capi.h.inc:35:29, please use with caution
function mlirCreateTransformsLocationSnapshot()
    ccall((:mlirCreateTransformsLocationSnapshot, mlir_c), MlirPass, ())
end

# no prototype is found for this function at Transforms.capi.h.inc:36:25, please use with caution
function mlirRegisterTransformsLocationSnapshot()
    ccall((:mlirRegisterTransformsLocationSnapshot, mlir_c), Cvoid, ())
end

# no prototype is found for this function at Transforms.capi.h.inc:40:29, please use with caution
function mlirCreateTransformsLoopInvariantCodeMotion()
    ccall((:mlirCreateTransformsLoopInvariantCodeMotion, mlir_c), MlirPass, ())
end

# no prototype is found for this function at Transforms.capi.h.inc:41:25, please use with caution
function mlirRegisterTransformsLoopInvariantCodeMotion()
    ccall((:mlirRegisterTransformsLoopInvariantCodeMotion, mlir_c), Cvoid, ())
end

# no prototype is found for this function at Transforms.capi.h.inc:45:29, please use with caution
function mlirCreateTransformsPrintOpStats()
    ccall((:mlirCreateTransformsPrintOpStats, mlir_c), MlirPass, ())
end

# no prototype is found for this function at Transforms.capi.h.inc:46:25, please use with caution
function mlirRegisterTransformsPrintOpStats()
    ccall((:mlirRegisterTransformsPrintOpStats, mlir_c), Cvoid, ())
end

# no prototype is found for this function at Transforms.capi.h.inc:50:29, please use with caution
function mlirCreateTransformsSCCP()
    ccall((:mlirCreateTransformsSCCP, mlir_c), MlirPass, ())
end

# no prototype is found for this function at Transforms.capi.h.inc:51:25, please use with caution
function mlirRegisterTransformsSCCP()
    ccall((:mlirRegisterTransformsSCCP, mlir_c), Cvoid, ())
end

# no prototype is found for this function at Transforms.capi.h.inc:55:29, please use with caution
function mlirCreateTransformsStripDebugInfo()
    ccall((:mlirCreateTransformsStripDebugInfo, mlir_c), MlirPass, ())
end

# no prototype is found for this function at Transforms.capi.h.inc:56:25, please use with caution
function mlirRegisterTransformsStripDebugInfo()
    ccall((:mlirRegisterTransformsStripDebugInfo, mlir_c), Cvoid, ())
end

# no prototype is found for this function at Transforms.capi.h.inc:60:29, please use with caution
function mlirCreateTransformsSymbolDCE()
    ccall((:mlirCreateTransformsSymbolDCE, mlir_c), MlirPass, ())
end

# no prototype is found for this function at Transforms.capi.h.inc:61:25, please use with caution
function mlirRegisterTransformsSymbolDCE()
    ccall((:mlirRegisterTransformsSymbolDCE, mlir_c), Cvoid, ())
end

# no prototype is found for this function at Transforms.capi.h.inc:65:29, please use with caution
function mlirCreateTransformsSymbolPrivatize()
    ccall((:mlirCreateTransformsSymbolPrivatize, mlir_c), MlirPass, ())
end

# no prototype is found for this function at Transforms.capi.h.inc:66:25, please use with caution
function mlirRegisterTransformsSymbolPrivatize()
    ccall((:mlirRegisterTransformsSymbolPrivatize, mlir_c), Cvoid, ())
end

# no prototype is found for this function at Transforms.capi.h.inc:70:29, please use with caution
function mlirCreateTransformsTopologicalSort()
    ccall((:mlirCreateTransformsTopologicalSort, mlir_c), MlirPass, ())
end

# no prototype is found for this function at Transforms.capi.h.inc:71:25, please use with caution
function mlirRegisterTransformsTopologicalSort()
    ccall((:mlirRegisterTransformsTopologicalSort, mlir_c), Cvoid, ())
end

# no prototype is found for this function at Transforms.capi.h.inc:75:29, please use with caution
function mlirCreateTransformsViewOpGraph()
    ccall((:mlirCreateTransformsViewOpGraph, mlir_c), MlirPass, ())
end

# no prototype is found for this function at Transforms.capi.h.inc:76:25, please use with caution
function mlirRegisterTransformsViewOpGraph()
    ccall((:mlirRegisterTransformsViewOpGraph, mlir_c), Cvoid, ())
end

