using CEnum

const intptr_t = Clong

struct MlirStringRef
    data::Cstring
    length::Csize_t
end

function mlirStringRefCreate(str, length)
    ccall((:mlirStringRefCreate, libMLIRPublicAPI), MlirStringRef, (Cstring, Csize_t), str, length)
end

function mlirStringRefCreateFromCString(str)
    ccall((:mlirStringRefCreateFromCString, libMLIRPublicAPI), MlirStringRef, (Cstring,), str)
end

# typedef void ( * MlirStringCallback ) ( MlirStringRef , void * )
const MlirStringCallback = Ptr{Cvoid}

struct MlirLogicalResult
    value::Int8
end

function mlirLogicalResultIsSuccess(res)
    ccall((:mlirLogicalResultIsSuccess, libMLIRPublicAPI), Bool, (MlirLogicalResult,), res)
end

function mlirLogicalResultIsFailure(res)
    ccall((:mlirLogicalResultIsFailure, libMLIRPublicAPI), Bool, (MlirLogicalResult,), res)
end

# no prototype is found for this function at Support.h:107:33, please use with caution
function mlirLogicalResultSuccess()
    ccall((:mlirLogicalResultSuccess, libMLIRPublicAPI), MlirLogicalResult, ())
end

# no prototype is found for this function at Support.h:113:33, please use with caution
function mlirLogicalResultFailure()
    ccall((:mlirLogicalResultFailure, libMLIRPublicAPI), MlirLogicalResult, ())
end

struct MlirContext
    ptr::Ptr{Cvoid}
end

struct MlirDialect
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

# no prototype is found for this function at IR.h:83:32, please use with caution
function mlirContextCreate()
    ccall((:mlirContextCreate, libMLIRPublicAPI), MlirContext, ())
end

function mlirContextEqual(ctx1, ctx2)
    ccall((:mlirContextEqual, libMLIRPublicAPI), Bool, (MlirContext, MlirContext), ctx1, ctx2)
end

function mlirContextIsNull(context)
    ccall((:mlirContextIsNull, libMLIRPublicAPI), Bool, (MlirContext,), context)
end

function mlirContextDestroy(context)
    ccall((:mlirContextDestroy, libMLIRPublicAPI), Cvoid, (MlirContext,), context)
end

function mlirContextSetAllowUnregisteredDialects(context, allow)
    ccall((:mlirContextSetAllowUnregisteredDialects, libMLIRPublicAPI), Cvoid, (MlirContext, Bool), context, allow)
end

function mlirContextGetAllowUnregisteredDialects(context)
    ccall((:mlirContextGetAllowUnregisteredDialects, libMLIRPublicAPI), Bool, (MlirContext,), context)
end

function mlirContextGetNumRegisteredDialects(context)
    ccall((:mlirContextGetNumRegisteredDialects, libMLIRPublicAPI), intptr_t, (MlirContext,), context)
end

function mlirContextGetNumLoadedDialects(context)
    ccall((:mlirContextGetNumLoadedDialects, libMLIRPublicAPI), intptr_t, (MlirContext,), context)
end

function mlirContextGetOrLoadDialect(context, name)
    ccall((:mlirContextGetOrLoadDialect, libMLIRPublicAPI), MlirDialect, (MlirContext, MlirStringRef), context, name)
end

function mlirDialectGetContext(dialect)
    ccall((:mlirDialectGetContext, libMLIRPublicAPI), MlirContext, (MlirDialect,), dialect)
end

function mlirDialectIsNull(dialect)
    ccall((:mlirDialectIsNull, libMLIRPublicAPI), Bool, (MlirDialect,), dialect)
end

function mlirDialectEqual(dialect1, dialect2)
    ccall((:mlirDialectEqual, libMLIRPublicAPI), Bool, (MlirDialect, MlirDialect), dialect1, dialect2)
end

function mlirDialectGetNamespace(dialect)
    ccall((:mlirDialectGetNamespace, libMLIRPublicAPI), MlirStringRef, (MlirDialect,), dialect)
end

function mlirLocationFileLineColGet(context, filename, line, col)
    ccall((:mlirLocationFileLineColGet, libMLIRPublicAPI), MlirLocation, (MlirContext, MlirStringRef, Cuint, Cuint), context, filename, line, col)
end

function mlirLocationCallSiteGet(callee, caller)
    ccall((:mlirLocationCallSiteGet, libMLIRPublicAPI), MlirLocation, (MlirLocation, MlirLocation), callee, caller)
end

function mlirLocationUnknownGet(context)
    ccall((:mlirLocationUnknownGet, libMLIRPublicAPI), MlirLocation, (MlirContext,), context)
end

function mlirLocationGetContext(location)
    ccall((:mlirLocationGetContext, libMLIRPublicAPI), MlirContext, (MlirLocation,), location)
end

function mlirLocationIsNull(location)
    ccall((:mlirLocationIsNull, libMLIRPublicAPI), Bool, (MlirLocation,), location)
end

function mlirLocationEqual(l1, l2)
    ccall((:mlirLocationEqual, libMLIRPublicAPI), Bool, (MlirLocation, MlirLocation), l1, l2)
end

function mlirLocationPrint(location, callback, userData)
    ccall((:mlirLocationPrint, libMLIRPublicAPI), Cvoid, (MlirLocation, MlirStringCallback, Ptr{Cvoid}), location, callback, userData)
end

function mlirModuleCreateEmpty(location)
    ccall((:mlirModuleCreateEmpty, libMLIRPublicAPI), MlirModule, (MlirLocation,), location)
end

function mlirModuleCreateParse(context, _module)
    ccall((:mlirModuleCreateParse, libMLIRPublicAPI), MlirModule, (MlirContext, MlirStringRef), context, _module)
end

function mlirModuleGetContext(_module)
    ccall((:mlirModuleGetContext, libMLIRPublicAPI), MlirContext, (MlirModule,), _module)
end

function mlirModuleGetBody(_module)
    ccall((:mlirModuleGetBody, libMLIRPublicAPI), MlirBlock, (MlirModule,), _module)
end

function mlirModuleIsNull(_module)
    ccall((:mlirModuleIsNull, libMLIRPublicAPI), Bool, (MlirModule,), _module)
end

function mlirModuleDestroy(_module)
    ccall((:mlirModuleDestroy, libMLIRPublicAPI), Cvoid, (MlirModule,), _module)
end

function mlirModuleGetOperation(_module)
    ccall((:mlirModuleGetOperation, libMLIRPublicAPI), MlirOperation, (MlirModule,), _module)
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
    ccall((:mlirOperationStateGet, libMLIRPublicAPI), MlirOperationState, (MlirStringRef, MlirLocation), name, loc)
end

function mlirOperationStateAddResults(state, n, results)
    ccall((:mlirOperationStateAddResults, libMLIRPublicAPI), Cvoid, (Ptr{MlirOperationState}, intptr_t, Ptr{MlirType}), state, n, results)
end

function mlirOperationStateAddOperands(state, n, operands)
    ccall((:mlirOperationStateAddOperands, libMLIRPublicAPI), Cvoid, (Ptr{MlirOperationState}, intptr_t, Ptr{MlirValue}), state, n, operands)
end

function mlirOperationStateAddOwnedRegions(state, n, regions)
    ccall((:mlirOperationStateAddOwnedRegions, libMLIRPublicAPI), Cvoid, (Ptr{MlirOperationState}, intptr_t, Ptr{MlirRegion}), state, n, regions)
end

function mlirOperationStateAddSuccessors(state, n, successors)
    ccall((:mlirOperationStateAddSuccessors, libMLIRPublicAPI), Cvoid, (Ptr{MlirOperationState}, intptr_t, Ptr{MlirBlock}), state, n, successors)
end

function mlirOperationStateAddAttributes(state, n, attributes)
    ccall((:mlirOperationStateAddAttributes, libMLIRPublicAPI), Cvoid, (Ptr{MlirOperationState}, intptr_t, Ptr{MlirNamedAttribute}), state, n, attributes)
end

function mlirOperationStateEnableResultTypeInference(state)
    ccall((:mlirOperationStateEnableResultTypeInference, libMLIRPublicAPI), Cvoid, (Ptr{MlirOperationState},), state)
end

# no prototype is found for this function at IR.h:270:40, please use with caution
function mlirOpPrintingFlagsCreate()
    ccall((:mlirOpPrintingFlagsCreate, libMLIRPublicAPI), MlirOpPrintingFlags, ())
end

function mlirOpPrintingFlagsDestroy(flags)
    ccall((:mlirOpPrintingFlagsDestroy, libMLIRPublicAPI), Cvoid, (MlirOpPrintingFlags,), flags)
end

function mlirOpPrintingFlagsElideLargeElementsAttrs(flags, largeElementLimit)
    ccall((:mlirOpPrintingFlagsElideLargeElementsAttrs, libMLIRPublicAPI), Cvoid, (MlirOpPrintingFlags, intptr_t), flags, largeElementLimit)
end

function mlirOpPrintingFlagsEnableDebugInfo(flags, prettyForm)
    ccall((:mlirOpPrintingFlagsEnableDebugInfo, libMLIRPublicAPI), Cvoid, (MlirOpPrintingFlags, Bool), flags, prettyForm)
end

function mlirOpPrintingFlagsPrintGenericOpForm(flags)
    ccall((:mlirOpPrintingFlagsPrintGenericOpForm, libMLIRPublicAPI), Cvoid, (MlirOpPrintingFlags,), flags)
end

function mlirOpPrintingFlagsUseLocalScope(flags)
    ccall((:mlirOpPrintingFlagsUseLocalScope, libMLIRPublicAPI), Cvoid, (MlirOpPrintingFlags,), flags)
end

function mlirOperationCreate(state)
    ccall((:mlirOperationCreate, libMLIRPublicAPI), MlirOperation, (Ptr{MlirOperationState},), state)
end

function mlirOperationDestroy(op)
    ccall((:mlirOperationDestroy, libMLIRPublicAPI), Cvoid, (MlirOperation,), op)
end

function mlirOperationIsNull(op)
    ccall((:mlirOperationIsNull, libMLIRPublicAPI), Bool, (MlirOperation,), op)
end

function mlirOperationEqual(op, other)
    ccall((:mlirOperationEqual, libMLIRPublicAPI), Bool, (MlirOperation, MlirOperation), op, other)
end

function mlirOperationGetName(op)
    ccall((:mlirOperationGetName, libMLIRPublicAPI), MlirIdentifier, (MlirOperation,), op)
end

function mlirOperationGetBlock(op)
    ccall((:mlirOperationGetBlock, libMLIRPublicAPI), MlirBlock, (MlirOperation,), op)
end

function mlirOperationGetParentOperation(op)
    ccall((:mlirOperationGetParentOperation, libMLIRPublicAPI), MlirOperation, (MlirOperation,), op)
end

function mlirOperationGetNumRegions(op)
    ccall((:mlirOperationGetNumRegions, libMLIRPublicAPI), intptr_t, (MlirOperation,), op)
end

function mlirOperationGetRegion(op, pos)
    ccall((:mlirOperationGetRegion, libMLIRPublicAPI), MlirRegion, (MlirOperation, intptr_t), op, pos)
end

function mlirOperationGetNextInBlock(op)
    ccall((:mlirOperationGetNextInBlock, libMLIRPublicAPI), MlirOperation, (MlirOperation,), op)
end

function mlirOperationGetNumOperands(op)
    ccall((:mlirOperationGetNumOperands, libMLIRPublicAPI), intptr_t, (MlirOperation,), op)
end

function mlirOperationGetOperand(op, pos)
    ccall((:mlirOperationGetOperand, libMLIRPublicAPI), MlirValue, (MlirOperation, intptr_t), op, pos)
end

function mlirOperationGetNumResults(op)
    ccall((:mlirOperationGetNumResults, libMLIRPublicAPI), intptr_t, (MlirOperation,), op)
end

function mlirOperationGetResult(op, pos)
    ccall((:mlirOperationGetResult, libMLIRPublicAPI), MlirValue, (MlirOperation, intptr_t), op, pos)
end

function mlirOperationGetNumSuccessors(op)
    ccall((:mlirOperationGetNumSuccessors, libMLIRPublicAPI), intptr_t, (MlirOperation,), op)
end

function mlirOperationGetSuccessor(op, pos)
    ccall((:mlirOperationGetSuccessor, libMLIRPublicAPI), MlirBlock, (MlirOperation, intptr_t), op, pos)
end

function mlirOperationGetNumAttributes(op)
    ccall((:mlirOperationGetNumAttributes, libMLIRPublicAPI), intptr_t, (MlirOperation,), op)
end

function mlirOperationGetAttribute(op, pos)
    ccall((:mlirOperationGetAttribute, libMLIRPublicAPI), MlirNamedAttribute, (MlirOperation, intptr_t), op, pos)
end

function mlirOperationGetAttributeByName(op, name)
    ccall((:mlirOperationGetAttributeByName, libMLIRPublicAPI), MlirAttribute, (MlirOperation, MlirStringRef), op, name)
end

function mlirOperationSetAttributeByName(op, name, attr)
    ccall((:mlirOperationSetAttributeByName, libMLIRPublicAPI), Cvoid, (MlirOperation, MlirStringRef, MlirAttribute), op, name, attr)
end

function mlirOperationRemoveAttributeByName(op, name)
    ccall((:mlirOperationRemoveAttributeByName, libMLIRPublicAPI), Bool, (MlirOperation, MlirStringRef), op, name)
end

function mlirOperationPrint(op, callback, userData)
    ccall((:mlirOperationPrint, libMLIRPublicAPI), Cvoid, (MlirOperation, MlirStringCallback, Ptr{Cvoid}), op, callback, userData)
end

function mlirOperationPrintWithFlags(op, flags, callback, userData)
    ccall((:mlirOperationPrintWithFlags, libMLIRPublicAPI), Cvoid, (MlirOperation, MlirOpPrintingFlags, MlirStringCallback, Ptr{Cvoid}), op, flags, callback, userData)
end

function mlirOperationDump(op)
    ccall((:mlirOperationDump, libMLIRPublicAPI), Cvoid, (MlirOperation,), op)
end

function mlirOperationVerify(op)
    ccall((:mlirOperationVerify, libMLIRPublicAPI), Bool, (MlirOperation,), op)
end

# no prototype is found for this function at IR.h:416:31, please use with caution
function mlirRegionCreate()
    ccall((:mlirRegionCreate, libMLIRPublicAPI), MlirRegion, ())
end

function mlirRegionDestroy(region)
    ccall((:mlirRegionDestroy, libMLIRPublicAPI), Cvoid, (MlirRegion,), region)
end

function mlirRegionIsNull(region)
    ccall((:mlirRegionIsNull, libMLIRPublicAPI), Bool, (MlirRegion,), region)
end

function mlirRegionGetFirstBlock(region)
    ccall((:mlirRegionGetFirstBlock, libMLIRPublicAPI), MlirBlock, (MlirRegion,), region)
end

function mlirRegionAppendOwnedBlock(region, block)
    ccall((:mlirRegionAppendOwnedBlock, libMLIRPublicAPI), Cvoid, (MlirRegion, MlirBlock), region, block)
end

function mlirRegionInsertOwnedBlock(region, pos, block)
    ccall((:mlirRegionInsertOwnedBlock, libMLIRPublicAPI), Cvoid, (MlirRegion, intptr_t, MlirBlock), region, pos, block)
end

function mlirRegionInsertOwnedBlockAfter(region, reference, block)
    ccall((:mlirRegionInsertOwnedBlockAfter, libMLIRPublicAPI), Cvoid, (MlirRegion, MlirBlock, MlirBlock), region, reference, block)
end

function mlirRegionInsertOwnedBlockBefore(region, reference, block)
    ccall((:mlirRegionInsertOwnedBlockBefore, libMLIRPublicAPI), Cvoid, (MlirRegion, MlirBlock, MlirBlock), region, reference, block)
end

function mlirBlockCreate(nArgs, args)
    ccall((:mlirBlockCreate, libMLIRPublicAPI), MlirBlock, (intptr_t, Ptr{MlirType}), nArgs, args)
end

function mlirBlockDestroy(block)
    ccall((:mlirBlockDestroy, libMLIRPublicAPI), Cvoid, (MlirBlock,), block)
end

function mlirBlockIsNull(block)
    ccall((:mlirBlockIsNull, libMLIRPublicAPI), Bool, (MlirBlock,), block)
end

function mlirBlockEqual(block, other)
    ccall((:mlirBlockEqual, libMLIRPublicAPI), Bool, (MlirBlock, MlirBlock), block, other)
end

function mlirBlockGetNextInRegion(block)
    ccall((:mlirBlockGetNextInRegion, libMLIRPublicAPI), MlirBlock, (MlirBlock,), block)
end

function mlirBlockGetFirstOperation(block)
    ccall((:mlirBlockGetFirstOperation, libMLIRPublicAPI), MlirOperation, (MlirBlock,), block)
end

function mlirBlockGetTerminator(block)
    ccall((:mlirBlockGetTerminator, libMLIRPublicAPI), MlirOperation, (MlirBlock,), block)
end

function mlirBlockAppendOwnedOperation(block, operation)
    ccall((:mlirBlockAppendOwnedOperation, libMLIRPublicAPI), Cvoid, (MlirBlock, MlirOperation), block, operation)
end

function mlirBlockInsertOwnedOperation(block, pos, operation)
    ccall((:mlirBlockInsertOwnedOperation, libMLIRPublicAPI), Cvoid, (MlirBlock, intptr_t, MlirOperation), block, pos, operation)
end

function mlirBlockInsertOwnedOperationAfter(block, reference, operation)
    ccall((:mlirBlockInsertOwnedOperationAfter, libMLIRPublicAPI), Cvoid, (MlirBlock, MlirOperation, MlirOperation), block, reference, operation)
end

function mlirBlockInsertOwnedOperationBefore(block, reference, operation)
    ccall((:mlirBlockInsertOwnedOperationBefore, libMLIRPublicAPI), Cvoid, (MlirBlock, MlirOperation, MlirOperation), block, reference, operation)
end

function mlirBlockGetNumArguments(block)
    ccall((:mlirBlockGetNumArguments, libMLIRPublicAPI), intptr_t, (MlirBlock,), block)
end

function mlirBlockGetArgument(block, pos)
    ccall((:mlirBlockGetArgument, libMLIRPublicAPI), MlirValue, (MlirBlock, intptr_t), block, pos)
end

function mlirBlockPrint(block, callback, userData)
    ccall((:mlirBlockPrint, libMLIRPublicAPI), Cvoid, (MlirBlock, MlirStringCallback, Ptr{Cvoid}), block, callback, userData)
end

function mlirValueIsNull(value)
    ccall((:mlirValueIsNull, libMLIRPublicAPI), Bool, (MlirValue,), value)
end

function mlirValueEqual(value1, value2)
    ccall((:mlirValueEqual, libMLIRPublicAPI), Bool, (MlirValue, MlirValue), value1, value2)
end

function mlirValueIsABlockArgument(value)
    ccall((:mlirValueIsABlockArgument, libMLIRPublicAPI), Bool, (MlirValue,), value)
end

function mlirValueIsAOpResult(value)
    ccall((:mlirValueIsAOpResult, libMLIRPublicAPI), Bool, (MlirValue,), value)
end

function mlirBlockArgumentGetOwner(value)
    ccall((:mlirBlockArgumentGetOwner, libMLIRPublicAPI), MlirBlock, (MlirValue,), value)
end

function mlirBlockArgumentGetArgNumber(value)
    ccall((:mlirBlockArgumentGetArgNumber, libMLIRPublicAPI), intptr_t, (MlirValue,), value)
end

function mlirBlockArgumentSetType(value, type)
    ccall((:mlirBlockArgumentSetType, libMLIRPublicAPI), Cvoid, (MlirValue, MlirType), value, type)
end

function mlirOpResultGetOwner(value)
    ccall((:mlirOpResultGetOwner, libMLIRPublicAPI), MlirOperation, (MlirValue,), value)
end

function mlirOpResultGetResultNumber(value)
    ccall((:mlirOpResultGetResultNumber, libMLIRPublicAPI), intptr_t, (MlirValue,), value)
end

function mlirValueGetType(value)
    ccall((:mlirValueGetType, libMLIRPublicAPI), MlirType, (MlirValue,), value)
end

function mlirValueDump(value)
    ccall((:mlirValueDump, libMLIRPublicAPI), Cvoid, (MlirValue,), value)
end

function mlirValuePrint(value, callback, userData)
    ccall((:mlirValuePrint, libMLIRPublicAPI), Cvoid, (MlirValue, MlirStringCallback, Ptr{Cvoid}), value, callback, userData)
end

function mlirTypeParseGet(context, type)
    ccall((:mlirTypeParseGet, libMLIRPublicAPI), MlirType, (MlirContext, MlirStringRef), context, type)
end

function mlirTypeGetContext(type)
    ccall((:mlirTypeGetContext, libMLIRPublicAPI), MlirContext, (MlirType,), type)
end

function mlirTypeIsNull(type)
    ccall((:mlirTypeIsNull, libMLIRPublicAPI), Bool, (MlirType,), type)
end

function mlirTypeEqual(t1, t2)
    ccall((:mlirTypeEqual, libMLIRPublicAPI), Bool, (MlirType, MlirType), t1, t2)
end

function mlirTypePrint(type, callback, userData)
    ccall((:mlirTypePrint, libMLIRPublicAPI), Cvoid, (MlirType, MlirStringCallback, Ptr{Cvoid}), type, callback, userData)
end

function mlirTypeDump(type)
    ccall((:mlirTypeDump, libMLIRPublicAPI), Cvoid, (MlirType,), type)
end

function mlirAttributeParseGet(context, attr)
    ccall((:mlirAttributeParseGet, libMLIRPublicAPI), MlirAttribute, (MlirContext, MlirStringRef), context, attr)
end

function mlirAttributeGetContext(attribute)
    ccall((:mlirAttributeGetContext, libMLIRPublicAPI), MlirContext, (MlirAttribute,), attribute)
end

function mlirAttributeGetType(attribute)
    ccall((:mlirAttributeGetType, libMLIRPublicAPI), MlirType, (MlirAttribute,), attribute)
end

function mlirAttributeIsNull(attr)
    ccall((:mlirAttributeIsNull, libMLIRPublicAPI), Bool, (MlirAttribute,), attr)
end

function mlirAttributeEqual(a1, a2)
    ccall((:mlirAttributeEqual, libMLIRPublicAPI), Bool, (MlirAttribute, MlirAttribute), a1, a2)
end

function mlirAttributePrint(attr, callback, userData)
    ccall((:mlirAttributePrint, libMLIRPublicAPI), Cvoid, (MlirAttribute, MlirStringCallback, Ptr{Cvoid}), attr, callback, userData)
end

function mlirAttributeDump(attr)
    ccall((:mlirAttributeDump, libMLIRPublicAPI), Cvoid, (MlirAttribute,), attr)
end

function mlirNamedAttributeGet(name, attr)
    ccall((:mlirNamedAttributeGet, libMLIRPublicAPI), MlirNamedAttribute, (MlirIdentifier, MlirAttribute), name, attr)
end

function mlirIdentifierGet(context, str)
    ccall((:mlirIdentifierGet, libMLIRPublicAPI), MlirIdentifier, (MlirContext, MlirStringRef), context, str)
end

function mlirIdentifierEqual(ident, other)
    ccall((:mlirIdentifierEqual, libMLIRPublicAPI), Bool, (MlirIdentifier, MlirIdentifier), ident, other)
end

function mlirIdentifierStr(ident)
    ccall((:mlirIdentifierStr, libMLIRPublicAPI), MlirStringRef, (MlirIdentifier,), ident)
end

struct MlirPass
    ptr::Ptr{Cvoid}
end

struct MlirPassManager
    ptr::Ptr{Cvoid}
end

struct MlirOpPassManager
    ptr::Ptr{Cvoid}
end

function mlirPassManagerCreate(ctx)
    ccall((:mlirPassManagerCreate, libMLIRPublicAPI), MlirPassManager, (MlirContext,), ctx)
end

function mlirPassManagerDestroy(passManager)
    ccall((:mlirPassManagerDestroy, libMLIRPublicAPI), Cvoid, (MlirPassManager,), passManager)
end

function mlirPassManagerIsNull(passManager)
    ccall((:mlirPassManagerIsNull, libMLIRPublicAPI), Bool, (MlirPassManager,), passManager)
end

function mlirPassManagerGetAsOpPassManager(passManager)
    ccall((:mlirPassManagerGetAsOpPassManager, libMLIRPublicAPI), MlirOpPassManager, (MlirPassManager,), passManager)
end

function mlirPassManagerRun(passManager, _module)
    ccall((:mlirPassManagerRun, libMLIRPublicAPI), MlirLogicalResult, (MlirPassManager, MlirModule), passManager, _module)
end

function mlirPassManagerGetNestedUnder(passManager, operationName)
    ccall((:mlirPassManagerGetNestedUnder, libMLIRPublicAPI), MlirOpPassManager, (MlirPassManager, MlirStringRef), passManager, operationName)
end

function mlirOpPassManagerGetNestedUnder(passManager, operationName)
    ccall((:mlirOpPassManagerGetNestedUnder, libMLIRPublicAPI), MlirOpPassManager, (MlirOpPassManager, MlirStringRef), passManager, operationName)
end

function mlirPassManagerAddOwnedPass(passManager, pass)
    ccall((:mlirPassManagerAddOwnedPass, libMLIRPublicAPI), Cvoid, (MlirPassManager, MlirPass), passManager, pass)
end

function mlirOpPassManagerAddOwnedPass(passManager, pass)
    ccall((:mlirOpPassManagerAddOwnedPass, libMLIRPublicAPI), Cvoid, (MlirOpPassManager, MlirPass), passManager, pass)
end

function mlirPrintPassPipeline(passManager, callback, userData)
    ccall((:mlirPrintPassPipeline, libMLIRPublicAPI), Cvoid, (MlirOpPassManager, MlirStringCallback, Ptr{Cvoid}), passManager, callback, userData)
end

function mlirParsePassPipeline(passManager, pipeline)
    ccall((:mlirParsePassPipeline, libMLIRPublicAPI), MlirLogicalResult, (MlirOpPassManager, MlirStringRef), passManager, pipeline)
end

# typedef void ( * MlirContextRegisterDialectHook ) ( MlirContext context )
const MlirContextRegisterDialectHook = Ptr{Cvoid}

# typedef MlirDialect ( * MlirContextLoadDialectHook ) ( MlirContext context )
const MlirContextLoadDialectHook = Ptr{Cvoid}

# typedef MlirStringRef ( * MlirDialectGetNamespaceHook ) ( )
const MlirDialectGetNamespaceHook = Ptr{Cvoid}

struct MlirDialectRegistrationHooks
    registerHook::MlirContextRegisterDialectHook
    loadHook::MlirContextLoadDialectHook
    getNamespaceHook::MlirDialectGetNamespaceHook
end

function mlirRegisterAllDialects(context)
    ccall((:mlirRegisterAllDialects, libMLIRPublicAPI), Cvoid, (MlirContext,), context)
end

function mlirContextRegisterStandardDialect(context)
    ccall((:mlirContextRegisterStandardDialect, libMLIRPublicAPI), Cvoid, (MlirContext,), context)
end

function mlirContextLoadStandardDialect(context)
    ccall((:mlirContextLoadStandardDialect, libMLIRPublicAPI), MlirDialect, (MlirContext,), context)
end

# no prototype is found for this function at Standard.h:27:1, please use with caution
function mlirStandardDialectGetNamespace()
    ccall((:mlirStandardDialectGetNamespace, libMLIRPublicAPI), MlirStringRef, ())
end

# no prototype is found for this function at Standard.h:27:1, please use with caution
function mlirGetDialectHooks__std__()
    ccall((:mlirGetDialectHooks__std__, libMLIRPublicAPI), Ptr{MlirDialectRegistrationHooks}, ())
end

