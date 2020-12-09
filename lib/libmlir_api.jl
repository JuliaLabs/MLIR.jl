# Julia wrapper for header: AffineExpr.h
# Automatically generated using Clang.jl


function mlirAffineExprPrint()
    ccall((:mlirAffineExprPrint, libmlir), Cint, ())
end

function mlirAffineExprDump()
    ccall((:mlirAffineExprDump, libmlir), Cint, ())
end
# Julia wrapper for header: AffineMap.h
# Automatically generated using Clang.jl


function mlirAffineMapIsNull()
    ccall((:mlirAffineMapIsNull, libmlir), Cint, ())
end

function mlirAffineMapPrint()
    ccall((:mlirAffineMapPrint, libmlir), Cint, ())
end

function mlirAffineMapDump()
    ccall((:mlirAffineMapDump, libmlir), Cint, ())
end
# Julia wrapper for header: BuiltinAttributes.h
# Automatically generated using Clang.jl


function mlirFloatAttrGetValueDouble()
    ccall((:mlirFloatAttrGetValueDouble, libmlir), Cint, ())
end

function mlirDenseElementsAttrGetBoolSplatValue()
    ccall((:mlirDenseElementsAttrGetBoolSplatValue, libmlir), Cint, ())
end

function mlirDenseElementsAttrGetFloatSplatValue()
    ccall((:mlirDenseElementsAttrGetFloatSplatValue, libmlir), Cint, ())
end

function mlirDenseElementsAttrGetDoubleSplatValue()
    ccall((:mlirDenseElementsAttrGetDoubleSplatValue, libmlir), Cint, ())
end

function mlirDenseElementsAttrGetFloatValue()
    ccall((:mlirDenseElementsAttrGetFloatValue, libmlir), Cint, ())
end

function mlirDenseElementsAttrGetDoubleValue()
    ccall((:mlirDenseElementsAttrGetDoubleValue, libmlir), Cint, ())
end

function mlirDenseElementsAttrGetRawData()
    ccall((:mlirDenseElementsAttrGetRawData, libmlir), Ptr{Cint}, ())
end
# Julia wrapper for header: BuiltinTypes.h
# Automatically generated using Clang.jl


function mlirIntegerTypeGetWidth()
    ccall((:mlirIntegerTypeGetWidth, libmlir), Cint, ())
end

function mlirMemRefTypeGetMemorySpace()
    ccall((:mlirMemRefTypeGetMemorySpace, libmlir), Cint, ())
end

function mlirUnrankedMemrefGetMemorySpace()
    ccall((:mlirUnrankedMemrefGetMemorySpace, libmlir), Cint, ())
end
# Julia wrapper for header: Diagnostics.h
# Automatically generated using Clang.jl


function mlirDiagnosticPrint()
    ccall((:mlirDiagnosticPrint, libmlir), Cint, ())
end

function mlirContextDetachDiagnosticHandler()
    ccall((:mlirContextDetachDiagnosticHandler, libmlir), Cint, ())
end

function mlirEmitError()
    ccall((:mlirEmitError, libmlir), Cint, ())
end
# Julia wrapper for header: IR.h
# Automatically generated using Clang.jl


function mlirContextEqual()
    ccall((:mlirContextEqual, libmlir), Cint, ())
end

function mlirContextIsNull(context)
    ccall((:mlirContextIsNull, libmlir), Bool, (MlirContext,), context)
end

function mlirContextDestroy()
    ccall((:mlirContextDestroy, libmlir), Cint, ())
end

function mlirContextSetAllowUnregisteredDialects()
    ccall((:mlirContextSetAllowUnregisteredDialects, libmlir), Cint, ())
end

function mlirContextGetAllowUnregisteredDialects()
    ccall((:mlirContextGetAllowUnregisteredDialects, libmlir), Cint, ())
end

function mlirDialectIsNull(dialect)
    ccall((:mlirDialectIsNull, libmlir), Bool, (MlirDialect,), dialect)
end

function mlirDialectEqual()
    ccall((:mlirDialectEqual, libmlir), Cint, ())
end

function mlirLocationIsNull(location)
    ccall((:mlirLocationIsNull, libmlir), Bool, (MlirLocation,), location)
end

function mlirLocationEqual()
    ccall((:mlirLocationEqual, libmlir), Cint, ())
end

function mlirLocationPrint()
    ccall((:mlirLocationPrint, libmlir), Cint, ())
end

function mlirModuleIsNull(_module)
    ccall((:mlirModuleIsNull, libmlir), Bool, (MlirModule,), _module)
end

function mlirModuleDestroy()
    ccall((:mlirModuleDestroy, libmlir), Cint, ())
end

function mlirOperationStateAddResults()
    ccall((:mlirOperationStateAddResults, libmlir), Cint, ())
end

function mlirOperationStateAddOperands()
    ccall((:mlirOperationStateAddOperands, libmlir), Cint, ())
end

function mlirOperationStateAddOwnedRegions()
    ccall((:mlirOperationStateAddOwnedRegions, libmlir), Cint, ())
end

function mlirOperationStateAddSuccessors()
    ccall((:mlirOperationStateAddSuccessors, libmlir), Cint, ())
end

function mlirOperationStateAddAttributes()
    ccall((:mlirOperationStateAddAttributes, libmlir), Cint, ())
end

function mlirOpPrintingFlagsDestroy()
    ccall((:mlirOpPrintingFlagsDestroy, libmlir), Cint, ())
end

function mlirOpPrintingFlagsElideLargeElementsAttrs()
    ccall((:mlirOpPrintingFlagsElideLargeElementsAttrs, libmlir), Cint, ())
end

function mlirOpPrintingFlagsEnableDebugInfo()
    ccall((:mlirOpPrintingFlagsEnableDebugInfo, libmlir), Cint, ())
end

function mlirOpPrintingFlagsPrintGenericOpForm()
    ccall((:mlirOpPrintingFlagsPrintGenericOpForm, libmlir), Cint, ())
end

function mlirOpPrintingFlagsUseLocalScope()
    ccall((:mlirOpPrintingFlagsUseLocalScope, libmlir), Cint, ())
end

function mlirOperationDestroy()
    ccall((:mlirOperationDestroy, libmlir), Cint, ())
end

function mlirOperationIsNull(op)
    ccall((:mlirOperationIsNull, libmlir), Bool, (MlirOperation,), op)
end

function mlirOperationEqual()
    ccall((:mlirOperationEqual, libmlir), Cint, ())
end

function mlirOperationSetAttributeByName()
    ccall((:mlirOperationSetAttributeByName, libmlir), Cint, ())
end

function mlirOperationRemoveAttributeByName()
    ccall((:mlirOperationRemoveAttributeByName, libmlir), Cint, ())
end

function mlirOperationPrint()
    ccall((:mlirOperationPrint, libmlir), Cint, ())
end

function mlirOperationPrintWithFlags()
    ccall((:mlirOperationPrintWithFlags, libmlir), Cint, ())
end

function mlirOperationDump()
    ccall((:mlirOperationDump, libmlir), Cint, ())
end

function mlirOperationVerify()
    ccall((:mlirOperationVerify, libmlir), Cint, ())
end

function mlirRegionDestroy()
    ccall((:mlirRegionDestroy, libmlir), Cint, ())
end

function mlirRegionIsNull(region)
    ccall((:mlirRegionIsNull, libmlir), Bool, (MlirRegion,), region)
end

function mlirRegionAppendOwnedBlock()
    ccall((:mlirRegionAppendOwnedBlock, libmlir), Cint, ())
end

function mlirRegionInsertOwnedBlock()
    ccall((:mlirRegionInsertOwnedBlock, libmlir), Cint, ())
end

function mlirRegionInsertOwnedBlockAfter()
    ccall((:mlirRegionInsertOwnedBlockAfter, libmlir), Cint, ())
end

function mlirRegionInsertOwnedBlockBefore()
    ccall((:mlirRegionInsertOwnedBlockBefore, libmlir), Cint, ())
end

function mlirBlockDestroy()
    ccall((:mlirBlockDestroy, libmlir), Cint, ())
end

function mlirBlockIsNull(block)
    ccall((:mlirBlockIsNull, libmlir), Bool, (MlirBlock,), block)
end

function mlirBlockEqual()
    ccall((:mlirBlockEqual, libmlir), Cint, ())
end

function mlirBlockAppendOwnedOperation()
    ccall((:mlirBlockAppendOwnedOperation, libmlir), Cint, ())
end

function mlirBlockInsertOwnedOperation()
    ccall((:mlirBlockInsertOwnedOperation, libmlir), Cint, ())
end

function mlirBlockInsertOwnedOperationAfter()
    ccall((:mlirBlockInsertOwnedOperationAfter, libmlir), Cint, ())
end

function mlirBlockInsertOwnedOperationBefore()
    ccall((:mlirBlockInsertOwnedOperationBefore, libmlir), Cint, ())
end

function mlirBlockPrint()
    ccall((:mlirBlockPrint, libmlir), Cint, ())
end

function mlirValueIsNull(value)
    ccall((:mlirValueIsNull, libmlir), Bool, (MlirValue,), value)
end

function mlirValueEqual(value1, value2)
    ccall((:mlirValueEqual, libmlir), Bool, (MlirValue, MlirValue), value1, value2)
end

function mlirValueIsABlockArgument()
    ccall((:mlirValueIsABlockArgument, libmlir), Cint, ())
end

function mlirValueIsAOpResult()
    ccall((:mlirValueIsAOpResult, libmlir), Cint, ())
end

function mlirBlockArgumentSetType()
    ccall((:mlirBlockArgumentSetType, libmlir), Cint, ())
end

function mlirValueDump()
    ccall((:mlirValueDump, libmlir), Cint, ())
end

function mlirValuePrint()
    ccall((:mlirValuePrint, libmlir), Cint, ())
end

function mlirTypeIsNull(type)
    ccall((:mlirTypeIsNull, libmlir), Bool, (MlirType,), type)
end

function mlirTypeEqual()
    ccall((:mlirTypeEqual, libmlir), Cint, ())
end

function mlirTypePrint()
    ccall((:mlirTypePrint, libmlir), Cint, ())
end

function mlirTypeDump()
    ccall((:mlirTypeDump, libmlir), Cint, ())
end

function mlirAttributeIsNull(attr)
    ccall((:mlirAttributeIsNull, libmlir), Bool, (MlirAttribute,), attr)
end

function mlirAttributeEqual()
    ccall((:mlirAttributeEqual, libmlir), Cint, ())
end

function mlirAttributePrint()
    ccall((:mlirAttributePrint, libmlir), Cint, ())
end

function mlirAttributeDump()
    ccall((:mlirAttributeDump, libmlir), Cint, ())
end

function mlirIdentifierEqual()
    ccall((:mlirIdentifierEqual, libmlir), Cint, ())
end
# Julia wrapper for header: Pass.h
# Automatically generated using Clang.jl


function mlirPassManagerDestroy()
    ccall((:mlirPassManagerDestroy, libmlir), Cint, ())
end

function mlirPassManagerIsNull()
    ccall((:mlirPassManagerIsNull, libmlir), Cint, ())
end

function mlirPassManagerAddOwnedPass()
    ccall((:mlirPassManagerAddOwnedPass, libmlir), Cint, ())
end

function mlirOpPassManagerAddOwnedPass()
    ccall((:mlirOpPassManagerAddOwnedPass, libmlir), Cint, ())
end

function mlirPrintPassPipeline()
    ccall((:mlirPrintPassPipeline, libmlir), Cint, ())
end
# Julia wrapper for header: Registration.h
# Automatically generated using Clang.jl


function mlirRegisterAllDialects()
    ccall((:mlirRegisterAllDialects, libmlir), Cint, ())
end
# Julia wrapper for header: StandardDialect.h
# Automatically generated using Clang.jl


function mlirContextRegisterStandardDialect()
    ccall((:mlirContextRegisterStandardDialect, libmlir), Cint, ())
end
# Julia wrapper for header: Support.h
# Automatically generated using Clang.jl


function mlirStringRefCreate(str, length)
    ccall((:mlirStringRefCreate, libmlir), MlirStringRef, (Cstring, Csize_t), str, length)
end

function mlirStringRefCreateFromCString(str)
    ccall((:mlirStringRefCreateFromCString, libmlir), MlirStringRef, (Cstring,), str)
end

function mlirLogicalResultIsSuccess()
    ccall((:mlirLogicalResultIsSuccess, libmlir), Cint, ())
end

function mlirLogicalResultIsFailure()
    ccall((:mlirLogicalResultIsFailure, libmlir), Cint, ())
end

function mlirLogicalResultSuccess()
    ccall((:mlirLogicalResultSuccess, libmlir), MlirLogicalResult, ())
end

function mlirLogicalResultFailure()
    ccall((:mlirLogicalResultFailure, libmlir), MlirLogicalResult, ())
end
# Julia wrapper for header: Transforms.h
# Automatically generated using Clang.jl

