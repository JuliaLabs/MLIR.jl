# Julia wrapper for header: AffineExpr.h
# Automatically generated using Clang.jl


function mlirAffineExprGetContext(affineExpr)
    ccall((:mlirAffineExprGetContext, libmlir), MlirContext, (MlirAffineExpr,), affineExpr)
end

function mlirAffineExprPrint(affineExpr, callback, userData)
    ccall((:mlirAffineExprPrint, libmlir), Cvoid, (MlirAffineExpr, MlirStringCallback, Ptr{Cvoid}), affineExpr, callback, userData)
end

function mlirAffineExprDump(affineExpr)
    ccall((:mlirAffineExprDump, libmlir), Cvoid, (MlirAffineExpr,), affineExpr)
end

function mlirAffineExprIsSymbolicOrConstant(affineExpr)
    ccall((:mlirAffineExprIsSymbolicOrConstant, libmlir), Bool, (MlirAffineExpr,), affineExpr)
end

function mlirAffineExprIsPureAffine(affineExpr)
    ccall((:mlirAffineExprIsPureAffine, libmlir), Bool, (MlirAffineExpr,), affineExpr)
end

function mlirAffineExprGetLargestKnownDivisor(affineExpr)
    ccall((:mlirAffineExprGetLargestKnownDivisor, libmlir), Int64, (MlirAffineExpr,), affineExpr)
end

function mlirAffineExprIsMultipleOf(affineExpr, factor)
    ccall((:mlirAffineExprIsMultipleOf, libmlir), Bool, (MlirAffineExpr, Int64), affineExpr, factor)
end

function mlirAffineExprIsFunctionOfDim(affineExpr, position)
    ccall((:mlirAffineExprIsFunctionOfDim, libmlir), Bool, (MlirAffineExpr, intptr_t), affineExpr, position)
end

function mlirAffineDimExprGet(ctx, position)
    ccall((:mlirAffineDimExprGet, libmlir), MlirAffineExpr, (MlirContext, intptr_t), ctx, position)
end

function mlirAffineDimExprGetPosition(affineExpr)
    ccall((:mlirAffineDimExprGetPosition, libmlir), intptr_t, (MlirAffineExpr,), affineExpr)
end

function mlirAffineSymbolExprGet(ctx, position)
    ccall((:mlirAffineSymbolExprGet, libmlir), MlirAffineExpr, (MlirContext, intptr_t), ctx, position)
end

function mlirAffineSymbolExprGetPosition(affineExpr)
    ccall((:mlirAffineSymbolExprGetPosition, libmlir), intptr_t, (MlirAffineExpr,), affineExpr)
end

function mlirAffineConstantExprGet(ctx, constant)
    ccall((:mlirAffineConstantExprGet, libmlir), MlirAffineExpr, (MlirContext, Int64), ctx, constant)
end

function mlirAffineConstantExprGetValue(affineExpr)
    ccall((:mlirAffineConstantExprGetValue, libmlir), Int64, (MlirAffineExpr,), affineExpr)
end

function mlirAffineExprIsAAdd(affineExpr)
    ccall((:mlirAffineExprIsAAdd, libmlir), Bool, (MlirAffineExpr,), affineExpr)
end

function mlirAffineAddExprGet(lhs, rhs)
    ccall((:mlirAffineAddExprGet, libmlir), MlirAffineExpr, (MlirAffineExpr, MlirAffineExpr), lhs, rhs)
end

function mlirAffineExprIsAMul(affineExpr)
    ccall((:mlirAffineExprIsAMul, libmlir), Bool, (MlirAffineExpr,), affineExpr)
end

function mlirAffineMulExprGet(lhs, rhs)
    ccall((:mlirAffineMulExprGet, libmlir), MlirAffineExpr, (MlirAffineExpr, MlirAffineExpr), lhs, rhs)
end

function mlirAffineExprIsAMod(affineExpr)
    ccall((:mlirAffineExprIsAMod, libmlir), Bool, (MlirAffineExpr,), affineExpr)
end

function mlirAffineModExprGet(lhs, rhs)
    ccall((:mlirAffineModExprGet, libmlir), MlirAffineExpr, (MlirAffineExpr, MlirAffineExpr), lhs, rhs)
end

function mlirAffineExprIsAFloorDiv(affineExpr)
    ccall((:mlirAffineExprIsAFloorDiv, libmlir), Bool, (MlirAffineExpr,), affineExpr)
end

function mlirAffineFloorDivExprGet(lhs, rhs)
    ccall((:mlirAffineFloorDivExprGet, libmlir), MlirAffineExpr, (MlirAffineExpr, MlirAffineExpr), lhs, rhs)
end

function mlirAffineExprIsACeilDiv(affineExpr)
    ccall((:mlirAffineExprIsACeilDiv, libmlir), Bool, (MlirAffineExpr,), affineExpr)
end

function mlirAffineCeilDivExprGet(lhs, rhs)
    ccall((:mlirAffineCeilDivExprGet, libmlir), MlirAffineExpr, (MlirAffineExpr, MlirAffineExpr), lhs, rhs)
end

function mlirAffineBinaryOpExprGetLHS(affineExpr)
    ccall((:mlirAffineBinaryOpExprGetLHS, libmlir), MlirAffineExpr, (MlirAffineExpr,), affineExpr)
end

function mlirAffineBinaryOpExprGetRHS(affineExpr)
    ccall((:mlirAffineBinaryOpExprGetRHS, libmlir), MlirAffineExpr, (MlirAffineExpr,), affineExpr)
end
# Julia wrapper for header: AffineMap.h
# Automatically generated using Clang.jl


function mlirAffineMapGetContext(affineMap)
    ccall((:mlirAffineMapGetContext, libmlir), MlirContext, (MlirAffineMap,), affineMap)
end

function mlirAffineMapIsNull(affineMap)
    ccall((:mlirAffineMapIsNull, libmlir), Bool, (MlirAffineMap,), affineMap)
end

function mlirAffineMapEqual(a1, a2)
    ccall((:mlirAffineMapEqual, libmlir), Bool, (MlirAffineMap, MlirAffineMap), a1, a2)
end

function mlirAffineMapPrint(affineMap, callback, userData)
    ccall((:mlirAffineMapPrint, libmlir), Cvoid, (MlirAffineMap, MlirStringCallback, Ptr{Cvoid}), affineMap, callback, userData)
end

function mlirAffineMapDump(affineMap)
    ccall((:mlirAffineMapDump, libmlir), Cvoid, (MlirAffineMap,), affineMap)
end

function mlirAffineMapEmptyGet(ctx)
    ccall((:mlirAffineMapEmptyGet, libmlir), MlirAffineMap, (MlirContext,), ctx)
end

function mlirAffineMapGet(ctx, dimCount, symbolCount)
    ccall((:mlirAffineMapGet, libmlir), MlirAffineMap, (MlirContext, intptr_t, intptr_t), ctx, dimCount, symbolCount)
end

function mlirAffineMapConstantGet(ctx, val)
    ccall((:mlirAffineMapConstantGet, libmlir), MlirAffineMap, (MlirContext, Int64), ctx, val)
end

function mlirAffineMapMultiDimIdentityGet(ctx, numDims)
    ccall((:mlirAffineMapMultiDimIdentityGet, libmlir), MlirAffineMap, (MlirContext, intptr_t), ctx, numDims)
end

function mlirAffineMapMinorIdentityGet(ctx, dims, results)
    ccall((:mlirAffineMapMinorIdentityGet, libmlir), MlirAffineMap, (MlirContext, intptr_t, intptr_t), ctx, dims, results)
end

function mlirAffineMapPermutationGet(ctx, size, permutation)
    ccall((:mlirAffineMapPermutationGet, libmlir), MlirAffineMap, (MlirContext, intptr_t, Ptr{UInt32}), ctx, size, permutation)
end

function mlirAffineMapIsIdentity(affineMap)
    ccall((:mlirAffineMapIsIdentity, libmlir), Bool, (MlirAffineMap,), affineMap)
end

function mlirAffineMapIsMinorIdentity(affineMap)
    ccall((:mlirAffineMapIsMinorIdentity, libmlir), Bool, (MlirAffineMap,), affineMap)
end

function mlirAffineMapIsEmpty(affineMap)
    ccall((:mlirAffineMapIsEmpty, libmlir), Bool, (MlirAffineMap,), affineMap)
end

function mlirAffineMapIsSingleConstant(affineMap)
    ccall((:mlirAffineMapIsSingleConstant, libmlir), Bool, (MlirAffineMap,), affineMap)
end

function mlirAffineMapGetSingleConstantResult(affineMap)
    ccall((:mlirAffineMapGetSingleConstantResult, libmlir), Int64, (MlirAffineMap,), affineMap)
end

function mlirAffineMapGetNumDims(affineMap)
    ccall((:mlirAffineMapGetNumDims, libmlir), intptr_t, (MlirAffineMap,), affineMap)
end

function mlirAffineMapGetNumSymbols(affineMap)
    ccall((:mlirAffineMapGetNumSymbols, libmlir), intptr_t, (MlirAffineMap,), affineMap)
end

function mlirAffineMapGetNumResults(affineMap)
    ccall((:mlirAffineMapGetNumResults, libmlir), intptr_t, (MlirAffineMap,), affineMap)
end

function mlirAffineMapGetNumInputs(affineMap)
    ccall((:mlirAffineMapGetNumInputs, libmlir), intptr_t, (MlirAffineMap,), affineMap)
end

function mlirAffineMapIsProjectedPermutation(affineMap)
    ccall((:mlirAffineMapIsProjectedPermutation, libmlir), Bool, (MlirAffineMap,), affineMap)
end

function mlirAffineMapIsPermutation(affineMap)
    ccall((:mlirAffineMapIsPermutation, libmlir), Bool, (MlirAffineMap,), affineMap)
end

function mlirAffineMapGetSubMap(affineMap, size, resultPos)
    ccall((:mlirAffineMapGetSubMap, libmlir), MlirAffineMap, (MlirAffineMap, intptr_t, Ptr{intptr_t}), affineMap, size, resultPos)
end

function mlirAffineMapGetMajorSubMap(affineMap, numResults)
    ccall((:mlirAffineMapGetMajorSubMap, libmlir), MlirAffineMap, (MlirAffineMap, intptr_t), affineMap, numResults)
end

function mlirAffineMapGetMinorSubMap(affineMap, numResults)
    ccall((:mlirAffineMapGetMinorSubMap, libmlir), MlirAffineMap, (MlirAffineMap, intptr_t), affineMap, numResults)
end
# Julia wrapper for header: BuiltinAttributes.h
# Automatically generated using Clang.jl


function mlirAttributeIsAAffineMap(attr)
    ccall((:mlirAttributeIsAAffineMap, libmlir), Bool, (MlirAttribute,), attr)
end

function mlirAffineMapAttrGet(map)
    ccall((:mlirAffineMapAttrGet, libmlir), MlirAttribute, (MlirAffineMap,), map)
end

function mlirAffineMapAttrGetValue(attr)
    ccall((:mlirAffineMapAttrGetValue, libmlir), MlirAffineMap, (MlirAttribute,), attr)
end

function mlirAttributeIsAArray(attr)
    ccall((:mlirAttributeIsAArray, libmlir), Bool, (MlirAttribute,), attr)
end

function mlirArrayAttrGet(ctx, numElements, elements)
    ccall((:mlirArrayAttrGet, libmlir), MlirAttribute, (MlirContext, intptr_t, Ptr{MlirAttribute}), ctx, numElements, elements)
end

function mlirArrayAttrGetNumElements(attr)
    ccall((:mlirArrayAttrGetNumElements, libmlir), intptr_t, (MlirAttribute,), attr)
end

function mlirArrayAttrGetElement(attr, pos)
    ccall((:mlirArrayAttrGetElement, libmlir), MlirAttribute, (MlirAttribute, intptr_t), attr, pos)
end

function mlirAttributeIsADictionary(attr)
    ccall((:mlirAttributeIsADictionary, libmlir), Bool, (MlirAttribute,), attr)
end

function mlirDictionaryAttrGet(ctx, numElements, elements)
    ccall((:mlirDictionaryAttrGet, libmlir), MlirAttribute, (MlirContext, intptr_t, Ptr{MlirNamedAttribute}), ctx, numElements, elements)
end

function mlirDictionaryAttrGetNumElements(attr)
    ccall((:mlirDictionaryAttrGetNumElements, libmlir), intptr_t, (MlirAttribute,), attr)
end

function mlirDictionaryAttrGetElement(attr, pos)
    ccall((:mlirDictionaryAttrGetElement, libmlir), MlirNamedAttribute, (MlirAttribute, intptr_t), attr, pos)
end

function mlirDictionaryAttrGetElementByName(attr, name)
    ccall((:mlirDictionaryAttrGetElementByName, libmlir), MlirAttribute, (MlirAttribute, MlirStringRef), attr, name)
end

function mlirAttributeIsAFloat(attr)
    ccall((:mlirAttributeIsAFloat, libmlir), Bool, (MlirAttribute,), attr)
end

function mlirFloatAttrDoubleGet(ctx, type, value)
    ccall((:mlirFloatAttrDoubleGet, libmlir), MlirAttribute, (MlirContext, MlirType, Cdouble), ctx, type, value)
end

function mlirFloatAttrDoubleGetChecked(type, value, loc)
    ccall((:mlirFloatAttrDoubleGetChecked, libmlir), MlirAttribute, (MlirType, Cdouble, MlirLocation), type, value, loc)
end

function mlirFloatAttrGetValueDouble(attr)
    ccall((:mlirFloatAttrGetValueDouble, libmlir), Cdouble, (MlirAttribute,), attr)
end

function mlirAttributeIsAInteger(attr)
    ccall((:mlirAttributeIsAInteger, libmlir), Bool, (MlirAttribute,), attr)
end

function mlirIntegerAttrGet(type, value)
    ccall((:mlirIntegerAttrGet, libmlir), MlirAttribute, (MlirType, Int64), type, value)
end

function mlirIntegerAttrGetValueInt(attr)
    ccall((:mlirIntegerAttrGetValueInt, libmlir), Int64, (MlirAttribute,), attr)
end

function mlirAttributeIsABool(attr)
    ccall((:mlirAttributeIsABool, libmlir), Bool, (MlirAttribute,), attr)
end

function mlirBoolAttrGet(ctx, value)
    ccall((:mlirBoolAttrGet, libmlir), MlirAttribute, (MlirContext, Cint), ctx, value)
end

function mlirBoolAttrGetValue(attr)
    ccall((:mlirBoolAttrGetValue, libmlir), Bool, (MlirAttribute,), attr)
end

function mlirAttributeIsAIntegerSet(attr)
    ccall((:mlirAttributeIsAIntegerSet, libmlir), Bool, (MlirAttribute,), attr)
end

function mlirAttributeIsAOpaque(attr)
    ccall((:mlirAttributeIsAOpaque, libmlir), Bool, (MlirAttribute,), attr)
end

function mlirOpaqueAttrGet(ctx, dialectNamespace, dataLength, data, type)
    ccall((:mlirOpaqueAttrGet, libmlir), MlirAttribute, (MlirContext, MlirStringRef, intptr_t, Cstring, MlirType), ctx, dialectNamespace, dataLength, data, type)
end

function mlirOpaqueAttrGetDialectNamespace(attr)
    ccall((:mlirOpaqueAttrGetDialectNamespace, libmlir), MlirStringRef, (MlirAttribute,), attr)
end

function mlirOpaqueAttrGetData(attr)
    ccall((:mlirOpaqueAttrGetData, libmlir), MlirStringRef, (MlirAttribute,), attr)
end

function mlirAttributeIsAString(attr)
    ccall((:mlirAttributeIsAString, libmlir), Bool, (MlirAttribute,), attr)
end

function mlirStringAttrGet(ctx, str)
    ccall((:mlirStringAttrGet, libmlir), MlirAttribute, (MlirContext, MlirStringRef), ctx, str)
end

function mlirStringAttrTypedGet(type, str)
    ccall((:mlirStringAttrTypedGet, libmlir), MlirAttribute, (MlirType, MlirStringRef), type, str)
end

function mlirStringAttrGetValue(attr)
    ccall((:mlirStringAttrGetValue, libmlir), MlirStringRef, (MlirAttribute,), attr)
end

function mlirAttributeIsASymbolRef(attr)
    ccall((:mlirAttributeIsASymbolRef, libmlir), Bool, (MlirAttribute,), attr)
end

function mlirSymbolRefAttrGet(ctx, symbol, numReferences, references)
    ccall((:mlirSymbolRefAttrGet, libmlir), MlirAttribute, (MlirContext, MlirStringRef, intptr_t, Ptr{MlirAttribute}), ctx, symbol, numReferences, references)
end

function mlirSymbolRefAttrGetRootReference(attr)
    ccall((:mlirSymbolRefAttrGetRootReference, libmlir), MlirStringRef, (MlirAttribute,), attr)
end

function mlirSymbolRefAttrGetLeafReference(attr)
    ccall((:mlirSymbolRefAttrGetLeafReference, libmlir), MlirStringRef, (MlirAttribute,), attr)
end

function mlirSymbolRefAttrGetNumNestedReferences(attr)
    ccall((:mlirSymbolRefAttrGetNumNestedReferences, libmlir), intptr_t, (MlirAttribute,), attr)
end

function mlirSymbolRefAttrGetNestedReference(attr, pos)
    ccall((:mlirSymbolRefAttrGetNestedReference, libmlir), MlirAttribute, (MlirAttribute, intptr_t), attr, pos)
end

function mlirAttributeIsAFlatSymbolRef(attr)
    ccall((:mlirAttributeIsAFlatSymbolRef, libmlir), Bool, (MlirAttribute,), attr)
end

function mlirFlatSymbolRefAttrGet(ctx, symbol)
    ccall((:mlirFlatSymbolRefAttrGet, libmlir), MlirAttribute, (MlirContext, MlirStringRef), ctx, symbol)
end

function mlirFlatSymbolRefAttrGetValue(attr)
    ccall((:mlirFlatSymbolRefAttrGetValue, libmlir), MlirStringRef, (MlirAttribute,), attr)
end

function mlirAttributeIsAType(attr)
    ccall((:mlirAttributeIsAType, libmlir), Bool, (MlirAttribute,), attr)
end

function mlirTypeAttrGet(type)
    ccall((:mlirTypeAttrGet, libmlir), MlirAttribute, (MlirType,), type)
end

function mlirTypeAttrGetValue(attr)
    ccall((:mlirTypeAttrGetValue, libmlir), MlirType, (MlirAttribute,), attr)
end

function mlirAttributeIsAUnit(attr)
    ccall((:mlirAttributeIsAUnit, libmlir), Bool, (MlirAttribute,), attr)
end

function mlirUnitAttrGet(ctx)
    ccall((:mlirUnitAttrGet, libmlir), MlirAttribute, (MlirContext,), ctx)
end

function mlirAttributeIsAElements(attr)
    ccall((:mlirAttributeIsAElements, libmlir), Bool, (MlirAttribute,), attr)
end

function mlirElementsAttrGetValue(attr, rank, idxs)
    ccall((:mlirElementsAttrGetValue, libmlir), MlirAttribute, (MlirAttribute, intptr_t, Ptr{UInt64}), attr, rank, idxs)
end

function mlirElementsAttrIsValidIndex(attr, rank, idxs)
    ccall((:mlirElementsAttrIsValidIndex, libmlir), Bool, (MlirAttribute, intptr_t, Ptr{UInt64}), attr, rank, idxs)
end

function mlirElementsAttrGetNumElements(attr)
    ccall((:mlirElementsAttrGetNumElements, libmlir), Int64, (MlirAttribute,), attr)
end

function mlirAttributeIsADenseElements(attr)
    ccall((:mlirAttributeIsADenseElements, libmlir), Bool, (MlirAttribute,), attr)
end

function mlirAttributeIsADenseIntElements(attr)
    ccall((:mlirAttributeIsADenseIntElements, libmlir), Bool, (MlirAttribute,), attr)
end

function mlirAttributeIsADenseFPElements(attr)
    ccall((:mlirAttributeIsADenseFPElements, libmlir), Bool, (MlirAttribute,), attr)
end

function mlirDenseElementsAttrGet(shapedType, numElements, elements)
    ccall((:mlirDenseElementsAttrGet, libmlir), MlirAttribute, (MlirType, intptr_t, Ptr{MlirAttribute}), shapedType, numElements, elements)
end

function mlirDenseElementsAttrSplatGet(shapedType, element)
    ccall((:mlirDenseElementsAttrSplatGet, libmlir), MlirAttribute, (MlirType, MlirAttribute), shapedType, element)
end

function mlirDenseElementsAttrBoolSplatGet(shapedType, element)
    ccall((:mlirDenseElementsAttrBoolSplatGet, libmlir), MlirAttribute, (MlirType, Bool), shapedType, element)
end

function mlirDenseElementsAttrUInt32SplatGet(shapedType, element)
    ccall((:mlirDenseElementsAttrUInt32SplatGet, libmlir), MlirAttribute, (MlirType, UInt32), shapedType, element)
end

function mlirDenseElementsAttrInt32SplatGet(shapedType, element)
    ccall((:mlirDenseElementsAttrInt32SplatGet, libmlir), MlirAttribute, (MlirType, Int32), shapedType, element)
end

function mlirDenseElementsAttrUInt64SplatGet(shapedType, element)
    ccall((:mlirDenseElementsAttrUInt64SplatGet, libmlir), MlirAttribute, (MlirType, UInt64), shapedType, element)
end

function mlirDenseElementsAttrInt64SplatGet(shapedType, element)
    ccall((:mlirDenseElementsAttrInt64SplatGet, libmlir), MlirAttribute, (MlirType, Int64), shapedType, element)
end

function mlirDenseElementsAttrFloatSplatGet(shapedType, element)
    ccall((:mlirDenseElementsAttrFloatSplatGet, libmlir), MlirAttribute, (MlirType, Cfloat), shapedType, element)
end

function mlirDenseElementsAttrDoubleSplatGet(shapedType, element)
    ccall((:mlirDenseElementsAttrDoubleSplatGet, libmlir), MlirAttribute, (MlirType, Cdouble), shapedType, element)
end

function mlirDenseElementsAttrBoolGet(shapedType, numElements, elements)
    ccall((:mlirDenseElementsAttrBoolGet, libmlir), MlirAttribute, (MlirType, intptr_t, Ptr{Cint}), shapedType, numElements, elements)
end

function mlirDenseElementsAttrUInt32Get(shapedType, numElements, elements)
    ccall((:mlirDenseElementsAttrUInt32Get, libmlir), MlirAttribute, (MlirType, intptr_t, Ptr{UInt32}), shapedType, numElements, elements)
end

function mlirDenseElementsAttrInt32Get(shapedType, numElements, elements)
    ccall((:mlirDenseElementsAttrInt32Get, libmlir), MlirAttribute, (MlirType, intptr_t, Ptr{Int32}), shapedType, numElements, elements)
end

function mlirDenseElementsAttrUInt64Get(shapedType, numElements, elements)
    ccall((:mlirDenseElementsAttrUInt64Get, libmlir), MlirAttribute, (MlirType, intptr_t, Ptr{UInt64}), shapedType, numElements, elements)
end

function mlirDenseElementsAttrInt64Get(shapedType, numElements, elements)
    ccall((:mlirDenseElementsAttrInt64Get, libmlir), MlirAttribute, (MlirType, intptr_t, Ptr{Int64}), shapedType, numElements, elements)
end

function mlirDenseElementsAttrFloatGet(shapedType, numElements, elements)
    ccall((:mlirDenseElementsAttrFloatGet, libmlir), MlirAttribute, (MlirType, intptr_t, Ptr{Cfloat}), shapedType, numElements, elements)
end

function mlirDenseElementsAttrDoubleGet(shapedType, numElements, elements)
    ccall((:mlirDenseElementsAttrDoubleGet, libmlir), MlirAttribute, (MlirType, intptr_t, Ptr{Cdouble}), shapedType, numElements, elements)
end

function mlirDenseElementsAttrStringGet(shapedType, numElements, strs)
    ccall((:mlirDenseElementsAttrStringGet, libmlir), MlirAttribute, (MlirType, intptr_t, Ptr{MlirStringRef}), shapedType, numElements, strs)
end

function mlirDenseElementsAttrReshapeGet(attr, shapedType)
    ccall((:mlirDenseElementsAttrReshapeGet, libmlir), MlirAttribute, (MlirAttribute, MlirType), attr, shapedType)
end

function mlirDenseElementsAttrIsSplat(attr)
    ccall((:mlirDenseElementsAttrIsSplat, libmlir), Bool, (MlirAttribute,), attr)
end

function mlirDenseElementsAttrGetSplatValue(attr)
    ccall((:mlirDenseElementsAttrGetSplatValue, libmlir), MlirAttribute, (MlirAttribute,), attr)
end

function mlirDenseElementsAttrGetBoolSplatValue(attr)
    ccall((:mlirDenseElementsAttrGetBoolSplatValue, libmlir), Cint, (MlirAttribute,), attr)
end

function mlirDenseElementsAttrGetInt32SplatValue(attr)
    ccall((:mlirDenseElementsAttrGetInt32SplatValue, libmlir), Int32, (MlirAttribute,), attr)
end

function mlirDenseElementsAttrGetUInt32SplatValue(attr)
    ccall((:mlirDenseElementsAttrGetUInt32SplatValue, libmlir), UInt32, (MlirAttribute,), attr)
end

function mlirDenseElementsAttrGetInt64SplatValue(attr)
    ccall((:mlirDenseElementsAttrGetInt64SplatValue, libmlir), Int64, (MlirAttribute,), attr)
end

function mlirDenseElementsAttrGetUInt64SplatValue(attr)
    ccall((:mlirDenseElementsAttrGetUInt64SplatValue, libmlir), UInt64, (MlirAttribute,), attr)
end

function mlirDenseElementsAttrGetFloatSplatValue(attr)
    ccall((:mlirDenseElementsAttrGetFloatSplatValue, libmlir), Cfloat, (MlirAttribute,), attr)
end

function mlirDenseElementsAttrGetDoubleSplatValue(attr)
    ccall((:mlirDenseElementsAttrGetDoubleSplatValue, libmlir), Cdouble, (MlirAttribute,), attr)
end

function mlirDenseElementsAttrGetStringSplatValue(attr)
    ccall((:mlirDenseElementsAttrGetStringSplatValue, libmlir), MlirStringRef, (MlirAttribute,), attr)
end

function mlirDenseElementsAttrGetBoolValue(attr, pos)
    ccall((:mlirDenseElementsAttrGetBoolValue, libmlir), Bool, (MlirAttribute, intptr_t), attr, pos)
end

function mlirDenseElementsAttrGetInt32Value(attr, pos)
    ccall((:mlirDenseElementsAttrGetInt32Value, libmlir), Int32, (MlirAttribute, intptr_t), attr, pos)
end

function mlirDenseElementsAttrGetUInt32Value(attr, pos)
    ccall((:mlirDenseElementsAttrGetUInt32Value, libmlir), UInt32, (MlirAttribute, intptr_t), attr, pos)
end

function mlirDenseElementsAttrGetInt64Value(attr, pos)
    ccall((:mlirDenseElementsAttrGetInt64Value, libmlir), Int64, (MlirAttribute, intptr_t), attr, pos)
end

function mlirDenseElementsAttrGetUInt64Value(attr, pos)
    ccall((:mlirDenseElementsAttrGetUInt64Value, libmlir), UInt64, (MlirAttribute, intptr_t), attr, pos)
end

function mlirDenseElementsAttrGetFloatValue(attr, pos)
    ccall((:mlirDenseElementsAttrGetFloatValue, libmlir), Cfloat, (MlirAttribute, intptr_t), attr, pos)
end

function mlirDenseElementsAttrGetDoubleValue(attr, pos)
    ccall((:mlirDenseElementsAttrGetDoubleValue, libmlir), Cdouble, (MlirAttribute, intptr_t), attr, pos)
end

function mlirDenseElementsAttrGetStringValue(attr, pos)
    ccall((:mlirDenseElementsAttrGetStringValue, libmlir), MlirStringRef, (MlirAttribute, intptr_t), attr, pos)
end

function mlirDenseElementsAttrGetRawData(attr)
    ccall((:mlirDenseElementsAttrGetRawData, libmlir), Ptr{Cvoid}, (MlirAttribute,), attr)
end

function mlirAttributeIsAOpaqueElements(attr)
    ccall((:mlirAttributeIsAOpaqueElements, libmlir), Bool, (MlirAttribute,), attr)
end

function mlirAttributeIsASparseElements(attr)
    ccall((:mlirAttributeIsASparseElements, libmlir), Bool, (MlirAttribute,), attr)
end

function mlirSparseElementsAttribute(shapedType, denseIndices, denseValues)
    ccall((:mlirSparseElementsAttribute, libmlir), MlirAttribute, (MlirType, MlirAttribute, MlirAttribute), shapedType, denseIndices, denseValues)
end

function mlirSparseElementsAttrGetIndices(attr)
    ccall((:mlirSparseElementsAttrGetIndices, libmlir), MlirAttribute, (MlirAttribute,), attr)
end

function mlirSparseElementsAttrGetValues(attr)
    ccall((:mlirSparseElementsAttrGetValues, libmlir), MlirAttribute, (MlirAttribute,), attr)
end
# Julia wrapper for header: BuiltinTypes.h
# Automatically generated using Clang.jl


function mlirTypeIsAInteger(type)
    ccall((:mlirTypeIsAInteger, libmlir), Bool, (MlirType,), type)
end

function mlirIntegerTypeGet(ctx, bitwidth)
    ccall((:mlirIntegerTypeGet, libmlir), MlirType, (MlirContext, UInt32), ctx, bitwidth)
end

function mlirIntegerTypeSignedGet(ctx, bitwidth)
    ccall((:mlirIntegerTypeSignedGet, libmlir), MlirType, (MlirContext, UInt32), ctx, bitwidth)
end

function mlirIntegerTypeUnsignedGet(ctx, bitwidth)
    ccall((:mlirIntegerTypeUnsignedGet, libmlir), MlirType, (MlirContext, UInt32), ctx, bitwidth)
end

function mlirIntegerTypeGetWidth(type)
    ccall((:mlirIntegerTypeGetWidth, libmlir), UInt32, (MlirType,), type)
end

function mlirIntegerTypeIsSignless(type)
    ccall((:mlirIntegerTypeIsSignless, libmlir), Bool, (MlirType,), type)
end

function mlirIntegerTypeIsSigned(type)
    ccall((:mlirIntegerTypeIsSigned, libmlir), Bool, (MlirType,), type)
end

function mlirIntegerTypeIsUnsigned(type)
    ccall((:mlirIntegerTypeIsUnsigned, libmlir), Bool, (MlirType,), type)
end

function mlirTypeIsAIndex(type)
    ccall((:mlirTypeIsAIndex, libmlir), Bool, (MlirType,), type)
end

function mlirIndexTypeGet(ctx)
    ccall((:mlirIndexTypeGet, libmlir), MlirType, (MlirContext,), ctx)
end

function mlirTypeIsABF16(type)
    ccall((:mlirTypeIsABF16, libmlir), Bool, (MlirType,), type)
end

function mlirBF16TypeGet(ctx)
    ccall((:mlirBF16TypeGet, libmlir), MlirType, (MlirContext,), ctx)
end

function mlirTypeIsAF16(type)
    ccall((:mlirTypeIsAF16, libmlir), Bool, (MlirType,), type)
end

function mlirF16TypeGet(ctx)
    ccall((:mlirF16TypeGet, libmlir), MlirType, (MlirContext,), ctx)
end

function mlirTypeIsAF32(type)
    ccall((:mlirTypeIsAF32, libmlir), Bool, (MlirType,), type)
end

function mlirF32TypeGet(ctx)
    ccall((:mlirF32TypeGet, libmlir), MlirType, (MlirContext,), ctx)
end

function mlirTypeIsAF64(type)
    ccall((:mlirTypeIsAF64, libmlir), Bool, (MlirType,), type)
end

function mlirF64TypeGet(ctx)
    ccall((:mlirF64TypeGet, libmlir), MlirType, (MlirContext,), ctx)
end

function mlirTypeIsANone(type)
    ccall((:mlirTypeIsANone, libmlir), Bool, (MlirType,), type)
end

function mlirNoneTypeGet(ctx)
    ccall((:mlirNoneTypeGet, libmlir), MlirType, (MlirContext,), ctx)
end

function mlirTypeIsAComplex(type)
    ccall((:mlirTypeIsAComplex, libmlir), Bool, (MlirType,), type)
end

function mlirComplexTypeGet(elementType)
    ccall((:mlirComplexTypeGet, libmlir), MlirType, (MlirType,), elementType)
end

function mlirComplexTypeGetElementType(type)
    ccall((:mlirComplexTypeGetElementType, libmlir), MlirType, (MlirType,), type)
end

function mlirTypeIsAShaped(type)
    ccall((:mlirTypeIsAShaped, libmlir), Bool, (MlirType,), type)
end

function mlirShapedTypeGetElementType(type)
    ccall((:mlirShapedTypeGetElementType, libmlir), MlirType, (MlirType,), type)
end

function mlirShapedTypeHasRank(type)
    ccall((:mlirShapedTypeHasRank, libmlir), Bool, (MlirType,), type)
end

function mlirShapedTypeGetRank(type)
    ccall((:mlirShapedTypeGetRank, libmlir), Int64, (MlirType,), type)
end

function mlirShapedTypeHasStaticShape(type)
    ccall((:mlirShapedTypeHasStaticShape, libmlir), Bool, (MlirType,), type)
end

function mlirShapedTypeIsDynamicDim(type, dim)
    ccall((:mlirShapedTypeIsDynamicDim, libmlir), Bool, (MlirType, intptr_t), type, dim)
end

function mlirShapedTypeGetDimSize(type, dim)
    ccall((:mlirShapedTypeGetDimSize, libmlir), Int64, (MlirType, intptr_t), type, dim)
end

function mlirShapedTypeIsDynamicSize(size)
    ccall((:mlirShapedTypeIsDynamicSize, libmlir), Bool, (Int64,), size)
end

function mlirShapedTypeIsDynamicStrideOrOffset(val)
    ccall((:mlirShapedTypeIsDynamicStrideOrOffset, libmlir), Bool, (Int64,), val)
end

function mlirTypeIsAVector(type)
    ccall((:mlirTypeIsAVector, libmlir), Bool, (MlirType,), type)
end

function mlirVectorTypeGet(rank, shape, elementType)
    ccall((:mlirVectorTypeGet, libmlir), MlirType, (intptr_t, Ptr{Int64}, MlirType), rank, shape, elementType)
end

function mlirVectorTypeGetChecked(rank, shape, elementType, loc)
    ccall((:mlirVectorTypeGetChecked, libmlir), MlirType, (intptr_t, Ptr{Int64}, MlirType, MlirLocation), rank, shape, elementType, loc)
end

function mlirTypeIsATensor(type)
    ccall((:mlirTypeIsATensor, libmlir), Bool, (MlirType,), type)
end

function mlirTypeIsARankedTensor(type)
    ccall((:mlirTypeIsARankedTensor, libmlir), Bool, (MlirType,), type)
end

function mlirTypeIsAUnrankedTensor(type)
    ccall((:mlirTypeIsAUnrankedTensor, libmlir), Bool, (MlirType,), type)
end

function mlirRankedTensorTypeGet(rank, shape, elementType)
    ccall((:mlirRankedTensorTypeGet, libmlir), MlirType, (intptr_t, Ptr{Int64}, MlirType), rank, shape, elementType)
end

function mlirRankedTensorTypeGetChecked(rank, shape, elementType, loc)
    ccall((:mlirRankedTensorTypeGetChecked, libmlir), MlirType, (intptr_t, Ptr{Int64}, MlirType, MlirLocation), rank, shape, elementType, loc)
end

function mlirUnrankedTensorTypeGet(elementType)
    ccall((:mlirUnrankedTensorTypeGet, libmlir), MlirType, (MlirType,), elementType)
end

function mlirUnrankedTensorTypeGetChecked(elementType, loc)
    ccall((:mlirUnrankedTensorTypeGetChecked, libmlir), MlirType, (MlirType, MlirLocation), elementType, loc)
end

function mlirTypeIsAMemRef(type)
    ccall((:mlirTypeIsAMemRef, libmlir), Bool, (MlirType,), type)
end

function mlirTypeIsAUnrankedMemRef(type)
    ccall((:mlirTypeIsAUnrankedMemRef, libmlir), Bool, (MlirType,), type)
end

function mlirMemRefTypeGet(elementType, rank, shape, numMaps, affineMaps, memorySpace)
    ccall((:mlirMemRefTypeGet, libmlir), MlirType, (MlirType, intptr_t, Ptr{Int64}, intptr_t, Ptr{MlirAttribute}, UInt32), elementType, rank, shape, numMaps, affineMaps, memorySpace)
end

function mlirMemRefTypeContiguousGet(elementType, rank, shape, memorySpace)
    ccall((:mlirMemRefTypeContiguousGet, libmlir), MlirType, (MlirType, intptr_t, Ptr{Int64}, UInt32), elementType, rank, shape, memorySpace)
end

function mlirMemRefTypeContiguousGetChecked(elementType, rank, shape, memorySpace, loc)
    ccall((:mlirMemRefTypeContiguousGetChecked, libmlir), MlirType, (MlirType, intptr_t, Ptr{Int64}, UInt32, MlirLocation), elementType, rank, shape, memorySpace, loc)
end

function mlirUnrankedMemRefTypeGet(elementType, memorySpace)
    ccall((:mlirUnrankedMemRefTypeGet, libmlir), MlirType, (MlirType, UInt32), elementType, memorySpace)
end

function mlirUnrankedMemRefTypeGetChecked(elementType, memorySpace, loc)
    ccall((:mlirUnrankedMemRefTypeGetChecked, libmlir), MlirType, (MlirType, UInt32, MlirLocation), elementType, memorySpace, loc)
end

function mlirMemRefTypeGetNumAffineMaps(type)
    ccall((:mlirMemRefTypeGetNumAffineMaps, libmlir), intptr_t, (MlirType,), type)
end

function mlirMemRefTypeGetAffineMap(type, pos)
    ccall((:mlirMemRefTypeGetAffineMap, libmlir), MlirAffineMap, (MlirType, intptr_t), type, pos)
end

function mlirMemRefTypeGetMemorySpace(type)
    ccall((:mlirMemRefTypeGetMemorySpace, libmlir), UInt32, (MlirType,), type)
end

function mlirUnrankedMemrefGetMemorySpace(type)
    ccall((:mlirUnrankedMemrefGetMemorySpace, libmlir), UInt32, (MlirType,), type)
end

function mlirTypeIsATuple(type)
    ccall((:mlirTypeIsATuple, libmlir), Bool, (MlirType,), type)
end

function mlirTupleTypeGet(ctx, numElements, elements)
    ccall((:mlirTupleTypeGet, libmlir), MlirType, (MlirContext, intptr_t, Ptr{MlirType}), ctx, numElements, elements)
end

function mlirTupleTypeGetNumTypes(type)
    ccall((:mlirTupleTypeGetNumTypes, libmlir), intptr_t, (MlirType,), type)
end

function mlirTupleTypeGetType(type, pos)
    ccall((:mlirTupleTypeGetType, libmlir), MlirType, (MlirType, intptr_t), type, pos)
end

function mlirTypeIsAFunction(type)
    ccall((:mlirTypeIsAFunction, libmlir), Bool, (MlirType,), type)
end

function mlirFunctionTypeGet(ctx, numInputs, inputs, numResults, results)
    ccall((:mlirFunctionTypeGet, libmlir), MlirType, (MlirContext, intptr_t, Ptr{MlirType}, intptr_t, Ptr{MlirType}), ctx, numInputs, inputs, numResults, results)
end

function mlirFunctionTypeGetNumInputs(type)
    ccall((:mlirFunctionTypeGetNumInputs, libmlir), intptr_t, (MlirType,), type)
end

function mlirFunctionTypeGetNumResults(type)
    ccall((:mlirFunctionTypeGetNumResults, libmlir), intptr_t, (MlirType,), type)
end

function mlirFunctionTypeGetInput(type, pos)
    ccall((:mlirFunctionTypeGetInput, libmlir), MlirType, (MlirType, intptr_t), type, pos)
end

function mlirFunctionTypeGetResult(type, pos)
    ccall((:mlirFunctionTypeGetResult, libmlir), MlirType, (MlirType, intptr_t), type, pos)
end
# Julia wrapper for header: Diagnostics.h
# Automatically generated using Clang.jl


function mlirDiagnosticPrint(diagnostic, callback, userData)
    ccall((:mlirDiagnosticPrint, libmlir), Cvoid, (MlirDiagnostic, MlirStringCallback, Ptr{Cvoid}), diagnostic, callback, userData)
end

function mlirDiagnosticGetLocation(diagnostic)
    ccall((:mlirDiagnosticGetLocation, libmlir), MlirLocation, (MlirDiagnostic,), diagnostic)
end

function mlirDiagnosticGetSeverity(diagnostic)
    ccall((:mlirDiagnosticGetSeverity, libmlir), MlirDiagnosticSeverity, (MlirDiagnostic,), diagnostic)
end

function mlirDiagnosticGetNumNotes(diagnostic)
    ccall((:mlirDiagnosticGetNumNotes, libmlir), intptr_t, (MlirDiagnostic,), diagnostic)
end

function mlirDiagnosticGetNote(diagnostic, pos)
    ccall((:mlirDiagnosticGetNote, libmlir), MlirDiagnostic, (MlirDiagnostic, intptr_t), diagnostic, pos)
end

function mlirContextAttachDiagnosticHandler(context, handler, userData, deleteUserData)
    ccall((:mlirContextAttachDiagnosticHandler, libmlir), MlirDiagnosticHandlerID, (MlirContext, MlirDiagnosticHandler, Ptr{Cvoid}, Ptr{Cvoid}), context, handler, userData, deleteUserData)
end

function mlirContextDetachDiagnosticHandler(context, id)
    ccall((:mlirContextDetachDiagnosticHandler, libmlir), Cvoid, (MlirContext, MlirDiagnosticHandlerID), context, id)
end

function mlirEmitError(location, message)
    ccall((:mlirEmitError, libmlir), Cvoid, (MlirLocation, Cstring), location, message)
end
# Julia wrapper for header: IR.h
# Automatically generated using Clang.jl


function mlirContextCreate()
    ccall((:mlirContextCreate, libmlir), MlirContext, ())
end

function mlirContextEqual(ctx1, ctx2)
    ccall((:mlirContextEqual, libmlir), Bool, (MlirContext, MlirContext), ctx1, ctx2)
end

function mlirContextIsNull(context)
    ccall((:mlirContextIsNull, libmlir), Bool, (MlirContext,), context)
end

function mlirContextDestroy(context)
    ccall((:mlirContextDestroy, libmlir), Cvoid, (MlirContext,), context)
end

function mlirContextSetAllowUnregisteredDialects(context, allow)
    ccall((:mlirContextSetAllowUnregisteredDialects, libmlir), Cvoid, (MlirContext, Bool), context, allow)
end

function mlirContextGetAllowUnregisteredDialects(context)
    ccall((:mlirContextGetAllowUnregisteredDialects, libmlir), Bool, (MlirContext,), context)
end

function mlirContextGetNumRegisteredDialects(context)
    ccall((:mlirContextGetNumRegisteredDialects, libmlir), intptr_t, (MlirContext,), context)
end

function mlirContextGetNumLoadedDialects(context)
    ccall((:mlirContextGetNumLoadedDialects, libmlir), intptr_t, (MlirContext,), context)
end

function mlirContextGetOrLoadDialect(context, name)
    ccall((:mlirContextGetOrLoadDialect, libmlir), MlirDialect, (MlirContext, MlirStringRef), context, name)
end

function mlirDialectGetContext(dialect)
    ccall((:mlirDialectGetContext, libmlir), MlirContext, (MlirDialect,), dialect)
end

function mlirDialectIsNull(dialect)
    ccall((:mlirDialectIsNull, libmlir), Bool, (MlirDialect,), dialect)
end

function mlirDialectEqual(dialect1, dialect2)
    ccall((:mlirDialectEqual, libmlir), Bool, (MlirDialect, MlirDialect), dialect1, dialect2)
end

function mlirDialectGetNamespace(dialect)
    ccall((:mlirDialectGetNamespace, libmlir), MlirStringRef, (MlirDialect,), dialect)
end

function mlirLocationFileLineColGet(context, filename, line, col)
    ccall((:mlirLocationFileLineColGet, libmlir), MlirLocation, (MlirContext, MlirStringRef, UInt32, UInt32), context, filename, line, col)
end

function mlirLocationUnknownGet(context)
    ccall((:mlirLocationUnknownGet, libmlir), MlirLocation, (MlirContext,), context)
end

function mlirLocationGetContext(location)
    ccall((:mlirLocationGetContext, libmlir), MlirContext, (MlirLocation,), location)
end

function mlirLocationIsNull(location)
    ccall((:mlirLocationIsNull, libmlir), Bool, (MlirLocation,), location)
end

function mlirLocationEqual(l1, l2)
    ccall((:mlirLocationEqual, libmlir), Bool, (MlirLocation, MlirLocation), l1, l2)
end

function mlirLocationPrint(location, callback, userData)
    ccall((:mlirLocationPrint, libmlir), Cvoid, (MlirLocation, MlirStringCallback, Ptr{Cvoid}), location, callback, userData)
end

function mlirModuleCreateEmpty(location)
    ccall((:mlirModuleCreateEmpty, libmlir), MlirModule, (MlirLocation,), location)
end

function mlirModuleCreateParse(context, _module)
    ccall((:mlirModuleCreateParse, libmlir), MlirModule, (MlirContext, MlirStringRef), context, _module)
end

function mlirModuleGetContext(_module)
    ccall((:mlirModuleGetContext, libmlir), MlirContext, (MlirModule,), _module)
end

function mlirModuleGetBody(_module)
    ccall((:mlirModuleGetBody, libmlir), MlirBlock, (MlirModule,), _module)
end

function mlirModuleIsNull(_module)
    ccall((:mlirModuleIsNull, libmlir), Bool, (MlirModule,), _module)
end

function mlirModuleDestroy(_module)
    ccall((:mlirModuleDestroy, libmlir), Cvoid, (MlirModule,), _module)
end

function mlirModuleGetOperation(_module)
    ccall((:mlirModuleGetOperation, libmlir), MlirOperation, (MlirModule,), _module)
end

function mlirOperationStateGet(name, loc)
    ccall((:mlirOperationStateGet, libmlir), MlirOperationState, (MlirStringRef, MlirLocation), name, loc)
end

function mlirOperationStateAddResults(state, n, results)
    ccall((:mlirOperationStateAddResults, libmlir), Cvoid, (Ptr{MlirOperationState}, intptr_t, Ptr{MlirType}), state, n, results)
end

function mlirOperationStateAddOperands(state, n, operands)
    ccall((:mlirOperationStateAddOperands, libmlir), Cvoid, (Ptr{MlirOperationState}, intptr_t, Ptr{MlirValue}), state, n, operands)
end

function mlirOperationStateAddOwnedRegions(state, n, regions)
    ccall((:mlirOperationStateAddOwnedRegions, libmlir), Cvoid, (Ptr{MlirOperationState}, intptr_t, Ptr{MlirRegion}), state, n, regions)
end

function mlirOperationStateAddSuccessors(state, n, successors)
    ccall((:mlirOperationStateAddSuccessors, libmlir), Cvoid, (Ptr{MlirOperationState}, intptr_t, Ptr{MlirBlock}), state, n, successors)
end

function mlirOperationStateAddAttributes(state, n, attributes)
    ccall((:mlirOperationStateAddAttributes, libmlir), Cvoid, (Ptr{MlirOperationState}, intptr_t, Ptr{MlirNamedAttribute}), state, n, attributes)
end

function mlirOpPrintingFlagsCreate()
    ccall((:mlirOpPrintingFlagsCreate, libmlir), MlirOpPrintingFlags, ())
end

function mlirOpPrintingFlagsDestroy(flags)
    ccall((:mlirOpPrintingFlagsDestroy, libmlir), Cvoid, (MlirOpPrintingFlags,), flags)
end

function mlirOpPrintingFlagsElideLargeElementsAttrs(flags, largeElementLimit)
    ccall((:mlirOpPrintingFlagsElideLargeElementsAttrs, libmlir), Cvoid, (MlirOpPrintingFlags, intptr_t), flags, largeElementLimit)
end

function mlirOpPrintingFlagsEnableDebugInfo(flags, prettyForm)
    ccall((:mlirOpPrintingFlagsEnableDebugInfo, libmlir), Cvoid, (MlirOpPrintingFlags, Bool), flags, prettyForm)
end

function mlirOpPrintingFlagsPrintGenericOpForm(flags)
    ccall((:mlirOpPrintingFlagsPrintGenericOpForm, libmlir), Cvoid, (MlirOpPrintingFlags,), flags)
end

function mlirOpPrintingFlagsUseLocalScope(flags)
    ccall((:mlirOpPrintingFlagsUseLocalScope, libmlir), Cvoid, (MlirOpPrintingFlags,), flags)
end

function mlirOperationCreate(state)
    ccall((:mlirOperationCreate, libmlir), MlirOperation, (Ptr{MlirOperationState},), state)
end

function mlirOperationDestroy(op)
    ccall((:mlirOperationDestroy, libmlir), Cvoid, (MlirOperation,), op)
end

function mlirOperationIsNull(op)
    ccall((:mlirOperationIsNull, libmlir), Bool, (MlirOperation,), op)
end

function mlirOperationEqual(op, other)
    ccall((:mlirOperationEqual, libmlir), Bool, (MlirOperation, MlirOperation), op, other)
end

function mlirOperationGetName(op)
    ccall((:mlirOperationGetName, libmlir), MlirIdentifier, (MlirOperation,), op)
end

function mlirOperationGetBlock(op)
    ccall((:mlirOperationGetBlock, libmlir), MlirBlock, (MlirOperation,), op)
end

function mlirOperationGetParentOperation(op)
    ccall((:mlirOperationGetParentOperation, libmlir), MlirOperation, (MlirOperation,), op)
end

function mlirOperationGetNumRegions(op)
    ccall((:mlirOperationGetNumRegions, libmlir), intptr_t, (MlirOperation,), op)
end

function mlirOperationGetRegion(op, pos)
    ccall((:mlirOperationGetRegion, libmlir), MlirRegion, (MlirOperation, intptr_t), op, pos)
end

function mlirOperationGetNextInBlock(op)
    ccall((:mlirOperationGetNextInBlock, libmlir), MlirOperation, (MlirOperation,), op)
end

function mlirOperationGetNumOperands(op)
    ccall((:mlirOperationGetNumOperands, libmlir), intptr_t, (MlirOperation,), op)
end

function mlirOperationGetOperand(op, pos)
    ccall((:mlirOperationGetOperand, libmlir), MlirValue, (MlirOperation, intptr_t), op, pos)
end

function mlirOperationGetNumResults(op)
    ccall((:mlirOperationGetNumResults, libmlir), intptr_t, (MlirOperation,), op)
end

function mlirOperationGetResult(op, pos)
    ccall((:mlirOperationGetResult, libmlir), MlirValue, (MlirOperation, intptr_t), op, pos)
end

function mlirOperationGetNumSuccessors(op)
    ccall((:mlirOperationGetNumSuccessors, libmlir), intptr_t, (MlirOperation,), op)
end

function mlirOperationGetSuccessor(op, pos)
    ccall((:mlirOperationGetSuccessor, libmlir), MlirBlock, (MlirOperation, intptr_t), op, pos)
end

function mlirOperationGetNumAttributes(op)
    ccall((:mlirOperationGetNumAttributes, libmlir), intptr_t, (MlirOperation,), op)
end

function mlirOperationGetAttribute(op, pos)
    ccall((:mlirOperationGetAttribute, libmlir), MlirNamedAttribute, (MlirOperation, intptr_t), op, pos)
end

function mlirOperationGetAttributeByName(op, name)
    ccall((:mlirOperationGetAttributeByName, libmlir), MlirAttribute, (MlirOperation, MlirStringRef), op, name)
end

function mlirOperationSetAttributeByName(op, name, attr)
    ccall((:mlirOperationSetAttributeByName, libmlir), Cvoid, (MlirOperation, MlirStringRef, MlirAttribute), op, name, attr)
end

function mlirOperationRemoveAttributeByName(op, name)
    ccall((:mlirOperationRemoveAttributeByName, libmlir), Bool, (MlirOperation, MlirStringRef), op, name)
end

function mlirOperationPrint(op, callback, userData)
    ccall((:mlirOperationPrint, libmlir), Cvoid, (MlirOperation, MlirStringCallback, Ptr{Cvoid}), op, callback, userData)
end

function mlirOperationPrintWithFlags(op, flags, callback, userData)
    ccall((:mlirOperationPrintWithFlags, libmlir), Cvoid, (MlirOperation, MlirOpPrintingFlags, MlirStringCallback, Ptr{Cvoid}), op, flags, callback, userData)
end

function mlirOperationDump(op)
    ccall((:mlirOperationDump, libmlir), Cvoid, (MlirOperation,), op)
end

function mlirOperationVerify(op)
    ccall((:mlirOperationVerify, libmlir), Bool, (MlirOperation,), op)
end

function mlirRegionCreate()
    ccall((:mlirRegionCreate, libmlir), MlirRegion, ())
end

function mlirRegionDestroy(region)
    ccall((:mlirRegionDestroy, libmlir), Cvoid, (MlirRegion,), region)
end

function mlirRegionIsNull(region)
    ccall((:mlirRegionIsNull, libmlir), Bool, (MlirRegion,), region)
end

function mlirRegionGetFirstBlock(region)
    ccall((:mlirRegionGetFirstBlock, libmlir), MlirBlock, (MlirRegion,), region)
end

function mlirRegionAppendOwnedBlock(region, block)
    ccall((:mlirRegionAppendOwnedBlock, libmlir), Cvoid, (MlirRegion, MlirBlock), region, block)
end

function mlirRegionInsertOwnedBlock(region, pos, block)
    ccall((:mlirRegionInsertOwnedBlock, libmlir), Cvoid, (MlirRegion, intptr_t, MlirBlock), region, pos, block)
end

function mlirRegionInsertOwnedBlockAfter(region, reference, block)
    ccall((:mlirRegionInsertOwnedBlockAfter, libmlir), Cvoid, (MlirRegion, MlirBlock, MlirBlock), region, reference, block)
end

function mlirRegionInsertOwnedBlockBefore(region, reference, block)
    ccall((:mlirRegionInsertOwnedBlockBefore, libmlir), Cvoid, (MlirRegion, MlirBlock, MlirBlock), region, reference, block)
end

function mlirBlockCreate(nArgs, args)
    ccall((:mlirBlockCreate, libmlir), MlirBlock, (intptr_t, Ptr{MlirType}), nArgs, args)
end

function mlirBlockDestroy(block)
    ccall((:mlirBlockDestroy, libmlir), Cvoid, (MlirBlock,), block)
end

function mlirBlockIsNull(block)
    ccall((:mlirBlockIsNull, libmlir), Bool, (MlirBlock,), block)
end

function mlirBlockEqual(block, other)
    ccall((:mlirBlockEqual, libmlir), Bool, (MlirBlock, MlirBlock), block, other)
end

function mlirBlockGetNextInRegion(block)
    ccall((:mlirBlockGetNextInRegion, libmlir), MlirBlock, (MlirBlock,), block)
end

function mlirBlockGetFirstOperation(block)
    ccall((:mlirBlockGetFirstOperation, libmlir), MlirOperation, (MlirBlock,), block)
end

function mlirBlockGetTerminator(block)
    ccall((:mlirBlockGetTerminator, libmlir), MlirOperation, (MlirBlock,), block)
end

function mlirBlockAppendOwnedOperation(block, operation)
    ccall((:mlirBlockAppendOwnedOperation, libmlir), Cvoid, (MlirBlock, MlirOperation), block, operation)
end

function mlirBlockInsertOwnedOperation(block, pos, operation)
    ccall((:mlirBlockInsertOwnedOperation, libmlir), Cvoid, (MlirBlock, intptr_t, MlirOperation), block, pos, operation)
end

function mlirBlockInsertOwnedOperationAfter(block, reference, operation)
    ccall((:mlirBlockInsertOwnedOperationAfter, libmlir), Cvoid, (MlirBlock, MlirOperation, MlirOperation), block, reference, operation)
end

function mlirBlockInsertOwnedOperationBefore(block, reference, operation)
    ccall((:mlirBlockInsertOwnedOperationBefore, libmlir), Cvoid, (MlirBlock, MlirOperation, MlirOperation), block, reference, operation)
end

function mlirBlockGetNumArguments(block)
    ccall((:mlirBlockGetNumArguments, libmlir), intptr_t, (MlirBlock,), block)
end

function mlirBlockGetArgument(block, pos)
    ccall((:mlirBlockGetArgument, libmlir), MlirValue, (MlirBlock, intptr_t), block, pos)
end

function mlirBlockPrint(block, callback, userData)
    ccall((:mlirBlockPrint, libmlir), Cvoid, (MlirBlock, MlirStringCallback, Ptr{Cvoid}), block, callback, userData)
end

function mlirValueIsNull(value)
    ccall((:mlirValueIsNull, libmlir), Bool, (MlirValue,), value)
end

function mlirValueEqual(value1, value2)
    ccall((:mlirValueEqual, libmlir), Bool, (MlirValue, MlirValue), value1, value2)
end

function mlirValueIsABlockArgument(value)
    ccall((:mlirValueIsABlockArgument, libmlir), Bool, (MlirValue,), value)
end

function mlirValueIsAOpResult(value)
    ccall((:mlirValueIsAOpResult, libmlir), Bool, (MlirValue,), value)
end

function mlirBlockArgumentGetOwner(value)
    ccall((:mlirBlockArgumentGetOwner, libmlir), MlirBlock, (MlirValue,), value)
end

function mlirBlockArgumentGetArgNumber(value)
    ccall((:mlirBlockArgumentGetArgNumber, libmlir), intptr_t, (MlirValue,), value)
end

function mlirBlockArgumentSetType(value, type)
    ccall((:mlirBlockArgumentSetType, libmlir), Cvoid, (MlirValue, MlirType), value, type)
end

function mlirOpResultGetOwner(value)
    ccall((:mlirOpResultGetOwner, libmlir), MlirOperation, (MlirValue,), value)
end

function mlirOpResultGetResultNumber(value)
    ccall((:mlirOpResultGetResultNumber, libmlir), intptr_t, (MlirValue,), value)
end

function mlirValueGetType(value)
    ccall((:mlirValueGetType, libmlir), MlirType, (MlirValue,), value)
end

function mlirValueDump(value)
    ccall((:mlirValueDump, libmlir), Cvoid, (MlirValue,), value)
end

function mlirValuePrint(value, callback, userData)
    ccall((:mlirValuePrint, libmlir), Cvoid, (MlirValue, MlirStringCallback, Ptr{Cvoid}), value, callback, userData)
end

function mlirTypeParseGet(context, type)
    ccall((:mlirTypeParseGet, libmlir), MlirType, (MlirContext, MlirStringRef), context, type)
end

function mlirTypeGetContext(type)
    ccall((:mlirTypeGetContext, libmlir), MlirContext, (MlirType,), type)
end

function mlirTypeIsNull(type)
    ccall((:mlirTypeIsNull, libmlir), Bool, (MlirType,), type)
end

function mlirTypeEqual(t1, t2)
    ccall((:mlirTypeEqual, libmlir), Bool, (MlirType, MlirType), t1, t2)
end

function mlirTypePrint(type, callback, userData)
    ccall((:mlirTypePrint, libmlir), Cvoid, (MlirType, MlirStringCallback, Ptr{Cvoid}), type, callback, userData)
end

function mlirTypeDump(type)
    ccall((:mlirTypeDump, libmlir), Cvoid, (MlirType,), type)
end

function mlirAttributeParseGet(context, attr)
    ccall((:mlirAttributeParseGet, libmlir), MlirAttribute, (MlirContext, MlirStringRef), context, attr)
end

function mlirAttributeGetContext(attribute)
    ccall((:mlirAttributeGetContext, libmlir), MlirContext, (MlirAttribute,), attribute)
end

function mlirAttributeGetType(attribute)
    ccall((:mlirAttributeGetType, libmlir), MlirType, (MlirAttribute,), attribute)
end

function mlirAttributeIsNull(attr)
    ccall((:mlirAttributeIsNull, libmlir), Bool, (MlirAttribute,), attr)
end

function mlirAttributeEqual(a1, a2)
    ccall((:mlirAttributeEqual, libmlir), Bool, (MlirAttribute, MlirAttribute), a1, a2)
end

function mlirAttributePrint(attr, callback, userData)
    ccall((:mlirAttributePrint, libmlir), Cvoid, (MlirAttribute, MlirStringCallback, Ptr{Cvoid}), attr, callback, userData)
end

function mlirAttributeDump(attr)
    ccall((:mlirAttributeDump, libmlir), Cvoid, (MlirAttribute,), attr)
end

function mlirNamedAttributeGet(name, attr)
    ccall((:mlirNamedAttributeGet, libmlir), MlirNamedAttribute, (MlirStringRef, MlirAttribute), name, attr)
end

function mlirIdentifierGet(context, str)
    ccall((:mlirIdentifierGet, libmlir), MlirIdentifier, (MlirContext, MlirStringRef), context, str)
end

function mlirIdentifierEqual(ident, other)
    ccall((:mlirIdentifierEqual, libmlir), Bool, (MlirIdentifier, MlirIdentifier), ident, other)
end

function mlirIdentifierStr(ident)
    ccall((:mlirIdentifierStr, libmlir), MlirStringRef, (MlirIdentifier,), ident)
end
# Julia wrapper for header: Pass.h
# Automatically generated using Clang.jl


function mlirPassManagerCreate(ctx)
    ccall((:mlirPassManagerCreate, libmlir), MlirPassManager, (MlirContext,), ctx)
end

function mlirPassManagerDestroy(passManager)
    ccall((:mlirPassManagerDestroy, libmlir), Cvoid, (MlirPassManager,), passManager)
end

function mlirPassManagerIsNull(passManager)
    ccall((:mlirPassManagerIsNull, libmlir), Bool, (MlirPassManager,), passManager)
end

function mlirPassManagerGetAsOpPassManager(passManager)
    ccall((:mlirPassManagerGetAsOpPassManager, libmlir), MlirOpPassManager, (MlirPassManager,), passManager)
end

function mlirPassManagerRun(passManager, _module)
    ccall((:mlirPassManagerRun, libmlir), MlirLogicalResult, (MlirPassManager, MlirModule), passManager, _module)
end

function mlirPassManagerGetNestedUnder(passManager, operationName)
    ccall((:mlirPassManagerGetNestedUnder, libmlir), MlirOpPassManager, (MlirPassManager, MlirStringRef), passManager, operationName)
end

function mlirOpPassManagerGetNestedUnder(passManager, operationName)
    ccall((:mlirOpPassManagerGetNestedUnder, libmlir), MlirOpPassManager, (MlirOpPassManager, MlirStringRef), passManager, operationName)
end

function mlirPassManagerAddOwnedPass(passManager, pass)
    ccall((:mlirPassManagerAddOwnedPass, libmlir), Cvoid, (MlirPassManager, MlirPass), passManager, pass)
end

function mlirOpPassManagerAddOwnedPass(passManager, pass)
    ccall((:mlirOpPassManagerAddOwnedPass, libmlir), Cvoid, (MlirOpPassManager, MlirPass), passManager, pass)
end

function mlirPrintPassPipeline(passManager, callback, userData)
    ccall((:mlirPrintPassPipeline, libmlir), Cvoid, (MlirOpPassManager, MlirStringCallback, Ptr{Cvoid}), passManager, callback, userData)
end

function mlirParsePassPipeline(passManager, pipeline)
    ccall((:mlirParsePassPipeline, libmlir), MlirLogicalResult, (MlirOpPassManager, MlirStringRef), passManager, pipeline)
end
# Julia wrapper for header: Registration.h
# Automatically generated using Clang.jl


function mlirRegisterAllDialects(context)
    ccall((:mlirRegisterAllDialects, libmlir), Cvoid, (MlirContext,), context)
end
# Julia wrapper for header: StandardDialect.h
# Automatically generated using Clang.jl


function mlirContextRegisterStandardDialect(context)
    ccall((:mlirContextRegisterStandardDialect, libmlir), Cvoid, (MlirContext,), context)
end

function mlirContextLoadStandardDialect(context)
    ccall((:mlirContextLoadStandardDialect, libmlir), MlirDialect, (MlirContext,), context)
end

function mlirStandardDialectGetNamespace()
    ccall((:mlirStandardDialectGetNamespace, libmlir), MlirStringRef, ())
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

