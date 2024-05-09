using MLIR
using Test

using LLVM: LLVM

function registerAllUpstreamDialects!(ctx)
    if LLVM.version() >= v"15"
        registry = MLIR.API.mlirDialectRegistryCreate()
        MLIR.API.mlirRegisterAllDialects(registry)
        MLIR.API.mlirContextAppendDialectRegistry(ctx, registry)
        MLIR.API.mlirDialectRegistryDestroy(registry)
    else
        MLIR.API.mlirRegisterAllDialects(ctx)
    end

    return nothing
end

function lowerModuleToLLVM(ctx, mod)
    pm = MLIR.API.mlirPassManagerCreate(ctx)
    if LLVM.version() >= v"15"
        op = "func.func"
    else
        op = "builtin.func"
    end
    opm = MLIR.API.mlirPassManagerGetNestedUnder(pm, op)
    if LLVM.version() >= v"15"
        MLIR.API.mlirPassManagerAddOwnedPass(
            pm, MLIR.API.mlirCreateConversionConvertFuncToLLVM()
        )
    else
        MLIR.API.mlirPassManagerAddOwnedPass(
            pm, MLIR.API.mlirCreateConversionConvertStandardToLLVM()
        )
    end

    if LLVM.version() >= v"16"
        MLIR.API.mlirOpPassManagerAddOwnedPass(
            opm, MLIR.API.mlirCreateConversionArithToLLVMConversionPass()
        )
    else
        MLIR.API.mlirOpPassManagerAddOwnedPass(
            opm, MLIR.API.mlirCreateConversionConvertArithmeticToLLVM()
        )
    end
    status = MLIR.API.mlirPassManagerRun(pm, mod)
    # undefined symbol: mlirLogicalResultIsFailure
    if status.value == 0
        error("Unexpected failure running pass failure")
    end
    return MLIR.API.mlirPassManagerDestroy(pm)
end

ctx = MLIR.API.mlirContextCreate()
registerAllUpstreamDialects!(ctx)

if LLVM.version() >= v"15"
    ir = """
        module {
            func.func @add(%arg0 : i32) -> i32 attributes { llvm.emit_c_interface } {
                %res = arith.addi %arg0, %arg0 : i32
                return %res : i32
            }
        }
        """
else
    ir = """
        module {
            func @add(%arg0 : i32) -> i32 attributes { llvm.emit_c_interface } {
                %res = arith.addi %arg0, %arg0 : i32
                return %res : i32
            }
        }
        """
end
mod = MLIR.API.mlirModuleCreateParse(ctx, ir)
lowerModuleToLLVM(ctx, mod)

MLIR.API.mlirRegisterAllLLVMTranslations(ctx)

# TODO add C-API for translateModuleToLLVMIR

jit = if LLVM.version() >= v"16"
    MLIR.API.mlirExecutionEngineCreate(mod, 2, 0, C_NULL, false) #= enableObjectDump =#
else
    MLIR.API.mlirExecutionEngineCreate(mod, 2, 0, C_NULL) #=sharedLibPaths=#
end

if jit == C_NULL
    error("Execution engine creation failed")
end

addr = MLIR.API.mlirExecutionEngineLookup(jit, "add")

if addr == C_NULL
    error("Lookup failed")
end

@test ccall(addr, Cint, (Cint,), 42) == 84

MLIR.API.mlirExecutionEngineDestroy(jit)
MLIR.API.mlirModuleDestroy(mod)
MLIR.API.mlirContextDestroy(ctx)
