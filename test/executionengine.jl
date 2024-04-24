using MLIR
using Test

import LLVM

function registerAllUpstreamDialects!(ctx)
    if LLVM.version() >= v"15"
        registry = MLIR.API.Dispatcher.mlirDialectRegistryCreate()
        MLIR.API.Dispatcher.mlirRegisterAllDialects(registry)
        MLIR.API.Dispatcher.mlirContextAppendDialectRegistry(ctx, registry)
        MLIR.API.Dispatcher.mlirDialectRegistryDestroy(registry)
    else
        MLIR.API.Dispatcher.mlirRegisterAllDialects(ctx)
    end

    return nothing
end

function lowerModuleToLLVM(ctx, mod)
    pm = MLIR.API.Dispatcher.mlirPassManagerCreate(ctx)
    if LLVM.version() >= v"15"
        op = "func.func"
    else
        op = "builtin.func"
    end
    opm = MLIR.API.Dispatcher.mlirPassManagerGetNestedUnder(pm, op)
    if LLVM.version() >= v"15"
        MLIR.API.Dispatcher.mlirPassManagerAddOwnedPass(pm,
            MLIR.API.Dispatcher.mlirCreateConversionConvertFuncToLLVM()
        )
    else
        MLIR.API.Dispatcher.mlirPassManagerAddOwnedPass(pm,
            MLIR.API.Dispatcher.mlirCreateConversionConvertStandardToLLVM()
        )
    end

    if LLVM.version() >= v"16"
        MLIR.API.Dispatcher.mlirOpPassManagerAddOwnedPass(opm,
            MLIR.API.Dispatcher.mlirCreateConversionArithToLLVMConversionPass()
        )
    else
        MLIR.API.Dispatcher.mlirOpPassManagerAddOwnedPass(opm,
            MLIR.API.Dispatcher.mlirCreateConversionConvertArithmeticToLLVM()
        )
    end
    status = MLIR.API.Dispatcher.mlirPassManagerRun(pm, mod)
    # undefined symbol: mlirLogicalResultIsFailure
    if status.value == 0
        error("Unexpected failure running pass failure")
    end
    MLIR.API.Dispatcher.mlirPassManagerDestroy(pm)
end

ctx = MLIR.API.Dispatcher.mlirContextCreate()
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
mod = MLIR.API.Dispatcher.mlirModuleCreateParse(ctx, ir)
lowerModuleToLLVM(ctx, mod)

MLIR.API.Dispatcher.mlirRegisterAllLLVMTranslations(ctx)

# TODO add C-API for translateModuleToLLVMIR

jit = if LLVM.version() >= v"16"
    MLIR.API.Dispatcher.mlirExecutionEngineCreate(
        mod, #=optLevel=# 2, #=numPaths=# 0, #=sharedLibPaths=# C_NULL, #= enableObjectDump =# false)
else
    MLIR.API.Dispatcher.mlirExecutionEngineCreate(
        mod, #=optLevel=# 2, #=numPaths=# 0, #=sharedLibPaths=# C_NULL)
end

if jit == C_NULL
    error("Execution engine creation failed")
end

addr = MLIR.API.Dispatcher.mlirExecutionEngineLookup(jit, "add")

if addr == C_NULL
    error("Lookup failed")
end

@test ccall(addr, Cint, (Cint,), 42) == 84

MLIR.API.Dispatcher.mlirExecutionEngineDestroy(jit)
MLIR.API.Dispatcher.mlirModuleDestroy(mod)
MLIR.API.Dispatcher.mlirContextDestroy(ctx)
