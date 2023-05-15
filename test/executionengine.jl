using MLIR
import LLVM

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

# TODO: Fix for LLVM 15
function lowerModuleToLLVM(ctx, mod)
    pm = MLIR.API.mlirPassManagerCreate(ctx)
    opm = MLIR.API.mlirPassManagerGetNestedUnder(
        pm, MLIR.API.mlirStringRefCreateFromCString("builtin.func"))
    MLIR.API.mlirPassManagerAddOwnedPass(pm,
        MLIR.API.mlirCreateConversionConvertStandardToLLVM()
    )
    MLIR.API.mlirOpPassManagerAddOwnedPass(opm,
        MLIR.API.mlirCreateConversionConvertArithmeticToLLVM()
    )
    status = MLIR.API.mlirPassManagerRun(pm, mod)
    # undefined symbol: mlirLogicalResultIsFailure
    if status.value == 0
        error("Unexpected failure running pass failure")
    end
end

ctx = MLIR.API.mlirContextCreate()
registerAllUpstreamDialects!(ctx)

ir = MLIR.API.mlirStringRefCreateFromCString(
    """
    module {
        func @add(%arg0 : i32) -> i32 attributes { llvm.emit_c_interface } {
            %res = arith.addi %arg0, %arg0 : i32
            return %res : i32
        }
    }
    """
)
mod = MLIR.API.mlirModuleCreateParse(ctx, ir)
lowerModuleToLLVM(ctx, mod)

MLIR.API.mlirRegisterAllLLVMTranslations(ctx)

# TODO: We are missing the mlirExecutionEngine or translateModuleToLLVMIR

MLIR.API.mlirModuleDestroy(mod)
MLIR.API.mlirContextDestroy(ctx)
