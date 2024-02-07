using LLVM
using MLIR.IR
using MLIR.Dialects: arith, linalg, func

n = 128
a = rand(Float64, n, n)
b = rand(Float64, n, n)

fptr = IR.context!(IR.Context()) do
    mod = IR.Module(Location())
    body = IR.get_body(mod)

    # Create a function
    block = IR.Block()
    op = linalg.matmul(...) # TODO
    push!(block, op)

    region = IR.Region()
    push!(region, block)

    ftype = IR.FunctionType( # TODO
        inputs=MLIRType[...],
        results=MLIRType[...],
    )
    f = func.func_(;
        sym_name=IR.Attribute("matmul_demo"),
        function_type=IR.Attribute(...), # TODO
        owned_regions=Region[region],
        result_inference=false,
    )
    push!(body, f)

    pm = IR.PassManager()
    opm = IR.OpPassManager(pm)

    IR.enable_ir_printing!(pm)
    IR.enable_verifier!(pm, true)

    MLIR.API.mlirRegisterAllPasses()
    MLIR.API.mlirRegisterAllLLVMTranslations(IR.context())
    IR.add_pipeline!(opm, "convert-linalg-to-loops,convert-func-to-llvm")

    IR.run!(pm, mod)

    jit = if LLVM.version() >= v"16"
        MLIR.API.mlirExecutionEngineCreate(mod, 0, 0, C_NULL, false)
    else
        MLIR.API.mlirExecutionEngineCreate(mod, 0, 0, C_NULL)
    end
    MLIR.API.mlirExecutionEngineLookup(jit, "matmul_demo")
end

@test ccall(fptr, Ptr{Float64}, (Ptr{Float64}, Ptr{Float64}), a, b) ≈ a * b
