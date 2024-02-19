using LLVM
using MLIR.IR
using MLIR.API
using MLIR.Dialects: arith, linalg, func, operandsegmentsizes

n = 2
a = ones(Float64, n, n)
b = ones(Float64, n, n)
c = zeros(Float64, n, n)

fptr = IR.context!(IR.Context()) do
    IR.enable_multithreading!(false)

    # for dialect in ["func", "linalg"]
    #     IR.get_or_load_dialect!(dialect)
    # end
    registry = API.mlirDialectRegistryCreate()
    API.mlirRegisterAllDialects(registry)
    API.mlirContextAppendDialectRegistry(IR.context(), registry)
    API.mlirDialectRegistryDestroy(registry)

    API.mlirContextLoadAllAvailableDialects(IR.context())

    API.mlirRegisterAllPasses()
    API.mlirRegisterAllLLVMTranslations(IR.context())

    mod = IR.Module(Location())
    body = IR.get_body(mod)

    # Create a function    
    scalartype = MLIRType(Float64)
    # mattype = MLIRType(API.mlirRankedTensorTypeGet(2, [n, n], scalartype, API.mlirAttributeGetNull()))
    mattype = MLIRType(API.mlirMemRefTypeContiguousGet(scalartype, 2, [n, n], API.mlirAttributeGetNull()))

    linalg_block = IR.Block()
    arg0 = IR.push_argument!(linalg_block, scalartype, IR.Location())
    arg1 = IR.push_argument!(linalg_block, scalartype, IR.Location())
    arg2 = IR.push_argument!(linalg_block, scalartype, IR.Location())
    op = arith.mulf(arg0, arg1; result=scalartype)
    push!(linalg_block, op)

    op = arith.addf(IR.get_result(op), arg2)
    push!(linalg_block, op)

    op = linalg.yield([IR.get_result(op)])
    push!(linalg_block, op)

    linalg_region = IR.Region()
    push!(linalg_region, linalg_block)

    block = IR.Block()
    a_ir = IR.push_argument!(block, mattype, IR.Location())
    b_ir = IR.push_argument!(block, mattype, IR.Location())
    c_ir = IR.push_argument!(block, mattype, IR.Location())

    op = linalg.matmul([a_ir, b_ir], [c_ir]; result_tensors=MLIRType[mattype], region=linalg_region)
    push!(block, op)

    push!(block, func.return_(IR.Value[]))

    region = IR.Region()
    push!(region, block)

    ftype = MLIRType(API.mlirFunctionTypeGet(IR.context(), 3, [mattype, mattype, mattype], 0, IR.Value[]))
    f = func.func_(;
        sym_name=IR.Attribute("matmul!"),
        function_type=IR.Attribute(ftype),
        body=region,
    )
    IR.verifyall(f)
    push!(body, f)

    pm = IR.PassManager()
    opm = IR.OpPassManager(pm)

    IR.enable_ir_printing!(pm)
    IR.enable_verifier!(pm, true)

    API.mlirRegisterAllPasses()
    API.mlirRegisterAllLLVMTranslations(IR.context())
    IR.add_pipeline!(opm, "func.func(convert-linalg-to-loops)")
    IR.add_pipeline!(opm, "func.func(convert-scf-to-cf)")
    IR.add_pipeline!(opm, "convert-linalg-to-llvm")
    IR.add_pipeline!(opm, "convert-memref-to-llvm")
    IR.add_pipeline!(opm, "func.func(convert-arith-to-llvm)")
    IR.add_pipeline!(opm, "convert-func-to-llvm")
    IR.add_pipeline!(opm, "reconcile-unrealized-casts")

    IR.run!(pm, mod)

    jit = if LLVM.version() >= v"16"
        API.mlirExecutionEngineCreate(mod, 0, 0, C_NULL, false)
    else
        API.mlirExecutionEngineCreate(mod, 0, 0, C_NULL)
    end
    API.mlirExecutionEngineLookup(jit, "matmul!")
end

matmul! = fptr

resptr = ccall(matmul!, Ptr{Float64}, (Ptr{Float64}, Ptr{Float64}, Ptr{Float64}), pointer(a), pointer(b), pointer(c))

@show a b c
@show pointer(a) pointer(b) pointer(c) resptr
@show unsafe_wrap(Array, resptr, (2, 2); own=true)
