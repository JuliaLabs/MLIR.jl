using MLIR.IR
using MLIR.API
using MLIR.Dialects: arith, linalg, func, operandsegmentsizes

n = 128
a = rand(Float64, n, n)
b = rand(Float64, n, n)

function linalg_yield(x::IR.Value; location=IR.Location())
    IR.create_operation(
        "linalg.yield", location;
        operands=[x],
        owned_regions=IR.Region[],
        successors=IR.Block[],
        results=nothing,
        result_inference=false,
        attributes=IR.NamedAttribute[operandsegmentsizes([1])],
    )
end

# TODO remove this when bindings are regenerated
function linalg_matmul(c::IR.Value, a::IR.Value, b::IR.Value; region, result::Union{Nothing,IR.MLIRType}=nothing, location=IR.Location())
    IR.create_operation(
        "linalg.matmul", location;
        operands=[a, b, c],
        owned_regions=IR.Region[region],
        successors=IR.Block[],
        results=isnothing(result) ? nothing : MLIRType[result],
        attributes=IR.NamedAttribute[operandsegmentsizes([2, 1])],
    )
end

fptr = IR.context!(IR.Context()) do
    IR.enable_multithreading!(false)

    for dialect in ["func", "linalg"]
        IR.get_or_load_dialect!(dialect)
    end

    mod = IR.Module(Location())
    body = IR.get_body(mod)

    # Create a function    
    scalartype = MLIRType(Float64)
    mattype = MLIRType(API.mlirRankedTensorTypeGet(2, [n, n], scalartype, API.mlirAttributeGetNull()))
    # mattype = MLIRType(API.mlirMemRefTypeContiguousGet(MLIRType(Float64), 2, [n, n], API.mlirAttributeGetNull()))

    linalg_block = IR.Block()
    arg0 = IR.push_argument!(linalg_block, scalartype, IR.Location())
    arg1 = IR.push_argument!(linalg_block, scalartype, IR.Location())
    arg2 = IR.push_argument!(linalg_block, scalartype, IR.Location())
    op = arith.mulf(arg0, arg1; result=scalartype)
    push!(linalg_block, op)

    op = arith.addf(arg2, IR.get_result(op))
    push!(linalg_block, op)

    op = linalg_yield(IR.get_result(op))
    push!(linalg_block, op)

    linalg_region = IR.Region()
    push!(linalg_region, linalg_block)

    block = IR.Block()
    c_ir = IR.push_argument!(block, mattype, IR.Location())
    a_ir = IR.push_argument!(block, mattype, IR.Location())
    b_ir = IR.push_argument!(block, mattype, IR.Location())

    op = linalg_matmul(c_ir, a_ir, b_ir; region=linalg_region, result=mattype) # TODO refactor to `linalg.matmul` when bindings are regenerated
    push!(block, op)

    push!(block, func.return_([IR.get_result(op)]))

    region = IR.Region()
    push!(region, block)

    ftype = MLIRType(API.mlirFunctionTypeGet(IR.context(), 2, [mattype, mattype], 1, [mattype]))
    f = func.func_(;
        sym_name=IR.Attribute("matmul_demo"),
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
    # IR.add_pipeline!(opm, "convert-func-to-llvm")

    IR.run!(pm, mod)

    # jit = if LLVM.version() >= v"16"
    #     API.mlirExecutionEngineCreate(mod, 0, 0, C_NULL, false)
    # else
    #     API.mlirExecutionEngineCreate(mod, 0, 0, C_NULL)
    # end
    # API.mlirExecutionEngineLookup(jit, "matmul_demo")
end

# @test ccall(fptr, Ptr{Float64}, (Ptr{Float64}, Ptr{Float64}), a, b) ≈ a * b
