using LLVM
using MLIR.IR
using MLIR.API
using MLIR.Dialects: arith, linalg, emitc, func, vector, memref, llvm, operandsegmentsizes, namedattribute
using MLIR_jll
using Libdl

n = 2
a = rand(Float64, n, n)
b = rand(Float64, n, n)
c = zeros(Float64, n, n)

function cfunc(; sym_name, function_type, sym_visibility=nothing, body::IR.Region, location=IR.Location())
    results = MLIRType[]
    operands = IR.Value[]
    owned_regions = IR.Region[body,]
    successors = IR.Block[]
    attributes = IR.NamedAttribute[namedattribute("sym_name", sym_name), namedattribute("function_type", function_type), namedattribute("llvm.emit_c_interface", API.mlirUnitAttrGet(IR.context()))]
    !isnothing(sym_visibility) && push!(attributes, namedattribute("sym_visibility", sym_visibility))

    IR.create_operation(
        "func.func", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

# required for vector.print -> imports "printF64", "printNewline" symbols
# dlopen(joinpath(MLIR_jll.artifact_dir, "lib", "libmlir_c_runner_utils.$(dlext)"))

fptr = IR.context!(IR.Context()) do
    IR.enable_multithreading!(false)

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
    # mattype = MLIRType(API.mlirMemRefTypeContiguousGet(scalartype, 2, [n, n], API.mlirAttributeGetNull()))
    layout_map_col_major = API.mlirAffineMapAttrGet(API.mlirAffineMapPermutationGet(IR.context(), 2, pointer([1, 0])))
    mattype = MLIRType(API.mlirMemRefTypeGet(scalartype, 2, [n, n], layout_map_col_major, API.mlirAttributeGetNull()))

    linalg_block = IR.Block()
    arg0 = IR.push_argument!(linalg_block, scalartype, IR.Location())
    arg1 = IR.push_argument!(linalg_block, scalartype, IR.Location())
    arg2 = IR.push_argument!(linalg_block, scalartype, IR.Location())

    # push!(linalg_block, vector.print(arg0))
    # push!(linalg_block, vector.print(arg1))
    # push!(linalg_block, vector.print(arg2))

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

    # call matmul
    op = linalg.matmul([a_ir, b_ir], [c_ir]; result_tensors=MLIRType[], region=linalg_region)
    push!(block, op)

    push!(block, func.return_(IR.Value[]))

    region = IR.Region()
    push!(region, block)

    # create "matmul!" function
    ftype = MLIRType(API.mlirFunctionTypeGet(IR.context(), 3, [mattype, mattype, mattype], 0, IR.MLIRType[]))
    f = cfunc(;
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
    # IR.add_pipeline!(opm, "one-shot-bufferize") # if using `tensor` types
    IR.add_pipeline!(opm, "func.func(convert-linalg-to-loops)")
    # IR.add_pipeline!(opm, "convert-vector-to-llvm") # required for vector.print
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
    # API.mlirExecutionEngineLookup(jit, "matmul!")
    API.mlirExecutionEngineLookup(jit, "_mlir_ciface_matmul!")
end

matmul! = fptr

@info "before" a b c

struct MemRefDescritor{T,N}
    allocated::Ptr{T}
    aligned::Ptr{T}
    offset::Int
    sizes::NTuple{N,Int}
    strides::NTuple{N,Int}
end

function MemRefDescritor(arr::Array{T,N}) where {T,N}
    MemRefDescritor(
        pointer(arr),
        pointer(arr),
        0,
        size(arr),
        strides(arr),
    )
end

ccall(matmul!, Cvoid, (MemRefDescritor{Float64,2}, MemRefDescritor{Float64,2}, MemRefDescritor{Float64,2}), MemRefDescritor(a), MemRefDescritor(b), MemRefDescritor(c))

@info "after" a b c

@info "" a * b (a' * b')'
