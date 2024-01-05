using LLVM_full_jll

println("Environment")
println("- llvm-config = $(LLVM_full_jll.get_llvm_config_path())")
println("- clang = $(LLVM_full_jll.get_clang_path())")

CXXFLAGS = `$(llvm_config()) --cxxflags` |> readchomp |> split
LDFLAGS = `$(llvm_config()) --ldflags` |> readchomp |> split
println("- CXXFLAGS = $CXXFLAGS")
println("- LDFLAGS = $LDFLAGS")

INCLUDE_PATH = joinpath(LLVM_full_jll.artifact_dir, "include")
DIALECTS_PATH = joinpath(INCLUDE_PATH, "mlir", "Dialect")
println("- INCLUDE_PATH = $INCLUDE_PATH")
println("- DIALECTS_PATH = $DIALECTS_PATH")

# compile TableGen generator
println("Compiling TableGen generator...")
files = [joinpath(@__DIR__, "tblgen", "mlir-jl-tblgen.cc"), joinpath(@__DIR__, "tblgen", "jl-generators.cc")]
output = ["-o", "mlir-jl-tblgen"]
libs = ["-lLLVM", "-lMLIR", "-lMLIRTableGen", "-lLLVMTableGen"]

extra = ["-rpath", joinpath(LLVM_full_jll.artifact_dir, "lib")]
if Base.Sys.isapple()
    isysroot = strip(read(`xcrun --show-sdk-path`, String))
    append!(extra, [
        "-isysroot",
        isysroot,
        "-lc++",
    ])
elseif Base.Sys.islinux()
    append!(extra, [
        "-lstdc++",
    ])
end
println("- extra flags = $extra")

run(`$(clang()) $files $CXXFLAGS $LDFLAGS $extra $libs $output`)

# generate bindings
println("Generating bindings...")

target_dialects = [
    ("Builtin.jl", "../IR/BuiltinOps.td"),
    ("AMDGPU.jl", "AMDGPU/AMDGPU.td"),
    ("AMX.jl", "AMX/AMX.td"),
    ("Affine.jl", "Affine/IR/AffineOps.td"),
    ("Arithmetic.jl", "Arithmetic/IR/ArithmeticOps.td"),
    # ("ArmNeon.jl", "ArmNeon/ArmNeon.td"),
    ("ArmSVE.jl", "ArmSVE/ArmSVE.td"),
    ("Async.jl", "Async/IR/AsyncOps.td"),
    ("Bufferization.jl", "Bufferization/IR/BufferizationOps.td"),
    ("Complex.jl", "Complex/IR/ComplexOps.td"),
    ("ControlFlow.jl", "ControlFlow/IR/ControlFlowOps.td"),
    # ("DLTI.jl", "DLTI/DLTI.td"),
    ("EmitC.jl", "EmitC/IR/EmitC.td"),
    ("Func.jl", "Func/IR/FuncOps.td"),
    # ("GPU.jl", "GPU/IR/GPUOps.td"),
    ("Linalg.jl", "Linalg/IR/LinalgOps.td"),
    # ("LinalgStructured.jl", "Linalg/IR/LinalgStructuredOps.td"),
    ("LLVMIR.jl", "LLVMIR/LLVMOps.td"),
    # ("MLProgram.jl", "MLProgram/IR/MLProgramOps.td"),
    ("Math.jl", "Math/IR/MathOps.td"),
    ("MemRef.jl", "MemRef/IR/MemRefOps.td"),
    ("NVGPU.jl", "NVGPU/IR/NVGPU.td"),
    # ("OpenACC.jl", "OpenACC/OpenACCOps.td"),
    # ("OpenMP.jl", "OpenMP/OpenMPOps.td"),
    # ("PDL.jl", "PDL/IR/PDLOps.td"),
    # ("PDLInterp.jl", "PDLInterp/IR/PDLInterpOps.td"),
    ("Quant.jl", "Quant/QuantOps.td"),
    # ("SCF.jl", "SCF/IR/SCFOps.td"),
    # ("SPIRV.jl", "SPIRV/IR/SPIRVOps.td"),
    ("Shape.jl", "Shape/IR/ShapeOps.td"),
    ("SparseTensor.jl", "SparseTensor/IR/SparseTensorOps.td"),
    ("Tensor.jl", "Tensor/IR/TensorOps.td"),
    # ("Tosa.jl", "Tosa/IR/TosaOps.td"),
    ("Transform.jl", "Transform/IR/TransformOps.td"),
    ("Vector.jl", "Vector/IR/VectorOps.td"),
    # ("X86Vector.jl", "X86Vector/X86Vector.td"),
]

for (file, path) in target_dialects
    output = joinpath(@__DIR__, "..", "src", "dialects", file)
    run(`./mlir-jl-tblgen --generator=jl-op-defs $(joinpath(DIALECTS_PATH, path)) -I$INCLUDE_PATH -o $output`)
    println("- Generated \"$output\" from \"$path\"")
end
