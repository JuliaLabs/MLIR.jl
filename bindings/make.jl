using Pkg
import BinaryBuilderBase: PkgSpec, Prefix, temp_prefix, setup_dependencies, cleanup_dependencies, destdir
using Clang.Generators

function mlir_dialects(version::VersionNumber)
    dialects = Tuple{String,String}[
        ("Builtin.jl", "../IR/BuiltinOps.td"),
    ]

    if version >= v"14"
        append!(dialects, [
            ("AMX.jl", "AMX/AMX.td"),
            ("Affine.jl", "Affine/IR/AffineOps.td"),
            ("ArmNeon.jl", "ArmNeon/ArmNeon.td"),
            ("ArmSVE.jl", "ArmSVE/ArmSVE.td"),
            ("Async.jl", "Async/IR/AsyncOps.td"),
            ("Bufferization.jl", "Bufferization/IR/BufferizationOps.td"),
            ("Complex.jl", "Complex/IR/ComplexOps.td"),
            ("DLTI.jl", "DLTI/DLTI.td"),
            ("EmitC.jl", "EmitC/IR/EmitC.td"),
            ("GPU.jl", "GPU/IR/GPUOps.td"),
            ("LLVMIR.jl", "LLVMIR/LLVMOps.td"),
            ("Linalg.jl", "Linalg/IR/LinalgOps.td"), # TODO include LinalgStructuredOps.td
            ("Math.jl", "Math/IR/MathOps.td"),
            ("MemRef.jl", "MemRef/IR/MemRefOps.td"),
            ("OpenACC.jl", "OpenACC/OpenACCOps.td"),
            ("OpenMP.jl", "OpenMP/OpenMPOps.td"),
            ("PDL.jl", "PDL/IR/PDLOps.td"),
            ("PDLInterp.jl", "PDLInterp/IR/PDLInterpOps.td"),
            ("Quant.jl", "Quant/QuantOps.td"),
            ("SCF.jl", "SCF/IR/SCFOps.td"),
            ("SPIRV.jl", "SPIRV/IR/SPIRVOps.td"),
            ("Shape.jl", "Shape/IR/ShapeOps.td"),
            ("SparseTensor.jl", "SparseTensor/IR/SparseTensorOps.td"),
            ("Tensor.jl", "Tensor/IR/TensorOps.td"),
            ("Tosa.jl", "Tosa/IR/TosaOps.td"),
            ("Vector.jl", "Vector/IR/VectorOps.td"),
            ("X86Vector.jl", "X86Vector/X86Vector.td"),
        ])
    end

    if v"14" <= version < v"15"
        append!(dialects, [
            ("StandardOps.jl", "Standard/IR/StandardOps.td"),
        ])
    end

    if v"14" <= version < v"16"
        append!(dialects, [
            ("Arithmetic.jl", "Arithmetic/IR/ArithmeticOps.td"), # renamed to 'Arith' in v16
        ])
    end

    if version >= v"15"
        append!(dialects, [
            ("AMDGPU.jl", "AMDGPU/AMDGPU.td"),
            ("ControlFlow.jl", "ControlFlow/IR/ControlFlowOps.td"),
            ("Func.jl", "Func/IR/FuncOps.td"),
            ("MLProgram.jl", "MLProgram/IR/MLProgramOps.td"),
            ("NVGPU.jl", "NVGPU/IR/NVGPU.td"),
            ("Transform.jl", "Transform/IR/TransformOps.td"),
        ])
    end

    if version >= v"16"
        append!(dialects, [
            ("Arith.jl", "Arith/IR/ArithOps.td"),
            ("Index.jl", "Index/IR/IndexOps.td"),
        ])
    end

    if version >= v"17"
        append!(dialects, [
            ("ArmSME.jl", "ArmSME/IR/ArmSME.td"),
            ("IRDL.jl", "IRDL/IR/IRDLOps.td"),
            ("UB.jl", "UB/IR/UBOps.td"),
        ])
    end

    if version >= v"18"
        append!(dialects, [
            ("Mesh.jl", "Mesh/IR/MeshOps.td"),
        ])
    end

    return dialects
end

function rewrite!(dag::ExprDAG) end

julia_llvm = Dict([v"1.9" => v"14.0.5", v"1.10" => v"15.0.7"])
options = load_options(joinpath(@__DIR__, "wrap.toml"))

@add_def off_t
@add_def MlirTypesCallback

for (julia_version, llvm_version) in julia_llvm
    println("Generating... julia version: $julia_version, llvm version: $llvm_version")

    temp_prefix() do prefix
        platform = Pkg.BinaryPlatforms.HostPlatform()
        platform["llvm_version"] = string(llvm_version.major)
        platform["julia_version"] = string(julia_version)

        # NOTE LLVM version is implicitly resolved by compatibility with the Julia version
        dependencies = PkgSpec[PkgSpec(; name="LLVM_full_jll")]

        artifact_paths = setup_dependencies(prefix, dependencies, platform; verbose=true)

        include_dir = joinpath(destdir(prefix, platform), "include")

        # generate MLIR API bindings
        mkpath(joinpath(@__DIR__, "..", "src", "API", string(llvm_version.major)))

        let options = deepcopy(options)
            output_file_path = joinpath(@__DIR__, "..", "src", "API", string(llvm_version.major), options["general"]["output_file_path"])
            isdir(dirname(output_file_path)) || mkpath(dirname(output_file_path))
            options["general"]["output_file_path"] = output_file_path

            libmlir_header_dir = joinpath(include_dir, "mlir-c")
            args = Generators.get_default_args()
            append!(args, ["-I", include_dir, "-x", "c++"])

            headers = detect_headers(libmlir_header_dir, args, Dict(), endswith("Python/Interop.h"))
            ctx = create_context(headers, args, options)

            # build without printing so we can do custom rewriting
            build!(ctx, BUILDSTAGE_NO_PRINTING)

            rewrite!(ctx.dag)

            # print
            build!(ctx, BUILDSTAGE_PRINTING_ONLY)
        end

        # generate MLIR dialect bindings
        mkpath(joinpath(@__DIR__, "..", "src", "Dialects", string(llvm_version.major)))

        for (binding, td) in mlir_dialects(llvm_version)
            flags = [
                "--generator=jl-op-defs",
                joinpath(include_dir, "mlir", "Dialect", td),
                "-I", include_dir,
                "-o", joinpath(@__DIR__, "..", "src", "Dialects", string(llvm_version.major), binding),
            ]
            run(`./mlir-jl-tblgen $flags`) # TODO replace for artifact
            println("- Generated \"$binding\" from \"$td\"")
        end
    end
end