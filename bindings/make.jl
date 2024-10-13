using Pkg
import BinaryBuilderBase:
    PkgSpec, Prefix, temp_prefix, setup_dependencies, cleanup_dependencies, destdir
using Clang.Generators

function mlir_dialects(version::VersionNumber)
    # construct set of dialects to generate bindings for
    # 1. dialect name
    # 2. bindings file name
    # 3. tablegen files

    dialects = if v"14" <= version < v"15"
        [
            ("acc", "OpenACC.jl", ["OpenACC/OpenACCOps.td"]),
            ("affine", "Affine.jl", ["Affine/IR/AffineOps.td"]),
            ("amx", "AMX.jl", ["AMX/AMX.td"]),
            ("arith", "Arithmetic.jl", ["Arithmetic/IR/ArithmeticOps.td"]), # folder renamed to 'Arith' in v16
            ("arm_neon", "ArmNeon.jl", ["ArmNeon/ArmNeon.td"]),
            ("arm_sve", "ArmSVE.jl", ["ArmSVE/ArmSVE.td"]),
            ("async", "Async.jl", ["Async/IR/AsyncOps.td"]),
            ("bufferization", "Bufferization.jl", ["Bufferization/IR/BufferizationOps.td"]),
            ("builtin", "Builtin.jl", ["../IR/BuiltinOps.td"]),
            ("complex", "Complex.jl", ["Complex/IR/ComplexOps.td"]),
            # ("dlti", "DLTI.jl", ["DLTI/DLTI.td"]), # TODO crashes
            ("emitc", "EmitC.jl", ["EmitC/IR/EmitC.td"]),
            ("gpu", "GPU.jl", ["GPU/GPUOps.td"]), # moved to IR subfolder in v15
            (
                "linalg",
                "Linalg.jl",
                ["Linalg/IR/LinalgOps.td", "Linalg/IR/LinalgStructuredOps.td"],
            ),
            (
                "llvm",
                "LLVMIR.jl",
                ["LLVMIR/LLVMOps.td", "LLVMIR/NVVMOps.td", "LLVMIR/ROCDLOps.td"],
            ),
            ("math", "Math.jl", ["Math/IR/MathOps.td"]),
            ("memref", "MemRef.jl", ["MemRef/IR/MemRefOps.td"]),
            ("omp", "OpenMP.jl", ["OpenMP/OpenMPOps.td"]),
            ("pdl_interp", "PDLInterp.jl", ["PDLInterp/IR/PDLInterpOps.td"]),
            ("pdl", "PDL.jl", ["PDL/IR/PDLOps.td"]),
            ("quant", "Quant.jl", ["Quant/QuantOps.td"]),
            ("scf", "SCF.jl", ["SCF/SCFOps.td"]), # moved to IR subfolder in v15
            ("shape", "Shape.jl", ["Shape/IR/ShapeOps.td"]),
            ("sparse_tensor", "SparseTensor.jl", ["SparseTensor/IR/SparseTensorOps.td"]),
            ("spv", "SPIRV.jl", ["SPIRV/IR/SPIRVOps.td"]), # dialect name renamed to 'spirv' in v16
            ("std", "StandardOps.jl", ["StandardOps/IR/Ops.td"]),
            ("tensor", "Tensor.jl", ["Tensor/IR/TensorOps.td"]),
            ("tosa", "Tosa.jl", ["Tosa/IR/TosaOps.td"]),
            ("vector", "Vector.jl", ["Vector/IR/VectorOps.td"]),
            ("x86vector", "X86Vector.jl", ["X86Vector/X86Vector.td"]),
        ]
    elseif v"15" <= version < v"16"
        [
            ("acc", "OpenACC.jl", ["OpenACC/OpenACCOps.td"]),
            ("affine", "Affine.jl", ["Affine/IR/AffineOps.td"]),
            ("amdgpu", "AMDGPU.jl", ["AMDGPU/AMDGPU.td"]),
            ("amx", "AMX.jl", ["AMX/AMX.td"]),
            ("arith", "Arithmetic.jl", ["Arithmetic/IR/ArithmeticOps.td"]), # folder renamed to 'Arith' in v16
            ("arm_neon", "ArmNeon.jl", ["ArmNeon/ArmNeon.td"]),
            ("arm_sve", "ArmSVE.jl", ["ArmSVE/ArmSVE.td"]),
            ("async", "Async.jl", ["Async/IR/AsyncOps.td"]),
            ("bufferization", "Bufferization.jl", ["Bufferization/IR/BufferizationOps.td"]),
            ("builtin", "Builtin.jl", ["../IR/BuiltinOps.td"]),
            ("cf", "ControlFlow.jl", ["ControlFlow/IR/ControlFlowOps.td"]),
            ("complex", "Complex.jl", ["Complex/IR/ComplexOps.td"]),
            # ("dlti", "DLTI.jl"[, "DLTI/DLTI.td"]), # TODO crashes
            ("emitc", "EmitC.jl", ["EmitC/IR/EmitC.td"]),
            ("func", "Func.jl", ["Func/IR/FuncOps.td"]),
            ("gpu", "GPU.jl", ["GPU/IR/GPUOps.td"]),
            (
                "linalg",
                "Linalg.jl",
                ["Linalg/IR/LinalgOps.td", "Linalg/IR/LinalgStructuredOps.td"],
            ),
            (
                "llvm",
                "LLVMIR.jl",
                [
                    "LLVMIR/LLVMOps.td",
                    "LLVMIR/LLVMIntrinsicOps.td",
                    "LLVMIR/NVVMOps.td",
                    "LLVMIR/ROCDLOps.td",
                ],
            ),
            ("math", "Math.jl", ["Math/IR/MathOps.td"]),
            ("memref", "MemRef.jl", ["MemRef/IR/MemRefOps.td"]),
            ("ml_program", "MLProgram.jl", ["MLProgram/IR/MLProgramOps.td"]),
            ("nvgpu", "NVGPU.jl", ["NVGPU/IR/NVGPU.td"]),
            ("omp", "OpenMP.jl", ["OpenMP/OpenMPOps.td"]),
            ("pdl_interp", "PDLInterp.jl", ["PDLInterp/IR/PDLInterpOps.td"]),
            ("pdl", "PDL.jl", ["PDL/IR/PDLOps.td"]),
            ("quant", "Quant.jl", ["Quant/QuantOps.td"]),
            ("scf", "SCF.jl", ["SCF/IR/SCFOps.td"]),
            ("shape", "Shape.jl", ["Shape/IR/ShapeOps.td"]),
            ("sparse_tensor", "SparseTensor.jl", ["SparseTensor/IR/SparseTensorOps.td"]),
            ("spv", "SPIRV.jl", ["SPIRV/IR/SPIRVOps.td"]), # dialect name renamed to 'spirv' in v16
            ("tensor", "Tensor.jl", ["Tensor/IR/TensorOps.td"]),
            ("tosa", "Tosa.jl", ["Tosa/IR/TosaOps.td"]),
            (
                "transform",
                "Transform.jl",
                [
                    "Bufferization/TransformOps/BufferizationTransformOps.td",
                    "Linalg/TransformOps/LinalgTransformOps.td",
                    "SCF/TransformOps/SCFTransformOps.td",
                    "Transform/IR/TransformOps.td",
                ],
            ), # more ops files in v16
            ("vector", "Vector.jl", ["Vector/IR/VectorOps.td"]),
            ("x86vector", "X86Vector.jl", ["X86Vector/X86Vector.td"]),
        ]
    elseif v"16" <= version < v"17"
        [
            ("acc", "OpenACC.jl", ["OpenACC/OpenACCOps.td"]),
            ("affine", "Affine.jl", ["Affine/IR/AffineOps.td"]),
            ("amdgpu", "AMDGPU.jl", ["AMDGPU/AMDGPU.td"]),
            ("amx", "AMX.jl", ["AMX/AMX.td"]),
            ("arith", "Arith.jl", ["Arith/IR/ArithOps.td"]),
            ("arm_neon", "ArmNeon.jl", ["ArmNeon/ArmNeon.td"]),
            ("arm_sve", "ArmSVE.jl", ["ArmSVE/ArmSVE.td"]),
            ("async", "Async.jl", ["Async/IR/AsyncOps.td"]),
            ("bufferization", "Bufferization.jl", ["Bufferization/IR/BufferizationOps.td"]),
            ("builtin", "Builtin.jl", ["../IR/BuiltinOps.td"]),
            ("cf", "ControlFlow.jl", ["ControlFlow/IR/ControlFlowOps.td"]),
            ("complex", "Complex.jl", ["Complex/IR/ComplexOps.td"]),
            # ("dlti", "DLTI.jl"[, "DLTI/DLTI.td"]), # TODO crashes
            ("emitc", "EmitC.jl", ["EmitC/IR/EmitC.td"]),
            ("func", "Func.jl", ["Func/IR/FuncOps.td"]),
            ("gpu", "GPU.jl", ["GPU/IR/GPUOps.td"]),
            ("index", "Index.jl", ["Index/IR/IndexOps.td"]),
            (
                "linalg",
                "Linalg.jl",
                ["Linalg/IR/LinalgOps.td", "Linalg/IR/LinalgStructuredOps.td"],
            ),
            (
                "llvm",
                "LLVMIR.jl",
                [
                    "LLVMIR/LLVMOps.td",
                    "LLVMIR/LLVMIntrinsicOps.td",
                    "LLVMIR/NVVMOps.td",
                    "LLVMIR/ROCDLOps.td",
                ],
            ),
            ("math", "Math.jl", ["Math/IR/MathOps.td"]),
            ("memref", "MemRef.jl", ["MemRef/IR/MemRefOps.td"]),
            ("ml_program", "MLProgram.jl", ["MLProgram/IR/MLProgramOps.td"]),
            ("nvgpu", "NVGPU.jl", ["NVGPU/IR/NVGPU.td"]),
            ("omp", "OpenMP.jl", ["OpenMP/OpenMPOps.td"]),
            ("pdl_interp", "PDLInterp.jl", ["PDLInterp/IR/PDLInterpOps.td"]),
            ("pdl", "PDL.jl", ["PDL/IR/PDLOps.td"]),
            ("quant", "Quant.jl", ["Quant/QuantOps.td"]),
            ("scf", "SCF.jl", ["SCF/IR/SCFOps.td"]),
            ("shape", "Shape.jl", ["Shape/IR/ShapeOps.td"]),
            ("sparse_tensor", "SparseTensor.jl", ["SparseTensor/IR/SparseTensorOps.td"]),
            ("spirv", "SPIRV.jl", ["SPIRV/IR/SPIRVOps.td"]),
            ("tensor", "Tensor.jl", ["Tensor/IR/TensorOps.td"]),
            ("tosa", "Tosa.jl", ["Tosa/IR/TosaOps.td"]),
            (
                "transform",
                "Transform.jl",
                [
                    "Affine/TransformOps/AffineTransformOps.td",
                    "Bufferization/TransformOps/BufferizationTransformOps.td",
                    "GPU/TransformOps/GPUTransformOps.td",
                    "Linalg/TransformOps/LinalgTransformOps.td",
                    "MemRef/TransformOps/MemRefTransformOps.td",
                    "SCF/TransformOps/SCFTransformOps.td",
                    "Transform/IR/TransformOps.td",
                    "Vector/TransformOps/VectorTransformOps.td",
                ],
            ), # more ops files in v17
            ("vector", "Vector.jl", ["Vector/IR/VectorOps.td"]),
            ("x86vector", "X86Vector.jl", ["X86Vector/X86Vector.td"]),
        ]
    elseif v"17" <= version < v"18"
        [
            ("acc", "OpenACC.jl", ["OpenACC/OpenACCOps.td"]),
            ("affine", "Affine.jl", ["Affine/IR/AffineOps.td"]),
            ("amdgpu", "AMDGPU.jl", ["AMDGPU/IR/AMDGPU.td"]),
            ("amx", "AMX.jl", ["AMX/AMX.td"]),
            ("arith", "Arith.jl", ["Arith/IR/ArithOps.td"]),
            ("arm_neon", "ArmNeon.jl", ["ArmNeon/ArmNeon.td"]),
            ("arm_sme", "ArmSME.jl", ["ArmSME/IR/ArmSME.td"]),
            ("arm_sve", "ArmSVE.jl", ["ArmSVE/ArmSVE.td"]),
            ("async", "Async.jl", ["Async/IR/AsyncOps.td"]),
            ("bufferization", "Bufferization.jl", ["Bufferization/IR/BufferizationOps.td"]),
            ("builtin", "Builtin.jl", ["../IR/BuiltinOps.td"]),
            ("cf", "ControlFlow.jl", ["ControlFlow/IR/ControlFlowOps.td"]),
            ("complex", "Complex.jl", ["Complex/IR/ComplexOps.td"]),
            # ("dlti", "DLTI.jl", ["DLTI/DLTI.td"]), # TODO crashes
            ("emitc", "EmitC.jl", ["EmitC/IR/EmitC.td"]),
            ("func", "Func.jl", ["Func/IR/FuncOps.td"]),
            ("gpu", "GPU.jl", ["GPU/IR/GPUOps.td"]),
            ("index", "Index.jl", ["Index/IR/IndexOps.td"]),
            ("irdl", "IRDL.jl", ["IRDL/IR/IRDLOps.td"]),
            (
                "linalg",
                "Linalg.jl",
                ["Linalg/IR/LinalgOps.td", "Linalg/IR/LinalgStructuredOps.td"],
            ),
            (
                "llvm",
                "LLVMIR.jl",
                [
                    "LLVMIR/LLVMOps.td",
                    "LLVMIR/LLVMIntrinsicOps.td",
                    "LLVMIR/NVVMOps.td",
                    "LLVMIR/ROCDLOps.td",
                ],
            ),
            ("math", "Math.jl", ["Math/IR/MathOps.td"]),
            ("memref", "MemRef.jl", ["MemRef/IR/MemRefOps.td"]),
            ("ml_program", "MLProgram.jl", ["MLProgram/IR/MLProgramOps.td"]),
            ("nvgpu", "NVGPU.jl", ["NVGPU/IR/NVGPU.td"]),
            ("omp", "OpenMP.jl", ["OpenMP/OpenMPOps.td"]),
            ("pdl_interp", "PDLInterp.jl", ["PDLInterp/IR/PDLInterpOps.td"]),
            ("pdl", "PDL.jl", ["PDL/IR/PDLOps.td"]),
            ("quant", "Quant.jl", ["Quant/QuantOps.td"]),
            ("scf", "SCF.jl", ["SCF/IR/SCFOps.td"]),
            ("shape", "Shape.jl", ["Shape/IR/ShapeOps.td"]),
            ("sparse_tensor", "SparseTensor.jl", ["SparseTensor/IR/SparseTensorOps.td"]),
            ("spirv", "SPIRV.jl", ["SPIRV/IR/SPIRVOps.td"]),
            ("tensor", "Tensor.jl", ["Tensor/IR/TensorOps.td"]),
            ("tosa", "Tosa.jl", ["Tosa/IR/TosaOps.td"]),
            (
                "transform",
                "Transform.jl",
                [
                    "Affine/TransformOps/AffineTransformOps.td",
                    "Bufferization/TransformOps/BufferizationTransformOps.td",
                    "GPU/TransformOps/GPUTransformOps.td",
                    "Linalg/TransformOps/LinalgMatchOps.td",
                    "Linalg/TransformOps/LinalgTransformOps.td",
                    "MemRef/TransformOps/MemRefTransformOps.td",
                    "NVGPU/TransformOps/NVGPUTransformOps.td",
                    "SCF/TransformOps/SCFTransformOps.td",
                    "Tensor/TransformOps/TensorTransformOps.td",
                    "Transform/IR/TransformOps.td",
                    "Vector/TransformOps/VectorTransformOps.td",
                ],
            ),
            ("ub", "UB.jl", ["UB/IR/UBOps.td"]),
            ("vector", "Vector.jl", ["Vector/IR/VectorOps.td"]),
            ("x86vector", "X86Vector.jl", ["X86Vector/X86Vector.td"]),
        ]
    elseif v"18" <= version < v"19"
        [
            ("acc", "OpenACC.jl", ["OpenACC/OpenACCOps.td"]),
            ("affine", "Affine.jl", ["Affine/IR/AffineOps.td"]),
            ("amdgpu", "AMDGPU.jl", ["AMDGPU/IR/AMDGPU.td"]),
            ("amx", "AMX.jl", ["AMX/AMX.td"]),
            ("arith", "Arith.jl", ["Arith/IR/ArithOps.td"]),
            ("arm_neon", "ArmNeon.jl", ["ArmNeon/ArmNeon.td"]),
            (
                "arm_sme",
                "ArmSME.jl",
                ["ArmSME/IR/ArmSMEOps.td", "ArmSME/IR/ArmSMEIntrinsicOps.td"],
            ),
            ("arm_sve", "ArmSVE.jl", ["ArmSVE/IR/ArmSVE.td"]),
            ("async", "Async.jl", ["Async/IR/AsyncOps.td"]),
            ("bufferization", "Bufferization.jl", ["Bufferization/IR/BufferizationOps.td"]),
            ("builtin", "Builtin.jl", ["../IR/BuiltinOps.td"]),
            ("cf", "ControlFlow.jl", ["ControlFlow/IR/ControlFlowOps.td"]),
            ("complex", "Complex.jl", ["Complex/IR/ComplexOps.td"]),
            # ("dlti", "DLTI.jl", ["DLTI/DLTI.td"]), # TODO crashes
            ("emitc", "EmitC.jl", ["EmitC/IR/EmitC.td"]),
            ("func", "Func.jl", ["Func/IR/FuncOps.td"]),
            ("gpu", "GPU.jl", ["GPU/IR/GPUOps.td"]),
            ("index", "Index.jl", ["Index/IR/IndexOps.td"]),
            ("irdl", "IRDL.jl", ["IRDL/IR/IRDLOps.td"]),
            (
                "linalg",
                "Linalg.jl",
                ["Linalg/IR/LinalgOps.td", "Linalg/IR/LinalgStructuredOps.td"],
            ),
            (
                "llvm",
                "LLVMIR.jl",
                [
                    "LLVMIR/LLVMOps.td",
                    "LLVMIR/LLVMIntrinsicOps.td",
                    "LLVMIR/NVVMOps.td",
                    "LLVMIR/ROCDLOps.td",
                ],
            ),
            ("math", "Math.jl", ["Math/IR/MathOps.td"]),
            ("memref", "MemRef.jl", ["MemRef/IR/MemRefOps.td"]),
            ("mesh", "Mesh.jl", ["Mesh/IR/MeshOps.td"]),
            ("ml_program", "MLProgram.jl", ["MLProgram/IR/MLProgramOps.td"]),
            ("nvgpu", "NVGPU.jl", ["NVGPU/IR/NVGPU.td"]),
            ("omp", "OpenMP.jl", ["OpenMP/OpenMPOps.td"]),
            ("pdl_interp", "PDLInterp.jl", ["PDLInterp/IR/PDLInterpOps.td"]),
            ("pdl", "PDL.jl", ["PDL/IR/PDLOps.td"]),
            ("quant", "Quant.jl", ["Quant/QuantOps.td"]),
            ("scf", "SCF.jl", ["SCF/IR/SCFOps.td"]),
            ("shape", "Shape.jl", ["Shape/IR/ShapeOps.td"]),
            ("sparse_tensor", "SparseTensor.jl", ["SparseTensor/IR/SparseTensorOps.td"]),
            ("spirv", "SPIRV.jl", ["SPIRV/IR/SPIRVOps.td"]),
            ("tensor", "Tensor.jl", ["Tensor/IR/TensorOps.td"]),
            ("tosa", "Tosa.jl", ["Tosa/IR/TosaOps.td"]),
            (
                "transform",
                "Transform.jl",
                [
                    "Affine/TransformOps/AffineTransformOps.td",
                    "Bufferization/TransformOps/BufferizationTransformOps.td",
                    "Func/TransformOps/FuncTransformOps.td",
                    "GPU/TransformOps/GPUTransformOps.td",
                    "Linalg/TransformOps/LinalgMatchOps.td",
                    "Linalg/TransformOps/LinalgTransformOps.td",
                    "MemRef/TransformOps/MemRefTransformOps.td",
                    "NVGPU/TransformOps/NVGPUTransformOps.td",
                    "SCF/TransformOps/SCFTransformOps.td",
                    "SparseTensor/TransformOps/SparseTensorTransformOps.td",
                    "Tensor/TransformOps/TensorTransformOps.td",
                    "Transform/IR/TransformOps.td",
                    "Transform/DebugExtension/DebugExtensionOps.td",
                    "Transform/LoopExtension/LoopExtensionOps.td",
                    "Transform/PDLExtension/PDLExtensionOps.td",
                    "Vector/TransformOps/VectorTransformOps.td",
                ],
            ),
            ("ub", "UB.jl", ["UB/IR/UBOps.td"]),
            ("vector", "Vector.jl", ["Vector/IR/VectorOps.td"]),
            ("x86vector", "X86Vector.jl", ["X86Vector/X86Vector.td"]),
        ]
    elseif v"19" <= version < v"20"
        [
            ("acc", "OpenACC.jl", ["OpenACC/OpenACCOps.td"]),
            ("affine", "Affine.jl", ["Affine/IR/AffineOps.td"]),
            ("amdgpu", "AMDGPU.jl", ["AMDGPU/IR/AMDGPU.td"]),
            ("amx", "AMX.jl", ["AMX/AMX.td"]),
            ("arith", "Arith.jl", ["Arith/IR/ArithOps.td"]),
            ("arm_neon", "ArmNeon.jl", ["ArmNeon/ArmNeon.td"]),
            (
                "arm_sme",
                "ArmSME.jl",
                ["ArmSME/IR/ArmSMEOps.td", "ArmSME/IR/ArmSMEIntrinsicOps.td"],
            ),
            ("arm_sve", "ArmSVE.jl", ["ArmSVE/IR/ArmSVE.td"]),
            ("async", "Async.jl", ["Async/IR/AsyncOps.td"]),
            ("bufferization", "Bufferization.jl", ["Bufferization/IR/BufferizationOps.td"]),
            ("builtin", "Builtin.jl", ["../IR/BuiltinOps.td"]),
            ("cf", "ControlFlow.jl", ["ControlFlow/IR/ControlFlowOps.td"]),
            ("complex", "Complex.jl", ["Complex/IR/ComplexOps.td"]),
            # ("dlti", "DLTI.jl", ["DLTI/DLTI.td"]), # TODO crashes
            ("emitc", "EmitC.jl", ["EmitC/IR/EmitC.td"]),
            ("func", "Func.jl", ["Func/IR/FuncOps.td"]),
            ("gpu", "GPU.jl", ["GPU/IR/GPUOps.td"]),
            ("index", "Index.jl", ["Index/IR/IndexOps.td"]),
            ("irdl", "IRDL.jl", ["IRDL/IR/IRDLOps.td"]),
            (
                "linalg",
                "Linalg.jl",
                ["Linalg/IR/LinalgOps.td", "Linalg/IR/LinalgStructuredOps.td"],
            ),
            (
                "llvm",
                "LLVMIR.jl",
                [
                    "LLVMIR/LLVMOps.td",
                    "LLVMIR/LLVMIntrinsicOps.td",
                    "LLVMIR/NVVMOps.td",
                    "LLVMIR/ROCDLOps.td",
                ],
            ),
            ("math", "Math.jl", ["Math/IR/MathOps.td"]),
            ("memref", "MemRef.jl", ["MemRef/IR/MemRefOps.td"]),
            ("mesh", "Mesh.jl", ["Mesh/IR/MeshOps.td"]),
            ("ml_program", "MLProgram.jl", ["MLProgram/IR/MLProgramOps.td"]),
            ("mpi", "MPI.jl", ["MPI/IR/MPIOps.td"]),
            ("nvgpu", "NVGPU.jl", ["NVGPU/IR/NVGPU.td"]),
            ("omp", "OpenMP.jl", ["OpenMP/OpenMPOps.td"]),
            ("pdl_interp", "PDLInterp.jl", ["PDLInterp/IR/PDLInterpOps.td"]),
            ("pdl", "PDL.jl", ["PDL/IR/PDLOps.td"]),
            ("polynomial", "Polynomial.jl", ["Polynomial/IR/Polynomial.td"]),
            # ("ptr", "Ptr.jl", ["Ptr/IR/PtrOps.td"]),
            ("quant", "Quant.jl", ["Quant/QuantOps.td"]),
            ("scf", "SCF.jl", ["SCF/IR/SCFOps.td"]),
            ("shape", "Shape.jl", ["Shape/IR/ShapeOps.td"]),
            ("sparse_tensor", "SparseTensor.jl", ["SparseTensor/IR/SparseTensorOps.td"]),
            ("spirv", "SPIRV.jl", ["SPIRV/IR/SPIRVOps.td"]),
            ("tensor", "Tensor.jl", ["Tensor/IR/TensorOps.td"]),
            ("tosa", "Tosa.jl", ["Tosa/IR/TosaOps.td"]),
            (
                "transform",
                "Transform.jl",
                [
                    "Affine/TransformOps/AffineTransformOps.td",
                    "Bufferization/TransformOps/BufferizationTransformOps.td",
                    "Func/TransformOps/FuncTransformOps.td",
                    "GPU/TransformOps/GPUTransformOps.td",
                    "Linalg/TransformOps/LinalgMatchOps.td",
                    "Linalg/TransformOps/LinalgTransformOps.td",
                    "MemRef/TransformOps/MemRefTransformOps.td",
                    "NVGPU/TransformOps/NVGPUTransformOps.td",
                    "SCF/TransformOps/SCFTransformOps.td",
                    "SparseTensor/TransformOps/SparseTensorTransformOps.td",
                    "Tensor/TransformOps/TensorTransformOps.td",
                    "Transform/IR/TransformOps.td",
                    "Transform/DebugExtension/DebugExtensionOps.td",
                    "Transform/LoopExtension/LoopExtensionOps.td",
                    "Transform/PDLExtension/PDLExtensionOps.td",
                    "Vector/TransformOps/VectorTransformOps.td",
                ],
            ),
            ("ub", "UB.jl", ["UB/IR/UBOps.td"]),
            ("vector", "Vector.jl", ["Vector/IR/VectorOps.td"]),
            ("x86vector", "X86Vector.jl", ["X86Vector/X86Vector.td"]),
            ("xegpu", "XeGPU.jl", ["XeGPU/IR/XeGPUOps.td"]),
        ]
    else
        error("Unsupported MLIR version: $version")
    end

    return dialects
end

function rewrite!(dag::ExprDAG) end

julia_llvm = [
    (v"1.9", v"14.0.5+3"),
    (v"1.10", v"15.0.7+10"),
    (v"1.11", v"16.0.6+2"),
    (v"1.12", v"17.0.6+3"),
    (v"1.12", v"18.1.7+2"),
    (v"1.12", v"19.1.1+0"),
]
options = load_options(joinpath(@__DIR__, "wrap.toml"))

@add_def off_t
@add_def MlirTypesCallback

for (julia_version, llvm_version) in julia_llvm
    println("Generating... julia version: $julia_version, llvm version: $llvm_version")

    temp_prefix() do prefix
        platform = Pkg.BinaryPlatforms.HostPlatform()
        platform["llvm_version"] = string(llvm_version.major)
        platform["julia_version"] = string(julia_version)

        # Note: 1.10
        dependencies = PkgSpec[
            PkgSpec(; name="LLVM_full_jll", version=llvm_version),
            PkgSpec(; name="mlir_jl_tblgen_jll"),
        ]

        artifact_paths = setup_dependencies(prefix, dependencies, platform; verbose=true)

        mlir_jl_tblgen = joinpath(destdir(prefix, platform), "bin", "mlir-jl-tblgen")
        include_dir = joinpath(destdir(prefix, platform), "include")

        # generate MLIR API bindings
        mkpath(joinpath(@__DIR__, "..", "src", "API", string(llvm_version.major)))

        let options = deepcopy(options)
            output_file_path = joinpath(
                @__DIR__,
                "..",
                "src",
                "API",
                string(llvm_version.major),
                options["general"]["output_file_path"],
            )
            isdir(dirname(output_file_path)) || mkpath(dirname(output_file_path))
            options["general"]["output_file_path"] = output_file_path

            libmlir_header_dir = joinpath(include_dir, "mlir-c")
            args = Generators.get_default_args(get_triple(); is_cxx=true)
            push!(args, "-I$include_dir")
            push!(args, "-xc++")

            headers = detect_headers(
                libmlir_header_dir, args, Dict(), endswith("Python/Interop.h")
            )
            ctx = create_context(headers, args, options)

            # build without printing so we can do custom rewriting
            build!(ctx, BUILDSTAGE_NO_PRINTING)

            rewrite!(ctx.dag)

            # print
            build!(ctx, BUILDSTAGE_PRINTING_ONLY)
        end

        # generate MLIR dialect bindings
        mkpath(joinpath(@__DIR__, "..", "src", "Dialects", string(llvm_version.major)))

        for (dialect_name, binding, tds) in mlir_dialects(llvm_version)
            tempfiles = map(tds) do td
                tempfile, _ = mktemp()
                tdpath = joinpath(include_dir, "mlir", "Dialect", td)
                flags = [
                    "--generator=jl-op-defs",
                    "--disable-module-wrap",
                    tdpath,
                    "-I",
                    dirname(tdpath),
                    "-I",
                    include_dir,
                    "-o",
                    tempfile,
                ]
                run(`$mlir_jl_tblgen $flags`)
                return tempfile
            end

            output = joinpath(
                @__DIR__, "..", "src", "Dialects", string(llvm_version.major), binding
            )
            open(output, "w") do io
                println(io, "module $dialect_name\n")
                for tempfile in tempfiles
                    write(io, read(tempfile, String))
                end
                println(io, "end # $dialect_name")
            end

            println("- Generated \"$binding\" from $(join(tds, ",", " and "))")
        end
    end
end
