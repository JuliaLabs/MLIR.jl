#! /usr/bin/env bash
set -e
cd $(dirname "$0")

make mlir-jl-tblgen

rm -f ./output/*.jl
mkdir -p ./output

LD_LIBRARY_PATH=/home/jumerckx/masterthesis/llvm-project/llvm/build_debug/lib/
DIALECTS_PATH=/home/jumerckx/masterthesis/llvm-project/llvm/install_debug/include/mlir/Dialect/
INCLUDE_PATH=/home/jumerckx/masterthesis/llvm-project/llvm/install_debug/include/

# LD_LIBRARY_PATH=/home/jumerckx/.julia/artifacts/7a30d5d08131c8d72e002314ee933895a1bed594/mlir/lib/
# INCLUDE_PATH=/home/jumerckx/.julia/artifacts/7a30d5d08131c8d72e002314ee933895a1bed594/mlir/include/

LD_LIBRARY_PATH=$LD_LIBRARY_PATH ./mlir-jl-tblgen --generator=jl-op-defs $DIALECTS_PATH/Arith/IR/ArithOps.td -I$INCLUDE_PATH > output/Arith.jl
LD_LIBRARY_PATH=$LD_LIBRARY_PATH ./mlir-jl-tblgen --generator=jl-op-defs $DIALECTS_PATH/Shape/IR/ShapeOps.td -I$INCLUDE_PATH > output/Shape.jl
LD_LIBRARY_PATH=$LD_LIBRARY_PATH ./mlir-jl-tblgen --generator=jl-op-defs $DIALECTS_PATH/MemRef/IR/MemRefOps.td -I$INCLUDE_PATH > output/MemRef.jl
LD_LIBRARY_PATH=$LD_LIBRARY_PATH ./mlir-jl-tblgen --generator=jl-op-defs $DIALECTS_PATH/Complex/IR/ComplexOps.td -I$INCLUDE_PATH > output/Complex.jl
LD_LIBRARY_PATH=$LD_LIBRARY_PATH ./mlir-jl-tblgen --generator=jl-op-defs $DIALECTS_PATH/Linalg/IR/LinalgOps.td -I$INCLUDE_PATH > output/Linalg_unstructured.jl
LD_LIBRARY_PATH=$LD_LIBRARY_PATH ./mlir-jl-tblgen --generator=jl-op-defs $DIALECTS_PATH/Linalg/IR/LinalgStructuredOps.td -I$INCLUDE_PATH > output/Linalg_structured.jl
LD_LIBRARY_PATH=$LD_LIBRARY_PATH ./mlir-jl-tblgen --generator=jl-op-defs $DIALECTS_PATH/LLVMIR/LLVMOps.td -I$INCLUDE_PATH > output/LLVM.jl
LD_LIBRARY_PATH=$LD_LIBRARY_PATH ./mlir-jl-tblgen --generator=jl-op-defs $DIALECTS_PATH/AMDGPU/IR/AMDGPU.td -I$INCLUDE_PATH > output/AMDGPU.jl
LD_LIBRARY_PATH=$LD_LIBRARY_PATH ./mlir-jl-tblgen --generator=jl-op-defs $DIALECTS_PATH/Affine/IR/AffineOps.td -I$INCLUDE_PATH > output/Affine.jl
LD_LIBRARY_PATH=$LD_LIBRARY_PATH ./mlir-jl-tblgen --generator=jl-op-defs $DIALECTS_PATH/ControlFlow/IR/ControlFlowOps.td -I$INCLUDE_PATH > output/ControlFlow.jl
LD_LIBRARY_PATH=$LD_LIBRARY_PATH ./mlir-jl-tblgen --generator=jl-op-defs $DIALECTS_PATH/Func/IR/FuncOps.td -I$INCLUDE_PATH > output/Func.jl
LD_LIBRARY_PATH=$LD_LIBRARY_PATH ./mlir-jl-tblgen --generator=jl-op-defs $DIALECTS_PATH/Quant/QuantOps.td -I$INCLUDE_PATH > output/Quant.jl
# LD_LIBRARY_PATH=$LD_LIBRARY_PATH ./mlir-jl-tblgen --generator=jl-op-defs $DIALECTS_PATH/IRDL/IR/IRDLOps.td -I$INCLUDE_PATH > output/IRDL.jl
LD_LIBRARY_PATH=$LD_LIBRARY_PATH ./mlir-jl-tblgen --generator=jl-op-defs $DIALECTS_PATH/PDL/IR/PDLOps.td -I$INCLUDE_PATH > output/PDL.jl
LD_LIBRARY_PATH=$LD_LIBRARY_PATH ./mlir-jl-tblgen --generator=jl-op-defs /home/jumerckx/masterthesis/llvm-project/mlir/include/mlir/IR/BuiltinOps.td -I$INCLUDE_PATH > output/Builtin.jl
LD_LIBRARY_PATH=$LD_LIBRARY_PATH ./mlir-jl-tblgen --generator=jl-op-defs /home/jumerckx/masterthesis/llvm-project/mlir/include/mlir/IR/BuiltinTypes.td -I$INCLUDE_PATH > output/BuiltinTypes.jl
LD_LIBRARY_PATH=$LD_LIBRARY_PATH ./mlir-jl-tblgen --generator=jl-op-defs /home/jumerckx/masterthesis/llvm-project/mlir/include/mlir/IR/OpBase.td -I$INCLUDE_PATH > output/Base.jl
