LLVM_DIR = llvm
llvmconfig = $(LLVM_DIR)/tools/llvm-config

CXXFLAGS = $(shell $(llvmconfig) --cxxflags)
LDFLAGS = $(shell $(llvmconfig) --ldflags)

mlir-jl-tblgen: mlir-jl-tblgen.cc jl-generators.cc 
	$(CXX) $^ $(CXXFLAGS) $(LDFLAGS) -I$(LLVM_DIR)/include -L$(LLVM_DIR)/lib -lMLIR -lLLVM -lMLIRTableGen -lLLVMTableGen -o $@ 

# LD_LIBRARY_PATH=llvm/lib ./mlir-jl-tblgen --generator=jl-op-defs llvm/include/mlir/Dialect/StandardOps/IR/Ops.td -Illvm/include