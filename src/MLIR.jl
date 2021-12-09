module MLIR

using Libdl
using LLVM
using LLVM: @checked, refcheck

# ------------ C API ------------ #

module API
using MLIR_jll
include(joinpath(@__DIR__, "..", "lib", string(Base.libllvm_version.major), "api.jl"))
end

# ----------- Core ------------ #

# Builds off C API.

include("core/IR.jl")

# ------------ Standard dialects ------------ #

include("dialects/affine.jl")

end # module
