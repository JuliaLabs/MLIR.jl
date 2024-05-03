module MLIR

using ScopedValues
import MLIR_jll

const MLIR_VERSION = ScopedValue(Base.libllvm_version)
const MLIR_C_PATH = ScopedValue(MLIR_jll.mlir_c)

struct MLIRException <: Exception
    msg::String
end

Base.showerror(io::IO, err::MLIRException) = print(io, err.msg)

include("API/API.jl")
include("IR/IR.jl")
include("Dialects/Dialects.jl")

end # module MLIR
