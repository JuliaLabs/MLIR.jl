module MLIR

using Preferences

const MLIR_VERSION = ScopedValue(VersionNumber(@load_preference("MLIR_VERSION", Base.libllvm_version_string)))

include("API.jl")
include("IR/IR.jl")
include("Dialects.jl")


end # module MLIR
