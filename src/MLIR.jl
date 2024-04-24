module MLIR

using Preferences

libmlir_version = VersionNumber(@load_preference("libmlir_version", Base.libllvm_version_string))

include("API.jl")
include("IR/IR.jl")
include("Dialects.jl")


end # module MLIR
