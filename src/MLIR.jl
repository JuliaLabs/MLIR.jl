module MLIR

using Preferences

libmlir_version = VersionNumber(@load_preference(:libmlir_version, Base.libllvm_version_string))

module API
using CEnum

# MLIR C API
using MLIR_jll
let
    ver = string(libmlir_version.major)
    dir = joinpath(@__DIR__, "API", ver)
    if !isdir(dir)
        error("""The MLIR API bindings for v$ver do not exist.
                You might need a newer version of MLIR.jl for this version of Julia.""")
    end

    include(joinpath(dir, "libMLIR_h.jl"))
end
end # module API

include("IR/IR.jl")

include("Dialects.jl")


end # module MLIR
