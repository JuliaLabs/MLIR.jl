module BindingsGenerator

using Clang.Generators

const MLIR_C_INCLUDE = joinpath(@__DIR__, "..", "julia/usr/include/mlir-c") |> normpath
const MLIR_C_HEADERS = [joinpath(MLIR_C_INCLUDE, header) for header in readdir(MLIR_C_INCLUDE) if endswith(header, ".h")]

println("Found headers:")
display(MLIR_C_HEADERS)
try
    mkdir("lib")
catch e
    println("/lib already exists. Writing into /lib.")
end

cd(@__DIR__)

args = ["-I", joinpath(MLIR_C_INCLUDE, "..")]

options = load_options(joinpath(@__DIR__, "bindings.toml"))

ctx = create_context(headers, args, options)

build!(ctx)

end # module
