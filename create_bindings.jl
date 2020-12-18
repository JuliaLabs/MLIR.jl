module BindingsGenerator

using Clang
using Clang_jll

const LIBCLANG_INCLUDE = joinpath(dirname(Clang_jll.libclang_path), "..", "include", "clang-c") |> normpath
const MLIR_C_INCLUDE = joinpath(@__DIR__, "..", "julia/usr/include/mlir-c") |> normpath
const MLIR_C_HEADERS = [joinpath(MLIR_C_INCLUDE, header) for header in readdir(MLIR_C_INCLUDE) if endswith(header, ".h")]

println("Found headers:")
display(MLIR_C_HEADERS)
try
    mkdir("lib")
catch e
    println("/lib already exists. Writing into /lib.")
end

wc = init(; headers = MLIR_C_HEADERS,
          output_file = joinpath(@__DIR__, "lib/libmlir_api.jl"),
          common_file = joinpath(@__DIR__, "lib/libmlir_common.jl"),
          clang_includes = vcat(LIBCLANG_INCLUDE, CLANG_INCLUDE),
          clang_args = ["-I", joinpath(LIBCLANG_INCLUDE, ".."),
                        "-I", joinpath(MLIR_C_INCLUDE, "..")],
          header_wrapped = (root, current) -> root == current,
          header_library = x -> "libmlir",
          clang_diagnostics = false,
         )

run(wc)

end # module
