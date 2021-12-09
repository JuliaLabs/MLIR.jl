# generate LLVM wrappers

using LLVM_full_jll
import MLIR_jll

using Clang.Generators
import Clang

options = load_options(joinpath(@__DIR__, "wrap.toml"))

prefix = joinpath("..", "lib")
output_path = joinpath(prefix, string(Base.libllvm_version.major), "api.jl")
options["general"]["output_file_path"] = output_path
mkpath(dirname(output_path))

if v"12" <= Base.libllvm_version < "v13"
    options["general"]["library_name"] = "libMLIRPublicAPI"
else
    options["general"]["library_name"] = "libMLIR"
end

includedir = LLVM_full_jll.llvm_config() do config
    readchomp(`$config --includedir`)
end
cppflags = LLVM_full_jll.llvm_config() do config
    split(readchomp(`$config --cppflags`))
end

args = get_default_args("x86_64-linux-gnu")
push!(args, "-Isystem$includedir")
append!(args, cppflags)

headers = ["Pass.h", joinpath("Dialect", "Standard.h")]

header_files = map(h->joinpath(MLIR_jll.artifact_dir, "include", "mlir-c", h), headers)

ctx = create_context(header_files, args, options)

build!(ctx, BUILDSTAGE_NO_PRINTING)

# # custom rewriter
# function rewrite!(dag::ExprDAG)
#     replace!(get_nodes(dag)) do node
#         filename = normpath(Clang.get_filename(node.cursor))
#         if !contains(filename, "LLVMExtra")
#             return ExprNode(node.id, Generators.Skip(), node.cursor, Expr[], node.adj)
#         end
#         return node
#     end
# end

# rewrite!(ctx.dag)

# print
build!(ctx, BUILDSTAGE_PRINTING_ONLY)

