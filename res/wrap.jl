module Wrap

# generate MLIR wrappers

using LLVM_full_jll
using Clang
using Clang: find_std_headers

function wrap()
    libmlir_path = normpath(joinpath(String(Base.libllvm_path()), "../libMLIR.so"))
    header_dir = normpath(joinpath(libmlir_path, "../../include/mlir-c"))

    cppflags = LLVM_full_jll.llvm_config() do config
        split(readchomp(`$config --cppflags`))
    end

    # Set-up arguments to clang
    clang_includes  = map(x->x[3:end], filter( x->startswith(x,"-I"), cppflags))
    clang_extraargs = filter(x->!startswith(x,"-I"), cppflags)
    for header in find_std_headers()
        push!(clang_extraargs, "-I"*header)
    end

    # Recursively discover MLIR C API headers (files ending in .h)
    header_dirs = String[header_dir]
    header_files = String[]
    while !isempty(header_dirs)
        parent = pop!(header_dirs)
        children = readdir(parent)
        for child in children
            path = joinpath(parent, child)
            if isdir(path)
                push!(header_dirs, path)
            elseif isfile(path) && endswith(path, ".h")
                push!(header_files, path)
            end
        end
    end
    display(header_files)

    context = init(;
                   headers = header_files,
                   output_file = "libMLIR_h.jl",
                   common_file = "libMLIR_common.jl",
                   clang_includes = convert(Vector{String}, clang_includes),
                   clang_args = convert(Vector{String}, clang_extraargs),
                   header_wrapped = (root, current) -> root == current,
                   header_library = x -> "libmlir",
                  )

    run(context)
end

function main()
    cd(joinpath(dirname(@__DIR__), "lib")) do
        wrap()
    end
end

main()

# Manual clean-up:
# - remove build-host details (LLVM_DEFAULT_TARGET_TRIPLE etc) in libLLVM_common.jl
# - remove "# Skipping ..." comments by Clang.jl
# - replace `const (LLVMOpaque.*) = Cvoid` with `struct $1 end`
# - use `gawk -i inplace '/^[[:blank:]]*$/ { print; next; }; {cur = seen[$0]; if(!seen[$0]++ || (/^end$/ && !prev) || /^.*Clang.*$/) print $0; prev=cur}' libLLVM_h.jl` to remove duplicates
# - use `cat -s` to remove duplicate empty lines

end # module
