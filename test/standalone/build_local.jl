using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using Pkg, Scratch, Preferences, Libdl, CMake_jll

MLIR = Base.UUID("bfde9dd4-8f40-4a1e-be09-1475335e1c92")

# get scratch directories
scratch_dir = get_scratch!(MLIR, "standalone")
isdir(scratch_dir) && rm(scratch_dir; recursive=true)
source_dir = @__DIR__

# get build directory
build_dir = if isempty(ARGS)
    mktempdir()
else
    ARGS[1]
end
mkpath(build_dir)

# download LLVM
Pkg.activate(; temp=true)
llvm_assertions = try
    cglobal((:_ZN4llvm24DisableABIBreakingChecksE, Base.libllvm_path()), Cvoid)
    false
catch
    true
end
llvm_pkg_version = "$(Base.libllvm_version.major).$(Base.libllvm_version.minor)"
LLVM = if llvm_assertions
    Pkg.add(name="LLVM_full_assert_jll", version=llvm_pkg_version)
    using LLVM_full_assert_jll
    LLVM_full_assert_jll
else
    Pkg.add(name="LLVM_full_jll", version=llvm_pkg_version)
    using LLVM_full_jll
    LLVM_full_jll
end
Pkg.add(name="mlir_jl_tblgen_jll")

LLVM_DIR = joinpath(LLVM.artifact_dir, "lib", "cmake", "llvm")
MLIR_DIR = joinpath(LLVM.artifact_dir, "lib", "cmake", "mlir")

# build and install
@info "Building" source_dir scratch_dir build_dir LLVM_DIR MLIR_DIR
cmake() do cmake_path
    config_opts = `-DLLVM_ROOT=$(LLVM_DIR) -DMLIR_ROOT=$(MLIR_DIR) -DCMAKE_INSTALL_PREFIX=$(scratch_dir)`
    if Sys.iswindows()
        # prevent picking up MSVC
        config_opts = `$config_opts -G "MSYS Makefiles"`
    end
    run(`$cmake_path $config_opts -B$(build_dir) -S$(source_dir)`)
    run(`$cmake_path --build $(build_dir) --target install`)
end

# # discover built binaries
# built_libs = filter(readdir(joinpath(scratch_dir, "lib"))) do file
#     endswith(file, "standalone.$(Libdl.dlext)")
# end

# libstandalone_path = joinpath(scratch_dir, "lib", only(built_libs))
# isfile(libstandalone_path) || error("Could not find library $libstandalone_path in build directory")

built_libs = filter(readdir(joinpath(scratch_dir, "lib"))) do file
    endswith(file, "standalone-c.$(Libdl.dlext)")
end

libstandalone_c_path = joinpath(scratch_dir, "lib", only(built_libs))
isfile(libstandalone_c_path) || error("Could not find library $libstandalone_c_path in build directory")

# tell Standalone.jl to load our library instead of the default artifact one
set_preferences!(
    joinpath(@__DIR__, "Standalone", "LocalPreferences.toml"),
    "Standalone",
    # "libstandalone" => libstandalone_path,
    "libstandalone_c" => libstandalone_c_path;
    force=true,
)

# include_dir = joinpath(LLVM.artifact_dir, "include")
# output = joinpath(@__DIR__, "Brutus", "src", "Dialects", string(Base.libllvm_version.major), "JuliaOps.jl")
# mkpath(dirname(output))
# using mlir_jl_tblgen_jll
# run(`$(mlir_jl_tblgen()) --generator=jl-op-defs include/brutus/Dialect/Julia/JuliaOps.td -Iinclude/ -I$include_dir -o $output`)
