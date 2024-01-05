using Pkg

# TODO(@mofeing) is this the best way to constraint LLVM version based on Julia version?
LLVM_version = if haskey(ENV, "JULIA_LLVM_VERSION")
    ENV["JULIA_LLVM_VERSION"]
elseif VERSION >= v"1.10"
    "15"
elseif VERSION >= v"1.9"
    "14"
else
    error("Unsupported Julia version: $(VERSION)")
end
Pkg.add(name="LLVM_full_jll", version=LLVM_version)

using LLVM_full_jll

println("- LLVM_full_jll v$(pkgversion(LLVM_full_jll))")
