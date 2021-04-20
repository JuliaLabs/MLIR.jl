#####
##### Initialization
#####

function __init__()
    @assert(VERSION >= v"1.6.0-DEV.1429")

    # Find the libMLIR public *.so.
    libmlir[] = begin
        path = joinpath(String(Base.libllvm_path()), "..", "libMLIRPublicAPI.so") |> normpath
        if path === nothing
            error("""Cannot find the LLVM library loaded by Julia.
                  Please use a version of Julia that has been built with USE_LLVM_SHLIB=1 (like the official binaries).
                  If you are, please file an issue and attach the output of `Libdl.dllist()`.""")
        end
        Symbol(path)
    end

    # Find the private *.so.
    libmlir_private[] = begin
        path = joinpath(String(Base.libllvm_path()), "..", "libMLIR.so") |> normpath
        if path === nothing
            error("""Cannot find the LLVM library loaded by Julia.
                  Please use a version of Julia that has been built with USE_LLVM_SHLIB=1 (like the official binaries).
                  If you are, please file an issue and attach the output of `Libdl.dllist()`.""")
        end
        Symbol(path)
    end
end
