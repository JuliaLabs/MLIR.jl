module API

include("Types.jl")
using .Types

# generate versioned API modules
for dir in Base.Filesystem.readdir(joinpath(@__DIR__))
    isdir(joinpath(@__DIR__, dir)) || continue
    @eval module $(Symbol(:v, dir))
    using ...MLIR: MLIR_VERSION, MLIR_C_PATH
    using ...API.Types
    using CEnum
    include(joinpath(@__DIR__, $dir, "libMLIR_h.jl"))
    end
end

# generate version-less API functions
begin
    local ops = mapreduce(∪, [v14, v15, v16]) do mod
        filter(names(mod; all=true)) do name
            name ∉ [nameof(mod), :eval, :include] && !startswith(string(name), '#')
        end
    end

    for op in ops
        @eval function $op(args...; kwargs...)
            if v"14" <= MLIR_VERSION[] < v"15"
                v14.$op(args...; kwargs...)
            elseif v"15" <= MLIR_VERSION[] < v"16"
                v15.$op(args...; kwargs...)
            elseif v"16" <= MLIR_VERSION[] < v"17"
                v16.$op(args...; kwargs...)
            else
                error("Unsupported MLIR version $version")
            end
        end
    end
end

end
