module Dialects

include("Utils.jl")

# generate versioned modules
for version in Base.Filesystem.readdir(joinpath(@__DIR__))
    isdir(joinpath(@__DIR__, version)) || continue
    includes = map(readdir(joinpath(@__DIR__, version); join=true)) do path
        :(include($path))
    end

    @eval module $(Symbol(:v, version))
    import ..Dialects
    $(includes...)
    end
end

begin
    # list dialect operations
    local dialectops = mapreduce(mergewith!(∪), [v14, v15, v16]) do mod
        dialects = filter(names(mod; all=true)) do dialect
            dialect ∉ [nameof(mod), :eval, :include] && !startswith(string(dialect), '#')
        end

        Dict(dialect => filter(names(getproperty(mod, dialect); all=true)) do name
            name ∉ [dialect, :eval, :include] && !startswith(string(name), '#')
        end for dialect in dialects)
    end

    # generate version-less dialect modules
    for (dialect, ops) in dialectops
        mod = @eval module $dialect
        using ...MLIR: MLIR_VERSION
        using ..Dialects: v14, v15, v16
        end

        for op in ops
            @eval mod function $op(args...; kwargs...)
                if v"14" <= MLIR_VERSION[] < v"15"
                    v14.$dialect.$op(args...; kwargs...)
                elseif v"15" <= MLIR_VERSION[] < v"16"
                    v15.$dialect.$op(args...; kwargs...)
                elseif v"16" <= MLIR_VERSION[] < v"17"
                    v16.$dialect.$op(args...; kwargs...)
                else
                    error("Unsupported MLIR version $version")
                end
            end
        end
    end
end

end # module Dialects
