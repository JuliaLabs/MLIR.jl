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
    local dialectops = mapreduce(mergewith!(∪), [v14, v15, v16, v17, v18, v19]) do mod
        dialects = filter(names(mod; all=true)) do dialect
            dialect ∉ [nameof(mod), :eval, :include] && !startswith(string(dialect), '#')
        end

        Dict(
            dialect => filter(names(getproperty(mod, dialect); all=true)) do name
                name ∉ [dialect, :eval, :include] && !startswith(string(name), '#')
            end for dialect in dialects
        )
    end

    # generate version-less dialect modules
    for (dialect, ops) in dialectops
        mod = @eval module $dialect
            using ...MLIR: MLIR_VERSION, MLIRException
            using ..Dialects: v14, v15, v16, v17, v18, v19
        end

        for op in ops
            container_mods = filter([v14, v15, v16, v17, v18, v19]) do mod
                dialect in names(mod; all=true) &&
                    op in names(getproperty(mod, dialect); all=true)
            end
            container_mods = map(container_mods) do mod
                mod, VersionNumber(string(nameof(mod)))
            end

            @eval mod function $op(args...; kwargs...)
                version = MLIR_VERSION[]
                if !(MLIR_VERSION_MIN <= version <= MLIR_VERSION_MAX)
                    error("Unsupported MLIR version $version")
                end

                $(
                    map(container_mods) do (mod, version)
                        :(
                            if @v_str($(version.major)) <=
                                version <
                                @v_str($(version.major + 1))
                                return $mod.$dialect.$op(args...; kwargs...)
                            end
                        )
                    end...
                )

                throw(
                    MLIRException(
                        string(
                            $dialect,
                            ".",
                            $op,
                            " is not implemented for MLIR $(version.major). You can find it in MLIR ",
                            $(join(
                                map(container_mods) do (_, version)
                                    "$(version.major)"
                                end,
                                ", ",
                                ", and ",
                            )),
                            ".",
                        ),
                    ),
                )
            end
        end
    end
end

end # module Dialects
