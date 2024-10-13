module API

using ..MLIR: MLIR_VERSION, MLIRException

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
    local ops = mapreduce(∪, [v14, v15, v16, v17, v18, v19]) do mod
        filter(names(mod; all=true)) do name
            name ∉ [nameof(mod), :eval, :include] && !startswith(string(name), '#')
        end
    end

    for op in ops
        container_mods = filter([v14, v15, v16, v17, v18, v19]) do mod
            op in names(mod; all=true)
        end
        container_mods = map(container_mods) do mod
            mod, VersionNumber(string(nameof(mod)))
        end

        @eval function $op(args...; kwargs...)
            version = MLIR_VERSION[]
            if !($MLIR_VERSION_MIN <= version <= $MLIR_VERSION_MAX)
                error("Unsupported MLIR version $version")
            end

            $(
                map(container_mods) do (mod, version)
                    :(
                        if @v_str($(version.major)) <=
                            version <
                            @v_str($(version.major + 1))
                            return $mod.$op(args...; kwargs...)
                        end
                    )
                end...
            )

            throw(
                MLIRException(
                    string(
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

end # module API
