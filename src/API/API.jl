module API

using ..MLIR: VersionDispatcher
using CEnum

# generate versioned API modules
for dir in Base.Filesystem.readdir(joinpath(@__DIR__, "API"))
    @eval module $(Symbol(:v, dir))
    using ...MLIR: MLIR_VERSION, MLIR_C_PATH
    include(joinpath(@__DIR__, "API", $dir, "libMLIR_h.jl"))
    end
end

# gather all type definitions
type_versions = Dict{Symbol,Vector{Module}}()
for mod in [v14, v15, v16]
    version = VersionNumber(string(nameof(mod)))

    defined_types = filter(names(mod; all=true)) do name
        ref = getproperty(mod, name)
        ref isa DataType && !(ref <: Function)
    end

    mergewith!(type_versions, Dict(type => [mod] for type in defined_types)) do a, b
        append!(a, b)
    end
end

# generate type aliases
for (type, mods) in type_versions
    if length(mods) == 1
        @eval const $type = $(only(mods)).$type
        continue
    end

    # check all type definitions are equal
    pred_names = allequal(Iterators.map(m -> fieldnames(getproperty(m, type)), mods))
    pred_types = allequal(Iterators.map(mods) do mod
        map(fieldtypes(getproperty(mod, type))) do type
            if type isa Ptr
                Symbol(:Ptr, '{', nameof(eltype(type)), '}')
            else
                nameof(type)
            end
        end
    end)
    if !(pred_names && pred_types)
        error("Type $type has different definitions in different versions")
    end

    # TODO maybe we need to add `convert` rules for versioned types?
    # when @ccall-ing, `convert` will be called and different versions might be used
    @eval const $type = $(mods[1]).$type
end

const Dispatcher = VersionDispatcher(@__MODULE__)

end
