module Dialects

using ..MLIR: VersionDispatcher
import ..IR: IR, Attribute, NamedAttribute, context
import ..API

namedattribute(name, val) = namedattribute(name, Attribute(val))
namedattribute(name, val::Attribute) = NamedAttribute(name, val)
function namedattribute(name, val::NamedAttribute)
    @assert true # TODO(jm): check whether name of attribute is correct, getting the name might need to be added to IR.jl?
    return val
end

operandsegmentsizes(segments) = namedattribute("operand_segment_sizes", Attribute(Int32.(segments)))

# generate versioned API modules
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

const Dispatcher = VersionDispatcher(@__MODULE__)

end # module Dialects
