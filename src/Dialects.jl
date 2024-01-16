module Dialects

import LLVM
import ..IR: Attribute, NamedAttribute, DenseArrayAttribute, context
import ..API

namedattribute(name, val) = namedattribute(name, Attribute(val))
namedattribute(name, val::Attribute) = NamedAttribute(name, val)
function namedattribute(name, val::NamedAttribute)
    @assert true # TODO(jm): check whether name of attribute is correct, getting the name might need to be added to IR.jl?
    return val
end

operandsegmentsizes(segments) =
    namedattribute("operand_segment_sizes",
        LLVM.version() <= v"15" ?
            Attribute(Int32.(segments)) :
            DenseArrayAttribute(Int32.(segments))
    )


let
    ver = string(LLVM.version().major)
    dir = joinpath(@__DIR__, "Dialects", ver)
    if !isdir(dir)
        error("""The MLIR dialect bindings for v$ver do not exist.
                You might need a newer version of MLIR.jl for this version of Julia.""")
    end

    for path in readdir(dir; join=true)
        include(path)
    end
end

end # module Dialects
