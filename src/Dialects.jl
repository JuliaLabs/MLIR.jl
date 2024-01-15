module Dialects

import LLVM
import ..IR: Attribute, NamedAttribute, context
import ..API

namedattribute(name, val) = namedattribute(name, Attribute(val))
namedattribute(name, val::Attribute) = NamedAttribute(name, val)
function namedattribute(name, val::NamedAttribute)
    @assert true # TODO(jm): check whether name of attribute is correct, getting the name might need to be added to IR.jl?
    return val
end

operandsegmentsizes(segments) = namedattribute(
    "operand_segment_sizes",
    Attribute(API.mlirDenseI32ArrayGet(
        context().context,
        length(segments),
        Int32.(segments)
    )))

let
    ver = string(LLVM.version().major)
    dir = joinpath(@__DIR__, "Dialects", ver)
    if !isdir(dir)
        error("""The MLIR dialect bindings for v$ver do not exist.
                You might need a newer version of MLIR.jl for this version of Julia.""")
    end

    for path in readdir(joinpath(@__DIR__, "Dialects"); join=true)
        include(path)
    end
end

# module arith

# module Predicates
# const eq = 0
# const ne = 1
# const slt = 2
# const sle = 3
# const sgt = 4
# const sge = 5
# const ult = 6
# const ule = 7
# const ugt = 8
# const uge = 9
# end

# end # module arith

end # module Dialects
