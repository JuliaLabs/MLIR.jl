module Dialects

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

include("dialects/builtin.jl")

include("dialects/llvm.jl")

include("dialects/arith.jl")

include("dialects/cf.jl")

include("dialects/func.jl")

# include("dialects/Gpu.jl")

# include("dialects/Memref.jl")

# include("dialects/Index.jl")

include("dialects/affine.jl")

# include("dialects/Ub.jl")

# include("dialects/SCF.jl")

end # module Dialects
