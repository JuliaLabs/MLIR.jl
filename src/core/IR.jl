module IR

using ..MLIR

import Base: ==, insert!, push!, getindex

include("utils.jl")

include("ir/context.jl")
include("ir/type.jl")
include("ir/value.jl")
include("ir/attribute.jl")
include("ir/location.jl")
include("ir/operation.jl")
include("ir/identifier.jl")
include("ir/block.jl")
include("ir/region.jl")
include("ir/dialect.jl")
include("ir/module.jl")
include("ir/builders.jl")

include("pass_manager.jl")

end # module
