module IR

using ..MLIR

import Base: ==, insert!, push!, getindex

include("utils.jl")

include("context.jl")
include("type.jl")
include("value.jl")
include("attribute.jl")
include("location.jl")
include("operation.jl")
include("identifier.jl")
include("block.jl")
include("region.jl")
include("operation_state.jl")
include("dialect.jl")
include("module.jl")
include("pass_manager.jl")

end # module
