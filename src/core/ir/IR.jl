module IR

using ..MLIR
using ..MLIR: StringRef

include("context.jl")
include("type.jl")
include("value.jl")
include("attribute.jl")
include("location.jl")
include("operation.jl")
include("block.jl")
include("region.jl")
include("dialect.jl")
include("module.jl")
include("builders.jl")

end # module
