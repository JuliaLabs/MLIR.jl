include("CodeInfoTools.jl")

module Generate

using ..IR
import ..Dialects

include("intrinsic.jl")
include("absint.jl")
include("transform.jl")
include("CodegenContext.jl")

end