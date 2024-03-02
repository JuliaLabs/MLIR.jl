module IR

using ..API
using LLVM: LLVM

# do not export `Type`, as it is already defined in Core
# also, use `Core.Type` inside this module to avoid clash with MLIR `Type`
export Attribute, Block, Context, Dialect, Location, Operation, Region, Value
export activate!, deactivate!, dispose!, enable_multithreading!, context!
export context, type, type!, location, typeid, block, dialect
export nattrs, attr, attr!, rmattr!, nregions, region, nresults, result, noperands, operand, operand!, nsuccessors, successor
export BlockIterator, RegionIterator, OperationIterator
export @affinemap

function mlirIsNull(val)
    val.ptr == C_NULL
end

function print_callback(str::API.MlirStringRef, userdata)
    data = unsafe_wrap(Array, Base.convert(Ptr{Cchar}, str.data), str.length; own=false)
    write(userdata isa Base.RefValue ? userdata[] : userdata, data)
    return Cvoid()
end

include("LogicalResult.jl")
include("Context.jl")
include("Dialect.jl")
include("Location.jl")
include("Type.jl")
include("TypeID.jl")
include("Operation.jl")
include("Module.jl")
include("Block.jl")
include("Region.jl")
include("Value.jl")

if LLVM.version() >= v"16"
    include("OpOperand.jl")
end

include("Identifier.jl")
include("SymbolTable.jl")
include("AffineExpr.jl")
include("AffineMap.jl")
include("Attribute.jl")
include("IntegerSet.jl")
include("Iterators.jl")

include("Pass.jl")

# MlirStringRef is a non-owning reference to a string,
# we thus need to ensure that the Julia string remains alive
# over the use. For that we use the cconvert/unsafe_convert mechanism
# for foreign-calls. The returned value of the cconvert is rooted across
# foreign-call.
Base.cconvert(::Core.Type{API.MlirStringRef}, s::Union{Symbol,String}) = s
Base.cconvert(::Core.Type{API.MlirStringRef}, s::AbstractString) = Base.cconvert(API.MlirStringRef, String(s)::String)

# Directly create `MlirStringRef` instead of adding an extra ccall.
function Base.unsafe_convert(::Core.Type{API.MlirStringRef}, s::Union{Symbol,String,AbstractVector{UInt8}})
    p = Base.unsafe_convert(Ptr{Cchar}, s)
    return API.MlirStringRef(p, sizeof(s))
end

function Base.String(str::API.MlirStringRef)
    Base.unsafe_string(pointer(str.data), str.length)
end

### Utils

function visit(f, op)
    for region in RegionIterator(op)
        for block in BlockIterator(region)
            for op in OperationIterator(block)
                f(op)
            end
        end
    end
end

"""
    verifyall(operation; debug=false)

Prints the operations which could not be verified.
"""
function verifyall(operation::Operation; debug=false)
    io = IOContext(stdout, :debug => debug)
    visit(operation) do op
        if !verify(op)
            show(io, op)
        end
    end
end
verifyall(module_::IR.Module) = Operation(module_) |> verifyall

end # module IR
