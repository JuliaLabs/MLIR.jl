module IR

using ..MLIR: MLIR_VERSION, MLIRException
using ..API

# do not export `Type`, as it is already defined in Core
# also, use `Core.Type` inside this module to avoid clash with MLIR `Type`
export Attribute, Block, Context, Dialect, Location, Operation, Region, Value
export activate!, deactivate!, dispose!, enable_multithreading!, context!
export context, type, type!, location, typeid, block, dialect
export nattrs,
    attr,
    attr!,
    rmattr!,
    regions,
    nregions,
    region,
    nresults,
    result,
    noperands,
    operand,
    operand!,
    nsuccessors,
    successor
export @affinemap

function mlirIsNull(val)
    return val.ptr == C_NULL
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
include("OpOperand.jl") # introduced in LLVM 16
include("Identifier.jl")
include("SymbolTable.jl")
include("AffineExpr.jl")
include("AffineMap.jl")
include("Attribute.jl")
include("IntegerSet.jl")

include("ExecutionEngine.jl")
include("Pass.jl")

### Utils

function visit(f, op)
    for region in regions(op)
        for block in region
            for op in block
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
verifyall(module_::IR.Module) = verifyall(Operation(module_))

const current = (; var"module"=ScopedValue{Union{Nothing,Core.Module}}(nothing),)
# const current_module = ScopedValue{Union{Nothing,Core.Module}}(nothing)

function var"@module"(__source__::LineNumberNode, __module__::Core.Module, block)
    location = :(Location(string($(__source__.file)), $(__source__.line), 0))
    quote
        with(current.module => Module($location)) do
            block
        end
    end
end

function var"@op"(__source__::LineNumberNode, __module__::Core.Module, opcall)
    @assert opcall.head === :call

    name = opcall.args[1]
    location = :(Location(string($(__source__.file)), $(__source__.line), 0))
    operands = opcall.args[2:end]

    quote
        operands = results.($tuple($(esc.(operands)...)))
        create_operation($name, $location; operands)
    end
end

function var"@op"(__source__::LineNumberNode, __module__::Core.Module, opcall, attributes)
    @assert opcall.head === :call
    @assert attributes.head === :tuple

    name = opcall.args[1]
    location = :(Location(string($(__source__.file)), $(__source__.line), 0))
    operands = opcall.args[2:end]

    quote
        operands = results.($tuple($(esc.(operands)...)))
        create_operation($name, $location; operands)
    end
end

end # module IR
