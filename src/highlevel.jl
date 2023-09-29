module Builder

export @Block, @Region

using ...IR

ctx = IR.Context()
loc = IR.Location()

struct BlockBuilder
    block::IR.Block
    expr::Expr
end
_has_blockbuilder() = haskey(task_local_storage(), :BlockBuilder) &&
                      !isempty(task_local_storage(:BlockBuilder))

function blockbuilder()
    if !_has_blockbuilder()
        error("No BlockBuilder is active")
        return nothing
    end
    last(task_local_storage(:BlockBuilder))
end
function activate(b::BlockBuilder)
    stack = get!(task_local_storage(), :BlockBuilder) do
        BlockBuilder[]
    end
    push!(stack, b)
end
function deactivate(b::BlockBuilder)
    blockbuilder() == b || error("Deactivating wrong RegionBuilder")
    pop!(task_local_storage(:BlockBuilder))
end

struct RegionBuilder
    region::IR.Region
    blockbuilders::Vector{BlockBuilder}
end
_has_regionbuilder() = haskey(task_local_storage(), :RegionBuilder) &&
                       !isempty(task_local_storage(:RegionBuilder))
function regionbuilder()
    if !_has_regionbuilder()
        error("No RegionBuilder is active")
        return nothing
    end
    last(task_local_storage(:RegionBuilder))
end
function activate(r::RegionBuilder)
    stack = get!(task_local_storage(), :RegionBuilder) do
        RegionBuilder[]
    end
    push!(stack, r)
end
function deactivate(r::RegionBuilder)
    regionbuilder() == r || error("Deactivating wrong RegionBuilder")
    pop!(task_local_storage(:RegionBuilder))
end

function Region(expr)
    exprs = Expr[]

    #= Create region =#
    region = IR.Region()
    #= Push region on the stack =#
    regionbuilder = RegionBuilder(region, BlockBuilder[])
    activate(regionbuilder)
    #=
    `expr` calls to @block.
    These calls will create the block variables that
    are referenced in control flow operations.
    Blocks are added to the region at the top of the
    stack and a queue of blocks is kept. The
    expressions to generate the operations in each
    block can't be executed yet since they can't
    reference the blocks before their creation.
    =#
    push!(exprs, expr)
    #=
    Once the blocks are created, the operation
    code can be run. This happens in order. All the
    operations are pushed to the block at the front
    of the queue
    =#
    push!(exprs, quote
        for blockbuilder in $regionbuilder.blockbuilders
            $activate(blockbuilder)
            eval(blockbuilder.expr)
            $deactivate(blockbuilder)
        end
    end)

    push!(exprs, quote
        $deactivate($regionbuilder)
        $region
    end)

    return Expr(:block, exprs...)
end
macro Region(expr)
    quote
        $(esc(Region(expr)))
    end
end

function Block(expr)
    block = IR.Block()
    blockbuilder = BlockBuilder(block, expr)

    if (_has_regionbuilder())
        #= Add block to current region =#
        push!(regionbuilder().region, block)
        #=
        Add blockbuilder to the queue to come back later to
        generate its operations.
        =#
        push!(regionbuilder().blockbuilders, blockbuilder)

        #=
        Only return the block, don't create the
        operations yet.
        =#
        return quote
            $block
        end
    else
        #=
        If there's no regionbuilder, the operations
        defined in `expr` can immediately get executed
        =#
        return quote
            $activate($blockbuilder)
            $expr
            $deactivate($blockbuilder)
            $block
        end
    end
end
macro Block(expr)
    quote
        $(esc(Block(expr)))
    end
end

end # Builder