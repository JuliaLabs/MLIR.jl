module AffineUtils

using ..MLIR.API
using ..MLIR.IR


walk(f, other) = f(other)
function walk(f, expr::Expr)
    expr = f(expr)
    Expr(expr.head, map(arg -> walk(f, arg), expr.args)...)
end

"""
    @map (d1, d2, d3, ...)[s1, s2, ...] -> (d0 + d1, ...)

Returns an affine map from the provided Julia expression.
On the right hand side are allowed the following function calls:

 - +, *, ÷, %, fld, cld

The rhs can only contains dimensions and symbols present on the left hand side or integer literals.

```juliadoctest
julia> using MLIR: IR, AffineUtils

julia> IR.context!(IR.Context()) do
           AffineUtils.@map (d1, d2)[s0] -> (d1 + s0, d2 % 10)
       end
MLIR.IR.AffineMap(#= (d0, d1)[s0] -> (d0 + s0, d1 mod 10) =#)
```
"""
macro map(ex)
    @assert Meta.isexpr(ex, :(->), 2) "invalid affine expression $ex"

    lhs, rhs = ex.args
    rhs = Meta.isexpr(rhs,:block ) ? rhs.args[end] : rhs
    if Meta.isexpr(lhs, :ref)
        lhs, symbols... = lhs.args
    else
        symbols = []
    end
    @assert Meta.isexpr(lhs, :tuple) "invalid expression lhs $(lhs) (expected tuple)"
    @assert Meta.isexpr(rhs, :tuple) "invalid expression rhs $(rhs) (expected tuple)"

    dimensions = lhs.args
    values = Dict{Symbol,Expr}()

    for (i, s) in enumerate(symbols)
        @assert s isa Symbol "invalid symbol $s in expression"
        values[s] = Expr(:call, API.mlirAffineSymbolExprGet, :context, i -1)
    end
    for (i, s) in enumerate(dimensions)
        @assert s isa Symbol "invalid dimension $s in expression"
        values[s] = Expr(:call, API.mlirAffineDimExprGet, :context, i -1)
    end

    calls_to_replace = Dict{Symbol,Function}(
        :+ => API.mlirAffineAddExprGet,
        :* => API.mlirAffineMulExprGet,
        :÷ => API.mlirAffineFloorDivExprGet, # <- not sure about this one since it is round to zero in julia
        :fld => API.mlirAffineFloorDivExprGet,
        :cld => API.mlirAffineCeilDivExprGet,
        :(%) => API.mlirAffineModExprGet,
    )

    # TODO: it would be useful to embed integer constants with $(myint).
    affine_exprs = Expr(:vect, map(rhs.args) do ex
        walk(ex) do v
            v isa Integer ?
                Expr(:call, API.mlirAffineConstantExprGet, :context, Int64(v)) :
            Meta.isexpr(v, :call) ?
                Expr(:call, get(calls_to_replace, v.args[1], v.args[1]), v.args[2:end]...) :
            haskey(values, v) ? values[v] :
            v isa Symbol ? error("unknown item $v") : v
        end
    end...)

    dimcount = length(dimensions)
    symcount = length(symbols)
    naffine_exprs = length(affine_exprs.args)

    quote
        local context = IR.context()
        map = API.mlirAffineMapGet(
            context, $dimcount,
            $symcount, $naffine_exprs, $(affine_exprs)::Vector{API.MlirAffineExpr})
        IR.AffineMap(map)
    end
end

end # module AffineUtils
