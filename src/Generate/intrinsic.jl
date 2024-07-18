import MacroTools: splitdef

is_intrinsic(::Any) = false

macro is_intrinsic(sig)
    return esc(:($(@__MODULE__).is_intrinsic(::Type{<:$sig}) = true))
end

function intrinsic_(expr)
    dict = splitdef(expr)
    # TODO: this can probably be fixed:
    length(dict[:kwargs]) == 0 || error("Intrinsic functions can't have keyword arguments\nDefine a regular function with kwargs that calls the intrinsic instead.")
    argtypes = map(dict[:args]) do arg
        if arg isa Symbol
            return :Any
        elseif arg.head == :(::)
            return arg.args[end]
        else
            error("Don't know how to handle argument type $arg")
        end
    end

    return quote
        $(reset_cache!)()
        $(esc(expr))
        $(@__MODULE__).is_intrinsic(::Type{<:Tuple{$(_typeof)($(esc(dict[:name]))), $(esc.(argtypes)...)}}) where {$(esc.(dict[:whereparams])...)} = true
    end
end

macro intrinsic(f)
    f = macroexpand(__module__, f)
    Base.is_function_def(f) || error("@intrinsic requires a function definition")
    intrinsic_(f)
end

