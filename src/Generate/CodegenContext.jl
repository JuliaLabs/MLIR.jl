abstract type AbstractCodegenContext end

for method in (
    :generate_return,
    :generate_goto,
    :generate_gotoifnot,
    :generate_function,
    :name,
    :generate_invoke
)
    @eval begin
        $method(cg::T, args...; kwargs...) where {T<:AbstractCodegenContext} = error("$method not implemented for type \$T")
    end
end

mutable struct CodegenContext{T} <: AbstractCodegenContext
    CodegenContext{T}() where T = new{T}()

    function (cg::CodegenContext)(f, types)
        mod = IR.Module()
        methods = collect_methods(f, types)
        for (mi, (ir, ret)) in methods
            mlir_func = generate!(cg, ir, ret; mi)
            push!(IR.body(mod), mlir_func)
        end
        return mod
    end
end

abstract type Default end
CodegenContext() = CodegenContext{Default}()

generate_return(cg::CodegenContext, values; location) = Dialects.func.return_(values; location)
generate_goto(cg::CodegenContext, args, dest; location) = Dialects.cf.br(args; dest, location)
generate_gotoifnot(cg::CodegenContext, cond;
    true_args, false_args, true_dest, false_dest, location
    ) = Dialects.cf.cond_br(
        cond, true_args, false_args;
        trueDest=true_dest, falseDest=false_dest, location
        )
generate_function(cg::CodegenContext{T}, types::Pair, reg::IR.Region; kwargs...) where {T} = generate_function(cg, types.first, types.second, reg; kwargs...)
function generate_function(cg::CodegenContext, argtypes, rettypes, reg; name="f")
    function_type = IR.FunctionType(argtypes, rettypes)
    op = Dialects.func.func_(;
        sym_name=string(name),
        sym_visibility="private",
        function_type,
        body=reg,
    )
    return op
end
function name(cg::CodegenContext, mi::MethodInstance)
    return string(mi.specTypes)
end
function generate_invoke(cg::CodegenContext, fname::String, ret, args)
    # return first(args)
    op = Dialects.func.call(args; result_0=IR.Type[IR.Type(ret)], callee=IR.FlatSymbolRefAttribute(fname))
    return ret(IR.result(op))
end