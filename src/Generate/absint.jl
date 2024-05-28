using Core: MethodInstance, CodeInstance, OpaqueClosure
const CC = Core.Compiler
using ..CodeInfoTools

## custom interpreter

struct MLIRInterpreter <: CC.AbstractInterpreter
    cache_token::Symbol
    world::UInt
    inf_params::CC.InferenceParams
    opt_params::CC.OptimizationParams
    inf_cache::Vector{CC.InferenceResult}

    active::Bool # when set to false, inlining_policy is not changed.

    function _MLIRInterpreter()
        return new(
            cache_token,
            Base.get_world_counter(),
            CC.InferenceParams(),
            CC.OptimizationParams(),
            CC.InferenceResult[],
            true
        )
    end

    function MLIRInterpreter(interp::MLIRInterpreter=_MLIRInterpreter();
                                cache_token::Symbol = interp.cache_token,
                                world::UInt = interp.world,
                                inf_params::CC.InferenceParams = interp.inf_params,
                                opt_params::CC.OptimizationParams = interp.opt_params,
                                inf_cache::Vector{CC.InferenceResult} = interp.inf_cache,
                                active::Bool = interp.active)
        @assert world <= Base.get_world_counter()

        return new(cache_token, world,
                inf_params, opt_params,
                inf_cache, active)
    end
end




cache_token = gensym(:MLIRInterpreterCache)
function reset_cache!()
    @debug "Resetting cache"
    
    global cache_token
    cache_token = gensym(:MLIRInterpreterCache)
end

CC.InferenceParams(interp::MLIRInterpreter) = interp.inf_params
CC.OptimizationParams(interp::MLIRInterpreter) = interp.opt_params
CC.get_inference_cache(interp::MLIRInterpreter) = interp.inf_cache
CC.get_inference_world(interp::MLIRInterpreter) = interp.world
CC.cache_owner(interp::MLIRInterpreter) = interp.cache_token

# # No need to do any locking since we're not putting our results into the runtime cache
CC.lock_mi_inference(interp::MLIRInterpreter, mi::MethodInstance) = nothing
CC.unlock_mi_inference(interp::MLIRInterpreter, mi::MethodInstance) = nothing

function CC.add_remark!(interp::MLIRInterpreter, sv::CC.InferenceState, msg)
    @debug "Inference remark during compilation of MethodInstance of $(sv.linfo): $msg"
end

CC.may_optimize(interp::MLIRInterpreter) = true
CC.may_compress(interp::MLIRInterpreter) = true
CC.may_discard_trees(interp::MLIRInterpreter) = true
CC.verbose_stmt_info(interp::MLIRInterpreter) = false

struct MLIRIntrinsicCallInfo <: CC.CallInfo
    info::CC.CallInfo
    MLIRIntrinsicCallInfo(@nospecialize(info::CC.CallInfo)) = new(info)
end
CC.nsplit_impl(info::MLIRIntrinsicCallInfo) = CC.nsplit(info.info)
CC.getsplit_impl(info::MLIRIntrinsicCallInfo, idx::Int) = CC.getsplit(info.info, idx)
CC.getresult_impl(info::MLIRIntrinsicCallInfo, idx::Int) = CC.getresult(info.info, idx)


function CC.abstract_call_gf_by_type(interp::MLIRInterpreter, @nospecialize(f), arginfo::CC.ArgInfo, si::CC.StmtInfo, @nospecialize(atype),
    sv::CC.AbsIntState, max_methods::Int)

    argtype_tuple = Tuple{map(_type, arginfo.argtypes)...}
    if interp.active && is_intrinsic(argtype_tuple)
        interp′ = MLIRInterpreter(interp; active=false)
    else
        interp′ = interp
    end

    cm = @invoke CC.abstract_call_gf_by_type(interp′::CC.AbstractInterpreter, f::Any,
        arginfo::CC.ArgInfo, si::CC.StmtInfo, atype::Any, sv::CC.AbsIntState, max_methods::Int)
            
    if interp.active && is_intrinsic(argtype_tuple)
        return CC.CallMeta(cm.rt, cm.exct, cm.effects, MLIRIntrinsicCallInfo(cm.info))
    else
        return cm
    end
end


"""
    _typeof(x)

Central definition of typeof, which is specific to the use-required in this package.
"""
_typeof(x) = Base._stable_typeof(x)
_typeof(x::Tuple) = Tuple{map(_typeof, x)...}
_typeof(x::NamedTuple{names}) where {names} = NamedTuple{names, _typeof(Tuple(x))}

_type(x) = x
_type(x::CC.Const) = _typeof(x.val)
_type(x::CC.PartialStruct) = _type(x.typ)
_type(x::CC.Conditional) = Union{_type(x.thentype), _type(x.elsetype)}

function CC.inlining_policy(interp::MLIRInterpreter,
    @nospecialize(src), @nospecialize(info::CC.CallInfo), stmt_flag::UInt32)
    if (!interp.active)
        return @invoke CC.inlining_policy(
            interp::CC.AbstractInterpreter,
            src::Any,
            info::CC.CallInfo,
            stmt_flag::UInt32,
        )
    elseif isa(info, MLIRIntrinsicCallInfo)
        return nothing
    else
        return src
    end
end


## utils

# create a MethodError from a function type
# TODO: fix upstream
function unsafe_function_from_type(ft::Type)
    if isdefined(ft, :instance)
        ft.instance
    else
        # HACK: dealing with a closure or something... let's do somthing really invalid,
        #       which works because MethodError doesn't actually use the function
        Ref{ft}()[]
    end
end
function MethodError(ft::Type{<:Function}, tt::Type, world::Integer=typemax(UInt))
    Base.MethodError(unsafe_function_from_type(ft), tt, world)
end
MethodError(ft, tt, world=typemax(UInt)) = Base.MethodError(ft, tt, world)

import Core.Compiler: retrieve_code_info, maybe_validate_code, InferenceState, InferenceResult
# Replace usage sites of `retrieve_code_info`, OptimizationState is one such, but in all interesting use-cases
# it is derived from an InferenceState. There is a third one in `typeinf_ext` in case the module forbids inference.
function InferenceState(result::InferenceResult, cache_mode::UInt8, interp::MLIRInterpreter)
    src = retrieve_code_info(result.linfo, interp.world)
    src === nothing && return nothing
    maybe_validate_code(result.linfo, src, "lowered")
    if (interp.active)
        src = transform(interp, result.linfo, src)
        maybe_validate_code(result.linfo, src, "transformed")
    end
    return InferenceState(result, src, cache_mode, interp)
end

"""
Datastructure to keep track of how value numbers need to be updated after insertions.
"""
struct DestinationOffsets
    indices::Vector{Int}
    DestinationOffsets() = new([])
end
function Base.insert!(d::DestinationOffsets, insertion::Int)
    candidateindex = d[insertion]+1
    if (length(d.indices) == 0)
        push!(d.indices, insertion)
    elseif candidateindex == length(d.indices)+1
        push!(d.indices, insertion)
    elseif (candidateindex == 1) || (d.indices[candidateindex-1] != insertion)
        insert!(d.indices, candidateindex, insertion)
    end
    return d
end
Base.getindex(d::DestinationOffsets, i::Int) = searchsortedlast(d.indices, i, lt= <=)

function insert_bool_conversions_pass(mi, src)
    offsets = DestinationOffsets()

    b = CodeInfoTools.Builder(src)
    for (v, st) in b
        if st isa Core.GotoIfNot
            arg = st.cond isa Core.SSAValue ? var(st.cond.id + offsets[st.cond.id]) : st.cond
            b[v] = Statement(Expr(:call, GlobalRef(@__MODULE__, :mlir_bool_conversion), arg))
            push!(b, Core.GotoIfNot(v, st.dest))
            insert!(offsets, v.id)
        elseif st isa Core.GotoNode
            b[v] = st
        end
    end

    # fix destinations and conditions
    for i in 1:length(b.to)
        st = b.to[i].node
        if st isa Core.GotoNode
            b.to[i] = Core.GotoNode(st.label + offsets[st.label])
        elseif st isa Core.GotoIfNot
            b.to[i] = Statement(Core.GotoIfNot(st.cond, st.dest + offsets[st.dest]))
        end
    end
    finish(b)
end

function transform(interp, mi, src)
    src = insert_bool_conversions_pass(mi, src)
    return src
end

abstract type BoolTrait end
struct NonBoollike <: BoolTrait end
struct Boollike <: BoolTrait end
BoolTrait(T) = NonBoollike()

@inline mlir_bool_conversion(x::Bool) = x
@inline mlir_bool_conversion(x::T) where T = mlir_bool_conversion(BoolTrait(T), x)

@intrinsic mlir_bool_conversion(::Boollike, x) = Base.inferencebarrier(x)::Bool
@intrinsic mlir_bool_conversion(::NonBoollike, x::T) where T = error("Type $T is not marked as Boollike.")
