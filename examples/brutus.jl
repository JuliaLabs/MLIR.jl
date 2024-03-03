"""
Brutus is a toy implementation of a Julia typed IR to MLIR conversion, the name
is a reference to the [brutus](https://github.com/JuliaLabs/brutus) project from
the MIT JuliaLabs which performs a similar conversion (with a lot more language constructs supported)
but from C++.
"""
module Brutus

import LLVM
using MLIR.IR
using MLIR.Dialects: arith, func, cf
using Core: PhiNode, GotoNode, GotoIfNot, SSAValue, Argument, ReturnNode, PiNode

const BrutusScalar = Union{Bool,Int64,Int32,Float32,Float64}

module Predicates
const eq = 0
const ne = 1
const slt = 2
const sle = 3
const sgt = 4
const sge = 5
const ult = 6
const ule = 7
const ugt = 8
const uge = 9
end

function cmpi_pred(predicate)
    function (ops...; location=Location())
        arith.cmpi(ops...; result=IR.Type(Bool), predicate, location)
    end
end

function single_op_wrapper(fop)
    (block::Block, args::Vector{Value}; location=Location()) -> push!(block, fop(args...; location))
end

const intrinsics_to_mlir = Dict([
    Base.add_int => single_op_wrapper(arith.addi),
    Base.sle_int => single_op_wrapper(cmpi_pred(Predicates.sle)),
    Base.slt_int => single_op_wrapper(cmpi_pred(Predicates.slt)),
    Base.:(===) => single_op_wrapper(cmpi_pred(Predicates.eq)),
    Base.mul_int => single_op_wrapper(arith.muli),
    Base.mul_float => single_op_wrapper(arith.mulf),
    Base.not_int => function (block, args; location=Location())
        arg = only(args)
        mT = IR.type(arg)
        T = IR.julia_type(mT)
        ones = push!(block, arith.constant(value=typemax(UInt64) % T;
            result=mT, location)) |> IR.result
        push!(block, arith.xori(arg, ones; location))
    end,
])

"Generates a block argument for each phi node present in the block."
function prepare_block(ir, bb)
    b = Block()

    for sidx in bb.stmts
        stmt = ir.stmts[sidx]
        inst = stmt[:inst]
        inst isa Core.PhiNode || continue

        type = stmt[:type]
        IR.push_argument!(b, IR.Type(type), Location())
    end

    return b
end

"Values to populate the Phi Node when jumping from `from` to `to`."
function collect_value_arguments(ir, from, to)
    to = ir.cfg.blocks[to]
    values = []
    for s in to.stmts
        stmt = ir.stmts[s]
        inst = stmt[:inst]
        inst isa Core.PhiNode || continue

        edge = findfirst(==(from), inst.edges)
        if isnothing(edge) # use dummy scalar val instead
            val = zero(stmt[:type])
            push!(values, val)
        else
            push!(values, inst.values[edge])
        end
    end
    values
end

"""
    code_mlir(f, types::Type{Tuple}) -> IR.Operation

Returns a `func.func` operation corresponding to the ircode of the provided method.
This only supports a few Julia Core primitives and scalar types of type $BrutusScalar.

!!! note
    The Julia SSAIR to MLIR conversion implemented is very primitive and only supports a
    handful of primitives. A better to perform this conversion would to create a dialect
    representing Julia IR and progressively lower it to base MLIR dialects.
"""
function code_mlir(f, types)
    ctx = context()
    ir, ret = Core.Compiler.code_ircode(f, types) |> only
    @assert first(ir.argtypes) isa Core.Const

    values = Vector{Value}(undef, length(ir.stmts))

    for dialect in (:func, :cf)
        IR.register_dialect!(IR.DialectHandle(dialect))
    end
    IR.load_all_available_dialects()

    blocks = [
        prepare_block(ir, bb)
        for bb in ir.cfg.blocks
    ]

    current_block = entry_block = blocks[begin]

    for argtype in types.parameters
        IR.push_argument!(entry_block, IR.Type(argtype), Location())
    end

    function get_value(x)::Value
        if x isa Core.SSAValue
            @assert isassigned(values, x.id) "value $x was not assigned"
            values[x.id]
        elseif x isa Core.Argument
            IR.argument(entry_block, x.n - 1)
        elseif x isa BrutusScalar
            IR.result(push!(current_block, arith.constant(; value=x)))
        else
            error("could not use value $x inside MLIR")
        end
    end

    for (block_id, (b, bb)) in enumerate(zip(blocks, ir.cfg.blocks))
        current_block = b
        n_phi_nodes = 0

        for sidx in bb.stmts
            stmt = ir.stmts[sidx]
            inst = stmt[:inst]
            line = ir.linetable[stmt[:line] + 1]

            if Meta.isexpr(inst, :call)
                val_type = stmt[:type]
                if !(val_type <: BrutusScalar)
                    error("type $val_type is not supported")
                end
                out_type = IR.Type(val_type)

                called_func = first(inst.args)
                if called_func isa GlobalRef # TODO: should probably use something else here
                    called_func = getproperty(called_func.mod, called_func.name)
                end

                fop! = intrinsics_to_mlir[called_func]
                args = get_value.(@view inst.args[begin+1:end])

                location = Location(string(line.file), line.line, 0)
                res = IR.result(fop!(current_block, args; location))

                values[sidx] = res
            elseif inst isa PhiNode
                values[sidx] = IR.argument(current_block, n_phi_nodes += 1)
            elseif inst isa PiNode
                values[sidx] = get_value(inst.val)
            elseif inst isa GotoNode
                args = get_value.(collect_value_arguments(ir, block_id, inst.label))
                dest = blocks[inst.label]
                location = Location(string(line.file), line.line, 0)
                push!(current_block, cf.br(args; dest, location))
            elseif inst isa GotoIfNot
                false_args = get_value.(collect_value_arguments(ir, block_id, inst.dest))
                cond = get_value(inst.cond)
                @assert length(bb.succs) == 2 # NOTE: We assume that length(bb.succs) == 2, this might be wrong
                other_dest = setdiff(bb.succs, inst.dest) |> only
                true_args = get_value.(collect_value_arguments(ir, block_id, other_dest))
                other_dest = blocks[other_dest]
                dest = blocks[inst.dest]

                location = Location(string(line.file), line.line, 0)
                cond_br = cf.cond_br(cond, true_args, false_args; trueDest=other_dest, falseDest=dest, location)
                push!(current_block, cond_br)
            elseif inst isa ReturnNode
                line = ir.linetable[stmt[:line]+1]
                location = Location(string(line.file), line.line, 0)
                push!(current_block, func.return_([get_value(inst.val)]; location))
            elseif Meta.isexpr(inst, :code_coverage_effect)
                # Skip
            else
                error("unhandled ir $(inst)")
            end
        end
    end

    func_name = nameof(f)

    region = Region()
    for b in blocks
        push!(region, b)
    end

    input_types = IR.Type[
        IR.type(IR.argument(entry_block, i))
        for i in 1:IR.nargs(entry_block)
    ]
    result_types = [IR.Type(ret)]

    ftype = IR.FunctionType(input_types, result_types)
    op = IR.create_operation(
        "func.func",
        Location();
        attributes=[
            IR.NamedAttribute("sym_name", IR.Attribute(string(func_name))),
            IR.NamedAttribute("function_type", IR.Attribute(ftype)),
        ],
        owned_regions=Region[region],
        result_inference=false,
    )

    IR.verifyall(op)

    op
end

"""
    @code_mlir f(args...)
"""
macro code_mlir(call)
    @assert Meta.isexpr(call, :call) "only calls are supported"

    f = first(call.args) |> esc
    args = Expr(:curly,
        Tuple,
        map(arg -> :($(Core.Typeof)($arg)),
            call.args[begin+1:end])...,
    ) |> esc

    quote
        code_mlir($f, $args)
    end
end

export code_mlir, @code_mlir

end # module Brutus

# ---

function pow(x::F, n) where {F}
    p = one(F)
    for _ in 1:n
        p *= x
    end
    p
end

function f(x)
    if x == 1
        2
    else
        3
    end
end

# ---

using Test, LLVM
using MLIR.IR, MLIR

fptr = IR.context!(IR.Context()) do
    op = Brutus.code_mlir(pow, Tuple{Int,Int})

    mod = IR.Module(Location())
    body = IR.body(mod)
    push!(body, op)

    pm = IR.PassManager()
    opm = IR.OpPassManager(pm)

    # IR.enable_ir_printing!(pm)
    IR.enable_verifier!(pm, true)

    MLIR.API.mlirRegisterAllPasses()
    MLIR.API.mlirRegisterAllLLVMTranslations(IR.context())
    IR.add_pipeline!(opm, "convert-arith-to-llvm,convert-func-to-llvm")

    IR.run!(pm, mod)

    jit = if LLVM.version() >= v"16"
        MLIR.API.mlirExecutionEngineCreate(mod, 0, 0, C_NULL, false)
    else
        MLIR.API.mlirExecutionEngineCreate(mod, 0, 0, C_NULL)
    end
    MLIR.API.mlirExecutionEngineLookup(jit, "pow")
end

x, y = 3, 4

@test ccall(fptr, Int, (Int, Int), x, y) == pow(x, y)
