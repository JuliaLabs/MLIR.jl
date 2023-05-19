module Brutus

import LLVM
using MLIR.IR
using MLIR.Dialects: arith, func, cf, std
using Core: PhiNode, GotoNode, GotoIfNot, SSAValue, Argument, ReturnNode, PiNode

const BrutusScalar = Union{Bool,Int64,Int32,Float32,Float64}

function cmpi_pred(predicate)
    function(ctx, ops; loc=Location(ctx))
        arith.cmpi(ctx, predicate, ops; loc)
    end
end

function single_op_wrapper(fop)
    (ctx::Context, block::Block, args::Vector{Value}; loc=Location(ctx)) -> push!(block, fop(ctx, args; loc))
end

const intrinsics_to_mlir = Dict([
    Base.add_int => single_op_wrapper(arith.addi),
    Base.sle_int => single_op_wrapper(cmpi_pred(arith.Predicates.sle)),
    Base.slt_int => single_op_wrapper(cmpi_pred(arith.Predicates.slt)),
    Base.:(===) =>  single_op_wrapper(cmpi_pred(arith.Predicates.eq)),
    Base.mul_int => single_op_wrapper(arith.muli),
    Base.mul_float => single_op_wrapper(arith.mulf),
    Base.not_int => function(ctx, block, args; loc=Location(ctx))
        arg = only(args)
        ones = push!(block, arith.constant(ctx, -1, IR.get_type(arg); loc)) |> IR.get_result
        push!(block, arith.xori(ctx, Value[arg, ones]; loc))
    end,
])

"Generates a block argument for each phi node present in the block."
function prepare_block(ctx, ir, bb)
    b = Block()

    for sidx in bb.stmts
        stmt = ir.stmts[sidx]
        inst = stmt[:inst]
        inst isa Core.PhiNode || continue

        type = stmt[:type]
        IR.push_argument!(b, MLIRType(ctx, type), Location(ctx))
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
    code_mlir(f, types::Type{Tuple}; ctx=Context()) -> IR.Operation

Returns a `func.func` operation corresponding to the ircode of the provided method.
This only supports a few Julia Core primitives and scalar types of type $BrutusScalar.

!!! note
    The Julia SSAIR to MLIR conversion implemented is very primitive and only supports a
    handful of primitives. A better to perform this conversion would to create a dialect
    representing Julia IR and progressively lower it to base MLIR dialects.
"""
function code_mlir(f, types; ctx=Context())
    ir, ret = Core.Compiler.code_ircode(f, types) |> only
    @assert first(ir.argtypes) isa Core.Const

    values = Vector{Value}(undef, length(ir.stmts))

    for dialect in (LLVM.version() >= v"15" ? ("func", "cf") : ("std",))
        IR.get_or_load_dialect!(ctx, dialect)
    end

    blocks = [
        prepare_block(ctx, ir, bb)
        for bb in ir.cfg.blocks
    ]

    current_block = entry_block = blocks[begin]

    for argtype in types.parameters
        IR.push_argument!(entry_block, MLIRType(ctx, argtype), Location(ctx))
    end

    function get_value(x)::Value
        if x isa Core.SSAValue
            @assert isassigned(values, x.id) "value $x was not assigned"
            values[x.id]
        elseif x isa Core.Argument
            IR.get_argument(entry_block, x.n - 1)
        elseif x isa BrutusScalar
            IR.get_result(push!(current_block, arith.constant(ctx, x)))
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
            line = ir.linetable[stmt[:line]]

            if Meta.isexpr(inst, :call)
                val_type = stmt[:type]
                if !(val_type <: BrutusScalar)
                    error("type $val_type is not supported")
                end
                out_type = MLIRType(ctx, val_type)

                called_func = first(inst.args)
                if called_func isa GlobalRef # TODO: should probably use something else here
                    called_func = getproperty(called_func.mod, called_func.name)
                end

                fop! = intrinsics_to_mlir[called_func]
                args = get_value.(@view inst.args[begin+1:end])

                loc = Location(ctx, string(line.file), line.line, 0)
                res = IR.get_result(fop!(ctx, current_block, args; loc))

                values[sidx] = res
            elseif inst isa PhiNode
                values[sidx] = IR.get_argument(current_block, n_phi_nodes += 1)
            elseif inst isa PiNode
                values[sidx] = get_value(inst.val)
            elseif inst isa GotoNode
                args = get_value.(collect_value_arguments(ir, block_id, inst.label))
                dest = blocks[inst.label]
                loc = Location(ctx, string(line.file), line.line, 0)
                brop = LLVM.version() >= v"15" ? cf.br : std.br
                push!(current_block, brop(ctx, dest, args; loc))
            elseif inst isa GotoIfNot
                false_args = get_value.(collect_value_arguments(ir, block_id, inst.dest))
                cond = get_value(inst.cond)
                @assert length(bb.succs) == 2 # NOTE: We assume that length(bb.succs) == 2, this might be wrong
                other_dest = setdiff(bb.succs, inst.dest) |> only
                true_args = get_value.(collect_value_arguments(ir, block_id, other_dest))
                other_dest = blocks[other_dest]
                dest = blocks[inst.dest]

                loc = Location(ctx, string(line.file), line.line, 0)
                cond_brop = LLVM.version() >= v"15" ? cf.cond_br : std.cond_br
                cond_br = cond_brop(ctx, cond, other_dest, dest, true_args, false_args; loc)
                push!(current_block, cond_br)
            elseif inst isa ReturnNode
                line = ir.linetable[stmt[:line]]
                retop = LLVM.version() >= v"15" ? func.return_ : std.return_
                loc = Location(ctx, string(line.file), line.line, 0)
                push!(current_block, retop(ctx, [get_value(inst.val)]; loc))
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

    LLVM15 = LLVM.version() >= v"15"

    input_types = MLIRType[
        IR.get_type(IR.get_argument(entry_block, i))
        for i in 1:IR.num_arguments(entry_block)
    ]
    result_types = [MLIRType(ctx, ret)]

    ftype = MLIRType(ctx, input_types => result_types)
    op = IR.create_operation(
        LLVM15 ? "func.func" : "builtin.func",
        Location(ctx);
        attributes = [
            NamedAttribute(ctx, "sym_name", IR.Attribute(ctx, string(func_name))),
            NamedAttribute(ctx, LLVM15 ? "function_type" : "type", IR.Attribute(ftype)),
        ],
        owned_regions = Region[region],
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

using Test
using MLIR.IR, MLIR

ctx = Context()
# IR.enable_multithreading!(ctx, false)

op = Brutus.code_mlir(pow, Tuple{Int, Int}; ctx)

mod = MModule(ctx, Location(ctx))
body = IR.get_body(mod)
push!(body, op)

pm = IR.PassManager(ctx)
opm = IR.OpPassManager(pm)

# IR.enable_ir_printing!(pm)
IR.enable_verifier!(pm, true)

MLIR.API.mlirRegisterAllPasses()
MLIR.API.mlirRegisterAllLLVMTranslations(ctx)
IR.add_pipeline!(opm, Brutus.LLVM.version() >= v"15" ? "convert-arith-to-llvm,convert-func-to-llvm" : "convert-std-to-llvm")

IR.run!(pm, mod)

jit = MLIR.API.mlirExecutionEngineCreate(mod, 0, 0, C_NULL)
fptr = MLIR.API.mlirExecutionEngineLookup(jit, "pow")

x, y = 3, 4

@test ccall(fptr, Int, (Int, Int), x, y) == pow(x, y)
