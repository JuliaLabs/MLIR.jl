module Dialects

module arith

using ...IR

for (f, t) in Iterators.product(
    (:add, :sub, :mul),
    (:i, :f),
)
    fname = Symbol(f, t)
    @eval function $fname(context, operands, type=IR.get_type(first(operands)); loc=Location(context))
        state = OperationState($(string("arith.", fname)), loc)
        IR.add_operands!(state, operands)
        IR.add_results!(state, [type])
        Operation(state)
    end
end

for fname in (:xori, :andi, :ori)
    @eval function $fname(context, operands, type=IR.get_type(first(operands)); loc=Location(context))
        state = OperationState($(string("arith.", fname)), loc)
        IR.add_operands!(state, operands)
        IR.add_results!(state, [type])
        Operation(state)
    end
end

for (f, t) in Iterators.product(
    (:div, :max, :min),
    (:si, :ui, :f),
)
    fname = Symbol(f, t)
    @eval function $fname(context, operands, type=IR.get_type(first(operands)); loc=Location(context))
        state = OperationState($(string("arith.", fname)), loc)
        IR.add_operands!(state, operands)
        IR.add_results!(state, [type])
        Operation(state)
    end
end

# https://mlir.llvm.org/docs/Dialects/ArithOps/#arithindex_cast-mlirarithindexcastop
for f in (:index_cast, :index_castui)
    @eval function $f(context, operand; loc=Location(context))
        state = OperationState($(string("arith.", f)), loc)
        add_operands!(state, [operand])
        add_results!(state, [IR.IndexType(context)])
        Operation(state)
    end
end

# https://mlir.llvm.org/docs/Dialects/ArithOps/#arithextf-mlirarithextfop
function extf(context, operand, type; loc=Location(context))
    state = OperationState("arith.exf", loc)
    IR.add_results!(state, [type])
    IR.add_operands!(state, [operand])
    Operation(state)
end

# https://mlir.llvm.org/docs/Dialects/ArithOps/#arithsitofp-mlirarithsitofpop
function sitofp(context, operand, ftype=float(julia_type(eltype(get_type(operand)))); loc=Location(context))
    state = OperationState("arith.sitofp", loc)
    type = get_type(operand)
    IR.add_results!(state, [
        IR.is_tensor(type) ?
        MType(context, ftype isa MType ? eltype(ftype) : MType(context, ftype), size(type)) :
        MType(context, ftype)
    ])
    IR.add_operands!(state, [operand])
    Operation(state)
end

# https://mlir.llvm.org/docs/Dialects/ArithOps/#arithfptosi-mlirarithfptosiop
function fptosi(context, operand, itype; loc=Location(context))
    state = OperationState("arith.fptosi", loc)
    type = get_type(operand)
    IR.add_results!(state, [
        IR.is_tensor(type) ?
        MType(context, itype isa MType ? itype : MType(context, itype), size(type)) :
        MType(context, itype)
    ])
    IR.add_operands!(state, [operand])
    Operation(state)
end


# https://mlir.llvm.org/docs/Dialects/ArithOps/#arithconstant-mlirarithconstantop
function constant(context, value, type=MType(context, typeof(value)); loc=Location(context))
    state = OperationState("arith.constant", loc)
    IR.add_results!(state, [type])
    IR.add_attributes!(state, [
        IR.NamedAttribute(context, "value",
            Attribute(context, value, type)),
    ])
    Operation(state)
end

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

function cmpi(context, predicate, operands; loc=Location(context))
    state = OperationState("arith.cmpi", loc)
    IR.add_operands!(state, operands)
    IR.add_attributes!(state, [
        IR.NamedAttribute(context, "predicate",
            Attribute(context, predicate))
    ])
    IR.add_results!(state, [MType(context, Bool)])
    Operation(state)
end

end # module arith

module std
# for llvm 14

using ...IR

function return_(context, operands; loc=Location(context))
    state = OperationState("std.return", loc)
    IR.add_operands!(state, operands)
    Operation(state)
end

function br(context, dest, operands; loc=Location(context))
    state = OperationState("std.br", loc)
    IR.add_successors!(state, [dest])
    IR.add_operands!(state, operands)
    Operation(state)
end

function cond_br(
    context, cond,
    true_dest, false_dest,
    true_dest_operands,
    false_dest_operands;
    loc=Location(context),
)
    state = OperationState("std.cond_br", loc)
    IR.add_successors!(state, [true_dest, false_dest])
    IR.add_operands!(state, [cond, true_dest_operands..., false_dest_operands...])
    IR.add_attributes!(state, [
        IR.NamedAttribute(context, "operand_segment_sizes",
            IR.Attribute(context, Int32[1, length(true_dest_operands), length(false_dest_operands)]))
    ])
    Operation(state)
end

end # module std

module func
# https://mlir.llvm.org/docs/Dialects/Func/

using ...IR

function return_(context, operands; loc=Location(context))
    state = OperationState("func.return", loc)
    IR.add_operands!(state, operands)
    Operation(state)
end

end # module func

module cf

using ...IR

function br(context, dest, operands; loc=Location(context))
    state = OperationState("cf.br", loc)
    IR.add_successors!(state, [dest])
    IR.add_operands!(state, operands)
    Operation(state)
end

function cond_br(
    context, cond,
    true_dest, false_dest,
    true_dest_operands,
    false_dest_operands;
    loc=Location(context),
)
    state = OperationState("cf.cond_br", loc)
    IR.add_successors!(state, [true_dest, false_dest])
    IR.add_operands!(state, [cond, true_dest_operands..., false_dest_operands...])
    IR.add_attributes!(state, [
        IR.NamedAttribute(context, "operand_segment_sizes",
            IR.Attribute(context, Int32[1, length(true_dest_operands), length(false_dest_operands)]))
    ])
    Operation(state)
end

end # module cf

end # module Dialects
