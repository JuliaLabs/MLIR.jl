module Dialects

module Arith

using ...IR
using ...Builder: blockbuilder, _has_blockbuilder

for (f, t) in Iterators.product(
    (:add, :sub, :mul),
    (:i, :f),
)
    fname = Symbol(f, t)
    @eval function $fname(operands, type=IR.get_type(first(operands)); loc=Location())
        op = IR.create_operation($(string("arith.", fname)), loc; operands, results=[type])
        push!(blockbuilder().block, op)
        return IR.get_result(op, 1)
    end
end

for fname in (:xori, :andi, :ori)
    @eval function $fname(operands, type=IR.get_type(first(operands)); loc=Location())
        op = IR.create_operation($(string("arith.", fname)), loc; operands, results=[type])
        push!(blockbuilder().block, op)
        return IR.get_result(op, 1)
    end
end

for (f, t) in Iterators.product(
    (:div, :max, :min),
    (:si, :ui, :f),
)
    fname = Symbol(f, t)
    @eval function $fname(operands, type=IR.get_type(first(operands)); loc=Location())
        op = IR.create_operation($(string("arith.", fname)), loc; operands, results=[type])
        push!(blockbuilder().block, op)
        return IR.get_result(op, 1)
    end
end

# https://mlir.llvm.org/docs/Dialects/ArithOps/#arithindex_cast-mlirarithindexcastop
for f in (:index_cast, :index_castui)
    @eval function $f(operand; loc=Location())
        op = IR.create_operation(
            $(string("arith.", f)),
            loc;
            operands=[operand],
            results=[IR.IndexType()],
        )
        push!(blockbuilder().block, op)
        return IR.get_result(op, 1)
    end
end

# https://mlir.llvm.org/docs/Dialects/ArithOps/#arithextf-mlirarithextfop
function extf(operand, type; loc=Location())
    op = IR.create_operation("arith.exf", loc; operands=[operand], results=[type])
    push!(blockbuilder().block, op)
    return IR.get_result(op , 1)
end

# https://mlir.llvm.org/docs/Dialects/ArithOps/#arithconstant-mlirarithconstantop
function constant(value, type=MLIRType(typeof(value)); loc=Location())
    op = IR.create_operation(
      "arith.constant",
      loc;
      results=[type],
      attributes=[
          IR.NamedAttribute("value",
              Attribute(value, type)),
      ],
    )
    push!(blockbuilder().block, op)
    return IR.get_result(op, 1)
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

function cmpi(predicate, operands; loc=Location())
    op = IR.create_operation(
        "arith.cmpi",
        loc;
        operands,
        results=[MLIRType(Bool)],
        attributes=[
            IR.NamedAttribute("predicate",
                Attribute(predicate))
        ],
    )
    push!(blockbuilder().block, op)
    return get_result(op, 1)
end

end # module arith

module STD
# for llvm 14

using ...IR

function return_(operands; loc=Location())
    IR.create_operation("std.return", loc; operands, result_inference=false)
end

function br(dest, operands; loc=Location())
    IR.create_operation("std.br", loc; operands, successors=[dest], result_inference=false)
end

function cond_br(
    cond,
    true_dest, false_dest,
    true_dest_operands,
    false_dest_operands;
    loc=Location(),
)
    IR.create_operation(
        "std.cond_br",
        loc;
        successors=[true_dest, false_dest],
        operands=[cond, true_dest_operands..., false_dest_operands...],
        attributes=[
            IR.NamedAttribute("operand_segment_sizes",
                IR.Attribute(Int32[1, length(true_dest_operands), length(false_dest_operands)]))
        ],
        result_inference=false,
    )
end

end # module std

module Func
# https://mlir.llvm.org/docs/Dialects/Func/

using ...IR

function return_(operands; loc=Location())
    IR.create_operation("func.return", loc; operands, result_inference=false)
end

end # module func

module CF

using ...IR
using ...Builder

function br(dest, operands=[]; loc=Location())
    op = IR.create_operation("cf.br", loc; operands, successors=[dest], result_inference=false)
    push!(Builder.blockbuilder().block, op)
    return op # no value so returning operation itself (?)
end

function cond_br(
    cond,
    true_dest, false_dest,
    true_dest_operands=[],
    false_dest_operands=[];
    loc=Location(),
)
    op = IR.create_operation(
        "cf.cond_br", loc; 
        operands=[cond, true_dest_operands..., false_dest_operands...],
        successors=[true_dest, false_dest],
        attributes=[
            IR.NamedAttribute("operand_segment_sizes",
                IR.Attribute(Int32[1, length(true_dest_operands), length(false_dest_operands)]))
        ],
        result_inference=false,
    )
    push!(blockbuilder().block, op)
    return op
end

end # module cf

end # module Dialects
