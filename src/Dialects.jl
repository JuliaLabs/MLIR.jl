module Dialects

module arith

using ...IR

for (f, t) in Iterators.product(
    (:add, :sub, :mul),
    (:i, :f),
)
    fname = Symbol(f, t)
    @eval function $fname(context, operands, type=IR.get_type(first(operands)); loc=Location(context))
        IR.create_operation($(string("arith.", fname)), loc; operands, results=[type])
    end
end

for fname in (:xori, :andi, :ori)
    @eval function $fname(context, operands, type=IR.get_type(first(operands)); loc=Location(context))
        IR.create_operation($(string("arith.", fname)), loc; operands, results=[type])
    end
end

for (f, t) in Iterators.product(
    (:div, :max, :min),
    (:si, :ui, :f),
)
    fname = Symbol(f, t)
    @eval function $fname(context, operands, type=IR.get_type(first(operands)); loc=Location(context))
        IR.create_operation($(string("arith.", fname)), loc; operands, results=[type])
    end
end

# https://mlir.llvm.org/docs/Dialects/ArithOps/#arithindex_cast-mlirarithindexcastop
for f in (:index_cast, :index_castui)
    @eval function $f(context, operand; loc=Location(context))
        IR.create_operation(
            $(string("arith.", f)),
            loc;
            operands=[operand],
            results=[IR.IndexType(context)],
        )
    end
end

# https://mlir.llvm.org/docs/Dialects/ArithOps/#arithextf-mlirarithextfop
function extf(context, operand, type; loc=Location(context))
    IR.create_operation("arith.exf", loc; operands=[operand], results=[type])
end

# https://mlir.llvm.org/docs/Dialects/ArithOps/#arithconstant-mlirarithconstantop
function constant(context, value, type=MLIRType(context, typeof(value)); loc=Location(context))
    IR.create_operation(
      "arith.constant",
      loc;
      results=[type],
      attributes=[
          IR.NamedAttribute(context, "value",
              Attribute(context, value, type)),
      ],
    )
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
    IR.create_operation(
        "arith.cmpi",
        loc;
        operands,
        results=[MLIRType(context, Bool)],
        attributes=[
            IR.NamedAttribute(context, "predicate",
                Attribute(context, predicate))
        ],
    )
end

end # module arith

module std
# for llvm 14

using ...IR

function return_(context, operands; loc=Location(context))
    IR.create_operation("std.return", loc; operands, result_inference=false)
end

function br(context, dest, operands; loc=Location(context))
    IR.create_operation("std.br", loc; operands, successors=[dest], result_inference=false)
end

function cond_br(
    context, cond,
    true_dest, false_dest,
    true_dest_operands,
    false_dest_operands;
    loc=Location(context),
)
    IR.create_operation(
        "std.cond_br",
        loc;
        successors=[true_dest, false_dest],
        operands=[cond, true_dest_operands..., false_dest_operands...],
        attributes=[
            IR.NamedAttribute(context, "operand_segment_sizes",
                IR.Attribute(context, Int32[1, length(true_dest_operands), length(false_dest_operands)]))
        ],
        result_inference=false,
    )
end

end # module std

module func
# https://mlir.llvm.org/docs/Dialects/Func/

using ...IR

function return_(context, operands; loc=Location(context))
    IR.create_operation("func.return", loc; operands, result_inference=false)
end

end # module func

module cf

using ...IR

function br(context, dest, operands; loc=Location(context))
    IR.create_operation("cf.br", loc; operands, successors=[dest], result_inference=false)
end

function cond_br(
    context, cond,
    true_dest, false_dest,
    true_dest_operands,
    false_dest_operands;
    loc=Location(context),
)
    IR.create_operation(
        "cf.cond_br", loc; 
        operands=[cond, true_dest_operands..., false_dest_operands...],
        successors=[true_dest, false_dest],
        attributes=[
            IR.NamedAttribute(context, "operand_segment_sizes",
                IR.Attribute(context, Int32[1, length(true_dest_operands), length(false_dest_operands)]))
        ],
        result_inference=false,
    )
end

end # module cf

end # module Dialects
