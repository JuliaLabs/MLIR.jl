using Test
@warn "Running MLIR code generation tests. This can take a few minutes."

# include type and intrinsic function definitions:
include("definitions.jl");

import MLIR: IR, Generate, Dialects, API
import MLIR.IR: Value, Operation, result, Convertible, context, ValueTrait

# administrative duties
function registerAllDialects!()
    ctx = IR.context()
    registry = API.mlirDialectRegistryCreate()
    API.mlirRegisterAllDialects(registry)
    API.mlirContextAppendDialectRegistry(ctx, registry)
    API.mlirDialectRegistryDestroy(registry)

    API.mlirContextLoadAllAvailableDialects(ctx)
    return registry
end
ctx = IR.Context()
registerAllDialects!();
API.mlirRegisterAllPasses()
API.mlirRegisterAllLLVMTranslations(ctx.context)

function compare_generated(generated::IR.Module, expected::String)
  # Don't directly compare expected with generated.
  # Parse first to ensure the same SSA value names are used.
  expected = parse(IR.Module, expected)

  expected = string.(IR.OperationIterator(IR.body(expected)))
  generated = string.(IR.OperationIterator(IR.body(generated)))
  
  return Set(expected) == Set(generated)
end

# A codegen context is used to specify how Julia IR is converted to MLIR.
cg = Generate.CodegenContext()

f1(a, b) = a+b

op1 = cg(f1, Tuple{i64,i64})

@test compare_generated(
    op1,
    """
func.func private @"Tuple{typeof(f1), i64, i64}"(%a: i64, %b: i64) -> i64 {
    %sum = arith.addi %a, %b : i64
    return %sum : i64
}
""",
)

# isbitstype structs can be used as arguments or return types.
# They will be unpacked into their constituent fields that are convertible to MLIR types.
# e.g. a function `Point{i64}` taking and returning a point, will be converted to a MLIR
# function that takes two `i64` arguments and returns two `i64` values.
struct Point{T}
    x::T
    y::T
end

struct Line{T}
    p1::Point{T}
    p2::Point{T}
end

@noinline sq_distance(l::Line) = (l.p1.x - l.p2.x)^2 + (l.p1.y - l.p2.y)^2

function f2(a::Point{i64}, b::Point{i64})
    l = Line(a, b)
    d_2 = sq_distance(l)

    Point(((a.x, a.y) .- d_2)...)
end

@time op2 = cg(f2, Tuple{Point{i64},Point{i64}})

# Note the large number of basic blocks generated.
# Simplification passes in MLIR can be used to clean these up.
@test compare_generated(
    op2,
    """
  func.func private @"Tuple{typeof(f2), Point{i64}, Point{i64}}"(%arg0: i64, %arg1: i64, %arg2: i64, %arg3: i64) -> (i64, i64) {
    %0 = call @"Tuple{typeof(sq_distance), Line{i64}}"(%arg0, %arg1, %arg2, %arg3) : (i64, i64, i64, i64) -> i64
    cf.br ^bb1
  ^bb1:  // pred: ^bb0
    cf.br ^bb2
  ^bb2:  // pred: ^bb1
    cf.br ^bb3
  ^bb3:  // pred: ^bb2
    cf.br ^bb4
  ^bb4:  // pred: ^bb3
    cf.br ^bb5
  ^bb5:  // pred: ^bb4
    %1 = arith.subi %arg0, %0 : i64
    cf.br ^bb6
  ^bb6:  // pred: ^bb5
    cf.br ^bb7
  ^bb7:  // pred: ^bb6
    cf.br ^bb8
  ^bb8:  // pred: ^bb7
    cf.br ^bb9
  ^bb9:  // pred: ^bb8
    cf.br ^bb10
  ^bb10:  // pred: ^bb9
    cf.br ^bb11
  ^bb11:  // pred: ^bb10
    cf.br ^bb12
  ^bb12:  // pred: ^bb11
    %2 = arith.subi %arg1, %0 : i64
    cf.br ^bb13
  ^bb13:  // pred: ^bb12
    cf.br ^bb14
  ^bb14:  // pred: ^bb13
    cf.br ^bb15
  ^bb15:  // pred: ^bb14
    cf.br ^bb16
  ^bb16:  // pred: ^bb15
    cf.br ^bb17
  ^bb17:  // pred: ^bb16
    return %1, %2 : i64, i64
  }
  func.func private @"Tuple{typeof(sq_distance), Line{i64}}"(%arg0: i64, %arg1: i64, %arg2: i64, %arg3: i64) -> i64 {
    %0 = arith.subi %arg0, %arg2 : i64
    %1 = arith.muli %0, %0 : i64
    %2 = arith.subi %arg1, %arg3 : i64
    %3 = arith.muli %2, %2 : i64
    %4 = arith.addi %1, %3 : i64
    return %4 : i64
  }
  """
)

# To customize how goto's or return statements are converted to MLIR, we can specialize
# particular methods for a new type.
# In this case, we create a codegen context that will generate `scf.yield` for return
# statements in Julia SSA IR.
# We also change the behaviour of `generate_function` to return the generated region
# instead of wrapping it in a `func.func` operation, as we want to nest it inside a
# `scf.for` operation.
import MLIR.Generate: generate_return, generate_function, aggregate_funcs
abstract type LoopBody end
function generate_return(cg::Generate.CodegenContext{LoopBody}, values; location)
    return Dialects.scf.yield(values)
end
generate_function(cg::Generate.CodegenContext{LoopBody}, argtypes, rettypes, reg; kwargs...) = reg
aggregate_funcs(cg::Generate.CodegenContext{LoopBody}, funcs) = only(funcs)

# This intrinsic function creates an operation containing a region.
# The region body is generated by the `body` function, which is passed the intrinsic function
# as an argument.
Generate.@intrinsic function scf_for(
    body, initial_value::T, start::index, stop::index, step::index
) where {T}
    region = Generate.CodegenContext{LoopBody}()(body, Tuple{index,T})
    op = Dialects.scf.for_(
        start, stop, step, [initial_value]; results_=IR.Type[IR.Type(T)], region
    )
    return T(IR.result(op))
end

function f3(N)
    val = i64(0)
    scf_for(val, index(0), N, index(1)) do i, val
        val + i64(i)
    end
end

op3 = cg(f3, Tuple{index})

@test compare_generated(op3, """
  func.func private @"Tuple{typeof(f3), MLIRIndex}"(%arg0: index) -> i64 {
    %c0_i64 = arith.constant 0 : i64
    %idx0 = index.constant 0
    %idx1 = index.constant 1
    %0 = scf.for %arg1 = %idx0 to %arg0 step %idx1 iter_args(%arg2 = %c0_i64) -> (i64) {
      %1 = index.casts %arg1 : index to i64
      %2 = arith.addi %arg2, %1 : i64
      scf.yield %2 : i64
    }
    return %0 : i64
  }
""")
