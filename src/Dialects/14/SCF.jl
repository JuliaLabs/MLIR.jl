module scf

import ...IR: IR, NamedAttribute, Value, Location, Block, Region, Attribute, context, IndexType
import ..Dialects: namedattribute, operandsegmentsizes


"""
`condition`

This operation accepts the continuation (i.e., inverse of exit) condition
of the `scf.while` construct. If its first argument is true, the \"after\"
region of `scf.while` is executed, with the remaining arguments forwarded
to the entry block of the region. Otherwise, the loop terminates.
"""
function condition(condition::Value, args::Vector{Value}; location=Location())
    results = IR.Type[]
    operands = Value[condition, args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "scf.condition", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`execute_region`

The `execute_region` operation is used to allow multiple blocks within SCF
and other operations which can hold only one block.  The `execute_region`
operation executes the region held exactly once and cannot have any operands.
As such, its region has no arguments. All SSA values that dominate the op can
be accessed inside the op. The op\'s region can have multiple blocks and the
blocks can have multiple distinct terminators. Values returned from this op\'s
region define the op\'s results.

# Example

```mlir
scf.for %i = 0 to 128 step %c1 {
  %y = scf.execute_region -> i32 {
    %x = load %A[%i] : memref<128xi32>
    scf.yield %x : i32
  }
}

affine.for %i = 0 to 100 {
  \"foo\"() : () -> ()
  %v = scf.execute_region -> i64 {
    cond_br %cond, ^bb1, ^bb2

  ^bb1:
    %c1 = arith.constant 1 : i64
    br ^bb3(%c1 : i64)

  ^bb2:
    %c2 = arith.constant 2 : i64
    br ^bb3(%c2 : i64)

  ^bb3(%x : i64):
    scf.yield %x : i64
  }
  \"bar\"(%v) : (i64) -> ()
}
```
"""
function execute_region(; result_0::Vector{IR.Type}, region::Region, location=Location())
    results = IR.Type[result_0..., ]
    operands = Value[]
    owned_regions = Region[region, ]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "scf.execute_region", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`for_`

The \"scf.for\" operation represents a loop taking 3 SSA value as operands
that represent the lower bound, upper bound and step respectively. The
operation defines an SSA value for its induction variable. It has one
region capturing the loop body. The induction variable is represented as an
argument of this region. This SSA value always has type index, which is the
size of the machine word. The step is a value of type index, required to be
positive.
The lower and upper bounds specify a half-open range: the range includes
the lower bound but does not include the upper bound.

The body region must contain exactly one block that terminates with
\"scf.yield\". Calling ForOp::build will create such a region and insert
the terminator implicitly if none is defined, so will the parsing even in
cases when it is absent from the custom format. For example:

```mlir
scf.for %iv = %lb to %ub step %step {
  ... // body
}
```

`scf.for` can also operate on loop-carried variables and returns the final
values after loop termination. The initial values of the variables are
passed as additional SSA operands to the \"scf.for\" following the 3 loop
control SSA values mentioned above (lower bound, upper bound and step). The
operation region has an argument for the induction variable, followed by
one argument for each loop-carried variable, representing the value of the
variable at the current iteration.

The region must terminate with a \"scf.yield\" that passes the current
values of all loop-carried variables to the next iteration, or to the
\"scf.for\" result, if at the last iteration. The static type of a
loop-carried variable may not change with iterations; its runtime type is
allowed to change. Note, that when the loop-carried variables are present,
calling ForOp::build will not insert the terminator implicitly. The caller
must insert \"scf.yield\" in that case.

\"scf.for\" results hold the final values after the last iteration.
For example, to sum-reduce a memref:

```mlir
func @reduce(%buffer: memref<1024xf32>, %lb: index,
             %ub: index, %step: index) -> (f32) {
  // Initial sum set to 0.
  %sum_0 = arith.constant 0.0 : f32
  // iter_args binds initial values to the loop\'s region arguments.
  %sum = scf.for %iv = %lb to %ub step %step
      iter_args(%sum_iter = %sum_0) -> (f32) {
    %t = load %buffer[%iv] : memref<1024xf32>
    %sum_next = arith.addf %sum_iter, %t : f32
    // Yield current iteration sum to next iteration %sum_iter or to %sum
    // if final iteration.
    scf.yield %sum_next : f32
  }
  return %sum : f32
}
```

If the \"scf.for\" defines any values, a yield must be explicitly present.
The number and types of the \"scf.for\" results must match the initial
values in the \"iter_args\" binding and the yield operands.

Another example with a nested \"scf.if\" (see \"scf.if\" for details) to
perform conditional reduction:

```mlir
func @conditional_reduce(%buffer: memref<1024xf32>, %lb: index,
                         %ub: index, %step: index) -> (f32) {
  %sum_0 = arith.constant 0.0 : f32
  %c0 = arith.constant 0.0 : f32
  %sum = scf.for %iv = %lb to %ub step %step
      iter_args(%sum_iter = %sum_0) -> (f32) {
    %t = load %buffer[%iv] : memref<1024xf32>
    %cond = arith.cmpf \"ugt\", %t, %c0 : f32
    %sum_next = scf.if %cond -> (f32) {
      %new_sum = arith.addf %sum_iter, %t : f32
      scf.yield %new_sum : f32
    } else {
      scf.yield %sum_iter : f32
    }
    scf.yield %sum_next : f32
  }
  return %sum : f32
}
```
"""
function for_(lowerBound::Value, upperBound::Value, step::Value, initArgs::Vector{Value}; results_::Vector{IR.Type}, region::Region, location=Location())
    results = IR.Type[results_..., ]
    operands = Value[lowerBound, upperBound, step, initArgs..., ]
    owned_regions = Region[region, ]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "scf.for", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`if_`

The `scf.if` operation represents an if-then-else construct for
conditionally executing two regions of code. The operand to an if operation
is a boolean value. For example:

```mlir
scf.if %b  {
  ...
} else {
  ...
}
```

`scf.if` may also return results that are defined in its regions. The
values defined are determined by which execution path is taken.

# Example

```mlir
%x, %y = scf.if %b -> (f32, f32) {
  %x_true = ...
  %y_true = ...
  scf.yield %x_true, %y_true : f32, f32
} else {
  %x_false = ...
  %y_false = ...
  scf.yield %x_false, %y_false : f32, f32
}
```

`scf.if` regions are always terminated with \"scf.yield\". If \"scf.if\"
defines no values, the \"scf.yield\" can be left out, and will be inserted
implicitly. Otherwise, it must be explicit.
Also, if \"scf.if\" defines one or more values, the \'else\' block cannot be
omitted.

# Example

```mlir
scf.if %b  {
  ...
}
```
"""
function if_(condition::Value; results_::Vector{IR.Type}, thenRegion::Region, elseRegion::Region, location=Location())
    results = IR.Type[results_..., ]
    operands = Value[condition, ]
    owned_regions = Region[thenRegion, elseRegion, ]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "scf.if", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`parallel`

The \"scf.parallel\" operation represents a loop nest taking 4 groups of SSA
values as operands that represent the lower bounds, upper bounds, steps and
initial values, respectively. The operation defines a variadic number of
SSA values for its induction variables. It has one region capturing the
loop body. The induction variables are represented as an argument of this
region. These SSA values always have type index, which is the size of the
machine word. The steps are values of type index, required to be positive.
The lower and upper bounds specify a half-open range: the range includes
the lower bound but does not include the upper bound. The initial values
have the same types as results of \"scf.parallel\". If there are no results,
the keyword `init` can be omitted.

Semantically we require that the iteration space can be iterated in any
order, and the loop body can be executed in parallel. If there are data
races, the behavior is undefined.

The parallel loop operation supports reduction of values produced by
individual iterations into a single result. This is modeled using the
scf.reduce operation (see scf.reduce for details). Each result of a
scf.parallel operation is associated with an initial value operand and
reduce operation that is an immediate child. Reductions are matched to
result and initial values in order of their appearance in the body.
Consequently, we require that the body region has the same number of
results and initial values as it has reduce operations.

The body region must contain exactly one block that terminates with
\"scf.yield\" without operands. Parsing ParallelOp will create such a region
and insert the terminator when it is absent from the custom format.

# Example

```mlir
%init = arith.constant 0.0 : f32
scf.parallel (%iv) = (%lb) to (%ub) step (%step) init (%init) -> f32 {
  %elem_to_reduce = load %buffer[%iv] : memref<100xf32>
  scf.reduce(%elem_to_reduce) : f32 {
    ^bb0(%lhs : f32, %rhs: f32):
      %res = arith.addf %lhs, %rhs : f32
      scf.reduce.return %res : f32
  }
}
```
"""
function parallel(lowerBound::Vector{Value}, upperBound::Vector{Value}, step::Vector{Value}, initVals::Vector{Value}; results_::Vector{IR.Type}, region::Region, location=Location())
    results = IR.Type[results_..., ]
    operands = Value[lowerBound..., upperBound..., step..., initVals..., ]
    owned_regions = Region[region, ]
    successors = Block[]
    attributes = NamedAttribute[]
    push!(attributes, operandsegmentsizes([length(lowerBound), length(upperBound), length(step), length(initVals), ]))
    
    IR.create_operation(
        "scf.parallel", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`reduce`

\"scf.reduce\" is an operation occurring inside \"scf.parallel\" operations.
It consists of one block with two arguments which have the same type as the
operand of \"scf.reduce\".

\"scf.reduce\" is used to model the value for reduction computations of a
\"scf.parallel\" operation. It has to appear as an immediate child of a
\"scf.parallel\" and is associated with a result value of its parent
operation.

Association is in the order of appearance in the body where the first
result of a parallel loop operation corresponds to the first \"scf.reduce\"
in the operation\'s body region. The reduce operation takes a single
operand, which is the value to be used in the reduction.

The reduce operation contains a region whose entry block expects two
arguments of the same type as the operand. As the iteration order of the
parallel loop and hence reduction order is unspecified, the result of
reduction may be non-deterministic unless the operation is associative and
commutative.

The result of the reduce operation\'s body must have the same type as the
operands and associated result value of the parallel loop operation.
# Example

```mlir
%operand = arith.constant 1.0 : f32
scf.reduce(%operand) : f32 {
  ^bb0(%lhs : f32, %rhs: f32):
    %res = arith.addf %lhs, %rhs : f32
    scf.reduce.return %res : f32
}
```
"""
function reduce(operand::Value; reductionOperator::Region, location=Location())
    results = IR.Type[]
    operands = Value[operand, ]
    owned_regions = Region[reductionOperator, ]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "scf.reduce", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`reduce_return`

\"scf.reduce.return\" is a special terminator operation for the block inside
\"scf.reduce\". It terminates the region. It should have the same type as
the operand of \"scf.reduce\". Example for the custom format:

```mlir
scf.reduce.return %res : f32
```
"""
function reduce_return(result::Value; location=Location())
    results = IR.Type[]
    operands = Value[result, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "scf.reduce.return", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`while_`

This operation represents a generic \"while\"/\"do-while\" loop that keeps
iterating as long as a condition is satisfied. There is no restriction on
the complexity of the condition. It consists of two regions (with single
block each): \"before\" region and \"after\" region. The names of regions
indicates whether they execute before or after the condition check.
Therefore, if the main loop payload is located in the \"before\" region, the
operation is a \"do-while\" loop. Otherwise, it is a \"while\" loop.

The \"before\" region terminates with a special operation, `scf.condition`,
that accepts as its first operand an `i1` value indicating whether to
proceed to the \"after\" region (value is `true`) or not. The two regions
communicate by means of region arguments. Initially, the \"before\" region
accepts as arguments the operands of the `scf.while` operation and uses them
to evaluate the condition. It forwards the trailing, non-condition operands
of the `scf.condition` terminator either to the \"after\" region if the
control flow is transferred there or to results of the `scf.while` operation
otherwise. The \"after\" region takes as arguments the values produced by the
\"before\" region and uses `scf.yield` to supply new arguments for the
\"before\" region, into which it transfers the control flow unconditionally.

A simple \"while\" loop can be represented as follows.

```mlir
%res = scf.while (%arg1 = %init1) : (f32) -> f32 {
  /* \"Before\" region.
   * In a \"while\" loop, this region computes the condition. */
  %condition = call @evaluate_condition(%arg1) : (f32) -> i1

  /* Forward the argument (as result or \"after\" region argument). */
  scf.condition(%condition) %arg1 : f32

} do {
^bb0(%arg2: f32):
  /* \"After region.
   * In a \"while\" loop, this region is the loop body. */
  %next = call @payload(%arg2) : (f32) -> f32

  /* Forward the new value to the \"before\" region.
   * The operand types must match the types of the `scf.while` operands. */
  scf.yield %next : f32
}
```

A simple \"do-while\" loop can be represented by reducing the \"after\" block
to a simple forwarder.

```mlir
%res = scf.while (%arg1 = %init1) : (f32) -> f32 {
  /* \"Before\" region.
   * In a \"do-while\" loop, this region contains the loop body. */
  %next = call @payload(%arg1) : (f32) -> f32

  /* And also evaluates the condition. */
  %condition = call @evaluate_condition(%arg1) : (f32) -> i1

  /* Loop through the \"after\" region. */
  scf.condition(%condition) %next : f32

} do {
^bb0(%arg2: f32):
  /* \"After\" region.
   * Forwards the values back to \"before\" region unmodified. */
  scf.yield %arg2 : f32
}
```

Note that the types of region arguments need not to match with each other.
The op expects the operand types to match with argument types of the
\"before\" region\"; the result types to match with the trailing operand types
of the terminator of the \"before\" region, and with the argument types of the
\"after\" region. The following scheme can be used to share the results of
some operations executed in the \"before\" region with the \"after\" region,
avoiding the need to recompute them.

```mlir
%res = scf.while (%arg1 = %init1) : (f32) -> i64 {
  /* One can perform some computations, e.g., necessary to evaluate the
   * condition, in the \"before\" region and forward their results to the
   * \"after\" region. */
  %shared = call @shared_compute(%arg1) : (f32) -> i64

  /* Evaluate the condition. */
  %condition = call @evaluate_condition(%arg1, %shared) : (f32, i64) -> i1

  /* Forward the result of the shared computation to the \"after\" region.
   * The types must match the arguments of the \"after\" region as well as
   * those of the `scf.while` results. */
  scf.condition(%condition) %shared : i64

} do {
^bb0(%arg2: i64) {
  /* Use the partial result to compute the rest of the payload in the
   * \"after\" region. */
  %res = call @payload(%arg2) : (i64) -> f32

  /* Forward the new value to the \"before\" region.
   * The operand types must match the types of the `scf.while` operands. */
  scf.yield %res : f32
}
```

The custom syntax for this operation is as follows.

```
op ::= `scf.while` assignments `:` function-type region `do` region
       `attributes` attribute-dict
initializer ::= /* empty */ | `(` assignment-list `)`
assignment-list ::= assignment | assignment `,` assignment-list
assignment ::= ssa-value `=` ssa-value
```
"""
function while_(inits::Vector{Value}; results_::Vector{IR.Type}, before::Region, after::Region, location=Location())
    results = IR.Type[results_..., ]
    operands = Value[inits..., ]
    owned_regions = Region[before, after, ]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "scf.while", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`yield`

\"scf.yield\" yields an SSA value from the SCF dialect op region and
terminates the regions. The semantics of how the values are yielded is
defined by the parent operation.
If \"scf.yield\" has any operands, the operands must match the parent
operation\'s results.
If the parent operation defines no values, then the \"scf.yield\" may be
left out in the custom syntax and the builders will insert one implicitly.
Otherwise, it has to be present in the syntax to indicate which values are
yielded.
"""
function yield(results_::Vector{Value}; location=Location())
    results = IR.Type[]
    operands = Value[results_..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "scf.yield", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

end # scf
