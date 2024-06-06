module acc

import ...IR: IR, NamedAttribute, Value, Location, Block, Region, Attribute, context, IndexType
import ..Dialects: namedattribute, operandsegmentsizes


"""
`data`

The \"acc.data\" operation represents a data construct. It defines vars to
be allocated in the current device memory for the duration of the region,
whether data should be copied from local memory to the current device
memory upon region entry , and copied from device memory to local memory
upon region exit.

# Example

```mlir
acc.data present(%a: memref<10x10xf32>, %b: memref<10x10xf32>,
    %c: memref<10xf32>, %d: memref<10xf32>) {
  // data region
}
```
"""
function data(ifCond=nothing::Union{Nothing, Value}; copyOperands::Vector{Value}, copyinOperands::Vector{Value}, copyinReadonlyOperands::Vector{Value}, copyoutOperands::Vector{Value}, copyoutZeroOperands::Vector{Value}, createOperands::Vector{Value}, createZeroOperands::Vector{Value}, noCreateOperands::Vector{Value}, presentOperands::Vector{Value}, deviceptrOperands::Vector{Value}, attachOperands::Vector{Value}, defaultAttr=nothing, region::Region, location=Location())
    results = IR.Type[]
    operands = Value[copyOperands..., copyinOperands..., copyinReadonlyOperands..., copyoutOperands..., copyoutZeroOperands..., createOperands..., createZeroOperands..., noCreateOperands..., presentOperands..., deviceptrOperands..., attachOperands..., ]
    owned_regions = Region[region, ]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(ifCond) && push!(operands, ifCond)
    push!(attributes, operandsegmentsizes([(ifCond==nothing) ? 0 : 1length(copyOperands), length(copyinOperands), length(copyinReadonlyOperands), length(copyoutOperands), length(copyoutZeroOperands), length(createOperands), length(createZeroOperands), length(noCreateOperands), length(presentOperands), length(deviceptrOperands), length(attachOperands), ]))
    !isnothing(defaultAttr) && push!(attributes, namedattribute("defaultAttr", defaultAttr))
    
    IR.create_operation(
        "acc.data", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`enter_data`

The \"acc.enter_data\" operation represents the OpenACC enter data directive.

# Example

```mlir
acc.enter_data create(%d1 : memref<10xf32>) attributes {async}
```
"""
function enter_data(ifCond=nothing::Union{Nothing, Value}; asyncOperand=nothing::Union{Nothing, Value}, waitDevnum=nothing::Union{Nothing, Value}, waitOperands::Vector{Value}, copyinOperands::Vector{Value}, createOperands::Vector{Value}, createZeroOperands::Vector{Value}, attachOperands::Vector{Value}, async=nothing, wait=nothing, location=Location())
    results = IR.Type[]
    operands = Value[waitOperands..., copyinOperands..., createOperands..., createZeroOperands..., attachOperands..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(ifCond) && push!(operands, ifCond)
    !isnothing(asyncOperand) && push!(operands, asyncOperand)
    !isnothing(waitDevnum) && push!(operands, waitDevnum)
    push!(attributes, operandsegmentsizes([(ifCond==nothing) ? 0 : 1(asyncOperand==nothing) ? 0 : 1(waitDevnum==nothing) ? 0 : 1length(waitOperands), length(copyinOperands), length(createOperands), length(createZeroOperands), length(attachOperands), ]))
    !isnothing(async) && push!(attributes, namedattribute("async", async))
    !isnothing(wait) && push!(attributes, namedattribute("wait", wait))
    
    IR.create_operation(
        "acc.enter_data", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`exit_data`

The \"acc.exit_data\" operation represents the OpenACC exit data directive.

# Example

```mlir
acc.exit_data delete(%d1 : memref<10xf32>) attributes {async}
```
"""
function exit_data(ifCond=nothing::Union{Nothing, Value}; asyncOperand=nothing::Union{Nothing, Value}, waitDevnum=nothing::Union{Nothing, Value}, waitOperands::Vector{Value}, copyoutOperands::Vector{Value}, deleteOperands::Vector{Value}, detachOperands::Vector{Value}, async=nothing, wait=nothing, finalize=nothing, location=Location())
    results = IR.Type[]
    operands = Value[waitOperands..., copyoutOperands..., deleteOperands..., detachOperands..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(ifCond) && push!(operands, ifCond)
    !isnothing(asyncOperand) && push!(operands, asyncOperand)
    !isnothing(waitDevnum) && push!(operands, waitDevnum)
    push!(attributes, operandsegmentsizes([(ifCond==nothing) ? 0 : 1(asyncOperand==nothing) ? 0 : 1(waitDevnum==nothing) ? 0 : 1length(waitOperands), length(copyoutOperands), length(deleteOperands), length(detachOperands), ]))
    !isnothing(async) && push!(attributes, namedattribute("async", async))
    !isnothing(wait) && push!(attributes, namedattribute("wait", wait))
    !isnothing(finalize) && push!(attributes, namedattribute("finalize", finalize))
    
    IR.create_operation(
        "acc.exit_data", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`init`

The \"acc.init\" operation represents the OpenACC init executable
directive.

# Example

```mlir
acc.init
acc.init device_num(%dev1 : i32)
```
"""
function init(deviceTypeOperands::Vector{Value}, deviceNumOperand=nothing::Union{Nothing, Value}; ifCond=nothing::Union{Nothing, Value}, location=Location())
    results = IR.Type[]
    operands = Value[deviceTypeOperands..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(deviceNumOperand) && push!(operands, deviceNumOperand)
    !isnothing(ifCond) && push!(operands, ifCond)
    push!(attributes, operandsegmentsizes([length(deviceTypeOperands), (deviceNumOperand==nothing) ? 0 : 1(ifCond==nothing) ? 0 : 1]))
    
    IR.create_operation(
        "acc.init", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`loop`

The \"acc.loop\" operation represents the OpenACC loop construct.

# Example

```mlir
acc.loop gang vector {
  scf.for %arg3 = %c0 to %c10 step %c1 {
    scf.for %arg4 = %c0 to %c10 step %c1 {
      scf.for %arg5 = %c0 to %c10 step %c1 {
        // ... body
      }
    }
  }
  acc.yield
} attributes { collapse = 3 }
```
"""
function loop(gangNum=nothing::Union{Nothing, Value}; gangStatic=nothing::Union{Nothing, Value}, workerNum=nothing::Union{Nothing, Value}, vectorLength=nothing::Union{Nothing, Value}, tileOperands::Vector{Value}, privateOperands::Vector{Value}, reductionOperands::Vector{Value}, results_::Vector{IR.Type}, collapse=nothing, seq=nothing, independent=nothing, auto_=nothing, reductionOp=nothing, exec_mapping=nothing, region::Region, location=Location())
    results = IR.Type[results_..., ]
    operands = Value[tileOperands..., privateOperands..., reductionOperands..., ]
    owned_regions = Region[region, ]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(gangNum) && push!(operands, gangNum)
    !isnothing(gangStatic) && push!(operands, gangStatic)
    !isnothing(workerNum) && push!(operands, workerNum)
    !isnothing(vectorLength) && push!(operands, vectorLength)
    push!(attributes, operandsegmentsizes([(gangNum==nothing) ? 0 : 1(gangStatic==nothing) ? 0 : 1(workerNum==nothing) ? 0 : 1(vectorLength==nothing) ? 0 : 1length(tileOperands), length(privateOperands), length(reductionOperands), ]))
    !isnothing(collapse) && push!(attributes, namedattribute("collapse", collapse))
    !isnothing(seq) && push!(attributes, namedattribute("seq", seq))
    !isnothing(independent) && push!(attributes, namedattribute("independent", independent))
    !isnothing(auto_) && push!(attributes, namedattribute("auto_", auto_))
    !isnothing(reductionOp) && push!(attributes, namedattribute("reductionOp", reductionOp))
    !isnothing(exec_mapping) && push!(attributes, namedattribute("exec_mapping", exec_mapping))
    
    IR.create_operation(
        "acc.loop", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`parallel`

The \"acc.parallel\" operation represents a parallel construct block. It has
one region to be executed in parallel on the current device.

# Example

```mlir
acc.parallel num_gangs(%c10) num_workers(%c10)
    private(%c : memref<10xf32>) {
  // parallel region
}
```
"""
function parallel(async=nothing::Union{Nothing, Value}; waitOperands::Vector{Value}, numGangs=nothing::Union{Nothing, Value}, numWorkers=nothing::Union{Nothing, Value}, vectorLength=nothing::Union{Nothing, Value}, ifCond=nothing::Union{Nothing, Value}, selfCond=nothing::Union{Nothing, Value}, reductionOperands::Vector{Value}, copyOperands::Vector{Value}, copyinOperands::Vector{Value}, copyinReadonlyOperands::Vector{Value}, copyoutOperands::Vector{Value}, copyoutZeroOperands::Vector{Value}, createOperands::Vector{Value}, createZeroOperands::Vector{Value}, noCreateOperands::Vector{Value}, presentOperands::Vector{Value}, devicePtrOperands::Vector{Value}, attachOperands::Vector{Value}, gangPrivateOperands::Vector{Value}, gangFirstPrivateOperands::Vector{Value}, asyncAttr=nothing, waitAttr=nothing, selfAttr=nothing, reductionOp=nothing, defaultAttr=nothing, region::Region, location=Location())
    results = IR.Type[]
    operands = Value[waitOperands..., reductionOperands..., copyOperands..., copyinOperands..., copyinReadonlyOperands..., copyoutOperands..., copyoutZeroOperands..., createOperands..., createZeroOperands..., noCreateOperands..., presentOperands..., devicePtrOperands..., attachOperands..., gangPrivateOperands..., gangFirstPrivateOperands..., ]
    owned_regions = Region[region, ]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(async) && push!(operands, async)
    !isnothing(numGangs) && push!(operands, numGangs)
    !isnothing(numWorkers) && push!(operands, numWorkers)
    !isnothing(vectorLength) && push!(operands, vectorLength)
    !isnothing(ifCond) && push!(operands, ifCond)
    !isnothing(selfCond) && push!(operands, selfCond)
    push!(attributes, operandsegmentsizes([(async==nothing) ? 0 : 1length(waitOperands), (numGangs==nothing) ? 0 : 1(numWorkers==nothing) ? 0 : 1(vectorLength==nothing) ? 0 : 1(ifCond==nothing) ? 0 : 1(selfCond==nothing) ? 0 : 1length(reductionOperands), length(copyOperands), length(copyinOperands), length(copyinReadonlyOperands), length(copyoutOperands), length(copyoutZeroOperands), length(createOperands), length(createZeroOperands), length(noCreateOperands), length(presentOperands), length(devicePtrOperands), length(attachOperands), length(gangPrivateOperands), length(gangFirstPrivateOperands), ]))
    !isnothing(asyncAttr) && push!(attributes, namedattribute("asyncAttr", asyncAttr))
    !isnothing(waitAttr) && push!(attributes, namedattribute("waitAttr", waitAttr))
    !isnothing(selfAttr) && push!(attributes, namedattribute("selfAttr", selfAttr))
    !isnothing(reductionOp) && push!(attributes, namedattribute("reductionOp", reductionOp))
    !isnothing(defaultAttr) && push!(attributes, namedattribute("defaultAttr", defaultAttr))
    
    IR.create_operation(
        "acc.parallel", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`shutdown`

The \"acc.shutdown\" operation represents the OpenACC shutdown executable
directive.

# Example

```mlir
acc.shutdown
acc.shutdown device_num(%dev1 : i32)
```
"""
function shutdown(deviceTypeOperands::Vector{Value}, deviceNumOperand=nothing::Union{Nothing, Value}; ifCond=nothing::Union{Nothing, Value}, location=Location())
    results = IR.Type[]
    operands = Value[deviceTypeOperands..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(deviceNumOperand) && push!(operands, deviceNumOperand)
    !isnothing(ifCond) && push!(operands, ifCond)
    push!(attributes, operandsegmentsizes([length(deviceTypeOperands), (deviceNumOperand==nothing) ? 0 : 1(ifCond==nothing) ? 0 : 1]))
    
    IR.create_operation(
        "acc.shutdown", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`terminator`

A terminator operation for regions that appear in the body of OpenACC
operation. Generic OpenACC construct regions are not expected to return any
value so the terminator takes no operands. The terminator op returns control
to the enclosing op.
"""
function terminator(; location=Location())
    results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "acc.terminator", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`update`

The \"acc.udpate\" operation represents the OpenACC update executable
directive.
As host and self clauses are synonyms, any operands for host and self are
add to \$hostOperands.

# Example

```mlir
acc.update device(%d1 : memref<10xf32>) attributes {async}
```
"""
function update(ifCond=nothing::Union{Nothing, Value}; asyncOperand=nothing::Union{Nothing, Value}, waitDevnum=nothing::Union{Nothing, Value}, waitOperands::Vector{Value}, deviceTypeOperands::Vector{Value}, hostOperands::Vector{Value}, deviceOperands::Vector{Value}, async=nothing, wait=nothing, ifPresent=nothing, location=Location())
    results = IR.Type[]
    operands = Value[waitOperands..., deviceTypeOperands..., hostOperands..., deviceOperands..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(ifCond) && push!(operands, ifCond)
    !isnothing(asyncOperand) && push!(operands, asyncOperand)
    !isnothing(waitDevnum) && push!(operands, waitDevnum)
    push!(attributes, operandsegmentsizes([(ifCond==nothing) ? 0 : 1(asyncOperand==nothing) ? 0 : 1(waitDevnum==nothing) ? 0 : 1length(waitOperands), length(deviceTypeOperands), length(hostOperands), length(deviceOperands), ]))
    !isnothing(async) && push!(attributes, namedattribute("async", async))
    !isnothing(wait) && push!(attributes, namedattribute("wait", wait))
    !isnothing(ifPresent) && push!(attributes, namedattribute("ifPresent", ifPresent))
    
    IR.create_operation(
        "acc.update", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`wait`

The \"acc.wait\" operation represents the OpenACC wait executable
directive.

# Example

```mlir
acc.wait(%value1: index)
acc.wait() async(%async1: i32)
```
"""
function wait(waitOperands::Vector{Value}, asyncOperand=nothing::Union{Nothing, Value}; waitDevnum=nothing::Union{Nothing, Value}, ifCond=nothing::Union{Nothing, Value}, async=nothing, location=Location())
    results = IR.Type[]
    operands = Value[waitOperands..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(asyncOperand) && push!(operands, asyncOperand)
    !isnothing(waitDevnum) && push!(operands, waitDevnum)
    !isnothing(ifCond) && push!(operands, ifCond)
    push!(attributes, operandsegmentsizes([length(waitOperands), (asyncOperand==nothing) ? 0 : 1(waitDevnum==nothing) ? 0 : 1(ifCond==nothing) ? 0 : 1]))
    !isnothing(async) && push!(attributes, namedattribute("async", async))
    
    IR.create_operation(
        "acc.wait", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`yield`

`acc.yield` is a special terminator operation for block inside regions in
acc ops (parallel and loop). It returns values to the immediately enclosing
acc op.
"""
function yield(operands_::Vector{Value}; location=Location())
    results = IR.Type[]
    operands = Value[operands_..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "acc.yield", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

end # acc
