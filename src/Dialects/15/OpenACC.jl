module acc

import ...IR: NamedAttribute, MLIRType, Value, Location, Block, Region, Attribute, create_operation, context, IndexType
import ..Dialects: namedattribute, operandsegmentsizes
import ...API


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
function data(ifCond=nothing::Union{Nothing, Value}, copyOperands::Vector{Value}, copyinOperands::Vector{Value}, copyinReadonlyOperands::Vector{Value}, copyoutOperands::Vector{Value}, copyoutZeroOperands::Vector{Value}, createOperands::Vector{Value}, createZeroOperands::Vector{Value}, noCreateOperands::Vector{Value}, presentOperands::Vector{Value}, deviceptrOperands::Vector{Value}, attachOperands::Vector{Value}; defaultAttr=nothing, region::Region, location=Location())
    results = MLIRType[]
    operands = Value[copyOperands..., copyinOperands..., copyinReadonlyOperands..., copyoutOperands..., copyoutZeroOperands..., createOperands..., createZeroOperands..., noCreateOperands..., presentOperands..., deviceptrOperands..., attachOperands..., ]
    owned_regions = Region[region, ]
    successors = Block[]
    attributes = NamedAttribute[]
    (ifCond != nothing) && push!(operands, ifCond)
    push!(attributes, operandsegmentsizes([(ifCond==nothing) ? 0 : 1length(copyOperands), length(copyinOperands), length(copyinReadonlyOperands), length(copyoutOperands), length(copyoutZeroOperands), length(createOperands), length(createZeroOperands), length(noCreateOperands), length(presentOperands), length(deviceptrOperands), length(attachOperands), ]))
    (defaultAttr != nothing) && push!(attributes, namedattribute("defaultAttr", defaultAttr))
    
    create_operation(
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
function enter_data(ifCond=nothing::Union{Nothing, Value}, asyncOperand=nothing::Union{Nothing, Value}, waitDevnum=nothing::Union{Nothing, Value}, waitOperands::Vector{Value}, copyinOperands::Vector{Value}, createOperands::Vector{Value}, createZeroOperands::Vector{Value}, attachOperands::Vector{Value}; async=nothing, wait=nothing, location=Location())
    results = MLIRType[]
    operands = Value[waitOperands..., copyinOperands..., createOperands..., createZeroOperands..., attachOperands..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (ifCond != nothing) && push!(operands, ifCond)
    (asyncOperand != nothing) && push!(operands, asyncOperand)
    (waitDevnum != nothing) && push!(operands, waitDevnum)
    push!(attributes, operandsegmentsizes([(ifCond==nothing) ? 0 : 1(asyncOperand==nothing) ? 0 : 1(waitDevnum==nothing) ? 0 : 1length(waitOperands), length(copyinOperands), length(createOperands), length(createZeroOperands), length(attachOperands), ]))
    (async != nothing) && push!(attributes, namedattribute("async", async))
    (wait != nothing) && push!(attributes, namedattribute("wait", wait))
    
    create_operation(
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
function exit_data(ifCond=nothing::Union{Nothing, Value}, asyncOperand=nothing::Union{Nothing, Value}, waitDevnum=nothing::Union{Nothing, Value}, waitOperands::Vector{Value}, copyoutOperands::Vector{Value}, deleteOperands::Vector{Value}, detachOperands::Vector{Value}; async=nothing, wait=nothing, finalize=nothing, location=Location())
    results = MLIRType[]
    operands = Value[waitOperands..., copyoutOperands..., deleteOperands..., detachOperands..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (ifCond != nothing) && push!(operands, ifCond)
    (asyncOperand != nothing) && push!(operands, asyncOperand)
    (waitDevnum != nothing) && push!(operands, waitDevnum)
    push!(attributes, operandsegmentsizes([(ifCond==nothing) ? 0 : 1(asyncOperand==nothing) ? 0 : 1(waitDevnum==nothing) ? 0 : 1length(waitOperands), length(copyoutOperands), length(deleteOperands), length(detachOperands), ]))
    (async != nothing) && push!(attributes, namedattribute("async", async))
    (wait != nothing) && push!(attributes, namedattribute("wait", wait))
    (finalize != nothing) && push!(attributes, namedattribute("finalize", finalize))
    
    create_operation(
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
function init(deviceTypeOperands::Vector{Value}, deviceNumOperand=nothing::Union{Nothing, Value}, ifCond=nothing::Union{Nothing, Value}; location=Location())
    results = MLIRType[]
    operands = Value[deviceTypeOperands..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (deviceNumOperand != nothing) && push!(operands, deviceNumOperand)
    (ifCond != nothing) && push!(operands, ifCond)
    push!(attributes, operandsegmentsizes([length(deviceTypeOperands), (deviceNumOperand==nothing) ? 0 : 1(ifCond==nothing) ? 0 : 1]))
    
    create_operation(
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
function loop(gangNum=nothing::Union{Nothing, Value}, gangStatic=nothing::Union{Nothing, Value}, workerNum=nothing::Union{Nothing, Value}, vectorLength=nothing::Union{Nothing, Value}, tileOperands::Vector{Value}, privateOperands::Vector{Value}, reductionOperands::Vector{Value}; results::Vector{MLIRType}, collapse=nothing, seq=nothing, independent=nothing, auto_=nothing, reductionOp=nothing, exec_mapping=nothing, region::Region, location=Location())
    results = MLIRType[results..., ]
    operands = Value[tileOperands..., privateOperands..., reductionOperands..., ]
    owned_regions = Region[region, ]
    successors = Block[]
    attributes = NamedAttribute[]
    (gangNum != nothing) && push!(operands, gangNum)
    (gangStatic != nothing) && push!(operands, gangStatic)
    (workerNum != nothing) && push!(operands, workerNum)
    (vectorLength != nothing) && push!(operands, vectorLength)
    push!(attributes, operandsegmentsizes([(gangNum==nothing) ? 0 : 1(gangStatic==nothing) ? 0 : 1(workerNum==nothing) ? 0 : 1(vectorLength==nothing) ? 0 : 1length(tileOperands), length(privateOperands), length(reductionOperands), ]))
    (collapse != nothing) && push!(attributes, namedattribute("collapse", collapse))
    (seq != nothing) && push!(attributes, namedattribute("seq", seq))
    (independent != nothing) && push!(attributes, namedattribute("independent", independent))
    (auto_ != nothing) && push!(attributes, namedattribute("auto_", auto_))
    (reductionOp != nothing) && push!(attributes, namedattribute("reductionOp", reductionOp))
    (exec_mapping != nothing) && push!(attributes, namedattribute("exec_mapping", exec_mapping))
    
    create_operation(
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
function parallel(async=nothing::Union{Nothing, Value}, waitOperands::Vector{Value}, numGangs=nothing::Union{Nothing, Value}, numWorkers=nothing::Union{Nothing, Value}, vectorLength=nothing::Union{Nothing, Value}, ifCond=nothing::Union{Nothing, Value}, selfCond=nothing::Union{Nothing, Value}, reductionOperands::Vector{Value}, copyOperands::Vector{Value}, copyinOperands::Vector{Value}, copyinReadonlyOperands::Vector{Value}, copyoutOperands::Vector{Value}, copyoutZeroOperands::Vector{Value}, createOperands::Vector{Value}, createZeroOperands::Vector{Value}, noCreateOperands::Vector{Value}, presentOperands::Vector{Value}, devicePtrOperands::Vector{Value}, attachOperands::Vector{Value}, gangPrivateOperands::Vector{Value}, gangFirstPrivateOperands::Vector{Value}; asyncAttr=nothing, waitAttr=nothing, selfAttr=nothing, reductionOp=nothing, defaultAttr=nothing, region::Region, location=Location())
    results = MLIRType[]
    operands = Value[waitOperands..., reductionOperands..., copyOperands..., copyinOperands..., copyinReadonlyOperands..., copyoutOperands..., copyoutZeroOperands..., createOperands..., createZeroOperands..., noCreateOperands..., presentOperands..., devicePtrOperands..., attachOperands..., gangPrivateOperands..., gangFirstPrivateOperands..., ]
    owned_regions = Region[region, ]
    successors = Block[]
    attributes = NamedAttribute[]
    (async != nothing) && push!(operands, async)
    (numGangs != nothing) && push!(operands, numGangs)
    (numWorkers != nothing) && push!(operands, numWorkers)
    (vectorLength != nothing) && push!(operands, vectorLength)
    (ifCond != nothing) && push!(operands, ifCond)
    (selfCond != nothing) && push!(operands, selfCond)
    push!(attributes, operandsegmentsizes([(async==nothing) ? 0 : 1length(waitOperands), (numGangs==nothing) ? 0 : 1(numWorkers==nothing) ? 0 : 1(vectorLength==nothing) ? 0 : 1(ifCond==nothing) ? 0 : 1(selfCond==nothing) ? 0 : 1length(reductionOperands), length(copyOperands), length(copyinOperands), length(copyinReadonlyOperands), length(copyoutOperands), length(copyoutZeroOperands), length(createOperands), length(createZeroOperands), length(noCreateOperands), length(presentOperands), length(devicePtrOperands), length(attachOperands), length(gangPrivateOperands), length(gangFirstPrivateOperands), ]))
    (asyncAttr != nothing) && push!(attributes, namedattribute("asyncAttr", asyncAttr))
    (waitAttr != nothing) && push!(attributes, namedattribute("waitAttr", waitAttr))
    (selfAttr != nothing) && push!(attributes, namedattribute("selfAttr", selfAttr))
    (reductionOp != nothing) && push!(attributes, namedattribute("reductionOp", reductionOp))
    (defaultAttr != nothing) && push!(attributes, namedattribute("defaultAttr", defaultAttr))
    
    create_operation(
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
function shutdown(deviceTypeOperands::Vector{Value}, deviceNumOperand=nothing::Union{Nothing, Value}, ifCond=nothing::Union{Nothing, Value}; location=Location())
    results = MLIRType[]
    operands = Value[deviceTypeOperands..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (deviceNumOperand != nothing) && push!(operands, deviceNumOperand)
    (ifCond != nothing) && push!(operands, ifCond)
    push!(attributes, operandsegmentsizes([length(deviceTypeOperands), (deviceNumOperand==nothing) ? 0 : 1(ifCond==nothing) ? 0 : 1]))
    
    create_operation(
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
    results = MLIRType[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
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
function update(ifCond=nothing::Union{Nothing, Value}, asyncOperand=nothing::Union{Nothing, Value}, waitDevnum=nothing::Union{Nothing, Value}, waitOperands::Vector{Value}, deviceTypeOperands::Vector{Value}, hostOperands::Vector{Value}, deviceOperands::Vector{Value}; async=nothing, wait=nothing, ifPresent=nothing, location=Location())
    results = MLIRType[]
    operands = Value[waitOperands..., deviceTypeOperands..., hostOperands..., deviceOperands..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (ifCond != nothing) && push!(operands, ifCond)
    (asyncOperand != nothing) && push!(operands, asyncOperand)
    (waitDevnum != nothing) && push!(operands, waitDevnum)
    push!(attributes, operandsegmentsizes([(ifCond==nothing) ? 0 : 1(asyncOperand==nothing) ? 0 : 1(waitDevnum==nothing) ? 0 : 1length(waitOperands), length(deviceTypeOperands), length(hostOperands), length(deviceOperands), ]))
    (async != nothing) && push!(attributes, namedattribute("async", async))
    (wait != nothing) && push!(attributes, namedattribute("wait", wait))
    (ifPresent != nothing) && push!(attributes, namedattribute("ifPresent", ifPresent))
    
    create_operation(
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
function wait(waitOperands::Vector{Value}, asyncOperand=nothing::Union{Nothing, Value}, waitDevnum=nothing::Union{Nothing, Value}, ifCond=nothing::Union{Nothing, Value}; async=nothing, location=Location())
    results = MLIRType[]
    operands = Value[waitOperands..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (asyncOperand != nothing) && push!(operands, asyncOperand)
    (waitDevnum != nothing) && push!(operands, waitDevnum)
    (ifCond != nothing) && push!(operands, ifCond)
    push!(attributes, operandsegmentsizes([length(waitOperands), (asyncOperand==nothing) ? 0 : 1(waitDevnum==nothing) ? 0 : 1(ifCond==nothing) ? 0 : 1]))
    (async != nothing) && push!(attributes, namedattribute("async", async))
    
    create_operation(
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
function yield(operands::Vector{Value}; location=Location())
    results = MLIRType[]
    operands = Value[operands..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "acc.yield", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

end # acc
