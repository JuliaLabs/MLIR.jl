module acc

import ...IR: IR, NamedAttribute, Value, value, Location, Block, Region, Attribute, context, IndexType
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
function data(ifCond=nothing; copyOperands, copyinOperands, copyinReadonlyOperands, copyoutOperands, copyoutZeroOperands, createOperands, createZeroOperands, noCreateOperands, presentOperands, deviceptrOperands, attachOperands, defaultAttr=nothing, region::Region, location=Location())
    results = IR.Type[]
    operands = Value[value.(copyOperands)..., value.(copyinOperands)..., value.(copyinReadonlyOperands)..., value.(copyoutOperands)..., value.(copyoutZeroOperands)..., value.(createOperands)..., value.(createZeroOperands)..., value.(noCreateOperands)..., value.(presentOperands)..., value.(deviceptrOperands)..., value.(attachOperands)..., ]
    owned_regions = Region[region, ]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(ifCond) && push!(operands, value(ifCond))
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
function enter_data(ifCond=nothing; asyncOperand=nothing, waitDevnum=nothing, waitOperands, copyinOperands, createOperands, createZeroOperands, attachOperands, async=nothing, wait=nothing, location=Location())
    results = IR.Type[]
    operands = Value[value.(waitOperands)..., value.(copyinOperands)..., value.(createOperands)..., value.(createZeroOperands)..., value.(attachOperands)..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(ifCond) && push!(operands, value(ifCond))
    !isnothing(asyncOperand) && push!(operands, value(asyncOperand))
    !isnothing(waitDevnum) && push!(operands, value(waitDevnum))
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
function exit_data(ifCond=nothing; asyncOperand=nothing, waitDevnum=nothing, waitOperands, copyoutOperands, deleteOperands, detachOperands, async=nothing, wait=nothing, finalize=nothing, location=Location())
    results = IR.Type[]
    operands = Value[value.(waitOperands)..., value.(copyoutOperands)..., value.(deleteOperands)..., value.(detachOperands)..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(ifCond) && push!(operands, value(ifCond))
    !isnothing(asyncOperand) && push!(operands, value(asyncOperand))
    !isnothing(waitDevnum) && push!(operands, value(waitDevnum))
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
function init(deviceTypeOperands, deviceNumOperand=nothing; ifCond=nothing, location=Location())
    results = IR.Type[]
    operands = Value[value.(deviceTypeOperands)..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(deviceNumOperand) && push!(operands, value(deviceNumOperand))
    !isnothing(ifCond) && push!(operands, value(ifCond))
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
function loop(gangNum=nothing; gangStatic=nothing, workerNum=nothing, vectorLength=nothing, tileOperands, privateOperands, reductionOperands, results_::Vector{IR.Type}, collapse=nothing, seq=nothing, independent=nothing, auto_=nothing, reductionOp=nothing, exec_mapping=nothing, region::Region, location=Location())
    results = IR.Type[results_..., ]
    operands = Value[value.(tileOperands)..., value.(privateOperands)..., value.(reductionOperands)..., ]
    owned_regions = Region[region, ]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(gangNum) && push!(operands, value(gangNum))
    !isnothing(gangStatic) && push!(operands, value(gangStatic))
    !isnothing(workerNum) && push!(operands, value(workerNum))
    !isnothing(vectorLength) && push!(operands, value(vectorLength))
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
function parallel(async=nothing; waitOperands, numGangs=nothing, numWorkers=nothing, vectorLength=nothing, ifCond=nothing, selfCond=nothing, reductionOperands, copyOperands, copyinOperands, copyinReadonlyOperands, copyoutOperands, copyoutZeroOperands, createOperands, createZeroOperands, noCreateOperands, presentOperands, devicePtrOperands, attachOperands, gangPrivateOperands, gangFirstPrivateOperands, asyncAttr=nothing, waitAttr=nothing, selfAttr=nothing, reductionOp=nothing, defaultAttr=nothing, region::Region, location=Location())
    results = IR.Type[]
    operands = Value[value.(waitOperands)..., value.(reductionOperands)..., value.(copyOperands)..., value.(copyinOperands)..., value.(copyinReadonlyOperands)..., value.(copyoutOperands)..., value.(copyoutZeroOperands)..., value.(createOperands)..., value.(createZeroOperands)..., value.(noCreateOperands)..., value.(presentOperands)..., value.(devicePtrOperands)..., value.(attachOperands)..., value.(gangPrivateOperands)..., value.(gangFirstPrivateOperands)..., ]
    owned_regions = Region[region, ]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(async) && push!(operands, value(async))
    !isnothing(numGangs) && push!(operands, value(numGangs))
    !isnothing(numWorkers) && push!(operands, value(numWorkers))
    !isnothing(vectorLength) && push!(operands, value(vectorLength))
    !isnothing(ifCond) && push!(operands, value(ifCond))
    !isnothing(selfCond) && push!(operands, value(selfCond))
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
function shutdown(deviceTypeOperands, deviceNumOperand=nothing; ifCond=nothing, location=Location())
    results = IR.Type[]
    operands = Value[value.(deviceTypeOperands)..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(deviceNumOperand) && push!(operands, value(deviceNumOperand))
    !isnothing(ifCond) && push!(operands, value(ifCond))
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
    _results = IR.Type[]
    _operands = Value[]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "acc.terminator",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
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
function update(ifCond=nothing; asyncOperand=nothing, waitDevnum=nothing, waitOperands, deviceTypeOperands, hostOperands, deviceOperands, async=nothing, wait=nothing, ifPresent=nothing, location=Location())
    results = IR.Type[]
    operands = Value[value.(waitOperands)..., value.(deviceTypeOperands)..., value.(hostOperands)..., value.(deviceOperands)..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(ifCond) && push!(operands, value(ifCond))
    !isnothing(asyncOperand) && push!(operands, value(asyncOperand))
    !isnothing(waitDevnum) && push!(operands, value(waitDevnum))
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
function wait(waitOperands, asyncOperand=nothing; waitDevnum=nothing, ifCond=nothing, async=nothing, location=Location())
    results = IR.Type[]
    operands = Value[value.(waitOperands)..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(asyncOperand) && push!(operands, value(asyncOperand))
    !isnothing(waitDevnum) && push!(operands, value(waitDevnum))
    !isnothing(ifCond) && push!(operands, value(ifCond))
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
function yield(operands_; location=Location())
    results = IR.Type[]
    operands = Value[value.(operands_)..., ]
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
