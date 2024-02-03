module acc

import ...IR: NamedAttribute, MLIRType, get_value, Location, Block, Region, Attribute, create_operation, context, IndexType
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
function data(ifCond=nothing; copyOperands, copyinOperands, copyinReadonlyOperands, copyoutOperands, copyoutZeroOperands, createOperands, createZeroOperands, noCreateOperands, presentOperands, deviceptrOperands, attachOperands, defaultAttr=nothing, region::Region, location=Location())
    results = MLIRType[]
    operands = API.MlirValue[get_value.(copyOperands)..., get_value.(copyinOperands)..., get_value.(copyinReadonlyOperands)..., get_value.(copyoutOperands)..., get_value.(copyoutZeroOperands)..., get_value.(createOperands)..., get_value.(createZeroOperands)..., get_value.(noCreateOperands)..., get_value.(presentOperands)..., get_value.(deviceptrOperands)..., get_value.(attachOperands)..., ]
    owned_regions = Region[region, ]
    successors = Block[]
    attributes = NamedAttribute[]
    (ifCond != nothing) && push!(operands, get_valueifCond)
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
function enter_data(ifCond=nothing; asyncOperand=nothing, waitDevnum=nothing, waitOperands, copyinOperands, createOperands, createZeroOperands, attachOperands, async=nothing, wait=nothing, location=Location())
    results = MLIRType[]
    operands = API.MlirValue[get_value.(waitOperands)..., get_value.(copyinOperands)..., get_value.(createOperands)..., get_value.(createZeroOperands)..., get_value.(attachOperands)..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (ifCond != nothing) && push!(operands, get_valueifCond)
    (asyncOperand != nothing) && push!(operands, get_valueasyncOperand)
    (waitDevnum != nothing) && push!(operands, get_valuewaitDevnum)
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
function exit_data(ifCond=nothing; asyncOperand=nothing, waitDevnum=nothing, waitOperands, copyoutOperands, deleteOperands, detachOperands, async=nothing, wait=nothing, finalize=nothing, location=Location())
    results = MLIRType[]
    operands = API.MlirValue[get_value.(waitOperands)..., get_value.(copyoutOperands)..., get_value.(deleteOperands)..., get_value.(detachOperands)..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (ifCond != nothing) && push!(operands, get_valueifCond)
    (asyncOperand != nothing) && push!(operands, get_valueasyncOperand)
    (waitDevnum != nothing) && push!(operands, get_valuewaitDevnum)
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
function init(deviceTypeOperands, deviceNumOperand=nothing; ifCond=nothing, location=Location())
    results = MLIRType[]
    operands = API.MlirValue[get_value.(deviceTypeOperands)..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (deviceNumOperand != nothing) && push!(operands, get_valuedeviceNumOperand)
    (ifCond != nothing) && push!(operands, get_valueifCond)
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
function loop(gangNum=nothing; gangStatic=nothing, workerNum=nothing, vectorLength=nothing, tileOperands, privateOperands, reductionOperands, results::Vector{MLIRType}, collapse=nothing, seq=nothing, independent=nothing, auto_=nothing, reductionOp=nothing, exec_mapping=nothing, region::Region, location=Location())
    results = MLIRType[results..., ]
    operands = API.MlirValue[get_value.(tileOperands)..., get_value.(privateOperands)..., get_value.(reductionOperands)..., ]
    owned_regions = Region[region, ]
    successors = Block[]
    attributes = NamedAttribute[]
    (gangNum != nothing) && push!(operands, get_valuegangNum)
    (gangStatic != nothing) && push!(operands, get_valuegangStatic)
    (workerNum != nothing) && push!(operands, get_valueworkerNum)
    (vectorLength != nothing) && push!(operands, get_valuevectorLength)
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
function parallel(async=nothing; waitOperands, numGangs=nothing, numWorkers=nothing, vectorLength=nothing, ifCond=nothing, selfCond=nothing, reductionOperands, copyOperands, copyinOperands, copyinReadonlyOperands, copyoutOperands, copyoutZeroOperands, createOperands, createZeroOperands, noCreateOperands, presentOperands, devicePtrOperands, attachOperands, gangPrivateOperands, gangFirstPrivateOperands, asyncAttr=nothing, waitAttr=nothing, selfAttr=nothing, reductionOp=nothing, defaultAttr=nothing, region::Region, location=Location())
    results = MLIRType[]
    operands = API.MlirValue[get_value.(waitOperands)..., get_value.(reductionOperands)..., get_value.(copyOperands)..., get_value.(copyinOperands)..., get_value.(copyinReadonlyOperands)..., get_value.(copyoutOperands)..., get_value.(copyoutZeroOperands)..., get_value.(createOperands)..., get_value.(createZeroOperands)..., get_value.(noCreateOperands)..., get_value.(presentOperands)..., get_value.(devicePtrOperands)..., get_value.(attachOperands)..., get_value.(gangPrivateOperands)..., get_value.(gangFirstPrivateOperands)..., ]
    owned_regions = Region[region, ]
    successors = Block[]
    attributes = NamedAttribute[]
    (async != nothing) && push!(operands, get_valueasync)
    (numGangs != nothing) && push!(operands, get_valuenumGangs)
    (numWorkers != nothing) && push!(operands, get_valuenumWorkers)
    (vectorLength != nothing) && push!(operands, get_valuevectorLength)
    (ifCond != nothing) && push!(operands, get_valueifCond)
    (selfCond != nothing) && push!(operands, get_valueselfCond)
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
function shutdown(deviceTypeOperands, deviceNumOperand=nothing; ifCond=nothing, location=Location())
    results = MLIRType[]
    operands = API.MlirValue[get_value.(deviceTypeOperands)..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (deviceNumOperand != nothing) && push!(operands, get_valuedeviceNumOperand)
    (ifCond != nothing) && push!(operands, get_valueifCond)
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
    operands = API.MlirValue[]
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
function update(ifCond=nothing; asyncOperand=nothing, waitDevnum=nothing, waitOperands, deviceTypeOperands, hostOperands, deviceOperands, async=nothing, wait=nothing, ifPresent=nothing, location=Location())
    results = MLIRType[]
    operands = API.MlirValue[get_value.(waitOperands)..., get_value.(deviceTypeOperands)..., get_value.(hostOperands)..., get_value.(deviceOperands)..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (ifCond != nothing) && push!(operands, get_valueifCond)
    (asyncOperand != nothing) && push!(operands, get_valueasyncOperand)
    (waitDevnum != nothing) && push!(operands, get_valuewaitDevnum)
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
function wait(waitOperands, asyncOperand=nothing; waitDevnum=nothing, ifCond=nothing, async=nothing, location=Location())
    results = MLIRType[]
    operands = API.MlirValue[get_value.(waitOperands)..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (asyncOperand != nothing) && push!(operands, get_valueasyncOperand)
    (waitDevnum != nothing) && push!(operands, get_valuewaitDevnum)
    (ifCond != nothing) && push!(operands, get_valueifCond)
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
function yield(operands; location=Location())
    results = MLIRType[]
    operands = API.MlirValue[get_value.(operands)..., ]
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
