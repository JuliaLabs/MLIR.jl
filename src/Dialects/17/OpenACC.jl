module acc

import ...IR: IR, NamedAttribute, Value, Location, Block, Region, Attribute, context, IndexType
import ..Dialects: namedattribute, operandsegmentsizes


"""
`attach`

Description of arguments:
- `varPtr`: The address of variable to copy.
- `varPtrPtr`: Specifies the address of varPtr - only used when the variable
copied is a field in a struct. This is important for OpenACC due to implicit
attach semantics on data clauses (2.6.4).
- `bounds`: Used when copying just slice of array or array\'s bounds are not
encoded in type. They are in rank order where rank 0 is inner-most dimension.
- `dataClause`: Keeps track of the data clause the user used. This is because
the acc operations are decomposed. So a \'copy\' clause is decomposed to both 
`acc.copyin` and `acc.copyout` operations, but both have dataClause that
specifies `acc_copy` in this field.
- `structured`: Flag to note whether this is associated with structured region
(parallel, kernels, data) or unstructured (enter data, exit data). This is
important due to spec specifically calling out structured and dynamic reference
counters (2.6.7).
- `implicit`: Whether this is an implicitly generated operation, such as copies
done to satisfy \"Variables with Implicitly Determined Data Attributes\" in 2.6.2.
- `name`: Holds the name of variable as specified in user clause (including bounds).
"""
function attach(varPtr::Value, varPtrPtr=nothing::Union{Nothing, Value}; bounds::Vector{Value}, accPtr::IR.Type, dataClause=nothing, structured=nothing, implicit=nothing, name=nothing, location=Location())
    results = IR.Type[accPtr, ]
    operands = Value[varPtr, bounds..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(varPtrPtr) && push!(operands, varPtrPtr)
    push!(attributes, operandsegmentsizes([1, (varPtrPtr==nothing) ? 0 : 1length(bounds), ]))
    !isnothing(dataClause) && push!(attributes, namedattribute("dataClause", dataClause))
    !isnothing(structured) && push!(attributes, namedattribute("structured", structured))
    !isnothing(implicit) && push!(attributes, namedattribute("implicit", implicit))
    !isnothing(name) && push!(attributes, namedattribute("name", name))
    
    IR.create_operation(
        "acc.attach", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`copyin`

Description of arguments:
- `varPtr`: The address of variable to copy.
- `varPtrPtr`: Specifies the address of varPtr - only used when the variable
copied is a field in a struct. This is important for OpenACC due to implicit
attach semantics on data clauses (2.6.4).
- `bounds`: Used when copying just slice of array or array\'s bounds are not
encoded in type. They are in rank order where rank 0 is inner-most dimension.
- `dataClause`: Keeps track of the data clause the user used. This is because
the acc operations are decomposed. So a \'copy\' clause is decomposed to both 
`acc.copyin` and `acc.copyout` operations, but both have dataClause that
specifies `acc_copy` in this field.
- `structured`: Flag to note whether this is associated with structured region
(parallel, kernels, data) or unstructured (enter data, exit data). This is
important due to spec specifically calling out structured and dynamic reference
counters (2.6.7).
- `implicit`: Whether this is an implicitly generated operation, such as copies
done to satisfy \"Variables with Implicitly Determined Data Attributes\" in 2.6.2.
- `name`: Holds the name of variable as specified in user clause (including bounds).
"""
function copyin(varPtr::Value, varPtrPtr=nothing::Union{Nothing, Value}; bounds::Vector{Value}, accPtr::IR.Type, dataClause=nothing, structured=nothing, implicit=nothing, name=nothing, location=Location())
    results = IR.Type[accPtr, ]
    operands = Value[varPtr, bounds..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(varPtrPtr) && push!(operands, varPtrPtr)
    push!(attributes, operandsegmentsizes([1, (varPtrPtr==nothing) ? 0 : 1length(bounds), ]))
    !isnothing(dataClause) && push!(attributes, namedattribute("dataClause", dataClause))
    !isnothing(structured) && push!(attributes, namedattribute("structured", structured))
    !isnothing(implicit) && push!(attributes, namedattribute("implicit", implicit))
    !isnothing(name) && push!(attributes, namedattribute("name", name))
    
    IR.create_operation(
        "acc.copyin", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`copyout`

- `varPtr`: The address of variable to copy back to. This only applies to
`acc.copyout`
- `accPtr`: The acc address of variable. This is the link from the data-entry
operation used.
- `bounds`: Used when copying just slice of array or array\'s bounds are not
encoded in type. They are in rank order where rank 0 is inner-most dimension.
- `dataClause`: Keeps track of the data clause the user used. This is because
the acc operations are decomposed. So a \'copy\' clause is decomposed to both 
`acc.copyin` and `acc.copyout` operations, but both have dataClause that
specifies `acc_copy` in this field.
- `structured`: Flag to note whether this is associated with structured region
(parallel, kernels, data) or unstructured (enter data, exit data). This is
important due to spec specifically calling out structured and dynamic reference
counters (2.6.7).
- `implicit`: Whether this is an implicitly generated operation, such as copies
done to satisfy \"Variables with Implicitly Determined Data Attributes\" in 2.6.2.
- `name`: Holds the name of variable as specified in user clause (including bounds).
"""
function copyout(accPtr::Value, varPtr=nothing::Union{Nothing, Value}; bounds::Vector{Value}, dataClause=nothing, structured=nothing, implicit=nothing, name=nothing, location=Location())
    results = IR.Type[]
    operands = Value[accPtr, bounds..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(varPtr) && push!(operands, varPtr)
    push!(attributes, operandsegmentsizes([1, (varPtr==nothing) ? 0 : 1length(bounds), ]))
    !isnothing(dataClause) && push!(attributes, namedattribute("dataClause", dataClause))
    !isnothing(structured) && push!(attributes, namedattribute("structured", structured))
    !isnothing(implicit) && push!(attributes, namedattribute("implicit", implicit))
    !isnothing(name) && push!(attributes, namedattribute("name", name))
    
    IR.create_operation(
        "acc.copyout", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`create`

Description of arguments:
- `varPtr`: The address of variable to copy.
- `varPtrPtr`: Specifies the address of varPtr - only used when the variable
copied is a field in a struct. This is important for OpenACC due to implicit
attach semantics on data clauses (2.6.4).
- `bounds`: Used when copying just slice of array or array\'s bounds are not
encoded in type. They are in rank order where rank 0 is inner-most dimension.
- `dataClause`: Keeps track of the data clause the user used. This is because
the acc operations are decomposed. So a \'copy\' clause is decomposed to both 
`acc.copyin` and `acc.copyout` operations, but both have dataClause that
specifies `acc_copy` in this field.
- `structured`: Flag to note whether this is associated with structured region
(parallel, kernels, data) or unstructured (enter data, exit data). This is
important due to spec specifically calling out structured and dynamic reference
counters (2.6.7).
- `implicit`: Whether this is an implicitly generated operation, such as copies
done to satisfy \"Variables with Implicitly Determined Data Attributes\" in 2.6.2.
- `name`: Holds the name of variable as specified in user clause (including bounds).
"""
function create(varPtr::Value, varPtrPtr=nothing::Union{Nothing, Value}; bounds::Vector{Value}, accPtr::IR.Type, dataClause=nothing, structured=nothing, implicit=nothing, name=nothing, location=Location())
    results = IR.Type[accPtr, ]
    operands = Value[varPtr, bounds..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(varPtrPtr) && push!(operands, varPtrPtr)
    push!(attributes, operandsegmentsizes([1, (varPtrPtr==nothing) ? 0 : 1length(bounds), ]))
    !isnothing(dataClause) && push!(attributes, namedattribute("dataClause", dataClause))
    !isnothing(structured) && push!(attributes, namedattribute("structured", structured))
    !isnothing(implicit) && push!(attributes, namedattribute("implicit", implicit))
    !isnothing(name) && push!(attributes, namedattribute("name", name))
    
    IR.create_operation(
        "acc.create", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`bounds`

This operation is used to record bounds used in acc data clause in a
normalized fashion (zero-based). This works well with the `PointerLikeType`
requirement in data clauses - since a `lowerbound` of 0 means looking
at data at the zero offset from pointer.

The operation must have an `upperbound` or `extent` (or both are allowed -
but not checked for consistency). When the source language\'s arrays are
not zero-based, the `startIdx` must specify the zero-position index.

Examples below show copying a slice of 10-element array except first element.
Note that the examples use extent in data clause for C++ and upperbound
for Fortran (as per 2.7.1). To simplify examples, the constants are used
directly in the acc.bounds operands - this is not the syntax of operation.

C++:
```
int array[10];
#pragma acc copy(array[1:9])
```
=>
```mlir
acc.bounds lb(1) ub(9) extent(9) startIdx(0)
```

Fortran:
```
integer :: array(1:10)
!\$acc copy(array(2:10))
```
=>
```mlir
acc.bounds lb(1) ub(9) extent(9) startIdx(1)
```
"""
function bounds(lowerbound=nothing::Union{Nothing, Value}; upperbound=nothing::Union{Nothing, Value}, extent=nothing::Union{Nothing, Value}, stride=nothing::Union{Nothing, Value}, startIdx=nothing::Union{Nothing, Value}, result::IR.Type, strideInBytes=nothing, location=Location())
    results = IR.Type[result, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(lowerbound) && push!(operands, lowerbound)
    !isnothing(upperbound) && push!(operands, upperbound)
    !isnothing(extent) && push!(operands, extent)
    !isnothing(stride) && push!(operands, stride)
    !isnothing(startIdx) && push!(operands, startIdx)
    push!(attributes, operandsegmentsizes([(lowerbound==nothing) ? 0 : 1(upperbound==nothing) ? 0 : 1(extent==nothing) ? 0 : 1(stride==nothing) ? 0 : 1(startIdx==nothing) ? 0 : 1]))
    !isnothing(strideInBytes) && push!(attributes, namedattribute("strideInBytes", strideInBytes))
    
    IR.create_operation(
        "acc.bounds", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

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
function data(ifCond=nothing::Union{Nothing, Value}; async=nothing::Union{Nothing, Value}, waitDevnum=nothing::Union{Nothing, Value}, waitOperands::Vector{Value}, dataClauseOperands::Vector{Value}, asyncAttr=nothing, waitAttr=nothing, defaultAttr=nothing, region::Region, location=Location())
    results = IR.Type[]
    operands = Value[waitOperands..., dataClauseOperands..., ]
    owned_regions = Region[region, ]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(ifCond) && push!(operands, ifCond)
    !isnothing(async) && push!(operands, async)
    !isnothing(waitDevnum) && push!(operands, waitDevnum)
    push!(attributes, operandsegmentsizes([(ifCond==nothing) ? 0 : 1(async==nothing) ? 0 : 1(waitDevnum==nothing) ? 0 : 1length(waitOperands), length(dataClauseOperands), ]))
    !isnothing(asyncAttr) && push!(attributes, namedattribute("asyncAttr", asyncAttr))
    !isnothing(waitAttr) && push!(attributes, namedattribute("waitAttr", waitAttr))
    !isnothing(defaultAttr) && push!(attributes, namedattribute("defaultAttr", defaultAttr))
    
    IR.create_operation(
        "acc.data", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`declare_device_resident`

Description of arguments:
- `varPtr`: The address of variable to copy.
- `varPtrPtr`: Specifies the address of varPtr - only used when the variable
copied is a field in a struct. This is important for OpenACC due to implicit
attach semantics on data clauses (2.6.4).
- `bounds`: Used when copying just slice of array or array\'s bounds are not
encoded in type. They are in rank order where rank 0 is inner-most dimension.
- `dataClause`: Keeps track of the data clause the user used. This is because
the acc operations are decomposed. So a \'copy\' clause is decomposed to both 
`acc.copyin` and `acc.copyout` operations, but both have dataClause that
specifies `acc_copy` in this field.
- `structured`: Flag to note whether this is associated with structured region
(parallel, kernels, data) or unstructured (enter data, exit data). This is
important due to spec specifically calling out structured and dynamic reference
counters (2.6.7).
- `implicit`: Whether this is an implicitly generated operation, such as copies
done to satisfy \"Variables with Implicitly Determined Data Attributes\" in 2.6.2.
- `name`: Holds the name of variable as specified in user clause (including bounds).
"""
function declare_device_resident(varPtr::Value, varPtrPtr=nothing::Union{Nothing, Value}; bounds::Vector{Value}, accPtr::IR.Type, dataClause=nothing, structured=nothing, implicit=nothing, name=nothing, location=Location())
    results = IR.Type[accPtr, ]
    operands = Value[varPtr, bounds..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(varPtrPtr) && push!(operands, varPtrPtr)
    push!(attributes, operandsegmentsizes([1, (varPtrPtr==nothing) ? 0 : 1length(bounds), ]))
    !isnothing(dataClause) && push!(attributes, namedattribute("dataClause", dataClause))
    !isnothing(structured) && push!(attributes, namedattribute("structured", structured))
    !isnothing(implicit) && push!(attributes, namedattribute("implicit", implicit))
    !isnothing(name) && push!(attributes, namedattribute("name", name))
    
    IR.create_operation(
        "acc.declare_device_resident", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`declare_enter`

The \"acc.declare_enter\" operation represents the OpenACC declare directive
and captures the entry semantics to the implicit data region.
This operation is modeled similarly to \"acc.enter_data\".

Example showing `acc declare create(a)`:

```mlir
%0 = acc.create varPtr(%a : !llvm.ptr<f32>) -> !llvm.ptr<f32>
acc.declare_enter dataOperands(%0 : !llvm.ptr<f32>)
```
"""
function declare_enter(dataClauseOperands::Vector{Value}; location=Location())
    results = IR.Type[]
    operands = Value[dataClauseOperands..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "acc.declare_enter", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`declare_exit`

The \"acc.declare_exit\" operation represents the OpenACC declare directive
and captures the exit semantics from the implicit data region.
This operation is modeled similarly to \"acc.exit_data\".

Example showing `acc declare device_resident(a)`:

```mlir
%0 = acc.getdeviceptr varPtr(%a : !llvm.ptr<f32>) -> !llvm.ptr<f32> {dataClause = #acc<data_clause declare_device_resident>}
acc.declare_exit dataOperands(%0 : !llvm.ptr<f32>)
acc.delete accPtr(%0 : !llvm.ptr<f32>) {dataClause = #acc<data_clause declare_device_resident>}
```
"""
function declare_exit(dataClauseOperands::Vector{Value}; location=Location())
    results = IR.Type[]
    operands = Value[dataClauseOperands..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "acc.declare_exit", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`declare_link`

Description of arguments:
- `varPtr`: The address of variable to copy.
- `varPtrPtr`: Specifies the address of varPtr - only used when the variable
copied is a field in a struct. This is important for OpenACC due to implicit
attach semantics on data clauses (2.6.4).
- `bounds`: Used when copying just slice of array or array\'s bounds are not
encoded in type. They are in rank order where rank 0 is inner-most dimension.
- `dataClause`: Keeps track of the data clause the user used. This is because
the acc operations are decomposed. So a \'copy\' clause is decomposed to both 
`acc.copyin` and `acc.copyout` operations, but both have dataClause that
specifies `acc_copy` in this field.
- `structured`: Flag to note whether this is associated with structured region
(parallel, kernels, data) or unstructured (enter data, exit data). This is
important due to spec specifically calling out structured and dynamic reference
counters (2.6.7).
- `implicit`: Whether this is an implicitly generated operation, such as copies
done to satisfy \"Variables with Implicitly Determined Data Attributes\" in 2.6.2.
- `name`: Holds the name of variable as specified in user clause (including bounds).
"""
function declare_link(varPtr::Value, varPtrPtr=nothing::Union{Nothing, Value}; bounds::Vector{Value}, accPtr::IR.Type, dataClause=nothing, structured=nothing, implicit=nothing, name=nothing, location=Location())
    results = IR.Type[accPtr, ]
    operands = Value[varPtr, bounds..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(varPtrPtr) && push!(operands, varPtrPtr)
    push!(attributes, operandsegmentsizes([1, (varPtrPtr==nothing) ? 0 : 1length(bounds), ]))
    !isnothing(dataClause) && push!(attributes, namedattribute("dataClause", dataClause))
    !isnothing(structured) && push!(attributes, namedattribute("structured", structured))
    !isnothing(implicit) && push!(attributes, namedattribute("implicit", implicit))
    !isnothing(name) && push!(attributes, namedattribute("name", name))
    
    IR.create_operation(
        "acc.declare_link", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`delete`

- `varPtr`: The address of variable to copy back to. This only applies to
`acc.copyout`
- `accPtr`: The acc address of variable. This is the link from the data-entry
operation used.
- `bounds`: Used when copying just slice of array or array\'s bounds are not
encoded in type. They are in rank order where rank 0 is inner-most dimension.
- `dataClause`: Keeps track of the data clause the user used. This is because
the acc operations are decomposed. So a \'copy\' clause is decomposed to both 
`acc.copyin` and `acc.copyout` operations, but both have dataClause that
specifies `acc_copy` in this field.
- `structured`: Flag to note whether this is associated with structured region
(parallel, kernels, data) or unstructured (enter data, exit data). This is
important due to spec specifically calling out structured and dynamic reference
counters (2.6.7).
- `implicit`: Whether this is an implicitly generated operation, such as copies
done to satisfy \"Variables with Implicitly Determined Data Attributes\" in 2.6.2.
- `name`: Holds the name of variable as specified in user clause (including bounds).
"""
function delete(accPtr::Value, varPtr=nothing::Union{Nothing, Value}; bounds::Vector{Value}, dataClause=nothing, structured=nothing, implicit=nothing, name=nothing, location=Location())
    results = IR.Type[]
    operands = Value[accPtr, bounds..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(varPtr) && push!(operands, varPtr)
    push!(attributes, operandsegmentsizes([1, (varPtr==nothing) ? 0 : 1length(bounds), ]))
    !isnothing(dataClause) && push!(attributes, namedattribute("dataClause", dataClause))
    !isnothing(structured) && push!(attributes, namedattribute("structured", structured))
    !isnothing(implicit) && push!(attributes, namedattribute("implicit", implicit))
    !isnothing(name) && push!(attributes, namedattribute("name", name))
    
    IR.create_operation(
        "acc.delete", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`detach`

- `varPtr`: The address of variable to copy back to. This only applies to
`acc.copyout`
- `accPtr`: The acc address of variable. This is the link from the data-entry
operation used.
- `bounds`: Used when copying just slice of array or array\'s bounds are not
encoded in type. They are in rank order where rank 0 is inner-most dimension.
- `dataClause`: Keeps track of the data clause the user used. This is because
the acc operations are decomposed. So a \'copy\' clause is decomposed to both 
`acc.copyin` and `acc.copyout` operations, but both have dataClause that
specifies `acc_copy` in this field.
- `structured`: Flag to note whether this is associated with structured region
(parallel, kernels, data) or unstructured (enter data, exit data). This is
important due to spec specifically calling out structured and dynamic reference
counters (2.6.7).
- `implicit`: Whether this is an implicitly generated operation, such as copies
done to satisfy \"Variables with Implicitly Determined Data Attributes\" in 2.6.2.
- `name`: Holds the name of variable as specified in user clause (including bounds).
"""
function detach(accPtr::Value, varPtr=nothing::Union{Nothing, Value}; bounds::Vector{Value}, dataClause=nothing, structured=nothing, implicit=nothing, name=nothing, location=Location())
    results = IR.Type[]
    operands = Value[accPtr, bounds..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(varPtr) && push!(operands, varPtr)
    push!(attributes, operandsegmentsizes([1, (varPtr==nothing) ? 0 : 1length(bounds), ]))
    !isnothing(dataClause) && push!(attributes, namedattribute("dataClause", dataClause))
    !isnothing(structured) && push!(attributes, namedattribute("structured", structured))
    !isnothing(implicit) && push!(attributes, namedattribute("implicit", implicit))
    !isnothing(name) && push!(attributes, namedattribute("name", name))
    
    IR.create_operation(
        "acc.detach", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`deviceptr`

Description of arguments:
- `varPtr`: The address of variable to copy.
- `varPtrPtr`: Specifies the address of varPtr - only used when the variable
copied is a field in a struct. This is important for OpenACC due to implicit
attach semantics on data clauses (2.6.4).
- `bounds`: Used when copying just slice of array or array\'s bounds are not
encoded in type. They are in rank order where rank 0 is inner-most dimension.
- `dataClause`: Keeps track of the data clause the user used. This is because
the acc operations are decomposed. So a \'copy\' clause is decomposed to both 
`acc.copyin` and `acc.copyout` operations, but both have dataClause that
specifies `acc_copy` in this field.
- `structured`: Flag to note whether this is associated with structured region
(parallel, kernels, data) or unstructured (enter data, exit data). This is
important due to spec specifically calling out structured and dynamic reference
counters (2.6.7).
- `implicit`: Whether this is an implicitly generated operation, such as copies
done to satisfy \"Variables with Implicitly Determined Data Attributes\" in 2.6.2.
- `name`: Holds the name of variable as specified in user clause (including bounds).
"""
function deviceptr(varPtr::Value, varPtrPtr=nothing::Union{Nothing, Value}; bounds::Vector{Value}, accPtr::IR.Type, dataClause=nothing, structured=nothing, implicit=nothing, name=nothing, location=Location())
    results = IR.Type[accPtr, ]
    operands = Value[varPtr, bounds..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(varPtrPtr) && push!(operands, varPtrPtr)
    push!(attributes, operandsegmentsizes([1, (varPtrPtr==nothing) ? 0 : 1length(bounds), ]))
    !isnothing(dataClause) && push!(attributes, namedattribute("dataClause", dataClause))
    !isnothing(structured) && push!(attributes, namedattribute("structured", structured))
    !isnothing(implicit) && push!(attributes, namedattribute("implicit", implicit))
    !isnothing(name) && push!(attributes, namedattribute("name", name))
    
    IR.create_operation(
        "acc.deviceptr", location;
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
function enter_data(ifCond=nothing::Union{Nothing, Value}; asyncOperand=nothing::Union{Nothing, Value}, waitDevnum=nothing::Union{Nothing, Value}, waitOperands::Vector{Value}, dataClauseOperands::Vector{Value}, async=nothing, wait=nothing, location=Location())
    results = IR.Type[]
    operands = Value[waitOperands..., dataClauseOperands..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(ifCond) && push!(operands, ifCond)
    !isnothing(asyncOperand) && push!(operands, asyncOperand)
    !isnothing(waitDevnum) && push!(operands, waitDevnum)
    push!(attributes, operandsegmentsizes([(ifCond==nothing) ? 0 : 1(asyncOperand==nothing) ? 0 : 1(waitDevnum==nothing) ? 0 : 1length(waitOperands), length(dataClauseOperands), ]))
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
function exit_data(ifCond=nothing::Union{Nothing, Value}; asyncOperand=nothing::Union{Nothing, Value}, waitDevnum=nothing::Union{Nothing, Value}, waitOperands::Vector{Value}, dataClauseOperands::Vector{Value}, async=nothing, wait=nothing, finalize=nothing, location=Location())
    results = IR.Type[]
    operands = Value[waitOperands..., dataClauseOperands..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(ifCond) && push!(operands, ifCond)
    !isnothing(asyncOperand) && push!(operands, asyncOperand)
    !isnothing(waitDevnum) && push!(operands, waitDevnum)
    push!(attributes, operandsegmentsizes([(ifCond==nothing) ? 0 : 1(asyncOperand==nothing) ? 0 : 1(waitDevnum==nothing) ? 0 : 1length(waitOperands), length(dataClauseOperands), ]))
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
`firstprivate`

Description of arguments:
- `varPtr`: The address of variable to copy.
- `varPtrPtr`: Specifies the address of varPtr - only used when the variable
copied is a field in a struct. This is important for OpenACC due to implicit
attach semantics on data clauses (2.6.4).
- `bounds`: Used when copying just slice of array or array\'s bounds are not
encoded in type. They are in rank order where rank 0 is inner-most dimension.
- `dataClause`: Keeps track of the data clause the user used. This is because
the acc operations are decomposed. So a \'copy\' clause is decomposed to both 
`acc.copyin` and `acc.copyout` operations, but both have dataClause that
specifies `acc_copy` in this field.
- `structured`: Flag to note whether this is associated with structured region
(parallel, kernels, data) or unstructured (enter data, exit data). This is
important due to spec specifically calling out structured and dynamic reference
counters (2.6.7).
- `implicit`: Whether this is an implicitly generated operation, such as copies
done to satisfy \"Variables with Implicitly Determined Data Attributes\" in 2.6.2.
- `name`: Holds the name of variable as specified in user clause (including bounds).
"""
function firstprivate(varPtr::Value, varPtrPtr=nothing::Union{Nothing, Value}; bounds::Vector{Value}, accPtr::IR.Type, dataClause=nothing, structured=nothing, implicit=nothing, name=nothing, location=Location())
    results = IR.Type[accPtr, ]
    operands = Value[varPtr, bounds..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(varPtrPtr) && push!(operands, varPtrPtr)
    push!(attributes, operandsegmentsizes([1, (varPtrPtr==nothing) ? 0 : 1length(bounds), ]))
    !isnothing(dataClause) && push!(attributes, namedattribute("dataClause", dataClause))
    !isnothing(structured) && push!(attributes, namedattribute("structured", structured))
    !isnothing(implicit) && push!(attributes, namedattribute("implicit", implicit))
    !isnothing(name) && push!(attributes, namedattribute("name", name))
    
    IR.create_operation(
        "acc.firstprivate", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`firstprivate_recipe`

Declares an OpenACC privatization recipe with copy of the initial value.
The operation requires two mandatory regions and one optional.

  1. The initializer region specifies how to allocate and initialize a new
     private value. For example in Fortran, a derived-type might have a
     default initialization. The region has an argument that contains the
     value that need to be privatized. This is useful if the type is not
     known at compile time and the private value is needed to create its
     copy.
  2. The copy region specifies how to copy the initial value to the newly
     created private value. It takes the initial value and the privatized
     value as arguments.
  3. The destroy region specifies how to destruct the value when it reaches
     its end of life. It takes the privatized value as argument. It is
     optional.

A single privatization recipe can be used for multiple operand if they have
the same type and do not require a specific default initialization.

# Example

```mlir
acc.firstprivate.recipe @privatization_f32 : f32 init {
^bb0(%0: f32):
  // init region contains a sequence of operations to create and
  // initialize the copy if needed. It yields the create copy.
} copy {
^bb0(%0: f32, %1: !llvm.ptr<f32>):
  // copy region contains a sequence of operations to copy the initial value
  // of the firstprivate value to the newly created value.
} destroy {
^bb0(%0: f32)
  // destroy region contains a sequences of operations to destruct the
  // created copy.
}

// The privatization symbol is then used in the corresponding operation.
acc.parallel firstprivate(@privatization_f32 -> %a : f32) {
}
```
"""
function firstprivate_recipe(; sym_name, type, initRegion::Region, copyRegion::Region, destroyRegion::Region, location=Location())
    results = IR.Type[]
    operands = Value[]
    owned_regions = Region[initRegion, copyRegion, destroyRegion, ]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("sym_name", sym_name), namedattribute("type", type), ]
    
    IR.create_operation(
        "acc.firstprivate.recipe", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`getdeviceptr`

This operation is used to get the `accPtr` for a variable. This is often
used in conjunction with data exit operations when the data entry
operation is not visible. This operation can have a `dataClause` argument
that is any of the valid `mlir::acc::DataClause` entries.
\\
    
    Description of arguments:
    - `varPtr`: The address of variable to copy.
    - `varPtrPtr`: Specifies the address of varPtr - only used when the variable
    copied is a field in a struct. This is important for OpenACC due to implicit
    attach semantics on data clauses (2.6.4).
    - `bounds`: Used when copying just slice of array or array\'s bounds are not
    encoded in type. They are in rank order where rank 0 is inner-most dimension.
    - `dataClause`: Keeps track of the data clause the user used. This is because
    the acc operations are decomposed. So a \'copy\' clause is decomposed to both 
    `acc.copyin` and `acc.copyout` operations, but both have dataClause that
    specifies `acc_copy` in this field.
    - `structured`: Flag to note whether this is associated with structured region
    (parallel, kernels, data) or unstructured (enter data, exit data). This is
    important due to spec specifically calling out structured and dynamic reference
    counters (2.6.7).
    - `implicit`: Whether this is an implicitly generated operation, such as copies
    done to satisfy \"Variables with Implicitly Determined Data Attributes\" in 2.6.2.
    - `name`: Holds the name of variable as specified in user clause (including bounds).
"""
function getdeviceptr(varPtr::Value, varPtrPtr=nothing::Union{Nothing, Value}; bounds::Vector{Value}, accPtr::IR.Type, dataClause=nothing, structured=nothing, implicit=nothing, name=nothing, location=Location())
    results = IR.Type[accPtr, ]
    operands = Value[varPtr, bounds..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(varPtrPtr) && push!(operands, varPtrPtr)
    push!(attributes, operandsegmentsizes([1, (varPtrPtr==nothing) ? 0 : 1length(bounds), ]))
    !isnothing(dataClause) && push!(attributes, namedattribute("dataClause", dataClause))
    !isnothing(structured) && push!(attributes, namedattribute("structured", structured))
    !isnothing(implicit) && push!(attributes, namedattribute("implicit", implicit))
    !isnothing(name) && push!(attributes, namedattribute("name", name))
    
    IR.create_operation(
        "acc.getdeviceptr", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`global_ctor`

The \"acc.global_ctor\" operation is used to capture OpenACC actions to apply
on globals (such as `acc declare`) at the entry to the implicit data region.
This operation is isolated and intended to be used in a module.

Example showing `declare create` of global:

```mlir
llvm.mlir.global external @globalvar() : i32 {
  %0 = llvm.mlir.constant(0 : i32) : i32
  llvm.return %0 : i32
}
acc.global_ctor @acc_constructor {
  %0 = llvm.mlir.addressof @globalvar : !llvm.ptr<i32>
  %1 = acc.create varPtr(%0 : !llvm.ptr<i32>) -> !llvm.ptr<i32>
  acc.declare_enter dataOperands(%1 : !llvm.ptr<i32>)
}
```
"""
function global_ctor(; sym_name, region::Region, location=Location())
    results = IR.Type[]
    operands = Value[]
    owned_regions = Region[region, ]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("sym_name", sym_name), ]
    
    IR.create_operation(
        "acc.global_ctor", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`global_dtor`

The \"acc.global_dtor\" operation is used to capture OpenACC actions to apply
on globals (such as `acc declare`) at the exit from the implicit data
region. This operation is isolated and intended to be used in a module.

Example showing delete associated with `declare create` of global:

```mlir
llvm.mlir.global external @globalvar() : i32 {
  %0 = llvm.mlir.constant(0 : i32) : i32
  llvm.return %0 : i32
}
acc.global_dtor @acc_destructor {
  %0 = llvm.mlir.addressof @globalvar : !llvm.ptr<i32>
  %1 = acc.getdeviceptr varPtr(%0 : !llvm.ptr<i32>) -> !llvm.ptr<i32> {dataClause = #acc<data_clause create>}
  acc.declare_exit dataOperands(%1 : !llvm.ptr<i32>)
  acc.delete accPtr(%1 : !llvm.ptr<i32>) {dataClause = #acc<data_clause create>}
}
```
"""
function global_dtor(; sym_name, region::Region, location=Location())
    results = IR.Type[]
    operands = Value[]
    owned_regions = Region[region, ]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("sym_name", sym_name), ]
    
    IR.create_operation(
        "acc.global_dtor", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`host_data`

The \"acc.host_data\" operation represents the OpenACC host_data construct.

# Example

```mlir
%0 = acc.use_device varPtr(%a : !llvm.ptr<f32>) -> !llvm.ptr<f32>
acc.host_data dataOperands(%0 : !llvm.ptr<f32>) {

}
```
"""
function host_data(ifCond=nothing::Union{Nothing, Value}; dataClauseOperands::Vector{Value}, ifPresent=nothing, region::Region, location=Location())
    results = IR.Type[]
    operands = Value[dataClauseOperands..., ]
    owned_regions = Region[region, ]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(ifCond) && push!(operands, ifCond)
    push!(attributes, operandsegmentsizes([(ifCond==nothing) ? 0 : 1length(dataClauseOperands), ]))
    !isnothing(ifPresent) && push!(attributes, namedattribute("ifPresent", ifPresent))
    
    IR.create_operation(
        "acc.host_data", location;
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
`kernels`

The \"acc.kernels\" operation represents a kernels construct block. It has
one region to be compiled into a sequence of kernels for execution on the
current device.

# Example

```mlir
acc.kernels num_gangs(%c10) num_workers(%c10)
    private(%c : memref<10xf32>) {
  // kernels region
}
```
"""
function kernels(async=nothing::Union{Nothing, Value}; waitOperands::Vector{Value}, numGangs::Vector{Value}, numWorkers=nothing::Union{Nothing, Value}, vectorLength=nothing::Union{Nothing, Value}, ifCond=nothing::Union{Nothing, Value}, selfCond=nothing::Union{Nothing, Value}, dataClauseOperands::Vector{Value}, asyncAttr=nothing, waitAttr=nothing, selfAttr=nothing, defaultAttr=nothing, region::Region, location=Location())
    results = IR.Type[]
    operands = Value[waitOperands..., numGangs..., dataClauseOperands..., ]
    owned_regions = Region[region, ]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(async) && push!(operands, async)
    !isnothing(numWorkers) && push!(operands, numWorkers)
    !isnothing(vectorLength) && push!(operands, vectorLength)
    !isnothing(ifCond) && push!(operands, ifCond)
    !isnothing(selfCond) && push!(operands, selfCond)
    push!(attributes, operandsegmentsizes([(async==nothing) ? 0 : 1length(waitOperands), length(numGangs), (numWorkers==nothing) ? 0 : 1(vectorLength==nothing) ? 0 : 1(ifCond==nothing) ? 0 : 1(selfCond==nothing) ? 0 : 1length(dataClauseOperands), ]))
    !isnothing(asyncAttr) && push!(attributes, namedattribute("asyncAttr", asyncAttr))
    !isnothing(waitAttr) && push!(attributes, namedattribute("waitAttr", waitAttr))
    !isnothing(selfAttr) && push!(attributes, namedattribute("selfAttr", selfAttr))
    !isnothing(defaultAttr) && push!(attributes, namedattribute("defaultAttr", defaultAttr))
    
    IR.create_operation(
        "acc.kernels", location;
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
function loop(gangNum=nothing::Union{Nothing, Value}; gangDim=nothing::Union{Nothing, Value}, gangStatic=nothing::Union{Nothing, Value}, workerNum=nothing::Union{Nothing, Value}, vectorLength=nothing::Union{Nothing, Value}, tileOperands::Vector{Value}, privateOperands::Vector{Value}, reductionOperands::Vector{Value}, results_::Vector{IR.Type}, collapse=nothing, seq=nothing, independent=nothing, auto_=nothing, hasGang=nothing, hasWorker=nothing, hasVector=nothing, privatizations=nothing, reductionRecipes=nothing, region::Region, location=Location())
    results = IR.Type[results_..., ]
    operands = Value[tileOperands..., privateOperands..., reductionOperands..., ]
    owned_regions = Region[region, ]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(gangNum) && push!(operands, gangNum)
    !isnothing(gangDim) && push!(operands, gangDim)
    !isnothing(gangStatic) && push!(operands, gangStatic)
    !isnothing(workerNum) && push!(operands, workerNum)
    !isnothing(vectorLength) && push!(operands, vectorLength)
    push!(attributes, operandsegmentsizes([(gangNum==nothing) ? 0 : 1(gangDim==nothing) ? 0 : 1(gangStatic==nothing) ? 0 : 1(workerNum==nothing) ? 0 : 1(vectorLength==nothing) ? 0 : 1length(tileOperands), length(privateOperands), length(reductionOperands), ]))
    !isnothing(collapse) && push!(attributes, namedattribute("collapse", collapse))
    !isnothing(seq) && push!(attributes, namedattribute("seq", seq))
    !isnothing(independent) && push!(attributes, namedattribute("independent", independent))
    !isnothing(auto_) && push!(attributes, namedattribute("auto_", auto_))
    !isnothing(hasGang) && push!(attributes, namedattribute("hasGang", hasGang))
    !isnothing(hasWorker) && push!(attributes, namedattribute("hasWorker", hasWorker))
    !isnothing(hasVector) && push!(attributes, namedattribute("hasVector", hasVector))
    !isnothing(privatizations) && push!(attributes, namedattribute("privatizations", privatizations))
    !isnothing(reductionRecipes) && push!(attributes, namedattribute("reductionRecipes", reductionRecipes))
    
    IR.create_operation(
        "acc.loop", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`nocreate`

Description of arguments:
- `varPtr`: The address of variable to copy.
- `varPtrPtr`: Specifies the address of varPtr - only used when the variable
copied is a field in a struct. This is important for OpenACC due to implicit
attach semantics on data clauses (2.6.4).
- `bounds`: Used when copying just slice of array or array\'s bounds are not
encoded in type. They are in rank order where rank 0 is inner-most dimension.
- `dataClause`: Keeps track of the data clause the user used. This is because
the acc operations are decomposed. So a \'copy\' clause is decomposed to both 
`acc.copyin` and `acc.copyout` operations, but both have dataClause that
specifies `acc_copy` in this field.
- `structured`: Flag to note whether this is associated with structured region
(parallel, kernels, data) or unstructured (enter data, exit data). This is
important due to spec specifically calling out structured and dynamic reference
counters (2.6.7).
- `implicit`: Whether this is an implicitly generated operation, such as copies
done to satisfy \"Variables with Implicitly Determined Data Attributes\" in 2.6.2.
- `name`: Holds the name of variable as specified in user clause (including bounds).
"""
function nocreate(varPtr::Value, varPtrPtr=nothing::Union{Nothing, Value}; bounds::Vector{Value}, accPtr::IR.Type, dataClause=nothing, structured=nothing, implicit=nothing, name=nothing, location=Location())
    results = IR.Type[accPtr, ]
    operands = Value[varPtr, bounds..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(varPtrPtr) && push!(operands, varPtrPtr)
    push!(attributes, operandsegmentsizes([1, (varPtrPtr==nothing) ? 0 : 1length(bounds), ]))
    !isnothing(dataClause) && push!(attributes, namedattribute("dataClause", dataClause))
    !isnothing(structured) && push!(attributes, namedattribute("structured", structured))
    !isnothing(implicit) && push!(attributes, namedattribute("implicit", implicit))
    !isnothing(name) && push!(attributes, namedattribute("name", name))
    
    IR.create_operation(
        "acc.nocreate", location;
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
function parallel(async=nothing::Union{Nothing, Value}; waitOperands::Vector{Value}, numGangs::Vector{Value}, numWorkers=nothing::Union{Nothing, Value}, vectorLength=nothing::Union{Nothing, Value}, ifCond=nothing::Union{Nothing, Value}, selfCond=nothing::Union{Nothing, Value}, reductionOperands::Vector{Value}, gangPrivateOperands::Vector{Value}, gangFirstPrivateOperands::Vector{Value}, dataClauseOperands::Vector{Value}, asyncAttr=nothing, waitAttr=nothing, selfAttr=nothing, reductionRecipes=nothing, privatizations=nothing, firstprivatizations=nothing, defaultAttr=nothing, region::Region, location=Location())
    results = IR.Type[]
    operands = Value[waitOperands..., numGangs..., reductionOperands..., gangPrivateOperands..., gangFirstPrivateOperands..., dataClauseOperands..., ]
    owned_regions = Region[region, ]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(async) && push!(operands, async)
    !isnothing(numWorkers) && push!(operands, numWorkers)
    !isnothing(vectorLength) && push!(operands, vectorLength)
    !isnothing(ifCond) && push!(operands, ifCond)
    !isnothing(selfCond) && push!(operands, selfCond)
    push!(attributes, operandsegmentsizes([(async==nothing) ? 0 : 1length(waitOperands), length(numGangs), (numWorkers==nothing) ? 0 : 1(vectorLength==nothing) ? 0 : 1(ifCond==nothing) ? 0 : 1(selfCond==nothing) ? 0 : 1length(reductionOperands), length(gangPrivateOperands), length(gangFirstPrivateOperands), length(dataClauseOperands), ]))
    !isnothing(asyncAttr) && push!(attributes, namedattribute("asyncAttr", asyncAttr))
    !isnothing(waitAttr) && push!(attributes, namedattribute("waitAttr", waitAttr))
    !isnothing(selfAttr) && push!(attributes, namedattribute("selfAttr", selfAttr))
    !isnothing(reductionRecipes) && push!(attributes, namedattribute("reductionRecipes", reductionRecipes))
    !isnothing(privatizations) && push!(attributes, namedattribute("privatizations", privatizations))
    !isnothing(firstprivatizations) && push!(attributes, namedattribute("firstprivatizations", firstprivatizations))
    !isnothing(defaultAttr) && push!(attributes, namedattribute("defaultAttr", defaultAttr))
    
    IR.create_operation(
        "acc.parallel", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`present`

Description of arguments:
- `varPtr`: The address of variable to copy.
- `varPtrPtr`: Specifies the address of varPtr - only used when the variable
copied is a field in a struct. This is important for OpenACC due to implicit
attach semantics on data clauses (2.6.4).
- `bounds`: Used when copying just slice of array or array\'s bounds are not
encoded in type. They are in rank order where rank 0 is inner-most dimension.
- `dataClause`: Keeps track of the data clause the user used. This is because
the acc operations are decomposed. So a \'copy\' clause is decomposed to both 
`acc.copyin` and `acc.copyout` operations, but both have dataClause that
specifies `acc_copy` in this field.
- `structured`: Flag to note whether this is associated with structured region
(parallel, kernels, data) or unstructured (enter data, exit data). This is
important due to spec specifically calling out structured and dynamic reference
counters (2.6.7).
- `implicit`: Whether this is an implicitly generated operation, such as copies
done to satisfy \"Variables with Implicitly Determined Data Attributes\" in 2.6.2.
- `name`: Holds the name of variable as specified in user clause (including bounds).
"""
function present(varPtr::Value, varPtrPtr=nothing::Union{Nothing, Value}; bounds::Vector{Value}, accPtr::IR.Type, dataClause=nothing, structured=nothing, implicit=nothing, name=nothing, location=Location())
    results = IR.Type[accPtr, ]
    operands = Value[varPtr, bounds..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(varPtrPtr) && push!(operands, varPtrPtr)
    push!(attributes, operandsegmentsizes([1, (varPtrPtr==nothing) ? 0 : 1length(bounds), ]))
    !isnothing(dataClause) && push!(attributes, namedattribute("dataClause", dataClause))
    !isnothing(structured) && push!(attributes, namedattribute("structured", structured))
    !isnothing(implicit) && push!(attributes, namedattribute("implicit", implicit))
    !isnothing(name) && push!(attributes, namedattribute("name", name))
    
    IR.create_operation(
        "acc.present", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`private`

Description of arguments:
- `varPtr`: The address of variable to copy.
- `varPtrPtr`: Specifies the address of varPtr - only used when the variable
copied is a field in a struct. This is important for OpenACC due to implicit
attach semantics on data clauses (2.6.4).
- `bounds`: Used when copying just slice of array or array\'s bounds are not
encoded in type. They are in rank order where rank 0 is inner-most dimension.
- `dataClause`: Keeps track of the data clause the user used. This is because
the acc operations are decomposed. So a \'copy\' clause is decomposed to both 
`acc.copyin` and `acc.copyout` operations, but both have dataClause that
specifies `acc_copy` in this field.
- `structured`: Flag to note whether this is associated with structured region
(parallel, kernels, data) or unstructured (enter data, exit data). This is
important due to spec specifically calling out structured and dynamic reference
counters (2.6.7).
- `implicit`: Whether this is an implicitly generated operation, such as copies
done to satisfy \"Variables with Implicitly Determined Data Attributes\" in 2.6.2.
- `name`: Holds the name of variable as specified in user clause (including bounds).
"""
function private(varPtr::Value, varPtrPtr=nothing::Union{Nothing, Value}; bounds::Vector{Value}, accPtr::IR.Type, dataClause=nothing, structured=nothing, implicit=nothing, name=nothing, location=Location())
    results = IR.Type[accPtr, ]
    operands = Value[varPtr, bounds..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(varPtrPtr) && push!(operands, varPtrPtr)
    push!(attributes, operandsegmentsizes([1, (varPtrPtr==nothing) ? 0 : 1length(bounds), ]))
    !isnothing(dataClause) && push!(attributes, namedattribute("dataClause", dataClause))
    !isnothing(structured) && push!(attributes, namedattribute("structured", structured))
    !isnothing(implicit) && push!(attributes, namedattribute("implicit", implicit))
    !isnothing(name) && push!(attributes, namedattribute("name", name))
    
    IR.create_operation(
        "acc.private", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`private_recipe`

Declares an OpenACC privatization recipe. The operation requires one
mandatory and one optional region.

  1. The initializer region specifies how to allocate and initialize a new
     private value. For example in Fortran, a derived-type might have a
     default initialization. The region has an argument that contains the
     value that need to be privatized. This is useful if the type is not
     known at compile time and the private value is needed to create its
     copy.
  2. The destroy region specifies how to destruct the value when it reaches
     its end of life. It takes the privatized value as argument.

A single privatization recipe can be used for multiple operand if they have
the same type and do not require a specific default initialization.

# Example

```mlir
acc.private.recipe @privatization_f32 : f32 init {
^bb0(%0: f32):
  // init region contains a sequence of operations to create and
  // initialize the copy if needed. It yields the create copy.
} destroy {
^bb0(%0: f32)
  // destroy region contains a sequences of operations to destruct the
  // created copy.
}

// The privatization symbol is then used in the corresponding operation.
acc.parallel private(@privatization_f32 -> %a : f32) {
}
```
"""
function private_recipe(; sym_name, type, initRegion::Region, destroyRegion::Region, location=Location())
    results = IR.Type[]
    operands = Value[]
    owned_regions = Region[initRegion, destroyRegion, ]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("sym_name", sym_name), namedattribute("type", type), ]
    
    IR.create_operation(
        "acc.private.recipe", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`reduction`

Description of arguments:
- `varPtr`: The address of variable to copy.
- `varPtrPtr`: Specifies the address of varPtr - only used when the variable
copied is a field in a struct. This is important for OpenACC due to implicit
attach semantics on data clauses (2.6.4).
- `bounds`: Used when copying just slice of array or array\'s bounds are not
encoded in type. They are in rank order where rank 0 is inner-most dimension.
- `dataClause`: Keeps track of the data clause the user used. This is because
the acc operations are decomposed. So a \'copy\' clause is decomposed to both 
`acc.copyin` and `acc.copyout` operations, but both have dataClause that
specifies `acc_copy` in this field.
- `structured`: Flag to note whether this is associated with structured region
(parallel, kernels, data) or unstructured (enter data, exit data). This is
important due to spec specifically calling out structured and dynamic reference
counters (2.6.7).
- `implicit`: Whether this is an implicitly generated operation, such as copies
done to satisfy \"Variables with Implicitly Determined Data Attributes\" in 2.6.2.
- `name`: Holds the name of variable as specified in user clause (including bounds).
"""
function reduction(varPtr::Value, varPtrPtr=nothing::Union{Nothing, Value}; bounds::Vector{Value}, accPtr::IR.Type, dataClause=nothing, structured=nothing, implicit=nothing, name=nothing, location=Location())
    results = IR.Type[accPtr, ]
    operands = Value[varPtr, bounds..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(varPtrPtr) && push!(operands, varPtrPtr)
    push!(attributes, operandsegmentsizes([1, (varPtrPtr==nothing) ? 0 : 1length(bounds), ]))
    !isnothing(dataClause) && push!(attributes, namedattribute("dataClause", dataClause))
    !isnothing(structured) && push!(attributes, namedattribute("structured", structured))
    !isnothing(implicit) && push!(attributes, namedattribute("implicit", implicit))
    !isnothing(name) && push!(attributes, namedattribute("name", name))
    
    IR.create_operation(
        "acc.reduction", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`reduction_recipe`

Declares an OpenACC reduction recipe. The operation requires two
mandatory regions.

  1. The initializer region specifies how to initialize the local reduction
     value. The region has a first argument that contains the value of the
     reduction accumulator at the start of the reduction. It is expected to
     `acc.yield` the new value. Extra arguments can be added to deal with
     dynamic arrays.
  2. The reduction region contains a sequences of operations to combine two
     values of the reduction type into one. It has at least two arguments
     and it is expected to `acc.yield` the combined value. Extra arguments
     can be added to deal with dynamic arrays.

# Example

```mlir
acc.reduction.recipe @reduction_add_i64 : i64 reduction_operator<add> init {
^bb0(%0: i64):
  // init region contains a sequence of operations to initialize the local
  // reduction value as specified in 2.5.15
  %c0 = arith.constant 0 : i64
  acc.yield %c0 : i64
} combiner {
^bb0(%0: i64, %1: i64)
  // combiner region contains a sequence of operations to combine
  // two values into one.
  %2 = arith.addi %0, %1 : i64
  acc.yield %2 : i64
}

// The reduction symbol is then used in the corresponding operation.
acc.parallel reduction(@reduction_add_i64 -> %a : i64) {
}
```

The following table lists the valid operators and the initialization values
according to OpenACC 3.3:

|------------------------------------------------|
|        C/C++          |        Fortran         |
|-----------------------|------------------------|
| operator | init value | operator | init value  |
|     +    |      0     |     +    |      0      |
|     *    |      1     |     *    |      1      |
|    max   |    least   |    max   |    least    |
|    min   |   largest  |    min   |   largest   |
|     &    |     ~0     |   iand   | all bits on |
|     |    |      0     |    ior   |      0      |
|     ^    |      0     |   ieor   |      0      |
|    &&    |      1     |   .and.  |    .true.   |
|    ||    |      0     |    .or.  |   .false.   |
|          |            |   .eqv.  |    .true.   |
|          |            |  .neqv.  |   .false.   |
-------------------------------------------------|
"""
function reduction_recipe(; sym_name, type, reductionOperator, initRegion::Region, combinerRegion::Region, location=Location())
    results = IR.Type[]
    operands = Value[]
    owned_regions = Region[initRegion, combinerRegion, ]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("sym_name", sym_name), namedattribute("type", type), namedattribute("reductionOperator", reductionOperator), ]
    
    IR.create_operation(
        "acc.reduction.recipe", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`serial`

The \"acc.serial\" operation represents a serial construct block. It has
one region to be executed in serial on the current device.

# Example

```mlir
acc.serial private(%c : memref<10xf32>) {
  // serial region
}
```
"""
function serial(async=nothing::Union{Nothing, Value}; waitOperands::Vector{Value}, ifCond=nothing::Union{Nothing, Value}, selfCond=nothing::Union{Nothing, Value}, reductionOperands::Vector{Value}, gangPrivateOperands::Vector{Value}, gangFirstPrivateOperands::Vector{Value}, dataClauseOperands::Vector{Value}, asyncAttr=nothing, waitAttr=nothing, selfAttr=nothing, reductionRecipes=nothing, privatizations=nothing, firstprivatizations=nothing, defaultAttr=nothing, region::Region, location=Location())
    results = IR.Type[]
    operands = Value[waitOperands..., reductionOperands..., gangPrivateOperands..., gangFirstPrivateOperands..., dataClauseOperands..., ]
    owned_regions = Region[region, ]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(async) && push!(operands, async)
    !isnothing(ifCond) && push!(operands, ifCond)
    !isnothing(selfCond) && push!(operands, selfCond)
    push!(attributes, operandsegmentsizes([(async==nothing) ? 0 : 1length(waitOperands), (ifCond==nothing) ? 0 : 1(selfCond==nothing) ? 0 : 1length(reductionOperands), length(gangPrivateOperands), length(gangFirstPrivateOperands), length(dataClauseOperands), ]))
    !isnothing(asyncAttr) && push!(attributes, namedattribute("asyncAttr", asyncAttr))
    !isnothing(waitAttr) && push!(attributes, namedattribute("waitAttr", waitAttr))
    !isnothing(selfAttr) && push!(attributes, namedattribute("selfAttr", selfAttr))
    !isnothing(reductionRecipes) && push!(attributes, namedattribute("reductionRecipes", reductionRecipes))
    !isnothing(privatizations) && push!(attributes, namedattribute("privatizations", privatizations))
    !isnothing(firstprivatizations) && push!(attributes, namedattribute("firstprivatizations", firstprivatizations))
    !isnothing(defaultAttr) && push!(attributes, namedattribute("defaultAttr", defaultAttr))
    
    IR.create_operation(
        "acc.serial", location;
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
`update_device`

Description of arguments:
- `varPtr`: The address of variable to copy.
- `varPtrPtr`: Specifies the address of varPtr - only used when the variable
copied is a field in a struct. This is important for OpenACC due to implicit
attach semantics on data clauses (2.6.4).
- `bounds`: Used when copying just slice of array or array\'s bounds are not
encoded in type. They are in rank order where rank 0 is inner-most dimension.
- `dataClause`: Keeps track of the data clause the user used. This is because
the acc operations are decomposed. So a \'copy\' clause is decomposed to both 
`acc.copyin` and `acc.copyout` operations, but both have dataClause that
specifies `acc_copy` in this field.
- `structured`: Flag to note whether this is associated with structured region
(parallel, kernels, data) or unstructured (enter data, exit data). This is
important due to spec specifically calling out structured and dynamic reference
counters (2.6.7).
- `implicit`: Whether this is an implicitly generated operation, such as copies
done to satisfy \"Variables with Implicitly Determined Data Attributes\" in 2.6.2.
- `name`: Holds the name of variable as specified in user clause (including bounds).
"""
function update_device(varPtr::Value, varPtrPtr=nothing::Union{Nothing, Value}; bounds::Vector{Value}, accPtr::IR.Type, dataClause=nothing, structured=nothing, implicit=nothing, name=nothing, location=Location())
    results = IR.Type[accPtr, ]
    operands = Value[varPtr, bounds..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(varPtrPtr) && push!(operands, varPtrPtr)
    push!(attributes, operandsegmentsizes([1, (varPtrPtr==nothing) ? 0 : 1length(bounds), ]))
    !isnothing(dataClause) && push!(attributes, namedattribute("dataClause", dataClause))
    !isnothing(structured) && push!(attributes, namedattribute("structured", structured))
    !isnothing(implicit) && push!(attributes, namedattribute("implicit", implicit))
    !isnothing(name) && push!(attributes, namedattribute("name", name))
    
    IR.create_operation(
        "acc.update_device", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`update_host`

- `varPtr`: The address of variable to copy back to. This only applies to
`acc.copyout`
- `accPtr`: The acc address of variable. This is the link from the data-entry
operation used.
- `bounds`: Used when copying just slice of array or array\'s bounds are not
encoded in type. They are in rank order where rank 0 is inner-most dimension.
- `dataClause`: Keeps track of the data clause the user used. This is because
the acc operations are decomposed. So a \'copy\' clause is decomposed to both 
`acc.copyin` and `acc.copyout` operations, but both have dataClause that
specifies `acc_copy` in this field.
- `structured`: Flag to note whether this is associated with structured region
(parallel, kernels, data) or unstructured (enter data, exit data). This is
important due to spec specifically calling out structured and dynamic reference
counters (2.6.7).
- `implicit`: Whether this is an implicitly generated operation, such as copies
done to satisfy \"Variables with Implicitly Determined Data Attributes\" in 2.6.2.
- `name`: Holds the name of variable as specified in user clause (including bounds).
"""
function update_host(accPtr::Value, varPtr=nothing::Union{Nothing, Value}; bounds::Vector{Value}, dataClause=nothing, structured=nothing, implicit=nothing, name=nothing, location=Location())
    results = IR.Type[]
    operands = Value[accPtr, bounds..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(varPtr) && push!(operands, varPtr)
    push!(attributes, operandsegmentsizes([1, (varPtr==nothing) ? 0 : 1length(bounds), ]))
    !isnothing(dataClause) && push!(attributes, namedattribute("dataClause", dataClause))
    !isnothing(structured) && push!(attributes, namedattribute("structured", structured))
    !isnothing(implicit) && push!(attributes, namedattribute("implicit", implicit))
    !isnothing(name) && push!(attributes, namedattribute("name", name))
    
    IR.create_operation(
        "acc.update_host", location;
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
function update(ifCond=nothing::Union{Nothing, Value}; asyncOperand=nothing::Union{Nothing, Value}, waitDevnum=nothing::Union{Nothing, Value}, waitOperands::Vector{Value}, deviceTypeOperands::Vector{Value}, dataClauseOperands::Vector{Value}, async=nothing, wait=nothing, ifPresent=nothing, location=Location())
    results = IR.Type[]
    operands = Value[waitOperands..., deviceTypeOperands..., dataClauseOperands..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(ifCond) && push!(operands, ifCond)
    !isnothing(asyncOperand) && push!(operands, asyncOperand)
    !isnothing(waitDevnum) && push!(operands, waitDevnum)
    push!(attributes, operandsegmentsizes([(ifCond==nothing) ? 0 : 1(asyncOperand==nothing) ? 0 : 1(waitDevnum==nothing) ? 0 : 1length(waitOperands), length(deviceTypeOperands), length(dataClauseOperands), ]))
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
`use_device`

Description of arguments:
- `varPtr`: The address of variable to copy.
- `varPtrPtr`: Specifies the address of varPtr - only used when the variable
copied is a field in a struct. This is important for OpenACC due to implicit
attach semantics on data clauses (2.6.4).
- `bounds`: Used when copying just slice of array or array\'s bounds are not
encoded in type. They are in rank order where rank 0 is inner-most dimension.
- `dataClause`: Keeps track of the data clause the user used. This is because
the acc operations are decomposed. So a \'copy\' clause is decomposed to both 
`acc.copyin` and `acc.copyout` operations, but both have dataClause that
specifies `acc_copy` in this field.
- `structured`: Flag to note whether this is associated with structured region
(parallel, kernels, data) or unstructured (enter data, exit data). This is
important due to spec specifically calling out structured and dynamic reference
counters (2.6.7).
- `implicit`: Whether this is an implicitly generated operation, such as copies
done to satisfy \"Variables with Implicitly Determined Data Attributes\" in 2.6.2.
- `name`: Holds the name of variable as specified in user clause (including bounds).
"""
function use_device(varPtr::Value, varPtrPtr=nothing::Union{Nothing, Value}; bounds::Vector{Value}, accPtr::IR.Type, dataClause=nothing, structured=nothing, implicit=nothing, name=nothing, location=Location())
    results = IR.Type[accPtr, ]
    operands = Value[varPtr, bounds..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(varPtrPtr) && push!(operands, varPtrPtr)
    push!(attributes, operandsegmentsizes([1, (varPtrPtr==nothing) ? 0 : 1length(bounds), ]))
    !isnothing(dataClause) && push!(attributes, namedattribute("dataClause", dataClause))
    !isnothing(structured) && push!(attributes, namedattribute("structured", structured))
    !isnothing(implicit) && push!(attributes, namedattribute("implicit", implicit))
    !isnothing(name) && push!(attributes, namedattribute("name", name))
    
    IR.create_operation(
        "acc.use_device", location;
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
