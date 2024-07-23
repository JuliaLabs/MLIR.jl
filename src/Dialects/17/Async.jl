module async

import ...IR:
    IR, NamedAttribute, Value, Location, Block, Region, Attribute, context, IndexType
import ..Dialects: namedattribute, operandsegmentsizes

"""
`add_to_group`

The `async.add_to_group` adds an async token or value to the async group.
Returns the rank of the added element in the group. This rank is fixed
for the group lifetime.

# Example

```mlir
%0 = async.create_group %size : !async.group
%1 = ... : !async.token
%2 = async.add_to_group %1, %0 : !async.token
```
"""
function add_to_group(
    operand::Value, group::Value; rank=nothing::Union{Nothing,IR.Type}, location=Location()
)
    _results = IR.Type[]
    _operands = Value[operand, group]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(rank) && push!(_results, rank)

    return IR.create_operation(
        "async.add_to_group",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=(length(_results) == 0 ? nothing : _results),
        result_inference=(length(_results) == 0 ? true : false),
    )
end

"""
`await_all`

The `async.await_all` operation waits until all the tokens or values in the
group become ready.

# Example

```mlir
%0 = async.create_group %size : !async.group

%1 = ... : !async.token
%2 = async.add_to_group %1, %0 : !async.token

%3 = ... : !async.token
%4 = async.add_to_group %2, %0 : !async.token

async.await_all %0
```
"""
function await_all(operand::Value; location=Location())
    _results = IR.Type[]
    _operands = Value[operand,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "async.await_all",
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
`await`

The `async.await` operation waits until the argument becomes ready, and for
the `async.value` arguments it unwraps the underlying value

# Example

```mlir
%0 = ... : !async.token
async.await %0 : !async.token

%1 = ... : !async.value<f32>
%2 = async.await %1 : !async.value<f32>
```
"""
function await(operand::Value; result=nothing::Union{Nothing,IR.Type}, location=Location())
    _results = IR.Type[]
    _operands = Value[operand,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(result) && push!(_results, result)

    return IR.create_operation(
        "async.await",
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
`call`

The `async.call` operation represents a direct call to an async function
that is within the same symbol scope as the call. The operands and result
types of the call must match the specified async function type. The callee
is encoded as a symbol reference attribute named \"callee\".

# Example

```mlir
%2 = async.call @my_add(%0, %1) : (f32, f32) -> !async.value<f32>
```
"""
function call(
    operands::Vector{Value}; result_0::Vector{IR.Type}, callee, location=Location()
)
    _results = IR.Type[result_0...,]
    _operands = Value[operands...,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[namedattribute("callee", callee),]

    return IR.create_operation(
        "async.call",
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
`coro_begin`

The `async.coro.begin` allocates a coroutine frame and returns a handle to
the coroutine.
"""
function coro_begin(id::Value; handle=nothing::Union{Nothing,IR.Type}, location=Location())
    _results = IR.Type[]
    _operands = Value[id,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(handle) && push!(_results, handle)

    return IR.create_operation(
        "async.coro.begin",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=(length(_results) == 0 ? nothing : _results),
        result_inference=(length(_results) == 0 ? true : false),
    )
end

"""
`coro_end`

The `async.coro.end` marks the point where a coroutine needs to return
control back to the caller if it is not an initial invocation of the
coroutine. It the start part of the coroutine is is no-op.
"""
function coro_end(handle::Value; location=Location())
    _results = IR.Type[]
    _operands = Value[handle,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "async.coro.end",
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
`coro_free`

The `async.coro.free` deallocates the coroutine frame created by the
async.coro.begin operation.
"""
function coro_free(id::Value, handle::Value; location=Location())
    _results = IR.Type[]
    _operands = Value[id, handle]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "async.coro.free",
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
`coro_id`

The `async.coro.id` returns a switched-resume coroutine identifier.
"""
function coro_id(; id=nothing::Union{Nothing,IR.Type}, location=Location())
    _results = IR.Type[]
    _operands = Value[]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(id) && push!(_results, id)

    return IR.create_operation(
        "async.coro.id",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=(length(_results) == 0 ? nothing : _results),
        result_inference=(length(_results) == 0 ? true : false),
    )
end

"""
`coro_save`

The `async.coro.saves` saves the coroutine state.
"""
function coro_save(
    handle::Value; state=nothing::Union{Nothing,IR.Type}, location=Location()
)
    _results = IR.Type[]
    _operands = Value[handle,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(state) && push!(_results, state)

    return IR.create_operation(
        "async.coro.save",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=(length(_results) == 0 ? nothing : _results),
        result_inference=(length(_results) == 0 ? true : false),
    )
end

"""
`coro_suspend`

The `async.coro.suspend` suspends the coroutine and transfers control to the
`suspend` successor. If suspended coroutine later resumed it will transfer
control to the `resume` successor. If it is destroyed it will transfer
control to the the `cleanup` successor.

In switched-resume lowering coroutine can be already in resumed state when
suspend operation is called, in this case control will be transferred to the
`resume` successor skipping the `suspend` successor.
"""
function coro_suspend(
    state::Value;
    suspendDest::Block,
    resumeDest::Block,
    cleanupDest::Block,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[state,]
    _owned_regions = Region[]
    _successors = Block[suspendDest, resumeDest, cleanupDest]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "async.coro.suspend",
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
`create_group`

The `async.create_group` allocates an empty async group. Async tokens or
values can be added to this group later. The size of the group must be
specified at construction time, and `await_all` operation will first
wait until the number of added tokens or values reaches the group size.

# Example

```mlir
%size = ... : index
%group = async.create_group %size : !async.group
...
async.await_all %group
```
"""
function create_group(
    size::Value; result=nothing::Union{Nothing,IR.Type}, location=Location()
)
    _results = IR.Type[]
    _operands = Value[size,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(result) && push!(_results, result)

    return IR.create_operation(
        "async.create_group",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=(length(_results) == 0 ? nothing : _results),
        result_inference=(length(_results) == 0 ? true : false),
    )
end

"""
`execute`

The `body` region attached to the `async.execute` operation semantically
can be executed concurrently with the successor operation. In the followup
example \"compute0\" can be executed concurrently with \"compute1\".

The actual concurrency semantics depends on the dialect lowering to the
executable format. Fully sequential execution (\"compute0\" completes before
\"compute1\" starts) is a completely legal execution.

Because concurrent execution is not guaranteed, it is illegal to create an
implicit dependency from \"compute1\" to \"compute0\" (e.g. via shared global
state). All dependencies must be made explicit with async execute arguments
(`async.token` or `async.value`).

   `async.execute` operation takes `async.token` dependencies and `async.value`
operands separately, and starts execution of the attached body region only
when all tokens and values become ready.

# Example

```mlir
%dependency = ... : !async.token
%value = ... : !async.value<f32>

%token, %results =
  async.execute [%dependency](%value as %unwrapped: !async.value<f32>)
             -> !async.value<!some.type>
  {
    %0 = \"compute0\"(%unwrapped): (f32) -> !some.type
    async.yield %0 : !some.type
  }

%1 = \"compute1\"(...) : !some.type
```

In the example above asynchronous execution starts only after dependency
token and value argument become ready. Unwrapped value passed to the
attached body region as an %unwrapped value of f32 type.
"""
function execute(
    dependencies::Vector{Value},
    bodyOperands::Vector{Value};
    token::IR.Type,
    bodyResults::Vector{IR.Type},
    bodyRegion::Region,
    location=Location(),
)
    _results = IR.Type[token, bodyResults...]
    _operands = Value[dependencies..., bodyOperands...]
    _owned_regions = Region[bodyRegion,]
    _successors = Block[]
    _attributes = NamedAttribute[]
    push!(attributes, operandsegmentsizes([length(dependencies), length(bodyOperands)]))

    return IR.create_operation(
        "async.execute",
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
`func`

An async function is like a normal function, but supports non-blocking
await. Internally, async function is lowered to the LLVM coroutinue with
async runtime intrinsic. It can return an async token and/or async values.
The token represents the execution state of async function and can be used
when users want to express dependencies on some side effects, e.g.,
the token becomes available once every thing in the func body is executed.

# Example

```mlir
// Async function can\'t return void, it always must be some async thing.
async.func @async.0() -> !async.token {
  return
}

// Function returns only async value.
async.func @async.1() -> !async.value<i32> {
  %0 = arith.constant 42 : i32
  return %0 : i32
}

// Implicit token can be added to return types.
async.func @async.2() -> !async.token, !async.value<i32> {
  %0 = arith.constant 42 : i32
  return %0 : i32
}
```
"""
function func(;
    sym_name,
    function_type,
    sym_visibility=nothing,
    arg_attrs=nothing,
    res_attrs=nothing,
    body::Region,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[]
    _owned_regions = Region[body,]
    _successors = Block[]
    _attributes = NamedAttribute[
        namedattribute("sym_name", sym_name), namedattribute("function_type", function_type)
    ]
    !isnothing(sym_visibility) &&
        push!(attributes, namedattribute("sym_visibility", sym_visibility))
    !isnothing(arg_attrs) && push!(attributes, namedattribute("arg_attrs", arg_attrs))
    !isnothing(res_attrs) && push!(attributes, namedattribute("res_attrs", res_attrs))

    return IR.create_operation(
        "async.func",
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
`return_`

The `async.return` is a special terminator operation for Async function.

# Example

```mlir
async.func @foo() : !async.token {
  return
}
```
"""
function return_(operands::Vector{Value}; location=Location())
    _results = IR.Type[]
    _operands = Value[operands...,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "async.return",
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
`runtime_add_ref`

The `async.runtime.add_ref` operation adds a reference(s) to async value
(token, value or group).
"""
function runtime_add_ref(operand::Value; count, location=Location())
    _results = IR.Type[]
    _operands = Value[operand,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[namedattribute("count", count),]

    return IR.create_operation(
        "async.runtime.add_ref",
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
`runtime_add_to_group`

The `async.runtime.add_to_group` adds an async token or value to the async
group. Returns the rank of the added element in the group.
"""
function runtime_add_to_group(
    operand::Value, group::Value; rank=nothing::Union{Nothing,IR.Type}, location=Location()
)
    _results = IR.Type[]
    _operands = Value[operand, group]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(rank) && push!(_results, rank)

    return IR.create_operation(
        "async.runtime.add_to_group",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=(length(_results) == 0 ? nothing : _results),
        result_inference=(length(_results) == 0 ? true : false),
    )
end

"""
`runtime_await_and_resume`

The `async.runtime.await_and_resume` operation awaits for the operand to
become available or error and resumes the coroutine on a thread managed by
the runtime.
"""
function runtime_await_and_resume(operand::Value, handle::Value; location=Location())
    _results = IR.Type[]
    _operands = Value[operand, handle]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "async.runtime.await_and_resume",
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
`runtime_await`

The `async.runtime.await` operation blocks the caller thread until the
operand becomes available or error.
"""
function runtime_await(operand::Value; location=Location())
    _results = IR.Type[]
    _operands = Value[operand,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "async.runtime.await",
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
`runtime_create_group`

The `async.runtime.create_group` operation creates an async dialect group
of the given size. Group created in the empty state.
"""
function runtime_create_group(
    size::Value; result=nothing::Union{Nothing,IR.Type}, location=Location()
)
    _results = IR.Type[]
    _operands = Value[size,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(result) && push!(_results, result)

    return IR.create_operation(
        "async.runtime.create_group",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=(length(_results) == 0 ? nothing : _results),
        result_inference=(length(_results) == 0 ? true : false),
    )
end

"""
`runtime_create`

The `async.runtime.create` operation creates an async dialect token or
value. Tokens and values are created in the non-ready state.
"""
function runtime_create(; result::IR.Type, location=Location())
    _results = IR.Type[result,]
    _operands = Value[]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "async.runtime.create",
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
`runtime_drop_ref`

The `async.runtime.drop_ref` operation drops a reference(s) to async value
(token, value or group).
"""
function runtime_drop_ref(operand::Value; count, location=Location())
    _results = IR.Type[]
    _operands = Value[operand,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[namedattribute("count", count),]

    return IR.create_operation(
        "async.runtime.drop_ref",
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
`runtime_is_error`

The `async.runtime.is_error` operation returns true if the token, value or
group (any of the async runtime values) is in the error state. It is the
caller responsibility to check error state after the call to `await` or
resuming after `await_and_resume`.
"""
function runtime_is_error(
    operand::Value; is_error=nothing::Union{Nothing,IR.Type}, location=Location()
)
    _results = IR.Type[]
    _operands = Value[operand,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(is_error) && push!(_results, is_error)

    return IR.create_operation(
        "async.runtime.is_error",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=(length(_results) == 0 ? nothing : _results),
        result_inference=(length(_results) == 0 ? true : false),
    )
end

"""
`runtime_load`

The `async.runtime.load` operation loads the value from the runtime
async.value storage.
"""
function runtime_load(storage::Value; result::IR.Type, location=Location())
    _results = IR.Type[result,]
    _operands = Value[storage,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "async.runtime.load",
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
`runtime_num_worker_threads`

The `async.runtime.num_worker_threads` operation gets the number of threads
in the threadpool from the runtime.
"""
function runtime_num_worker_threads(;
    result=nothing::Union{Nothing,IR.Type}, location=Location()
)
    _results = IR.Type[]
    _operands = Value[]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(result) && push!(_results, result)

    return IR.create_operation(
        "async.runtime.num_worker_threads",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=(length(_results) == 0 ? nothing : _results),
        result_inference=(length(_results) == 0 ? true : false),
    )
end

"""
`runtime_resume`

The `async.runtime.resume` operation resumes the coroutine on a thread
managed by the runtime.
"""
function runtime_resume(handle::Value; location=Location())
    _results = IR.Type[]
    _operands = Value[handle,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "async.runtime.resume",
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
`runtime_set_available`

The `async.runtime.set_available` operation switches async token or value
state to available.
"""
function runtime_set_available(operand::Value; location=Location())
    _results = IR.Type[]
    _operands = Value[operand,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "async.runtime.set_available",
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
`runtime_set_error`

The `async.runtime.set_error` operation switches async token or value
state to error.
"""
function runtime_set_error(operand::Value; location=Location())
    _results = IR.Type[]
    _operands = Value[operand,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "async.runtime.set_error",
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
`runtime_store`

The `async.runtime.store` operation stores the value into the runtime
async.value storage.
"""
function runtime_store(value::Value, storage::Value; location=Location())
    _results = IR.Type[]
    _operands = Value[value, storage]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "async.runtime.store",
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
`yield`

The `async.yield` is a special terminator operation for the block inside
`async.execute` operation.
"""
function yield(operands::Vector{Value}; location=Location())
    _results = IR.Type[]
    _operands = Value[operands...,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "async.yield",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

end # async
