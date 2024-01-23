export
    Operation,
    OperationState,
    Location,
    Context,
    MModule,
    Value,
    MLIRType,
    Region,
    Block,
    Attribute,
    NamedAttribute

import Base: ==, String
using .API:
    MlirDialectRegistry,
    MlirDialectHandle,
    MlirAttribute,
    MlirNamedAttribute,
    MlirDialect,
    MlirStringRef,
    MlirOperation,
    MlirOperationState,
    MlirLocation,
    MlirBlock,
    MlirRegion,
    MlirModule,
    MlirContext,
    MlirType,
    MlirValue,
    MlirIdentifier,
    MlirPassManager,
    MlirOpPassManager

function print_callback(str::MlirStringRef, userdata)
    data = unsafe_wrap(Array, Base.convert(Ptr{Cchar}, str.data), str.length; own=false)
    write(userdata isa Base.RefValue ? userdata[] : userdata, data)
    return Cvoid()
end

### Dialect

struct Dialect
    dialect::MlirDialect

    Dialect(dialect) = begin
        @assert !mlirIsNull(dialect) "cannot create Dialect from null MlirDialect"
        new(dialect)
    end
end

Base.convert(::Type{MlirDialect}, dialect::Dialect) = dialect.dialect
function Base.show(io::IO, dialect::Dialect)
    print(io, "Dialect(\"", String(API.mlirDialectGetNamespace(dialect)), "\")")
end

### DialectHandle

struct DialectHandle
    handle::API.MlirDialectHandle
end

function DialectHandle(s::Symbol)
    s = Symbol("mlirGetDialectHandle__", s, "__")
    DialectHandle(getproperty(API, s)())
end

Base.convert(::Type{MlirDialectHandle}, handle::DialectHandle) = handle.handle

### Dialect Registry

mutable struct DialectRegistry
    registry::MlirDialectRegistry
end
function DialectRegistry()
    registry = API.mlirDialectRegistryCreate()
    @assert !mlirIsNull(registry) "cannot create DialectRegistry with null MlirDialectRegistry"
    finalizer(DialectRegistry(registry)) do registry
        API.mlirDialectRegistryDestroy(registry.registry)
    end
end

function Base.insert!(registry::DialectRegistry, handle::DialectHandle)
    API.mlirDialectHandleInsertDialect(registry, handle)
end

### Context

struct Context
    context::MlirContext
end

function Context()
    context = API.mlirContextCreate()
    @assert !mlirIsNull(context) "cannot create Context with null MlirContext"
    context = Context(context)
    activate(context)
    context
end

function dispose(ctx::Context)
    deactivate(ctx)
    API.mlirContextDestroy(ctx.context)
end

function Context(f::Core.Function)
    ctx = Context()
    try
        f(ctx)
    finally
        dispose(ctx)
    end
end

Base.convert(::Type{MlirContext}, c::Context) = c.context

num_loaded_dialects() = API.mlirContextGetNumLoadedDialects(context())
function get_or_load_dialect!(handle::DialectHandle)
    mlir_dialect = API.mlirDialectHandleLoadDialect(handle, context())
    if mlirIsNull(mlir_dialect)
        error("could not load dialect from handle $handle")
    else
        Dialect(mlir_dialect)
    end
end
function get_or_load_dialect!(dialect::String)
    get_or_load_dialect!(DialectHandle(Symbol(dialect)))
end

function enable_multithreading!(enable=true)
    API.mlirContextEnableMultithreading(context(), enable)
    context()
end

is_registered_operation(opname) = API.mlirContextIsRegisteredOperation(context(), opname)

### Location

struct Location
    location::MlirLocation

    Location(location) = begin
        @assert !mlirIsNull(location) "cannot create Location with null MlirLocation"
        new(location)
    end
end

Location() = Location(API.mlirLocationUnknownGet(context()))
Location(filename, line, column) =
    Location(API.mlirLocationFileLineColGet(context(), filename, line, column))

Base.convert(::Type{MlirLocation}, location::Location) = location.location

function Base.show(io::IO, location::Location)
    c_print_callback = @cfunction(print_callback, Cvoid, (MlirStringRef, Any))
    ref = Ref(io)
    print(io, "Location(#= ")
    API.mlirLocationPrint(location, c_print_callback, ref)
    print(io, " =#)")
end

### Type

struct MLIRType
    type::MlirType

    MLIRType(type) = begin
        @assert !mlirIsNull(type)
        new(type)
    end
end

MLIRType(t::MLIRType) = t
MLIRType(T::Type{<:Signed}) =
    MLIRType(API.mlirIntegerTypeGet(context(), sizeof(T) * 8))
MLIRType(T::Type{<:Unsigned}) =
    MLIRType(API.mlirIntegerTypeGet(context(), sizeof(T) * 8))
MLIRType(::Type{Bool}) =
    MLIRType(API.mlirIntegerTypeGet(context(), 1))
MLIRType(::Type{Float32}) =
    MLIRType(API.mlirF32TypeGet(context()))
MLIRType(::Type{Float64}) =
    MLIRType(API.mlirF64TypeGet(context()))
MLIRType(ft::Pair) =
    MLIRType(API.mlirFunctionTypeGet(context(),
        length(ft.first), [MLIRType(t) for t in ft.first],
        length(ft.second), [MLIRType(t) for t in ft.second]))
MLIRType(a::AbstractArray{T}) where {T} = MLIRType(MLIRType(T), size(a))
MLIRType(::Type{<:AbstractArray{T,N}}, dims) where {T,N} =
    MLIRType(API.mlirRankedTensorTypeGetChecked(
        Location(),
        N, collect(dims),
        MLIRType(T),
        Attribute(),
    ))
MLIRType(element_type::MLIRType, dims) =
    MLIRType(API.mlirRankedTensorTypeGetChecked(
        Location(),
        length(dims), collect(dims),
        element_type,
        Attribute(),
    ))
MLIRType(::T) where {T<:Real} = MLIRType(T)
MLIRType(_, type::MLIRType) = type

IndexType() = MLIRType(API.mlirIndexTypeGet(context()))

Base.convert(::Type{MlirType}, mtype::MLIRType) = mtype.type
Base.parse(::Type{MLIRType}, s) =
    MLIRType(API.mlirTypeParseGet(context(), s))

function Base.eltype(type::MLIRType)
    if API.mlirTypeIsAShaped(type)
        MLIRType(API.mlirShapedTypeGetElementType(type))
    else
        type
    end
end

function Base.show(io::IO, type::MLIRType)
    print(io, "MLIRType(#= ")
    c_print_callback = @cfunction(print_callback, Cvoid, (MlirStringRef, Any))
    ref = Ref(io)
    API.mlirTypePrint(type, c_print_callback, ref)
    print(io, " =#)")
end

function inttype(size, issigned)
    size == 1 && issigned && return Bool
    ints = (Int8, Int16, Int32, Int64, Int128)
    IT = ints[Int(log2(size))-2]
    issigned ? IT : unsigned(IT)
end

function julia_type(type::MLIRType)
    if API.mlirTypeIsAInteger(type)
        is_signed = API.mlirIntegerTypeIsSigned(type) ||
                    API.mlirIntegerTypeIsSignless(type)
        width = API.mlirIntegerTypeGetWidth(type)

        try
            inttype(width, is_signed)
        catch
            t = is_signed ? "i" : "u"
            throw("could not convert type $(t)$(width) to julia")
        end
    elseif API.mlirTypeIsAF32(type)
        Float32
    elseif API.mlirTypeIsAF64(type)
        Float64
    else
        throw("could not convert type $type to julia")
    end
end

Base.ndims(type::MLIRType) =
    if API.mlirTypeIsAShaped(type) && API.mlirShapedTypeHasRank(type)
        API.mlirShapedTypeGetRank(type)
    else
        0
    end

Base.size(type::MLIRType, i::Int) = API.mlirShapedTypeGetDimSize(type, i - 1)
Base.size(type::MLIRType) = Tuple(size(type, i) for i in 1:ndims(type))

function is_tensor(type::MLIRType)
    API.mlirTypeIsAShaped(type)
end

function is_integer(type::MLIRType)
    API.mlirTypeIsAInteger(type)
end

is_function_type(mtype) = API.mlirTypeIsAFunction(mtype)

function num_inputs(ftype::MLIRType)
    @assert is_function_type(ftype) "cannot get the number of inputs on type $(ftype), expected a function type"
    API.mlirFunctionTypeGetNumInputs(ftype)
end
function num_results(ftype::MLIRType)
    @assert is_function_type(ftype) "cannot get the number of results on type $(ftype), expected a function type"
    API.mlirFunctionTypeGetNumResults(ftype)
end

function get_input(ftype::MLIRType, pos)
    @assert is_function_type(ftype) "cannot get input on type $(ftype), expected a function type"
    MLIRType(API.mlirFunctionTypeGetInput(ftype, pos - 1))
end
function get_result(ftype::MLIRType, pos=1)
    @assert is_function_type(ftype) "cannot get result on type $(ftype), expected a function type"
    MLIRType(API.mlirFunctionTypeGetResult(ftype, pos - 1))
end

### Attribute

struct Attribute
    attribute::MlirAttribute
end

Attribute() = Attribute(API.mlirAttributeGetNull())
Attribute(s::AbstractString) = Attribute(API.mlirStringAttrGet(context(), s))
Attribute(type::MLIRType) = Attribute(API.mlirTypeAttrGet(type))
Attribute(f::F, type=MLIRType(F)) where {F<:AbstractFloat} = Attribute(
    API.mlirFloatAttrDoubleGet(context(), type, Float64(f))
)
Attribute(i::T) where {T<:Integer} = Attribute(
    API.mlirIntegerAttrGet(MLIRType(T), Int64(i))
)
function Attribute(values::T) where {T<:AbstractArray{Int32}}
    type = MLIRType(T, size(values))
    Attribute(
        API.mlirDenseElementsAttrInt32Get(type, length(values), values)
    )
end
function Attribute(values::T) where {T<:AbstractArray{Int64}}
    type = MLIRType(T, size(values))
    Attribute(
        API.mlirDenseElementsAttrInt64Get(type, length(values), values)
    )
end
function Attribute(values::T) where {T<:AbstractArray{Float64}}
    type = MLIRType(T, size(values))
    Attribute(
        API.mlirDenseElementsAttrDoubleGet(type, length(values), values)
    )
end
function Attribute(values::T) where {T<:AbstractArray{Float32}}
    type = MLIRType(T, size(values))
    Attribute(
        API.mlirDenseElementsAttrFloatGet(type, length(values), values)
    )
end
function Attribute(values::AbstractArray{Int32}, type)
    Attribute(
        API.mlirDenseElementsAttrInt32Get(type, length(values), values)
    )
end
function Attribute(values::AbstractArray{Int}, type)
    Attribute(
        API.mlirDenseElementsAttrInt64Get(type, length(values), values)
    )
end
function Attribute(values::AbstractArray{Float32}, type)
    Attribute(
        API.mlirDenseElementsAttrFloatGet(type, length(values), values)
    )
end
function ArrayAttribute(values::AbstractVector{Int})
    elements = Attribute.(values)
    Attribute(
        API.mlirArrayAttrGet(context(), length(elements), elements)
    )
end
function ArrayAttribute(attributes::Vector{Attribute})
    Attribute(
        API.mlirArrayAttrGet(context(), length(attributes), attributes),
    )
end
function DenseArrayAttribute(values::AbstractVector{Int32})
    Attribute(
        API.mlirDenseI32ArrayGet(context(), length(values), collect(values))
    )
end
function DenseArrayAttribute(values::AbstractVector{Int})
    Attribute(
        API.mlirDenseI64ArrayGet(context(), length(values), collect(values))
    )
end
function Attribute(value::Int, type::MLIRType)
    Attribute(
        API.mlirIntegerAttrGet(type, value)
    )
end
function Attribute(value::Bool)
    Attribute(
        API.mlirBoolAttrGet(context(), value)
    )
end

Base.convert(::Type{MlirAttribute}, attribute::Attribute) = attribute.attribute
Base.parse(::Type{Attribute}, s) =
    Attribute(API.mlirAttributeParseGet(context(), s))

function get_type(attribute::Attribute)
    MLIRType(API.mlirAttributeGetType(attribute))
end
function type_value(attribute)
    @assert API.mlirAttributeIsAType(attribute) "attribute $(attribute) is not a type"
    MLIRType(API.mlirTypeAttrGetValue(attribute))
end
function bool_value(attribute)
    @assert API.mlirAttributeIsABool(attribute) "attribute $(attribute) is not a boolean"
    API.mlirBoolAttrGetValue(attribute)
end
function string_value(attribute)
    @assert API.mlirAttributeIsAString(attribute) "attribute $(attribute) is not a string attribute"
    String(API.mlirStringAttrGetValue(attribute))
end

function Base.show(io::IO, attribute::Attribute)
    print(io, "Attribute(#= ")
    c_print_callback = @cfunction(print_callback, Cvoid, (MlirStringRef, Any))
    ref = Ref(io)
    API.mlirAttributePrint(attribute, c_print_callback, ref)
    print(io, " =#)")
end

### Named Attribute

struct NamedAttribute
    named_attribute::MlirNamedAttribute
end

function NamedAttribute(name, attribute)
    @assert !mlirIsNull(attribute.attribute)
    NamedAttribute(API.mlirNamedAttributeGet(
        API.mlirIdentifierGet(context(), name),
        attribute
    ))
end

Base.convert(::Type{MlirAttribute}, named_attribute::NamedAttribute) =
    named_attribute.named_attribute

### Value

struct Value
    value::MlirValue

    Value(value) = begin
        @assert !mlirIsNull(value) "cannot create Value with null MlirValue"
        new(value)
    end
end

get_type(value) = MLIRType(API.mlirValueGetType(value))

Base.convert(::Type{MlirValue}, value::Value) = value.value
Base.size(value::Value) = Base.size(get_type(value))
Base.ndims(value::Value) = Base.ndims(get_type(value))

function Base.show(io::IO, value::Value)
    c_print_callback = @cfunction(print_callback, Cvoid, (MlirStringRef, Any))
    ref = Ref(io)
    API.mlirValuePrint(value, c_print_callback, ref)
end

is_a_op_result(value) = API.mlirValueIsAOpResult(value)
is_a_block_argument(value) = API.mlirValueIsABlockArgument(value)

function set_type!(value, type)
    @assert is_a_block_argument(value) "could not set type, value is not a block argument"
    API.mlirBlockArgumentSetType(value, type)
    value
end

function get_owner(value::Value)
    if is_a_block_argument(value)
        raw_block = API.mlirBlockArgumentGetOwner(value)
        if mlirIsNull(raw_block)
            return nothing
        end

        return Block(raw_block, false)
    end

    raw_op = API.mlirOpResultGetOwner(value)
    if mlirIsNull(raw_op)
        return nothing
    end

    return Operation(raw_op, false)
end

### Operation

mutable struct Operation
    operation::MlirOperation
    @atomic owned::Bool

    Operation(operation, owned=true) = begin
        @assert !mlirIsNull(operation) "cannot create Operation with null MlirOperation"
        finalizer(new(operation, owned)) do op
            if op.owned
                API.mlirOperationDestroy(op.operation)
            end
        end
    end
end

function create_operation(
    name, loc;
    results=nothing,
    operands=nothing,
    owned_regions=nothing,
    successors=nothing,
    attributes=nothing,
    result_inference=isnothing(results)
)
    GC.@preserve name loc begin
        state = Ref(API.mlirOperationStateGet(name, loc))
        if !isnothing(results)
            if result_inference
                error("Result inference and provided results conflict")
            end
            API.mlirOperationStateAddResults(state, length(results), results)
        end
        if !isnothing(operands)
            API.mlirOperationStateAddOperands(state, length(operands), operands)
        end
        if !isnothing(owned_regions)
            lose_ownership!.(owned_regions)
            GC.@preserve owned_regions begin
                mlir_regions = Base.unsafe_convert.(MlirRegion, owned_regions)
                API.mlirOperationStateAddOwnedRegions(state, length(mlir_regions), mlir_regions)
            end
        end
        if !isnothing(successors)
            GC.@preserve successors begin
                mlir_blocks = Base.unsafe_convert.(MlirBlock, successors)
                API.mlirOperationStateAddSuccessors(
                    state,
                    length(mlir_blocks),
                    mlir_blocks,
                )
            end
        end
        if !isnothing(attributes)
            API.mlirOperationStateAddAttributes(state, length(attributes), attributes)
        end
        if result_inference
            API.mlirOperationStateEnableResultTypeInference(state)
        end
        op = API.mlirOperationCreate(state)
        if mlirIsNull(op)
            error("Create Operation failed")
        end
        Operation(op, true)
    end
end

Base.copy(operation::Operation) = Operation(API.mlirOperationClone(operation))

num_regions(operation) = API.mlirOperationGetNumRegions(operation)
function get_region(operation, i)
    i ∉ 1:num_regions(operation) && throw(BoundsError(operation, i))
    Region(API.mlirOperationGetRegion(operation, i - 1), false)
end
num_results(operation::Operation) = API.mlirOperationGetNumResults(operation)
get_results(operation) = [
    get_result(operation, i)
    for i in 1:num_results(operation)
]
function get_result(operation::Operation, i=1)
    i ∉ 1:num_results(operation) && throw(BoundsError(operation, i))
    Value(API.mlirOperationGetResult(operation, i - 1))
end
num_operands(operation) = API.mlirOperationGetNumOperands(operation)
function get_operand(operation, i=1)
    i ∉ 1:num_operands(operation) && throw(BoundsError(operation, i))
    Value(API.mlirOperationGetOperand(operation, i - 1))
end
function set_operand!(operation, i, value)
    i ∉ 1:num_operands(operation) && throw(BoundsError(operation, i))
    API.mlirOperationSetOperand(operation, i - 1, value)
    value
end

function get_attribute_by_name(operation, name)
    raw_attr = API.mlirOperationGetAttributeByName(operation, name)
    if mlirIsNull(raw_attr)
        return nothing
    end
    Attribute(raw_attr)
end
function set_attribute_by_name!(operation, name, attribute)
    API.mlirOperationSetAttributeByName(operation, name, attribute)
    operation
end

location(operation) = Location(API.mlirOperationGetLocation(operation))
name(operation) = String(API.mlirOperationGetName(operation))
block(operation) = Block(API.mlirOperationGetBlock(operation), false)
parent_operation(operation) = Operation(API.mlirOperationGetParentOperation(operation), false)
dialect(operation) = first(split(name(operation), '.')) |> Symbol

function get_first_region(op::Operation)
    reg = iterate(RegionIterator(op))
    isnothing(reg) && return nothing
    first(reg)
end
function get_first_block(op::Operation)
    reg = get_first_region(op)
    isnothing(reg) && return nothing
    block = iterate(BlockIterator(reg))
    isnothing(block) && return nothing
    first(block)
end
function get_first_child_op(op::Operation)
    block = get_first_block(op)
    isnothing(block) && return nothing
    cop = iterate(OperationIterator(block))
    first(cop)
end

op::Operation == other::Operation = API.mlirOperationEqual(op, other)

Base.cconvert(::Type{MlirOperation}, operation::Operation) = operation
Base.unsafe_convert(::Type{MlirOperation}, operation::Operation) = operation.operation

function lose_ownership!(operation::Operation)
    @assert operation.owned
    @atomic operation.owned = false
    operation
end

function Base.show(io::IO, operation::Operation)
    c_print_callback = @cfunction(print_callback, Cvoid, (MlirStringRef, Any))

    buffer = IOBuffer()
    ref = Ref(buffer)

    flags = API.mlirOpPrintingFlagsCreate()
    get(io, :debug, false) && API.mlirOpPrintingFlagsEnableDebugInfo(flags, true, true)
    API.mlirOperationPrintWithFlags(operation, flags, c_print_callback, ref)
    API.mlirOpPrintingFlagsDestroy(flags)

    write(io, rstrip(String(take!(buffer))))
end

verify(operation::Operation) = API.mlirOperationVerify(operation)

### Block

mutable struct Block
    block::MlirBlock
    @atomic owned::Bool

    Block(block::MlirBlock, owned::Bool=true) = begin
        @assert !mlirIsNull(block) "cannot create Block with null MlirBlock"
        finalizer(new(block, owned)) do block
            if block.owned
                API.mlirBlockDestroy(block.block)
            end
        end
    end
end

Block() = Block(MLIRType[], Location[])
function Block(args::Vector{MLIRType}, locs::Vector{Location})
    @assert length(args) == length(locs) "there should be one args for each locs (got $(length(args)) & $(length(locs)))"
    Block(API.mlirBlockCreate(length(args), args, locs))
end

function Base.push!(block::Block, op::Operation)
    API.mlirBlockAppendOwnedOperation(block, lose_ownership!(op))
    op
end
function Base.insert!(block::Block, pos, op::Operation)
    API.mlirBlockInsertOwnedOperation(block, pos - 1, lose_ownership!(op))
    op
end
function Base.pushfirst!(block::Block, op::Operation)
    insert!(block, 1, op)
    op
end
function insert_after!(block::Block, reference::Operation, op::Operation)
    API.mlirBlockInsertOwnedOperationAfter(block, reference, lose_ownership!(op))
    op
end
function insert_before!(block::Block, reference::Operation, op::Operation)
    API.mlirBlockInsertOwnedOperationBefore(block, reference, lose_ownership!(op))
    op
end

num_arguments(block::Block) =
    API.mlirBlockGetNumArguments(block)
function get_argument(block::Block, i)
    i ∉ 1:num_arguments(block) && throw(BoundsError(block, i))
    Value(API.mlirBlockGetArgument(block, i - 1))
end
push_argument!(block::Block, type, loc) =
    Value(API.mlirBlockAddArgument(block, type, loc))

Base.cconvert(::Type{MlirBlock}, block::Block) = block
Base.unsafe_convert(::Type{MlirBlock}, block::Block) = block.block

function lose_ownership!(block::Block)
    @assert block.owned
    @atomic block.owned = false
    block
end

function Base.show(io::IO, block::Block)
    c_print_callback = @cfunction(print_callback, Cvoid, (MlirStringRef, Any))
    ref = Ref(io)
    API.mlirBlockPrint(block, c_print_callback, ref)
end

### Region

mutable struct Region
    region::MlirRegion
    @atomic owned::Bool

    Region(region, owned=true) = begin
        @assert !mlirIsNull(region)
        finalizer(new(region, owned)) do region
            if region.owned
                API.mlirRegionDestroy(region.region)
            end
        end
    end
end

Region() = Region(API.mlirRegionCreate())

function Base.push!(region::Region, block::Block)
    API.mlirRegionAppendOwnedBlock(region, lose_ownership!(block))
    block
end
function Base.insert!(region::Region, pos, block::Block)
    API.mlirRegionInsertOwnedBlock(region, pos - 1, lose_ownership!(block))
    block
end
function Base.pushfirst!(region::Region, block)
    insert!(region, 1, block)
    block
end
insert_after!(region::Region, reference::Block, block::Block) =
    API.mlirRegionInsertOwnedBlockAfter(region, reference, lose_ownership!(block))
insert_before!(region::Region, reference::Block, block::Block) =
    API.mlirRegionInsertOwnedBlockBefore(region, reference, lose_ownership!(block))

function get_first_block(region::Region)
    block = iterate(BlockIterator(region))
    isnothing(block) && return nothing
    first(block)
end

function lose_ownership!(region::Region)
    @assert region.owned
    @atomic region.owned = false
    region
end

Base.cconvert(::Type{MlirRegion}, region::Region) = region
Base.unsafe_convert(::Type{MlirRegion}, region::Region) = region.region

### Module

mutable struct MModule
    module_::MlirModule

    MModule(module_) = begin
        @assert !mlirIsNull(module_) "cannot create MModule with null MlirModule"
        finalizer(API.mlirModuleDestroy, new(module_))
    end
end

MModule(loc::Location=Location()) =
    MModule(API.mlirModuleCreateEmpty(loc))
MModule(op::Operation) = MModule(API.mlirModuleFromOperation(op))
get_operation(module_) = Operation(API.mlirModuleGetOperation(module_), false)
get_body(module_) = Block(API.mlirModuleGetBody(module_), false)
get_first_child_op(mod::MModule) = get_first_child_op(get_operation(mod))

Base.convert(::Type{MlirModule}, module_::MModule) = module_.module_
Base.parse(::Type{MModule}, module_) = MModule(API.mlirModuleCreateParse(context(), module_), context())

macro mlir_str(code)
    quote
        ctx = Context()
        parse(MModule, code)
    end
end

function Base.show(io::IO, module_::MModule)
    println(io, "MModule:")
    show(io, get_operation(module_))
end

### TypeID

struct TypeID
    typeid::API.MlirTypeID
end

Base.hash(typeid::TypeID) = API.mlirTypeIDHashValue(typeid.typeid)
Base.convert(::Type{API.MlirTypeID}, typeid::TypeID) = typeid.typeid

@static if isdefined(API, :MlirTypeIDAllocator)

    ### TypeIDAllocator

    mutable struct TypeIDAllocator
        allocator::API.MlirTypeIDAllocator

        function TypeIDAllocator()
            ptr = API.mlirTypeIDAllocatorCreate()
            @assert ptr != C_NULL "cannot create TypeIDAllocator"
            finalizer(API.mlirTypeIDAllocatorDestroy, new(ptr))
        end
    end

    Base.cconvert(::Type{API.MlirTypeIDAllocator}, allocator::TypeIDAllocator) = allocator
    Base.unsafe_convert(::Type{API.MlirTypeIDAllocator}, allocator) = allocator.allocator

    TypeID(allocator::TypeIDAllocator) = TypeID(API.mlirTypeIDCreate(allocator))

else

    struct TypeIDAllocator end

end

include("./Support.jl")
include("./Pass.jl")

