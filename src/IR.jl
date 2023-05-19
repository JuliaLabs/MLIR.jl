module IR

import ..API: API

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

function mlirIsNull(val)
   val.ptr == C_NULL 
end

function print_callback(str::MlirStringRef, userdata)
    data = unsafe_wrap(Array, Base.convert(Ptr{Cchar}, str.data), str.length; own=false)
    write(userdata isa Base.RefValue ? userdata[] : userdata, data)
    return Cvoid()
end

### Identifier

String(ident::MlirIdentifier) = String(API.mlirIdentifierStr(ident))

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

mutable struct Context
    context::MlirContext
end
function Context()
    context = API.mlirContextCreate()
    @assert !mlirIsNull(context) "cannot create Context with null MlirContext"
    finalizer(Context(context)) do context
        API.mlirContextDestroy(context.context)
    end
end

Base.convert(::Type{MlirContext}, c::Context) = c.context

num_loaded_dialects(context) = API.mlirContextGetNumLoadedDialects(context)
function get_or_load_dialect!(context, handle::DialectHandle)
    mlir_dialect = API.mlirDialectHandleLoadDialect(handle, context)
    if mlirIsNull(mlir_dialect)
        error("could not load dialect from handle $handle")
    else
        Dialect(mlir_dialect)
    end
end
function get_or_load_dialect!(context, dialect::String)
    get_or_load_dialect!(context, DialectHandle(Symbol(dialect)))
end

function enable_multithreading!(context, enable=true)
    API.mlirContextEnableMultithreading(context, enable)
    context
end

is_registered_operation(context, opname) = API.mlirContextIsRegisteredOperation(context, opname)

### Location

struct Location
    location::MlirLocation

    Location(location) = begin
        @assert !mlirIsNull(location) "cannot create Location with null MlirLocation"
        new(location)
    end
end

Location(context::Context) = Location(API.mlirLocationUnknownGet(context))
Location(context::Context, filename, line, column) =
    Location(API.mlirLocationFileLineColGet(context, filename, line, column))

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
MLIRType(context::Context, T::Type{<:Signed}) =
    MLIRType(API.mlirIntegerTypeGet(context, sizeof(T) * 8))
MLIRType(context::Context, T::Type{<:Unsigned}) =
    MLIRType(API.mlirIntegerTypeGet(context, sizeof(T) * 8))
MLIRType(context::Context, ::Type{Bool}) =
    MLIRType(API.mlirIntegerTypeGet(context, 1))
MLIRType(context::Context, ::Type{Float32}) =
    MLIRType(API.mlirF32TypeGet(context))
MLIRType(context::Context, ::Type{Float64}) =
    MLIRType(API.mlirF64TypeGet(context))
MLIRType(context::Context, ft::Pair) =
    MLIRType(API.mlirFunctionTypeGet(context,
        length(ft.first), [MLIRType(t) for t in ft.first],
        length(ft.second), [MLIRType(t) for t in ft.second]))
MLIRType(context, a::AbstractArray{T}) where {T} = MLIRType(context, MLIRType(context, T), size(a))
MLIRType(context, ::Type{<:AbstractArray{T,N}}, dims) where {T,N} =
    MLIRType(API.mlirRankedTensorTypeGetChecked(
        Location(context),
        N, collect(dims),
        MLIRType(context, T),
        Attribute(),
    ))
MLIRType(context, element_type::MLIRType, dims) =
    MLIRType(API.mlirRankedTensorTypeGetChecked(
        Location(context),
        length(dims), collect(dims),
        element_type,
        Attribute(),
    ))
MLIRType(context, ::T) where {T<:Real} = MLIRType(context, T)
MLIRType(_, type::MLIRType) = type

IndexType(context) = MLIRType(API.mlirIndexTypeGet(context))

Base.convert(::Type{MlirType}, mtype::MLIRType) = mtype.type

function Base.eltype(type::MLIRType)
    if API.mlirTypeIsAShaped(type)
        MLIRType(API.mlirShapedTypeGetElementType(type))
    else
        type
    end
end

function show_inner(io::IO, type::MLIRType)
    if API.mlirTypeIsAInteger(type)
        is_signless = API.mlirIntegerTypeIsSignless(type)
        is_signed = API.mlirIntegerTypeIsSigned(type)

        width = API.mlirIntegerTypeGetWidth(type)
        t = if is_signed
            "si"
        elseif is_signless
            "i"
        else
            "u"
        end
        print(io, t, width)
    elseif API.mlirTypeIsAF64(type)
        print(io, "f64")
    elseif API.mlirTypeIsAF32(type)
        print(io, "f32")
    elseif API.mlirTypeIsARankedTensor(type)
        print(io, "tensor<")
        s = size(type)
        print(io, join(s, "x"), "x")
        show_inner(io, eltype(type))
        print(io, ">")
    elseif API.mlirTypeIsAIndex(type)
        print(io, "index")
    else
        print(io, "unknown")
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
   IT = ints[Int(log2(size)) - 2]
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
Attribute(context, s::AbstractString) = Attribute(API.mlirStringAttrGet(context, s))
Attribute(type::MLIRType) = Attribute(API.mlirTypeAttrGet(type))
Attribute(context, f::F, type=MLIRType(context, F)) where {F<:AbstractFloat} = Attribute(
    API.mlirFloatAttrDoubleGet(context, type, Float64(f))
)
Attribute(context, i::T) where {T<:Integer} = Attribute(
    API.mlirIntegerAttrGet(MLIRType(context, T), Int64(i))
)
function Attribute(context, values::T) where {T<:AbstractArray{Int32}}
    type = MLIRType(context, T, size(values))
    Attribute(
        API.mlirDenseElementsAttrInt32Get(type, length(values), values)
    )
end
function Attribute(context, values::T) where {T<:AbstractArray{Int64}}
    type = MLIRType(context, T, size(values))
    Attribute(
        API.mlirDenseElementsAttrInt64Get(type, length(values), values)
    )
end
function Attribute(context, values::T) where {T<:AbstractArray{Float64}}
    type = MLIRType(context, T, size(values))
    Attribute(
        API.mlirDenseElementsAttrDoubleGet(type, length(values), values)
    )
end
function Attribute(context, values::T) where {T<:AbstractArray{Float32}}
    type = MLIRType(context, T, size(values))
    Attribute(
        API.mlirDenseElementsAttrFloatGet(type, length(values), values)
    )
end
function Attribute(context, values::AbstractArray{Int32}, type)
    Attribute(
        API.mlirDenseElementsAttrInt32Get(type, length(values), values)
    )
end
function Attribute(context, values::AbstractArray{Int}, type)
    Attribute(
        API.mlirDenseElementsAttrInt64Get(type, length(values), values)
    )
end
function Attribute(context, values::AbstractArray{Float32}, type)
    Attribute(
        API.mlirDenseElementsAttrFloatGet(type, length(values), values)
    )
end
function ArrayAttribute(context, values::AbstractVector{Int})
    elements = Attribute.((context,), values)
    Attribute(
        API.mlirArrayAttrGet(context, length(elements), elements)
    )
end
function ArrayAttribute(context, attributes::Vector{Attribute})
    Attribute(
        API.mlirArrayAttrGet(context, length(attributes), attributes),
    )
end
function DenseArrayAttribute(context, values::AbstractVector{Int})
    Attribute(
        API.mlirDenseI64ArrayGet(context, length(values), collect(values))
    )
end
function Attribute(context, value::Int, type::MLIRType)
    Attribute(
        API.mlirIntegerAttrGet(type, value)
    )
end
function Attribute(context, value::Bool, ::MLIRType=nothing)
    Attribute(
        API.mlirBoolAttrGet(context, value)
    )
end

Base.convert(::Type{MlirAttribute}, attribute::Attribute) = attribute.attribute
Base.parse(::Type{Attribute}, context, s) =
    Attribute(API.mlirAttributeParseGet(context, s))

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

function NamedAttribute(context, name, attribute)
    @assert !mlirIsNull(attribute.attribute)
    NamedAttribute(API.mlirNamedAttributeGet(
        API.mlirIdentifierGet(context, name),
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
    result_inference=isnothing(results),
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
    i ∈ 1:num_regions(operation) && throw(BoundsError(operation, i))
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
dialect(operation) = first(split(get_name(operation), '.')) |> Symbol

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
    ref = Ref(io)
    flags = API.mlirOpPrintingFlagsCreate()
    get(io, :debug, false) && API.mlirOpPrintingFlagsEnableDebugInfo(flags, true, true)
    API.mlirOperationPrintWithFlags(operation, flags, c_print_callback, ref)
    println(io)
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
    context::Context

    MModule(module_, context) = begin
        @assert !mlirIsNull(module_) "cannot create MModule with null MlirModule"
        finalizer(API.mlirModuleDestroy, new(module_, context))
    end
end

MModule(context::Context, loc=Location(context)) =
    MModule(API.mlirModuleCreateEmpty(loc), context)
get_operation(module_) = Operation(API.mlirModuleGetOperation(module_), false)
get_body(module_) = Block(API.mlirModuleGetBody(module_), false)
get_first_child_op(mod::MModule) = get_first_child_op(get_operation(mod))

Base.convert(::Type{MlirModule}, module_::MModule) = module_.module_
Base.parse(::Type{MModule}, context, module_) = MModule(API.mlirModuleCreateParse(context, module_), context)

macro mlir_str(code)
    quote
        ctx = Context()
        parse(MModule, ctx, code)
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

Base.convert(::Type{API.MlirTypeIDAllocator}, allocator::TypeIDAllocator) = allocator.allocator

TypeID(allocator::TypeIDAllocator) = TypeID(API.mlirTypeIDCreate(allocator))

else

struct TypeIDAllocator end

end

### Pass Manager

abstract type AbstractPass end

mutable struct ExternalPassHandle
    ctx::Union{Nothing,Context}
    pass::AbstractPass
end

mutable struct PassManager
    pass::MlirPassManager
    context::Context
    allocator::TypeIDAllocator
    passes::Dict{TypeID,ExternalPassHandle}

    PassManager(pm::MlirPassManager, context) = begin
        @assert !mlirIsNull(pm) "cannot create PassManager with null MlirPassManager"
        finalizer(new(pm, context, TypeIDAllocator(), Dict{TypeID,ExternalPassHandle}())) do pm
            API.mlirPassManagerDestroy(pm.pass)
        end
    end
end

function enable_ir_printing!(pm)
    API.mlirPassManagerEnableIRPrinting(pm)
    pm
end
function enable_verifier!(pm, enable=true)
    API.mlirPassManagerEnableVerifier(pm, enable)
    pm
end

PassManager(context) =
    PassManager(API.mlirPassManagerCreate(context), context)

function run!(pm::PassManager, module_)
    status = API.mlirPassManagerRun(pm, module_)
    if mlirLogicalResultIsFailure(status)
        throw("failed to run pass manager on module")
    end
    module_
end

Base.convert(::Type{MlirPassManager}, pass::PassManager) = pass.pass

### Op Pass Manager

struct OpPassManager
    op_pass::MlirOpPassManager
    pass::PassManager

    OpPassManager(op_pass, pass) = begin
        @assert !mlirIsNull(op_pass) "cannot create OpPassManager with null MlirOpPassManager"
        new(op_pass, pass)
    end
end

OpPassManager(pm::PassManager) = OpPassManager(API.mlirPassManagerGetAsOpPassManager(pm), pm)
OpPassManager(pm::PassManager, opname) = OpPassManager(API.mlirPassManagerGetNestedUnder(pm, opname), pm)
OpPassManager(opm::OpPassManager, opname) = OpPassManager(API.mlirOpPassManagerGetNestedUnder(opm, opname), opm.pass)

Base.convert(::Type{MlirOpPassManager}, op_pass::OpPassManager) = op_pass.op_pass

function Base.show(io::IO, op_pass::OpPassManager)
    c_print_callback = @cfunction(print_callback, Cvoid, (MlirStringRef, Any))
    ref = Ref(io)
    println(io, "OpPassManager(\"\"\"")
    API.mlirPrintPassPipeline(op_pass, c_print_callback, ref)
    println(io)
    print(io, "\"\"\")")
end

struct AddPipelineException <: Exception
    message::String
end

function Base.showerror(io::IO, err::AddPipelineException)
    print(io, "failed to add pipeline:", err.message)
    nothing
end

mlirLogicalResultIsFailure(result) = result.value == 0

function add_pipeline!(op_pass::OpPassManager, pipeline)
    @static if isdefined(API, :mlirOpPassManagerAddPipeline)
        io = IOBuffer()
        c_print_callback = @cfunction(print_callback, Cvoid, (MlirStringRef, Any))
        result = GC.@preserve io API.mlirOpPassManagerAddPipeline(op_pass, pipeline, c_print_callback, io)
        if API.mlirLogicalResultIsFailure(result)
            exc = AddPipelineException(String(take!(io)))
            throw(exc)
        end
    else
        result = API.mlirParsePassPipeline(op_pass, pipeline)
        if mlirLogicalResultIsFailure(result)
            throw(AddPipelineException(" " * pipeline))
        end
    end
    op_pass
end
 
function add_owned_pass!(pm::PassManager, pass)
    API.mlirPassManagerAddOwnedPass(pm, pass)
    pm
end

function add_owned_pass!(opm::OpPassManager, pass)
    API.mlirOpPassManagerAddOwnedPass(opm, pass)
    opm
end


@static if isdefined(API, :mlirCreateExternalPass)

### Pass

# AbstractPass interface:
opname(::AbstractPass) = ""
function pass_run(::Context, ::P, op) where {P<:AbstractPass}
    error("pass $P does not implement `MLIR.pass_run`")
end

function _pass_construct(ptr::ExternalPassHandle)
    nothing
end

function _pass_destruct(ptr::ExternalPassHandle)
    nothing
end

function _pass_initialize(ctx, handle::ExternalPassHandle)
    try
        handle.ctx = Context(ctx)
        API.mlirLogicalResultSuccess()
    catch
        API.mlirLogicalResultFailure()
    end
end

function _pass_clone(handle::ExternalPassHandle)
    ExternalPassHandle(handle.ctx, deepcopy(handle.pass))
end

function _pass_run(rawop, external_pass, handle::ExternalPassHandle)
    op = Operation(rawop, false)
    try
        pass_run(handle.ctx, handle.pass, op)
    catch ex
        @error "Something went wrong running pass" exception=(ex,catch_backtrace())
        API.mlirExternalPassSignalFailure(external_pass)
    end
    nothing
end

function create_external_pass!(oppass::OpPassManager, args...)
    create_external_pass!(oppass.pass, args...)
end
function create_external_pass!(manager, pass, name, argument,
                               description, opname=opname(pass),
                               dependent_dialects=MlirDialectHandle[])
    passid = TypeID(manager.allocator)
    callbacks = API.MlirExternalPassCallbacks(
            @cfunction(_pass_construct, Cvoid, (Any,)),
            @cfunction(_pass_destruct, Cvoid, (Any,)),
            @cfunction(_pass_initialize, API.MlirLogicalResult, (MlirContext, Any,)),
            @cfunction(_pass_clone, Any, (Any,)),
            @cfunction(_pass_run, Cvoid, (MlirOperation, API.MlirExternalPass, Any))
    )
    pass_handle = manager.passes[passid] = ExternalPassHandle(nothing, pass)
    userdata = Base.pointer_from_objref(pass_handle)
    mlir_pass = API.mlirCreateExternalPass(passid, name, argument, description, opname,
                                               length(dependent_dialects), dependent_dialects,
                                               callbacks, userdata)
    mlir_pass
end

end

### Iterators

"""
    BlockIterator(region::Region)

Iterates over all blocks in the given region.
"""
struct BlockIterator
    region::Region
end

function Base.iterate(it::BlockIterator)
    reg = it.region
    raw_block = API.mlirRegionGetFirstBlock(reg)
    if mlirIsNull(raw_block)
        nothing
    else
        b = Block(raw_block, false)
        (b, b)
    end
end

function Base.iterate(it::BlockIterator, block)
    raw_block = API.mlirBlockGetNextInRegion(block)
    if mlirIsNull(raw_block)
        nothing
    else
        b = Block(raw_block, false)
        (b, b)
    end
end

"""
    OperationIterator(block::Block)

Iterates over all operations for the given block.
"""
struct OperationIterator
    block::Block
end

function Base.iterate(it::OperationIterator)
    raw_op = API.mlirBlockGetFirstOperation(it.block)
    if mlirIsNull(raw_op)
        nothing
    else
        op = Operation(raw_op, false)
        (op, op)
    end
end

function Base.iterate(it::OperationIterator, op)
    raw_op = API.mlirOperationGetNextInBlock(op)
    if mlirIsNull(raw_op)
        nothing
    else
        op = Operation(raw_op, false)
        (op, op)
    end
end

"""
    RegionIterator(::Operation)

Iterates over all sub-regions for the given operation.
"""
struct RegionIterator
    op::Operation
end

function Base.iterate(it::RegionIterator)
    raw_region = API.mlirOperationGetFirstRegion(it.op)
    if mlirIsNull(raw_region)
        nothing
    else
        region = Region(raw_region, false)
        (region, region)
    end
end

function Base.iterate(it::RegionIterator, region)
    raw_region = API.mlirRegionGetNextInOperation(region)
    if mlirIsNull(raw_region)
        nothing
    else
        region = Region(raw_region, false)
        (region, region)
    end
end

### Utils

function visit(f, op)
    for region in RegionIterator(op)
        for block in BlockIterator(region)
            for op in OperationIterator(block)
                f(op)
            end
        end
    end
end

"""
    verifyall(operation; debug=false)

Prints the operations which could not be verified.
"""
function verifyall(operation::Operation; debug=false)
    io = IOContext(stdout, :debug => debug)
    visit(operation) do op
        if !verify(op)
            show(io, op)
        end
    end
end
verifyall(module_::MModule) = get_operation(module_) |> verifyall

end # module IR
