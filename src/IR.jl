module IR

import ..API: API as LibMLIR

export
    Operation,
    OperationState,
    Location,
    Context,
    MModule,
    Value,
    MType,
    Region,
    Block,
    Attribute,
    NamedAttribute

export
    add_results!,
    add_attributes!,
    add_owned_regions!,
    add_successors!


import Base: ==, String
using .LibMLIR:
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

### String Ref

String(strref::MlirStringRef) =
    Base.unsafe_string(Base.convert(Ptr{Cchar}, strref.data), strref.length)
Base.convert(::Type{MlirStringRef}, s::String) =
    MlirStringRef(Base.unsafe_convert(Cstring, s), sizeof(s))

### Identifier

String(ident::MlirIdentifier) = String(LibMLIR.mlirIdentifierStr(ident))

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
    print(io, "Dialect(\"", String(LibMLIR.mlirDialectGetNamespace(dialect)), "\")")
end

### DialectHandle

struct DialectHandle
    handle::LibMLIR.MlirDialectHandle
end

function DialectHandle(s::Symbol)
    s = Symbol("mlirGetDialectHandle__", s, "__")
    DialectHandle(getproperty(LibMLIR, s)())
end

Base.convert(::Type{MlirDialectHandle}, handle::DialectHandle) = handle.handle

### Dialect Registry

mutable struct DialectRegistry
    registry::MlirDialectRegistry
end
function DialectRegistry()
    registry = LibMLIR.mlirDialectRegistryCreate()
    @assert !mlirIsNull(registry) "cannot create DialectRegistry with null MlirDialectRegistry"
    finalizer(DialectRegistry(registry)) do registry
        LibMLIR.mlirDialectRegistryDestroy(registry.registry)
    end
end

function Base.insert!(registry::DialectRegistry, handle::DialectHandle)
    LibMLIR.mlirDialectHandleInsertDialect(registry, handle)
end

### Context

mutable struct Context
    context::MlirContext
end
function Context()
    context = LibMLIR.mlirContextCreate()
    @assert !mlirIsNull(context) "cannot create Context with null MlirContext"
    finalizer(Context(context)) do context
        LibMLIR.mlirContextDestroy(context.context)
    end
end

Base.convert(::Type{MlirContext}, c::Context) = c.context

num_loaded_dialects(context) = LibMLIR.mlirContextGetNumLoadedDialects(context)
function get_or_load_dialect!(context, handle::DialectHandle)
    mlir_dialect = LibMLIR.mlirDialectHandleLoadDialect(handle, context)
    if mlirIsNull(mlir_dialect)
        error("could not load dialect from handle $handle")
    else
        Dialect(mlir_dialect)
    end
end
function get_or_load_dialect!(context, dialect::String)
    get_or_load_dialect!(context, DialectHandle(Symbol(dialect)))
end

is_registered_operation(context, opname) = LibMLIR.mlirContextIsRegisteredOperation(context, opname)

### Location

struct Location
    location::MlirLocation

    Location(location) = begin
        @assert !mlirIsNull(location) "cannot create Location with null MlirLocation"
        new(location)
    end
end

Location(context::Context) = Location(LibMLIR.mlirLocationUnknownGet(context))
Location(context::Context, filename, line, column=0) =
    Location(LibMLIR.mlirLocationFileLineColGet(context, filename, line, column))
Location(context::Context, lin::Core.LineInfoNode) =
    Location(context, string(lin.file), lin.line)
Location(context::Context, lin::LineNumberNode) =
    isnothing(lin.file) ?
    Location(context) :
    Location(context, string(lin.file), lin.line)
Location(context::Context, ::Nothing) = Location(context)

Base.convert(::Type{MlirLocation}, location::Location) = location.location

function Base.show(io::IO, location::Location)
    c_print_callback = @cfunction(print_callback, Cvoid, (MlirStringRef, Any))
    ref = Ref(io)
    print(io, "Location(#= ")
    GC.@preserve ref LibMLIR.mlirLocationPrint(location, c_print_callback, ref)
    print(io, " =#)")
end

### Type

struct MType
    type::MlirType

    MType(type) = begin
        @assert !mlirIsNull(type)
        new(type)
    end
end

MType(t::MType) = t
MType(context::Context, T::Type{<:Signed}) =
    MType(LibMLIR.mlirIntegerTypeGet(context, sizeof(T) * 8))
MType(context::Context, T::Type{<:Unsigned}) =
    MType(LibMLIR.mlirIntegerTypeGet(context, sizeof(T) * 8))
MType(context::Context, ::Type{Bool}) =
    MType(LibMLIR.mlirIntegerTypeGet(context, 1))
MType(context::Context, ::Type{Float32}) =
    MType(LibMLIR.mlirF32TypeGet(context))
MType(context::Context, ::Type{Float64}) =
    MType(LibMLIR.mlirF64TypeGet(context))
MType(context::Context, ft::Pair) =
    MType(LibMLIR.mlirFunctionTypeGet(context,
        length(ft.first), [MType(t) for t in ft.first],
        length(ft.second), [MType(t) for t in ft.second]))
MType(context, a::AbstractArray{T}) where {T} = MType(context, MType(context, T), size(a))
MType(context, ::Type{<:AbstractArray{T,N}}, dims) where {T,N} =
    MType(LibMLIR.mlirRankedTensorTypeGetChecked(
        Location(context),
        N, collect(dims),
        MType(context, T),
        Attribute(),
    ))
MType(context, element_type::MType, dims) =
    MType(LibMLIR.mlirRankedTensorTypeGetChecked(
        Location(context),
        length(dims), collect(dims),
        element_type,
        Attribute(),
    ))
MType(context, ::T) where {T<:Real} = MType(context, T)
MType(_, type::MType) = type

IndexType(context) = MType(LibMLIR.mlirIndexTypeGet(context))

Base.convert(::Type{MlirType}, mtype::MType) = mtype.type

function Base.eltype(type::MType)
    if LibMLIR.mlirTypeIsAShaped(type)
        MType(LibMLIR.mlirShapedTypeGetElementType(type))
    else
        type
    end
end

function show_inner(io::IO, type::MType)
    if LibMLIR.mlirTypeIsAInteger(type)
        is_signless = LibMLIR.mlirIntegerTypeIsSignless(type)
        is_signed = LibMLIR.mlirIntegerTypeIsSigned(type)

        width = LibMLIR.mlirIntegerTypeGetWidth(type)
        t = if is_signed
            "si"
        elseif is_signless
            "i"
        else
            "u"
        end
        print(io, t, width)
    elseif LibMLIR.mlirTypeIsAF64(type)
        print(io, "f64")
    elseif LibMLIR.mlirTypeIsAF32(type)
        print(io, "f32")
    elseif LibMLIR.mlirTypeIsARankedTensor(type)
        print(io, "tensor<")
        s = size(type)
        print(io, join(s, "x"), "x")
        show_inner(io, eltype(type))
        print(io, ">")
    elseif LibMLIR.mlirTypeIsAIndex(type)
        print(io, "index")
    else
        print(io, "unknown")
    end
end

function Base.show(io::IO, type::MType)
    print(io, "MType(#= ")
    c_print_callback = @cfunction(print_callback, Cvoid, (MlirStringRef, Any))
    ref = Ref(io)
    GC.@preserve ref LibMLIR.mlirTypePrint(type, c_print_callback, ref)
    print(io, " =#)")
end

function inttype(size, issigned)
   size == 1 && issigned && return Bool
   ints = (Int8, Int16, Int32, Int64, Int128)
   IT = ints[Int(log2(size)) - 2]
   issigned ? IT : unsigned(IT)
end

function julia_type(type::MType)
    if LibMLIR.mlirTypeIsAInteger(type)
        is_signed = LibMLIR.mlirIntegerTypeIsSigned(type) ||
                    LibMLIR.mlirIntegerTypeIsSignless(type)
        width = LibMLIR.mlirIntegerTypeGetWidth(type)

        try
            inttype(width, is_signed)
        catch
            t = is_signed ? "i" : "u"
            throw("could not convert type $(t)$(width) to julia")
        end
    elseif LibMLIR.mlirTypeIsAF32(type)
        Float32
    elseif LibMLIR.mlirTypeIsAF64(type)
        Float64
    else
        throw("could not convert type $type to julia")
    end
end

Base.ndims(type::MType) =
    if LibMLIR.mlirTypeIsAShaped(type) && LibMLIR.mlirShapedTypeHasRank(type)
        LibMLIR.mlirShapedTypeGetRank(type)
    else
        0
    end

Base.size(type::MType, i::Int) = LibMLIR.mlirShapedTypeGetDimSize(type, i - 1)
Base.size(type::MType) = Tuple(size(type, i) for i in 1:ndims(type))

function is_tensor(type::MType)
    LibMLIR.mlirTypeIsAShaped(type)
end

function is_integer(type::MType)
    LibMLIR.mlirTypeIsAInteger(type)
end

is_function_type(mtype) = LibMLIR.mlirTypeIsAFunction(mtype)

function get_num_inputs(ftype)
    @assert is_function_type(ftype) "cannot get the number of inputs on type $(ftype), expected a function type"
    LibMLIR.mlirFunctionTypeGetNumInputs(ftype)
end
function get_num_results(ftype)
    @assert is_function_type(ftype) "cannot get the number of results on type $(ftype), expected a function type"
    LibMLIR.mlirFunctionTypeGetNumResults(ftype)
end

function get_input(ftype::MType, pos)
    @assert is_function_type(ftype) "cannot get input on type $(ftype), expected a function type"
    MType(LibMLIR.mlirFunctionTypeGetInput(ftype, pos - 1))
end
function get_result(ftype::MType, pos=1)
    @assert is_function_type(ftype) "cannot get result on type $(ftype), expected a function type"
    MType(LibMLIR.mlirFunctionTypeGetResult(ftype, pos - 1))
end

### Attribute

struct Attribute
    attribute::MlirAttribute
end

Attribute() = Attribute(LibMLIR.mlirAttributeGetNull())
Attribute(context, s::AbstractString) = Attribute(LibMLIR.mlirStringAttrGet(context, s))
Attribute(type::MType) = Attribute(LibMLIR.mlirTypeAttrGet(type))
Attribute(context, f::F, type=MType(context, F)) where {F<:AbstractFloat} = Attribute(
    LibMLIR.mlirFloatAttrDoubleGet(context, type, Float64(f))
)
Attribute(context, i::T) where {T<:Integer} = Attribute(
    LibMLIR.mlirIntegerAttrGet(MType(context, T), Int64(i))
)
function Attribute(context, values::T) where {T<:AbstractArray{Int32}}
    type = MType(context, T, size(values))
    Attribute(
        LibMLIR.mlirDenseElementsAttrInt32Get(type, length(values), values)
    )
end
function Attribute(context, values::T) where {T<:AbstractArray{Int64}}
    type = MType(context, T, size(values))
    Attribute(
        LibMLIR.mlirDenseElementsAttrInt64Get(type, length(values), values)
    )
end
function Attribute(context, values::T) where {T<:AbstractArray{Float64}}
    type = MType(context, T, size(values))
    Attribute(
        LibMLIR.mlirDenseElementsAttrDoubleGet(type, length(values), values)
    )
end
function Attribute(context, values::T) where {T<:AbstractArray{Float32}}
    type = MType(context, T, size(values))
    Attribute(
        LibMLIR.mlirDenseElementsAttrFloatGet(type, length(values), values)
    )
end
function Attribute(context, values::AbstractArray{Int32}, type)
    Attribute(
        LibMLIR.mlirDenseElementsAttrInt32Get(type, length(values), values)
    )
end
function Attribute(context, values::AbstractArray{Int}, type)
    Attribute(
        LibMLIR.mlirDenseElementsAttrInt64Get(type, length(values), values)
    )
end
function Attribute(context, values::AbstractArray{Float32}, type)
    Attribute(
        LibMLIR.mlirDenseElementsAttrFloatGet(type, length(values), values)
    )
end
function ArrayAttribute(context, values::AbstractVector{Int})
    elements = Attribute.((context,), values)
    Attribute(
        LibMLIR.mlirArrayAttrGet(context, length(elements), elements)
    )
end
function ArrayAttribute(context, attributes::Vector{Attribute})
    Attribute(
        LibMLIR.mlirArrayAttrGet(context, length(attributes), attributes),
    )
end
function DenseArrayAttribute(context, values::AbstractVector{Int})
    Attribute(
        LibMLIR.mlirDenseI64ArrayGet(context, length(values), collect(values))
    )
end
function Attribute(context, value::Int, type::MType)
    Attribute(
        LibMLIR.mlirIntegerAttrGet(type, value)
    )
end
function Attribute(context, value::Bool, ::MType=nothing)
    Attribute(
        LibMLIR.mlirBoolAttrGet(context, value)
    )
end

Base.convert(::Type{MlirAttribute}, attribute::Attribute) = attribute.attribute
Base.parse(::Type{Attribute}, context, s) =
    Attribute(LibMLIR.mlirAttributeParseGet(context, s))

function get_type(attribute::Attribute)
    MType(LibMLIR.mlirAttributeGetType(attribute))
end
function get_type_value(attribute)
    @assert LibMLIR.mlirAttributeIsAType(attribute) "attribute $(attribute) is not a type"
    MType(LibMLIR.mlirTypeAttrGetValue(attribute))
end
function get_bool_value(attribute)
    @assert LibMLIR.mlirAttributeIsABool(attribute) "attribute $(attribute) is not a boolean"
    LibMLIR.mlirBoolAttrGetValue(attribute)
end
function get_string_value(attribute)
    @assert LibMLIR.mlirAttributeIsAString(attribute) "attribute $(attribute) is not a string attribute"
    String(LibMLIR.mlirStringAttrGetValue(attribute))
end

function Base.show(io::IO, attribute::Attribute)
    print(io, "Attribute(#= ")
    c_print_callback = @cfunction(print_callback, Cvoid, (MlirStringRef, Any))
    ref = Ref(io)
    GC.@preserve ref LibMLIR.mlirAttributePrint(attribute, c_print_callback, ref)
    print(io, " =#)")
end

### Named Attribute

struct NamedAttribute
    named_attribute::MlirNamedAttribute
end

function NamedAttribute(context, name, attribute)
    @assert !mlirIsNull(attribute.attribute)
    NamedAttribute(LibMLIR.mlirNamedAttributeGet(
        LibMLIR.mlirIdentifierGet(context, name),
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

get_type(value) = MType(LibMLIR.mlirValueGetType(value))

Base.convert(::Type{MlirValue}, value::Value) = value.value
Base.size(value::Value) = Base.size(get_type(value))
Base.ndims(value::Value) = Base.ndims(get_type(value))

function Base.show(io::IO, value::Value)
    c_print_callback = @cfunction(print_callback, Cvoid, (MlirStringRef, Any))
    ref = Ref(io)
    GC.@preserve ref LibMLIR.mlirValuePrint(value, c_print_callback, ref)
end

is_a_op_result(value) = LibMLIR.mlirValueIsAOpResult(value)
is_a_block_argument(value) = LibMLIR.mlirValueIsABlockArgument(value)

function set_type!(value, type)
    @assert is_a_block_argument(value) "could not set type, value is not a block argument"
    LibMLIR.mlirBlockArgumentSetType(value, type)
    value
end

function get_owner(value::Value)
    if is_a_block_argument(value)
        raw_block = LibMLIR.mlirBlockArgumentGetOwner(value)
        if mlirIsNull(raw_block)
            return nothing
        end

        return Block(raw_block, false)
    end

    raw_op = LibMLIR.mlirOpResultGetOwner(value)
    if mlirIsNull(raw_op)
        return nothing
    end

    return Operation(raw_op, false)
end

### OperationState

struct OperationState
    opstate::MlirOperationState
end

OperationState(name, location) = OperationState(LibMLIR.mlirOperationStateGet(name, location))

add_results!(state, results) =
    LibMLIR.mlirOperationStateAddResults(state, length(results), results)
add_operands!(state, operands) =
    LibMLIR.mlirOperationStateAddOperands(state, length(operands), operands)
function add_owned_regions!(state, regions)
    mlir_regions = Base.convert.(MlirRegion, regions)
    lose_ownership!.(regions)
    LibMLIR.mlirOperationStateAddOwnedRegions(state, length(mlir_regions), mlir_regions)
end
add_attributes!(state, attributes) =
    LibMLIR.mlirOperationStateAddAttributes(state, length(attributes), attributes)
add_successors!(state, successors) =
    LibMLIR.mlirOperationStateAddSuccessors(
        state, length(successors),
        convert(Vector{LibMLIR.MlirBlock}, successors),
    )

enable_type_inference!(state) =
    LibMLIR.mlirOperationStateEnableResultTypeInference(state)

Base.unsafe_convert(::Type{Ptr{MlirOperationState}}, state::OperationState) =
    Base.unsafe_convert(Ptr{MlirOperationState}, Base.pointer_from_objref(state.opstate))

### Operation

mutable struct Operation
    operation::MlirOperation
    @atomic owned::Bool

    Operation(operation, owned=true) = begin
        @assert !mlirIsNull(operation) "cannot create Operation with null MlirOperation"
        finalizer(new(operation, owned)) do op
            if op.owned
                LibMLIR.mlirOperationDestroy(op.operation)
            end
        end
    end
end

Operation(state::OperationState) = Operation(LibMLIR.mlirOperationCreate(state), true)

Base.copy(operation::Operation) = Operation(LibMLIR.mlirOperationClone(operation))

num_regions(operation) = LibMLIR.mlirOperationGetNumRegions(operation)
function get_region(operation, i)
    i ∈ 1:num_regions(operation) && throw(BoundsError(operation, i))
    Region(LibMLIR.mlirOperationGetRegion(operation, i - 1), false)
end
num_results(operation) = LibMLIR.mlirOperationGetNumResults(operation)
get_results(operation) = [
    get_result(operation, i)
    for i in 1:num_results(operation)
]
function get_result(operation::Operation, i=1)
    i ∉ 1:num_results(operation) && throw(BoundsError(operation, i))
    Value(LibMLIR.mlirOperationGetResult(operation, i - 1))
end
num_operands(operation) = LibMLIR.mlirOperationGetNumOperands(operation)
function get_operand(operation, i=1)
    i ∉ 1:num_operands(operation) && throw(BoundsError(operation, i))
    Value(LibMLIR.mlirOperationGetOperand(operation, i - 1))
end
function set_operand!(operation, i, value)
    i ∉ 1:num_operands(operation) && throw(BoundsError(operation, i))
    LibMLIR.mlirOperationSetOperand(operation, i - 1, value)
    value
end

function get_attribute_by_name(operation, name)
    raw_attr = LibMLIR.mlirOperationGetAttributeByName(operation, name)
    if mlirIsNull(raw_attr)
        return nothing
    end
    Attribute(raw_attr)
end
function set_attribute_by_name!(operation, name, attribute)
    LibMLIR.mlirOperationSetAttributeByName(operation, name, attribute)
    operation
end

get_location(operation) = Location(LibMLIR.mlirOperationGetLocation(operation))
get_name(operation) = String(LibMLIR.mlirOperationGetName(operation))
get_block(operation) = Block(LibMLIR.mlirOperationGetBlock(operation), false)
get_parent_operation(operation) = Operation(LibMLIR.mlirOperationGetParentOperation(operation), false)
get_dialect(operation) = first(split(get_name(operation), '.')) |> Symbol

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

op::Operation == other::Operation = LibMLIR.mlirOperationEqual(op, other)

Base.convert(::Type{MlirOperation}, op::Operation) = op.operation

function lose_ownership!(operation::Operation)
    @assert operation.owned
    @atomic operation.owned = false
    operation
end

function Base.show(io::IO, operation::Operation)
    c_print_callback = @cfunction(print_callback, Cvoid, (MlirStringRef, Any))
    ref = Ref(io)
    flags = LibMLIR.mlirOpPrintingFlagsCreate()
    get(io, :debug, false) && LibMLIR.mlirOpPrintingFlagsEnableDebugInfo(flags, true, true)
    GC.@preserve ref LibMLIR.mlirOperationPrintWithFlags(operation, flags, c_print_callback, ref)
    println(io)
end

verify(operation::Operation) = LibMLIR.mlirOperationVerify(operation)

### Block

mutable struct Block
    block::MlirBlock
    @atomic owned::Bool

    Block(block::MlirBlock, owned::Bool=true) = begin
        @assert !mlirIsNull(block) "cannot create Block with null MlirBlock"
        finalizer(new(block, owned)) do block
            if block.owned
                LibMLIR.mlirBlockDestroy(block.block)
            end
        end
    end
end

Block() = Block(MType[], Location[])
function Block(args::Vector{MType}, locs::Vector{Location})
    @assert length(args) == length(locs) "there should be one args for each locs (got $(length(args)) & $(length(locs)))"
    Block(LibMLIR.mlirBlockCreate(length(args), args, locs))
end

function Base.push!(block::Block, op::Operation)
    LibMLIR.mlirBlockAppendOwnedOperation(block, lose_ownership!(op))
    op
end
function Base.insert!(block::Block, pos, op::Operation)
    LibMLIR.mlirBlockInsertOwnedOperation(block, pos - 1, lose_ownership!(op))
    op
end
function Base.pushfirst!(block::Block, op::Operation)
    insert!(block, 1, op)
    op
end
function insert_after!(block::Block, reference::Operation, op::Operation)
    LibMLIR.mlirBlockInsertOwnedOperationAfter(block, reference, lose_ownership!(op))
    op
end
function insert_before!(block::Block, reference::Operation, op::Operation)
    LibMLIR.mlirBlockInsertOwnedOperationBefore(block, reference, lose_ownership!(op))
    op
end

num_arguments(block::Block) =
    LibMLIR.mlirBlockGetNumArguments(block)
function get_argument(block::Block, i)
    i ∉ 1:num_arguments(block) && throw(BoundsError(block, i))
    Value(LibMLIR.mlirBlockGetArgument(block, i - 1))
end
push_argument!(block::Block, type, loc) =
    Value(LibMLIR.mlirBlockAddArgument(block, type, loc))

Base.convert(::Type{MlirBlock}, block::Block) = block.block

function lose_ownership!(block::Block)
    @assert block.owned
    @atomic block.owned = false
    block
end

function Base.show(io::IO, block::Block)
    c_print_callback = @cfunction(print_callback, Cvoid, (MlirStringRef, Any))
    ref = Ref(io)
    GC.@preserve ref LibMLIR.mlirBlockPrint(block, c_print_callback, ref)
end

### Region

mutable struct Region
    region::MlirRegion
    @atomic owned::Bool # TODO: make atomic?

    Region(region, owned=true) = begin
        @assert !mlirIsNull(region)
        finalizer(new(region, owned)) do region
            if region.owned
                LibMLIR.mlirRegionDestroy(region.region)
            end
        end
    end
end

Region() = Region(LibMLIR.mlirRegionCreate())

function Base.push!(region::Region, block::Block)
    LibMLIR.mlirRegionAppendOwnedBlock(region, lose_ownership!(block))
    block
end
function Base.insert!(region::Region, pos, block::Block)
    LibMLIR.mlirRegionInsertOwnedBlock(region, pos - 1, lose_ownership!(block))
    block
end
function Base.pushfirst!(region::Region, block)
    insert!(region, 1, block)
    block
end
insert_after!(region::Region, reference::Block, block::Block) =
    LibMLIR.mlirRegionInsertOwnedBlockAfter(region, reference, lose_ownership!(block))
insert_before!(region::Region, reference::Block, block::Block) =
    LibMLIR.mlirRegionInsertOwnedBlockBefore(region, reference, lose_ownership!(block))

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

Base.convert(::Type{MlirRegion}, region::Region) = region.region

### Module

mutable struct MModule
    module_::MlirModule
    context::Context

    MModule(module_, context) = begin
        @assert !mlirIsNull(module_) "cannot create MModule with null MlirModule"
        finalizer(LibMLIR.mlirModuleDestroy, new(module_, context))
    end
end

MModule(context::Context, loc=Location(context)) =
    MModule(LibMLIR.mlirModuleCreateEmpty(loc), context)
get_operation(module_) = Operation(LibMLIR.mlirModuleGetOperation(module_), false)
get_body(module_) = Block(LibMLIR.mlirModuleGetBody(module_), false)
get_first_child_op(mod::MModule) = get_first_child_op(get_operation(mod))

Base.convert(::Type{MlirModule}, module_::MModule) = module_.module_
Base.parse(::Type{MModule}, context, module_) = MModule(LibMLIR.mlirModuleCreateParse(context, module_), context)

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
    typeid::LibMLIR.MlirTypeID
end

Base.hash(typeid::TypeID) = LibMLIR.mlirTypeIDHashValue(typeid.typeid)
Base.convert(::Type{LibMLIR.MlirTypeID}, typeid::TypeID) = typeid.typeid

@static if isdefined(LibMLIR, :MlirTypeIDAllocator)

### TypeIDAllocator

mutable struct TypeIDAllocator
    allocator::LibMLIR.MlirTypeIDAllocator

    function TypeIDAllocator()
        ptr = LibMLIR.mlirTypeIDAllocatorCreate()
        @assert ptr != C_NULL "cannot create TypeIDAllocator"
        finalizer(LibMLIR.mlirTypeIDAllocatorDestroy, new(ptr))
    end
end

Base.convert(::Type{LibMLIR.MlirTypeIDAllocator}, allocator::TypeIDAllocator) = allocator.allocator

TypeID(allocator::TypeIDAllocator) = TypeID(LibMLIR.mlirTypeIDCreate(allocator))

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
            LibMLIR.mlirPassManagerDestroy(pm.pass)
        end
    end
end

function enable_verifier!(pm)
    LibMLIR.mlirPassManagerEnableVerifier(pm)
    pm
end

PassManager(context) =
    PassManager(LibMLIR.mlirPassManagerCreate(context), context)

function run(pm::PassManager, module_)
    status = LibMLIR.mlirPassManagerRun(pm, module_)
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

OpPassManager(pm::PassManager) = OpPassManager(LibMLIR.mlirPassManagerGetAsOpPassManager(pm), pm)
OpPassManager(pm::PassManager, opname) = OpPassManager(LibMLIR.mlirPassManagerGetNestedUnder(pm, opname), pm)
OpPassManager(opm::OpPassManager, opname) = OpPassManager(LibMLIR.mlirOpPassManagerGetNestedUnder(opm, opname), opm.pass)

Base.convert(::Type{MlirOpPassManager}, op_pass::OpPassManager) = op_pass.op_pass

function Base.show(io::IO, op_pass::OpPassManager)
    c_print_callback = @cfunction(print_callback, Cvoid, (MlirStringRef, Any))
    ref = Ref(io)
    println(io, "OpPassManager(\"\"\"")
    GC.@preserve ref LibMLIR.mlirPrintPassPipeline(op_pass, c_print_callback, ref)
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
    @static if isdefined(LibMLIR, :mlirOpPassManagerAddPipeline)
        io = IOBuffer()
        c_print_callback = @cfunction(print_callback, Cvoid, (MlirStringRef, Any))
        result = GC.@preserve io LibMLIR.mlirOpPassManagerAddPipeline(op_pass, pipeline, c_print_callback, io)
        if LibMLIR.mlirLogicalResultIsFailure(result)
            exc = AddPipelineException(String(take!(io)))
            throw(exc)
        end
    else
        result = LibMLIR.mlirParsePassPipeline(op_pass, pipeline)
        if mlirLogicalResultIsFailure(result)
            throw(AddPipelineException(" " * pipeline))
        end
    end
    op_pass
end

@static if isdefined(LibMLIR, :mlirCreateExternalPass)

### Pass

# AbstractPass interface:
get_opname(::AbstractPass) = ""
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
        LibMLIR.mlirLogicalResultSuccess()
    catch
        LibMLIR.mlirLogicalResultFailure()
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
        LibMLIR.mlirExternalPassSignalFailure(external_pass)
    end
    nothing
end

function create_external_pass!(oppass::OpPassManager, args...)
    create_external_pass!(oppass.pass, args...)
end
function create_external_pass!(manager, pass, name, argument,
                               description, opname=get_opname(pass),
                               dependent_dialects=MlirDialectHandle[])
    passid = TypeID(manager.allocator)
    callbacks = LibMLIR.MlirExternalPassCallbacks(
            @cfunction(_pass_construct, Cvoid, (Any,)),
            @cfunction(_pass_destruct, Cvoid, (Any,)),
            @cfunction(_pass_initialize, LibMLIR.MlirLogicalResult, (MlirContext, Any,)),
            @cfunction(_pass_clone, Any, (Any,)),
            @cfunction(_pass_run, Cvoid, (MlirOperation, LibMLIR.MlirExternalPass, Any))
    )
    pass_handle = manager.passes[passid] = ExternalPassHandle(nothing, pass)
    userdata = Base.pointer_from_objref(pass_handle)
    mlir_pass = LibMLIR.mlirCreateExternalPass(passid, name, argument, description, opname,
                                               length(dependent_dialects), dependent_dialects,
                                               callbacks, userdata)
    mlir_pass
end

function add_owned_pass!(pm::PassManager, pass)
    LibMLIR.mlirPassManagerAddOwnedPass(pm, pass)
    pm
end

function add_owned_pass!(opm::OpPassManager, pass)
    LibMLIR.mlirOpPassManagerAddOwnedPass(opm, pass)
    opm
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
    raw_block = LibMLIR.mlirRegionGetFirstBlock(reg)
    if mlirIsNull(raw_block)
        nothing
    else
        b = Block(raw_block, false)
        (b, b)
    end
end

function Base.iterate(it::BlockIterator, block)
    raw_block = LibMLIR.mlirBlockGetNextInRegion(block)
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
    raw_op = LibMLIR.mlirBlockGetFirstOperation(it.block)
    if mlirIsNull(raw_op)
        nothing
    else
        op = Operation(raw_op, false)
        (op, op)
    end
end

function Base.iterate(it::OperationIterator, op)
    raw_op = LibMLIR.mlirOperationGetNextInBlock(op)
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
    raw_region = LibMLIR.mlirOperationGetFirstRegion(it.op)
    if mlirIsNull(raw_region)
        nothing
    else
        region = Region(raw_region, false)
        (region, region)
    end
end

function Base.iterate(it::RegionIterator, region)
    raw_region = LibMLIR.mlirRegionGetNextInOperation(region)
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

function get_dialects!(dialects::Set{Symbol}, op::Operation)
    push!(dialects, get_dialect(op))

    visit(op) do op
        get_dialects!(dialects, op)
    end

    dialects
end

function get_input_type(module_)
    dialects = Set{Symbol}()

    op = get_operation(module_)
    get_dialects!(dialects, op)

    if :mhlo ∈ dialects
        # :tosa ∉ dialects || throw("cannot have both tosa and mhlo operations")
        :mhlo
    elseif :tosa ∈ dialects
        :tosa
    else
        :none
    end
end

end # module IR
