struct Type
    type::API.MlirType

    function Type(type)
        @assert !mlirIsNull(type) "cannot create Type with null MlirType"
        new(type)
    end
end

Base.convert(::Core.Type{API.MlirType}, type::Type) = type.type

"""
    parse(type; context=context())

Parses a type. The type is owned by the context.
"""
Base.parse(::Core.Type{Type}, s; context::Context=context()) = Type(API.Dispatcher.mlirTypeParseGet(context, s))

"""
    ==(t1, t2)

Checks if two types are equal.
"""
Base.:(==)(a::Type, b::Type) = API.Dispatcher.mlirTypeEqual(a, b)

"""
    context(type)

Gets the context that a type was created with.
"""
context(type::Type) = Context(API.Dispatcher.mlirTypeGetContext(type))

"""
    typeid(type)

Gets the type ID of the type.
"""
typeid(type::Type) = TypeID(API.Dispatcher.mlirTypeGetTypeID(type))

# None type
"""
    Type(::Core.Type{Nothing}; context=context())

Creates a None type in the given context. The type is owned by the context.
"""
Type(::Core.Type{Nothing}; context::Context=context()) = Type(API.Dispatcher.mlirNoneTypeGet(context))

"""
    mlirTypeIsANone(type)

Checks whether the given type is a None type.
"""
isnone(type::Type) = API.Dispatcher.mlirTypeIsANone(type)

# Index type
"""
    IndexType(; context=context())

Creates an index type in the given context. The type is owned by the context.
"""
IndexType(; context::Context=context()) = Type(API.Dispatcher.mlirIndexTypeGet(context))

"""
    isindex(type)

Checks whether the given type is an index type.
"""
isindex(type::Type) = API.Dispatcher.mlirTypeIsAIndex(type)

"""
    Type(T::Core.Type{Bool}; context=context()

Creates a 1-bit signless integer type in the context. The type is owned by the context.
"""
Type(::Core.Type{Bool}; context::Context=context()) = Type(API.Dispatcher.mlirIntegerTypeGet(context, 1))

# Integer types
"""
    Type(T::Core.Type{<:Integer}; context=context()

Creates a signless integer type of the given bitwidth in the context. The type is owned by the context.
"""
Type(T::Core.Type{<:Integer}; context::Context=context()) = Type(API.Dispatcher.mlirIntegerTypeGet(context, sizeof(T) * 8))

"""
    Type(T::Core.Type{<:Signed}; context=context()

Creates a signed integer type of the given bitwidth in the context. The type is owned by the context.
"""
Type(T::Core.Type{<:Signed}; context::Context=context()) = Type(API.Dispatcher.mlirIntegerTypeGet(context, sizeof(T) * 8))

"""
    Type(T::Core.Type{<:Unsigned}; context=context()

Creates an unsigned integer type of the given bitwidth in the context. The type is owned by the context.
"""
Type(T::Core.Type{<:Unsigned}; context::Context=context()) = Type(API.Dispatcher.mlirIntegerTypeUnsignedGet(context, sizeof(T) * 8))

"""
    isinteger(type)

Checks whether the given type is an integer type.
"""
isinteger(type::Type) = API.Dispatcher.mlirTypeIsAInteger(type)

"""
    issigned(type)

Checks whether the given integer type is signed.
"""
issigned(type::Type) = API.Dispatcher.mlirIntegerTypeIsSigned(type)

"""
    issignless(type)

Checks whether the given integer type is signless.
"""
issignless(type::Type) = API.Dispatcher.mlirIntegerTypeIsSignless(type)

"""
    isunsigned(type)

Checks whether the given integer type is unsigned.
"""
isunsigned(type::Type) = API.Dispatcher.mlirIntegerTypeIsUnsigned(type)

"""
    bitwidth(type)

Returns the bitwidth of an integer type.
"""
function bitwidth(type::Type)
    @assert isinteger(type) "expected an integer type"
    API.Dispatcher.mlirIntegerTypeGetWidth(type)
end

# Floating point types
"""
      Float8E5M2(; context=context())

  Creates an f8E5M2 type in the given context. The type is owned by the context.
  """
function Float8E5M2(; context::Context=context())
    MLIR_VERSION[] >= v"16" || throw(UndefVarError("`Float8E5M2()` requires MLIR version 16 or later"))
    Type(API.Dispatcher.mlirFloat8E5M2TypeGet(context))
end

"""
      Float8E4M3FN(; context=context())

  Creates an f8E4M3FN type in the given context. The type is owned by the context.
  """
function Float8E4M3FN(; context::Context=context())
    MLIR_VERSION[] >= v"16" || throw(UndefVarError("`Float8E4M3FN()` requires MLIR version 16 or later"))
    Type(API.Dispatcher.mlirFloat8E4M3FNTypeGet(context))
end

"""
BFloat16Type(; context=context())

Creates a bf16 type in the given context. The type is owned by the context.
"""
BFloat16Type(; context::Context=context()) = Type(API.Dispatcher.mlirBF16TypeGet(context))

"""
    Type(::Core.Type{Float16}; context=context())

Creates an f16 type in the given context. The type is owned by the context.
"""
Type(::Core.Type{Float16}; context::Context=context()) = Type(API.Dispatcher.mlirF16TypeGet(context))

"""
    Type(Core.Type{Float32}; context=context())

Creates an f32 type in the given context. The type is owned by the context.
"""
Type(::Core.Type{Float32}; context::Context=context()) = Type(API.Dispatcher.mlirF32TypeGet(context))

"""
    Type(Core.Type{Float64}; context=context())

Creates a f64 type in the given context. The type is owned by the context.
"""
Type(::Core.Type{Float64}; context::Context=context()) = Type(API.Dispatcher.mlirF64TypeGet(context))

"""
      isf8e5m2(type)

  Checks whether the given type is an f8E5M2 type.
  """
function isf8e5m2(type::Type)
    MLIR_VERSION[] >= v"16" || throw(UndefVarError("`isf8e5m2()` requires MLIR version 16 or later"))
    API.Dispatcher.mlirTypeIsAFloat8E5M2(type)
end

"""
      isf8e4m3fn(type)

  Checks whether the given type is an f8E4M3FN type.
  """
function isf8e4m3fn(type::Type)
    MLIR_VERSION[] >= v"16" || throw(UndefVarError("`isf8e4m3fn()` requires MLIR version 16 or later"))
    API.Dispatcher.mlirTypeIsAFloat8E4M3FN(type)
end

"""
    isbf16(type)

Checks whether the given type is a bf16 type.
"""
isbf16(type::Type) = API.Dispatcher.mlirTypeIsABF16(type)

"""
    isf16(type)

Checks whether the given type is an f16 type.
"""
isf16(type::Type) = API.Dispatcher.mlirTypeIsAF16(type)

"""
    isf32(type)

Checks whether the given type is an f32 type.
"""
isf32(type::Type) = API.Dispatcher.mlirTypeIsAF32(type)

"""
    isf64(type)

Checks whether the given type is an f64 type.
"""
isf64(type::Type) = API.Dispatcher.mlirTypeIsAF64(type)

# Complex types
"""
    Type(Complex{T}) where {T}

Creates a complex type with the given element type in the same context as the element type. The type is owned by the context.
"""
Type(::Core.Type{Complex{T}}) where {T} = Type(API.Dispatcher.mlirComplexTypeGet(Type(T)))

"""
    iscomplex(type)

Checks whether the given type is a Complex type.
"""
iscomplex(type::Type) = API.Dispatcher.mlirTypeIsAComplex(type)

# Shaped types
"""
    isshaped(type)

Checks whether the given type is a Shaped type.
"""
isshaped(type::Type) = API.Dispatcher.mlirTypeIsAShaped(type)

"""
    hasrank(type)

Checks whether the given shaped type is ranked.
"""
hasrank(type::Type) = API.Dispatcher.mlirShapedTypeHasRank(type)

"""
    ndims(type)

Returns the rank of the given ranked shaped type.
"""
function Base.ndims(type::Type)
    @assert isshaped(type) && hasrank(type) "expected a ranked shaped type"
    API.Dispatcher.mlirShapedTypeGetRank(type)
end

"""
    hasstaticshape(type)

Checks whether the given shaped type has a static shape.
"""
hasstaticshape(type::Type) = API.Dispatcher.mlirShapedTypeHasStaticShape(type)

"""
    isdyndim(type, i)

Checks wither the `i`-th dimension of the given shaped type is dynamic.
"""
isdyndim(type::Type, i::Int) = API.Dispatcher.mlirShapedTypeIsDynamicDim(type, i - 1)

"""
    size(type, i)

Returns the `i`-th dimension of the given ranked shaped type.
"""
function Base.size(type::Type, i::Int)
    @assert isshaped(type) "expected a shaped type"
    API.Dispatcher.mlirShapedTypeGetDimSize(type, i - 1)
end

Base.size(type::Type) = Tuple(size(type, i) for i in 1:ndims(type))

"""
    isdynsize(size)

Checks whether the given value is used as a placeholder for dynamic sizes in shaped types.
"""
isdynsize(size) = API.Dispatcher.mlirShapedTypeIsDynamicSize(size)

"""
    dynsize()

Returns the value indicating a dynamic size in a shaped type. Prefer [`isdynsize`](@ref) to direct comparisons with this value.
"""
dynsize() = API.Dispatcher.mlirShapedTypeGetDynamicSize()

"""
    mlirShapedTypeIsDynamicStrideOrOffset(val)

Checks whether the given value is used as a placeholder for dynamic strides and offsets in shaped types.
"""
isdynstrideoroffset(val) = API.Dispatcher.mlirShapedTypeIsDynamicStrideOrOffset(val)

"""
    mlirShapedTypeGetDynamicStrideOrOffset()

Returns the value indicating a dynamic stride or offset in a shaped type. Prefer [`isdynstrideoroffset`](@ref) to direct comparisons with this value.
"""
dynstrideoroffset() = API.Dispatcher.mlirShapedTypeGetDynamicStrideOrOffset()

# Vector type
"""
    VectorType(rank, shape, elementType; location=Location(), check=false)

Creates a vector type of the shape identified by its rank and dimensions, with the given element type in the same context as the element type.
The type is owned by the context. If `check=true`, emits appropriate diagnostics on illegal arguments.
"""
function VectorType(rank, shape, elem_type; location::Location=Location(), check::Bool=false)
    Type(check ? API.Dispatcher.mlirVectorTypeGetChecked(location, rank, shape, elem_type) : API.Dispatcher.mlirVectorTypeGet(rank, shape, elem_type))
end

"""
    isvector(type)

Checks whether the given type is a Vector type.
"""
isvector(type::Type) = API.Dispatcher.mlirTypeIsAVector(type)

# Tensor type
"""
    TensorType(shape, elementType, encoding=Attribute(); location=Location(), check=false)

Creates a tensor type of a fixed rank with the given shape, element type, and optional encoding in the same context as the element type.
The type is owned by the context. Tensor types without any specific encoding field should assign [`mlirAttributeGetNull`](@ref) to this parameter.
If `check=true`, emits appropriate diagnostics on illegal arguments.
"""
function TensorType(shape, elem_type, encoding=Attribute(); location::Location=Location(), check::Bool=false)
    rank = length(shape)
    shape = shape isa AbstractVector ? shape : collect(shape)
    Type(check ? API.Dispatcher.mlirRankedTensorTypeGetChecked(location, rank, shape, elem_type, encoding) : API.Dispatcher.mlirRankedTensorTypeGet(rank, shape, elem_type, encoding))
end

"""
    TensorType(elementType)

Creates an unranked tensor type with the given element type in the same context as the element type. The type is owned by the context.
If `check=true`, emits appropriate diagnostics on illegal arguments.
"""
function TensorType(elem_type; location::Location=Location(), check::Bool=false)
    Type(check ? API.Dispatcher.mlirUnrankedTensorTypeGetChecked(location, elem_type) : API.Dispatcher.mlirUnrankedTensorTypeGet(elem_type))
end

# TODO maybe add these helper methods?
# Type(a::AbstractArray{T}) where {T} = Type(Type(T), size(a))
# Type(::Core.Type{<:AbstractArray{T,N}}, dims) where {T,N} =
#     Type(API.Dispatcher.mlirRankedTensorTypeGetChecked(
#         Location(),
#         N, collect(dims),
#         Type(T),
#         Attribute(),
#     ))
# Type(element_type::Type, dims) =
#     Type(API.Dispatcher.mlirRankedTensorTypeGetChecked(
#         Location(),
#         length(dims), collect(dims),
#         element_type,
#         Attribute(),
#     ))

"""
    istensor(type)

Checks whether the given type is a Tensor type.
"""
istensor(type::Type) = API.Dispatcher.mlirTypeIsATensor(type)

"""
    isrankedtensor(type)

Checks whether the given type is a ranked tensor type.
"""
isrankedtensor(type::Type) = API.Dispatcher.mlirTypeIsARankedTensor(type)

"""
    isunrankedtensor(type)

Checks whether the given type is an unranked tensor type.
"""
isunrankedtensor(type::Type) = API.Dispatcher.mlirTypeIsAUnrankedTensor(type)

"""
    encoding(type)

Gets the 'encoding' attribute from the ranked tensor type, returning a `nothing` if none.
"""
function encoding(type::Type)
    @assert isrankedtensor(type) "expected a ranked tensor type"
    attr = API.Dispatcher.mlirRankedTensorTypeGetEncoding(type)
    mlirIsNull(attr) ? nothing : Attribute(attr)
end

# Memref type
"""
    MemRefType(elementType, rank, shape, layout, memorySpace; location=Location(), check=false)

Creates a MemRef type with the given rank and shape, a potentially empty list of affine layout maps, the given memory space and element type, in the same context as element type.
The type is owned by the context. If `check=true`, emits appropriate diagnostics on illegal arguments.
"""
function MemRefType(elem_type::Type, shape, layout, memspace; location::Location=Location(), check::Bool=false)
    if check
        Type(API.Dispatcher.mlirMemRefTypeGetChecked(location, elem_type, length(shape), pointer(shape), layout, memspace))
    else
        Type(API.Dispatcher.mlirMemRefTypeGet(elem_type, length(shape), pointer(shape), layout, memspace))
    end
end

"""
    MemRefType(elementType, rank, shape, memorySpace; location=Location(), check=false)

Creates a MemRef type with the given rank, shape, memory space and element type in the same context as the element type.
The type has no affine maps, i.e. represents a default row-major contiguous memref. The type is owned by the context.
If `check=true`, emits appropriate diagnostics on illegal arguments.
"""
function MemRefType(elem_type::Type, shape, memspace; location::Location=Location(), check::Bool=false)
    if check
        Type(API.Dispatcher.mlirMemRefTypeContiguousGetChecked(location, elem_type, length(shape), pointer(shape), memspace))
    else
        Type(API.Dispatcher.mlirMemRefTypeContiguousGet(elem_type, length(shape), pointer(shape), memspace))
    end
end

"""
    MemRefType(elementType, memorySpace)

Creates an Unranked MemRef type with the given element type and in the given memory space. The type is owned by the context of element type.
If `check=true`, emits appropriate diagnostics on illegal arguments.
"""
function MemRefType(elem_type::Type, memspace; location::Location=Location(), check::Bool=false)
    if check
        Type(API.Dispatcher.mlirUnrankedMemRefTypeGetChecked(location, elem_type, memspace))
    else
        Type(API.Dispatcher.mlirUnrankedMemRefTypeGet(elem_type, memspace))
    end
end

MemRefType(T::Core.Type, args...; kwargs...) = MemRefType(Type(T), args...; kwargs)

"""
    ismemref(type)

Checks whether the given type is a MemRef type.
"""
ismemref(type::Type) = API.Dispatcher.mlirTypeIsAMemRef(type)

"""
    mlirTypeIsAUnrankedMemRef(type)

Checks whether the given type is an UnrankedMemRef type.
"""
isunrankedmemref(type::Type) = API.Dispatcher.mlirTypeIsAUnrankedMemRef(type)

"""
    layout(type)

Returns the layout of the given MemRef type.
"""
function layout(type::Type)
    @assert ismemref(type) "expected a MemRef type"
    Attribute(API.Dispatcher.mlirMemRefTypeGetLayout(type))
end

"""
    affinemap(type)

Returns the affine map of the given MemRef type.
"""
function affinemap(type::Type)
    @assert ismemref(type) "expected a MemRef type"
    AffineMap(API.Dispatcher.mlirMemRefTypeGetAffineMaps(type))
end

"""
    mlirMemRefTypeGetMemorySpace(type)

Returns the memory space of the given MemRef type.
"""
function memspace(type::Type)
    @assert ismemref(type) "expected a MemRef type"
    if isunrankedmemref(type)
        Attribute(API.Dispatcher.mlirUnrankedMemrefGetMemorySpace(type))
    else
        Attribute(API.Dispatcher.mlirMemRefTypeGetMemorySpace(type))
    end
end

# Tuple type
"""
    Type(elements; context=context())
    Type(::Core.Type{<:Tuple{T...}}; context=context())

Creates a tuple type that consists of the given list of elemental types. The type is owned by the context.
"""
Type(elements::Vector{Type}; context::Context=context()) = Type(API.Dispatcher.mlirTupleTypeGet(context, length(elements), pointer(elements)))
Type(@nospecialize(elements::NTuple{N,Type}); context::Context=context()) where {N} = Type(collect(elements); context)
Type(T::Core.Type{<:Tuple}; context::Context=context()) = Type(map(Type, T.parameters); context)

"""
    istuple(type)

Checks whether the given type is a tuple type.
"""
istuple(type::Type) = API.Dispatcher.mlirTypeIsATuple(type)

# Function type
"""
    isfunction(type)

Checks whether the given type is a function type.
"""
isfunction(type::Type) = API.Dispatcher.mlirTypeIsAFunction(type)

"""
    FunctionType(inputs, results; context=context())

Creates a function type, mapping a list of input types to result types.
"""
function FunctionType(inputs, results; context::Context=context())
    Type(API.Dispatcher.mlirFunctionTypeGet(context, length(inputs), pointer(inputs), length(results), pointer(results)))
end

# TODO maybe add this helper method?
# Type(ft::Pair) = Type(API.Dispatcher.mlirFunctionTypeGet(context(),
#     length(ft.first), [Type(t) for t in ft.first],
#     length(ft.second), [Type(t) for t in ft.second]))

"""
    ninputs(type)

Returns the number of input types.
"""
function ninputs(type::Type)
    @assert isfunction(type) "cannot get the number of inputs on type $(type), expected a function type"
    API.Dispatcher.mlirFunctionTypeGetNumInputs(type)
end

"""
    nresults(type)

Returns the number of result types.
"""
function nresults(type::Type)
    @assert isfunction(type) "cannot get the number of results on type $(type), expected a function type"
    API.Dispatcher.mlirFunctionTypeGetNumResults(type)
end

"""
    input(type, i)

Returns the `i`-th input type.
"""
function input(type::Type, i)
    @assert isfunction(type) "cannot get input on type $(type), expected a function type"
    Type(API.Dispatcher.mlirFunctionTypeGetInput(type, i - 1))
end

"""
    result(type, i)

Returns the `i`-th result type.
"""
function result(type::Type, i=1)
    @assert isfunction(type) "cannot get result on type $(type), expected a function type"
    Type(API.Dispatcher.mlirFunctionTypeGetResult(type, i - 1))
end

# Opaque type
"""
    OpaqueType(dialectNamespace, typeData; context=context())

Creates an opaque type in the given context associated with the dialect identified by its namespace. The type contains opaque byte data of the specified length (data need not be null-terminated).
"""
OpaqueType(namespace, data; context::Context=context()) = Type(API.Dispatcher.mlirOpaqueTypeGet(context, namespace, data))

"""
    isopaque(type)

Checks whether the given type is an opaque type.
"""
isopaque(type::Type) = API.Dispatcher.mlirTypeIsAOpaque(type)

"""
    mlirOpaqueTypeGetDialectNamespace(type)

Returns the namespace of the dialect with which the given opaque type is associated. The namespace string is owned by the context.
"""
namespace(type::Type) = String(API.Dispatcher.mlirOpaqueTypeGetDialectNamespace(type))

"""
    mlirOpaqueTypeGetData(type)

Returns the raw data as a string reference. The data remains live as long as the context in which the type lives.
"""
data(type::Type) = String(API.Dispatcher.mlirOpaqueTypeGetData(type))

function Base.eltype(type::Type)
    if isshaped(type)
        Type(API.Dispatcher.mlirShapedTypeGetElementType(type))
    elseif iscomplex(type)
        Type(API.Dispatcher.mlirComplexTypeGetElementType(type))
    else
        type
    end
end

function Base.length(type::Type)
    if istuple(type)
        API.Dispatcher.mlirTupleTypeGetNumTypes(type)
    else
        nothing
    end
end

function Base.getindex(type::Type, i)
    if istuple(type)
        Type(API.Dispatcher.mlirTupleTypeGetType(type, i - 1))
    else
        nothing
    end
end

function julia_type(type::Type)
    if isinteger(type)
        width = bitwidth(type)
        if issignless(type) || issigned(type)
            if width == 1
                return Bool
            elseif width == 8
                return Int8
            elseif width == 16
                return Int16
            elseif width == 32
                return Int32
            elseif width == 64
                return Int64
            elseif width == 128
                return Int128
            else
                throw("could not convert signed $width-bit integer type to julia")
            end
        else
            if width == 8
                return UInt8
            elseif width == 16
                return UInt16
            elseif width == 32
                return UInt32
            elseif width == 64
                return UInt64
            elseif width == 128
                return UInt128
            else
                throw("could not convert unsigned $width-bit integer type to julia")
            end
        end
    elseif isf16(type)
        Float16
    elseif isf32(type)
        Float32
    elseif isf64(type)
        Float64
    elseif isnone(type)
        Nothing
    elseif iscomplex(type)
        Complex{julia_type(eltype(type))}
    elseif isshaped(type)
        if !hasrank(type)
            throw("don't know yet how to convert unranked tensor type to julia")
        end
        T = julia_type(eltype(type))
        N = ndims(type)
        AbstractArray{T,N}
    elseif istuple(type)
        Tuple{[julia_type(type[i]) for i in 1:length(type)]...}
    else
        throw("could not convert type $type to julia")
    end
end

function Base.show(io::IO, type::Type)
    print(io, "Type(#= ")
    c_print_callback = @cfunction(print_callback, Cvoid, (API.MlirStringRef, Any))
    ref = Ref(io)
    API.Dispatcher.mlirTypePrint(type, c_print_callback, ref)
    print(io, " =#)")
end
