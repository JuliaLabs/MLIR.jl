# EXCLUDE FROM TESTING

import MLIR.IR
using MLIR.IR: Value, Attribute, result, Operation, Convertible, context, IndexType, ValueTrait
import MLIR.Dialects
using MLIR.API: mlirMemRefTypeGet, mlirStridedLayoutAttrGet, mlirRankedTensorTypeGet, mlirIntegerTypeGet, mlirShapedTypeGetDynamicSize, mlirF64TypeGet, mlirF32TypeGet, mlirF16TypeGet
import MLIR.Generate
import MLIR.Generate: @intrinsic, BoolTrait

### int ###
struct MLIRInteger{N} <: Integer
    value::Value
    MLIRInteger{N}(i::Value) where {N} = new(i)
end
ValueTrait(::Type{<:MLIRInteger}) = Convertible()
IR.Type(::Type{MLIRInteger{N}}) where {N} = IR.Type(mlirIntegerTypeGet(context(), N))

const i1 = MLIRInteger{1}
BoolTrait(::Type{i1}) = Generate.Boollike()

const i8 = MLIRInteger{8}
const i16 = MLIRInteger{16}
const i32 = MLIRInteger{32}
const i64 = MLIRInteger{64}

@intrinsic Base.:+(a::T, b::T) where {T<:MLIRInteger} = T(Dialects.arith.addi(a, b)|>result)
@intrinsic Base.:-(a::T, b::T) where {T<:MLIRInteger} = T(Dialects.arith.subi(a, b)|>result)
@intrinsic Base.:*(a::T, b::T) where {T<:MLIRInteger} = T(Dialects.arith.muli(a, b)|>result)
@intrinsic Base.:/(a::T, b::T) where {T<:MLIRInteger} = T(Dialects.arith.divi(a, b)|>result)

@intrinsic Base.:>(a::T, b::T) where {T<:MLIRInteger} = i1(Dialects.arith.cmpi(a, b, result=IR.Type(i1), predicate=4))
@intrinsic Base.:>=(a::T, b::T) where {T<:MLIRInteger} = i1(Dialects.arith.cmpi(a, b, result=IR.Type(i1), predicate=5))
@intrinsic Base.:<(a::T, b::T) where {T<:MLIRInteger} = i1(Dialects.arith.cmpi(a, b, result=IR.Type(i1), predicate=2))
@intrinsic Base.:<=(a::T, b::T) where {T<:MLIRInteger} = i1(Dialects.arith.cmpi(a, b, result=IR.Type(i1), predicate=3))


# promote constant julia integers to int
@intrinsic i64(x::Integer) = i64(Dialects.arith.constant(value=Attribute(Int64(x)), result=IR.Type(i64))|>result)
@intrinsic i32(x::Integer) = i32(Dialects.arith.constant(value=Attribute(Int32(x)), result=IR.Type(i32))|>result)
@intrinsic i16(x::Integer) = i16(Dialects.arith.constant(value=Attribute(Int16(x)), result=IR.Type(i16))|>result)
@intrinsic i8(x::Integer) = i8(Dialects.arith.constant(value=Attribute(Int8(x)), result=IR.Type(i8))|>result)

Base.promote_rule(::Type{T}, ::Type{I}) where {T<:MLIRInteger, I<:Integer} = T
Base.convert(::Type{T}, x::T) where {T <: MLIRInteger} = x
@intrinsic function Base.convert(::Type{T}, x::Integer)::T where {T<:MLIRInteger}
    op = Dialects.arith.constant(value=Attribute(x), result=IR.Type(T))
    T(result(op))
end

### float ###
abstract type MLIRFloat <: AbstractFloat end
ValueTrait(::Type{<:MLIRFloat}) = Convertible()

struct MLIRF64 <: MLIRFloat
    value::Value
end
struct MLIRF32 <: MLIRFloat
    value::Value
end
struct MLIRF16 <: MLIRFloat
    value::Value
end

const f64 = MLIRF64
const f32 = MLIRF32
const f16 = MLIRF16

IR.Type(::Type{MLIRF64}) = IR.Type(mlirF64TypeGet(context()))
IR.Type(::Type{MLIRF32}) = IR.Type(mlirF32TypeGet(context()))
IR.Type(::Type{MLIRF16}) = IR.Type(mlirF16TypeGet(context()))

@intrinsic (Base.:+(a::T, b::T)::T) where {T<:MLIRFloat} = T(Dialects.arith.addf(a, b)|>result)
@intrinsic (Base.:-(a::T, b::T)::T) where {T<:MLIRFloat} = T(Dialects.arith.subf(a, b)|>result)
@intrinsic (Base.:*(a::T, b::T)::T) where {T<:MLIRFloat} = T(Dialects.arith.mulf(a, b)|>result)
@intrinsic (Base.:/(a::T, b::T)::T) where {T<:MLIRFloat} = T(Dialects.arith.divf(a, b)|>result)

# TODO: 
# @intrinsic Base.:>(a::T, b::T)::i1 where {T<:MLIRFloat} = i1(Dialects.arith.cmpf(a, b, predicate=...))
# @intrinsic Base.:>=(a::T, b::T)::i1 where {T<:MLIRFloat} = i1(Dialects.arith.cmpf(a, b, predicate=...))
# @intrinsic Base.:<(a::T, b::T)::i1 where {T<:MLIRFloat} = i1(Dialects.arith.cmpf(a, b, predicate=...))
# @intrinsic Base.:<=(a::T, b::T)::i1 where {T<:MLIRFloat} = i1(Dialects.arith.cmpf(a, b, predicate=...))

f32(x::f32) = x
@intrinsic f32(x::Real) = f32(Dialects.arith.constant(value=IR.Attribute(Float32(x)), result=IR.Type(f32)) |> result)
Base.convert(::Type{f32}, x::Real) = f32(x)
Base.promote_rule(::Type{f32}, ::Type{<:Real}) = f32

### index  ###
struct MLIRIndex <: Integer
    value::Value
end
const index = MLIRIndex
IR.Type(::Type{MLIRIndex}) = IndexType()
ValueTrait(::Type{<:MLIRIndex}) = Convertible()

@intrinsic Base.:+(a::index, b::index)::index = index(Dialects.index.add(a, b)|>result)
@intrinsic Base.:-(a::index, b::index)::index = index(Dialects.index.sub(a, b)|>result)
@intrinsic Base.:*(a::index, b::index)::index = index(Dialects.index.mul(a, b)|>result)
@intrinsic Base.:/(a::index, b::index)::index = index(Dialects.index.divs(a, b)|>result)

# TODO:
# @intrinsic Base.:>(a::index, b::index)::i1 = i1(Dialects.index.cmp(a, b, predicate=...)|>result)
# @intrinsic Base.:>=(a::index, b::index)::i1 = i1(Dialects.index.cmp(a, b, predicate=...)|>result)
# @intrinsic Base.:<(a::index, b::index)::i1 = i1(Dialects.index.cmp(a, b, predicate=...)|>result)
# @intrinsic Base.:<=(a::index, b::index)::i1 = i1(Dialects.index.cmp(a, b, predicate=...)|>result)

# promote constant julia integers to index
@intrinsic index(x::Integer) = index(Dialects.index.constant(value=Attribute(x, IR.Type(index)), result=IR.Type(index))|>result)
Base.promote_rule(::Type{index}, ::Type{I}) where {I<:Integer} = index
function Base.convert(::Type{index}, x::Integer)::index
    index(x)
end

@intrinsic i64(x::index) = i64(Dialects.index.casts(x, output=IR.Type(i64))|>result)
@intrinsic index(x::i64) = index(Dialects.index.casts(x, output=IR.Type(index))|>result)

### abstract type for array-like types ###
abstract type MLIRArrayLike{T, N} <: AbstractArray{T, N} end

# implementation detail: reinterpret shouldn't try reinterpreting individual elements:
function Base.reinterpret(::Type{Tuple{A}}, array::A) where {A<:MLIRArrayLike}
    return (array, )
end
ValueTrait(::Type{<:MLIRArrayLike}) = Convertible()
Base.show(io::IO, a::A) where {A<:MLIRArrayLike{T, N}} where {T, N} = print(io, "$A[...]")
Base.show(io::IO, ::MIME{Symbol("text/plain")}, a::A) where {A<:MLIRArrayLike{T, N}} where {T, N} = print(io, "$A[...]")

### memref ###
struct MLIRMemref{T, N, Shape, Memspace, Stride, Offset} <: MLIRArrayLike{T, N}
    value::Value
end
function IR.Type(::Type{MLIRMemref{T, N, Shape, Memspace, Stride, Offset}}) where {T, N, Shape, Memspace, Stride, Offset}
    memspace(a::Attribute) = a
    memspace(::Nothing) = Attribute()
    memspace(i::Integer) = Attribute(i)

    shape(::Nothing) = Int[mlirShapedTypeGetDynamicSize() for _ in 1:N]
    shape(s) = Int[s.parameters...]

    # default to column-major layout
    stride(::Nothing) = Int[1, [mlirShapedTypeGetDynamicSize() for _ in 2:N]...]
    stride(s) = shape(s)

    offset(::Nothing) = mlirShapedTypeGetDynamicSize()
    offset(i::Integer) = i

    IR.Type(mlirMemRefTypeGet(
        IR.Type(T),
        N,
        shape(Shape),
        Attribute(mlirStridedLayoutAttrGet(
            context().context,
            offset(Offset),
            N,
            stride(Stride))),
        memspace(Memspace)
    ))

end
const memref{T, N} = MLIRMemref{T, N, nothing, nothing, nothing, 0}

Base.size(A::MLIRMemref{T, N, Shape}) where {T, N, Shape} = Tuple(Shape.parameters)

@intrinsic function Base.getindex(A::MLIRMemref{T, 1}, i::index)::T where T
    oneoff = Dialects.index.constant(; value=Attribute(1, IndexType())) |> result
    new_index = Dialects.index.sub(i, oneoff) |> result
    T(Dialects.memref.load(A, [new_index]) |> result)
end
function Base.getindex(A::MLIRMemref{T}, i::Int)::T where T
    A[index(i)]
end

@intrinsic function Base.setindex!(A::MLIRMemref{T, 1}, v::T, i::index)::T where T
    oneoff = Dialects.index.constant(; value=Attribute(1, IndexType())) |> result
    new_index = Dialects.index.sub(i, oneoff) |> IR.result
    Dialects.memref.store(v, A, [new_index])
    return v
end
@intrinsic function Base.setindex!(A::MLIRMemref{T, 1}, v, i::Int)::T where {T}
    # this method can only be called with constant i since we assume arguments to `code_mlir` to be MLIR types, not Julia types.
    i = index(Dialects.index.constant(; value=Attribute(i, IndexType())) |> result)
    A[i] = v
end

### tensor ###
struct MLIRTensor{T, N} <: MLIRArrayLike{T, N}
    value::Value
end
IR.Type(::Type{MLIRTensor{T, N}}) where {T, N} = mlirRankedTensorTypeGet(
    N,
    Int[mlirShapedTypeGetDynamicSize() for _ in 1:N],
    IR.Type(T),
    Attribute()) |> IR.Type
const tensor = MLIRTensor
