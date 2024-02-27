struct Value
    value::API.MlirValue

    Value(value) = begin
        @assert !API.mlirValueIsNull(value) "cannot create Value with null MlirValue"
        new(value)
    end
end

Base.convert(::Core.Type{API.MlirValue}, value::Value) = value.value
Base.size(value::Value) = Base.size(get_type(value))
Base.ndims(value::Value) = Base.ndims(get_type(value))

Base.:(==)(a::Value, b::Value) = API.mlirValueEqual(a, b)
is_block_arg(value::Value) = API.mlirValueIsABlockArgument(value)
is_op_res(value::Value) = API.mlirValueIsAOpResult(value)

function block_owner(value::Value)
    @assert is_block_arg(value) "could not get owner, value is not a block argument"
    Block(API.mlirBlockArgumentGetOwner(value), false)
end

function op_owner(value::Value)
    @assert is_op_res(value) "could not get owner, value is not an op result"
    Operation(API.mlirOpResultGetOwner(value), false)
end

function owner(value::Value)
    if is_block_arg(value)
        raw_block = API.mlirBlockArgumentGetOwner(value)
        mlirValueIsNull(raw_block) && return nothing
        return Block(raw_block, false)
    elseif is_op_res(value)
        raw_op = API.mlirOpResultGetOwner(value)
        mlirValueIsNull(raw_op) && return nothing
        return Operation(raw_op, false)
    else
        error("Value is neither a block argument nor an op result")
    end
end

function block_arg_num(value::Value)
    @assert is_block_arg(value) "could not get arg number, value is not a block argument"
    API.mlirBlockArgumentGetArgNumber(value)
end

function op_res_num(value::Value)
    @assert is_op_res(value) "could not get result number, value is not an op result"
    API.mlirOpResultGetResultNumber(value)
end

function position(value::Value)
    if is_block_arg(value)
        return block_arg_num(value)
    elseif is_op_res(value)
        return op_res_num(value)
    else
        error("Value is neither a block argument nor an op result")
    end
end

Type(value::Value) = Type(API.mlirValueGetType(value))

function set_type!(value, type)
    @assert is_a_block_argument(value) "could not set type, value is not a block argument"
    API.mlirBlockArgumentSetType(value, type)
    value
end

function Base.show(io::IO, value::Value)
    c_print_callback = @cfunction(print_callback, Cvoid, (MlirStringRef, Any))
    ref = Ref(io)
    API.mlirValuePrint(value, c_print_callback, ref)
end
