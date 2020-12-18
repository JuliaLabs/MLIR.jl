module IR

using ..MLIR

# ------------ Context alias and APIs ------------ #

const Context = MLIR.API.MlirContext

create_context() = MLIR.API.mlirContextCreate()
num_loaded_dialects(ctx::Context) = convert(Int64, MLIR.API.mlirContextGetNumLoadedDialects(ctx))
num_registered_dialects(ctx::Context) = convert(Int64, MLIR.API.mlirContextGetNumRegisteredDialects(ctx))
is_null(ctx::Context) = MLIR.API.mlirContextIsNull(ctx)
destroy!(ctx::Context) = MLIR.API.mlirContextDestroy(ctx)
get_allow_unregistered_dialects(ctx::Context) = MLIR.API.mlirContextGetAllowUnregisteredDialects(ctx)
unknown_get(ctx::Context) = MLIR.API.mlirLocationUnknownGet(ctx)

register_all_dialects!(ctx::Context) = MLIR.API.MLIR.API.mlirRegisterAllDialects(ctx)
register_standard_dialect!(ctx::Context) = MLIR.API.MLIR.API.mlirContextRegisterStandardDialect(ctx)
load_standard_dialect!(ctx::Context) = MLIR.API.MLIR.API.mlirContextLoadStandardDialect(ctx)

Context() = create_context()

@doc(
"""
const Context = MLIR.API.MlirContext
""", Context)

# ------------ Dialect alias and APIs ------------ #

const Dialect = MLIR.API.MlirDialect

get_context(d::Dialect) = MLIR.API.mlirDialectGetContext(d)
is_null(d::Dialect) = MLIR.API.mlirDialectIsNull(d)
Base.:(==)(d1::Dialect, d2::Dialect) = MLIR.API.mlirDialectEqual(d1, d2)
get_namespace(d::Dialect) = MLIR.API.mlirDialectGetNamespace(d)

@doc(
"""
const Dialect = MLIR.API.MlirDialect
""", Dialect)

# ------------ Type alias and APIs ------------ #

const Type = MLIR.API.MlirType

get_context(t::Type) = MLIR.API.mlirTypeGetContext(t)
is_null(t::Type) = MLIR.API.mlirTypeIsNull(t)
Base.:(==)(t1::Type, t2::Type) = MLIR.API.mlirTypeEqual(t1, t2)
dump(t::Type) = MLIR.API.mlirTypeDump(t)
parse_get(ctx::Context, t::Type) = MLIR.API.mlirTypeParseGet(ctx, t)

@doc(
"""
const Type = MLIR.API.MlirType
""", Type)

# ------------ Location alias and APIs ------------ #

const Location = MLIR.API.MlirLocation

get_context(l::Location) = MLIR.API.mlirLocationGetContext(l)
is_null(l::Location) = MLIR.API.mlirLocationIsNull(l)
Base.:(==)(l1::Location, l2::Location) = MLIR.API.mlirLocationEquation(l1, l2)

@doc(
"""
const Location = MLIR.API.MlirLocation
""", Location)

# ------------ Attribute alias and APIs ------------ #

const Attribute = MLIR.API.MlirAttribute

get_context(attr::Attribute) = MLIR.API.mlirAttributeGetContext(attr)
get_type(attr::Attribute) = MLIR.API.mlirAttributeGetType(attr)
is_null(attr::Attribute) = MLIR.API.mlirAttributeIsNull(attr)
Base.:(==)(attr1::Attribute, attr2::Attribute) = MLIR.API.mlirAttributeEqual(attr1, attr2)
dump(attr::Attribute) = MLIR.API.mlirAttributeDump(attr)
parse_get(ctx::Context, attr::Attribute) = MLIR.API.mlirAttributeParseGet(ctx, attr)

@doc(
"""
const Attribute = MLIR.API.MlirAttribute
""", Attribute)

# ------------ Operation state alias and APIs ------------ #

const OperationState = MLIR.API.MlirOperationState

@doc(
"""
const OperationState = MLIR.API.MlirOperationState
""", OperationState)

# ------------ Operation alias and APIs ------------ #

const Operation = MLIR.API.MlirOperation

@doc(
"""
const Operation = MLIR.API.MlirOperation
""", Operation)

# ------------ Value alias and APIs ------------ #

const Value = MLIR.API.MlirValue

get_type(v::Value) = MLIR.API.mlirValueGetType(v)
get_owner_block(v::Value) = MLIR.API.mlirBlockArgumentGetOwner(v)
get_owner_block_arg_index(v::Value) = MLIR.API.mlirBlockArgumentGetArgNumber(v)
get_owner_op(v::Value) = MLIR.API.mlirOpResultGetOwner(v)
get_owner_op_result_index(v::Value) = MLIR.API.mlirOpResultGetResultNumber(v)
is_null(v::Value) = MLIR.API.mlirValueIsNull(v)
Base.:(==)(v1::Value, v2::Value) = MLIR.API.mlirValueEqual(v1, v2)
is_block_args(v::Value) = MLIR.API.mlirValueIsABlockArgument(v)
is_op_result(v::Value) = MLIR.API.mlirValueIsAOpResult(v)
dump(v::Value) = MLIR.API.mlirValueDump(v)
set_type!(v::Value, t::Type) = MLIR.API.mlirBlockArgumentSetType(v, t)

@doc(
"""
const Value = MLIR.API.MlirValue
""", Value)

# ------------ Block alias and APIs ------------ #

const Block = MLIR.API.MlirBlock

create_block(n::Int, t::Type) = MLIR.API.mlirBlockCreate(Ptr(n), t) # TODO.
destroy!(b::Block) = MLIR.API.mlirBlockDestroy(b)
is_null(b::Block) = MLIR.API.mlirBlockIsNull(b)
Base.:(==)(b1::Block, b2::Block) = MLIR.API.mlirBlockEqual(b1, b2)
get_args(b::Block, pos::Int) = MLIR.API.mlirBlockGetArgument(block, Ptr{Int}(pos))
get_num_args(b::Block) = MLIR.API.mlirBlockGetNumArguments(b)
get_next_in_region(b::Block) = MLIR.API.mlirBlockGetNextInRegion(b)
get_first_operation(b::Block) = MLIR.API.mlirBlockGetFirstOperation(b)
get_terminator(b::Block) = MLIR.API.mlirBlockGetTerminator(b)
append!(b::Block, op::Operation) = MLIR.API.mlirBlockAppendOwnedOperation(b, op)
insert!(b::Block, pos::Int, op::Operation) = MLIR.API.mlirBlockInsertOwnedoperation(b, Ptr(pos), op)
insertbefore!(b::Block, ref::Operation, op::Operation) = MLIR.API.mlirBlockInsertOwnedoperationAfter(b, ref, op)
insertafter!(b::Block, ref::Operation, op::Operation) = MLIR.API.mlirBlockInsertOwnedoperation(b, ref, op)

@doc(
"""
const Block = MLIR.API.MlirBlock
""", Block)

# ------------ Module alias and APIs ------------ #

const Module = MLIR.API.MlirModule

create_empty(l::Location) = MLIR.API.mlirModuleCreateEmpty(l)
create_parse(ctx::Context, m::Module) = MLIR.API.mlirModuleCreateParse(ctx, m)
get_context(m::Module) = MLIR.API.mlirModuleGetContext(m)
get_body(m::Module) = MLIR.API.mlirModuleGetBody(m)
is_null(m::Module) = MLIR.API.mlirModuleIsNull(m)
destroy!(m::Module) = MLIR.API.mlirModuleDestroy(m)
get_operation(m::Module) = MLIR.API.mlirModuleGetOperation(m)

Module(l::Location) = create_empty(l)

@doc(
"""
const Module = MLIR.API.MlirModule
""", Module)

# ------------ Region alias and APIs ------------ #

const Region = MLIR.API.MlirRegion

create_region() = MLIR.API.mlirRegionCreate()
destroy!(r::Region) = MLIR.API.mlirRegionDestroy(r)
is_null(r::Region) = MLIR.API.mlirRegionIsNull(r)
get_first_block(r::Region) = MLIR.API.mlirRegionGetFirstBlock(r)
append!(r::Region, b::Block) = MLIR.API.mlirRegionAppendOwnedBlock(r, b)
insert!(r::Region, pos::Int, b::Block) = MLIR.API.mlirRegionAppendOwnedBlock(r, Ptr(pos), b)
insertafter!(r::Region, ref::Block, b::Block) = MLIR.API.mlirRegionInsertOwnedBlockAfter(r, ref, b)
insertbefore!(r::Region, ref::Block, b::Block) = MLIR.API.mlirRegionInsertOwnedBlockBefore(r, ref, b)

Region() = create_region()

@doc(
"""
const Region = MLIR.API.MlirRegion
""", Region)

end # module
