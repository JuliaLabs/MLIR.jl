struct Dialect
    dialect::API.MlirDialect

    function Dialect(dialect)
        @assert !API.mlirDialectIsNull(dialect) "cannot create Dialect from null MlirDialect"
        new(dialect)
    end
end

Base.convert(::Core.Type{API.MlirDialect}, dialect::Dialect) = dialect.dialect
Base.:(==)(a::Dialect, b::Dialect) = API.mlirDialectEqual(a, b)

context(dialect::Dialect) = Context(API.mlirDialectGetContext(dialect))
namespace(dialect::Dialect) = String(API.mlirDialectGetNamespace(dialect))

function Base.show(io::IO, dialect::Dialect)
    print(io, "Dialect(\"", namespace(dialect), "\")")
end

allow_unregistered_dialects(ctx::Context=context()) = API.mlirContextGetAllowUnregisteredDialects(ctx)
allow_unregistered_dialects!(ctx::Context=context(), allow::Bool=true) = API.mlirContextSetAllowUnregisteredDialects(ctx, allow)

num_registered_dialects(ctx::Context=context()) = API.mlirContextGetNumRegisteredDialects(ctx)
num_loaded_dialects(ctx::Context=context()) = API.mlirContextGetNumLoadedDialects(ctx)

load_all_available_dialects(ctx::Context=context()) = API.mlirContextLoadAllAvailableDialects(ctx)

function get_or_load_dialect!(ctx::Context, name::String)
    dialect = API.mlirContextGetOrLoadDialect(ctx, name)
    API.mlirDialectIsNull(dialect) && error("could not load dialect $name")
    Dialect(dialect)
end

struct DialectHandle
    handle::API.MlirDialectHandle
end

function DialectHandle(s::Symbol)
    s = Symbol("mlirGetDialectHandle__", s, "__")
    DialectHandle(getproperty(API, s)())
end

Base.convert(::Core.Type{API.MlirDialectHandle}, handle::DialectHandle) = handle.handle

namespace(handle::DialectHandle) = String(API.mlirDialectHandleGetNamespace(handle))

function get_or_load_dialect!(ctx::Context, handle::DialectHandle)
    dialect = API.mlirDialectHandleLoadDialect(handle, ctx)
    API.mlirDialectIsNull(dialect) && error("could not load dialect from handle $handle")
    Dialect(dialect)
end

function get_or_load_dialect!(dialect::String)
    get_or_load_dialect!(DialectHandle(Symbol(dialect)))
end

register_dialect!(ctx::Context, handle::DialectHandle) = API.mlirDialectHandleRegisterDialect(handle, ctx)
load_dialect!(ctx::Context, handle::DialectHandle) = Dialect(API.mlirDialectHandleLoadDialect(handle, ctx))

mutable struct DialectRegistry
    registry::API.MlirDialectRegistry
end

function DialectRegistry()
    registry = API.mlirDialectRegistryCreate()
    @assert !API.mlirDialectRegistryIsNull(registry) "cannot create DialectRegistry with null MlirDialectRegistry"
    finalizer(DialectRegistry(registry)) do registry
        API.mlirDialectRegistryDestroy(registry.registry)
    end
end

Base.convert(::Core.Type{API.MlirDialectRegistry}, registry::DialectRegistry) = registry.registry
Base.push!(registry::DialectRegistry, handle::DialectHandle) = API.mlirDialectHandleInsertDialect(handle, registry)

# TODO is `append!` the right name?
Base.append!(ctx::Context, registry::DialectRegistry) = API.mlirContextAppendDialectRegistry(ctx, registry)
