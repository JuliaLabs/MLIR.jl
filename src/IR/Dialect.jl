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

allow_unregistered_dialects(context::Context=context()) = API.mlirContextGetAllowUnregisteredDialects(context)
allow_unregistered_dialects!(allow::Bool=true; context::Context=context()) = API.mlirContextSetAllowUnregisteredDialects(context, allow)

num_registered_dialects(ctx::Context=context()) = API.mlirContextGetNumRegisteredDialects(ctx)
num_loaded_dialects(ctx::Context=context()) = API.mlirContextGetNumLoadedDialects(ctx)

load_all_available_dialects(ctx::Context=context()) = API.mlirContextLoadAllAvailableDialects(ctx)

function get_or_load_dialect!(name::String; context::Context=context())
    dialect = API.mlirContextGetOrLoadDialect(context, name)
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

function get_or_load_dialect!(handle::DialectHandle; context::Context=context())
    dialect = API.mlirDialectHandleLoadDialect(handle, context)
    API.mlirDialectIsNull(dialect) && error("could not load dialect from handle $handle")
    Dialect(dialect)
end

get_or_load_dialect!(dialect::String; context::Context=context()) = get_or_load_dialect!(DialectHandle(Symbol(dialect)); context)

register_dialect!(handle::DialectHandle; context::Context=context()) = API.mlirDialectHandleRegisterDialect(handle, context)
load_dialect!(handle::DialectHandle; context::Context=context()) = Dialect(API.mlirDialectHandleLoadDialect(handle, context))

mutable struct DialectRegistry
    registry::API.MlirDialectRegistry

    function DialectRegistry(registry)
        @assert !API.mlirDialectRegistryIsNull(registry) "cannot create DialectRegistry with null MlirDialectRegistry"
        finalizer(DialectRegistry(registry)) do registry
            API.mlirDialectRegistryDestroy(registry.registry)
        end
    end
end

DialectRegistry() = DialectRegistry(API.mlirDialectRegistryCreate())

Base.convert(::Core.Type{API.MlirDialectRegistry}, registry::DialectRegistry) = registry.registry
Base.push!(registry::DialectRegistry, handle::DialectHandle) = API.mlirDialectHandleInsertDialect(handle, registry)

# TODO is `append!` the right name?
Base.append!(ctx::Context, registry::DialectRegistry) = API.mlirContextAppendDialectRegistry(ctx, registry)
