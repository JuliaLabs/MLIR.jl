struct Dialect
    dialect::API.MlirDialect

    function Dialect(dialect)
        @assert !mlirIsNull(dialect) "cannot create Dialect from null MlirDialect"
        new(dialect)
    end
end

Base.convert(::Core.Type{API.MlirDialect}, dialect::Dialect) = dialect.dialect
Base.:(==)(a::Dialect, b::Dialect) = API.Dispatcher.mlirDialectEqual(a, b)

context(dialect::Dialect) = Context(API.Dispatcher.mlirDialectGetContext(dialect))
namespace(dialect::Dialect) = String(API.Dispatcher.mlirDialectGetNamespace(dialect))

function Base.show(io::IO, dialect::Dialect)
    print(io, "Dialect(\"", namespace(dialect), "\")")
end

allow_unregistered_dialects(; context::Context=context()) = API.Dispatcher.mlirContextGetAllowUnregisteredDialects(context)
allow_unregistered_dialects!(allow::Bool=true; context::Context=context()) = API.Dispatcher.mlirContextSetAllowUnregisteredDialects(context, allow)

num_registered_dialects(; context::Context=context()) = API.Dispatcher.mlirContextGetNumRegisteredDialects(context)
num_loaded_dialects(; context::Context=context()) = API.Dispatcher.mlirContextGetNumLoadedDialects(context)

load_all_available_dialects(; context::Context=context()) = API.Dispatcher.mlirContextLoadAllAvailableDialects(context)

function get_or_load_dialect!(name::String; context::Context=context())
    dialect = API.Dispatcher.mlirContextGetOrLoadDialect(context, name)
    mlirIsNull(dialect) && error("could not load dialect $name")
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

namespace(handle::DialectHandle) = String(API.Dispatcher.mlirDialectHandleGetNamespace(handle))

function get_or_load_dialect!(handle::DialectHandle; context::Context=context())
    dialect = API.Dispatcher.mlirDialectHandleLoadDialect(handle, context)
    mlirIsNull(dialect) && error("could not load dialect from handle $handle")
    Dialect(dialect)
end

register_dialect!(handle::DialectHandle; context::Context=context()) = API.Dispatcher.mlirDialectHandleRegisterDialect(handle, context)
load_dialect!(handle::DialectHandle; context::Context=context()) = Dialect(API.Dispatcher.mlirDialectHandleLoadDialect(handle, context))

mutable struct DialectRegistry
    registry::API.MlirDialectRegistry

    function DialectRegistry(registry)
        @assert !mlirIsNull(registry) "cannot create DialectRegistry with null MlirDialectRegistry"
        finalizer(DialectRegistry(registry)) do registry
            API.Dispatcher.mlirDialectRegistryDestroy(registry.registry)
        end
    end
end

DialectRegistry() = DialectRegistry(API.Dispatcher.mlirDialectRegistryCreate())

Base.convert(::Core.Type{API.MlirDialectRegistry}, registry::DialectRegistry) = registry.registry
Base.push!(registry::DialectRegistry, handle::DialectHandle) = API.Dispatcher.mlirDialectHandleInsertDialect(handle, registry)

# TODO is `append!` the right name?
Base.append!(registry::DialectRegistry; context::Context) = API.Dispatcher.mlirContextAppendDialectRegistry(context, registry)
