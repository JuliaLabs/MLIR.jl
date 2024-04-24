mutable struct Module
    module_::API.MlirModule

    function Module(module_)
        @assert !mlirIsNull(module_) "cannot create Module with null MlirModule"
        finalizer(API.Dispatcher.mlirModuleDestroy, new(module_))
    end
end

"""
    Module(location=Location())

Creates a new, empty module and transfers ownership to the caller.
"""
Module(loc::Location=Location()) = Module(API.Dispatcher.mlirModuleCreateEmpty(loc))

Module(op::Operation) = Module(API.Dispatcher.mlirModuleFromOperation(lose_ownership!(op)))

Base.convert(::Core.Type{API.MlirModule}, module_::Module) = module_.module_

"""
    parse(::Type{Module}, module; context=context())

Parses a module from the string and transfers ownership to the caller.
"""
Base.parse(::Core.Type{Module}, module_; context::Context=context()) = Module(API.Dispatcher.mlirModuleCreateParse(context, module_))

macro mlir_str(code)
    quote
        ctx = Context()
        parse(Module, $code)
    end
end

"""
    context(module)

Gets the context that a module was created with.
"""
context(module_::Module) = Context(API.Dispatcher.mlirModuleGetContext(module_))

"""
    body(module)

Gets the body of the module, i.e. the only block it contains.
"""
body(module_) = Block(API.Dispatcher.mlirModuleGetBody(module_), false)

"""
    Operation(module)

Views the module as a generic operation.
"""
Operation(module_::Module) = Operation(API.Dispatcher.mlirModuleGetOperation(module_), false)

function Base.show(io::IO, module_::Module)
    println(io, "Module:")
    show(io, Operation(module_))
end
