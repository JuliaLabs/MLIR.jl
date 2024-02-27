mutable struct Module
    module_::API.MlirModule

    Module(module_) = begin
        @assert !API.mlirModuleIsNull(module_) "cannot create Module with null MlirModule"
        finalizer(API.mlirModuleDestroy, new(module_))
    end
end

Module(loc::Location=Location()) = Module(API.mlirModuleCreateEmpty(loc))

Module(op::Operation) = Module(API.mlirModuleFromOperation(lose_ownership!(op)))

Base.convert(::Type{API.MlirModule}, module_::Module) = module_.module_

Base.parse(::Type{Module}, module_; context::Context=context()) = Module(API.mlirModuleCreateParse(context, module_))
macro mlir_str(code)
    quote
        ctx = Context()
        parse(Module, code)
    end
end

context(module_::Module) = Context(API.mlirModuleGetContext(module_))
body(module_) = Block(API.mlirModuleGetBody(module_), false)

Operation(module_::Module) = Operation(API.mlirModuleGetOperation(module_), false)
Module(op::Operation) = Module(API.mlirModuleFromOperation(op))

# get_first_child_op(mod::Module) = get_first_child_op(get_operation(mod))

function Base.show(io::IO, module_::Module)
    println(io, "Module:")
    show(io, get_operation(module_))
end
