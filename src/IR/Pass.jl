### Pass Manager

abstract type AbstractPass end

mutable struct ExternalPassHandle
    ctx::Union{Nothing,Context}
    pass::AbstractPass
end

mutable struct PassManager
    pass::MlirPassManager
    allocator::TypeIDAllocator
    passes::Dict{TypeID,ExternalPassHandle}

    PassManager(pm::MlirPassManager) = begin
        @assert !mlirIsNull(pm) "cannot create PassManager with null MlirPassManager"
        finalizer(new(pm, TypeIDAllocator(), Dict{TypeID,ExternalPassHandle}())) do pm
            API.mlirPassManagerDestroy(pm.pass)
        end
    end
end

function enable_ir_printing!(pm)
    API.mlirPassManagerEnableIRPrinting(pm)
    pm
end
function enable_verifier!(pm, enable=true)
    API.mlirPassManagerEnableVerifier(pm, enable)
    pm
end

PassManager() =
    PassManager(API.mlirPassManagerCreate(context()))

function run!(pm::PassManager, module_)
    status = API.mlirPassManagerRun(pm, module_)
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

OpPassManager(pm::PassManager) = OpPassManager(API.mlirPassManagerGetAsOpPassManager(pm), pm)
OpPassManager(pm::PassManager, opname) = OpPassManager(API.mlirPassManagerGetNestedUnder(pm, opname), pm)
OpPassManager(opm::OpPassManager, opname) = OpPassManager(API.mlirOpPassManagerGetNestedUnder(opm, opname), opm.pass)

Base.convert(::Type{MlirOpPassManager}, op_pass::OpPassManager) = op_pass.op_pass

function Base.show(io::IO, op_pass::OpPassManager)
    c_print_callback = @cfunction(print_callback, Cvoid, (MlirStringRef, Any))
    ref = Ref(io)
    println(io, "OpPassManager(\"\"\"")
    API.mlirPrintPassPipeline(op_pass, c_print_callback, ref)
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

function add_pipeline!(op_pass::OpPassManager, pipeline)
    @static if isdefined(API, :mlirOpPassManagerAddPipeline)
        io = IOBuffer()
        c_print_callback = @cfunction(print_callback, Cvoid, (MlirStringRef, Any))
        result = GC.@preserve io API.mlirOpPassManagerAddPipeline(op_pass, pipeline, c_print_callback, Ref(io))
        if mlirLogicalResultIsFailure(result)
            exc = AddPipelineException(String(take!(io)))
            throw(exc)
        end
    else
        result = API.mlirParsePassPipeline(op_pass, pipeline)
        if mlirLogicalResultIsFailure(result)
            throw(AddPipelineException(" " * pipeline))
        end
    end
    op_pass
end

function add_owned_pass!(pm::PassManager, pass)
    API.mlirPassManagerAddOwnedPass(pm, pass)
    pm
end

function add_owned_pass!(opm::OpPassManager, pass)
    API.mlirOpPassManagerAddOwnedPass(opm, pass)
    opm
end


@static if isdefined(API, :mlirCreateExternalPass)

    ### Pass

    # AbstractPass interface:
    opname(::AbstractPass) = ""
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
            mlirLogicalResultSuccess()
        catch
            mlirLogicalResultFailure()
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
            @error "Something went wrong running pass" exception = (ex, catch_backtrace())
            API.mlirExternalPassSignalFailure(external_pass)
        end
        nothing
    end

    function create_external_pass!(oppass::OpPassManager, args...)
        create_external_pass!(oppass.pass, args...)
    end
    function create_external_pass!(manager, pass, name, argument,
        description, opname=opname(pass),
        dependent_dialects=MlirDialectHandle[])
        passid = TypeID(manager.allocator)
        callbacks = API.MlirExternalPassCallbacks(
            @cfunction(_pass_construct, Cvoid, (Any,)),
            @cfunction(_pass_destruct, Cvoid, (Any,)),
            @cfunction(_pass_initialize, API.MlirLogicalResult, (MlirContext, Any,)),
            @cfunction(_pass_clone, Any, (Any,)),
            @cfunction(_pass_run, Cvoid, (MlirOperation, API.MlirExternalPass, Any))
        )
        pass_handle = manager.passes[passid] = ExternalPassHandle(nothing, pass)
        userdata = Base.pointer_from_objref(pass_handle)
        mlir_pass = API.mlirCreateExternalPass(passid, name, argument, description, opname,
            length(dependent_dialects), dependent_dialects,
            callbacks, userdata)
        mlir_pass
    end

end