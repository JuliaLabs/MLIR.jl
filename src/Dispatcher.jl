struct VersionDispatcher
    mod::Module
end

function modulefromversion(dispatcher::VersionDispatcher, version::VersionNumber)
    if v"14" <= version < v"15"
        getproperty(getfield(dispatcher, :mod), :v14)
    elseif v"15" <= version < v"16"
        getproperty(getfield(dispatcher, :mod), :v15)
    elseif v"16" <= version < v"17"
        getproperty(getfield(dispatcher, :mod), :v16)
    else
        error("Unsupported MLIR version $version")
    end
end

function Base.getproperty(dispatcher::VersionDispatcher, name::Symbol)
    version = MLIR_VERSION[]
    mod = modulefromversion(dispatcher, version)
    getproperty(mod, name)
end

function Base.propertynames(dispatcher::VersionDispatcher, private::Bool=true)
    version = MLIR_VERSION[]
    mod = modulefromversion(dispatcher, version)
    propertynames(mod, private)
end

function Base.names(dispatcher::VersionDispatcher; kwargs...)
    version = MLIR_VERSION[]
    mod = modulefromversion(dispatcher, version)
    names(mod; kwargs...)
end
