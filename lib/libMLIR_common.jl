# Automatically generated using Clang.jl


# Skipping MacroDefinition: DEFINE_C_API_STRUCT ( name , storage ) @checked struct name { storage * ptr ; } ; typedef @checked struct name name

@checked struct MlirAffineExpr
    ref::Ptr{Cvoid}
end

@checked struct MlirAffineMap
    ref::Ptr{Cvoid}
end

@checked struct MlirDiagnostic
    ref::Ptr{Cvoid}
end

@cenum MlirDiagnosticSeverity::UInt32 begin
    MlirDiagnosticError = 0
    MlirDiagnosticWarning = 1
    MlirDiagnosticNote = 2
    MlirDiagnosticRemark = 3
end


const MlirDiagnosticHandlerID = UInt64
const MlirDiagnosticHandler = Ptr{Cvoid}

@checked struct MlirContext
    ref::Ptr{Cvoid}
end

@checked struct MlirDialect
    ref::Ptr{Cvoid}
end

@checked struct MlirOperation
    ref::Ptr{Cvoid}
end

@checked struct MlirOpPrintingFlags
    ref::Ptr{Cvoid}
end

@checked struct MlirBlock
    ref::Ptr{Cvoid}
end

@checked struct MlirRegion
    ref::Ptr{Cvoid}
end

@checked struct MlirAttribute
    ref::Ptr{Cvoid}
end

@checked struct MlirIdentifier
    ref::Ptr{Cvoid}
end

@checked struct MlirLocation
    ref::Ptr{Cvoid}
end

@checked struct MlirModule
    ref::Ptr{Cvoid}
end

@checked struct MlirType
    ref::Ptr{Cvoid}
end

@checked struct MlirValue
    ref::Ptr{Cvoid}
end

struct MlirNamedAttribute
    name::MlirIdentifier
    attribute::MlirAttribute
end

struct MlirStringRef
    data::Cstring
    length::Csize_t
end

mutable struct MlirOperationState
    name::MlirStringRef
    location::MlirLocation
    nResults::intptr_t
    results::Ptr{MlirType}
    nOperands::intptr_t
    operands::Ptr{MlirValue}
    nRegions::intptr_t
    regions::Ptr{MlirRegion}
    nSuccessors::intptr_t
    successors::Ptr{MlirBlock}
    nAttributes::intptr_t
    attributes::Ptr{MlirNamedAttribute}
    enableResultTypeInference::Bool
end

@checked struct MlirIntegerSet
    ref::Ptr{Cvoid}
end

@checked struct MlirPass
    ref::Ptr{Cvoid}
end

@checked struct MlirPassManager
    ref::Ptr{Cvoid}
end

@checked struct MlirOpPassManager
    ref::Ptr{Cvoid}
end

# Skipping MacroDefinition: MLIR_DECLARE_CAPI_DIALECT_REGISTRATION ( Name , Namespace ) MLIR_CAPI_EXPORTED void mlirContextRegister ## Name ## Dialect ( MlirContext context ) ; MLIR_CAPI_EXPORTED MlirDialect mlirContextLoad ## Name ## Dialect ( MlirContext context ) ; MLIR_CAPI_EXPORTED MlirStringRef mlir ## Name ## DialectGetNamespace ( ) ; MLIR_CAPI_EXPORTED const MlirDialectRegistrationHooks * mlirGetDialectHooks__ ## Namespace ## __ ( )

const MlirContextRegisterDialectHook = Ptr{Cvoid}
const MlirContextLoadDialectHook = Ptr{Cvoid}
const MlirDialectGetNamespaceHook = Ptr{Cvoid}

struct MlirDialectRegistrationHooks
    registerHook::MlirContextRegisterDialectHook
    loadHook::MlirContextLoadDialectHook
    getNamespaceHook::MlirDialectGetNamespaceHook
end

# Skipping MacroDefinition: MLIR_CAPI_EXPORTED __attribute__ ( ( visibility ( "default" ) ) )

const MlirStringCallback = Ptr{Cvoid}

struct MlirLogicalResult
    value::Int8
end
