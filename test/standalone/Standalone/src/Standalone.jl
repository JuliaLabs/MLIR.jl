module Standalone

using MLIR
using Preferences

const libstandalone_c = @load_preference("libstandalone_c")

module StandaloneAPI
    import ..Standalone: libstandalone_c
    import MLIR.API: MlirDialectHandle
    function mlirGetDialectHandle__standalone__()
        @ccall libstandalone_c.mlirGetDialectHandle__standalone__()::MlirDialectHandle
    end
end

import MLIR: API, IR, Dialects

function load_dialect(ctx)
    dialect = IR.DialectHandle(StandaloneAPI.mlirGetDialectHandle__standalone__())
    API.mlirDialectHandleRegisterDialect(dialect, ctx)
    API.mlirDialectHandleLoadDialect(dialect, ctx)
end

end # module Standalone
