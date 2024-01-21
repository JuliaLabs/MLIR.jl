using Standalone
using MLIR

MLIR.IR.context!(MLIR.IR.Context()) do
    for dialect in ("func", "cf")
        MLIR.IR.get_or_load_dialect!(dialect)
    end
    Standalone.load_dialect(MLIR.IR.context())
end
