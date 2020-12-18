module Simple

using MLIR

# Create and destroy.
ctx = MLIR.IR.create_context()
MLIR.IR.num_loaded_dialects(ctx) |> println
MLIR.IR.num_registered_dialects(ctx) |> println
MLIR.IR.destroy!(ctx)

# Create and register standard.
ctx =  MLIR.IR.create_context()
MLIR.IR.register_standard_dialect!(ctx)
MLIR.IR.load_standard_dialect!(ctx)
MLIR.IR.num_loaded_dialects(ctx) |> println
MLIR.IR.num_registered_dialects(ctx) |> println
MLIR.IR.destroy!(ctx)

end # module
