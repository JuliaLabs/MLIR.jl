# ------------ Pass API ------------ #

const Pass = MLIR.API.MlirPass

# ------------ Pass manager API ------------ #

const PassManager = MLIR.API.MlirPassManager

create_pass_manager(ctx::Context) = mlirPassManagerCreate(ctx)
destroy(pm::PassManager) = mlirPassManagerDestroy(pm)
is_null(pm::PassManager) = mlirPassManagerIsNull(pm)
run(pm::PassManager, m::IR.Module) = mlirPassManagerRun(pm, m)
get_nested_under(pm::PassManager, op_name::StringRef) = mlirPassManagerGetNestedUnder(pm, op_name)
add_owned_pass(pm::PassManager, p::Pass) = mlirPassManagerAddOwnedPass(pm, p)

const OperationPassManager = MLIR.API.MlirOpPassManager
