#####
##### Mlir(Pass + PassManager) alias and APIs
#####

const Pass = MLIR.API.MlirPass
const PassManager = MLIR.API.MlirPassManager

create_pass_manager(ctx::Context) = MLIR.API.mlirPassManagerCreate(ctx)
destroy(pm::PassManager) = MLIR.API.mlirPassManagerDestroy(pm)
is_null(pm::PassManager) = MLIR.API.mlirPassManagerIsNull(pm)
get_nested_under(pm::PassManager, op_name::StringRef) = MLIR.API.mlirPassManagerGetNestedUnder(pm, op_name)
add_owned_pass(pm::PassManager, p::Pass) = MLIR.API.mlirPassManagerAddOwnedPass(pm, p)
run(pm::PassManager, mod::Module) = MLIR.API.mlirPassManagerRun(pm, mod)

# Constructor.
PassManager(ctx::Context) = create_pass_manager(ctx)

#####
##### MlirOpPassManager alias and APIs
#####

const OperationPassManager = MLIR.API.MlirOpPassManager

get_as_op_pm(pm::PassManager) = MLIR.API.mlirPassManagerGetAsOpPassManager(pm)
add_owned_pass(opm::OperationPassManager, pass::Pass) = MLIR.API.mlirOpPassManagerAddOwnedPass(opm, pass)
parse_pass_pipeline(opm::OperationPassManager, pipeline::StringRef) = MLIR.API.mlirParsePassPipeline(opm, pipeline)
print(opm, callback, userdata) = MLIR.API.mlirPrintPassPipeline(opm, callback, userdata)

# Constructor.
OperationPassManager(pm::PassManager) = get_as_op_pm(pm)
