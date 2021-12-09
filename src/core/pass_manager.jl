# ------------ Pass API ------------ #

const Pass = API.MlirPass

# ------------ Pass manager API ------------ #

const PassManager = API.MlirPassManager

create_pass_manager(ctx::Context) = API.mlirPassManagerCreate(ctx)
destroy(pm::PassManager) = API.mlirPassManagerDestroy(pm)
is_null(pm::PassManager) = API.mlirPassManagerIsNull(pm)
get_nested_under(pm::PassManager, op_name::StringRef) = API.mlirPassManagerGetNestedUnder(pm, op_name)
add_owned_pass(pm::PassManager, p::Pass) = API.mlirPassManagerAddOwnedPass(pm, p)
run(pm::PassManager, mod::Module) = API.mlirPassManagerRun(pm, mod)

# Constructor.
PassManager(ctx::Context) = create_pass_manager(ctx)

# ------------ Operation pass manager API ------------ #

const OperationPassManager = API.MlirOpPassManager

get_as_op_pm(pm::PassManager) = API.mlirPassManagerGetAsOpPassManager(pm)
add_owned_pass(opm::OperationPassManager, pass::Pass) = API.mlirOpPassManagerAddOwnedPass(opm, pass)
parse_pass_pipeline(opm::OperationPassManager, pipeline::StringRef) = API.mlirParsePassPipeline(opm, pipeline)
print(opm, callback, userdata) = API.mlirPrintPassPipeline(opm, callback, userdata)

# Constructor.
OperationPassManager(pm::PassManager) = get_as_op_pm(pm)
