# MLIR.jl

## Design

### String and MlirStringRef

`MlirStringRef` is a non-owning pointer, the caller is in charge of performing necessary
copies or ensuring that the pointee outlives all uses of `MlirStringRef`.
Since Julia is a GC'd language special care must be taken around the live-time of Julia
objects such as `String`s when interacting with foreign libraries.

For convenience and safty sake, users of the API should use Julia `String` or `Symbol` as
the argument to the C-ABI of MLIR instead of directly using `MlirStringRef`. We translate
(cheaply) between Julia `String` and `MlirStringRef`.
