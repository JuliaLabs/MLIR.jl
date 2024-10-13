module mpi

import ...IR:
    IR, NamedAttribute, Value, Location, Block, Region, Attribute, context, IndexType
import ..Dialects: namedattribute, operandsegmentsizes

"""
`comm_rank`

Communicators other than `MPI_COMM_WORLD` are not supported for now.

This operation can optionally return an `!mpi.retval` value that can be used
to check for errors.
"""
function comm_rank(;
    retval=nothing::Union{Nothing,IR.Type}, rank::IR.Type, location=Location()
)
    _results = IR.Type[rank,]
    _operands = Value[]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(retval) && push!(_results, retval)

    return IR.create_operation(
        "mpi.comm_rank",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`error_class`

`MPI_Error_class` maps return values from MPI calls to a set of well-known
MPI error classes.
"""
function error_class(val::Value; errclass::IR.Type, location=Location())
    _results = IR.Type[errclass,]
    _operands = Value[val,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "mpi.error_class",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`finalize`

This function cleans up the MPI state. Afterwards, no MPI methods may 
be invoked (excpet for MPI_Get_version, MPI_Initialized, and MPI_Finalized).
Notably, MPI_Init cannot be called again in the same program.

This operation can optionally return an `!mpi.retval` value that can be used
to check for errors.
"""
function finalize(; retval=nothing::Union{Nothing,IR.Type}, location=Location())
    _results = IR.Type[]
    _operands = Value[]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(retval) && push!(_results, retval)

    return IR.create_operation(
        "mpi.finalize",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`init`

This operation must preceed most MPI calls (except for very few exceptions,
please consult with the MPI specification on these).

Passing &argc, &argv is not supported currently.

This operation can optionally return an `!mpi.retval` value that can be used
to check for errors.
"""
function init(; retval=nothing::Union{Nothing,IR.Type}, location=Location())
    _results = IR.Type[]
    _operands = Value[]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(retval) && push!(_results, retval)

    return IR.create_operation(
        "mpi.init",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`recv`

MPI_Recv performs a blocking receive of `size` elements of type `dtype` 
from rank `dest`. The `tag` value and communicator enables the library to 
determine the matching of multiple sends and receives between the same 
ranks.

Communicators other than `MPI_COMM_WORLD` are not supprted for now.
The MPI_Status is set to `MPI_STATUS_IGNORE`, as the status object 
is not yet ported to MLIR.

This operation can optionally return an `!mpi.retval` value that can be used
to check for errors.
"""
function recv(
    ref::Value,
    tag::Value,
    rank::Value;
    retval=nothing::Union{Nothing,IR.Type},
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[ref, tag, rank]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(retval) && push!(_results, retval)

    return IR.create_operation(
        "mpi.recv",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`retval_check`

This operation compares MPI status codes to known error class
constants such as `MPI_SUCCESS`, or `MPI_ERR_COMM`.
"""
function retval_check(val::Value; res::IR.Type, errclass, location=Location())
    _results = IR.Type[res,]
    _operands = Value[val,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[namedattribute("errclass", errclass),]

    return IR.create_operation(
        "mpi.retval_check",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`send`

MPI_Send performs a blocking send of `size` elements of type `dtype` to rank
`dest`. The `tag` value and communicator enables the library to determine 
the matching of multiple sends and receives between the same ranks.

Communicators other than `MPI_COMM_WORLD` are not supprted for now.

This operation can optionally return an `!mpi.retval` value that can be used
to check for errors.
"""
function send(
    ref::Value,
    tag::Value,
    rank::Value;
    retval=nothing::Union{Nothing,IR.Type},
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[ref, tag, rank]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(retval) && push!(_results, retval)

    return IR.create_operation(
        "mpi.send",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

end # mpi
