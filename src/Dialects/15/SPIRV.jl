module spv

import ...IR: NamedAttribute, MLIRType, Value, Location, Block, Region, Attribute, create_operation, context, IndexType
import ..Dialects: namedattribute, operandsegmentsizes
import ...API


"""
`AccessChain`

Result Type must be an OpTypePointer. Its Type operand must be the type
reached by walking the Base’s type hierarchy down to the last provided
index in Indexes, and its Storage Class operand must be the same as the
Storage Class of Base.

Base must be a pointer, pointing to the base of a composite object.

Indexes walk the type hierarchy to the desired depth, potentially down
to scalar granularity. The first index in Indexes will select the top-
level member/element/component/element of the base composite. All
composite constituents use zero-based numbering, as described by their
OpType… instruction. The second index will apply similarly to that
result, and so on. Once any non-composite type is reached, there must be
no remaining (unused) indexes.

 Each index in Indexes

- must be a scalar integer type,

- is treated as a signed count, and

- must be an OpConstant when indexing into a structure.

<!-- End of AutoGen section -->
```
access-chain-op ::= ssa-id `=` `spv.AccessChain` ssa-use
                    `[` ssa-use (\',\' ssa-use)* `]`
                    `:` pointer-type
```

#### Example:

```mlir
%0 = \"spv.Constant\"() { value = 1: i32} : () -> i32
%1 = spv.Variable : !spv.ptr<!spv.struct<f32, !spv.array<4xf32>>, Function>
%2 = spv.AccessChain %1[%0] : !spv.ptr<!spv.struct<f32, !spv.array<4xf32>>, Function>
%3 = spv.Load \"Function\" %2 [\"Volatile\"] : !spv.array<4xf32>
```
"""
function AccessChain(base_ptr::Value, indices::Vector{Value}; component_ptr::MLIRType, location=Location())
    results = MLIRType[component_ptr, ]
    operands = Value[base_ptr, indices..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "spv.AccessChain", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`mlir_addressof`

Variables in module scope are defined using symbol names. This op generates
an SSA value that can be used to refer to the symbol within function scope
for use in ops that expect an SSA value. This operation has no corresponding
SPIR-V instruction; it\'s merely used for modelling purpose in the SPIR-V
dialect. Since variables in module scope in SPIR-V dialect are of pointer
type, this op returns a pointer type as well, and the type is the same as
the variable referenced.

<!-- End of AutoGen section -->

```
spv-address-of-op ::= ssa-id `=` `spv.mlir.addressof` symbol-ref-id
                                 `:` spirv-pointer-type
```

#### Example:

```mlir
%0 = spv.mlir.addressof @global_var : !spv.ptr<f32, Input>
```
"""
function mlir_addressof(; pointer::MLIRType, variable, location=Location())
    results = MLIRType[pointer, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("variable", variable), ]
    
    create_operation(
        "spv.mlir.addressof", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`AssumeTrueKHR`



<!-- End of AutoGen section -->

```
assumetruekhr-op ::= `spv.AssumeTrueKHR` ssa-use
```mlir

#### Example:

```
spv.AssumeTrueKHR %arg
```
"""
function AssumeTrueKHR(condition::Value; location=Location())
    results = MLIRType[]
    operands = Value[condition, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "spv.AssumeTrueKHR", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`AtomicAnd`

1) load through Pointer to get an Original Value,

2) get a New Value by the bitwise AND of Original Value and Value, and

3) store the New Value back through Pointer.

The instruction’s result is the Original Value.

Result Type must be an integer type scalar.

 The type of Value must be the same as Result Type.  The type of the
value pointed to by Pointer must be the same as Result Type.

Memory must be a valid memory Scope.

<!-- End of AutoGen section -->

```
scope ::= `\"CrossDevice\"` | `\"Device\"` | `\"Workgroup\"` | ...

memory-semantics ::= `\"None\"` | `\"Acquire\"` | \"Release\"` | ...

atomic-and-op ::=
    `spv.AtomicAnd` scope memory-semantics
                    ssa-use `,` ssa-use `:` spv-pointer-type
```

#### Example:

```mlir
%0 = spv.AtomicAnd \"Device\" \"None\" %pointer, %value :
                   !spv.ptr<i32, StorageBuffer>
```
"""
function AtomicAnd(pointer::Value, value::Value; result::MLIRType, memory_scope, semantics, location=Location())
    results = MLIRType[result, ]
    operands = Value[pointer, value, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("memory_scope", memory_scope), namedattribute("semantics", semantics), ]
    
    create_operation(
        "spv.AtomicAnd", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`AtomicCompareExchange`

1) load through Pointer to get an Original Value,

2) get a New Value from Value only if Original Value equals Comparator,
and

3) store the New Value back through Pointer\'only if \'Original Value
equaled Comparator.

The instruction\'s result is the Original Value.

Result Type must be an integer type scalar.

Use Equal for the memory semantics of this instruction when Value and
Original Value compare equal.

Use Unequal for the memory semantics of this instruction when Value and
Original Value compare unequal. Unequal must not be set to Release or
Acquire and Release. In addition, Unequal cannot be set to a stronger
memory-order then Equal.

 The type of Value must be the same as Result Type.  The type of the
value pointed to by Pointer must be the same as Result Type.  This type
must also match the type of Comparator.

Memory is a memory Scope.

<!-- End of AutoGen section -->

```
atomic-compare-exchange-op ::=
    `spv.AtomicCompareExchange` scope memory-semantics memory-semantics
                                ssa-use `,` ssa-use `,` ssa-use
                                `:` spv-pointer-type
```mlir

#### Example:

```
%0 = spv.AtomicCompareExchange \"Workgroup\" \"Acquire\" \"None\"
                                %pointer, %value, %comparator
                                : !spv.ptr<i32, WorkGroup>
```
"""
function AtomicCompareExchange(pointer::Value, value::Value, comparator::Value; result::MLIRType, memory_scope, equal_semantics, unequal_semantics, location=Location())
    results = MLIRType[result, ]
    operands = Value[pointer, value, comparator, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("memory_scope", memory_scope), namedattribute("equal_semantics", equal_semantics), namedattribute("unequal_semantics", unequal_semantics), ]
    
    create_operation(
        "spv.AtomicCompareExchange", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`AtomicCompareExchangeWeak`

Has the same semantics as OpAtomicCompareExchange.

Memory must be a valid memory Scope.

<!-- End of AutoGen section -->

```
atomic-compare-exchange-weak-op ::=
    `spv.AtomicCompareExchangeWeak` scope memory-semantics memory-semantics
                                    ssa-use `,` ssa-use `,` ssa-use
                                    `:` spv-pointer-type
```

#### Example:

```mlir
%0 = spv.AtomicCompareExchangeWeak \"Workgroup\" \"Acquire\" \"None\"
                                   %pointer, %value, %comparator
                                   : !spv.ptr<i32, WorkGroup>
```
"""
function AtomicCompareExchangeWeak(pointer::Value, value::Value, comparator::Value; result::MLIRType, memory_scope, equal_semantics, unequal_semantics, location=Location())
    results = MLIRType[result, ]
    operands = Value[pointer, value, comparator, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("memory_scope", memory_scope), namedattribute("equal_semantics", equal_semantics), namedattribute("unequal_semantics", unequal_semantics), ]
    
    create_operation(
        "spv.AtomicCompareExchangeWeak", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`AtomicExchange`

1) load through Pointer to get an Original Value,

2) get a New Value from copying Value, and

3) store the New Value back through Pointer.

The instruction\'s result is the Original Value.

Result Type must be a scalar of integer type or floating-point type.

 The type of Value must be the same as Result Type.  The type of the
value pointed to by Pointer must be the same as Result Type.

Memory is a memory Scope.

<!-- End of AutoGen section -->

 ```
atomic-exchange-op ::=
    `spv.AtomicCompareExchange` scope memory-semantics
                                ssa-use `,` ssa-use `:` spv-pointer-type
```mlir

#### Example:

```
%0 = spv.AtomicExchange \"Workgroup\" \"Acquire\" %pointer, %value,
                        : !spv.ptr<i32, WorkGroup>
```
"""
function AtomicExchange(pointer::Value, value::Value; result::MLIRType, memory_scope, semantics, location=Location())
    results = MLIRType[result, ]
    operands = Value[pointer, value, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("memory_scope", memory_scope), namedattribute("semantics", semantics), ]
    
    create_operation(
        "spv.AtomicExchange", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`AtomicFAddEXT`



<!-- End of AutoGen section -->

Perform the following steps atomically with respect to any other atomic
accesses within Scope to the same location:

1) load through Pointer to get an Original Value,

2) get a New Value by float addition of Original Value and Value, and

3) store the New Value back through Pointer.

The instruction’s result is the Original Value.

Result Type must be a floating-point type scalar.

The type of Value must be the same as Result Type. The type of the value
pointed to by Pointer must be the same as Result Type.

Memory must be a valid memory Scope.

```
atomic-fadd-op ::=
    `spv.AtomicFAddEXT` scope memory-semantics
                        ssa-use `,` ssa-use `:` spv-pointer-type
```

#### Example:

```mlir
%0 = spv.AtomicFAddEXT \"Device\" \"None\" %pointer, %value :
                       !spv.ptr<f32, StorageBuffer>
```mlir
"""
function AtomicFAddEXT(pointer::Value, value::Value; result::MLIRType, memory_scope, semantics, location=Location())
    results = MLIRType[result, ]
    operands = Value[pointer, value, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("memory_scope", memory_scope), namedattribute("semantics", semantics), ]
    
    create_operation(
        "spv.AtomicFAddEXT", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`AtomicIAdd`

1) load through Pointer to get an Original Value,

2) get a New Value by integer addition of Original Value and Value, and

3) store the New Value back through Pointer.

The instruction’s result is the Original Value.

Result Type must be an integer type scalar.

 The type of Value must be the same as Result Type.  The type of the
value pointed to by Pointer must be the same as Result Type.

Memory must be a valid memory Scope.

<!-- End of AutoGen section -->

```
atomic-iadd-op ::=
    `spv.AtomicIAdd` scope memory-semantics
                     ssa-use `,` ssa-use `:` spv-pointer-type
```

#### Example:

```mlir
%0 = spv.AtomicIAdd \"Device\" \"None\" %pointer, %value :
                    !spv.ptr<i32, StorageBuffer>
```
"""
function AtomicIAdd(pointer::Value, value::Value; result::MLIRType, memory_scope, semantics, location=Location())
    results = MLIRType[result, ]
    operands = Value[pointer, value, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("memory_scope", memory_scope), namedattribute("semantics", semantics), ]
    
    create_operation(
        "spv.AtomicIAdd", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`AtomicIDecrement`

1) load through Pointer to get an Original Value,

2) get a New Value through integer subtraction of 1 from Original Value,
and

3) store the New Value back through Pointer.

The instruction’s result is the Original Value.

Result Type must be an integer type scalar.  The type of the value
pointed to by Pointer must be the same as Result Type.

Memory must be a valid memory Scope.

<!-- End of AutoGen section -->

```
atomic-idecrement-op ::=
    `spv.AtomicIDecrement` scope memory-semantics ssa-use
                           `:` spv-pointer-type
```

#### Example:

```mlir
%0 = spv.AtomicIDecrement \"Device\" \"None\" %pointer :
                          !spv.ptr<i32, StorageBuffer>
```
"""
function AtomicIDecrement(pointer::Value; result::MLIRType, memory_scope, semantics, location=Location())
    results = MLIRType[result, ]
    operands = Value[pointer, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("memory_scope", memory_scope), namedattribute("semantics", semantics), ]
    
    create_operation(
        "spv.AtomicIDecrement", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`AtomicIIncrement`

1) load through Pointer to get an Original Value,

2) get a New Value through integer addition of 1 to Original Value, and

3) store the New Value back through Pointer.

The instruction’s result is the Original Value.

Result Type must be an integer type scalar.  The type of the value
pointed to by Pointer must be the same as Result Type.

Memory must be a valid memory Scope.

<!-- End of AutoGen section -->

```
atomic-iincrement-op ::=
    `spv.AtomicIIncrement` scope memory-semantics ssa-use
                           `:` spv-pointer-type
```

#### Example:

```mlir
%0 = spv.AtomicIncrement \"Device\" \"None\" %pointer :
                         !spv.ptr<i32, StorageBuffer>
```
"""
function AtomicIIncrement(pointer::Value; result::MLIRType, memory_scope, semantics, location=Location())
    results = MLIRType[result, ]
    operands = Value[pointer, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("memory_scope", memory_scope), namedattribute("semantics", semantics), ]
    
    create_operation(
        "spv.AtomicIIncrement", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`AtomicISub`

1) load through Pointer to get an Original Value,

2) get a New Value by integer subtraction of Value from Original Value,
and

3) store the New Value back through Pointer.

The instruction’s result is the Original Value.

Result Type must be an integer type scalar.

 The type of Value must be the same as Result Type.  The type of the
value pointed to by Pointer must be the same as Result Type.

Memory must be a valid memory Scope.

<!-- End of AutoGen section -->

```
atomic-isub-op ::=
    `spv.AtomicISub` scope memory-semantics
                     ssa-use `,` ssa-use `:` spv-pointer-type
```

#### Example:

```mlir
%0 = spv.AtomicISub \"Device\" \"None\" %pointer, %value :
                    !spv.ptr<i32, StorageBuffer>
```
"""
function AtomicISub(pointer::Value, value::Value; result::MLIRType, memory_scope, semantics, location=Location())
    results = MLIRType[result, ]
    operands = Value[pointer, value, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("memory_scope", memory_scope), namedattribute("semantics", semantics), ]
    
    create_operation(
        "spv.AtomicISub", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`AtomicOr`

1) load through Pointer to get an Original Value,

2) get a New Value by the bitwise OR of Original Value and Value, and

3) store the New Value back through Pointer.

The instruction’s result is the Original Value.

Result Type must be an integer type scalar.

 The type of Value must be the same as Result Type.  The type of the
value pointed to by Pointer must be the same as Result Type.

Memory must be a valid memory Scope.

<!-- End of AutoGen section -->

```
atomic-or-op ::=
    `spv.AtomicOr` scope memory-semantics
                   ssa-use `,` ssa-use `:` spv-pointer-type
```

#### Example:

```mlir
%0 = spv.AtomicOr \"Device\" \"None\" %pointer, %value :
                  !spv.ptr<i32, StorageBuffer>
```
"""
function AtomicOr(pointer::Value, value::Value; result::MLIRType, memory_scope, semantics, location=Location())
    results = MLIRType[result, ]
    operands = Value[pointer, value, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("memory_scope", memory_scope), namedattribute("semantics", semantics), ]
    
    create_operation(
        "spv.AtomicOr", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`AtomicSMax`

1) load through Pointer to get an Original Value,

2) get a New Value by finding the largest signed integer of Original
Value and Value, and

3) store the New Value back through Pointer.

The instruction’s result is the Original Value.

Result Type must be an integer type scalar.

 The type of Value must be the same as Result Type.  The type of the
value pointed to by Pointer must be the same as Result Type.

Memory must be a valid memory Scope.

<!-- End of AutoGen section -->

```
atomic-smax-op ::=
    `spv.AtomicSMax` scope memory-semantics
                     ssa-use `,` ssa-use `:` spv-pointer-type
```

#### Example:

```mlir
%0 = spv.AtomicSMax \"Device\" \"None\" %pointer, %value :
                    !spv.ptr<i32, StorageBuffer>
```
"""
function AtomicSMax(pointer::Value, value::Value; result::MLIRType, memory_scope, semantics, location=Location())
    results = MLIRType[result, ]
    operands = Value[pointer, value, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("memory_scope", memory_scope), namedattribute("semantics", semantics), ]
    
    create_operation(
        "spv.AtomicSMax", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`AtomicSMin`

1) load through Pointer to get an Original Value,

2) get a New Value by finding the smallest signed integer of Original
Value and Value, and

3) store the New Value back through Pointer.

The instruction’s result is the Original Value.

Result Type must be an integer type scalar.

 The type of Value must be the same as Result Type.  The type of the
value pointed to by Pointer must be the same as Result Type.

Memory must be a valid memory Scope.

<!-- End of AutoGen section -->

```
atomic-smin-op ::=
    `spv.AtomicSMin` scope memory-semantics
                     ssa-use `,` ssa-use `:` spv-pointer-type
```

#### Example:

```mlir
%0 = spv.AtomicSMin \"Device\" \"None\" %pointer, %value :
                    !spv.ptr<i32, StorageBuffer>
```
"""
function AtomicSMin(pointer::Value, value::Value; result::MLIRType, memory_scope, semantics, location=Location())
    results = MLIRType[result, ]
    operands = Value[pointer, value, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("memory_scope", memory_scope), namedattribute("semantics", semantics), ]
    
    create_operation(
        "spv.AtomicSMin", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`AtomicUMax`

1) load through Pointer to get an Original Value,

2) get a New Value by finding the largest unsigned integer of Original
Value and Value, and

3) store the New Value back through Pointer.

The instruction’s result is the Original Value.

Result Type must be an integer type scalar.

 The type of Value must be the same as Result Type.  The type of the
value pointed to by Pointer must be the same as Result Type.

Memory must be a valid memory Scope.

<!-- End of AutoGen section -->

```
atomic-umax-op ::=
    `spv.AtomicUMax` scope memory-semantics
                     ssa-use `,` ssa-use `:` spv-pointer-type
```

#### Example:

```mlir
%0 = spv.AtomicUMax \"Device\" \"None\" %pointer, %value :
                    !spv.ptr<i32, StorageBuffer>
```
"""
function AtomicUMax(pointer::Value, value::Value; result::MLIRType, memory_scope, semantics, location=Location())
    results = MLIRType[result, ]
    operands = Value[pointer, value, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("memory_scope", memory_scope), namedattribute("semantics", semantics), ]
    
    create_operation(
        "spv.AtomicUMax", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`AtomicUMin`

1) load through Pointer to get an Original Value,

2) get a New Value by finding the smallest unsigned integer of Original
Value and Value, and

3) store the New Value back through Pointer.

The instruction’s result is the Original Value.

Result Type must be an integer type scalar.

 The type of Value must be the same as Result Type.  The type of the
value pointed to by Pointer must be the same as Result Type.

Memory must be a valid memory Scope.

<!-- End of AutoGen section -->

```
atomic-umin-op ::=
    `spv.AtomicUMin` scope memory-semantics
                     ssa-use `,` ssa-use `:` spv-pointer-type
```

#### Example:

```mlir
%0 = spv.AtomicUMin \"Device\" \"None\" %pointer, %value :
                    !spv.ptr<i32, StorageBuffer>
```
"""
function AtomicUMin(pointer::Value, value::Value; result::MLIRType, memory_scope, semantics, location=Location())
    results = MLIRType[result, ]
    operands = Value[pointer, value, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("memory_scope", memory_scope), namedattribute("semantics", semantics), ]
    
    create_operation(
        "spv.AtomicUMin", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`AtomicXor`

1) load through Pointer to get an Original Value,

2) get a New Value by the bitwise exclusive OR of Original Value and
Value, and

3) store the New Value back through Pointer.

The instruction’s result is the Original Value.

Result Type must be an integer type scalar.

 The type of Value must be the same as Result Type.  The type of the
value pointed to by Pointer must be the same as Result Type.

Memory must be a valid memory Scope.

<!-- End of AutoGen section -->

```
atomic-xor-op ::=
    `spv.AtomicXor` scope memory-semantics
                    ssa-use `,` ssa-use `:` spv-pointer-type
```

#### Example:

```mlir
%0 = spv.AtomicXor \"Device\" \"None\" %pointer, %value :
                   !spv.ptr<i32, StorageBuffer>
```
"""
function AtomicXor(pointer::Value, value::Value; result::MLIRType, memory_scope, semantics, location=Location())
    results = MLIRType[result, ]
    operands = Value[pointer, value, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("memory_scope", memory_scope), namedattribute("semantics", semantics), ]
    
    create_operation(
        "spv.AtomicXor", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`BitCount`

Results are computed per component.

    Result Type must be a scalar or vector of integer type.  The components
    must be wide enough to hold the unsigned Width of Base as an unsigned
    value. That is, no sign bit is needed or counted when checking for a
    wide enough result width.

    Base must be a scalar or vector of integer type.  It must have the same
    number of components as Result Type.

    The result is the unsigned value that is the number of bits in Base that
    are 1.

    <!-- End of AutoGen section -->

    ```
    integer-scalar-vector-type ::= integer-type |
                             `vector<` integer-literal `x` integer-type `>`
    bitcount-op ::= ssa-id `=` `spv.BitCount` ssa-use
                          `:` integer-scalar-vector-type
    ```

    #### Example:

    ```mlir
    %2 = spv.BitCount %0: i32
    %3 = spv.BitCount %1: vector<4xi32>
    ```
"""
function BitCount(operand::Value; result=nothing::Union{Nothing, MLIRType}, location=Location())
    results = MLIRType[]
    operands = Value[operand, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (result != nothing) && push!(results, result)
    
    create_operation(
        "spv.BitCount", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`BitFieldInsert`

Results are computed per component.

    Result Type must be a scalar or vector of integer type.

The type of Base and Insert must be the same as Result Type.

    Any result bits numbered outside [Offset, Offset + Count -  1]
    (inclusive) will come from the corresponding bits in Base.

    Any result bits numbered in [Offset, Offset + Count -  1] come, in
    order, from the bits numbered [0, Count - 1] of Insert.

    Count  must be an integer type scalar. Count is the number of bits taken
    from Insert. It will be consumed as an unsigned value. Count can be 0,
    in which case the result will be Base.

    Offset  must be an integer type scalar. Offset is the lowest-order bit
    of the bit field.  It will be consumed as an unsigned value.

    The resulting value is undefined if Count or Offset or their sum is
    greater than the number of bits in the result.

    <!-- End of AutoGen section -->

    ```
    integer-scalar-vector-type ::= integer-type |
                             `vector<` integer-literal `x` integer-type `>`
    bitfield-insert-op ::= ssa-id `=` `spv.BitFieldInsert` ssa-use `,` ssa-use
                                 `,` ssa-use `,` ssa-use
                                 `:` integer-scalar-vector-type
                                 `,` integer-type `,` integer-type
    ```

    #### Example:

    ```mlir
    %0 = spv.BitFieldInsert %base, %insert, %offset, %count : vector<3xi32>, i8, i8
    ```
"""
function BitFieldInsert(base::Value, insert::Value, offset::Value, count::Value; result=nothing::Union{Nothing, MLIRType}, location=Location())
    results = MLIRType[]
    operands = Value[base, insert, offset, count, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (result != nothing) && push!(results, result)
    
    create_operation(
        "spv.BitFieldInsert", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`BitFieldSExtract`

Results are computed per component.

    Result Type must be a scalar or vector of integer type.

The type of Base must be the same as Result Type.

    If Count is greater than 0: The bits of Base numbered in [Offset, Offset
    + Count -  1] (inclusive) become the bits numbered [0, Count - 1] of the
    result. The remaining bits of the result will all be the same as bit
    Offset + Count -  1 of Base.

    Count  must be an integer type scalar. Count is the number of bits
    extracted from Base. It will be consumed as an unsigned value. Count can
    be 0, in which case the result will be 0.

    Offset  must be an integer type scalar. Offset is the lowest-order bit
    of the bit field to extract from Base.  It will be consumed as an
    unsigned value.

    The resulting value is undefined if Count or Offset or their sum is
    greater than the number of bits in the result.

    <!-- End of AutoGen section -->

    ```
    integer-scalar-vector-type ::= integer-type |
                             `vector<` integer-literal `x` integer-type `>`
    bitfield-extract-s-op ::= ssa-id `=` `spv.BitFieldSExtract` ssa-use
                                    `,` ssa-use `,` ssa-use
                                    `:` integer-scalar-vector-type
                                    `,` integer-type `,` integer-type
    ```

    #### Example:

    ```mlir
    %0 = spv.BitFieldSExtract %base, %offset, %count : vector<3xi32>, i8, i8
    ```
"""
function BitFieldSExtract(base::Value, offset::Value, count::Value; result=nothing::Union{Nothing, MLIRType}, location=Location())
    results = MLIRType[]
    operands = Value[base, offset, count, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (result != nothing) && push!(results, result)
    
    create_operation(
        "spv.BitFieldSExtract", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`BitFieldUExtract`

The semantics are the same as with OpBitFieldSExtract with the exception
that there is no sign extension. The remaining bits of the result will
all be 0.

<!-- End of AutoGen section -->

```
integer-scalar-vector-type ::= integer-type |
                              `vector<` integer-literal `x` integer-type `>`
bitfield-extract-u-op ::= ssa-id `=` `spv.BitFieldUExtract` ssa-use
                                     `,` ssa-use `,` ssa-use
                                     `:` integer-scalar-vector-type
                                     `,` integer-type `,` integer-type
```

#### Example:

```mlir
%0 = spv.BitFieldUExtract %base, %offset, %count : vector<3xi32>, i8, i8
```
"""
function BitFieldUExtract(base::Value, offset::Value, count::Value; result=nothing::Union{Nothing, MLIRType}, location=Location())
    results = MLIRType[]
    operands = Value[base, offset, count, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (result != nothing) && push!(results, result)
    
    create_operation(
        "spv.BitFieldUExtract", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`BitReverse`

Results are computed per component.

    Result Type must be a scalar or vector of integer type.

The type of Base must be the same as Result Type.

    The bit-number n of the result will be taken from bit-number Width - 1 -
    n of Base, where Width is the OpTypeInt operand of the Result Type.

    <!-- End of AutoGen section -->

    ```
    integer-scalar-vector-type ::= integer-type |
                              `vector<` integer-literal `x` integer-type `>`
    bitreverse-op ::= ssa-id `=` `spv.BitReverse` ssa-use
                            `:` integer-scalar-vector-type
    ```

    #### Example:

    ```mlir
    %2 = spv.BitReverse %0 : i32
    %3 = spv.BitReverse %1 : vector<4xi32>
    ```
"""
function BitReverse(operand::Value; result=nothing::Union{Nothing, MLIRType}, location=Location())
    results = MLIRType[]
    operands = Value[operand, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (result != nothing) && push!(results, result)
    
    create_operation(
        "spv.BitReverse", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`Bitcast`

Result Type must be an OpTypePointer, or a scalar or vector of
numerical-type.

Operand must have a type of OpTypePointer, or a scalar or vector of
numerical-type. It must be a different type than Result Type.

If either Result Type or Operand is a pointer, the other must be a
pointer (diverges from the SPIR-V spec).

If Result Type has a different number of components than Operand, the
total number of bits in Result Type must equal the total number of bits
in Operand. Let L be the type, either Result Type or Operand’s type,
that has the larger number of components. Let S be the other type, with
the smaller number of components. The number of components in L must be
an integer multiple of the number of components in S. The first
component (that is, the only or lowest-numbered component) of S maps to
the first components of L, and so on,  up to the last component of S
mapping to the last components of L. Within this mapping, any single
component of S (mapping to multiple components of L) maps its lower-
ordered bits to the lower-numbered components of L.

<!-- End of AutoGen section -->

```
bitcast-op ::= ssa-id `=` `spv.Bitcast` ssa-use
               `:` operand-type `to` result-type
```

#### Example:

```mlir
%1 = spv.Bitcast %0 : f32 to i32
%1 = spv.Bitcast %0 : vector<2xf32> to i64
%1 = spv.Bitcast %0 : !spv.ptr<f32, Function> to !spv.ptr<i32, Function>
```
"""
function Bitcast(operand::Value; result::MLIRType, location=Location())
    results = MLIRType[result, ]
    operands = Value[operand, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "spv.Bitcast", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`BitwiseAnd`

Results are computed per component, and within each component, per bit.

    Result Type must be a scalar or vector of integer type.  The type of
    Operand 1 and Operand 2  must be a scalar or vector of integer type.
    They must have the same number of components as Result Type. They must
    have the same component width as Result Type.

    <!-- End of AutoGen section -->

    ```
    integer-scalar-vector-type ::= integer-type |
                             `vector<` integer-literal `x` integer-type `>`
    bitwise-and-op ::= ssa-id `=` `spv.BitwiseAnd` ssa-use, ssa-use
                             `:` integer-scalar-vector-type
    ```

    #### Example:

    ```mlir
    %2 = spv.BitwiseAnd %0, %1 : i32
    %2 = spv.BitwiseAnd %0, %1 : vector<4xi32>
    ```
"""
function BitwiseAnd(operand1::Value, operand2::Value; result=nothing::Union{Nothing, MLIRType}, location=Location())
    results = MLIRType[]
    operands = Value[operand1, operand2, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (result != nothing) && push!(results, result)
    
    create_operation(
        "spv.BitwiseAnd", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`BitwiseOr`

Results are computed per component, and within each component, per bit.

    Result Type must be a scalar or vector of integer type.  The type of
    Operand 1 and Operand 2  must be a scalar or vector of integer type.
    They must have the same number of components as Result Type. They must
    have the same component width as Result Type.

    <!-- End of AutoGen section -->

    ```
    integer-scalar-vector-type ::= integer-type |
                             `vector<` integer-literal `x` integer-type `>`
    bitwise-or-op ::= ssa-id `=` `spv.BitwiseOr` ssa-use, ssa-use
                             `:` integer-scalar-vector-type
    ```

    #### Example:

    ```mlir
    %2 = spv.BitwiseOr %0, %1 : i32
    %2 = spv.BitwiseOr %0, %1 : vector<4xi32>
    ```
"""
function BitwiseOr(operand1::Value, operand2::Value; result=nothing::Union{Nothing, MLIRType}, location=Location())
    results = MLIRType[]
    operands = Value[operand1, operand2, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (result != nothing) && push!(results, result)
    
    create_operation(
        "spv.BitwiseOr", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`BitwiseXor`

Results are computed per component, and within each component, per bit.

    Result Type must be a scalar or vector of integer type.  The type of
    Operand 1 and Operand 2  must be a scalar or vector of integer type.
    They must have the same number of components as Result Type. They must
    have the same component width as Result Type.

    <!-- End of AutoGen section -->

    ```
    integer-scalar-vector-type ::= integer-type |
                             `vector<` integer-literal `x` integer-type `>`
    bitwise-xor-op ::= ssa-id `=` `spv.BitwiseXor` ssa-use, ssa-use
                             `:` integer-scalar-vector-type
    ```

    #### Example:

    ```mlir
    %2 = spv.BitwiseXor %0, %1 : i32
    %2 = spv.BitwiseXor %0, %1 : vector<4xi32>
    ```
"""
function BitwiseXor(operand1::Value, operand2::Value; result=nothing::Union{Nothing, MLIRType}, location=Location())
    results = MLIRType[]
    operands = Value[operand1, operand2, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (result != nothing) && push!(results, result)
    
    create_operation(
        "spv.BitwiseXor", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`BranchConditional`

Condition must be a Boolean type scalar.

Branch weights are unsigned 32-bit integer literals. There must be
either no Branch Weights or exactly two branch weights. If present, the
first is the weight for branching to True Label, and the second is the
weight for branching to False Label. The implied probability that a
branch is taken is its weight divided by the sum of the two Branch
weights. At least one weight must be non-zero. A weight of zero does not
imply a branch is dead or permit its removal; branch weights are only
hints. The two weights must not overflow a 32-bit unsigned integer when
added together.

This instruction must be the last instruction in a block.

<!-- End of AutoGen section -->

```
branch-conditional-op ::= `spv.BranchConditional` ssa-use
                          (`[` integer-literal, integer-literal `]`)?
                          `,` successor `,` successor
successor ::= bb-id branch-use-list?
branch-use-list ::= `(` ssa-use-list `:` type-list-no-parens `)`
```

#### Example:

```mlir
spv.BranchConditional %condition, ^true_branch, ^false_branch
spv.BranchConditional %condition, ^true_branch(%0: i32), ^false_branch(%1: i32)
```
"""
function BranchConditional(condition::Value, trueTargetOperands::Vector{Value}, falseTargetOperands::Vector{Value}; branch_weights=nothing, trueTarget::Block, falseTarget::Block, location=Location())
    results = MLIRType[]
    operands = Value[condition, trueTargetOperands..., falseTargetOperands..., ]
    owned_regions = Region[]
    successors = Block[trueTarget, falseTarget, ]
    attributes = NamedAttribute[]
    push!(attributes, operandsegmentsizes([1, length(trueTargetOperands), length(falseTargetOperands), ]))
    (branch_weights != nothing) && push!(attributes, namedattribute("branch_weights", branch_weights))
    
    create_operation(
        "spv.BranchConditional", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`Branch`

This instruction must be the last instruction in a block.

<!-- End of AutoGen section -->

```
branch-op ::= `spv.Branch` successor
successor ::= bb-id branch-use-list?
branch-use-list ::= `(` ssa-use-list `:` type-list-no-parens `)`
```

#### Example:

```mlir
spv.Branch ^target
spv.Branch ^target(%0, %1: i32, f32)
```
"""
function Branch(targetOperands::Vector{Value}; target::Block, location=Location())
    results = MLIRType[]
    operands = Value[targetOperands..., ]
    owned_regions = Region[]
    successors = Block[target, ]
    attributes = NamedAttribute[]
    
    create_operation(
        "spv.Branch", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`CL_ceil`

Result Type and x must be floating-point or vector(2,3,4,8,16) of
floating-point values.

All of the operands, including the Result Type operand, must be of the
same type.

<!-- End of AutoGen section -->

```
float-scalar-vector-type ::= float-type |
                             `vector<` integer-literal `x` float-type `>`
ceil-op ::= ssa-id `=` `spv.CL.ceil` ssa-use `:`
           float-scalar-vector-type
```mlir

#### Example:

```
%2 = spv.CL.ceil %0 : f32
%3 = spv.CL.ceil %1 : vector<3xf16>
```
"""
function CL_ceil(operand::Value; result=nothing::Union{Nothing, MLIRType}, location=Location())
    results = MLIRType[]
    operands = Value[operand, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (result != nothing) && push!(results, result)
    
    create_operation(
        "spv.CL.ceil", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`CL_cos`

Result Type and x must be floating-point or vector(2,3,4,8,16) of
floating-point values.

All of the operands, including the Result Type operand, must be of the
same type.

<!-- End of AutoGen section -->

```
float-scalar-vector-type ::= float-type |
                             `vector<` integer-literal `x` float-type `>`
cos-op ::= ssa-id `=` `spv.CL.cos` ssa-use `:`
           float-scalar-vector-type
```mlir

#### Example:

```
%2 = spv.CL.cos %0 : f32
%3 = spv.CL.cos %1 : vector<3xf16>
```
"""
function CL_cos(operand::Value; result=nothing::Union{Nothing, MLIRType}, location=Location())
    results = MLIRType[]
    operands = Value[operand, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (result != nothing) && push!(results, result)
    
    create_operation(
        "spv.CL.cos", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`CL_erf`

Result Type and x must be floating-point or vector(2,3,4,8,16) of
floating-point values.

All of the operands, including the Result Type operand, must be of the
same type.

<!-- End of AutoGen section -->

```
float-scalar-vector-type ::= float-type |
                             `vector<` integer-literal `x` float-type `>`
erf-op ::= ssa-id `=` `spv.CL.erf` ssa-use `:`
           float-scalar-vector-type
```mlir

#### Example:

```
%2 = spv.CL.erf %0 : f32
%3 = spv.CL.erf %1 : vector<3xf16>
```
"""
function CL_erf(operand::Value; result=nothing::Union{Nothing, MLIRType}, location=Location())
    results = MLIRType[]
    operands = Value[operand, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (result != nothing) && push!(results, result)
    
    create_operation(
        "spv.CL.erf", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`CL_exp`

Compute the base-e exponential of x. (i.e. ex)

Result Type and x must be floating-point or vector(2,3,4,8,16) of
floating-point values.

All of the operands, including the Result Type operand,
must be of the same type.

<!-- End of AutoGen section -->
```
float-scalar-vector-type ::= float-type |
                             `vector<` integer-literal `x` float-type `>`
exp-op ::= ssa-id `=` `spv.CL.exp` ssa-use `:`
           float-scalar-vector-type
```
#### Example:

```mlir
%2 = spv.CL.exp %0 : f32
%3 = spv.CL.exp %1 : vector<3xf16>
```
"""
function CL_exp(operand::Value; result=nothing::Union{Nothing, MLIRType}, location=Location())
    results = MLIRType[]
    operands = Value[operand, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (result != nothing) && push!(results, result)
    
    create_operation(
        "spv.CL.exp", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`CL_fabs`

Compute the absolute value of x.

Result Type and x must be floating-point or vector(2,3,4,8,16) of
floating-point values.

All of the operands, including the Result Type operand,
must be of the same type.

<!-- End of AutoGen section -->
```
float-scalar-vector-type ::= float-type |
                             `vector<` integer-literal `x` float-type `>`
abs-op ::= ssa-id `=` `spv.CL.fabs` ssa-use `:`
           float-scalar-vector-type
```
#### Example:

```mlir
%2 = spv.CL.fabs %0 : f32
%3 = spv.CL.fabs %1 : vector<3xf16>
```
"""
function CL_fabs(operand::Value; result=nothing::Union{Nothing, MLIRType}, location=Location())
    results = MLIRType[]
    operands = Value[operand, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (result != nothing) && push!(results, result)
    
    create_operation(
        "spv.CL.fabs", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`CL_floor`

Result Type and x must be floating-point or vector(2,3,4,8,16) of
floating-point values.

All of the operands, including the Result Type operand, must be of the
same type.

<!-- End of AutoGen section -->

```
float-scalar-vector-type ::= float-type |
                             `vector<` integer-literal `x` float-type `>`
floor-op ::= ssa-id `=` `spv.CL.floor` ssa-use `:`
           float-scalar-vector-type
```mlir

#### Example:

```
%2 = spv.CL.floor %0 : f32
%3 = spv.CL.ceifloorl %1 : vector<3xf16>
```
"""
function CL_floor(operand::Value; result=nothing::Union{Nothing, MLIRType}, location=Location())
    results = MLIRType[]
    operands = Value[operand, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (result != nothing) && push!(results, result)
    
    create_operation(
        "spv.CL.floor", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`CL_fma`

Result Type, a, b and c must be floating-point or vector(2,3,4,8,16) of
floating-point values.

All of the operands, including the Result Type operand, must be of the
same type.

<!-- End of AutoGen section -->

```
fma-op ::= ssa-id `=` `spv.CL.fma` ssa-use, ssa-use, ssa-use `:`
           float-scalar-vector-type
```mlir

```
%0 = spv.CL.fma %a, %b, %c : f32
%1 = spv.CL.fma %a, %b, %c : vector<3xf16>
```
"""
function CL_fma(x::Value, y::Value, z::Value; result=nothing::Union{Nothing, MLIRType}, location=Location())
    results = MLIRType[]
    operands = Value[x, y, z, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (result != nothing) && push!(results, result)
    
    create_operation(
        "spv.CL.fma", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`CL_log`

Result Type and x must be floating-point or vector(2,3,4,8,16) of
floating-point values.

All of the operands, including the Result Type operand, must be of the
same type.

<!-- End of AutoGen section -->

```
float-scalar-vector-type ::= float-type |
                             `vector<` integer-literal `x` float-type `>`
log-op ::= ssa-id `=` `spv.CL.log` ssa-use `:`
           float-scalar-vector-type
```mlir

#### Example:

```
%2 = spv.CL.log %0 : f32
%3 = spv.CL.log %1 : vector<3xf16>
```
"""
function CL_log(operand::Value; result=nothing::Union{Nothing, MLIRType}, location=Location())
    results = MLIRType[]
    operands = Value[operand, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (result != nothing) && push!(results, result)
    
    create_operation(
        "spv.CL.log", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`CL_pow`

Result Type, x and y must be floating-point or vector(2,3,4,8,16) of
floating-point values.

All of the operands, including the Result Type operand, must be of the
same type.

<!-- End of AutoGen section -->

```
restricted-float-scalar-type ::=  `f16` | `f32`
restricted-float-scalar-vector-type ::=
  restricted-float-scalar-type |
  `vector<` integer-literal `x` restricted-float-scalar-type `>`
pow-op ::= ssa-id `=` `spv.CL.pow` ssa-use `:`
           restricted-float-scalar-vector-type
```
#### Example:

```mlir
%2 = spv.CL.pow %0, %1 : f32
%3 = spv.CL.pow %0, %1 : vector<3xf16>
```
"""
function CL_pow(lhs::Value, rhs::Value; result=nothing::Union{Nothing, MLIRType}, location=Location())
    results = MLIRType[]
    operands = Value[lhs, rhs, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (result != nothing) && push!(results, result)
    
    create_operation(
        "spv.CL.pow", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`CL_round`

Result Type and x must be floating-point or vector(2,3,4,8,16) of
floating-point values.

All of the operands, including the Result Type operand, must be of the
same type.

<!-- End of AutoGen section -->

```
float-scalar-vector-type ::= float-type |
                             `vector<` integer-literal `x` float-type `>`
round-op ::= ssa-id `=` `spv.CL.round` ssa-use `:`
           float-scalar-vector-type
```
#### Example:

```mlir
%2 = spv.CL.round %0 : f32
%3 = spv.CL.round %0 : vector<3xf16>
```
"""
function CL_round(operand::Value; result=nothing::Union{Nothing, MLIRType}, location=Location())
    results = MLIRType[]
    operands = Value[operand, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (result != nothing) && push!(results, result)
    
    create_operation(
        "spv.CL.round", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`CL_rsqrt`

Result Type and x must be floating-point or vector(2,3,4,8,16) of
floating-point values.

All of the operands, including the Result Type operand, must be of the
same type.

<!-- End of AutoGen section -->

```
float-scalar-vector-type ::= float-type |
                             `vector<` integer-literal `x` float-type `>`
rsqrt-op ::= ssa-id `=` `spv.CL.rsqrt` ssa-use `:`
           float-scalar-vector-type
```mlir

#### Example:

```
%2 = spv.CL.rsqrt %0 : f32
%3 = spv.CL.rsqrt %1 : vector<3xf16>
```
"""
function CL_rsqrt(operand::Value; result=nothing::Union{Nothing, MLIRType}, location=Location())
    results = MLIRType[]
    operands = Value[operand, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (result != nothing) && push!(results, result)
    
    create_operation(
        "spv.CL.rsqrt", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`CL_s_abs`

Returns |x|, where x is treated as signed integer.

Result Type and x must be integer or vector(2,3,4,8,16) of
integer values.

All of the operands, including the Result Type operand,
must be of the same type.

<!-- End of AutoGen section -->
```
integer-scalar-vector-type ::= integer-type |
                               `vector<` integer-literal `x` integer-type `>`
abs-op ::= ssa-id `=` `spv.CL.s_abs` ssa-use `:`
           integer-scalar-vector-type
```
#### Example:

```mlir
%2 = spv.CL.s_abs %0 : i32
%3 = spv.CL.s_abs %1 : vector<3xi16>
```
"""
function CL_s_abs(operand::Value; result=nothing::Union{Nothing, MLIRType}, location=Location())
    results = MLIRType[]
    operands = Value[operand, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (result != nothing) && push!(results, result)
    
    create_operation(
        "spv.CL.s_abs", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`CL_sin`

Result Type and x must be floating-point or vector(2,3,4,8,16) of
floating-point values.

All of the operands, including the Result Type operand, must be of the
same type.

<!-- End of AutoGen section -->

```
float-scalar-vector-type ::= float-type |
                             `vector<` integer-literal `x` float-type `>`
sin-op ::= ssa-id `=` `spv.CL.sin` ssa-use `:`
           float-scalar-vector-type
```mlir

#### Example:

```
%2 = spv.CL.sin %0 : f32
%3 = spv.CL.sin %1 : vector<3xf16>
```
"""
function CL_sin(operand::Value; result=nothing::Union{Nothing, MLIRType}, location=Location())
    results = MLIRType[]
    operands = Value[operand, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (result != nothing) && push!(results, result)
    
    create_operation(
        "spv.CL.sin", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`CL_sqrt`

Result Type and x must be floating-point or vector(2,3,4,8,16) of
floating-point values.

All of the operands, including the Result Type operand, must be of the
same type.

<!-- End of AutoGen section -->

```
float-scalar-vector-type ::= float-type |
                             `vector<` integer-literal `x` float-type `>`
sqrt-op ::= ssa-id `=` `spv.CL.sqrt` ssa-use `:`
           float-scalar-vector-type
```mlir

#### Example:

```
%2 = spv.CL.sqrt %0 : f32
%3 = spv.CL.sqrt %1 : vector<3xf16>
```
"""
function CL_sqrt(operand::Value; result=nothing::Union{Nothing, MLIRType}, location=Location())
    results = MLIRType[]
    operands = Value[operand, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (result != nothing) && push!(results, result)
    
    create_operation(
        "spv.CL.sqrt", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`CL_tanh`

Result Type and x must be floating-point or vector(2,3,4,8,16) of
floating-point values.

All of the operands, including the Result Type operand, must be of the
same type.

<!-- End of AutoGen section -->

```
float-scalar-vector-type ::= float-type |
                             `vector<` integer-literal `x` float-type `>`
tanh-op ::= ssa-id `=` `spv.CL.tanh` ssa-use `:`
           float-scalar-vector-type
```mlir

#### Example:

```
%2 = spv.CL.tanh %0 : f32
%3 = spv.CL.tanh %1 : vector<3xf16>
```
"""
function CL_tanh(operand::Value; result=nothing::Union{Nothing, MLIRType}, location=Location())
    results = MLIRType[]
    operands = Value[operand, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (result != nothing) && push!(results, result)
    
    create_operation(
        "spv.CL.tanh", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`CompositeConstruct`

Result Type must be a composite type, whose top-level
members/elements/components/columns have the same type as the types of
the operands, with one exception. The exception is that for constructing
a vector, the operands may also be vectors with the same component type
as the Result Type component type. When constructing a vector, the total
number of components in all the operands must equal the number of
components in Result Type.

Constituents will become members of a structure, or elements of an
array, or components of a vector, or columns of a matrix. There must be
exactly one Constituent for each top-level
member/element/component/column of the result, with one exception. The
exception is that for constructing a vector, a contiguous subset of the
scalars consumed can be represented by a vector operand instead. The
Constituents must appear in the order needed by the definition of the
type of the result. When constructing a vector, there must be at least
two Constituent operands.

<!-- End of AutoGen section -->

```
composite-construct-op ::= ssa-id `=` `spv.CompositeConstruct`
                           (ssa-use (`,` ssa-use)* )? `:` composite-type
```

#### Example:

```mlir
%0 = spv.CompositeConstruct %1, %2, %3 : vector<3xf32>
```
"""
function CompositeConstruct(constituents::Vector{Value}; result::MLIRType, location=Location())
    results = MLIRType[result, ]
    operands = Value[constituents..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "spv.CompositeConstruct", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`CompositeExtract`

Result Type must be the type of object selected by the last provided
index.  The instruction result is the extracted object.

Composite is the composite to extract from.

Indexes walk the type hierarchy, potentially down to component
granularity, to select the part to extract. All indexes must be in
bounds.  All composite constituents use zero-based numbering, as
described by their OpType… instruction.

<!-- End of AutoGen section -->

```
composite-extract-op ::= ssa-id `=` `spv.CompositeExtract` ssa-use
                         `[` integer-literal (\',\' integer-literal)* `]`
                         `:` composite-type
```

#### Example:

```mlir
%0 = spv.Variable : !spv.ptr<!spv.array<4x!spv.array<4xf32>>, Function>
%1 = spv.Load \"Function\" %0 [\"Volatile\"] : !spv.array<4x!spv.array<4xf32>>
%2 = spv.CompositeExtract %1[1 : i32] : !spv.array<4x!spv.array<4xf32>>
```
"""
function CompositeExtract(composite::Value; component::MLIRType, indices, location=Location())
    results = MLIRType[component, ]
    operands = Value[composite, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("indices", indices), ]
    
    create_operation(
        "spv.CompositeExtract", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`CompositeInsert`

Result Type must be the same type as Composite.

Object is the object to use as the modified part.

Composite is the composite to copy all but the modified part from.

Indexes walk the type hierarchy of Composite to the desired depth,
potentially down to component granularity, to select the part to modify.
All indexes must be in bounds. All composite constituents use zero-based
numbering, as described by their OpType… instruction. The type of the
part selected to modify must match the type of Object.

<!-- End of AutoGen section -->

```
composite-insert-op ::= ssa-id `=` `spv.CompositeInsert` ssa-use, ssa-use
                        `[` integer-literal (\',\' integer-literal)* `]`
                        `:` object-type `into` composite-type
```

#### Example:

```mlir
%0 = spv.CompositeInsert %object, %composite[1 : i32] : f32 into !spv.array<4xf32>
```
"""
function CompositeInsert(object::Value, composite::Value; result::MLIRType, indices, location=Location())
    results = MLIRType[result, ]
    operands = Value[object, composite, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("indices", indices), ]
    
    create_operation(
        "spv.CompositeInsert", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`Constant`

This op declares a SPIR-V normal constant. SPIR-V has multiple constant
instructions covering different constant types:

* `OpConstantTrue` and `OpConstantFalse` for boolean constants
* `OpConstant` for scalar constants
* `OpConstantComposite` for composite constants
* `OpConstantNull` for null constants
* ...

Having such a plethora of constant instructions renders IR transformations
more tedious. Therefore, we use a single `spv.Constant` op to represent
them all. Note that conversion between those SPIR-V constant instructions
and this op is purely mechanical; so it can be scoped to the binary
(de)serialization process.

<!-- End of AutoGen section -->

```
spv.Constant-op ::= ssa-id `=` `spv.Constant` attribute-value
                    (`:` spirv-type)?
```

#### Example:

```mlir
%0 = spv.Constant true
%1 = spv.Constant dense<[2, 3]> : vector<2xf32>
%2 = spv.Constant [dense<3.0> : vector<2xf32>] : !spv.array<1xvector<2xf32>>
```

TODO: support constant structs
"""
function Constant(; constant::MLIRType, value, location=Location())
    results = MLIRType[constant, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("value", value), ]
    
    create_operation(
        "spv.Constant", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`ControlBarrier`

All invocations of this module within Execution scope must reach this
point of execution before any invocation will proceed beyond it.

When Execution is Workgroup or larger, behavior is undefined if this
instruction is used in control flow that is non-uniform within
Execution. When Execution is Subgroup or Invocation, the behavior of
this instruction in non-uniform control flow is defined by the client
API.

If Semantics is not None, this instruction also serves as an
OpMemoryBarrier instruction, and must also perform and adhere to the
description and semantics of an OpMemoryBarrier instruction with the
same Memory and Semantics operands.  This allows atomically specifying
both a control barrier and a memory barrier (that is, without needing
two instructions). If Semantics is None, Memory is ignored.

Before version 1.3, it is only valid to use this instruction with
TessellationControl, GLCompute, or Kernel execution models. There is no
such restriction starting with version 1.3.

When used with the TessellationControl execution model, it also
implicitly synchronizes the Output Storage Class:  Writes to Output
variables performed by any invocation executed prior to a
OpControlBarrier will be visible to any other invocation after return
from that OpControlBarrier.

<!-- End of AutoGen section -->

```
scope ::= `\"CrossDevice\"` | `\"Device\"` | `\"Workgroup\"` | ...

memory-semantics ::= `\"None\"` | `\"Acquire\"` | \"Release\"` | ...

control-barrier-op ::= `spv.ControlBarrier` scope, scope, memory-semantics
```

#### Example:

```mlir
spv.ControlBarrier \"Workgroup\", \"Device\", \"Acquire|UniformMemory\"

```
"""
function ControlBarrier(; execution_scope, memory_scope, memory_semantics, location=Location())
    results = MLIRType[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("execution_scope", execution_scope), namedattribute("memory_scope", memory_scope), namedattribute("memory_semantics", memory_semantics), ]
    
    create_operation(
        "spv.ControlBarrier", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`ConvertFToS`

Result Type must be a scalar or vector of integer type.

Float Value must be a scalar or vector of floating-point type.  It must
have the same number of components as Result Type.

 Results are computed per component.

<!-- End of AutoGen section -->

```
convert-f-to-s-op ::= ssa-id `=` `spv.ConvertFToSOp` ssa-use
                      `:` operand-type `to` result-type
```

#### Example:

```mlir
%1 = spv.ConvertFToS %0 : f32 to i32
%3 = spv.ConvertFToS %2 : vector<3xf32> to vector<3xi32>
```
"""
function ConvertFToS(operand::Value; result::MLIRType, location=Location())
    results = MLIRType[result, ]
    operands = Value[operand, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "spv.ConvertFToS", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`ConvertFToU`

Result Type must be a scalar or vector of integer type, whose Signedness
operand is 0.

Float Value must be a scalar or vector of floating-point type.  It must
have the same number of components as Result Type.

 Results are computed per component.

<!-- End of AutoGen section -->

```
convert-f-to-u-op ::= ssa-id `=` `spv.ConvertFToUOp` ssa-use
                      `:` operand-type `to` result-type
```

#### Example:

```mlir
%1 = spv.ConvertFToU %0 : f32 to i32
%3 = spv.ConvertFToU %2 : vector<3xf32> to vector<3xi32>
```
"""
function ConvertFToU(operand::Value; result::MLIRType, location=Location())
    results = MLIRType[result, ]
    operands = Value[operand, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "spv.ConvertFToU", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`ConvertSToF`

Result Type must be a scalar or vector of floating-point type.

Signed Value must be a scalar or vector of integer type.  It must have
the same number of components as Result Type.

 Results are computed per component.

<!-- End of AutoGen section -->

```
convert-s-to-f-op ::= ssa-id `=` `spv.ConvertSToFOp` ssa-use
                      `:` operand-type `to` result-type
```

#### Example:

```mlir
%1 = spv.ConvertSToF %0 : i32 to f32
%3 = spv.ConvertSToF %2 : vector<3xi32> to vector<3xf32>
```
"""
function ConvertSToF(operand::Value; result::MLIRType, location=Location())
    results = MLIRType[result, ]
    operands = Value[operand, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "spv.ConvertSToF", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`ConvertUToF`

Result Type must be a scalar or vector of floating-point type.

Unsigned Value must be a scalar or vector of integer type.  It must have
the same number of components as Result Type.

 Results are computed per component.

<!-- End of AutoGen section -->

```
convert-u-to-f-op ::= ssa-id `=` `spv.ConvertUToFOp` ssa-use
                      `:` operand-type `to` result-type
```

#### Example:

```mlir
%1 = spv.ConvertUToF %0 : i32 to f32
%3 = spv.ConvertUToF %2 : vector<3xi32> to vector<3xf32>
```
"""
function ConvertUToF(operand::Value; result::MLIRType, location=Location())
    results = MLIRType[result, ]
    operands = Value[operand, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "spv.ConvertUToF", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`CooperativeMatrixLengthNV`

Number of components of a cooperative matrix type accessible to each
invocation when treated as a composite.

Result Type must be an OpTypeInt with 32-bit Width and 0 Signedness.

Type is a cooperative matrix type.

``` {.ebnf}
cooperative-matrix-length-op ::= ssa-id `=` `spv.CooperativeMatrixLengthNV
                                ` : ` cooperative-matrix-type
```

For example:

```
%0 = spv.CooperativeMatrixLengthNV : !spv.coopmatrix<Subgroup, i32, 8, 16>
```
"""
function CooperativeMatrixLengthNV(; result=nothing::Union{Nothing, MLIRType}, type, location=Location())
    results = MLIRType[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("type", type), ]
    (result != nothing) && push!(results, result)
    
    create_operation(
        "spv.CooperativeMatrixLengthNV", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`CooperativeMatrixLoadNV`

Load a cooperative matrix through a pointer.

Result Type is the type of the loaded object. It must be a cooperative
matrix type.

Pointer is a pointer into an array. Its type must be an OpTypePointer whose
Type operand is a scalar or vector type. The storage class of Pointer must
be Workgroup, StorageBuffer, or (if SPV_EXT_physical_storage_buffer is
supported) PhysicalStorageBufferEXT.

Stride is the number of elements in the array in memory between the first
component of consecutive rows (or columns) in the result. It must be a
scalar integer type.

ColumnMajor indicates whether the values loaded from memory are arranged in
column-major or row-major order. It must be a boolean constant instruction,
with false indicating row major and true indicating column major.

Memory Access must be a Memory Access literal. If not present, it is the
same as specifying None.

If ColumnMajor is false, then elements (row,*) of the result are taken in
order from contiguous locations starting at Pointer[row*Stride]. If
ColumnMajor is true, then elements (*,col) of the result are taken in order
from contiguous locations starting from Pointer[col*Stride]. Any ArrayStride
decoration on Pointer is ignored.

For a given dynamic instance of this instruction, all operands of this
instruction must be the same for all invocations in a given scope instance
(where the scope is the scope the cooperative matrix type was created with).
All invocations in a given scope instance must be active or all must be
inactive.

### Custom assembly form

``` {.ebnf}
cooperative-matrixload-op ::= ssa-id `=` `spv.CooperativeMatrixLoadNV`
                          ssa-use `,` ssa-use `,` ssa-use
                          (`[` memory-access `]`)? ` : `
                          pointer-type `as`
                          cooperative-matrix-type
```

For example:

```
%0 = spv.CooperativeMatrixLoadNV %ptr, %stride, %colMajor
     : !spv.ptr<i32, StorageBuffer> as !spv.coopmatrix<i32, Workgroup, 16, 8>
```
"""
function CooperativeMatrixLoadNV(pointer::Value, stride::Value, columnmajor::Value; result::MLIRType, memory_access=nothing, location=Location())
    results = MLIRType[result, ]
    operands = Value[pointer, stride, columnmajor, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (memory_access != nothing) && push!(attributes, namedattribute("memory_access", memory_access))
    
    create_operation(
        "spv.CooperativeMatrixLoadNV", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`CooperativeMatrixMulAddNV`

Linear-algebraic matrix multiply of A by B and then component-wise add C.
The order of the operations is implementation-dependent. The internal
precision of floating-point operations is defined by the client API.
Integer operations are performed at the precision of the Result Type and are
exact unless there is overflow or underflow, in which case the result is
undefined.

Result Type must be a cooperative matrix type with M rows and N columns.

A is a cooperative matrix with M rows and K columns.

B is a cooperative matrix with K rows and N columns.

C is a cooperative matrix with M rows and N columns.

The values of M, N, and K must be consistent across the result and operands.
This is referred to as an MxNxK matrix multiply.

A, B, C, and Result Type must have the same scope, and this defines the
scope of the operation. A, B, C, and Result Type need not necessarily have
the same component type, this is defined by the client API.

If the Component Type of any matrix operand is an integer type, then its
components are treated as signed if its Component Type has Signedness of 1
and are treated as unsigned otherwise.

For a given dynamic instance of this instruction, all invocations in a given
scope instance must be active or all must be inactive (where the scope is
the scope of the operation).

``` {.ebnf}
cooperative-matrixmuladd-op ::= ssa-id `=` `spv.CooperativeMatrixMulAddNV`
                          ssa-use `,` ssa-use `,` ssa-use ` : `
                          a-cooperative-matrix-type,
                          b-cooperative-matrix-type ->
                          result-cooperative-matrix-type
```
For example:

```
%0 = spv.CooperativeMatrixMulAddNV %arg0, %arg1, %arg2,  :
  !spv.coopmatrix<Subgroup, i32, 8, 16>
```
"""
function CooperativeMatrixMulAddNV(a::Value, b::Value, c::Value; result=nothing::Union{Nothing, MLIRType}, location=Location())
    results = MLIRType[]
    operands = Value[a, b, c, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (result != nothing) && push!(results, result)
    
    create_operation(
        "spv.CooperativeMatrixMulAddNV", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`CooperativeMatrixStoreNV`

Store a cooperative matrix through a pointer.

Pointer is a pointer into an array. Its type must be an OpTypePointer whose
Type operand is a scalar or vector type. The storage class of Pointer must
be Workgroup, StorageBuffer, or (if SPV_EXT_physical_storage_buffer is
supported) PhysicalStorageBufferEXT.

Object is the object to store. Its type must be an
OpTypeCooperativeMatrixNV.

Stride is the number of elements in the array in memory between the first
component of consecutive rows (or columns) in the result. It must be a
scalar integer type.

ColumnMajor indicates whether the values stored to memory are arranged in
column-major or row-major order. It must be a boolean constant instruction,
with false indicating row major and true indicating column major.

Memory Access must be a Memory Access literal. If not present, it is the
same as specifying None.

``` {.ebnf}
coop-matrix-store-op ::= `spv.CooperativeMatrixStoreNV `
                          ssa-use `, ` ssa-use `, `
                          ssa-use `, ` ssa-use `, `
                          (`[` memory-access `]`)? `:`
                          pointer-type `,` spirv-element-type
```

For example:

```
  spv.CooperativeMatrixStoreNV %arg0, %arg2, %arg1, %arg3 :
    !spv.ptr<i32, StorageBuffer>, !spv.coopmatrix<Workgroup, i32, 16, 8>
```
"""
function CooperativeMatrixStoreNV(pointer::Value, object::Value, stride::Value, columnmajor::Value; memory_access=nothing, location=Location())
    results = MLIRType[]
    operands = Value[pointer, object, stride, columnmajor, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (memory_access != nothing) && push!(attributes, namedattribute("memory_access", memory_access))
    
    create_operation(
        "spv.CooperativeMatrixStoreNV", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`CopyMemory`

If present, any Memory Operands must begin with a memory operand
literal. If not present, it is the same as specifying the memory operand
None. Before version 1.4, at most one memory operands mask can be
provided. Starting with version 1.4 two masks can be provided, as
described in Memory Operands. If no masks or only one mask is present,
it applies to both Source and Target. If two masks are present, the
first applies to Target and cannot include MakePointerVisible, and the
second applies to Source and cannot include MakePointerAvailable.

<!-- End of AutoGen section -->

```
copy-memory-op ::= `spv.CopyMemory ` storage-class ssa-use
                   storage-class ssa-use
                   (`[` memory-access `]` (`, [` memory-access `]`)?)?
                   ` : ` spirv-element-type
```

#### Example:

```mlir
%0 = spv.Variable : !spv.ptr<f32, Function>
%1 = spv.Variable : !spv.ptr<f32, Function>
spv.CopyMemory \"Function\" %0, \"Function\" %1 : f32
```
"""
function CopyMemory(target::Value, source::Value; memory_access=nothing, alignment=nothing, source_memory_access=nothing, source_alignment=nothing, location=Location())
    results = MLIRType[]
    operands = Value[target, source, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (memory_access != nothing) && push!(attributes, namedattribute("memory_access", memory_access))
    (alignment != nothing) && push!(attributes, namedattribute("alignment", alignment))
    (source_memory_access != nothing) && push!(attributes, namedattribute("source_memory_access", source_memory_access))
    (source_alignment != nothing) && push!(attributes, namedattribute("source_alignment", source_alignment))
    
    create_operation(
        "spv.CopyMemory", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`EntryPoint`

Execution Model is the execution model for the entry point and its
static call tree. See Execution Model.

Entry Point must be the Result <id> of an OpFunction instruction.

Name is a name string for the entry point. A module cannot have two
OpEntryPoint instructions with the same Execution Model and the same
Name string.

Interface is a list of symbol references to `spv.GlobalVariable`
operations. These declare the set of global variables from a
module that form the interface of this entry point. The set of
Interface symbols must be equal to or a superset of the
`spv.GlobalVariable`s referenced by the entry point’s static call
tree, within the interface’s storage classes.  Before version 1.4,
the interface’s storage classes are limited to the Input and
Output storage classes. Starting with version 1.4, the interface’s
storage classes are all storage classes used in declaring all
global variables referenced by the entry point’s call tree.

<!-- End of AutoGen section -->

```
execution-model ::= \"Vertex\" | \"TesellationControl\" |
                    <and other SPIR-V execution models...>

entry-point-op ::= ssa-id `=` `spv.EntryPoint` execution-model
                   symbol-reference (`, ` symbol-reference)*
```

#### Example:

```mlir
spv.EntryPoint \"GLCompute\" @foo
spv.EntryPoint \"Kernel\" @foo, @var1, @var2

```
"""
function EntryPoint(; execution_model, fn, interface, location=Location())
    results = MLIRType[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("execution_model", execution_model), namedattribute("fn", fn), namedattribute("interface", interface), ]
    
    create_operation(
        "spv.EntryPoint", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`ExecutionMode`

Entry Point must be the Entry Point <id> operand of an OpEntryPoint
instruction.

Mode is the execution mode. See Execution Mode.

This instruction is only valid when the Mode operand is an execution
mode that takes no Extra Operands, or takes Extra Operands that are not
<id> operands.

<!-- End of AutoGen section -->

```
execution-mode ::= \"Invocations\" | \"SpacingEqual\" |
                   <and other SPIR-V execution modes...>

execution-mode-op ::= `spv.ExecutionMode ` ssa-use execution-mode
                      (integer-literal (`, ` integer-literal)* )?
```

#### Example:

```mlir
spv.ExecutionMode @foo \"ContractionOff\"
spv.ExecutionMode @bar \"LocalSizeHint\", 3, 4, 5
```
"""
function ExecutionMode(; fn, execution_mode, values, location=Location())
    results = MLIRType[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("fn", fn), namedattribute("execution_mode", execution_mode), namedattribute("values", values), ]
    
    create_operation(
        "spv.ExecutionMode", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`FAdd`

Result Type must be a scalar or vector of floating-point type.

 The types of Operand 1 and Operand 2 both must be the same as Result
Type.

 Results are computed per component.

<!-- End of AutoGen section -->
```
float-scalar-vector-type ::= float-type |
                             `vector<` integer-literal `x` float-type `>`
fadd-op ::= ssa-id `=` `spv.FAdd` ssa-use, ssa-use
                      `:` float-scalar-vector-type
```
#### Example:

```mlir
%4 = spv.FAdd %0, %1 : f32
%5 = spv.FAdd %2, %3 : vector<4xf32>
```
"""
function FAdd(operand1::Value, operand2::Value; result=nothing::Union{Nothing, MLIRType}, location=Location())
    results = MLIRType[]
    operands = Value[operand1, operand2, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (result != nothing) && push!(results, result)
    
    create_operation(
        "spv.FAdd", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`FConvert`

Result Type must be a scalar or vector of floating-point type.

Float Value must be a scalar or vector of floating-point type.  It must
have the same number of components as Result Type.  The component width
cannot equal the component width in Result Type.

 Results are computed per component.

<!-- End of AutoGen section -->

```
f-convert-op ::= ssa-id `=` `spv.FConvertOp` ssa-use
                 `:` operand-type `to` result-type
```

#### Example:

```mlir
%1 = spv.FConvertOp %0 : f32 to f64
%3 = spv.FConvertOp %2 : vector<3xf32> to vector<3xf64>
```
"""
function FConvert(operand::Value; result::MLIRType, location=Location())
    results = MLIRType[result, ]
    operands = Value[operand, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "spv.FConvert", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`FDiv`

Result Type must be a scalar or vector of floating-point type.

 The types of Operand 1 and Operand 2 both must be the same as Result
Type.

 Results are computed per component.  The resulting value is undefined
if Operand 2 is 0.

<!-- End of AutoGen section -->
```
float-scalar-vector-type ::= float-type |
                             `vector<` integer-literal `x` float-type `>`
fdiv-op ::= ssa-id `=` `spv.FDiv` ssa-use, ssa-use
                      `:` float-scalar-vector-type
```

#### Example:

```mlir
%4 = spv.FDiv %0, %1 : f32
%5 = spv.FDiv %2, %3 : vector<4xf32>
```
"""
function FDiv(operand1::Value, operand2::Value; result=nothing::Union{Nothing, MLIRType}, location=Location())
    results = MLIRType[]
    operands = Value[operand1, operand2, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (result != nothing) && push!(results, result)
    
    create_operation(
        "spv.FDiv", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`FMod`

Result Type must be a scalar or vector of floating-point type.

 The types of Operand 1 and Operand 2 both must be the same as Result
Type.

 Results are computed per component.  The resulting value is undefined
if Operand 2 is 0.  Otherwise, the result is the remainder r of Operand
1 divided by Operand 2 where if r ≠ 0, the sign of r is the same as the
sign of Operand 2.

<!-- End of AutoGen section -->
```
float-scalar-vector-type ::= float-type |
                             `vector<` integer-literal `x` float-type `>`
fmod-op ::= ssa-id `=` `spv.FMod` ssa-use, ssa-use
                      `:` float-scalar-vector-type
```
#### Example:

```mlir
%4 = spv.FMod %0, %1 : f32
%5 = spv.FMod %2, %3 : vector<4xf32>
```
"""
function FMod(operand1::Value, operand2::Value; result=nothing::Union{Nothing, MLIRType}, location=Location())
    results = MLIRType[]
    operands = Value[operand1, operand2, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (result != nothing) && push!(results, result)
    
    create_operation(
        "spv.FMod", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`FMul`

Result Type must be a scalar or vector of floating-point type.

 The types of Operand 1 and Operand 2 both must be the same as Result
Type.

 Results are computed per component.

<!-- End of AutoGen section -->

```
float-scalar-vector-type ::= float-type |
                             `vector<` integer-literal `x` float-type `>`
fmul-op ::= `spv.FMul` ssa-use, ssa-use
                      `:` float-scalar-vector-type
```

#### Example:

```mlir
%4 = spv.FMul %0, %1 : f32
%5 = spv.FMul %2, %3 : vector<4xf32>
```
"""
function FMul(operand1::Value, operand2::Value; result=nothing::Union{Nothing, MLIRType}, location=Location())
    results = MLIRType[]
    operands = Value[operand1, operand2, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (result != nothing) && push!(results, result)
    
    create_operation(
        "spv.FMul", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`FNegate`

Result Type must be a scalar or vector of floating-point type.

 The type of Operand must be the same as Result Type.

 Results are computed per component.

<!-- End of AutoGen section -->

```
float-scalar-vector-type ::= float-type |
                             `vector<` integer-literal `x` float-type `>`
fmul-op ::= `spv.FNegate` ssa-use `:` float-scalar-vector-type
```

#### Example:

```mlir
%1 = spv.FNegate %0 : f32
%3 = spv.FNegate %2 : vector<4xf32>
```
"""
function FNegate(operand::Value; result=nothing::Union{Nothing, MLIRType}, location=Location())
    results = MLIRType[]
    operands = Value[operand, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (result != nothing) && push!(results, result)
    
    create_operation(
        "spv.FNegate", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`FOrdEqual`

Result Type must be a scalar or vector of Boolean type.

 The type of Operand 1 and Operand 2  must be a scalar or vector of
floating-point type.  They must have the same type, and they must have
the same number of components as Result Type.

 Results are computed per component.

<!-- End of AutoGen section -->

```
float-scalar-vector-type ::= float-type |
                             `vector<` integer-literal `x` float-type `>`
fordequal-op ::= ssa-id `=` `spv.FOrdEqual` ssa-use, ssa-use
```

#### Example:

```mlir
%4 = spv.FOrdEqual %0, %1 : f32
%5 = spv.FOrdEqual %2, %3 : vector<4xf32>
```
"""
function FOrdEqual(operand1::Value, operand2::Value; result::MLIRType, location=Location())
    results = MLIRType[result, ]
    operands = Value[operand1, operand2, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "spv.FOrdEqual", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`FOrdGreaterThanEqual`

Result Type must be a scalar or vector of Boolean type.

 The type of Operand 1 and Operand 2  must be a scalar or vector of
floating-point type.  They must have the same type, and they must have
the same number of components as Result Type.

 Results are computed per component.

<!-- End of AutoGen section -->

```
float-scalar-vector-type ::= float-type |
                             `vector<` integer-literal `x` float-type `>`
fordgte-op ::= ssa-id `=` `spv.FOrdGreaterThanEqual` ssa-use, ssa-use
```

#### Example:

```mlir
%4 = spv.FOrdGreaterThanEqual %0, %1 : f32
%5 = spv.FOrdGreaterThanEqual %2, %3 : vector<4xf32>
```
"""
function FOrdGreaterThanEqual(operand1::Value, operand2::Value; result::MLIRType, location=Location())
    results = MLIRType[result, ]
    operands = Value[operand1, operand2, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "spv.FOrdGreaterThanEqual", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`FOrdGreaterThan`

Result Type must be a scalar or vector of Boolean type.

 The type of Operand 1 and Operand 2  must be a scalar or vector of
floating-point type.  They must have the same type, and they must have
the same number of components as Result Type.

 Results are computed per component.

<!-- End of AutoGen section -->

```
float-scalar-vector-type ::= float-type |
                             `vector<` integer-literal `x` float-type `>`
fordgt-op ::= ssa-id `=` `spv.FOrdGreaterThan` ssa-use, ssa-use
```

#### Example:

```mlir
%4 = spv.FOrdGreaterThan %0, %1 : f32
%5 = spv.FOrdGreaterThan %2, %3 : vector<4xf32>
```
"""
function FOrdGreaterThan(operand1::Value, operand2::Value; result::MLIRType, location=Location())
    results = MLIRType[result, ]
    operands = Value[operand1, operand2, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "spv.FOrdGreaterThan", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`FOrdLessThanEqual`

Result Type must be a scalar or vector of Boolean type.

 The type of Operand 1 and Operand 2  must be a scalar or vector of
floating-point type.  They must have the same type, and they must have
the same number of components as Result Type.

 Results are computed per component.

<!-- End of AutoGen section -->

```
float-scalar-vector-type ::= float-type |
                             `vector<` integer-literal `x` float-type `>`
fordlte-op ::= ssa-id `=` `spv.FOrdLessThanEqual` ssa-use, ssa-use
```

#### Example:

```mlir
%4 = spv.FOrdLessThanEqual %0, %1 : f32
%5 = spv.FOrdLessThanEqual %2, %3 : vector<4xf32>
```
"""
function FOrdLessThanEqual(operand1::Value, operand2::Value; result::MLIRType, location=Location())
    results = MLIRType[result, ]
    operands = Value[operand1, operand2, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "spv.FOrdLessThanEqual", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`FOrdLessThan`

Result Type must be a scalar or vector of Boolean type.

 The type of Operand 1 and Operand 2  must be a scalar or vector of
floating-point type.  They must have the same type, and they must have
the same number of components as Result Type.

 Results are computed per component.

<!-- End of AutoGen section -->

```
float-scalar-vector-type ::= float-type |
                             `vector<` integer-literal `x` float-type `>`
fordlt-op ::= ssa-id `=` `spv.FOrdLessThan` ssa-use, ssa-use
```

#### Example:

```mlir
%4 = spv.FOrdLessThan %0, %1 : f32
%5 = spv.FOrdLessThan %2, %3 : vector<4xf32>
```
"""
function FOrdLessThan(operand1::Value, operand2::Value; result::MLIRType, location=Location())
    results = MLIRType[result, ]
    operands = Value[operand1, operand2, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "spv.FOrdLessThan", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`FOrdNotEqual`

Result Type must be a scalar or vector of Boolean type.

 The type of Operand 1 and Operand 2  must be a scalar or vector of
floating-point type.  They must have the same type, and they must have
the same number of components as Result Type.

 Results are computed per component.

<!-- End of AutoGen section -->

```
float-scalar-vector-type ::= float-type |
                             `vector<` integer-literal `x` float-type `>`
fordneq-op ::= ssa-id `=` `spv.FOrdNotEqual` ssa-use, ssa-use
```

#### Example:

```mlir
%4 = spv.FOrdNotEqual %0, %1 : f32
%5 = spv.FOrdNotEqual %2, %3 : vector<4xf32>
```
"""
function FOrdNotEqual(operand1::Value, operand2::Value; result::MLIRType, location=Location())
    results = MLIRType[result, ]
    operands = Value[operand1, operand2, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "spv.FOrdNotEqual", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`FRem`

Result Type must be a scalar or vector of floating-point type.

 The types of Operand 1 and Operand 2 both must be the same as Result
Type.

 Results are computed per component.  The resulting value is undefined
if Operand 2 is 0.  Otherwise, the result is the remainder r of Operand
1 divided by Operand 2 where if r ≠ 0, the sign of r is the same as the
sign of Operand 1.

<!-- End of AutoGen section -->
```
float-scalar-vector-type ::= float-type |
                             `vector<` integer-literal `x` float-type `>`
frem-op ::= ssa-id `=` `spv.FRemOp` ssa-use, ssa-use
                      `:` float-scalar-vector-type
```

#### Example:

```mlir
%4 = spv.FRemOp %0, %1 : f32
%5 = spv.FRemOp %2, %3 : vector<4xf32>
```
"""
function FRem(operand1::Value, operand2::Value; result=nothing::Union{Nothing, MLIRType}, location=Location())
    results = MLIRType[]
    operands = Value[operand1, operand2, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (result != nothing) && push!(results, result)
    
    create_operation(
        "spv.FRem", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`FSub`

Result Type must be a scalar or vector of floating-point type.

 The types of Operand 1 and Operand 2 both must be the same as Result
Type.

 Results are computed per component.

<!-- End of AutoGen section -->
```
float-scalar-vector-type ::= float-type |
                             `vector<` integer-literal `x` float-type `>`
fsub-op ::= ssa-id `=` `spv.FRemOp` ssa-use, ssa-use
                      `:` float-scalar-vector-type
```

#### Example:

```mlir
%4 = spv.FRemOp %0, %1 : f32
%5 = spv.FRemOp %2, %3 : vector<4xf32>
```
"""
function FSub(operand1::Value, operand2::Value; result=nothing::Union{Nothing, MLIRType}, location=Location())
    results = MLIRType[]
    operands = Value[operand1, operand2, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (result != nothing) && push!(results, result)
    
    create_operation(
        "spv.FSub", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`FUnordEqual`

Result Type must be a scalar or vector of Boolean type.

 The type of Operand 1 and Operand 2  must be a scalar or vector of
floating-point type.  They must have the same type, and they must have
the same number of components as Result Type.

 Results are computed per component.

<!-- End of AutoGen section -->

```
float-scalar-vector-type ::= float-type |
                             `vector<` integer-literal `x` float-type `>`
funordequal-op ::= ssa-id `=` `spv.FUnordEqual` ssa-use, ssa-use
```

#### Example:

```mlir
%4 = spv.FUnordEqual %0, %1 : f32
%5 = spv.FUnordEqual %2, %3 : vector<4xf32>
```
"""
function FUnordEqual(operand1::Value, operand2::Value; result::MLIRType, location=Location())
    results = MLIRType[result, ]
    operands = Value[operand1, operand2, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "spv.FUnordEqual", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`FUnordGreaterThanEqual`

Result Type must be a scalar or vector of Boolean type.

 The type of Operand 1 and Operand 2  must be a scalar or vector of
floating-point type.  They must have the same type, and they must have
the same number of components as Result Type.

 Results are computed per component.

<!-- End of AutoGen section -->

```
float-scalar-vector-type ::= float-type |
                             `vector<` integer-literal `x` float-type `>`
funordgte-op ::= ssa-id `=` `spv.FUnordGreaterThanEqual` ssa-use, ssa-use
```

#### Example:

```mlir
%4 = spv.FUnordGreaterThanEqual %0, %1 : f32
%5 = spv.FUnordGreaterThanEqual %2, %3 : vector<4xf32>
```
"""
function FUnordGreaterThanEqual(operand1::Value, operand2::Value; result::MLIRType, location=Location())
    results = MLIRType[result, ]
    operands = Value[operand1, operand2, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "spv.FUnordGreaterThanEqual", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`FUnordGreaterThan`

Result Type must be a scalar or vector of Boolean type.

 The type of Operand 1 and Operand 2  must be a scalar or vector of
floating-point type.  They must have the same type, and they must have
the same number of components as Result Type.

 Results are computed per component.

<!-- End of AutoGen section -->

```
float-scalar-vector-type ::= float-type |
                             `vector<` integer-literal `x` float-type `>`
funordgt-op ::= ssa-id `=` `spv.FUnordGreaterThan` ssa-use, ssa-use
```

#### Example:

```mlir
%4 = spv.FUnordGreaterThan %0, %1 : f32
%5 = spv.FUnordGreaterThan %2, %3 : vector<4xf32>
```
"""
function FUnordGreaterThan(operand1::Value, operand2::Value; result::MLIRType, location=Location())
    results = MLIRType[result, ]
    operands = Value[operand1, operand2, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "spv.FUnordGreaterThan", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`FUnordLessThanEqual`

Result Type must be a scalar or vector of Boolean type.

 The type of Operand 1 and Operand 2  must be a scalar or vector of
floating-point type.  They must have the same type, and they must have
the same number of components as Result Type.

 Results are computed per component.

<!-- End of AutoGen section -->

```
float-scalar-vector-type ::= float-type |
                             `vector<` integer-literal `x` float-type `>`
funordlte-op ::= ssa-id `=` `spv.FUnordLessThanEqual` ssa-use, ssa-use
```

#### Example:

```mlir
%4 = spv.FUnordLessThanEqual %0, %1 : f32
%5 = spv.FUnordLessThanEqual %2, %3 : vector<4xf32>
```
"""
function FUnordLessThanEqual(operand1::Value, operand2::Value; result::MLIRType, location=Location())
    results = MLIRType[result, ]
    operands = Value[operand1, operand2, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "spv.FUnordLessThanEqual", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`FUnordLessThan`

Result Type must be a scalar or vector of Boolean type.

 The type of Operand 1 and Operand 2  must be a scalar or vector of
floating-point type.  They must have the same type, and they must have
the same number of components as Result Type.

 Results are computed per component.

<!-- End of AutoGen section -->

```
float-scalar-vector-type ::= float-type |
                             `vector<` integer-literal `x` float-type `>`
funordlt-op ::= ssa-id `=` `spv.FUnordLessThan` ssa-use, ssa-use
```

#### Example:

```mlir
%4 = spv.FUnordLessThan %0, %1 : f32
%5 = spv.FUnordLessThan %2, %3 : vector<4xf32>
```
"""
function FUnordLessThan(operand1::Value, operand2::Value; result::MLIRType, location=Location())
    results = MLIRType[result, ]
    operands = Value[operand1, operand2, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "spv.FUnordLessThan", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`FUnordNotEqual`

Result Type must be a scalar or vector of Boolean type.

 The type of Operand 1 and Operand 2  must be a scalar or vector of
floating-point type.  They must have the same type, and they must have
the same number of components as Result Type.

 Results are computed per component.

<!-- End of AutoGen section -->

```
float-scalar-vector-type ::= float-type |
                             `vector<` integer-literal `x` float-type `>`
funordneq-op ::= ssa-id `=` `spv.FUnordNotEqual` ssa-use, ssa-use
```

#### Example:

```mlir
%4 = spv.FUnordNotEqual %0, %1 : f32
%5 = spv.FUnordNotEqual %2, %3 : vector<4xf32>
```
"""
function FUnordNotEqual(operand1::Value, operand2::Value; result::MLIRType, location=Location())
    results = MLIRType[result, ]
    operands = Value[operand1, operand2, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "spv.FUnordNotEqual", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`func`

This op declares or defines a SPIR-V function using one region, which
contains one or more blocks.

Different from the SPIR-V binary format, this op is not allowed to
implicitly capture global values, and all external references must use
function arguments or symbol references. This op itself defines a symbol
that is unique in the enclosing module op.

This op itself takes no operands and generates no results. Its region
can take zero or more arguments and return zero or one values.

<!-- End of AutoGen section -->

```
spv-function-control ::= \"None\" | \"Inline\" | \"DontInline\" | ...
spv-function-op ::= `spv.func` function-signature
                     spv-function-control region
```

#### Example:

```mlir
spv.func @foo() -> () \"None\" { ... }
spv.func @bar() -> () \"Inline|Pure\" { ... }
```
"""
function func(; function_type, sym_name, function_control, body::Region, location=Location())
    results = MLIRType[]
    operands = Value[]
    owned_regions = Region[body, ]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("function_type", function_type), namedattribute("sym_name", sym_name), namedattribute("function_control", function_control), ]
    
    create_operation(
        "spv.func", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`FunctionCall`

Result Type is the type of the return value of the function. It must be
the same as the Return Type operand of the Function Type operand of the
Function operand.

Function is an OpFunction instruction.  This could be a forward
reference.

Argument N is the object to copy to parameter N of Function.

Note: A forward call is possible because there is no missing type
information: Result Type must match the Return Type of the function, and
the calling argument types must match the formal parameter types.

<!-- End of AutoGen section -->

```
function-call-op ::= `spv.FunctionCall` function-id `(` ssa-use-list `)`
                 `:` function-type
```

#### Example:

```mlir
spv.FunctionCall @f_void(%arg0) : (i32) ->  ()
%0 = spv.FunctionCall @f_iadd(%arg0, %arg1) : (i32, i32) -> i32
```
"""
function FunctionCall(arguments::Vector{Value}; result=nothing::Union{Nothing, MLIRType}, callee, location=Location())
    results = MLIRType[]
    operands = Value[arguments..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("callee", callee), ]
    (result != nothing) && push!(results, result)
    
    create_operation(
        "spv.FunctionCall", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`GL_Acos`

The standard trigonometric arc cosine of x radians.

Result is an angle, in radians, whose cosine is x. The range of result
values is [0, π]. Result is undefined if abs x > 1.

The operand x must be a scalar or vector whose component type is 16-bit or
32-bit floating-point.

Result Type and the type of x must be the same type. Results are computed
per component.
<!-- End of AutoGen section -->
```
restricted-float-scalar-type ::=  `f16` | `f32`
restricted-float-scalar-vector-type ::=
  restricted-float-scalar-type |
  `vector<` integer-literal `x` restricted-float-scalar-type `>`
acos-op ::= ssa-id `=` `spv.GL.Acos` ssa-use `:`
            restricted-float-scalar-vector-type
```
#### Example:

```mlir
%2 = spv.GL.Acos %0 : f32
%3 = spv.GL.Acos %1 : vector<3xf16>
```
"""
function GL_Acos(operand::Value; result=nothing::Union{Nothing, MLIRType}, location=Location())
    results = MLIRType[]
    operands = Value[operand, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (result != nothing) && push!(results, result)
    
    create_operation(
        "spv.GL.Acos", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`GL_Asin`

The standard trigonometric arc sine of x radians.

Result is an angle, in radians, whose sine is x. The range of result values
is [-π / 2, π / 2]. Result is undefined if abs x > 1.

The operand x must be a scalar or vector whose component type is 16-bit or
32-bit floating-point.

Result Type and the type of x must be the same type. Results are computed
per component.
<!-- End of AutoGen section -->
```
restricted-float-scalar-type ::=  `f16` | `f32`
restricted-float-scalar-vector-type ::=
  restricted-float-scalar-type |
  `vector<` integer-literal `x` restricted-float-scalar-type `>`
asin-op ::= ssa-id `=` `spv.GL.Asin` ssa-use `:`
            restricted-float-scalar-vector-type
```
#### Example:

```mlir
%2 = spv.GL.Asin %0 : f32
%3 = spv.GL.Asin %1 : vector<3xf16>
```
"""
function GL_Asin(operand::Value; result=nothing::Union{Nothing, MLIRType}, location=Location())
    results = MLIRType[]
    operands = Value[operand, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (result != nothing) && push!(results, result)
    
    create_operation(
        "spv.GL.Asin", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`GL_Atan`

The standard trigonometric arc tangent of x radians.

Result is an angle, in radians, whose tangent is y_over_x. The range of
result values is [-π / 2, π / 2].

The operand x must be a scalar or vector whose component type is 16-bit or
32-bit floating-point.

Result Type and the type of x must be the same type. Results are computed
per component.
<!-- End of AutoGen section -->
```
restricted-float-scalar-type ::=  `f16` | `f32`
restricted-float-scalar-vector-type ::=
  restricted-float-scalar-type |
  `vector<` integer-literal `x` restricted-float-scalar-type `>`
atan-op ::= ssa-id `=` `spv.GL.Atan` ssa-use `:`
            restricted-float-scalar-vector-type
```
#### Example:

```mlir
%2 = spv.GL.Atan %0 : f32
%3 = spv.GL.Atan %1 : vector<3xf16>
```
"""
function GL_Atan(operand::Value; result=nothing::Union{Nothing, MLIRType}, location=Location())
    results = MLIRType[]
    operands = Value[operand, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (result != nothing) && push!(results, result)
    
    create_operation(
        "spv.GL.Atan", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`GL_Ceil`

Result is the value equal to the nearest whole number that is greater than
or equal to x.

The operand x must be a scalar or vector whose component type is
floating-point.

Result Type and the type of x must be the same type. Results are computed
per component.

<!-- End of AutoGen section -->
```
float-scalar-vector-type ::= float-type |
                             `vector<` integer-literal `x` float-type `>`
ceil-op ::= ssa-id `=` `spv.GL.Ceil` ssa-use `:`
            float-scalar-vector-type
```
#### Example:

```mlir
%2 = spv.GL.Ceil %0 : f32
%3 = spv.GL.Ceil %1 : vector<3xf16>
```
"""
function GL_Ceil(operand::Value; result=nothing::Union{Nothing, MLIRType}, location=Location())
    results = MLIRType[]
    operands = Value[operand, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (result != nothing) && push!(results, result)
    
    create_operation(
        "spv.GL.Ceil", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`GL_Cos`

The standard trigonometric cosine of x radians.

The operand x must be a scalar or vector whose component type is 16-bit or
32-bit floating-point.

Result Type and the type of x must be the same type. Results are computed
per component.

<!-- End of AutoGen section -->
```
restricted-float-scalar-type ::=  `f16` | `f32`
restricted-float-scalar-vector-type ::=
  restricted-float-scalar-type |
  `vector<` integer-literal `x` restricted-float-scalar-type `>`
cos-op ::= ssa-id `=` `spv.GL.Cos` ssa-use `:`
           restricted-float-scalar-vector-type
```
#### Example:

```mlir
%2 = spv.GL.Cos %0 : f32
%3 = spv.GL.Cos %1 : vector<3xf16>
```
"""
function GL_Cos(operand::Value; result=nothing::Union{Nothing, MLIRType}, location=Location())
    results = MLIRType[]
    operands = Value[operand, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (result != nothing) && push!(results, result)
    
    create_operation(
        "spv.GL.Cos", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`GL_Cosh`

Hyperbolic cosine of x radians.

The operand x must be a scalar or vector whose component type is 16-bit or
32-bit floating-point.

Result Type and the type of x must be the same type. Results are computed
per component.

<!-- End of AutoGen section -->
```
restricted-float-scalar-type ::=  `f16` | `f32`
restricted-float-scalar-vector-type ::=
  restricted-float-scalar-type |
  `vector<` integer-literal `x` restricted-float-scalar-type `>`
cosh-op ::= ssa-id `=` `spv.GL.Cosh` ssa-use `:`
            restricted-float-scalar-vector-type
```
#### Example:

```mlir
%2 = spv.GL.Cosh %0 : f32
%3 = spv.GL.Cosh %1 : vector<3xf16>
```
"""
function GL_Cosh(operand::Value; result=nothing::Union{Nothing, MLIRType}, location=Location())
    results = MLIRType[]
    operands = Value[operand, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (result != nothing) && push!(results, result)
    
    create_operation(
        "spv.GL.Cosh", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`GL_Exp`

Result is the natural exponentiation of x; e^x.

The operand x must be a scalar or vector whose component type is
16-bit or 32-bit floating-point.

Result Type and the type of x must be the same type. Results are
computed per component.\";

<!-- End of AutoGen section -->
```
restricted-float-scalar-type ::=  `f16` | `f32`
restricted-float-scalar-vector-type ::=
  restricted-float-scalar-type |
  `vector<` integer-literal `x` restricted-float-scalar-type `>`
exp-op ::= ssa-id `=` `spv.GL.Exp` ssa-use `:`
           restricted-float-scalar-vector-type
```
#### Example:

```mlir
%2 = spv.GL.Exp %0 : f32
%3 = spv.GL.Exp %1 : vector<3xf16>
```
"""
function GL_Exp(operand::Value; result=nothing::Union{Nothing, MLIRType}, location=Location())
    results = MLIRType[]
    operands = Value[operand, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (result != nothing) && push!(results, result)
    
    create_operation(
        "spv.GL.Exp", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`GL_FAbs`

Result is x if x >= 0; otherwise result is -x.

The operand x must be a scalar or vector whose component type is
floating-point.

Result Type and the type of x must be the same type. Results are computed
per component.

<!-- End of AutoGen section -->
```
float-scalar-vector-type ::= float-type |
                             `vector<` integer-literal `x` float-type `>`
abs-op ::= ssa-id `=` `spv.GL.FAbs` ssa-use `:`
           float-scalar-vector-type
```
#### Example:

```mlir
%2 = spv.GL.FAbs %0 : f32
%3 = spv.GL.FAbs %1 : vector<3xf16>
```
"""
function GL_FAbs(operand::Value; result=nothing::Union{Nothing, MLIRType}, location=Location())
    results = MLIRType[]
    operands = Value[operand, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (result != nothing) && push!(results, result)
    
    create_operation(
        "spv.GL.FAbs", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`GL_FClamp`

Result is min(max(x, minVal), maxVal). The resulting value is undefined if
minVal > maxVal. The semantics used by min() and max() are those of FMin and
FMax.

The operands must all be a scalar or vector whose component type is
floating-point.

Result Type and the type of all operands must be the same type. Results are
computed per component.

<!-- End of AutoGen section -->
```
fclamp-op ::= ssa-id `=` `spv.GL.FClamp` ssa-use, ssa-use, ssa-use `:`
           float-scalar-vector-type
```
#### Example:

```mlir
%2 = spv.GL.FClamp %x, %min, %max : f32
%3 = spv.GL.FClamp %x, %min, %max : vector<3xf16>
```
"""
function GL_FClamp(x::Value, y::Value, z::Value; result::MLIRType, location=Location())
    results = MLIRType[result, ]
    operands = Value[x, y, z, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "spv.GL.FClamp", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`GL_FMax`

Result is y if x < y; otherwise result is x. Which operand is the
result is undefined if one of the operands is a NaN.

The operands must all be a scalar or vector whose component type
is floating-point.

Result Type and the type of all operands must be the same
type. Results are computed per component.

<!-- End of AutoGen section -->
```
float-scalar-vector-type ::= float-type |
                             `vector<` integer-literal `x` float-type `>`
fmax-op ::= ssa-id `=` `spv.GL.FMax` ssa-use `:`
            float-scalar-vector-type
```
#### Example:

```mlir
%2 = spv.GL.FMax %0, %1 : f32
%3 = spv.GL.FMax %0, %1 : vector<3xf16>
```
"""
function GL_FMax(lhs::Value, rhs::Value; result=nothing::Union{Nothing, MLIRType}, location=Location())
    results = MLIRType[]
    operands = Value[lhs, rhs, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (result != nothing) && push!(results, result)
    
    create_operation(
        "spv.GL.FMax", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`GL_FMin`

Result is y if y < x; otherwise result is x. Which operand is the result is
undefined if one of the operands is a NaN.

The operands must all be a scalar or vector whose component type is
floating-point.

Result Type and the type of all operands must be the same type. Results are
computed per component.

<!-- End of AutoGen section -->
```
float-scalar-vector-type ::= float-type |
                             `vector<` integer-literal `x` float-type `>`
fmin-op ::= ssa-id `=` `spv.GL.FMin` ssa-use `:`
            float-scalar-vector-type
```
#### Example:

```mlir
%2 = spv.GL.FMin %0, %1 : f32
%3 = spv.GL.FMin %0, %1 : vector<3xf16>
```
"""
function GL_FMin(lhs::Value, rhs::Value; result=nothing::Union{Nothing, MLIRType}, location=Location())
    results = MLIRType[]
    operands = Value[lhs, rhs, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (result != nothing) && push!(results, result)
    
    create_operation(
        "spv.GL.FMin", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`GL_FMix`

Result is the linear blend of x and y, i.e., x * (1 - a) + y * a.

The operands must all be a scalar or vector whose component type is floating-point.

Result Type and the type of all operands must be the same type. Results are computed per component.

<!-- End of AutoGen section -->

#### Example:

```mlir
%0 = spv.GL.FMix %x : f32, %y : f32, %a : f32 -> f32
%0 = spv.GL.FMix %x : vector<4xf32>, %y : vector<4xf32>, %a : vector<4xf32> -> vector<4xf32>
```
"""
function GL_FMix(x::Value, y::Value, a::Value; result=nothing::Union{Nothing, MLIRType}, location=Location())
    results = MLIRType[]
    operands = Value[x, y, a, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (result != nothing) && push!(results, result)
    
    create_operation(
        "spv.GL.FMix", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`GL_FSign`

Result is 1.0 if x > 0, 0.0 if x = 0, or -1.0 if x < 0.

The operand x must be a scalar or vector whose component type is
floating-point.

Result Type and the type of x must be the same type. Results are computed
per component.

<!-- End of AutoGen section -->
```
float-scalar-vector-type ::= float-type |
                             `vector<` integer-literal `x` float-type `>`
sign-op ::= ssa-id `=` `spv.GL.FSign` ssa-use `:`
            float-scalar-vector-type
```
#### Example:

```mlir
%2 = spv.GL.FSign %0 : f32
%3 = spv.GL.FSign %1 : vector<3xf16>
```
"""
function GL_FSign(operand::Value; result=nothing::Union{Nothing, MLIRType}, location=Location())
    results = MLIRType[]
    operands = Value[operand, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (result != nothing) && push!(results, result)
    
    create_operation(
        "spv.GL.FSign", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`GL_FindUMsb`

Results in the bit number of the most-significant 1-bit in the binary
representation of Value. If Value is 0, the result is -1.

Result Type and the type of Value must both be integer scalar or
integer vector types. Result Type and operand types must have the
same number of components with the same component width. Results are
computed per component.

This instruction is currently limited to 32-bit width components.
"""
function GL_FindUMsb(operand::Value; result=nothing::Union{Nothing, MLIRType}, location=Location())
    results = MLIRType[]
    operands = Value[operand, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (result != nothing) && push!(results, result)
    
    create_operation(
        "spv.GL.FindUMsb", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`GL_Floor`

Result is the value equal to the nearest whole number that is less than or
equal to x.

The operand x must be a scalar or vector whose component type is
floating-point.

Result Type and the type of x must be the same type. Results are computed
per component.

<!-- End of AutoGen section -->
```
float-scalar-vector-type ::= float-type |
                             `vector<` integer-literal `x` float-type `>`
floor-op ::= ssa-id `=` `spv.GL.Floor` ssa-use `:`
            float-scalar-vector-type
```
#### Example:

```mlir
%2 = spv.GL.Floor %0 : f32
%3 = spv.GL.Floor %1 : vector<3xf16>
```
"""
function GL_Floor(operand::Value; result=nothing::Union{Nothing, MLIRType}, location=Location())
    results = MLIRType[]
    operands = Value[operand, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (result != nothing) && push!(results, result)
    
    create_operation(
        "spv.GL.Floor", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`GL_Fma`

In uses where this operation is decorated with NoContraction:

- fma is considered a single operation, whereas the expression a * b + c
  is considered two operations.
- The precision of fma can differ from the precision of the expression
  a * b + c.
- fma will be computed with the same precision as any other fma decorated
  with NoContraction, giving invariant results for the same input values
  of a, b, and c.

Otherwise, in the absence of a NoContraction decoration, there are no
special constraints on the number of operations or difference in precision
between fma and the expression a * b +c.

The operands must all be a scalar or vector whose component type is
floating-point.

Result Type and the type of all operands must be the same type. Results
are computed per component.

<!-- End of AutoGen section -->
```
fma-op ::= ssa-id `=` `spv.GL.Fma` ssa-use, ssa-use, ssa-use `:`
           float-scalar-vector-type
```
#### Example:

```mlir
%0 = spv.GL.Fma %a, %b, %c : f32
%1 = spv.GL.Fma %a, %b, %c : vector<3xf16>
```
"""
function GL_Fma(x::Value, y::Value, z::Value; result::MLIRType, location=Location())
    results = MLIRType[result, ]
    operands = Value[x, y, z, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "spv.GL.Fma", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`GL_FrexpStruct`

Result is a structure containing x split into a floating-point significand
in the range (-1.0, 0.5] or [0.5, 1.0) and an integral exponent of 2, such that:

x = significand * 2^exponent

If x is a zero, the exponent is 0.0. If x is an infinity or a NaN, the
exponent is undefined. If x is 0.0, the significand is 0.0. If x is -0.0,
the significand is -0.0

Result Type must be an OpTypeStruct with two members. Member 0 must have
the same type as the type of x. Member 0 holds the significand. Member 1
must be a scalar or vector with integer component type, with 32-bit
component width. Member 1 holds the exponent. These two members and x must
have the same number of components.

The operand x must be a scalar or vector whose component type is
floating-point.

<!-- End of AutoGen section -->
```
float-scalar-vector-type ::= float-type |
                             `vector<` integer-literal `x` float-type `>`
integer-scalar-vector-type ::= integer-type |
                             `vector<` integer-literal `x` integer-type `>`
frexpstruct-op ::= ssa-id `=` `spv.GL.FrexpStruct` ssa-use `:`
                              `!spv.struct<` float-scalar-vector-type `,`
                                              integer-scalar-vector-type `>`
```
#### Example:

```mlir
%2 = spv.GL.FrexpStruct %0 : f32 -> !spv.struct<f32, i32>
%3 = spv.GL.FrexpStruct %0 : vector<3xf32> -> !spv.struct<vector<3xf32>, vector<3xi32>>
```
"""
function GL_FrexpStruct(operand::Value; result::MLIRType, location=Location())
    results = MLIRType[result, ]
    operands = Value[operand, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "spv.GL.FrexpStruct", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`GL_InverseSqrt`

Result is the reciprocal of sqrt x. Result is undefined if x <= 0.

The operand x must be a scalar or vector whose component type is
floating-point.

Result Type and the type of x must be the same type. Results are computed
per component.

<!-- End of AutoGen section -->
```
float-scalar-vector-type ::= float-type |
                             `vector<` integer-literal `x` float-type `>`
rsqrt-op ::= ssa-id `=` `spv.GL.InverseSqrt` ssa-use `:`
             float-scalar-vector-type
```
#### Example:

```mlir
%2 = spv.GL.InverseSqrt %0 : f32
%3 = spv.GL.InverseSqrt %1 : vector<3xf16>
```
"""
function GL_InverseSqrt(operand::Value; result=nothing::Union{Nothing, MLIRType}, location=Location())
    results = MLIRType[]
    operands = Value[operand, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (result != nothing) && push!(results, result)
    
    create_operation(
        "spv.GL.InverseSqrt", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`GL_Ldexp`

Builds a floating-point number from x and the corresponding
integral exponent of two in exp:

significand * 2^exponent

If this product is too large to be represented in the floating-point
type, the resulting value is undefined. If exp is greater than +128
(single precision) or +1024 (double precision), the resulting value is
undefined. If exp is less than -126 (single precision) or -1022 (double precision),
the result may be flushed to zero. Additionally, splitting the value
into a significand and exponent using frexp and then reconstructing a
floating-point value using ldexp should yield the original input for
zero and all finite non-denormalized values.

The operand x must be a scalar or vector whose component type is floating-point.

The exp operand must be a scalar or vector with integer component type.
The number of components in x and exp must be the same.

Result Type must be the same type as the type of x. Results are computed per
component.

<!-- End of AutoGen section -->

#### Example:

```mlir
%y = spv.GL.Ldexp %x : f32, %exp : i32 -> f32
%y = spv.GL.Ldexp %x : vector<3xf32>, %exp : vector<3xi32> -> vector<3xf32>
```
"""
function GL_Ldexp(x::Value, exp::Value; y=nothing::Union{Nothing, MLIRType}, location=Location())
    results = MLIRType[]
    operands = Value[x, exp, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (y != nothing) && push!(results, y)
    
    create_operation(
        "spv.GL.Ldexp", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`GL_Log`

Result is the natural logarithm of x, i.e., the value y which satisfies the
equation x = ey. Result is undefined if x <= 0.

The operand x must be a scalar or vector whose component type is 16-bit or
32-bit floating-point.

Result Type and the type of x must be the same type. Results are computed
per component.

<!-- End of AutoGen section -->
```
restricted-float-scalar-type ::=  `f16` | `f32`
restricted-float-scalar-vector-type ::=
  restricted-float-scalar-type |
  `vector<` integer-literal `x` restricted-float-scalar-type `>`
log-op ::= ssa-id `=` `spv.GL.Log` ssa-use `:`
           restricted-float-scalar-vector-type
```
#### Example:

```mlir
%2 = spv.GL.Log %0 : f32
%3 = spv.GL.Log %1 : vector<3xf16>
```
"""
function GL_Log(operand::Value; result=nothing::Union{Nothing, MLIRType}, location=Location())
    results = MLIRType[]
    operands = Value[operand, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (result != nothing) && push!(results, result)
    
    create_operation(
        "spv.GL.Log", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`GL_Pow`

Result is x raised to the y power; x^y.

Result is undefined if x = 0 and y ≤ 0.

The operand x and y must be a scalar or vector whose component type is
16-bit or 32-bit floating-point.

Result Type and the type of all operands must be the same type. Results are
computed per component.

<!-- End of AutoGen section -->
```
restricted-float-scalar-type ::=  `f16` | `f32`
restricted-float-scalar-vector-type ::=
  restricted-float-scalar-type |
  `vector<` integer-literal `x` restricted-float-scalar-type `>`
pow-op ::= ssa-id `=` `spv.GL.Pow` ssa-use `:`
           restricted-float-scalar-vector-type
```
#### Example:

```mlir
%2 = spv.GL.Pow %0, %1 : f32
%3 = spv.GL.Pow %0, %1 : vector<3xf16>
```
"""
function GL_Pow(lhs::Value, rhs::Value; result=nothing::Union{Nothing, MLIRType}, location=Location())
    results = MLIRType[]
    operands = Value[lhs, rhs, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (result != nothing) && push!(results, result)
    
    create_operation(
        "spv.GL.Pow", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`GL_Round`

Result is the value equal to the nearest whole number.

The operand x must be a scalar or vector whose component type is
floating-point.

Result Type and the type of x must be the same type. Results are computed
per component.

<!-- End of AutoGen section -->
```
float-scalar-vector-type ::= float-type |
                             `vector<` integer-literal `x` float-type `>`
floor-op ::= ssa-id `=` `spv.GL.Round` ssa-use `:`
            float-scalar-vector-type
```
#### Example:

```mlir
%2 = spv.GL.Round %0 : f32
%3 = spv.GL.Round %1 : vector<3xf16>
```
"""
function GL_Round(operand::Value; result=nothing::Union{Nothing, MLIRType}, location=Location())
    results = MLIRType[]
    operands = Value[operand, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (result != nothing) && push!(results, result)
    
    create_operation(
        "spv.GL.Round", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`GL_SAbs`

Result is x if x ≥ 0; otherwise result is -x, where x is interpreted as a
signed integer.

Result Type and the type of x must both be integer scalar or integer vector
types. Result Type and operand types must have the same number of components
with the same component width. Results are computed per component.

<!-- End of AutoGen section -->
```
integer-scalar-vector-type ::= integer-type |
                               `vector<` integer-literal `x` integer-type `>`
abs-op ::= ssa-id `=` `spv.GL.SAbs` ssa-use `:`
           integer-scalar-vector-type
```
#### Example:

```mlir
%2 = spv.GL.SAbs %0 : i32
%3 = spv.GL.SAbs %1 : vector<3xi16>
```
"""
function GL_SAbs(operand::Value; result=nothing::Union{Nothing, MLIRType}, location=Location())
    results = MLIRType[]
    operands = Value[operand, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (result != nothing) && push!(results, result)
    
    create_operation(
        "spv.GL.SAbs", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`GL_SClamp`

Result is min(max(x, minVal), maxVal), where x, minVal and maxVal are
interpreted as signed integers. The resulting value is undefined if
minVal > maxVal.

Result Type and the type of the operands must both be integer scalar or
integer vector types. Result Type and operand types must have the same number
of components with the same component width. Results are computed per
component.

<!-- End of AutoGen section -->
```
uclamp-op ::= ssa-id `=` `spv.GL.UClamp` ssa-use, ssa-use, ssa-use `:`
           sgined-scalar-vector-type
```
#### Example:

```mlir
%2 = spv.GL.SClamp %x, %min, %max : si32
%3 = spv.GL.SClamp %x, %min, %max : vector<3xsi16>
```
"""
function GL_SClamp(x::Value, y::Value, z::Value; result::MLIRType, location=Location())
    results = MLIRType[result, ]
    operands = Value[x, y, z, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "spv.GL.SClamp", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`GL_SMax`

Result is y if x < y; otherwise result is x, where x and y are interpreted
as signed integers.

Result Type and the type of x and y must both be integer scalar or integer
vector types. Result Type and operand types must have the same number of
components with the same component width. Results are computed per
component.

<!-- End of AutoGen section -->
```
integer-scalar-vector-type ::= integer-type |
                               `vector<` integer-literal `x` integer-type `>`
smax-op ::= ssa-id `=` `spv.GL.SMax` ssa-use `:`
            integer-scalar-vector-type
```
#### Example:

```mlir
%2 = spv.GL.SMax %0, %1 : i32
%3 = spv.GL.SMax %0, %1 : vector<3xi16>
```
"""
function GL_SMax(lhs::Value, rhs::Value; result=nothing::Union{Nothing, MLIRType}, location=Location())
    results = MLIRType[]
    operands = Value[lhs, rhs, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (result != nothing) && push!(results, result)
    
    create_operation(
        "spv.GL.SMax", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`GL_SMin`

Result is y if y < x; otherwise result is x, where x and y are interpreted
as signed integers.

Result Type and the type of x and y must both be integer scalar or integer
vector types. Result Type and operand types must have the same number of
components with the same component width. Results are computed per
component.

<!-- End of AutoGen section -->
```
integer-scalar-vector-type ::= integer-type |
                               `vector<` integer-literal `x` integer-type `>`
smin-op ::= ssa-id `=` `spv.GL.SMin` ssa-use `:`
            integer-scalar-vector-type
```
#### Example:

```mlir
%2 = spv.GL.SMin %0, %1 : i32
%3 = spv.GL.SMin %0, %1 : vector<3xi16>
```
"""
function GL_SMin(lhs::Value, rhs::Value; result=nothing::Union{Nothing, MLIRType}, location=Location())
    results = MLIRType[]
    operands = Value[lhs, rhs, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (result != nothing) && push!(results, result)
    
    create_operation(
        "spv.GL.SMin", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`GL_SSign`

Result is 1 if x > 0, 0 if x = 0, or -1 if x < 0, where x is interpreted as
a signed integer.

Result Type and the type of x must both be integer scalar or integer vector
types. Result Type and operand types must have the same number of components
with the same component width. Results are computed per component.

<!-- End of AutoGen section -->
```
integer-scalar-vector-type ::= integer-type |
                               `vector<` integer-literal `x` integer-type `>`
sign-op ::= ssa-id `=` `spv.GL.SSign` ssa-use `:`
            integer-scalar-vector-type
```
#### Example:

```mlir
%2 = spv.GL.SSign %0 : i32
%3 = spv.GL.SSign %1 : vector<3xi16>
```
"""
function GL_SSign(operand::Value; result=nothing::Union{Nothing, MLIRType}, location=Location())
    results = MLIRType[]
    operands = Value[operand, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (result != nothing) && push!(results, result)
    
    create_operation(
        "spv.GL.SSign", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`GL_Sin`

The standard trigonometric sine of x radians.

The operand x must be a scalar or vector whose component type is 16-bit or
32-bit floating-point.

Result Type and the type of x must be the same type. Results are computed
per component.

<!-- End of AutoGen section -->
```
restricted-float-scalar-type ::=  `f16` | `f32`
restricted-float-scalar-vector-type ::=
  restricted-float-scalar-type |
  `vector<` integer-literal `x` restricted-float-scalar-type `>`
sin-op ::= ssa-id `=` `spv.GL.Sin` ssa-use `:`
           restricted-float-scalar-vector-type
```
#### Example:

```mlir
%2 = spv.GL.Sin %0 : f32
%3 = spv.GL.Sin %1 : vector<3xf16>
```
"""
function GL_Sin(operand::Value; result=nothing::Union{Nothing, MLIRType}, location=Location())
    results = MLIRType[]
    operands = Value[operand, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (result != nothing) && push!(results, result)
    
    create_operation(
        "spv.GL.Sin", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`GL_Sinh`

Hyperbolic sine of x radians.

The operand x must be a scalar or vector whose component type is 16-bit or
32-bit floating-point.

Result Type and the type of x must be the same type. Results are computed
per component.

<!-- End of AutoGen section -->
```
restricted-float-scalar-type ::=  `f16` | `f32`
restricted-float-scalar-vector-type ::=
  restricted-float-scalar-type |
  `vector<` integer-literal `x` restricted-float-scalar-type `>`
sinh-op ::= ssa-id `=` `spv.GL.Sinh` ssa-use `:`
            restricted-float-scalar-vector-type
```
#### Example:

```mlir
%2 = spv.GL.Sinh %0 : f32
%3 = spv.GL.Sinh %1 : vector<3xf16>
```
"""
function GL_Sinh(operand::Value; result=nothing::Union{Nothing, MLIRType}, location=Location())
    results = MLIRType[]
    operands = Value[operand, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (result != nothing) && push!(results, result)
    
    create_operation(
        "spv.GL.Sinh", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`GL_Sqrt`

Result is the square root of x. Result is undefined if x < 0.

The operand x must be a scalar or vector whose component type is
floating-point.

Result Type and the type of x must be the same type. Results are computed
per component.

<!-- End of AutoGen section -->
```
float-scalar-vector-type ::= float-type |
                             `vector<` integer-literal `x` float-type `>`
sqrt-op ::= ssa-id `=` `spv.GL.Sqrt` ssa-use `:`
            float-scalar-vector-type
```
#### Example:

```mlir
%2 = spv.GL.Sqrt %0 : f32
%3 = spv.GL.Sqrt %1 : vector<3xf16>
```
"""
function GL_Sqrt(operand::Value; result=nothing::Union{Nothing, MLIRType}, location=Location())
    results = MLIRType[]
    operands = Value[operand, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (result != nothing) && push!(results, result)
    
    create_operation(
        "spv.GL.Sqrt", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`GL_Tan`

The standard trigonometric tangent of x radians.

The operand x must be a scalar or vector whose component type is 16-bit or
32-bit floating-point.

Result Type and the type of x must be the same type. Results are computed
per component.

<!-- End of AutoGen section -->
```
restricted-float-scalar-type ::=  `f16` | `f32`
restricted-float-scalar-vector-type ::=
  restricted-float-scalar-type |
  `vector<` integer-literal `x` restricted-float-scalar-type `>`
tan-op ::= ssa-id `=` `spv.GL.Tan` ssa-use `:`
           restricted-float-scalar-vector-type
```
#### Example:

```mlir
%2 = spv.GL.Tan %0 : f32
%3 = spv.GL.Tan %1 : vector<3xf16>
```
"""
function GL_Tan(operand::Value; result=nothing::Union{Nothing, MLIRType}, location=Location())
    results = MLIRType[]
    operands = Value[operand, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (result != nothing) && push!(results, result)
    
    create_operation(
        "spv.GL.Tan", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`GL_Tanh`

Hyperbolic tangent of x radians.

The operand x must be a scalar or vector whose component type is 16-bit or
32-bit floating-point.

Result Type and the type of x must be the same type. Results are computed
per component.

<!-- End of AutoGen section -->
```
restricted-float-scalar-type ::=  `f16` | `f32`
restricted-float-scalar-vector-type ::=
  restricted-float-scalar-type |
  `vector<` integer-literal `x` restricted-float-scalar-type `>`
tanh-op ::= ssa-id `=` `spv.GL.Tanh` ssa-use `:`
            restricted-float-scalar-vector-type
```
#### Example:

```mlir
%2 = spv.GL.Tanh %0 : f32
%3 = spv.GL.Tanh %1 : vector<3xf16>
```
"""
function GL_Tanh(operand::Value; result=nothing::Union{Nothing, MLIRType}, location=Location())
    results = MLIRType[]
    operands = Value[operand, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (result != nothing) && push!(results, result)
    
    create_operation(
        "spv.GL.Tanh", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`GL_UClamp`

Result is min(max(x, minVal), maxVal), where x, minVal and maxVal are
interpreted as unsigned integers. The resulting value is undefined if
minVal > maxVal.

Result Type and the type of the operands must both be integer scalar or
integer vector types. Result Type and operand types must have the same number
of components with the same component width. Results are computed per
component.

<!-- End of AutoGen section -->
```
uclamp-op ::= ssa-id `=` `spv.GL.UClamp` ssa-use, ssa-use, ssa-use `:`
           unsigned-signless-scalar-vector-type
```
#### Example:

```mlir
%2 = spv.GL.UClamp %x, %min, %max : i32
%3 = spv.GL.UClamp %x, %min, %max : vector<3xui16>
```
"""
function GL_UClamp(x::Value, y::Value, z::Value; result::MLIRType, location=Location())
    results = MLIRType[result, ]
    operands = Value[x, y, z, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "spv.GL.UClamp", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`GL_UMax`

Result is y if x < y; otherwise result is x, where x and y are interpreted
as unsigned integers.

Result Type and the type of x and y must both be integer scalar or integer
vector types. Result Type and operand types must have the same number of
components with the same component width. Results are computed per
component.

<!-- End of AutoGen section -->
```
integer-scalar-vector-type ::= integer-type |
                               `vector<` integer-literal `x` integer-type `>`
smax-op ::= ssa-id `=` `spv.GL.UMax` ssa-use `:`
            integer-scalar-vector-type
```
#### Example:

```mlir
%2 = spv.GL.UMax %0, %1 : i32
%3 = spv.GL.UMax %0, %1 : vector<3xi16>
```
"""
function GL_UMax(lhs::Value, rhs::Value; result=nothing::Union{Nothing, MLIRType}, location=Location())
    results = MLIRType[]
    operands = Value[lhs, rhs, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (result != nothing) && push!(results, result)
    
    create_operation(
        "spv.GL.UMax", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`GL_UMin`

Result is y if y < x; otherwise result is x, where x and y are interpreted
as unsigned integers.

Result Type and the type of x and y must both be integer scalar or integer
vector types. Result Type and operand types must have the same number of
components with the same component width. Results are computed per
component.

<!-- End of AutoGen section -->
```
integer-scalar-vector-type ::= integer-type |
                               `vector<` integer-literal `x` integer-type `>`
smin-op ::= ssa-id `=` `spv.GL.UMin` ssa-use `:`
            integer-scalar-vector-type
```
#### Example:

```mlir
%2 = spv.GL.UMin %0, %1 : i32
%3 = spv.GL.UMin %0, %1 : vector<3xi16>
```
"""
function GL_UMin(lhs::Value, rhs::Value; result=nothing::Union{Nothing, MLIRType}, location=Location())
    results = MLIRType[]
    operands = Value[lhs, rhs, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (result != nothing) && push!(results, result)
    
    create_operation(
        "spv.GL.UMin", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`GlobalVariable`

The variable type must be an OpTypePointer. Its type operand is the type of
object in memory.

Storage Class is the Storage Class of the memory holding the object. It
cannot be Generic. It must be the same as the Storage Class operand of
the variable types. Only those storage classes that are valid at module
scope (like Input, Output, StorageBuffer, etc.) are valid.

Initializer is optional.  If Initializer is present, it will be
the initial value of the variable’s memory content. Initializer
must be an symbol defined from a constant instruction or other
`spv.GlobalVariable` operation in module scope. Initializer must
have the same type as the type of the defined symbol.

<!-- End of AutoGen section -->

```
variable-op ::= `spv.GlobalVariable` spirv-type symbol-ref-id
                (`initializer(` symbol-ref-id `)`)?
                (`bind(` integer-literal, integer-literal `)`)?
                (`built_in(` string-literal `)`)?
                attribute-dict?
```

where `initializer` specifies initializer and `bind` specifies the
descriptor set and binding number. `built_in` specifies SPIR-V
BuiltIn decoration associated with the op.

#### Example:

```mlir
spv.GlobalVariable @var0 : !spv.ptr<f32, Input> @var0
spv.GlobalVariable @var1 initializer(@var0) : !spv.ptr<f32, Output>
spv.GlobalVariable @var2 bind(1, 2) : !spv.ptr<f32, Uniform>
spv.GlobalVariable @var3 built_in(\"GlobalInvocationId\") : !spv.ptr<vector<3xi32>, Input>
```
"""
function GlobalVariable(; type, sym_name, initializer=nothing, location_=nothing, binding=nothing, descriptor_set=nothing, builtin=nothing, location=Location())
    results = MLIRType[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("type", type), namedattribute("sym_name", sym_name), ]
    (initializer != nothing) && push!(attributes, namedattribute("initializer", initializer))
    (location != nothing) && push!(attributes, namedattribute("location", location_))
    (binding != nothing) && push!(attributes, namedattribute("binding", binding))
    (descriptor_set != nothing) && push!(attributes, namedattribute("descriptor_set", descriptor_set))
    (builtin != nothing) && push!(attributes, namedattribute("builtin", builtin))
    
    create_operation(
        "spv.GlobalVariable", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`GroupBroadcast`

All invocations of this module within Execution must reach this point of
execution.

Behavior is undefined if this instruction is used in control flow that
is non-uniform within Execution.

Result Type  must be a scalar or vector of floating-point type, integer
type, or Boolean type.

Execution must be Workgroup or Subgroup Scope.

 The type of Value must be the same as Result Type.

LocalId must be an integer datatype. It can be a scalar, or a vector
with 2 components or a vector with 3 components. LocalId must be the
same for all invocations in the group.

<!-- End of AutoGen section -->

```
scope ::= `\"Workgroup\"` | `\"Subgroup\"`
integer-float-scalar-vector-type ::= integer-type | float-type |
                           `vector<` integer-literal `x` integer-type `>` |
                           `vector<` integer-literal `x` float-type `>`
localid-type ::= integer-type |
               `vector<` integer-literal `x` integer-type `>`
group-broadcast-op ::= ssa-id `=` `spv.GroupBroadcast` scope ssa_use,
               ssa_use `:` integer-float-scalar-vector-type `,` localid-type
```mlir

#### Example:

```
%scalar_value = ... : f32
%vector_value = ... : vector<4xf32>
%scalar_localid = ... : i32
%vector_localid = ... : vector<3xi32>
%0 = spv.GroupBroadcast \"Subgroup\" %scalar_value, %scalar_localid : f32, i32
%1 = spv.GroupBroadcast \"Workgroup\" %vector_value, %vector_localid :
  vector<4xf32>, vector<3xi32>
```
"""
function GroupBroadcast(value::Value, localid::Value; result=nothing::Union{Nothing, MLIRType}, execution_scope, location=Location())
    results = MLIRType[]
    operands = Value[value, localid, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("execution_scope", execution_scope), ]
    (result != nothing) && push!(results, result)
    
    create_operation(
        "spv.GroupBroadcast", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`GroupNonUniformBallot`

Result Type  must be a vector of four components of integer type scalar,
whose Signedness operand is 0.

Result is a set of bitfields where the first invocation is represented
in the lowest bit of the first vector component and the last (up to the
size of the group) is the higher bit number of the last bitmask needed
to represent all bits of the group invocations.

Execution must be Workgroup or Subgroup Scope.

Predicate must be a Boolean type.

<!-- End of AutoGen section -->

```
scope ::= `\"Workgroup\"` | `\"Subgroup\"`
non-uniform-ballot-op ::= ssa-id `=` `spv.GroupNonUniformBallot` scope
                          ssa-use `:` `vector` `<` 4 `x` `integer-type` `>`
```

#### Example:

```mlir
%0 = spv.GroupNonUniformBallot \"SubGroup\" %predicate : vector<4xi32>
```
"""
function GroupNonUniformBallot(predicate::Value; result::MLIRType, execution_scope, location=Location())
    results = MLIRType[result, ]
    operands = Value[predicate, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("execution_scope", execution_scope), ]
    
    create_operation(
        "spv.GroupNonUniformBallot", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`GroupNonUniformBroadcast`

Result Type  must be a scalar or vector of floating-point type, integer
type, or Boolean type.

Execution must be Workgroup or Subgroup Scope.

 The type of Value must be the same as Result Type.

Id  must be a scalar of integer type, whose Signedness operand is 0.

Before version 1.5, Id must come from a constant instruction. Starting
with version 1.5, Id must be dynamically uniform.

The resulting value is undefined if Id is an inactive invocation, or is
greater than or equal to the size of the group.

<!-- End of AutoGen section -->

```
scope ::= `\"Workgroup\"` | `\"Subgroup\"`
integer-float-scalar-vector-type ::= integer-type | float-type |
                           `vector<` integer-literal `x` integer-type `>` |
                           `vector<` integer-literal `x` float-type `>`
group-non-uniform-broadcast-op ::= ssa-id `=` 
	            `spv.GroupNonUniformBroadcast` scope ssa_use,
            ssa_use `:` integer-float-scalar-vector-type `,` integer-type
```mlir

#### Example:

```
%scalar_value = ... : f32
%vector_value = ... : vector<4xf32>
%id = ... : i32
%0 = spv.GroupNonUniformBroadcast \"Subgroup\" %scalar_value, %id : f32, i32
%1 = spv.GroupNonUniformBroadcast \"Workgroup\" %vector_value, %id :
  vector<4xf32>, i32
```
"""
function GroupNonUniformBroadcast(value::Value, id::Value; result=nothing::Union{Nothing, MLIRType}, execution_scope, location=Location())
    results = MLIRType[]
    operands = Value[value, id, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("execution_scope", execution_scope), ]
    (result != nothing) && push!(results, result)
    
    create_operation(
        "spv.GroupNonUniformBroadcast", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`GroupNonUniformElect`

Result Type must be a Boolean type.

Execution must be Workgroup or Subgroup Scope.

<!-- End of AutoGen section -->

```
scope ::= `\"Workgroup\"` | `\"Subgroup\"`
non-uniform-elect-op ::= ssa-id `=` `spv.GroupNonUniformElect` scope
                         `:` `i1`
```

#### Example:

```mlir
%0 = spv.GroupNonUniformElect : i1
```
"""
function GroupNonUniformElect(; result=nothing::Union{Nothing, MLIRType}, execution_scope, location=Location())
    results = MLIRType[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("execution_scope", execution_scope), ]
    (result != nothing) && push!(results, result)
    
    create_operation(
        "spv.GroupNonUniformElect", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`GroupNonUniformFAdd`

Result Type  must be a scalar or vector of floating-point type.

Execution must be Workgroup or Subgroup Scope.

The identity I for Operation is 0. If Operation is ClusteredReduce,
ClusterSize must be specified.

 The type of Value must be the same as Result Type.  The method used to
perform the group operation on the contributed Value(s) from active
invocations is implementation defined.

ClusterSize is the size of cluster to use. ClusterSize must be a scalar
of integer type, whose Signedness operand is 0. ClusterSize must come
from a constant instruction. ClusterSize must be at least 1, and must be
a power of 2. If ClusterSize is greater than the declared SubGroupSize,
executing this instruction results in undefined behavior.

<!-- End of AutoGen section -->

```
scope ::= `\"Workgroup\"` | `\"Subgroup\"`
operation ::= `\"Reduce\"` | `\"InclusiveScan\"` | `\"ExclusiveScan\"` | ...
float-scalar-vector-type ::= float-type |
                             `vector<` integer-literal `x` float-type `>`
non-uniform-fadd-op ::= ssa-id `=` `spv.GroupNonUniformFAdd` scope operation
                        ssa-use ( `cluster_size` `(` ssa_use `)` )?
                        `:` float-scalar-vector-type
```

#### Example:

```mlir
%four = spv.Constant 4 : i32
%scalar = ... : f32
%vector = ... : vector<4xf32>
%0 = spv.GroupNonUniformFAdd \"Workgroup\" \"Reduce\" %scalar : f32
%1 = spv.GroupNonUniformFAdd \"Subgroup\" \"ClusteredReduce\" %vector cluster_size(%four) : vector<4xf32>
```
"""
function GroupNonUniformFAdd(value::Value, cluster_size=nothing::Union{Nothing, Value}; result::MLIRType, execution_scope, group_operation, location=Location())
    results = MLIRType[result, ]
    operands = Value[value, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("execution_scope", execution_scope), namedattribute("group_operation", group_operation), ]
    (cluster_size != nothing) && push!(operands, cluster_size)
    
    create_operation(
        "spv.GroupNonUniformFAdd", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`GroupNonUniformFMax`

Result Type  must be a scalar or vector of floating-point type.

Execution must be Workgroup or Subgroup Scope.

The identity I for Operation is -INF. If Operation is ClusteredReduce,
ClusterSize must be specified.

 The type of Value must be the same as Result Type.  The method used to
perform the group operation on the contributed Value(s) from active
invocations is implementation defined. From the set of Value(s) provided
by active invocations within a subgroup, if for any two Values one of
them is a NaN, the other is chosen. If all Value(s) that are used by the
current invocation are NaN, then the result is an undefined value.

ClusterSize is the size of cluster to use. ClusterSize must be a scalar
of integer type, whose Signedness operand is 0. ClusterSize must come
from a constant instruction. ClusterSize must be at least 1, and must be
a power of 2. If ClusterSize is greater than the declared SubGroupSize,
executing this instruction results in undefined behavior.

<!-- End of AutoGen section -->

```
scope ::= `\"Workgroup\"` | `\"Subgroup\"`
operation ::= `\"Reduce\"` | `\"InclusiveScan\"` | `\"ExclusiveScan\"` | ...
float-scalar-vector-type ::= float-type |
                             `vector<` integer-literal `x` float-type `>`
non-uniform-fmax-op ::= ssa-id `=` `spv.GroupNonUniformFMax` scope operation
                        ssa-use ( `cluster_size` `(` ssa_use `)` )?
                        `:` float-scalar-vector-type
```

#### Example:

```mlir
%four = spv.Constant 4 : i32
%scalar = ... : f32
%vector = ... : vector<4xf32>
%0 = spv.GroupNonUniformFMax \"Workgroup\" \"Reduce\" %scalar : f32
%1 = spv.GroupNonUniformFMax \"Subgroup\" \"ClusteredReduce\" %vector cluster_size(%four) : vector<4xf32>
```
"""
function GroupNonUniformFMax(value::Value, cluster_size=nothing::Union{Nothing, Value}; result::MLIRType, execution_scope, group_operation, location=Location())
    results = MLIRType[result, ]
    operands = Value[value, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("execution_scope", execution_scope), namedattribute("group_operation", group_operation), ]
    (cluster_size != nothing) && push!(operands, cluster_size)
    
    create_operation(
        "spv.GroupNonUniformFMax", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`GroupNonUniformFMin`

Result Type  must be a scalar or vector of floating-point type.

Execution must be Workgroup or Subgroup Scope.

The identity I for Operation is +INF. If Operation is ClusteredReduce,
ClusterSize must be specified.

 The type of Value must be the same as Result Type.  The method used to
perform the group operation on the contributed Value(s) from active
invocations is implementation defined. From the set of Value(s) provided
by active invocations within a subgroup, if for any two Values one of
them is a NaN, the other is chosen. If all Value(s) that are used by the
current invocation are NaN, then the result is an undefined value.

ClusterSize is the size of cluster to use. ClusterSize must be a scalar
of integer type, whose Signedness operand is 0. ClusterSize must come
from a constant instruction. ClusterSize must be at least 1, and must be
a power of 2. If ClusterSize is greater than the declared SubGroupSize,
executing this instruction results in undefined behavior.

<!-- End of AutoGen section -->

```
scope ::= `\"Workgroup\"` | `\"Subgroup\"`
operation ::= `\"Reduce\"` | `\"InclusiveScan\"` | `\"ExclusiveScan\"` | ...
float-scalar-vector-type ::= float-type |
                             `vector<` integer-literal `x` float-type `>`
non-uniform-fmin-op ::= ssa-id `=` `spv.GroupNonUniformFMin` scope operation
                        ssa-use ( `cluster_size` `(` ssa_use `)` )?
                        `:` float-scalar-vector-type
```

#### Example:

```mlir
%four = spv.Constant 4 : i32
%scalar = ... : f32
%vector = ... : vector<4xf32>
%0 = spv.GroupNonUniformFMin \"Workgroup\" \"Reduce\" %scalar : f32
%1 = spv.GroupNonUniformFMin \"Subgroup\" \"ClusteredReduce\" %vector cluster_size(%four) : vector<4xf32>
```
"""
function GroupNonUniformFMin(value::Value, cluster_size=nothing::Union{Nothing, Value}; result::MLIRType, execution_scope, group_operation, location=Location())
    results = MLIRType[result, ]
    operands = Value[value, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("execution_scope", execution_scope), namedattribute("group_operation", group_operation), ]
    (cluster_size != nothing) && push!(operands, cluster_size)
    
    create_operation(
        "spv.GroupNonUniformFMin", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`GroupNonUniformFMul`

Result Type  must be a scalar or vector of floating-point type.

Execution must be Workgroup or Subgroup Scope.

The identity I for Operation is 1. If Operation is ClusteredReduce,
ClusterSize must be specified.

 The type of Value must be the same as Result Type.  The method used to
perform the group operation on the contributed Value(s) from active
invocations is implementation defined.

ClusterSize is the size of cluster to use. ClusterSize must be a scalar
of integer type, whose Signedness operand is 0. ClusterSize must come
from a constant instruction. ClusterSize must be at least 1, and must be
a power of 2. If ClusterSize is greater than the declared SubGroupSize,
executing this instruction results in undefined behavior.

<!-- End of AutoGen section -->

```
scope ::= `\"Workgroup\"` | `\"Subgroup\"`
operation ::= `\"Reduce\"` | `\"InclusiveScan\"` | `\"ExclusiveScan\"` | ...
float-scalar-vector-type ::= float-type |
                             `vector<` integer-literal `x` float-type `>`
non-uniform-fmul-op ::= ssa-id `=` `spv.GroupNonUniformFMul` scope operation
                        ssa-use ( `cluster_size` `(` ssa_use `)` )?
                        `:` float-scalar-vector-type
```

#### Example:

```mlir
%four = spv.Constant 4 : i32
%scalar = ... : f32
%vector = ... : vector<4xf32>
%0 = spv.GroupNonUniformFMul \"Workgroup\" \"Reduce\" %scalar : f32
%1 = spv.GroupNonUniformFMul \"Subgroup\" \"ClusteredReduce\" %vector cluster_size(%four) : vector<4xf32>
```
"""
function GroupNonUniformFMul(value::Value, cluster_size=nothing::Union{Nothing, Value}; result::MLIRType, execution_scope, group_operation, location=Location())
    results = MLIRType[result, ]
    operands = Value[value, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("execution_scope", execution_scope), namedattribute("group_operation", group_operation), ]
    (cluster_size != nothing) && push!(operands, cluster_size)
    
    create_operation(
        "spv.GroupNonUniformFMul", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`GroupNonUniformIAdd`

Result Type  must be a scalar or vector of integer type.

Execution must be Workgroup or Subgroup Scope.

The identity I for Operation is 0. If Operation is ClusteredReduce,
ClusterSize must be specified.

 The type of Value must be the same as Result Type.

ClusterSize is the size of cluster to use. ClusterSize must be a scalar
of integer type, whose Signedness operand is 0. ClusterSize must come
from a constant instruction. ClusterSize must be at least 1, and must be
a power of 2. If ClusterSize is greater than the declared SubGroupSize,
executing this instruction results in undefined behavior.

<!-- End of AutoGen section -->

```
scope ::= `\"Workgroup\"` | `\"Subgroup\"`
operation ::= `\"Reduce\"` | `\"InclusiveScan\"` | `\"ExclusiveScan\"` | ...
integer-scalar-vector-type ::= integer-type |
                             `vector<` integer-literal `x` integer-type `>`
non-uniform-iadd-op ::= ssa-id `=` `spv.GroupNonUniformIAdd` scope operation
                        ssa-use ( `cluster_size` `(` ssa_use `)` )?
                        `:` integer-scalar-vector-type
```

#### Example:

```mlir
%four = spv.Constant 4 : i32
%scalar = ... : i32
%vector = ... : vector<4xi32>
%0 = spv.GroupNonUniformIAdd \"Workgroup\" \"Reduce\" %scalar : i32
%1 = spv.GroupNonUniformIAdd \"Subgroup\" \"ClusteredReduce\" %vector cluster_size(%four) : vector<4xi32>
```
"""
function GroupNonUniformIAdd(value::Value, cluster_size=nothing::Union{Nothing, Value}; result::MLIRType, execution_scope, group_operation, location=Location())
    results = MLIRType[result, ]
    operands = Value[value, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("execution_scope", execution_scope), namedattribute("group_operation", group_operation), ]
    (cluster_size != nothing) && push!(operands, cluster_size)
    
    create_operation(
        "spv.GroupNonUniformIAdd", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`GroupNonUniformIMul`

Result Type  must be a scalar or vector of integer type.

Execution must be Workgroup or Subgroup Scope.

The identity I for Operation is 1. If Operation is ClusteredReduce,
ClusterSize must be specified.

 The type of Value must be the same as Result Type.

ClusterSize is the size of cluster to use. ClusterSize must be a scalar
of integer type, whose Signedness operand is 0. ClusterSize must come
from a constant instruction. ClusterSize must be at least 1, and must be
a power of 2. If ClusterSize is greater than the declared SubGroupSize,
executing this instruction results in undefined behavior.

<!-- End of AutoGen section -->

```
scope ::= `\"Workgroup\"` | `\"Subgroup\"`
operation ::= `\"Reduce\"` | `\"InclusiveScan\"` | `\"ExclusiveScan\"` | ...
integer-scalar-vector-type ::= integer-type |
                             `vector<` integer-literal `x` integer-type `>`
non-uniform-imul-op ::= ssa-id `=` `spv.GroupNonUniformIMul` scope operation
                        ssa-use ( `cluster_size` `(` ssa_use `)` )?
                        `:` integer-scalar-vector-type
```

#### Example:

```mlir
%four = spv.Constant 4 : i32
%scalar = ... : i32
%vector = ... : vector<4xi32>
%0 = spv.GroupNonUniformIMul \"Workgroup\" \"Reduce\" %scalar : i32
%1 = spv.GroupNonUniformIMul \"Subgroup\" \"ClusteredReduce\" %vector cluster_size(%four) : vector<4xi32>
```
"""
function GroupNonUniformIMul(value::Value, cluster_size=nothing::Union{Nothing, Value}; result::MLIRType, execution_scope, group_operation, location=Location())
    results = MLIRType[result, ]
    operands = Value[value, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("execution_scope", execution_scope), namedattribute("group_operation", group_operation), ]
    (cluster_size != nothing) && push!(operands, cluster_size)
    
    create_operation(
        "spv.GroupNonUniformIMul", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`GroupNonUniformSMax`

Result Type  must be a scalar or vector of integer type.

Execution must be Workgroup or Subgroup Scope.

The identity I for Operation is INT_MIN. If Operation is
ClusteredReduce, ClusterSize must be specified.

 The type of Value must be the same as Result Type.

ClusterSize is the size of cluster to use. ClusterSize must be a scalar
of integer type, whose Signedness operand is 0. ClusterSize must come
from a constant instruction. ClusterSize must be at least 1, and must be
a power of 2. If ClusterSize is greater than the declared SubGroupSize,
executing this instruction results in undefined behavior.

<!-- End of AutoGen section -->

```
scope ::= `\"Workgroup\"` | `\"Subgroup\"`
operation ::= `\"Reduce\"` | `\"InclusiveScan\"` | `\"ExclusiveScan\"` | ...
integer-scalar-vector-type ::= integer-type |
                             `vector<` integer-literal `x` integer-type `>`
non-uniform-smax-op ::= ssa-id `=` `spv.GroupNonUniformSMax` scope operation
                        ssa-use ( `cluster_size` `(` ssa_use `)` )?
                        `:` integer-scalar-vector-type
```

#### Example:

```mlir
%four = spv.Constant 4 : i32
%scalar = ... : i32
%vector = ... : vector<4xi32>
%0 = spv.GroupNonUniformSMax \"Workgroup\" \"Reduce\" %scalar : i32
%1 = spv.GroupNonUniformSMax \"Subgroup\" \"ClusteredReduce\" %vector cluster_size(%four) : vector<4xi32>
```
"""
function GroupNonUniformSMax(value::Value, cluster_size=nothing::Union{Nothing, Value}; result::MLIRType, execution_scope, group_operation, location=Location())
    results = MLIRType[result, ]
    operands = Value[value, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("execution_scope", execution_scope), namedattribute("group_operation", group_operation), ]
    (cluster_size != nothing) && push!(operands, cluster_size)
    
    create_operation(
        "spv.GroupNonUniformSMax", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`GroupNonUniformSMin`

Result Type  must be a scalar or vector of integer type.

Execution must be Workgroup or Subgroup Scope.

The identity I for Operation is INT_MAX. If Operation is
ClusteredReduce, ClusterSize must be specified.

 The type of Value must be the same as Result Type.

ClusterSize is the size of cluster to use. ClusterSize must be a scalar
of integer type, whose Signedness operand is 0. ClusterSize must come
from a constant instruction. ClusterSize must be at least 1, and must be
a power of 2. If ClusterSize is greater than the declared SubGroupSize,
executing this instruction results in undefined behavior.

<!-- End of AutoGen section -->

```
scope ::= `\"Workgroup\"` | `\"Subgroup\"`
operation ::= `\"Reduce\"` | `\"InclusiveScan\"` | `\"ExclusiveScan\"` | ...
integer-scalar-vector-type ::= integer-type |
                             `vector<` integer-literal `x` integer-type `>`
non-uniform-smin-op ::= ssa-id `=` `spv.GroupNonUniformSMin` scope operation
                        ssa-use ( `cluster_size` `(` ssa_use `)` )?
                        `:` integer-scalar-vector-type
```

#### Example:

```mlir
%four = spv.Constant 4 : i32
%scalar = ... : i32
%vector = ... : vector<4xi32>
%0 = spv.GroupNonUniformSMin \"Workgroup\" \"Reduce\" %scalar : i32
%1 = spv.GroupNonUniformSMin \"Subgroup\" \"ClusteredReduce\" %vector cluster_size(%four) : vector<4xi32>
```
"""
function GroupNonUniformSMin(value::Value, cluster_size=nothing::Union{Nothing, Value}; result::MLIRType, execution_scope, group_operation, location=Location())
    results = MLIRType[result, ]
    operands = Value[value, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("execution_scope", execution_scope), namedattribute("group_operation", group_operation), ]
    (cluster_size != nothing) && push!(operands, cluster_size)
    
    create_operation(
        "spv.GroupNonUniformSMin", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`GroupNonUniformUMax`

Result Type  must be a scalar or vector of integer type, whose
Signedness operand is 0.

Execution must be Workgroup or Subgroup Scope.

The identity I for Operation is 0. If Operation is ClusteredReduce,
ClusterSize must be specified.

 The type of Value must be the same as Result Type.

ClusterSize is the size of cluster to use. ClusterSize must be a scalar
of integer type, whose Signedness operand is 0. ClusterSize must come
from a constant instruction. ClusterSize must be at least 1, and must be
a power of 2. If ClusterSize is greater than the declared SubGroupSize,
executing this instruction results in undefined behavior.

<!-- End of AutoGen section -->

```
scope ::= `\"Workgroup\"` | `\"Subgroup\"`
operation ::= `\"Reduce\"` | `\"InclusiveScan\"` | `\"ExclusiveScan\"` | ...
integer-scalar-vector-type ::= integer-type |
                             `vector<` integer-literal `x` integer-type `>`
non-uniform-umax-op ::= ssa-id `=` `spv.GroupNonUniformUMax` scope operation
                        ssa-use ( `cluster_size` `(` ssa_use `)` )?
                        `:` integer-scalar-vector-type
```

#### Example:

```mlir
%four = spv.Constant 4 : i32
%scalar = ... : i32
%vector = ... : vector<4xi32>
%0 = spv.GroupNonUniformUMax \"Workgroup\" \"Reduce\" %scalar : i32
%1 = spv.GroupNonUniformUMax \"Subgroup\" \"ClusteredReduce\" %vector cluster_size(%four) : vector<4xi32>
```
"""
function GroupNonUniformUMax(value::Value, cluster_size=nothing::Union{Nothing, Value}; result::MLIRType, execution_scope, group_operation, location=Location())
    results = MLIRType[result, ]
    operands = Value[value, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("execution_scope", execution_scope), namedattribute("group_operation", group_operation), ]
    (cluster_size != nothing) && push!(operands, cluster_size)
    
    create_operation(
        "spv.GroupNonUniformUMax", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`GroupNonUniformUMin`

Result Type  must be a scalar or vector of integer type, whose
Signedness operand is 0.

Execution must be Workgroup or Subgroup Scope.

The identity I for Operation is UINT_MAX. If Operation is
ClusteredReduce, ClusterSize must be specified.

 The type of Value must be the same as Result Type.

ClusterSize is the size of cluster to use. ClusterSize must be a scalar
of integer type, whose Signedness operand is 0. ClusterSize must come
from a constant instruction. ClusterSize must be at least 1, and must be
a power of 2. If ClusterSize is greater than the declared SubGroupSize,
executing this instruction results in undefined behavior.

<!-- End of AutoGen section -->

```
scope ::= `\"Workgroup\"` | `\"Subgroup\"`
operation ::= `\"Reduce\"` | `\"InclusiveScan\"` | `\"ExclusiveScan\"` | ...
integer-scalar-vector-type ::= integer-type |
                             `vector<` integer-literal `x` integer-type `>`
non-uniform-umin-op ::= ssa-id `=` `spv.GroupNonUniformUMin` scope operation
                        ssa-use ( `cluster_size` `(` ssa_use `)` )?
                        `:` integer-scalar-vector-type
```

#### Example:

```mlir
%four = spv.Constant 4 : i32
%scalar = ... : i32
%vector = ... : vector<4xi32>
%0 = spv.GroupNonUniformUMin \"Workgroup\" \"Reduce\" %scalar : i32
%1 = spv.GroupNonUniformUMin \"Subgroup\" \"ClusteredReduce\" %vector cluster_size(%four) : vector<4xi32>
```
"""
function GroupNonUniformUMin(value::Value, cluster_size=nothing::Union{Nothing, Value}; result::MLIRType, execution_scope, group_operation, location=Location())
    results = MLIRType[result, ]
    operands = Value[value, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("execution_scope", execution_scope), namedattribute("group_operation", group_operation), ]
    (cluster_size != nothing) && push!(operands, cluster_size)
    
    create_operation(
        "spv.GroupNonUniformUMin", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`IAdd`

Result Type must be a scalar or vector of integer type.

 The type of Operand 1 and Operand 2  must be a scalar or vector of
integer type.  They must have the same number of components as Result
Type. They must have the same component width as Result Type.

The resulting value will equal the low-order N bits of the correct
result R, where N is the component width and R is computed with enough
precision to avoid overflow and underflow.

 Results are computed per component.

<!-- End of AutoGen section -->
```
integer-scalar-vector-type ::= integer-type |
                             `vector<` integer-literal `x` integer-type `>`
iadd-op ::= ssa-id `=` `spv.IAdd` ssa-use, ssa-use
                      `:` integer-scalar-vector-type
```

#### Example:

```mlir
%4 = spv.IAdd %0, %1 : i32
%5 = spv.IAdd %2, %3 : vector<4xi32>

```
"""
function IAdd(operand1::Value, operand2::Value; result=nothing::Union{Nothing, MLIRType}, location=Location())
    results = MLIRType[]
    operands = Value[operand1, operand2, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (result != nothing) && push!(results, result)
    
    create_operation(
        "spv.IAdd", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`IEqual`

Result Type must be a scalar or vector of Boolean type.

 The type of Operand 1 and Operand 2  must be a scalar or vector of
integer type.  They must have the same component width, and they must
have the same number of components as Result Type.

 Results are computed per component.

<!-- End of AutoGen section -->
```
integer-scalar-vector-type ::= integer-type |
                             `vector<` integer-literal `x` integer-type `>`
iequal-op ::= ssa-id `=` `spv.IEqual` ssa-use, ssa-use
                         `:` integer-scalar-vector-type
```
#### Example:

```mlir
%4 = spv.IEqual %0, %1 : i32
%5 = spv.IEqual %2, %3 : vector<4xi32>

```
"""
function IEqual(operand1::Value, operand2::Value; result::MLIRType, location=Location())
    results = MLIRType[result, ]
    operands = Value[operand1, operand2, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "spv.IEqual", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`IMul`

Result Type must be a scalar or vector of integer type.

 The type of Operand 1 and Operand 2  must be a scalar or vector of
integer type.  They must have the same number of components as Result
Type. They must have the same component width as Result Type.

The resulting value will equal the low-order N bits of the correct
result R, where N is the component width and R is computed with enough
precision to avoid overflow and underflow.

 Results are computed per component.

<!-- End of AutoGen section -->
```
integer-scalar-vector-type ::= integer-type |
                             `vector<` integer-literal `x` integer-type `>`
imul-op ::= ssa-id `=` `spv.IMul` ssa-use, ssa-use
                      `:` integer-scalar-vector-type
```

#### Example:

```mlir
%4 = spv.IMul %0, %1 : i32
%5 = spv.IMul %2, %3 : vector<4xi32>

```
"""
function IMul(operand1::Value, operand2::Value; result=nothing::Union{Nothing, MLIRType}, location=Location())
    results = MLIRType[]
    operands = Value[operand1, operand2, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (result != nothing) && push!(results, result)
    
    create_operation(
        "spv.IMul", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`INotEqual`

Result Type must be a scalar or vector of Boolean type.

 The type of Operand 1 and Operand 2  must be a scalar or vector of
integer type.  They must have the same component width, and they must
have the same number of components as Result Type.

 Results are computed per component.

<!-- End of AutoGen section -->
```
integer-scalar-vector-type ::= integer-type |
                             `vector<` integer-literal `x` integer-type `>`
inot-equal-op ::= ssa-id `=` `spv.INotEqual` ssa-use, ssa-use
                             `:` integer-scalar-vector-type
```
#### Example:

```mlir
%4 = spv.INotEqual %0, %1 : i32
%5 = spv.INotEqual %2, %3 : vector<4xi32>

```
"""
function INotEqual(operand1::Value, operand2::Value; result::MLIRType, location=Location())
    results = MLIRType[result, ]
    operands = Value[operand1, operand2, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "spv.INotEqual", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`ISubBorrow`

Result Type must be from OpTypeStruct.  The struct must have two
members, and the two members must be the same type.  The member type
must be a scalar or vector of integer type, whose Signedness operand is
0.

Operand 1 and Operand 2 must have the same type as the members of Result
Type. These are consumed as unsigned integers.

 Results are computed per component.

Member 0 of the result gets the low-order bits (full component width) of
the subtraction. That is, if Operand 1 is larger than Operand 2, member
0 gets the full value of the subtraction;  if Operand 2 is larger than
Operand 1, member 0 gets 2w + Operand 1 - Operand 2, where w is the
component width.

Member 1 of the result gets 0 if Operand 1 ≥ Operand 2, and gets 1
otherwise.

<!-- End of AutoGen section -->

#### Example:

```mlir
%2 = spv.ISubBorrow %0, %1 : !spv.struct<(i32, i32)>
%2 = spv.ISubBorrow %0, %1 : !spv.struct<(vector<2xi32>, vector<2xi32>)>
```
"""
function ISubBorrow(operand1::Value, operand2::Value; result::MLIRType, location=Location())
    results = MLIRType[result, ]
    operands = Value[operand1, operand2, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "spv.ISubBorrow", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`ISub`

Result Type must be a scalar or vector of integer type.

 The type of Operand 1 and Operand 2  must be a scalar or vector of
integer type.  They must have the same number of components as Result
Type. They must have the same component width as Result Type.

The resulting value will equal the low-order N bits of the correct
result R, where N is the component width and R is computed with enough
precision to avoid overflow and underflow.

 Results are computed per component.

<!-- End of AutoGen section -->
```
integer-scalar-vector-type ::= integer-type |
                             `vector<` integer-literal `x` integer-type `>`
isub-op ::= `spv.ISub` ssa-use, ssa-use
                      `:` integer-scalar-vector-type
```

#### Example:

```mlir
%4 = spv.ISub %0, %1 : i32
%5 = spv.ISub %2, %3 : vector<4xi32>

```
"""
function ISub(operand1::Value, operand2::Value; result=nothing::Union{Nothing, MLIRType}, location=Location())
    results = MLIRType[]
    operands = Value[operand1, operand2, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (result != nothing) && push!(results, result)
    
    create_operation(
        "spv.ISub", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`ImageDrefGather`

Result Type must be a vector of four components of floating-point type
or integer type.  Its components must be the same as Sampled Type of the
underlying OpTypeImage (unless that underlying Sampled Type is
OpTypeVoid). It has one component per gathered texel.

Sampled Image must be an object whose type is OpTypeSampledImage. Its
OpTypeImage must have a Dim of 2D, Cube, or Rect. The MS operand of the
underlying OpTypeImage must be 0.

Coordinate  must be a scalar or vector of floating-point type.  It
contains (u[, v] … [, array layer]) as needed by the definition of
Sampled Image.

Dref is the depth-comparison reference value. It must be a 32-bit
floating-point type scalar.

Image Operands encodes what operands follow, as per Image Operands.

<!-- End of AutoGen section -->
```
image-operands ::= `\"None\"` | `\"Bias\"` | `\"Lod\"` | `\"Grad\"`
                  | `\"ConstOffset\"` | `\"Offser\"` | `\"ConstOffsets\"`
                  | `\"Sample\"` | `\"MinLod\"` | `\"MakeTexelAvailable\"`
                  | `\"MakeTexelVisible\"` | `\"NonPrivateTexel\"`
                  | `\"VolatileTexel\"` | `\"SignExtend\"` | `\"ZeroExtend\"`
#### Example:
```

```mlir
%0 = spv.ImageDrefGather %1 : !spv.sampled_image<!spv.image<i32, Dim2D, NoDepth, NonArrayed, SingleSampled, NoSampler, Unknown>>, %2 : vector<4xf32>, %3 : f32 -> vector<4xi32>
%0 = spv.ImageDrefGather %1 : !spv.sampled_image<!spv.image<i32, Dim2D, NoDepth, NonArrayed, SingleSampled, NoSampler, Unknown>>, %2 : vector<4xf32>, %3 : f32 [\"NonPrivateTexel\"] : f32, f32 -> vector<4xi32>
```
"""
function ImageDrefGather(sampledimage::Value, coordinate::Value, dref::Value, operand_arguments::Vector{Value}; result::MLIRType, imageoperands=nothing, location=Location())
    results = MLIRType[result, ]
    operands = Value[sampledimage, coordinate, dref, operand_arguments..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (imageoperands != nothing) && push!(attributes, namedattribute("imageoperands", imageoperands))
    
    create_operation(
        "spv.ImageDrefGather", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`Image`

Result Type must be OpTypeImage.

Sampled Image must have type OpTypeSampledImage whose Image Type is the
same as Result Type.

<!-- End of AutoGen section -->

#### Example:

```mlir
%0 = spv.Image %1 : !spv.sampled_image<!spv.image<f32, Cube, NoDepth, NonArrayed, SingleSampled, NoSampler, Unknown>>
```
"""
function Image(sampledimage::Value; result::MLIRType, location=Location())
    results = MLIRType[result, ]
    operands = Value[sampledimage, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "spv.Image", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`ImageQuerySize`

Result Type must be an integer type scalar or vector.  The number of
components must be:

1 for the 1D and Buffer dimensionalities,

2 for the 2D, Cube, and Rect dimensionalities,

3 for the 3D dimensionality,

plus 1 more if the image type is arrayed. This vector is filled in with
(width [, height] [, elements]) where elements is the number of layers
in an image array or the number of cubes in a cube-map array.

Image must be an object whose type is OpTypeImage. Its Dim operand must
be one of those listed under Result Type, above. Additionally, if its
Dim is 1D, 2D, 3D, or Cube, it must also have either an MS of 1 or a
Sampled of 0 or 2. There is no implicit level-of-detail consumed by this
instruction. See OpImageQuerySizeLod for querying images having level of
detail. This operation is allowed on an image decorated as NonReadable.
See the client API specification for additional image type restrictions.

<!-- End of AutoGen section -->

#### Example:

```mlir
%3 = spv.ImageQuerySize %0 : !spv.image<i32, Dim1D, NoDepth, NonArrayed, SingleSampled, NoSampler, Unknown> -> i32
%4 = spv.ImageQuerySize %1 : !spv.image<i32, Dim2D, NoDepth, NonArrayed, SingleSampled, NoSampler, Unknown> -> vector<2xi32>
%5 = spv.ImageQuerySize %2 : !spv.image<i32, Dim2D, NoDepth, Arrayed, SingleSampled, NoSampler, Unknown> -> vector<3xi32>
```
"""
function ImageQuerySize(image::Value; result::MLIRType, location=Location())
    results = MLIRType[result, ]
    operands = Value[image, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "spv.ImageQuerySize", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`InBoundsPtrAccessChain`



<!-- End of AutoGen section -->

```
access-chain-op ::= ssa-id `=` `spv.InBoundsPtrAccessChain` ssa-use
                    `[` ssa-use (\',\' ssa-use)* `]`
                    `:` pointer-type
```mlir

#### Example:

```
func @inbounds_ptr_access_chain(%arg0: !spv.ptr<f32, CrossWorkgroup>, %arg1 : i64) -> () {
  %0 = spv.InBoundsPtrAccessChain %arg0[%arg1] : !spv.ptr<f32, CrossWorkgroup>, i64
  ...
}
```
"""
function InBoundsPtrAccessChain(base_ptr::Value, element::Value, indices::Vector{Value}; result::MLIRType, location=Location())
    results = MLIRType[result, ]
    operands = Value[base_ptr, element, indices..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "spv.InBoundsPtrAccessChain", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`IsInf`

Result Type must be a scalar or vector of Boolean type.

x must be a scalar or vector of floating-point type.  It must have the
same number of components as Result Type.

 Results are computed per component.

<!-- End of AutoGen section -->

```
float-scalar-vector-type ::= float-type |
                             `vector<` integer-literal `x` float-type `>`
isinf-op ::= ssa-id `=` `spv.IsInf` ssa-use
                        `:` float-scalar-vector-type
```

#### Example:

```mlir
%2 = spv.IsInf %0: f32
%3 = spv.IsInf %1: vector<4xi32>
```
"""
function IsInf(operand::Value; result::MLIRType, location=Location())
    results = MLIRType[result, ]
    operands = Value[operand, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "spv.IsInf", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`IsNan`

Result Type must be a scalar or vector of Boolean type.

x must be a scalar or vector of floating-point type.  It must have the
same number of components as Result Type.

 Results are computed per component.

<!-- End of AutoGen section -->

```
float-scalar-vector-type ::= float-type |
                             `vector<` integer-literal `x` float-type `>`
isnan-op ::= ssa-id `=` `spv.IsNan` ssa-use
                        `:` float-scalar-vector-type
```

#### Example:

```mlir
%2 = spv.IsNan %0: f32
%3 = spv.IsNan %1: vector<4xi32>
```
"""
function IsNan(operand::Value; result::MLIRType, location=Location())
    results = MLIRType[result, ]
    operands = Value[operand, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "spv.IsNan", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`Load`

Result Type is the type of the loaded object. It must be a type with
fixed size; i.e., it cannot be, nor include, any OpTypeRuntimeArray
types.

Pointer is the pointer to load through.  Its type must be an
OpTypePointer whose Type operand is the same as Result Type.

If present, any Memory Operands must begin with a memory operand
literal. If not present, it is the same as specifying the memory operand
None.

<!-- End of AutoGen section -->

```
memory-access ::= `\"None\"` | `\"Volatile\"` | `\"Aligned\", ` integer-literal
                | `\"NonTemporal\"`

load-op ::= ssa-id ` = spv.Load ` storage-class ssa-use
            (`[` memory-access `]`)? ` : ` spirv-element-type
```

#### Example:

```mlir
%0 = spv.Variable : !spv.ptr<f32, Function>
%1 = spv.Load \"Function\" %0 : f32
%2 = spv.Load \"Function\" %0 [\"Volatile\"] : f32
%3 = spv.Load \"Function\" %0 [\"Aligned\", 4] : f32
```
"""
function Load(ptr::Value; value::MLIRType, memory_access=nothing, alignment=nothing, location=Location())
    results = MLIRType[value, ]
    operands = Value[ptr, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (memory_access != nothing) && push!(attributes, namedattribute("memory_access", memory_access))
    (alignment != nothing) && push!(attributes, namedattribute("alignment", alignment))
    
    create_operation(
        "spv.Load", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`LogicalAnd`

Result Type must be a scalar or vector of Boolean type.

 The type of Operand 1 must be the same as Result Type.

 The type of Operand 2 must be the same as Result Type.

 Results are computed per component.

<!-- End of AutoGen section -->

```
logical-and ::= `spv.LogicalAnd` ssa-use `,` ssa-use
                `:` operand-type
```

#### Example:

```mlir
%2 = spv.LogicalAnd %0, %1 : i1
%2 = spv.LogicalAnd %0, %1 : vector<4xi1>
```
"""
function LogicalAnd(operand1::Value, operand2::Value; result::MLIRType, location=Location())
    results = MLIRType[result, ]
    operands = Value[operand1, operand2, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "spv.LogicalAnd", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`LogicalEqual`

Result Type must be a scalar or vector of Boolean type.

 The type of Operand 1 must be the same as Result Type.

 The type of Operand 2 must be the same as Result Type.

 Results are computed per component.

<!-- End of AutoGen section -->

```
logical-equal ::= `spv.LogicalEqual` ssa-use `,` ssa-use
                  `:` operand-type
```

#### Example:

```mlir
%2 = spv.LogicalEqual %0, %1 : i1
%2 = spv.LogicalEqual %0, %1 : vector<4xi1>
```
"""
function LogicalEqual(operand1::Value, operand2::Value; result::MLIRType, location=Location())
    results = MLIRType[result, ]
    operands = Value[operand1, operand2, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "spv.LogicalEqual", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`LogicalNotEqual`

Result Type must be a scalar or vector of Boolean type.

 The type of Operand 1 must be the same as Result Type.

 The type of Operand 2 must be the same as Result Type.

 Results are computed per component.

<!-- End of AutoGen section -->

```
logical-not-equal ::= `spv.LogicalNotEqual` ssa-use `,` ssa-use
                      `:` operand-type
```

#### Example:

```mlir
%2 = spv.LogicalNotEqual %0, %1 : i1
%2 = spv.LogicalNotEqual %0, %1 : vector<4xi1>
```
"""
function LogicalNotEqual(operand1::Value, operand2::Value; result::MLIRType, location=Location())
    results = MLIRType[result, ]
    operands = Value[operand1, operand2, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "spv.LogicalNotEqual", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`LogicalNot`

Result Type must be a scalar or vector of Boolean type.

 The type of Operand must be the same as Result Type.

 Results are computed per component.

<!-- End of AutoGen section -->

```
logical-not ::= `spv.LogicalNot` ssa-use `:` operand-type
```

#### Example:

```mlir
%2 = spv.LogicalNot %0 : i1
%2 = spv.LogicalNot %0 : vector<4xi1>
```
"""
function LogicalNot(operand::Value; result::MLIRType, location=Location())
    results = MLIRType[result, ]
    operands = Value[operand, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "spv.LogicalNot", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`LogicalOr`

Result Type must be a scalar or vector of Boolean type.

 The type of Operand 1 must be the same as Result Type.

 The type of Operand 2 must be the same as Result Type.

 Results are computed per component.

<!-- End of AutoGen section -->

```
logical-or ::= `spv.LogicalOr` ssa-use `,` ssa-use
                `:` operand-type
```

#### Example:

```mlir
%2 = spv.LogicalOr %0, %1 : i1
%2 = spv.LogicalOr %0, %1 : vector<4xi1>
```
"""
function LogicalOr(operand1::Value, operand2::Value; result::MLIRType, location=Location())
    results = MLIRType[result, ]
    operands = Value[operand1, operand2, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "spv.LogicalOr", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`mlir_loop`

SPIR-V can explicitly declare structured control-flow constructs using merge
instructions. These explicitly declare a header block before the control
flow diverges and a merge block where control flow subsequently converges.
These blocks delimit constructs that must nest, and can only be entered
and exited in structured ways. See \"2.11. Structured Control Flow\" of the
SPIR-V spec for more details.

Instead of having a `spv.LoopMerge` op to directly model loop merge
instruction for indicating the merge and continue target, we use regions
to delimit the boundary of the loop: the merge target is the next op
following the `spv.mlir.loop` op and the continue target is the block that
has a back-edge pointing to the entry block inside the `spv.mlir.loop`\'s region.
This way it\'s easier to discover all blocks belonging to a construct and
it plays nicer with the MLIR system.

The `spv.mlir.loop` region should contain at least four blocks: one entry block,
one loop header block, one loop continue block, one loop merge block.
The entry block should be the first block and it should jump to the loop
header block, which is the second block. The loop merge block should be the
last block. The merge block should only contain a `spv.mlir.merge` op.
The continue block should be the second to last block and it should have a
branch to the loop header block. The loop continue block should be the only
block, except the entry block, branching to the header block.
"""
function mlir_loop(; loop_control, body::Region, location=Location())
    results = MLIRType[]
    operands = Value[]
    owned_regions = Region[body, ]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("loop_control", loop_control), ]
    
    create_operation(
        "spv.mlir.loop", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`MatrixTimesMatrix`

Result Type must be an OpTypeMatrix whose Column Type is a vector of
floating-point type.

LeftMatrix must be a matrix whose Column Type is the same as the Column
Type in Result Type.

RightMatrix must be a matrix with the same Component Type as the
Component Type in Result Type. Its number of columns must equal the
number of columns in Result Type. Its columns must have the same number
of components as the number of columns in LeftMatrix.

<!-- End of AutoGen section -->

```
matrix-times-matrix-op ::= ssa-id `=` `spv.MatrixTimesMatrix` ssa-use,
ssa-use `:` matrix-type `,` matrix-type `->` matrix-type
```mlir

#### Example:

```
%0 = spv.MatrixTimesMatrix %matrix_1, %matrix_2 :
    !spv.matrix<4 x vector<3xf32>>, !spv.matrix<3 x vector<4xf32>> ->
    !spv.matrix<4 x vector<4xf32>>
```
"""
function MatrixTimesMatrix(leftmatrix::Value, rightmatrix::Value; result::MLIRType, location=Location())
    results = MLIRType[result, ]
    operands = Value[leftmatrix, rightmatrix, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "spv.MatrixTimesMatrix", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`MatrixTimesScalar`

Result Type must be an OpTypeMatrix whose Column Type is a vector of
floating-point type.

 The type of Matrix must be the same as Result Type. Each component in
each column in Matrix is multiplied by Scalar.

Scalar must have the same type as the Component Type in Result Type.

<!-- End of AutoGen section -->

```
matrix-times-scalar-op ::= ssa-id `=` `spv.MatrixTimesScalar` ssa-use,
ssa-use `:` matrix-type `,` float-type `->` matrix-type

```

#### Example:

```mlir

%0 = spv.MatrixTimesScalar %matrix, %scalar :
!spv.matrix<3 x vector<3xf32>>, f32 -> !spv.matrix<3 x vector<3xf32>>

```
"""
function MatrixTimesScalar(matrix::Value, scalar::Value; result::MLIRType, location=Location())
    results = MLIRType[result, ]
    operands = Value[matrix, scalar, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "spv.MatrixTimesScalar", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`MemoryBarrier`

Ensures that memory accesses issued before this instruction will be
observed before memory accesses issued after this instruction. This
control is ensured only for memory accesses issued by this invocation
and observed by another invocation executing within Memory scope. If the
Vulkan memory model is declared, this ordering only applies to memory
accesses that use the NonPrivatePointer memory operand or
NonPrivateTexel image operand.

Semantics declares what kind of memory is being controlled and what kind
of control to apply.

To execute both a memory barrier and a control barrier, see
OpControlBarrier.

<!-- End of AutoGen section -->

```
scope ::= `\"CrossDevice\"` | `\"Device\"` | `\"Workgroup\"` | ...

memory-semantics ::= `\"None\"` | `\"Acquire\"` | `\"Release\"` | ...

memory-barrier-op ::= `spv.MemoryBarrier` scope, memory-semantics
```

#### Example:

```mlir
spv.MemoryBarrier \"Device\", \"Acquire|UniformMemory\"

```
"""
function MemoryBarrier(; memory_scope, memory_semantics, location=Location())
    results = MLIRType[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("memory_scope", memory_scope), namedattribute("memory_semantics", memory_semantics), ]
    
    create_operation(
        "spv.MemoryBarrier", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`mlir_merge`

We use `spv.mlir.selection`/`spv.mlir.loop` for modelling structured selection/loop.
This op is a terminator used inside their regions to mean jumping to the
merge point, which is the next op following the `spv.mlir.selection` or
`spv.mlir.loop` op. This op does not have a corresponding instruction in the
SPIR-V binary format; it\'s solely for structural purpose.
"""
function mlir_merge(; location=Location())
    results = MLIRType[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "spv.mlir.merge", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`module_`

This op defines a SPIR-V module using a MLIR region. The region contains
one block. Module-level operations, including functions definitions,
are all placed in this block.

Using an op with a region to define a SPIR-V module enables \"embedding\"
SPIR-V modules in other dialects in a clean manner: this op guarantees
the validity and serializability of a SPIR-V module and thus serves as
a clear-cut boundary.

This op takes no operands and generates no results. This op should not
implicitly capture values from the enclosing environment.

This op has only one region, which only contains one block. The block
has no terminator.

<!-- End of AutoGen section -->

```
addressing-model ::= `Logical` | `Physical32` | `Physical64` | ...
memory-model ::= `Simple` | `GLSL450` | `OpenCL` | `Vulkan` | ...
spv-module-op ::= `spv.module` addressing-model memory-model
                  (requires  spirv-vce-attribute)?
                  (`attributes` attribute-dict)?
                  region
```

#### Example:

```mlir
spv.module Logical GLSL450  {}

spv.module Logical Vulkan
    requires #spv.vce<v1.0, [Shader], [SPV_KHR_vulkan_memory_model]>
    attributes { some_additional_attr = ... } {
  spv.func @do_nothing() -> () {
    spv.Return
  }
}
```
"""
function module_(; addressing_model, memory_model, vce_triple=nothing, sym_name=nothing, region_0::Region, location=Location())
    results = MLIRType[]
    operands = Value[]
    owned_regions = Region[region_0, ]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("addressing_model", addressing_model), namedattribute("memory_model", memory_model), ]
    (vce_triple != nothing) && push!(attributes, namedattribute("vce_triple", vce_triple))
    (sym_name != nothing) && push!(attributes, namedattribute("sym_name", sym_name))
    
    create_operation(
        "spv.module", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`Not`

Results are computed per component, and within each component, per bit.

    Result Type must be a scalar or vector of integer type.

    Operand\'s type  must be a scalar or vector of integer type.  It must
    have the same number of components as Result Type.  The component width
    must equal the component width in Result Type.

    <!-- End of AutoGen section -->

    ```
    integer-scalar-vector-type ::= integer-type |
                             `vector<` integer-literal `x` integer-type `>`
    not-op ::= ssa-id `=` `spv.BitNot` ssa-use `:` integer-scalar-vector-type
    ```

    #### Example:

    ```mlir
    %2 = spv.Not %0 : i32
    %3 = spv.Not %1 : vector<4xi32>
    ```
"""
function Not(operand::Value; result=nothing::Union{Nothing, MLIRType}, location=Location())
    results = MLIRType[]
    operands = Value[operand, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (result != nothing) && push!(results, result)
    
    create_operation(
        "spv.Not", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`Ordered`

Result Type must be a scalar or vector of Boolean type.

x must be a scalar or vector of floating-point type.  It must have the
same number of components as Result Type.

y must have the same type as x.

 Results are computed per component.

<!-- End of AutoGen section -->

```
float-scalar-vector-type ::= float-type |
                             `vector<` integer-literal `x` float-type `>`
ordered-op ::= ssa-id `=` `spv.Ordered` ssa-use, ssa-use
```mlir

#### Example:

```
%4 = spv.Ordered %0, %1 : f32
%5 = spv.Ordered %2, %3 : vector<4xf32>
```
"""
function Ordered(operand1::Value, operand2::Value; result::MLIRType, location=Location())
    results = MLIRType[result, ]
    operands = Value[operand1, operand2, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "spv.Ordered", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`PtrAccessChain`

Element is used to do an initial dereference of Base: Base is treated as
the address of an element in an array, and a new element address is
computed from Base and Element to become the OpAccessChain Base to
dereference as per OpAccessChain. This computed Base has the same type
as the originating Base.

To compute the new element address, Element is treated as a signed count
of elements E, relative to the original Base element B, and the address
of element B + E is computed using enough precision to avoid overflow
and underflow. For objects in the Uniform, StorageBuffer, or
PushConstant storage classes, the element\'s address or location is
calculated using a stride, which will be the Base-type\'s Array Stride if
the Base type is decorated with ArrayStride. For all other objects, the
implementation calculates the element\'s address or location.

With one exception, undefined behavior results when B + E is not an
element in the same array (same innermost array, if array types are
nested) as B. The exception being when B + E = L, where L is the length
of the array: the address computation for element L is done with the
same stride as any other B + E computation that stays within the array.

Note: If Base is typed to be a pointer to an array and the desired
operation is to select an element of that array, OpAccessChain should be
directly used, as its first Index selects the array element.

<!-- End of AutoGen section -->

```
[access-chain-op ::= ssa-id `=` `spv.PtrAccessChain` ssa-use
                    `[` ssa-use (\',\' ssa-use)* `]`
                    `:` pointer-type
```mlir

#### Example:

```
func @ptr_access_chain(%arg0: !spv.ptr<f32, CrossWorkgroup>, %arg1 : i64) -> () {
  %0 = spv.PtrAccessChain %arg0[%arg1] : !spv.ptr<f32, CrossWorkgroup>, i64
  ...
}
```
"""
function PtrAccessChain(base_ptr::Value, element::Value, indices::Vector{Value}; result::MLIRType, location=Location())
    results = MLIRType[result, ]
    operands = Value[base_ptr, element, indices..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "spv.PtrAccessChain", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`mlir_referenceof`

Specialization constants in module scope are defined using symbol names.
This op generates an SSA value that can be used to refer to the symbol
within function scope for use in ops that expect an SSA value.
This operation has no corresponding SPIR-V instruction; it\'s merely used
for modelling purpose in the SPIR-V dialect. This op\'s return type is
the same as the specialization constant.

<!-- End of AutoGen section -->

```
spv-reference-of-op ::= ssa-id `=` `spv.mlir.referenceof` symbol-ref-id
                                   `:` spirv-scalar-type
```

#### Example:

```mlir
%0 = spv.mlir.referenceof @spec_const : f32
```

TODO Add support for composite specialization constants.
"""
function mlir_referenceof(; reference::MLIRType, spec_const, location=Location())
    results = MLIRType[reference, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("spec_const", spec_const), ]
    
    create_operation(
        "spv.mlir.referenceof", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`Return`

This instruction must be the last instruction in a block.

<!-- End of AutoGen section -->

```
return-op ::= `spv.Return`
```
"""
function Return(; location=Location())
    results = MLIRType[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "spv.Return", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`ReturnValue`

Value is the value returned, by copy, and must match the Return Type
operand of the OpTypeFunction type of the OpFunction body this return
instruction is in.

This instruction must be the last instruction in a block.

<!-- End of AutoGen section -->

```
return-value-op ::= `spv.ReturnValue` ssa-use `:` spirv-type
```

#### Example:

```mlir
spv.ReturnValue %0 : f32
```
"""
function ReturnValue(value::Value; location=Location())
    results = MLIRType[]
    operands = Value[value, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "spv.ReturnValue", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`SConvert`

Result Type must be a scalar or vector of integer type.

Signed Value must be a scalar or vector of integer type.  It must have
the same number of components as Result Type.  The component width
cannot equal the component width in Result Type.

 Results are computed per component.

<!-- End of AutoGen section -->

```
s-convert-op ::= ssa-id `=` `spv.SConvertOp` ssa-use
                 `:` operand-type `to` result-type
```

#### Example:

```mlir
%1 = spv.SConvertOp %0 : i32 to i64
%3 = spv.SConvertOp %2 : vector<3xi32> to vector<3xi64>
```
"""
function SConvert(operand::Value; result::MLIRType, location=Location())
    results = MLIRType[result, ]
    operands = Value[operand, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "spv.SConvert", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`SDiv`

Result Type must be a scalar or vector of integer type.

 The type of Operand 1 and Operand 2  must be a scalar or vector of
integer type.  They must have the same number of components as Result
Type. They must have the same component width as Result Type.

 Results are computed per component.  The resulting value is undefined
if Operand 2 is 0.

<!-- End of AutoGen section -->
```
integer-scalar-vector-type ::= integer-type |
                             `vector<` integer-literal `x` integer-type `>`
sdiv-op ::= ssa-id `=` `spv.SDiv` ssa-use, ssa-use
                       `:` integer-scalar-vector-type
```

#### Example:

```mlir
%4 = spv.SDiv %0, %1 : i32
%5 = spv.SDiv %2, %3 : vector<4xi32>

```
"""
function SDiv(operand1::Value, operand2::Value; result=nothing::Union{Nothing, MLIRType}, location=Location())
    results = MLIRType[]
    operands = Value[operand1, operand2, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (result != nothing) && push!(results, result)
    
    create_operation(
        "spv.SDiv", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`SGreaterThanEqual`

Result Type must be a scalar or vector of Boolean type.

 The type of Operand 1 and Operand 2  must be a scalar or vector of
integer type.  They must have the same component width, and they must
have the same number of components as Result Type.

 Results are computed per component.

<!-- End of AutoGen section -->
```
integer-scalar-vector-type ::= integer-type |
                             `vector<` integer-literal `x` integer-type `>`
sgreater-than-equal-op ::= ssa-id `=` `spv.SGreaterThanEqual` ssa-use, ssa-use
                                      `:` integer-scalar-vector-type
```
#### Example:

```
%4 = spv.SGreaterThanEqual %0, %1 : i32
%5 = spv.SGreaterThanEqual %2, %3 : vector<4xi32>

```
"""
function SGreaterThanEqual(operand1::Value, operand2::Value; result::MLIRType, location=Location())
    results = MLIRType[result, ]
    operands = Value[operand1, operand2, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "spv.SGreaterThanEqual", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`SGreaterThan`

Result Type must be a scalar or vector of Boolean type.

 The type of Operand 1 and Operand 2  must be a scalar or vector of
integer type.  They must have the same component width, and they must
have the same number of components as Result Type.

 Results are computed per component.

<!-- End of AutoGen section -->
```
integer-scalar-vector-type ::= integer-type |
                             `vector<` integer-literal `x` integer-type `>`
sgreater-than-op ::= ssa-id `=` `spv.SGreaterThan` ssa-use, ssa-use
                                `:` integer-scalar-vector-type
```
#### Example:

```mlir
%4 = spv.SGreaterThan %0, %1 : i32
%5 = spv.SGreaterThan %2, %3 : vector<4xi32>

```
"""
function SGreaterThan(operand1::Value, operand2::Value; result::MLIRType, location=Location())
    results = MLIRType[result, ]
    operands = Value[operand1, operand2, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "spv.SGreaterThan", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`SLessThanEqual`

Result Type must be a scalar or vector of Boolean type.

 The type of Operand 1 and Operand 2  must be a scalar or vector of
integer type.  They must have the same component width, and they must
have the same number of components as Result Type.

 Results are computed per component.

<!-- End of AutoGen section -->
```
integer-scalar-vector-type ::= integer-type |
                             `vector<` integer-literal `x` integer-type `>`
sless-than-equal-op ::= ssa-id `=` `spv.SLessThanEqual` ssa-use, ssa-use
                                   `:` integer-scalar-vector-type
```
#### Example:

```mlir
%4 = spv.SLessThanEqual %0, %1 : i32
%5 = spv.SLessThanEqual %2, %3 : vector<4xi32>

```
"""
function SLessThanEqual(operand1::Value, operand2::Value; result::MLIRType, location=Location())
    results = MLIRType[result, ]
    operands = Value[operand1, operand2, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "spv.SLessThanEqual", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`SLessThan`

Result Type must be a scalar or vector of Boolean type.

 The type of Operand 1 and Operand 2  must be a scalar or vector of
integer type.  They must have the same component width, and they must
have the same number of components as Result Type.

 Results are computed per component.

<!-- End of AutoGen section -->
```
integer-scalar-vector-type ::= integer-type |
                             `vector<` integer-literal `x` integer-type `>`
sless-than-op ::= ssa-id `=` `spv.SLessThan` ssa-use, ssa-use
                             `:` integer-scalar-vector-type
```
#### Example:

```mlir
%4 = spv.SLessThan %0, %1 : i32
%5 = spv.SLessThan %2, %3 : vector<4xi32>

```
"""
function SLessThan(operand1::Value, operand2::Value; result::MLIRType, location=Location())
    results = MLIRType[result, ]
    operands = Value[operand1, operand2, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "spv.SLessThan", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`SMod`

Result Type must be a scalar or vector of integer type.

 The type of Operand 1 and Operand 2  must be a scalar or vector of
integer type.  They must have the same number of components as Result
Type. They must have the same component width as Result Type.

 Results are computed per component.  The resulting value is undefined
if Operand 2 is 0.  Otherwise, the result is the remainder r of Operand
1 divided by Operand 2 where if r ≠ 0, the sign of r is the same as the
sign of Operand 2.

<!-- End of AutoGen section -->
```
integer-scalar-vector-type ::= integer-type |
                             `vector<` integer-literal `x` integer-type `>`
smod-op ::= ssa-id `=` `spv.SMod` ssa-use, ssa-use
                       `:` integer-scalar-vector-type
```
#### Example:

```mlir
%4 = spv.SMod %0, %1 : i32
%5 = spv.SMod %2, %3 : vector<4xi32>

```
"""
function SMod(operand1::Value, operand2::Value; result=nothing::Union{Nothing, MLIRType}, location=Location())
    results = MLIRType[]
    operands = Value[operand1, operand2, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (result != nothing) && push!(results, result)
    
    create_operation(
        "spv.SMod", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`SNegate`

Result Type must be a scalar or vector of integer type.

Operand’s type  must be a scalar or vector of integer type.  It must
have the same number of components as Result Type.  The component width
must equal the component width in Result Type.

 Results are computed per component.

<!-- End of AutoGen section -->

#### Example:

```mlir
%1 = spv.SNegate %0 : i32
%3 = spv.SNegate %2 : vector<4xi32>
```
"""
function SNegate(operand::Value; result=nothing::Union{Nothing, MLIRType}, location=Location())
    results = MLIRType[]
    operands = Value[operand, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (result != nothing) && push!(results, result)
    
    create_operation(
        "spv.SNegate", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`SRem`

Result Type must be a scalar or vector of integer type.

 The type of Operand 1 and Operand 2  must be a scalar or vector of
integer type.  They must have the same number of components as Result
Type. They must have the same component width as Result Type.

 Results are computed per component.  The resulting value is undefined
if Operand 2 is 0.  Otherwise, the result is the remainder r of Operand
1 divided by Operand 2 where if r ≠ 0, the sign of r is the same as the
sign of Operand 1.

<!-- End of AutoGen section -->
```
integer-scalar-vector-type ::= integer-type |
                             `vector<` integer-literal `x` integer-type `>`
srem-op ::= ssa-id `=` `spv.SRem` ssa-use, ssa-use
                       `:` integer-scalar-vector-type
```
#### Example:

```mlir
%4 = spv.SRem %0, %1 : i32
%5 = spv.SRem %2, %3 : vector<4xi32>

```
"""
function SRem(operand1::Value, operand2::Value; result=nothing::Union{Nothing, MLIRType}, location=Location())
    results = MLIRType[]
    operands = Value[operand1, operand2, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (result != nothing) && push!(results, result)
    
    create_operation(
        "spv.SRem", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`Select`

Before version 1.4, Result Type must be a pointer, scalar, or vector.

 The types of Object 1 and Object 2 must be the same as Result Type.

Condition must be a scalar or vector of Boolean type.

If Condition is a scalar and true, the result is Object 1. If Condition
is a scalar and false, the result is Object 2.

If Condition is a vector, Result Type must be a vector with the same
number of components as Condition and the result is a mix of Object 1
and Object 2: When a component of Condition is true, the corresponding
component in the result is taken from Object 1, otherwise it is taken
from Object 2.

<!-- End of AutoGen section -->

```
scalar-type ::= integer-type | float-type | boolean-type
select-object-type ::= scalar-type
                       | `vector<` integer-literal `x` scalar-type `>`
                       | pointer-type
select-condition-type ::= boolean-type
                          | `vector<` integer-literal `x` boolean-type `>`
select-op ::= ssa-id `=` `spv.Select` ssa-use, ssa-use, ssa-use
              `:` select-condition-type `,` select-object-type
```

#### Example:

```mlir
%3 = spv.Select %0, %1, %2 : i1, f32
%3 = spv.Select %0, %1, %2 : i1, vector<3xi32>
%3 = spv.Select %0, %1, %2 : vector<3xi1>, vector<3xf32>
```
"""
function Select(condition::Value, true_value::Value, false_value::Value; result=nothing::Union{Nothing, MLIRType}, location=Location())
    results = MLIRType[]
    operands = Value[condition, true_value, false_value, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (result != nothing) && push!(results, result)
    
    create_operation(
        "spv.Select", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`mlir_selection`

SPIR-V can explicitly declare structured control-flow constructs using merge
instructions. These explicitly declare a header block before the control
flow diverges and a merge block where control flow subsequently converges.
These blocks delimit constructs that must nest, and can only be entered
and exited in structured ways. See \"2.11. Structured Control Flow\" of the
SPIR-V spec for more details.

Instead of having a `spv.SelectionMerge` op to directly model selection
merge instruction for indicating the merge target, we use regions to delimit
the boundary of the selection: the merge target is the next op following the
`spv.mlir.selection` op. This way it\'s easier to discover all blocks belonging to
the selection and it plays nicer with the MLIR system.

The `spv.mlir.selection` region should contain at least two blocks: one selection
header block, and one selection merge. The selection header block should be
the first block. The selection merge block should be the last block.
The merge block should only contain a `spv.mlir.merge` op.
"""
function mlir_selection(; selection_control, body::Region, location=Location())
    results = MLIRType[]
    operands = Value[]
    owned_regions = Region[body, ]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("selection_control", selection_control), ]
    
    create_operation(
        "spv.mlir.selection", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`ShiftLeftLogical`

Result Type must be a scalar or vector of integer type.

 The type of each Base and Shift must be a scalar or vector of integer
type. Base and Shift must have the same number of components.  The
number of components and bit width of the type of Base must be the same
as in Result Type.

Shift is treated as unsigned. The result is undefined if Shift is
greater than or equal to the bit width of the components of Base.

The number of components and bit width of Result Type must match those
Base type. All types must be integer types.

 Results are computed per component.

<!-- End of AutoGen section -->

```
integer-scalar-vector-type ::= integer-type |
                              `vector<` integer-literal `x` integer-type `>`
shift-left-logical-op ::= ssa-id `=` `spv.ShiftLeftLogical`
                                      ssa-use `,` ssa-use `:`
                                      integer-scalar-vector-type `,`
                                      integer-scalar-vector-type
```

#### Example:

```mlir
%2 = spv.ShiftLeftLogical %0, %1 : i32, i16
%5 = spv.ShiftLeftLogical %3, %4 : vector<3xi32>, vector<3xi16>
```
"""
function ShiftLeftLogical(operand1::Value, operand2::Value; result=nothing::Union{Nothing, MLIRType}, location=Location())
    results = MLIRType[]
    operands = Value[operand1, operand2, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (result != nothing) && push!(results, result)
    
    create_operation(
        "spv.ShiftLeftLogical", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`ShiftRightArithmetic`

Result Type must be a scalar or vector of integer type.

 The type of each Base and Shift must be a scalar or vector of integer
type. Base and Shift must have the same number of components.  The
number of components and bit width of the type of Base must be the same
as in Result Type.

Shift is treated as unsigned. The result is undefined if Shift is
greater than or equal to the bit width of the components of Base.

 Results are computed per component.

<!-- End of AutoGen section -->

```
integer-scalar-vector-type ::= integer-type |
                              `vector<` integer-literal `x` integer-type `>`
shift-right-arithmetic-op ::= ssa-id `=` `spv.ShiftRightArithmetic`
                                          ssa-use `,` ssa-use `:`
                                          integer-scalar-vector-type `,`
                                          integer-scalar-vector-type
```

#### Example:

```mlir
%2 = spv.ShiftRightArithmetic %0, %1 : i32, i16
%5 = spv.ShiftRightArithmetic %3, %4 : vector<3xi32>, vector<3xi16>
```
"""
function ShiftRightArithmetic(operand1::Value, operand2::Value; result=nothing::Union{Nothing, MLIRType}, location=Location())
    results = MLIRType[]
    operands = Value[operand1, operand2, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (result != nothing) && push!(results, result)
    
    create_operation(
        "spv.ShiftRightArithmetic", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`ShiftRightLogical`

Result Type must be a scalar or vector of integer type.

 The type of each Base and Shift must be a scalar or vector of integer
type. Base and Shift must have the same number of components.  The
number of components and bit width of the type of Base must be the same
as in Result Type.

Shift is consumed as an unsigned integer. The result is undefined if
Shift is greater than or equal to the bit width of the components of
Base.

 Results are computed per component.

<!-- End of AutoGen section -->

```
integer-scalar-vector-type ::= integer-type |
                              `vector<` integer-literal `x` integer-type `>`
shift-right-logical-op ::= ssa-id `=` `spv.ShiftRightLogical`
                                       ssa-use `,` ssa-use `:`
                                       integer-scalar-vector-type `,`
                                       integer-scalar-vector-type
```

#### Example:

```mlir
%2 = spv.ShiftRightLogical %0, %1 : i32, i16
%5 = spv.ShiftRightLogical %3, %4 : vector<3xi32>, vector<3xi16>
```
"""
function ShiftRightLogical(operand1::Value, operand2::Value; result=nothing::Union{Nothing, MLIRType}, location=Location())
    results = MLIRType[]
    operands = Value[operand1, operand2, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (result != nothing) && push!(results, result)
    
    create_operation(
        "spv.ShiftRightLogical", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`SpecConstantComposite`

This op declares a SPIR-V composite specialization constant. This covers
the `OpSpecConstantComposite` SPIR-V instruction. Scalar constants are
covered by `spv.SpecConstant`.

A constituent of a spec constant composite can be:
- A symbol referring of another spec constant.
- The SSA ID of a non-specialization constant (i.e. defined through
  `spv.SpecConstant`).
- The SSA ID of a `spv.Undef`.

```
spv-spec-constant-composite-op ::= `spv.SpecConstantComposite` symbol-ref-id ` (`
                                   symbol-ref-id (`, ` symbol-ref-id)*
                                   `) :` composite-type
```

 where `composite-type` is some non-scalar type that can be represented in the `spv`
 dialect: `spv.struct`, `spv.array`, or `vector`.

 #### Example:

 ```mlir
 spv.SpecConstant @sc1 = 1   : i32
 spv.SpecConstant @sc2 = 2.5 : f32
 spv.SpecConstant @sc3 = 3.5 : f32
 spv.SpecConstantComposite @scc (@sc1, @sc2, @sc3) : !spv.struct<i32, f32, f32>
 ```

TODO Add support for constituents that are:
- regular constants.
- undef.
- spec constant composite.
"""
function SpecConstantComposite(; type, sym_name, constituents, location=Location())
    results = MLIRType[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("type", type), namedattribute("sym_name", sym_name), namedattribute("constituents", constituents), ]
    
    create_operation(
        "spv.SpecConstantComposite", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`SpecConstant`

This op declares a SPIR-V scalar specialization constant. SPIR-V has
multiple constant instructions covering different scalar types:

* `OpSpecConstantTrue` and `OpSpecConstantFalse` for boolean constants
* `OpSpecConstant` for scalar constants

Similar as `spv.Constant`, this op represents all of the above cases.
`OpSpecConstantComposite` and `OpSpecConstantOp` are modelled with
separate ops.

<!-- End of AutoGen section -->

```
spv-spec-constant-op ::= `spv.SpecConstant` symbol-ref-id
                         `spec_id(` integer `)`
                         `=` attribute-value (`:` spirv-type)?
```

where `spec_id` specifies the SPIR-V SpecId decoration associated with
the op.

#### Example:

```mlir
spv.SpecConstant @spec_const1 = true
spv.SpecConstant @spec_const2 spec_id(5) = 42 : i32
```
"""
function SpecConstant(; sym_name, default_value, location=Location())
    results = MLIRType[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("sym_name", sym_name), namedattribute("default_value", default_value), ]
    
    create_operation(
        "spv.SpecConstant", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`SpecConstantOperation`

This op declares a SPIR-V specialization constant that results from
doing an operation on other constants (specialization or otherwise).

In the `spv` dialect, this op is modelled as follows:

```
spv-spec-constant-operation-op ::= `spv.SpecConstantOperation` `wraps`
                                     generic-spirv-op `:` function-type
```

In particular, an `spv.SpecConstantOperation` contains exactly one
region. In turn, that region, contains exactly 2 instructions:
- One of SPIR-V\'s instructions that are allowed within an
OpSpecConstantOp.
- An `spv.mlir.yield` instruction as the terminator.

The following SPIR-V instructions are valid:
- OpSConvert,
- OpUConvert,
- OpFConvert,
- OpSNegate,
- OpNot,
- OpIAdd,
- OpISub,
- OpIMul,
- OpUDiv,
- OpSDiv,
- OpUMod,
- OpSRem,
- OpSMod
- OpShiftRightLogical,
- OpShiftRightArithmetic,
- OpShiftLeftLogical
- OpBitwiseOr,
- OpBitwiseXor,
- OpBitwiseAnd
- OpVectorShuffle,
- OpCompositeExtract,
- OpCompositeInsert
- OpLogicalOr,
- OpLogicalAnd,
- OpLogicalNot,
- OpLogicalEqual,
- OpLogicalNotEqual
- OpSelect
- OpIEqual,
- OpINotEqual
- OpULessThan,
- OpSLessThan
- OpUGreaterThan,
- OpSGreaterThan
- OpULessThanEqual,
- OpSLessThanEqual
- OpUGreaterThanEqual,
- OpSGreaterThanEqual

TODO Add capability-specific ops when supported.

#### Example:
```mlir
%0 = spv.Constant 1: i32
%1 = spv.Constant 1: i32

%2 = spv.SpecConstantOperation wraps \"spv.IAdd\"(%0, %1) : (i32, i32) -> i32
```
"""
function SpecConstantOperation(; result::MLIRType, body::Region, location=Location())
    results = MLIRType[result, ]
    operands = Value[]
    owned_regions = Region[body, ]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "spv.SpecConstantOperation", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`Store`

Pointer is the pointer to store through.  Its type must be an
OpTypePointer whose Type operand is the same as the type of Object.

Object is the object to store.

If present, any Memory Operands must begin with a memory operand
literal. If not present, it is the same as specifying the memory operand
None.

<!-- End of AutoGen section -->

```
store-op ::= `spv.Store ` storage-class ssa-use `, ` ssa-use `, `
              (`[` memory-access `]`)? `:` spirv-element-type
```

#### Example:

```mlir
%0 = spv.Variable : !spv.ptr<f32, Function>
%1 = spv.FMul ... : f32
spv.Store \"Function\" %0, %1 : f32
spv.Store \"Function\" %0, %1 [\"Volatile\"] : f32
spv.Store \"Function\" %0, %1 [\"Aligned\", 4] : f32
```
"""
function Store(ptr::Value, value::Value; memory_access=nothing, alignment=nothing, location=Location())
    results = MLIRType[]
    operands = Value[ptr, value, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (memory_access != nothing) && push!(attributes, namedattribute("memory_access", memory_access))
    (alignment != nothing) && push!(attributes, namedattribute("alignment", alignment))
    
    create_operation(
        "spv.Store", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`SubgroupBallotKHR`

Computes a bitfield value combining the Predicate value from all invocations
in the current Subgroup that execute the same dynamic instance of this
instruction. The bit is set to one if the corresponding invocation is active
and the predicate is evaluated to true; otherwise, it is set to zero.

Predicate must be a Boolean type.

Result Type must be a 4 component vector of 32 bit integer types.

Result is a set of bitfields where the first invocation is represented in bit
0 of the first vector component and the last (up to SubgroupSize) is the
higher bit number of the last bitmask needed to represent all bits of the
subgroup invocations.

<!-- End of AutoGen section -->

```
subgroup-ballot-op ::= ssa-id `=` `spv.SubgroupBallotKHR`
                            ssa-use `:` `vector` `<` 4 `x` `i32` `>`
```

#### Example:

```mlir
%0 = spv.SubgroupBallotKHR %predicate : vector<4xi32>
```
"""
function SubgroupBallotKHR(predicate::Value; result::MLIRType, location=Location())
    results = MLIRType[result, ]
    operands = Value[predicate, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "spv.SubgroupBallotKHR", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`SubgroupBlockReadINTEL`

Reads one or more components of Result data for each invocation in the
subgroup from the specified Ptr as a block operation.

The data is read strided, so the first value read is:
Ptr[ SubgroupLocalInvocationId ]

and the second value read is:
Ptr[ SubgroupLocalInvocationId + SubgroupMaxSize ]
etc.

Result Type may be a scalar or vector type, and its component type must be
equal to the type pointed to by Ptr.

The type of Ptr must be a pointer type, and must point to a scalar type.

<!-- End of AutoGen section -->

```
subgroup-block-read-INTEL-op ::= ssa-id `=` `spv.SubgroupBlockReadINTEL`
                            storage-class ssa_use `:` spirv-element-type
```mlir

#### Example:

```
%0 = spv.SubgroupBlockReadINTEL \"StorageBuffer\" %ptr : i32
```
"""
function SubgroupBlockReadINTEL(ptr::Value; value::MLIRType, location=Location())
    results = MLIRType[value, ]
    operands = Value[ptr, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "spv.SubgroupBlockReadINTEL", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`SubgroupBlockWriteINTEL`

Writes one or more components of Data for each invocation in the subgroup
from the specified Ptr as a block operation.

The data is written strided, so the first value is written to:
Ptr[ SubgroupLocalInvocationId ]

and the second value written is:
Ptr[ SubgroupLocalInvocationId + SubgroupMaxSize ]
etc.

The type of Ptr must be a pointer type, and must point to a scalar type.

The component type of Data must be equal to the type pointed to by Ptr.

<!-- End of AutoGen section -->

```
subgroup-block-write-INTEL-op ::= ssa-id `=` `spv.SubgroupBlockWriteINTEL`
                  storage-class ssa_use `,` ssa-use `:` spirv-element-type
```mlir

#### Example:

```
spv.SubgroupBlockWriteINTEL \"StorageBuffer\" %ptr, %value : i32
```
"""
function SubgroupBlockWriteINTEL(ptr::Value, value::Value; location=Location())
    results = MLIRType[]
    operands = Value[ptr, value, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "spv.SubgroupBlockWriteINTEL", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`Transpose`

Result Type must be an OpTypeMatrix.

Matrix must be an object of type OpTypeMatrix. The number of columns and
the column size of Matrix must be the reverse of those in Result Type.
The types of the scalar components in Matrix and Result Type must be the
same.

Matrix must have of type of OpTypeMatrix.

<!-- End of AutoGen section -->

```
transpose-op ::= ssa-id `=` `spv.Transpose` ssa-use `:` matrix-type `->`
matrix-type

```mlir

#### Example:

```
%0 = spv.Transpose %matrix: !spv.matrix<2 x vector<3xf32>> ->
!spv.matrix<3 x vector<2xf32>>

```
"""
function Transpose(matrix::Value; result::MLIRType, location=Location())
    results = MLIRType[result, ]
    operands = Value[matrix, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "spv.Transpose", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`UConvert`

Result Type must be a scalar or vector of integer type, whose Signedness
operand is 0.

Unsigned Value must be a scalar or vector of integer type.  It must have
the same number of components as Result Type.  The component width
cannot equal the component width in Result Type.

 Results are computed per component.

<!-- End of AutoGen section -->

```
u-convert-op ::= ssa-id `=` `spv.UConvertOp` ssa-use
             `:` operand-type `to` result-type
```

#### Example:

```mlir
%1 = spv.UConvertOp %0 : i32 to i64
%3 = spv.UConvertOp %2 : vector<3xi32> to vector<3xi64>
```
"""
function UConvert(operand::Value; result::MLIRType, location=Location())
    results = MLIRType[result, ]
    operands = Value[operand, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "spv.UConvert", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`UDiv`

Result Type must be a scalar or vector of integer type, whose Signedness
operand is 0.

 The types of Operand 1 and Operand 2 both must be the same as Result
Type.

 Results are computed per component.  The resulting value is undefined
if Operand 2 is 0.

<!-- End of AutoGen section -->
```
integer-scalar-vector-type ::= integer-type |
                             `vector<` integer-literal `x` integer-type `>`
udiv-op ::= ssa-id `=` `spv.UDiv` ssa-use, ssa-use
                       `:` integer-scalar-vector-type
```
#### Example:

```mlir
%4 = spv.UDiv %0, %1 : i32
%5 = spv.UDiv %2, %3 : vector<4xi32>

```
"""
function UDiv(operand1::Value, operand2::Value; result=nothing::Union{Nothing, MLIRType}, location=Location())
    results = MLIRType[]
    operands = Value[operand1, operand2, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (result != nothing) && push!(results, result)
    
    create_operation(
        "spv.UDiv", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`UGreaterThanEqual`

Result Type must be a scalar or vector of Boolean type.

 The type of Operand 1 and Operand 2  must be a scalar or vector of
integer type.  They must have the same component width, and they must
have the same number of components as Result Type.

 Results are computed per component.

<!-- End of AutoGen section -->
```
integer-scalar-vector-type ::= integer-type |
                             `vector<` integer-literal `x` integer-type `>`
ugreater-than-equal-op ::= ssa-id `=` `spv.UGreaterThanEqual` ssa-use, ssa-use
                                      `:` integer-scalar-vector-type
```
#### Example:

```mlir
%4 = spv.UGreaterThanEqual %0, %1 : i32
%5 = spv.UGreaterThanEqual %2, %3 : vector<4xi32>

```
"""
function UGreaterThanEqual(operand1::Value, operand2::Value; result::MLIRType, location=Location())
    results = MLIRType[result, ]
    operands = Value[operand1, operand2, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "spv.UGreaterThanEqual", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`UGreaterThan`

Result Type must be a scalar or vector of Boolean type.

 The type of Operand 1 and Operand 2  must be a scalar or vector of
integer type.  They must have the same component width, and they must
have the same number of components as Result Type.

 Results are computed per component.

<!-- End of AutoGen section -->
```
integer-scalar-vector-type ::= integer-type |
                             `vector<` integer-literal `x` integer-type `>`
ugreater-than-op ::= ssa-id `=` `spv.UGreaterThan` ssa-use, ssa-use
                                `:` integer-scalar-vector-type
```
#### Example:

```mlir
%4 = spv.UGreaterThan %0, %1 : i32
%5 = spv.UGreaterThan %2, %3 : vector<4xi32>

```
"""
function UGreaterThan(operand1::Value, operand2::Value; result::MLIRType, location=Location())
    results = MLIRType[result, ]
    operands = Value[operand1, operand2, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "spv.UGreaterThan", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`ULessThanEqual`

Result Type must be a scalar or vector of Boolean type.

 The type of Operand 1 and Operand 2  must be a scalar or vector of
integer type.  They must have the same component width, and they must
have the same number of components as Result Type.

 Results are computed per component.

<!-- End of AutoGen section -->
```
integer-scalar-vector-type ::= integer-type |
                             `vector<` integer-literal `x` integer-type `>`
uless-than-equal-op ::= ssa-id `=` `spv.ULessThanEqual` ssa-use, ssa-use
                                   `:` integer-scalar-vector-type
```
#### Example:

```mlir
%4 = spv.ULessThanEqual %0, %1 : i32
%5 = spv.ULessThanEqual %2, %3 : vector<4xi32>

```
"""
function ULessThanEqual(operand1::Value, operand2::Value; result::MLIRType, location=Location())
    results = MLIRType[result, ]
    operands = Value[operand1, operand2, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "spv.ULessThanEqual", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`ULessThan`

Result Type must be a scalar or vector of Boolean type.

 The type of Operand 1 and Operand 2  must be a scalar or vector of
integer type.  They must have the same component width, and they must
have the same number of components as Result Type.

 Results are computed per component.

<!-- End of AutoGen section -->
```
integer-scalar-vector-type ::= integer-type |
                             `vector<` integer-literal `x` integer-type `>`
uless-than-op ::= ssa-id `=` `spv.ULessThan` ssa-use, ssa-use
                             `:` integer-scalar-vector-type
```
#### Example:

```mlir
%4 = spv.ULessThan %0, %1 : i32
%5 = spv.ULessThan %2, %3 : vector<4xi32>

```
"""
function ULessThan(operand1::Value, operand2::Value; result::MLIRType, location=Location())
    results = MLIRType[result, ]
    operands = Value[operand1, operand2, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "spv.ULessThan", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`UMod`

Result Type must be a scalar or vector of integer type, whose Signedness
operand is 0.

 The types of Operand 1 and Operand 2 both must be the same as Result
Type.

 Results are computed per component.  The resulting value is undefined
if Operand 2 is 0.

<!-- End of AutoGen section -->
```
integer-scalar-vector-type ::= integer-type |
                             `vector<` integer-literal `x` integer-type `>`
umod-op ::= ssa-id `=` `spv.UMod` ssa-use, ssa-use
                       `:` integer-scalar-vector-type
```
#### Example:

```mlir
%4 = spv.UMod %0, %1 : i32
%5 = spv.UMod %2, %3 : vector<4xi32>

```
"""
function UMod(operand1::Value, operand2::Value; result=nothing::Union{Nothing, MLIRType}, location=Location())
    results = MLIRType[]
    operands = Value[operand1, operand2, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (result != nothing) && push!(results, result)
    
    create_operation(
        "spv.UMod", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`Undef`

Result Type is the type of object to make.

Each consumption of Result <id> yields an arbitrary, possibly different
bit pattern or abstract value resulting in possibly different concrete,
abstract, or opaque values.

<!-- End of AutoGen section -->

```
undef-op ::= `spv.Undef` `:` spirv-type
```

#### Example:

```mlir
%0 = spv.Undef : f32
%1 = spv.Undef : !spv.struct<!spv.array<4 x vector<4xi32>>>
```
"""
function Undef(; result::MLIRType, location=Location())
    results = MLIRType[result, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "spv.Undef", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`Unordered`

Result Type must be a scalar or vector of Boolean type.

x must be a scalar or vector of floating-point type.  It must have the
same number of components as Result Type.

y must have the same type as x.

 Results are computed per component.

<!-- End of AutoGen section -->

```
float-scalar-vector-type ::= float-type |
                             `vector<` integer-literal `x` float-type `>`
unordered-op ::= ssa-id `=` `spv.Unordered` ssa-use, ssa-use
```mlir

#### Example:

```
%4 = spv.Unordered %0, %1 : f32
%5 = spv.Unordered %2, %3 : vector<4xf32>
```
"""
function Unordered(operand1::Value, operand2::Value; result::MLIRType, location=Location())
    results = MLIRType[result, ]
    operands = Value[operand1, operand2, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "spv.Unordered", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`Unreachable`

This instruction must be the last instruction in a block.

<!-- End of AutoGen section -->

```
unreachable-op ::= `spv.Unreachable`
```
"""
function Unreachable(; location=Location())
    results = MLIRType[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "spv.Unreachable", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`Variable`

Result Type must be an OpTypePointer. Its Type operand is the type of
object in memory.

Storage Class is the Storage Class of the memory holding the object.
Since the op is used to model function-level variables, the storage class
must be the `Function` Storage Class.

Initializer is optional. If Initializer is present, it will be the
initial value of the variable’s memory content. Initializer must be an
<id> from a constant instruction or a global (module scope) OpVariable
instruction. Initializer must have the same type as the type pointed to
by Result Type.

<!-- End of AutoGen section -->

```
variable-op ::= ssa-id `=` `spv.Variable` (`init(` ssa-use `)`)?
                attribute-dict? `:` spirv-pointer-type
```

where `init` specifies initializer.

#### Example:

```mlir
%0 = spv.Constant ...

%1 = spv.Variable : !spv.ptr<f32, Function>
%2 = spv.Variable init(%0): !spv.ptr<f32, Function>
```
"""
function Variable(initializer=nothing::Union{Nothing, Value}; pointer::MLIRType, storage_class, location=Location())
    results = MLIRType[pointer, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("storage_class", storage_class), ]
    (initializer != nothing) && push!(operands, initializer)
    
    create_operation(
        "spv.Variable", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`VectorExtractDynamic`

Result Type must be a scalar type.

Vector must have a type OpTypeVector whose Component Type is Result
Type.

Index must be a scalar integer. It is interpreted as a 0-based index of
which component of Vector to extract.

Behavior is undefined if Index\'s value is less than zero or greater than
or equal to the number of components in Vector.

<!-- End of AutoGen section -->

#### Example:

```
%2 = spv.VectorExtractDynamic %0[%1] : vector<8xf32>, i32
```
"""
function VectorExtractDynamic(vector::Value, index::Value; result::MLIRType, location=Location())
    results = MLIRType[result, ]
    operands = Value[vector, index, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "spv.VectorExtractDynamic", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`VectorInsertDynamic`

Result Type must be an OpTypeVector.

Vector must have the same type as Result Type and is the vector that the
non-written components are copied from.

Component is the value supplied for the component selected by Index. It
must have the same type as the type of components in Result Type.

Index must be a scalar integer. It is interpreted as a 0-based index of
which component to modify.

Behavior is undefined if Index\'s value is less than zero or greater than
or equal to the number of components in Vector.

<!-- End of AutoGen section -->

```
scalar-type ::= integer-type | float-type | boolean-type
vector-insert-dynamic-op ::= `spv.VectorInsertDynamic ` ssa-use `,`
                              ssa-use `[` ssa-use `]`
                              `:` `vector<` integer-literal `x` scalar-type `>` `,`
                              integer-type
```mlir

#### Example:

```
%scalar = ... : f32
%2 = spv.VectorInsertDynamic %scalar %0[%1] : f32, vector<8xf32>, i32
```
"""
function VectorInsertDynamic(vector::Value, component::Value, index::Value; result=nothing::Union{Nothing, MLIRType}, location=Location())
    results = MLIRType[]
    operands = Value[vector, component, index, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (result != nothing) && push!(results, result)
    
    create_operation(
        "spv.VectorInsertDynamic", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`VectorShuffle`

Result Type must be an OpTypeVector. The number of components in Result
Type must be the same as the number of Component operands.

Vector 1 and Vector 2 must both have vector types, with the same
Component Type as Result Type. They do not have to have the same number
of components as Result Type or with each other. They are logically
concatenated, forming a single vector with Vector 1’s components
appearing before Vector 2’s. The components of this logical vector are
logically numbered with a single consecutive set of numbers from 0 to N
- 1, where N is the total number of components.

Components are these logical numbers (see above), selecting which of the
logically numbered components form the result. Each component is an
unsigned 32-bit integer.  They can select the components in any order
and can repeat components. The first component of the result is selected
by the first Component operand,  the second component of the result is
selected by the second Component operand, etc. A Component literal may
also be FFFFFFFF, which means the corresponding result component has no
source and is undefined. All Component literals must either be FFFFFFFF
or in [0, N - 1] (inclusive).

Note: A vector “swizzle” can be done by using the vector for both Vector
operands, or using an OpUndef for one of the Vector operands.

<!-- End of AutoGen section -->

#### Example:

```mlir
%0 = spv.VectorShuffle [1: i32, 3: i32, 5: i32]
                       %vector1: vector<4xf32>, %vector2: vector<2xf32>
                    -> vector<3xf32>
```
"""
function VectorShuffle(vector1::Value, vector2::Value; result::MLIRType, components, location=Location())
    results = MLIRType[result, ]
    operands = Value[vector1, vector2, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("components", components), ]
    
    create_operation(
        "spv.VectorShuffle", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`VectorTimesScalar`

Result Type must be a vector of floating-point type.

 The type of Vector must be the same as Result Type. Each component of
Vector is multiplied by Scalar.

Scalar must have the same type as the Component Type in Result Type.

<!-- End of AutoGen section -->

#### Example:

```mlir
%0 = spv.VectorTimesScalar %vector, %scalar : vector<4xf32>
```
"""
function VectorTimesScalar(vector::Value, scalar::Value; result::MLIRType, location=Location())
    results = MLIRType[result, ]
    operands = Value[vector, scalar, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "spv.VectorTimesScalar", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`mlir_yield`

This op is a special terminator whose only purpose is to terminate
an `spv.SpecConstantOperation`\'s enclosed region. It accepts a
single operand produced by the preceeding (and only other) instruction
in its parent block (see SPV_SpecConstantOperation for further
details). This op has no corresponding SPIR-V instruction.

```
spv.mlir.yield ::= `spv.mlir.yield` ssa-id : spirv-type
```

#### Example:
```mlir
%0 = ... (some op supported by SPIR-V OpSpecConstantOp)
spv.mlir.yield %0
```
"""
function mlir_yield(operand::Value; location=Location())
    results = MLIRType[]
    operands = Value[operand, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "spv.mlir.yield", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

end # spv
