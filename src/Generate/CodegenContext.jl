abstract type AbstractCodegenContext end

for method in (
    :generate_return,
    :generate_goto,
    :generate_gotoifnot,
    :generate_function
)
    @eval begin
        $method(cg::T) where {T<:AbstractCodegenContext} = error("$method not implemented for type \$T")
    end
end

mutable struct CodegenContext{T} <: AbstractCodegenContext
    CodegenContext{T}() where T = new{T}()

    function (cg::CodegenContext)(f, types)
        ir, ret = Core.Compiler.code_ircode(f, types, interp=MLIRInterpreter()) |> only

        blocks = [
            prepare_block(ir, bb)
            for bb in ir.cfg.blocks
        ]

        argtypes = ir.argtypes[begin+1:end] # the first argument is the type of the function itself, we don't need this.
        args = map(enumerate(argtypes)) do i, argtype
            temp = []
            for t in unpack(argtype)
                arg = IR.push_argument!(entryblock, IR.Type(t))
                push!(temp, t(arg))
            end
            reinterpret(argtype, Tuple(temp))
        end

        builder = create_builder!(ir, blocks, cg)

        return builder(args)
    end
end

abstract type Default end
CodegenContext() = CodegenContext{Default}()
