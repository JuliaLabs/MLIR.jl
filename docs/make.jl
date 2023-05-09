pushfirst!(LOAD_PATH, joinpath(@__DIR__, "..")) # add MLIR to environment stack

using MLIR
using Documenter

DocMeta.setdocmeta!(MLIR, :DocTestSetup, :(using MLIR); recursive=true)

# Generate examples

using Literate

const EXAMPLES_DIR = joinpath(@__DIR__, "..", "examples")
const OUTPUT_DIR = joinpath(@__DIR__, "src/generated")

examples = Pair{String,String}[
]

for (_, name) in examples
    example_filepath = joinpath(EXAMPLES_DIR, string(name, ".jl"))
    Literate.markdown(example_filepath, OUTPUT_DIR, documenter = true)
end

examples = [title => joinpath("generated", string(name, ".md")) for (title, name) in examples]

makedocs(;
    modules=[MLIR],
    authors="Valentin Churavy <vchuravy@mit.edu>",
    repo="https://github.com/JuliaLabs/MLIR.jl/blob/{commit}{path}#{line}",
    sitename="MLIR.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://julia.mit.edu/MLIR/",
        assets = [
            asset("https://plausible.io/js/plausible.js",
                    class=:js,
                    attributes=Dict(Symbol("data-domain") => "julia.mit.edu", :defer => "")
                )
	    ],
    ),
    pages = [
        "Home" => "index.md",
        "Examples" => examples,
        "API" => "api.md",
    ],
    doctest = true,
    strict = true,
)

deploydocs(;
    repo="github.com/JuliaLabs/MLIR.jl",
    devbranch = "main",
    push_preview = true,
)