using MultiIndexing
using Documenter

DocMeta.setdocmeta!(MultiIndexing, :DocTestSetup, :(using MultiIndexing); recursive = true)

makedocs(;
    modules = [MultiIndexing],
    authors = "Daniel Sharp",
    repo = "https://github.com/dannys4/MultiIndexing.jl/blob/{commit}{path}#{line}",
    sitename = "MultiIndexing.jl",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://dannys4.github.io/MultiIndexing.jl",
        edit_link = "main",
        assets = String[]
    ),
    pages = [
        "Home" => "index.md"
    ]
)

deploydocs(;
    repo = "github.com/dannys4/MultiIndexing.jl"
)