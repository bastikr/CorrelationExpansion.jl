using Documenter, CorrelationExpansion

pages = [
    "index.md",
    "api.md"
]

makedocs(
    modules = [CorrelationExpansion],
    checkdocs = :exports,
    format = :html,
    sitename = "CorrelationExpansion.jl",
    pages = pages
    )

deploydocs(
    repo = "github.com/bastikr/CorrelationExpansion.jl.git",
    target = "build",
    deps = nothing,
    make = nothing,
    julia = "0.6"
)
