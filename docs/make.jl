using Pkg

Pkg.develop(PackageSpec(path = dirname(@__DIR__)))

using Documenter
using NonLocalMeans

makedocs(
    modules = [NonLocalMeans],
    sitename = "NonLocalMeans.jl",
    pages = [
        "Home" => "index.md",
        "API" => "api.md",
    ],
)

deploydocs(
    repo = "github.com/FerreolS/NonLocalMeans.jl.git",
    devbranch = "master",
)
