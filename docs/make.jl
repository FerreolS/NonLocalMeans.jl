using Pkg

Pkg.develop(PackageSpec(path = dirname(@__DIR__)))

using Documenter
using NonlocalMeans

makedocs(
    modules = [NonlocalMeans],
    sitename = "NonlocalMeans.jl",
    pages = [
        "Home" => "index.md",
        "API" => "api.md",
    ],
)

deploydocs(
    repo = "github.com/FerreolS/NonlocalMeans.jl.git",
    devbranch = "master",
)
