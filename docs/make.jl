using SpectralLines
using Documenter

DocMeta.setdocmeta!(SpectralLines, :DocTestSetup, :(using SpectralLines); recursive=true)

makedocs(;
    modules=[SpectralLines],
    authors="Jan Kuhfeld <jankuhfeld@plasma-matters.nl> and contributors",
    sitename="SpectralLines.jl",
    format=Documenter.HTML(;
        canonical="https://optical-diagnostics.github.io/SpectralLines.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/Optical-Diagnostics/SpectralLines.jl",
    devbranch="main",
)
