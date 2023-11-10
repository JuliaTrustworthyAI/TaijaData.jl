using TaijaData
using Documenter

DocMeta.setdocmeta!(TaijaData, :DocTestSetup, :(using TaijaData); recursive=true)

makedocs(;
    modules=[TaijaData],
    authors="Patrick Altmeyer",
    repo="https://github.com/JuliaTrustworthyAI/TaijaData.jl/blob/{commit}{path}#{line}",
    sitename="TaijaData.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://JuliaTrustworthyAI.github.io/TaijaData.jl",
        edit_link="master",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/JuliaTrustworthyAI/TaijaData.jl",
    devbranch="master",
)
