using Documenter

using Pkg
Pkg.activate("../")
using SpeechFeatures


makedocs(
    sitename="SpeechFeatures",
    format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true"),
    pages = [
        "Home" => "index.md",
        "Manual" => Any[
            "Installation" => "install.md",
            "Features extraction" => "feaextract.md"
        ],
        "API" => "api.md"
    ]
)
