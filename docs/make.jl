push!(LOAD_PATH, "../src/")

using Documenter
using SpeechFeatures

DocMeta.setdocmeta!(SpeechFeatures, :DocTestSetup,
                    :(using SpeechFeatures), recursive = true)

makedocs(
    sitename="SpeechFeatures",
    format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true"),
    pages = [
        "Home" => "index.md",
        "Extracting Features" => "feaextract.md"
    ]
)

deploydocs(
    repo = "github.com/lucasondel/SpeechFeatures.jl.git",
)

