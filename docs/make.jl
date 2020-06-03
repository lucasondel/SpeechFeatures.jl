using Documenter

push!(LOAD_PATH, "../src/")
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

deploydocs(
    repo = "github.com/BUTSpeechFIT/SpeechFeatures.git",
)

