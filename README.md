# SpeechFeatures.jl

SpeechFeatures.jl is a Julia package for extracting acoustic features
for speech technologies.

| **Documentation**  | **Test Status**   |
|:------------------:|:-----------------:|
| [![](https://img.shields.io/badge/docs-stable-blue.svg)](https://lucasondel.github.io/SpeechFeatures.jl/stable) [![](https://img.shields.io/badge/docs-dev-blue.svg)](https://lucasondel.github.io/SpeechFeatures.jl/dev) | ![](https://github.com/lucasondel/SpeechFeatures.jl/workflows/Test/badge.svg) |

See the [changelog file](CHANGELOG.md) to check what's new since the
last release.

## Installation

The package can be installed with the Julia package manager. From the
Julia REPL, type ] to enter the Pkg REPL mode and run:

```
pkg> add SpeechFeatures
```

## Example

```julia
julia> # x = ... extracted signal
julia> lms = LogMelSpectrum()
julia> x |> lms
```

![](docs/src/images/lms.svg)

Have a look at the [documentation](https://lucasondel.github.io/SpeechFeatures.jl/stable/) or the [example Jupyter notebook](demo.ipynb)
to get started.

## Authors

Lucas Ondel 2021

