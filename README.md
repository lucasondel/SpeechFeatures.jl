# SpeechFeatures

SpeechFeatures is a Julia package for extracting acoustic features
for speech technologies.

| **Documentation**  | **Test Status**   |
|:------------------:|:-----------------:|
| [![](https://img.shields.io/badge/docs-stable-blue.svg)](https://lucasondel.github.io/SpeechFeatures/stable) [![](https://img.shields.io/badge/docs-dev-blue.svg)](https://lucasondel.github.io/SpeechFeatures/dev) | ![](https://github.com/lucasondel/SpeechFeatures/workflows/Test/badge.svg) |

See the [changelog file](CHANGELOG.md) to check what's new since the
last release.

## Installation

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

Have a look at the [documentation](https://lucasondel.github.io/SpeechFeatures/stable/)
to get started.

## Authors

Lucas Ondel 2020

