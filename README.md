# SpeechFeatures.jl

*SpeechFeatures* is a Julia package for extracting acoustic features
for speech technologies.

| **Test Status**   |
|:-----------------:|
| ![](https://github.com/lucasondel/SpeechFeatures.jl/workflows/Test/badge.svg) |

See the [changelog file](CHANGELOG.md) to check what's new since the
last release.

## Installation

The package can be installed with the Julia package manager. From the
Julia REPL, type ] to enter the Pkg REPL mode and run:

```
pkg> add SpeechFeatures
```

## Quick start

To get the [MFCC](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum)
features:

```julia
using SpeechFeatures

# x = ... extracted signal
# fs = ... sampling frequency

S, fftlen = stft(x; srate=fs) # Complex short-term spectrum.
fbank = filterbank(26; fftlen=fftlen)
mS = fbank * abs.(S) # Magnitude of the Mel-spectrum.
MFCCs = mfcc(mS; nceps=13) # Standard MFCCs.
MFCCs_Δ_ΔΔ = add_deltas(MFCCs; order=2) # MFCCs + 1st and 2nd order derivatives.
```

Have a look at the [examples](https://github.com/lucasondel/SpeechFeatures.jl/tree/master/examples)
to get started.

## Author

[Lucas Ondel](https://lucasondel.github.io), [LISN](https://www.lisn.upsaclay.fr/) 2021

