# SpeechFeatures

A Julia package to extract speech features.

See the full documentation here: [![](https://img.shields.io/badge/docs-dev-blue.svg)](https://butspeechfit.github.io/SpeechFeatures/dev)

## Installation

```
    julia> Pkg.add("https://github.com/BUTSpeechFIT/SpeechFeatures")
```

## Extracting MFCC features

```
    # x = ... # signal stored as a Vector
    julia> mfcc = MFCC(srate = 16000, nfilters = 26)
    julia> features = x |> mfcc
```

For a more details examples see the [example notebook](https://github.com/BUTSpeechFIT/SpeechFeatures/blob/master/examples/demo.ipynb).

