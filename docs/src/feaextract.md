# Features extraction

## Loading an audio file

To extract any type of speech features you will need the audio signal
stored in an `Array`-like object and the sampling rate in Hertz.
SpeechFeatures does not provide a way to load these two elements from
audio files directly but there are several Julia packages to do this.
In this tutorial, we will use [WAV.jl](https://github.com/dancasimiro/WAV.jl).
For the rest of the tutorial, we assumed that you have installed the
WAV.jl package in your Julia environment.

First of all, as an example, we download an audio file from the
[TIMIT](https://catalog.ldc.upenn.edu/LDC93S1) corpus. In the Julia
REPL type:

```juliashowcase
julia> run(`wget https://catalog.ldc.upenn.edu/desc/addenda/LDC93S1.wav`)
```

Now, we load the audio waveform:
```
julia> using WAV
julia> channels, srate = wavread("LDC93S1.wav", format = "double")
```
Where `channels` is a `N`x`C` matrix where `N` is the length of the audio in
samples and `C` is the number of channels. Since TIMIT is mono recorded
it has only one channel. `format = "double"` indicates that the
signals in `channels` will be encoded with double precision and each
sample of the signal will be between `1.0` and `-1.0`.

!!! warning
    The `wavread` function also accepts `format = "native"` which will
    return the data in the format it is stored in the WAV file. We
    discourage its use as extracting the features from integer or
    floating point encoded signal can lead to drastically different
    output.

We get the signal from the `channels` matrix:
```julia
julia> x = channels[:, 1]
```

As a sanity check, we print the sampling rate and duration of the
signal:
```julia
julia> println("sampling freq: $srate Hz\nduration: $(round(length(x) / srate, digits=2)) s")
sampling freq: 16000.0 Hz
duration: 2.92 s
```
and we plot the waveform:
```julia
julia> using Plots
julia> pyplot()
julia> t = range(0, length(x) / srate, length=length(x))
julia> plot(t, x, size = (1000, 300), xlabel = "time (seconds)", legend = false)
```
![](images/signal.svg)

## Extracting the features

All the different types of features supported by this package follow
the same extraction scheme.
1. create a the feature extractor object with a specific configuration
2. send the signal(s) to this extractor to get the features.

SpeechFeatures provides the following feature extractor:

| Extractor  | Constructor | Description |
|:-----------|:------------|:------------|
| Log magnitude spectrum | `LogMagnitudeSpectrum([options])` | Logarithm of the magnitude of the Short Term Fourier Transform (STFT) |
| Log Mel Spectrum | `LogMelSpectrum([options])` | Logarithm of the STFT transformed via a mel-spaced filter bank. |
| Mel Cepsral Coefficients (MFCCs) | `MFCC([options])` | Classical MFCC features |

As an example, we will use the popular Mel Frequency Cepstral
Coefficients (MFCC) features. First we create the extractor
with the default configuration:
```julia
julia> mfcc = MFCC()
```
and then, we extract and plot the features from our TIMIT sample:
```julia
julia> fea = x |> mfcc
julia> heatmap(range(0, length(x) / srate, length = size(fea, 2)),
               1:size(fea, 1), fea, xlabel = "time (s)", c = :viridis)
```
![](images/mfcc.svg)

Here is the list of possible options for each extractor

| Option name | Default | Supported by | Description  |
|:------------|:--------|:-------------|:-------------|
| `removedc`  | `true`  | all          | Remove the direct component from the signal. |
| `dithering` | `true`  | all          | Add Gaussian white noise with `dithering` stdandard deviation. |
| `srate`     | `16000` | all          | Sampling rate in Hz of the input signal |
| `frameduration` | `0.025` | all        | Frame duration in seconds. |
| `framestep` | `0.011` | all          | Frame step (hop size) in seconds. |
| `preemphasis` | `0.97` | all         | Preemphasis filter coefficient. |
| `windowfn` | `SpeechFeatures.HannWindow` | all | Windowing function (others are `HammingWindow` or `RectangularWindow`). |
| `windowpower` | `0.85` | all         | Sharpening exponent of the window. |
| `nfilters` | `26` | LogMelSpectrum \| MFCC | Number of filters in the filter bank. |
| `lofreq` | `80` | LogMelSpectrum \| MFCC | Low cut-off frequency in Hz for the filter bank. |
| `hifreq` | `7600` | LogMelSpectrum \| MFCC | High cut-off frequency in Hz for the filter bank. |
| `addenergy` | `true` | MFCC | Append the per-frame energy to the features. |
| `nceps` | `12` | MFCC | Number of cepstral coefficients. |
| `liftering` | `22` | MFCC | Liftering coefficient. |

## Deltas and mean normalization

The deltas and acceleration coefficients (i.e. "double deltas") can
be computed by chaining the features extraction with the
deltas features extractor:
```julia
julia> Δ_ΔΔ = DeltaCoeffs(order = 2, deltawin = 2)
julia> fea = x |> mfcc |> Δ_ΔΔ
```
The `order` parameter is the order of the deltas coefficients, i.e.
`order = 2` means that the first and second deltas (acceleration)
coefficients will be computed. `deltawin` is the length of the delta
window.

Similarly, to remove the mean of the utterance you can add one more
element to the chain:
```julia
julia> mnorm = MeanNorm()
julia> fea = x |> mfcc |> Δ_ΔΔ |> mnorm
```

