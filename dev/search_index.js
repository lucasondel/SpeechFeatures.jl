var documenterSearchIndex = {"docs":
[{"location":"feaextract/#Features-extraction-1","page":"Extracting Features","title":"Features extraction","text":"","category":"section"},{"location":"feaextract/#Loading-an-audio-file-1","page":"Extracting Features","title":"Loading an audio file","text":"","category":"section"},{"location":"feaextract/#","page":"Extracting Features","title":"Extracting Features","text":"To extract any type of speech features you will need the audio signal stored in an Array-like object and the sampling rate in Hertz. SpeechFeatures does not provide a way to load these two elements from audio files directly but there are several Julia packages to do this. In this tutorial, we will use WAV.jl. For the rest of the tutorial, we assumed that you have installed the WAV.jl package in your Julia environment.","category":"page"},{"location":"feaextract/#","page":"Extracting Features","title":"Extracting Features","text":"First of all, as an example, we download an audio file from the TIMIT corpus. In the Julia REPL type:","category":"page"},{"location":"feaextract/#","page":"Extracting Features","title":"Extracting Features","text":"julia> run(`wget https://catalog.ldc.upenn.edu/desc/addenda/LDC93S1.wav`)","category":"page"},{"location":"feaextract/#","page":"Extracting Features","title":"Extracting Features","text":"Now, we load the audio waveform:","category":"page"},{"location":"feaextract/#","page":"Extracting Features","title":"Extracting Features","text":"julia> using WAV\njulia> channels, srate = wavread(\"LDC93S1.wav\", format = \"double\")","category":"page"},{"location":"feaextract/#","page":"Extracting Features","title":"Extracting Features","text":"Where channels is a NxC matrix where N is the length of the audio in samples and C is the number of channels. Since TIMIT is mono recorded it has only one channel. format = \"double\" indicates that the signals in channels will be encoded with double precision and each sample of the signal will be between 1.0 and -1.0.","category":"page"},{"location":"feaextract/#","page":"Extracting Features","title":"Extracting Features","text":"warning: Warning\nThe wavread function also accepts format = \"native\" which will return the data in the format it is stored in the WAV file. We discourage its use as extracting the features from integer or floating point encoded signal can lead to drastically different output.","category":"page"},{"location":"feaextract/#","page":"Extracting Features","title":"Extracting Features","text":"We get the signal from the channels matrix:","category":"page"},{"location":"feaextract/#","page":"Extracting Features","title":"Extracting Features","text":"julia> x = channels[:, 1]","category":"page"},{"location":"feaextract/#","page":"Extracting Features","title":"Extracting Features","text":"As a sanity check, we print the sampling rate and duration of the signal:","category":"page"},{"location":"feaextract/#","page":"Extracting Features","title":"Extracting Features","text":"julia> println(\"sampling freq: $srate Hz\\nduration: $(round(length(x) / srate, digits=2)) s\")\nsampling freq: 16000.0 Hz\nduration: 2.92 s","category":"page"},{"location":"feaextract/#","page":"Extracting Features","title":"Extracting Features","text":"and we plot the waveform:","category":"page"},{"location":"feaextract/#","page":"Extracting Features","title":"Extracting Features","text":"julia> using Plots\njulia> pyplot()\njulia> t = range(0, length(x) / srate, length=length(x))\njulia> plot(t, x, size = (1000, 300), xlabel = \"time (seconds)\", legend = false)","category":"page"},{"location":"feaextract/#","page":"Extracting Features","title":"Extracting Features","text":"(Image: )","category":"page"},{"location":"feaextract/#Extracting-the-features-1","page":"Extracting Features","title":"Extracting the features","text":"","category":"section"},{"location":"feaextract/#","page":"Extracting Features","title":"Extracting Features","text":"All the different types of features supported by this package follow the same extraction scheme.","category":"page"},{"location":"feaextract/#","page":"Extracting Features","title":"Extracting Features","text":"create a the feature extractor object with a specific configuration\nsend the signal(s) to this extractor to get the features.","category":"page"},{"location":"feaextract/#","page":"Extracting Features","title":"Extracting Features","text":"SpeechFeatures provides the following feature extractor:","category":"page"},{"location":"feaextract/#","page":"Extracting Features","title":"Extracting Features","text":"Extractor Constructor Description\nLog magnitude spectrum LogMagnitudeSpectrum([options]) Logarithm of the magnitude of the Short Term Fourier Transform (STFT)\nLog Mel Spectrum LogMelSpectrum([options]) Logarithm of the STFT transformed via a mel-spaced filter bank.\nMel Cepsral Coefficients (MFCCs) MFCC([options]) Classical MFCC features","category":"page"},{"location":"feaextract/#","page":"Extracting Features","title":"Extracting Features","text":"As an example, we will use the popular Mel Frequency Cepstral Coefficients (MFCC) features. First we create the extractor with the default configuration:","category":"page"},{"location":"feaextract/#","page":"Extracting Features","title":"Extracting Features","text":"julia> mfcc = MFCC()","category":"page"},{"location":"feaextract/#","page":"Extracting Features","title":"Extracting Features","text":"and then, we extract and plot the features from our TIMIT sample:","category":"page"},{"location":"feaextract/#","page":"Extracting Features","title":"Extracting Features","text":"julia> fea = x |> mfcc","category":"page"},{"location":"feaextract/#","page":"Extracting Features","title":"Extracting Features","text":"Here is the list of possible options for each extractor","category":"page"},{"location":"feaextract/#","page":"Extracting Features","title":"Extracting Features","text":"Option name Default Supported by Description\nremovedc true all Remove the direct component from the signal.\ndithering true all Add Gaussian white noise with dithering stdandard deviation.\nsrate 16000 all Sampling rate in Hz of the input signal\nframeduration 0.025 all Frame duration in seconds.\nframestep 0.011 all Frame step (hop size) in seconds.\npreemphasis 0.97 all Preemphasis filter coefficient.\nwindowfn SpeechFeatures.HannWindow all Windowing function (others are HammingWindow or RectangularWindow).\nwindowpower 0.85 all Sharpening exponent of the window.\nnfilters 26 LogMelSpectrum | MFCC Number of filters in the filter bank.\nlofreq 80 LogMelSpectrum | MFCC Low cut-off frequency in Hz for the filter bank.\nhifreq 7600 LogMelSpectrum | MFCC High cut-off frequency in Hz for the filter bank.\naddenergy true MFCC Append the per-frame energy to the features.\nnceps 12 MFCC Number of cepstral coefficients.\nliftering 22 MFCC Liftering coefficient.","category":"page"},{"location":"feaextract/#Deltas-and-mean-normalization-1","page":"Extracting Features","title":"Deltas and mean normalization","text":"","category":"section"},{"location":"feaextract/#","page":"Extracting Features","title":"Extracting Features","text":"The deltas and acceleration coefficients (i.e. \"double deltas\") can be computed by chaining the features extraction with the deltas features extractor:","category":"page"},{"location":"feaextract/#","page":"Extracting Features","title":"Extracting Features","text":"julia> Δ_ΔΔ = DeltaCoeffs(order = 2, deltawin = 2)\njulia> fea = x |> mfcc |> Δ_ΔΔ","category":"page"},{"location":"feaextract/#","page":"Extracting Features","title":"Extracting Features","text":"The order parameter is the order of the deltas coefficients, i.e. order = 2 means that the first and second deltas (acceleration) coefficients will be computed. deltawin is the length of the delta window.","category":"page"},{"location":"feaextract/#","page":"Extracting Features","title":"Extracting Features","text":"Similarly, to remove the mean of the utterance you can add one more element to the chain:","category":"page"},{"location":"feaextract/#","page":"Extracting Features","title":"Extracting Features","text":"julia> mnorm = MeanNorm()\njulia> fea = x |> mfcc |> Δ_ΔΔ |> mnorm","category":"page"},{"location":"#SpeechFeatures.jl-1","page":"Home","title":"SpeechFeatures.jl","text":"","category":"section"},{"location":"#","page":"Home","title":"Home","text":"SpeechFeatures.jl is a Julia package for extracting acoustic features for speech technologies.","category":"page"},{"location":"#Authors-1","page":"Home","title":"Authors","text":"","category":"section"},{"location":"#","page":"Home","title":"Home","text":"Lucas Ondel","category":"page"},{"location":"#Installation-1","page":"Home","title":"Installation","text":"","category":"section"},{"location":"#","page":"Home","title":"Home","text":"The package can be installed with the Julia package manager. From the Julia REPL, type ] to enter the Pkg REPL mode and run:","category":"page"},{"location":"#","page":"Home","title":"Home","text":"pkg> add SpeechFeatures","category":"page"},{"location":"#Manual-Outline-1","page":"Home","title":"Manual Outline","text":"","category":"section"},{"location":"#","page":"Home","title":"Home","text":"Pages = [\"feaextract.md\"]","category":"page"}]
}
