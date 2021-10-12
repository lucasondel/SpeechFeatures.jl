module SpeechFeatures

using PaddedViews
using FFTW

include("utils.jl")
include("features.jl")

export FilterBank, stft, mfcc, add_deltas

end
