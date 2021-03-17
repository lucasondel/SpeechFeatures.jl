module SpeechFeatures

using PaddedViews
using FFTW

#######################################################################
# Internal API

include("utils.jl")
include("windows.jl")
include("fbank.jl")
include("framing.jl")

#######################################################################
# Features

export DeltaCoeffs
export LogMagnitudeSpectrum
export LogMelSpectrum
export MFCC
export MeanNorm

include("features.jl")

end
