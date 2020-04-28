module SpeechFeatures


include("windows.jl")
include("dsp.jl")
include("fbank.jl")

export LogMagnitudeSpectrum
export LogMelSpectrum

function LogMagnitudeSpectrum(T::Type, x::Vector{<:Real};
                              fftlen::Union{Integer, FFTLengthAutoConfig} = fftlen_auto,
                              srate::Real = 16000,
                              frameduration::Real = 0.025,
                              framestep::Real = 0.01,
                              removedc::Bool = true,
                              dithering::Real = 0.,
                              preemphasis::Real = 0.97,
                              windowfn::WindowFunction = hamming,
                              windowpower::Real = 1.0)
    S = stft(T, x;
             fftlen = fftlen,
             srate = srate,
             frameduration = frameduration,
             framestep = framestep,
             removedc = removedc,
             dithering = dithering,
             preemphasis = preemphasis,
             windowfn = windowfn,
             windowpower = windowpower)
    log.(abs.(S) .+ nextfloat(0.))
end
LogMagnitudeSpectrum(x::Vector{<:Real}; kwargs...) = LogMagnitudeSpectrum(Float64, x; kwargs...)

function LogMelSpectrum(T::Type, x::Vector{<:Real};
                        fftlen::Union{Integer, FFTLengthAutoConfig} = fftlen_auto,
                        srate::Real = 16000,
                        frameduration::Real = 0.025,
                        framestep::Real = 0.01,
                        removedc::Bool = true,
                        dithering::Real = 0.,
                        preemphasis::Real = 0.97,
                        windowfn::WindowFunction = hamming,
                        windowpower::Real = 1.0,
                        nfilters::Integer = 26,
                        lofreq::Real = 80,
                        hifreq::Real = 8000)
    S = stft(T, x;
             fftlen = fftlen,
             srate = srate,
             frameduration = frameduration,
             framestep = framestep,
             removedc = removedc,
             dithering = dithering,
             preemphasis = preemphasis,
             windowfn = windowfn,
             windowpower = windowpower)
    F = FilterBank(T, nfilters;
                   srate = srate,
                   fftlen = 2 * size(S, 1),
                   lofreq = lofreq,
                   hifreq = hifreq)
    log.( F' * abs.(S) .+ nextfloat(0.))
end
LogMelSpectrum(x::Vector{<:Real}; kwargs...) = LogMelSpectrum(Float64, x; kwargs...)

end

