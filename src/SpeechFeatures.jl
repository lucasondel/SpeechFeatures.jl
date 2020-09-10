module SpeechFeatures

include("windows.jl")
include("dsp.jl")
include("fbank.jl")

export STFT
export LogMagnitudeSpectrum
export LogMelSpectrum
export cepsenergy
export rawenergy
export MFCC
export DeltaCoeffs
export fftlen_auto

export frames


#######################################################################
# Short Term Fourer Spectrum

# Constant to tell the stft function to automatically determine
# the length of the FFT.
struct FFTLengthAutoConfig end
fftlen_auto = FFTLengthAutoConfig()
(::FFTLengthAutoConfig)(framelength) = Int(2^ceil(log2(framelength)))

struct STFT
    T::Type
    fftlen::Union{Integer, FFTLengthAutoConfig}
    srate::Real
    frameduration::Real
    framestep::Real
    removedc::Bool
    dithering::Real
    preemphasis::Real
    windowfn::WindowFunction
    windowpower::Real
end


function STFT(;T = Float64,
              fftlen = fftlen_auto,
              srate = 16000,
              frameduration = 0.025,
              framestep = 0.01,
              removedc = true,
              dithering = 0.,
              preemphasis = 0.97,
              windowfn = hamming,
              windowpower = 1.0)
    STFT(T, fftlen, srate, frameduration, framestep, removedc, dithering,
         preemphasis, windowfn, windowpower)
end


function (stft::STFT)(signal::Vector{<:Real}; return_energy::Bool = false)
    # Copy the signal not to modify the original
    x = deepcopy(signal)

    # Get the signal in the requested floating point precision)
    x = convert(Vector{stft.T}, x)

    # Dithering
    x .+= randn(length(x)) .* stft.dithering

    # Remove DC offset
    if stft.removedc x .-= sum(x) / length(x) end

    # Split the signal in overlapping frames
    # X = frame length x no. frames
    X = hcat(frames(x, stft.srate, stft.frameduration, stft.framestep)...)

    # Sharpen/flatten the window by exponentiating it
    window = stft.windowfn(stft.T, size(X, 1)) .^ stft.windowpower

    # Iterate of the column, i.e. the frames of the signal
    energy = zeros(stft.T, size(X, 2))
    for i in 1:size(X, 2)
        # Pre-emphasis
        px = vcat([X[1, i] ], X[1:end-1, i])
        X[:, i] .-= px * stft.preemphasis

        # Windowing
        X[:, i] .*= window

        # Per-frame energy
        energy[i] = log(sum(X[:, i] .^ 2))
    end

    fftlen = stft.fftlen
    if fftlen == fftlen_auto
        # Get the closest power of 2 of the frame length
        fftlen = fftlen_auto(size(X, 1))
    end

    # Pad the frames with 0 to get the correct length FFT
    pX = PaddedView(0, X, (fftlen, size(X, 2)))

    # Compute (half of) the Fourier transform over the first dimension
    S = rfft(pX, 1)

    # If the length of FFT is even then we discard the value at the
    # Nyquist frequency.
    if fftlen % 2 == 0
        S = S[1:end-1, :]
    end

    if return_energy
        return S, energy
    else
        return S
    end
end

#######################################################################
# Log magnitude spectrum

struct LogMagnitudeSpectrum
    stft::STFT
end

function LogMagnitudeSpectrum(;T = Float64,
                              fftlen = fftlen_auto,
                              srate = 16000,
                              frameduration = 0.025,
                              framestep = 0.01,
                              removedc = true,
                              dithering = 0.,
                              preemphasis = 0.97,
                              windowfn = hamming,
                              windowpower = 1.0)
    stft = STFT(T, fftlen, srate, frameduration, framestep, removedc,
                dithering, preemphasis, windowfn, windowpower)
    LogMagnitudeSpectrum(stft)
end

function (lms::LogMagnitudeSpectrum)(x::Vector{<:Real})
    S = lms.stft(x)
    log.(abs.(S) .+ nextfloat(lms.stft.T(0.)))
end

#######################################################################
# Log Mel spectrum

struct LogMelSpectrum
    stft::STFT
    fbank::AbstractMatrix
end

function LogMelSpectrum(;T = Float64,
                        fftlen = fftlen_auto,
                        srate = 16000,
                        frameduration = 0.025,
                        framestep = 0.01,
                        removedc = true,
                        dithering = 0.,
                        preemphasis = 0.97,
                        windowfn = hamming,
                        windowpower = 1.0,
                        nfilters = 26,
                        lofreq = 80,
                        hifreq = 8000)
    stft = STFT(T, fftlen, srate, frameduration, framestep, removedc, dithering,
                preemphasis, windowfn, windowpower)

    if fftlen == fftlen_auto fftlen = fftlen_auto(srate * frameduration) end
    if fftlen % 2 == 0 fftlen -= 1 end

    F = FilterBank(T, nfilters;
                   srate = srate,
                   fftlen = fftlen,
                   lofreq = lofreq,
                   hifreq = hifreq)
    LogMelSpectrum(stft, F)
end

function (lmels::LogMelSpectrum)(x::Vector{<:Real})
    S = lmels.stft(x)
    log.( lmels.fbank' * abs.(S) .+ nextfloat(lmels.stft.T(0.)))
end

#######################################################################
# MFCC

struct RawEnergy end
const rawenergy = RawEnergy()

struct CepsEnergy end
const cepsenergy = CepsEnergy()

struct MFCC
    stft::STFT
    fbank::AbstractMatrix
    dct::AbstractMatrix
    lift::AbstractVector
    htkscaling::Bool
    energy::Union{RawEnergy, CepsEnergy, Nothing}
    energyfloor::Float64
end

function MFCC(;T = Float64,
              fftlen = fftlen_auto,
              srate = 16000,
              frameduration = 0.025,
              framestep = 0.01,
              removedc = true,
              dithering = 0.,
              preemphasis = 0.97,
              windowfn = hamming,
              windowpower = 1.0,
              nfilters = 26,
              lofreq = 80,
              hifreq = 8000,
              nceps = 12,
              liftering = 22,
              htkscaling = true,
              energy = cepsenergy,
              energyfloor = 0.0
             )

    stft = STFT(T, fftlen, srate, frameduration, framestep, removedc, dithering,
                preemphasis, windowfn, windowpower)

    if fftlen == fftlen_auto fftlen = fftlen_auto(srate * frameduration) end
    if fftlen % 2 == 0 fftlen -= 1 end
    F = FilterBank(T, nfilters; srate = srate, fftlen = fftlen, lofreq = lofreq,
                   hifreq = hifreq)

    dct = dctbases(T, nceps, nfilters)
    lift = lifter(T, nceps, liftering)

    MFCC(stft, F, dct, lift, htkscaling, energy, energyfloor)
end

function (mfcc::MFCC)(x::Vector{<:Real})
    S, e = mfcc.stft(x, return_energy = true)
    melspec = log.( mfcc.fbank' * abs.(S) .+ nextfloat(mfcc.stft.T(0.)))
    fea = mfcc.lift .* mfcc.dct * melspec

    mfnorm = sqrt(2. / size(mfcc.fbank, 2))
    if mfcc.htkscaling fea .*= mfnorm end

    if mfcc.energy ≠ nothing
        if mfcc.energy == rawenergy
            e = e[[CartesianIndex()], :]
        else
            println("cepsenergy")
            e = sum(melspec, dims = 1)
            if mfcc.htkscaling e .*= mfnorm end
        end
        floor = eltype(e)(mfcc.energyfloor)
        e[ e .< floor] .= floor
        fea = vcat(e, fea)
    end
    return fea
end

#######################################################################
# Delta features

struct DeltaCoeffs
    order::Integer
    win::Integer
end
DeltaCoeffs(;order = 2, win = 2) = DeltaCoeffs(order, win)

function (dcoeffs::DeltaCoeffs)(X::Matrix{T}) where T <: AbstractFloat
    X_and_deltas = [X]
    for order = 1:dcoeffs.order
        push!(X_and_deltas, getdelta(X_and_deltas[end]))
    end
    vcat(X_and_deltas...)
end

end
