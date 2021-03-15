module SpeechFeatures

using PaddedViews
using FFTW

#######################################################################
# Utilities

include("dsp.jl")

#include("fbank.jl")
#include("windows.jl")

#######################################################################
# Short Term Fouriere Spectral analysis

#export STFT

#include("stft.jl")

#######################################################################
# Log Mel spectrum

#struct LogMelSpectrum
#    stft::STFT
#    fbank::AbstractMatrix
#end
#
#function LogMelSpectrum(;T = Float64,
#                        fftlen = fftlen_auto,
#                        srate = 16000,
#                        frameduration = 0.025,
#                        framestep = 0.01,
#                        removedc = true,
#                        dithering = 0.,
#                        preemphasis = 0.97,
#                        windowfn = hamming,
#                        windowpower = 1.0,
#                        nfilters = 26,
#                        lofreq = 80,
#                        hifreq = 8000)
#    stft = STFT(T, fftlen, srate, frameduration, framestep, removedc, dithering,
#                preemphasis, windowfn, windowpower)
#
#    if fftlen == fftlen_auto fftlen = fftlen_auto(srate * frameduration) end
#    if fftlen % 2 == 0 fftlen -= 1 end
#
#    F = FilterBank(T, nfilters;
#                   srate = srate,
#                   fftlen = fftlen,
#                   lofreq = lofreq,
#                   hifreq = hifreq)
#    LogMelSpectrum(stft, F)
#end
#
#function (lmels::LogMelSpectrum)(x::Vector{<:Real})
#    S = lmels.stft(x)
#    log.( lmels.fbank' * abs.(S) .+ nextfloat(lmels.stft.T(0.)))
#end
#
########################################################################
## MFCC
#
#struct RawEnergy end
#const rawenergy = RawEnergy()
#
#struct CepsEnergy end
#const cepsenergy = CepsEnergy()
#
#struct MFCC
#    stft::STFT
#    fbank::AbstractMatrix
#    dct::AbstractMatrix
#    lift::AbstractVector
#    htkscaling::Bool
#    energy::Union{RawEnergy, CepsEnergy, Nothing}
#    energyfloor::Float64
#end
#
#function MFCC(;T = Float64,
#              fftlen = fftlen_auto,
#              srate = 16000,
#              frameduration = 0.025,
#              framestep = 0.01,
#              removedc = true,
#              dithering = 0.,
#              preemphasis = 0.97,
#              windowfn = hamming,
#              windowpower = 1.0,
#              nfilters = 26,
#              lofreq = 80,
#              hifreq = 8000,
#              nceps = 12,
#              liftering = 22,
#              htkscaling = true,
#              energy = cepsenergy,
#              energyfloor = 0.0
#             )
#
#    stft = STFT(T, fftlen, srate, frameduration, framestep, removedc, dithering,
#                preemphasis, windowfn, windowpower)
#
#    if fftlen == fftlen_auto fftlen = fftlen_auto(srate * frameduration) end
#    if fftlen % 2 == 0 fftlen -= 1 end
#    F = FilterBank(T, nfilters; srate = srate, fftlen = fftlen, lofreq = lofreq,
#                   hifreq = hifreq)
#
#    dct = dctbases(T, nceps, nfilters)
#    lift = lifter(T, nceps, liftering)
#
#    MFCC(stft, F, dct, lift, htkscaling, energy, energyfloor)
#end
#
#function (mfcc::MFCC)(x::Vector{<:Real})
#    S, e = mfcc.stft(x, return_energy = true)
#    melspec = log.( mfcc.fbank' * abs.(S) .+ nextfloat(mfcc.stft.T(0.)))
#    fea = mfcc.lift .* mfcc.dct * melspec
#
#    mfnorm = sqrt(2. / size(mfcc.fbank, 2))
#    if mfcc.htkscaling fea .*= mfnorm end
#
#    if mfcc.energy ≠ nothing
#        if mfcc.energy == rawenergy
#            e = e[[CartesianIndex()], :]
#        else
#            println("cepsenergy")
#            e = sum(melspec, dims = 1)
#            if mfcc.htkscaling e .*= mfnorm end
#        end
#        floor = eltype(e)(mfcc.energyfloor)
#        e[ e .< floor] .= floor
#        fea = vcat(e, fea)
#    end
#    return fea
#end
#
########################################################################
## Delta features
#
#struct DeltaCoeffs
#    order::Integer
#    win::Integer
#end
#DeltaCoeffs(;order = 2, win = 2) = DeltaCoeffs(order, win)
#
#function (dcoeffs::DeltaCoeffs)(X::Matrix{T}) where T <: AbstractFloat
#    X_and_deltas = [X]
#    for order = 1:dcoeffs.order
#        push!(X_and_deltas, getdelta(X_and_deltas[end]))
#    end
#    vcat(X_and_deltas...)
#end

end
