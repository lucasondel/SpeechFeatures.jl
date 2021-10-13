# SPDX-License-Identifier: MIT

"""
    stft(x, fs; kwargs...) -> S, fftlen

Compute the short-term Fourier transform of the signal `x`. `fs` is
the sampling frequency of the input signal. In addition to the spectrum,
the function also return the length of the FFT used.
"""
function stft(x_in; srate=16000, dithering=0, removedc=true, frameduration=0.025,
              framestep=0.01, windowfn=HannWindow, windowexp=0.85,
              preemph=0.97)

    x = x_in .+ randn(length(x_in)) * dithering
    x .-= sum(x) / length(x)
    X = hcat(eachframe(x; srate, frameduration, framestep)...)
    X = hcat(map(preemphasis, eachcol(X))...)
    window = windowfn(Int64(srate*frameduration)) .^ windowexp
    X = hcat(map(x -> x .* window, eachcol(X))...)

    fftlen = Int64(2^ceil(log2(size(X, 1))))
    pX = PaddedView(0, X, (fftlen, size(X, 2)))
    rfft(pX, 1)[1:end-1,:], fftlen
end

"""
    mfcc(mS; kwargs...)

Compute the cepstral coefficient from the the magnitude of the
mel-spectrum `mS`.
"""
function mfcc(mS; nceps=13, liftering=22)
    C = dct(log.(mS), 1)[1:nceps,:]
    if liftering > 0
        lifter = makelifter(size(C,1), liftering)
        C = hcat(map(x -> x .* lifter, eachcol(C))...)
    end
    C
end

"""
    add_deltas(X; order=2, winlen=2)

Add the derivatives to the features.
"""
function add_deltas(X; order=2, winlen=2)
    X_and_deltas = [X]
    for o in 1:order
        push!(X_and_deltas, delta(X_and_deltas[end], winlen))
    end
    vcat(X_and_deltas...)
end

