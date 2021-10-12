# SPDX-License-Identifier: MIT

#======================================================================
Liftering
======================================================================#

function makelifter(N, L)
	t = Vector(1:N)
	1 .+ L/2 * sin.(π * t / L)
end

#======================================================================
Pre-emphasis
======================================================================#

function preemphasis(x; k=0.97)
    y = similar(x)
	y[1] = x[1]
	prev = x[1]
	for i in 2:length(x)
		y[i] = x[i] - k*prev
		prev = x[i]
	end
	y
end

#======================================================================
Delta features
======================================================================#

function delta(X::AbstractMatrix{T}, deltawin::Int = 2) where T
    D, N = size(X)
    Δ = zeros(T, D, N)
    norm = T(2 * sum(collect(1:deltawin).^2))
    for n in 1:N
        for θ in 1:deltawin
            Δ[:, n] += (θ * (X[:,min(N, n+θ)] - X[:,max(1,n-θ)])) / norm
        end
    end
    Δ
end

#======================================================================
Framing
======================================================================#

struct FrameIterator{T<:AbstractVector}
    x::T
    framesize::Int64
    hopsize::Int64
end

function Base.length(it::FrameIterator)
    N = length(it.x)
    N > it.framesize ? 1 + (N - it.framesize) ÷ it.hopsize : 0
end

function Base.iterate(it::FrameIterator, idx=1)
    1 + length(it.x) - idx < it.framesize &&  return nothing
    (view(it.x, idx:(idx + it.framesize - 1)), idx + it.hopsize)
end

eachframe(x::AbstractVector; srate=16000, frameduration=0.025, framestep=0.01) =
    FrameIterator(x, Int64(srate * frameduration), Int64(srate * framestep))

#======================================================================
Window functions
======================================================================#

function HannWindow(T::Type, N::Int)
    T(.5) .* (1 .- cos.(T(2π) .* Vector{T}(0:N-1) ./ (N-1)))
end
HannWindow(N::Int) = HannWindow(Float64, N)

function HammingWindow(T::Type, N::Int)
    T(0.54) .- T(0.46) .* cos.(T(2π) .* Vector{T}(0:N-1) ./ (N-1))
end
HammingWindow(N::Int) = HammingWindow(Float64, N)

RectangularWindow(T::Type, N::Int) = ones(T, N)
RectangularWindow(N::Int) = RectangularWindow(Float64, N)

#======================================================================
Filter bank
======================================================================#

mel2freq(mel::Real) = 700 * (exp(mel / 1127) - 1)
freq2mel(freq::Real) = 1127 * (log(1 + (freq / 700)))

# Create a set of triangular filters
function FilterBank(n::Int; srate::Real = 16000, fftlen::Int = 512,
                    lofreq::Real = 80, hifreq::Real = 7600)

    # Convert the cut-off frequencies into mel
    lomel = freq2mel(lofreq)
    himel = freq2mel(hifreq)

    # Centers (in mel and freqs) of the filterbank
    melcenters = range(lomel, himel, length = n + 2)
    freqcenters = mel2freq.(melcenters)

    # Now get the centers in terms of FFT bins
    bincenters = 1 .+ Int64.(floor.( fftlen .* freqcenters ./ srate ))

    # Allocate the matrix which will store the filters
    D = Int64(ceil(fftlen / 2))
    F = zeros(n, D)

    # Construct the "triangle"
    for f = 1:n
        d1 = bincenters[f + 1] - bincenters[f]
        d2 = bincenters[f + 2] - bincenters[f + 1]

        s = bincenters[f]
        e = bincenters[f + 1]
        F[f, s:e] = range(0, 1, length = d1 + 1)

        s = bincenters[f + 1]
        e = bincenters[f + 2]
        F[f, s:e] = range(1, 0, length = d2 + 1)
    end
    F
end

