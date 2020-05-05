# Basic DSP operations.

using PaddedViews
using FFTW

struct FrameIterator
    signal::Vector{<:Real}
    framelength::Int64
    hopsize::Int64
end

function Base.length(it::FrameIterator)
    if length(it.signal) <= it.framelength
        return 0
    end
    1 + (length(it.signal) - it.framelength) ÷ it.hopsize
end

function Base.iterate(it::FrameIterator, state::Int64=1)
    if state > length(it)
        return nothing
    end
    framestart = (state - 1) * it.hopsize + 1
    frameend = framestart + it.framelength - 1
    (it.signal[framestart:frameend], state + 1)
end

# Return an iterator over the frames of the signal `x`
function frames(x::Vector{<:Real}, sr::Real, t::Real, Δt::Real)
    FrameIterator(x, Int64(sr * t), Int64(sr * Δt))
end

# Generate the DCT bases, `d` is the number of samples per base
# and `n` the number of bases.
function dctbases(T::Type, n::Integer, d::Integer)
    retval = zeros(T, n, d)
    t = range(0, d-1, length = d) .+ 0.5
    for i = 1:n
        retval[i, :] = cos.(i * π * t / d)
    end
    retval
end


# Generate the liftering function. `n` is the number of cepstral
# coefficients and `l` the liftering parameter.
function lifter(T::Type, n::Integer, l::Real)
    t = Vector{T}(1:n)
    retval = 1 .+ T(l/2) * sin.(π * t / l)
    return retval
end


# Calculate the delta coefficient of a matrix of features
function getdelta(X::Matrix{T}, deltawin::Integer = 2) where T <: AbstractFloat
    D, N = size(X)
    Δ = zeros(T, D, N)
    norm = T(2. * sum(collect(1:deltawin) .^ 2))
    for i = 1:N
        for t = 1:deltawin
            tm = i - t
            tp = i + t
            if tm < 1 tm = 1 end
            if tp > N tp = N end
            Δ[:, i] = (t * (X[:, tp] - X[:, tm])) / norm
        end
    end
    return Δ
end

