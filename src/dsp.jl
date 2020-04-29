# Basic DSP operations.

using PaddedViews
using FFTW

struct FrameIterator signal::Vector{<:AbstractFloat}
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
function frames(x::Vector{<:AbstractFloat}, sr::Real, t::Real, Δt::Real)
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

# Generate the liftering function. `n` if the number of cepstral
# coefficients and `l` the liftering parameter.
function lifter(T::Type, n::Integer, l::Real)
    t = Vector{T}(1:n)
    1 .+ (l/2) * sin.(π * t / l)
end

