# Basic DSP operations.

using PaddedViews

export frames
export preemphasis!
export removedc!


struct FrameIterator
    signal::Vector{<:Real}
    framelength::Int64
    hopsize::Int64
end

function Base.length(it::FrameIterator)
    if length(it.signal) <= it.framelength
        return 0
    end
    (length(it.signal) - it.framelength) ÷ it.hopsize
end

function Base.iterate(it::FrameIterator, state::Int64=1)
    if state > length(it)
        return nothing
    end
    framestart = (state - 1) * it.hopsize + 1
    frameend = framestart + it.framelength - 1
    (it.signal[framestart:frameend], state + 1)
end

"""
    frames(x, sr, t, Δt)

Return an iterator over the frames of the signal `x`. Each frame will
have `t` time span in second and `Δt` time shift in second as well.
`sr` is the sampling rate of the `x` in Hertz.
"""
function frames(x::Vector{<:Real}, sr::Real, t::Real, Δt::Real)
    FrameIterator(x, Int64(sr * t), Int64(sr * Δt))
end


"""
    preemphasis!(x, k)

Apply a preemphasis filter in-place on the signal `x` with
preemphasis coefficient `k`. The filter is defined as:
``
    x̂[n] = x[n] - k x[n-t]
``
"""
function preemphasis!(x::Vector{<:Real}, k::Float64)
    psignal = PaddedView(signal[1], signal[1:end-1], (length(signal),))
    signal .-= psignal * k
end

"""
    removedc!(x)

Remove the direct component of the signal `x` in-place.
"""
removedc!(x::Vector{<:Real}) = x .-= sum(x) / length(x)


