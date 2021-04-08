_framelen(srate, fduration) = Int64(srate * fduration)

struct FrameIterator{T<:AbstractVector}
    signal::T
    framelength::Int64
    hopsize::Int64
end

function Base.length(it::FrameIterator)
    N = length(it.signal)
    N > it.framelength ? 1 + (N - it.framelength) ÷ it.hopsize : 0
end

Base.eltype(it::FrameIterator{T}) where T = T

function Base.iterate(it::FrameIterator, state::Int64 = 1)
    if state > length(it)
        return nothing
    end
    framestart = (state - 1) * it.hopsize + 1
    frameend = framestart + it.framelength - 1
    (it.signal[framestart:frameend], state + 1)
end

# Split the signal in overlapping frames
function frames(x::AbstractVector, sr::Real, t::Real, Δt::Real)
    FrameIterator(x, _framelen(sr, t), Int64(sr * Δt))
end

struct FrameExtractor
    srate::Int64
    frameduration::Float64
    framestep::Float64
    preemphasis::Float64
    windowfn::Function
    windowpower::Float64
end

function FrameExtractor(; srate = 16000, frameduration = 0.025, framestep = 0.01,
                         preemphasis = 0.97, windowfn = HannWindow,
                         windowpower = 0.85)
    FrameExtractor(srate, frameduration, framestep, preemphasis, windowfn,
                   windowpower)
end

function (fx::FrameExtractor)(x::AbstractVector)
    itframes = frames(x, fx.srate, fx.frameduration, fx.framestep)
    N = length(itframes)
    W = _framelen(fx.srate, fx.frameduration)
    window = fx.windowfn(Float64, W) .^ fx.windowpower
    retval = Matrix{Float64}(undef, W, N)
    for (n,frame) in enumerate(itframes)
        h = vcat(frame[1], frame[1:end-1]) * fx.preemphasis
        retval[:, n] .= window .* (frame .- h)
    end
    retval
end

