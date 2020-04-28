# Basic DSP operations.

using PaddedViews
using FFTW

export frames
export fftlen_auto
export stft


struct FrameIterator
    signal::Vector{<:AbstractFloat}
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

# Constant to indicate the stft function to automatically determine
# the length of the FFT.
struct FFTLengthAutoConfig end
fftlen_auto = FFTLengthAutoConfig()

(::FFTLengthAutoConfig)(framelength) = Int(2^ceil(framelength))

# Extract the short-term Fourier transform of a signal.
function stft(T::Type, signal::Vector{<:Real};
              fftlen::Union{Integer, FFTLengthAutoConfig} = fftlen_auto,
              srate::Real = 16000,
              frameduration::Real = 0.025,
              framestep::Real = 0.01,
              removedc::Bool = true,
              dithering::Real = 0.,
              preemphasis::Real = 0.97,
              windowfn::WindowFunction = hamming,
              windowpower::Real = 1.0)

    # Copy the signal not to modify the original
    x = deepcopy(signal)

    # Get the signal in the requested floating point precision)
    x = convert(Vector{T}, x)

    # Dithering
    x .+= randn() .* dithering

    # Remove DC offset
    if removedc x .-= sum(x) / length(x) end

    # Split the signal in overlapping frames
    # X = frame length x no. frames
    X = hcat(frames(x, srate, frameduration, framestep)...)

    # Sharpen/flatten the window by exponentiating it
    window = windowfn(size(X, 1)) .^ windowpower

    # Iterate of the column, i.e. the frames of the signal
    for i in 1:size(X, 2)
        # Pre-emphasis
        px = vcat([X[1, i] ], X[1:end-1, i])
        X[:, i] .-= px * preemphasis

        # Windowing
        X[:, i] .*= window
    end

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
    S
end
stft(signal::Vector{<:Real}; kwargs...) = stft(Float64, signal; kwargs...)

