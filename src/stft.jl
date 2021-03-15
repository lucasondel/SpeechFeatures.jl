# Short Term Fourier Spectrum.
# Lucas Ondel, 2021

struct STFT{W<:WindowFunction}
    srate::Int64
    frameduration::Float64
    framestep::Float64
    removedc::Bool
    dithering::Float64
    preemphasis::Float64
    window::Vector{Float64}
    windowpower::Float64
end


function STFT(;
              srate = 16000,
              frameduration = 0.025,
              framestep = 0.01,
              removedc = true,
              dithering = 0.,
              preemphasis = 0.97,
              windowfn = hamming,
              windowpower = 1.0)
    STFT(srate, frameduration, framestep, removedc, dithering,
         preemphasis, windowfn, windowpower)
end


function (stft::STFT)(signal)
    # Dithering: add negligeable noise to the signal to avoid 0 energy.
    x = signal .+ randn(length(signal)) .* stft.dithering

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

    # Set the FFT length to the nearest power of 2.
    fftlen = Int(2^ceil(log2(framelength)))

    # Pad the frames with 0 to get the correct length FFT
    pX = PaddedView(0, X, (fftlen, size(X, 2)))

    # Compute (half of) the Fourier transform over the first dimension
    S = rfft(pX, 1)

    # If the length of FFT is even then we discard the value at the
    # Nyquist frequency.
    if fftlen % 2 == 0
        S = S[1:end-1, :]
    end
end

