# Short Term Fourier Spectrum.
# Lucas Ondel, 2021

struct FFT
    fftlen::Int64
end
FFT(; siglen) = FFT(Int( 2^ceil(log2(siglen))))

Base.broadcastable(trans::FFT) = Ref(trans)

function (transform::FFT)(x::AbstractVector)
    pX = PaddedView(0, x, (transform.fftlen,))
    S = rfft(pX)

    # If the length of FFT is even then we discard the value at the
    # Nyquist frequency.
    if transform.fftlen % 2 == 0 S = S[1:end-1] end
end

