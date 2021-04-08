# Generate the DCT bases, `d` is the number of samples per base
# and `n` the number of bases.
function dctbases(T::Type, N::Int, D::Int)
    retval = zeros(T, N, D)
    t = range(0, D-1, length = D) .+ 0.5
    for i = 1:N
        retval[i, :] = cos.(i * π * t / D)
    end
    retval
end
dctbases(N, D) = dctbases(Float64, N, D)

function lifter(T::Type, N::Int, L::Real)
    t = Vector{T}(1:N)
    1 .+ T(L/2) * sin.(π * t / L)
end
lifter(N, L) = lifter(Float64, N, L)

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

struct Preprocessor
    removedc::Bool
    dithering::Float64
end
Preprocessor(;removedc = true, dithering = 0.0) = Preprocessor(removedc, dithering)

function (preproc::Preprocessor)(x::AbstractVector)
    N = length(x)
    x̂ = x .+ randn(N) .* preproc.dithering
    if preproc.removedc x̂ .-= sum(x̂) / N end
    x̂
end

struct FFT
    fftlen::Int64
end
FFT(; siglen) = FFT(Int( 2^ceil(log2(siglen))))

function (transform::FFT)(X::AbstractMatrix)
    pX = PaddedView(0, X, (transform.fftlen, size(X,2)))
    S = rfft(pX, 1)

    # If the length of FFT is even then we discard the value at the
    # Nyquist frequency.
    if transform.fftlen % 2 == 0 S = S[1:end-1, :] end
end

