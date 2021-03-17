# Signal preprocessing (before the the spectral analysis)
# Lucas Ondel, 2021

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

