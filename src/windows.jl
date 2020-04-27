# Common window functions

export hann
export hamming
export rectangular

abstract type WindowFunction <: Function end

struct HannWindow <: WindowFunction end
struct HammingWindow <: WindowFunction end
struct RectangularWindow <: WindowFunction end

"""
    hann([T=Float64,] N)

Create a [Hann window](https://en.wikipedia.org/wiki/Window_function#Hann_and_Hamming_windows)
of length `N` samples.
"""
const hann = HannWindow()
(::HannWindow)(T::Type, N::Int64) = sin.((π .* Vector{T}(1:N)) / N) .^2
(::HannWindow)(N::Int64) = hann(Float64, N)


"""
    HammingWindow([T=Float64,] N)

Create a [Hamming window](https://en.wikipedia.org/wiki/Window_function#Hann_and_Hamming_windows)
of length `N` samples.
"""
const hamming = HammingWindow()
function (::HammingWindow)(T::Type, N::Int64)
    a₀ = 25. /46
    a₁ = 1 - a₀
    n = Vector{T}(1:N)
    a₀ .- a₁ * cos.(2π * n / N)
end
(::HammingWindow)(N::Int64) = hamming(Float64, N)

"""
    RectangularWindow([T=Float64,] N)

Create a [rectangular window](https://en.wikipedia.org/wiki/Window_function#Rectangular_window)
of length `N` samples.
"""
const rectangular = RectangularWindow()
(::RectangularWindow)(T::Type, N::Int64) = zeros(T, N)
(::RectangularWindow)(N::Int64) = rectangular(Float64, N)

