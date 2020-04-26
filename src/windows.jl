# Common window functions

export HannWindow
export HammingWindow
export RectangularWindow

"""
    HannWindow([T=Float64,] N)

Create a [Hann window](https://en.wikipedia.org/wiki/Window_function#Hann_and_Hamming_windows)
of length `N` samples.
"""
HannWindow(T::Type, N::Int64) = sin.((π .* Vector{T}(1:N)) / N) .^2
HannWindow(N::Int64) = HannWindow(Float64, N)

"""
    HammingWindow([T=Float64,] N)

Create a [Hamming window](https://en.wikipedia.org/wiki/Window_function#Hann_and_Hamming_windows)
of length `N` samples.
"""
function HammingWindow(T::Type, N::Int64)
    a₀ = 25. /46
    a₁ = 1 - a₀
    n = Vector{T}(1:N)
    a₀ .- a₁ * cos.(2π * n / N)
end
HammingWindow(N::Int64) = HammingWindow(Float64, N)

"""
    RectangularWindow([T=Float64,] N)

Create a [rectangular window](https://en.wikipedia.org/wiki/Window_function#Rectangular_window)
of length `N` samples.
"""
RectangularWindow(T::Type, N::Int64) = ones(T, N)
RectangularWindow(N::Int64) = RectangularWindow(Float64, N)

