# Common window functions
# Window function for the STFT

"""
    HannWindow([T=Float64,] N)

Create a [Hann window](https://en.wikipedia.org/wiki/Window_function#Hann_and_Hamming_windows)
of length `N` samples.
"""
function HannWindow(T::Type, N::Int)
    T(.5) .* (1 .- cos.(T(2π) .* Vector{T}(0:N-1) ./ (N-1)))
end
HannWindow(N::Int) = HannWindow(Float64, N)

"""
    HammingWindow([T=Float64,] N)

Create a [Hamming window](https://en.wikipedia.org/wiki/Window_function#Hann_and_Hamming_windows)
of length `N` samples.
"""
function HammingWindow(T::Type, N::Int)
    T(0.54) .- T(0.46) .* cos.(T(2π) .* Vector{T}(0:N-1) ./ (N-1))
end
HammingWindow(N::Int) = HammingWindow(Float64, N)

"""
    RectangularWindow([T=Float64,] N)

Create a [rectangular window](https://en.wikipedia.org/wiki/Window_function#Rectangular_window)
of length `N` samples.
"""
RectangularWindow(T::Type, N::Int) = ones(T, N)
RectangularWindow(N::Int) = RectangularWindow(Float64, N)

