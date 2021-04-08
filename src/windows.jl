function HannWindow(T::Type, N::Int)
    T(.5) .* (1 .- cos.(T(2π) .* Vector{T}(0:N-1) ./ (N-1)))
end
HannWindow(N::Int) = HannWindow(Float64, N)

function HammingWindow(T::Type, N::Int)
    T(0.54) .- T(0.46) .* cos.(T(2π) .* Vector{T}(0:N-1) ./ (N-1))
end
HammingWindow(N::Int) = HammingWindow(Float64, N)

RectangularWindow(T::Type, N::Int) = ones(T, N)
RectangularWindow(N::Int) = RectangularWindow(Float64, N)

