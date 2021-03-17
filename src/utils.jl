# Basic DSP operations.

#######################################################################
# DCT bases for the cosine transform

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

#######################################################################
# Liftering

function lifter(T::Type, N::Int, L::Real)
    t = Vector{T}(1:N)
    1 .+ T(L/2) * sin.(π * t / L)
end
lifter(N, L) = lifter(Float64, N, L)

#######################################################################
# delta coefficient of a matrix of features

function delta(T::Type, x::AbstractVector, deltawin::Int = 2)
    N = length(x)
    y = zeros(T, N)
    norm = 2*sum(collect(1:deltawin).^2)
    for n in 1:N
        for θ in 1:deltawin
            y[n] += (θ * (x[min(N, n+θ)] - x[max(1, n-θ)])) / norm
        end
    end
    y
end
delta(x; deltawin = 2) = delta(Float64, x, deltawin)

