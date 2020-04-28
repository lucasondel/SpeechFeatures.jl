# Filter bank

export FilterBank

mel2freq(mel::Real) = 700. * (exp(mel / 1127.) - 1)

freq2mel(freq::Real) = 1127. * (log(1 + (freq / 700.)))

# Create a set of triangular filters
function FilterBank(T::Type, n::Integer; srate::Real, fftlen::Integer,
                    lofreq::Real, hifreq::Real)
    # Convert the cut-off frequencies into mel
    lomel = freq2mel(lofreq)
    himel = freq2mel(hifreq)

    # Centers (in mel and freqs) of the filterbank
    melcenters = range(lomel, himel, length = n + 2)
    freqcenters = mel2freq.(melcenters)

    # Now get the centers in terms of FFT bins
    bincenters = Int64.(floor.( fftlen .* freqcenters ./ srate ))

    # Allocate the matrix which will store the filters
    D = Int64(ceil(fftlen / 2))
    F = zeros(T, D, n)

    # Construct the "triangle"
    for f = 1:n
        d1 = bincenters[f + 1] - bincenters[f]
        d2 = bincenters[f + 2] - bincenters[f + 1]

        s = bincenters[f]
        e = bincenters[f + 1]
        F[s:e, f] = range(0, 1, length = d1 + 1)

        s = bincenters[f + 1]
        e = bincenters[f + 2]
        F[s:e, f] = range(1, 0, length = d2 + 1)
    end
    F
end
FilterBank(n::Integer, args...) = FilterBank(Float64, n, args...)

