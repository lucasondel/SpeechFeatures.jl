
using Documenter
using PyCall
using SpeechFeatures
using Test

DocMeta.setdocmeta!(SpeechFeatures, :DocTestSetup, :(using SpeechFeatures),
                    recursive = true)

doctest(SpeechFeatures)

#######################################################################
# Utilities

@testset "Utils" begin
    x = Vector(1:10)
    f1 = collect(SpeechFeatures.eachframe(x; srate=10, frameduration=0.3, framestep=0.2))
    f2 = [[1, 2, 3], [3, 4, 5], [5, 6, 7], [7, 8, 9]]
    @test all(f1 .== f2)

    lifter1 = SpeechFeatures.makelifter(10, 22)
    lifter2 = [2.56546322,  4.09905813,  5.56956514,  6.94704899,  8.20346807,
               9.31324532, 10.25378886, 11.00595195, 11.55442271, 11.88803586]
    @test all(lifter1 .≈ lifter2)

    X = Float64[1 2 3; 2 3 4]
    Y1 = SpeechFeatures.delta(X)
    println(Y1)
    Y2 = [5/10 6/10 5/10; 5/10 6/10 5/10]
    @test all(Y1 .≈ Y2)
end

#######################################################################
# Window functions

@testset "Window functions" begin
    N = 10

    w1 = SpeechFeatures.RectangularWindow(N)
    w2 = ones(N)
    @test all(w1 .≈ w2)
    w1 = SpeechFeatures.RectangularWindow(Float32, N)
    @test eltype(w1) ==  Float32

    w1 = SpeechFeatures.HannWindow(N)
    w2 = 0.5 .* (1 .- cos.(2π .* Vector(0:N-1) ./ (N-1) ))
    @test all(w1 .≈ w2)
    w1 = SpeechFeatures.HannWindow(Float32, N)
    @test eltype(w1) ==  Float32

    w1 = SpeechFeatures.HammingWindow(N)
    w2 = 0.54 .- 0.46 .* cos.(2π .* Vector(0:N-1) ./ (N-1) )
    @test all(w1 .≈ w2)
    w1 = SpeechFeatures.HammingWindow(Float32, N)
    @test eltype(w1) ==  Float32
end

#######################################################################
# Filter bank

py"""
import numpy as np

def create_filter(num, fft_len, lo_freq, hi_freq, samp_freq):
    filter_num = num
    filter_mat = np.zeros((fft_len // 2, filter_num))

    mel2freq = lambda mel: 700.0 * (np.exp(mel / 1127.0) - 1)
    freq2mel = lambda freq: 1127 * (np.log(1 + (freq / 700.0)))

    lo_mel = freq2mel(lo_freq);
    hi_mel = freq2mel(hi_freq);

    mel_c = np.linspace(lo_mel, hi_mel, filter_num + 2)
    freq_c = mel2freq(mel_c);

    point_c = freq_c / float(samp_freq) * fft_len
    point_c = np.floor(point_c).astype('int')

    for f in range(filter_num):
        d1 = point_c[f + 1] - point_c[f]
        d2 = point_c[f + 2] - point_c[f + 1]

        filter_mat[point_c[f]:point_c[f + 1] + 1, f] = np.linspace(0, 1, d1 + 1)
        filter_mat[point_c[f + 1]:point_c[f + 2] + 1, f] = np.linspace(1, 0, d2 + 1)

    return filter_mat
"""

@testset "FBANK" begin

    m = 12.75
    f = 100.12
    @test SpeechFeatures.mel2freq(m) ≈ 700 * (exp(m / 1127) - 1)
    @test typeof(SpeechFeatures.mel2freq(Float32(m))) == Float32
    @test SpeechFeatures.freq2mel(f) ≈ 1127 * log(1 + (f / 700))
    @test typeof(SpeechFeatures.freq2mel(Float32(f))) == Float32

    fbank1 = SpeechFeatures.FilterBank(26; srate = 16000, fftlen = 512,
                                       lofreq = 80, hifreq = 7600);
    fbank2 = py"create_filter(26, 512, 80, 7600, 16000)"
    @test all(fbank1 .≈ fbank2')
end
