
using Documenter
using SpeechFeatures
using Test

DocMeta.setdocmeta!(SpeechFeatures, :DocTestSetup, :(using SpeechFeatures),
                    recursive = true)

doctest(SpeechFeatures)

#######################################################################
# Utilities

@testset "Utils" begin
    x = Vector(1:10)
    f1 = collect(SpeechFeatures.frames(x, 10, 0.3, 0.2))
    f2 = [[1, 2, 3], [3, 4, 5], [5, 6, 7], [7, 8, 9]]
    @test all(f1 .== f2)

    DCT1 = SpeechFeatures.dctbases(2, 3)
    DCT2 = [8.66025404e-01 6.12323400e-17 -8.66025404e-01;
            0.5            -1.0            0.5]
    @test all(DCT1 .≈ DCT2)
    DCT1 = SpeechFeatures.dctbases(Float32, 2, 3)
    @test eltype(DCT1) == Float32

    lifter1 = SpeechFeatures.lifter(10, 22)
    lifter2 = [2.56546322,  4.09905813,  5.56956514,  6.94704899,  8.20346807,
               9.31324532, 10.25378886, 11.00595195, 11.55442271, 11.88803586]
    @test all(lifter1 .≈ lifter2)
    lifter1 = SpeechFeatures.lifter(Float32, 2, 3)
    @test eltype(lifter1) == Float32

    x = [1, 2, 3]
    y1 = SpeechFeatures.delta(x)
    y2 = [5/10, 6/10, 5/10]
    @test all(y1 .≈ y2)
    y1 = SpeechFeatures.delta(Float32, x)
    @test eltype(y1) == Float32
end

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

