using SpectralLines, Test
using Random
import SpectralLines: grid_chunked!, sample_serial!

@testset "direct evaluation and wing_cut" begin
    xs = collect(range(-20.0, 20.0; length = 4001))
    lines = [Line(0.0, 1.0, Gaussian(0.5)), Line(2.0, 0.8, Lorentzian(0.5))]
    @test calculate_lines(xs, lines, PointSample()) ≈ [sum(l(x) for l in lines) for x in xs]
    tight = calculate_lines(xs, lines[2:2], PointSample(wing_cut = 2.0))
    @test all(iszero, tight[abs.(xs .- 2.0) .> 1.0])
    @test all(>(0), tight[abs.(xs .- 2.0) .< 0.99])
end

@testset "show" begin
    @test sprint(show, PointSample(wing_cut = 3.0, min_amplitude = 0.5)) == "PointSample(wing_cut=3.0, min_amplitude=0.5)"
end

@testset "allocations" begin
    xs = collect(range(-10.0, 10.0; length = 2000))
    lines = [Line(Float64(i), 1.0, Voigt(0.3, 0.2)) for i in 1:20]
    ret = zeros(length(xs))
    for method in (PointSample(), PointSample(wing_cut = 10.0))
        calculate_lines!(ret, xs, lines, method)
        @test (@allocated calculate_lines!(ret, xs, lines, method)) == 0
    end
end

@testset "chunked vs. serial" begin
    Random.seed!(7)
    lines = [Line(-4 + 8 * rand(), rand(), Voigt(0.3 + rand(), 0.1 + rand())) for _ in 1:97]
    for method in (PointSample(), PointSample(wing_cut = 8.0, min_amplitude = 0.2))
        for xs in (range(-5.0, 5.0, length = 1013), collect(range(-5.0, 5.0, length = 1013)))
            ref = sample_serial!(zeros(length(xs)), xs, lines, method)
            for nchunks in (3, 32)
                @test grid_chunked!(zeros(length(xs)), xs, lines, method, nchunks) == ref
            end
        end
    end
    xs = range(-5.0, 5.0, length = 6000)
    @test calculate_lines(xs, lines, PointSample()) == sample_serial!(zeros(length(xs)), xs, lines, PointSample())
end
