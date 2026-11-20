using SpectralLines, Test
using QuadGK, Random
import SpectralLines: grid_chunked!, bin_serial!

@testset "area conservation" begin
    # With bin_width equal to the step, the averages times Δx sum to the
    # integral over [xs[1] − Δx/2, xs[end] + Δx/2].
    xs = collect(range(-20.0, 20.0; length = 4001))
    Δx = xs[2] - xs[1]
    for shape in (Lorentzian(0.5), Gaussian(0.02), Voigt(0.3, 0.1))
        line = Line(0.37, 2.0, shape)
        expected = quadgk(line, xs[1] - Δx / 2, line.position, xs[end] + Δx / 2; rtol = 1e-12)[1]
        @test sum(calculate_lines(xs, [line], BinAverage(bin_width = Δx))) * Δx ≈ expected rtol = 1e-6
    end
end

@testset "bins wider than the line" begin
    xs = range(-5.0, 5.0, step = 1.0)
    for shape in (Voigt(1e-3, 1e-5), Triangular(0.8))
        @test sum(calculate_lines(xs, [Line(0.13, 1.0, shape)], BinAverage(bin_width = 1.0))) ≈ 1 rtol = 1e-5
    end
end

@testset "narrow-bin limit" begin
    xs = collect(range(-20.0, 20.0; length = 6401))
    lines = [Line(0.0, 1.0, Lorentzian(0.5))]
    s = calculate_lines(xs, lines, PointSample())
    @test maximum(abs, calculate_lines(xs, lines, BinAverage(bin_width = xs[2] - xs[1])) .- s) < 1e-3 * maximum(s)
end

@testset "min_amplitude" begin
    xs = collect(range(-20.0, 20.0; length = 2001))
    strong = Line(-5.0, 1.0, Lorentzian(0.5))
    weak = Line(5.0, 1e-6, Lorentzian(0.5))
    gain = Line(0.0, -1.0, Lorentzian(0.5))   # a population inversion is strong
    for m in (PointSample(min_amplitude = 1e-3), BinAverage(bin_width = xs[2] - xs[1], min_amplitude = 1e-3))
        without = calculate_lines(xs, [strong], m)
        @test calculate_lines(xs, [strong, weak], m) == without
        @test minimum(calculate_lines(xs, [strong, gain], m)) < 0
    end
end

@testset "show" begin
    @test sprint(show, BinAverage(bin_width = 0.5, wing_cut = 3.0)) == "BinAverage(bin_width=0.5, wing_cut=3.0)"
end

@testset "allocations" begin
    xs = collect(range(-10.0, 10.0; length = 2000))
    lines = [Line(Float64(i), 1.0, Voigt(0.3, 0.2)) for i in 1:20]
    ret = zeros(length(xs))
    method = BinAverage(bin_width = xs[2] - xs[1], wing_cut = 10.0)
    calculate_lines!(ret, xs, lines, method)
    @test (@allocated calculate_lines!(ret, xs, lines, method)) == 0
end

@testset "chunked vs. serial" begin
    Random.seed!(11)
    lines = [Line(-4 + 8 * rand(), rand(), Voigt(0.3 + rand(), 0.1 + rand())) for _ in 1:61]
    for method in (BinAverage(bin_width = 0.05), BinAverage(bin_width = 0.05, wing_cut = 10.0, min_amplitude = 0.3))
        for xs in (range(-5.0, 5.0, length = 509), collect(range(-5.0, 5.0, length = 509)))
            ref = bin_serial!(zeros(length(xs)), xs, lines, method)
            for nchunks in (3, 32)
                @test grid_chunked!(zeros(length(xs)), xs, lines, method, nchunks) == ref
            end
            @test calculate_lines(xs, lines, method) == ref
        end
    end
end
