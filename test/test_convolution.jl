using SpectralLines, Test
import SpectralLines: SpectrumConvolution, LineConvolution, conv_full

@testset "convolve" begin
    g, l, v = Gaussian(0.3), Lorentzian(0.2), Voigt(1.0, 0.5)
    @test convolve(g, Gaussian(0.4)).sigma ≈ 0.5
    @test convolve(l, Lorentzian(0.5)).hwhm ≈ 0.7
    @test convolve(g, l) == convolve(l, g) == Voigt(0.3, 0.2)
    @test convolve(v, g) == convolve(g, v)
    @test convolve(v, g).sigma ≈ sqrt(1.0^2 + 0.3^2)
    @test convolve(v, l).gamma ≈ 0.7
    @test convolve(v, Voigt(0.4, 0.1)).gamma ≈ 0.6
    @test convolve(g, nothing) === g
    line = Line(1.5, 2.0, g)
    @test convolve(line, l) == Line(1.5, 2.0, convolve(g, l))
    @test all(n -> Base.ispublic(SpectralLines, n), (:SpectrumConvolution, :LineConvolution, :default_fftw_threads))
end

@testset "calculate_lines with an instrument function" begin
    xs = collect(range(-10.0, 10.0; step = 0.01))
    lines = [Line(0.0, 1.0, Gaussian(0.2))]
    instrument = Gaussian(0.5)
    expected = convolve(Gaussian(0.2), instrument).(xs)
    @test calculate_lines(xs, lines, instrument, LineConvolution()) ≈ expected atol = 1e-10
    @test calculate_lines(xs, lines, instrument) ≈ expected atol = 1e-4

    xs = collect(range(-3.0, 3.0; step = 0.001))
    result = calculate_lines(xs, [Line(0.0, 1.0, Gaussian(0.01))], Triangular(1.0))
    @test maximum(abs, result .- Triangular(1.0).(xs)) < 1e-2
end

@testset "SpectrumConvolution at the grid edges" begin
    # Lines near and beyond the edges.
    σl, σi = 0.20, 0.15
    combined = Gaussian(sqrt(σl^2 + σi^2))
    xs = range(-4.0, 4.0, length = 401)
    for pos in (-3.9, -4.6, -7.0, 4.6)
        result = calculate_lines(xs, [Line(pos, 1.0, Gaussian(σl))], Gaussian(σi))
        @test result ≈ combined.(xs .- pos) atol = 1e-12
    end
end

@testset "keyword arguments" begin
    xs = range(-4.0, 4.0, length = 401)
    lines = [Line(0.0, 1.0, Gaussian(0.2))]
    @test_throws MethodError calculate_lines(xs, lines, Gaussian(0.15); nonsense_kwarg = 42)
    @test calculate_lines(xs, lines, Gaussian(0.15); wing_cut = 50.0) ≈
          calculate_lines(xs, lines, Gaussian(0.15)) atol = 1e-9
end

@testset "bad grids" begin
    lines = [Line(0.0, 1.0, Gaussian(0.2))]
    @test_throws ArgumentError calculate_lines([0.0, 0.1, 0.5, 0.6], lines, Gaussian(0.15))
    # Too coarse to resolve the narrowest line.
    @test_throws ArgumentError calculate_lines(range(-4.0, 4.0, length = 5), lines, Gaussian(0.15))
end

@testset "conv_full" begin
    naive(u, v) = [sum(u[i] * v[k - i + 1] for i in max(1, k + 1 - length(v)):min(k, length(u)))
                   for k in 1:(length(u) + length(v) - 1)]
    u = sin.(0.1 .* (1:100))
    v = exp.(-abs2.(range(-2.0, 2.0, length = 31)))
    @test conv_full(u, v) ≈ naive(u, v) rtol = 1e-12
    @test conv_full([1, 2, 3], [1, 1]) == [1, 3, 5, 3]
end
