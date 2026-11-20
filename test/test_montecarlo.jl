using SpectralLines, Test
using MonteCarloMeasurements

@testset "Voigt with uncertain widths" begin
    w = 1.0 ± 0.3
    for v in (Voigt(w, 0.5), Voigt(0.5, w), Voigt(w, w))
        ref = [Voigt(s, g)(0.2) for (s, g) in zip(v.sigma.particles, v.gamma.particles)]
        @test v(0.2).particles ≈ ref
    end
    spec = calculate_lines(-2:0.5:2, [Line(0.0, 1.0, Voigt(w, 0.5))], PointSample())
    @test spec[5].particles ≈ Voigt(w, 0.5)(0.0).particles
end
