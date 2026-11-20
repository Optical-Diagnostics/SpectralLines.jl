using SpectralLines, Test
using QuadGK
using SpecialFunctions: faddeeva

@testset "Faddeeva definition" begin
    # Checks the scaling; faddeyeva's own accuracy is tested in test_faddeyeva.jl.
    for σ in (0.01, 1.0, 100.0), γ in (0.01, 1.0, 100.0), x in (0.5, 10.0)
        expected = real(faddeeva((x + im * γ) / σ / sqrt(2))) / σ / sqrt(2π)
        @test Voigt(σ, γ)(x) ≈ expected rtol = 1e-4
    end
end

@testset "zero widths" begin
    @test Voigt(0.0, 0.35)(0.3) === Lorentzian(0.35)(0.3)
    @test Voigt(0.4, 0.0)(0.3) === Gaussian(0.4)(0.3)
    @test Voigt(1, 2) isa Voigt{Float64}
end

@testset "integrate: additivity" begin
    v = Voigt(1.0, 0.01)
    hw = hwhm(v)
    @test sum(integrate(v, (i - 1.5) * hw, (i - 0.5) * hw) for i in -50:50) ≈ 1 atol = 2e-3
    v = Voigt(1.0, 2.0)
    @test integrate(v, -5.0, 0.0) + integrate(v, 0.0, 5.0) ≈ integrate(v, -5.0, 5.0) atol = 1e-15
end

@testset "integrate over bins wider than the line" begin
    for (σ, γ) in ((0.1, 0.0001), (0.02, 0.2)), k in (0.5, 4.0, 16.0)
        v = Voigt(σ, γ)
        f = fwhm(v)
        Δ = k * f
        @test integrate(v, -Δ / 2, Δ / 2) ≈ quadgk(v, -Δ / 2, Δ / 2; rtol = 1e-12)[1] rtol = 1e-3
        @test integrate(v, f, f + Δ) ≈ quadgk(v, f, f + Δ; rtol = 1e-12)[1] rtol = 1e-3
    end
end

@testset "integrate over very wide intervals" begin
    # Gaussian-dominated profiles (y < 0.7), out to 10⁶ FWHM.
    for γ in (1e-4, 0.98)
        v = Voigt(1.0, γ)
        f = fwhm(v)
        for (a, b) in ((-5000f, 5000f), (-1e6f, 1e6f), (500f, 1500f), (1200f, 5e4f))
            ref = a < 0 ? quadgk(v, a, 0.0, b; rtol = 1e-12)[1] : quadgk(v, a, b; rtol = 1e-12)[1]
            @test integrate(v, a, b) ≈ ref rtol = 1e-5
        end
    end
end
