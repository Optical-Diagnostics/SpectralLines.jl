using SpectralLines, Test
using QuadGK
import SpectralLines: numtype_convert

# Properties every shape shares. Voigt is here in both integrate branches
# (y = γ/(σ√2) below and above 0.7).
const SHAPES = (Gaussian(0.3), Lorentzian(0.3), Voigt(1.0, 0.3), Voigt(1.0, 2.0),
                PseudoVoigt(0.3, 0.2), Triangular(0.3))

@testset "$(nameof(typeof(s)))" for s in SHAPES
    f = fwhm(s)
    @test quadgk(s, -Inf, -f, 0.0, f, Inf; rtol = 1e-10)[1] ≈ 1 atol = 1e-6
    @test s(f / 2) / s(0.0) ≈ 0.5 rtol = 1e-3
    @test hwhm(s) ≈ f / 2
    @test s(0.7f) == s(-0.7f)
    @test integrate(s, -Inf, Inf) ≈ 1 rtol = 1e-5
    for (a, b) in ((-0.1f, 0.2f), (0.5f, 3.0f), (-4.0f, 0.0))
        @test integrate(s, a, b) ≈ quadgk(s, a, 0.0, b; rtol = 1e-12)[1] rtol = 1e-4
    end
    # Float32 in, Float32 out.
    @test numtype_convert(Float32, s)(0.1f0) isa Float32
end

@testset "closed forms" begin
    @test Gaussian(1.0)(1.0) ≈ exp(-0.5) / sqrt(2π) rtol = 1e-14
    @test Lorentzian(1.0)(2.0) ≈ 1 / 5π rtol = 1e-14
    @test Triangular(1.0)(0.5) ≈ 0.5
    @test Triangular(1.0)(1.0) == 0
end

@testset "integrate in the far wings" begin
    gauss(x) = exp(-x^2 / 2) / sqrt(2 * big(π))
    lorentz(x) = 1 / (big(π) * (1 + x^2))
    for (s, ref, a, b) in ((Gaussian(1.0), gauss, 7.0, 7.01), (Lorentzian(1.0), lorentz, -1e5 - 1, -1e5))
        @test integrate(s, a, b) ≈ Float64(quadgk(ref, big(a), big(b); rtol = 1e-30)[1]) rtol = 1e-12
    end
end

@testset "Triangular" begin
    w = 0.8
    t = Triangular(w)
    @test convolution_support(t) == (-w, w)
    cdf(x) = x <= -w ? 0.0 : x < 0 ? (x + w)^2 / (2w^2) : x < w ? 1 - (w - x)^2 / (2w^2) : 1.0
    for (a, b) in ((-0.5, 0.3), (-0.79, -0.2), (-2.0, -0.9))
        @test integrate(t, a, b) ≈ cdf(b) - cdf(a) atol = 1e-15
    end
    @test integrate(t, w - 1e-6, w) ≈ (1e-6)^2 / (2w^2) rtol = 1e-9
end
