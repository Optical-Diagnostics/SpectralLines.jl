using SpectralLines, Test
using QuadGK

const MODELS = (Thompson(), Kielkopf(), Olivero(), LiuLin(), Rocco())
const RATIOS = (0.01, 0.1, 0.3, 1.0, 3.0, 10.0, 100.0)   # γ/σ

# FWHM of the exact Voigt profile, by bisection on half maximum.
function voigt_fwhm(σ, γ)
    v = Voigt(σ, γ)
    half = v(0.0) / 2
    lo, hi = 0.0, 100.0 * (σ + γ)
    for _ in 1:200
        m = (lo + hi) / 2
        v(m) > half ? (lo = m) : (hi = m)
    end
    return 2lo
end

# Per-model accuracy, about 1.5× the measured worst case over RATIOS: the
# combined FWHM against the exact Voigt FWHM, and the profile against the exact
# Voigt profile as a fraction of its peak.
const FWHM_TOL = Dict(Thompson => 5e-3, Kielkopf => 3.5e-4, Olivero => 2e-4,
                      LiuLin => 3.5e-4, Rocco => 3.5e-4)
const PROFILE_TOL = Dict(Thompson => 2e-2, Kielkopf => 2e-2, Olivero => 2e-2,
                         LiuLin => 2e-2, Rocco => 5e-2)

@testset "accuracy per model" begin
    for m in MODELS
        fwhm_err = maximum(abs(fwhm(PseudoVoigt(1.0, r, m)) / voigt_fwhm(1.0, r) - 1) for r in RATIOS)
        @test fwhm_err < FWHM_TOL[typeof(m)]
        profile_err = maximum(RATIOS) do r
            pv, v = PseudoVoigt(1.0, r, m), Voigt(1.0, r)
            xs = range(-3fwhm(v), 3fwhm(v), length = 241)
            maximum(abs.(pv.(xs) .- v.(xs))) / v(0.0)
        end
        @test profile_err < PROFILE_TOL[typeof(m)]
    end
end

@testset "models are distinct" begin
    params = [(PseudoVoigt(1.0, 1.0, m).eta, fwhm(PseudoVoigt(1.0, 1.0, m))) for m in MODELS]
    @test all(abs(p[1] - q[1]) > 1e-5 || abs(p[2] - q[2]) > 1e-5
               for (i, p) in enumerate(params) for q in params[i+1:end])
    @test PseudoVoigt(1.0, 0.5)(0.3) == PseudoVoigt(1.0, 0.5, Thompson())(0.3)   # the default
end

@testset "limits γ → 0 and σ → 0" begin
    xs = range(-4.0, 4.0, length = 81)
    g, l = Gaussian(1.0), Lorentzian(1.0)
    for m in MODELS
        tol_g = m isa LiuLin ? 5e-4 : 1e-12
        tol_l = m isa LiuLin ? 5e-3 : 1e-5
        @test maximum(abs.(PseudoVoigt(1.0, 0.0, m).(xs) .- g.(xs))) / g(0.0) < tol_g
        @test maximum(abs.(PseudoVoigt(0.0, 1.0, m).(xs) .- l.(xs))) / l(0.0) < tol_l
    end
end

