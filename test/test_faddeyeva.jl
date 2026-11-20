using SpectralLines, Test
using SpecialFunctions: faddeeva
using ForwardDiff
import SpectralLines: faddeyeva

# Reference: SpecialFunctions.faddeeva (S. G. Johnson's Faddeeva package),
# accurate to ~1e-13.
ref(x, y) = faddeeva(complex(x, y))

# A grid over the domain of Zaghloul (2017), y from 1e-20 to 1e4, plus points on
# every region border s = x² + y².
const XS = vcat(range(-60, 60, length = 481), [sqrt(s) for s in (2.5, 30, 62, 256, 3.8e4)])
const YS = exp10.(range(-20, 4, length = 97))

@testset "accuracy in the upper half-plane" begin
    err_re = 0.0
    err_im = 0.0
    for x in XS, y in YS
        r = ref(x, y)
        w = faddeyeva(complex(x, y))
        err_re = max(err_re, abs(real(w) - real(r)) / abs(real(r)))
        err_im = max(err_im, abs(imag(w) - imag(r)) / abs(r))
    end
    @test err_re < 4e-5
    @test err_im < 4e-5
end

@testset "Weideman region" begin
    err = 0.0
    for x in range(-5.4, 5.4, length = 217), y in range(0.0, 5.4, length = 109)
        s = x^2 + y^2
        (s < 30 && !(s >= 2.5 && y^2 < 0.072)) || continue
        err = max(err, abs(faddeyeva(x, y) - real(ref(x, y))) / abs(real(ref(x, y))))
    end
    @test err < 3e-6
end

@testset "real-part form" begin
    err = maximum(abs(faddeyeva(x, y) - real(faddeyeva(complex(x, y)))) /
                  abs(real(faddeyeva(complex(x, y)))) for x in XS[1:4:end], y in YS)
    @test err < 1e-12
end

@testset "lower half-plane" begin
    err = 0.0
    for x in range(-6, 6, length = 49), y in -exp10.(range(-6, 0.5, length = 14))
        r = ref(x, y)
        err = max(err, abs(faddeyeva(complex(x, y)) - r) / abs(r), abs(faddeyeva(x, y) - real(r)) / abs(r))
    end
    @test err < 1e-4
end

@testset "types and extreme arguments" begin
    @test faddeyeva(1.0f0, 0.5f0) isa Float32
    @test faddeyeva(complex(1.0f0, 0.5f0)) isa ComplexF32
    @test faddeyeva(1.0f0, 0.5f0) ≈ real(ref(1.0, 0.5)) rtol = 1e-5
    @test faddeyeva(1, 0) ≈ real(ref(1.0, 0.0)) rtol = 4e-5
    # |z|² overflows; i/(√π z) must not.
    @test imag(faddeyeva(complex(1e160, 1.0))) ≈ 1 / (sqrt(π) * 1e160) rtol = 1e-12
    @test faddeyeva(Inf, 1.0) == 0
    @test faddeyeva(complex(Inf, 1.0)) == 0
    @test isnan(faddeyeva(NaN, 1.0))
end

@testset "ForwardDiff derivatives" begin
    err = 0.0
    for x in range(-8, 8, length = 161), y in vcat(-0.3, exp10.(range(-12, 2, length = 29)))
        r = ref(x, y)
        wp = -2 * complex(x, y) * r + 2im / sqrt(π)
        dx = ForwardDiff.derivative(t -> faddeyeva(t, y), x)
        dy = ForwardDiff.derivative(t -> faddeyeva(x, t), y)
        err = max(err, abs(dx - real(wp)) / abs(wp), abs(dy + imag(wp)) / abs(wp))
    end
    @test err < 1e-4
    # Complex form and second derivatives (nested duals).
    x, y = 0.7, 0.3
    r = ref(x, y)
    wp = -2 * complex(x, y) * r + 2im / sqrt(π)
    wpp = -2r - 2 * complex(x, y) * wp
    @test ForwardDiff.derivative(t -> imag(faddeyeva(complex(t, y))), x) ≈ imag(wp) rtol = 1e-4
    d2 = ForwardDiff.derivative(a -> ForwardDiff.derivative(t -> faddeyeva(t, y), a), x)
    @test d2 ≈ real(wpp) rtol = 1e-4
end
