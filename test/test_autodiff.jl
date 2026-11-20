using SpectralLines, Test
using ForwardDiff

# Forward-mode AD through every synthesis method. A dual position or width on a
# plain Float64 grid must still give a dual spectrum.

const XS = collect(range(-5.0, 5.0, length = 201))
const XR = range(-5.0, 5.0, length = 201)

dual(x) = ForwardDiff.Dual{Nothing,Float64,1}(x, ForwardDiff.Partials((1.0,)))

# Central difference of a scalar function of a vector, in component j.
function fd(f, p, j; h = 1e-6)
    pp = copy(p); pp[j] += h
    pm = copy(p); pm[j] -= h
    return (f(pp) - f(pm)) / (2h)
end

# Two lines sharing parameters, so every partial reaches positions, amplitudes
# and both widths. Off-grid centre: the position derivative jumps at grid nodes.
mklines(p) = [Line(p[1], p[2], Voigt(p[3], p[4])),
              Line(p[1] + 0.9137, 0.5 * p[2], Voigt(1.3 * p[3], 0.8 * p[4]))]
const P0 = [0.3137, 1.0, 0.7, 0.5]
obj(xs, m) = p -> sum(abs2, calculate_lines(xs, mklines(p), m))

@testset "PointSample and BinAverage" begin
    for m in (PointSample(), PointSample(wing_cut = 30.0), BinAverage(bin_width = 0.05))
        g = ForwardDiff.gradient(obj(XS, m), P0)
        for j in eachindex(P0)
            @test g[j] ≈ fd(obj(XS, m), P0, j) rtol = 1e-4
        end
    end
end

@testset "every shape" begin
    for mk in (p -> Gaussian(p), p -> Lorentzian(p), p -> PseudoVoigt(p, 0.5),
               p -> PseudoVoigt(p, 0.5, LiuLin()), p -> Triangular(p))
        f = p -> sum(abs2, calculate_lines(XS, [Line(0.3137, 1.0, mk(p[1]))], PointSample()))
        @test ForwardDiff.gradient(f, [1.0])[1] ≈ fd(f, [1.0], 1) rtol = 1e-4
    end
end

@testset "dual amplitude only" begin
    @test eltype(calculate_lines(XS, [Line(0.3, dual(1.0), Voigt(1.0, 0.5))], PointSample())) <: ForwardDiff.Dual
end

@testset "LDM and SparseLDM" begin
    for m in (LDM(), SparseLDM(truncation = 3.0))
        g = ForwardDiff.gradient(obj(XR, m), P0)
        for j in eachindex(P0)
            @test g[j] ≈ fd(obj(XR, m), P0, j) rtol = 1e-4
        end
    end
    for m in (LDM(), SparseLDM(truncation = 3.0))
        @test ForwardDiff.value.(calculate_lines(XR, mklines(dual.(P0)), m)) ≈
              calculate_lines(XR, mklines(P0), m) rtol = 1e-14
        # Nested duals: the Hessian matches finite differences of the gradient.
        gσ = p -> ForwardDiff.gradient(obj(XR, m), p)[3]
        @test ForwardDiff.hessian(obj(XR, m), P0)[3, 3] ≈ fd(gσ, P0, 3; h = 1e-5) rtol = 1e-4
    end

    ws = SparseLDMWorkspace()
    @test ForwardDiff.jacobian(p -> calculate_lines(XR, mklines(p), SparseLDM(truncation = 3.0), ws), P0) ==
          ForwardDiff.jacobian(p -> calculate_lines(XR, mklines(p), SparseLDM(truncation = 3.0)), P0)
    D4 = ForwardDiff.Dual{Nothing,Float64,4}
    dl = mklines([D4(x, ForwardDiff.Partials(ntuple(i -> Float64(i == j), 4))) for (j, x) in enumerate(P0)])
    ret = zeros(D4, length(XR))
    calculate_lines!(ret, XR, dl, SparseLDM(truncation = 3.0), ws)
    @test @allocated(calculate_lines!(ret, XR, dl, SparseLDM(truncation = 3.0), ws)) == 0
end

@testset "Float32 duals" begin
    D = ForwardDiff.Dual{Nothing,Float32,1}
    xs32 = range(0.0f0, 100.0f0, length = 2^16)
    xs64 = range(0.0, 100.0, length = 2^16)
    for m in (LDM(), SparseLDM())
        d = D(50.0137f0, ForwardDiff.Partials((1.0f0,)))
        r = ForwardDiff.value.(calculate_lines(xs32, [Line(d, 1.0f0, Voigt(0.03f0, 0.02f0))], m))
        r64 = calculate_lines(xs64, [Line(50.0137, 1.0, Voigt(0.03, 0.02))], m)
        @test maximum(abs, r .- r64) < 1e-4 * maximum(r64)
    end
end

@testset "duals over unsupported number types" begin
    d = ForwardDiff.Dual{Nothing,BigFloat,1}(big(0.3), ForwardDiff.Partials((big(1.0),)))
    lines = [Line(d, big(1.0), Voigt(big(1.0), big(0.5)))]
    for m in (LDM(), SparseLDM())
        @test_throws r"PointSample.*BinAverage" calculate_lines(XR, lines, m)
    end
end
