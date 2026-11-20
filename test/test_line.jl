using SpectralLines, Test
using QuadGK

@testset "evaluate, integrate, convert" begin
    for shape in (Gaussian(0.5), PseudoVoigt(0.5, 0.3, Rocco()))
        l = Line(1, 2, shape)
        @test l isa Line{Float64,Float64}
        @test l(1.3) ≈ 2 * shape(0.3) rtol = 1e-14
        @test integrate(l, -0.2, 1.4) ≈ quadgk(l, -0.2, 1.4; rtol = 1e-12)[1] rtol = 1e-6
    end
end

@testset "update_lines!" begin
    mkl() = [Line(0.5, 1.0, Voigt(0.20, 0.10)),
             Line(1.5, 2.0, Voigt(0.30, 0.10)),
             Line(2.5, 3.0, Voigt(0.40, 0.20))]
    ps = [0.6, 1.4, 2.7]
    ss = [Voigt(0.25, 0.12), Voigt(0.35, 0.15), Voigt(0.45, 0.22)]

    lines = mkl()
    @test update_lines!(lines; positions = ps, amplitudes = [9, 8, 7], shapes = ss) === lines
    @test lines == [Line(p, a, s) for (p, a, s) in zip(ps, [9.0, 8.0, 7.0], ss)]

    # One kwarg keeps the other fields, also on a mixed-shape vector.
    mixed = [Line(0.5, 1.0, Gaussian(0.2)), Line(1.5, 2.0, Lorentzian(0.4))]
    @test update_lines!(copy(mixed); amplitudes = [3.0, 4.0]) ==
          [Line(0.5, 3.0, Gaussian(0.2)), Line(1.5, 4.0, Lorentzian(0.4))]

    # The element type cannot change in place.
    @test_throws InexactError update_lines!(mkl(); amplitudes = [1.5im, 2.0im, 3.0im])
    @test_throws MethodError update_lines!(mkl(); shapes = [Gaussian(0.1), Gaussian(0.2), Gaussian(0.3)])
    @test_throws DimensionMismatch update_lines!(mkl(); positions = ps, amplitudes = [1.0, 2.0])
end
