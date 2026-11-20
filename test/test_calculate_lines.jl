using SpectralLines, Test

struct NotImplemented <: LineSynthesisMethod end

@testset "unimplemented method errors" begin
    @test_throws ErrorException calculate_lines(range(0.0, 1.0, 9), [Line(0.5, 1.0, Gaussian(0.1))], NotImplemented())
end

@testset "in place, and an empty line list" begin
    xs = range(-1.0, 1.0, length = 9)
    lines = [Line(0.0, 1.0, Voigt(0.3, 0.1))]
    for method in (PointSample(), BinAverage(bin_width = 0.05), LDM(), SparseLDM())
        ret = ones(length(xs))
        @test calculate_lines!(ret, xs, empty(lines), method) === ret
        @test all(iszero, ret)
    end
end

@testset "descending grids" begin
    lines = [Line(1000.0, 1.0, Voigt(0.05, 0.02))]
    xd = range(1001.0, 999.0, length = 2001)
    for method in (PointSample(wing_cut = 10.0), BinAverage(bin_width = 0.001, wing_cut = 10.0), LDM(), SparseLDM())
        @test_throws ArgumentError calculate_lines(xd, lines, method)
    end
    @test_throws ArgumentError calculate_lines(collect(xd), lines, PointSample(wing_cut = 10.0))
    @test_throws ArgumentError calculate_lines(collect(xd), lines, Gaussian(0.01))
    @test_throws ArgumentError calculate_lines(xd, lines, LDM(), LDMWorkspace())
    # Pointwise methods accept any order.
    @test calculate_lines(xd, lines, BinAverage(bin_width = 0.001)) ≈
          reverse(calculate_lines(reverse(xd), lines, BinAverage(bin_width = 0.001)))
end

@testset "ret/xs index mismatch" begin
    xs = range(-1.0, 1.0, length = 9)
    lines = [Line(0.0, 1.0, Voigt(0.3, 0.1))]
    for method in (PointSample(), BinAverage(bin_width = 0.05), LDM(), SparseLDM()), n in (3, 20)
        @test_throws DimensionMismatch calculate_lines!(zeros(n), xs, lines, method)
    end
end
