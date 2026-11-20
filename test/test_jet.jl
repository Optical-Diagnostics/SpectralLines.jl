using SpectralLines, Test
using JET
import SpectralLines: faddeyeva, LineConvolution

JET.test_package(SpectralLines; target_modules = (SpectralLines,),
broken = VERSION >= v"1.14-"
)

const TM = (SpectralLines,)

@testset "line shapes" begin
    for s in (Gaussian(1.0), Lorentzian(1.0), Voigt(1.0, 0.5), Triangular(1.0),
              PseudoVoigt(1.0, 0.5), PseudoVoigt(1.0, 0.5, Rocco()))
        @test_call target_modules = TM s(1.0)
        @test_call target_modules = TM integrate(s, -1.0, 1.0)
    end
    @test_call target_modules = TM faddeyeva(1.0 + 0.5im)
end

@testset "calculate_lines" begin
    xs = collect(range(-5.0, 5.0, length = 50))
    xr = range(-5.0, 5.0, length = 50)
    lines = [Line(float(i), 1.0, Lorentzian(0.1)) for i in 1:5]
    for (x, m) in ((xs, PointSample(wing_cut = 10.0)), (xs, BinAverage(bin_width = 0.1)), (xr, LDM()), (xr, SparseLDM()),
                   (xs, Gaussian(0.5)))
        @test_call target_modules = TM calculate_lines(x, lines, m)
    end
    @test_call target_modules = TM calculate_lines(xs, lines, Gaussian(0.5), LineConvolution())
end
