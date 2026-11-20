# LDM, and what it shares with SparseLDM. SparseLDM-only tests are in
# test_sparse_ldm.jl.
using SpectralLines, Test
using Random
import SpectralLines: FFTW, init_width_axis, frac_index, lorentzian_step, fill_template!, voigt_ft

const METHODS = (LDM(), SparseLDM())

relmax(a, ref) = maximum(abs, a .- ref) / maximum(abs, ref)

@testset "zero-width bin" begin
    step_L = lorentzian_step(0.01)
    widths = [0.04, 0.05, 0.06]
    without_zero = init_width_axis(widths, step_L)
    with_zero = init_width_axis(vcat(0.0, widths), step_L)
    @test with_zero[2:end] ≈ without_zero
    @test 0 < with_zero[1] < 1e-6 * without_zero[1]
    @test length(init_width_axis(widths .* 1e4, step_L)) == length(without_zero)
end

@testset "frac_index" begin
    grid = [1.0, 2.0, 4.0]
    @test frac_index(1.5, grid) == (1, 2, 0.5)
    @test frac_index(0.5, grid) == (1, 1, 0.0)
    @test frac_index(10.0, grid) == (3, 3, 0.0)
    @test frac_index(1.0, [1.0]) == (1, 1, 0.0)
end

@testset "fill_template!" begin
    n = 16385
    Δ = 3.05e-4
    for (σ, γ) in ((0.5, 0.2), (0.01, 0.0), (0.0, 0.005), (0.0, 0.0))
        direct = [voigt_ft((i - 1) * Δ, σ, γ) for i in 1:n]
        rec = fill_template!(zeros(n), Δ, σ, γ, n)
        @test all(abs(rec[i] - direct[i]) <= 1e-7 * direct[i] + 1e-280 for i in 1:n)
    end
    n32 = 100001
    Δ32, σ32, γ32 = Float32(1 / (200002 * 0.01)), 0.2f0, 0.05f0
    rec32 = fill_template!(zeros(Float32, n32), Δ32, σ32, γ32, n32)
    direct32 = [voigt_ft((i - 1) * Float64(Δ32), Float64(σ32), Float64(γ32)) for i in 1:n32]
    @test maximum(abs, rec32 .- direct32) < 1e-6
end

@testset "random spread of widths" begin
    Random.seed!(42)
    xs = range(-100.0, 100.0; length = 4000)
    n = 300
    positions = rand(n) .* 160.0 .- 80.0
    amps = 0.5 .+ rand(n)
    sigmas = 0.05 .+ 0.5 .* rand(n)
    gammas = 0.02 .+ 0.3 .* rand(n)
    lines = [Line(positions[i], amps[i], Voigt(sigmas[i], gammas[i])) for i in 1:n]
    # SparseLDM drops the wings beyond ±10 FWHM; compare like with like.
    for (m, ref) in ((LDM(), calculate_lines(xs, lines, PointSample())),
                     (SparseLDM(), calculate_lines(xs, lines, PointSample(wing_cut = 20.0))))
        r = calculate_lines(xs, lines, m)
        @test relmax(r, ref) < 0.05
        @test sqrt(sum(abs2, r .- ref) / sum(abs2, ref)) < 0.01
    end
end

@testset "pure and zero-width shapes" begin
    xs = range(1000.0, 1002.0, length = 2048)
    lines = [Line(1000.5, 1.0, Voigt(0.01, 0.02)),
             Line(1001.2, 2.0, Voigt(0.012, 0.0)),   # γ = 0
             Line(1001.6, 0.5, Voigt(0.0, 0.03)),    # σ = 0
             Line(1000.3, 1.0, Gaussian(0.01)),
             Line(1001.4, 1.0, Lorentzian(0.02))]
    exact = calculate_lines(xs, lines, PointSample())
    for m in METHODS
        @test relmax(calculate_lines(xs, lines, m), exact) < 0.02
    end
end

@testset "Float32 accuracy" begin
    xs64 = range(0.0, 1000.0, length = 100001)
    xs32 = range(0.0f0, 1000.0f0, length = 100001)
    ps = range(100.0, 900.0, length = 50)
    exact = calculate_lines(xs64, [Line(p, 1.0, Gaussian(0.2)) for p in ps], PointSample())
    lines32 = [Line(Float32(p), 1.0f0, Gaussian(0.2f0)) for p in ps]
    for m in METHODS
        @test relmax(calculate_lines(xs32, lines32, m), exact) < 1e-3
    end
end

@testset "lines at and past the grid end" begin
    for (a, b, n) in ((999.0, 1001.0, 2001), (12.3, 45.6, 5000), (0.0f0, 10.0f0, 1001))
        xs = range(a, b, length = n)
        lines = [Line(last(xs), one(a), Gaussian(10 * step(xs)))]
        ref = calculate_lines(xs, lines, PointSample())
        for m in METHODS
            @test calculate_lines(xs, lines, m)[end] ≈ ref[end] rtol = 1e-3
        end
        outside = [Line(nextfloat(last(xs)), one(a), Gaussian(10 * step(xs)))]
        @test all(iszero, calculate_lines(xs, outside, LDM()))
    end
end

@testset "odd grid lengths" begin
    lines = [Line(0.0, 1.0, Voigt(0.3, 0.1)), Line(0.5, 2.0, Gaussian(0.2))]
    for n in (33, 129)
        xs = range(-2.0, 2.0, length = n)
        @test relmax(calculate_lines(xs, lines, LDM()), calculate_lines(xs, lines, PointSample())) < 0.02
    end
end

@testset "unsupported grids and number types" begin
    xs = collect(range(-50.0, 50.0; length = 200))
    lines = [Line(0.0, 1.0, Voigt(0.3, 0.2))]
    for m in METHODS
        @test_throws r"AbstractRange.*PointSample\(\)" calculate_lines(xs, lines, m)
        big_lines = [Line(big(0.0), big(1.0), Voigt(big(0.3), big(0.2)))]
        @test_throws r"PointSample.*BinAverage" calculate_lines(range(big(-1.0), big(1.0), 64), big_lines, m)
    end
end

@testset "FFTW thread count" begin
    previous = FFTW.get_num_threads()
    try
        FFTW.set_num_threads(3)
        xs = range(0.0, 10.0, length = 512)
        lines = [Line(5.0, 1.0, Voigt(0.05, 0.02))]
        for m in METHODS
            calculate_lines(xs, lines, m)
            @test FFTW.get_num_threads() == 3
            # Also after concurrent calls, each planning on its own workspace.
            for _ in 1:20
                @sync for _ in 1:4
                    Threads.@spawn calculate_lines(xs, lines, m)
                end
            end
            @test FFTW.get_num_threads() == 3
        end
    finally
        FFTW.set_num_threads(previous)
    end
end

@testset "show" begin
    @test sprint(show, LDM(res_G = 0.02, res_L = 0.03)) == "LDM(res_G=0.02, res_L=0.03)"
end

@testset "identical shapes" begin
    xs = range(0.0, 10.0, length = 512)
    ws = LDMWorkspace()
    for shape in (Voigt(0.05, 0.02), Gaussian(0.05), Lorentzian(0.02))
        calculate_lines(xs, [Line(x, 1.0, shape) for x in 1.0:8.0], LDM(), ws)
        @test count(ws.state[].occupied) == 1
    end
end
