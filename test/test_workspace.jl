using SpectralLines, Test
using ForwardDiff, Random

Random.seed!(42)

mkline(f) = Line(1000 + 100 * rand(), f * rand(), Voigt(0.01 + 0.01 * rand(), 0.005 + 0.01 * rand()))
mklines(n; f = 1.0) = [mkline(f) for _ in 1:n]

closeto(a, b) = isapprox(a, b; rtol = 1e-12)

const PAIRS = ((LDM(), LDMWorkspace()), (SparseLDM(), SparseLDMWorkspace()))

@testset "with and without workspace" begin
    for (m, ws) in PAIRS
        xs = range(1000.0, 1100.0, length = 512)
        lines = mklines(500)
        ref = calculate_lines(xs, lines, m)
        @test closeto(calculate_lines(xs, lines, m, ws), ref)
        lines2 = mklines(500; f = 2.0)
        @test closeto(calculate_lines(xs, lines2, m, ws), calculate_lines(xs, lines2, m))
        # Grid length and line count grow, then shrink back.
        xs3 = range(1000.0, 1100.0, length = 700)
        lines3 = mklines(2000)
        @test closeto(calculate_lines(xs3, lines3, m, ws), calculate_lines(xs3, lines3, m))
        @test closeto(calculate_lines(xs, lines, m, ws), ref)
        # Float32 rebuilds the state.
        xs32 = range(1000.0f0, 1100.0f0, length = 512)
        l32 = [Line(Float32(l.position), Float32(l.amplitude), Voigt(Float32(l.shape.sigma), Float32(l.shape.gamma)))
               for l in lines]
        @test calculate_lines(xs32, l32, m, ws) ≈ calculate_lines(xs32, l32, m) rtol = 1e-5
        ret = ones(length(xs))
        calculate_lines!(ret, xs, empty(lines), m, ws)
        @test all(iszero, ret)
    end
end

function alloc_after_warmup(ret, xs, lines, method, ws)
    calculate_lines!(ret, xs, lines, method, ws)
    calculate_lines!(ret, xs, lines, method, ws)
    return @allocated calculate_lines!(ret, xs, lines, method, ws)
end

@testset "steady-state allocations" begin
    xs = range(1000.0, 1100.0, length = 512)
    lines = mklines(2000)
    for (m, ws) in PAIRS
        @test alloc_after_warmup(zeros(length(xs)), xs, lines, m, ws) <= 4096
    end
end

@testset "duals, then floats, on one LDMWorkspace" begin
    ws = LDMWorkspace()
    xs = range(-2.0, 2.0, length = 256)
    d = ForwardDiff.Dual{Nothing,Float64,1}(1.0, ForwardDiff.Partials((1.0,)))
    dlines = [Line(0.1, d, Voigt(0.5, 0.2))]
    @test calculate_lines(xs, dlines, LDM(), ws) == calculate_lines(xs, dlines, LDM())
    flines = [Line(0.1, 1.0, Voigt(0.5, 0.2))]
    @test closeto(calculate_lines(xs, flines, LDM(), ws), calculate_lines(xs, flines, LDM()))
end

@testset "shrinking n_x at fixed n_pad" begin
    # 512 and 505 points both pad to n_pad = 1024.
    ws = LDMWorkspace()
    lines = mklines(300)
    calculate_lines(range(1000.0, 1100.0, length = 512), lines, LDM(), ws)
    xs = range(1000.0, 1100.0, length = 505)
    @test closeto(calculate_lines(xs, lines, LDM(), ws), calculate_lines(xs, lines, LDM()))
end

@testset "threaded Fourier stage" begin
    ws = LDMWorkspace()
    xs = range(1000.0, 1100.0, length = 512)
    calculate_lines(xs, mklines(400), LDM(), ws)
    st = ws.state[]
    n_freq = st.n_pad ÷ 2 + 1
    freqs = (0:(n_freq - 1)) ./ (st.n_pad * step(xs))
    SpectralLines.ldm_fourier_ws!(st, freqs, n_freq, false)
    acc_serial = copy(st.acc)
    SpectralLines.ldm_fourier_ws!(st, freqs, n_freq, true)
    @test st.acc ≈ acc_serial rtol = 1e-12
end

@testset "changing fftw_threads" begin
    lines = mklines(200)
    xs = range(1000.0, 1100.0, length = 2^16)   # n_pad = 2^17: threading engages
    ref = calculate_lines(xs, lines, LDM(fftw_threads = 1))
    ws = LDMWorkspace()
    @test closeto(calculate_lines(xs, lines, LDM(fftw_threads = 2), ws), ref)
    @test closeto(calculate_lines(xs, lines, LDM(fftw_threads = 1), ws), ref)
    @test closeto(calculate_lines(xs, lines, LDM(), ws), ref)
end
