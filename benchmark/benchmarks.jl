using BenchmarkTools
using SpectralLines
using Random: MersenneTwister

import SpectralLines: faddeyeva, integrate, convolve

# Evaluation counts, written by tune.jl.
const PARAMS_PATH = joinpath(@__DIR__, "params.json")

function bench_integrate(shape, as, bs)
    s = 0.0
    @inbounds for k in eachindex(as)
        s += integrate(shape, as[k], bs[k])
    end
    s
end

const SUITE = BenchmarkGroup()

# Each shape over 100 points.
SUITE["lineshapes"] = let g = BenchmarkGroup()
    xs = collect(range(-5.0, 5.0, length = 100))

    g["lorentzian"] = @benchmarkable (Lorentzian(1.0)).($xs)
    g["gaussian"] = @benchmarkable (Gaussian(1.0)).($xs)
    g["pseudo_voigt"] = @benchmarkable (PseudoVoigt(1.0, 0.5)).($xs)
    g["voigt"] = @benchmarkable (Voigt(1.0, 0.5)).($xs)
    g["triangular"] = @benchmarkable (Triangular(1.0)).($xs)

    # 50 x-values at y = 0.2 that cross all approximation regions.
    xs_fad = vcat(collect(range(0.0, 8.0, length = 30)), collect(range(9.0, 200.0, length = 20)))
    g["faddeyeva"] = @benchmarkable faddeyeva.($xs_fad, 0.2)
    g
end

# N lines on 500 points, with a cheap shape and an expensive one.
SUITE["calculate_lines"] = let g = BenchmarkGroup()
    xs = collect(range(-50.0, 50.0, length = 500))

    for N in [10, 100, 1000]
        lines_l = [Line(float(i), 1.0, Lorentzian(0.1)) for i in 1:N]
        lines_v = [Line(float(i), 1.0, Voigt(0.1, 0.05)) for i in 1:N]
        ret_l = zeros(length(xs))
        ret_v = zeros(length(xs))

        g["lorentzian_$(N)_alloc"] = @benchmarkable calculate_lines($xs, $lines_l, PointSample())
        g["lorentzian_$(N)_inplace"] = @benchmarkable calculate_lines!($ret_l, $xs, $lines_l, PointSample())
        g["voigt_$(N)_alloc"] = @benchmarkable calculate_lines($xs, $lines_v, PointSample())
        g["voigt_$(N)_inplace"] = @benchmarkable calculate_lines!($ret_v, $xs, $lines_v, PointSample())
    end
    g
end

# 100 intervals per benchmark.
SUITE["integrate"] = let g = BenchmarkGroup()
    centres = range(-5.0, 5.0; length = 100)
    half = step(centres) / 2
    as = collect(centres .- half)
    bs = collect(centres .+ half)

    g["gaussian"] = @benchmarkable bench_integrate($(Gaussian(1.0)), $as, $bs)
    g["lorentzian"] = @benchmarkable bench_integrate($(Lorentzian(1.0)), $as, $bs)
    g["voigt"] = @benchmarkable bench_integrate($(Voigt(1.0, 0.5)), $as, $bs)
    g["pseudo_voigt"] = @benchmarkable bench_integrate($(PseudoVoigt(1.0, 0.5)), $as, $bs)
    g
end

# wing_cut and BinAverage. All lines lie inside the grid, unlike in the group above.
SUITE["calculate_lines_wc"] = let g = BenchmarkGroup()
    xr = range(-50.0, 50.0; length = 500)
    xs = collect(xr)
    Δx = step(xr)

    for N in [100, 1000]
        pos = collect(range(-48.0, 48.0; length = N))
        lines_l = [Line(p, 1.0, Lorentzian(0.1)) for p in pos]
        lines_v = [Line(p, 1.0, Voigt(0.1, 0.05)) for p in pos]
        ret_l = zeros(length(xs))
        ret_v = zeros(length(xs))

        g["lorentzian_$(N)_baseline"] = @benchmarkable calculate_lines!($ret_l, $xs, $lines_l, PointSample())
        g["voigt_$(N)_baseline"] = @benchmarkable calculate_lines!($ret_v, $xs, $lines_v, PointSample())

        g["lorentzian_$(N)_wc25"] = @benchmarkable calculate_lines!($ret_l, $xs, $lines_l, PointSample(wing_cut=25.0))
        g["voigt_$(N)_wc25"] = @benchmarkable calculate_lines!($ret_v, $xs, $lines_v, PointSample(wing_cut=25.0))

        g["lorentzian_$(N)_binned"] = @benchmarkable calculate_lines!($ret_l, $xs, $lines_l, BinAverage(bin_width=$Δx))
        g["voigt_$(N)_binned"] = @benchmarkable calculate_lines!($ret_v, $xs, $lines_v, BinAverage(bin_width=$Δx))

        g["voigt_$(N)_wc25_binned"] = @benchmarkable calculate_lines!($ret_v, $xs, $lines_v, BinAverage(bin_width=$Δx, wing_cut=25.0))
    end
    g
end

# LDM against PointSample on a 5000-point grid.
SUITE["ldm"] = let g = BenchmarkGroup()
    xr = range(0.0, 500.0; length = 5000)
    xs = collect(xr)

    for N in [1000, 10_000, 100_000]
        rng = MersenneTwister(1)
        positions = rand(rng, N) .* 500.0
        amps = 0.5 .+ rand(rng, N)
        sigmas = 0.02 .+ 0.05 .* rand(rng, N)
        gammas = 0.01 .+ 0.03 .* rand(rng, N)
        lines = [Line(positions[i], amps[i], Voigt(sigmas[i], gammas[i])) for i in 1:N]

        g["exact_$(N)"] = @benchmarkable calculate_lines($xs, $lines, PointSample())
        g["ldm_$(N)"] = @benchmarkable calculate_lines($xr, $lines, LDM())
    end
    g
end

# SparseLDM against LDM. "uniform": the lines of every width bin cover the
# whole grid. "clustered": two separated bands, each with its own width range.
SUITE["sparse_ldm"] = let g = BenchmarkGroup()
    xr = range(0.0, 2000.0; length = 40_000)

    for N in [10_000, 100_000]
        rng = MersenneTwister(1)
        positions = rand(rng, N) .* 2000.0
        amps = 0.5 .+ rand(rng, N)
        sigmas = 0.15 .+ 0.05 .* rand(rng, N)
        gammas = 0.05 .+ 0.02 .* rand(rng, N)
        lines_uniform = [Line(positions[i], amps[i], Voigt(sigmas[i], gammas[i])) for i in 1:N]

        g["ldm_uniform_$(N)"] = @benchmarkable calculate_lines($xr, $lines_uniform, LDM())
        g["sparse_ldm_uniform_$(N)"] = @benchmarkable calculate_lines($xr, $lines_uniform, SparseLDM())

        rng2 = MersenneTwister(2)
        n_half = N ÷ 2
        lines_clustered = vcat(
            [Line(50.0 + 20 * rand(rng2), 0.5 + rand(rng2), Voigt(0.15 + 0.05 * rand(rng2), 0.05 + 0.02 * rand(rng2)))
             for _ in 1:n_half],
            [Line(1950.0 + 20 * rand(rng2), 0.5 + rand(rng2), Voigt(1.0 + 0.3 * rand(rng2), 0.4 + 0.1 * rand(rng2)))
             for _ in 1:n_half],
        )

        g["ldm_clustered_$(N)"] = @benchmarkable calculate_lines($xr, $lines_clustered, LDM())
        g["sparse_ldm_clustered_$(N)"] = @benchmarkable calculate_lines($xr, $lines_clustered, SparseLDM())
    end
    g
end

# `convolve` on 500 shape pairs, and the FFT convolution of a spectrum.
SUITE["convolution"] = let g = BenchmarkGroup()
    σs = collect(range(0.5, 2.0, length = 500))
    γs = collect(range(0.1, 1.0, length = 500))
    glines = [Line(float(i) * 0.01, 1.0, Gaussian(σs[i])) for i in 1:500]

    g["gauss_lorentz_analytical"] = @benchmarkable convolve.(Gaussian.($σs), Lorentzian.($γs))
    g["line_lorentz_analytical"] = @benchmarkable convolve.($glines, Lorentzian(0.5))

    xs = collect(range(-10.0, 10.0, length = 500))
    lines = [Line(float(i) * 0.5, 1.0, Lorentzian(0.05)) for i in -20:20]
    g["spectrum_convolution_fft"] = @benchmarkable calculate_lines($xs, $lines, Gaussian(0.5))
    g
end

if isfile(PARAMS_PATH)
    loadparams!(SUITE, first(BenchmarkTools.load(PARAMS_PATH)), :evals)
end
