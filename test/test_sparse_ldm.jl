# SparseLDM-only tests; those shared with LDM are in test_ldm.jl.
using SpectralLines, Test
using Random
import SpectralLines: flatten_lines, init_width_axis, gaussian_step, lorentzian_step,
                      sparse_ldm_bins, cluster_contributions

relmax(a, ref) = maximum(abs, a .- ref) / maximum(abs, ref)

# The clusters of each non-empty width bin, as SparseLDM forms them.
function bin_clusters(xs, lines)
    positions, amplitudes, wG_dat, wL_dat = flatten_lines(lines)
    width_axis_G = init_width_axis(wG_dat, gaussian_step(0.01))
    width_axis_L = init_width_axis(wL_dat, lorentzian_step(0.01))
    dx = step(xs)
    bins = sparse_ldm_bins(first(xs), last(xs), dx, length(xs), positions, amplitudes,
                           wG_dat, wL_dat, width_axis_G, width_axis_L)
    map(findall(!isempty, bins)) do I
        l, m = Tuple(I)
        lo, hi = convolution_support(Voigt(width_axis_G[l] / (2 * sqrt(2 * log(2))), width_axis_L[m] / 2))
        cluster_contributions(bins[I], ceil(Int, max(-lo, hi) / dx))
    end
end

@testset "no wrap-around inside a cluster" begin
    # A strong line at one end of a band that forms one cluster.
    xs = range(666.485, 800.0; step = 0.036)
    v = Voigt(0.0006, 0.07)
    lines = [Line(667.4, 1e4, v); [Line(p, 1e-2, v) for p in 668.0:1.5:765.0]]
    sparse = calculate_lines(xs, lines, SparseLDM())
    cut = calculate_lines(xs, lines, PointSample(wing_cut = 20.0))
    far = 690 .<= xs .<= 770
    @test maximum(abs, sparse[far] .- cut[far]) < 0.01
    # Beyond the last line's window the spectrum is exactly zero.
    @test all(iszero, sparse[xs .> 768])
end

@testset "fixed truncation" begin
    # A wing beyond 10 FWHM that still matters, as in a saturated band.
    xs = range(0.0, 100.0; step = 0.01)
    v = Voigt(0.01, 0.05)
    lines = [Line(50.0, 1.0, v), Line(53.0, 0.1, v)]
    sparse = calculate_lines(xs, lines, SparseLDM(truncation = 20.0))
    cut = calculate_lines(xs, lines, PointSample(wing_cut = 20.0 / hwhm(v)))
    wing = (35 .<= xs .<= 49) .| (54 .<= xs .<= 69)
    @test maximum(abs.(sparse[wing] .- cut[wing]) ./ cut[wing]) < 0.02
    @test all(iszero, sparse[(xs .< 29.9) .| (xs .> 73.1)])
    @test iszero(calculate_lines(xs, lines, SparseLDM())[findfirst(>=(45.0), xs)])
end

@testset "two bands" begin
    # Each band has its own (σ, γ) range, as when the Doppler width scales
    # with wavenumber.
    Random.seed!(7)
    xs = range(0.0, 2000.0; length = 40_000)
    lines = typeof(Line(0.0, 1.0, Voigt(0.1, 0.05)))[]
    for _ in 1:150
        push!(lines, Line(50.0 + 20 * rand(), 0.5 + rand(), Voigt(0.15 + 0.05 * rand(), 0.05 + 0.02 * rand())))
    end
    for _ in 1:150
        push!(lines, Line(1950.0 + 20 * rand(), 0.5 + rand(), Voigt(1.0 + 0.3 * rand(), 0.4 + 0.1 * rand())))
    end
    exact = calculate_lines(xs, lines, PointSample())
    sparse = calculate_lines(xs, lines, SparseLDM())
    @test relmax(sparse, exact) < 0.02
    for clusters in bin_clusters(xs, lines)
        @test length(clusters) == 1
        @test all(c -> c[2] - c[1] < length(xs) ÷ 2, clusters)
    end
end

@testset "one bin, several disjoint clusters" begin
    Random.seed!(11)
    xs = range(0.0, 3000.0; length = 60_000)
    lines = [Line(c + 5 * rand(), 0.5 + rand(), Voigt(0.1, 0.05)) for c in (50.0, 1500.0, 2950.0) for _ in 1:50]
    @test relmax(calculate_lines(xs, lines, SparseLDM()), calculate_lines(xs, lines, PointSample())) < 0.02
    bc = bin_clusters(xs, lines)
    @test length(bc) == 1
    @test length(only(bc)) == 3
    @test all(c -> c[2] - c[1] < length(xs) ÷ 10, only(bc))
end

@testset "truncation: show and validation" begin
    @test sprint(show, SparseLDM(truncation = 50.0)) == "SparseLDM(res_G=0.01, res_L=0.01, truncation=50.0)"
    @test_throws ArgumentError SparseLDM(truncation = 0.0)
end
