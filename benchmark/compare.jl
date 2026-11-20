include(joinpath(@__DIR__, "benchmarks.jl"))

using BenchmarkTools

baseline_path = joinpath(@__DIR__, "..", "_research", "benchmark_baseline.json")
isfile(baseline_path) || error("No baseline found. Run benchmark/save_baseline.jl first.")

baseline = first(BenchmarkTools.load(baseline_path))
results = run(SUITE; verbose=true)

# A ratio above 1 is slower than the baseline.
show(stdout, MIME("text/plain"), judge(minimum(results), minimum(baseline); time_tolerance=0.20))
println()
