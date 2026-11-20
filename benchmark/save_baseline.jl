include(joinpath(@__DIR__, "benchmarks.jl"))

using BenchmarkTools

results = run(SUITE; verbose=true)

baseline_path = joinpath(@__DIR__, "..", "_research", "benchmark_baseline.json")
mkpath(dirname(baseline_path))
BenchmarkTools.save(baseline_path, results)

show(stdout, MIME("text/plain"), results)
println()
