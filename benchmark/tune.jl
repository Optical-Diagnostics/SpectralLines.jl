include(joinpath(@__DIR__, "benchmarks.jl"))

using BenchmarkTools

tune!(SUITE)
BenchmarkTools.save(PARAMS_PATH, params(SUITE))
