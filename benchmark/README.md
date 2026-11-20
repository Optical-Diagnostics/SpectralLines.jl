# Benchmarks

The suite is defined in `benchmarks.jl`. Evaluation counts are read from
`params.json`. Regenerate that file after changing the benchmarks:

```
julia --project=benchmark benchmark/tune.jl
```

To compare a change against the previous state, save a baseline first:

```
julia --project=benchmark benchmark/save_baseline.jl
```

then make the change and run

```
julia --project=benchmark benchmark/compare.jl
```

The baseline goes to `_research/benchmark_baseline.json`, which is not tracked.

The in-place `calculate_lines!` benchmarks for `PointSample` and `BinAverage` should not
allocate. The `ldm_*` benchmarks do.
