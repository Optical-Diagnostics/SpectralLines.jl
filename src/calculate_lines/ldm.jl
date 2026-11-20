using FFTW: FFTW, plan_rfft, plan_irfft
using LinearAlgebra: mul!

# Next length ≥ n with only the prime factors 2, 3, 5 and 7, which FFTW
# transforms fast.
function nextfastfft(n::Integer)
    m = max(Int(n), 1)
    while true
        k = m
        for p in (2, 3, 5, 7)
            while k % p == 0
                k ÷= p
            end
        end
        k == 1 && return m
        m += 1
    end
end

# LDM bins lines by (Gaussian σ, Lorentzian γ), so it supports Voigt, Gaussian
# (γ = 0) and Lorentzian (σ = 0).
voigt_params(shape::Voigt) = (shape.sigma, shape.gamma)
voigt_params(shape::Gaussian) = (shape.sigma, zero(shape.sigma))
voigt_params(shape::Lorentzian) = (zero(shape.hwhm), shape.hwhm)

@inline gaussian_ft(w, σ) = exp(-2 * (π * σ * w)^2)
@inline lorentzian_ft(w, γ) = exp(-2 * π * w * γ)
@inline voigt_ft(w, σ, γ) = gaussian_ft(w, σ) * lorentzian_ft(w, γ)

@inline bin_sigma_gamma(fwhm_G, fwhm_L) = (fwhm_G / fwhm_over_sigma(fwhm_G), fwhm_L / 2)

# Float64 copy of a number. The ForwardDiff extension adds the `Dual` method.
to_float64(x::Real) = Float64(x)

# The first `n` values of `voigt_ft` on a frequency grid of step `Δ`, by
# recurrence. Runs in Float64 also for a Float32 template.
function fill_template!(template, Δ, σ, γ, n::Int)
    Δ, σ, γ = to_float64(Δ), to_float64(σ), to_float64(γ)
    g2 = exp(-4 * (π * σ * Δ)^2)
    r = exp(-2 * (π * σ * Δ)^2 - 2 * π * γ * Δ)
    v = one(r)
    @inbounds for i in 1:n
        template[i] = v
        v *= r
        r *= g2
    end
    return template
end

# Log-spaced width-axis step size targeting a `res` relative error (as given in RADIS
# loader.py `lorentzian_step`/`gaussian_step`).
# TODO: RADIS states that these values are 'horribly outdated'
lorentzian_step(res_L::Real=0.01) = log(sqrt(res_L / 0.20) + 1)
gaussian_step(res_G::Real=0.01) = log(sqrt(res_G / 0.46) + 1)

# Flatten a line list into vectors: positions, amplitudes, and the two component
# FWHMs.
function flatten_lines(lines)
    first_line = first(lines)
    S = typeof(first_line.position)
    A = typeof(first_line.amplitude)
    return flatten_lines!(Vector{S}(), Vector{A}(), Vector{S}(), Vector{S}(), lines)
end

# In-place variant for workspace reuse.
function flatten_lines!(positions::Vector{S}, amplitudes, fwhm_G_dat, fwhm_L_dat,
                       lines) where {S}
    n = length(lines)
    resize!(positions, n)
    resize!(amplitudes, n)
    resize!(fwhm_G_dat, n)
    resize!(fwhm_L_dat, n)
    @inbounds for (i, line) in enumerate(lines)
        σ, γ = voigt_params(line.shape)
        positions[i] = line.position
        amplitudes[i] = line.amplitude
        fwhm_G_dat[i] = max(fwhm_over_sigma(σ) * σ, zero(S))
        fwhm_L_dat[i] = max(2 * γ, zero(S))
    end
    return positions, amplitudes, fwhm_G_dat, fwhm_L_dat
end

# Providing better error messages for unsupported types in fft.
function fft_eltype_check(::Type{T}, method_name) where {T}
    T <: Union{Float32,Float64} && return nothing
    throw(ArgumentError(
        "$method_name needs a Float32 or Float64 element type, or a " *
        "ForwardDiff.Dual over one, got $T: FFTW cannot plan a transform over " *
        "it. Use `PointSample()` or `BinAverage()` instead, which take any number type."))
end

# Relative size of the dedicated zero-width bin, as a fraction of the narrowest
# real width present. Small enough that the bin's Voigt template is a delta at
# grid resolution, since the narrowest real line is itself resolved by the grid.
const ZERO_WIDTH_FRACTION = 1e-8

init_width_axis(w_dat::AbstractVector{T}, log_p::Real) where {T} =
    init_width_axis!(Vector{T}(), w_dat, log_p)

function init_width_axis!(axis::Vector{T}, w_dat::AbstractVector{T}, log_p::Real) where {T}
    empty!(axis)
    w_min = w_max = zero(T)
    has_zero = false
    seen = false
    @inbounds for w in w_dat
        if w > zero(T)
            if seen
                w_min = min(w_min, w)
                w_max = max(w_max, w)
            else
                w_min = w_max = w
                seen = true
            end
        else
            has_zero = true
        end
    end
    if !seen
        push!(axis, zero(T))
        return axis
    end
    w_max *= 1 + T(1e-6)
    n = max(1, ceil(Int, (log(w_max) - log(w_min)) / log_p) + 1)
    has_zero && push!(axis, w_min * T(ZERO_WIDTH_FRACTION))
    for k in 0:(n - 1)
        push!(axis, w_min * exp(log_p * k))
    end
    return axis
end

# Fractional index of `value` on a monotonic `grid`, as 1-based (i0, i1, frac).
# A clamped value returns its one bin twice, so no empty neighbour is marked
# occupied.
function frac_index(value, grid::AbstractVector)
    n = length(grid)
    # A one-bin axis takes every value. It is also the exact zero-width bin,
    # whose log is -Inf: comparing against it is not reliable for dual numbers.
    n == 1 && return 1, 1, zero(value)
    value <= grid[1] && return 1, 1, zero(value)
    value >= grid[end] && return n, n, zero(value)
    i = searchsortedlast(grid, value)
    frac = (value - grid[i]) / (grid[i + 1] - grid[i])
    return i, i + 1, frac
end

# Grid cell of a line at `p`: the 0-based left corner `cell_lo` and the fraction
# `cell_frac` toward the right one, or `nothing` outside the grid. The bounds compare
# positions, since `(x_end - x0) / dx` can round above `n_x - 1`. A line on
# `xs[end]` takes the last cell.
@inline function grid_cell(p, x0, x_end, dx, n_x::Int)
    (p < x0 || p > x_end) && return nothing
    pos = (p - x0) / dx
    cell_lo = clamp(floor(Int, pos), 0, max(n_x - 2, 0))
    return cell_lo, pos - cell_lo
end

# Scatters each line's amplitude onto its 8 neighbouring corners (position ×
# Gaussian width × Lorentzian width) of `ldm` with trilinear weights. Rows
# beyond `n_x` are zero padding. `occupied` marks the width bins that received
# a line; the FFT stage skips the others.
function scatter_lines!(ldm::Array{T,3}, x0, x_end, dx, n_x::Int, positions, amplitudes,
                    fwhm_G_dat, fwhm_L_dat, width_axis_G::AbstractVector, width_axis_L::AbstractVector,
                    log_width_axis_G::AbstractVector, log_width_axis_L::AbstractVector,
                    occupied::BitMatrix) where {T}
    fill!(occupied, false)

    @inbounds for i in eachindex(positions)
        fwhm_G = max(fwhm_G_dat[i], width_axis_G[1])
        fwhm_L = max(fwhm_L_dat[i], width_axis_L[1])

        cell = grid_cell(positions[i], x0, x_end, dx, n_x)
        cell === nothing && continue
        cell_lo, cell_frac = cell
        cell_hi = cell_lo + 1

        g_lo, g_hi, g_frac = frac_index(log(fwhm_G), log_width_axis_G)
        l_lo, l_hi, l_frac = frac_index(log(fwhm_L), log_width_axis_L)

        amp_lo = amplitudes[i] * (1 - cell_frac)
        amp_hi = amplitudes[i] * cell_frac

        wgt_g0l0 = (1 - g_frac) * (1 - l_frac)
        wgt_g0l1 = (1 - g_frac) * l_frac
        wgt_g1l0 = g_frac * (1 - l_frac)
        wgt_g1l1 = g_frac * l_frac

        ldm[cell_lo + 1, g_lo, l_lo] += amp_lo * wgt_g0l0
        ldm[cell_lo + 1, g_lo, l_hi] += amp_lo * wgt_g0l1
        ldm[cell_lo + 1, g_hi, l_lo] += amp_lo * wgt_g1l0
        ldm[cell_lo + 1, g_hi, l_hi] += amp_lo * wgt_g1l1
        ldm[cell_hi + 1, g_lo, l_lo] += amp_hi * wgt_g0l0
        ldm[cell_hi + 1, g_lo, l_hi] += amp_hi * wgt_g0l1
        ldm[cell_hi + 1, g_hi, l_lo] += amp_hi * wgt_g1l0
        ldm[cell_hi + 1, g_hi, l_hi] += amp_hi * wgt_g1l1

        occupied[g_lo, l_lo] = true
        occupied[g_lo, l_hi] = true
        occupied[g_hi, l_lo] = true
        occupied[g_hi, l_hi] = true
    end
    return occupied
end

"""
    default_fftw_threads() -> Int

Default for `LDM`'s `fftw_threads`: half the machine's CPU threads, at least 1.
"""
default_fftw_threads() = max(1, Sys.CPU_THREADS ÷ 2)

"""
    LDM(; res_G=0.01, res_L=0.01, fftw_threads=default_fftw_threads())

Fast approximate line synthesis by the discrete integral transform (DIT) of
van den Bekerom & Pannier (2021), RADIS's default method. Its cost is nearly
independent of the line count, so it targets large line lists (10⁵+);
`PointSample`/`BinAverage` are simpler and exact for small ones.

`res_G` and `res_L` set the resolution of the Gaussian and Lorentzian width
grids: smaller values are more accurate and slower.

Requires an ascending `xs::AbstractRange` and `Voigt`/`Gaussian`/`Lorentzian` line shapes;
lines centred outside `xs` are dropped. `calculate_lines!` allocates
intermediate buffers; pass a [`LDMWorkspace`](@ref) to reuse them across calls.
Memory grows with the grid length times the number of distinct Gaussian and
Lorentzian widths present.

Forward-mode differentiable once `ForwardDiff` is loaded, at `1 + n_partials`
times the FFT cost. The derivative is that of the LDM approximation, so it
differs from the true derivative at the `res_G`/`res_L` level, and the position
derivative is piecewise constant per grid cell. `PointSample`/`BinAverage` differentiate
exactly.

`fftw_threads` sets the FFTW thread count (default: half the machine's CPU
threads, see [`default_fftw_threads`](@ref)). It is used only for large grids
(2¹⁷ points or more after padding) and only when Julia runs single-threaded.
With `julia -t N` the work is parallelized over Julia threads instead. Set
`fftw_threads = 1` to keep FFTW serial.

`LDM` also convolves a stick spectrum with one instrument profile: give every
line the same shape.

```julia
lines = [Line(λ[i], amp[i], Gaussian(σ_inst)) for i in eachindex(λ)]
spec = calculate_lines(grid, lines, LDM())
```
"""
Base.@kwdef struct LDM <: LineSynthesisMethod
    res_G::Float64 = 0.01
    res_L::Float64 = 0.01
    fftw_threads::Int = default_fftw_threads()
end

function Base.show(io::IO, m::LDM)
    print(io, "LDM(res_G=", m.res_G, ", res_L=", m.res_L)
    m.fftw_threads == default_fftw_threads() || print(io, ", fftw_threads=", m.fftw_threads)
    print(io, ")")
end

calculate_lines!(ret, xs::AbstractRange, lines, method::LDM) =
    calculate_lines!(ret, xs, lines, method, LDMWorkspace())

# The two FFT seams of `ldm_generic!`. The ForwardDiff extension adds the `Dual`
# methods. Hardware floats never reach them; these fallbacks only throw.
ldm_fourier_accumulate(ldm::Array{T,3}, occupied, width_axis_G, width_axis_L, freqs) where {T} =
    throw(ArgumentError("LDM has no FFT path for element type $T."))
ldm_irfft(acc::AbstractVector, n_pad::Int) =
    throw(ArgumentError("LDM has no inverse-FFT path for element type $(eltype(acc))."))

# Element-type-generic pipeline for non-hardware floats, in practice
# `ForwardDiff.Dual`. The caller handles the empty-list case.
function ldm_generic!(ret, xs::AbstractRange, lines, method::LDM)
    positions, amplitudes, fwhm_G_dat, fwhm_L_dat = flatten_lines(lines)
    T = promote_type(eltype(positions), eltype(amplitudes))
    fft_eltype_check(T, "LDM")

    width_axis_G = init_width_axis(fwhm_G_dat, gaussian_step(method.res_G))
    width_axis_L = init_width_axis(fwhm_L_dat, lorentzian_step(method.res_L))

    n_x = length(xs)
    x0 = first(xs)
    x_end = last(xs)
    dx = step(xs)
    n_pad = nextfastfft(2 * n_x)

    ldm = zeros(T, n_pad, length(width_axis_G), length(width_axis_L))::Array{T,3}
    occupied = falses(length(width_axis_G), length(width_axis_L))
    scatter_lines!(ldm, x0, x_end, dx, n_x, positions, amplitudes,
               fwhm_G_dat, fwhm_L_dat, width_axis_G, width_axis_L, log.(width_axis_G), log.(width_axis_L), occupied)

    n_freq = n_pad ÷ 2 + 1
    freqs = (0:(n_freq - 1)) ./ (n_pad * dx)

    acc = ldm_fourier_accumulate(ldm, occupied, width_axis_G, width_axis_L, freqs)
    ret .= view(ldm_irfft(acc, n_pad), 1:n_x) .* inv(dx)
    return ret
end

calculate_lines!(ret, xs, lines, ::LDM) = error(
    "LDM requires an evenly-spaced grid: pass `xs` as an `AbstractRange`, " *
    "or use `PointSample()` for an arbitrary grid."
)

"""
    LDMWorkspace()

Reusable workspace for [`LDM`](@ref): `calculate_lines!(ret, xs, lines, LDM(), ws)`.
Repeated calls, as in a fitting loop, then allocate next to nothing.

Not thread-safe: use one workspace per task. With `ForwardDiff.Dual` inputs the
workspace is not used and the call allocates.
"""
struct LDMWorkspace
    # Untyped: the state type depends on the line list's element types. A
    # function barrier recovers a concrete type.
    state::Base.RefValue{Any}
end
LDMWorkspace() = LDMWorkspace(Ref{Any}(nothing))

const PLAN_FLAGS = FFTW.ESTIMATE | FFTW.UNALIGNED
const FWD_PLAN_F64 = typeof(plan_rfft(Vector{Float64}(undef, 2); flags = PLAN_FLAGS))
const FWD_PLAN_F32 = typeof(plan_rfft(Vector{Float32}(undef, 2); flags = PLAN_FLAGS))
const INV_PLAN_F64 = typeof(plan_irfft(Vector{Complex{Float64}}(undef, 2), 2; flags = PLAN_FLAGS))
const INV_PLAN_F32 = typeof(plan_irfft(Vector{Complex{Float32}}(undef, 2), 2; flags = PLAN_FLAGS))
fwd_plan_type(::Type{Float64}) = FWD_PLAN_F64
fwd_plan_type(::Type{Float32}) = FWD_PLAN_F32
inv_plan_type(::Type{Float64}) = INV_PLAN_F64
inv_plan_type(::Type{Float32}) = INV_PLAN_F32

# Smallest padded length at which FFTW threads pay.
const FFT_THREADS_MIN_N_PAD = 2^17

# FFTW threads are used only for large transforms and only when Julia is
# single-threaded. Otherwise the bins run in parallel.
plan_thread_count(method::LDM, n_pad::Int) =
    Threads.nthreads() == 1 && n_pad >= FFT_THREADS_MIN_N_PAD ? method.fftw_threads : 1

ldm_plans(out::Vector, acc::Vector, n_pad::Int, nthreads::Int) =
    (plan_rfft(out; flags = PLAN_FLAGS, num_threads = nthreads),
     plan_irfft(acc, n_pad; flags = PLAN_FLAGS, num_threads = nthreads))

mutable struct LDMState{S,A,T,PF,PI}
    positions::Vector{S}
    amplitudes::Vector{A}
    fwhm_G::Vector{S}
    fwhm_L::Vector{S}
    width_axis_G::Vector{S}
    width_axis_L::Vector{S}
    log_width_axis_G::Vector{S}
    log_width_axis_L::Vector{S}
    ldm::Array{T,3}
    occupied::BitMatrix
    prev_occupied::BitMatrix
    acc::Vector{Complex{T}}
    template::Vector{T}
    fft_out::Vector{Complex{T}}
    out::Vector{T}
    fwd_plan::PF
    inv_plan::PI
    n_pad::Int
    # The previous call's n_x: its scatter wrote only rows 1:n_x. Tracked
    # because n_x can shrink while n_pad stays the same.
    n_x_written::Int
    # Scratch for the threaded fourier stage: occupied-bin list and per-thread
    # template/FFT/accumulator buffers, sized lazily on the first threaded call.
    occ_list::Vector{Tuple{Int,Int}}
    thread_templates::Vector{Vector{T}}
    thread_ffts::Vector{Vector{Complex{T}}}
    thread_accs::Vector{Vector{Complex{T}}}
    plan_threads::Int
end

function new_ldm_state(::Type{S}, ::Type{A}, ::Type{T}, n_pad::Int,
                       method::LDM) where {S,A,T}
    n_freq = n_pad ÷ 2 + 1
    ldm = zeros(T, n_pad, 1, 1)
    acc = zeros(Complex{T}, n_freq)
    out = Vector{T}(undef, n_pad)
    # Plans are built from plain vectors so their types survive a rebuild.
    #
    # UNALIGNED: the forward plan runs over every (l, m) column of `ldm`, and an
    # odd n_pad leaves those columns misaligned. FFTW would reject them.
    #
    # ESTIMATE: the default MEASURE overwrites its input while planning.
    plan_threads = plan_thread_count(method, n_pad)
    fwd, inv = ldm_plans(out, acc, n_pad, plan_threads)
    return LDMState{S,A,T,typeof(fwd),typeof(inv)}(
        Vector{S}(), Vector{A}(), Vector{S}(), Vector{S}(),
        Vector{S}(), Vector{S}(), Vector{S}(), Vector{S}(),
        ldm, falses(1, 1), falses(1, 1),
        acc, Vector{T}(undef, n_freq), Vector{Complex{T}}(undef, n_freq),
        out, fwd, inv, n_pad, 0,
        Vector{Tuple{Int,Int}}(),
        [Vector{T}() for _ in 1:Threads.nthreads()],
        [Vector{Complex{T}}() for _ in 1:Threads.nthreads()],
        [Vector{Complex{T}}() for _ in 1:Threads.nthreads()],
        plan_threads)
end

function calculate_lines!(ret, xs::AbstractRange, lines, method::LDM, ws::LDMWorkspace)
    check_axes(ret, xs)
    check_ascending(xs, "LDM")
    if isempty(lines)
        fill!(ret, zero(eltype(ret)))
        return ret
    end
    l1 = first(lines)
    S = typeof(l1.position)
    A = typeof(l1.amplitude)
    T = promote_type(S, A)
    T <: Union{Float32,Float64} || return ldm_generic!(ret, xs, lines, method)
    st = ws.state[]
    if !(st isa LDMState{S,A,T,fwd_plan_type(T),inv_plan_type(T)})
        st = new_ldm_state(S, A, T, nextfastfft(2 * length(xs)), method)
        ws.state[] = st
    end
    return ldm_ws!(ret, xs, lines, method, st)
end

calculate_lines!(ret, xs, lines, method::LDM, ::LDMWorkspace) =
    calculate_lines!(ret, xs, lines, method)

# Flatten `lines` into the state's vectors and build its two width axes. Shared
# with `SparseLDMState`, which has the same fields.
function load_lines!(st, lines, method)
    flatten_lines!(st.positions, st.amplitudes, st.fwhm_G, st.fwhm_L, lines)
    init_width_axis!(st.width_axis_G, st.fwhm_G, gaussian_step(method.res_G))
    init_width_axis!(st.width_axis_L, st.fwhm_L, lorentzian_step(method.res_L))
    resize!(st.log_width_axis_G, length(st.width_axis_G))
    resize!(st.log_width_axis_L, length(st.width_axis_L))
    st.log_width_axis_G .= log.(st.width_axis_G)
    st.log_width_axis_L .= log.(st.width_axis_L)
    return st
end

function ldm_ws!(ret, xs, lines, method::LDM, st::LDMState{S,A,T}) where {S,A,T}
    load_lines!(st, lines, method)

    n_x = length(xs)
    x0 = first(xs)
    x_end = last(xs)
    dx = step(xs)
    n_pad = nextfastfft(2 * n_x)
    n_freq = n_pad ÷ 2 + 1

    if n_pad != st.n_pad
        resize!(st.acc, n_freq)
        resize!(st.template, n_freq)
        resize!(st.fft_out, n_freq)
        resize!(st.out, n_pad)
        st.n_pad = n_pad
        st.plan_threads = -1  # size changed: force the plan rebuild below
    end
    plan_threads = plan_thread_count(method, n_pad)
    if plan_threads != st.plan_threads
        st.fwd_plan, st.inv_plan = ldm_plans(st.out, st.acc, n_pad, plan_threads)
        st.plan_threads = plan_threads
    end
    if size(st.ldm) != (n_pad, length(st.width_axis_G), length(st.width_axis_L))
        st.ldm = zeros(T, n_pad, length(st.width_axis_G), length(st.width_axis_L))
        st.occupied = falses(length(st.width_axis_G), length(st.width_axis_L))
        st.prev_occupied = falses(length(st.width_axis_G), length(st.width_axis_L))
    else
        @inbounds for m in axes(st.ldm, 3), l in axes(st.ldm, 2)
            st.prev_occupied[l, m] || continue
            fill!(view(st.ldm, 1:st.n_x_written, l, m), zero(T))
        end
    end

    scatter_lines!(st.ldm, x0, x_end, dx, n_x, st.positions, st.amplitudes,
               st.fwhm_G, st.fwhm_L, st.width_axis_G, st.width_axis_L, st.log_width_axis_G, st.log_width_axis_L, st.occupied)
    st.n_x_written = n_x
    copyto!(st.prev_occupied, st.occupied)

    freqs = (0:(n_freq - 1)) ./ (n_pad * dx)
    ldm_fourier_ws!(st, freqs, n_freq,
                     Threads.nthreads() > 1 && count(st.occupied) >= 2 * Threads.nthreads())

    mul!(st.out, st.inv_plan, st.acc)
    ret .= view(st.out, 1:n_x) .* inv(dx)
    return ret
end

# One bin's contribution: template, forward FFT of its `ldm` column,
# multiply-accumulate.
@inline function ldm_bin_accumulate!(acc, template, fft_out, st, l, m, freqs, n_freq)
    σ_lm, γ_lm = bin_sigma_gamma(st.width_axis_G[l], st.width_axis_L[m])
    fill_template!(template, step(freqs), σ_lm, γ_lm, n_freq)
    mul!(fft_out, st.fwd_plan, view(st.ldm, :, l, m))
    @inbounds @simd for i in eachindex(acc)
        acc[i] += fft_out[i] * template[i]
    end
    return acc
end

# The Fourier stage: a serial loop over the occupied bins or, when `threaded`,
# one chunk of them per task. The threaded sum is deterministic for a given
# thread count, but differs from the serial one in rounding.
function ldm_fourier_ws!(st::LDMState{S,A,T}, freqs, n_freq::Int, threaded::Bool) where {S,A,T}
    fill!(st.acc, zero(Complex{T}))
    if !threaded
        @inbounds for m in eachindex(st.width_axis_L), l in eachindex(st.width_axis_G)
            st.occupied[l, m] || continue
            ldm_bin_accumulate!(st.acc, st.template, st.fft_out, st, l, m, freqs, n_freq)
        end
        return st
    end
    empty!(st.occ_list)
    for m in eachindex(st.width_axis_L), l in eachindex(st.width_axis_G)
        st.occupied[l, m] && push!(st.occ_list, (l, m))
    end
    nt = Threads.nthreads()
    bounds = chunk_bounds(length(st.occ_list), nt)
    Threads.@threads for t in 1:nt
        template = st.thread_templates[t]
        fft_out = st.thread_ffts[t]
        acc = st.thread_accs[t]
        length(template) == n_freq || resize!(template, n_freq)
        length(fft_out) == n_freq || resize!(fft_out, n_freq)
        length(acc) == n_freq || resize!(acc, n_freq)
        fill!(acc, zero(Complex{T}))
        for i in (bounds[t] + 1):bounds[t + 1]
            l, m = st.occ_list[i]
            ldm_bin_accumulate!(acc, template, fft_out, st, l, m, freqs, n_freq)
        end
    end
    @inbounds for t in 1:nt
        a = st.thread_accs[t]
        @simd for i in eachindex(st.acc)
            st.acc[i] += a[i]
        end
    end
    return st
end
