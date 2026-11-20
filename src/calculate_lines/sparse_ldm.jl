using FFTW: FFTW, plan_rfft, plan_irfft
using LinearAlgebra: mul!

# One line in a (σ, γ) bin: its two grid cells (0-based) and the amplitude on each.
const Contribution{T} = NamedTuple{(:cell_lo, :cell_hi, :amp_lo, :amp_hi),Tuple{Int,Int,T,T}}

# One cluster: its bin (l, m), its padded cell range, and its index range in the
# bin's contribution list.
const Cluster = NamedTuple{(:l, :m, :lo, :hi, :first_idx, :last_idx),NTuple{6,Int}}

# Like `scatter_lines!`, but collects each line's contributions in one list per
# (σ, γ) bin.
function sparse_ldm_bins(x0, x_end, dx, n_x::Int, positions, amplitudes, fwhm_G_dat, fwhm_L_dat,
                          width_axis_G::AbstractVector{T}, width_axis_L::AbstractVector{T}) where {T}
    contributions = [Contribution{T}[] for _ in 1:length(width_axis_G), _ in 1:length(width_axis_L)]
    return sparse_ldm_bins!(contributions, x0, x_end, dx, n_x, positions, amplitudes,
                             fwhm_G_dat, fwhm_L_dat, width_axis_G, width_axis_L, log.(width_axis_G), log.(width_axis_L))
end

function sparse_ldm_bins!(contributions, x0, x_end, dx, n_x::Int, positions, amplitudes,
                           fwhm_G_dat, fwhm_L_dat, width_axis_G::AbstractVector{T},
                           width_axis_L::AbstractVector{T}, log_width_axis_G, log_width_axis_L) where {T}
    for v in contributions
        empty!(v)
    end

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

        for (l, m, aw) in ((g_lo, l_lo, (1 - g_frac) * (1 - l_frac)), (g_lo, l_hi, (1 - g_frac) * l_frac),
                           (g_hi, l_lo, g_frac * (1 - l_frac)), (g_hi, l_hi, g_frac * l_frac))
            iszero(aw) && continue
            push!(contributions[l, m], (; cell_lo, cell_hi, amp_lo = amp_lo * aw, amp_hi = amp_hi * aw))
        end
    end
    return contributions
end

# Sorts `contribs` by position and partitions it into clusters: groups of lines
# more than `2*tp` grid points apart, each with its own FFT. Returns
# `(cluster_lo, cluster_hi, first_idx, last_idx)` per cluster. The range is
# padded by `tp` on both sides, so a kernel truncated at ±`tp` does not wrap.
cluster_contributions(contribs::Vector{Contribution{T}}, tp::Int) where {T} =
    cluster_contributions!(Tuple{Int,Int,Int,Int}[], contribs, tp, nothing)

function cluster_contributions!(clusters::Vector{Tuple{Int,Int,Int,Int}},
                        contribs::Vector{Contribution{T}}, tp::Int, scratch) where {T}
    sort!(contribs; by=first, scratch)

    empty!(clusters)
    cluster_start = 1
    cluster_lo = contribs[1].cell_lo - tp
    cluster_hi = contribs[1].cell_hi + tp
    @inbounds for i in 2:length(contribs)
        lo_i = contribs[i].cell_lo - tp
        hi_i = contribs[i].cell_hi + tp
        if lo_i <= cluster_hi + 1
            cluster_hi = max(cluster_hi, hi_i)
        else
            push!(clusters, (cluster_lo, cluster_hi, cluster_start, i - 1))
            cluster_start = i
            cluster_lo = lo_i
            cluster_hi = hi_i
        end
    end
    push!(clusters, (cluster_lo, cluster_hi, cluster_start, length(contribs)))
    return clusters
end

# FFTW plans per transform length, for hardware floats `V`. `r2c!` is the
# forward real transform and `c2r!` the inverse, which destroys its input. The
# ForwardDiff extension adds a cache and both methods for duals.
struct FFTPlanCache{V,P1,P2}
    rbuf::Vector{V}             # planning targets only, grown to each new size
    cbuf::Vector{Complex{V}}
    fwd_plans::Dict{Int,P1}
    inv_plans::Dict{Int,P2}
end

fft_cache_type(::Type{V}) where {V<:Union{Float32,Float64}} =
    FFTPlanCache{V,fwd_plan_type(V),inv_plan_type(V)}
fft_cache(::Type{V}) where {V<:Union{Float32,Float64}} =
    fft_cache_type(V)(V[], Complex{V}[], Dict(), Dict())

function rfft_plan!(c::FFTPlanCache, n::Int)
    plan = get(c.fwd_plans, n, nothing)
    if plan === nothing
        length(c.rbuf) < n && resize!(c.rbuf, n)
        plan = plan_rfft(view(c.rbuf, 1:n); flags = PLAN_FLAGS, num_threads = 1)
        c.fwd_plans[n] = plan
    end
    return plan
end
function irfft_plan!(c::FFTPlanCache, n::Int)
    plan = get(c.inv_plans, n, nothing)
    if plan === nothing
        length(c.cbuf) < n ÷ 2 + 1 && resize!(c.cbuf, n ÷ 2 + 1)
        plan = plan_irfft(view(c.cbuf, 1:(n ÷ 2 + 1)), n; flags = PLAN_FLAGS, num_threads = 1)
        c.inv_plans[n] = plan
    end
    return plan
end

r2c!(out, c::FFTPlanCache, x, n::Int) = mul!(out, rfft_plan!(c, n), x)
c2r!(out, c::FFTPlanCache, X, n::Int) = mul!(out, irfft_plan!(c, n), X)

# Scratch buffers and the plan cache, shared by all clusters of a call. Buffers
# are sized to the largest transform; smaller ones use a prefix view.
struct SparseFFTBuffers{T,C}
    scatter_buf::Vector{T}
    fft::Vector{Complex{T}}
    conv::Vector{T}
    kernel::Vector{T}               # current bin's kernel at offsets -tp:tp
    kernel_fft::Vector{Complex{T}}  # its rfft at the current cluster FFT length
    plans::C
end

sparse_fft_buffers(::Type{T}) where {T} =
    SparseFFTBuffers(zeros(T, 2), Vector{Complex{T}}(undef, 2), Vector{T}(undef, 2),
                        Vector{T}(undef, 1), Vector{Complex{T}}(undef, 2), fft_cache(T))

cluster_fft_length(cluster_lo::Int, cluster_hi::Int) = nextfastfft(cluster_hi - cluster_lo + 1)

# The automatic truncation changes in whole grid points with the width, so the
# spectrum is not differentiable in it. The ForwardDiff extension adds a method
# that throws for widths carrying derivatives.
check_width_derivatives(fwhm_G, fwhm_L) = nothing

# FFT length for a bin's kernel: long enough that its periodic images are
# negligible within ±tp, also for a truncation narrower than the line.
function kernel_fft_length(tp::Int, dx, fwhm_G, fwhm_L)
    width_pts = ceil(Int, 3 * max(fwhm_G, fwhm_L) / dx)
    return nextfastfft(8 * (2 * max(tp, width_pts) + 1))
end

# The bin's Voigt kernel on the grid, truncated to offsets -tp:tp, into
# `ws.kernel`. It is the inverse transform of LDM's template. The truncation
# keeps the Lorentzian wings from wrapping around in the cluster FFTs.
function sparse_ldm_kernel!(ws::SparseFFTBuffers{T}, tp::Int, n::Int, dx, σ_lm, γ_lm) where {T}
    n_freq = n ÷ 2 + 1
    fft_view = view(ws.fft, 1:n_freq)
    fill_template!(fft_view, 1 / (n * dx), σ_lm, γ_lm, n_freq)
    conv_view = view(ws.conv, 1:n)
    c2r!(conv_view, ws.plans, fft_view, n)
    resize!(ws.kernel, 2tp + 1)
    @inbounds for j in -tp:tp
        ws.kernel[j + tp + 1] = conv_view[mod(j, n) + 1]
    end
    return ws.kernel
end

# rfft of the current kernel, laid out circularly on `n_pad` points, into
# `ws.kernel_fft`. Uses `ws.conv` as scratch.
function sparse_ldm_kernel_fft!(ws::SparseFFTBuffers{T}, tp::Int, n_pad::Int) where {T}
    kpad = view(ws.conv, 1:n_pad)
    fill!(kpad, zero(T))
    @inbounds for j in -tp:tp
        kpad[mod(j, n_pad) + 1] = ws.kernel[j + tp + 1]
    end
    r2c!(view(ws.kernel_fft, 1:(n_pad ÷ 2 + 1)), ws.plans, kpad, n_pad)
    return ws.kernel_fft
end

function convolve_cluster!(ret::AbstractVector, contribs::Vector{Contribution{T}},
                            first_idx::Int, last_idx::Int, cluster_lo::Int, cluster_hi::Int,
                            n_pad_local::Int, n_x::Int, inv_dx,
                            ws::SparseFFTBuffers{T}) where {T}
    n_freq_local = n_pad_local ÷ 2 + 1

    ldm_view = view(ws.scatter_buf, 1:n_pad_local)
    fill!(ldm_view, zero(T))
    @inbounds for i in first_idx:last_idx
        cell_lo, cell_hi, c0, c1 = contribs[i]
        ldm_view[cell_lo - cluster_lo + 1] += c0
        ldm_view[cell_hi - cluster_lo + 1] += c1
    end

    fft_view = view(ws.fft, 1:n_freq_local)
    r2c!(fft_view, ws.plans, ldm_view, n_pad_local)
    fft_view .*= view(ws.kernel_fft, 1:n_freq_local)

    conv_view = view(ws.conv, 1:n_pad_local)
    c2r!(conv_view, ws.plans, fft_view, n_pad_local)

    out_lo = max(cluster_lo, 0)
    out_hi = min(cluster_hi, n_x - 1)
    @inbounds @simd for k in out_lo:out_hi
        ret[k + 1] += conv_view[k - cluster_lo + 1] * inv_dx
    end
    return ret
end

"""
    SparseLDM(; res_G=0.01, res_L=0.01, truncation=nothing)

Variant of [`LDM`](@ref) that transforms only the parts of `xs` near each group
of lines. It takes the same grids and line shapes, and
`res_G`/`res_L` have the same meaning.

It is faster than `LDM` when lines of similar width cover only part of `xs`, as
in a real molecular spectrum, where the Doppler width scales with wavenumber.
Lines whose widths do not correlate with position favor `LDM`.

`truncation` sets the half-width of the kept wings, in units of `xs`, the same
for every line (RADIS's `truncation`). The default `nothing` uses the
[`convolution_support`](@ref) of each line's shape.

Forward-mode differentiable once `ForwardDiff` is loaded, like `LDM` and with
its caveats. Differentiating with respect to a line width needs a fixed
`truncation`: the default one changes in whole grid points with the width, so
the spectrum is not a smooth function of it, and such a call throws.

!!! warning
    The wings are zero beyond the truncation, by default ±10×FWHM. A Lorentzian
    has about 3% of its area out there. If the far wings matter, use `LDM` or a
    wider `truncation`.
"""
struct SparseLDM{W} <: LineSynthesisMethod
    res_G::Float64
    res_L::Float64
    truncation::W
end
function SparseLDM(; res_G = 0.01, res_L = 0.01, truncation = nothing)
    truncation === nothing || truncation > zero(truncation) ||
        throw(ArgumentError("truncation must be positive, got $truncation"))
    return SparseLDM{cutoff_type(truncation)}(res_G, res_L, truncation)
end

function Base.show(io::IO, m::SparseLDM)
    print(io, "SparseLDM(res_G=", m.res_G, ", res_L=", m.res_L)
    isnothing(m.truncation) || print(io, ", truncation=", m.truncation)
    print(io, ")")
end

calculate_lines!(ret, xs::AbstractRange, lines, method::SparseLDM) =
    calculate_lines!(ret, xs, lines, method, SparseLDMWorkspace())

"""
    SparseLDMWorkspace()

Reusable workspace for [`SparseLDM`](@ref):
`calculate_lines!(ret, xs, lines, SparseLDM(), ws)`. Repeated calls, as in a
fitting loop, then allocate next to nothing, also with `ForwardDiff.Dual`
inputs.

Not thread-safe: use one workspace per task.
"""
struct SparseLDMWorkspace
    state::Base.RefValue{Any}
end
SparseLDMWorkspace() = SparseLDMWorkspace(Ref{Any}(nothing))

mutable struct SparseLDMState{S,A,T,C}
    positions::Vector{S}
    amplitudes::Vector{A}
    fwhm_G::Vector{S}
    fwhm_L::Vector{S}
    width_axis_G::Vector{S}
    width_axis_L::Vector{S}
    log_width_axis_G::Vector{S}
    log_width_axis_L::Vector{S}
    trunc_pts::Matrix{Int}
    contributions::Matrix{Vector{Contribution{T}}}
    sort_scratch::Vector{Contribution{T}}
    bin_clusters::Vector{Tuple{Int,Int,Int,Int}}
    cluster_scratch::Vector{Tuple{Int,Int,Int,Int}}
    clusters::Vector{Cluster}
    fftws::SparseFFTBuffers{T,C}
end

function new_sparse_ldm_state(::Type{S}, ::Type{A}, ::Type{T}) where {S,A,T}
    fftws = sparse_fft_buffers(T)
    return SparseLDMState{S,A,T,fft_cache_type(T)}(
        Vector{S}(), Vector{A}(), Vector{S}(), Vector{S}(),
        Vector{S}(), Vector{S}(), Vector{S}(), Vector{S}(),
        Matrix{Int}(undef, 0, 0), Matrix{Vector{Contribution{T}}}(undef, 0, 0),
        Vector{Contribution{T}}(), Vector{Tuple{Int,Int,Int,Int}}(),
        Vector{Tuple{Int,Int,Int,Int}}(), Vector{Cluster}(), fftws)
end

# Grow the shared FFT buffers to the largest transform of this call.
function ensure_sparse_capacity!(ws::SparseFFTBuffers, max_n_pad::Int)
    max_n_pad = max(max_n_pad, 2)
    length(ws.scatter_buf) >= max_n_pad && return ws
    max_n_freq = max_n_pad ÷ 2 + 1
    resize!(ws.scatter_buf, max_n_pad)
    resize!(ws.fft, max_n_freq)
    resize!(ws.conv, max_n_pad)
    resize!(ws.kernel_fft, max_n_freq)
    return ws
end

function calculate_lines!(ret, xs::AbstractRange, lines, method::SparseLDM,
                          ws::SparseLDMWorkspace)
    check_axes(ret, xs)
    check_ascending(xs, "SparseLDM")
    if isempty(lines)
        fill!(ret, zero(eltype(ret)))
        return ret
    end
    l1 = first(lines)
    S = typeof(l1.position)
    A = typeof(l1.amplitude)
    T = promote_type(S, A)
    fft_eltype_check(T, "SparseLDM")
    st = ws.state[]
    if !(st isa SparseLDMState{S,A,T,fft_cache_type(T)})
        st = new_sparse_ldm_state(S, A, T)
        ws.state[] = st
    end
    return sparse_ldm_ws!(ret, xs, lines, method, st)
end

calculate_lines!(ret, xs, lines, method::SparseLDM, ::SparseLDMWorkspace) =
    calculate_lines!(ret, xs, lines, method)

function sparse_ldm_ws!(ret, xs, lines, method::SparseLDM,
                         st::SparseLDMState{S,A,T}) where {S,A,T}
    load_lines!(st, lines, method)

    n_x = length(xs)
    x0 = first(xs)
    x_end = last(xs)
    dx = step(xs)
    inv_dx = inv(dx)
    n_G, n_L = length(st.width_axis_G), length(st.width_axis_L)

    if size(st.trunc_pts) != (n_G, n_L)
        st.trunc_pts = Matrix{Int}(undef, n_G, n_L)
        st.contributions = [Contribution{T}[] for _ in 1:n_G, _ in 1:n_L]
    end
    if method.truncation === nothing
        check_width_derivatives(st.fwhm_G, st.fwhm_L)
        @inbounds for m in 1:n_L, l in 1:n_G
            σ_lm, γ_lm = bin_sigma_gamma(st.width_axis_G[l], st.width_axis_L[m])
            lo, hi = convolution_support(Voigt(σ_lm, γ_lm))
            st.trunc_pts[l, m] = ceil(Int, max(-lo, hi) / dx)
        end
    else
        fill!(st.trunc_pts, ceil(Int, Float64(method.truncation / dx)))
    end

    sparse_ldm_bins!(st.contributions, x0, x_end, dx, n_x, st.positions, st.amplitudes,
                      st.fwhm_G, st.fwhm_L, st.width_axis_G, st.width_axis_L, st.log_width_axis_G, st.log_width_axis_L)

    # All clusters are found first, to size the shared buffers. Each bin's
    # clusters are sorted by length, so equal FFT lengths share one kernel FFT.
    empty!(st.clusters)
    max_n_pad = 1
    @inbounds for m in 1:n_L, l in 1:n_G
        contribs = st.contributions[l, m]
        isempty(contribs) && continue
        tp = st.trunc_pts[l, m]
        max_n_pad = max(max_n_pad, kernel_fft_length(tp, dx, st.width_axis_G[l], st.width_axis_L[m]))
        cluster_contributions!(st.bin_clusters, contribs, tp, st.sort_scratch)
        sort!(st.bin_clusters; by = c -> c[2] - c[1], scratch = st.cluster_scratch)
        for (cluster_lo, cluster_hi, first_idx, last_idx) in st.bin_clusters
            max_n_pad = max(max_n_pad, cluster_fft_length(cluster_lo, cluster_hi))
            push!(st.clusters, (; l, m, lo = cluster_lo, hi = cluster_hi, first_idx, last_idx))
        end
    end
    ensure_sparse_capacity!(st.fftws, max_n_pad)

    fill!(ret, zero(eltype(ret)))
    bin = (0, 0)
    kernel_n_pad = 0
    @inbounds for (l, m, cluster_lo, cluster_hi, first_idx, last_idx) in st.clusters
        tp = st.trunc_pts[l, m]
        if (l, m) != bin
            σ_lm, γ_lm = bin_sigma_gamma(st.width_axis_G[l], st.width_axis_L[m])
            n_kernel = kernel_fft_length(tp, dx, st.width_axis_G[l], st.width_axis_L[m])
            sparse_ldm_kernel!(st.fftws, tp, n_kernel, dx, σ_lm, γ_lm)
            bin = (l, m)
            kernel_n_pad = 0
        end
        n_pad = cluster_fft_length(cluster_lo, cluster_hi)
        if n_pad != kernel_n_pad
            sparse_ldm_kernel_fft!(st.fftws, tp, n_pad)
            kernel_n_pad = n_pad
        end
        convolve_cluster!(ret, st.contributions[l, m], first_idx, last_idx,
                          cluster_lo, cluster_hi, n_pad, n_x, inv_dx, st.fftws)
    end
    return ret
end

calculate_lines!(ret, xs, lines, ::SparseLDM) = error(
    "SparseLDM requires an evenly-spaced grid: pass `xs` as an `AbstractRange`, " *
    "or use `PointSample()` for an arbitrary grid."
)

calculate_lines(xs, lines, method::LDM, ws::LDMWorkspace) =
    calculate_lines!(zeros(spectrum_eltype(xs, lines), length(xs)), xs, lines, method, ws)
calculate_lines(xs, lines, method::SparseLDM, ws::SparseLDMWorkspace) =
    calculate_lines!(zeros(spectrum_eltype(xs, lines), length(xs)), xs, lines, method, ws)
