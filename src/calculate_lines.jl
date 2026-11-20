"""
    LineSynthesisMethod

Abstract supertype for spectrum-synthesis methods: [`PointSample`](@ref),
[`BinAverage`](@ref), [`LDM`](@ref), and [`SparseLDM`](@ref). A method implements
[`calculate_lines!`](@ref).
"""
abstract type LineSynthesisMethod end

"""
    calculate_lines!(ret, xs, lines, method [, workspace]) -> ret

In-place form of [`calculate_lines`](@ref): overwrite `ret` with the spectrum
synthesized on `xs` and return it. `LDM` and `SparseLDM` additionally accept a
[`LDMWorkspace`](@ref) / [`SparseLDMWorkspace`](@ref) as fifth argument, which
makes repeated calls (near-)allocation-free.
"""
calculate_lines!(ret, xs, lines, method::LineSynthesisMethod) =
    error("$(typeof(method)) must implement calculate_lines!.")

"""
    calculate_lines(xs, lines, method::LineSynthesisMethod) -> Vector

Allocating variant of `calculate_lines!`.

## Automatic differentiation

`PointSample` and `BinAverage` are forward-mode differentiable: a `ForwardDiff.Dual` in a
line's position, amplitude or shape parameters propagates through. `LDM` and
`SparseLDM` are differentiable too once `ForwardDiff` is loaded (see
[`LDM`](@ref) for the accuracy caveats).

Reverse mode (Zygote, Enzyme) is unsupported for now.
"""
function calculate_lines(xs, lines, method::LineSynthesisMethod)
    ret = zeros(spectrum_eltype(xs, lines), length(xs))
    calculate_lines!(ret, xs, lines, method)
    return ret
end

line_amplitude_type(::Type{<:Line{Tx,Ta}}, _) where {Tx,Ta} = Ta
line_amplitude_type(::Type, x_T) = x_T
line_position_type(::Type{<:Line{Tx,Ta}}, _) where {Tx,Ta} = Tx
line_position_type(::Type, x_T) = x_T

function spectrum_eltype(xs, lines)
    x_T = typeof(first(xs))
    if isempty(lines)
        LT = eltype(lines)
        return promote_type(x_T, line_position_type(LT, x_T), line_amplitude_type(LT, x_T))
    end
    line = first(lines)
    return promote_type(x_T, typeof(line.position), typeof(line.amplitude))
end

function check_axes(ret, xs)
    axes(ret) == axes(xs) || throw(DimensionMismatch(
        "`ret` has indices $(axes(ret)), but `xs` has $(axes(xs))"))
    return nothing
end

# Whether `line` is too weak to synthesize. Compares `|amplitude|`: a large
# negative amplitude (gain from a population inversion) is a strong line.
@inline below_cutoff(line, min_amplitude) =
    min_amplitude !== nothing && abs(line.amplitude) < min_amplitude

cutoff_type(::Nothing) = Nothing
cutoff_type(x) = typeof(x)

is_ascending(xs::AbstractRange) = length(xs) <= 1 || step(xs) > zero(step(xs))
is_ascending(xs) = issorted(xs)
function check_ascending(xs, what)
    is_ascending(xs) || throw(ArgumentError(
        "$what needs an ascending grid; reverse `xs` (and the result) instead"))
    return nothing
end

# Index range [lo, hi] of xs within wing_cut half-widths of the line centre.
function line_index_range(xs, line, wing_cut)::Tuple{Int,Int}
    wing_cut === nothing && return (firstindex(xs), lastindex(xs))
    hw = fwhm(line.shape) / 2
    lo = searchsortedfirst(xs, line.position - wing_cut * hw)
    hi = searchsortedlast(xs, line.position + wing_cut * hw)
    return lo, hi
end

should_thread(xs, lines, method) =
    Threads.nthreads() > 1 && length(xs) * length(lines) >= thread_min_work(method)

chunk_bounds(n::Int, nchunks::Int) = round.(Int, range(0, n, length = nchunks + 1))

# Grid-chunked driver for `PointSample` and `BinAverage`: one task per chunk, each running
# `grid_serial!` on views.
function grid_chunked!(ret, xs, lines, method,
                        nchunks::Int = min(4 * Threads.nthreads(), length(xs)))
    b0 = firstindex(xs) - 1
    bounds = chunk_bounds(length(xs), nchunks)
    Threads.@threads for c in 1:nchunks
        lo = b0 + bounds[c] + 1
        hi = b0 + bounds[c + 1]
        lo > hi && continue
        grid_serial!(view(ret, lo:hi), view(xs, lo:hi), lines, method)
    end
    return ret
end
