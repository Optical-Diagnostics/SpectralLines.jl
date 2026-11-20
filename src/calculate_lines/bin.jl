"""
    BinAverage(; bin_width, wing_cut=nothing, min_amplitude=nothing)

Bin-averaged line synthesis: each grid point accumulates the profile's mean
value over a bin of width `bin_width` centred on the point,
`(1/bin_width) ∫ φ(x) dx`, rather than a point sample. If `wing_cut` is set,
each line is only evaluated within `±wing_cut × hwhm` of its centre; this
needs an ascending grid.

Use `BinAverage` when the grid spacing is comparable to or coarser than the linewidth.
There point samples miss most of a narrow line or overweight it, depending on where its
centre falls between grid points; with `bin_width` equal to the grid step, the
bins tile the axis and each line keeps its area. Use `PointSample` when the grid
resolves the profile.
"""
struct BinAverage{W,A} <: LineSynthesisMethod
    bin_width::W
    wing_cut::Union{Nothing,Float64}
    min_amplitude::A
end
function BinAverage(; bin_width, wing_cut = nothing, min_amplitude = nothing)
    w = float(bin_width)
    return BinAverage{typeof(w),cutoff_type(min_amplitude)}(w, wing_cut, min_amplitude)
end

function Base.show(io::IO, m::BinAverage)
    print(io, "BinAverage(bin_width=", m.bin_width)
    isnothing(m.wing_cut) || print(io, ", wing_cut=", m.wing_cut)
    isnothing(m.min_amplitude) || print(io, ", min_amplitude=", m.min_amplitude)
    print(io, ")")
end

@inline function add_line_binned!(ret, xs, line, lo::Int, hi::Int, half_bin_width, inv_bin_width)
    pos = line.position
    amp = line.amplitude * inv_bin_width
    @inbounds for i in lo:hi
        offset = xs[i] - pos
        ret[i] += amp * integrate(line.shape, offset - half_bin_width, offset + half_bin_width)
    end
    return nothing
end

thread_min_work(::BinAverage) = 512

grid_serial!(ret, xs, lines, method::BinAverage) = bin_serial!(ret, xs, lines, method)

function calculate_lines!(ret, xs, lines, method::BinAverage)
    check_axes(ret, xs)
    method.wing_cut === nothing || check_ascending(xs, "BinAverage with wing_cut")
    should_thread(xs, lines, method) && return grid_chunked!(ret, xs, lines, method)
    return bin_serial!(ret, xs, lines, method)
end

function bin_serial!(ret, xs, lines, method::BinAverage)
    fill!(ret, zero(eltype(ret)))
    half_bin_width = method.bin_width / 2
    inv_bin_width = inv(method.bin_width)
    min_amp = method.min_amplitude
    for line in lines
        below_cutoff(line, min_amp) && continue
        lo, hi = line_index_range(xs, line, method.wing_cut)
        lo > hi && continue
        add_line_binned!(ret, xs, line, lo, hi, half_bin_width, inv_bin_width)
    end
    return ret
end
