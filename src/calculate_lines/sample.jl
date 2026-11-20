"""
    PointSample(; wing_cut=nothing, min_amplitude=nothing)

Point-sampling line synthesis: each grid point takes the line shape's value
directly. If `wing_cut` is set, each line is only evaluated within
`±wing_cut × hwhm` of its centre; this needs an ascending grid.

`min_amplitude` skips lines whose `|amplitude|` falls below it. The threshold
is absolute, in the units of the amplitudes. Weak lines that are never wanted
are better filtered out before the `Line` objects are built.

Multithreaded when Julia runs with threads (`julia -t N`).
"""
struct PointSample{A} <: LineSynthesisMethod
    wing_cut::Union{Nothing,Float64}
    min_amplitude::Union{Nothing,A}
end
PointSample(; wing_cut = nothing, min_amplitude = nothing) =
    PointSample{cutoff_type(min_amplitude)}(wing_cut, min_amplitude)

function Base.show(io::IO, m::PointSample)
    print(io, "PointSample(")
    parts = String[]
    isnothing(m.wing_cut) || push!(parts, "wing_cut=$(m.wing_cut)")
    isnothing(m.min_amplitude) || push!(parts, "min_amplitude=$(m.min_amplitude)")
    print(io, join(parts, ", "), ")")
end

@inline function add_line_range!(ret, xs, line, lo::Int, hi::Int)
    rv = @view ret[lo:hi]
    xv = @view xs[lo:hi]
    @. rv += line(xv)
    return nothing
end

thread_min_work(::PointSample) = 10_000

grid_serial!(ret, xs, lines, method::PointSample) = sample_serial!(ret, xs, lines, method)

function calculate_lines!(ret, xs, lines, method::PointSample)
    check_axes(ret, xs)
    method.wing_cut === nothing || check_ascending(xs, "PointSample with wing_cut")
    if isempty(lines)
        fill!(ret, zero(eltype(ret)))
        return ret
    end
    should_thread(xs, lines, method) && return grid_chunked!(ret, xs, lines, method)
    return sample_serial!(ret, xs, lines, method)
end

# Outer loop over lines, broadcast inner over xs: best when length(xs) >> length(lines).
function sample_serial!(ret, xs, lines, method::PointSample)
    wing_cut = method.wing_cut
    min_amp = method.min_amplitude
    if wing_cut === nothing && min_amp === nothing
        @. ret = lines[1](xs)
        @inbounds for i in 2:length(lines)
            add_line_range!(ret, xs, lines[i], firstindex(xs), lastindex(xs))
        end
    else
        fill!(ret, zero(eltype(ret)))
        for line in lines
            below_cutoff(line, min_amp) && continue
            lo, hi = line_index_range(xs, line, wing_cut)
            lo > hi && continue
            add_line_range!(ret, xs, line, lo, hi)
        end
    end
    return ret
end
