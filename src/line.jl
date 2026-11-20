"""
    Line{Tx, Ta, LS<:LineShape{Tx}}

A single spectral line: a `shape` (any [`LineShape`](@ref)) centred at
`position` and scaled by `amplitude`. Callable as `line(x)`, evaluating
`amplitude * shape(x - position)`.

Construct with `Line(position, amplitude, shape)`. Integer arguments are
converted to floating point.

[`calculate_lines`](@ref) and `calculate_lines!` sum many `Line`s over a grid to
build a spectrum.

# Fields
- `position::Tx`: line centre
- `amplitude::Ta`: scale factor
- `shape::LS`: the (possibly instrument-convolved, see [`convolve`](@ref))
  line-shape function
"""
struct Line{Tx,Ta,LS<:LineShape{Tx}}
    position::Tx
    amplitude::Ta
    shape::LS
    # Only the parametrized form, so `Line(p, a, s)` always runs the outer
    # constructor and its conversions.
    Line{Tx,Ta,LS}(position, amplitude, shape) where {Tx,Ta,LS<:LineShape{Tx}} =
        new{Tx,Ta,LS}(position, amplitude, shape)
end
Base.broadcastable(x::Line) = Ref(x)
function Base.show(io::IO, l::Line)
    print(io, "Line(pos=", l.position, ", amp=", l.amplitude, ", shape=")
    show(io, l.shape)
    print(io, ")")
end


function Line(position, amplitude, shape::LS) where {T,LS<:LineShape{T}}
    Tx = promote_type(typeof(position), T)
    Ta = float(typeof(amplitude))
    new_shape = numtype_convert(Tx, shape)
    return Line{Tx,Ta,typeof(new_shape)}(position, amplitude, new_shape)
end

@inline (l::Line)(x) = l.amplitude * l.shape(x - l.position)

integrate(l::Line, a, b) = l.amplitude * integrate(l.shape, a - l.position, b - l.position)


"""
    update_lines!(lines; positions=nothing, amplitudes=nothing, shapes=nothing) -> lines

Replace the given per-line fields of `lines` in place; omitted fields are kept.
For fitting loops that re-synthesize with new parameters without rebuilding the
line list.

New values are converted to the types already stored in `lines`. A
different type (a `ForwardDiff.Dual`, another shape family) needs a freshly
built line list. Throws `DimensionMismatch` unless every supplied vector shares
indices with `lines`.

Use it only when all lines have the same shape type. For a mixed-shape vector,
rebuilding the list is faster.
"""
update_lines!(lines; positions = nothing, amplitudes = nothing, shapes = nothing) =
    _update!(lines, positions, amplitudes, shapes)

_update!(lines, ::Nothing, ::Nothing, ::Nothing) = lines

function _update!(lines, positions, amplitudes, shapes)
    @inbounds for i in eachindex(lines, index_args(positions, amplitudes, shapes)...)
        l = lines[i]
        pos = new_position(positions, i, l)
        lines[i] = Line(pos, new_amplitude(amplitudes, i, l), new_shape(shapes, i, l, pos))
    end
    return lines
end

index_args() = ()
index_args(::Nothing, rest...) = index_args(rest...)
index_args(x, rest...) = (x, index_args(rest...)...)

new_position(::Nothing, i, l) = l.position
Base.@propagate_inbounds new_position(ps, i, l) = oftype(l.position, ps[i])
new_amplitude(::Nothing, i, l) = l.amplitude
Base.@propagate_inbounds new_amplitude(as, i, l) = oftype(l.amplitude, as[i])
new_shape(::Nothing, i, l, pos) = l.shape
Base.@propagate_inbounds new_shape(ss, i, l, pos) = numtype_convert(typeof(pos), ss[i])

