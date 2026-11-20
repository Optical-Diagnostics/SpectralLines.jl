"""
    LineShape{T}

Abstract supertype for normalized line-shape profiles, callable as `shape(x)`.
"""
abstract type LineShape{T} end

# interface

Base.broadcastable(x::LineShape) = Ref(x)

(ls::LineShape)(x)= error("$(typeof(ls)) must implement the call operator.")

"""
    fwhm(shape) -> w

Full width at half maximum of `shape`. Exact for every shape except `Voigt`,
which uses the Olivero-Longbothum combination formula (accurate to ~0.02%).
"""
fwhm(ls::LineShape) = error("$(typeof(ls)) must implement a fwhm function")

"""
    hwhm(shape) -> w

Half width at half maximum, `fwhm(shape) / 2`.
"""
hwhm(ls::LineShape) = fwhm(ls)/2

# Constants in the float type of `x`, so a Float32 width is not widened by a
# Float64 literal.
@inline _pi(x) = oftype(float(x), π)
@inline twopi(x) = 2 * _pi(x)
@inline sqrt2(x) = sqrt(oftype(float(x), 2))
# 2√(2 ln 2), the Gaussian FWHM-to-σ ratio.
@inline fwhm_over_sigma(x) = 2 * sqrt(2 * log(oftype(float(x), 2)))

"""
    numtype_convert(::Type{T}, shape) -> LineShape{T}

Rebuild `shape` with its numeric type parameter changed to `T`, leaving any
non-numeric type parameters intact. Used by [`Line`](@ref) to promote a shape to
the numeric type of the line's position.

The generic fallback calls `Shape{T}(shape)`, which binds `T` to the first type
parameter. Every shape in this package leads with its numeric parameter. A
custom shape that does not must define its own method.
"""
numtype_convert(::Type{T}, ls::LS) where {T,LS<:LineShape} =
    Base.typename(LS).wrapper{T}(ls)

"""
    integrate(shape, a, b) -> T

The integral ``\\int_a^b \\text{shape}(x)\\,dx``, with `a` and `b` positions
relative to the line centre. [`BinAverage`](@ref) uses it to average each line over a
grid cell, which matters once the grid spacing is comparable to the linewidth.

Exact to rounding for every shape except `Voigt`, whose relative error is below
0.02%.
"""
integrate(ls::LineShape, a, b) = error("$(typeof(ls)) must implement an integrate function")

"""
    convolution_support(shape) -> (lo, hi)

The offset range `(lo, hi)` around the shape's centre (`lo ≤ 0 ≤ hi`) outside of
which `shape` is zero or negligible. Sizes the kernels and padding of the FFT
convolutions.

The fallback, for shapes with unbounded support, is `±10 × fwhm(shape)`. 
Shapes with finite support override it, e.g. `Triangular`.
"""
function convolution_support(ls)
    w = 10 * fwhm(ls)
    return (-w, w)
end

