"""
    Lorentzian{T} <: LineShape{T}

Cauchy (Lorentzian) line-shape profile parameterized by its half-width at half-maximum.

```math
L(x;\\,\\gamma) = \\frac{1}{\\pi}\\frac{\\gamma}{x^2 + \\gamma^2}
```

Normalized to unit area. Construct with `Lorentzian(γ)`.

# Fields
- `hwhm::T`: half-width at half-maximum ``\\gamma``
"""
struct Lorentzian{T} <: LineShape{T}
    hwhm::T
end
Lorentzian{T}(l::Lorentzian) where {T} = Lorentzian{T}(l.hwhm) # copy constructor

@inline lorentzian(x, γ) = γ / ((abs2(γ) + abs2(x)) * _pi(γ))
@inline (L::Lorentzian{T})(x) where {T} = lorentzian(x, L.hwhm)
@inline fwhm(l::Lorentzian{T}) where {T} = 2 * l.hwhm
# atan(u) - atan(v). Far in one wing both are ≈ ±π/2 and their difference loses
# relative precision. For finite u, v of the same sign, uv ≥ 0 and the
# subtraction formula atan((u - v) / (1 + uv)) holds exactly without cancelling.
@inline function atan_diff(u, v)
    ((u < zero(u)) == (v < zero(v)) && isfinite(u) && isfinite(v)) ||
        return atan(u) - atan(v)
    return atan((u - v) / (1 + u * v))
end
integrate(l::Lorentzian, a, b) = atan_diff(b / l.hwhm, a / l.hwhm) / _pi(l.hwhm)

Base.show(io::IO, l::Lorentzian) = print(io, "Lorentzian(γ=", l.hwhm, ")")




