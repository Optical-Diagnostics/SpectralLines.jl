"""
    Gaussian{T} <: LineShape{T}

Gaussian line-shape profile parameterized by its standard deviation `sigma`.

```math
G(x;\\,\\sigma) = \\frac{1}{\\sigma\\sqrt{2\\pi}}\\exp\\left(-\\frac{x^2}{2\\sigma^2}\\right)
```

Normalized to unit area. Construct with `Gaussian(σ)`.

# Fields
- `sigma::T`: standard deviation ``\\sigma``
"""
struct Gaussian{T} <: LineShape{T}
    sigma::T
    normalization::T
    inv_sigma_sqrt2::T
end
Gaussian(σ) = Gaussian{typeof(float(σ))}(float(σ))
Gaussian{T}(σ) where {T} = Gaussian{T}(σ, inv(sqrt(twopi(σ)) * σ), inv(σ * sqrt2(σ)))
Gaussian{T}(g::Gaussian) where {T} = Gaussian{T}(g.sigma, g.normalization, g.inv_sigma_sqrt2) # copy constructor

# interface
@inline (L::Gaussian)(x) = L.normalization * exp(-abs2(x * L.inv_sigma_sqrt2))
@inline fwhm(g::Gaussian) = fwhm_over_sigma(g.sigma) * g.sigma
# erf(u) - erf(v). Two values far in the same tail are both ≈ ±1 and their
# difference loses relative precision; the difference of the erfc does not.
@inline function erf_diff(u, v)
    u > zero(u) && v > zero(v) && return erfc(v) - erfc(u)
    u < zero(u) && v < zero(v) && return erfc(-u) - erfc(-v)
    return erf(u) - erf(v)
end
integrate(g::Gaussian, a, b) = erf_diff(b * g.inv_sigma_sqrt2, a * g.inv_sigma_sqrt2) / 2

Base.show(io::IO, g::Gaussian) = print(io, "Gaussian(σ=", g.sigma, ")")


@inline gaussian(x, σ) = inv(sqrt(twopi(σ)) * σ) * exp(-abs2(x / (σ * sqrt2(σ))))
