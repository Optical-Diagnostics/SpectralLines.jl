"""
    Triangular{T} <: LineShape{T}

Triangular line-shape profile: zero at ``|x| \\geq w``, linearly rising to a
peak at ``x=0``, with `w` the full width at half maximum (so the base spans
``2w``). Normalized to unit area. Construct with `Triangular(w)`.

```math
T(x;\\,w) = \\begin{cases} \\frac{1}{w}\\left(1 - \\frac{|x|}{w}\\right) & |x| < w \\\\ 0 & |x| \\geq w \\end{cases}
```

Its convolution with a `Gaussian`, `Lorentzian` or `Voigt` is not one of this
package's shapes, so it has no [`convolve`](@ref) method. Apply it as an
instrument with `calculate_lines(xs, lines, instrument_function)`, which
FFT-convolves the synthesized spectrum.

The derivative with respect to `w` jumps at ``|x| = w``. A grid point exactly on
that edge gets a zero derivative.

# Fields
- `fwhm::T`: full width at half maximum ``w``
"""
struct Triangular{T} <: LineShape{T}
    fwhm::T
end
Triangular{T}(t::Triangular) where {T} = Triangular{T}(t.fwhm) # copy constructor

@inline function (L::Triangular)(x)
    w = L.fwhm
    ax = abs(x)
    return ax < w ? (1 - ax / w) / w : zero(ax / w)
end

@inline fwhm(t::Triangular) = t.fwhm

# Split at the peak into one piece per flank, each integrated in factored form,
# which does not cancel near the edges or the peak.
function integrate(t::Triangular, a, b)
    b < a && return -integrate(t, b, a)
    w = t.fwhm
    lo, hi = clamp(a, -w, w), clamp(b, -w, w)
    z = zero(lo)
    return triangle_flank(w, max(lo, z), max(hi, z)) +
           triangle_flank(w, max(-hi, z), max(-lo, z))
end

# ∫_p^q (1 - x/w) / w dx for 0 ≤ p ≤ q ≤ w.
@inline triangle_flank(w, p, q) = (q - p) * (2w - p - q) / (2 * abs2(w))

convolution_support(t::Triangular) = (-t.fwhm, t.fwhm)

Base.show(io::IO, t::Triangular) = print(io, "Triangular(fwhm=", t.fwhm, ")")
