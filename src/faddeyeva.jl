# The Faddeeva function w(z) = exp(-z²) erfc(-iz), evaluated piecewise in the
# upper half-plane and extended to the lower one by reflection.
#
# Region layout after Zaghloul (2017), ACM TOMS Algorithm 985, Table 1: Laplace
# continued-fraction convergents away from the origin, Humlíček's (1982) w4
# region-IV formula along the real axis, and in the remaining inner region
# Weideman's (1994) rational approximation, which here takes the place of the
# Hui et al. (1978) formula Zaghloul uses.

using FFTW: fft, fftshift

# Region borders on s = |z|² and y², Zaghloul (2017) Table 1.
const S_I   = 3.8e4    # s ≥ S_I:          1 convergent
const S_II  = 256.0    # S_II ≤ s < S_I:   2 convergents
const S_III = 62.0     # S_III ≤ s < S_II: 3 convergents
const S_IV  = 30.0     # S_IV ≤ s < S_III: 4 convergents, if y² ≥ Y2_IV
const S_V   = 2.5      # S_V ≤ s < S_IV:   Humlíček, if y² < Y2_V
const Y2_IV = 1.0e-13  # below it, S_IV ≤ s < S_III is Humlíček's region too
const Y2_V  = 0.072
# Humlíček's exp(-z²) term is replaced by exp(-x²) for y² below this
# (Zaghloul 2017, Sec. 3.2).
const Y2_EXP = 1.0e-13

@inline invsqrtpi(::Type{T}) where {T} = inv(sqrt(T(π)))

# n/d without the rescaling of Julia's complex division, which guards against
# overflow. Used only where |z|² < 256, so operands stay moderate.
@inline unguarded_cdiv(n::Complex, d::Complex) = n * conj(d) / abs2(d)
@inline unguarded_cdiv_re(n::Complex, d::Complex) = (real(n) * real(d) + imag(n) * imag(d)) / abs2(d)

# Laplace continued fraction, w(z) = (i/√π) / (z − ½/(z − 1/(z − (3/2)/(z − …)))).
# Its first four convergents, simplified:
#   i/(√π z),   i z/(√π (z² − ½)),   i (z² − 1)/(√π z (z² − 3/2)),
#   i z (z² − 5/2)/(√π (z⁴ − 3z² + 3/4)).
@inline function _laplace(z::Complex{T}, n::Int) where {T<:Real}
    c = im * invsqrtpi(T)
    n == 1 && return c * inv(z)
    z2 = z * z
    n == 2 && return unguarded_cdiv(c * z, z2 - T(0.5))
    n == 3 && return unguarded_cdiv(c * (z2 - 1), z * (z2 - T(1.5)))
    return unguarded_cdiv(c * z * (z2 - T(2.5)), z2 * (z2 - 3) + T(0.75))
end

# Humlíček (1982), w4 region IV, in his variables t = y − ix and u = t² = −z²:
#   w ≈ exp(u) − t P(u)/Q(u).
const HUMLICEK_P = (36183.31, -3321.99, 1540.787, -219.0313, 35.76683, -1.320522, 0.56419)
const HUMLICEK_Q = (32066.6, -24322.84, 9022.228, -2186.181, 364.2191, -61.57037, 1.841439, -1.0)

@inline function _humlicek(x::T, y::T) where {T<:Real}
    t = complex(y, -x)
    u = t * t
    r = unguarded_cdiv(t * evalpoly(u, map(T, HUMLICEK_P)), evalpoly(u, map(T, HUMLICEK_Q)))
    # exp(u) = exp(y² − x²) (cos 2xy − i sin 2xy); for tiny y, exp(−x²).
    e = y * y < Y2_EXP ? complex(exp(-x * x)) : exp(u)
    return e - r
end

# Real part of `_humlicek`: Re exp(u) = exp(y² − x²) cos 2xy needs no complex exp.
@inline function _humlicek_re(x::T, y::T) where {T<:Real}
    t = complex(y, -x)
    u = t * t
    r = unguarded_cdiv_re(t * evalpoly(u, map(T, HUMLICEK_P)), evalpoly(u, map(T, HUMLICEK_Q)))
    y2 = y * y
    e = y2 < Y2_EXP ? exp(-x * x) : exp(y2 - x * x) * cos(2 * x * y)
    return e - r
end

# Weideman (1994): with Z = (L + iz)/(L − iz),
#   w(z) ≈ 2 p(Z)/(L − iz)² + 1/(√π (L − iz)),
# p a polynomial of degree N − 1 whose coefficients are the discrete Fourier
# coefficients of (L² + t²) exp(−t²) at t = L tan(θ/2), θ = kπ/M, M = 2N.
# Uniform in absolute error, so used only where |w| is not small.
function weideman_coefficients(N::Int)
    M = 2N
    L = sqrt(N / sqrt(2))
    t = [L * tan(k * π / (2M)) for k in (-M + 1):(M - 1)]
    f = vcat(0.0, exp.(-t .^ 2) .* (L^2 .+ t .^ 2))
    a = real.(fft(fftshift(f))) ./ (2M)
    return L, Tuple(a[(N + 1):-1:2])    # highest degree first
end

const WEIDEMAN_N = 16
const WEIDEMAN_L, WEIDEMAN_A = weideman_coefficients(WEIDEMAN_N)

@inline function _weideman(z::Complex{T}) where {T<:Real}
    L = T(WEIDEMAN_L)
    g = inv(L - im * z)
    Z = (L + im * z) * g
    p = zero(Z)
    for a in WEIDEMAN_A
        p = muladd(p, Z, T(a))
    end
    return 2p * g * g + invsqrtpi(T) * g
end

@inline function _weideman_re(x::T, y::T) where {T<:Real}
    L = T(WEIDEMAN_L)
    dr, di = L + y, -x                       # L − iz
    n = dr * dr + di * di
    gr, gi = dr / n, -di / n                 # g = 1/(L − iz)
    ar, ai = L - y, x                        # L + iz
    Zr, Zi = ar * gr - ai * gi, ar * gi + ai * gr
    pr, pi_ = zero(T), zero(T)
    for a in WEIDEMAN_A
        pr, pi_ = muladd(pr, Zr, muladd(-pi_, Zi, T(a))), muladd(pr, Zi, pi_ * Zr)
    end
    g2r, g2i = gr * gr - gi * gi, 2 * gr * gi
    return 2 * (pr * g2r - pi_ * g2i) + invsqrtpi(T) * gr
end

# Regions V (`humlicek`) and VI, near the origin. Kept apart so the ForwardDiff
# extension can give region VI a derivative rule; the other regions
# differentiate well as they are.
@inline _w_inner(x::T, y::T, humlicek::Bool) where {T<:Real} =
    humlicek ? _humlicek(x, y) : _weideman(complex(x, y))
@inline _w_inner_re(x::T, y::T, humlicek::Bool) where {T<:Real} =
    humlicek ? _humlicek_re(x, y) : _weideman_re(x, y)

# w(z) for y ≥ 0.
@inline function _w_upper(x::T, y::T) where {T<:Real}
    z = complex(x, y)
    s = x * x + y * y
    s >= S_I && return _laplace(z, 1)
    s >= S_II && return _laplace(z, 2)
    s >= S_III && return _laplace(z, 3)
    y2 = y * y
    s >= S_IV && y2 >= Y2_IV && return _laplace(z, 4)
    return _w_inner(x, y, s >= S_IV || (s >= S_V && y2 < Y2_V))
end

# Re w(z) for y ≥ 0. The first two convergents have short real forms:
#   Re i/(√π z)          = y / (√π s)
#   Re i z/(√π (z² − ½)) = y (s + ½) / (√π (s² + y² − x² + ¼))
@inline function _w_upper_re(x::T, y::T) where {T<:Real}
    s = x * x + y * y
    if s >= S_I
        isfinite(s) || return real(_laplace(complex(x, y), 1))
        return y * invsqrtpi(T) / s
    end
    s >= S_II && return y * (s + T(0.5)) * invsqrtpi(T) / (s * s + (y * y - x * x) + T(0.25))
    z = complex(x, y)
    c = im * invsqrtpi(T)
    if s >= S_III
        z2 = z * z
        return unguarded_cdiv_re(c * (z2 - 1), z * (z2 - T(1.5)))
    end
    y2 = y * y
    if s >= S_IV && y2 >= Y2_IV
        z2 = z * z
        return unguarded_cdiv_re(c * z * (z2 - T(2.5)), z2 * (z2 - 3) + T(0.75))
    end
    return _w_inner_re(x, y, s >= S_IV || (s >= S_V && y2 < Y2_V))
end

# Lower half-plane by reflection: w(z) = 2 exp(−z²) − w(−z), with −z in the
# upper half-plane.
@inline function _w(x::T, y::T) where {T<:Real}
    y >= zero(y) && return _w_upper(x, y)
    return 2 * exp(-complex(x, y)^2) - _w_upper(-x, -y)
end

@inline function _w_re(x::T, y::T) where {T<:Real}
    y >= zero(y) && return _w_upper_re(x, y)
    # Re exp(−z²) = exp(y² − x²) cos 2xy; Re w is even in x.
    return 2 * exp(y * y - x * x) * cos(2 * x * y) - _w_upper_re(x, -y)
end

"""
    faddeyeva(z::Complex) -> Complex
    faddeyeva(x, y) -> Real

The Faddeeva (scaled complex complementary error) function

```math
w(z) = e^{-z^2} \\operatorname{erfc}(-iz), \\quad z = x + iy,
```

whose real part gives the Voigt profile:
`V(x; σ, γ) = faddeyeva(x/(σ√2), γ/(σ√2)) / (σ√(2π))`. The two-argument form
returns `real(w(x + iy))` with real arithmetic, about twice as fast.

Relative error is below 4×10⁻⁵, and the value jumps by up to that error where
the approximation switches between regions of the complex plane. The method
combines a Laplace continued fraction, Humlíček's w4 formula and Weideman's
rational approximation, with regions chosen as in Zaghloul's Algorithm 985.

Float32 and Float64 arguments give results of the same type. Differentiable
with ForwardDiff, with derivative errors below 4×10⁻⁵ relative to ``|w'|``.

## References

- Zaghloul, M. R. (2017). Algorithm 985: Simple, efficient, and relatively
  accurate approximation for the evaluation of the Faddeyeva function. *ACM
  Trans. Math. Softw.*, **44**(2), 22. doi:10.1145/3119904
- Humlíček, J. (1982). Optimized computation of the Voigt and complex
  probability functions. *J. Quant. Spectrosc. Radiat. Transfer*, **27**(4),
  437–444. doi:10.1016/0022-4073(82)90078-4
- Weideman, J. A. C. (1994). Computation of the complex error function. *SIAM J.
  Numer. Anal.*, **31**(5), 1497–1518. doi:10.1137/0731077
"""
faddeyeva(z::Complex) = _w(float(real(z)), float(imag(z)))
faddeyeva(x::Real, y::Real) = _w_re(promote(float(x), float(y))...)
