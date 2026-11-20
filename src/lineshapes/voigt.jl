"""
    Voigt{T} <: LineShape{T}

Voigt line-shape profile: the convolution of a Gaussian (standard deviation
`sigma`) and a Lorentzian (HWHM `gamma`), evaluated via the real part of the
Faddeeva function [`faddeyeva`](@ref). Its relative error is below 4×10⁻⁵;
`sigma == 0` and `gamma == 0` give the exact Lorentzian and Gaussian.

```math
V(x;\\,\\sigma,\\gamma) = \\frac{1}{\\sigma\\sqrt{2\\pi}}\\,
\\operatorname{Re}\\!\\left[w\\!\\left(\\frac{x}{\\sigma\\sqrt{2}}+i\\frac{\\gamma}{\\sigma\\sqrt{2}}\\right)\\right]
```

Normalized to unit area. Construct with `Voigt(σ, γ)`.

# Fields
- `sigma::T`: Gaussian width ``\\sigma``
- `gamma::T`: Lorentzian HWHM ``\\gamma``
"""
struct Voigt{T} <: LineShape{T}
    sigma::T
    gamma::T
    inv_sigma_sqrt2::T
    y::T
end
function Voigt(s, g)
    T = float(promote_type(typeof(s), typeof(g)))
    return Voigt{T}(convert(T, s), convert(T, g))
end
function Voigt{T}(s, g) where {T}
    rt2 = sqrt2(s)
    return Voigt{T}(s, g, inv(s * rt2), g / (s * rt2))
end
Voigt{T}(v::Voigt) where {T} = Voigt{T}(v.sigma, v.gamma, v.inv_sigma_sqrt2, v.y) # copy constructor


# sigma == 0 would give NaN at x = 0 (0 * Inf), so it takes the exact
# Lorentzian. gamma == 0 takes the exact Gaussian, which is free.
@inline function (L::Voigt)(x)
    iszero(L.sigma) && return lorentzian(x, L.gamma)
    iszero(L.gamma) && return gaussian(x, L.sigma)
    return faddeyeva(x * L.inv_sigma_sqrt2, L.y) * L.inv_sigma_sqrt2 * inv(sqrt(_pi(x)))
end

# Olivero & Longbothum (1977), https://doi.org/10.1016/0022-4073(77)90161-3,
# accurate to ~0.02%. In terms of the two component FWHMs:
# fV = c₁·fL + √(c₂·fL² + fG²).
@inline function fwhm(v::Voigt)
    fG = fwhm_over_sigma(v.sigma) * v.sigma
    fL = 2 * v.gamma
    c1 = oftype(float(fL), 0.5343)
    c2 = oftype(float(fL), 0.2169)
    return c1 * fL + sqrt(c2 * fL^2 + fG^2)
end


# 20-point Gauss-Hermite nodes/weights (physicist's convention):
# ∫_{-∞}^{∞} e^{-x²} f(x) dx ≈ Σ_k w_k f(x_k); Σ w_k = √π.
# Used by integrate(Voigt, a, b) in the Lorentzian-dominated branch (y ≥ 0.7).
const GH20 = let
    n = 20
    J = SymTridiagonal(zeros(n), sqrt.(1:n-1) ./ sqrt(2.0))
    F = eigen(J)
    perm = sortperm(F.values)
    (nodes=F.values[perm], weights=sqrt(Float64(π)) .* F.vectors[1, perm] .^ 2)
end
const GH20_NODES = GH20.nodes
const GH20_WEIGHTS = GH20.weights
const INV_PI_SQRTPI = inv(Float64(π) * sqrt(Float64(π)))

# 10-point Gauss-Legendre nodes/weights on (-1, 1):
# ∫_{-1}^{1} f(t) dt ≈ Σ_k w_k f(t_k); Σ w_k = 2.
# Used by integrate(Voigt, a, b) in the Gaussian-dominated branch (y < 0.7).
const GL10 = let
    n = 10
    β = [k / sqrt(4k^2 - 1.0) for k in 1:n-1]
    J = SymTridiagonal(zeros(n), β)
    F = eigen(J)
    perm = sortperm(F.values)
    (nodes=F.values[perm], weights=2.0 .* F.vectors[1, perm] .^ 2)
end
const GL10_NODES = GL10.nodes
const GL10_WEIGHTS = GL10.weights
# Beyond this many Gaussian FWHMs from the centre, `integrate(::Voigt)` uses the
# Lorentzian wing in closed form. Its relative error there is ~3σ²/x² < 1e-6.
const GL10_FAR_WING = 1000

@inline quad_zero(v, a, b) =
    zero(promote_type(eltype(GH20_WEIGHTS), typeof(a), typeof(b), typeof(v.sigma)))

function integrate(v::Voigt, a, b)
    # Two branches on the Faddeeva y-parameter (y = γ/(σ√2)):
    #
    # y ≥ 0.7, Lorentzian-dominated: 20-pt Gauss-Hermite on the Gaussian kernel.
    #   ∫_a^b V dx = (1/π√π) Σ_k w_k [atan((b−σ√2·x_k)/γ) − atan((a−σ√2·x_k)/γ)]
    #   Error < 0.01%.
    #
    # y < 0.7, Gaussian-dominated: 10-pt Gauss-Legendre panels, see
    # `voigt_gl_side`.
    if v.y >= 0.7
        sigma_rt2 = v.sigma * sqrt2(v.sigma)
        s = quad_zero(v, a, b)
        @inbounds for k in 1:20
            t = sigma_rt2 * GH20_NODES[k]
            s += GH20_WEIGHTS[k] * atan_diff((b - t) / v.gamma, (a - t) / v.gamma)
        end
        return s * INV_PI_SQRTPI
    end
    iszero(v.gamma) && return erf_diff(b * v.inv_sigma_sqrt2, a * v.inv_sigma_sqrt2) / 2
    b < a && return -integrate(v, b, a)
    fG = fwhm_over_sigma(v.sigma) * v.sigma
    b - a <= 2 * fG && return voigt_gl_panel(v, a, b)
    # V is even: fold the interval onto x ≥ 0, one side at a time.
    a >= zero(a) && return voigt_gl_side(v, a, b)
    b <= zero(b) && return voigt_gl_side(v, -b, -a)
    return voigt_gl_side(v, zero(a), -a) + voigt_gl_side(v, zero(b), b)
end

# ∫_p^q V dx for 0 ≤ p ≤ q. A 10-point rule holds to ~1e-4 relative on a panel
# of up to 2×FWHM, so panels start at that width at the centre. Farther out the
# profile varies on the scale of x itself, so a panel spans [x, 2x] there.
function voigt_gl_side(v::Voigt, p, q)
    fG = fwhm_over_sigma(v.sigma) * v.sigma
    p, q, h_min, far = promote(p, q, 2 * fG, GL10_FAR_WING * fG)
    s = quad_zero(v, p, q)
    if q > far
        lo = max(p, far)
        s += atan_diff(q / v.gamma, lo / v.gamma) / _pi(v.gamma)
        q = lo
    end
    x = p
    while x < q
        xe = min(x + max(h_min, x), q)
        s += voigt_gl_panel(v, x, xe)
        x = xe
    end
    return s
end

# 10-point Gauss-Legendre on [lo, hi].
@inline function voigt_gl_panel(v::Voigt, lo, hi)
    half = (hi - lo) / 2
    mid = lo + half
    s = quad_zero(v, lo, hi)
    @inbounds for k in 1:10
        s += GL10_WEIGHTS[k] * v(mid + half * GL10_NODES[k])
    end
    return s * half
end


Base.show(io::IO, v::Voigt) = print(io, "Voigt(σ=", v.sigma, ", γ=", v.gamma, ")")
