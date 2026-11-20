"""Abstract supertype for pseudo-Voigt mixing-parameter models."""
abstract type PseudoVoigtModel end

"""Thompson et al. (1987) 5th-order combined FWHM + cubic η. Default model."""
struct Thompson <: PseudoVoigtModel end

"""Kielkopf (1973) combined FWHM `f ≈ 0.5346 f_L + √(0.2166 f_L² + f_G²)` + Thompson cubic η. Cheaper to construct than Thompson; evaluation costs the same."""
struct Kielkopf <: PseudoVoigtModel end

"""Olivero & Longbothum (1977) damping-factor combined FWHM + Thompson cubic η."""
struct Olivero <: PseudoVoigtModel end

"""Liu et al. (2001) Kielkopf FWHM + cubic-in-d η (d = (fL-fG)/(fL+fG))."""
struct LiuLin <: PseudoVoigtModel end

"""Di Rocco & Cruzado (2012) erfcx-based η; η → 1 for large y. Kielkopf FWHM."""
struct Rocco <: PseudoVoigtModel end


"""
    PseudoVoigt{T, M<:PseudoVoigtModel} <: LineShape{T}

Pseudo-Voigt approximation: a linear blend ``\\eta\\,L + (1-\\eta)\\,G`` of a Lorentzian
and a Gaussian sharing a common combined HWHM.  The mixing parameter ``\\eta`` and the
combined FWHM are determined by the mixing model `M`.

Construct with `PseudoVoigt(σ, γ)` (default: `Thompson()`) or
`PseudoVoigt(σ, γ, model)` to select a specific mixing model.

Available models: [`Thompson`](@ref), [`Kielkopf`](@ref), [`Olivero`](@ref),
[`LiuLin`](@ref), [`Rocco`](@ref).

# Fields
- `sigma::T`: Gaussian width ``\\sigma``
- `gamma::T`: Lorentzian HWHM ``\\gamma``
"""
struct PseudoVoigt{T,M<:PseudoVoigtModel} <: LineShape{T}
    sigma::T
    gamma::T
    eta::T
    lorentz::Lorentzian{T}
    gauss::Gaussian{T}
end
Base.show(io::IO, pv::PseudoVoigt{<:Any,M}) where {M} = print(io, "PseudoVoigt{", M, "}(σ=", pv.sigma, ", γ=", pv.gamma, ")")

PseudoVoigt(σ, γ) = PseudoVoigt(σ, γ, Thompson())
function PseudoVoigt(σ, γ, ::M) where {M<:PseudoVoigtModel}
    T = float(promote_type(typeof(σ), typeof(γ)))
    return pseudo_voigt(M, convert(T, σ), convert(T, γ))
end

function pseudo_voigt(::Type{M}, σ::T, γ::T) where {M<:PseudoVoigtModel,T}
    α = fwhm_over_sigma(σ) * σ / 2                   # Gaussian HWHM
    hV, η = pseudo_voigt_params(M(), α, γ)
    σ_eff = 2 * hV / fwhm_over_sigma(hV)             # Gaussian σ matching the combined width
    U = promote_type(T, typeof(hV), typeof(σ_eff), typeof(η))
    return PseudoVoigt{U,M}(σ, γ, η, Lorentzian{U}(hV), Gaussian{U}(σ_eff))
end

# Copy constructor; keeps the mixing model.
PseudoVoigt{T}(v::PseudoVoigt{<:Any,M}) where {T,M} =
    PseudoVoigt{T,M}(v.sigma, v.gamma, v.eta, Lorentzian{T}(v.lorentz), Gaussian{T}(v.gauss))

@inline (L::PseudoVoigt)(x) = L.eta * L.lorentz(x) + (1 - L.eta) * L.gauss(x)

@inline fwhm(v::PseudoVoigt) = fwhm(v.lorentz)   # both components share the combined width

integrate(pv::PseudoVoigt, a, b) =
    pv.eta * integrate(pv.lorentz, a, b) + (1 - pv.eta) * integrate(pv.gauss, a, b)



# √(ln 2) and 1 − √(π ln 2) at the numeric type of `x`, for the Rocco model.
@inline sqrt_ln2(x) = sqrt(log(oftype(float(x), 2)))
@inline one_minus_sqrt_pi_ln2(x) = 1 - sqrt(_pi(x) * log(oftype(float(x), 2)))

@inline coeffs(x, c::Tuple) = map(v -> oftype(float(x), v), c)

@inline thompson_eta(r) = r * evalpoly(r, coeffs(r, (1.36603, -0.47719, 0.11116)))

@inline function kielkopf_width(α, γ)
    c = coeffs(α, (0.5346, 0.2166))
    return muladd(c[1], γ, sqrt(muladd(c[2], γ^2, α^2)))
end

# pseudo_voigt_params(model, α, γ) → (hV, η)
# α = Gaussian HWHM = √(2 ln 2)·σ;  γ = Lorentzian HWHM;  hV = combined HWHM
@inline function pseudo_voigt_params(::Thompson, α, γ)
    c = coeffs(α, (2.69269, 2.42843, 4.47163, 0.07842))
    p = α^5 + c[1] * α^4 * γ + c[2] * α^3 * γ^2 + c[3] * α^2 * γ^3 + c[4] * α * γ^4 + γ^5
    hV = max(zero(p), p)^oftype(float(α), 0.2)
    return hV, thompson_eta(γ / hV)
end

@inline function pseudo_voigt_params(::Kielkopf, α, γ)
    hV = kielkopf_width(α, γ)
    return hV, thompson_eta(γ / hV)
end

@inline function pseudo_voigt_params(::Olivero, α, γ)
    d = (γ - α) / (γ + α)
    c = coeffs(α, (0.18121, 0.023665, 0.6, 0.00418, -1.9))
    hV = (1 - c[1] * (1 - d^2) - (c[2] * exp(c[3] * d) + c[4] * exp(c[5] * d)) * sin(_pi(α) * d)) * (α + γ)
    return hV, thompson_eta(γ / hV)
end

@inline function pseudo_voigt_params(::LiuLin, α, γ)
    d = (γ - α) / (γ + α)
    hV = kielkopf_width(α, γ)
    return hV, evalpoly(d, coeffs(α, (0.68188, 0.61293, -0.18384, -0.11568)))
end

@inline function pseudo_voigt_params(::Rocco, α, γ)
    rt_ln2 = sqrt_ln2(α)
    y = γ * rt_ln2 / α      # = Faddeeva y parameter (γ / (σ√2))
    ys = min(y, oftype(y, 10))
    ex = ys * evalpoly(ys, coeffs(y, (-0.6055, 0.0718, -0.0049, 0.000136)))
    Vy = (ys + rt_ln2 * exp(ex)) * erfcx(ys)
    η = y > 10 ? one(Vy) : (Vy - rt_ln2) / (Vy * one_minus_sqrt_pi_ln2(α))
    hV = kielkopf_width(α, γ)  # Kielkopf width
    return hV, η
end


