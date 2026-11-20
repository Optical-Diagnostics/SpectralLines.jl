# Makes `LDM` and `SparseLDM` forward-mode differentiable. FFTW cannot transform
# `ForwardDiff.Dual` arrays, but the FFT is linear: a Dual transform is one
# hardware-float FFT for the value and one for each partial.
#
# LDM has a fast path for one Dual level over hardware floats (gradients and
# Jacobians) and a recursive path for nested duals (Hessians). SparseLDM uses
# one `DualFFTCache` per Dual level, kept in its workspace.
module SpectralLinesForwardDiffExt

using SpectralLines: SpectralLines, FFTW
using ForwardDiff: Dual, Partials, value, partials, npartials, tagtype
using LinearAlgebra: mul!

SpectralLines.fft_eltype_check(::Type{<:Dual{T,V}}, method_name) where {T,V} =
    SpectralLines.fft_eltype_check(V, method_name)

# SparseLDM's automatic truncation is not differentiable in the widths (see the
# core method). Widths that are duals only by promotion, with every partial
# zero at every nesting level, pass: differentiating in positions or amplitudes
# alone is fine.
_has_derivative(x::Real) = false
_has_derivative(x::Dual) = !iszero(partials(x)) || _has_derivative(value(x))

function SpectralLines.check_width_derivatives(fwhm_G::AbstractVector{<:Dual},
                                                fwhm_L::AbstractVector{<:Dual})
    any(_has_derivative, fwhm_G) || any(_has_derivative, fwhm_L) || return nothing
    throw(ArgumentError(
        "SparseLDM with the default truncation cannot be differentiated with respect " *
        "to a line width: the truncation changes in whole grid points with the width. " *
        "Set a fixed width, `SparseLDM(truncation = ...)`, or use `LDM()`."))
end

SpectralLines.to_float64(x::Dual{T}) where {T} =
    Dual{T}(SpectralLines.to_float64(value(x)),
            Partials(map(SpectralLines.to_float64, partials(x).values)))

# LDM fast path: one Dual level over hardware floats

function SpectralLines.ldm_fourier_accumulate(ldm::Array{Dual{Tg,V,N},3}, occupied,
                                               width_axis_G, width_axis_L, freqs) where {Tg,V<:Union{Float32,Float64},N}
    D = Dual{Tg,V,N}
    n_pad = size(ldm, 1)
    n_freq = length(freqs)

    comp = Vector{V}(undef, n_pad)         # one real component of an `ldm` column
    plan = FFTW.plan_rfft(comp; flags = FFTW.ESTIMATE)
    Fv = Vector{Complex{V}}(undef, n_freq)
    Fp = ntuple(_ -> Vector{Complex{V}}(undef, n_freq), Val(N))
    template = Vector{D}(undef, n_freq)
    acc = zeros(Complex{D}, n_freq)

    @inbounds for m in eachindex(width_axis_L), l in eachindex(width_axis_G)
        occupied[l, m] || continue
        σ_lm, γ_lm = SpectralLines.bin_sigma_gamma(width_axis_G[l], width_axis_L[m])
        SpectralLines.fill_template!(template, step(freqs), σ_lm, γ_lm, n_freq)

        col = view(ldm, :, l, m)
        for k in 1:n_pad
            comp[k] = value(col[k])
        end
        mul!(Fv, plan, comp)
        for j in 1:N
            for k in 1:n_pad
                comp[k] = partials(col[k], j)
            end
            mul!(Fp[j], plan, comp)
        end

        for i in 1:n_freq
            re = Dual{Tg}(real(Fv[i]), Partials(ntuple(j -> real(Fp[j][i]), Val(N))))
            im = Dual{Tg}(imag(Fv[i]), Partials(ntuple(j -> imag(Fp[j][i]), Val(N))))
            acc[i] += Complex(re, im) * template[i]
        end
    end
    return acc
end

function SpectralLines.ldm_irfft(acc::AbstractVector{Complex{Dual{Tg,V,N}}},
                                  n_pad::Int) where {Tg,V<:Union{Float32,Float64},N}
    n_freq = length(acc)
    cbuf = Vector{Complex{V}}(undef, n_freq)
    plan = FFTW.plan_irfft(cbuf, n_pad; flags = FFTW.ESTIMATE)
    vout = Vector{V}(undef, n_pad)
    pout = ntuple(_ -> Vector{V}(undef, n_pad), Val(N))
    # cbuf is refilled per component, so the c2r transform destroying its input
    # (FFTW's default for inverse real transforms) is harmless.
    @inbounds for k in 1:n_freq
        cbuf[k] = Complex(value(real(acc[k])), value(imag(acc[k])))
    end
    mul!(vout, plan, cbuf)
    for j in 1:N
        @inbounds for k in 1:n_freq
            cbuf[k] = Complex(partials(real(acc[k]), j), partials(imag(acc[k]), j))
        end
        mul!(pout[j], plan, cbuf)
    end
    return [Dual{Tg}(vout[k], Partials(ntuple(j -> pout[j][k], Val(N)))) for k in 1:n_pad]
end

# LDM generic tier: nested duals via one-layer-per-call recursion

dual_rfft(A::AbstractVector{<:Union{Float32,Float64}}) = FFTW.rfft(A)
dual_irfft(A::AbstractVector{<:Complex{<:Union{Float32,Float64}}}, n::Int) = FFTW.irfft(A, n)

function dual_rfft(A::AbstractVector{D}) where {D<:Dual}
    N = npartials(D)
    Fv = dual_rfft(value.(A))
    Fp = ntuple(i -> dual_rfft(partials.(A, i)), N)
    return [Complex(Dual{tagtype(D)}(real(Fv[k]), Partials(ntuple(i -> real(Fp[i][k]), N))),
                    Dual{tagtype(D)}(imag(Fv[k]), Partials(ntuple(i -> imag(Fp[i][k]), N))))
            for k in eachindex(Fv)]
end

function dual_irfft(A::AbstractVector{Complex{D}}, n::Int) where {D<:Dual}
    N = npartials(D)
    v = dual_irfft([Complex(value(real(a)), value(imag(a))) for a in A], n)
    p = ntuple(i -> dual_irfft([Complex(partials(real(a), i), partials(imag(a), i)) for a in A], n), N)
    return [Dual{tagtype(D)}(v[k], Partials(ntuple(i -> p[i][k], N))) for k in eachindex(v)]
end

function SpectralLines.ldm_fourier_accumulate(ldm::Array{D,3}, occupied, width_axis_G, width_axis_L,
                                               freqs) where {D<:Dual}
    acc = zeros(Complex{D}, length(freqs))
    for m in eachindex(width_axis_L), l in eachindex(width_axis_G)
        occupied[l, m] || continue
        σ_lm, γ_lm = SpectralLines.bin_sigma_gamma(width_axis_G[l], width_axis_L[m])
        F = dual_rfft(view(ldm, :, l, m))
        @inbounds for i in eachindex(acc)
            acc[i] += F[i] * SpectralLines.voigt_ft(freqs[i], σ_lm, γ_lm)
        end
    end
    return acc
end

SpectralLines.ldm_irfft(acc::AbstractVector{Complex{D}}, n_pad::Int) where {D<:Dual} =
    dual_irfft(acc, n_pad)


# SparseLDM: FFT seams over Duals
#
# Each transform runs once per component, primal then partials, through the
# inner cache of the value type V: the plan cache of a hardware float, or
# another `DualFFTCache` for nested duals.

struct DualFFTCache{V,N,C}
    inner::C
    rin::Vector{V}                       # one component of a real input
    cin::Vector{Complex{V}}              # one component of a complex input
    rout::Vector{Vector{V}}              # every component of a real output
    cout::Vector{Vector{Complex{V}}}     # every component of a complex output
end

SpectralLines.fft_cache_type(::Type{Dual{T,V,N}}) where {T,V,N} =
    DualFFTCache{V,N,SpectralLines.fft_cache_type(V)}
SpectralLines.fft_cache(::Type{Dual{T,V,N}}) where {T,V,N} =
    SpectralLines.fft_cache_type(Dual{T,V,N})(SpectralLines.fft_cache(V), V[], Complex{V}[],
                                              [V[] for _ in 0:N], [Complex{V}[] for _ in 0:N])

_prefix!(v::Vector, n::Int) = view(length(v) < n ? resize!(v, n) : v, 1:n)

# Component j of a Dual: its value for j = 0, else its j-th partial.
@inline _component(d::Dual, j::Int) = j == 0 ? value(d) : partials(d, j)

function SpectralLines.r2c!(out::AbstractVector{Complex{Dual{T,V,N}}}, c::DualFFTCache{V,N},
                             x::AbstractVector{Dual{T,V,N}}, n::Int) where {T,V,N}
    nf = n ÷ 2 + 1
    rin = _prefix!(c.rin, n)
    for j in 0:N
        @inbounds for k in 1:n
            rin[k] = _component(x[k], j)
        end
        SpectralLines.r2c!(_prefix!(c.cout[j + 1], nf), c.inner, rin, n)
    end
    cout = c.cout
    @inbounds for k in 1:nf
        re = Dual{T}(real(cout[1][k]), Partials(ntuple(j -> real(cout[j + 1][k]), Val(N))))
        im = Dual{T}(imag(cout[1][k]), Partials(ntuple(j -> imag(cout[j + 1][k]), Val(N))))
        out[k] = Complex(re, im)
    end
    return out
end

function SpectralLines.c2r!(out::AbstractVector{Dual{T,V,N}}, c::DualFFTCache{V,N},
                             X::AbstractVector{Complex{Dual{T,V,N}}}, n::Int) where {T,V,N}
    nf = n ÷ 2 + 1
    cin = _prefix!(c.cin, nf)
    for j in 0:N
        @inbounds for k in 1:nf
            cin[k] = Complex(_component(real(X[k]), j), _component(imag(X[k]), j))
        end
        SpectralLines.c2r!(_prefix!(c.rout[j + 1], n), c.inner, cin, n)
    end
    rout = c.rout
    @inbounds for k in 1:n
        out[k] = Dual{T}(rout[1][k], Partials(ntuple(j -> rout[j + 1][k], Val(N))))
    end
    return out
end


# Faddeeva function
#
# In Weideman's region VI the derivative comes from w itself through
# w'(z) = −2z w(z) + 2i/√π, which is more accurate there than differentiating
# the approximation. Elsewhere Duals pass straight through. Nested duals recurse.

function SpectralLines._w_inner(x::D, y::D, humlicek::Bool) where {T,D<:Dual{T}}
    humlicek && return SpectralLines._humlicek(x, y)
    v = complex(value(x), value(y))
    w = SpectralLines._w_inner(value(x), value(y), humlicek)
    wp = -2 * v * w + 2im / sqrt(oftype(real(v), π))
    wr, wi = real(wp), imag(wp)
    px, py = partials(x), partials(y)
    # dw = w' (dx + i dy)
    return complex(Dual{T}(real(w), wr * px - wi * py),
                   Dual{T}(imag(w), wi * px + wr * py))
end

SpectralLines._w_inner_re(x::D, y::D, humlicek::Bool) where {D<:Dual} =
    humlicek ? SpectralLines._humlicek_re(x, y) : real(SpectralLines._w_inner(x, y, false))

end
