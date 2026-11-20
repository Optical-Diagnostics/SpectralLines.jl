using FFTW: rfft, irfft

"""
    convolve(shape1, shape2) -> LineShape
    convolve(line::Line, shape) -> Line

Analytically combine two line shapes into the closed-form `LineShape` their
convolution equals: `Gaussian⊛Gaussian → Gaussian`, `Lorentzian⊛Lorentzian →
Lorentzian`, `Gaussian⊛Lorentzian → Voigt`, and `Voigt` with any of the three.
These are exact. A `Line` is convolved through its shape, keeping position and
amplitude. `convolve(x, nothing) = x` is a no-op, for when no instrumental
broadening is requested.

`PseudoVoigt` and `Triangular` have no `convolve` method. For a `Triangular`
instrument, use `calculate_lines(xs, lines, instrument_function)`, which
FFT-convolves the synthesized spectrum.
"""
convolve(x, ::Nothing) = x

convolve(g1::Gaussian{T1}, g2::Gaussian{T2}) where {T1,T2} = Gaussian{promote_type(T1,T2)}(sqrt(abs2(g1.sigma) + abs2(g2.sigma)))
convolve(l1::Lorentzian{T1}, l2::Lorentzian{T2}) where {T1,T2} = Lorentzian{promote_type(T1,T2)}(l1.hwhm + l2.hwhm)

convolve(g::Gaussian{T1}, l::Lorentzian{T2}) where {T1,T2} = Voigt(g.sigma, l.hwhm)
convolve(l::Lorentzian{T2}, g::Gaussian{T1}) where {T1,T2} = Voigt(g.sigma, l.hwhm)

convolve(v::Voigt{T1}, g::Gaussian{T2}) where {T1,T2} = Voigt(sqrt(abs2(v.sigma) + abs2(g.sigma)), v.gamma)
convolve(g::Gaussian{T1}, v::Voigt{T2}) where {T1,T2} = convolve(v, g)
convolve(v::Voigt{T1}, l::Lorentzian{T2}) where {T1,T2} = Voigt(v.sigma, v.gamma + l.hwhm)
convolve(l::Lorentzian{T1}, v::Voigt{T2}) where {T1,T2} = convolve(v, l)
convolve(v1::Voigt{T1}, v2::Voigt{T2}) where {T1,T2} = Voigt(sqrt(abs2(v1.sigma) + abs2(v2.sigma)), v1.gamma + v2.gamma)

convolve(l::Line, ls::LineShape) = Line(l.position, l.amplitude, convolve(l.shape, ls))

abstract type ConvolutionType end

"""
    SpectrumConvolution()

Convolution strategy for [`calculate_lines`](@ref) with an instrument function:
synthesize, then FFT-convolve the spectrum. Works for any instrument shape. The
default.
"""
struct SpectrumConvolution <: ConvolutionType end

"""
    LineConvolution()

Convolution strategy for [`calculate_lines`](@ref) with an instrument function:
combine each line's shape with the instrument via [`convolve`](@ref), then
synthesize exactly. Needs a closed form for every pair, e.g. `Gaussian`,
`Lorentzian` or `Voigt` throughout.
"""
struct LineConvolution <: ConvolutionType end

"""
    calculate_lines(xs, lines, instrument_function[, strategy]; wing_cut=nothing, min_amplitude=nothing) -> Vector

Spectrum of `lines` on `xs`, broadened by `instrument_function` (any callable
with an `fwhm` method). Synthesis uses [`PointSample`](@ref) with the given keywords.

`strategy` is [`SpectrumConvolution`](@ref) (default) or
[`LineConvolution`](@ref). `SpectrumConvolution` needs an evenly-spaced,
ascending `xs` that resolves the narrowest line.
"""
calculate_lines(xs, lines, instrument_function; kwargs...) =
    calculate_lines(xs, lines, instrument_function, SpectrumConvolution(); kwargs...)

# Not always faster than the FFT path below.
function calculate_lines(x, lines, instrument_function::IF, ::LineConvolution; kwargs...) where {IF<:LineShape}
    conv_lines = convolve.(lines, instrument_function)
    calculate_lines(x, conv_lines, PointSample(; kwargs...))
end

# Grid spacing of an evenly-spaced `xs`; throws if it is not.
uniform_step(xs::AbstractRange) = step(xs)
function uniform_step(xs)
    length(xs) < 2 && throw(ArgumentError(
        "convolution needs at least two grid points, got $(length(xs))"))
    dx = xs[begin+1] - xs[begin]
    @inbounds for i in (firstindex(xs)+2):lastindex(xs)
        d = xs[i] - xs[i-1]
        isapprox(d, dx, rtol = 1e-2) || throw(ArgumentError(
            "convolution requires an evenly-spaced grid; spacing $d at index $i " *
            "differs from the leading spacing $dx by more than 1%"))
    end
    return dx
end

function calculate_lines(xs, lines, instrument_function, ::SpectrumConvolution; kwargs...)
    isempty(lines) && throw(ArgumentError(
        "convolution needs at least one line to size the kernel against"))
    check_ascending(xs, "the FFT convolution")
    fw = minimum(fwhm(l.shape) for l in lines)
    dx = uniform_step(xs)
    dx < fw || throw(ArgumentError(
        "grid spacing $dx does not resolve the narrowest line (FWHM $fw); the FFT " *
        "convolution needs each profile sampled before it is convolved"))

    lo, hi = convolution_support(instrument_function)
    n_pad = ceil(Int, max(-lo, hi) / dx)
    left = range(xs[begin] - n_pad * dx, step = dx, length = n_pad)
    right = range(xs[end] + dx, step = dx, length = n_pad)
    padded_xs = vcat(left, xs, right)
    n_kernel = n_pad * 2 + 1
    kernel = instrument_function.(range(-n_pad * dx, n_pad * dx, length = n_kernel)) * dx
    kernel ./= sum(kernel)
    padded_spec = calculate_lines(padded_xs, lines, PointSample(; kwargs...))

    return conv_full(padded_spec, kernel)[(1 + 2 * n_pad):(2 * n_pad + length(xs))]
end

# Full linear convolution, of length length(u) + length(v) - 1. Hardware floats
# use zero-padded rfft/irfft; other types (e.g. ForwardDiff.Dual) the direct
# O(n·m) loop below.
function conv_full(u::AbstractVector{T}, v::AbstractVector{T}) where {T<:Union{Float32,Float64}}
    n_full = length(u) + length(v) - 1
    n_pad = nextfastfft(n_full)
    up = zeros(T, n_pad)
    copyto!(up, u)
    vp = zeros(T, n_pad)
    copyto!(vp, v)
    return irfft(rfft(up) .* rfft(vp), n_pad)[1:n_full]
end

function conv_full(u::AbstractVector{<:Union{Float32,Float64}},
                    v::AbstractVector{<:Union{Float32,Float64}})
    T = promote_type(eltype(u), eltype(v))
    return conv_full(T.(u), T.(v))
end

function conv_full(u::AbstractVector, v::AbstractVector)
    T = promote_type(eltype(u), eltype(v))
    out = zeros(T, length(u) + length(v) - 1)
    for j in eachindex(v)
        vj = v[j]
        iszero(vj) && continue
        @inbounds @simd for i in eachindex(u)
            out[i + j - 1] += u[i] * vj
        end
    end
    return out
end
