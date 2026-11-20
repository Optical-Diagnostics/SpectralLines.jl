module SpectralLines

import SpecialFunctions: erf, erfc

import SpecialFunctions: erfcx

# For the quadrature nodes and weights of integrate(::Voigt, a, b).
using LinearAlgebra: SymTridiagonal, eigen

include("faddeyeva.jl")

include("lineshape.jl")
export fwhm, hwhm, integrate, convolution_support
include("lineshapes/gaussian.jl")
export Gaussian
include("lineshapes/lorentzian.jl")
export Lorentzian
include("lineshapes/voigt.jl")
export Voigt
include("lineshapes/pseudo_voigt.jl")
export PseudoVoigt, PseudoVoigtModel, Thompson, Kielkopf, Olivero, LiuLin, Rocco
include("lineshapes/triangular.jl")
export Triangular


include("line.jl")
export Line, update_lines!


include("calculate_lines.jl")
export LineSynthesisMethod, calculate_lines, calculate_lines!
include("calculate_lines/sample.jl")
export PointSample
include("calculate_lines/bin.jl")
export BinAverage
include("calculate_lines/ldm.jl")
export LDM, LDMWorkspace
include("calculate_lines/sparse_ldm.jl")
export SparseLDM, SparseLDMWorkspace

include("convolution.jl")
export convolve
public LineShape, numtype_convert, faddeyeva
public SpectrumConvolution, LineConvolution, default_fftw_threads


end
