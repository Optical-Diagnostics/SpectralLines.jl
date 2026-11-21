# SpectralLines

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://optical-diagnostics.github.io/SpectralLines.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://optical-diagnostics.github.io/SpectralLines.jl/dev/)
[![Build Status](https://github.com/Optical-Diagnostics/SpectralLines.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/jqfeld/SpectralLines.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/jqfeld/SpectralLines.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/jqfeld/SpectralLines.jl)
[![Aqua](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

Fast synthesis of spectra from lists of spectral lines.

The line shapes are `Gaussian`, `Lorentzian`, `Voigt`, `PseudoVoigt` and
`Triangular`. All are callable and normalized to unit area, with `integrate` and
`fwhm`. `Voigt` uses a Faddeeva approximation after Zaghloul (2017) with a
relative error below 4×10⁻⁵.

A spectrum is synthesized by point sampling (`PointSample`), by bin averaging for
coarse grids (`BinAverage`), or by one of two FFT-based methods for large line lists:
`LDM`, the discrete integral transform behind RADIS, and its windowed variant
`SparseLDM`. An instrument profile is applied per line with `convolve` where a
closed form exists, and by FFT convolution of the spectrum otherwise.

All four methods are forward-mode differentiable, second derivatives included.
`LDM` and `SparseLDM` need ForwardDiff loaded. Widths may be
`MonteCarloMeasurements.Particles`. `PointSample` and `BinAverage` are multithreaded, and
`LDM` and `SparseLDM` take reusable workspaces for fitting loops.

## Example

```julia
using SpectralLines

lines = [Line(1000.0, 1.0, Voigt(0.05, 0.02)),
         Line(1000.4, 0.5, Voigt(0.03, 0.01))]
xs = range(999.0, 1001.0, length = 2001)

spectrum = calculate_lines(xs, lines, PointSample())   # direct evaluation
spectrum = calculate_lines(xs, lines, LDM())      # fast for 10⁵+ lines

```

See the [documentation](https://jqfeld.github.io/SpectralLines.jl/dev/) for the full API.
