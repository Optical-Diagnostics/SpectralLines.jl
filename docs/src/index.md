```@meta
CurrentModule = SpectralLines
```

# SpectralLines

Fast synthesis of spectra from lists of spectral lines.

A spectrum is a sum of [`Line`](@ref)s, each a unit-area [`LineShape`](@ref)
centred at a position and scaled by an amplitude. [`calculate_lines`](@ref)
evaluates that sum on a grid with one of four methods:

| Method | Use when |
|:-------|:---------|
| [`PointSample`](@ref) | the grid resolves every line |
| [`BinAverage`](@ref) | the grid spacing is comparable to or coarser than the linewidths |
| [`LDM`](@ref) | the line list is large (10⁵ lines and more) and the shapes are `Voigt`, `Gaussian` or `Lorentzian` |
| [`SparseLDM`](@ref) | as `LDM`, when each width bin's lines cover only part of the grid |

An instrument profile is applied per line with [`convolve`](@ref) where a closed
form exists, or to the synthesized spectrum with
`calculate_lines(xs, lines, instrument_function)`.

## Example

```julia
using SpectralLines

lines = [Line(1000.0, 1.0, Voigt(0.05, 0.02)),
         Line(1000.4, 0.5, Voigt(0.03, 0.01))]
xs = range(999.0, 1001.0, length = 2001)

spectrum = calculate_lines(xs, lines, PointSample())   # direct evaluation
spectrum = calculate_lines(xs, lines, LDM())      # fast for 10⁵+ lines

# instrument broadening, analytic per line
broadened = calculate_lines(xs, convolve.(lines, Gaussian(0.02)), PointSample())

```

## Reference

```@index
```

```@autodocs
Modules = [SpectralLines]
```
