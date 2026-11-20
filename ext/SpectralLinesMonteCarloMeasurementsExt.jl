# Uncertain line widths (`MonteCarloMeasurements.Particles`) for `Voigt`.
#
# `Gaussian` and `Lorentzian` need nothing: their formulas are plain arithmetic,
# which `Particles` propagate. `Voigt` evaluates `faddeyeva`, whose region
# choice compares its arguments, and comparing particles throws. Registering it
# as a primitive applies it particle by particle.
module SpectralLinesMonteCarloMeasurementsExt

using MonteCarloMeasurements
import SpectralLines

MonteCarloMeasurements.register_primitive(SpectralLines.faddeyeva, eval)

end
