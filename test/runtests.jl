using Test, SafeTestsets

@safetestset "Faddeyeva" include("test_faddeyeva.jl")

@safetestset "Line shapes" begin
    @safetestset "All shapes" include("test_lineshapes.jl")
    @safetestset "Voigt" include("test_voigt.jl")
    @safetestset "PseudoVoigt" include("test_pseudo_voigt.jl")
end

@safetestset "Line" include("test_line.jl")

@safetestset "calculate_lines" begin
    @safetestset "Interface" include("test_calculate_lines.jl")
    @safetestset "PointSample" include("test_sample.jl")
    @safetestset "BinAverage" include("test_bin.jl")
    @safetestset "LDM and SparseLDM" include("test_ldm.jl")
    @safetestset "SparseLDM" include("test_sparse_ldm.jl")
    @safetestset "Workspaces" include("test_workspace.jl")
end

@safetestset "Convolution" include("test_convolution.jl")
@safetestset "Autodiff" include("test_autodiff.jl")
@safetestset "MonteCarloMeasurements" include("test_montecarlo.jl")

@safetestset "Aqua" begin
    using Aqua, SpectralLines
    Aqua.test_all(SpectralLines)
end

@safetestset "JET" include("test_jet.jl")
