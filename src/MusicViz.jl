module MusicViz

using Statistics, Printf
using DSP, FFTW, Flux, PaddedViews
using WAV, ProgressMeter, PyPlot

using Bplus;
using Bplus.BplusCore, Bplus.Utilities, Bplus.Math,
      Bplus.BplusApp, Bplus.GL, Bplus.GUI, Bplus.Input, Bplus.ModernGLbp,
      Bplus.BplusTools, Bplus.ECS, Bplus.Fields, Bplus.SceneTree

include("basic_utils.jl")
include("fingerprints.jl")
include("instruments.jl")
include("nn.jl")
include("wav_utils.jl")


end # module MusicViz
