module MusicViz

using Statistics
using DSP, WAV, PyPlot

using Bplus;
using Bplus.BplusCore, Bplus.Utilities, Bplus.Math,
      Bplus.BplusApp, Bplus.GL, Bplus.GUI, Bplus.Input, Bplus.ModernGLbp,
      Bplus.BplusTools, Bplus.ECS, Bplus.Fields, Bplus.SceneTree

include("basic_utils.jl")
include("fingerprints.jl")
include("instruments.jl")
include("wav_utils.jl")


end # module MusicViz
