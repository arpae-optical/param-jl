using FastAI, Plots, Flux, StaticArrays, LearnBase, Mongoc, CSV, DataFrames, Distributions
using Mongoc: BSONObjectId, BSON

"""
# 99.999 percentile
- scatter    8.640875e-01
- absorp     3.257729e-11
"""

abstract type Input end
abstract type GeometryClass <: Input end
"Geometry types that have analytically solveable emissivity."
abstract type ExactGeometry <: GeometryClass end
abstract type Output end

const Wavelen = Float64
const Scatter = Float64
const Absorption = Float64
const Emiss = Tuple{Scatter,Absorption}
"Interpolated is a separate data type for safety reasons."
# TODO make generic over NUM_WAVELENS
struct InterpolatedEmissPlot <: Output
    emiss::SVector{NUM_WAVELENS,Pair{Wavelen,Emiss}}
end
Base.length(p::InterpolatedEmissPlot) = length(p.emiss)
Base.size(p::InterpolatedEmissPlot) = size(p.emiss)
Base.iterate(p::InterpolatedEmissPlot) = iterate(p.emiss)
Base.iterate(p::InterpolatedEmissPlot, state) = iterate(p.emiss, state)

abstract type ForwardTask <: LearningTask end # {Input,Output}
abstract type BackwardTask <: LearningTask end # {Output,Input}

struct ForwardMethod <: LearningMethod{ForwardTask} end
struct BackwardMethod <: LearningMethod{BackwardTask}
    simulators::Dict{Type{<:GeometryClass},Any}
end

# TODO Find unit package; use microns
# TODO Might want to take log(wavelen).
"""Absolute percentage error"""
function ape(pred::Number, target::Number)
    abs((pred - target) / target)
end
function mape(pred::AbstractArray, target::AbstractArray)
    mean(abs.((pred - target) ./ target))
end

MAPE = Metric(mape, device = Flux.gpu)

# TRAINING
#           encode            lossfn(model(X), Y)
# ::(I, O) -------> ::(X, Y) --------------------> loss

# INFERENCE
#     encode       model       decode
#::I -------> ::X ------> ::Ŷ -------> ::T

struct Spheroid <: GeometryClass
    """ry := rz"""
    rx::Float64
    rz::Float64
end

struct HexCavity <: ExactGeometry
    diam::Float64
    height::Float64
    len::Float64
end

struct HexSphere <: ExactGeometry
    diam::Float64
    len::Float64
end

struct TriGroove <: ExactGeometry
    depth::Float64
    height::Float64
    len::Float64
end

# TODO use units library
struct LaserParams <: Input
    power # Watts
    polarization::Base.String
    rep_rate:: # (khz)


    freq::Float64 # Repetition rates (kHz)	100
    wavelen::Float64 # TODO make into microns
    amplitude::Float64
end
