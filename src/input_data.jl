using FastAI
using Flux
using StaticArrays
using LearnBase

"""# Data"""

using Mongoc
using Mongoc: BSONObjectId, BSON
function interpolate_emiss(emiss_in) end

"""
# 99.999 percentile
- scatter    8.640875e-01
- absorp     3.257729e-11
"""
const GOLD = BSONObjectId("5f5a83183c9d9fd8800ce8a3")
const VACUUM = BSONObjectId("5f5a831c3c9d9fd8800ce92c")
const BATCH_SIZE = 32

const NUM_WAVELENS = 150
const NUM_SIMULATORS = 1


const FILTER = Dict(
    # Skip multiple geometries for now by only taking meshes with 1 geometry (len == 1) that have gold as their id.
    "material_geometry_mesh" => Dict(raw"$size" => 1),
    "material_geometry_mesh.material" => GOLD,
    # TODO make this work
    # "material_geometry_mesh_detailed.name": "spheroid",
    # XXX full spectra only (for now). This is why we can directly construct an InterpolatedEmissPlot in `getobs_all`
    "results" => Dict(raw"$size" => NUM_WAVELENS),
    "surrounding_material" => VACUUM,
)

const PROJECTION = Dict(
    "_id" => false,
    "material_geometry_mesh" => true,
    "results.wavelength_micron" => true,
    "results.orientation_av_emissivity" => true,
    "results.orientation_av_absorption_CrossSection_m2" => true,
    "results.orientation_av_scattering_CrossSection_m2" => true,
)

const MIN_CLIP = 1e-19

const MAX_SCATTER_CUTOFF = 1e-2
const MIN_SCATTER_CUTOFF = 1e-14

const MAX_ABSORPTION_CUTOFF = 1e-11

abstract type Input end
abstract type Output end

const ForwardTask = LearningTask{Input,Output}
const BackwardTask = LearningTask{Output,Input}

# TRAINING
#           encode            lossfn(model(X), Y)
# ::(I, O) -------> ::(X, Y) --------------------> loss

# INFERENCE
#     encode       model       decode
#::I -------> ::X ------> ::Ŷ -------> ::T

struct ForwardMethod <: LearningMethod{ForwardTask} end
struct BackwardMethod <: LearningMethod{BackwardTask}
    simulators::Any
end

abstract type GeometryClass <: Input end
"Geometry types that have analytically solveable emissivity"
abstract type ExactGeometry <: GeometryClass end

mape(pred, target) = abs((pred - target) / target)
Metric(mape, device = Flux.gpu)

const Scatter = Float64
const Absorption = Float64
# TODO might want to take log(wavelen). TODO: Find unit package; use microns
"""wavelen=>emiss"""
const Wavelen = Float64
const Emiss = (Scatter, Absorption)
"Interpolated is a separate data type for safety reasons."
const InterpolatedEmissPlot <: Output = SVector{NUM_WAVELENS,Pair{Wavelen,Emiss}}

function rand(::HexSphere)
    HexSphere(len = 10^rand(Uniform(0, 0.7), N), diam = rand(Uniform(1, 10), N))
end
function rand(::HexCavity)
    l_d = 10^Uniform(0.065, 0.4).sample((N,))
    height = Uniform(0, 10).sample((N,))
    simulator = hex_cavity_emiss_out
end
function rand(::TriGroove)
    l_d = 10^Uniform(0, 0.4).sample((N,))
    height = Uniform(0, 10).sample((N,))
    simulator = tri_groove_emiss_out
end

struct Spheroid <: GeometryClass
    """ry := rz"""
    rx::Float64
    rz::Float64
end

struct HexCavity <: ExactGeometry
    """height over diam (ratio)"""
    height::Float64
    """len over diam (ratio)"""
    len::Float64
    diam::Float64
end

struct HexSphere <: ExactGeometry
    """len over diam (ratio)"""
    len::Float64
    diam::Float64
end

struct TriGroove <: ExactGeometry
    """Height"""
    height::Float64
    "depth"
    len::Float64
    diam::Float64
end

struct LaserParams <: Input end # TODO fill out

function (g::HexCavity)(emiss_in::InterpolatedEmissPlot)
    D = g.diam
    R = D / 2
    A₂ = π * R^2
    A₁ = A₂ + 2π * R * g.height / g.diam
    F₁₁ = 1 - A₂ / A₁
    A₂₃ = 3(g.len / D)^2 / (2 * √(3))
    A₃ = A₂₃ - A₂
    (A₁ / A₂₃) * (1 - F₁₁) / (1 / emiss_in - F₁₁ * (1 / emiss_in - 1)) + 1 / emiss_in * A₃ / A₂₃
end

function (g::HexSphere)(emiss_in::InterpolatedEmissPlot)
    """TODO make it pretty (pi, *, etc)"""

    R = g.diam / 2
    L = g.len
    s = sqrt(L^2 / 3) #length of one side of hexagon |

    Aₛ = 4π * R^2 #surface area of sphere
    Aₕ = 3 / 2 * s * L #surface area of hexagon (side times base over two times six)
    A₁ = Aₛ + Aₕ #surface area of solid

    A₁ = A₁ #emitting surface
    A₂ = Aₕ #apparent area
    # K=1-F₁₁-A₂/A₁;
    1 / (1 + A₂ / A₁ * (1 / emiss_in - 1))
end

function (g::TriGroove)(emiss_in::InterpolatedEmissPlot)
    # hexagonal stack, cylindrical cavity
    D = g.diam
    A₁ = sqrt(4 * g.height^2 + D^2)
    A₂ = D
    A₂₃ = g.len
    A₃ = A₂₃ - A₂
    F₁₁ = 1 - A₂ / A₁
    (A₁ / A₂₃) * (1 - F₁₁) / (1 / emiss_in - F₁₁ * (1 / emiss_in - 1)) + emiss_in * A₃ / A₂₃
end

function getobs_all(::Spheroid)
    client = Mongoc.Client("mongodb://propopt_admin:ww11122wfg64b1aaa@mongodb07.nersc.gov/propopt")
    db = client["propopt"]
    simulations = db["simulations"]


    out = SpheroidData[]
    for (i, sim) in enumerate(Mongoc.find(simulations, BSON(FILTER), options = BSON(PROJECTION)))
        geometry = sim["material_geometry_mesh"][1]["geometry"] #Dict type?


        spheroid_filter = Dict(
            "_id" => geometry,
            "name" => "spheroid",
            "dims.rx" => Dict(raw"$exists" => true),
            "dims.rz" => (raw"$exists" => true),
        )

        spheroid_projection = Dict("_id" => false, "dims.rx" => true, "dims.rz" => true)


        # XXX only handle spherical geoms for now
        # TODO convert old type assertion geom: Dict[str, Dict[str, float]]
        geom = Mongoc.find_one(geometry, BSON(spheroid_filter), options = BSON(spheroid_projection))

        results = sim["results"]

        wavelen = [r["wavelength_micron"] for r in results]
        absorption = [r["orientation_av_absorption_CrossSection_m2"] for r in results]
        scatter = [r["orientation_av_scattering_CrossSection_m2"] for r in results]

        if any(isnan(x) for x in Iterators.flatten([wavelen, scatter, absorption])) ||
           any(absorption .> MAX_ABSORPTION_CUTOFF) ||
           any(scatter .> MAX_SCATTER_CUTOFF)
            continue
        end


        clamp!(scatter, MIN_CLIP, Inf)
        clamp!(absorption, MIN_CLIP, Inf)

        push!(
            out,
            (
                Spheroid(rx = geom["dims"]["rx"], rz = geom["dims"]["rz"]),
                InterpolatedEmissPlot(wavelen => collect(zip(scatter, absorption))),
            ),
        )
    end

    out

end

DLPipelines.encodeinput(method::ForwardMethod, ctx, input::Spheroid) = (input.rx, input.rz)

function DLPipelines.encodetarget(task::ForwardTask, target::InterpolatedEmissPlot)
    [t.second for t in target]
end

function DLPipelines.encodeinput(method::BackwardMethod, ctx, input::InterpolatedEmissPlot)
    [i.second for i in input]
end

DLPipelines.encodetarget(task::BackwardTask, target::Spheroid) = (target.rx, target.rz)

DLPipelines.decodeŷ(method::ForwardMethod, ctx, pred) = pred


function DLPipelines.methodmodel(method::ForwardMethod, backbone)
    Chain(
        Dense(2, 32),
        gelu,
        Dense(32, 64),
        gelu,
        Dense(64, 128),
        gelu,
        Dense(128, 64),
        gelu,
        Dense(64, 32),
        gelu,
        Dense(32, NUM_WAVELENS),
        sigmoid,
    )
end

DLPipelines.methodlossfn(method::ForwardMethod) = Flux.Losses.mse

function DLPipelines.methodmodel(method::BackwardMethod, backbone)

    encoder = Chain(
        Conv((3,), 1 => 8, gelu),
        # BatchNorm1d(8),
        Conv((3,), 8 => 16, gelu),
        # BatchNorm1d(16),
        Conv((3,), 16 => 64, gelu),
        # BatchNorm1d(64),
        Conv((3,), 64 => 256),
        flatten,
    )

    Z = 1024
    mean_head = Dense(71 * 512, Z)
    std_head = Chain(Dense(71 * 512, Z),)

    decoder = Chain(
        Dense(Z, 512, gelu),
        # BatchNorm1d(512),
        Dense(512, 256, gelu),
        # BatchNorm1d(256),
        Dense(256, 128),
    )

    geom_heads = [
        Chain(
            Dense(128, 96, gelu),
            # BatchNorm1d(96),
            Dense(96, 64, gelu),
            # BatchNorm1d(64),
            Dense(64, 32, gelu),
            # BatchNorm1d(32),
            Dense(32, 2),
        ) for _ in range(NUM_SIMULATORS)
    ]

    classifier = Chain(
        # TODO(cc) make 2*.. the sum of the outputs of geom heads
        Dense(150, 64, gelu),
        # BatchNorm1d(64),
        Dense(64, 32, gelu),
        # BatchNorm1d(32),
        Dense(32, NUM_SIMULATORS),
    )

    return (structured_emiss,) => begin
        h = reshape(structured_emiss, :, 1, NUM_WAVELENS)

        h = encoder(h)
        mean, std = mean_head(h), std_head(h)
        std = (log_var / 2).exp()

        dist = Normal(mean, std)
        zs = rand(dist)

        decoded = decoder(zs)

        geoms = [g(decoded) for g in geom_heads]

        # TODO use typeddict instead
        ModelOutput(geoms = geoms, mean = mean, std = std)
    end

end

struct ForwardPred
    geoms::Any
    mean::Any
    std::Any
end

function DLPipelines.methodlossfn(method::BackwardMethod)
    (pred, target) -> begin
        geom, mean, std = pred.geoms, pred.mean, pred.std

        # TODO generalize to simulators
        # TODO make simulators an enum
        pred_emiss = method.simulators[1](geom[1])
        mape_loss = mape(pred_emiss, target)

        var = std^2

        kl_loss = mean(-sum((1 + log(var - mean^2 - std), -1) / 2))

        aspect_ratio_loss = mean(max(g.rx, g.rz) / min(g.rz, g.rz) for g in geom)
        total_loss = mape_loss + kl_loss + aspect_ratio_loss

        total_loss
    end
end


dataset = getobs_all(Spheroid)

fwd_learner = methodlearner(ForwardMethod(), dataset, nothing, ToGPU(), Metrics(accuracy))

fitonecycle!(fwd_learner, 5)

backward_learner = methodlearner(
    BackwardMethod(simulators = [fwd_learner.model]),
    dataset,
    nothing,
    ToGPU(),
    Metrics(accuracy),
)

fitonecycle!(backward_learner, 5)
