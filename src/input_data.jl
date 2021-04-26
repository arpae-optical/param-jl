using FastAI
using StaticArrays



dataset = loadtaskdata(datasetpath("imagenette2-160"), ImageClassificationTask)
method = ImageClassification(Datasets.getclassesclassification("imagenette2-160"), (160, 160))
data_loaders = methoddataloaders(dataset, method, 16)
model = methodmodel(method, Models.xresnet18())
learner = Learner(model, data_loaders, ADAM(), methodlossfn(method), ToGPU(), Metrics(accuracy))
fitonecycle!(learner, 5)

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
# ::(I, 0) -------> ::(X, Y) --------------------> loss
# INFERENCE
#     encode       model       decode
#::I -------> ::X ------> ::Ŷ -------> ::T

struct DirectMethod <: LearningMethod{ForwardTask} end
struct IndirectMethod <: LearningMethod{BackwardTask} end

abstract type GeometryClass <: Input end
"""Geometry types that have analytically solveable emissivity"""
abstract type ExactGeometry <: GeometryClass

end

# TODO might want to take log(wavelen). TODO: Find unit package; use microns
"""wavelen=>emiss"""
const EmissivityCurve = SVector{150,Pair{Float64,Float64}}
struct Spheroid <: GeometryClass
    """ry := rz"""
    rx::Float64
    rz::Float64
end

struct SpheroidData
    s::Spheroid
    emiss::EmissivityCurve
end

struct HexSphere <: ExactGeometry
    """len over diam (ratio)"""
    l_d_ratio::Float64 #real number, length/diameter
end

struct TriGroove <: ExactGeometry
    """Height"""
    H::Float64
    "depth"
    L::Float64
end

struct HexCavity <: ExactGeometry
    """height over diam (ratio)"""
    h_d::Float64
    """len over diam (ratio)"""
    l_d::Float64
end

struct LaserParams <: Input end # TODO fill out



function emissivity_of(emiss_in::EmissivityCurve, g::HexSphere)
    """TODO make it pretty (pi, *, etc)"""

    D = 1 #diameter of sphere
    R = D / 2 #radius of sphere    /\
    L = g.l_d_ratio * D #width of hexagon |  | distance between |  |
    #                             \/
    s = sqrt(L^2 / 3) #length of one side of hexagon |

    As = 4 * pi * R^2 #surface area of sphere
    Ah = 3 / 2 * s * L #surface area of hexagon (side times base over two times six)
    A1 = As + Ah #surface area of solid

    A_1 = A1 #emitting surface
    A_2 = Ah #apparent area
    # K=1-F11-A_2/A_1;
    1 / (1 + A_2 / A_1 * (1 / emiss_in - 1))
end

function emissivity_of(emiss_in::EmissivityCurve, g::TriGroove)
    # hexagonal stack, cylindrical cavity
    D = 1
    A_1 = sqrt(4 * g.H^2 + D^2)
    A_2 = D
    A23 = g.L
    A_3 = A23 - A_2
    F11 = 1 - A_2 / A_1
    (A_1 / A23) * (1 - F11) / (1 / emiss_in - F11 * (1 / emiss_in - 1)) + emiss_in * A_3 / A23
end

function emissivity_of(emiss_in::EmissivityCurve, g::HexCavity)
    D = 1
    R = D / 2
    A_2 = pi * R^2
    A_1 = A_2 + 2 * pi * R * g.h_d_ratio
    F11 = 1 - A_2 / A_1
    A23 = 3 * g.l_d_ratio^2 / 2 / sqrt(3)
    A_3 = A23 - A_2
    (A_1 / A23) * (1 - F11) / (1 / emiss_in - F11 * (1 / emiss_in - 1)) + epp * A_3 / A23
end
function emissivity_of(s::SpheroidData)
    s.emiss
end

function get_spheroid_data()
    client = Mongoc.Client("mongodb://propopt_admin:ww11122wfg64b1aaa@mongodb07.nersc.gov/propopt")
    db = client["propopt"]
    simulations = db["simulations"]

    filter = BSON(FILTER)

    projection = BSON(PROJECTION)

    all_emisses = Dict()
    all_rs = Spheroid[]
    for (i, sim) in enumerate(Mongoc.find(simulations, filter = FILTER, options = projection))
        material_handle = sim["material_geometry_mesh"][1] #Dict type?

        geometry = material_handle["geometry"]


        spheroid_filter = Dict(
            "_id" => geometry,
            "name" => "spheroid",
            "dims.rx" => Dict(raw"$exists" => true),
            "dims.rz" => (raw"$exists" => true),
        )

        spheroid_projection = Dict("_id" => false, "dims.rx" => true, "dims.rz" => true)

        mongo_sph_filter = BSON(spheroid_filter)

        mongo_sph_proj = BSON(spheroid_projection)

        # XXX only handle spherical geoms for now
        # TODO convert old type assertion geom: Dict[str, Dict[str, float]]
        geom = Mongoc.find_one(geometry, spheroid_filter, options = mongo_sph_proj)

        results = sim["results"]

        wavelen = [r["wavelength_micron"] for r in results]

        absorption = [r["orientation_av_absorption_CrossSection_m2"] for r in results]

        scatter = [r["orientation_av_scattering_CrossSection_m2"] for r in results]

        if any(any.(isnan.(arr)) for arr in [wavelen, scatter, absorption]) #dots may need shuffling?
            continue
        end

        if any(absorption .> MAX_ABSORPTION_CUTOFF) || any(scatter .> MAX_SCATTER_CUTOFF)
            continue
        end

        clamp!(scatter, MIN_CLIP, Inf)

        clamp!(absorption, MIN_CLIP, Inf)

        push!(all_rs, Spheroid(rx = geom["dims"]["rx"], rz = geom["dims"]["rz"]))

        all_emisses[wavelen] = (scatter, absorption) #TODO sort by wavelen (use not a dict)
    end
    SpheroidData(all_rs, all_emisses)
end


function emissivity_of(emiss_in, g::Spheroid)

end

function emissivity_of(emiss_in, g::LaserParams)

end
