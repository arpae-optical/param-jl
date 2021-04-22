"""Interpolates emiss curve to 150"""
using BSON
import Mongoc
import Mongoc: BSONObjectId
function interpolate_emiss(emiss_in) end

"""
# 99.999 percentile
- scatter    8.640875e-01
- absorp     3.257729e-11
"""

const FILTER = Dict(
    # Skip multiple geometries for now by only taking meshes with 1 geometry (len == 1) that have gold as their id.
    "material_geometry_mesh" => Dict(raw"$size" => 1),
    "material_geometry_mesh.material" => GOLD,
    # TODO make this work
    # "material_geometry_mesh_detailed.name": "spheroid",
    # XXX full spectra only (for now)
    "results" => Dict(raw"$size" => 150),
    "surrounding_material" => VACUUM,
)

const PROJECTION = Dict(
    "_id" => False,
    "material_geometry_mesh" => True,
    "results.wavelength_micron" => True,
    "results.orientation_av_emissivity" => True,
    "results.orientation_av_absorption_CrossSection_m2" => True,
    "results.orientation_av_scattering_CrossSection_m2" => True,
)

const MIN_CLIP = 1e-19

const MAX_SCATTER_CUTOFF = 1e-2
const MIN_SCATTER_CUTOFF = 1e-14
const MAX_ABSORPTION_CUTOFF = 1e-11

const GOLD = BSONObjectId("5f5a83183c9d9fd8800ce8a3")
const VACUUM = BSONObjectId("5f5a831c3c9d9fd8800ce92c")

struct SpheroidData
    rs::#Array(Tuple{Float64, Float64}) ?
    emiss::#Array(Float64)  ?
end

abstract type Input end

abstract type GeometryClass <: Input end
"""Geometry types that have analytically solveable emissivity"""
abstract type ExactGeometry <: GeometryClass

end


struct HexSphere <: ExactGeometry
    l_d_ratio::Float64 #real number, length/diameter
end

struct TriGroove <: ExactGeometry
    H::Float64
    L::Float64
end

struct HexCavity <: ExactGeometry
    h_d_ratio::Float64
    l_d_ratio::Float64
end

struct Spheroid <: GeometryClass
    rx::Float64
    rz::Float64
end

# TODO fill out
struct LaserParams <: Input end
struct Wavelength
    #might want to take log. TODO: Find unit package; use microns
end
struct EmissivityValue
    emiss::Float64 # TODO should be in (0,1)
end
const Emissivity = Pair{Wavelength,EmissivityValue}
const EmissivityCurve = AbstractVector{Emissivity}
#TODO add a length param in type signature (150)


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

function emissivity_of(emiss_in, g::TriGroove)
    # hexagonal stack, cylindrical cavity
    D = 1
    A_1 = sqrt(4 * g.H^2 + D^2)
    A_2 = D
    A23 = g.L
    A_3 = A23 - A_2
    F11 = 1 - A_2 / A_1
    (A_1 / A23) * (1 - F11) / (1 / emissin - F11 * (1 / emissin - 1)) + emissin * A_3 / A23
end






function emissivity_of(emiss_in, g::HexCavity)
    D = 1
    R = D / 2
    A_2 = pi * R^2
    A_1 = A_2 + 2 * pi * R * g.h_d_ratio
    F11 = 1 - A_2 / A_1
    A23 = 3 * g.l_d_ratio^2 / 2 / sqrt(3)
    A_3 = A23 - A_2
    (A_1 / A23) * (1 - F11) / (1 / emissin - F11 * (1 / emissin - 1)) + epp * A_3 / A23
end

function get_spheroid_data()
    client = Mongoc.Client("mongodb://propopt_admin:ww11122wfg64b1aaa@mongodb07.nersc.gov/propopt")
    db = client["propopt"]
    simulations = db["simulations"]
    
    filter = Mongoc.BSON(FILTER)

    projection = Monogc.BSON(PROJECTION)

    all_emisses = Dict()
    all_rs = Spheroid[]
    for i, sim in enumerate(Mongoc.find(simulations, filter=FILTER, options = projection))
        material_handle = sim["material_geometry_mesh"][1] #Dict type?
    
        geometry = material_handle["geometry"]
    

        spheroid_filter = Dict(
            "_id" => geometry,
            "name" => "spheroid",
            "dims.rx" => Dict(raw"$exists" => True),
            "dims.rz" => (raw"$exists" => True),
        )

        spheroid_projection = Dict(
            "_id" => False, 
            "dims.rx" => True, 
            "dims.rz"=> True
        )

        mongo_sph_filter = Mongoc.BSON(spheroid_filter)
        
        mongo_sph_proj = Mongoc.BSON(spheroid_projection)

        # XXX only handle spherical geoms for now
        # TODO convert old type assertion geom: Dict[str, Dict[str, float]]
        geom = Mongoc.find_one(geometry,
            spheroid_filter, options = mongo_sph_proj
        )

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
    
        push!(all_rs, Spheroid(rx=geom["dims"]["rx"], rz=geom["dims"]["rz"]))

        all_emisses[wavelen] = (scatter, absorption) #TODO sort by wavelen (use not a dict)
    end
    SpheroidData(all_rs, all_emisses)
end


function emissivity_of(emiss_in, g::Spheroid)

end

function emissivity_of(emiss_in, g::LaserParams)

end
