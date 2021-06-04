getobs_all(::Type{T}; num_obs::Integer = 10^4) where {T<:ExactGeometry} = rand(T, num_obs)
using FastAI, Plots, Flux, StaticArrays, LearnBase, Mongoc, CSV, DataFrames, Distributions, Interpolations

include("constants.jl")
include("types.jl")

function getobs_laser()
    client = Mongoc.Client("mongodb://propopt_ro:2vsz634dwrwwsq@mongodb07.nersc.gov/propopt")
    db = client["propopt"]
    laser_samples = db["laser_samples"]

    emiss_list = []
    wavelength_list = []
    laser_power_W_list = []
    x_speed_list = []
    y_spacing_list = []
    frequency_list = []
    
    for entry in laser_samples
        push!(wavelength_list, entry["laser_wavelength_nm"])
        push!(laser_power_W_list, entry["laser_power_W"])
        push!(x_speed_list, entry["laser_scanning_speed_x_dir_mm_per_s"])
        push!(y_spacing_list, entry["laser_scanning_line_spacing_y_dir_micron"])
        push!(frequency_list, entry["laser_repetition_rate_kHz"])
        EmissPlot = []
        emiss = entry["emissivity_spectrum"]
        for ex in emiss
            push!(EmissPlot, ex["normal_emissivity"])
        end
        interpolated_values = [1:(size(EmissPlot)[1]-1)/NUM_WAVELENS:size(EmissPlot)[1]]

        InterpolatedEmissPlot = interpolate(EmissPlot, BSpline(Linear()))
        InterpolatedSample = [InterpolatedEmissPlot(value) for value in interpolated_values]
        push!(emiss_list, InterpolatedSample)
    end

    out = Tuple{LaserParams,Any}[]


    for i in 1:size(emiss_list)[1]

        push!(
            out,
            (
                LaserParams(parse(Float64, wavelength_list[i]), laser_power_W_list[i], x_speed_list[i], y_spacing_list[i], parse(Float64, frequency_list[i])), #make the laserparams
                emiss_list[i]
                )
            )
            
    end

    out

end


function getobs_all(::Type{Spheroid})
    client = Mongoc.Client("mongodb://propopt_admin:ww11122wfg64b1aaa@mongodb07.nersc.gov/propopt")
    db = client["propopt"]
    simulations = db["simulations"]
    geometries = db["geometries"]


    out = Tuple{Spheroid,InterpolatedEmissPlot}[]

    for (i, sim) in enumerate(Mongoc.find(
        simulations,
        BSON(Dict(
            # Skip multiple geometries for now by only taking meshes with 1 geometry (len == 1) that have gold as their id.
            "material_geometry_mesh" => Dict(raw"$size" => 1),
            "material_geometry_mesh.material" => GOLD,
            # TODO make this work
            # "material_geometry_mesh_detailed.name": "spheroid",
            # XXX full spectra only (for now). This is why we can directly construct an InterpolatedEmissPlot in `getobs_all`
            "results" => Dict(raw"$size" => NUM_WAVELENS),
            "surrounding_material" => VACUUM,
        )),
        options = BSON(Dict(
            "projection" => Dict(
                "_id" => false,
                "material_geometry_mesh" => true,
                "results.wavelength_micron" => true,
                "results.orientation_av_emissivity" => true,
                "results.orientation_av_absorption_CrossSection_m2" => true,
                "results.orientation_av_scattering_CrossSection_m2" => true,
            ),
        )),
    ))
        geometry = sim["material_geometry_mesh"][1]["geometry"] #Dict type?

        # XXX only handle spherical geoms for now
        # TODO convert old type assertion geom: Dict[str, Dict[str, float]]
        geom = Mongoc.find_one(
            geometries,
            BSON(Dict(
                "_id" => geometry,
                "name" => "spheroid",
                # "dims.rx" => Dict(raw"$exists" => true),
                # "dims.rz" => (raw"$exists" => true),
            )),
            #= options = BSON(
                "projection" => Dict("_id" => false, "dims.rx" => true, "dims.rz" => true),
            ), =#
        )

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

        # emiss plot needs to be sorted
        push!(
            out,
            (
                Spheroid(geom["dims"]["rx"], geom["dims"]["rz"]),
                InterpolatedEmissPlot(sort(
                    [w => (s, a) for (w, s, a) in (zip(wavelen, scatter, absorption))],
                )),
            ),
        )
    end

    out

end
