unction getobs_all(::Type{Spheroid})
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
