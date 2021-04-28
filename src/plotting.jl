using FastAI, Plots, Flux, StaticArrays, LearnBase, Mongoc, CSV, DataFrames, Distributions
using Mongoc: BSONObjectId, BSON

#= function plotsample!(f, method::ForwardMethod, sample)
    # TODO make work
    g, emiss = sample
    f[1, 1] = ax1 = imageaxis(f, title = g)
    plotimage!(ax1, emiss)
end
function plotsample!(f, method::BackwardMethod, sample)
    # TODO make work
    emiss, g = sample
    f[1, 1] = ax1 = imageaxis(f, title = g)
    plot!(ax1, emiss)
end =#

#= function plotsample!(f, method, sample)
    #sample::((rx, rz), emiss)
    # TODO make sure this actually works
    wmesh = MeshMaker.mesh(sample[1][1], sample[1][2], 100)
    open("data/mesh_$(mesh_index).obj", "w") do mesh_target
        pts = [Tuple(p.coords) for p in wmesh.points]
        edges = [Tuple(p.list) for p in wmesh.connec]
        point_index = 0
        for point in pts
            write(mesh_target, "v $(point[1]) $(point[2]) $(point[3]) \n")
        end
        for edge in edges
            write(mesh_target, "f $(edge[1]) $(edge[2]) $(edge[3]) \n")
        end
    end
    visualize(sample) #assumes sample_mesh is .obj; can use save_as_obj from MeshMaker.jl to make it happen
end =#

#= function plotxy!(f, method::ForwardMethod, (g, emiss)) end
function plotxy!(f, method::BackwardMethod, (emiss, g)) end =#
