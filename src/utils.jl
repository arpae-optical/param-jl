using FastAI, Plots, Flux, StaticArrays, LearnBase, Mongoc, CSV, DataFrames
using Mongoc: BSONObjectId, BSON

# TODO make this a core function (called "freeze!" probably)
function Base.delete!(ps::Flux.Params)
    for x in ps.params
        delete!(ps.params, x)
    end
    ps.params
end

