using FastAI,
    Debugger,
    Plots,
    Flux,
    StaticArrays,
    LearnBase,
    Mongoc,
    CSV,
    DataFrames,
    Distributions,
    JLD2,
    FileIO,
    CUDA
using Mongoc: BSONObjectId, BSON
CUDA.allowscalar(false)

include("constants.jl")
include("utils.jl")
include("types.jl")
include("emissivity.jl")
include("data.jl")
include("learner.jl")
include("plotting.jl")

const SPHEROID_DATA_FILE = "data/spheroid_data.jld2"
function run()

    dataset = try
        load(SPHEROID_DATA_FILE, "dataset")
    catch
        dataset = getobs_all(Spheroid)
        jldsave(SPHEROID_DATA_FILE; dataset = dataset)
        dataset
    end

    fwd_learner = methodlearner(
        ForwardMethod(),
        dataset,
        nothing,
        ToGPU(),
        Metrics(MAPE);
        batchsize = BATCH_SIZE,
    )

    fitonecycle!(fwd_learner, 1)

    # freeze learner
    fwd_learner.params = delete!(fwd_learner.params)

    backward_learner = methodlearner(
        BackwardMethod(Dict(Spheroid => fwd_learner.model)),
        dataset,
        nothing,
        ToGPU(),
        Metrics(MAPE);
        batchsize = BATCH_SIZE,
    )

    FastAI.fit!(backward_learner, 2)

end
