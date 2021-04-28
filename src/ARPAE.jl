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
if CUDA.functional()
    CUDA.allowscalar(false)
end
break_on(:error)

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

    # fitonecycle!(fwd_learner, 1)

    backward_learner = methodlearner(
        BackwardMethod(Dict(Spheroid => fwd_learner.model)),
        dataset,
        nothing,
        ToGPU(),
        Metrics(MAPE);
        batchsize = BATCH_SIZE,
    )

    (x, y), _ = iterate(backward_learner.data.training)
    ŷ = backward_learner.model(x)
    loss = backward_learner.lossfn(ŷ, y)

    fitonecycle!(backward_learner, 1)

end
