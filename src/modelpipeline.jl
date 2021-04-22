using FastAI
using FastAI.Datasets
using FastAI: DLPipelines

num_wavelens = 150

abstract type EmissForwardTask <: DLPipelines.LearningTask end

struct EmissForward <: DLPipelines.LearningMethod{EmissForwardTask}
    #granularity? Mesh?
    
end

method = EmissForward()

"""Turns input into model-ready form"""
function DLPipelines.encodeinput(method::EmissForward,input)

    #convert input to array
    encoded_data = f(input)
    return(encoded_data)
end

function DLPipelines.encodetarget(method::EmissForward, emiss_target)
    #"convert output" to target emissivity

    return(emiss_target)
end

function DLPipelines.decodeŷ(method::MyImageClassification, ŷ)
    #"convert y hat" to predicted emissivity
    
    return(ŷ)
end

model = Chain(Dense(x, 32, gelu), #sizeof input data
Dense(32, 64, gelu),
Dense(64, 128, gelu),
Dense(128, 64, gelu),
Dense(64, 32, gelu),
Dense(32, num_wavelens, sigmoid)) #same chain as others?

abstract type EmissBackwardsTask <: DLPipelines.LearningTask end
    #granularity, mesh shape?
end

struct EmissBackwards <: DLPipelines.LearningMethod{EmissBackwardsTask}

end

