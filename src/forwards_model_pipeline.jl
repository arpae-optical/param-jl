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
function DLPipelines.encodeinput(method::EmissForward,surface_input)

    #convert input to array
    encoded_surface = f(surface_input)
    return(encoded_surface)
end

function DLPipelines.encodetarget(method::EmissForward, emiss_target)
    #"convert output" to target emissivity

    return(emiss_target)
end

function DLPipelines.decodeŷ(method::EmissForward, ŷ)
    #"convert y hat" to predicted emissivity
    
    return(ŷ)
end

model = Chain(Dense(x, 32, gelu), #sizeof input data
Dense(32, 64, gelu),
Dense(64, 128, gelu),
Dense(128, 64, gelu),
Dense(64, 32, gelu),
Dense(32, num_wavelens, sigmoid)) #same chain as others?

opt = ADAM

#TODO: loss is based on mape and I don't know what that is, so figure that out, configure optimizers, train/validation/test

function plotsample!(f, method::EmissForward, sample_mesh)
    visualize(sample_mesh) #assumes sample_mesh is .obj; can use save_as_obj from MeshMaker.jl to make it happen
end

function plotxy!(f, method::EmissForward, (x,y))
    #TODO figure out what this does
end