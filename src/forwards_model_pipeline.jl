using FastAI
using FastAI.Datasets
using FastAI: DLPipelines

num_wavelens = 150

mape(pred, true_value) = mean(abs(((pred - true_value)/true_value))) 

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

model = Chain(Dense(2, 32, gelu), #sizeof input data
Dense(32, 64, gelu),
Dense(64, 128, gelu),
Dense(128, 64, gelu),
Dense(64, 32, gelu),
Dense(32, num_wavelens, sigmoid)) #same chain as others?

opt = ADAM

function loss(self, batch)
    geom, structured_emiss = batch
    pred_emiss = self(geom)
    if stage == "train" 
        loss = self.train_metric(pred_emiss, structured_emiss)
    else 
        loss = self.val_metric(pred_emiss, structured_emiss)
    end
    return loss
end

function plotsample!(f, method::EmissForward, sample)
    #sample::((rx, rz), emiss)
    #TODO make sure this actually works
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
end
