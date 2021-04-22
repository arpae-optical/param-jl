using FastAI
using FastAI.Datasets
using FastAI: DLPipelines

num_wavelens = 150

abstract type EmissBackwardsTask <: DLPipelines.LearningTask 
    #granularity, mesh shape?
end

struct EmissBackwards <: DLPipelines.LearningMethod{EmissBackwardsTask}

end

function DLPipelines.encodeinput(method::EmissBackwards, emiss_input)
    encoding = Chain(
            Dense(150, 1) #Convert from 150 -> 1?
            #TODO fix model syntax
            Conv1d(1, 8, 3),
            GELU(),
            # BatchNorm1d(8),
            Conv1d(8, 16, 3),
            GELU(),
            # BatchNorm1d(16),
            Conv1d(16, 64, 3),
            GELU(),
            # BatchNorm1d(64),
            Conv1d(64, 256, 3),
            Flatten(),
        )
    encoding(emiss_input)

end

function DLPipelines.encodetarget(method::EmissBackwards, surface_target)
    #turn surface target into something: mesh?
    encoded_mesh = f^-1(surface_target)
    return(encoded_mesh)
end

function DLPipelines.decodeŷ(method::EmissBackwards, ŷ)
    #TODO fix decoder syntax
    decoder = Sequential(
            Linear(Z, 512),
            GELU(),
            # BatchNorm1d(512),
            Linear(512, 256),
            GELU(),
            # BatchNorm1d(256),
            Linear(256, 128),
        )

    decoder(ŷ)
end