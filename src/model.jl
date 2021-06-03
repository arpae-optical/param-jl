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
include("data.jl")
include("constants.jl")

data = getobs_all(::Type{LaserParams}) 

input = [
    [entry[1].freq, 
    entry[1].wavelen, 
    entry[1].laser_power_W, 
    entry[1].laser_repetition_rate_kHz, 
    entry[1].laser_scan_spacing_x, 
    entry[1].laser_scan_spacing_y, 
    entry[1].laser_x_speed, 
    entry[1].laser_y_speed
    ] for entry in data
    ] |> gpu
#deconstruct data back into vector

labels = [entry[2][2][2] for entry in data] |> gpu #each entry is SVector{NUM_WAVELENS,Pair{Wavelen,Emiss}}, so this should give just Emiss.

model = Chain(DenseChain(9, 32, 64, 128, 64, 32), Dense(32, NUM_WAVELENS, sigmoid))

loss(x,y) = mse(model(x), y)

params = Flux.params(model)

optimizer = Descent(0.1)

Flux.train!(loss, params, [(input,labels)], optimizer)
