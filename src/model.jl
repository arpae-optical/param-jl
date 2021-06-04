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
include("types.jl")
include("constants.jl")
include("data.jl")

function backwards_model()
    data = getobs_laser() 

    labels = [
        [entry[1].freq, 
        entry[1].wavelen, 
        entry[1].laser_power_W,
        entry[1].laser_scan_spacing_x,
        entry[1].laser_y_speed
        ] for entry in data
        ] |> gpu
    #deconstruct data back into vector

    input = [entry[2] for entry in data] |> gpu 

    data = zip(labels, input)

    model = Chain(Dense(NUM_WAVELENS, 32), Dense(32, 64), Dense(64, 128), Dense(128, 64), Dense(64, 32), Dense(32, 5, sigmoid))

    θ = Flux.params(model)

    optimizer = Descent(0.1)

    validation_list = []

    for epoch in 1:100
        for (input_data, label) in training_data
            prediction = model(input_data)
            grads = gradient(θ) do 
                current_loss = Flux.mse(prediction,label)
            end
            Flux.update!(optimizer, θ, grads)
        end
        test_data, test_label = first(training_data)
        push!(validation_list, Flux.mse(model(test_data), test_label))
    end

    plot([1:100], validation_list)
end


function forwards_model()
    data = getobs_laser() 

    input = [
        [entry[1].freq, 
        entry[1].wavelen, 
        entry[1].laser_power_W,
        entry[1].laser_scan_spacing_x,
        entry[1].laser_y_speed
        ] for entry in data
        ] |> gpu
    #deconstruct data back into vector

    labels = [entry[2] for entry in data] |> gpu 

    data = zip(labels, input)

    model = Chain(Dense(5, 32), Dense(32, 64), Dense(64, 128), Dense(128, 64), Dense(64, 32), Dense(32, NUM_WAVELENS, sigmoid))

    θ = Flux.params(model)

    optimizer = Descent(0.1)

    validation_list = []

    for epoch in 1:100
        for (input_data, label) in training_data
            prediction = model(input_data)
            grads = gradient(θ) do 
                current_loss = Flux.mse(prediction,label)
            end
            Flux.update!(optimizer, θ, grads)
        end
        test_data, test_label = first(training_data)
        push!(validation_list, Flux.mse(model(test_data), test_label))
    end

    plot([1:100], validation_list)
end

forwards_model()
