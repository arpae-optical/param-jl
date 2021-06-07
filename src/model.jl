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
    CUDA,
    Statistics
include("types.jl")
include("constants.jl")
include("data.jl")

function backwards_model()
    data = getobs_laser() 
    labels = Float64[]
    input = Vector{Float64}[]
    validation_labels = Float64[]
    validation_input = Vector{Float64}[]
    for (i, entry) in enumerate(data)
        if i < 900
            push!(labels, entry[1].laser_power_W::Float64)
            push!(labels, entry[1].laser_scanning_speed_x_dir_mm_per_s::Float64)
            push!(labels, entry[1].laser_scanning_line_spacing_y_dir_micron::Float64)
            push!(input, entry[2]::Vector{Float64})
        else
            push!(validation_labels, entry[1].laser_power_W::Float64)
            push!(validation_labels, entry[1].laser_scanning_speed_x_dir_mm_per_s::Float64)
            push!(validation_labels, entry[1].laser_scanning_line_spacing_y_dir_micron::Float64)
            push!(validation_input, entry[2]::Vector{Float64})
        end

    end

    data = zip(input, labels)

    validation = zip(validation_input, validation_labels) 

    model = Chain(Dense(935, 32), Dense(32, 64), Dense(64, 128), Dense(128, 64), Dense(64, 32), Dense(32, 3, sigmoid)) 

    θ = Flux.params(model)

    loss(x, y) = sum(Flux.crossentropy(model(x), y))

    optimizer = Descent(0.1)

    validation_list = []

    for epoch in 1:100
        Flux.train!(loss, θ, data, optimizer)

        validation_loss = mean(Flux.mse(model(input_data), label) for (input_data, label) in validation)
        println(validation_loss)
        push!(validation_list, validation_loss)
    end

    plot([1:100], validation_list)
    savefig("test_plot2.png")

end


function forwards_model()
    data = getobs_laser() 
    labels = Vector{Float64}[]
    input = Float64[]
    validation_labels = Vector{Float64}[]
    validation_input = Float64[]
    for (i, entry) in enumerate(data)
        if i < 900
            push!(input, entry[1].laser_power_W::Float64)
            push!(input, entry[1].laser_scanning_speed_x_dir_mm_per_s::Float64)
            push!(input, entry[1].laser_scanning_line_spacing_y_dir_micron::Float64)
            push!(labels, entry[2]::Vector{Float64})
        else
            push!(validation_input, entry[1].laser_power_W::Float64)
            push!(validation_input, entry[1].laser_scanning_speed_x_dir_mm_per_s::Float64)
            push!(validation_input, entry[1].laser_scanning_line_spacing_y_dir_micron::Float64)
            push!(validation_labels, entry[2]::Vector{Float64})
        end

    end

    data = zip(input, labels)

    validation = zip(validation_input, validation_labels) 

    model = Chain(Dense(935, 32), Dense(32, 64), Dense(64, 128), Dense(128, 64), Dense(64, 32), Dense(32, 3, sigmoid)) 

    θ = Flux.params(model)

    loss(x, y) = sum(Flux.crossentropy(model(x), y))

    optimizer = Descent(0.1)

    validation_list = []

    for epoch in 1:100
        Flux.train!(loss, θ, data, optimizer)

        validation_loss = mean(Flux.mse(model(input_data), label) for (input_data, label) in validation)
        println(validation_loss)
        push!(validation_list, validation_loss)
    end

    plot([1:100], validation_list)
    savefig("test_plot2.png")
end

backwards_model()
