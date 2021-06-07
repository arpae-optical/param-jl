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

    #using for loop instead of just [f(x) for x in y] because the data is being weird about using indexing. 
    #i.e. if the full data is data = [f(x) for x in y] and I try to take data[901:], it errors due to invalid indexing.
    #so I'm using enumerate and push! instead of worrying about it.
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

    data = zip(input, labels) #|> gpu

    validation = zip(validation_input, validation_labels) #|> gpu

    model = Chain(Dense(935, 32), Dense(32, 64), Dense(64, 128), Dense(128, 64), Dense(64, 32), Dense(32, 3, sigmoid)) #|> gpu

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
    #plot claims it shouldn't display unless returned. Since it's not the last line it shouldn't return. Still (only sometimes) erroring, but it saves anyway, so it's getting past the line.
    plot([1:100], validation_list)
    savefig("test_plot2.png")

end


function forwards_model()
    data = getobs_laser() 
    labels = Vector{Float64}[]
    input = Float64[]
    validation_labels = Vector{Float64}[]
    validation_input = Float64[]

    #Only difference is that this has input and labels swapped, and the model goes 935 -> 3 instead of 3 -> 935
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

    #aforementioned changes in the model
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
