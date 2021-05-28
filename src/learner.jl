using Statistics
function DLPipelines.encodeinput(method::ForwardMethod, ctx, input::Spheroid)
    x = [input.rx; input.rz]
end
function DLPipelines.encodeinput(method::ForwardMethod, ctx, input::HexCavity)
    [input.diam; input.height; input.len]
end
function DLPipelines.encodeinput(method::ForwardMethod, ctx, input::HexSphere)
    [input.diam; input.len]
end
function DLPipelines.encodeinput(method::ForwardMethod, ctx, input::TriGroove)
    [input.depth; input.height; input.len]
end

function DLPipelines.encodetarget(method::ForwardMethod, ctx, target::InterpolatedEmissPlot)
    t = [t.second[1] for t in target.emiss]
end
function DLPipelines.encodeinput(method::BackwardMethod, ctx, input::InterpolatedEmissPlot)
    [i.second[1] for i in input.emiss]
end

function DLPipelines.encode(
    method::BackwardMethod,
    ctx,
    (g, emiss)::Tuple{Spheroid,InterpolatedEmissPlot},
)
    encodeinput(method, ctx, emiss), encodetarget(method, ctx, g)
end

function DLPipelines.encodetarget(method::BackwardMethod, ctx, target::Spheroid)
    [target.rx; target.rz]
end
function DLPipelines.encodetarget(method::BackwardMethod, ctx, target::HexCavity)
    [target.diam; target.height; target.len]
end
function DLPipelines.encodetarget(method::BackwardMethod, ctx, target::HexSphere)
    [target.diam; target.len]
end
function DLPipelines.encodetarget(method::BackwardMethod, ctx, target::TriGroove)
    [target.depth; target.height; target.len]
end

function DLPipelines.decodeŷ(method::BackwardMethod, ctx, pred)
    pred
end

function DenseChain(dims...; activation::Function = gelu)
    t = Chain([Dense(i, o, activation) for (i, o) in zip(dims[1:(end - 1)], dims[2:end])]...)
end

function DLPipelines.methodmodel(method::ForwardMethod, backbone)
    Chain(DenseChain(2, 32, 64, 128, 64, 32), Dense(32, NUM_WAVELENS, sigmoid))
end

DLPipelines.methodlossfn(method::ForwardMethod) = Flux.Losses.mse

function DLPipelines.methodmodel(method::BackwardMethod, backbone)
    encoder = Chain(
        Conv((3,), 1 => 8, gelu),
        # BatchNorm1d(8),
        Conv((3,), 8 => 16, gelu),
        # BatchNorm1d(16),
        Conv((3,), 16 => 64, gelu),
        # BatchNorm1d(64),
        Conv((3,), 64 => 256),
        Flux.flatten,
    )

    Z = 1024
    mean_head = Dense(71 * 512, Z)
    std_head = Chain(Dense(71 * 512, Z),)

    decoder = Chain(
        Dense(Z, 512, gelu),
        # BatchNorm1d(512),
        Dense(512, 256, gelu),
        # BatchNorm1d(256),
        Dense(256, 128),
    )

    geom_heads = Dict(
        type => Chain(
            Dense(128, 96, gelu),
            # BatchNorm1d(96),
            Dense(96, 64, gelu),
            # BatchNorm1d(64),
            Dense(64, 32, gelu),
            # BatchNorm1d(32),
            Dense(32, length(fieldnames(type))),
        ) for type in (Spheroid, HexCavity, HexSphere, TriGroove)
    )

    classifier = Chain(
        Dense(NUM_WAVELENS, 64, gelu),
        # BatchNorm1d(64),
        Dense(64, 32, gelu),
        # BatchNorm1d(32),
        Dense(32, NUM_SIMULATORS),
    )

    return (structured_emiss,) -> begin
        h = encoder(reshape(structured_emiss, NUM_WAVELENS, 1, :))
        mean, std = mean_head(h), std_head(h)
        # TODO get rid of scalar indexing here (zip, eachcol)
        samples =
            Flux.stack([rand(MvNormal(m, s)) for (m, s) in zip(eachcol(mean), eachcol(std))], 2)
        decoded = decoder(samples)
        ForwardPred(Dict(k => g(decoded) for (k, g) in geom_heads), mean, std, structured_emiss)
    end

end

struct ForwardPred
    geoms::Dict
    mean::Any
    std::Any
    true_emiss::Union{Any,Nothing}
end

function DLPipelines.methodlossfn(method::BackwardMethod)
    (pred, target) -> begin
        geom, mean, std, true_emiss = pred.geoms, pred.mean, pred.std, pred.true_emiss
        true_emiss = gpu(true_emiss)
        target = gpu(target)
        gg = gpu(geom[Spheroid])

        # TODO generalize to any simulators
        pred_emiss = method.simulators[Spheroid](gg)

        mape_loss = mape(pred_emiss, true_emiss)

        var = std .^ 2
        kl_term = -sum(1 .+ log.(var) .- mean .^ 2 .- std; dims = 1) ./ 2
        kl_loss = Statistics.mean(dropdims(kl_term; dims = 1))

        aspect_ratio_loss = Statistics.mean(maximum(g) / minimum(g) for g in eachcol(gg))
        total_loss = mape_loss + kl_loss + aspect_ratio_loss

        total_loss
    end
end
