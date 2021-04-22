using FastAI
using FastAI.Datasets
using FastAI: DLPipelines
using Distributions
using Statistics

num_wavelens = 150
NUM_SIMULATORS = size(simulators) #gonna have to import this somewhere, pl.LightningModule in the class arguments makes this a little opaque

struct ModelOutput
    geoms
    mean
    log_variance
    class_logits
end

struct Loss
    aspect_ratio_loss
    structured_emiss_mape_loss
    kl_loss
    loss
end
abstract type EmissBackwardsTask <: DLPipelines.LearningTask 
    #granularity, mesh shape?
end

struct EmissBackwards <: DLPipelines.LearningMethod{EmissBackwardsTask}

end

function DLPipelines.encodeinput(method::EmissBackwards, emiss_input)
    
    encoding(emiss_input)

end

function DLPipelines.encodetarget(method::EmissBackwards, surface_target)
    #turn surface target into something: mesh?
    encoded_mesh = f^-1(surface_target)
    return(encoded_mesh)
end

function DLPipelines.decodeŷ(method::EmissBackwards, ŷ)
    #TODO fix decoder syntax
    

    decoder(ŷ)
end

function backwards_model(structured_emiss)
    #
    h = structured_emiss.reshape(-1, 1, self.num_wavelens) #TODO make sure this works: this was a structured_emiss method

    encoder = Chain(
            #TODO fix model syntax
            Conv((3,), 1 => 8, gelu),
            GELU(),
            # BatchNorm1d(8),
            Conv((3,), 8 => 16, gelu),
            GELU(),
            # BatchNorm1d(16),
            Conv((3,), 16 => 64, gelu),
            GELU(),
            # BatchNorm1d(64),
            Conv((3,), 64 => 256, gelu),
            Flatten()
        )



    Z = 1024

    self.mean_head = Chain(Dense(71 * 512, Z))

    self.log_var_head = Chain(
        Dense(71 * 512, Z),
    )

    decoder = Sequential(
            Linear(Z, 512),
            GELU(),
            # BatchNorm1d(512),
            Linear(512, 256),
            GELU(),
            # BatchNorm1d(256),
            Linear(256, 128),
        )

    geom_heads = ( #TODO make sure modulelist wasn't doing anything important
            [
                Chain(
                    Dense(128, 96, gelu),
                    # BatchNorm1d(96),
                    Dense(96, 64, gelu),
                    # BatchNorm1d(64),
                    Dense(64, 32, gelu),
                    # BatchNorm1d(32),
                    Dense(32, 2)
                )
                for _ 1:NUM_SIMULATORS
            ]
        )


    h = structured_emiss.reshape(-1, 1, self.num_wavelens) #TODO make work

    h = self.encoder(h)
    mean, log_var = self.mean_head(h), self.log_var_head(h)
    std = e^(log_var / 2)

    dist = Normal(
        μ=mean,
        σ=std
    )
    zs = gradient(mean -> rand(dist), 0) == (1,) #TODO make sure this works, see rsample https://github.com/FluxML/Flux.jl/issues/1431

    decoded = self.decoder(zs)

    geoms = [g(decoded) for g in geom_heads]
    class_logits = self.classifier(structured_emiss)
 
    ModelOutput(geoms, mean, log_var, class_logits)
end

function _loss(self, batch, true_batch_idx::Int64, stage::String)

    structured_emiss, *_ = batch #TODO does this work

    preds = model(structured_emiss)

    if stage == "train"
        self.simulators[true_batch_idx](preds.geoms[tre_batch_idx]) #use Symbol here
        structured_emiss_mape_loss = args.structured_emiss_weight*mape(pred_emiss, structured_emiss)
    else:
        pred_emiss = [sim(g) for sim, g in zip(self.simulators, preds.geoms)]
        structured_emiss_mape_loss = min(
            mape(p, structured_emiss) for p in pred_emiss
        )
    end

    kl_loss = (
        mean(
        args.kl_weight
        * (
                -(
                    1 + preds.log_variance - preds.mean^2 - e^preds.log_variance
                ) #TODO figure out .sum(dim=-1); what did it do?
                / 2
            ))
    )

    total_loss = structured_emiss_mape_loss + kl_loss

    if true_batch_idx == 0
        spheroids = preds.geoms[0]
        aspect_ratios = spheroids.max(1).values / spheroids.min(1).values #TODO figure out what the max method is doing
        aspect_ratio_loss = mean(aspect_ratios)
    else
        aspect_ratio_loss = zeros(size(total_loss)) #TODO is this equivalent to torch.tensor(0)
    end

    return(Loss(aspect_ratio_loss, structured_emiss_mape_loss, kl_loss, total_loss))

function _log
    #No idea how to do this
end

function _step(self, batch, *, stage:: String)
    #also this
end

opt = ADAM #scheduler, monitor?

#TODO training step, validation step, test step