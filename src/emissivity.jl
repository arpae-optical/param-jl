using FastAI, Plots, Flux, StaticArrays, LearnBase, Mongoc, CSV, DataFrames, Distributions
using Mongoc: BSONObjectId, BSON

function Base.rand(::HexSphere)
    HexSphere(len = 10^rand(Uniform(0.0, 0.7)), diam = rand(Uniform(1.0, 10.0)))
end

function Base.rand(::HexCavity)
    HexCavity(
        len = 10^rand(Uniform(0.065, 0.4)),
        height = rand(Uniform(0.0, 10.0)),
        diam = rand(Uniform(0.0, 10.0)),
    )
end
function Base.rand(::TriGroove)
    TriGroove(
        len = 10^rand(Uniform(0.0, 0.4)),
        height = rand(Uniform(0.0, 10.0)),
        diam = 10^rand(Uniform(0.0, 0.4)),
    )
end

function emiss(g::HexCavity, emiss_in::InterpolatedEmissPlot)
    R = g.diam / 2
    A₂ = π * R^2
    A₁ = A₂ + 2π * R * g.height / g.diam
    F₁₁ = 1 - A₂ / A₁
    A₂₃ = 3(g.len / g.diam)^2 / (2 * √(3))
    A₃ = A₂₃ - A₂
    (A₁ / A₂₃) * (1 - F₁₁) / (1 / emiss_in - F₁₁ * (1 / emiss_in - 1)) + 1 / emiss_in * A₃ / A₂₃
end

function emiss(g::HexSphere, emiss_in::InterpolatedEmissPlot)
    R = g.diam / 2
    L = g.len
    s = √(L^2 / 3) #length of one side of hexagon |

    Aₛ = 4π * R^2 #surface area of sphere
    Aₕ = 3 / 2 * s * L #surface area of hexagon (side times base over two times six)
    A₁ = Aₛ + Aₕ #surface area of solid
    A₂ = Aₕ #apparent area
    # K=1-F₁₁-A₂/A₁;
    1 / (1 + A₂ / A₁ * (1 / emiss_in - 1))
end

function emiss(g::TriGroove, emiss_in::InterpolatedEmissPlot)
    # hexagonal stack, cylindrical cavity
    A₁ = √(4g.height^2 + g.depth^2)
    A₂ = g.depth
    A₂₃ = g.len
    A₃ = A₂₃ - A₂
    F₁₁ = 1 - A₂ / A₁
    (A₁ / A₂₃) * (1 - F₁₁) / (1 / emiss_in - F₁₁ * (1 / emiss_in - 1)) + emiss_in * A₃ / A₂₃
end
