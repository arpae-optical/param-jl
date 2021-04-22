function plotsample!(f, method::ImageClassification, sample)
    image, class = sample
    f[1, 1] = ax1 = imageaxis(f, title = class)
    plotimage!(ax1, image)
end