"""
    load_blobs(n=250; seed=data_seed, kwrgs...)

Loads overlapping synthetic data.
"""
function load_blobs_raw(n=250; seed=data_seed, k=2, centers=2, kwrgs...)
    if isa(seed, Random.AbstractRNG)
        X, y = MLJBase.make_blobs(n, k; centers=centers, rng=seed, kwrgs...)
    else
        Random.seed!(seed)
        X, y = MLJBase.make_blobs(n, k; centers=centers, kwrgs...)
    end

    return (X, y)
end