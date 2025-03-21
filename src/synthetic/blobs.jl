"""
    load_blobs(n=250; seed=data_seed, k=2, centers=2, kwrgs...)

Loads overlapping synthetic data.
"""
function load_blobs(n=250; seed=data_seed, k=2, centers=2, kwrgs...)
    seed = get_rng(seed)

    X, y = MLJBase.make_blobs(n, k; centers=centers, rng=seed, kwrgs...)
    X = permutedims(MLJBase.matrix(X))
    y = DataAPI.unwrap.(y)

    return (X, y)
end
