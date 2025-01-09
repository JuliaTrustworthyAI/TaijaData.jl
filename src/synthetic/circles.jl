"""
    load_circles(n=250; seed=data_seed, noise=0.15, factor=0.01)

Loads synthetic circles data.
"""
function load_circles(n=250; seed=data_seed, noise=0.15, factor=0.01)
    if isa(seed, Int)
        seed = Xoshiro(seed)
    end

    X, y = MLJBase.make_circles(n; rng=seed, noise=noise, factor=factor)
    X = permutedims(MLJBase.matrix(X))
    y = DataAPI.unwrap.(y)

    return (X, y)
end
