"""
    load_moons(n=250; seed=data_seed, kwrgs...)

Loads synthetic moons data.
"""
function load_moons(n=250; seed=data_seed, kwrgs...)
    if isa(seed, Int)
        seed = Xoshiro(seed)
    end
    X, y = MLJBase.make_moons(n; rng=seed, kwrgs...)
    X = permutedims(MLJBase.matrix(X))
    y = DataAPI.unwrap.(y)
    return (X, y)
end
