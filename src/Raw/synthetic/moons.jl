"""
    load_moons_raw(n=250; seed=data_seed, kwrgs...)

Loads synthetic moons data.
"""
function load_moons_raw(n=250; seed=data_seed, kwrgs...)
    if isa(seed, Random.AbstractRNG)
        X, y = MLJBase.make_moons(n; rng=seed, kwrgs...)
    else
        Random.seed!(seed)
        X, y = MLJBase.make_moons(n; kwrgs...)
    end

    X = permutedims(MLJBase.matrix(X))
    y = DataAPI.unwrap.(y)

    return (X, y)
end
