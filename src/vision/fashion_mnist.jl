"""
    load_fashion_mnist(n::Union{Nothing,Int}=nothing; seed=data_seed)

Loads FashionMNIST data.
"""
function load_fashion_mnist(n::Union{Nothing,Int}=nothing; seed=data_seed)

    # Assertions:
    ensure_positive(n)

    X, y = MLDatasets.FashionMNIST(:train)[:]
    X = Flux.flatten(X)
    X = X .* 2.0f0 .- 1.0f0
    y = MLJBase.categorical(y)
    y = DataAPI.unwrap.(y)
    # counterfactual_data = CounterfactualExplanations.CounterfactualData(
    #     X, y; domain=(-1.0, 1.0), standardize=false
    # )

    # Checks and warnings
    request_more_than_available(n, size(X, 2))

    # Randomly under-/over-sample:
    rng = get_rng(seed)
    if !isnothing(n) && n != size(X)[2]
        X, y = subsample(rng, X, y, n)
    end

    return (X, y)
end

"""
    load_fashion_mnist_test()

Loads FashionMNIST test data.
"""
function load_fashion_mnist_test()
    X, y = MLDatasets.FashionMNIST(:test)[:]
    X = Flux.flatten(X)
    X = X .* 2.0f0 .- 1.0f0
    y = MLJBase.categorical(y)
    y = DataAPI.unwrap.(y)
    # counterfactual_data = CounterfactualExplanations.CounterfactualData(
    #     X, y; domain=(-1.0, 1.0)
    # )
    return (X, y)
end
