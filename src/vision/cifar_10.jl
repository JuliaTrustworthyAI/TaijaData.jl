"""
    load_cifar_10(n::Union{Nothing, Int}=nothing; seed=data_seed)

Loads data from the CIFAR-10 dataset.
"""
function load_cifar_10(n::Union{Nothing,Int}=nothing; seed=data_seed)

    # Assertions:
    ensure_positive(n)

    X, y = MLDatasets.CIFAR10()[:] # [:] gives us X, y
    X = Flux.flatten(X)
    X = X .* 2 .- 1                 # normalization between [-1, 1]
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
    load_cifar_10_test()

Loads test data from the CIFAR-10 dataset.
"""
function load_cifar_10_test()
    X, y = MLDatasets.CIFAR10(:test)[:]
    X = Flux.flatten(X)
    X = X .* 2 .- 1 # normalization between [-1, 1]
    y = MLJBase.categorical(y)
    y = DataAPI.unwrap.(y)
    # counterfactual_data = CounterfactualExplanations.CounterfactualData(
    #     X, y; domain=(-1.0, 1.0)
    # )
    return (X, y)
end
