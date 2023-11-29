"""
    load_fashion_mnist(n::Union{Nothing,Int}=nothing)

Loads FashionMNIST data.
"""
function load_fashion_mnist(n::Union{Nothing,Int}=nothing)
    X, y = MLDatasets.FashionMNIST(:train)[:]
    X = Flux.flatten(X)
    X = X .* 2.0f0 .- 1.0f0
    y = MLJBase.categorical(y)
    y = DataAPI.unwrap.(y)
    # counterfactual_data = CounterfactualExplanations.CounterfactualData(
    #     X, y; domain=(-1.0, 1.0), standardize=false
    # )

    # Undersample:
    if !isnothing(n)
        X, y = subsample(X, y, n)
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
