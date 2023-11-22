"""
    load_mnist()

Loads and prepares MNIST data.
"""
function load_mnist_raw(n::Union{Nothing,Int}=nothing)
    X, y = MLDatasets.MNIST(:train)[:]
    X = Flux.flatten(X)
    X = X .* 2.0f0 .- 1.0f0
    y = MLJBase.categorical(y)
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
    load_mnist_test()

Loads and prepares MNIST test data.
"""
function load_mnist_test_raw()
    X, y = MLDatasets.MNIST(:test)[:]
    X = Flux.flatten(X)
    X = X .* 2.0f0 .- 1.0f0
    y = MLJBase.categorical(y)
    # counterfactual_data = CounterfactualExplanations.CounterfactualData(
    #     X, y; domain=(-1.0, 1.0)
    # )
    return (X, y)
end
