"""
    load_fashion_mnist(n::Union{Nothing,Int}=nothing)

Loads and prepares FashionMNIST data.
"""
function load_fashion_mnist_raw(n::Union{Nothing,Int}=nothing)
    X, y = MLDatasets.FashionMNIST(:train)[:]
    X = Flux.flatten(X)
    X = X .* 2.0f0 .- 1.0f0
    y = MLJBase.categorical(y)
    # counterfactual_data = CounterfactualExplanations.CounterfactualData(
    #     X, y; domain=(-1.0, 1.0), standardize=false
    # )
    # counterfactual_data.X = Float32.(counterfactual_data.X)
    # # Undersample:
    # if !isnothing(n)
    #     counterfactual_data = CounterfactualExplanations.DataPreprocessing.subsample(
    #         counterfactual_data, n
    #     )
    # end
    return (X, y)
end

"""
    load_fashion_mnist_test()

Loads and prepares FashionMNIST test data.
"""
function load_fashion_mnist_test_raw()
    X, y = MLDatasets.FashionMNIST(:test)[:]
    X = Flux.flatten(X)
    X = X .* 2.0f0 .- 1.0f0
    y = MLJBase.categorical(y)
    # counterfactual_data = CounterfactualExplanations.CounterfactualData(
    #     X, y; domain=(-1.0, 1.0)
    # )
    # counterfactual_data.X = Float32.(counterfactual_data.X)
    return (X, y)
end
