module TaijaData

using CategoricalArrays
using Random
using LazyArtifacts
# using CounterfactualExplanations
using MLJBase
using CSV
using DataFrames
using MLJModels
# using CounterfactualExplanations.DataPreprocessing
using Flux
using MLDatasets
using StatsBase

const data_seed = 42
data_dir = joinpath(artifact"data-tabular", "data-tabular")

include("Raw/synthetic/blobs.jl")
include("Raw/synthetic/circles.jl")
include("Raw/synthetic/linearly_separable.jl")
include("Raw/synthetic/moons.jl")
include("Raw/synthetic/multi_class.jl")
include("Raw/synthetic/overlapping.jl")

include("Raw/tabular/adult.jl")
include("Raw/tabular/california_housing.jl")
include("Raw/tabular/credit_default.jl")
include("Raw/tabular/gmsc.jl")
include("Raw/tabular/german_credit.jl")

include("Raw/vision/cifar_10.jl")
include("Raw/vision/fashion_mnist.jl")
include("Raw/vision/mnist.jl")

include("Raw/utils.jl")

"A dictionary that provides an overview of the various benchmark datasets and the methods to load them."
const data_catalogue_raw = Dict(
    :synthetic => Dict(
        :linearly_separable => load_linearly_separable_raw,
        :overlapping => load_overlapping_raw,
        :multi_class => load_multi_class_raw,
        :blobs => load_blobs_raw,
        :moons => load_moons_raw,
        :circles => load_circles_raw,
    ),
    :tabular => Dict(
        :california_housing => load_california_housing_raw,
        :credit_default => load_credit_default_raw,
        :gmsc => load_gmsc_raw,
        :german_credit => load_german_credit_raw,
        :adult => load_uci_adult_raw,
    ),
    :vision => Dict(
        :mnist => load_mnist_raw,
        :fashion_mnist => load_fashion_mnist_raw,
        :cifar_10 => load_cifar_10_raw,
    ),
)

"""
    load_synthetic_data(n=250; seed=data_seed)

Loads all synthetic datasets and wraps them in a dictionary.
"""
function load_synthetic_data_raw(n=250; seed=data_seed, drop=nothing)
    _dict = data_catalogue[:synthetic]
    if !isnothing(drop)
        drop = drop isa Vector ? drop : [drop]
        @assert all(_drop in keys(_dict) for _drop in drop)
    else
        drop = []
    end
    _dict = filter(((k, v),) -> k ∉ [drop..., :blobs], _dict)
    data = Dict(key => fun(n; seed=seed) for (key, fun) in _dict)
    return data
end

# """
#     load_tabular_data(n=nothing; drop=nothing)

# Loads all tabular datasets and wraps them in a dictionary.
# """
# function load_tabular_data(n=nothing; drop=nothing)
#     _dict = data_catalogue[:tabular]
#     if !isnothing(drop)
#         drop = drop isa Vector ? drop : [drop]
#         @assert all(_drop in keys(_dict) for _drop in drop)
#     else
#         drop = []
#     end
#     _dict = filter(((k, v),) -> k ∉ drop, _dict)
#     data = Dict(key => fun(n) for (key, fun) in _dict)
#     return data
# end

export data_catalogue_raw
export load_linearly_separable_raw, load_overlapping_raw, load_multi_class_raw
export load_blobs_raw, load_circles_raw, load_moons_raw, load_multi_class_raw
export load_synthetic_data_raw
export load_california_housing_raw, load_credit_default_raw, load_gmsc_raw
export load_german_credit_raw, load_uci_adult_raw
export load_uci_adult_raw
export load_tabular_data_raw
export load_mnist_raw, load_mnist_test_raw
export load_fashion_mnist_raw, load_fashion_mnist_test_raw
export load_cifar_10_raw, load_cifar_10_test_raw

end
