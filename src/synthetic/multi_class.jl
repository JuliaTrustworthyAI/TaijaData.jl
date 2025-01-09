"""
    load_multi_class(n=250; seed=data_seed, centers=4)

Loads multi-class synthetic data.

!!! note
    This calls the [`load_blobs`](@ref) function with specific parameters and the seed set to [`data_seed`](@ref).
"""
function load_multi_class(n=250; seed=data_seed, centers=4)
    data = load_blobs(n; seed=seed, centers=centers, cluster_std=0.5)
    return data
end
