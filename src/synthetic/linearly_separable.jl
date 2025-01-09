"""
    load_linearly_separable(n=250)

Loads linearly separable synthetic data.

!!! note
    This calls the [`load_blobs`](@ref) function with specific parameters and the seed set to [`data_seed`](@ref). The function returns the synthetic data with two clusters that are linearly separable. For more flexibility, you can use [`load_blobs`](@ref) directly with different parameters if needed.
"""
function load_linearly_separable(n=250)
    data = load_blobs(n; seed=data_seed, centers=2, cluster_std=0.5)
    return data
end
