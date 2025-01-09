"""
    load_overlapping(n=250)

Loads overlapping synthetic data.

!!! note
    This calls the [`load_blobs`](@ref) function with specific parameters and the seed set to [`data_seed`](@ref). The function returns the synthetic data with two clusters that are linearly separable. For more flexibility, you can use [`load_blobs`](@ref) directly with different parameters if needed.
"""
function load_overlapping(n=250)
    data = load_blobs(n; seed=data_seed, centers=2, cluster_std=2.0)
    return data
end
