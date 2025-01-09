"""
    load_overlapping(n=250; seed=data_seed)

Loads overlapping synthetic data.

!!! note
    This calls the [`load_blobs`](@ref) function with specific parameters and the seed set to [`data_seed`](@ref). To ensure overlapping clusters and reproducibility, setting the `seed` keyword argument has no effect. For more flexibility, you can use [`load_blobs`](@ref) directly with different parameters if needed.
"""
function load_overlapping(n=250; seed=data_seed)
    data = load_blobs(n; seed=data_seed, centers=2, cluster_std=2.0)
    return data
end
