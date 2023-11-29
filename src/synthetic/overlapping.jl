"""
    load_overlapping(n=250; seed=data_seed)

Loads overlapping synthetic data.
"""
function load_overlapping(n=250; seed=data_seed)
    data = load_blobs(n; seed=seed, centers=2, cluster_std=2.0)

    return data
end
