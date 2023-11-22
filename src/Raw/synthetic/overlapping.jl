"""
    load_overlapping(n=250; seed=data_seed)

Loads overlapping synthetic data.
"""
function load_overlapping_raw(n=250; seed=data_seed)
    raw_data = load_blobs_raw(n; seed=seed, centers=2, cluster_std=2.0)

    return raw_data
end
