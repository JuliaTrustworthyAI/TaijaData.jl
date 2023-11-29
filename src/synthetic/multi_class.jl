"""
    load_multi_class(n=250; seed=data_seed)

Loads multi-class synthetic data.
"""
function load_multi_class(n=250; seed=data_seed, centers=4)
    data = load_blobs(n; seed=seed, centers=centers, cluster_std=0.5)

    return data
end
