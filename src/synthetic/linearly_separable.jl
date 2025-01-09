"""
    load_linearly_separable(n=250; seed=data_seed)

Loads linearly separable synthetic data.
"""
function load_linearly_separable(n=250; seed=data_seed)
    data = load_blobs(n; seed=seed, centers=2, cluster_std=0.5)
    return data
end
