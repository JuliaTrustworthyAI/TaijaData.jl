"""
    load_california_housing(n::Union{Nothing,Int}=5000; seed=data_seed)

Loads California Housing data.
"""
function load_california_housing(n::Union{Nothing,Int}=5000; seed=data_seed)

    # Assertions:
    ensure_positive(n)

    # Load:
    df = CSV.read(joinpath(data_dir, "cal_housing.csv"), DataFrames.DataFrame)
    # Pre-process features:
    transformer = MLJModels.Standardizer(; count=true)
    mach = MLJBase.fit!(MLJBase.machine(transformer, df[:, DataFrames.Not(:target)]))
    X = MLJBase.transform(mach, df[:, DataFrames.Not(:target)])
    X = Matrix(X)
    X = permutedims(X)

    # Counterfactual data:
    y = Int.(df.target)

    # Checks and warnings
    request_more_than_available(n, size(X, 2))

    # Randomly under-/over-sample:
    rng = get_rng(seed)
    if !isnothing(n) && n != size(X)[2]
        X, y = subsample(rng, X, y, n)
    end

    return (X, y)
end
