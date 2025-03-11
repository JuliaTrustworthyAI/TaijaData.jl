"""
    load_uci_adult(n::Union{Nothing,Int}=1000; seed=data_seed)

Loads data from the UCI 'Adult' dataset.
"""
function load_uci_adult(n::Union{Nothing,Int}=1000; seed=data_seed, return_cats::Bool=false)

    # Assertions:
    ensure_positive(n)

    # Load data
    df = CSV.read(joinpath(data_dir, "adult.csv"), DataFrames.DataFrame) |> format_header!

    # Categoricals:
    cats = [
        "workclass",
        "education",
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "country",
    ]
    df = coerce(df, [catvar => Multiclass for catvar in cats]...)

    # Transformations:
    transformer = MLJModels.Standardizer(; count=true) |> MLJModels.ContinuousEncoder()
    mach = MLJBase.fit!(machine(transformer, df[:, DataFrames.Not(:target)]))
    X = MLJBase.transform(mach, df[:, DataFrames.Not(:target)])

    # Store indices for categorical features for use with CE.jl:
    features_categorical = get_categorical_indices(X, cats)

    X = Matrix(X)
    X = permutedims(X)
    y = df.target

    # Checks and warnings
    request_more_than_available(n, size(X, 2))

    # Randomly under-/over-sample:
    rng = get_rng(seed)
    if !isnothing(n) && n != size(X)[2]
        X, y = subsample(rng, X, y, n)
    end

    # Return categorical indices:
    if return_cats
        return (X, y), features_categorical
    end

    return (X, y)
end
