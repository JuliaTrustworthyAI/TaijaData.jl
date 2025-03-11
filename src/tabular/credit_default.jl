"""
    load_credit_default(n::Union{Nothing,Int}=5000; seed=data_seed)

Loads UCI Credit Default data.
"""
function load_credit_default(
    n::Union{Nothing,Int}=5000; seed=data_seed, return_cats::Bool=false
)

    # Assertions:
    ensure_positive(n)

    # Load:
    df =
        CSV.read(joinpath(data_dir, "credit_default.csv"), DataFrames.DataFrame) |>
        format_header!

    # Categoricals:
    cats = ["sex", "education", "marriage"]
    df = coerce(df, [catvar => Multiclass for catvar in cats]...)

    # Transformations:
    transformer = MLJModels.Standardizer(; count=true) |> MLJModels.ContinuousEncoder()
    mach = MLJBase.fit!(MLJBase.machine(transformer, df[:, DataFrames.Not(:target)]))
    X = MLJBase.transform(mach, df[:, DataFrames.Not(:target)])

    # Store indices for categorical features for use with CE.jl:
    features_categorical = get_categorical_indices(X, cats)

    X = permutedims(Matrix(X))
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
