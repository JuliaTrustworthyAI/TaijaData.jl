"Abstract type for tabular datasets."
abstract type TabularData end

"""
    pre_pre_process(
        fname::String,
        n::Union{Nothing,Int};
        rng::AbstractRNG,
        shuffle::Bool,
        train_test_split::Union{Nothing,Real},
    )

Common intitial pre-processing workflow used for tabular datasets.
"""
function pre_pre_process(
    fname::String,
    n::Union{Nothing,Int};
    rng::AbstractRNG,
    shuffle::Bool,
    train_test_split::Union{Nothing,Real},
    cats::Vector=[],
)
    df = CSV.read(joinpath(data_dir, fname), DataFrames.DataFrame) |> format_header!
    ntotal = size(df, 1)
    request_more_than_available(n, ntotal)
    nfinal_train, nfinal_test = nfinal(n, ntotal, train_test_split)
    if !isnothing(nfinal_train) && !isnothing(nfinal_test)
        nreq = nfinal_train + nfinal_test
    else
        nreq = nothing
    end
    df = shuffle_rows(rng, df, shuffle)

    # Categoricals:
    if length(cats) > 0
        df = coerce(df, [catvar => Multiclass for catvar in cats]...)
    end

    df_train, df_test = apply_split(train_test_split, df)

    return df_train, df_test, nfinal_train, nfinal_test, ntotal, nreq
end

"""
    get_feature_names(fname::String)

Helper function to get feature names.
"""
function get_feature_names(fname::String)
    df = CSV.read(joinpath(data_dir, fname), DataFrames.DataFrame) |> format_header!
    return names(df)[names(df) .!= "target"]
end

"""
    pre_process(
        transformer,
        df_train::DataFrame,
        df_test::Union{Nothing,DataFrame};
        rng::AbstractRNG,
        nfinal_train::Union{Nothing,Int},
        nfinal_test::Union{Nothing,Int},
        ntotal::Int,
        nreq::Union{Nothing,Int},
        return_cats::Bool=false,
        cats::Union{Nothing,Vector{<:String}}=nothing,
    )

Common final pre-processing workflow used for tabular datasets.
"""
function pre_process(
    transformer,
    df_train::DataFrame,
    df_test::Union{Nothing,DataFrame};
    rng::AbstractRNG,
    nfinal_train::Union{Nothing,Int},
    nfinal_test::Union{Nothing,Int},
    ntotal::Int,
    nreq::Union{Nothing,Int},
    return_cats::Bool=false,
    cats::Union{Nothing,Vector{<:String}}=nothing,
)
    output = []

    mach = MLJBase.fit!(machine(transformer, df_train[:, DataFrames.Not(:target)]))

    # Transform training data:
    X, y, df_trans = apply_transformations(df_train, mach)
    # Randomly under-/over-sample (training set):
    X, y = subsample(rng, X, y, nfinal_train, nreq, ntotal)

    # Add training data to output:
    push!(output, X, y)

    # Add test data to output, if applicable:
    if !isnothing(df_test) && nfinal_test > 0
        # Transform test data:
        Xtest, ytest, _ = apply_transformations(df_test, mach)
        # Randomly under-/over-sample (test set):
        Xtest, ytest = subsample(rng, Xtest, ytest, nfinal_test, nreq, ntotal)
        push!(output, Xtest, ytest)
    end

    # Add categorical indices, if applicable:
    if return_cats
        push!(output, get_categorical_indices(df_trans, cats))
    end

    return tuple(output...)
end

include("adult.jl")
include("california_housing.jl")
include("credit_default.jl")
include("gmsc.jl")
include("german_credit.jl")
