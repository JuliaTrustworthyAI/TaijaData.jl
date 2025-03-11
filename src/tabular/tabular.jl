function pre_pre_process(
    fname::String,
    n::Union{Nothing,Int};
    rng::AbstractRNG,
    shuffle::Bool,
    train_test_split::Union{Nothing,Real},
)
    df = CSV.read(joinpath(data_dir, fname), DataFrames.DataFrame) |> format_header!
    ntotal = size(df, 1)
    request_more_than_available(n, ntotal)
    nfinal_train, nfinal_test = nfinal(n, ntotal, train_test_split)
    df = shuffle_rows(rng, df, shuffle)
    df_train, df_test = apply_split(train_test_split, df)
    nreq = nfinal_train + nfinal_test
    
    return df, df_train, df_test, nfinal_train, nfinal_test, ntotal, nreq
end

function pre_process(
    transformer,
    df_train::DataFrame,
    df_test::DataFrame;
    rng::AbstractRNG,
    nfinal_train::Int,
    nfinal_test::Int,
    ntotal::Int,
    nreq::Int,
    return_cats::Bool=false,
    cats::Union{Nothing,Vector{<:String}}=nothing,
    train_test_split::Union{Nothing,Real}
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
    if !isnothing(train_test_split)
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