struct Adult <: TabularData end

get_original_feature_names(data::Adult) = get_original_feature_names("adult.csv")

load_data(data::Adult; kwrgs...) = load_uci_adult(; kwrgs...)

get_feature_names(data::Adult) = load_data(data; feature_names=true)

"""
    load_uci_adult(
        n::Union{Nothing,Int}=1000;
        seed=data_seed,
        return_cats::Bool=false,
        train_test_split::Union{Nothing,Real}=nothing,
        shuffle::Bool=false,
    )

Loads data from the UCI 'Adult' dataset.
"""
function load_uci_adult(
    n::Union{Nothing,Int}=1000;
    seed=data_seed,
    return_cats::Bool=false,
    train_test_split::Union{Nothing,Real}=nothing,
    shuffle::Bool=false,
    kwrgs...,
)

    # Setup:
    rng = get_rng(seed)
    ensure_positive(n)
    ensure_bounded(train_test_split)

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

    # Load data
    df_train, df_test, nfinal_train, nfinal_test, ntotal, nreq = pre_pre_process(
        "adult.csv", n; rng, shuffle, train_test_split, cats
    )

    # Fit on train set only to avoid leakage:
    transformer = MLJModels.Standardizer(; count=true) |> MLJModels.ContinuousEncoder()

    # Pre-process:
    output = pre_process(
        transformer,
        df_train,
        df_test;
        rng,
        nfinal_train,
        nfinal_test,
        ntotal,
        nreq,
        return_cats,
        cats,
        kwrgs...,
    )

    return output
end
