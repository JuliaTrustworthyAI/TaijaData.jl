struct CreditDefault <: TabularData end

function get_original_feature_names(data::CreditDefault)
    get_original_feature_names("credit_default.csv")
end

load_data(data::CreditDefault; kwrgs...) = load_credit_default(; kwrgs...)

get_feature_names(data::CreditDefault) = load_data(data; feature_names=true)

"""
    load_credit_default(
        n::Union{Nothing,Int}=5000;
        seed=data_seed,
        train_test_split::Union{Nothing,Real}=nothing,
        shuffle::Bool=false,
        return_cats::Bool=false,
    )

Loads UCI Credit Default data.
"""
function load_credit_default(
    n::Union{Nothing,Int}=5000;
    seed=data_seed,
    train_test_split::Union{Nothing,Real}=nothing,
    shuffle::Bool=false,
    return_cats::Bool=false,
    kwrgs...,
)

    # Setup:
    rng = get_rng(seed)
    ensure_positive(n)
    ensure_bounded(train_test_split)

    # Categoricals:
    cats = ["sex", "education", "marriage"]

    # Load data
    df_train, df_test, nfinal_train, nfinal_test, ntotal, nreq = pre_pre_process(
        "credit_default.csv", n; rng, shuffle, train_test_split, cats
    )

    # Transformer:
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
