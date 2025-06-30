struct GMSC <: TabularData end

get_original_feature_names(data::GMSC) = get_original_feature_names("gmsc.csv")

load_data(data::GMSC; kwrgs...) = load_gmsc(; kwrgs...)

get_feature_names(data::GMSC) = load_data(data; feature_names=true)

"""
    load_gmsc(
        n::Union{Nothing,Int}=5000;
        seed=data_seed,
        train_test_split::Union{Nothing,Real}=nothing,
        shuffle::Bool=false,
    )

Loads Give Me Some Credit (GMSC) data.
"""
function load_gmsc(
    n::Union{Nothing,Int}=5000;
    seed=data_seed,
    train_test_split::Union{Nothing,Real}=nothing,
    shuffle::Bool=false,
    kwrgs...,
)

    # Setup:
    rng = get_rng(seed)
    ensure_positive(n)
    ensure_bounded(train_test_split)

    # Load data
    df_train, df_test, nfinal_train, nfinal_test, ntotal, nreq = pre_pre_process(
        "gmsc.csv", n; rng, shuffle, train_test_split
    )

    # Transformer:
    transformer = MLJModels.Standardizer(; count=true)

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
        kwrgs...,
    )

    return output
end
