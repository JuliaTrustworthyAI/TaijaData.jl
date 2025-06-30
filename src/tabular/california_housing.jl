struct CaliHousing <: TabularData end

get_feature_names(data::CaliHousing) = get_feature_names("cal_housing.csv")

"""
    load_california_housing(
        n::Union{Nothing,Int}=5000;
        seed=data_seed,
        train_test_split::Union{Nothing,Real}=nothing,
        shuffle::Bool=false,
    )

Loads California Housing data.
"""
function load_california_housing(
    n::Union{Nothing,Int}=5000;
    seed=data_seed,
    train_test_split::Union{Nothing,Real}=nothing,
    shuffle::Bool=false,
)

    # Setup:
    rng = get_rng(seed)
    ensure_positive(n)
    ensure_bounded(train_test_split)

    # Load data
    df_train, df_test, nfinal_train, nfinal_test, ntotal, nreq = pre_pre_process(
        "cal_housing.csv", n; rng, shuffle, train_test_split
    )

    # Transformer:
    transformer = MLJModels.Standardizer(; count=true)

    # Pre-process:
    output = pre_process(
        transformer, df_train, df_test; rng, nfinal_train, nfinal_test, ntotal, nreq
    )

    return output
end
