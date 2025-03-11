"""
    load_german_credit(
        n::Union{Nothing,Int}=nothing;
        seed=data_seed,
        train_test_split::Union{Nothing,Real}=nothing,
        shuffle::Bool=false,
    )

Loads UCI German Credit data.
"""
function load_german_credit(
    n::Union{Nothing,Int}=nothing;
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
        "german_credit.csv", n; rng, shuffle, train_test_split
    )

    # Transformer:
    transformer = MLJModels.Standardizer(; count=true)

    # Pre-process:
    output = pre_process(
        transformer, df_train, df_test; rng, nfinal_train, nfinal_test, ntotal, nreq
    )

    return output
end
