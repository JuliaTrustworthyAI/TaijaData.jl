"""
    load_uci_adult(n::Union{Nothing,Int}=1000; seed=data_seed)

Loads data from the UCI 'Adult' dataset.
"""
function load_uci_adult(n::Union{Nothing,Int}=1000; seed=data_seed)
    # Throw an exception if n < 1:
    if !isnothing(n) && n < 1
        throw(ArgumentError("n must be >= 1"))
    end
    if !isnothing(n) && n > 32000
        throw(ArgumentError("n must not exceed size of dataset (<=32000)"))
    end

    # Load data
    df = CSV.read(joinpath(data_dir, "adult.csv"), DataFrames.DataFrame)
    DataFrames.rename!(
        df,
        [
            :age,
            :workclass,
            :fnlwgt,
            :education,
            :education_num,
            :marital_status,
            :occupation,
            :relationship,
            :race,
            :sex,
            :capital_gain,
            :capital_loss,
            :hours_per_week,
            :native_country,
            :target,
        ],
    )

    # Preprocessing
    transformer = MLJModels.Standardizer(; count=true)
    mach = MLJBase.fit!(machine(transformer, df[:, DataFrames.Not(:target)]))
    X = MLJBase.transform(mach, df[:, DataFrames.Not(:target)])
    X = Matrix(X)
    X = permutedims(X)

    y = df.target

    # Randomly under-/over-sample:
    rng = get_rng(seed)
    if !isnothing(n) && n != size(X)[2]
        X, y = subsample(rng, X, y, n)
    end

    return (X, y)
end
