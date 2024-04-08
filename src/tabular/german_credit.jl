"""
    load_german_credit(n::Union{Nothing, Int}=nothing)

Loads UCI German Credit data.
"""
function load_german_credit(n::Union{Nothing,Int}=nothing)
    # Throw an exception if n > 1000:
    if !isnothing(n) && n > 1000
        throw(ArgumentError("n must be <= 1000"))
    end

    # Throw an exception if n < 1:
    if !isnothing(n) && n < 1
        throw(ArgumentError("n must be >= 1"))
    end

    # Load:
    df = CSV.read(joinpath(data_dir, "german_credit.csv"), DataFrames.DataFrame)

    # Pre-process features:
    transformer = MLJModels.Standardizer(; count=true)
    mach = MLJBase.fit!(MLJBase.machine(transformer, df[:, DataFrames.Not(:target)]))
    X = MLJBase.transform(mach, df[:, DataFrames.Not(:target)])
    X = Matrix(X)
    X = permutedims(X)

    # Counterfactual data:
    y = convert(Vector,df.target)

    # Undersample:
    if !isnothing(n)
        X, y = subsample(X, y, n)
    end

    return (X, y)
end
