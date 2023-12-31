"""
    load_california_housing(n::Union{Nothing,Int}=5000)

Loads California Housing data.
"""
function load_california_housing(n::Union{Nothing,Int}=5000)

    # check that n is > 0
    if !isnothing(n) && n <= 0
        throw(ArgumentError("n must be > 0"))
    end

    # Load:
    df = CSV.read(joinpath(data_dir, "cal_housing.csv"), DataFrames.DataFrame)
    # Pre-process features:
    transformer = MLJModels.Standardizer(; count=true)
    mach = MLJBase.fit!(MLJBase.machine(transformer, df[:, DataFrames.Not(:target)]))
    X = MLJBase.transform(mach, df[:, DataFrames.Not(:target)])
    X = Matrix(X)
    X = permutedims(X)

    # Counterfactual data:
    y = Int.(df.target)

    # Undersample:
    if !isnothing(n)
        X, y = subsample(X, y, n)
    end
    
    return (X, y)
end
