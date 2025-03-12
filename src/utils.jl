using Random

"""
    get_rng(seed::Union{Int,AbstractRNG})

Returns a random number generator based on the provided seed, if seed is an integer, or returns the seed itself if it's already an `AbstractRNG`.
"""
function get_rng(seed::Union{Nothing,Int,AbstractRNG})
    rng = seed
    # Integer provided:
    if isa(seed, Int)
        rng = Xoshiro(seed)
    end
    # Nothing provided
    if isnothing(seed)
        rng = Random.default_rng()
    end
    return rng
end

function request_more_than_available(nreq, navailable)
    if !isnothing(nreq) && nreq > navailable
        @warn "Requested $nreq samples but only $navailable are available. Will resort to random oversampling."
    end
end

function ensure_positive(n::Union{Nothing,Int})
    if !isnothing(n) && n < 1
        throw(ArgumentError("`n` must be >= 1"))
    end
end

function subsample(rng::AbstractRNG, X::AbstractMatrix, y::AbstractVector, n::Int)
    # Get the unique classes in `y`.
    classes_ = unique(y)

    # Calculate the number of classes.
    n_classes = length(classes_)

    # Calculate the number of samples per class.
    n_per_class = Int(round(n / n_classes))

    # Find the indices of the samples for each class, sample `n_per_class` indices from them (with replacement), and concatenate the results for all classes. The resulting array of indices is then sorted.
    idx = sort(
        reduce(
            vcat,
            [
                StatsBase.sample(rng, findall(y .== cls), n_per_class; replace=true) for
                cls in classes_
            ],
        ),
    )

    # Subset `X` and `y` using the indices found above.
    X = X[:, idx]
    y = y[idx]

    return (X, y)
end

function subsample(
    rng::AbstractRNG,
    X::AbstractMatrix,
    y::AbstractVector,
    n::Union{Nothing,Int},
    nreq::Union{Nothing,Int},
    ntotal::Int,
)
    if isnothing(n)
        return X, y
    end
    if !isnothing(n) && nreq != ntotal
        X, y = subsample(rng, X, y, n)
    end
    return X, y
end

"""
    format_header!(df::DataFrame)

Helper function to apply some standard formatting to headers.
"""
function format_header!(df::DataFrame)
    return DataFrames.rename!(
        df,
        (
            x -> lowercase(x) |> x -> replace(x, " " => "_") |> x -> replace(x, "-" => "_")
        ).(names(df)),
    )
end

function get_categorical_indices(df::DataFrame, cats::Vector{String})
    return [findall((x -> contains(x, catvar)).(names(df))) for catvar in cats]
end

function ensure_bounded(x::Union{Nothing,Real})
    @assert isnothing(x) ? true : 0.0 <= x <= 1.0 "Value should be inside [0,1]."
end

function nfinal(n, ntotal, train_test_split)

    if !isnothing(n) && n != ntotal
        ntotal = n
    end

    if isnothing(train_test_split)
        return ntotal, 0
    else
        nfinal_train = Int(round(train_test_split * ntotal))
        return nfinal_train, ntotal - nfinal_train
    end

end

function apply_split(train_test_split, df)
    ntotal = size(df, 1)
    if isnothing(train_test_split)
        df_train = df
        ntrain = ntotal
        df_test = nothing
    else
        ntrain = Int(round(train_test_split * ntotal))
        df_train = df[1:ntrain, :]
        df_test = df[(ntrain + 1):end, :]
    end
    return df_train, df_test
end

function apply_transformations(df::DataFrame, mach)
    df_trans = MLJBase.transform(mach, df[:, DataFrames.Not(:target)])
    X = Matrix(df_trans)
    X = permutedims(X)
    y = df.target
    return X, y, df_trans
end

"""
    shuffle_rows([rng::AbstractRNG], df::DataFrame, shuffle::Bool)

Shuffle the rows of the DataFrame.
"""
function shuffle_rows(rng::AbstractRNG, df::DataFrame, shuffle::Bool)
    shuffle_warning(rng, shuffle)
    if shuffle
        row_idx = randperm(rng, size(df, 1))
        df = df[row_idx, :]
    end
    return df
end

shuffle_rows(df::DataFrame, shuffle::Bool) = shuffle_rows(Random.default_rng(), df, shuffle)

function shuffle_warning(rng::AbstractRNG, shuffle::Bool)
    if shuffle && rng != Random.default_rng()
        @warn "Rows will be shuffled using non-default RNG. Repeated calls will yield the same row order. If you're loading data, try setting `seed=nothing` nothing to use the default RNG."
    end
end
