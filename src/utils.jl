using Random

"""
    get_rng(seed::Union{Int,AbstractRNG})

Returns a random number generator based on the provided seed, if seed is an integer, or returns the seed itself if it's already an `AbstractRNG`.
"""
function get_rng(seed::Union{Int,AbstractRNG})
    if isa(seed, Int)
        seed = Xoshiro(seed)
    end
    return seed
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
