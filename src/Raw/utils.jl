function subsample(X::AbstractMatrix, y::AbstractVector, n::Int)
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
                StatsBase.sample(findall(y .== cls), n_per_class; replace=true) for
                cls in classes_
            ],
        ),
    )

    # Subset `X` and `y` using the indices found above.
    X = X[:, idx]
    y = y[idx]

    return (X, y)

return (X, y)
end
