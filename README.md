# TaijaData

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://JuliaTrustworthyAI.github.io/TaijaData.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://JuliaTrustworthyAI.github.io/TaijaData.jl/dev/)
[![Build Status](https://github.com/JuliaTrustworthyAI/TaijaData.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/JuliaTrustworthyAI/TaijaData.jl/actions/workflows/CI.yml?query=branch%3Amaster)
[![Coverage](https://codecov.io/gh/JuliaTrustworthyAI/TaijaData.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/JuliaTrustworthyAI/TaijaData.jl)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)

TaijaData is a package designed to simplify the process of fetching data for training and testing AI models.

## How-to

### Listing Available Functions/Datasets

To explore all the available functions and datasets, use:

```julia
data_catalogue
```

### Choosing a Dataset

Once you've reviewed the catalog, select a dataset, such as the `synthetic` dataset:

```julia
data_catalogue[:synthetic]
```

This will display a list of available synthetic datasets. For example, if you wish to import the blobs data:

```julia
x, y = load_blobs()
```

Here, `x` represents the input features matrix, and `y` contains the labels for each entry.

### Specifying the Number of Rows

If you want to fetch a specific number of rows, you can specify the desired count:

```julia
x, y = load_blobs(100)
```

This command fetches 100 rows of data from the blobs dataset.