ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

@testset "Construction" begin
    for (name, loader) in merge(values(data_catalogue)...)
        @testset "$name" begin
            @test typeof(loader()) == CounterfactualExplanations.CounterfactualData
        end
    end
end

@testset "Data loading tests" begin
    @testset "load_tabular_data tests" begin
        # Test loading all tabular datasets
        data = TaijaData.load_tabular_data()
        @test typeof(data) == Dict{Symbol, CounterfactualExplanations.CounterfactualData}
        @test all(k in keys(data) for k in keys(TaijaData.data_catalogue[:tabular]))

        # Test dropping a specific dataset
        data = TaijaData.load_tabular_data(; drop=:california_housing)
        @test !haskey(data, :california_housing)

        # Test loading with specified number of samples
        data = TaijaData.load_tabular_data(1000)
        for dataset in values(data)
            @test size(dataset.X)[2] == 1000  # replace ... with actual size of your data
        end
    end

    @testset "load_synthetic_data tests" begin
        # Test dropping a specific dataset
        data = TaijaData.load_synthetic_data(; drop=:linearly_separable)
        @test !haskey(data, :linearly_separable)

        # Test loading with specified number of samples
        data = TaijaData.load_synthetic_data(500)
        for dataset in values(data)
            @test size(dataset.X)[2] == 500  # replace ... with actual size of your data
        end
    end
end
