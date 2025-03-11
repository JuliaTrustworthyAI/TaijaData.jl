using TaijaData

@testset "Tabular tests" begin
    @testset "load_california_housing tests" begin
        @test_throws ArgumentError load_california_housing(-1)  # n must be a positive integer or Nothing

        @testset "Check output types and sizes" begin
            result = load_california_housing(100)

            @test isa(result, Tuple{Matrix,Vector})

            @test size(result[1], 2) == 100  # there should be 100 observations

            # Check that the dimensions of X and y match:
            @test size(result[1], 2) == size(result[2], 1)
        end

        @testset "Check data consistency" begin
            # example row:
            # MedInc,HouseAge,AveRooms,AveBedrms,Population,AveOccup,Latitude,Longitude,target
            # 3.4698,31.0,5.390243902439025,1.1986062717770034,956.0,3.3310104529616726,33.9,-118.35,1.0
            # we use a hardcoded value for the expected dimensions because the # dataset is very stable and this will greatly speed
            # things up. However, it is still something to be aware of.
            X_dim_expected = 8  # we expect one less column in X, as the target column is not included
            result = load_california_housing(100)

            # Check that the dimension of X is correct:
            @test size(result[1], 1) == X_dim_expected
        end
    end

    @testset "German credit statlog dataset" begin
        # Test loading german_credit dataset with default parameters
        data = load_german_credit()
        @test size(data[1])[2] == 1000
        @test size(data[1])[1] == 20
        @test size(data[2])[1] == 1000

        # Test loading german_credit dataset with subsampled data
        data = load_german_credit(500)
        @test size(data[1])[2] == 500
        @test size(data[1])[1] == 20
        @test size(data[2])[1] == 500

        # Test case: Load data with n < 1, expecting an error
        @test_throws ArgumentError load_german_credit(0)
        @test_throws ArgumentError load_german_credit(-100)
    end

    @testset "UCI Adult dataset" begin
        data = load_uci_adult()
        @test size(data[1])[2] == 1000
        @test size(data[1])[1] == 108
        @test size(data[2])[1] == 1000

        data = load_uci_adult(500)
        @test size(data[1])[2] == 500
        @test size(data[1])[1] == 108
        @test size(data[2])[1] == 500

        @test_throws ArgumentError load_uci_adult(0)
        @test_throws ArgumentError load_uci_adult(-1)

        data, cats = load_uci_adult(; return_cats=true)
        @test true
    end

    @testset "Credit Default" begin
        data, cats = load_uci_adult(; return_cats=true)
        @test true
    end
end
