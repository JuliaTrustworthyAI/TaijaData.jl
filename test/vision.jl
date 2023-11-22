@testset "Vision tests" begin
    # Test loading CIFAR10 dataset with default parameters
    @testset "cifar 10" begin
        raw_data = load_cifar_10_raw()
        @test size(raw_data[1])[2] == 50000
        @test size(raw_data[1])[1] == 3072
        @test size(raw_data[2])[1] == 50000
        # @test all(
        #     raw_data.domain[i] == (-1.0, 1.0) for
        #     i in eachindex(raw_data.domain)
        # )
        # @test raw_data.standardize == false

        # Test loading CIFAR10 dataset with subsampled data
        raw_data = load_cifar_10_raw(1000)
        @test size(raw_data[1])[2] == 1000
        @test size(raw_data[1])[1] == 3072
        @test size(raw_data[2])[1] == 1000
        # @test all(
        #     raw_data.domain[i] == (-1.0, 1.0) for
        #     i in eachindex(raw_data.domain)
        # )
        # @test raw_data.standardize == false

        # Test loading CIFAR10 test dataset
        raw_data = load_cifar_10_test_raw()
        @test size(raw_data[1])[2] == 10000
        @test size(raw_data[1])[1] == 3072
        @test size(raw_data[2])[1] == 10000
        # @test raw_data.standardize == false
    end

    @testset "fashion mnist" begin
        raw_data = load_fashion_mnist_raw()
        @test size(raw_data[1])[2] == 60000
        @test size(raw_data[1])[1] == 784
        @test size(raw_data[2])[1] == 60000
        # @test all(
        #     raw_data.domain[i] == (-1.0, 1.0) for
        #     i in eachindex(raw_data.domain)
        # )
        # @test raw_data.standardize == false
        # Test loading Fashion MNIST dataset with subsampled data
        raw_data = load_fashion_mnist_raw(1000)
        @test size(raw_data[1])[2] == 1000
        @test size(raw_data[1])[1] == 784
        @test size(raw_data[2])[1] == 1000
        # @test all(
        #     raw_data.domain[i] == (-1.0, 1.0) for
        #     i in eachindex(raw_data.domain)
        # )
        # @test raw_data.standardize == false
        # Test loading Fashion MNIST test dataset
        raw_data = load_fashion_mnist_test_raw()
        @test size(raw_data[1])[2] == 10000
        @test size(raw_data[1])[1] == 784
        @test size(raw_data[2])[1] == 10000
        # @test raw_data.standardize == false
    end

    @testset "mnist" begin
        raw_data = load_mnist_raw()
        @test size(raw_data[1])[2] == 60000
        @test size(raw_data[1])[1] == 784
        @test size(raw_data[2])[1] == 60000
        # @test all(
        #     raw_data.domain[i] == (-1.0, 1.0) for
        #     i in eachindex(raw_data.domain)
        # )
        # @test raw_data.standardize == false

        # Test loading MNIST dataset with subsampled data
        raw_data = load_mnist_raw(1000)
        @test size(raw_data[1])[2] == 1000
        @test size(raw_data[1])[1] == 784
        @test size(raw_data[2])[1] == 1000
        # @test all(
        #     raw_data.domain[i] == (-1.0, 1.0) for
        #     i in eachindex(raw_data.domain)
        # )
        # @test raw_data.standardize == false

        # Test loading MNIST test dataset
        raw_data = load_mnist_test_raw()
        @test size(raw_data[1])[2] == 10000
        @test size(raw_data[1])[1] == 784
        @test size(raw_data[2])[1] == 10000
        # @test raw_data.standardize == false
    end
end
