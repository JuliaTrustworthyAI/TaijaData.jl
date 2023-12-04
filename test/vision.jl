@testset "Vision tests" begin
    # Test loading CIFAR10 dataset with default parameters
    @testset "cifar 10" begin
        data = load_cifar_10()
        @test size(data[1])[2] == 50000
        @test size(data[1])[1] == 3072
        @test size(data[2])[1] == 50000
        # @test all(
        #     data.domain[i] == (-1.0, 1.0) for
        #     i in eachindex(data.domain)
        # )
        # @test data.standardize == false

        # Test loading CIFAR10 dataset with subsampled data
        data = load_cifar_10(1000)
        @test size(data[1])[2] == 1000
        @test size(data[1])[1] == 3072
        @test size(data[2])[1] == 1000
        # @test all(
        #     data.domain[i] == (-1.0, 1.0) for
        #     i in eachindex(data.domain)
        # )
        # @test data.standardize == false

        # Test loading CIFAR10 test dataset
        data = load_cifar_10_test()
        @test size(data[1])[2] == 10000
        @test size(data[1])[1] == 3072
        @test size(data[2])[1] == 10000
        # @test data.standardize == false
    end

    @testset "fashion mnist" begin
        data = load_fashion_mnist()
        @test size(data[1])[2] == 60000
        @test size(data[1])[1] == 784
        @test size(data[2])[1] == 60000
        # @test all(
        #     data.domain[i] == (-1.0, 1.0) for
        #     i in eachindex(data.domain)
        # )
        # @test data.standardize == false
        # Test loading Fashion MNIST dataset with subsampled data
        data = load_fashion_mnist(1000)
        @test size(data[1])[2] == 1000
        @test size(data[1])[1] == 784
        @test size(data[2])[1] == 1000
        # @test all(
        #     data.domain[i] == (-1.0, 1.0) for
        #     i in eachindex(data.domain)
        # )
        # @test data.standardize == false
        # Test loading Fashion MNIST test dataset
        data = load_fashion_mnist_test()
        @test size(data[1])[2] == 10000
        @test size(data[1])[1] == 784
        @test size(data[2])[1] == 10000
        # @test data.standardize == false
    end

    @testset "mnist" begin
        data = load_mnist()
        @test size(data[1])[2] == 60000
        @test size(data[1])[1] == 784
        @test size(data[2])[1] == 60000
        # @test all(
        #     data.domain[i] == (-1.0, 1.0) for
        #     i in eachindex(data.domain)
        # )
        # @test data.standardize == false

        # Test loading MNIST dataset with subsampled data
        data = load_mnist(1000)
        @test size(data[1])[2] == 1000
        @test size(data[1])[1] == 784
        @test size(data[2])[1] == 1000
        # @test all(
        #     data.domain[i] == (-1.0, 1.0) for
        #     i in eachindex(data.domain)
        # )
        # @test data.standardize == false

        # Test loading MNIST test dataset
        data = load_mnist_test()
        @test size(data[1])[2] == 10000
        @test size(data[1])[1] == 784
        @test size(data[2])[1] == 10000
        # @test data.standardize == false
    end
end
