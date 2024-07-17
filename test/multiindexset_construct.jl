@testset "Basic Total Order" begin
    d, p = 2, 3
    hardcoded_mset = [0 1 0 2 1 0 3 2 1 0; 0 0 1 0 1 2 0 1 2 3]
    harcoded_last_start = 7

    # Test matrix creation
    mset, last_start = MultiIndexing.CreateTotalOrder_matrix(d, p)
    @test all(sum(mset[:, last_start:end], dims = 1) .== p) # Last start is wrong
    @test all(sum(mset[:, 1:(last_start - 1)], dims = 1) .< p) # First part is wrong
    @test mset == hardcoded_mset # Matrix is wrong
    @test last_start == harcoded_last_start # Last start is wrong

    # Test basic set creation
    d, p = 5, 4
    # Test Tensor order
    mset = MultiIndexing.CreateTensorOrder(d, p)
    mis_matrix = reduce(hcat, mset.indices)
    @test d^(p+1) == size(mis_matrix,2)
    @test all(mis_matrix .<= p)

    # Create a slightly larger set to capture reduced margin
    mset, last_start = MultiIndexing.CreateTotalOrder_matrix(d, p + 1)
    mis = MultiIndexing.CreateTotalOrder(d, p)
    mis_matrix = reduce(hcat, mis.indices)
    @test mset[:, 1:(last_start - 1)] == mis_matrix # Indices are wrong
    @test mis.isDownwardClosed # Set is not downward closed
    @test mis.limit == MultiIndexing.NoLimiter # Set has a limiter
    @test collect(mis.maxDegrees) == fill(p, d) # Max degrees are wrong
    rm_matrix = reduce(hcat, mis.reduced_margin)
    @test rm_matrix == mset[:, last_start:end] # Reduced margin is wrong
end

@testset "Different Limiters" begin
    d, p = 5, 10

    # Test Empty limiter Set creation
    empty_limiter = Returns(false)
    @test_logs (:warn, "No valid reduced margin found on frontier") (mis=CreateTotalOrder(
        d, p, empty_limiter))
    @test length(mis.indices) == 0 # Set is nonempty
    @test mis.isDownwardClosed # Set is not downward closed
    @test mis.limit == empty_limiter # Set has a limiter
    @test collect(mis.maxDegrees) == zeros(Int, d) # Max degrees are wrong
    @test length(mis.reduced_margin) == 0 # Reduced margin is nonempty

    # Test Anisotropic curved limiter Set creation
    rng = Xoshiro(80284)
    mis = create_example_curved(rng, d, p)
    @test mis.limit isa MultiIndexing.CurvedLimiter # Set has a limiter
    @test mis.isDownwardClosed # Set is downward closed
    @test all(collect(mis.maxDegrees) .<= fill(p, d)) # Max degrees at most p
    @test mis.maxDegrees[1] == p # Max degree of first index is p
    @test all(collect(mis.maxDegrees) .>= 0) # Max degrees at least 0
    @test all(mis.limit(idx, p) for idx in mis.indices) # All indices pass limiter
    @test all(mis.limit(idx, p + 1) for idx in mis.reduced_margin) # All reduced margin pass limiter
    @test all(!in(idx, mis.indices) for idx in mis.reduced_margin) # Reduced margin not in indices
    @test all(!in(idx, mis.reduced_margin) for idx in mis.indices) # Indices not in reduced margin
end