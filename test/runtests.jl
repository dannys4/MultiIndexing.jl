using MultiIndexing
using Test, StaticArrays, Random

function create_example_curved(rng, d, p)
    ani_weights = 0.5 .+ rand(rng, d)
    ani_weights[1] = 0.
    ani_weights = SVector{d}(ani_weights * (d-1)/sum(ani_weights))
    aniso_limiter = MultiIndexing.AnisotropicLimiter(ani_weights)
    curved_weights = 0.5 .+ rand(rng, d)
    curved_weights[1] = 0.
    curved_weights = SVector{d}(curved_weights * (d-1)/sum(curved_weights))
    curved_limiter = MultiIndexing.CurvedLimiter(curved_weights, aniso_limiter)
    mis = MultiIndexing.CreateTotalOrder(d, p, curved_limiter)
    mis
end

@testset "MultiIndexing.jl" begin
    @testset "CreateTotalOrder" begin
        @testset "Basic Total Order" begin
            d,p = 2, 3
            hardcoded_mset = [0 1 0 2 1 0 3 2 1 0; 0 0 1 0 1 2 0 1 2 3]
            harcoded_last_start = 7

            # Test matrix creation
            mset, last_start = MultiIndexing.CreateTotalOrder_matrix(d, p)
            @test all(sum(mset[:,last_start:end], dims=1) .== p) # Last start is wrong
            @test all(sum(mset[:,1:last_start-1], dims=1) .< p) # First part is wrong
            @test mset == hardcoded_mset # Matrix is wrong
            @test last_start == harcoded_last_start # Last start is wrong

            # Test basic set creation
            d, p = 5, 4
            # Create a slightly larger set to capture reduced margin
            mset, last_start = MultiIndexing.CreateTotalOrder_matrix(d, p+1)
            mis = MultiIndexing.CreateTotalOrder(d, p)
            mis_matrix = reduce(hcat, mis.indices)
            @test mset[:,1:last_start-1] == mis_matrix # Indices are wrong
            @test mis.isDownwardClosed # Set is not downward closed
            @test mis.limit == MultiIndexing.NoLimiter # Set has a limiter
            @test collect(mis.maxDegrees) == fill(p,d) # Max degrees are wrong
            rm_matrix = reduce(hcat, mis.reduced_margin)
            @test rm_matrix == mset[:,last_start:end] # Reduced margin is wrong
        end
        @testset "Different Limiters" begin
            d, p = 5, 10

            # Test Empty limiter Set creation
            empty_limiter = Returns(false)
            @test_logs (:warn, "No valid reduced margin found on frontier") (mis = CreateTotalOrder(d, p, empty_limiter))
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
            @test all(collect(mis.maxDegrees) .<= fill(p,d)) # Max degrees at most p
            @test mis.maxDegrees[1] == p # Max degree of first index is p
            @test all(collect(mis.maxDegrees) .>= 0) # Max degrees at least 0
            @test all(mis.limit(idx, p) for idx in mis.indices) # All indices pass limiter
            @test all(mis.limit(idx, p+1) for idx in mis.reduced_margin) # All reduced margin pass limiter
            @test all(!in(idx, mis.indices) for idx in mis.reduced_margin) # Reduced margin not in indices
            @test all(!in(idx, mis.reduced_margin) for idx in mis.indices) # Indices not in reduced margin
        end
    end

    @testset "Backward Ancestors" begin
        d, p = 5, 10
        mset = CreateTotalOrder(d, p)
        idx = SVector{d}(fill(p÷d,d))
        j = findfirst(isequal(idx), mset.indices)
        @test !isnothing(j) # Index must be in set
        @test mset[j] == idx # Index must be at j
        ancestors = allBackwardAncestors(mset, j)
        @test length(ancestors) == prod(idx .+ 1) - 1 # Must have d*p ancestors
        @test all(ancestors .< length(mset.indices)) # All ancestors must be in set
        @test all(all(mset[i] .<= idx) for i in ancestors) # All ancestors must be in tensor product box
        @test !in(idx, ancestors) # Target index must not be in ancestors
        exclusion = collect(1:length(mset.indices))
        deleteat!(exclusion, sort([ancestors; j]))
        @test all(any(mset[i] .> idx) for i in exclusion) # All non-ancestors must be outside tensor product box
    end

    @testset "Reduced Frontier" begin
        rng, d, p = Xoshiro(820482), 5, 10
        mis = create_example_curved(rng, d, p)
        frontier = findReducedFrontier(mis)
        @test all(mis.limit(mis.indices[i], p) for i in frontier) # All frontier indices pass limiter
        @test all(frontier isa Vector{Int}) # Frontier is a vector of integers
        @test all(frontier .<= length(mis.indices)) # All frontier indices are in the set
        @test all(frontier .> 0) # All frontier indices are positive
        check_valid = true
        all_ancestors = Set{Int}(frontier)
        @test length(all_ancestors) == length(frontier) # All indices are unique
        for j in eachindex(mis.indices)
            back_neighbors = allBackwardAncestors(mis, j)
            if j in frontier
                for b in back_neighbors
                    push!(all_ancestors, b)
                end
                continue
            end
            # Check that the reduced frontier is made of indices who are not the backward
            # neighbors of any other index
            check_valid &= !any(in(f, back_neighbors) for f in frontier)
        end
        @test check_valid
        @test length(all_ancestors) == length(mis.indices) # All indices are a backward ancestor of at least one frontier index
    end
end
