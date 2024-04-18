@testset "isDownwardClosed" begin
    d, p = 5, 10
    mset = CreateTotalOrder(d, p)
    @test isDownwardClosed(mset)
    mset_2_mat = Int[zeros(5) 1:5]'
    mset_2 = MultiIndexSet(mset_2_mat)
    @test !isDownwardClosed(mset_2)
    mset_3_mat = Int[0:5 zeros(6)]'
    mset_3 = MultiIndexSet(mset_3_mat)
    @test isDownwardClosed(mset_3)
    mset_4_mat = vcat(mset_3_mat', Int[zeros(6) 1:6])'
    mset_4 = MultiIndexSet(mset_4_mat)
    @test isDownwardClosed(mset_4)
    mset_5_mat = hcat(mset_3_mat, Int[3, 3])'
    mset_5 = MultiIndexSet(mset_5_mat)
    @test !isDownwardClosed(mset_5)
end

@testset "Backward Ancestors" begin
    d, p = 5, 10
    mset = CreateTotalOrder(d, p)
    idx = SVector{d}(fill(p ÷ d, d))
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

@testset "Subset Completion" begin
    mset = CreateTotalOrder(5, 10)
    subset = 1:(length(mset) ÷ 5):length(mset)
    mset2 = MultiIndexSet(mset[subset])
    completion_indices = MultiIndexing.subsetCompletion(mset, subset)
    completion = MultiIndexSet(mset[completion_indices], mset.limit)
    @test !isDownwardClosed(mset2) # Not necessary in general, but required for the test
    @test isDownwardClosed(completion)
    @test length(completion) > length(mset2)
    @test all(c in mset.indices for c in completion)
    @test all(c in completion for c in mset2.indices)
end