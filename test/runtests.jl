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

function create_example_hyperbolic2d(p)
    log2p = floor(Int, log2(p))
    pd2 = p ÷ 2
    midx_rep = reduce(hcat, [MultiIndexing.tens_prod_mat(@SVector[p_1, pd2 ÷ p_1]) for p_1 in 2 .^ (0:log2p-1)])
    midx_ext = Int[midx_rep [pd2+1:p zeros(pd2)]' [zeros(pd2) pd2+1:p]']
    MultiIndexing.MultiIndexSet(unique(midx_ext, dims=2))
end

function create_example_hyperbolic3d(p)
    log2p = floor(Int, log2(p))
    pd2 = p ÷ 2
    mset_rep = []
    for logp_1 in 0:log2p-1
        for logp_2 in 0:(log2p-1-logp_1)
            p_1 = 2^logp_1
            p_2 = 2^logp_2
            p_3 = 2^(log2p - (logp_1 + logp_2 + 1))
            push!(mset_rep, MultiIndexing.tens_prod_mat(@SVector[p_1, p_2, p_3]))
        end
    end
    mset_rep = reduce(hcat, mset_rep)
    mset_ext = Int[mset_rep [pd2+1:p zeros(pd2) zeros(pd2)]' [zeros(pd2) pd2+1:p zeros(pd2)]' [zeros(pd2) zeros(pd2) pd2+1:p]']
    MultiIndexing.MultiIndexSet(unique(mset_ext, dims=2))
end

function viz_2d(mset_mat, markers = 'X')
    if mset_mat isa MultiIndexing.MultiIndexSet
        mset_mat = reduce(hcat, mset_mat.indices)
    end
    @assert size(mset_mat, 1) == 2
    chars = fill(' ', (maximum(mset_mat,dims=2) .+ 1)...)
    for j in axes(mset_mat,2)
        mark = markers isa Char ? markers : markers[j]
        chars[end - mset_mat[1,j], mset_mat[2,j]+1] = mark
    end
    rows = [join(c, ' ') for c in eachrow(chars)]
    join(rows, "\n")
end

function rgb_char(r, g, b, char)
    "\e[1m\e[38;2;$r;$g;$b;249m$char\e[0m"
end

function viz_smolyak_2d(mset::MultiIndexing.MultiIndexSet)
    smolyak_indices = MultiIndexing.smolyakIndexing(mset, true)
    rgb1, rgb2 = [100, 100, 0], [0, 100, 100]
    mset_max = [maximum(j[1] for j in mset.indices), maximum(j[2] for j in mset.indices)]
    chars = fill(" ", (mset_max .+ 1)...)
    for (j,level) in enumerate(smolyak_indices)
        col = rgb1 * (j-1) + rgb2 * (length(smolyak_indices) - j)
        for idx in level
            m_idx = mset.indices[idx]
            chars[(m_idx .+ 1)...] = rgb_char(col..., 'X')
            back_indices = MultiIndexing.allBackwardAncestors(mset, idx)
            for bidx in back_indices
                m_idx_b = mset.indices[bidx]
                chars[(m_idx_b .+1)...] = rgb_char(col..., 'o')
            end
        end
    end
    join([join(c, ' ') for c in eachrow(chars)][end:-1:1], "\n")
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
            @test_logs (:warn, "No valid reduced margin found on frontier") (mis = MultiIndexing.CreateTotalOrder(d, p, empty_limiter))
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
        mset = MultiIndexing.CreateTotalOrder(d, p)
        idx = SVector{d}(fill(p÷d,d))
        j = findfirst(isequal(idx), mset.indices)
        @test !isnothing(j) # Index must be in set
        @test mset[j] == idx # Index must be at j
        ancestors = MultiIndexing.allBackwardAncestors(mset, j)
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
        frontier = MultiIndexing.findReducedFrontier(mis)
        @test all(mis.limit(mis.indices[i], p) for i in frontier) # All frontier indices pass limiter
        @test all(frontier isa Vector{Int}) # Frontier is a vector of integers
        @test all(frontier .<= length(mis.indices)) # All frontier indices are in the set
        @test all(frontier .> 0) # All frontier indices are positive
        check_valid = true
        all_ancestors = Set{Int}(frontier)
        @test length(all_ancestors) == length(frontier) # All indices are unique
        for j in eachindex(mis.indices)
            back_neighbors = MultiIndexing.allBackwardAncestors(mis, j)
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
