using FFTW, FastGaussQuadrature

@testset "Smolyak Indexing" begin
    @testset "Total Order Formulae" begin
        dp_vec = [(3, 2), (2, 4), (4, 5), (5, 4), (10, 2)]
        for (d, p) in dp_vec
            mset = CreateTotalOrder(d, p)
            smolyak = smolyakIndexing(mset)
            correct_length = sum(sum(midx) >= p - d + 1 for midx in mset)
            @test length(smolyak) == correct_length
            correct_midxs = true
            correct_counts = true
            for (idx, count) in smolyak
                midx = mset[idx]
                correct_midxs &= sum(midx) >= p - d + 1
                correct_constant = binomial(d - 1, p - sum(midx)) * (-1)^(p - sum(midx))
                correct_counts &= count == correct_constant
            end
            @test correct_midxs
            @test correct_counts
        end
    end

    @testset "Non total order examples" begin
        mset_1 = [zeros(9) 1:9]
        mset_2 = [1:7 zeros(7)]
        mset_mat = Int[zeros(2) ones(2) mset_1' mset_2']
        mset = MultiIndexSet(mset_mat)
        correct_indices = [2, 3, 11, 12, 18]
        correct_counts = [1, -1, 1, -1, 1]
        smolyak = smolyakIndexing(mset)
        @test length(smolyak) == length(correct_indices)
        for (idx, count) in smolyak
            @test idx in correct_indices
            count_idx = findfirst(correct_indices .== idx)
            @test !isnothing(count_idx) && count == correct_counts[count_idx]
        end

        mset = create_example_hyperbolic2d(8)
        correct_midxs = [8 0; 1 4; 2 2; 4 1; 0 8; # First order additions
                         4 0; 2 1; 1 2; 0 4] # Correction terms
        correct_counts = [1, 1, 1, 1, 1,
            -1, -1, -1, -1]
        smolyak = smolyakIndexing(mset)
        @test length(smolyak) == size(correct_midxs, 1)
        correct_index_and_count = true
        for j in axes(correct_midxs, 1)
            midx = SVector{2}(correct_midxs[j, :])
            idx = findfirst(isequal(midx), mset.indices)
            smolyak_idx = findfirst(s -> s[1] == idx, smolyak)
            correct_index_and_count &= !isnothing(smolyak_idx) &&
                                       (smolyak[smolyak_idx][2] == correct_counts[j])
        end
        @test correct_index_and_count

        # Arbitrary example
        mset = create_example_curved(Xoshiro(80284), 5, 10, false, 0.8)
        @test isDownwardClosed(mset)
        smolyak = smolyakIndexing(mset)
        occurrences = zeros(Int, length(mset))
        for (idx, count) in smolyak
            occurrences[idx] += count
            ancestors = allBackwardAncestors(mset, idx)
            occurrences[ancestors] .+= count
        end
    end
end

function unifquad01(exactness::Int)
    N = ceil(Int, (exactness + 1)/2)
    x, w = gausslegendre(N)
    return (x.+1)/2, w/2
end

# Inspired by implementation in ChaosPy: https://github.com/jonathf/chaospy/blob/53000bbb04f8d3f9908ebbf1be6bf139a21c2e6e/chaospy/quadrature/clenshaw_curtis.py#L76
function clenshawcurtis01(order::Int)
    if order == 0
        return [0.5], [1.0]
    elseif order == 1
        return [0.0, 1.0], [0.5, 0.5]
    end

    theta = (order .- (0:order)) .* π / order
    abscissas = 0.5 .* cos.(theta) .+ 0.5

    steps = 1:2:order-1
    L = length(steps)
    remains = order - L

    beta = vcat(2.0 ./ (steps .* (steps .- 2)), [1.0 / steps[end]], zeros(remains))
    beta = -beta[1:end-1] .- reverse(beta[2:end])

    gamma = -ones(order)
    gamma[L+1] += order
    gamma[remains+1] += order
    gamma ./= order^2 - 1 + (order % 2)

    weights = rfft(beta + gamma)/order
    @assert maximum(imag.(weights)) < 1e-15
    weights = real.(weights)
    weights = vcat(weights, reverse(weights)[2 - (order % 2):end]) ./ 2

    return abscissas, weights
end

function monomialEval(midx::SVector{d}, x::AbstractMatrix) where {d}
    evals = ones(size(x, 2))
    for i in eachindex(midx)
        evals .*= (x[i, :] .^ midx[i])
    end
    evals
end

@testset "Smolyak Quadrature" begin

    @testset "Total Order Quadrature" begin
        dp_vec = [(3, 2), (2, 4), (4, 5), (5, 4), (10, 2)]
        for (d, p) in dp_vec
            mset = CreateTotalOrder(d, p)
            pts, wts = SmolyakQuadrature(mset, unifquad01)
            quad_int = sum(monomialEval(midx, pts) for midx in mset)' * wts
            exact_int = sum(prod(1 ./ (midx .+ 1)) for midx in mset)
            @test isapprox(quad_int, exact_int, atol=1e-10)
        end
    end

    @testset "Hyperbolic Quadrature" begin
        for j in 4:8

            mset = create_example_hyperbolic2d(2^j)
            prev_mset = create_example_hyperbolic2d(2^(j-1))
            # Gaussian quadrature
            pts, wts = SmolyakQuadrature(mset, unifquad01)
            quad_int = sum(monomialEval(midx, pts) for midx in mset)' * wts
            exact_int = sum(exp(sum(k->log(1/(k+1)), midx)) for midx in mset)
            @test isapprox(quad_int, exact_int, atol=1e-10)
            # Clenshaw-Curtis quadrature
            pts, wts = SmolyakQuadrature(mset, clenshawcurtis01)
            # For N points, exact on N-1 polynomials; therefore, not actually exact on mset
            quad_int = sum(monomialEval(midx, pts) for midx in mset)' * wts
            exact_int = sum(exp(sum(k->-log(k+1), midx)) for midx in mset)
            @test isapprox(quad_int, exact_int, atol=1e-10)
        end
    end
end