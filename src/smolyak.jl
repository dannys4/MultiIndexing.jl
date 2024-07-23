export smolyakIndexing, SmolyakQuadrature

"""
    smolyakIndexing(mset) -> Vector{Tuple{Int,Int}}

Gets the smolyak indexing for a multi-index set.
Returns a vector of tuples with each first index as the index
of the multi-index representing a tensor product rule and each
second index representing its count in the smolyak construction.

# Examples
```jldoctest
julia> d = 2;

julia> mis = MultiIndexSet([(0:7)'; zeros(Int,1,8)]);

julia> quad_rules = smolyakIndexing(mis); # Creates quad rule exact on x^7

julia> quad_rules[1], length(quad_rules) # Counts the highest index once
((8, 1), 1)

julia> mis = CreateTotalOrder(d, 10);

julia> print(visualize_smolyak_2d(mis, false))
X
X X
o X X
o o X X
o o o X X
o o o o X X
o o o o o X X
o o o o o o X X
o o o o o o o X X
o o o o o o o o X X
o o o o o o o o o X X
```
"""
function smolyakIndexing(mset::MultiIndexSet{d, T}) where {d, T}
    mset_full = mset # Alias to clarify in code what's going on
    N = length(mset_full)
    quad_rules = []
    occurrences = zeros(Int, N)
    loop_indices = collect(1:N)
    while sum(loop_indices) > 0
        # Form the mset for this loop
        loop_indices_subset = (1:N)[loop_indices]
        loop_indices_full_map = subsetCompletion(mset_full, loop_indices_subset)
        mset_loop = MultiIndexSet(mset_full[loop_indices_full_map], mset_full.limit)
        frontier_loop = findReducedFrontier(mset_loop)

        # Adjust each frontier member and reindex their ancestors' number of occurrences
        for idx_midx_loop in frontier_loop
            # Access the adjustment for this frontier member
            idx_midx_full = loop_indices_full_map[idx_midx_loop]
            j = 1 - occurrences[idx_midx_full] # Enforce o[i] + j = 1
            occurrences[idx_midx_full] = 1

            # Reindex the number of occurrences of the backward ancestors
            ancestors = allBackwardAncestors(mset_loop, idx_midx_loop)
            for idx_back_loop in ancestors
                idx_back_full = loop_indices_full_map[idx_back_loop]
                occurrences[idx_back_full] += j
            end
            push!(quad_rules, (idx_midx_full, j))
        end
        loop_indices = occurrences .!= 1
    end
    quad_rules
end

function tensor_prod_quad(pts_wts_zipped,::Val{d}) where {d}
    points_1d = [p for (p,_) in pts_wts_zipped]
    log_wts_1d = [[(sign(w),log(abs(w))) for w in wts] for (_,wts) in pts_wts_zipped]
    # Create all indices for the tensor product rule
    lengths_1d = ntuple(k->length(points_1d[k]), d)
    idxs = CartesianIndices(lengths_1d)
    points = Vector{SVector{d, Float64}}(undef, length(idxs))
    weights = zeros(Float64, length(idxs))
    @inbounds for (j, idx) in enumerate(idxs)
        points[j] = SVector{d}(ntuple(k->points_1d[k][idx[k]], d))
        weights[j] = exp(sum(log_wts_1d[k][idx[k]][2] for k in 1:d))*prod(log_wts_1d[k][idx[k]][1] for k in 1:d)
    end
    points, weights
end

function tensor_prod_quad(midx::SVector{d, Int}, rules::Union{<:AbstractVector,<:Tuple}) where {d}
    rules_eval = ntuple(i->rules[i](midx[i]), d)
    tensor_prod_quad(rules_eval, Val{d}())
end

"""
    SmolyakQuadrature(mset, rules)

Create a Smolyak quadrature rule from a multi-index set and a set of rules

# Arguments
- `mset`: MultiIndexSet
- `rules`: Vector of rules for each dimension. `rules[j](n::Int)` should return a quadrature rule `(pts,wts)` for dimension `j` exact up to order `n`

"""
function SmolyakQuadrature(mset::MultiIndexSet{d}, rules::Union{<:AbstractVector,<:Tuple}) where {d}
    if length(rules) != d
        throw(ArgumentError("Number of rules must match dimension"))
    end
    quad_rules = smolyakIndexing(mset)
    unique_elems = Dict{SVector{d, Float64}, Float64}()
    for (idx, count) in quad_rules
        midx = mset[idx]
        pts_idx, wts_idx = tensor_prod_quad(midx, rules)
        for (pt, wt) in zip(pts_idx, wts_idx)
            entry = get(unique_elems, pt, 0.0)
            unique_elems[pt] = entry + wt * count
        end
    end
    points = Matrix{Float64}(undef, d, length(unique_elems))
    weights = Vector{Float64}(undef, length(unique_elems))
    for (i, (pt, wt)) in enumerate(unique_elems)
        points[:, i] .= pt
        weights[i] = wt
    end
    points, weights
end

function SmolyakQuadrature(mset::MultiIndexSet{d}, rule::Function) where {d}
    SmolyakQuadrature(mset, ntuple(_->rule,d))
end
