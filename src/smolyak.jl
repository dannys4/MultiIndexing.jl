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