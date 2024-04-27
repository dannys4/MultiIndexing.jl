export MultiIndexSet
export allBackwardAncestors, findReducedFrontier
export isDownwardClosed, subsetCompletion

using StaticArrays, LinearAlgebra
import Base: getindex, push!, length, iterate

"""
Represents a set of multi-indices
vector of {d} dimensional multi-indices with length N
"""
struct MultiIndexSet{d, T}
    indices::Vector{SVector{d, Int}}
    reduced_margin::Vector{StaticVector{d, Int}}
    limit::T
    isDownwardClosed::Bool
    maxDegrees::MVector{d, Int}
end

function MultiIndexSet(indices::Vector{SVector{d, Int}}, limit::T = NoLimiter,
        calc_reduced_margin = false) where {d, T}
    reduced_margin = []
    if calc_reduced_margin
        throw(ArgumentError("Reduced margin calculation not implemented"))
    end
    maxDegrees = MVector{d}(reduce((x, y) -> max.(x, y), indices, init = zeros(Int, d)))
    MultiIndexSet{d, T}(indices, reduced_margin, limit, true, maxDegrees)
end

"""
    MultiIndexSet(indices, limit, calc_reduced_margin)

Create a multi-index set from a matrix of multi-indices

Arguments:
- `indices`: dim x N matrix of multi-indices
- `limit`: function that takes a multi-index and a limit and returns true if the index is admissible
- `calc_reduced_margin`: if true, calculate the reduced margin of the set, otherwise leave it empty
"""
function MultiIndexSet(indices_mat::AbstractMatrix{Int}, limit::T = NoLimiter,
        calc_reduced_margin = false) where {T}
    d = size(indices_mat, 1)
    indices = [SVector{d}(indices_mat[:, i]) for i in axes(indices_mat, 2)]
    MultiIndexSet(indices, limit, calc_reduced_margin)
end

# Some helper functions
function Base.getindex(mis::MultiIndexSet{d}, i) where {d}
    mis.indices[i]
end

function Base.length(mis::MultiIndexSet{d}) where {d}
    length(mis.indices)
end

function Base.iterate(mis::MultiIndexSet{d},
        state::Int = 1)::Union{Tuple{SVector{d, Int}, Int}, Nothing} where {d}
    return state > length(mis.indices) ? nothing : (mis.indices[state], state + 1)
end

function Base.vec(mis::MultiIndexSet{d}) where {d}
    collect.(mis.indices)
end

# Searches backward through the mset backward for an index
function in_mset_backward(mis::MultiIndexSet{d}, idx::StaticVector{d, Int}) where {d}
    any(idx .> mis.maxDegrees) && return false
    for i in length(mis.indices):-1:1
        mis.indices[i] == idx && return true
    end
    false
end

"""
    checkIndexAdmissible(mis, idx, check_indices)

Check if an index is admissible to add to a reduced margin
"""
function checkIndexAdmissible(mis::MultiIndexSet{d}, idx::StaticVector{d, Int},
        check_indices::Bool = false) where {d}
    # If any index is greater than the limit, return false
    mis.limit(idx) && return false
    # If more than one subindex is greater than its max degree, return false
    sum(idx .>= mis.maxDegrees .+ 1) > 1 && return false
    # Check if the index is in the reduced margin
    idx in mis.reduced_margin && return false
    if check_indices # Check if the index is in the indices (expensive)
        idx in mis.indices && return false
    end
    # Otherwise, check that all backward neighbors of the index is in the set
    idx_tmp = Vector(idx)
    for i in 1:d
        idx[i] == 0 && continue
        idx_tmp[i] -= 1
        !(SVector{d}(idx_tmp) in mis.indices) && return false
        idx_tmp[i] += 1
    end
    true
end

function isDownwardClosed(mis::MultiIndexSet{d}) where {d}
    for idx in mis.indices
        tmp_idx = Vector(idx)
        for i in 1:d
            tmp_idx[i] == 0 && continue
            tmp_idx[i] -= 1
            if !(SVector{d}(tmp_idx) in mis.indices)
                return false
            end
            tmp_idx[i] += 1
        end
    end
    true
end

"""
    allBackwardAncestors(mis, j)

Get all indices of multi-indices in mset limited by the tensor product box of the mset at idx

Arguments:
- mis: MultiIndexSet to get the backward ancestors from
- j: index of the multi-index to get the backward ancestors of

```jldoctest
julia> d, p = 2, 5;

julia> mis = CreateTotalOrder(d, p);

julia> println(visualize_2d(mis))
X
X X
X X X
X X X X
X X X X X
X X X X X X

julia> midx = [2,2]; idx = findfirst(isequal(midx), vec(mis.indices));

julia> ancestors_idx = allBackwardAncestors(mis, idx);

julia> ancestors = MultiIndexSet(mis[ancestors_idx]);

julia> println(visualize_2d(ancestors))
X X
X X X
X X X

```
"""
function allBackwardAncestors(mis::MultiIndexSet{d}, j::Int) where {d}
    idx = mis.indices[j]
    back_indices = CartesianIndices(ntuple(k -> 0:idx[k], d))
    j_ancestors = Vector{Int}(undef, length(back_indices) - 1)
    for (i, back_index) in enumerate(back_indices)
        if i < length(back_indices)
            check_back_index = isequal(SVector(Tuple(back_index)))
            ancestor_i = findfirst(check_back_index, mis.indices)
            if isnothing(ancestor_i)
                @info "No back index:" back_index idx
            end
            j_ancestors[i] = ancestor_i
        end
    end
    j_ancestors
end

# Update the max degrees of a set after adding an index
function updateMaxDegrees!(mis::MultiIndexSet{d}, idx::StaticVector{d, Int}) where {d}
    mis.maxDegrees .= max.(mis.maxDegrees, idx)
end

# Update the reduced margin of a set after adding an index
function updateReducedMargin!(mis::MultiIndexSet{d}, idx::StaticVector{d, Int}) where {d}
    j = findfirst(isequal(idx), mis.reduced_margin)
    # delete idx from reduced margin
    deleteat!(mis.reduced_margin, j)
    # for each neighbor of idx, check valid and add to reduced margin
    idx_tmp = Vector(idx)
    for i in 1:d
        idx_tmp[i] += 1
        tmp_i = SVector{d}(idx_tmp)
        if checkIndexValid(mis, tmp_i, false)
            push!(mis.reduced_margin, tmp_i)
        end
        idx_tmp[i] -= 1
    end
end

function Base.push!(mis::MultiIndexSet{d}, idx::StaticVector{d, Int}) where {d}
    (idx in mis.indices) && return true
    isValid = checkIndexValid(mis, idx)
    mis.isDownwardClosed = isValid
    push!(mis.indices, idx)
    updateMaxDegrees!(mis, idx)
    # Update reduced margin if downward closed
    if mis.isDownwardClosed
        updateReducedMargin!(mis, idx)
    end
    isValid
end

"""
    findReducedFrontier(mis)

Find the subset of multi-indices that are not the backward neighbors of any other multi-index in the set.

# Examples
```jldoctest
julia> d, p = 2, 3;

julia> mis = CreateTotalOrder(d, p);

julia> frontier = findReducedFrontier(mis);

julia> length(frontier)
4

julia> expected_indices = [[0,3], [1,2], [2,1], [3,0]];

julia> all(e in vec(mis) for e in expected_indices)
true
```
"""
function findReducedFrontier(mis::MultiIndexSet{d}) where {d}
    frontier = Int[]
    unmarked = ones(Bool, length(mis.indices))
    sorted_idxs = sortperm(mis.indices, lt = (x, y) -> sum(x) < sum(y))
    inv_sorted_idxs = invperm(sorted_idxs)
    while true
        last_unmarked = findlast(unmarked)
        isnothing(last_unmarked) && break
        # Mark all backward ancestors of last_unmarked
        ancestors = allBackwardAncestors(mis, sorted_idxs[last_unmarked])
        unmarked[inv_sorted_idxs[ancestors]] .= false
        unmarked[last_unmarked] = false
        push!(frontier, sorted_idxs[last_unmarked])
    end
    frontier
end

# Given a subset of indices, find the minimal subset of the mset that contains them
function subsetCompletion(mset::MultiIndexSet{d}, indices::AbstractVector{Int}) where {d}
    completion = Set{Int}(indices)
    for idx in indices
        ancestors = allBackwardAncestors(mset, idx)
        for a in ancestors
            push!(completion, a)
        end
    end
    collect(completion)
end
