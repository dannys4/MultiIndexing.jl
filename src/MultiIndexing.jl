module MultiIndexing

using StaticArrays, LinearAlgebra
import Base: getindex, push!, length, iterate

"""
Represents a set of multi-indices
vector of {d} dimensional multi-indices with length N
"""
struct MultiIndexSet{d,T}
    indices::Vector{SVector{d,Int}}
    reduced_margin::Vector{StaticVector{d,Int}}
    limit::T
    isDownwardClosed::Bool
    maxDegrees::MVector{d,Int}
end

function MultiIndexSet(indices::Vector{SVector{d,Int}}, limit::T=NoLimiter, calc_reduced_margin=false) where {d,T}
    reduced_margin = []
    if calc_reduced_margin
        throw(ArgumentError("Reduced margin calculation not implemented"))
    end
    maxDegrees = MVector{d}(reduce((x,y)->max.(x,y), indices, init=zeros(Int, d)))
    MultiIndexSet{d,T}(indices, reduced_margin, limit, true, maxDegrees)
end

"""
Create a multi-index set from a matrix of multi-indices

    MultiIndexSet(indices, limit, calc_reduced_margin)

- indices: dim x N matrix of multi-indices
- limit: function that takes a multi-index and a limit and returns true if the index is admissible
- calc_reduced_margin: if true, calculate the reduced margin of the set, otherwise leave it empty
"""
function MultiIndexSet(indices_mat::Matrix{Int}, limit::T=NoLimiter, calc_reduced_margin=false) where {T}
    d = size(indices_mat,1)
    indices = [SVector{d}(indices_mat[:,i]) for i in axes(indices_mat,2)]
    MultiIndexSet(indices, limit, calc_reduced_margin)
end

# Some helper functions
function Base.getindex(mis::MultiIndexSet{d}, i) where {d}
    mis.indices[i]
end

function Base.length(mis::MultiIndexSet{d}) where {d}
    length(mis.indices)
end

function Base.iterate(mis::MultiIndexSet{d}, state::Int = 1)::Union{Tuple{SVector{d,Int},Int},Nothing} where {d}
    return state > length(mis.indices) ? nothing : (mis.indices[state], state+1)
end

function Base.vec(mis::MultiIndexSet{d}) where {d}
    collect.(mis.indices)
end


# Creates a matrix of multi-indices of total order p
# returns the matrix and the index where the frontier starts
function CreateTotalOrder_matrix(d::Int, p::Int)
    Mk = zeros(Int, d, 1)
    M = Mk
    last_start = -1
    for j in 1:p
        Mk = repeat(Mk, 1, d) + kron(I(d), ones(Int, 1, size(Mk, 2)))
        Mk = unique(Mk, dims=2)
        M = hcat(M, Mk)
        j == p-1 && (last_start = size(M, 2)+1)
    end
    M, last_start
end

NoLimiter = Returns(true)

SumLimiter = (x::StaticVector,p) -> sum(x) <= p

struct AnisotropicLimiter{d,T}
    weights::StaticVector{d,Float64}
    limit::T
    function AnisotropicLimiter(weights::AbstractVector{Float64}, limit::_T = SumLimiter) where {_T}
        _d = length(weights)
        new{_d,_T}(SVector{_d}(weights), limit)
    end
end

function (lim::AnisotropicLimiter{d})(index::StaticVector{d}, p) where {d}
    lim.limit(index .* lim.weights, p)
end

struct CurvedLimiter{d,T}
    curve_weights::StaticVector{d,Float64}
    limit::T
    function CurvedLimiter(curve_weights::AbstractVector{Float64}, limit::_T = SumLimiter) where {_T}
        _d = length(curve_weights)
        new{_d,_T}(SVector{_d}(curve_weights), limit)
    end
end

function (lim::CurvedLimiter{d})(index::StaticVector{d}, p) where {d}
    lim.limit(index + lim.curve_weights .* log1p.(index), p)
end

"""
Create a multi-index set with total order p

    CreateTotalOrder(d, p, limit)
"""
function CreateTotalOrder(d::Int, p, limit=NoLimiter)
    mset_mat, last_start = CreateTotalOrder_matrix(d, ceil(p))
    frontier = @view mset_mat[:, last_start:end]
    indices = [SVector{d}(mset_mat[:,i]) for i in axes(mset_mat,2) if limit(SVector{d}(mset_mat[:,i]), p)]
    reduced_margin = Vector{StaticVector{d,Int}}(undef, size(frontier, 2)*d)
    rm_idx = 1
    max_degrees = zeros(Int, d)
    full_limited = true
    @inbounds for i in axes(frontier, 2)
        m_idx = frontier[:,i]
        limit(SVector{d}(m_idx), p) && for j in 1:d
            max_degrees[j] = max(max_degrees[j], m_idx[j])
            m_idx[j] += 1
            static_m_idx = SVector{d}(m_idx)
            if limit(static_m_idx, p+1)
                reduced_margin[rm_idx] = static_m_idx
                rm_idx += 1
            end
            m_idx[j] -= 1
            full_limited &= false;
        end
    end
    full_limited && @warn "No valid reduced margin found on frontier"
    MultiIndexSet{d,typeof(limit)}(indices, unique(reduced_margin[1:rm_idx-1]), limit, true, SVector{d}(max_degrees))
end

function tens_prod_mat(idx)
    reduce(hcat, collect.(Tuple.(vec(CartesianIndices(ntuple(k->0:idx[k], length(idx)))))))
end
function tens_prod_mat(p,d)
    tens_prod_mat(fill(p,d))
end

"""
Create a multi-index set with tensor order p

    CreateTensorOrder(d, p, limit)
"""
function CreateTensorOrder(d::Int, p::Int, limit::T=NoLimiter) where {T}
    indices_arr = CartesianIndices(ntuple(_->0:p, d))
    indices = [SVector(Tuple(idx)) for idx in vec(indices_arr) if limit(SVector(Tuple(idx)), p)]
    reduced_margin_arr = CartesianIndices(ntuple(_->0:p+1, d))
    reduced_margin = [SVector(Tuple(idx)) for idx in vec(reduced_margin_arr) if any(idx .> p) && limit(SVector(Tuple(idx)), p+1)]
    MultiIndexSet{d,T}(indices, reduced_margin, limit, true, SVector{d}(fill(p,d)))
end

# Searches backward through the mset backward for an index
function in_mset_backward(mis::MultiIndexSet{d}, idx::StaticVector{d,Int}) where {d}
    any(idx .> mis.maxDegrees) && return false
    for i in length(mis.indices):-1:1
        mis.indices[i] == idx && return true
    end
    false
end

"""
Check if an index is admissible to add to a reduced margin

    checkIndexAdmissible(mis, idx, check_indices)
"""
function checkIndexAdmissible(mis::MultiIndexSet{d}, idx::StaticVector{d,Int}, check_indices::Bool = false) where {d}
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

"""
Get all indices of multi-indices in mset limited by the tensor product box of the mset at idx

    allBackwardAncestors(mis, j)

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
    back_indices = CartesianIndices(ntuple(k->0:idx[k], d))
    j_ancestors = Vector{Int}(undef, length(back_indices) - 1)
    for (i, back_index) in enumerate(back_indices)
        if i < length(back_indices)
            check_back_index = isequal(SVector(Tuple(back_index)))
            j_ancestors[i] = findfirst(check_back_index, mis.indices)
        end
    end
    j_ancestors
end

# Update the max degrees of a set after adding an index
function updateMaxDegrees!(mis::MultiIndexSet{d}, idx::StaticVector{d,Int}) where {d}
    mis.maxDegrees .= max.(mis.maxDegrees, idx)
end

# Update the reduced margin of a set after adding an index
function updateReducedMargin!(mis::MultiIndexSet{d}, idx::StaticVector{d,Int}) where {d}
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

function Base.push!(mis::MultiIndexSet{d}, idx::StaticVector{d,Int}) where {d}
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
Find the subset of multi-indices that are not the backward neighbors of any other multi-index in the set.

    findReducedFrontier(mis)

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
    sorted_idxs = sortperm(mis.indices, lt=(x,y)->sum(x) < sum(y))
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

"""
Visualize a 2D multi-index set as a string with markers for each index.

    visualize_2d(mset_mat, markers)

Arguments:
- mset: MultiIndexSet{2} or 2 x N matrix of multi-indices
- markers: a character or array of characters (length N) to use as markers for each index

```jldoctest
julia> mis = CreateTotalOrder(2, 4);

julia> println(visualize_2d(mis))
X
X X
X X X
X X X X
X X X X X

"""
function visualize_2d(mset, markers = 'X')
    if mset isa MultiIndexSet
        mset = reduce(hcat, mset.indices)
    end
    @assert size(mset, 1) == 2 "Only 2D visualization supported"
    chars = fill(' ', (maximum(mset,dims=2) .+ 1)...)
    for j in axes(mset,2)
        mark = markers isa Char ? markers : markers[j]
        chars[end - mset[1,j], mset[2,j]+1] = mark
    end
    rows = [join(c, ' ') for c in eachrow(chars)]
    join(rows, "\n")
end

function rgb_char(r, g, b, char)
    "\e[1m\e[38;2;$r;$g;$b;249m$char\e[0m"
end

"""
Visualize a two-dimensional mset's smolyak decomposition with colors!

        visualize_smolyak_2d(mset)

```jldoctest
julia> mis = MultiIndexSet([0 1 4 3 2 1 0 0 0 0; 0 1 0 0 0 0 1 2 3 4]);

julia> println(visualize_smolyak_2d(mis)) # Note colors on [4,0], [1,1] and [0,4] match
X
o
o
X X
X X o o X

"""
function visualize_smolyak_2d(mset::MultiIndexSet)
    smolyak_indices = smolyakIndexing(mset, true)
    rgb1, rgb2 = [100, 100, 0], [0, 100, 100]
    mset_max = [maximum(j[1] for j in mset.indices), maximum(j[2] for j in mset.indices)]
    chars = fill(" ", (mset_max .+ 1)...)
    for (j,level) in enumerate(smolyak_indices)
        col = rgb1 * (j-1) + rgb2 * (length(smolyak_indices) - j)
        for idx in level
            m_idx = mset.indices[idx]
            chars[(m_idx .+ 1)...] = rgb_char(col..., 'X')
            back_indices = allBackwardAncestors(mset, idx)
            for bidx in back_indices
                m_idx_b = mset.indices[bidx]
                chars[(m_idx_b .+1)...] = rgb_char(col..., 'o')
            end
        end
    end
    join([join(c, ' ') for c in eachrow(chars)][end:-1:1], "\n")
end

"""
Gets the smolyak indexing for a multi-index set. Each index represents a tensor-product box.

    smolyakIndexing(mis, keep_levels)

Arguments:
- mis: MultiIndexSet to get the Smolyak indexing for
- keep_levels: if true, return index vectors at each level of the Smolyak grid,
where we subtract every index in level j if even and add if j is odd. If false, return the positive and negative pairs

```jldoctest
julia> mis = MultiIndexSet([0 1 4 3 2 1 0 0 0 0; 0 1 0 0 0 0 1 2 3 4]);

julia> println(visualize_2d(mis))
X
X
X
X X
X X X X X

julia> level_indices = smolyakIndexing(mis, true);

julia> level_sets = [MultiIndexSet(mis[li]) for li in level_indices];

julia> println(visualize_2d(level_sets[1]))
X


  X
        X

julia> println(visualize_2d(level_sets[2]))
X
  X

julia> println(visualize_2d(level_sets[3]))
X
```
"""
function smolyakIndexing(mis::MultiIndexSet{d,T}, keep_levels::Bool=false) where {d,T}
    mi_loop = mis
    frontiers = keep_levels ? Vector{Int}[] : ntuple(_->Int[], 2)
    original_indices = collect(1:length(mis.indices))
    frontiers_index = 1
    while length(mi_loop.indices) > 0
        frontier = findReducedFrontier(mi_loop)
        occurrences = zeros(Int, length(mi_loop.indices))
        for idx in frontier
            back_idxs = allBackwardAncestors(mi_loop, idx)
            occurrences[back_idxs] .+= 1
        end
        sort!(frontier)
        if keep_levels
            push!(frontiers, original_indices[frontier])
        else
            append!(frontiers[frontiers_index], original_indices[frontier])
        end
        repeated_indices = mi_loop.indices[occurrences .> 1]
        original_indices = original_indices[occurrences .> 1]
        mi_loop = MultiIndexSet{d,T}(repeated_indices, [], mis.limit, true, mis.maxDegrees)
        frontiers_index = 3-frontiers_index # map 2 to 1 and 1 to 2
    end
    frontiers
end

# Create a two-dimensional multi-index set with a hyperbolic limiter
function create_example_hyperbolic2d(p)
    log2p = floor(Int, log2(p))
    pd2 = p ÷ 2
    midx_rep = reduce(hcat, [tens_prod_mat(@SVector[p_1, pd2 ÷ p_1]) for p_1 in 2 .^ (0:log2p-1)])
    midx_ext = Int[midx_rep [pd2+1:p zeros(pd2)]' [zeros(pd2) pd2+1:p]']
    MultiIndexSet(unique(midx_ext, dims=2))
end

# Create a three-dimensional multi-index set with a hyperbolic limiter
function create_example_hyperbolic3d(p)
    log2p = floor(Int, log2(p))
    pd2 = p ÷ 2
    mset_rep = []
    for logp_1 in 0:log2p-1
        for logp_2 in 0:(log2p-1-logp_1)
            p_1 = 2^logp_1
            p_2 = 2^logp_2
            p_3 = 2^(log2p - (logp_1 + logp_2 + 1))
            push!(mset_rep, tens_prod_mat(@SVector[p_1, p_2, p_3]))
        end
    end
    mset_rep = reduce(hcat, mset_rep)
    mset_ext = Int[mset_rep [pd2+1:p zeros(pd2) zeros(pd2)]' [zeros(pd2) pd2+1:p zeros(pd2)]' [zeros(pd2) zeros(pd2) pd2+1:p]']
    MultiIndexSet(unique(mset_ext, dims=2))
end

export CreateTensorOrder, CreateTotalOrder, MultiIndexSet
export findReducedFrontier, allBackwardAncestors, smolyakIndexing
export visualize_2d, visualize_smolyak_2d
export create_example_hyperbolic2d, create_example_hyperbolic3d
end
