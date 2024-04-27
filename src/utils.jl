export visualize_2d, visualize_smolyak_2d
export create_example_hyperbolic2d, create_example_hyperbolic3d

"""
    visualize_2d(mset_mat, markers)

Visualize a 2D multi-index set as a string with markers for each index.

# Arguments:
- `mset`: MultiIndexSet{2} or 2 x N matrix of multi-indices
- `markers`: a character or array of characters (length N) to use as markers for each index

# Examples
```jldoctest
julia> mis = CreateTotalOrder(2, 4);

julia> println(visualize_2d(mis))
X
X X
X X X
X X X X
X X X X X
```
"""
function visualize_2d(mset, markers = 'X')
    if mset isa MultiIndexSet
        mset = reduce(hcat, mset.indices)
    end
    @assert size(mset, 1)==2 "Only 2D visualization supported"
    chars = fill(' ', (maximum(mset, dims = 2) .+ 1)...)
    for j in axes(mset, 2)
        mark = markers isa Char ? markers : markers[j]
        chars[end - mset[1, j], mset[2, j] + 1] = mark
    end
    rows = [join(c, ' ') for c in eachrow(chars)]
    join(rows, "\n")
end

function rgb_char(r, g, b, char, colored)
    colored ? "\e[1m\e[38;2;$r;$g;$b;249m$char\e[0m" : string(char)
end

"""
    visualize_smolyak_2d(mset)

Visualize a two-dimensional mset's smolyak decomposition with colors!

# Examples
```jldoctest
julia> mis = MultiIndexSet([0 1 4 3 2 1 0 0 0 0; 0 1 0 0 0 0 1 2 3 4]);

julia> println(visualize_smolyak_2d(mis, false))
X
o
o
X X
o X o o X
```
"""
function visualize_smolyak_2d(mset::MultiIndexSet, colored::Bool = true)
    smolyak_rules = smolyakIndexing(mset)
    min_count, max_count = extrema(j[2] for j in smolyak_rules)
    rgb1, rgb2 = [100, 100, 0], [0, 100, 100]
    mset_max = [maximum(j[1] for j in mset.indices), maximum(j[2] for j in mset.indices)]
    chars = fill(" ", (mset_max .+ 1)...)
    for (idx, j) in smolyak_rules
        interp_idx = (j - min_count) / (max_count - min_count)
        col = round.(Int, rgb1 * (1 - interp_idx) + rgb2 * interp_idx)
        m_idx = mset.indices[idx]
        chars[(m_idx .+ 1)...] = rgb_char(col..., 'X', colored)
        back_indices = allBackwardAncestors(mset, idx)
        for bidx in back_indices
            m_idx_b = mset.indices[bidx]
            chars[(m_idx_b .+ 1)...] = rgb_char(col..., 'o', colored)
        end
    end
    join([join(c, ' ') for c in eachrow(chars)][end:-1:1], "\n")
end

# Create a two-dimensional multi-index set with a hyperbolic limiter
function create_example_hyperbolic2d(p)
    log2p = floor(Int, log2(p))
    pd2 = p ÷ 2
    midx_rep = reduce(
        hcat, [tens_prod_mat(@SVector[p_1, pd2 ÷ p_1]) for p_1 in 2 .^ (0:(log2p - 1))])
    midx_ext = Int[midx_rep [(pd2 + 1):p zeros(pd2)]' [zeros(pd2) (pd2 + 1):p]']
    MultiIndexSet(unique(midx_ext, dims = 2))
end

# Create a three-dimensional multi-index set with a hyperbolic limiter
function create_example_hyperbolic3d(p)
    log2p = floor(Int, log2(p))
    pd2 = p ÷ 2
    mset_rep = []
    for logp_1 in 0:(log2p - 1)
        for logp_2 in 0:(log2p - 1 - logp_1)
            p_1 = 2^logp_1
            p_2 = 2^logp_2
            p_3 = 2^(log2p - (logp_1 + logp_2 + 1))
            push!(mset_rep, tens_prod_mat(@SVector[p_1, p_2, p_3]))
        end
    end
    mset_rep = reduce(hcat, mset_rep)
    mset_ext = Int[mset_rep [(pd2 + 1):p zeros(pd2) zeros(pd2)]' [zeros(pd2) (pd2 + 1):p zeros(pd2)]' [zeros(pd2) zeros(pd2) (pd2 + 1):p]']
    MultiIndexSet(unique(mset_ext, dims = 2))
end