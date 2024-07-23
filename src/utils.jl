export visualize_2d, visualize_smolyak_2d

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
    chars = fill(' ', (reverse(maximum(mset, dims = 2)) .+ 1)...)
    for j in axes(mset, 2)
        mark = markers isa Char ? markers : markers[j]
        chars[end - mset[2, j], mset[1, j] + 1] = mark
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

using Base: collect, pop!, push!, peek, isempty

mutable struct SortedListNode{P}
    payload::Union{Nothing,P}
    next::Union{Nothing,SortedListNode{P}}
end

mutable struct SortedList{P,C}
    comp::C
    head::SortedListNode{P}
    function SortedList{_P}(comp::_C) where {_P,_C}
        tail = SortedListNode{_P}(nothing,nothing)
        head = SortedListNode{_P}(nothing,tail)
        new{_P,_C}(comp, head)
    end
end

function Base.push!(list::SortedList{T}, val::T) where {T}
    prev = list.head
    node = prev.next
    while !isnothing(node.payload)
        list.comp(val, node.payload) && break
        prev = node
        node = prev.next
    end
    new_node = SortedListNode(val, node)
    prev.next = new_node
end

function Base.pop!(list::SortedList)
    ret_node = list.head.next
    isnothing(ret_node.payload) && return nothing
    list.head.next = ret_node.next
    ret_node.payload
end

function Base.peek(list::SortedList)
    ret_node = list.head.next
    isnothing(ret_node.payload) && return nothing
    ret_node.payload
end

function Base.isempty(list::SortedList)
    isnothing(peek(list))
end

function Base.collect(list::SortedList{T}) where {T}
    ret = T[]
    node = list.head.next
    while !isnothing(node.payload)
        push!(ret, node.payload)
        node = node.next
    end
    ret
end