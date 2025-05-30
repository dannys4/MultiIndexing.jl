export FixedMultiIndexSet

import Base: length, size

"""
A FixedMultiIndexSet is a sparse representation of a multi-index set,
where we only keep track of nonzero values.

starts: Vector{Int} - the starting index of each multi-index
nz_indices: Vector{Int} - the nonzero indices of each multi-index
nz_values: Vector{Int} - the nonzero values of each multi-index

nz_indices and nz_values will have the same length `M`, and
if there are `N` multi-indices, then `length(starts) == N + 1`.
By construction, `starts[1] == 1` and `starts[N + 1] == M + 1`.
"""
struct FixedMultiIndexSet{d, T <: AbstractVector{<:Integer}}
    starts::T
    nz_indices::T
    nz_values::T
    max_orders::SVector{d}
end

function FixedMultiIndexSet(mset::MultiIndexSet{d}) where {d}
    indices = mset.indices
    M = sum(sum(idx .!= 0) for idx in indices)
    starts = Vector{Int}(undef, length(indices) + 1)
    nz_indices = Vector{Int}(undef, M)
    nz_values = Vector{Int}(undef, M)
    max_orders_ = zeros(Int, d)
    nz_idx = 1
    @inbounds for (i, idx) in enumerate(indices)
        starts[i] = nz_idx
        @inbounds for (j, val) in enumerate(idx)
            if val != 0
                nz_indices[nz_idx] = j
                nz_values[nz_idx] = val
                max_orders_[j] = max(max_orders_[j], val)
                nz_idx += 1
            end
        end
    end
    max_orders = SVector{d}(max_orders_)
    starts[end] = M + 1
    FixedMultiIndexSet{d, Vector{Int}}(starts, nz_indices, nz_values, max_orders)
end

"""
    length(fmset::FixedMultiIndexSet)
Number of multi-indices in the fixed multi-index set
"""
Base.length(fmset::FixedMultiIndexSet) = length(fmset.starts) - 1

"""
    size(fmset::FixedMultiIndexSet, [dims])
(Dimension, length) of fixed multi-index set
"""
function Base.size(fmset::FixedMultiIndexSet{d}) where {d}
    d, length(fmset)
end
Base.size(fmset::FixedMultiIndexSet, dims) = size(fmset)[dims]