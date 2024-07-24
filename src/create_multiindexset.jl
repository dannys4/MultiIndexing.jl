export create_example_hyperbolic2d, create_example_hyperbolic3d
export CreateTensorOrder, CreateTotalOrder

# Creates a matrix of multi-indices of total order p
# returns the matrix and the index where the frontier starts
function CreateTotalOrder_matrix(d::Int, p::Int)
    Mk = zeros(Int, d, 1)
    M = Mk
    last_start = -1
    for j in 1:p
        Mk = repeat(Mk, 1, d) + kron(I(d), ones(Int, 1, size(Mk, 2)))
        Mk = unique(Mk, dims = 2)
        M = hcat(M, Mk)
        j == p - 1 && (last_start = size(M, 2) + 1)
    end
    M, last_start
end

NoLimiter = Returns(true)

SumLimiter = (x::StaticVector, p) -> sum(x) <= p

struct AnisotropicLimiter{d, T}
    weights::StaticVector{d, Float64}
    limit::T
    function AnisotropicLimiter(
            weights::AbstractVector{Float64}, limit::_T = SumLimiter) where {_T}
        _d = length(weights)
        new{_d, _T}(SVector{_d}(weights), limit)
    end
end

function (lim::AnisotropicLimiter{d})(index::StaticVector{d}, p) where {d}
    lim.limit(index .* lim.weights, p)
end

struct CurvedLimiter{d, T}
    curve_weights::StaticVector{d, Float64}
    limit::T
    function CurvedLimiter(
            curve_weights::AbstractVector{Float64}, limit::_T = SumLimiter) where {_T}
        _d = length(curve_weights)
        new{_d, _T}(SVector{_d}(curve_weights), limit)
    end
end

function (lim::CurvedLimiter{d})(index::StaticVector{d}, p) where {d}
    lim.limit(index + lim.curve_weights .* log1p.(index), p)
end

"""
    CreateTotalOrder(d, p, limit)

Create a multi-index set with total order p
"""
function CreateTotalOrder(d::Int, p, limit::T = NoLimiter) where {T}
    p = ceil(Int, p)
    p < 0 && throw(ArgumentError("Invalid total order $p"))
    if p == 0
        rm = Vector{StaticVector{d, Int}}(undef, d)
        midx_tmp = zeros(Int, d)
        for j in 1:d
            midx_tmp[j] += 1
            rm[j] = SVector{d}(midx_tmp)
            midx_tmp[j] -= 1
        end
        rm = filter(midx -> limit(midx, p), rm)
        return MultiIndexSet{d, T}(
            [SVector{d}(zeros(Int, d))], rm, limit, MVector{d}(ntuple(Returns(0), d)))
    elseif p == 1
        rm = StaticVector{d, Int}[]
        indices = StaticVector{d, Int}[SVector{d}(ntuple(Returns(0), d))]
        sizehint!(indices, d + 1)
        sizehint!(rm, d * d)
        rm_idx = 1
        tmp_midx = zeros(Int, d)
        for j1 in 1:d
            tmp_midx[j1] += 1
            midx_j1 = SVector{d}(tmp_midx)
            limit(midx_j1, p) && push!(indices, midx_j1)
            for j2 in j1:d
                tmp_midx[j2] += 1
                limit(tmp_midx, p) && push!(rm, SVector{d}(tmp_midx))
                tmp_midx[j2] -= 1
            end
            tmp_midx[j1] -= 1
        end
        return MultiIndexSet{d, T}(indices, rm, limit, MVector{d}(ntuple(Returns(1), d)))
    end
    mset_mat, last_start = CreateTotalOrder_matrix(d, p)
    frontier = @view mset_mat[:, last_start:end]
    indices = [SVector{d}(mset_mat[:, i])
               for i in axes(mset_mat, 2) if limit(SVector{d}(mset_mat[:, i]), p)]
    reduced_margin = Vector{StaticVector{d, Int}}(undef, size(frontier, 2) * d)
    rm_idx = 1
    max_degrees = zeros(Int, d)
    full_limited = true
    @inbounds for i in axes(frontier, 2)
        m_idx = frontier[:, i]
        limit(SVector{d}(m_idx), p) && for j in 1:d
            max_degrees[j] = max(max_degrees[j], m_idx[j])
            m_idx[j] += 1
            static_m_idx = SVector{d}(m_idx)
            if limit(static_m_idx, p + 1)
                reduced_margin[rm_idx] = static_m_idx
                rm_idx += 1
            end
            m_idx[j] -= 1
            full_limited &= false
        end
    end
    full_limited && @warn "No valid reduced margin found on frontier"
    MultiIndexSet{d, T}(indices, unique(reduced_margin[1:(rm_idx - 1)]),
        limit, SVector{d}(max_degrees))
end

function tens_prod_mat(idx)
    reduce(
        hcat, collect.(Tuple.(vec(CartesianIndices(ntuple(k -> 0:idx[k], length(idx)))))))
end

"""
    CreateTensorOrder(d, p, limit)

Create a multi-index set with tensor order p
"""
function CreateTensorOrder(d::Int, p::Int, limit::T = NoLimiter) where {T}
    indices_arr = CartesianIndices(ntuple(_ -> 0:p, d))
    indices = [SVector(Tuple(idx))
               for idx in vec(indices_arr) if limit(SVector(Tuple(idx)), p)]
    reduced_margin_arr = CartesianIndices(ntuple(_ -> 0:(p + 1), d))
    reduced_margin = [SVector(Tuple(idx))
                      for idx in vec(reduced_margin_arr)
                      if any(Tuple(idx) .> p) && limit(SVector(Tuple(idx)), p + 1)]
    MultiIndexSet{d, T}(indices, reduced_margin, limit, SVector{d}(fill(p, d)))
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