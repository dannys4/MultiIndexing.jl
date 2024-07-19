struct AdaptiveSparseGrid_4{d,O,R,D,T,V,C}
    base_mset::MultiIndexSet{d}
    diagnostic_output_type::Type{O}
    diagnostic::D
    force_nested::Bool
    quad_rules::R
    tol::Float64
    max_degrees::V
    directions::T
    diagnostic_comparison::C
end
AdaptiveSparseGrid = AdaptiveSparseGrid_4


function formDifference1d(q_rule, order)
    # Ensure each quadrature rule is sorted
    pts_hi, wts_hi = q_rule(order)
    sort_hi = sortperm(pts_hi)

    pts_lo, wts_lo = q_rule(order-1)
    sort_lo = sortperm(pts_lo)

    diff_quad_pts = similar(pts_hi)
    diff_quad_wts = similar(wts_hi)

    # Match up points
    lo_idx = 1
    @inbounds for hi_idx in eachindex(pts_hi)
        p_hi = pts_hi[sort_hi[hi_idx]]
        p_lo = pts_lo[sort_lo[lo_idx]]
        diff_quad_pts[hi_idx] = p_hi
        diff_quad_wts[hi_idx] = wts_hi[sort_hi[hi_idx]]
        lo_idx > length(sort_lo) && continue
        if p_hi == p_lo
            diff_quad_wts[hi_idx] -= wts_lo[sort_lo[lo_idx]]
            lo_idx += 1
        elseif p_hi > p_lo
            throw(ArgumentError("Points are not nested"))            
        end
    end
    diff_quad_pts, diff_quad_wts
end

function formDifference!(eval_dict, rules, midx::SVector{d, Int}, fcn=nothing) where {d}
    differences_1d = [formDifference1d(r,idx) for (r,idx) in zip(rules, midx)]
    pts, wts = tensor_prod_quad(differences_1d, Val{d}())
    pts_evals = nothing
    if isnothing(fcn)
        pts_evals = [get(eval_dict,pts[:,j],nothing) for j in axes(pts,2)]
        any(isnothing.(pts_evals)) && throw(ArgumentError("Point found which function was not evaluated at."))
    else
        pts_evals = [fcn(pts[:,j]) for j in axes(pts,2)]
        for (j,pts_evals) in eachindex(pts_evals)
            eval_dict[pts[:,j]] = pts_evals[j]
        end
    end
    sum(p*w for (p,w) in zip(pts_evals, wts))
end

function adaptiveIntegrate(fcn, asg::AdaptiveSparseGrid{d,T}) where {d, T}
    eval_dict = Dict()
    mset = asg.base_mset
    rules = asg.rules
    active_midxs = MultiIndexing.findReducedFrontier(mset)
    L = Tuple{SVector{d, Int}, T}
    sorted_midxs = SortedList{L}(asg.comp)
    for midx in active_midxs
        midx_diff = formDifference!(eval_dict, rules, midx, fcn)
        push!(sorted_midxs, midx_diff)
    end
    
end