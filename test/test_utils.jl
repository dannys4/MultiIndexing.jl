using FFTW, FastGaussQuadrature

function create_example_curved(rng, d, p, zero_dim = true, base=0.5)
    ani_weights = base .+ rand(rng, d)
    zero_dim && (ani_weights[1] = 0.)
    ani_weights = SVector{d}(ani_weights * (d-zero_dim)/sum(ani_weights))
    aniso_limiter = MultiIndexing.AnisotropicLimiter(ani_weights)
    curved_weights = base .+ rand(rng, d)
    zero_dim && (curved_weights[1] = 0.)
    curved_weights = SVector{d}(curved_weights * (d-zero_dim)/sum(curved_weights))
    curved_limiter = MultiIndexing.CurvedLimiter(curved_weights, aniso_limiter)
    mis = MultiIndexing.CreateTotalOrder(d, p, curved_limiter)
    mis
end

function unifquad01(exactness::Int)
    N = ceil(Int, (exactness + 1)/2)
    x, w = gausslegendre(N)
    return (x.+1)/2, w/2
end

function monomialEval(midx::SVector{d}, x::AbstractMatrix) where {d}
    evals = ones(size(x, 2))
    for i in eachindex(midx)
        evals .*= (x[i, :] .^ midx[i])
    end
    evals
end