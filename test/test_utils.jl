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

# Inspired by implementation in ChaosPy: https://github.com/jonathf/chaospy/blob/53000bbb04f8d3f9908ebbf1be6bf139a21c2e6e/chaospy/quadrature/clenshaw_curtis.py#L76
function clenshawcurtis01(order::Int)
    if order == 0
        return [0.5], [1.0]
    elseif order == 1
        return [0.0, 1.0], [0.5, 0.5]
    end

    theta = (order .- (0:order)) .* π / order
    abscissas = 0.5 .* cos.(theta) .+ 0.5

    steps = 1:2:order-1
    L = length(steps)
    remains = order - L

    beta = vcat(2.0 ./ (steps .* (steps .- 2)), [1.0 / steps[end]], zeros(remains))
    beta = -beta[1:end-1] .- reverse(beta[2:end])

    gamma = -ones(order)
    gamma[L+1] += order
    gamma[remains+1] += order
    gamma ./= order^2 - 1 + (order % 2)

    weights = rfft(beta + gamma)/order
    @assert maximum(imag.(weights)) < 1e-15
    weights = real.(weights)
    weights = vcat(weights, reverse(weights)[2 - (order % 2):end]) ./ 2

    return abscissas, weights
end

function monomialEval(midx::SVector{d}, x::AbstractMatrix) where {d}
    evals = ones(size(x, 2))
    for i in eachindex(midx)
        evals .*= (x[i, :] .^ midx[i])
    end
    evals
end