using MultiIndexing
import AcceleratedKernels as AK
using Test, StaticArrays, Random

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

@testset "MultiIndexing.jl" begin
    @testset "MultiIndexSet Creation" begin
        include("multiindexset_construct.jl")
    end

    @testset "MultiIndexSet Methods" begin
        include("multiindexset_fcn.jl")
    end

    @testset "FixedMultiIndexSet" begin
        include("fixedmultiindexset.jl")
    end

    @testset "Smolyak" begin
        include("smolyak.jl")
    end
end
