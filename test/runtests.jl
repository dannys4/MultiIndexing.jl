using MultiIndexing
using Test, StaticArrays, Random

include("test_utils.jl")

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

    @testset "Adaptive Sparse Grid" begin
        include("adaptiveSparseGrid.jl")
    end
end
