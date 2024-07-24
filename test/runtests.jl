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

    @testset "Smolyak Indexing" begin
        include("smolyak.jl")
    end
end
