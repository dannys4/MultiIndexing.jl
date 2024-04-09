using MultiIndexing
using Test
using JET

@testset "MultiIndexing.jl" begin
    @testset "Code linting (JET.jl)" begin
        JET.test_package(MultiIndexing; target_defined_modules = true)
    end

end
