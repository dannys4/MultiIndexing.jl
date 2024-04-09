using MultiIndexing
using Test

can_use_jet = VERSION >= v"1.10"
can_use_jet && using JET

@testset "MultiIndexing.jl" begin
    @testset "Code linting (JET.jl)" begin
        can_use_jet && JET.test_package(MultiIndexing; target_defined_modules = true)
    end
end
