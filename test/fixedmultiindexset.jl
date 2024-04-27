@testset "FixedMultiIndexSet Creation" begin
    d, p = 5, 10
    mset = CreateTotalOrder(d, p)
    all_indices = [reduce(vcat, mset.indices) repeat(1:d, length(mset))]
    nz_per_index = [sum(idx .!= 0) for idx in mset.indices]
    starts = cumsum([1;nz_per_index])
    nz_info = all_indices[all_indices[:,1] .!= 0, :]
    nz_values = nz_info[:,1]
    nz_indices = nz_info[:,2]
    fmset = FixedMultiIndexSet(mset)
    @test fmset.starts == starts
    @test fmset.nz_indices == nz_indices
    @test fmset.nz_values == nz_values
    @test fmset.max_orders == SVector{d}(fill(p, d))
end