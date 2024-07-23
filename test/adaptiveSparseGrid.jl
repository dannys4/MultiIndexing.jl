
test_fcn_1d = x->sin(0.4pi*cos(0.4pi*x[]))^2/(x[]+1)
clenshawcurtis01_nested = j->clenshawcurtis01(j == 0 ? 0 : 2^j)
@testset "Difference Operators" begin
    order = 5
    @testset "One-dimensional function" begin
        diff_quad_pts, diff_quad_wts = MultiIndexing.formDifference1dRule(clenshawcurtis01_nested, order)
        pts_hi, wts_hi = clenshawcurtis01_nested(order)
        pts_lo, wts_lo = clenshawcurtis01_nested(order-1)
        test_fcn_diff_true = test_fcn_1d.(pts_hi)'wts_hi - test_fcn_1d.(pts_lo)'wts_lo
        test_fcn_diff_est = test_fcn_1d.(diff_quad_pts)'diff_quad_wts
        @test test_fcn_diff_true ≈ test_fcn_diff_est atol=1e-10

        eval_count = 0
        mset_1d = CreateTotalOrder(1, order)
        function test_fcn_counter(pt)
            eval_count += 1
            test_fcn_1d(pt[])
        end
        eval_dict = Dict()
        rules = (clenshawcurtis01_nested,)
        midx = @SVector[order]
        diff = MultiIndexing.differenceDiagnostic!(eval_dict, rules, midx, test_fcn_counter)
        @test eval_count == length(pts_hi)
        @test diff ≈ test_fcn_diff_est/length(pts_hi) atol=1e-10
    end
    @testset "two-dimensional function" begin
        eval_counter = 0
        function test_fcn_2d(x)
            eval_counter += 1
            (test_fcn_1d(x[1])^2 + 1)*test_fcn_1d(x[2])
        end
        orders = [3, 4]
        midx_test = SVector((orders...))
        
        # Calculate exact difference based on true formula
        U1_hi, U2_hi = clenshawcurtis01_nested.(orders)
        U1_lo, U2_lo = clenshawcurtis01_nested.(orders .- 1)
        TPQ = MultiIndexing.tensor_prod_quad
        pts_hi_hi, wts_hi_hi = TPQ((U1_hi, U2_hi), Val{2}())
        pts_hi_lo, wts_hi_lo = TPQ((U1_hi, U2_lo), Val{2}())
        pts_lo_hi, wts_lo_hi = TPQ((U1_lo, U2_hi), Val{2}())
        pts_lo_lo, wts_lo_lo = TPQ((U1_lo, U2_lo), Val{2}())
        diff_true  = test_fcn_2d.(pts_hi_hi)'wts_hi_hi
        diff_true -= test_fcn_2d.(pts_hi_lo)'wts_hi_lo
        diff_true -= test_fcn_2d.(pts_lo_hi)'wts_lo_hi
        diff_true += test_fcn_2d.(pts_lo_lo)'wts_lo_lo

        # Use MultiIndexing method
        eval_dict = Dict()
        eval_counter = 0
        rules = (clenshawcurtis01_nested,clenshawcurtis01_nested)
        diff_approx = MultiIndexing.differenceDiagnostic!(eval_dict, rules, midx_test, test_fcn_2d)
        expected_evals = length(pts_hi_hi)
        @test eval_counter == expected_evals
        @test diff_true/length(pts_hi_hi) ≈ diff_approx atol=1e-10
    end
end

@testset "AdaptiveSparseGrid" begin
    @testset "One dimensional" begin
        mset_base = CreateTotalOrder(1,1)
        num_evals = 0
        function test_fcn_1d_poly(x)
            num_evals += 1
            x[]^2 + x[] + x[]^10
        end
        test_fcn_int = (1/3) + (1/2) + (1/11)
        asg = AdaptiveSparseGrid(mset_base, (clenshawcurtis01_nested,), tol=5eps())
        result, eval_dict, final_pts, final_wts = adaptiveIntegrate!(asg, test_fcn_1d_poly; verbose=false)
        sp = sortperm(final_pts[:])
        final_pts = final_pts[sp]
        final_wts = final_wts[sp]
        @test result ≈ test_fcn_int atol=1e-10
        needed_level = ceil(Int, log2(10))
        @test mset_base.maxDegrees[1] >= needed_level
        # Make sure we keep track of every evaluation
        @test length(eval_dict) == num_evals
        # Make sure that the level doesn't go more than 1 above
        # level required to integrate fcn _exactly_
        @test length(eval_dict) == 2^(needed_level+1)+1
        expect_pts, expect_wts = clenshawcurtis01_nested(needed_level+1)
        @test all(isapprox.(expect_pts, final_pts, atol=1e-10))
        @test all(isapprox.(expect_wts, final_wts, atol=1e-10))
    end
    @testset "Two dimensional" begin
        mset_base = CreateTotalOrder(2,1)
        num_evals = 0
        function test_fcn_2d_poly(x)
            num_evals += 1
            x[1]^16 + x[2]^8 + x[1]*x[2]
        end
        test_fcn_2d_int = (1/17) + (1/9) + (1/4)
        # Limit interaction indices to be total order < 3
        # In 2d this means [1,1] can be the only interaction midx
        function is_valid_midx(midx)
            sum(midx .> 0) <= 1 && return true
            sum(midx) < 3
        end
        asg = AdaptiveSparseGrid(mset_base, (clenshawcurtis01_nested,clenshawcurtis01_nested), tol=5eps(); is_valid_midx)
        result, eval_dict, final_pts, final_wts = adaptiveIntegrate!(asg, test_fcn_2d_poly; verbose=false)
        
        @test length(eval_dict) == num_evals
        @test result ≈ test_fcn_2d_int atol=1e-10
        # Required level for integrating each monomial in fcn
        @test all(mset_base.maxDegrees .== [5,4])

        # Check the rule it returns is actually the rule associated with the new mset
        lt_slice = mat->((i,j)->any(mat[k,i] < mat[k,j] for k in axes(mat,1)))
        sp_test = sort(axes(final_pts,2), lt=lt_slice(final_pts))
        final_pts = final_pts[:,sp_test]
        final_wts = final_wts[sp_test]

        ref_pts, ref_wts = SmolyakQuadrature(mset_base, clenshawcurtis01_nested)
        sp_ref = sort(axes(ref_pts,2), lt=lt_slice(ref_pts))
        ref_pts = ref_pts[:,sp_ref]
        ref_wts = ref_wts[sp_ref]
        @test all(isapprox.(ref_pts, final_pts, atol=1e-10))
        @test all(isapprox.(ref_wts, final_wts, atol=1e-10))
    end
end

@testset "High dimensional" begin
    dim = 10
    pow = 16
    mset_base = CreateTotalOrder(dim,1)
    num_evals = 0
    function test_fcn_nd_poly(x)
        num_evals += 1
        sum(x->x^pow, x)+prod(x[1:end÷3])
    end
    function is_valid_midx(midx)
        sum(midx) > 10 && return false
        num_nz = sum(midx .> 0)
        num_nz <= 1 && return true
        num_nz == sum(midx[1:end÷3] .> 0) && return sum(midx) < 5
        sum(midx) < 3
    end
    test_fcn_nd_int = dim/(pow+1)+2. ^(-(dim÷3))
    asg = AdaptiveSparseGrid(mset_base, ntuple(_->clenshawcurtis01_nested,dim), tol=5eps(); is_valid_midx)
    result, eval_dict, final_pts, final_wts = adaptiveIntegrate!(asg, test_fcn_nd_poly; verbose=false)
    @test result ≈ test_fcn_nd_int atol=1e-10
    @test length(eval_dict) == num_evals
    Main._pts = final_pts
end