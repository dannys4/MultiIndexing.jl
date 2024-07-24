var documenterSearchIndex = {"docs":
[{"location":"","page":"Home","title":"Home","text":"CurrentModule = MultiIndexing","category":"page"},{"location":"#MultiIndexing","page":"Home","title":"MultiIndexing","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for MultiIndexing.jl.","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [MultiIndexing]\nOrder   = [:type, :function]","category":"page"},{"location":"#MultiIndexing.FixedMultiIndexSet","page":"Home","title":"MultiIndexing.FixedMultiIndexSet","text":"A FixedMultiIndexSet is a sparse representation of a multi-index set, where we only keep track of nonzero values.\n\nstarts: Vector{Int} - the starting index of each multi-index nzindices: Vector{Int} - the nonzero indices of each multi-index nzvalues: Vector{Int} - the nonzero values of each multi-index\n\nnzindices and nzvalues will have the same length M, and if there are N multi-indices, then length(starts) == N + 1. By construction, starts[1] == 1 and starts[N + 1] == M + 1.\n\n\n\n\n\n","category":"type"},{"location":"#MultiIndexing.MultiIndexSet","page":"Home","title":"MultiIndexing.MultiIndexSet","text":"Represents a set of multi-indices vector of {d} dimensional multi-indices with length N\n\n\n\n\n\n","category":"type"},{"location":"#MultiIndexing.MultiIndexSet-Union{Tuple{AbstractMatrix{Int64}}, Tuple{T}, Tuple{AbstractMatrix{Int64}, T}, Tuple{AbstractMatrix{Int64}, T, Any}} where T","page":"Home","title":"MultiIndexing.MultiIndexSet","text":"MultiIndexSet(indices, limit, calc_reduced_margin)\n\nCreate a multi-index set from a matrix of multi-indices\n\nArguments:\n\nindices: dim x N matrix of multi-indices\nlimit: function that takes a multi-index and a limit and returns true if the index is admissible\ncalc_reduced_margin: if true, calculate the reduced margin of the set, otherwise leave it empty\n\n\n\n\n\n","category":"method"},{"location":"#MultiIndexing.CreateTensorOrder-Union{Tuple{Int64, Int64}, Tuple{T}, Tuple{Int64, Int64, T}} where T","page":"Home","title":"MultiIndexing.CreateTensorOrder","text":"CreateTensorOrder(d, p, limit)\n\nCreate a multi-index set with tensor order p\n\n\n\n\n\n","category":"method"},{"location":"#MultiIndexing.CreateTotalOrder-Union{Tuple{T}, Tuple{Int64, Any}, Tuple{Int64, Any, T}} where T","page":"Home","title":"MultiIndexing.CreateTotalOrder","text":"CreateTotalOrder(d, p, limit)\n\nCreate a multi-index set with total order p\n\n\n\n\n\n","category":"method"},{"location":"#MultiIndexing.allBackwardAncestors-Union{Tuple{d}, Tuple{MultiIndexSet{d}, Int64}} where d","page":"Home","title":"MultiIndexing.allBackwardAncestors","text":"allBackwardAncestors(mis, j)\n\nGet all indices of multi-indices in mset limited by the tensor product box of the mset at idx\n\nArguments:\n\nmis: MultiIndexSet to get the backward ancestors from\nj: index of the multi-index to get the backward ancestors of\n\njulia> d, p = 2, 5;\n\njulia> mis = CreateTotalOrder(d, p);\n\njulia> println(visualize_2d(mis))\nX\nX X\nX X X\nX X X X\nX X X X X\nX X X X X X\n\njulia> midx = [2,2]; idx = findfirst(isequal(midx), vec(mis.indices));\n\njulia> ancestors_idx = allBackwardAncestors(mis, idx);\n\njulia> ancestors = MultiIndexSet(mis[ancestors_idx]);\n\njulia> println(visualize_2d(ancestors))\nX X\nX X X\nX X X\n\n\n\n\n\n\n","category":"method"},{"location":"#MultiIndexing.findReducedFrontier-Union{Tuple{MultiIndexSet{d}}, Tuple{d}} where d","page":"Home","title":"MultiIndexing.findReducedFrontier","text":"findReducedFrontier(mis)\n\nFind the subset of multi-indices that are not the backward neighbors of any other multi-index in the set.\n\nExamples\n\njulia> d, p = 2, 3;\n\njulia> mis = CreateTotalOrder(d, p);\n\njulia> frontier = findReducedFrontier(mis);\n\njulia> length(frontier)\n4\n\njulia> expected_indices = [[0,3], [1,2], [2,1], [3,0]];\n\njulia> all(e in vec(mis) for e in expected_indices)\ntrue\n\n\n\n\n\n","category":"method"},{"location":"#MultiIndexing.isAdmissible-Union{Tuple{d}, Tuple{MultiIndexSet{d}, StaticArraysCore.StaticArray{Tuple{d}, Int64, 1}}, Tuple{MultiIndexSet{d}, StaticArraysCore.StaticArray{Tuple{d}, Int64, 1}, Bool}} where d","page":"Home","title":"MultiIndexing.isAdmissible","text":"isAdmissible(mis, idx, check_indices)\n\nCheck if an index is admissible to add to a reduced margin\n\n\n\n\n\n","category":"method"},{"location":"#MultiIndexing.smolyakIndexing-Union{Tuple{MultiIndexSet{d, T}}, Tuple{T}, Tuple{d}} where {d, T}","page":"Home","title":"MultiIndexing.smolyakIndexing","text":"smolyakIndexing(mset) -> Vector{Tuple{Int,Int}}\n\nGets the smolyak indexing for a multi-index set. Returns a vector of tuples with each first index as the index of the multi-index representing a tensor product rule and each second index representing its count in the smolyak construction.\n\nExamples\n\njulia> d = 2;\n\njulia> mis = MultiIndexSet([(0:7)'; zeros(Int,1,8)]);\n\njulia> quad_rules = smolyakIndexing(mis); # Creates quad rule exact on x^7\n\njulia> quad_rules[1], length(quad_rules) # Counts the highest index once\n((8, 1), 1)\n\njulia> mis = CreateTotalOrder(d, 10);\n\njulia> print(visualize_smolyak_2d(mis, false))\nX\nX X\no X X\no o X X\no o o X X\no o o o X X\no o o o o X X\no o o o o o X X\no o o o o o o X X\no o o o o o o o X X\no o o o o o o o o X X\n\n\n\n\n\n","category":"method"},{"location":"#MultiIndexing.visualize_2d","page":"Home","title":"MultiIndexing.visualize_2d","text":"visualize_2d(mset_mat, markers)\n\nVisualize a 2D multi-index set as a string with markers for each index.\n\nArguments:\n\nmset: MultiIndexSet{2} or 2 x N matrix of multi-indices\nmarkers: a character or array of characters (length N) to use as markers for each index\n\nExamples\n\njulia> mis = CreateTotalOrder(2, 4);\n\njulia> println(visualize_2d(mis))\nX\nX X\nX X X\nX X X X\nX X X X X\n\n\n\n\n\n","category":"function"},{"location":"#MultiIndexing.visualize_smolyak_2d","page":"Home","title":"MultiIndexing.visualize_smolyak_2d","text":"visualize_smolyak_2d(mset)\n\nVisualize a two-dimensional mset's smolyak decomposition with colors!\n\nExamples\n\njulia> mis = MultiIndexSet([0 1 4 3 2 1 0 0 0 0; 0 1 0 0 0 0 1 2 3 4]);\n\njulia> println(visualize_smolyak_2d(mis, false))\nX\no\no\nX X\no X o o X\n\n\n\n\n\n","category":"function"}]
}
