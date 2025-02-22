module MultiIndexingAcceleratedKernelsExt
    using MultiIndexing
    using MultiIndexing: FixedMultiIndexSet
    import AcceleratedKernels as AK

    function MultiIndexing.FixedMultiIndexSet(fmset::FixedMultiIndexSet{d,T}, backend::AK.Backend) where {d,U,T<:AbstractVector{U}}
        (;starts,nz_indices,nz_values,max_orders) = fmset
        starts_dev = AK.allocate(backend, U, size(starts))
        nz_indices_dev = AK.allocate(backend, U, size(nz_indices))
        nz_values_dev = AK.allocate(backend, U, size(nz_values))
        T_dev = typeof(starts_dev)
        FixedMultiIndexSet{d,T_dev}(starts_dev, nz_indices_dev, nz_values_dev, max_orders)
    end
end