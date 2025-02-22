module MultiIndexingKernelAbstractionsExt
    using MultiIndexing
    using MultiIndexing: FixedMultiIndexSet
    using KernelAbstractions

    function MultiIndexing.FixedMultiIndexSet(fmset::FixedMultiIndexSet{d,T}, backend::AK.Backend) where {d,U,T<:AbstractVector{U}}
        (;starts,nz_indices,nz_values,max_orders) = fmset
        starts_dev = allocate(backend, U, size(starts))
        nz_indices_dev = allocate(backend, U, size(nz_indices))
        nz_values_dev = allocate(backend, U, size(nz_values))
        KernelAbstractions.copyto!(backend, starts_dev, starts)
        KernelAbstractions.copyto!(backend, nz_indices_dev, nz_indices)
        KernelAbstractions.copyto!(backend, nz_values_dev, nz_values)
        T_dev = typeof(starts_dev)
        FixedMultiIndexSet{d,T_dev}(starts_dev, nz_indices_dev, nz_values_dev, max_orders)
        synchronize(backend)
    end
end