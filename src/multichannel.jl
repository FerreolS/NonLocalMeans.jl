"""
    nonlocalmeans_multichannel(value, precision = ones; kwargs...)

Non-local means of a multichannel array whose **first** dimension indexes the
channels (`size(value) == (C, spatial...)`). All channels are denoised jointly:
one weight per pair of pixels is computed from the precision-weighted squared
differences of all channels, and applied to every channel. `precision` (and the
stored output precision) has the same size as `value`, i.e. one precision per
channel and pixel. Keywords are those of [`nonlocalmeans`](@ref). Returns
`(output, output_precision)`, the latter being `nothing` unless
`store_precision=true`.
"""
function nonlocalmeans_multichannel(
        value::AbstractArray{T, M},
        precision::AbstractArray{T, M} = FastUniformArray{T, M}(one(T), size(value));
        patch_radius = 2,
        search_radius = 7,
        h::Real = 1,
        skip_zero_offset::Bool = false,
        store_precision = false,
    ) where {T, M}
    output_precision = store_precision ? similar(value) : nothing
    return nonlocalmeans_multichannel!(
        similar(value), value, precision, output_precision;
        patch_radius, search_radius, h, skip_zero_offset,
    )
end

"""
    nonlocalmeans_multichannel!(output, value, precision = ones, output_precision = nothing; kwargs...)

In-place version of [`nonlocalmeans_multichannel`](@ref); returns
`(output, output_precision)`.
"""
function nonlocalmeans_multichannel!(
        output::AbstractArray{<:Any, M},
        value::AbstractArray{T, M},
        precision::AbstractArray{T, M} = FastUniformArray{T, M}(one(T), size(value)),
        output_precision = nothing;
        patch_radius = 3,
        search_radius = 7,
        h::Real = 10,
        skip_zero_offset::Bool = false,
    ) where {T, M}
    axes(value) == axes(precision) ||
        throw(DimensionMismatch("value and precision must have the same axes"))
    axes(output) == axes(value) ||
        throw(DimensionMismatch("output and value must have the same axes"))
    Base.require_one_based_indexing(output, value, precision)
    patch_radius = _as_radius(patch_radius, Val(M - 1), "patch_radius")
    search_radius = _as_radius(search_radius, Val(M - 1), "search_radius")

    if output_precision !== nothing
        axes(output_precision) == axes(value) ||
            throw(DimensionMismatch("output_precision and value must have the same axes"))
        Base.require_one_based_indexing(output_precision)
        Base.mightalias(output_precision, output) &&
            throw(ArgumentError("output_precision must not alias output"))
    end
    store_precision = output_precision !== nothing
    denominator = store_precision ? output_precision : output

    source = Base.mightalias(output, value) ? copy(value) : value
    source_precision = Base.mightalias(output, precision) ? copy(precision) : precision

    backend = KernelAbstractions.get_backend(value)
    kernel! = _nlmeans_multichannel_kernel!(backend)
    kernel!(
        output, source, source_precision, denominator, store_precision,
        patch_radius, search_radius, skip_zero_offset, T(h), Val(size(value, 1));
        ndrange = size(value)[2:end],
    )
    KernelAbstractions.synchronize(backend)
    return output, output_precision
end

@kernel function _nlmeans_multichannel_kernel!(output, value, precision, output_precision, store_precision, patch_radius, search_radius, skip_zero_offset, h, ::Val{C}) where {C}
    index = @index(Global, Cartesian)
    dims = size(value)[2:end]
    nspatial = length(dims)
    zero_offset = CartesianIndex(ntuple(_ -> 0, nspatial))
    z = zero(eltype(value))
    channels = ntuple(identity, Val(C))
    numerator = ntuple(_ -> z, Val(C))
    denom = ntuple(_ -> z, Val(C))
    squares = ntuple(_ -> z, Val(C))

    search_ranges = ntuple(
        dimension ->
        max(1, index[dimension] - search_radius[dimension]):min(dims[dimension], index[dimension] + search_radius[dimension]),
        nspatial,
    )
    @inbounds for candidate in CartesianIndices(search_ranges)
        distance = z
        distance_weight = z

        patch_ranges = ntuple(
            dimension -> max(
                -patch_radius[dimension],
                1 - index[dimension],
                1 - candidate[dimension],
            ):min(
                patch_radius[dimension],
                dims[dimension] - index[dimension],
                dims[dimension] - candidate[dimension],
            ),
            nspatial,
        )
        for offset in CartesianIndices(patch_ranges)
            skip_zero_offset && offset == zero_offset && continue
            center_patch = index + offset
            candidate_patch = candidate + offset
            for c in channels
                p1 = @inbounds precision[c, center_patch]
                p2 = @inbounds precision[c, candidate_patch]
                patch_weight = (p1 * p2) / (p1 + p2)
                difference = @inbounds value[c, center_patch] - value[c, candidate_patch]
                distance += difference^2 * patch_weight
                distance_weight += patch_weight
            end
        end

        weight = exp(-distance / distance_weight / h)
        pc = ntuple(c -> @inbounds(precision[c, candidate]), Val(C))
        vc = ntuple(c -> @inbounds(value[c, candidate]), Val(C))
        numerator = map((n, p, v) -> n + p * v * weight, numerator, pc, vc)
        denom = map((d, p) -> d + p * weight, denom, pc)
        squares = map((q, p) -> q + p * weight^2, squares, pc)
    end

    for c in 1:C
        output[c, index] = numerator[c] / denom[c]
        if store_precision
            output_precision[c, index] = denom[c]^2 / squares[c]
        end
    end
end
