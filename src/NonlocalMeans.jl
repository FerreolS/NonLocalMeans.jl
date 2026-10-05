module NonlocalMeans

export nonlocalmeans, nonlocalmeans!

using StructuredArrays
using KernelAbstractions

"""
    nonlocalmeans(value, precision = ones; patch_radius=2, search_radius=7, h=1,
              skip_zero_offset=true, store_precision=false)

Non-local means denoising of an N-dimensional array using KernelAbstractions.
`patch_radius` and `search_radius` are radii: an integer (same in every
dimension) or a `Tuple` of integers, or a `CartesianIndex` with one radius per dimension.
If `store_precision` is true, the precision of each output sample is stored in a separate array and returned.
"""
function nonlocalmeans(
        value::AbstractArray{T, N},
        precision::AbstractArray{T, N} = FastUniformArray{T, N}(one(T), size(value));
        patch_radius = 2,
        search_radius = 7,
        h::Real = 1,
        skip_zero_offset::Bool = false,
        store_precision = false,
    ) where {T, N}
    output = similar(value)
    output_precision = store_precision ? similar(value) : nothing

    return nonlocalmeans!(
        output,
        value,
        precision,
        output_precision;
        patch_radius,
        search_radius,
        h,
        skip_zero_offset,
    )
end

"""
    nonlocalmeans!(output, value, precision = ones; kwargs...)

In-place version of [`nonlocalmeans`](@ref); returns  `output`. Inputs aliasing
`output` are copied first.

If `output_precision` (an array of the same size as `value`) is given, it is
filled with the precision of each output sample, `(Σ pⱼwⱼ)² / Σ pⱼwⱼ²`,
treating the weights as fixed.
"""
function nonlocalmeans!(
        output::AbstractArray{T, N},
        value::AbstractArray{T, N},
        precision::AbstractArray{T, N} = FastUniformArray{T, N}(one(T), size(value)),
        output_precision = nothing;
        patch_radius = 3,
        search_radius = 7,
        h::Real = 10,
        skip_zero_offset::Bool = false
    ) where {T, N}
    axes(value) == axes(precision) ||
        throw(DimensionMismatch("value and precision must have the same axes"))
    axes(output) == axes(value) ||
        throw(DimensionMismatch("output and value must have the same axes"))
    Base.require_one_based_indexing(output, value, precision)
    patch_radius = _as_radius(patch_radius, Val(N), "patch_radius")
    search_radius = _as_radius(search_radius, Val(N), "search_radius")

    if output_precision !== nothing
        axes(output_precision) == axes(value) ||
            throw(DimensionMismatch("output_precision and value must have the same axes"))
        Base.require_one_based_indexing(output_precision)
        Base.mightalias(output_precision, output) &&
            throw(ArgumentError("output_precision must not alias output"))
    end
    store_precision = output_precision !== nothing
    output_precision = store_precision ? output_precision : output

    source = Base.mightalias(output, value) ? copy(value) : value
    source_precision = Base.mightalias(output, precision) ? copy(precision) : precision

    backend = KernelAbstractions.get_backend(value)
    kernel! = _nlmeans_kernel!(backend)
    kernel!(
        output,
        source,
        source_precision,
        output_precision,
        store_precision,
        patch_radius,
        search_radius,
        skip_zero_offset,
        T(h);
        ndrange = size(source),
    )
    KernelAbstractions.synchronize(backend)
    return output, output_precision
end

function _as_radius(r::Integer, ::Val{N}, name) where {N}
    0 <= r <= typemax(Int) || throw(ArgumentError("$name must be nonnegative"))
    return CartesianIndex(ntuple(_ -> Int(r), Val(N)))
end

function _as_radius(r::CartesianIndex{N}, ::Val{N}, name) where {N}
    all(>=(0), Tuple(r)) || throw(ArgumentError("$name must be nonnegative"))
    return r
end

function _as_radius(r::NTuple{N, T}, ::Val{N}, name) where {N, T}
    all(>=(0), Tuple(r)) || throw(ArgumentError("$name must be nonnegative"))
    return r
end


_as_radius(r, ::Val, name) =
    throw(ArgumentError("$name must be a nonnegative integer, a tuple of nonnegative integers, or a CartesianIndex of matching dimension"))

@kernel function _nlmeans_kernel!(output, value, precision, output_precision, store_precision, patch_radius, search_radius, skip_zero_offset, h)
    index = @index(Global, Cartesian)
    dims = size(value)
    zero_offset = CartesianIndex(ntuple(_ -> 0, ndims(value)))
    numerator = zero(eltype(value))
    denominator = zero(eltype(value))
    squared_sum = zero(eltype(value))

    search_ranges = ntuple(
        dimension ->
        max(1, index[dimension] - search_radius[dimension]):min(dims[dimension], index[dimension] + search_radius[dimension]),
        ndims(value),
    )
    for candidate in CartesianIndices(search_ranges)
        distance = zero(eltype(value))
        distance_weight = zero(eltype(value))

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
            ndims(value),
        )
        for offset in CartesianIndices(patch_ranges)
            skip_zero_offset && offset == zero_offset && continue
            center_patch = index + offset
            candidate_patch = candidate + offset
            p1 = precision[center_patch]
            p2 = precision[candidate_patch]
            patch_weight = (p1 * p2) / (p1 + p2)
            difference = value[center_patch] - value[candidate_patch]
            distance += difference^2 * patch_weight
            distance_weight += patch_weight
        end

        weight = exp(-distance / distance_weight / h)
        candidate_precision = precision[candidate]
        numerator += candidate_precision * value[candidate] * weight
        denominator += candidate_precision * weight
        squared_sum += candidate_precision * weight^2
    end

    output[index] = numerator / denominator
    if store_precision
        output_precision[index] = denominator^2 / squared_sum
    end
end


#include("legacy.jl")

end # module NonlocalMeans
