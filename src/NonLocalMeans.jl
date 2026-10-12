module NonLocalMeans

export nonlocalmeans, nonlocalmeans!
public skip_central_pixel_kernel, gaussian_kernel

import StructuredArrays: FastUniformArray
import KernelAbstractions
using KernelAbstractions: @index, @kernel, @Const
using OffsetArrays: OffsetArray
using Adapt:adapt
using LinearAlgebra: norm

include("kernel.jl")
"""
    nonlocalmeans(value[, precision]; patch = 2, neighborhood = 7, h = 1,
                   store_precision = false,
                  channel_dim = nothing, selected_area = nothing)

Apply non-local means denoising to an `N`-dimensional real-valued array using
`KernelAbstractions.jl`.

`patch` is the patch radius: an integer (same in every dimension), a `Tuple` of
integers, or a `CartesianIndex` with one radius per dimension. `neighborhood`
defines the search window and accepts the same radius forms, or a
`CartesianIndices` of offsets relative to each sample, e.g.
`CartesianIndices((-2:2, 0:3))` for an asymmetric window.

If `precision` is supplied, it weights the patch distances and is used to
compute the denoising weights. When `precision` is omitted, a unit precision array
is used and only `output` is returned unless `store_precision = true`.
With an explicit `precision`, the function returns `(output, output_precision)`;
`output_precision` is `nothing` unless `store_precision = true`.

`selected_area` optionally restricts the denoising to a subset of spatial
positions; pass `nothing` (default) to use the full spatial domain or a
`CartesianIndices`/array of `CartesianIndex`s to limit the valid search and patch
locations. The weights are `exp(-patch_distance / h)`, where `patch_distance` is
the precision-weighted mean squared difference between patches.

Inputs must use one-based indexing. If `channel_dim` is an integer, that
permutation is treated as the channel axis and denoised jointly using a single
set of weights; patches and search windows then span the remaining dimensions.
`precision` and any returned precision array have the same size as `value` (one
precision value per channel and sample). With `channel_dim = nothing` (default),
all dimensions are spatial.
"""
function nonlocalmeans(
        value::AbstractArray{T, N},
        precision::AbstractArray{T, N};
        patch = 2,
        kernel = nothing,
        neighborhood = 7,
        h::Real = 1,
        store_precision = false,
        channel_dim::Union{Nothing, Integer} = nothing,
        selected_area = nothing
    ) where {T, N}
    output = similar(value)
    output_precision = store_precision ? similar(value) : nothing

    return nonlocalmeans!(
        output,
        value,
        precision,
        output_precision;
        patch,
        kernel,
        neighborhood,
        h,
        channel_dim,
        selected_area,
    )
end

function nonlocalmeans(value::AbstractArray{T, N}; store_precision = false, kwargs...) where {T, N}
    precision = FastUniformArray{T, N}(one(T), size(value))
    return nonlocalmeans(value, precision; store_precision, kwargs...)
end

# permutation moving `channel_dim` to the front, keeping the other dims in order
function _channel_permutation(channel_dim::Integer, ::Val{N}) where {N}
    1 <= channel_dim <= N || throw(ArgumentError("channel_dim must be between 1 and $N"))
    return (Int(channel_dim), (d for d in 1:N if d != channel_dim)...)
end

"""
    nonlocalmeans!(output, value[, precision]; patch = 2, neighborhood = 7,
                  h = 1, channel_dim = nothing,
                  selected_area = nothing, output_precision = nothing)

In-place version of [`nonlocalmeans`](@ref). Inputs aliasing `output` are copied
first.

When `output_precision` is provided, it is filled with the estimated precision of
each output sample, `(Σ pⱼwⱼ)² / Σ pⱼwⱼ²`, treating the weights as fixed.
The return value is `output` unless `output_precision` is supplied, in which case
it is `(output, output_precision)`.

`channel_dim` has the same meaning as in [`nonlocalmeans`](@ref). `selected_area`
restricts valid patch and search locations to the supplied spatial subset.
"""
function nonlocalmeans!(
        output::AbstractArray{T, N},
        value::AbstractArray{T, N},
        precision::AbstractArray{T, N} = FastUniformArray{T, N}(one(T), size(value)),
        output_precision = nothing;
        patch = 2,
        kernel = nothing,
        neighborhood = 7,
        h::Real = 1,
        channel_dim::Union{Nothing, Integer} = nothing,
        selected_area = nothing,
    ) where {T, N}
    if channel_dim !== nothing
        axes(output) == axes(value) ||
            throw(DimensionMismatch("output and value must have the same axes"))
        axes(precision) == axes(value) ||
            throw(DimensionMismatch("value and precision must have the same axes"))
        perm = _channel_permutation(channel_dim, Val(N))
        permuted_output = permutedims(output, perm)
        permuted_precision = permutedims(precision, perm)
        permuted_op = output_precision === nothing ? nothing : permutedims(output_precision, perm)
        _nonlocalmeans!(
            permuted_output, permutedims(value, perm), permuted_precision, permuted_op;
            patch, kernel, neighborhood, h, selected_area,
        )
        copyto!(output, permutedims(permuted_output, invperm(perm)))
        permuted_op === nothing || copyto!(output_precision, permutedims(permuted_op, invperm(perm)))
        if isnothing(output_precision)
            return output
        else
            return (output, output_precision)
        end
    end
    axes(value) == axes(precision) ||
        throw(DimensionMismatch("value and precision must have the same axes"))
    axes(output) == axes(value) ||
        throw(DimensionMismatch("output and value must have the same axes"))
    Base.require_one_based_indexing(output, value, precision)
    if output_precision !== nothing
        axes(output_precision) == axes(value) ||
            throw(DimensionMismatch("output_precision and value must have the same axes"))
        Base.require_one_based_indexing(output_precision)
    end
    # a single channel: add a leading dimension of size one
    lift(a) = reshape(a, 1, size(a)...)
    _nonlocalmeans!(
        lift(output), lift(value), lift(precision),
        output_precision === nothing ? nothing : lift(output_precision);
        patch, kernel, neighborhood, h, selected_area,
    )
    if isnothing(output_precision)
        return output
    else
        return (output, output_precision)
    end
end


skip_central_pixel_kernel(p::NTuple{N}) where {N} = (p != ntuple(_ -> 0, N))

gaussian_kernel(r::NTuple; σ = 1.0) = exp(-norm(r)^2 / (2 * σ^2))

include("legacy.jl")

end
