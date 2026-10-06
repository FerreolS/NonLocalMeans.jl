module NonLocalMeans

export nonlocalmeans, nonlocalmeans!

import StructuredArrays: FastUniformArray
import KernelAbstractions
using KernelAbstractions: @index, @kernel

include("kernel.jl")
"""
    nonlocalmeans(value, precision = ones; patch_radius=2, search_radius=7, h=1,
              skip_zero_offset=false, store_precision=false)

Non-local means denoising of an N-dimensional array using KernelAbstractions.
`patch_radius` and `search_radius` are radii: an integer (same in every
dimension) or a `Tuple` of integers, or a `CartesianIndex` with one radius per dimension.
If `store_precision` is true, the precision of each output sample is stored in a separate array and returned.
With an explicit `precision`, the function returns `(output, output_precision)`,
with `output_precision` equal to `nothing` unless `store_precision` is true.
Without `precision` (unit precision everywhere) only `output` is returned,
unless `store_precision` is true, in which case `(output, output_precision)` is
returned. Inputs must use one-based indexing.

If `channel_dim` is an integer, that dimension of `value` indexes channels that
are denoised jointly (one common set of weights, see
joint-channel denoising); patches and search windows then span the
remaining dimensions only. `precision` and the returned precision always have the
same size as `value` (one precision per channel and pixel). With `channel_dim = nothing` (default) all dimensions are
spatial.
"""
function nonlocalmeans(
        value::AbstractArray{T, N},
        precision::AbstractArray{T, N};
        patch_radius = 2,
        search_radius = 7,
        h::Real = 1,
        skip_zero_offset::Bool = false,
        store_precision = false,
        channel_dim::Union{Nothing, Integer} = nothing,
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
        channel_dim,
    )
end

function nonlocalmeans(value::AbstractArray{T, N}; store_precision = false, kwargs...) where {T, N}
    precision = FastUniformArray{T, N}(one(T), size(value))
    result = nonlocalmeans(value, precision; store_precision, kwargs...)
    return store_precision ? result : first(result)
end

# permutation moving `channel_dim` to the front, keeping the other dims in order
function _channel_permutation(channel_dim::Integer, ::Val{N}) where {N}
    1 <= channel_dim <= N || throw(ArgumentError("channel_dim must be between 1 and $N"))
    return (Int(channel_dim), (d for d in 1:N if d != channel_dim)...)
end

"""
    nonlocalmeans!(output, value, precision = ones; kwargs...)

In-place version of [`nonlocalmeans`](@ref); returns `(output,
output_precision)`. Inputs aliasing `output` are copied first.

If `output_precision` (an array of the same size as `value`) is given, it is
filled with the precision of each output sample, `(Σ pⱼwⱼ)² / Σ pⱼwⱼ²`,
treating the weights as fixed.

`channel_dim` has the same meaning as in [`nonlocalmeans`](@ref).
"""
function nonlocalmeans!(
        output::AbstractArray{T, N},
        value::AbstractArray{T, N},
        precision::AbstractArray{T, N} = FastUniformArray{T, N}(one(T), size(value)),
        output_precision = nothing;
        patch_radius = 2,
        search_radius = 7,
        h::Real = 1,
        skip_zero_offset::Bool = false,
        channel_dim::Union{Nothing, Integer} = nothing,
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
            patch_radius, search_radius, h, skip_zero_offset,
        )
        copyto!(output, permutedims(permuted_output, invperm(perm)))
        permuted_op === nothing || copyto!(output_precision, permutedims(permuted_op, invperm(perm)))
        return output, output_precision
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
        patch_radius, search_radius, h, skip_zero_offset,
    )
    return output, output_precision
end

include("legacy.jl")

end
