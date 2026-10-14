module NonLocalMeans

import StructuredArrays: FastUniformArray
import KernelAbstractions
using KernelAbstractions: @index, @kernel, @Const
using OffsetArrays: OffsetArray
using Adapt: adapt
using LinearAlgebra: norm


export nonlocalmeans, nonlocalmeans!
if VERSION >= v"1.11"
    Core.eval(
        @__MODULE__, Expr(
            :public,
            :skip_central_pixel_kernel,
            :gaussian_kernel
        )
    )
else
    @eval begin
        export skip_central_pixel_kernel,
            gaussian_kernel
    end
end

include("kernel.jl")
"""
    nonlocalmeans(value[, precision]; patch = 2, neighborhood = 7, h = 1,
                  kernel = nothing, channel_dim = nothing)

Apply non-local means denoising to an `N`-dimensional real-valued array using
`KernelAbstractions.jl`.

`patch` is the patch radius: an integer (same in every dimension), a `Tuple` of
integers, or a `CartesianIndex` with one radius per dimension. `neighborhood`
defines the search window and accepts the same radius forms, or a
`CartesianIndices` of offsets relative to each sample, e.g.
`CartesianIndices((-2:2, 0:3))` for an asymmetric window.

`kernel` optionally applies spatial weights within each patch. It can be a
function of a spatial offset tuple, or an `OffsetArray` whose axes match the
patch offsets. The function is evaluated at each patch offset. `nothing`
(default) gives uniform patch weights; [`gaussian_kernel`](@ref) and
[`skip_central_pixel_kernel`](@ref) are provided as useful functions.

If `precision` is supplied, it weights patch comparisons and the output
averages, and the function returns `(output, output_precision)`. If it is
omitted, unit precision is used and only `output` is returned. The comparison
weight is `exp(-patch_distance / h^2)`, where `patch_distance` is the precision-
and kernel-weighted mean squared difference between patches.

Inputs must use one-based indexing. If `channel_dim` is an integer, that
dimension is treated as the channel axis and channels are denoised jointly using
a single set of weights; patches and search windows span the remaining
dimensions. With `channel_dim = nothing` (default), all dimensions are spatial.
"""
function nonlocalmeans(
        value::AbstractArray{T, N},
        precision::Union{AbstractArray{T, N}, Nothing} = nothing;
        patch = 2,
        kernel = nothing,
        neighborhood = 7,
        h::Real = 1,
        channel_dim::Union{Nothing, Integer} = nothing,
    ) where {T, N}
    output = similar(value)
    if precision isa AbstractArray{T, N}
        output_precision = similar(precision)
        return nonlocalmeans!(output, value, precision, output_precision; patch, kernel, neighborhood, h, channel_dim)
    else
        return nonlocalmeans!(output, value; patch, kernel, neighborhood, h, channel_dim)
    end
end


# permutation moving `channel_dim` to the front, keeping the other dims in order
function _channel_permutation(channel_dim::Integer, ::Val{N}) where {N}
    1 <= channel_dim <= N || throw(ArgumentError("channel_dim must be between 1 and $N"))
    return (Int(channel_dim), (d for d in 1:N if d != channel_dim)...)
end

"""
    nonlocalmeans!(output, value[, precision[, output_precision]];
                  patch = 2, neighborhood = 7, h = 1, kernel = nothing,
                  channel_dim = nothing)

In-place version of [`nonlocalmeans`](@ref). Inputs aliasing `output` are copied
first.

The keywords have the same meanings as in [`nonlocalmeans`](@ref). If
`output_precision` is supplied as the fourth positional argument, it is filled
with the estimated precision of each output sample,
`(Σ pⱼwⱼ)² / Σ pⱼwⱼ²`, treating the weights as fixed, and the function returns
`(output, output_precision)`. Otherwise, it returns `output`.

`channel_dim` has the same meaning as in [`nonlocalmeans`](@ref).
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
    ) where {T, N}
    selected_area = nothing
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


"""
    skip_central_pixel_kernel(offset)

Return `false` for the zero offset and `true` for every other offset.

Pass this function as `kernel` to exclude the center of each patch from the
patch-distance calculation.
"""
skip_central_pixel_kernel(p::NTuple{N}) where {N} = (p != ntuple(_ -> 0, N))

"""
    gaussian_kernel(offset; fwhm = 1.0)

Evaluate a Gaussian spatial weight for a patch offset:
`exp(-4 * log(2) * norm(offset)^2 / fwhm^2)`.

Pass this function as `kernel` to weight patches spatial distance.
"""
gaussian_kernel(r::NTuple; fwhm = 1.0) = exp(-4 * log(2) * norm(r)^2 / fwhm^2)


include("legacy.jl")

end
