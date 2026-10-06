function NLmeans_legacy(value::AbstractArray{T, N}, precision::AbstractArray{T, N} = FastUniformArray{T, N}(one(T), size(value)); patch_size = 2, search_size = 7, h = 1, skip_zero_offset::Bool = false) where {T, N}
    size(value) == size(precision) || throw(DimensionMismatch("value and precision must have the same size"))
    output = similar(value)
    patch_size isa Int && (patch_size = CartesianIndex(ntuple(_ -> patch_size, N)))
    search_size isa Int && (search_size = CartesianIndex(ntuple(_ -> search_size, N)))
    idx = CartesianIndices(value)
    firstI, lastI = first(idx), last(idx)
    for i in idx
        numerator = zero(eltype(value))
        denominator = zero(eltype(value))
        for j in max(firstI, i - search_size):min(lastI, i + search_size)
            w = compute_weights(value, precision, i, j, patch_size, h; skip_zero_offset)
            numerator += precision[j] * value[j] * w
            denominator += precision[j] * w
        end
        output[i] = numerator / denominator
    end
    return output
end

function compute_weights(value::AbstractArray{T, N}, precision::AbstractArray{T, N}, center1::CartesianIndex{N}, center2::CartesianIndex{N}, patch_radius::CartesianIndex{N}, h::Real; skip_zero_offset::Bool = false) where {T, N}
    axes(value) == axes(precision) || throw(DimensionMismatch("value and precision must have the same axes"))
    dist = zero(T)
    weight = zero(T)
    card = zero(Int)
    zeroindex = CartesianIndex(ntuple(_ -> 0, N))

    # clip offsets once so that both shifted patches stay inside the array
    offsets = CartesianIndices(
        map(axes(value), Tuple(center1), Tuple(center2), Tuple(patch_radius)) do ax, c1, c2, r
            (-r:r) ∩ (ax .- c1) ∩ (ax .- c2)
        end
    )

    @inbounds for offset in offsets
        skip_zero_offset && offset == zeroindex && continue
        i = center1 + offset
        j = center2 + offset
        p1p2 = (precision[i] + precision[j])
        w = p1p2 == 0 ? zero(p1p2) : (precision[i] * precision[j]) / p1p2
        dist += (value[i] - value[j])^2 * w
        weight += w
        card += 1
    end

    iszero(weight) && return one(T)
    return exp(- dist / weight / h)
end
