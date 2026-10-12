function _to_CartesianIndices(r::Integer, ::Val{N}, name) where {N}
    return _to_CartesianIndices(ntuple(_ -> Int(r), Val(N)), Val(N), name)
end

function _to_CartesianIndices(r::NTuple{N, <:Integer}, ::Val{N}, name) where {N}
    return _to_CartesianIndices(CartesianIndex(r), Val(N), name)
end

function _to_CartesianIndices(r::CartesianIndex{N}, ::Val{N}, name) where {N}
    r >= CartesianIndex(ntuple(_ -> 0, Val(N))) || throw(ArgumentError("$name must be nonnegative"))
    return -r:r
end


_to_CartesianIndices(r::Union{CartesianIndices{N}, <:AbstractArray{CartesianIndex{N}}}, ::Val{N}, _) where {N} = r

_to_CartesianIndices(r, ::Val, name) =
    throw(ArgumentError("$name must be a nonnegative integer, a tuple of nonnegative integers, or a CartesianIndex of matching dimension"))


function _nonlocalmeans!(
        output::AbstractArray{<:Any, M},
        value::AbstractArray{T, M},
        precision::AbstractArray{T, M} = FastUniformArray{T, M}(one(T), size(value)),
        output_precision = nothing;
        patch = 2,
        kernel = nothing,
        neighborhood = 7,
        h::Real = 1,
        selected_area = nothing,
    ) where {T, M}
    axes(value) == axes(precision) ||
        throw(DimensionMismatch("value and precision must have the same axes"))
    axes(output) == axes(value) ||
        throw(DimensionMismatch("output and value must have the same axes"))
    Base.require_one_based_indexing(output, value, precision)
    patch = _to_CartesianIndices(patch, Val(M - 1), "patch")
    if kernel isa Function
        kernel = OffsetArray(T.(kernel.(Tuple.(patch))), patch)
    elseif kernel isa OffsetArray
        CartesianIndices(kernel) == patch || throw(DimensionMismatch("kernel must have the same size as patch"))
        kernel = T.(kernel)
    elseif kernel !== nothing
        throw(ArgumentError("kernel must be a function or an OffsetArray of matching size to patch"))
    end
    neighborhood = _to_CartesianIndices(neighborhood, Val(M - 1), "neighborhood")
    if isnothing(selected_area)
        selected_area = CartesianIndices(ntuple(n -> axes(value, n + 1), M - 1))
    end

    if !isnothing(output_precision)
        axes(output_precision) == axes(value) ||
            throw(DimensionMismatch("output_precision and value must have the same axes"))
        Base.require_one_based_indexing(output_precision)
        Base.mightalias(output_precision, output) &&
            throw(ArgumentError("output_precision must not alias output"))
        (Base.mightalias(output_precision, value) || Base.mightalias(output_precision, precision)) &&
            throw(ArgumentError("output_precision must not alias value or precision"))
    end
    store_precision = !isnothing(output_precision)
    denominator = store_precision ? output_precision : output

    source = Base.mightalias(output, value) ? copy(value) : value
    source_precision = Base.mightalias(output, precision) ? copy(precision) : precision

    backend = KernelAbstractions.get_backend(value)
    _nlmeans_kernel!(backend)(
        output, source, source_precision, denominator, store_precision,
        patch, kernel, neighborhood, selected_area, T(h^2), Val(size(value, 1));
        ndrange = size(value)[2:end],
    )
    KernelAbstractions.synchronize(backend)

    isnothing(output_precision) && return output
    return output, output_precision
end

@kernel  function _nlmeans_kernel!(
        output,
        @Const(value::AbstractArray{T, M}),
        @Const(precision::AbstractArray{T, M}),
        output_precision,
        store_precision,
        patch::PT,
        kernel::K,
        neighborhood::NT,
        selected_area::ST,
        h²,
        ::Val{C}
    ) where {C, T, M, PT, NT, ST, K}
    index = @index(Global, Cartesian)
    channels = ntuple(identity, Val(C))
    numerator = ntuple(_ -> zero(T), Val(C))
    denom = ntuple(_ -> zero(T), Val(C))
    squares = ntuple(_ -> zero(T), Val(C))

    patch_CartesianIndices = (PT === CartesianIndices && ST === CartesianIndices)
    search_CartesianIndices = (NT === CartesianIndices && ST === CartesianIndices)
    search_ranges = search_CartesianIndices ? (neighborhood .+ index) ∩ selected_area : neighborhood .+ index

    @inbounds for candidate in search_ranges

        search_CartesianIndices || (candidate ∉ selected_area && continue)
        distance = zero(T)
        distance_weight = zero(T)

        patch_ranges = patch_CartesianIndices ? patch ∩ (selected_area .- index) ∩ (selected_area .- candidate) : patch
        for offset in patch_ranges

            center_patch = index + offset
            patch_CartesianIndices || (center_patch ∉ selected_area && continue)
            candidate_patch = candidate + offset
            patch_CartesianIndices || (candidate_patch ∉ selected_area && continue)

            for c in channels
                p1 = precision[c, center_patch]
                p2 = precision[c, candidate_patch]
                p1p2 = p1 + p2
                patch_weight = (p1p2 == 0 ? zero(p1p2) : (p1 * p2) / p1p2)
                K === Nothing ||  (patch_weight *= kernel[offset])
                difference = value[c, center_patch] - value[c, candidate_patch]
                distance += difference^2 * patch_weight
                distance_weight += patch_weight
            end
        end

        weight = iszero(distance_weight) ? zero(distance_weight) : exp(-distance / distance_weight / h²)
        pc = ntuple(c -> @inbounds(precision[c, candidate]), Val(C))
        vc = ntuple(c -> @inbounds(value[c, candidate]), Val(C))
        numerator = map((n, p, v) -> n + p * v * weight, numerator, pc, vc)
        denom = map((d, p) -> d + p * weight, denom, pc)
        if store_precision
            squares = map((q, p) -> q + p * weight^2, squares, pc)
        end
    end

    @inbounds for c in 1:C
        output[c, index] = numerator[c] / (denom[c] == zero(denom[c]) ? one(denom[c]) : denom[c])
        if store_precision
            output_precision[c, index] = denom[c]^2 / (squares[c] == zero(squares[c]) ? one(squares[c]) : squares[c])
        end
    end
end
