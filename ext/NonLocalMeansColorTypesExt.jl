module NonLocalMeansColorTypesExt

import NonLocalMeans: nonlocalmeans
using ColorTypes: Colorant, base_colorant_type

# (channels, spatial...) float array and the colorant type to convert back to
function _to_channels(img::AbstractArray{C, N}) where {C <: Colorant, N}
    T = float(eltype(C))
    raw = reinterpret(reshape, eltype(C), img)
    data = ndims(raw) == N ? reshape(T.(raw), 1, size(img)...) : T.(raw)
    return data, T
end

function _to_colors(out, ::Type{C}, ::Type{T}) where {C <: Colorant, T}
    out = size(out, 1) == 1 && length(C) == 1 ? dropdims(out; dims = 1) : out
    return collect(reinterpret(reshape, base_colorant_type(C){T}, out))
end

"""
    nonlocalmeans(img::AbstractArray{<:Colorant}; kwargs...)

Denoise a color image (e.g. `Array{RGB{Float32},2}`). The color channels are
denoised jointly using `channel_dim=1`. Returns `output`.
"""
function nonlocalmeans(img::AbstractArray{C, N}; kwargs...) where {C <: Colorant, N}
    data, T = _to_channels(img)
    out, _ = nonlocalmeans(data; channel_dim = 1, kwargs...)
    return _to_colors(out, C, T)
end


end
