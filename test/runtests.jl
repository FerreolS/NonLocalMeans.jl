using NonlocalMeans
using ColorTypes
using WeightedData
using Aqua
using ExplicitImports: check_no_implicit_imports, check_no_stale_explicit_imports
import WeightedData: get_value, get_precision
using Test

@testset "package quality" begin
    Aqua.test_all(NonlocalMeans; ambiguities = false)
    check_no_implicit_imports(NonlocalMeans)
    check_no_stale_explicit_imports(NonlocalMeans)
end

# brute-force reference: returns (output, output_precision)
function reference(value, precision; patch_radius, search_radius, h, skip_zero_offset = false)
    R = CartesianIndices(value)
    pr = CartesianIndex(ntuple(_ -> patch_radius, ndims(value)))
    sr = CartesianIndex(ntuple(_ -> search_radius, ndims(value)))
    out = similar(value)
    outp = similar(value)
    for i in R
        num = den = sq = zero(eltype(value))
        for j in R[max(first(R), i - sr):min(last(R), i + sr)]
            dist = wsum = zero(eltype(value))
            for o in CartesianIndices(ntuple(d -> -patch_radius:patch_radius, ndims(value)))
                skip_zero_offset && iszero(o) && continue
                a, b = i + o, j + o
                (a in R && b in R) || continue
                p1, p2 = precision[a], precision[b]
                w = p1 * p2 / (p1 + p2)
                dist += (value[a] - value[b])^2 * w
                wsum += w
            end
            w = exp(-dist / wsum / h)
            num += precision[j] * value[j] * w
            den += precision[j] * w
            sq += precision[j] * w^2
        end
        out[i] = num / den
        outp[i] = den^2 / sq
    end
    return out, outp
end

@testset "nonlocalmeans" begin
    for image in (fill(2.5, 7), fill(2.5, 5, 6), fill(2.5, 3, 4, 2))
        @test nonlocalmeans(image; patch_radius = 1, search_radius = 2)[1] ≈ image
    end

    clean = fill(1.0, 9, 9)
    noisy = [1 + 0.2 * (-1)^(i + j) for i in 1:9, j in 1:9]
    result = nonlocalmeans(noisy; patch_radius = 1, search_radius = 2, h = 0.4)[1]
    @test sum(abs2, result .- clean) < sum(abs2, noisy .- clean)
end

@testset "reference implementation, N-D" begin
    values = (
        Float32.(1:9) .^ 1.5f0,
        reshape(Float32.(1:30) .^ 1.2f0, 5, 6),
        reshape(Float32.(1:60), 3, 4, 5),
    )
    for value in values
        precision = reshape(Float32.(0.2 .+ mod.(1:length(value), 5)), size(value))
        for patch_radius in (1, 3), search_radius in (0, 2), skip_zero_offset in (true, false)
            kw = (; patch_radius, search_radius, h = 4, skip_zero_offset)
            expected, expected_precision = reference(value, precision; kw...)
            actual, actual_precision = nonlocalmeans(value, precision; kw..., store_precision = true)
            @test actual ≈ expected
            @test actual_precision ≈ expected_precision
        end
    end

    # An empty patch comparison has no evidence of a difference, so it gets
    # unit weight rather than propagating NaNs.
    singleton = reshape(Float32[3], 1, 1)
    @test nonlocalmeans(singleton; patch_radius = 0, search_radius = 0, skip_zero_offset = true)[1] == singleton
end

@testset "in place and aliasing" begin
    value = rand(Float32, 5, 6)
    precision = rand(Float32, 5, 6) .+ 1
    default_expected = nonlocalmeans(value, precision)[1]
    default_output = similar(value)
    nonlocalmeans!(default_output, value, precision)
    @test default_output ≈ default_expected

    kw = (patch_radius = 1, search_radius = 2, h = 4)
    expected = nonlocalmeans(value, precision; kw...)[1]

    output = similar(value)
    @test nonlocalmeans!(output, value, precision; kw...)[1] === output
    @test output ≈ expected

    aliased_value = copy(value)
    nonlocalmeans!(aliased_value, aliased_value, precision; kw...)
    @test aliased_value ≈ expected

    aliased_precision = copy(precision)
    nonlocalmeans!(aliased_precision, value, aliased_precision; kw...)
    @test aliased_precision ≈ expected

    @test_throws DimensionMismatch nonlocalmeans(zeros(3, 3), ones(2, 3))
    @test_throws ArgumentError nonlocalmeans(zeros(3, 3), ones(3, 3); patch_radius = -1)
    @test_throws ArgumentError nonlocalmeans(zeros(3, 3), ones(3, 3); search_radius = -1)
    @test_throws DimensionMismatch nonlocalmeans!(zeros(2, 3), zeros(3, 3), ones(3, 3))
end

@testset "output_precision" begin
    x = rand(Float32, 9, 8)
    p = rand(Float32, 9, 8) .+ 1
    op = similar(x)
    a, ret = nonlocalmeans!(similar(x), x, p, op; patch_radius = 1, search_radius = 2, h = 1)
    @test ret === op
    @test a ≈ nonlocalmeans(x, p; patch_radius = 1, search_radius = 2)[1]
    @test nonlocalmeans(x, p; store_precision = true)[2] isa Matrix{Float32}
    @test nonlocalmeans(x, p)[2] === nothing
    # equal weights: precisions add up over the search window
    nonlocalmeans!(similar(x), x, p, op; patch_radius = 0, search_radius = 1, h = 1.0f12)
    @test op[5, 4] ≈ sum(p[4:6, 3:5]) rtol = 1.0e-4
    @test_throws ArgumentError nonlocalmeans!(x, x, p, x)
    @test_throws ArgumentError nonlocalmeans!(similar(x), x, p, x; patch_radius = 1)
end

@testset "multichannel / channel_dim" begin

    default_value = reshape(Float32.(1:90) .^ 1.2f0, 3, 5, 6)
    default_precision = reshape(Float32.(1:90) ./ 90 .+ 1, 3, 5, 6)
    default_expected = nonlocalmeans(default_value, default_precision; channel_dim = 1)[1]
    default_output = similar(default_value)
    nonlocalmeans!(default_output, default_value, default_precision; channel_dim = 1)
    @test default_output ≈ default_expected

    kw = (patch_radius = 1, search_radius = 2, h = 0.5f0)
    x = rand(Float32, 3, 9, 8)
    P = rand(Float32, 3, 9, 8) .+ 1
    ref, refp = nonlocalmeans(x, P; channel_dim = 1, store_precision = true, kw...)

    # one channel equals the plain function
    v = rand(Float32, 9, 8)
    q = rand(Float32, 9, 8) .+ 1
    a, ap = nonlocalmeans(v, q; store_precision = true, kw...)
    b, bp = nonlocalmeans(
        reshape(v, 1, 9, 8), reshape(q, 1, 9, 8);
        channel_dim = 1, store_precision = true, kw...,
    )
    @test b[1, :, :] ≈ a && bp[1, :, :] ≈ ap

    # channels share one weight, so differ from independent denoising
    indep = cat((nonlocalmeans(x[c, :, :], P[c, :, :]; kw...)[1] for c in 1:3)...; dims = 3)
    @test !(permutedims(indep, (3, 1, 2)) ≈ ref)

    z = permutedims(x, (2, 1, 3))
    Pz = permutedims(P, (2, 1, 3))
    o, op = nonlocalmeans(z, Pz; channel_dim = 2, store_precision = true, kw...)
    @test o ≈ permutedims(ref, (2, 1, 3))
    @test op ≈ permutedims(refp, (2, 1, 3))

    out, op2 = similar(z), similar(z)
    nonlocalmeans!(out, z, Pz, op2; channel_dim = 2, kw...)
    @test out ≈ o && op2 ≈ op

    o3 = copy(z)
    nonlocalmeans!(o3, o3; channel_dim = 2, kw...)
    @test o3 ≈ nonlocalmeans(z; channel_dim = 2, kw...)[1]

    @test_throws ArgumentError nonlocalmeans(v; channel_dim = 3)
    @test_throws DimensionMismatch nonlocalmeans(z, rand(Float32, size(z) .+ 1); channel_dim = 2)
end

@testset "ColorTypes extension" begin
    img = [RGB{Float32}(rand(3)...) for i in 1:10, j in 1:12]
    kw = (patch_radius = 1, search_radius = 2)
    o = nonlocalmeans(img; kw...)
    @test o isa Matrix{RGB{Float32}}
    a = permutedims(Float32.(reinterpret(reshape, Float32, img)), (2, 3, 1))
    r = nonlocalmeans(a; channel_dim = 3, kw...)[1]
    @test red.(o) ≈ r[:, :, 1] && green.(o) ≈ r[:, :, 2] && blue.(o) ≈ r[:, :, 3]
end

@testset "WeightedData extension" begin
    x = rand(8, 9)
    p = rand(8, 9) .+ 1
    kw = (patch_radius = 1, search_radius = 2, h = 1.0)
    expected, expected_precision = nonlocalmeans(x, p; kw..., store_precision = true)

    w = nonlocalmeans(WeightedArray(x, p); kw...)
    @test w isa WeightedArray
    @test get_value(w) ≈ expected
    @test get_precision(w) ≈ expected_precision

    out = WeightedArray(similar(x), similar(p))
    @test nonlocalmeans!(out, WeightedArray(x, p); kw...) isa WeightedArray
    @test get_value(out) ≈ expected
    @test get_precision(out) ≈ expected_precision
end
