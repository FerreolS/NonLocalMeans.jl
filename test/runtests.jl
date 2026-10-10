using NonLocalMeans
using ColorTypes
using WeightedData
import WeightedData: get_value, get_precision
using Test


@testset "nonlocalmeans" begin
    for value in (fill(2.5, 7), fill(2.5, 5, 6), fill(2.5, 3, 4, 2))
        @test nonlocalmeans(value; patch_radius = 1, neighborhood = 2) ≈ value
    end

    clean = fill(1.0, 9, 9)
    noisy = [1 + 0.2 * (-1)^(i + j) for i in 1:9, j in 1:9]
    result = nonlocalmeans(noisy; patch_radius = 1, neighborhood = 2, h = 0.4)
    @test sum(abs2, result .- clean) < sum(abs2, noisy .- clean)
end

@testset "Legacy implementation, N-D" begin
    values = (
        Float32.(1:9) .^ 1.5f0,
        reshape(Float32.(1:30) .^ 1.2f0, 5, 6),
        reshape(Float32.(1:60), 3, 4, 5),
    )
    for value in values
        for patch_radius in (1, 3), neighborhood in (0, 2), skip_zero_offset in (true, false)
            kw = (; patch_radius, neighborhood, h = 4, skip_zero_offset)
            expected = NonLocalMeans.NLmeans_legacy(value; kw...)
            actual = nonlocalmeans(value; kw...)
            @test actual ≈ expected
        end
    end

    # An empty patch comparison has no evidence of a difference, so it gets
    # unit weight rather than propagating NaNs.
    singleton = reshape(Float32[3], 1, 1)
    @test nonlocalmeans(singleton; patch_radius = 0, neighborhood = 0, skip_zero_offset = true) == reshape(Float32[0], 1, 1)
end

@testset "in place and aliasing" begin
    value = rand(Float32, 5, 6)
    precision = rand(Float32, 5, 6) .+ 1
    default_expected = nonlocalmeans(value, precision)[1]
    default_output = similar(value)
    nonlocalmeans!(default_output, value, precision)
    @test default_output ≈ default_expected

    kw = (patch_radius = 1, neighborhood = 2, h = 4)
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
    @test_throws ArgumentError nonlocalmeans(zeros(3, 3), ones(3, 3); neighborhood = -1)
    @test_throws DimensionMismatch nonlocalmeans!(zeros(2, 3), zeros(3, 3), ones(3, 3))
end

@testset "output_precision" begin
    value = rand(Float32, 9, 8)
    precision = rand(Float32, 9, 8) .+ 1
    op = similar(value)
    a, ret = nonlocalmeans!(similar(value), value, precision, op; patch_radius = 1, neighborhood = 2, h = 1)
    @test ret === op
    @test a ≈ nonlocalmeans(value, precision; patch_radius = 1, neighborhood = 2)[1]
    @test nonlocalmeans(value, precision; store_precision = true)[2] isa Matrix{Float32}
    @test nonlocalmeans(value, precision)[2] === nothing
    # equal weights: precisions add up over the search window
    nonlocalmeans!(similar(value), value, precision, op; patch_radius = 0, neighborhood = 1, h = 1.0f12)
    @test op[5, 4] ≈ sum(precision[4:6, 3:5]) rtol = 1.0e-4
    @test_throws ArgumentError nonlocalmeans!(value, value, precision, value)
    @test_throws ArgumentError nonlocalmeans!(similar(value), value, precision, value; patch_radius = 1)
end

@testset "multichannel / channel_dim" begin

    default_value = reshape(Float32.(1:90) .^ 1.2f0, 3, 5, 6)
    default_precision = reshape(Float32.(1:90) ./ 90 .+ 1, 3, 5, 6)
    default_expected = nonlocalmeans(default_value, default_precision; channel_dim = 1)[1]
    default_output = similar(default_value)
    nonlocalmeans!(default_output, default_value, default_precision; channel_dim = 1)
    @test default_output ≈ default_expected

    kw = (patch_radius = 1, neighborhood = 2, h = 0.5f0)
    value = rand(Float32, 3, 9, 8)
    precision = rand(Float32, 3, 9, 8) .+ 1
    ref, refp = nonlocalmeans(value, precision; channel_dim = 1, store_precision = true, kw...)

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
    indep = cat((nonlocalmeans(value[c, :, :], precision[c, :, :]; kw...)[1] for c in 1:3)...; dims = 3)
    @test !(permutedims(indep, (3, 1, 2)) ≈ ref)

    z = permutedims(value, (2, 1, 3))
    Pz = permutedims(precision, (2, 1, 3))
    o, op = nonlocalmeans(z, Pz; channel_dim = 2, store_precision = true, kw...)
    @test o ≈ permutedims(ref, (2, 1, 3))
    @test op ≈ permutedims(refp, (2, 1, 3))

    out, op2 = similar(z), similar(z)
    nonlocalmeans!(out, z, Pz, op2; channel_dim = 2, kw...)
    @test out ≈ o && op2 ≈ op

    o3 = copy(z)
    nonlocalmeans!(o3, o3; channel_dim = 2, kw...)
    @test o3 ≈ nonlocalmeans(z; channel_dim = 2, kw...)

    @test_throws ArgumentError nonlocalmeans(v; channel_dim = 3)
    @test_throws DimensionMismatch nonlocalmeans(z, rand(Float32, size(z) .+ 1); channel_dim = 2)
end

@testset "ColorTypes extension" begin
    value = [RGB{Float32}(rand(3)...) for i in 1:10, j in 1:12]
    kw = (patch_radius = 1, neighborhood = 2)
    o = nonlocalmeans(value; kw...)
    @test o isa Matrix{RGB{Float32}}
    a = permutedims(Float32.(reinterpret(reshape, Float32, value)), (2, 3, 1))
    r = nonlocalmeans(a; channel_dim = 3, kw...)
    @test red.(o) ≈ r[:, :, 1] && green.(o) ≈ r[:, :, 2] && blue.(o) ≈ r[:, :, 3]
end

@testset "WeightedData extension" begin
    value = rand(8, 9)
    precision = rand(8, 9) .+ 1
    kw = (patch_radius = 1, neighborhood = 2, h = 1.0)
    expected, expected_precision = nonlocalmeans(value, precision; kw..., store_precision = true)

    w = nonlocalmeans(WeightedArray(value, precision); kw...)
    @test w isa WeightedArray
    @test get_value(w) ≈ expected
    @test get_precision(w) ≈ expected_precision

    out = WeightedArray(similar(value), similar(precision))
    @test nonlocalmeans!(out, WeightedArray(value, precision); kw...) isa WeightedArray
    @test get_value(out) ≈ expected
    @test get_precision(out) ≈ expected_precision
end

@testset "neighborhood" begin
    value = rand(Float32, 9, 8)
    kw = (patch_radius = 1, h = 1)
    @test nonlocalmeans(value; neighborhood = 2, kw...) ≈ nonlocalmeans(value; neighborhood = CartesianIndex(2, 2), kw...)
    @test nonlocalmeans(value; neighborhood = 2, kw...) ≈ nonlocalmeans(value; neighborhood = CartesianIndices((-2:2, -2:2)), kw...)
    asym = nonlocalmeans(value; neighborhood = CartesianIndices((0:2, -1:1)), kw...)
    @test asym != nonlocalmeans(value; neighborhood = 2, kw...)
    @test nonlocalmeans!(similar(value), value; neighborhood = CartesianIndices((0:2, -1:1)), kw...)[1] ≈ asym
    @test asym ≈ NonLocalMeans.NLmeans_legacy(value; neighborhood = CartesianIndices((0:2, -1:1)), kw...)
    @test nonlocalmeans(value; neighborhood = (2, 1), kw...) ≈ nonlocalmeans(value; neighborhood = CartesianIndex(2, 1), kw...)
    @test_throws ArgumentError nonlocalmeans(value; neighborhood = CartesianIndices((-1:1,)))
end

@testset "patch window" begin
    value = rand(Float32, 9, 8)
    @test nonlocalmeans(value; patch_radius = (1, 2), neighborhood = 2) ≈ nonlocalmeans(value; patch_radius = CartesianIndices((-1:1, -2:2)), neighborhood = 2)
    @test nonlocalmeans(value; patch_radius = (1, 2), neighborhood = 2) ≈ NonLocalMeans.NLmeans_legacy(value; patch_radius = CartesianIndex(1, 2), neighborhood = 2)
end
