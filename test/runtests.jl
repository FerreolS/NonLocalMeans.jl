using NonlocalMeans
using KernelAbstractions
using Test

@testset "shifted patch distance" begin
    v = [1.0, 2.0, 4.0, 7.0, 11.0]
    w = ones(5)
    r = CartesianIndex(1)
    cw(c1, c2) = NonlocalMeans.compute_weights(
        v, w, CartesianIndex(c1), CartesianIndex(c2), r, 1.0; skip_zero_offset = false
    )
    @test cw(2, 3) ≈ exp(-(1.0 + 4.0 + 9.0) / 3)
    # patches are clipped at the border
    @test cw(1, 2) ≈ exp(-(1.0 + 4.0) / 2)
    @test cw(2, 2) == 1.0
end

@testset "NLmeansKA behavior" begin
    for image in (fill(2.5, 7), fill(2.5, 5, 6), fill(2.5, 3, 4, 2))
        @test nonlocalmeans(image; patch_size = 1, search_size = 2) ≈ image
    end

    clean = fill(1.0, 9, 9)
    noisy = [1 + 0.2 * (-1)^(i + j) for i in 1:9, j in 1:9]
    result = nonlocalmeans(noisy; patch_size = 1, search_size = 2, h = 0.4)
    @test sum(abs2, result .- clean) < sum(abs2, noisy .- clean)
end

@testset "KernelAbstractions implementation" begin
    values = (
        Float32.(1:9),
        reshape(Float32.(1:30), 5, 6),
        reshape(Float32.(1:60), 3, 4, 5),
    )

    for value in values
        precision = reshape(Float32.(0.2 .+ mod.(1:length(value), 5)), size(value))
        for patch_size in (1, 3), search_size in (0, 2)
            for skip_zero_offset in (true, false)
                expected = NonlocalMeans.NLmeans_legacy(
                    value, precision;
                    patch_size, search_size, h = 4, skip_zero_offset
                )
                actual = nonlocalmeans(
                    value, precision;
                    patch_size, search_size, h = 4, skip_zero_offset,
                    backend = KernelAbstractions.CPU()
                )
                @test actual ≈ expected
            end
        end

        expected = nonlocalmeans(value, precision; patch_size = 1, search_size = 2, h = 4)
        output = similar(value)
        @test nonlocalmeans!(
            output, value, precision;
            patch_size = 1, search_size = 2, h = 4
        ) === output
        @test output ≈ expected

        aliased_value = copy(value)
        @test nonlocalmeans!(
            aliased_value, aliased_value, precision;
            patch_size = 1, search_size = 2, h = 4
        ) === aliased_value
        @test aliased_value ≈ expected

        aliased_precision = copy(precision)
        @test nonlocalmeans!(
            aliased_precision, value, aliased_precision;
            patch_size = 1, search_size = 2, h = 4
        ) === aliased_precision
        @test aliased_precision ≈ expected
    end

    @test_throws DimensionMismatch nonlocalmeans(zeros(3, 3), ones(2, 3))
    @test_throws ArgumentError nonlocalmeans(zeros(3, 3), ones(3, 3); patch_size = -1)
    @test_throws ArgumentError nonlocalmeans(zeros(3, 3), ones(3, 3); search_size = -1)
    @test_throws DimensionMismatch nonlocalmeans!(zeros(2, 3), zeros(3, 3), ones(3, 3))
end

@testset "output_precision" begin
    x = rand(Float32, 9, 8)
    p = rand(Float32, 9, 8) .+ 1
    op = similar(x)
    a = nonlocalmeans(x, p; patch_radius = 1, search_radius = 2, output_precision = op)
    @test a ≈ nonlocalmeans(x, p; patch_radius = 1, search_radius = 2)
    # equal weights: precisions add up over the search window
    nonlocalmeans(x, p; patch_radius = 0, search_radius = 1, h = 1.0f12, output_precision = op)
    @test op[5, 4] ≈ sum(p[4:6, 3:5]) rtol = 1.0e-4
    @test_throws ArgumentError nonlocalmeans!(x, x, p; output_precision = x)
end
