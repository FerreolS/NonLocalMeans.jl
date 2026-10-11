# NonLocalMeans.jl

[![Build Status][github-ci-img]][github-ci-url] [![Coverage][codecov-img]][codecov-url] [![Aqua QA][aqua-img]][aqua-url] [![Docs][docs-img]][docs-url]

`NonLocalMeans.jl` is a Julia package for non-local means denoising of
real-valued arrays of any dimensionality, implemented with
[KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl).

```julia
using NonLocalMeans

img = randn(Float32, 64, 64)

denoised = nonlocalmeans(img; patch = 2, neighborhood = 7, h = 1)
nonlocalmeans!(output, img; patch = 2, neighborhood = 7, h = 1)
```

Each sample is replaced by a weighted average of the samples in its search
neighborhood. `precision` is optional and defaults to a uniform array. The
functions return `output` unless `store_precision = true` (or an
`output_precision` array is supplied to the in-place variant), in which case they
return `(output, output_precision)`. The weights are `exp(-patch_distance / h)`,
where the patch distance is the precision-weighted mean squared difference
between patches.

- `patch` is a nonnegative integer radius. A tuple or `CartesianIndex` can
  specify a different radius in each dimension.
- `neighborhood` defines the search window: a radius (same forms as `patch`) or
a `CartesianIndices` of offsets, e.g. `CartesianIndices((-2:2, 0:3))` for an
asymmetric window.
- `h` controls the weight decay.
- `skip_zero_offset` (default `false`) omits the center pixel from the distance
  computation.
- `selected_area` limits denoising to a subset of spatial coordinates, e.g. a
  `CartesianIndices` slice or an array of `CartesianIndex`s.

`nonlocalmeans!` writes into `output`; if `output` aliases an input, that input is
copied before the kernel is launched. Inputs must use one-based indexing. Set
`channel_dim` to the channel axis to denoise channels jointly; patches and search
windows span the remaining dimensions.

Colorant arrays are supported through the optional ColorTypes extension.
WeightedArray values from WeightedData can be passed directly; their precision is
used for denoising and returned with the result.

[github-ci-img]: https://github.com/FerreolS/NonLocalMeans.jl/actions/workflows/CI.yml/badge.svg?branch=master
[github-ci-url]: https://github.com/FerreolS/NonLocalMeans.jl/actions/workflows/CI.yml?query=branch%3Amaster
[codecov-img]: http://codecov.io/github/FerreolS/NonLocalMeans.jl/coverage.svg?branch=master
[codecov-url]: http://codecov.io/github/FerreolS/NonLocalMeans.jl?branch=master
[aqua-img]: https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg
[aqua-url]: https://github.com/JuliaTesting/Aqua.jl
[docs-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-url]: https://ferreols.github.io/NonLocalMeans.jl/dev/
