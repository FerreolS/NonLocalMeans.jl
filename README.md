# NonLocalMeans.jl

[![Build Status][github-ci-img]][github-ci-url] [![Coverage][codecov-img]][codecov-url] [![Aqua QA][aqua-img]][aqua-url] [![Docs][docs-img]][docs-url]

`NonLocalMeans.jl` is a Julia package for non-local means denoising of
real-valued arrays of any dimensionality, implemented with
[KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl).

```julia
using NonLocalMeans

denoised, _ = nonlocalmeans(value; patch_radius = 2, search_radius = 7, h = 1)
nonlocalmeans!(output, value; patch_radius = 2, search_radius = 7, h = 1)
```

Each sample is replaced by a weighted average of the samples in its search
neighborhood. `precision` is optional and defaults to a uniform array. The
functions return `(output, output_precision)`. The second value is `nothing`
unless `store_precision=true` or an output-precision array is provided to the
in-place function. The weights are `exp(-patch_distance / h)`, where the patch
distance is the precision-weighted mean squared difference between patches.

- `patch_radius` and `search_radius` are nonnegative integer radii. A tuple or
  `CartesianIndex` can specify a different radius in each dimension.
- `h` controls the weight decay.
- `skip_zero_offset` (default `false`) omits the patch center from the distance.
- The KernelAbstractions backend is inferred from `value`; regular Julia
  arrays use the CPU backend.

`nonlocalmeans!` writes into `output`; if `output` aliases an input, that input
is copied before the kernel is launched. Inputs must use one-based indexing.
Set `channel_dim` to the channel axis to denoise channels jointly; patches and
search windows span the remaining dimensions.

Colorant arrays are supported through the optional ColorTypes extension.
WeightedArray values from WeightedData can be passed directly; their precision
is used for denoising and returned with the result.


[github-ci-img]: https://github.com/FerreolS/NonLocalMeans.jl/actions/workflows/CI.yml/badge.svg?branch=master
[github-ci-url]: https://github.com/FerreolS/NonLocalMeans.jl/actions/workflows/CI.yml?query=branch%3Amaster
[codecov-img]: http://codecov.io/github/FerreolS/NonLocalMeans.jl/coverage.svg?branch=master
[codecov-url]: http://codecov.io/github/FerreolS/NonLocalMeans.jl?branch=master
[aqua-img]: https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg
[aqua-url]: https://github.com/JuliaTesting/Aqua.jl
[docs-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-url]: https://ferreols.github.io/NonLocalMeans.jl/dev/
