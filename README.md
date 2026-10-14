# NonLocalMeans.jl

[![Build Status][github-ci-img]][github-ci-url] [![Coverage][codecov-img]][codecov-url] [![Aqua QA][aqua-img]][aqua-url] [![Docs][docs-img]][docs-url]

This `NonLocalMeans.jl` package implements the (pixel-wise) non-local means denoising algorithm proposed in Buades (2008)[^1]. In this algorithm, each sample is replaced by a weighted average of the samples in its search neighborhood. The package exports only the function `nonlocalmeans`  and its  in-place counter part `nonlocalmeans!`.  It can operate on any array of any dimension and is almost allocations free (few kB). Thanks to [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl) it can be runned on  CPU and GPUs backends.

```julia
using NonLocalMeans, TestImages
img = Float64.(testimage("cameraman"))*255 + 5 .* randn(Float64,512,512)
denoised = nonlocalmeans(img)
output = similar(img)
nonlocalmeans!(output, img)
```

`nonlocalmeans!` writes into `output`; if `output` aliases an input, that input is copied before the kernel is launched.

## Keywords

- `h = 1` controls the weight decay in patch weight computation:
  `exp(-patch_distance / h^2)`.
- `patch = 2` controls the size of the patch.
- `kernel = nothing` optionally supplies spatial weights within each patch; the default uses uniform weights. It can be a function of an offset `NTuple` or an `OffsetArray` whose axes match the patch offsets. The `NonLocalMeans.gaussian_kernel` and `NonLocalMeans.skip_central_pixel_kernel` functions provide Gaussian weighting and center-pixel exclusion, respectively.
- `neighborhood` controls the size of the search window. It accepts the same
 radius forms as `patch`.
- `channel_dim = nothing` specifies the channel axis for joint channel
 denoising. Patches and search windows span the remaining dimensions.

`patch` and `neighborhood` accept:

- An `Int` (non negative) radius, applied in every spatial dimension.
- An `NTuple{N,Int}` or `CartesianIndex{N}` radius, with one radius per spatial dimension. For example, `patch = (1, 1, 0)` uses a 3-by-3 patch in  the first two dimensions and no extent in the third.
- `CartesianIndices` or an array of `CartesianIndex` offsets to specify a custom patch or search window.

## Uncertainties estimates

Each sample can be associated with a precision (inverse variance). When a precision array is supplied, `nonlocalmeans` can also return the estimated precision of the denoised array.

```julia
denoised, denoised_precision = nonlocalmeans(img, precision)
nonlocalmeans!(denoised, img, precision, denoised_precision)
```

The computed precision is an approximation derived from the weighted average of the samples; it does not propagate uncertainty from the estimation of the weights. Precision values are also useful for excluding unmeasured or erroneous pixels by setting their precision to zero.

## Extensions

### ColorTypes

Colorant arrays are supported through the optional [ColorTypes](https://github.com/JuliaGraphics/ColorTypes.jl) extension. Color channels are processed jointly using color patches.

```julia
using NonLocalMeans, ImageInTerminal, TestImages
img = testimage("mandril_color")
data = img .+ 0.5 .* rand(eltype(img), size(img))
nonlocalmeans(data; h = 0.1)
```

### WeightedData

When [WeightedData](https://github.com/FerreolS/WeightedData.jl) is loaded, `WeightedArray` values can be passed directly; their precision is used for denoising and returned with the result.

[github-ci-img]: https://github.com/FerreolS/NonLocalMeans.jl/actions/workflows/CI.yml/badge.svg?branch=master
[github-ci-url]: https://github.com/FerreolS/NonLocalMeans.jl/actions/workflows/CI.yml?query=branch%3Amaster
[codecov-img]: http://codecov.io/github/FerreolS/NonLocalMeans.jl/coverage.svg?branch=master
[codecov-url]: http://codecov.io/github/FerreolS/NonLocalMeans.jl?branch=master
[aqua-img]: https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg
[aqua-url]: https://github.com/JuliaTesting/Aqua.jl
[docs-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-url]: https://ferreols.github.io/NonLocalMeans.jl/dev/

[^1]: A. Buades, B. Coll and J. . -M. Morel, "A non-local algorithm for image denoising," 2005 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR'05), San Diego, CA, USA, 2005, pp. 60-65 vol. 2, [doi](https://doi.org/10.1109/CVPR.2005.38)
