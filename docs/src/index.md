# NonLocalMeans.jl

NonLocalMeans.jl provides N-dimensional non-local means denoising for
real-valued arrays, with optional joint denoising across channels.

## Basic usage

```julia
using NonLocalMeans

img = randn(Float32, 64, 64)
output = nonlocalmeans(img; patch = 2, neighborhood = 7, h = 1)
```

Supply a precision array when samples have different confidence values. When a
precision array is supplied, the function returns both the denoised array and
its estimated precision:

```julia
precision = ones(Float32, size(img))
output, output_precision = nonlocalmeans(img, precision)
```

Without a precision array, `nonlocalmeans` returns only the denoised array. The
in-place variant can write estimated precision to an array passed as its fourth
positional argument:

```julia
output = similar(img)
output_precision = similar(img)
nonlocalmeans!(output, img, precision, output_precision)
```

## Patch weights

By default, all samples in each patch contribute equally to its patch distance.
Use `kernel` to weight patch offsets with a function, such as the built-in
Gaussian kernel:

```julia
output = nonlocalmeans(img; patch = 3, kernel = NonLocalMeans.gaussian_kernel)
```

`NonLocalMeans.skip_central_pixel_kernel` is another built-in function; it
excludes the center offset from patch-distance calculations. You can also pass
an `OffsetArray` of weights whose axes match the patch offsets.

## Joint-channel denoising

Set `channel_dim` to the channel axis to compute one set of denoising weights
jointly across channels. Patches and search neighborhoods span the other
dimensions:

```julia
# `image` has dimensions (height, width, channels).
output = nonlocalmeans(image; channel_dim = 3)
```

The same keyword is available on `nonlocalmeans!`.

Colorant arrays are supported when ColorTypes is loaded, and `WeightedArray`
values from WeightedData use their attached precision values automatically.

See the [API reference](api.md) for complete function and patch-kernel
documentation.
