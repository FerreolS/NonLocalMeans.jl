# NonLocalMeans.jl

NonLocalMeans.jl provides N-dimensional non-local means denoising for
real-valued arrays, with optional joint denoising across channels.

```@docs
nonlocalmeans
nonlocalmeans!
```

## Basic usage

```julia
using NonLocalMeans

output = nonlocalmeans(image; patch_radius = 2, neighborhood = 7, h = 1)
```

Supply a precision array when samples have different confidence values.
Request estimated output precision with `store_precision=true`:

```julia
output, output_precision = nonlocalmeans(
    image,
    precision;
    store_precision = true,
)
```

## Joint-channel denoising

Set `channel_dim` to the channel axis to compute one set of denoising weights
jointly across channels. Patches and search neighborhoods span the other
dimensions:

```julia
# `image` has dimensions (height, width, channels).
output = nonlocalmeans(image; channel_dim = 3)
```

The same keyword is available on `nonlocalmeans!`. Colorant arrays are also
supported when ColorTypes is loaded.

See the [API reference](api.md) for complete function documentation.
