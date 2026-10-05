# NonlocalMeans.jl

`NonlocalMeans.jl` is a Julia package for non-local means denoising of
real-valued arrays of any dimensionality, implemented with
[KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl).

```julia
using NonlocalMeans

denoised = NLmeansKA(value, precision; patch_size = 3, search_size = 7, h = 10)
NLmeansKA!(output, value, precision; patch_size = 3, search_size = 7, h = 10)
```

Each sample is replaced by a weighted average of the samples in its search
neighborhood. `precision` is optional and defaults to a uniform array. The
weights are `exp(-patch_distance / h)`, where the patch distance is a
precision-weighted mean squared difference between patches.

- `patch_size` and `search_size` are nonnegative integer radii.
- `h` controls the weight decay.
- `skip_zero_offset` (default `true`) omits the patch center from the distance.
- `backend` selects the KernelAbstractions backend; it defaults to the one
  associated with `value` (regular Julia arrays use the CPU backend).

`NLmeansKA!` writes into `output` and returns it; if `output` aliases an input,
that input is copied before the kernel is launched. Inputs must use one-based
indexing.

The original serial implementation remains as the non-exported
`NonlocalMeans.NLmeans`, used as a reference in the tests.

## Testing

Run the package tests with:

```julia
using Pkg
Pkg.test()
```
