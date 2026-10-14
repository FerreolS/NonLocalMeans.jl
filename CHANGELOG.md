# Changelog

Notable changes to NonLocalMeans.jl are documented here.

## [0.3.0] - 2026-10-14

### Added

- Optional spatial weighting of patch comparisons through the `kernel`
  keyword, including the public `gaussian_kernel` and
  `skip_central_pixel_kernel` helpers.
- Support for supplying custom patch-weight arrays as `OffsetArray`s.

### Changed

- Supplying a precision array now returns `(output, output_precision)`.
  `nonlocalmeans` without an explicit precision array continues to return only
  the denoised output.

### Removed

- The `store_precision` keyword; output precision is returned automatically
  when an explicit precision array is supplied.
