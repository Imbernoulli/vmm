# Independent-Initialization Alignment Barrier

This is a small surrogate for the proposal's alignment-before-merge visualization. Two one-hidden-layer MLPs are trained on the same sklearn digits task from different random initializations. The second model is then permutation-aligned to the first by matching hidden-unit feature vectors with the Hungarian algorithm.

## Result

- Model A accuracy: 0.967.
- Model B accuracy: 0.976.
- Midpoint accuracy before alignment: 0.944.
- Midpoint accuracy after alignment: 0.971.
- Loss barrier before alignment: 0.064.
- Loss barrier after alignment: 0.006.

## Interpretation

If the before-alignment midpoint is poor while the after-alignment midpoint improves, the apparent barrier was partly a coordinate/permutation artifact rather than pure functional incompatibility. This is the small-model analogue of why Git Re-Basin style alignment matters before weight-space merging.

See `interpolation_alignment.png` and `path_metrics.csv` in this directory.
