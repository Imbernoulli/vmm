# Proposal: Visualizing Model Merging

## 0. One-line summary

Build an interactive visualization framework for model merging that shows **where merged models land in weight space**, **whether they fall into the intersection of multiple task basins**, and **why some merges fail due to task interference**.

The core reframe is:

> Instead of visualizing a random 2D slice of one model’s loss landscape, visualize the task-vector subspace where model merging actually happens.

---

## 1. Motivation

The classic reference is Li et al., *Visualizing the Loss Landscape of Neural Nets*. Its key technical idea is **filter-wise normalization**, introduced because random perturbation directions in neural-network weight space are not directly comparable, especially under scale invariances. The paper uses normalized 2D/3D slices to study sharpness, flatness, architectures, and training dynamics. ([NIPS 会议论文][1])

Model merging gives us a more semantically meaningful setting. We do not need to choose arbitrary random directions. Given a base model:

[
\theta_0
]

and task-specific fine-tuned models:

[
\theta_i
]

we naturally get task vectors:

[
\tau_i = \theta_i - \theta_0
]

This is exactly the framing used in task arithmetic: a task vector is the weight-space direction induced by fine-tuning on a task, and adding task vectors can steer behavior across tasks. ([arXiv][2])

So the central object becomes the **merge subspace**:

[
\theta(c) = \theta_0 + \sum_i c_i \tau_i
]

Rather than asking:

> What does the loss landscape around one model look like?

we ask:

> Where are the low-loss regions for multiple tasks, and do different merging methods land inside their intersection?

That makes this a distinct project from “Visualizing Loss Landscape.” The goal is not just prettier contours. The goal is to explain **model merging geometry**.

---

## 2. Core hypothesis

Model merging succeeds when the merge point falls into a shared low-loss region across tasks.

More concretely:

> A good merged model lies in the intersection, or near-intersection, of multiple task-specific low-loss basins inside the task-vector subspace.

This connects naturally to several existing observations:

1. **Model Soups** showed that averaging weights of fine-tuned models can improve accuracy without additional inference cost, and argued that many fine-tuned models from the same pre-trained model lie in a shared low-error basin. ([arXiv][3])
2. **Task Arithmetic** showed that fine-tuning deltas can be treated as meaningful directions in weight space. ([arXiv][2])
3. **TIES-Merging** identified interference between model parameters, especially sign conflicts, as a major reason naive merging fails. ([OpenReview][4])
4. **DARE** showed that many delta parameters are redundant and can be dropped/rescaled before merging to reduce interference. ([arXiv][5])
5. **Git Re-Basin** showed that permutation symmetries can create artificial barriers between models, and that aligning models can make weight-space merging more meaningful. ([arXiv][6])

Our project should turn those scattered intuitions into one visual story.

---

## 3. Project goals

### Primary goal

Create a visualization toolkit / explainer for model merging that makes the following visible:

* where the base model, expert models, and merged models sit in a shared subspace;
* where each task’s low-loss basin lies;
* whether the basins overlap;
* whether naive average, task arithmetic, TIES, DARE, SLERP, Fisher merging, etc. land in useful regions;
* which layers or parameters create interference.

### Secondary goal

Use the visualization to generate empirical claims such as:

* task-vector alignment predicts merge success;
* sign conflict predicts merge failure;
* layer-wise conflicts are localized rather than uniform;
* some merge methods work because they move the final point away from conflict ridges;
* alignment matters when models do not share the same initialization.

### Non-goal

This is not initially about proposing a new merging algorithm. The first version should be an **artifact / explainer / diagnostic tool**. A new algorithm can come later if the visualizations expose a consistent failure pattern.

---

## 4. Key visualizations

## 4.1 Merge landscape

For two task vectors:

[
\tau_A = \theta_A - \theta_0
]

[
\tau_B = \theta_B - \theta_0
]

define a 2D merge plane:

[
\theta(\alpha, \beta) = \theta_0 + \alpha \tau_A + \beta \tau_B
]

Evaluate every grid point on task A and task B:

[
L_A(\alpha, \beta)
]

[
L_B(\alpha, \beta)
]

Then visualize:

[
L_{\text{avg}} = \frac{L_A + L_B}{2}
]

and:

[
L_{\text{worst}} = \max(L_A, L_B)
]

This produces the central figure:

> two task basins overlaid in the same merge plane, with the merge point shown relative to their intersection.

Plot points for:

* base model: ((0, 0))
* expert A: ((1, 0))
* expert B: ((0, 1))
* naive average: ((0.5, 0.5))
* task arithmetic sweep
* TIES merge
* DARE merge
* TIES + DARE
* SLERP
* Fisher merge
* validation-searched best merge

This should be the “hero figure.”

---

## 4.2 Per-task basin overlay

For each task, draw separate contour lines:

* task A loss contour;
* task B loss contour;
* joint objective contour;
* worst-task contour.

The main message:

> Merging is not about finding one flat minimum. It is about finding the overlap of several task-compatible regions.

This is probably the cleanest conceptual distinction from the Li et al. loss-landscape paper.

---

## 4.3 Lambda sweep path

For task arithmetic:

[
\theta(\lambda) = \theta_0 + \lambda(\tau_A + \tau_B)
]

sweep:

[
\lambda \in [0, 1.5]
]

Plot:

* task A accuracy vs. (\lambda);
* task B accuracy vs. (\lambda);
* average accuracy vs. (\lambda);
* worst-task accuracy vs. (\lambda);
* loss barrier along the path.

This is the simplest diagnostic for:

> how far can we move along the combined task direction before falling out of the shared basin?

This also connects to linear mode connectivity, where researchers study whether different trained solutions can be connected by low-loss linear paths. Frankle et al. studied when networks become linearly connected under different SGD randomness. ([arXiv][7])

---

## 4.4 Merge method overlay

Overlay the outputs of different merging methods in the same 2D plane.

Methods to include:

* linear average / model soup;
* task arithmetic;
* SLERP;
* Fisher merging;
* RegMean;
* TIES;
* DARE;
* TIES + DARE;
* layer-wise task arithmetic;
* validation-searched weighted average.

Fisher merging is worth including because it weights parameters by estimated Fisher information instead of treating all parameters equally. ([arXiv][8])

The purpose is not merely to rank methods. The purpose is to visually explain:

> Did a better method actually move the merged model into a better basin region, or did it improve performance through something invisible in this projection?

If a method performs well but does not look better in the 2D plot, that is also useful: it tells us the selected plane is missing important structure.

---

## 4.5 Interference atlas

This is the part that makes the project specifically about model merging rather than generic loss visualization.

For each layer (\ell), compute:

### Cosine alignment

[
\cos(\tau_A^\ell, \tau_B^\ell)
==============================

\frac{
\langle \tau_A^\ell, \tau_B^\ell \rangle
}{
|\tau_A^\ell| |\tau_B^\ell|
}
]

### Sign conflict rate

[
\text{conflict}_{A,B}^{\ell}
============================

\Pr[
\text{sign}(\tau_A^\ell) \neq \text{sign}(\tau_B^\ell)
]
]

### Magnitude-weighted conflict

[
\text{weighted-conflict}_{A,B}^{\ell}
=====================================

\sum_p
|\tau_{A,p}^{\ell}|
|\tau_{B,p}^{\ell}|
\cdot
\mathbf{1}
[
\text{sign}(\tau_{A,p}^{\ell})
\neq
\text{sign}(\tau_{B,p}^{\ell})
]
]

Visualize these as heatmaps:

* x-axis: layer;
* y-axis: task pair;
* color: conflict or alignment score.

This directly explains what TIES is trying to fix: it trims redundant updates, elects a sign, and merges only sign-consistent parameters to reduce interference. ([OpenReview][4])

---

## 4.6 Alignment-before-merge visualization

If models do not share the same base initialization, visualize:

* before alignment;
* after permutation alignment;
* loss barrier before alignment;
* loss barrier after alignment;
* merge accuracy before/after alignment.

This is important because apparent barriers in weight space can be artifacts of neuron/channel permutation symmetry. Git Re-Basin explicitly studies merging models modulo permutation symmetries and shows that alignment can make models much more linearly connected. ([arXiv][6])

This could become a strong demo:

> The landscape looks impossible before alignment; after alignment, the basin becomes connected.

---

## 5. Experimental plan

## 5.1 Phase 1: Small controlled prototype

Use a small CV setup first. The goal is not SOTA; the goal is clean visualization.

### Candidate setup

* Architecture: ResNet-18 or ViT-small.
* Base model: one shared pre-trained or jointly trained checkpoint.
* Experts: fine-tune from the same base on related classification tasks.
* Datasets: CIFAR-10 subsets, CIFAR-100 superclasses, MNIST variants, SVHN, Fashion-MNIST, or small image-domain tasks.
* Evaluation: use fixed validation subsets for speed.

### Why start here?

* Grid evaluation is cheap.
* Full weight loading is manageable.
* We can run (41 \times 41) or (51 \times 51) landscapes.
* Failure cases are easier to debug.
* We can validate whether the visual story is real before spending LLM compute.

### Outputs

* 2D merge landscape for task pair A/B.
* Per-task basin overlay.
* Lambda sweep.
* Method overlay.
* Layer-wise interference heatmap.

---

## 5.2 Phase 2: CLIP / ViT task-vector version

Move to a setting closer to existing task-vector and model-soup literature.

### Candidate setup

* Base: CLIP ViT-B/32 or ViT-B/16.
* Experts: fine-tuned checkpoints on image classification tasks.
* Tasks: Cars, DTD, EuroSAT, GTSRB, MNIST, RESISC45, SUN397, SVHN, etc.
* Merge methods: average, task arithmetic, TIES, DARE, Fisher, RegMean if feasible.

This phase is more publishable because task vectors and model soups were both studied heavily in large pre-trained vision models. Model Soups specifically demonstrated weight averaging for fine-tuned large pre-trained models including CLIP and ViT-family models. ([arXiv][3])

### Outputs

* Pairwise merge landscapes.
* 3-task simplex / triangle plots.
* PCA projection of many task vectors.
* Interference atlas across many task pairs.
* Correlation between conflict metrics and merge performance drop.

---

## 5.3 Phase 3: LLM / instruction model case study

Only after the visualization works on smaller models, run a limited LLM experiment.

### Candidate setup

* Base: same-size open-weight LLM.
* Experts: instruction-tuned, code-tuned, math-tuned, safety-tuned, or domain-tuned variants sharing the same base.
* Merge methods: linear, SLERP, task arithmetic, TIES, DARE, TIES+DARE.
* Evaluation: small but representative benchmark slices.

### Metrics

* MMLU / knowledge subset;
* GSM8K or math subset;
* HumanEval / MBPP subset;
* instruction-following score;
* toxicity / refusal / safety metrics if relevant;
* pairwise KL on prompts;
* calibration or confidence drift.

### Caveat

For LLMs, grid evaluation will be expensive and noisy. We probably should not start there. The first LLM version should use sparse path sweeps rather than dense 2D grids.

---

## 6. Technical design

## 6.1 Weight-space utilities

Implement utilities for:

* loading checkpoints;
* flattening weights into vectors;
* reconstructing model weights from vectors;
* computing task vectors;
* computing layer-wise task vectors;
* projecting arbitrary checkpoints into a selected 2D plane;
* generating grid checkpoints;
* evaluating grid checkpoints.

Pseudo-interface:

```python
theta0 = load_weights(base_ckpt)
theta_a = load_weights(task_a_ckpt)
theta_b = load_weights(task_b_ckpt)

tau_a = theta_a - theta0
tau_b = theta_b - theta0

plane = TaskVectorPlane(origin=theta0, directions=[tau_a, tau_b])

theta_grid = plane.point(alpha=0.5, beta=0.5)
metrics = evaluate(theta_grid, tasks=["A", "B"])
```

---

## 6.2 Plane construction

Support three plane types.

### Plane type A: raw task-vector plane

[
\theta(\alpha,\beta)=\theta_0+\alpha\tau_A+\beta\tau_B
]

Best for interpretability.

### Plane type B: orthonormalized task-vector plane

Use Gram-Schmidt:

[
e_1 = \frac{\tau_A}{|\tau_A|}
]

[
e_2 = \text{orthogonalize}(\tau_B, e_1)
]

Best for geometric comparison.

### Plane type C: PCA task-vector plane

For (n > 2) experts, stack task vectors:

[
T = [\tau_1, \tau_2, ..., \tau_n]
]

Run PCA and visualize the top two directions.

Best for many-task visualization.

---

## 6.3 Normalization policy

We should report two versions where possible.

### Raw merge landscape

Use the original task vectors and coefficients.

This answers:

> What happens under the actual merge operation?

### Normalized diagnostic landscape

Normalize per layer or per tensor, similar in spirit to filter-wise normalization.

This answers:

> Is the visual geometry robust to scale artifacts?

Important: unlike Li et al., normalization is not the main foundation here because task-vector directions already have semantic meaning. But diagnostic normalization is still useful for sanity checks.

---

## 6.4 BatchNorm / running-stat handling

For models with BatchNorm, each grid point should refresh BN statistics before evaluation.

Otherwise, a grid point may inherit stale running means/variances from a different checkpoint, making the loss surface misleading.

Practical options:

* avoid BN models in the first version by using LayerNorm-based ViTs;
* or recompute BN stats on a calibration subset before evaluating each grid point;
* or freeze BN and explicitly document the choice.

For clean early results, using ViT / CLIP-style models is probably safer.

---

## 7. Deliverables

## 7.1 MVP deliverables

The MVP should produce:

1. one clean 2D merge landscape;
2. one per-task basin overlay;
3. one lambda-sweep plot;
4. one method-overlay plot;
5. one layer-wise interference heatmap;
6. a short writeup explaining the geometry.

The MVP is successful if a reader can understand:

> naive averaging works when the midpoint falls inside the shared low-loss region, and fails when task basins are separated by an interference ridge.

---

## 7.2 Full demo deliverables

The full version should include an interactive UI.

### UI components

Left panel:

* 2D landscape;
* contours for task A, task B, joint loss;
* draggable merge point;
* method overlay points.

Right panel:

* task metrics;
* average / worst-task score;
* layer-wise conflict heatmap;
* selected checkpoint details;
* lambda slider;
* merge method selector.

### Interaction

Users should be able to:

* drag (\alpha, \beta);
* switch task pairs;
* switch merge method;
* toggle raw vs normalized plane;
* toggle loss / accuracy / worst-task objective;
* inspect per-layer interference;
* compare before vs after alignment.

This would make a strong blog artifact or paper companion.

---

## 8. Research questions

The project should answer these questions:

### RQ1: Do successful merges visually correspond to shared low-loss regions?

Expected result:

* successful average / soup points lie in basin intersections;
* failed merges lie on ridges or outside at least one task basin.

### RQ2: Does task-vector conflict predict merge failure?

Expected result:

* high sign conflict and low cosine alignment correlate with larger merge drop;
* conflict is concentrated in certain layers rather than evenly distributed.

### RQ3: Do TIES / DARE / Fisher move the merge into better regions?

Expected result:

* TIES and DARE reduce destructive interference;
* Fisher helps when parameter importance differs strongly across tasks;
* some improvements may not be visible in a 2D slice, which indicates limits of the visualization.

### RQ4: Does alignment change the apparent merge geometry?

Expected result:

* unaligned independently trained models show artificial barriers;
* after permutation alignment, interpolation barriers shrink.

### RQ5: Can the visualization guide better merge choices?

Expected result:

* visual diagnostics suggest better coefficients, better layers to merge, or layers to exclude.

---

## 9. Evaluation metrics

## 9.1 Model performance metrics

For each checkpoint or grid point:

* task A loss;
* task B loss;
* task A accuracy;
* task B accuracy;
* average accuracy;
* worst-task accuracy;
* calibration error if useful;
* OOD accuracy if available.

## 9.2 Landscape metrics

* basin overlap area;
* minimum joint loss in the plane;
* loss barrier along interpolation path;
* distance from merge point to nearest joint-loss minimum;
* curvature / sharpness around merge point;
* sensitivity to plane choice.

## 9.3 Interference metrics

* global task-vector cosine similarity;
* per-layer cosine similarity;
* sign conflict fraction;
* magnitude-weighted sign conflict;
* Fisher-weighted conflict;
* sparsity of delta parameters;
* overlap of high-magnitude delta coordinates.

## 9.4 Correlation analysis

Correlate:

[
\text{merge drop}
]

with:

[
\text{cosine alignment}
]

[
\text{sign conflict}
]

[
\text{weighted conflict}
]

[
\text{basin overlap}
]

The useful claim would be:

> visual and geometric diagnostics predict when merging will fail.

---

## 10. Implementation plan

## Week 1: Prototype core pipeline

* Pick small model and two tasks.
* Train or load base + two experts.
* Implement task-vector extraction.
* Implement 2D grid generation.
* Implement evaluation loop.
* Produce first merge landscape.

Exit criterion:

* We can generate a (41 \times 41) grid and evaluate every point on both tasks.

---

## Week 2: Add merge methods

Implement and overlay:

* average;
* task arithmetic;
* SLERP;
* TIES;
* DARE;
* TIES + DARE;
* simple validation-searched weighted merge.

Exit criterion:

* One plot shows all merge methods as points or trajectories on the same landscape.

---

## Week 3: Add interference atlas

Compute:

* layer-wise cosine;
* sign conflict rate;
* magnitude-weighted conflict;
* optional Fisher-weighted conflict.

Exit criterion:

* We can explain at least one failed merge using conflict heatmaps.

---

## Week 4: Expand to more tasks

* Add more expert models.
* Run pairwise landscapes.
* Add PCA plane for (n > 2) experts.
* Add simplex plot for three tasks.

Exit criterion:

* We can compare easy-to-merge task pairs vs hard-to-merge task pairs.

---

## Week 5: Interactive demo

* Build a lightweight Streamlit / Gradio / Plotly app.
* Add sliders for (\alpha, \beta, \lambda).
* Add method toggles.
* Add task-pair selector.
* Add conflict heatmap panel.

Exit criterion:

* Someone can understand model merging behavior by playing with the demo for five minutes.

---

## Week 6: Writeup

Write a blog-style or paper-style artifact:

Suggested title:

> Visualizing Model Merging: Task Basins, Interference, and Merge Geometry in Weight Space

Structure:

1. From loss landscapes to merge landscapes.
2. Task vectors define meaningful directions.
3. Merging succeeds in basin intersections.
4. Interference creates ridges.
5. TIES / DARE / Fisher as geometric interventions.
6. Alignment matters.
7. Limitations of 2D slices.
8. Future: using visualization to design new merge algorithms.

---

## 11. Risks and mitigations

## Risk 1: The 2D slice is misleading

A 2D plane is only a slice through a very high-dimensional space.

Mitigation:

* use multiple planes;
* compare raw task-vector plane, orthonormalized plane, and PCA plane;
* report that the visualization is diagnostic, not a full description of the basin.

---

## Risk 2: Different models are not alignable by default

If models come from different initializations, barriers may reflect permutation mismatch rather than real functional incompatibility.

Mitigation:

* start with same-base fine-tuned models;
* add alignment experiments later;
* explicitly separate same-base merging from independently trained model merging.

---

## Risk 3: BatchNorm or stale statistics distort the landscape

Mitigation:

* start with LayerNorm-based models if possible;
* recompute BN stats if using ResNet;
* document evaluation protocol carefully.

---

## Risk 4: LLM evaluation is too noisy or expensive

Mitigation:

* start with CV / CLIP models;
* for LLMs, use path sweeps rather than dense grids;
* use benchmark subsets and repeated sampling only for final case studies.

---

## Risk 5: The novelty is more artifact than algorithm

This is true, but acceptable.

The individual ingredients exist across loss landscapes, model soups, task arithmetic, TIES, DARE, Fisher merging, and Git Re-Basin. The novelty is the unified visual framing:

> model merging as geometry of task-vector basins and interference.

That is a strong explainer / artifact contribution even before introducing a new method.

---

## 12. Recommended MVP

The fastest useful version:

### Setup

* Use two or three same-base fine-tuned models.
* Prefer ViT / CLIP-style models to avoid BatchNorm complications.
* Evaluate on small validation subsets.

### Build

* Task-vector plane.
* 2D grid.
* Per-task loss contours.
* Average / worst-task contours.
* Overlay merge methods.
* Layer-wise conflict heatmap.

### First figure to aim for

A single plot showing:

* base model outside both task basins;
* expert A inside task A basin;
* expert B inside task B basin;
* average merge near the basin intersection;
* failed merge on a ridge;
* TIES / DARE shifted toward a safer region.

That figure is the whole pitch.

---

## 13. Final positioning

The project should be positioned as:

> A visualization and diagnostic framework for understanding when and why model merging works.

Not:

> Another loss landscape visualization.

The conceptual difference is:

| Loss landscape visualization     | Model merging visualization            |
| -------------------------------- | -------------------------------------- |
| random or normalized directions  | task-vector directions                 |
| one model / one objective        | multiple experts / multiple objectives |
| sharp vs flat minima             | basin intersection vs interference     |
| architecture/training comparison | merge-method comparison                |
| loss surface                     | merge geometry                         |

The strongest claim would be:

> Model merging can be understood as searching for low-loss intersections in task-vector subspaces, while failures often appear as conflict-induced ridges between task basins.

My recommendation: start with the CV/CLIP prototype, get the hero figure working, then decide whether this becomes a blog/demo artifact or a paper-style diagnostic framework.

[1]: https://papers.nips.cc/paper/7875-visualizing-the-loss-landscape-of-neural-nets?utm_source=chatgpt.com "Visualizing the Loss Landscape of Neural Nets"
[2]: https://arxiv.org/abs/2212.04089?utm_source=chatgpt.com "Editing Models with Task Arithmetic"
[3]: https://arxiv.org/abs/2203.05482?utm_source=chatgpt.com "Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time"
[4]: https://openreview.net/forum?id=xtaX3WyCj1&utm_source=chatgpt.com "TIES-Merging: Resolving Interference When Merging Models"
[5]: https://arxiv.org/abs/2311.03099?utm_source=chatgpt.com "Language Models are Super Mario: Absorbing Abilities from Homologous Models as a Free Lunch"
[6]: https://arxiv.org/abs/2209.04836?utm_source=chatgpt.com "Git Re-Basin: Merging Models modulo Permutation Symmetries"
[7]: https://arxiv.org/abs/1912.05671?utm_source=chatgpt.com "Linear Mode Connectivity and the Lottery Ticket Hypothesis"
[8]: https://arxiv.org/abs/2111.09832?utm_source=chatgpt.com "Merging Models with Fisher-Weighted Averaging"

