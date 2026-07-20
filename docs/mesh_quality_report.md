# Mesh quality: where the roughness comes from, and what the existing knobs can do about it

**Date:** 2026-07-20
**Plan:** `heAICare/docs/ACTION_PLAN_mesh_quality.md`
**Session:** `video_test_20260716_115516_arm1_baseline` (intact MVS workspace, 532 depth maps)
**Repository state:** `master` @ `73b1dfc`
**Recommendation:** **one small change** — `taubin_smoothing.iterations: 10 → 0` (smoothing off).
Every other parameter in `mesh.yaml` and `colmap.yaml` stays where it is.
**Principal finding:** both visible mesh defects are non-configurable; their causes are
capture-side.

---

## Summary

Neither visible defect is fixable from `configs/mesh.yaml`. Both hypotheses the plan was
built to test came back negative, and the measurements point the same direction: the
defects originate in the capture, not in the mesh stage.

**Membranes are not under-trimmed.** `density_threshold` is a *global quantile* on Poisson
density, so raising it shaves the entire surface uniformly. Going 0.01 → 0.05 removes 15.6%
of pale/membrane-class area and 15.1% of genuine dark head surface — in lockstep — while the
pale fraction stays pinned at 6.9% at every threshold. In the lower head, where the membrane
actually lives, pale area is *identical* (2386 mm²) at all four thresholds: the trim never
touches it. This independently reproduces the 2026-07-15 diagnosis on this session's data.

**Roughness is upstream, and Poisson is already suppressing it.** At a 2.5 mm neighbourhood
the dense cloud is *rougher* than the mesh fitted to it (median 0.487 mm vs 0.243 mm). The
two are strongly co-located (Pearson 0.67, Spearman 0.70): where the mesh sits in its
roughest decile, cloud scatter is 0.696 mm against 0.346 mm across the smoothest half.
Poisson is acting as a low-pass filter, not a roughness source — so `depth` and Taubin are
treating a symptom.

**The roughness has a specific physical location: the ArUco marker borders.** The heat maps
show the marker squares outlined in red on an otherwise smoother surface, and the numbers
confirm it — mesh roughness rises monotonically with local colour contrast, from 0.196 mm in
the flattest decile to 0.425 mm at the high-contrast borders (**2.2×**). Roughness also
tracks frame coverage monotonically (2–3 views 0.289 mm → 10+ views 0.191 mm). Taped-on
markers on a matt-black textureless phantom, plus thin coverage, are capture-protocol
properties. No parameter in this pipeline reaches them.

Every upstream filter that *does* reduce roughness pays for it in coverage or in dimensions,
which the plan's fidelity constraint rules out.

The only configuration change is therefore to turn Taubin smoothing **off** (10 → 0). Taubin
turned out to be radially unbiased, so it was not doing harm — but it can remove at most ~10% of
a roughness whose cause is now known to be capture-side, and shipping no smoothing means every
morphometric measurement traces to fused geometry with no intervening filter to defend. The
tuned `lambda_filter`/`mu` and the measured knee (5 iterations) are retained in the config and
below, so re-enabling is a one-token change if morphometry later calls for it.

---

## What situation applied

PatchMatch did **not** re-run. Every arm consumed the identical depth maps already in
`arm1_baseline/mvs/`, so fusion-filter differences are attributable to the filter alone.
COLMAP's stereo fusion is CPU-only, so all of this ran natively on Windows (Open3D 0.19.0,
pycolmap 4.1.0, `has_cuda False`) — PatchMatch would have required WSL2 + CUDA.

**Control.** Re-fusing with the shipped defaults and replaying SOR → scale → crop reproduced
the shipped cloud to 865,087 points vs 864,928 (**+0.02%**), and re-running the mesh stage on
the shipped cloud reproduced `mesh.ply` at 506,716 vertices vs 506,710 (**+0.001%**). The
harness is faithful to the pipeline; `_apply_lcc` and `_apply_taubin` are imported directly
from `mesh/surface_reconstruction.py` rather than reimplemented.

### Provenance

| Item | Value |
|---|---|
| Session | `video_test_20260716_115516_arm1_baseline` |
| Scale | 61.710147820611766 mm/unit |
| Head crop | radius 173.81 mm, `center_source: aruco_centroid`, `radius_clamped: false` |
| Mesh input | `dense_filtered_cropped.ply`, 864,928 points (mm) |
| Shipped mesh | 506,710 verts / 1,013,239 tris |
| Config under test | `depth: 9`, `density_threshold: 0.01`, `taubin.iterations: 10`, SOR `std_ratio: 2.0` |
| Fusion under test | `min_num_pixels: 5`, `max_reproj_error: 2.0` |

**Metric scale is unvalidated.** `known_distances_mm` is still empty, so every millimetre
figure in this report inherits that caveat. All conclusions rest on *relative* comparisons
within one shared scale factor, which is exactly where they are safe.

---

## T1 — Roughness metric and the cloud-vs-mesh verdict

**Metric: local plane-fit residual** — the RMS distance of a neighbourhood to its best-fit
plane (√ of the smallest covariance eigenvalue) within a fixed *metric* radius.

Chosen over discrete mean curvature because the entire question is "mesh vs cloud": the
plane-fit residual is defined identically on mesh vertices and on unstructured points, in the
same units, over the same physical neighbourhood. Mean curvature needs a triangulation, has
no point-cloud analogue, and its magnitude moves with triangle density. A fixed radius rather
than fixed *k* keeps the comparison honest where the two densities differ.

Both fields are evaluated at the **same** points (a uniform random subsample of mesh
vertices — the whole surface, no hand-picked regions), measured at 1.0 mm and 2.5 mm.

| Field | p25 | **p50** | p75 | p90 | p95 |
|---|---|---|---|---|---|
| Mesh roughness (Taubin off), 2.5 mm | 0.136 | **0.243** | 0.476 | 0.676 | 0.775 |
| Dense-cloud scatter, 2.5 mm | 0.333 | **0.487** | 0.623 | 0.741 | 0.808 |
| Mesh roughness, 1.0 mm | — | **0.058** | — | 0.179 | — |
| Dense-cloud scatter, 1.0 mm | — | **0.210** | — | 0.313 | — |

All mm. The cloud is rougher than the mesh at **both** scales — 2.0× at 2.5 mm, 3.6× at
1.0 mm.

**Spatial correspondence** (the part a global mean would hide):

| Region | Cloud scatter (median) |
|---|---|
| Where mesh is in its roughest decile (≥ 0.549 mm) | 0.696 mm |
| Where mesh is in its smoothest half | 0.346 mm |

Verdict: **UPSTREAM**. The cloud is already noisy exactly where the mesh is rough. The two
causes do not coexist — there is no region where the cloud is locally clean but the mesh is
wrinkled, which is what a Poisson/normals/`depth` problem would look like.

### Where the roughness sits: marker borders

The heat maps (`heAICare/analysis/results/mesh_quality/figures/heat_mesh_roughness.png` and
`heat_cloud_scatter.png`, rendered
on a shared colour scale) show the ArUco squares outlined in red. Tested numerically using
local colour contrast as a marker-border proxy — the phantom is matt black with white markers,
so the only strong brightness gradients on the surface *are* the borders:

| Contrast decile | Mean brightness | Mesh roughness | Cloud scatter |
|---|---|---|---|
| 1 (flattest) | 58.3 | 0.196 | 0.358 |
| 5 | 66.3 | 0.232 | 0.484 |
| 8 | 86.4 | 0.434 | 0.592 |
| 10 (borders) | 111.0 | 0.425 | 0.574 |

Monotonic, 2.2× top-to-bottom on the mesh (Pearson 0.31). The markers are physically raised
tape presenting a high-contrast step to a PatchMatch window of radius 5 — the roughness is
manufactured at capture time.

### Coverage overlay

| Contributing views | Mesh roughness (median) | Cloud scatter (median) |
|---|---|---|
| 2–3 | 0.289 | 0.463 |
| 4–5 | 0.269 | 0.421 |
| 6–9 | 0.239 | 0.382 |
| 10+ | 0.191 | 0.323 |

Monotonic. Thin coverage means rougher surface. **Recorded as a capture-protocol limitation,
not acted on here** (per the plan's scope), and consistent with the standing "textureless-surface
MVS" weakness: the black phantom yields sparse coverage between markers.

---

## T2 — Density-threshold sweep (Taubin held at 0)

Poisson ran **once**; the density array was reused across all four thresholds, so the trim is
the only variable. Areas in mm², inside the head sphere (r = 150 mm).

| `density_threshold` | Verts | Head area | Pale area | Dark area | Δ pale | Δ dark | Pale % |
|---|---|---|---|---|---|---|---|
| **0.01** (current) | 506,716 | 115,685 | 8,008 | 98,690 | — | — | 6.9% |
| 0.02 | 501,821 | 109,022 | 7,599 | 93,070 | −5.1% | −5.7% | 7.0% |
| 0.035 | 491,594 | 102,932 | 7,137 | 88,141 | −10.9% | −10.7% | 6.9% |
| 0.05 | 480,343 | 97,766 | 6,757 | 83,815 | −15.6% | −15.1% | 6.9% |

The tradeoff curve is the finding: **the two curves are the same curve.** Pale and dark area
fall together within 0.5 percentage points at every step, and the pale fraction never moves
off 6.9%. A useful trim would drop pale much faster than dark; this one has no selectivity at
all.

In the lower head — the membrane's actual location, below the crop centre:

| `density_threshold` | Lower pale area | Δ |
|---|---|---|
| 0.01 / 0.02 / 0.035 / 0.05 | 2386 / 2386 / 2386 / 2386 mm² | **0.0% at every step** |

Not approximately unchanged — *identical*. Membrane vertices sit at healthy Poisson density
(median around the 0.21 quantile), far above even a 5% cutoff, so no threshold in a sane range
reaches them.

**Pick: keep `density_threshold: 0.01`.** It is the lowest value tested and every increase is
pure loss of genuine surface. Figure: `heAICare/analysis/results/mesh_quality/figures/fig2_density_threshold_tradeoff.png`.

---

## T3 — Addressing roughness at its source

T1 said upstream, so the fusion and SOR filters were A/B'd on the same depth maps. Fidelity is
measured as **surface deviation**, not surface area: flattening noise legitimately lowers area,
so an area drop cannot distinguish "removed wrinkles" from "moved the surface". Deviation is
the unsigned distance between each arm's surface and the baseline surface, both directions,
inside the head sphere.

| Arm | Change | Crop pts | Head area | Mesh p50 | Δ roughness | arm→ref p50 / p99 | ref→arm p99 / max | Δ extents (x,y,z) |
|---|---|---|---|---|---|---|---|---|
| **base** | `mnp 5`, `reproj 2.0` | 865,087 | 123,022 | 0.243 | — | — | — | — |
| mnp7 | `min_num_pixels 5→7` | 577,145 | 113,785 | 0.208 | **−14%** | 0.043 / 0.611 | 1.433 / 13.27 | +0.19, +0.45, −0.43 |
| reproj1.0 | `max_reproj_error 2.0→1.0` | 1,201,849 | 114,594 | 0.200 | **−18%** | 0.055 / 0.768 | 2.048 / 12.73 | −0.10, +0.62, +0.24 |
| sor1.5 | `std_ratio 2.0→1.5` | 864,479 | 121,020 | 0.240 | −1.2% | 0.018 / 0.262 | 0.368 / 11.74 | −0.01, −0.02, −0.47 |
| sor1.0 | `std_ratio 2.0→1.0` | 863,133 | 118,818 | 0.237 | −2.5% | 0.023 / 0.378 | 0.579 / 12.70 | +0.34, −0.06, +0.13 |

All mm.

**SOR is inert — reject.** Tightening `std_ratio` all the way to 1.0 removes 0.2% of points
inside the crop and buys 2.5% roughness for 3.4% of surface area. The outliers SOR removes are
far-field; the head region has essentially none for it to catch.

**Both fusion filters are rejected on coverage, not on displacement.** They genuinely denoise
(−14% and −18%) and they barely move the surface where it survives (median deviation
0.043–0.055 mm). The cost is in the *reverse* direction: ref→arm p99 of 1.4 mm and 2.0 mm
with maxima above 12 mm means baseline surface with no counterpart in the arm — localized
holes opening. `min_num_pixels 7` also discards **33% of all points inside the crop**. On a
pipeline whose standing top weakness is sparse coverage on a textureless dark surface, trading
a third of the points to smooth a marker-edge artifact is the wrong direction.

A note on `max_reproj_error`, since the result is counter-intuitive: tightening it *increased*
fused points by 39%. Stricter reprojection agreement means fewer observations merge into each
fused point, so clusters split rather than being rejected — more points, each backed by fewer
views, and holes where nothing reaches consensus.

**Poisson `depth` 9 → 8 — reject.** The strongest roughness reduction measured anywhere
(0.243 → 0.169 mm, **−30%**) and the surface barely moves (median deviation 0.071 mm). But it
inflates head-sphere extents by **+1.93 mm (x) and +3.23 mm (z)** — 1.7% of a ~190 mm
dimension — because a coarser octree bridges gaps and pushes the hull outward. Surface area
rises 129,286 vs 122,811 mm² despite being visibly smoother, which is the tell. This is exactly
the failure mode the plan's fidelity constraint targets: looks better, measures worse.

### Taubin, quantified

All five variants were built from **one** Poisson solve, so vertices are index-aligned and
displacement is a direct per-vertex difference.

| Iterations | Roughness (1.0 mm) | Roughness (2.5 mm) | Mean disp | p95 disp | Max disp | Area | **Inward mean** |
|---|---|---|---|---|---|---|---|
| **0** | 0.0582 | 0.2433 | — | — | — | — | — |
| 2 | 0.0543 | 0.2407 | 0.024 | 0.057 | 0.96 | −1.13% | +0.0001 |
| **5** | 0.0529 | 0.2393 | 0.046 | 0.108 | 1.66 | −1.77% | +0.0001 |
| 10 (current) | 0.0526 | 0.2381 | 0.067 | 0.160 | 2.13 | −2.25% | +0.0001 |
| 20 | 0.0532 | 0.2378 | 0.091 | 0.217 | 2.79 | −2.58% | +0.0000 |

All mm. Two things fall out.

**The smoothing bias accepted is, to measurement precision, zero.** Taubin's displacement is
radially **unbiased**: mean inward movement +0.0001 mm with a 50/50 inward/outward split at
every setting. This is the λ/μ pass-band filter behaving as designed, and it is a materially
better result than the plan anticipated — the 2.25% area loss at 10 iterations is wrinkle
flattening, *not* head shrinkage. Head extents move ≤ 0.30 mm. Taubin is not biasing the
morphometry.

**But it earns very little, and 10 iterations is past the knee.** Total available reduction is
only 9.6% at 1.0 mm and 2.2% at 2.5 mm. Five iterations captures 91% of it; going 5 → 10 buys a
further 0.6% while raising mean displacement 46% (0.046 → 0.067 mm) and area loss to 2.25%.
Twenty iterations is *worse* at 1.0 mm than ten — it starts reintroducing residual as it
flattens genuine structure.

Figure: `heAICare/analysis/results/mesh_quality/figures/fig3_taubin_tradeoff.png`.

### Why the shipped value is 0 and not the measured optimum of 5

The measurements above identify **5** as the knee. The shipped value is **0**. That is a
deliberate choice against the local optimum, and the reasoning belongs on the record.

The shrinkage objection that motivated examining Taubin in the first place turned out to be
**wrong** — the filter is radially unbiased, so nothing here argues that smoothing damages the
morphometry. Five iterations would be defensible on the numbers.

What changed the decision is what the roughness *is*. T1 attributes it to marker-border relief
and thin frame coverage: physical properties of how this capture was performed. Smoothing does
not remove that roughness, it renders it less visible, and the most it can remove is ~10%.
Spending any surface displacement — even unbiased, even 0.046 mm — to cosmetically attenuate an
artifact whose cause is now known and documented buys nothing the morphometry needs.

The concrete benefit of 0 is evidential rather than geometric: a mesh that has had no smoothing
applied requires no argument that its displacement was harmless. Every measurement taken from it
traces to fused geometry without an intervening filter, so the "is this an artifact of
smoothing?" question cannot be raised against a downstream morphometric result. At a ~10% ceiling
on roughness reduction, that is a better trade than the smoothing.

The parameter is not removed. `lambda_filter` and `mu` keep their tuned values, `iterations: 0`
is a one-token change, and 5 is recorded above as the knee if morphometry later shows smoothing
is actually needed. This is a reversible default, chosen to keep the mesh evidentially clean —
not a finding that Taubin is harmful.

---

## Recommended configuration

```diff
--- a/configs/mesh.yaml
+++ b/configs/mesh.yaml
@@ poisson_surface_reconstruction:
   taubin_smoothing:
-    iterations: 10
+    iterations: 0
     lambda_filter: 0.5
     mu: -0.53
```

That is the entire diff: smoothing off, tuned `lambda_filter`/`mu` retained so re-enabling is a
one-token change (see "Why the shipped value is 0 and not the measured optimum of 5" above).
Unchanged and now positively justified rather than merely inherited:

| Parameter | Value | Why it stays |
|---|---|---|
| `density_threshold` | 0.01 | Every increase removes real surface 1:1 with membrane and never touches the membrane region |
| `depth` | 9 | Depth 8 smooths 30% but inflates extents up to +3.23 mm |
| `scale`, `linear_fit`, `keep_largest_component` | unchanged | Not implicated |
| SOR `nb_neighbors`/`std_ratio` | 20 / 2.0 | Tightening is inert inside the head crop |
| `stereo_fusion.min_num_pixels` | 5 | 7 costs 33% of points and opens holes |
| `stereo_fusion.max_reproj_error` | 2.0 | 1.0 denoises but opens holes (ref→arm p99 2.05 mm) |

**Smoothing bias accepted at the shipped setting: none — zero, by construction.** With
`iterations: 0` no smoothing filter runs, so mesh vertices are the Poisson solution over the
fused cloud and no displacement is introduced at any point in the mesh stage. Nothing about a
downstream morphometric measurement needs to be defended against a smoothing artifact.

For reference, had 5 iterations been shipped the accepted cost would have been: mean vertex
displacement 0.046 mm, p95 0.108 mm, max 1.66 mm, surface area −1.77%, radial bias +0.0001 mm
(statistically no direction), head extents within 0.26 mm. That cost was small and unbiased —
it was declined because what it buys (~10% less roughness, on a capture-side artifact) is worth
less than keeping the mesh free of any filter that has to be argued about.

---

## Conclusion: both defects are non-configurable

**This is the finding of this work, not a failure to find one.** Both visible mesh defects were
traced to their origin, and neither origin is reachable from any parameter in `mesh.yaml` or
`colmap.yaml`. The question the plan asked — "which already-present parameters best reduce both
defects?" — has a definite and useful answer: **none of them do, and here is why.**

That is a result with teeth. It closes parameter tuning as an avenue, it redirects effort to
where the causes actually are, and it means the mesh stage can be treated as settled while the
morphometric module is built on top of it.

### Defect 1 — the pale membrane

Occlusion-boundary background bleed: the white mug and bright desk sit directly behind and below
the head contour, and PatchMatch mixes them in at the silhouette. Contaminated points sit a
median ~5 px from the dark silhouette.

Non-configurable because the membrane vertices carry *healthy* Poisson density (median around
the 0.21 quantile). `density_threshold` is a global quantile, so no value in a sane range reaches
them, and every increase removes genuine surface at a 1:1 rate with membrane. The defect has now
resisted, on measurement: ArUco hull masks at feature extraction, ArUco hull masks at fusion, and
density trimming — three independent attempts, all failing for the same underlying reason (mask
boundaries and trim thresholds are both far from the ~5 px bleed band).

**Capture-protocol remedies:** a dark, non-reflective backdrop placed under the chin and behind
the head contour, removing the bright background from the silhouette region entirely; and raising
the head off pale supporting surfaces so no bright object sits within the occlusion boundary. In
processing, the one untried lever is an eroded tight silhouette mask applied at fusion only — the
plumbing exists and is opt-in — or silhouette-aware point filtering.

### Defect 2 — surface roughness

Marker-border depth discontinuities (2.2× the roughness of flat surface, rising monotonically
with local colour contrast) and thin frame coverage (2–3 views is 1.5× rougher than 10+).

Non-configurable because it is already present in the fused cloud before the mesh stage begins —
the cloud is 2.0–3.6× rougher than the mesh fitted to it, so Poisson is attenuating this
roughness, not producing it. Every upstream filter that reduces it further does so by discarding
coverage or by deforming dimensions, which the fidelity constraint rejects.

**Capture-protocol remedies:**

1. **Non-raised markers printed directly into the cap fabric**, replacing taped-on squares. This
   addresses the dominant term. The current markers present both a physical step (tape relief)
   and a high-contrast intensity edge to a PatchMatch window of radius 5; printing the pattern
   into the fabric removes the relief entirely and keeps the ArUco detection the scale recovery
   depends on. Lower-contrast marker ink would further soften the intensity step, subject to
   keeping detection reliable.
2. **More camera passes over low-view regions.** Roughness falls monotonically with view count
   across the whole measured range (0.289 mm at 2–3 views → 0.191 mm at 10+), with no sign of
   saturating, so additional passes over thinly-covered regions should continue to pay. The
   per-region view counts needed to target those passes are already recoverable from
   `dense.ply.vis`.

Neither remedy is started here — capture-protocol changes are explicitly out of scope for this
plan, and both are recorded as limitations for the capture track to pick up.

### Where this leaves the mesh stage

On this evidence the mesh stage is as good as its input allows, and its parameters are now
justified by measurement rather than inherited by default. The mesh it produces is trustworthy
for the morphometric module to consume, within the standing unvalidated-scale caveat. The
productive next work is upstream of it, in the capture.

---

## Reproducing

Tooling lives in `heAICare/analysis/` (outside this repo); results and figures in
`heAICare/analysis/results/mesh_quality/`.

| Script | Purpose |
|---|---|
| `mesh_variants.py` | Build mesh variants from one cloud; Poisson solved once per depth |
| `mesh_roughness.py` | T1 plane-fit residual on mesh + cloud, coverage, heat maps |
| `membrane_metrics.py` | T2 pale/dark area split by threshold |
| `fusion_ab.py` | T3 re-fusion + SOR arms on the same depth maps |
| `mesh_deviation.py` | Surface deviation between meshes from different clouds |
| `smoothing_displacement.py` | Index-aligned Taubin displacement |
| `marker_edge_roughness.py` | Roughness vs local colour contrast |
| `mesh_quality_figures.py`, `render_roughness.py` | Figures |

One methodological trap worth keeping, which produced plausible-looking wrong numbers before
being caught:

- **The kNN cap must not bind.** The roughness query is fixed-radius but implemented as
  capped kNN; when the cap saturates, the effective radius silently shrinks in dense regions
  and the mesh/cloud comparison stops being like-for-like. `mesh_roughness.py` reports
  `mesh_saturated_frac` / `cloud_saturated_frac`; both must be ~0. The first pass saturated at
  96 neighbours (median 96 for both fields) and had to be re-run at 1024.

The second trap encountered — non-deterministic Poisson vertex ordering — is written up as a
standalone limitation below, because it constrains reproducibility beyond this report.

---

## Limitation: Open3D Poisson vertex ordering is not deterministic across processes

**Relevant to the reproducibility track. Cite this rather than the methodology note above.**

### Statement

`open3d.geometry.TriangleMesh.create_from_point_cloud_poisson` does not produce a
bit-reproducible vertex ordering across separate process invocations. Given the identical input
cloud and identical parameters, two runs in two processes yield meshes that agree in vertex and
triangle **count** but not in vertex **index**: vertex *i* of one mesh is not the same surface
point as vertex *i* of the other.

Observed with Open3D 0.19.0 on Windows, depth 9, 864,928-point input. Both runs returned
506,716 vertices / 1,013,252–1,013,253 triangles — a count agreement close enough to look like
determinism to any check based on counts.

### Why it matters

It silently invalidates **index-based vertex correspondence**, which is the natural and
efficient way to compare two meshes built from the same cloud — exactly what a smoothing,
filtering, or regression comparison wants to do. The failure is not loud. Differencing
positions vertex-by-vertex across two solves returns a well-formed array of plausible-looking
floats; in this work it reported a mean vertex displacement of **67 mm** for a filter whose true
displacement is **0.067 mm** — three orders of magnitude, from code that raised no error.

The danger is that count agreement reads as a successful sanity check. `len(a.vertices) ==
len(b.vertices)` passed in every case here and proves nothing about correspondence.

### Consequences for reproducibility claims

- A mesh SHA-256 is **not** a valid reproducibility check for this pipeline's mesh stage. Two
  correct runs of identical input and configuration can produce different bytes. Reproducibility
  must be asserted on geometric quantities (surface deviation, area, extents, vertex/triangle
  counts, derived measurements), never on file digests or vertex arrays.
- Any per-vertex comparison is valid **only** within a single process, sharing one Poisson solve.
  Across processes, correspondence must be re-established geometrically — nearest-point or
  ray-cast distance, as in `analysis/mesh_deviation.py`.
- Regression tests over meshes must assert on geometric invariants with tolerances, not on exact
  vertex data.

### Mitigations applied here

Variants requiring per-vertex comparison are built in a single `mesh_variants.py` invocation, so
they share one Poisson solve and one vertex ordering. `smoothing_displacement.py` refuses to
report a result when the median displacement exceeds 5 mm — far above any plausible smoothing
effect, far below the object-scale nonsense that misalignment produces — and instructs the
caller to rebuild the variants together. Comparisons that legitimately span clouds
(`mesh_deviation.py`) use geometric correspondence and are unaffected.

### Not investigated

Whether the non-determinism originates in Open3D's parallel octree traversal or in the upstream
PoissonRecon isosurface extraction was not pursued — out of scope here. Nor was it tested
whether single-threaded execution restores deterministic ordering, which would be the natural
first probe if a reproducible ordering is ever required.
