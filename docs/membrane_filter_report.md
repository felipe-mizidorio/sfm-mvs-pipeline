# Membranes: diagnosis and a colour-based dense-cloud filter

**Date:** 2026-07-20
**Plan:** `heAICare/docs/ACTION_PLAN_membrane_filter.md`
**Repository state:** `master` @ `82ebdc4`
**Session:** `video_test_20260716_115516` (intact MVS workspace, 532 depth maps)
**Recommendation:** **ADOPT, opt-in — keep the default OFF.** The filter removes 100%
of the membrane at the target location while leaving the ArUco marker faces intact
(−0.05% area). Its assumption is scene-specific, which is why it should be enabled
explicitly rather than silently.

---

## Related investigations

This is investigation **4 of 4** into the same defect. The four are one line of inquiry,
not four unconnected documents:

| # | Investigation | Stage attacked | Outcome |
|---|---|---|---|
| 1 | ArUco hull masks at feature extraction | SfM features | Negative — `heAICare/docs/three_way_masking_comparison.md` |
| 2 | The same masks at stereo fusion | Fusion | Negative — [`fusion_masks_report.md`](fusion_masks_report.md) |
| 3 | Poisson density trimming | Meshing | Negative — [`mesh_quality_report.md`](mesh_quality_report.md) |
| 4 | Colour-based dense-cloud filter | Cropped cloud, pre-Poisson | **Positive, for this scene** — this report |

The methodological result is in *why* the split falls where it does, and is set out in the
Summary below: 1–3 each attacked the symptom at a stage where the contamination is not
separable from genuine surface, and each failed for that same reason. 4 proved the source
first (the T1 gate) and only then chose a stage — which is what changed the outcome.

The success is **scene-specific**. It establishes that the membranes are removable
contamination rather than Poisson hallucination, which is a real finding about their
nature; it does not make the filter a general remedy. The generalisable fix is still
capture-side, per `mesh_quality_report.md`. See also [`NEXT_STEPS.md`](NEXT_STEPS.md).

---

## Summary

The membranes are **dense-cloud contamination, not Poisson hallucination**, and they
can be removed by colour without damaging the head surface or the markers.

This is the first of the four membrane-related investigations to return a positive
result. The three before it (ArUco hull masks at feature extraction, the same masks
at fusion, and density trimming) all failed for the same reason: they attacked the
symptom at the wrong stage. Proving the source first — the structure this plan
imposed — is what changed the outcome.

**T1 (gate) confirmed the hypothesis.** Membrane mesh vertices are backed by real
points: 94.2% have a dense point within 2 mm, at a median nearest distance of
0.535 mm. Those supporting points are overwhelmingly pale — median 93.5% of the
neighbourhood above the pale threshold, against **0.0%** for genuine head surface.
The membranes are bright background points fused at the head silhouette and left
inside the spherical crop.

**T3 (A/B) is decisive.** With the filter on, pale area at the membrane location
falls **3245.3 → 0.0 mm²** (−100%), while marker faces lose **0.05%** of their area
(worst single marker −2.75%), genuine dark head surface *increases* 2.40%, and the
mesh has **fewer** hole boundaries than the control (−50 boundary edges, −3.16%
boundary length). Of the surface that disappears by more than 2 mm, 92.5% lies in
the membrane region and 82.6% of it is pale. Collateral loss is 0.441% of head
vertices.

The reason the default stays OFF is not performance but generality: the filter
assumes a dark subject against pale contamination, which is a property of this
phantom capture, not of the problem.

---

## What situation applied

PatchMatch did **not** re-run. Both arms re-fused the identical depth maps from the
`arm1_baseline` MVS workspace, applied the identical ArUco-derived crop, and used the
current `mesh.yaml` (`depth: 9`, `density_threshold: 0.01`, `taubin.iterations: 0`).
The only variable is `--membrane-filter`. COLMAP fusion is CPU-only, so both arms ran
natively on Windows.

The workspace was mounted into each arm's directory as a junction rather than copied,
so both arms read byte-identical depth maps and neither can perturb the other.

| Item | Value |
|---|---|
| Control arm | `data/processed/video_test_20260716_115516_membrane_off` |
| Filtered arm | `data/processed/video_test_20260716_115516_membrane_on` |
| Scale | 61.710148 mm/unit in both arms (independently recovered, identical) |
| Head crop | radius 173.81 mm, `aruco_auto`, `radius_clamped: false`, `center_source: aruco_centroid` |
| Markers triangulated | 19 |

**Metric scale is unvalidated.** `known_distances_mm` is still empty, so every
millimetre figure below carries that caveat; conclusions rest on relative comparisons
within one shared scale factor.

**Mesh comparisons are geometric only.** Open3D's Poisson is not order-deterministic
across processes (see `mesh_quality_report.md`), so nothing here uses vertex indices
or mesh hashes. Deviation is computed by ray-cast distance in both directions.

---

## T1 — GATE: are the membrane vertices point-supported, and what colour?

### How the membrane was localized

Geometrically, never by eye. Pale mesh vertices (mean RGB ≥ 150), minus everything
within 25 mm of a triangulated marker centroid, then DBSCAN (ε = 3 mm, min 30 points);
clusters below 200 vertices are discarded as speckle. Result: **11,167 vertices in 8
clusters**.

Subtracting the marker zones is essential rather than cosmetic — **the ArUco markers
are themselves white**, and without that step they dominate the pale set and the
measurement becomes a measurement of markers.

Three groups are compared, so the membrane numbers have a scale to be read against:

| Group | Definition |
|---|---|
| `membrane` | pale, off-marker, in a surviving cluster |
| `head` | dark (≤ 100), off-marker — genuine phantom surface |
| `marker` | pale, on-marker — genuine surface that happens to be pale |

### Support and supporting-point colour

| Group | n | nearest point p50 | p95 | density (pts / 2 mm ball) p50 | no point within 2 mm | supporting pale fraction p50 | supporting brightness p50 |
|---|---|---|---|---|---|---|---|
| **membrane** | 11,167 | **0.5351** | 2.1463 | **24** | **5.80%** | **0.935** | **171.0** |
| head | 60,000 | 0.3072 | 1.5778 | 55 | 3.33% | 0.000 | 63.1 |
| marker | 25,568 | 0.1937 | 0.6154 | 167 | 0.10% | 0.483 | 136.3 |

Distances in mm *(unvalidated)*.

**Verdict: hypothesis CONFIRMED — proceed to T2.** Membrane vertices are supported by
real points (94.2% have a neighbour within 2 mm), and those points are pale: a median
93.5% pale fraction against exactly 0.0% for genuine head surface. This is the
"supported by real, predominantly pale points" branch of the gate.

Two honest qualifications:

- Membrane support is **weaker** than genuine surface — density 24 versus 55 points
  per 2 mm ball, nearest distance 0.535 versus 0.307 mm. The membrane is a thin veil
  of contamination, not a solid second surface.
- **5.80% of membrane vertices have no point within 2 mm**, against 3.33% for head
  surface. A minority genuinely is Poisson bridging over the contamination. So the
  defect is mostly contamination *plus* some hallucination spanning it — removing the
  points should remove both, and the A/B confirms it does.

This also independently corroborates the density-trim result that motivated the plan:
pale area at the membrane location was identical to six decimals (2385.697941 mm²)
across `density_threshold` 0.01 → 0.05, verified again here at full precision. Points
at healthy density are exactly what a density quantile cannot reach.

---

## T2 — Filter design

`src/sfm_mvs_pipeline/postprocess/membrane_filter.py`. Runs on the cropped cloud,
before Poisson.

**Rule.** A point is removed if and only if both hold:

1. its mean RGB ≥ `pale_threshold` (default 150), and
2. it lies outside the protection sphere of **every** triangulated marker.

**Marker protection, and why it is per-marker.** The plan offered three options
(project ArUco hulls into 3D; require spatial coherence; combine colour with
distance-from-marker-centroid). The third was chosen, with one change: the protection
radius is derived **per marker from that marker's own triangulated corners**
(max corner-to-centroid distance + 5 mm margin) rather than from a global constant.

The reason is measurable. Across the 19 markers, max corner-to-centroid distance is
14.0–14.6 mm with a mean side of 20.0 mm — matching the ruler-measured 20 mm marker —
**except marker 17 at 24.7 mm**, the known-bad triangulation. A single global radius
must either be tight (and expose marker 17) or padded for the worst case (and
over-protect the other 18, sheltering membrane near them). A per-marker radius fits
each marker's real extent and automatically widens for a marker whose triangulation
is inflated, which fails in the safe direction. It also removes a hard-coded constant
that would silently mis-fit any other marker size.

Rejected alternatives: 3D hull projection reintroduces exactly the mask machinery that
`fusion_masks_report.md` showed does not reach the ~5 px bleed band; pure spatial
coherence cannot distinguish a coherent membrane sheet from coherent head surface,
and the membranes here *are* coherent sheets.

**Warn-don't-abort.** With no colours, no triangulated markers, or no scale factor,
the filter returns the cloud **unfiltered** and records the reason. Returning
unfiltered is the safe failure: removing pale points with no marker protection
available would delete the metric-scale anchors.

**Opt-in.** `--membrane-filter`, off by default, on both `run_pipeline.py` and
`resume_from_mvs.py`, matching how fusion masking was landed. The manifest always
records a `membrane_filter` block — including `enabled: false` — so a filtered run is
never mistaken for an unfiltered one.

### Failure modes

1. **Pale genuine subject surface is deleted.** Any real surface brighter than the
   threshold and farther than the protection radius from a marker will be removed.
   Harmless on a matt-black phantom; fatal on a pale subject.
2. **A marker missing from the sparse model gets no protection.** Its face is pale and
   unprotected, so it would be deleted — costing a scale anchor. Here all 19
   triangulated, but a capture with poorer marker coverage may not.
3. **Poisson can bridge the resulting gaps.** Removing points can leave a hole that
   Poisson spans with new surface — trading one membrane for another. Measured here:
   it does *not* happen (boundary length falls 3.16%), but that is an empirical result
   on this capture, not a guarantee.
4. **Threshold is a hard cut.** Contamination just below 150 survives; genuine surface
   just above it does not. Residual tan/brown streaks in the filtered mesh are exactly
   this — mid-tone (100–150) contamination the rule does not classify as pale.
5. **Over-protection near markers.** With 19 markers, the protection spheres cover a
   substantial part of the vault, so membrane immediately adjacent to a marker
   survives by design.

---

## T3 — A/B evaluation

Same depth maps, same crop, filter off vs on.

**Filter action:** of 864,952 cropped points, 68,894 were pale; 57,933 of those (84%)
fell inside a marker protection sphere and were kept; **10,961 points removed (1.27%)**.
Protection radii 19.0–29.7 mm *(unvalidated)*, median 19.2 mm.

Mesh: 506,645 → 492,631 vertices; 1,013,093 → 985,039 triangles.

### Metrics

The membrane region is derived once from the **control** arm and then held fixed for
both arms, so the two are measured over identical geometry rather than each arm
re-finding its own membrane.

| Metric | Control | Filtered | Change |
|---|---|---|---|
| **Membrane-region pale area** (target) | 3245.3 | **0.0** | **−100.00%** |
| Head-sphere dark area (genuine surface) | 98,681.0 | 101,045.4 | **+2.40%** |
| Head-sphere pale area | 8032.8 | 4354.7 | −45.79% |
| **Marker-sphere total area** | 37,440.0 | 37,422.0 | **−0.05%** |
| Marker-sphere pale area | 4439.1 | 4424.2 | −0.34% |
| Boundary length (hole rims) | 1991.3 | 1928.4 | −3.16% |
| Boundary edges | 1185 | 1135 | **−50** |

Areas in mm² *(unvalidated)*. Worst single marker: −2.75%.

### Directed surface deviation

| Direction | median | p90 | p99 | max | > 1 mm |
|---|---|---|---|---|---|
| filtered → control | 0.0202 | — | 0.8084 | 8.58 | 0.74% |
| control → filtered | 0.0214 | — | 2.4514 | 13.63 | 2.64% |

mm *(unvalidated)*. The surface that both arms share is essentially unmoved (median
~0.02 mm). The asymmetric tail in **control → filtered** is control surface with no
counterpart in the filtered arm — which is either the intended membrane removal or
collateral damage, and the distinction matters more than the number:

| Control surface lost by | vertices | % of head | in membrane region | pale | dark |
|---|---|---|---|---|---|
| > 0.5 mm | 19,307 | 3.89% | 74.3% | 56.7% | 24.1% |
| > 1.0 mm | 13,089 | 2.64% | 83.3% | 69.4% | 14.2% |
| > 2.0 mm | 6,701 | 1.35% | **92.5%** | **82.6%** | 6.2% |

The deeper the loss, the more certainly it is membrane. **Collateral loss — lost by
more than 1 mm *and* outside the membrane region — is 2,187 vertices, 0.441% of head
vertices**, mid-tone (mean brightness 111.4; 30.5% pale, 45.1% dark), median 28.4 mm
from the nearest marker. Much of this is membrane fringe just beyond the 3 mm region
tolerance rather than clean surface.

Against that, genuine dark head surface *increased* 2.40%: with the bright veil gone,
Poisson fits the underlying dark surface where contamination previously dominated.

### Visual

`heAICare/analysis/results/membrane_filter/mesh_off.png` and `mesh_on.png` (four
azimuths, true colour). The white/cream sheets at the cheeks, ears and skull base are
gone; all 19 marker faces remain crisp and complete; residual mid-tone tan streaks
persist near the base, consistent with failure mode 4.

---

## Recommendation: ADOPT, opt-in, default OFF

The A/B supports the filter on every metric the plan named, including the two that
were hard constraints — marker integrity (−0.05%) and new holes (there are fewer).
It should be **used for this capture series**, enabled explicitly:

```bash
uv run python scripts/resume_from_mvs.py --output-dir <session> \
    --image-dir <frames> --frames-manifest <frames>/manifest.json \
    --membrane-filter
```

**The default stays OFF, deliberately.** The plan allowed flipping it if the A/B
justified it, and on the numbers it does — for this scene. But the filter's premise is
"the subject is dark and the contamination is pale", which is a fact about this
phantom against this desk, not about cranial morphometry. Applied unexamined to an
infant capture with pale skin, light hair, or a dark background, rule 1 deletes the
subject. A silent default would make that failure invisible; an explicit flag makes it
a decision. The cost of the flag is one command-line argument; the cost of the wrong
default is a destroyed reconstruction that still looks plausible.

This is a **scene-dependent remedy, not a general solution.** It should be described
as such in any write-up: it demonstrates that the membranes are removable
contamination, which is a real finding about their nature, but a pipeline intended to
generalize needs the capture-side fix (dark backdrop under the chin, subject raised off
pale surfaces) that `mesh_quality_report.md` identified. The filter buys a clean mesh
for the current phantom data; it does not remove the need for that change.

---

## Reproducing

Tooling in `heAICare/analysis/` (outside this repo); results in
`heAICare/analysis/results/membrane_filter/`.

| Script | Purpose |
|---|---|
| `membrane_support.py` | T1 gate: localization, support, supporting-point colour |
| `membrane_ab.py` | T3: target metric, marker integrity, deviation, boundary/holes |
| `render_roughness.py` | Four-azimuth renders of either arm |

Unit tests for the filter live in `tests/test_membrane_filter.py` (8 tests) and cover
marker protection, the dark-point invariant, per-marker radius derivation, threshold
behaviour, and every warn-don't-abort path.
