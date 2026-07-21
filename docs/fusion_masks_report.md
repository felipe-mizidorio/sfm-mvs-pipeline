# Fusion masks: before/after evaluation and merge recommendation

**Date:** 2026-07-20
**Plan:** `heAICare/docs/ACTION_PLAN_fusion_masks_integration.md`
**Branch evaluated:** `feat/fusion-masks` @ `11b2adf` (1 commit ahead of `master` @ `76ffc0a`, 0 behind, rebases trivially)
**Recommendation:** **MERGE WITH CHANGES** — keep the plumbing, make it opt-in and off by default.

---

## Related investigations

This is investigation **2 of 4** into the same defect — the pale "membrane" contamination.
The four are one line of inquiry:

| # | Investigation | Stage attacked | Outcome |
|---|---|---|---|
| 1 | ArUco hull masks at feature extraction | SfM features | Negative — `heAICare/docs/three_way_masking_comparison.md` |
| 2 | The same masks at stereo fusion | Fusion | Negative — this report |
| 3 | Poisson density trimming | Meshing | Negative — [`mesh_quality_report.md`](mesh_quality_report.md) |
| 4 | Colour-based dense-cloud filter | Cropped cloud, pre-Poisson | **Positive, for this scene** — [`membrane_filter_report.md`](membrane_filter_report.md) |

Investigations 1–3 all failed for the same underlying reason, which this report states in
its own terms below: they attacked the symptom at a stage where the contamination is not
separable from genuine surface. Here that shows up as the mask boundary running through the
background while the bleed band sits ~5 px from the head silhouette — the masks and the
contamination are simply not the same pixels.

Investigation 4 succeeded by establishing what the membranes *are* before choosing a stage
to act at — and it succeeded **for this scene only**; the generalisable remedy remains
capture-side. See [`membrane_filter_report.md`](membrane_filter_report.md) for that framing
and [`NEXT_STEPS.md`](NEXT_STEPS.md) for current status.

---

## Summary

Fusion masks work exactly as designed, and the design does not help this capture.

Restricting stereo fusion to the ArUco convex-hull masks removes 22.5% of all fused
points, but essentially all of that removal is *outside* the head-crop sphere, where
the existing head crop already deletes it for free. Inside the crop sphere — the only
region that reaches the mesh — background contamination is **unchanged** (0.7540% →
0.7488% of head points, a 0.7% relative move), while mesh surface area within the head
sphere drops **9.3%** and the cranial-vault corner bands lose **2.7%** of their points.

The masks do not touch the thing they were meant to fix. The pale "membrane" is
occlusion-boundary bleed that sits a median of **5.3 px** from the dark head silhouette
(5.27 px unmasked → 5.25 px masked — unmoved). The hull mask boundary is nowhere near
those pixels; it runs far out in the background. So the masks cut distant background
that the crop would have removed anyway, and pay for it by cutting genuine surface in
the parts of the head that carry no markers.

This reproduces, at fusion, the same result the three-way masking comparison found at
feature extraction (`heAICare/docs/three_way_masking_comparison.md`): erosion-safe mask
margins and the bleed band are not the same pixels.

---

## What situation applied

**Workspace preserved.** The MVS workspace from `video_test_20260716_115516_arm1_baseline`
was intact (532 depth maps, 266 undistorted images), so PatchMatch did **not** re-run.
Both arms re-fused the *same* depth maps from one shared copied workspace
(`video_test_20260716_115516_fusionmask_ab`), so the comparison is exact: identical
PatchMatch output, identical sparse model, identical scale
(61.71014782061133 mm/unit in both arms), identical crop geometry
(radius 173.809 mm, `center_source: aruco_centroid`, `radius_clamped: false`).
The only variable is `StereoFusionOptions.mask_path`. No non-determinism was introduced.

`arm1_baseline` was chosen over the plan's suggested `video_test_20260714_213712`
because it is the most recent reconciled-branch session, and because it already used
hull masks at **feature extraction only** — making fusion masking the single clean
variable rather than a second change layered on a different masking state.

Both arms ran through `resume_from_mvs.py` rather than reusing the stored `arm1_baseline`
`dense.ply`, because `run_pipeline.py` leaves `dense.ply` in SfM units while
`resume_from_mvs.py` scales it to millimetres in place. Running both arms through the
same script removes that unit asymmetry from the comparison.

### Provenance

| Item | Value |
|---|---|
| Session | `video_test_20260716_115516` |
| Frames manifest | `manifest.json`, sha256 `2c89ed4ec4912e76…` (266 frames, `mask_dir: masks`) |
| Masks | ArUco convex hull, 266 files, `frame_XXXX.jpg.png` |
| sfm-mvs-pipeline | `feat/fusion-masks` @ `11b2adf` (+ changes below) |
| aruco-frame-preprocessing | `191821f` (masks predate it; unmodified here) |
| pycolmap | 4.1.0, CUDA available (RTX 4060) |
| `marker_length_mm` | 20.0 (corrected value, per plan) |
| Arm A output | `data/processed/video_test_20260716_115516_fusion_nomask` |
| Arm B output | `data/processed/video_test_20260716_115516_fusion_masked` |
| Metrics | `heAICare/analysis/results/fusion_{nomask,masked}_metrics.json`, `fusion_vault_integrity.json`, `mask_warp_verification.json` |

### Shared workspace removed (2026-07-20)

The scratch workspace both arms fused from,
`data/processed/video_test_20260716_115516_fusionmask_ab/`, was **deleted on 2026-07-20**
to reclaim 3.4 GB. It held only `mvs/` and `sparse/`, and was a copy of
`video_test_20260716_115516_arm1_baseline`'s — verified identical before deletion by
file counts, exact byte totals, and matching SHA-256 over both sparse models. It was not
a byte-for-byte duplicate: it also held a generated `mvs/fusion_masks/` (266 warped
masks) that regenerates in ~7 s, so nothing irreplaceable was lost.

Every metric in this report is stated in the tables above, so none of it depends on that
directory. Recomputing the metrics does **not** require re-fusing either: both arms'
clouds, meshes and manifests are preserved in their own directories, and
`--workspace-dir` can be pointed at `..._arm1_baseline` instead. That substitution was
checked before deletion and reproduced the no-mask arm's figures exactly. Per-arm
instructions are in `WORKSPACE_NOTE.md` inside each arm directory.

---

## T1 — Static review of the diff

The mapping is **correct**. `undistortion_maps` builds the map in the direction
`cv2.remap` actually needs: for each *undistorted* pixel it normalises through the
undistorted pinhole intrinsics, applies the *original* model's forward radial
distortion, and projects through the original intrinsics. No iterative inversion is
needed or attempted, and the radial polynomial matches COLMAP's `SIMPLE_RADIAL`
(`1 + k·r²`) and `RADIAL` (`1 + k₁·r² + k₂·r⁴`) conventions. `INTER_NEAREST` keeps
masks strictly binary; `BORDER_CONSTANT` with value 0 discards undistortion border
regions, which carry no image content. Maps are cached per camera-pair, so the cost is
one `remap` per image.

Verified against real data rather than only the branch's synthetic `MagicMock` tests
(`analysis/verify_mask_warp.py`, 14 frames sampled): every warped mask was strictly
`{0, 255}`, and mask keep-fraction was preserved (0.5755 → 0.5750).

Points confirmed rather than assumed:

- **Camera model coverage is adequate.** The run's self-calibration produced
  `SIMPLE_RADIAL`, which is supported. (The plan flagged this as unverified.)
- **Filename convention matches.** The module reads `<image name>.png` and the
  preprocessing repo writes `frame_0000.jpg.png`. Had these disagreed, every mask would
  have been silently "missing" and fusion would have run unmasked while reporting success.
- **`pycolmap.StereoFusionOptions.mask_path` exists** in 4.1.0 and accepts a string.
  The branch's real wiring is live, not a no-op.
- **Warping is mandatory, not a refinement.** Undistorted images are 861×479 while the
  original masks are 850×478, so masks cannot be handed to fusion unwarped at all.
  The geometric correction itself is small (IoU 0.982 against naive reuse, ~4.3 k pixels
  changed per frame), but the resize alone makes the module necessary.

### Issues found and fixed

1. **Unsupported camera models aborted the whole run.** `_unpack_intrinsics` raised
   `ValueError`, and mask warping happens *after* PatchMatch Stereo — so an
   unsupported model would have discarded the expensive GPU stage over an optional
   refinement. This also contradicts the established warn-don't-abort convention
   (`layout_check.py`, `self_consistency.py`). Added `undistort_masks_safe`, which logs
   the failure at ERROR level and continues with unmasked fusion, recording the reason
   in the manifest.
2. **Missing masks degraded quietly.** COLMAP fuses an image with no mask file at full
   frame, so each missing mask is a silent hole in the masking. The per-image warnings
   scrolled past among 266 lines; now a single summary WARNING reports the count, and
   `masks_missing` reaches the manifest. (This run: 266 written, 0 missing.)
3. **Provenance was asymmetric.** `fusion_masks` was written only when masks were
   active, making an unmasked run indistinguishable from one produced before the
   feature existed. Now `with_fusion_mask_provenance` always writes the block,
   including `{"enabled": false}` — the reproducibility requirement in T4.
4. **`resume_from_mvs.py` had no fusion-mask support**, so the branch's own feature
   could not be re-run from an existing workspace. Added `--fusion-masks` (this is what
   made the cheap exact comparison possible at all).

---

## T2 — Before/after results

### Point counts through the chain

| Stage | A: no fusion masks | B: fusion masks | Δ |
|---|---:|---:|---:|
| Fused dense | 1,057,182 | 819,735 | **−22.46%** |
| After SOR | 1,040,773 | 809,358 | −22.23% |
| After head crop | 865,033 | 791,407 | −8.51% |
| **Crop recovery** | **83.1%** | **97.8%** | +14.7 pp |
| Mesh triangles (post-LCC) | 1,015,331 | 1,069,949 | +5.38% |
| Mesh vertices | 507,756 | 535,138 | +5.39% |

The headline −22% and the improved crop recovery look like a win, but they are the same
fact stated twice: the masks delete far-field background *before* the crop instead of
the crop deleting it after. Crop recovery rises because the denominator shrank, not
because more head survived. What reaches the mesh is 8.5% *fewer* points.

### Background contamination — the metric the masks target

Measured with the unchanged provenance test from the 2026-07-15 cheek diagnosis
(`analysis/arm_metrics.py`; identical thresholds for both arms).

| Metric | A: no masks | B: masks | Δ |
|---|---:|---:|---:|
| Head points (r = 150 mm) | 860,624 | 790,755 | −8.12% |
| **Dark head points (coverage guard)** | **733,885** | **674,953** | **−8.03%** |
| Pale candidates tested | 9,384 | 8,107 | −13.61% |
| **Contaminated points** | **6,489** | **5,921** | −8.75% |
| **Contaminated fraction of head** | **0.7540%** | **0.7488%** | **−0.69% relative** |
| Contamination median distance to dark silhouette | 5.27 px | 5.25 px | unmoved |

Absolute contaminated points fall 8.75%, but head points fall 8.12% — contamination
drops only in proportion to the points removed everywhere. As a *rate*, contamination
is flat. The masks did not preferentially remove membrane points; they removed points
indiscriminately and the membrane came along at its existing share.

The unchanged 5.3 px median distance is the mechanism, stated directly: contaminated
points hug the dark head silhouette, and the hull mask boundary is nowhere near there.

### Vault integrity and where the loss falls

| Metric | A: no masks | B: masks | Δ |
|---|---:|---:|---:|
| Corner-band points (≤15 mm of 76 triangulated corners, 19 markers) | 766,249 | 745,686 | −2.68% |
| Worst single marker (id 14) | 20,585 | 16,299 | **−20.82%** |
| Mesh area inside head sphere | 113,463 mm² | 102,960 mm² | **−9.26%** |

Cropped-cloud points binned by distance to the nearest triangulated marker centroid,
split by colour (dark = black phantom surface, pale = mug/desk background):

| Distance from nearest marker | All Δ | Dark Δ | Pale Δ | Dark points lost |
|---|---:|---:|---:|---:|
| 0–20 mm | −3.0% | −2.5% | −3.8% | 7,251 |
| 20–40 mm | −1.8% | −1.4% | −2.7% | 4,830 |
| 40–60 mm | −10.3% | −10.6% | −3.5% | 4,624 |
| 60–80 mm | −66.0% | **−66.6%** | −46.7% | 22,254 |
| 80–100 mm | −86.1% | **−87.4%** | −78.3% | 14,950 |
| 100–125 mm | −89.2% | **−90.5%** | −75.3% | 7,344 |
| 125–150 mm | −70.6% | −70.7% | −100% | 1,235 |

This is the decisive table. Near the markers the masks are harmless — the hull contains
them, and the anchor holds (0–40 mm loses 1–3%). Beyond 60 mm the masks remove
two thirds to nine tenths of everything, and **~62.5 k of the ~66 k points removed
inside the crop sphere are dark**, against only ~3.6 k pale. A roughly **17:1 ratio of
dark-to-pale removal** is the opposite of what a background filter should do: the mug
and desk in this scene are pale.

One honest caveat: colour alone cannot prove every far-field dark point is phantom
surface rather than shadow. But the direction is unambiguous, and the 9.3% drop in mesh
area *inside the head sphere* is an independent, colour-free confirmation that real
reconstructed surface was lost.

### Mask warping quality

266 masks written, **0 missing**, no fallbacks, no unsupported-model degradations.
All warped masks strictly binary. Camera model `SIMPLE_RADIAL` throughout.

### Cost

| Item | Value |
|---|---|
| Mask warping (266 images) | **7.1 s** |
| Arm A total wall clock (fusion → SOR → crop → scale → Poisson) | 135 s |
| Arm B total wall clock | 164 s |
| Net overhead | **+29 s (+21%)**, of which 7.1 s is warping |
| Disk | +266 mask files in `mvs/fusion_masks/` |
| New dependencies | **none** (cv2, numpy, pycolmap already required) |

The cost is genuinely modest. Cost is not the reason for the recommendation.

---

## T3 — Recommendation

### MERGE WITH CHANGES — plumbing in, default off

Against the plan's success criterion — *"Do fusion masks measurably reduce background
contamination in the dense cloud without harming vault coverage, at acceptable cost?"* —
the answer is **no**: contamination is statistically unchanged and coverage is
measurably harmed. Merging the branch as written would silently make every masked run
9.3% worse in mesh coverage, because it activates whenever the manifest has a
`mask_dir` with no way to decline.

But the module itself is correct, tested, cheap, dependency-free, and is the exact
plumbing the three-way comparison already identified as the prerequisite for the next
experiment ("eroded tight mask at FUSION only"). Deleting it would mean rebuilding it.

So: merge the capability, do not merge the default. Changes made on the branch:

1. **`--fusion-masks` flag on `run_pipeline.py`, off by default.** Fusion masking no
   longer activates implicitly. Feature-extraction masking is unaffected.
2. **`--fusion-masks` on `resume_from_mvs.py`**, so the experiment is reproducible from
   an existing workspace without re-running PatchMatch.
3. **`undistort_masks_safe`** — warn and continue unmasked rather than aborting after
   PatchMatch.
4. **Symmetric `fusion_masks` provenance**, always recorded including `enabled: false`,
   with `masks_written` / `masks_missing` / `warp_seconds`.
5. **Loud missing-mask summary warning.**

The flag's help text states the measured penalty and points here, so the next person
who reaches for it sees the evidence before spending GPU time.

### What would actually fix the membrane

Not this mask. The contamination sits ~5 px from the silhouette; any mask whose boundary
is erosion-safe for the head is, by construction, far outside that band. The remaining
levers are unchanged from the three-way comparison: a **tight silhouette mask eroded to
the head contour** applied at fusion (the plumbing now exists and is opt-in), a
**silhouette-aware point filter** after fusion, or capture-side changes (dark backdrop
under the chin, raising the head off pale surfaces).

---

## Notes and limitations

- **Millimetre values are unvalidated.** `known_distances_mm` is still empty and no
  physical measurement was taken. Every mm figure here (crop radius 173.8 mm, head
  sphere 150 mm, mesh area 102,960 mm², distance bins) inherits the ArUco-derived scale
  and is **not independently verified**. All conclusions rest on relative comparisons
  between two arms that share one scale factor exactly, so the verdict does not depend
  on absolute accuracy.
- **Single session, single capture.** One phantom, one lighting setup, one mask style.
  The conclusion "ArUco hull masks do not help at fusion" is well supported; "fusion
  masking never helps" is not tested and is not claimed.
- `scale_self_consistency` fired its known warning (CV 6.6%, outlier marker 17) in both
  arms identically — pre-existing, matches the recorded baseline, not affected here.
- `tests/test_plotly_viz.py` has a pre-existing unused-import lint error on `master`,
  untouched here (out of scope: one variable at a time).
- Nothing in the MobileSAM arm2/arm3 branches was touched, enabled, or tested, and no
  Poisson/SOR/crop parameters were tuned.
