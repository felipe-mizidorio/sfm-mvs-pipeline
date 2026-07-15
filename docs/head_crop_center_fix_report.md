# Head-crop center fix — before/after validation report

**Plan:** `heAICare/docs/ACTION_PLAN_head_crop_center_fix.md` · **Branch:** `feat/phase-b-intrinsics-prior`
**Run compared:** `video_test_20260714_213712` — `_baseline` (before, `run_pipeline.py` 2026-07-15) vs `_centerfix` (after, `resume_from_mvs.py --skip-fusion` on the identical `dense.ply` + sparse model, 2026-07-15). Fusion was not re-run, so the comparison is exact: same input cloud, same scale (144.4943 mm/unit), only the crop centre logic changed.

## What changed (T1 + T2)

- Crop centre = centroid of the triangulated ArUco marker corners when ≥ `HEAD_CROP_MIN_MARKER_CORNERS` (8, i.e. two markers) are available; the optical-axis intersection is retained only as a fallback below that threshold. Manifest records `center_source`.
- Radius clamp unchanged (min 140 / max 250 mm) but now a logged, manifest-recorded sentinel: `radius_clamped` (`false | "min" | "max"`) + `radius_unclamped_mm`.

## Point counts through the chain

| Stage | Before | After |
|---|---|---|
| `dense.ply` | 188,088 | 188,088 (same file) |
| `dense_filtered.ply` (SOR) | 184,201 | 184,201 |
| `dense_filtered_cropped.ply` | **54,778 (29.7%)** | **162,049 (88.0%)** |
| `mesh.ply` | 147,618 verts / 295,160 tris | 336,887 verts / 673,032 tris |

## Manifest (`head_crop`, after)

- `center_source`: `aruco_centroid` (76 corners)
- `center_sfm`: `[0.2429, 1.7136, 1.8261]` — 186.6 mm from the old optical-axis centre `[0.3566, 2.5803, 2.7770]`
- `radius_sfm_units`: 1.7302 (250.0 mm)
- `radius_clamped`: `"max"`, `radius_unclamped_mm`: 285.0

## Vault recovery (the original failure symptom)

- Cropped cloud points within 40 mm of a triangulated marker corner (corners sit on the crown cap): **31,929 → 133,633**.
- Final-mesh vertices in the crown region (>50 mm toward the crown from the centroid, >140 mm radius): **40 → 53,164**.
- Visual projection along the old→new centre axis: the before cloud terminates at the old crop boundary with the entire corner band empty above it (vault sliced off); the after cloud forms a complete dome enclosing all 76 corners.

**Verdict: the cranial vault is recovered.** The crop keeps 88.0% of the SOR-filtered points, matching the plan's predicted 88.0%.

## Sentinel finding (expected, out of scope here)

The clamp sentinel **fired** on this run: median corner distance from the centroid is 185 mm, +100 mm margin = 285 mm > 250 mm ceiling. Per the T2 rationale this now signals an upstream problem rather than a crop bug — a 185 mm median corner-to-centroid distance implies a marker cloud ~2× larger than a neonatal head, consistent with the plan's carried-forward limitation that **metric scale is unvalidated** (`known_distances_mm` is empty, so the A3 layout check is inert). The mesh is geometrically complete but its millimetre scale must not be trusted until a measured rigid cap layout exists. Do not tune Taubin/density thresholds against this run's mm values.
