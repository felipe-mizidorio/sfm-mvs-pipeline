# Next steps and known limitations

**Last updated:** 2026-07-21

A living planning document: what is open, what it is blocked on, and what a reader should
not assume. It deliberately does **not** repeat the analysis in the three reports — where a
question has been investigated, this file states the status and links to the report that
did the work.

| Report | Question it settled |
|---|---|
| [`mesh_quality_report.md`](mesh_quality_report.md) | Where mesh roughness comes from; what `mesh.yaml` can and cannot reach |
| [`fusion_masks_report.md`](fusion_masks_report.md) | Whether ArUco hull masks at fusion remove silhouette contamination |
| [`membrane_filter_report.md`](membrane_filter_report.md) | What the membranes actually are, and a colour filter that removes them |

The working principle: **no stacking unvalidated complexity before the GPU baseline.**

---

## 1. Metric scale is unvalidated — read every millimetre figure with this in mind

`configs/aruco.yaml` has `layout_check.known_distances_mm: []`. The layout sanity check
therefore has nothing to compare against and never fires.

Scale is recovered from triangulated ArUco corners against `marker_length_mm`, so it is
**internally consistent but externally unchecked**. Every millimetre figure in all three
reports inherits this, and each of them says so. In practice this means the reports measure
**precision, not accuracy**: relative comparisons within one shared scale factor are safe;
absolute distances are not.

This is the single most important caveat in the project. A morphometric measurement
produced today would carry an unquantified systematic scale error.

**Blocked on:** physical access to the cap and a measuring instrument, to populate
`known_distances_mm` with at least two independent inter-marker distances.

**Related risk:** scale recovery is non-fatal by design — a `RuntimeError` is caught and
logged as a warning, and the layout check warns without aborting. A run whose scale
recovery failed still produces output that looks entirely plausible. When scale is finally
validated, decide whether these should stay warn-only or become opt-in hard gates.

## 2. No morphometry module exists

**Read this before searching commit messages.** Commit `981722f` is titled *"feat: add
neonatal cranial morphometry support"*. It did **not** add morphometric measurement. What
it added was the **scale** module — `src/sfm_mvs_pipeline/scale/aruco_scale.py`, plus
`configs/aruco.yaml` entries, feature-extraction changes and `tests/test_scale.py`.

There is no cephalic-index, no circumference, no landmark extraction, no asymmetry metric
anywhere in `src/sfm_mvs_pipeline/`. The package directories are `sfm`, `mvs`, `mesh`,
`scale`, `postprocess`, `evaluation`, `visualization`, `pipeline` — there is no
`morphometry`.

Anyone reading the commit log alone will conclude the opposite. That is the reason this
entry exists.

**Gated on:** the GPU baseline, and on limitation 1 — morphometry on an unvalidated scale
would produce numbers with no defensible units.

## 3. The membrane colour filter is scene-dependent

Opt-in via `--membrane-filter`, **default OFF**. Documented in
[`membrane_filter_report.md`](membrane_filter_report.md).

The limitation is **generality, not efficacy**. On the target scene the filter removes 100%
of the pale area at the membrane location while marker faces lose 0.05% — it works. But its
premise is "the subject is dark and the contamination is pale", which is a property of this
phantom against this desk, not of cranial morphometry.

That premise does not hold for a real domestic capture. It specifically fails **if the cap
fabric is white**: the marker-protection sphere stops being separable from background, and
the rule that currently protects the scale anchors would instead be deleting pale genuine
surface. On a pale-skinned or fair-haired infant, the filter deletes the subject.

The default stays OFF so that this is a decision rather than an invisible failure. The
generalisable remedy is capture-side (section 6), not a better threshold.

## 4. Fusion masking is opt-in and off

Opt-in via `--fusion-masks`, **default OFF**. Documented in
[`fusion_masks_report.md`](fusion_masks_report.md).

Measured to remove genuine head surface far out of proportion to what it removes from
background — roughly 17:1 against background — because the hull mask boundary runs through
the background while the contamination it was meant to catch sits ~5 px from the head
silhouette. The masks and the bleed band are not the same pixels.

The plumbing is **retained deliberately**, as the prerequisite for the eroded tight-mask
experiment (not started — see *Out of scope*). It should not be enabled in the meantime.

## 5. The DeepArUco++ comparison — code now pinned, weights still not

Provenance for the vendored DeepArUco++ code was resolved on 2026-07-21 and is recorded in
`aruco-frame-preprocessing/src/deeparuco_vendor/PROVENANCE.md`: upstream
`AVAuco/deeparuco` `impl/` at tag `IMAVIS` (`03f4982`), vendored 2026-06-01, with one
post-vendoring `arctan2` fix that makes the local copy diverge from stock upstream.

**Two things remain open:**

- The comparison runs against *patched* DeepArUco++, not stock. Any claim resting on it
  must be described that way.
- The three model weight files are downloaded at run time from a URL pinned to a **branch,
  not a commit**, and are not checksummed. They are the remaining unpinned input: a future
  download could differ from the one that produced a given `comparison.json` without that
  being detectable.

Recording weight hashes in the comparison output, or pinning the URL to a commit SHA, would
close this. Both are behavioural changes to `deeparuco_comparison.py`.

## 6. Capture-protocol remedies are outstanding

Both [`mesh_quality_report.md`](mesh_quality_report.md) and
[`membrane_filter_report.md`](membrane_filter_report.md) converge on the same conclusion
from different directions: the remaining defects are **capture-side and non-configurable**.
No parameter in this pipeline reaches them.

- **Markers printed into the cap fabric without relief.** Taped-on markers on a matt-black
  textureless phantom are the measured source of the roughness — mesh roughness rises 2.2×
  from the flattest decile to the high-contrast marker borders.
- **More camera passes over thin-coverage regions.** Roughness tracks frame coverage
  monotonically (2–3 views 0.289 mm → 10+ views 0.191 mm).
- **Dark non-reflective backdrop under the chin.**
- **Subject raised off pale surfaces.**

The last two are what actually fixes the membranes in general; the colour filter (section 3)
buys a clean mesh for the current phantom data without removing the need for them.

## 7. Environment debt: 44 pyright errors

`uv run pyright src/` reports **44 errors, 0 warnings** (verified 2026-07-21). Every one is
`reportAttributeAccessIssue` against `pycolmap` — `extract_features`, `Device`,
`Reconstruction`, `incremental_mapping`, `match_exhaustive`, and similar — across
`sfm/feature_extraction.py`, `sfm/feature_matching.py` and `sfm/reconstruction.py`.

These are stub gaps in the CPU `pycolmap` wheel, not defects in this code. Verified
identical on pristine `master`. **Deliberately not suppressed:** a blanket ignore would also
hide real attribute errors in the same modules. Revisit if a wheel ships working type stubs.

## 8. Poisson is not order-deterministic across processes

Recorded in [`mesh_quality_report.md`](mesh_quality_report.md). Open3D's Poisson
reconstruction does not produce a stable vertex ordering between processes, so **mesh
hashes and vertex indices are not valid comparison tools** in this project. Any future A/B
must compare geometry — ray-cast distance in both directions — as the existing reports do.

Worth knowing before writing any regression test that hashes a mesh.

## 9. Taubin smoothing is off, and that is a decision to revisit

`configs/mesh.yaml` has `taubin_smoothing.iterations: 0`. Turned off so that every
morphometric measurement traces to fused geometry with no intervening filter. The tuned
`lambda_filter: 0.5` / `mu: -0.53` and the measured knee (5 iterations) are retained in the
config, so re-enabling is a one-token change.

Revisit when morphometry exists and can say whether smoothing helps or harms a measurement
— which is the only evidence that should decide it.

---

## Out of scope right now

Not started, and deliberately so:

- **Eroded tight-mask experiment** — the reason the fusion-mask plumbing is retained.
- **Populating `known_distances_mm`** — blocked on physical access to the cap.
- **Promoting DeepArUco++ to production detection** — the comparison is not yet a
  defensible basis for it (section 5).
- **Any change to mask geometry, fusion, or mesh parameters** — gated on the GPU baseline.

---

## Reproducibility note

Analysis tooling lives in `heAICare/analysis/`, outside this repository, with results under
`heAICare/analysis/results/`. The pipeline manifest records library versions, the SHA-256
of the frames manifest, resolved configs and known non-determinism sources.

One gap: the shared MVS workspace both fusion-mask arms fused from
(`video_test_20260716_115516_fusionmask_ab/`) was **deleted on 2026-07-20** to reclaim
3.4 GB. It was verified identical to `arm1_baseline`'s before deletion, and every metric is
stated in the report, so no conclusion depends on it — but re-running that A/B means
regenerating the workspace first.
