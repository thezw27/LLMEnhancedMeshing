"""
feature_extraction.py — turn a raw CFD solution field into a small, LLM-readable
JSON summary instead of a firehose of per-vertex numbers.

This is deliberately "classical, not clever": a least-squares vertex gradient
(no full Hessian — that's the expensive step this pipeline is trying to avoid
paying for on every adaptation cycle) plus percentile thresholding and
connected-component grouping to find candidate feature regions (shocks,
strong thermal gradients, wakes, etc). The LLM's job starts *after* this —
reading the summary, describing it in plain English, asking the human for
input, and deciding refinement intent. This module never decides anything
about mesh sizing; it only observes and reports.

Detection is multi-scale/iterative ("peel off"), not a single global
percentile pass, specifically so a strong dominant feature (e.g. the main
shock on a wedge) doesn't hide a much weaker one (a small secondary shock, a
diffuse wake) in the same field: a single global percentile cutoff is a
majority-vote-shaped statistic, and once one feature's gradient dwarfs
everything else, a weaker-but-real feature simply never reaches that same
cutoff. Each pass instead computes its percentile only over vertices no
earlier pass has already claimed, so removing a dominant feature from the
pool lets the next-strongest one surface on its own terms. This is generic —
no assumption about geometry, field, or number of features — unlike an
earlier version of this pipeline that had ~600 lines of size-field-building
logic overtly tuned against one specific test case (see size_field_builder.py
history); nothing here should be tuned to any one case's specific numbers.

None of this uses anything beyond the input mesh's own topology/geometry and
the requested driver_fields' point-data values — no CAD/geometry model, no
case metadata (flow conditions, known analytic shock angles, etc.), no mesh
quality data. Purely data-driven detection like this cannot possibly agree
with known physics/geometry on every case; it's a complement to (not a
replacement for) a human or LLM sanity-checking candidate regions against
the actual case before writing region_spec.json, not a guarantee of a
correct answer by itself.

Output shape (see compute_feature_summary below) is capped in size regardless
of mesh resolution (a handful of regions with summary stats each) so it reads
fine as LLM context even for a 100k-vertex case.
"""

from __future__ import annotations

import json
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from vtu_io import vertex_adjacency

DEFAULT_GRADIENT_PERCENTILE = 90.0
DEFAULT_GRADIENT_PERCENTILE_REASON = (
    "90th percentile of |grad(field)|, recomputed each pass over only the vertices no "
    "earlier pass has already claimed (see module docstring) -- a broad, generically "
    "reasonable per-pass cutoff with no a priori knowledge of the case. Override via "
    "case_config['gradient_percentile'] if a case's real features are much weaker/"
    "stronger relative to the bulk of the field than that (e.g. a very smooth flow "
    "where even mild gradients are meaningful, or very noisy solution data where the "
    "90th percentile still lets noise through)."
)

DEFAULT_MIN_REGION_FRACTION = 0.0002
DEFAULT_MIN_REGION_FRACTION_REASON = (
    "the 'drop this connected component as noise' cutoff scales with mesh vertex "
    "count (fraction * n_vertices, floor 3) instead of a flat number, so a small but "
    "real feature isn't dropped just because the mesh is sparse, and a bigger mesh "
    "doesn't let bigger noise blobs through just because they clear a fixed count. "
    "Override via case_config['min_region_fraction'], or case_config['min_region_size'] "
    "for a direct absolute-vertex-count override, if a case's real features are "
    "smaller/larger than this implies."
)
_MIN_REGION_SIZE_FLOOR = 3

DEFAULT_MAX_DETECTION_PASSES = 6
DEFAULT_MAX_DETECTION_PASSES_REASON = (
    "each pass re-thresholds only the mesh not yet claimed by an earlier pass, so a "
    "much weaker secondary feature isn't hidden by a dominant one's skew on the "
    "gradient distribution; passes stop early once one finds nothing new. Override "
    "via case_config['max_detection_passes'] for a case with many distinct feature "
    "scales, or to cap runtime on a very large mesh."
)

DEFAULT_MIN_RELATIVE_THRESHOLD = 0.01
DEFAULT_MIN_RELATIVE_THRESHOLD_REASON = (
    "stops peeling off further passes once a pass's threshold drops below this "
    "fraction of the *first* (strongest) pass's threshold for the same field -- real "
    "solution data is never perfectly flat away from actual features (turbulence, "
    "interpolation, solver noise), so without a floor the peel-off keeps finding "
    "technically-above-percentile-but-not-actually-a-feature connected components "
    "indefinitely (confirmed: an early version of this function returned 40-56 "
    "'candidate regions' on a real CFD case before this floor was added, defeating "
    "the point of a compact, LLM-readable summary). 1% is a broad two-orders-of-"
    "magnitude default, not tuned to any one case -- override via "
    "case_config['min_relative_threshold'] if a case has a genuinely weaker-but-real "
    "feature beyond that range, or noisier data that needs a tighter floor."
)


def auto_min_region_size(n_vertices: int, min_region_fraction: float = DEFAULT_MIN_REGION_FRACTION) -> int:
    """Density-scaled floor for 'this connected component is real, not noise'
    -- see DEFAULT_MIN_REGION_FRACTION_REASON."""
    return max(_MIN_REGION_SIZE_FLOOR, int(round(n_vertices * min_region_fraction)))


def compute_vertex_gradient(points: np.ndarray, adjacency: list[set], field: np.ndarray) -> np.ndarray:
    """Per-vertex gradient of a scalar field via local weighted least squares
    over the 1-ring neighborhood: for vertex i with neighbors j,
    solve for g in   (x_j - x_i) . g  ~=  f_j - f_i   for all j,
    weighted by 1/|x_j - x_i| to downweight distant/skewed neighbors.

    This is the standard cheap alternative to a full Hessian recovery pass —
    good enough to *locate and orient* features, not intended to feed a
    curvature-accurate sizing formula.

    Performance note: this solves the 3-unknown least-squares system via the
    normal equations (A^T A) g = A^T b (a direct 3x3 solve) instead of
    np.linalg.lstsq's general SVD-based path -- lstsq was the single biggest
    line-level cost in feature detection on profiling (a few thousand tiny
    SVDs adds up). A^T A can be singular for a genuinely degenerate
    neighborhood (fewer than 3 neighbors, or all of them collinear); a small
    ridge term keeps the solve well-posed in that case instead of raising,
    at the cost of a slightly damped-toward-zero gradient there rather than
    lstsq's minimum-norm answer -- an acceptable trade for a "cheap,
    locate-and-orient-only" estimate per the module's own docstring above.
    """
    n = len(points)
    grad = np.zeros((n, 3))
    neighbor_arrays = [np.fromiter(adjacency[i], dtype=np.int64, count=len(adjacency[i]))
                       for i in range(n)]
    eye3 = np.eye(3)
    for i in range(n):
        nbrs = neighbor_arrays[i]
        if len(nbrs) == 0:
            continue
        dx = points[nbrs] - points[i]
        df = field[nbrs] - field[i]
        dist = np.linalg.norm(dx, axis=1)
        dist[dist < 1e-12] = 1e-12
        w = 1.0 / dist
        A = dx * w[:, None]
        b = df * w
        AtA = A.T @ A
        Atb = A.T @ b
        grad[i] = np.linalg.solve(AtA + 1e-12 * eye3, Atb)
    return grad


def compute_vertex_gradients_multi(points: np.ndarray, adjacency: list[set],
                                    fields: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Same per-vertex weighted-least-squares gradient as
    compute_vertex_gradient, but for several fields at once, sharing the
    geometry-only work (neighbor lookup, displacement vectors, distance
    weights, and A^T A) across all of them instead of recomputing it once
    per field. Only A^T b and the final solve actually depend on which
    field's values are being differenced.

    Worth it specifically because this pipeline always runs feature
    detection over several driver fields together (e.g. mach_number,
    pressure, temperature) -- profiling showed the per-field geometry setup
    was the majority of compute_vertex_gradient's cost, so with N fields
    this cuts that part from N x to 1x.
    """
    n = len(points)
    field_names = list(fields.keys())
    grads = {name: np.zeros((n, 3)) for name in field_names}
    eye3 = np.eye(3)

    for i in range(n):
        nbrs = np.fromiter(adjacency[i], dtype=np.int64, count=len(adjacency[i]))
        if len(nbrs) == 0:
            continue
        dx = points[nbrs] - points[i]
        dist = np.linalg.norm(dx, axis=1)
        dist[dist < 1e-12] = 1e-12
        w = 1.0 / dist
        A = dx * w[:, None]
        AtA = A.T @ A
        AtA_reg = AtA + 1e-12 * eye3
        for name in field_names:
            field = fields[name]
            df = field[nbrs] - field[i]
            b = df * w
            Atb = A.T @ b
            grads[name][i] = np.linalg.solve(AtA_reg, Atb)
    return grads


def _connected_components_on_mask(adjacency: list[set], mask: np.ndarray) -> np.ndarray:
    """Label connected components among vertices where mask is True, using
    mesh topology (not spatial proximity) so regions respect actual connectivity."""
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return np.full(len(mask), -1)
    remap = {v: k for k, v in enumerate(idx)}
    rows, cols = [], []
    for v in idx:
        for nb in adjacency[v]:
            if mask[nb]:
                rows.append(remap[v])
                cols.append(remap[nb])
    n_sub = len(idx)
    graph = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_sub, n_sub)) if rows else csr_matrix((n_sub, n_sub))
    n_comp, labels_sub = connected_components(graph, directed=False)
    labels = np.full(len(mask), -1)
    labels[idx] = labels_sub
    return labels


def _build_region_entry(points: np.ndarray, field: np.ndarray, grad: np.ndarray,
                         grad_mag: np.ndarray, region_mask: np.ndarray) -> dict:
    """Build one candidate-region summary dict from a connected-component mask."""
    pts = points[region_mask]
    g = grad[region_mask]
    g_mean = g.mean(axis=0)
    g_dir = (g_mean / np.linalg.norm(g_mean)).tolist() if np.linalg.norm(g_mean) > 1e-12 else [0, 0, 0]

    # How much this region's own detected vertices actually scatter
    # cross-feature (perpendicular to its dominant/along-feature PCA
    # direction) -- this is real, measured data about THIS region on THIS
    # case, meant to inform core_distance/falloff_distance instead of
    # reusing a value tuned on a previous, possibly very different case. A
    # region whose real detected vertices scatter by several mm needs a
    # core+falloff wide enough to cover most of that scatter, or most of
    # its own real vertices will only partially blend toward the requested
    # tight size regardless of how good detection/smoothing is -- that's a
    # property of the region's own data, not something core_distance
    # tuning alone can outrun.
    scatter_stats = None
    if len(pts) >= 4:
        centroid = pts.mean(axis=0)
        _, _, vt = np.linalg.svd(pts - centroid, full_matrices=False)
        along = vt[0]
        perp = (pts - centroid) - np.outer((pts - centroid) @ along, along)
        perp_dist = np.linalg.norm(perp, axis=1)
        scatter_stats = {
            "median": float(np.median(perp_dist)),
            "p90": float(np.percentile(perp_dist, 90)),
            "max": float(perp_dist.max()),
            "note": ("perpendicular distance of this region's own detected vertices to their "
                     "dominant (along-feature) PCA direction -- a data-driven starting point for "
                     "core_distance (~median) and core_distance+falloff_distance (~p90), not a "
                     "hard rule"),
        }

    return {
        "n_vertices": int(region_mask.sum()),
        "bbox_min": pts.min(axis=0).tolist(),
        "bbox_max": pts.max(axis=0).tolist(),
        "centroid": pts.mean(axis=0).tolist(),
        "mean_gradient_magnitude": float(grad_mag[region_mask].mean()),
        "max_gradient_magnitude": float(grad_mag[region_mask].max()),
        "dominant_gradient_direction": g_dir,  # normal-ish direction across the feature
        "field_value_range": [float(field[region_mask].min()), float(field[region_mask].max())],
        "cross_feature_scatter": scatter_stats,
    }


def _detect_regions_multiscale(points: np.ndarray, adjacency: list[set], field: np.ndarray,
                                grad: np.ndarray, grad_mag: np.ndarray, gradient_percentile: float,
                                min_region_size: int, max_passes: int,
                                min_relative_threshold: float = DEFAULT_MIN_RELATIVE_THRESHOLD
                                ) -> tuple[list[dict], list[float]]:
    """Iterative 'peel off' multi-scale detection -- see module docstring for
    why one global percentile can't see a weak feature once a strong one
    dominates. Each pass thresholds only vertices no earlier pass claimed,
    then removes whatever it finds from the pool before the next pass.
    Stops once a pass's threshold falls below min_relative_threshold times
    the first (strongest) pass's threshold -- see
    DEFAULT_MIN_RELATIVE_THRESHOLD_REASON for why that floor exists."""
    n = len(points)
    remaining = np.ones(n, dtype=bool)
    regions = []
    thresholds_used = []
    first_threshold = None

    for _ in range(max_passes):
        remaining_idx = np.where(remaining)[0]
        if len(remaining_idx) < min_region_size:
            break
        threshold = float(np.percentile(grad_mag[remaining_idx], gradient_percentile))
        # A threshold of ~0 means the remaining pool has no real gradient
        # structure left to separate (pure flat/background, or numerical
        # noise) -- stop instead of reporting "everything left over" as a
        # region, which is what a degenerate near-zero threshold would do.
        if threshold <= 1e-9:
            break
        if first_threshold is None:
            first_threshold = threshold
        elif threshold < min_relative_threshold * first_threshold:
            break
        mask = remaining & (grad_mag >= threshold)
        if not mask.any():
            break
        labels = _connected_components_on_mask(adjacency, mask)

        found_any = False
        for label in sorted(set(labels[labels >= 0])):
            region_mask = labels == label
            if region_mask.sum() < min_region_size:
                continue
            found_any = True
            regions.append(_build_region_entry(points, field, grad, grad_mag, region_mask))
            remaining[region_mask] = False

        if not found_any:
            break
        thresholds_used.append(threshold)

    return regions, thresholds_used


def compute_feature_summary(
    mesh: dict,
    driver_fields: list[str],
    gradient_percentile: float = DEFAULT_GRADIENT_PERCENTILE,
    min_region_size: int | None = None,
    min_region_fraction: float = DEFAULT_MIN_REGION_FRACTION,
    max_passes: int = DEFAULT_MAX_DETECTION_PASSES,
    min_relative_threshold: float = DEFAULT_MIN_RELATIVE_THRESHOLD,
) -> dict:
    """Build the compact JSON feature summary for one or more driver fields.

    Parameters
    ----------
    mesh : dict from vtu_io.read_vtu
    driver_fields : e.g. ["temperature", "pressure"] — must exist in mesh point_data
    gradient_percentile : per-pass percentile of |grad(field)| (over
        whatever vertices no earlier pass has claimed) that counts as a
        "candidate feature" vertex. See DEFAULT_GRADIENT_PERCENTILE_REASON.
    min_region_size : drop connected components smaller than this (noise).
        If None (default), auto_min_region_size(n_vertices, min_region_fraction)
        is used instead of a flat number -- see DEFAULT_MIN_REGION_FRACTION_REASON.
    min_region_fraction : only used when min_region_size is None.
    max_passes : safety cap on peel-off iterations; see DEFAULT_MAX_DETECTION_PASSES_REASON.
    min_relative_threshold : stop peeling off once a pass's threshold falls
        below this fraction of the field's first (strongest) pass threshold;
        see DEFAULT_MIN_RELATIVE_THRESHOLD_REASON.

    Returns
    -------
    A JSON-serializable dict: global stats per field, plus a list of candidate
    regions (bounding box, centroid, dominant gradient direction, size, and
    representative field values) — this is what gets handed to the LLM. Each
    field's region list can include regions found across several detection
    passes at different effective thresholds (see gradient_thresholds_used),
    not just whatever cleared one single global cutoff.
    """
    points = mesh["points"]
    adjacency = vertex_adjacency(mesh)

    resolved_min_region_size = (
        min_region_size if min_region_size is not None
        else auto_min_region_size(len(points), min_region_fraction)
    )

    summary = {"n_vertices": len(points), "fields": {}}

    for field_name in driver_fields:
        if field_name not in mesh["point_data"]:
            continue
        field = mesh["point_data"][field_name]
        if field.ndim > 1:
            field = np.linalg.norm(field, axis=1)  # vector field -> magnitude

        grad = compute_vertex_gradient(points, adjacency, field)
        grad_mag = np.linalg.norm(grad, axis=1)

        regions, thresholds_used = _detect_regions_multiscale(
            points, adjacency, field, grad, grad_mag,
            gradient_percentile, resolved_min_region_size, max_passes,
            min_relative_threshold,
        )

        # sort strongest features first so the LLM sees the important ones up front
        regions.sort(key=lambda r: r["max_gradient_magnitude"], reverse=True)

        summary["fields"][field_name] = {
            "value_range": [float(field.min()), float(field.max())],
            "mean": float(field.mean()),
            "std": float(field.std()),
            "gradient_magnitude_range": [float(grad_mag.min()), float(grad_mag.max())],
            "gradient_thresholds_used": thresholds_used,
            "min_region_size_used": resolved_min_region_size,
            "candidate_regions": regions,
        }

    return summary


def _self_test_multiscale_detection():
    """Synthetic proof that peel-off multi-pass detection finds a weak
    feature a single global-percentile pass would miss entirely: a 1D chain
    of 1000 vertices with three constant-slope zones -- a strong 'shock'
    (slope 10, 300 vertices), a flat background (slope 0, 600 vertices), and
    a weak-but-real 'shock' (slope 2, ~99 vertices). A single 90th-percentile
    pass over the whole chain lands right at the strong zone's own value (its
    300 vertices are exactly the top 30%), so the weak zone (2 << 10) never
    clears it -- but after peeling the strong zone off, the weak zone is the
    top ~14% of what's left and clears easily."""
    n = 1000
    x = np.arange(n, dtype=float)
    points = np.column_stack([x, np.zeros(n), np.zeros(n)])
    cells = [("line", np.array([[i, i + 1] for i in range(n - 1)]))]

    field = np.zeros(n)
    for i in range(1, n):
        if i <= 300:
            step = 10.0     # strong feature zone
        elif i <= 900:
            step = 0.0      # flat background
        else:
            step = 2.0      # weak feature zone
        field[i] = field[i - 1] + step

    mesh = {"points": points, "cells": cells, "point_data": {"f": field}, "cell_data": {}}

    summary = compute_feature_summary(mesh, ["f"])
    regions = summary["fields"]["f"]["candidate_regions"]
    thresholds = summary["fields"]["f"]["gradient_thresholds_used"]
    print(f"detection passes used: {len(thresholds)}, thresholds: {thresholds}")
    print(f"regions found: {[(r['n_vertices'], round(r['mean_gradient_magnitude'], 2)) for r in regions]}")

    assert len(thresholds) >= 2, (
        f"expected at least 2 detection passes to find both the strong and weak zones, got {len(thresholds)}"
    )
    assert len(regions) >= 2, f"expected >=2 candidate regions (strong + weak), got {len(regions)}"

    strong = max(regions, key=lambda r: r["mean_gradient_magnitude"])
    weak = min(regions, key=lambda r: r["mean_gradient_magnitude"])
    assert strong is not weak
    assert strong["mean_gradient_magnitude"] > 8.0, strong
    assert 1.0 < weak["mean_gradient_magnitude"] < 4.0, weak
    assert strong["n_vertices"] > 200, strong
    assert 50 < weak["n_vertices"] < 150, weak

    # single-pass-equivalent sanity check: a plain 90th percentile over the
    # WHOLE chain should have caught only the strong zone, confirming this
    # synthetic case actually exercises the failure mode multi-pass fixes.
    single_pass_threshold = float(np.percentile(
        np.abs(np.diff(field, prepend=field[0])), 90.0
    ))
    assert single_pass_threshold > 4.0, (
        "synthetic case doesn't actually separate strong/weak under one global "
        f"percentile (threshold {single_pass_threshold}) -- test isn't exercising the "
        "intended failure mode"
    )

    print("self-test OK: multi-pass detection found both the strong feature "
          "(single global percentile would have caught this alone) and the "
          "weak one (single global percentile would have missed this entirely)")


if __name__ == "__main__":
    import sys

    _self_test_multiscale_detection()

    if len(sys.argv) >= 3:
        from vtu_io import read_vtu
        mesh = read_vtu(sys.argv[1])
        fields = sys.argv[2:]
        summary = compute_feature_summary(mesh, fields)
        print(json.dumps(summary, indent=2))
    elif len(sys.argv) == 2:
        print("usage: python feature_extraction.py <mesh.vtu> <field1> [field2 ...]")
        sys.exit(1)
