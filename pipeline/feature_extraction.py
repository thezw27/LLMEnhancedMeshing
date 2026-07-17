"""
feature_extraction.py — turn a raw CFD solution field into a small, LLM-readable
JSON summary instead of a firehose of per-vertex numbers.

This is deliberately "classical, not clever": a least-squares vertex gradient
(no full Hessian — that's the expensive step this pipeline is trying to avoid
paying for on every adaptation cycle) plus simple percentile thresholding and
connected-component grouping to find candidate feature regions (shocks,
strong thermal gradients, etc). The LLM's job starts *after* this — reading
the summary, describing it in plain English, asking the human for input, and
deciding refinement intent. This module never decides anything about mesh
sizing; it only observes and reports.

Output shape (see feature_summary_schema below) is capped in size regardless
of mesh resolution (a handful of regions with summary stats each) so it reads
fine as LLM context even for a 100k-vertex case.
"""

from __future__ import annotations

import json
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from vtu_io import vertex_adjacency


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


def compute_feature_summary(
    mesh: dict,
    driver_fields: list[str],
    gradient_percentile: float = 90.0,
    min_region_size: int = 5,
) -> dict:
    """Build the compact JSON feature summary for one or more driver fields.

    Parameters
    ----------
    mesh : dict from vtu_io.read_vtu
    driver_fields : e.g. ["temperature", "pressure"] — must exist in mesh point_data
    gradient_percentile : vertices above this percentile of |grad(field)| are
        treated as "candidate feature" vertices (shock fronts, sharp thermal
        gradients, etc). 90 is a reasonable default; tune per case.
    min_region_size : drop connected components smaller than this (noise).

    Returns
    -------
    A JSON-serializable dict: global stats per field, plus a list of candidate
    regions (bounding box, centroid, dominant gradient direction, size, and
    representative field values) — this is what gets handed to the LLM.
    """
    points = mesh["points"]
    adjacency = vertex_adjacency(mesh)

    summary = {"n_vertices": len(points), "fields": {}}

    for field_name in driver_fields:
        if field_name not in mesh["point_data"]:
            continue
        field = mesh["point_data"][field_name]
        if field.ndim > 1:
            field = np.linalg.norm(field, axis=1)  # vector field -> magnitude

        grad = compute_vertex_gradient(points, adjacency, field)
        grad_mag = np.linalg.norm(grad, axis=1)

        threshold = float(np.percentile(grad_mag, gradient_percentile))
        mask = grad_mag >= threshold
        labels = _connected_components_on_mask(adjacency, mask)

        regions = []
        for label in sorted(set(labels[labels >= 0])):
            region_mask = labels == label
            if region_mask.sum() < min_region_size:
                continue
            pts = points[region_mask]
            g = grad[region_mask]
            g_mean = g.mean(axis=0)
            g_dir = (g_mean / np.linalg.norm(g_mean)).tolist() if np.linalg.norm(g_mean) > 1e-12 else [0, 0, 0]

            # How much this region's own detected vertices actually scatter
            # cross-feature (perpendicular to its dominant/along-feature PCA
            # direction) -- this is real, measured data about THIS region on
            # THIS case, meant to inform core_distance/falloff_distance
            # instead of reusing a value tuned on a previous, possibly very
            # different case. A region whose real detected vertices scatter
            # by several mm needs a core+falloff wide enough to cover most
            # of that scatter, or most of its own real vertices will only
            # partially blend toward the requested tight size regardless of
            # how good detection/smoothing is -- that's a property of the
            # region's own data, not something core_distance tuning alone
            # can outrun.
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

            regions.append({
                "n_vertices": int(region_mask.sum()),
                "bbox_min": pts.min(axis=0).tolist(),
                "bbox_max": pts.max(axis=0).tolist(),
                "centroid": pts.mean(axis=0).tolist(),
                "mean_gradient_magnitude": float(grad_mag[region_mask].mean()),
                "max_gradient_magnitude": float(grad_mag[region_mask].max()),
                "dominant_gradient_direction": g_dir,  # normal-ish direction across the feature
                "field_value_range": [float(field[region_mask].min()), float(field[region_mask].max())],
                "cross_feature_scatter": scatter_stats,
            })

        # sort strongest features first so the LLM sees the important ones up front
        regions.sort(key=lambda r: r["max_gradient_magnitude"], reverse=True)

        summary["fields"][field_name] = {
            "value_range": [float(field.min()), float(field.max())],
            "mean": float(field.mean()),
            "std": float(field.std()),
            "gradient_magnitude_range": [float(grad_mag.min()), float(grad_mag.max())],
            "gradient_threshold_used": threshold,
            "candidate_regions": regions,
        }

    return summary


if __name__ == "__main__":
    import sys
    from vtu_io import read_vtu

    if len(sys.argv) < 3:
        print("usage: python feature_extraction.py <mesh.vtu> <field1> [field2 ...]")
        sys.exit(1)

    mesh = read_vtu(sys.argv[1])
    fields = sys.argv[2:]
    summary = compute_feature_summary(mesh, fields)
    print(json.dumps(summary, indent=2))
