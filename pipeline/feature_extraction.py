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
    """
    n = len(points)
    grad = np.zeros((n, 3))
    for i in range(n):
        neighbors = adjacency[i]
        if not neighbors:
            continue
        dx = points[list(neighbors)] - points[i]
        df = field[list(neighbors)] - field[i]
        dist = np.linalg.norm(dx, axis=1)
        dist[dist < 1e-12] = 1e-12
        w = 1.0 / dist
        A = dx * w[:, None]
        b = df * w
        # solve least squares A g = b  (3 unknowns)
        g, *_ = np.linalg.lstsq(A, b, rcond=None)
        grad[i] = g
    return grad


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
            regions.append({
                "n_vertices": int(region_mask.sum()),
                "bbox_min": pts.min(axis=0).tolist(),
                "bbox_max": pts.max(axis=0).tolist(),
                "centroid": pts.mean(axis=0).tolist(),
                "mean_gradient_magnitude": float(grad_mag[region_mask].mean()),
                "max_gradient_magnitude": float(grad_mag[region_mask].max()),
                "dominant_gradient_direction": g_dir,  # normal-ish direction across the feature
                "field_value_range": [float(field[region_mask].min()), float(field[region_mask].max())],
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
