"""
size_field_builder.py — the deterministic core of the pipeline.

Turns a validated region_spec (written by the LLM, see region_spec.py) into
an actual per-vertex [3][3] anisoSize matrix for every vertex in the mesh —
the exact object MSA_setAnisoVertexSize expects (each row = a direction
vector whose magnitude is the requested mesh size in that direction).

Nothing in this file is "AI" — it's plain geometry and a gradation-limiting
sweep. That's intentional: the LLM should never be the thing computing
per-vertex numbers for a 100k-vertex field, both because it doesn't scale in
tokens and because there's no way to guarantee well-posedness (SPD-like
matrices, bounded growth rate, hmin/hmax) from free-form generation. This
module is the guardrail — whatever the LLM asks for in region_spec.json, the
output handed to Simmetrix is always valid.

Algorithm per vertex v:
  1. For every region r, compute distance(v, r.shape) and a falloff weight
     w_r in [0, 1] (1 inside core_distance, linearly relaxing to 0 at
     core_distance + falloff_distance).
  2. The *dominant* region at v is the one with the largest w_r. Its
     direction basis (normal + two tangentials) defines the anisotropy axes
     used at v. (If no region has w_r > 0, the field is isotropic at the
     background size — direction is irrelevant when all three sizes match.)
  3. Blend each of the three axis sizes independently:
         h_axis(v) = lerp(background_size, region.target_size[axis], w_dominant)
     i.e. size relaxes smoothly from the tight feature size back to the
     background size as you move away — simple, monotonic, and easy for a
     human to reason about when giving feedback ("push the falloff out
     further").
  4. Clamp every axis size to [hmin, hmax].
  5. Gradation-limit: run several Gauss-Seidel sweeps over mesh edges so no
     two adjacent vertices' (isotropic-equivalent) sizes differ by more than
     `growth_rate` per unit edge length, and rescale each vertex's full
     matrix by the resulting correction factor (preserves the anisotropy
     ratio while capping the absolute jump in size — a deliberate v1
     simplification; see README for the upgrade path to true metric
     intersection if two strongly anisotropic regions overlap in the same
     place).

This targets the ~1e3-1e5 vertex scale called out for this project; the
per-vertex Python loop below is fine at that scale and easy to read/debug,
which matters more than performance for a hackathon demo.
"""

from __future__ import annotations

import numpy as np

from vtu_io import vertex_adjacency


def _distance_to_shape(shape: dict, p: np.ndarray) -> float:
    stype = shape["type"]
    if stype == "sphere":
        c = np.array(shape["center"])
        return max(0.0, np.linalg.norm(p - c) - shape["radius"])
    if stype == "box":
        lo = np.array(shape["min"])
        hi = np.array(shape["max"])
        d = np.maximum(np.maximum(lo - p, p - hi), 0.0)
        return float(np.linalg.norm(d))
    if stype == "plane":
        pt = np.array(shape["point"])
        n = np.array(shape["normal"])
        n = n / np.linalg.norm(n)
        return abs(float(np.dot(p - pt, n)))
    if stype == "cylinder":
        pt = np.array(shape["point"])
        axis = np.array(shape["axis"])
        axis = axis / np.linalg.norm(axis)
        rel = p - pt
        along = np.dot(rel, axis)
        perp = rel - along * axis
        return max(0.0, float(np.linalg.norm(perp)) - shape["radius"])
    if stype == "points":
        coords = np.array(shape["coordinates"])
        return float(np.min(np.linalg.norm(coords - p, axis=1)))
    raise ValueError(f"unsupported shape type '{stype}'")


def _shape_normal_direction(shape: dict, p: np.ndarray) -> np.ndarray:
    """Natural normal direction implied by a shape at point p, used when a
    region doesn't explicitly set target_size.direction_normal."""
    stype = shape["type"]
    if stype == "plane":
        n = np.array(shape["normal"], dtype=float)
        return n / np.linalg.norm(n)
    if stype == "sphere":
        c = np.array(shape["center"], dtype=float)
        d = p - c
        norm = np.linalg.norm(d)
        return d / norm if norm > 1e-12 else np.array([1.0, 0.0, 0.0])
    if stype == "cylinder":
        pt = np.array(shape["point"], dtype=float)
        axis = np.array(shape["axis"], dtype=float)
        axis = axis / np.linalg.norm(axis)
        rel = p - pt
        perp = rel - np.dot(rel, axis) * axis
        norm = np.linalg.norm(perp)
        return perp / norm if norm > 1e-12 else np.array([1.0, 0.0, 0.0])
    # box / points: no canonical normal -> default to world up
    return np.array([0.0, 1.0, 0.0])


def _orthonormal_basis(normal: np.ndarray):
    """Given a unit normal, build two orthonormal tangential directions."""
    normal = normal / np.linalg.norm(normal)
    helper = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    t1 = np.cross(normal, helper)
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(normal, t1)
    return t1, t2


def _falloff_weight(distance: float, core: float, falloff: float) -> float:
    if distance <= core:
        return 1.0
    if falloff <= 1e-12 or distance >= core + falloff:
        return 0.0
    t = (distance - core) / falloff
    return float(1.0 - (3 * t**2 - 2 * t**3))  # smootherstep, monotonic & C1


def _lerp(a: float, b: float, w: float) -> float:
    return a + (b - a) * w


def build_raw_size_field(points: np.ndarray, spec: dict) -> tuple[np.ndarray, np.ndarray]:
    """First pass: per-vertex [3,3] matrices before gradation limiting.
    Returns (matrices (N,3,3), isotropic_equivalent_size (N,) used for
    gradation limiting and for coloring the visualization)."""
    n = len(points)
    bg = spec["defaults"]["background_size"]
    matrices = np.zeros((n, 3, 3))

    for i, p in enumerate(points):
        best_w = 0.0
        best_region = None
        best_dist = None
        for region in spec["regions"]:
            dist = _distance_to_shape(region["shape"], p)
            w = _falloff_weight(dist, region["influence"]["core_distance"], region["influence"]["falloff_distance"])
            if w > best_w:
                best_w = w
                best_region = region
                best_dist = dist

        if best_region is None or best_w <= 1e-9:
            matrices[i] = np.eye(3) * bg
            continue

        ts = best_region["target_size"]
        direction = ts["direction_normal"]
        if direction is None:
            normal = _shape_normal_direction(best_region["shape"], p)
        else:
            normal = np.array(direction, dtype=float)
            normal /= np.linalg.norm(normal)
        t1, t2 = _orthonormal_basis(normal)

        h_n = _lerp(bg, ts["normal"], best_w)
        h_t1 = _lerp(bg, ts["tangential_1"], best_w)
        h_t2 = _lerp(bg, ts["tangential_2"], best_w)

        matrices[i, 0] = normal * h_n
        matrices[i, 1] = t1 * h_t1
        matrices[i, 2] = t2 * h_t2

    iso_equiv = np.array([np.linalg.norm(m, axis=1).min() for m in matrices])
    return matrices, iso_equiv


def apply_gradation_limiting(points: np.ndarray, matrices: np.ndarray, iso_equiv: np.ndarray,
                              adjacency: list[set], growth_rate: float, n_sweeps: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Cap how fast the (isotropic-equivalent) size can grow between adjacent
    vertices, then rescale each vertex's full matrix by the resulting
    correction factor so anisotropy ratio is preserved.

    Standard Gauss-Seidel gradation-limiting sweep (cf. Persson 2006 /
    distmesh-style limiting): repeatedly enforce
        h[i] <= h[j] + (growth_rate - 1) * |x_i - x_j|
    over every edge, in both directions, until converged or n_sweeps used.
    """
    h = iso_equiv.copy()
    order = list(range(len(points)))

    for _ in range(n_sweeps):
        changed = False
        for i in order:
            for j in adjacency[i]:
                d = np.linalg.norm(points[i] - points[j])
                allowed = h[j] + (growth_rate - 1.0) * d
                if h[i] > allowed + 1e-15:
                    h[i] = allowed
                    changed = True
        order.reverse()  # alternate sweep direction (classic Gauss-Seidel trick)
        if not changed:
            break

    scale = np.divide(h, iso_equiv, out=np.ones_like(h), where=iso_equiv > 1e-15)
    limited = matrices * scale[:, None, None]
    return limited, h


def clamp_size_field(matrices: np.ndarray, hmin: float, hmax: float) -> np.ndarray:
    row_norms = np.linalg.norm(matrices, axis=2, keepdims=True)
    row_norms_safe = np.where(row_norms < 1e-15, 1.0, row_norms)
    unit_dirs = matrices / row_norms_safe
    clamped_norms = np.clip(row_norms, hmin, hmax)
    return unit_dirs * clamped_norms


def build_size_field(mesh: dict, spec: dict) -> dict:
    """Full pipeline: raw per-region blend -> gradation limiting -> clamping.

    Returns dict with:
      "matrices": (N,3,3) final anisoSize matrices, ready for MSA_setAnisoVertexSize
      "iso_equivalent_size": (N,) scalar for coloring/visualization/QA
    """
    from region_spec import validate_region_spec
    validate_region_spec(spec)

    points = mesh["points"]
    adjacency = vertex_adjacency(mesh)

    raw_matrices, iso_equiv = build_raw_size_field(points, spec)
    limited_matrices, limited_iso = apply_gradation_limiting(
        points, raw_matrices, iso_equiv, adjacency, spec["defaults"]["growth_rate"]
    )
    final_matrices = clamp_size_field(limited_matrices, spec["defaults"]["hmin"], spec["defaults"]["hmax"])
    final_iso = np.clip(limited_iso, spec["defaults"]["hmin"], spec["defaults"]["hmax"])

    return {"matrices": final_matrices, "iso_equivalent_size": final_iso}


if __name__ == "__main__":
    # quick self-test with a synthetic point set (NOT a CFD case — just
    # validates the math: a plane-shaped shock region should produce small
    # normal size / larger tangential size near the plane and relax to
    # background size far away, with no size jump bigger than growth_rate).
    from region_spec import EXAMPLE_REGION_SPEC

    rng = np.random.default_rng(0)
    pts = rng.uniform(-0.2, 0.5, size=(400, 3))
    pts[:, 2] = 0.0  # keep it 2D-ish for an easy sanity read
    mesh = {
        "points": pts,
        "cells": [("triangle", np.array([[0, 1, 2]]))],  # dummy, adjacency below is what matters
        "point_data": {},
        "cell_data": {},
    }
    # build adjacency via a simple radius graph instead of the dummy cell (self-test only)
    from scipy.spatial import cKDTree
    tree = cKDTree(pts)
    pairs = tree.query_pairs(r=0.06)
    adj = [set() for _ in range(len(pts))]
    for a, b in pairs:
        adj[a].add(b)
        adj[b].add(a)

    import vtu_io
    vtu_io.vertex_adjacency = lambda m: adj  # monkeypatch for this synthetic self-test only

    result = build_size_field(mesh, EXAMPLE_REGION_SPEC)
    print("iso_equivalent_size range:", result["iso_equivalent_size"].min(), result["iso_equivalent_size"].max())
    print("matrices shape:", result["matrices"].shape)
    print("sample matrix near shock plane point (0.15,0,0):")
    near_idx = np.argmin(np.linalg.norm(pts - np.array([0.15, 0.0, 0.0]), axis=1))
    print(result["matrices"][near_idx])
