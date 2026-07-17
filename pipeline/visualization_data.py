"""
visualization_data.py — build the payload behind the "whole geometry" size
field view: a continuous shaded surface colored by size field magnitude
(geometric mean of the three requested axis sizes at each vertex), plus a
subsampled set of anisotropy glyphs.

Unlike the point-cloud/glyph-only view, this needs an actual surface
triangulation to shade continuously. Two cases are handled:

  - mesh cells are already "triangle" (2D case, or a surface mesh) -> used
    directly.
  - mesh cells are "tetra" (3D volume mesh) -> the boundary surface is
    extracted: a face (3 of a tet's 4 vertices) that appears in exactly one
    tet is a boundary face; faces shared by two tets are interior and
    dropped. Standard boundary-extraction algorithm, no external mesh
    library needed.

If neither cell type is present (e.g. a bare point cloud with no
connectivity), triangles come back empty and the caller should fall back to
point-cloud-only rendering.
"""

from __future__ import annotations

import numpy as np

from vtu_io import vertex_adjacency


def _extract_tetra_boundary_faces(tetra_conn: np.ndarray) -> np.ndarray:
    """Faces that belong to exactly one tet are boundary faces."""
    face_count = {}
    face_orientation = {}
    local_faces = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]

    for tet in tetra_conn:
        for a, b, c in local_faces:
            face = (int(tet[a]), int(tet[b]), int(tet[c]))
            key = tuple(sorted(face))
            face_count[key] = face_count.get(key, 0) + 1
            if key not in face_orientation:
                face_orientation[key] = face

    boundary = [face_orientation[k] for k, cnt in face_count.items() if cnt == 1]
    return np.array(boundary, dtype=int) if boundary else np.zeros((0, 3), dtype=int)


def extract_surface_triangles(mesh: dict) -> np.ndarray:
    """Return an (M,3) int array of triangle vertex indices representing the
    renderable surface of the mesh, or an empty array if none can be
    determined from the available cell types."""
    for ctype, conn in mesh["cells"]:
        if ctype == "triangle":
            return np.asarray(conn, dtype=int)

    for ctype, conn in mesh["cells"]:
        if ctype == "tetra":
            return _extract_tetra_boundary_faces(np.asarray(conn, dtype=int))

    return np.zeros((0, 3), dtype=int)


def build_whole_geometry_payload(mesh: dict, size_field_result: dict, glyph_target_count: int = 70) -> dict:
    """Assemble the JSON-serializable payload for the whole-geometry widget:
    points, full anisoSize matrices (for glyphs), a scalar "magnitude" field
    (geometric mean of the three axis sizes — the standard single-number
    stand-in for an anisotropic size at a point), the surface triangulation,
    and a subsample stride for glyph placement so dense meshes don't render
    thousands of overlapping ellipsoids.
    """
    points = mesh["points"]
    matrices = size_field_result["matrices"]
    n = len(points)

    row_norms = np.linalg.norm(matrices, axis=2)  # (n,3)
    magnitude = np.cbrt(np.clip(row_norms[:, 0] * row_norms[:, 1] * row_norms[:, 2], 1e-30, None))

    triangles = extract_surface_triangles(mesh)
    glyph_stride = max(1, round(n / max(glyph_target_count, 1)))

    return {
        "points": points,
        "matrices": matrices,
        "magnitude": magnitude,
        "triangles": triangles,
        "glyph_stride": glyph_stride,
    }


def local_edge_length_scalar(mesh: dict) -> np.ndarray:
    """Per-vertex average incident edge length -- a simple proxy for "local
    mesh size" that doesn't require a target size_field_result. Used for
    reviewing a mesh you only have the *actual* geometry for (e.g. the
    adapted mesh handed back from SCOREC), as opposed to visualizing a
    *requested* size field you built yourself."""
    points = mesh["points"]
    adjacency = vertex_adjacency(mesh)
    n = len(points)
    out = np.zeros(n)
    for i in range(n):
        nbrs = adjacency[i]
        if not nbrs:
            continue
        d = np.linalg.norm(points[list(nbrs)] - points[i], axis=1)
        out[i] = d.mean()
    # fall back to the global mean for any isolated vertex (avoids a 0 that
    # would otherwise wreck the color scale)
    nonzero = out[out > 0]
    fallback = nonzero.mean() if len(nonzero) else 1.0
    out[out == 0] = fallback
    return out


def build_adapted_mesh_payload(mesh: dict, glyph_target_count: int = 70) -> dict:
    """Like build_whole_geometry_payload, but for a mesh you don't have a
    target size_field_result for (e.g. the adapted mesh handed back from
    SCOREC for a human-review pass) -- colors/glyphs by *actual* local edge
    length instead of a *requested* size, by wrapping it in the same [3,3]
    isotropic-matrix shape the rest of the visualization pipeline expects."""
    edge_len = local_edge_length_scalar(mesh)
    matrices = np.stack([np.eye(3) * max(float(h), 1e-9) for h in edge_len])
    fake_result = {"matrices": matrices}
    return build_whole_geometry_payload(mesh, fake_result, glyph_target_count=glyph_target_count)


def payload_to_json_dict(payload: dict, round_ndigits: int = 4) -> dict:
    """Round + convert numpy arrays to plain lists for embedding in an HTML/JS
    widget (keeps the payload compact)."""
    def r(x):
        arr = np.asarray(x)
        return np.round(arr, round_ndigits).tolist()

    return {
        "points": r(payload["points"]),
        "matrices": r(payload["matrices"]),
        "magnitude": r(payload["magnitude"]),
        "triangles": np.asarray(payload["triangles"]).tolist(),
        "glyph_stride": int(payload["glyph_stride"]),
    }


if __name__ == "__main__":
    # self-test: a tiny synthetic tet mesh (a single tetrahedron) should
    # produce exactly its 4 faces as the boundary surface.
    pts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    tetra_conn = np.array([[0, 1, 2, 3]])
    mesh = {"points": pts, "cells": [("tetra", tetra_conn)], "point_data": {}, "cell_data": {}}
    faces = extract_surface_triangles(mesh)
    assert faces.shape == (4, 3), f"expected 4 boundary faces for a single tet, got {faces.shape}"
    print("single-tet boundary extraction OK:", faces.tolist())

    # two tets sharing one interior face -> 6 boundary faces, not 8
    pts2 = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]], dtype=float)
    tetra_conn2 = np.array([[0, 1, 2, 3], [1, 2, 3, 4]])
    mesh2 = {"points": pts2, "cells": [("tetra", tetra_conn2)], "point_data": {}, "cell_data": {}}
    faces2 = extract_surface_triangles(mesh2)
    assert faces2.shape == (6, 3), f"expected 6 boundary faces for two tets sharing a face, got {faces2.shape}"
    print("two-tet shared-face boundary extraction OK:", faces2.shape)

    # magnitude field sanity check on a trivial isotropic size field
    n = 5
    matrices = np.stack([np.eye(3) * h for h in [0.01, 0.02, 0.03, 0.04, 0.05]])
    mesh3 = {"points": np.zeros((n, 3)), "cells": [], "point_data": {}, "cell_data": {}}
    payload = build_whole_geometry_payload(mesh3, {"matrices": matrices})
    expected = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
    assert np.allclose(payload["magnitude"], expected), payload["magnitude"]
    print("magnitude field OK (isotropic case reduces to the diagonal size):", payload["magnitude"].tolist())

    # adapted-mesh payload self-test: a small triangle strip, edge length ~ known
    pts4 = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=float)
    tris4 = np.array([[0, 1, 2], [1, 3, 2]])
    mesh4 = {"points": pts4, "cells": [("triangle", tris4)], "point_data": {}, "cell_data": {}}
    adapted_payload = build_adapted_mesh_payload(mesh4)
    assert adapted_payload["matrices"].shape == (4, 3, 3)
    assert np.all(adapted_payload["magnitude"] > 0)
    print("build_adapted_mesh_payload OK, magnitudes:", adapted_payload["magnitude"].tolist())
