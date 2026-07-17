"""
vtu_io.py — read/write VTU files for the LLM-driven anisotropic adaptation pipeline.

Wraps meshio so the rest of the pipeline works with plain numpy arrays instead
of VTK objects. Deliberately minimal: this is the boundary layer between the
real CFD output (VTU) and everything downstream (feature extraction, size
field construction, visualization export).

Mesh representation used throughout the pipeline (a plain dict, not a class,
so it's trivially JSON-summarizable / picklable):

    mesh = {
        "points": (N, 3) float64 array of vertex coordinates,
        "cells": list of (cell_type: str, connectivity: (M, k) int array),
        "point_data": {field_name: (N,) or (N, k) float64 array, ...},
        "cell_data": {field_name: [(M,) array per cell block], ...},
    }

Vertex indices are 0-based and correspond to row indices in "points" and to
positions in every point_data array — this index is also what you use as the
"vertex" argument shape for MSA_setAnisoVertexSize once the real Simmetrix
vertex handles are mapped in on the remote side (see export_for_simmetrix.py).
"""

from __future__ import annotations

import numpy as np

try:
    import meshio
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "meshio is required. Run `bash pipeline/setup_env.sh` once, then "
        "`conda activate llmesh` before running anything in this directory "
        "(the system Python here has no working pip and lacks meshio/scipy/numpy). "
        "meshio itself is a thin, dependency-light reader/writer for VTU/VTP/etc "
        "— no full VTK build needed."
    ) from e


def read_vtu(path: str) -> dict:
    """Read a .vtu file into the pipeline's plain-dict mesh representation."""
    m = meshio.read(path)

    points = np.asarray(m.points, dtype=np.float64)
    if points.shape[1] == 2:
        # tolerate 2D-only VTUs (e.g. quick ramp/wedge test cases) by padding z=0
        points = np.column_stack([points, np.zeros(len(points))])

    cells = [(block.type, np.asarray(block.data, dtype=np.int64)) for block in m.cells]

    point_data = {k: np.asarray(v, dtype=np.float64) for k, v in m.point_data.items()}

    cell_data = {}
    for k, v in m.cell_data.items():
        cell_data[k] = [np.asarray(block, dtype=np.float64) for block in v]

    return {
        "points": points,
        "cells": cells,
        "point_data": point_data,
        "cell_data": cell_data,
    }


def write_vtu(path: str, mesh: dict, extra_point_data: dict | None = None) -> None:
    """Write a mesh dict back out to .vtu, optionally merging in extra per-vertex
    fields (e.g. feature scalars, or the 9-component aniso_size field) so the
    result can be opened directly in ParaView for a sanity check."""
    point_data = dict(mesh.get("point_data", {}))
    if extra_point_data:
        point_data.update(extra_point_data)

    cells = [meshio.CellBlock(ctype, conn) for ctype, conn in mesh["cells"]]

    out = meshio.Mesh(
        points=mesh["points"],
        cells=cells,
        point_data=point_data,
        cell_data=mesh.get("cell_data") or None,
    )
    out.write(path)


def vertex_adjacency(mesh: dict):
    """Build a symmetric vertex-vertex adjacency list from cell connectivity.

    Returned as a list-of-sets (length N, one set of neighbor vertex ids per
    vertex). This is the only "mesh topology" primitive the rest of the
    pipeline needs — used for cheap least-squares gradients in
    feature_extraction.py and for gradation limiting in size_field_builder.py.
    Deliberately not a scipy sparse matrix so it stays easy to reason about
    and cheap to build for the ~1e3-1e5 vertex scale this pipeline targets.
    """
    n = len(mesh["points"])
    adj = [set() for _ in range(n)]

    # supported cell types and the edges implied by their connectivity
    edge_table = {
        "tetra": [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)],
        "hexahedron": [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
                       (0, 4), (1, 5), (2, 6), (3, 7)],
        "wedge": [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3),
                  (0, 3), (1, 4), (2, 5)],
        "pyramid": [(0, 1), (1, 2), (2, 3), (3, 0), (0, 4), (1, 4), (2, 4), (3, 4)],
        "triangle": [(0, 1), (1, 2), (2, 0)],
        "quad": [(0, 1), (1, 2), (2, 3), (3, 0)],
        "line": [(0, 1)],
    }

    for ctype, conn in mesh["cells"]:
        edges = edge_table.get(ctype)
        if edges is None:
            continue
        for a, b in edges:
            for row in conn:
                va, vb = int(row[a]), int(row[b])
                adj[va].add(vb)
                adj[vb].add(va)

    return adj


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("usage: python vtu_io.py <mesh.vtu>  (quick sanity check / summary)")
        sys.exit(1)
    mesh = read_vtu(sys.argv[1])
    print(f"points: {mesh['points'].shape}")
    print(f"cells: {[(t, c.shape) for t, c in mesh['cells']]}")
    print(f"point_data fields: {list(mesh['point_data'].keys())}")
