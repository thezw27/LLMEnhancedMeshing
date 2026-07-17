"""
export_for_simmetrix.py — serialize the final per-vertex anisoSize matrices
into the exact transfer format apply_aniso_size_field.cpp expects.

That program exists at simmetrix/src/apply_aniso_size_field.cpp (usage:
apply_aniso_size_field <native_model> <model.smd> <mesh.sms>
<size_field.txt> <output_dir>), and run_adaptation.py runs it directly as
a subprocess (this pipeline and the Simmetrix side run on the same
machine — no scp/staging step). Vertex correspondence is coordinate-based,
always — it builds a KD-tree (nanoflann) over the existing Simmetrix
mesh's vertices and maps every line of size_field.txt to its
nearest-coordinate vertex, then calls MSA_setAnisoVertexSize. So our
export only needs (x, y, z, matrix); no vertex index/ID is read or needed
on the C++ side.

IMPORTANT FORMAT NOTE: the C++ reader (readSizeFieldFile) parses each line
as exactly 12 whitespace/comma-separated numbers — "X Y Z m00 m01 m02 m10
m11 m12 m20 m21 m22" — with no header and no leading id column. An earlier
version of this file wrote a 13-column CSV with a "vertex_id" first column;
that shifts every field over by one and silently corrupts the data (the
reader doesn't crash on a header line since it fails a numeric parse and
warns/skips it, but on the *data* rows the vertex_id parses as a valid
double and every subsequent field lands in the wrong slot). write_size_field
below now writes the correct plain-text format as the primary output.

Three things are written:

  1. <name>.txt   — the actual file apply_aniso_size_field reads: one line
                    per vertex, "x y z m00 m01 m02 m10 m11 m12 m20 m21 m22",
                    space-separated, no header.
  2. <name>.csv   — human-readable reference copy (with header + a row
                    index) for inspecting what was sent; NOT the file
                    apply_aniso_size_field reads.
  3. <name>.npz   — compact round-trip format for re-loading back into this
                    pipeline (e.g. to re-visualize exactly what was sent).
"""

from __future__ import annotations

import numpy as np


def write_size_field(path_prefix: str, mesh: dict, size_field_result: dict) -> None:
    points = mesh["points"]
    matrices = size_field_result["matrices"]
    n = len(points)
    assert matrices.shape == (n, 3, 3)

    # .txt — the file apply_aniso_size_field.cpp actually reads. Exactly 12
    # fields per line, no header.
    with open(f"{path_prefix}.txt", "w") as f:
        for i in range(n):
            x, y, z = points[i]
            m = matrices[i].flatten()
            f.write(f"{x:.9g} {y:.9g} {z:.9g} " + " ".join(f"{v:.9g}" for v in m) + "\n")

    # .csv — human-readable reference only (extra columns apply_aniso_size_field
    # does NOT expect — don't pass this file to it).
    with open(f"{path_prefix}.csv", "w") as f:
        f.write("vertex_id,x,y,z,m00,m01,m02,m10,m11,m12,m20,m21,m22\n")
        for i in range(n):
            x, y, z = points[i]
            m = matrices[i].flatten()
            f.write(f"{i},{x:.9g},{y:.9g},{z:.9g}," + ",".join(f"{v:.9g}" for v in m) + "\n")

    # .npz — compact round-trip format
    np.savez(f"{path_prefix}.npz", points=points, matrices=matrices,
             iso_equivalent_size=size_field_result["iso_equivalent_size"])

    print(f"wrote {path_prefix}.txt (fed to apply_aniso_size_field), {path_prefix}.csv (reference), "
          f"{path_prefix}.npz ({n} vertices)")


def read_size_field_txt(path: str):
    """Round-trip reader for the .txt format (mirrors what the C++ side parses)."""
    pts, mats = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            vals = [float(v) for v in line.replace(",", " ").split()]
            if len(vals) != 12:
                continue
            pts.append(vals[0:3])
            mats.append(np.array(vals[3:12]).reshape(3, 3))
    return np.array(pts), np.array(mats)


def read_size_field_csv(path: str):
    """Round-trip reader for the human-readable .csv reference copy."""
    ids, pts, mats = [], [], []
    with open(path) as f:
        next(f)  # header
        for line in f:
            vals = line.strip().split(",")
            ids.append(int(vals[0]))
            pts.append([float(v) for v in vals[1:4]])
            mats.append(np.array([float(v) for v in vals[4:13]]).reshape(3, 3))
    return np.array(ids), np.array(pts), np.array(mats)


if __name__ == "__main__":
    # quick self-test: round-trip the .txt format and confirm it matches
    # what write_size_field produced, with the exact column layout
    # apply_aniso_size_field.cpp's readSizeFieldFile expects.
    import numpy as _np

    rng = _np.random.default_rng(0)
    pts = rng.uniform(-1, 1, size=(20, 3))
    mats = _np.stack([_np.eye(3) * h for h in rng.uniform(0.001, 0.02, size=20)])
    mesh = {"points": pts}
    result = {"matrices": mats, "iso_equivalent_size": _np.linalg.norm(mats, axis=2).min(axis=1)}

    write_size_field("/tmp/_export_selftest", mesh, result)
    pts2, mats2 = read_size_field_txt("/tmp/_export_selftest.txt")
    assert _np.allclose(pts2, pts) and _np.allclose(mats2, mats), "round-trip mismatch"
    with open("/tmp/_export_selftest.txt") as f:
        first_line_fields = f.readline().split()
    assert len(first_line_fields) == 12, f"expected 12 fields per line, got {len(first_line_fields)}"
    print("self-test OK: .txt format is exactly 12 space-separated fields per line, round-trips cleanly")
