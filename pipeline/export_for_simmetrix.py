"""
export_for_simmetrix.py — serialize the final per-vertex anisoSize matrices
into a small, dependency-free transfer format for the SCOREC remote machine
(where the Simmetrix library actually lives — nothing in this repo links
against Simmetrix directly).

Two things are written, both keyed by the *same* 0-based vertex index used
in vtu_io's mesh dict (i.e. row order in mesh["points"]):

  1. <name>.csv   — human-readable: vertex_id,x,y,z,m00,m01,m02,m10,m11,m12,m20,m21,m22
  2. <name>.npz   — same data, compact, for round-tripping back into this
                    pipeline (e.g. to re-visualize exactly what was sent)

OPEN QUESTION for the remote side (flagged, not solved here): Simmetrix
mesh vertices are identified by its own internal handles/IDs, not by our
"row index into points". Whatever we send needs a correspondence rule back
to real MSA vertex objects. The two usual options:
  (a) index correspondence — valid only if the VTU we read is literally the
      same mesh (same vertex ordering) that Simmetrix will adapt, i.e. this
      is the *first* adaptation pass on the solver's native mesh.
  (b) coordinate matching — look up each Simmetrix vertex by nearest
      coordinate (needed once the mesh has already been adapted at least
      once, since new/adapted meshes renumber vertices).
This file always ships (x,y,z) alongside the index specifically so either
strategy works on the remote side without re-exporting.

remote_apply_size_field_stub.c below is a placeholder for the actual
MSA_setAnisoVertexSize() loop — intentionally left unfinished pending your
directions on SCOREC connectivity/build setup.
"""

from __future__ import annotations

import numpy as np


def write_size_field(path_prefix: str, mesh: dict, size_field_result: dict) -> None:
    points = mesh["points"]
    matrices = size_field_result["matrices"]
    n = len(points)
    assert matrices.shape == (n, 3, 3)

    # .npz — compact round-trip format
    np.savez(f"{path_prefix}.npz", points=points, matrices=matrices,
             iso_equivalent_size=size_field_result["iso_equivalent_size"])

    # .csv — human-readable / easy for a remote script in any language to parse
    with open(f"{path_prefix}.csv", "w") as f:
        f.write("vertex_id,x,y,z,m00,m01,m02,m10,m11,m12,m20,m21,m22\n")
        for i in range(n):
            x, y, z = points[i]
            m = matrices[i].flatten()
            f.write(f"{i},{x:.9g},{y:.9g},{z:.9g}," + ",".join(f"{v:.9g}" for v in m) + "\n")

    print(f"wrote {path_prefix}.npz and {path_prefix}.csv ({n} vertices)")


def read_size_field_csv(path: str):
    """Round-trip reader, e.g. for re-visualizing exactly what was exported."""
    ids, pts, mats = [], [], []
    with open(path) as f:
        next(f)  # header
        for line in f:
            vals = line.strip().split(",")
            ids.append(int(vals[0]))
            pts.append([float(v) for v in vals[1:4]])
            mats.append(np.array([float(v) for v in vals[4:13]]).reshape(3, 3))
    return np.array(ids), np.array(pts), np.array(mats)


REMOTE_STUB = '''\
/* remote_apply_size_field_stub.c
 *
 * STUB — not runnable yet. Placeholder for the SCOREC-side program that
 * reads the .csv this pipeline exports and calls MSA_setAnisoVertexSize()
 * for every vertex, then drives Simmetrix's adaptation.
 *
 * TODO (pending your direction on SCOREC access / build setup):
 *   - confirm the exact Simmetrix headers/link libraries for this call
 *   - decide vertex correspondence strategy: row-index match (first pass
 *     on the solver's native mesh) vs. nearest-coordinate lookup (after
 *     at least one prior adaptation has renumbered vertices)
 *   - confirm how the adapted mesh gets written back out / SCP'd home
 *
 * Sketch of the loop once the above is settled:
 *
 *   FILE *f = fopen("size_field.csv", "r");
 *   char line[512];
 *   fgets(line, sizeof(line), f); // skip header
 *   while (fgets(line, sizeof(line), f)) {
 *       int vid; double x,y,z, m[9];
 *       sscanf(line, "%d,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf",
 *              &vid, &x,&y,&z, &m[0],&m[1],&m[2],&m[3],&m[4],&m[5],&m[6],&m[7],&m[8]);
 *
 *       pVertex v = lookup_vertex(vid, x, y, z);   // <-- correspondence strategy goes here
 *       double anisoSize[3][3] = {
 *           {m[0], m[1], m[2]},
 *           {m[3], m[4], m[5]},
 *           {m[6], m[7], m[8]},
 *       };
 *       MSA_setAnisoVertexSize(v, anisoSize);      // real Simmetrix call
 *   }
 *
 *   // then: run the adapter, write out the adapted mesh, done on SCOREC side
 */
'''

if __name__ == "__main__":
    with open("remote_apply_size_field_stub.c", "w") as f:
        f.write(REMOTE_STUB)
    print("wrote remote_apply_size_field_stub.c (placeholder, not runnable)")
