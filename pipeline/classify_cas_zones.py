"""
classify_cas_zones.py -- classify a Fluent ASCII .cas file's boundary face
zones by geometry, since the .cas this pipeline's SCOREC side produces
(simmetrix/translateToCas.py -> meshExporter("FLUENT")) is a bare mesh
export: every boundary zone comes out generically typed as "wall" (bc-type
3), no inlet/outlet/symmetry assignment, no Simmetrix model-face names
carried over (confirmed by inspecting testCases/ramp/rampInit.cas directly
-- `strings ramp.smd` has no inlet/outlet/wall labels, and every face zone
header in the .cas has bc-type=3).

This parses the case file's own node coordinate section (10) and boundary
face zone sections (13) directly (values are hex-encoded per Fluent's
ASCII case format; node coordinates themselves are plain decimal) and
reports each zone's bounding box, then classifies zones using simple
axis-aligned rules you provide (e.g. "the inlet is the planar face at
global-min X", "symmetry planes are the ones with a near-zero Y extent").

This does NOT talk to Fluent at all -- it's a standalone reader of the
plain-text .cas format, useful for figuring out which auto-generated zone
ID/name (e.g. "wall-3") corresponds to which physical boundary before
writing a BC-setup journal by hand.

Usage:
    python classify_cas_zones.py <path/to/file.cas> \
        [--inlet-axis x] [--outlet-axis x] [--symmetry-axis y] [--tol 1e-6]

Prints, per boundary zone: a bounding box and a best-guess classification
(inlet / outlet / symmetry / wall). Verify against the actual geometry
(e.g. in ParaView) before trusting it blindly -- this is a simple
axis-aligned-planar-face heuristic, not a general classifier.
"""

from __future__ import annotations

import argparse
import re


def parse_cas_zones(path: str):
    with open(path) as f:
        text = f.read()

    # Node coordinates: "(10 (<zone> 1 <last-hex> 1 <dim>)(\n x y z\n ...\n))"
    # -- the declaration zone (zone id 0) has no data; the real data zone
    # has the same header shape but a nonzero zone id and an opening "("
    # immediately after the header (same line), unlike face zones below.
    node_match = None
    for m in re.finditer(r'\(10 \(([0-9a-f]+) 1 ([0-9a-f]+) 1 (\d+)\)\(\n(.*?)\n\)\)', text, re.DOTALL):
        if m.group(1) != "0":
            node_match = m
            break
    if node_match is None:
        raise ValueError(f"could not find a node coordinate data section in {path}")

    last_idx = int(node_match.group(2), 16)
    dim = int(node_match.group(3))
    coord_lines = node_match.group(4).strip().split("\n")
    if len(coord_lines) != last_idx:
        raise ValueError(f"expected {last_idx} node coordinate lines, found {len(coord_lines)}")

    nodes = {}
    for i, line in enumerate(coord_lines, start=1):
        vals = [float(v) for v in line.split()]
        nodes[i] = tuple(vals[:3]) if dim == 3 else (vals[0], vals[1], 0.0)

    xs = [p[0] for p in nodes.values()]
    ys = [p[1] for p in nodes.values()]
    zs = [p[2] for p in nodes.values()]
    global_bbox = {"x": (min(xs), max(xs)), "y": (min(ys), max(ys)), "z": (min(zs), max(zs))}

    # Boundary face zones: "(13 (<zone-hex> <first-hex> <last-hex> <bctype-hex> <facetype-hex>)\n(\n ...face lines...\n))"
    # -- note the body's opening "(" is on its OWN line here, unlike the node section above.
    zone_headers = re.findall(
        r'\(13 \(([0-9a-f]+) ([0-9a-f]+) ([0-9a-f]+) ([0-9a-f]+) ([0-9a-f]+)\)\n\(\n(.*?)\n\)\)',
        text, re.DOTALL,
    )

    zones = {}
    for zid_hex, _first, _last, bctype_hex, facetype_hex, body in zone_headers:
        zid = int(zid_hex, 16)
        bctype = int(bctype_hex, 16)
        if bctype == 2:
            continue  # interior faces, not a boundary zone
        node_ids = set()
        for line in body.strip().split("\n"):
            toks = line.split()
            if facetype_hex == "0":
                n = int(toks[0], 16)
                face_nodes = [int(t, 16) for t in toks[1:1 + n]]
            else:
                n = int(facetype_hex, 16)
                face_nodes = [int(t, 16) for t in toks[0:n]]
            node_ids.update(face_nodes)
        pts = [nodes[nid] for nid in node_ids]
        bbox = {
            "x": (min(p[0] for p in pts), max(p[0] for p in pts)),
            "y": (min(p[1] for p in pts), max(p[1] for p in pts)),
            "z": (min(p[2] for p in pts), max(p[2] for p in pts)),
        }
        zones[zid] = {"bctype": bctype, "n_nodes": len(node_ids), "bbox": bbox}

    return global_bbox, zones


def classify(global_bbox: dict, zones: dict, inlet_axis: str, outlet_axis: str,
             symmetry_axis: str, tol: float) -> dict:
    labels = {}
    gmin = global_bbox[inlet_axis][0]
    gmax = global_bbox[outlet_axis][1]
    for zid, info in zones.items():
        bbox = info["bbox"]
        axis_range = {a: bbox[a][1] - bbox[a][0] for a in ("x", "y", "z")}
        if axis_range[inlet_axis] < tol and abs(bbox[inlet_axis][0] - gmin) < tol:
            labels[zid] = f"INLET (planar, {inlet_axis}={bbox[inlet_axis][0]:.6g})"
        elif axis_range[outlet_axis] < tol and abs(bbox[outlet_axis][1] - gmax) < tol:
            labels[zid] = f"OUTLET (planar, {outlet_axis}={bbox[outlet_axis][0]:.6g})"
        elif axis_range[symmetry_axis] < tol:
            labels[zid] = f"SYMMETRY (planar, {symmetry_axis}={bbox[symmetry_axis][0]:.6g})"
        else:
            labels[zid] = "WALL"
    return labels


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cas_path")
    ap.add_argument("--inlet-axis", default="x", choices=["x", "y", "z"])
    ap.add_argument("--outlet-axis", default="x", choices=["x", "y", "z"])
    ap.add_argument("--symmetry-axis", default="y", choices=["x", "y", "z"])
    ap.add_argument("--tol", type=float, default=1e-6)
    args = ap.parse_args()

    global_bbox, zones = parse_cas_zones(args.cas_path)
    print(f"Global bbox: X {global_bbox['x']}  Y {global_bbox['y']}  Z {global_bbox['z']}")
    print(f"{len(zones)} boundary zone(s) found (interior zones excluded)\n")

    labels = classify(global_bbox, zones, args.inlet_axis, args.outlet_axis, args.symmetry_axis, args.tol)
    for zid in sorted(zones):
        info = zones[zid]
        print(f"zone {zid}: bctype={info['bctype']} n_nodes={info['n_nodes']} "
              f"bbox={info['bbox']}")
        print(f"    -> {labels[zid]}")


if __name__ == "__main__":
    main()
