"""
region_spec.py — schema + validator for the structured refinement-intent file
the LLM writes after reading the feature summary (and the human's comments).

This is the *only* artifact the LLM produces about mesh sizing. It never
emits per-vertex numbers — it emits a short list of named regions with a
shape, an influence falloff, and a target anisotropic size. size_field_builder.py
is the deterministic code that turns this into actual per-vertex [3][3]
matrices, so nothing downstream depends on the LLM getting arithmetic right.

Shape types (v1 — enough for ramp/wedge/shock-tube style cases; extend as
needed for more complex geometry):

  "sphere":  {"center": [x,y,z], "radius": r}
  "box":     {"min": [x,y,z], "max": [x,y,z]}
  "plane":   {"point": [x,y,z], "normal": [nx,ny,nz]}   -- infinite plane, distance = |signed dist|
  "cylinder":{"point": [x,y,z], "axis": [ax,ay,az], "radius": r}  -- infinite along axis
  "points":  {"coordinates": [[x,y,z], ...]}  -- distance = distance to nearest listed point
             (typically populated directly from a feature_extraction.py region's
             bounding box corners / centroid, so the LLM can just say "use
             candidate_regions[0]" conceptually, but the concrete numbers still
             get written into the spec so it stays a self-contained, auditable file)

Full spec shape:

{
  "defaults": {
    "hmin": <float>, "hmax": <float>,
    "background_size": <float>,       # isotropic size far from every region
    "growth_rate": <float>            # max allowed size ratio between adjacent vertices (>1.0)
  },
  "regions": [
    {
      "id": "shock_1",
      "description": "<free text, for humans>",
      "shape": {"type": "plane", ...params...},
      "influence": {"core_distance": <float>, "falloff_distance": <float>},
      "target_size": {
        "normal": <float>,             # size across the feature (small = tight anisotropy)
        "tangential_1": <float>,       # size along the feature, direction 1
        "tangential_2": <float>,       # size along the feature, direction 2
        "direction_normal": [nx,ny,nz] or null   # null => inferred from shape (e.g. plane normal)
      }
    },
    ...
  ],
  "human_feedback_log": [
    {"turn": <int>, "instruction": "<what the human said>", "applied_as": "<what changed>"}
  ]
}
"""

from __future__ import annotations

_SHAPE_REQUIRED_KEYS = {
    "sphere": {"center", "radius"},
    "box": {"min", "max"},
    "plane": {"point", "normal"},
    "cylinder": {"point", "axis", "radius"},
    "points": {"coordinates"},
}


class RegionSpecError(ValueError):
    pass


def validate_region_spec(spec: dict) -> None:
    """Raise RegionSpecError with a specific message on any structural problem.
    Deliberately strict — this file is the contract between the LLM's
    reasoning and the deterministic size-field math, so silent tolerance of
    malformed specs is the wrong default."""

    if "defaults" not in spec:
        raise RegionSpecError("missing top-level 'defaults'")
    d = spec["defaults"]
    for key in ("hmin", "hmax", "background_size", "growth_rate"):
        if key not in d:
            raise RegionSpecError(f"defaults missing '{key}'")
    if not (0 < d["hmin"] <= d["background_size"] <= d["hmax"]):
        raise RegionSpecError("expected 0 < hmin <= background_size <= hmax")
    if d["growth_rate"] <= 1.0:
        raise RegionSpecError("growth_rate must be > 1.0 (it's a max size ratio between neighbors)")

    if "regions" not in spec:
        raise RegionSpecError("missing top-level 'regions'")

    seen_ids = set()
    for i, region in enumerate(spec["regions"]):
        rid = region.get("id", f"<region {i}>")
        if rid in seen_ids:
            raise RegionSpecError(f"duplicate region id '{rid}'")
        seen_ids.add(rid)

        if "shape" not in region:
            raise RegionSpecError(f"region '{rid}' missing 'shape'")
        shape = region["shape"]
        stype = shape.get("type")
        if stype not in _SHAPE_REQUIRED_KEYS:
            raise RegionSpecError(f"region '{rid}' has unknown shape type '{stype}'")
        missing = _SHAPE_REQUIRED_KEYS[stype] - set(shape.keys())
        if missing:
            raise RegionSpecError(f"region '{rid}' shape '{stype}' missing keys {missing}")

        if "influence" not in region:
            raise RegionSpecError(f"region '{rid}' missing 'influence'")
        inf = region["influence"]
        for key in ("core_distance", "falloff_distance"):
            if key not in inf:
                raise RegionSpecError(f"region '{rid}' influence missing '{key}'")
        if inf["falloff_distance"] < 0 or inf["core_distance"] < 0:
            raise RegionSpecError(f"region '{rid}' influence distances must be >= 0")

        if "target_size" not in region:
            raise RegionSpecError(f"region '{rid}' missing 'target_size'")
        ts = region["target_size"]
        for key in ("normal", "tangential_1", "tangential_2"):
            if key not in ts:
                raise RegionSpecError(f"region '{rid}' target_size missing '{key}'")
            if ts[key] <= 0:
                raise RegionSpecError(f"region '{rid}' target_size.{key} must be > 0")
        if "direction_normal" not in ts:
            ts["direction_normal"] = None  # default: infer from shape at build time


def defaults_from_case_config(case_config: dict, growth_rate: float = 2.5, background_size: float | None = None) -> dict:
    """Build the region_spec `defaults` block from a case_config's hmin/hmax
    (see case_config.py) — this is the hmin/hmax -> region_spec wiring: the
    human supplies hmin/hmax once per case, and every region_spec authored
    for that case starts from these same bounds.

    background_size defaults to hmax (i.e. "coarse everywhere except the
    regions the LLM/human explicitly call out as needing refinement") unless
    overridden — ask the human if a different background makes more sense
    for a given case (e.g. a uniformly finer far-field for an unsteady case).

    growth_rate defaults to 2.5, not the more textbook-conservative ~1.2 this
    used to default to. Measured directly on this pipeline's own ramp2 case
    against a real reference adapted mesh: at growth_rate=1.2-1.5 the
    background never gets a real chance to relax back out to hmax across
    most of the domain (gradation limiting's per-edge cap dominates over
    the raw coarsening target almost everywhere except right at hmax's own
    edge), which is exactly the "way too many small elements in places we
    don't need them" failure mode. Pushing growth_rate up to ~2.5-3.0 nearly
    doubled the fraction of the domain actually reaching hmax with no
    measurable change to anisotropy/aspect ratio at the real features
    (gradation limiting rescales a vertex's full matrix by one scalar, so it
    cannot change aspect ratio -- only how far the background gets to
    relax). Returns essentially plateaued past ~2.5-3.0, so this is picked as
    a reasonably conservative point on that curve rather than the extreme
    end -- still worth confirming/adjusting per case rather than assuming,
    especially once real adapted meshes exist to check element quality
    against.
    """
    hmin, hmax = case_config["hmin"], case_config["hmax"]
    return {
        "hmin": hmin,
        "hmax": hmax,
        "background_size": background_size if background_size is not None else hmax,
        "growth_rate": growth_rate,
    }


EXAMPLE_REGION_SPEC = {
    "defaults": {
        "hmin": 0.0005,
        "hmax": 0.05,
        "background_size": 0.02,
        "growth_rate": 2.5,
    },
    "regions": [
        {
            "id": "shock_1",
            "description": "Oblique shock off the ramp leading edge (from temperature/pressure gradient region 0)",
            "shape": {"type": "plane", "point": [0.15, 0.0, 0.0], "normal": [0.8, 0.6, 0.0]},
            "influence": {"core_distance": 0.003, "falloff_distance": 0.02},
            "target_size": {
                "normal": 0.0008,
                "tangential_1": 0.02,
                "tangential_2": 0.02,
                "direction_normal": None,
            },
        },
        {
            "id": "ramp_boundary_layer",
            "description": "Thin high-gradient layer along the ramp surface",
            "shape": {"type": "box", "min": [0.0, 0.0, -1.0], "max": [0.3, 0.01, 1.0]},
            "influence": {"core_distance": 0.001, "falloff_distance": 0.005},
            "target_size": {
                "normal": 0.0003,
                "tangential_1": 0.01,
                "tangential_2": 0.01,
                "direction_normal": [0.0, 1.0, 0.0],
            },
        },
    ],
    "human_feedback_log": [],
}

if __name__ == "__main__":
    validate_region_spec(EXAMPLE_REGION_SPEC)
    print("EXAMPLE_REGION_SPEC is valid.")
