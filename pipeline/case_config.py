"""
case_config.py — the per-case configuration a human supplies when starting
a new case with a VTU: where the matching Simmetrix model/mesh live, the
meshing bounds, and which fields drive feature detection.

This is deliberately separate from region_spec.json: region_spec is what
the LLM writes *after* reading the feature summary (refinement intent).
case_config is set once per case, by the human, before any of that — it's
the addressing information (VTU, model/mesh, run_on_scorec_script) plus
the meshing bounds (hmin/hmax) that flow into region_spec's defaults.

Everything in this pipeline (feature extraction through mesh adaptation)
now runs on one machine — there is no separate "local sandbox" vs. "remote
SCOREC machine" split, so every path below is a plain local absolute path.
run_adaptation.py invokes run_on_scorec_script directly as a subprocess;
no ssh/scp involved (see run_adaptation.py). The Fluent .cas export
(translateToCas.py via SimModelerScript) still runs as part of that same
script and still produces a real .cas file — it's just a different
toolchain under the hood (see run_on_scorec.sh), not a separate machine.

`native_model` is the CAD kernel's native geometry file (e.g. a Parasolid
.x_t) — needed only for the Fluent .cas export step (translateToCas.py),
not by apply_aniso_size_field itself, which loads model_smd directly.
`run_on_scorec_script` is simmetrix/run_on_scorec.sh: it runs
apply_aniso_size_field (writes adapted.vtu + adapted_mesh/adapted.sms),
then runs translateToCas.py via SimModelerScript (writes adapted.cas).

The adapted VTU and Fluent .cas are always named adapted.vtu/adapted.cas
(apply_aniso_size_field.cpp and run_on_scorec.sh both hardcode these
names) — ADAPTED_VTU_NAME/ADAPTED_CAS_NAME below exist so other modules
don't repeat the literal strings, not because they're configurable.

Optional per-case overrides (all validated here if present, but none are
required -- a documented default is used and its reasoning printed by
driver.py if omitted; see region_spec.DEFAULT_GROWTH_RATE_REASON /
feature_extraction.DEFAULT_GRADIENT_PERCENTILE_REASON /
DEFAULT_MIN_REGION_FRACTION_REASON / DEFAULT_MAX_DETECTION_PASSES_REASON
for exactly what each default is and why):
  "background_size"       -- isotropic size far from every region (default: hmax)
  "growth_rate"            -- max size ratio between adjacent vertices, >1.0 (default: 2.5)
  "gradient_percentile"    -- per-pass feature-detection cutoff, 0-100 (default: 90.0)
  "min_region_size"        -- absolute vertex-count noise floor for a detected region
  "min_region_fraction"    -- same, as a fraction of mesh vertex count instead of an
                              absolute count (ignored if min_region_size is also given)
  "max_detection_passes"   -- cap on peel-off detection iterations (default: 6)
  "min_relative_threshold" -- stop peel-off once a pass's threshold drops below this
                              fraction of the first (strongest) pass's, 0-1 (default: 0.01)
"""

from __future__ import annotations

import json


class CaseConfigError(ValueError):
    pass


_REQUIRED_TOP = {
    "case_name", "vtu_path", "driver_fields", "hmin", "hmax",
    "native_model", "model_smd", "mesh_sms", "run_on_scorec_script", "work_dir",
}
ADAPTED_VTU_NAME = "adapted.vtu"
ADAPTED_CAS_NAME = "adapted.cas"


def validate_case_config(cfg: dict) -> None:
    missing = _REQUIRED_TOP - set(cfg.keys())
    if missing:
        raise CaseConfigError(f"case_config missing top-level keys: {sorted(missing)}")

    if not isinstance(cfg["driver_fields"], list) or not cfg["driver_fields"]:
        raise CaseConfigError("driver_fields must be a non-empty list, e.g. ['temperature']")

    if not (0 < cfg["hmin"] <= cfg["hmax"]):
        raise CaseConfigError(f"expected 0 < hmin <= hmax, got hmin={cfg['hmin']} hmax={cfg['hmax']}")

    for key in ("native_model", "model_smd", "mesh_sms", "run_on_scorec_script", "work_dir"):
        if not cfg[key] or not cfg[key].startswith("/"):
            raise CaseConfigError(f"{key} should be an absolute path, got {cfg[key]!r}")

    # apply_aniso_exe is optional: run_on_scorec.sh defaults it to
    # <script_dir>/build/apply_aniso_size_field on its own. Only validated
    # if a case wants to override that (e.g. pointing at a different build).
    if "apply_aniso_exe" in cfg:
        if not cfg["apply_aniso_exe"] or not cfg["apply_aniso_exe"].startswith("/"):
            raise CaseConfigError(f"apply_aniso_exe should be an absolute path, got {cfg['apply_aniso_exe']!r}")

    # Optional detection/sizing overrides -- see module docstring. Checked
    # here so a typo/bad value fails fast with a case_config-specific
    # message instead of surfacing later as a confusing region_spec or
    # feature_extraction error.
    if "background_size" in cfg:
        bg = cfg["background_size"]
        if not (cfg["hmin"] <= bg <= cfg["hmax"]):
            raise CaseConfigError(f"background_size ({bg}) must satisfy hmin <= background_size <= hmax")
    if "growth_rate" in cfg and cfg["growth_rate"] <= 1.0:
        raise CaseConfigError(f"growth_rate must be > 1.0, got {cfg['growth_rate']}")
    if "gradient_percentile" in cfg and not (0.0 < cfg["gradient_percentile"] < 100.0):
        raise CaseConfigError(f"gradient_percentile must be in (0, 100), got {cfg['gradient_percentile']}")
    if "min_region_size" in cfg and cfg["min_region_size"] < 1:
        raise CaseConfigError(f"min_region_size must be >= 1, got {cfg['min_region_size']}")
    if "min_region_fraction" in cfg and not (0.0 < cfg["min_region_fraction"] < 1.0):
        raise CaseConfigError(f"min_region_fraction must be in (0, 1), got {cfg['min_region_fraction']}")
    if "max_detection_passes" in cfg and cfg["max_detection_passes"] < 1:
        raise CaseConfigError(f"max_detection_passes must be >= 1, got {cfg['max_detection_passes']}")
    if "min_relative_threshold" in cfg and not (0.0 < cfg["min_relative_threshold"] < 1.0):
        raise CaseConfigError(f"min_relative_threshold must be in (0, 1), got {cfg['min_relative_threshold']}")


def load_case_config(path: str) -> dict:
    with open(path) as f:
        cfg = json.load(f)
    validate_case_config(cfg)
    return cfg


EXAMPLE_CASE_CONFIG = {
    "case_name": "double_ramp_test1",
    "vtu_path": "case.vtu",
    "driver_fields": ["temperature"],
    "hmin": 0.0005,
    "hmax": 0.05,
    "native_model": "/users/gordoz2/lore.scorec.rpi.edu/adaptiveController/cases/double_ramp_test1/model_nat.x_t",
    "model_smd": "/users/gordoz2/lore.scorec.rpi.edu/adaptiveController/cases/double_ramp_test1/model.smd",
    "mesh_sms": "/users/gordoz2/lore.scorec.rpi.edu/adaptiveController/cases/double_ramp_test1/mesh.sms",
    "run_on_scorec_script": "/users/gordoz2/lore.scorec.rpi.edu/LLMEnhancedMeshing/simmetrix/run_on_scorec.sh",
    "work_dir": "/users/gordoz2/lore.scorec.rpi.edu/adaptiveController/runs/double_ramp_test1",
}

if __name__ == "__main__":
    validate_case_config(EXAMPLE_CASE_CONFIG)
    with_override = {**EXAMPLE_CASE_CONFIG, "apply_aniso_exe": "/some/other/build/apply_aniso_size_field"}
    validate_case_config(with_override)
    try:
        validate_case_config({**EXAMPLE_CASE_CONFIG, "apply_aniso_exe": "relative/path"})
        raise AssertionError("expected CaseConfigError for a non-absolute apply_aniso_exe override")
    except CaseConfigError:
        pass
    print("EXAMPLE_CASE_CONFIG is valid; apply_aniso_exe override validation verified.")

    # optional detection/sizing override validation
    good_overrides = {
        **EXAMPLE_CASE_CONFIG,
        "background_size": 0.01, "growth_rate": 1.8, "gradient_percentile": 85.0,
        "min_region_size": 4, "min_region_fraction": 0.001, "max_detection_passes": 4,
        "min_relative_threshold": 0.05,
    }
    validate_case_config(good_overrides)

    bad_cases = [
        {"background_size": 0.5},       # outside [hmin, hmax]
        {"growth_rate": 1.0},           # not > 1.0
        {"gradient_percentile": 100.0}, # not < 100
        {"min_region_size": 0},         # not >= 1
        {"min_region_fraction": 1.0},   # not < 1
        {"max_detection_passes": 0},    # not >= 1
        {"min_relative_threshold": 1.0},  # not < 1
    ]
    for bad in bad_cases:
        try:
            validate_case_config({**EXAMPLE_CASE_CONFIG, **bad})
            raise AssertionError(f"expected CaseConfigError for {bad}")
        except CaseConfigError:
            pass
    print("optional detection/sizing override validation verified (good case passes, each bad case rejected).")
