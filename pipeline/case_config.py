"""
case_config.py — the per-case configuration a human supplies when starting
a new case with a VTU: where the matching Simmetrix model/mesh live on
SCOREC, the meshing bounds, and which fields drive feature detection.

This is deliberately separate from region_spec.json: region_spec is what
the LLM writes *after* reading the feature summary (refinement intent).
case_config is set once per case, by the human, before any of that —
it's the addressing information (local VTU, remote model/mesh, remote
executable) plus the meshing bounds (hmin/hmax) that flow into
region_spec's defaults.

Note on the sandbox this pipeline currently runs in: it cannot reach
arbitrary outbound hosts (confirmed — SCOREC's hostname doesn't even
resolve from here, network egress is allowlisted). So nothing in this
pipeline calls ssh/scp directly. generate_scorec_run_script.py below
takes a CaseConfig and produces a plain shell script that a human runs
from a machine that *does* have SCOREC access.

SCOREC's handoff back is two files, not one: an adapted **VTU** (for
visualizing the new mesh here and getting human approve/reject feedback)
and an adapted Fluent **.cas** (only used downstream, to actually run
Fluent on the new mesh, once the human is happy with what the VTU shows).
`adapted_vtu_name`/`adapted_cas_name` below are the remote filenames
`run_on_scorec_script` writes into `remote_work_dir/output/` — override
them per case if SCOREC uses different names; otherwise the defaults below
are used.

`native_model` is the CAD kernel's native geometry file (e.g. a Parasolid
.x_t) — needed only for the Fluent .cas export step (translateToCas.py),
not by apply_aniso_size_field itself, which loads model_smd directly.
`run_on_scorec_script` is simmetrix/run_on_scorec.sh: it runs
apply_aniso_size_field (writes adapted.vtu + adapted_mesh/adapted.sms),
then runs translateToCas.py via SimModelerScript (writes adapted.cas) —
see that script for the exact module/version requirements. This is what
generate_scorec_run_script.py invokes remotely, not apply_aniso_exe
directly, since the plain executable alone never produces the .cas.
"""

from __future__ import annotations

import json


class CaseConfigError(ValueError):
    pass


_REQUIRED_TOP = {"case_name", "vtu_path", "driver_fields", "hmin", "hmax", "scorec"}
_REQUIRED_SCOREC = {
    "host", "user", "native_model", "model_smd", "mesh_sms",
    "apply_aniso_exe", "run_on_scorec_script", "remote_work_dir",
}
_DEFAULT_ADAPTED_VTU_NAME = "adapted.vtu"
_DEFAULT_ADAPTED_CAS_NAME = "adapted.cas"


def validate_case_config(cfg: dict) -> None:
    missing = _REQUIRED_TOP - set(cfg.keys())
    if missing:
        raise CaseConfigError(f"case_config missing top-level keys: {sorted(missing)}")

    if not isinstance(cfg["driver_fields"], list) or not cfg["driver_fields"]:
        raise CaseConfigError("driver_fields must be a non-empty list, e.g. ['temperature']")

    if not (0 < cfg["hmin"] <= cfg["hmax"]):
        raise CaseConfigError(f"expected 0 < hmin <= hmax, got hmin={cfg['hmin']} hmax={cfg['hmax']}")

    scorec = cfg["scorec"]
    missing_scorec = _REQUIRED_SCOREC - set(scorec.keys())
    if missing_scorec:
        raise CaseConfigError(f"case_config['scorec'] missing keys: {sorted(missing_scorec)}")

    for key in ("native_model", "model_smd", "mesh_sms", "apply_aniso_exe",
                "run_on_scorec_script", "remote_work_dir"):
        if not scorec[key] or not scorec[key].startswith("/"):
            raise CaseConfigError(f"scorec.{key} should be an absolute remote path, got {scorec[key]!r}")

    # adapted_vtu_name / adapted_cas_name are optional (defaults filled in by
    # scorec_output_names below), but if present they must be plain filenames,
    # not paths -- they're joined onto remote_work_dir/output/ downstream.
    for key in ("adapted_vtu_name", "adapted_cas_name"):
        if key in scorec and ("/" in scorec[key] or not scorec[key]):
            raise CaseConfigError(f"scorec.{key} should be a bare filename (no '/'), got {scorec[key]!r}")


def scorec_output_names(cfg: dict) -> tuple[str, str]:
    """Return (adapted_vtu_name, adapted_cas_name), filling in defaults for
    whichever one(s) the case_config didn't override."""
    scorec = cfg["scorec"]
    return (
        scorec.get("adapted_vtu_name", _DEFAULT_ADAPTED_VTU_NAME),
        scorec.get("adapted_cas_name", _DEFAULT_ADAPTED_CAS_NAME),
    )


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
    "scorec": {
        "host": "lore.scorec.rpi.edu",
        "user": "gordoz2",
        "native_model": "/users/gordoz2/lore.scorec.rpi.edu/adaptiveController/cases/double_ramp_test1/model_nat.x_t",
        "model_smd": "/users/gordoz2/lore.scorec.rpi.edu/adaptiveController/cases/double_ramp_test1/model.smd",
        "mesh_sms": "/users/gordoz2/lore.scorec.rpi.edu/adaptiveController/cases/double_ramp_test1/mesh.sms",
        "apply_aniso_exe": "/users/gordoz2/lore.scorec.rpi.edu/adaptiveController/build/apply_aniso_size_field",
        "run_on_scorec_script": "/users/gordoz2/lore.scorec.rpi.edu/adaptiveController/run_on_scorec.sh",
        "remote_work_dir": "/users/gordoz2/lore.scorec.rpi.edu/adaptiveController/runs/double_ramp_test1",
        # optional -- shown explicitly here even though they match the
        # defaults, so it's obvious where to override them per case:
        "adapted_vtu_name": "adapted.vtu",
        "adapted_cas_name": "adapted.cas",
    },
}

if __name__ == "__main__":
    validate_case_config(EXAMPLE_CASE_CONFIG)
    assert scorec_output_names(EXAMPLE_CASE_CONFIG) == ("adapted.vtu", "adapted.cas")
    stripped = {**EXAMPLE_CASE_CONFIG, "scorec": {k: v for k, v in EXAMPLE_CASE_CONFIG["scorec"].items()
                                                    if k not in ("adapted_vtu_name", "adapted_cas_name")}}
    validate_case_config(stripped)
    assert scorec_output_names(stripped) == ("adapted.vtu", "adapted.cas")
    print("EXAMPLE_CASE_CONFIG is valid; scorec_output_names() defaults verified.")
