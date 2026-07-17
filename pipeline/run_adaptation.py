"""
run_adaptation.py — runs simmetrix/run_on_scorec.sh directly as a
subprocess. This pipeline and the Simmetrix C++ side run on the same
machine (see case_config.py), so no ssh/scp staging is needed — the size
field file this pipeline just wrote is already sitting on the same
filesystem run_on_scorec.sh reads from.

run_on_scorec.sh itself runs apply_aniso_size_field (adapts the mesh,
writes adapted.vtu + adapted_mesh/adapted.sms) and then translateToCas.py
via SimModelerScript (writes adapted.cas) — see that script for the
module/version requirements it loads. Both steps run here, in one call;
this module doesn't distinguish between them.

Writes into <case_config['work_dir']>/output/:
  adapted.vtu               -- from apply_aniso_size_field
  adapted_mesh/adapted.sms  -- from apply_aniso_size_field (raw Simmetrix mesh, reference only)
  logs/simmetrix.log        -- from apply_aniso_size_field
  adapted.cas               -- from translateToCas.py (adapted Fluent case)
"""

from __future__ import annotations

import os
import subprocess

from case_config import validate_case_config, ADAPTED_VTU_NAME, ADAPTED_CAS_NAME


class AdaptationError(RuntimeError):
    pass


def run_adaptation(case_config: dict, size_field_txt: str) -> dict:
    """Runs run_on_scorec_script and returns the paths it produced.

    Returns {"output_dir", "adapted_vtu", "adapted_cas", "adapted_sms"}.
    Raises AdaptationError if the script fails or doesn't produce the
    adapted VTU (the .cas is reported but not required to exist — Fluent
    export can be treated as best-effort by callers that only care about
    the mesh).
    """
    validate_case_config(case_config)

    output_dir = os.path.join(case_config["work_dir"], "output")
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        case_config["run_on_scorec_script"],
        case_config["native_model"],
        case_config["model_smd"],
        case_config["mesh_sms"],
        os.path.abspath(size_field_txt),
        output_dir,
    ]
    env = dict(os.environ)
    if "apply_aniso_exe" in case_config:
        env["APPLY_ANISO_EXE"] = case_config["apply_aniso_exe"]

    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        raise AdaptationError(f"run_on_scorec_script exited {result.returncode}: {' '.join(cmd)}")

    adapted_vtu = os.path.join(output_dir, ADAPTED_VTU_NAME)
    adapted_cas = os.path.join(output_dir, ADAPTED_CAS_NAME)
    adapted_sms = os.path.join(output_dir, "adapted_mesh", "adapted.sms")
    if not os.path.exists(adapted_vtu):
        raise AdaptationError(f"run_on_scorec_script exited 0 but {adapted_vtu} does not exist")

    return {
        "output_dir": output_dir,
        "adapted_vtu": adapted_vtu,
        "adapted_cas": adapted_cas,
        "adapted_sms": adapted_sms,
    }


if __name__ == "__main__":
    # Self-test without actually invoking Simmetrix: point
    # run_on_scorec_script at a stub that fakes the two output files, and
    # confirm run_adaptation() builds the right command and finds them.
    import stat
    import tempfile

    from case_config import EXAMPLE_CASE_CONFIG

    with tempfile.TemporaryDirectory() as tmp:
        stub_path = os.path.join(tmp, "fake_run_on_scorec.sh")
        work_dir = os.path.join(tmp, "run")
        with open(stub_path, "w") as f:
            f.write(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                'echo "args: $@" > "$5/args.txt"\n'
                'mkdir -p "$5/adapted_mesh"\n'
                'echo fake_vtu > "$5/adapted.vtu"\n'
                'echo fake_sms > "$5/adapted_mesh/adapted.sms"\n'
                'echo fake_cas > "$5/adapted.cas"\n'
            )
        os.chmod(stub_path, os.stat(stub_path).st_mode | stat.S_IEXEC)

        size_field_txt = os.path.join(tmp, "size_field.txt")
        with open(size_field_txt, "w") as f:
            f.write("0 0 0 0.01 0 0 0 0.01 0 0 0 0.01\n")

        cfg = {**EXAMPLE_CASE_CONFIG, "run_on_scorec_script": stub_path, "work_dir": work_dir}
        result = run_adaptation(cfg, size_field_txt)

        assert os.path.exists(result["adapted_vtu"])
        assert os.path.exists(result["adapted_cas"])
        assert os.path.exists(result["adapted_sms"])
        with open(os.path.join(result["output_dir"], "args.txt")) as f:
            args_line = f.read()
        assert cfg["native_model"] in args_line
        assert cfg["model_smd"] in args_line
        assert cfg["mesh_sms"] in args_line
        assert size_field_txt in args_line

    print("self-test OK: run_adaptation() invokes run_on_scorec_script with the right args "
          "and locates adapted.vtu/adapted.cas/adapted.sms afterward")
