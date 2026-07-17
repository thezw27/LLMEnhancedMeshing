"""
driver.py -- end-to-end CLI orchestrator for the LLM-enhanced anisotropic
adaptation pipeline. Runs entirely on one machine (this one) -- there is
no separate "local sandbox" that hands a run script to a human to execute
elsewhere; run_adaptation.py invokes the Simmetrix side directly. Ties
together every stage that used to be a separate hand-run snippet:

  read_vtu -> feature_extraction -> region_spec authoring (LLM or human) ->
  size_field_builder -> standalone 3D visualization -> human approve/revise
  loop -> export_for_simmetrix -> run_adaptation (adapt + Fluent .cas export)

...and a second command, `review`, for looking at what came out of
adaptation: visualize the adapted VTU, get an approve/reject on the mesh
itself, and point at the adapted Fluent .cas once approved.

No Cowork dependency: this is plain Python (argparse + the pipeline
modules: vtu_io, feature_extraction, region_spec, size_field_builder,
export_for_simmetrix, case_config, run_adaptation, visualization_data,
visualize_standalone), runs from any terminal.

The system Python here has no working pip and lacks meshio/scipy/numpy,
so this needs its own conda env: `bash setup_env.sh` once, then
`conda activate llmesh` before running this (see environment.yml).

region_spec authoring can go two ways:
  --auto-llm    call Claude directly via the Anthropic API (llm_region_spec.py)
                -- needs `pip install anthropic` and ANTHROPIC_API_KEY. Fully
                unattended after that (still subject to the approve/revise
                loop below, which also runs automatically at --yes).
  (default)     pause and ask a human -- or an LLM you're chatting with, e.g.
                inside Cowork -- to write/edit region_spec.json by hand at
                the path this prints, then re-run the same command.

Usage:
  python driver.py plan   --case-config case_config.json [--auto-llm] [--notes "..."] \\
                          [--yes] [--max-rounds 3] [--out-dir DIR]
  python driver.py review --case-config case_config.json --adapted-dir results/<case_name> [--yes]

Verified end-to-end against the synthetic double-ramp test case in
pipeline/tests/ (double_ramp.vtu): `plan` correctly pauses the first time
with no region_spec.json (prints the feature summary + defaults + where to
write it), completes the whole build->visualize->export->adapt flow once
one exists, and `review` correctly visualizes an adapted VTU and reports
the paired .cas path on approval (or tells you to go back and revise on
rejection).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from case_config import load_case_config, CaseConfigError, ADAPTED_VTU_NAME, ADAPTED_CAS_NAME
from vtu_io import read_vtu
from feature_extraction import compute_feature_summary
from region_spec import validate_region_spec, defaults_from_case_config, RegionSpecError
from size_field_builder import build_size_field
from export_for_simmetrix import write_size_field
from run_adaptation import run_adaptation, AdaptationError
from visualization_data import build_whole_geometry_payload, build_adapted_mesh_payload, payload_to_json_dict
from visualize_standalone import write_standalone_viewer


def _prompt(msg: str) -> str:
    return input(msg)


def _load_or_author_region_spec(path: str, summary: dict, cfg: dict, args) -> dict:
    """Load an existing region_spec.json if present (validating it), else
    either draft one via --auto-llm, or pause and hand the human/chat-LLM
    the feature summary + defaults + instructions to author one by hand."""
    if os.path.exists(path):
        with open(path) as f:
            spec = json.load(f)
        try:
            validate_region_spec(spec)
        except RegionSpecError as e:
            print(f"ERROR: {path} exists but is invalid: {e}")
            sys.exit(1)
        spec.setdefault("human_feedback_log", [])
        print(f"==> loaded existing {path}")
        return spec

    if args.auto_llm:
        from llm_region_spec import draft_region_spec
        print("==> drafting region_spec.json via the Anthropic API (--auto-llm)")
        spec = draft_region_spec(summary, cfg, human_notes=args.notes or "")
        spec.setdefault("human_feedback_log", [])
        with open(path, "w") as f:
            json.dump(spec, f, indent=2)
        print(f"    wrote {path}")
        return spec

    defaults = defaults_from_case_config(cfg)
    print(f"\nNo region_spec.json yet at {path}.")
    print("Author one now: read the feature summary above, decide which candidate regions matter, "
          "and write a region for each (see region_spec.EXAMPLE_REGION_SPEC for the exact shape).")
    print(f"defaults to start from (do not change hmin/hmax/background_size/growth_rate): "
          f"{json.dumps(defaults)}")
    print("If you're running this from inside a chat with an LLM, this is the step where you'd ask "
          "it to write the file for you, based on the feature summary printed above.")
    print(f"Re-run this exact command once {path} exists.")
    sys.exit(0)


def cmd_plan(args):
    try:
        cfg = load_case_config(args.case_config)
    except CaseConfigError as e:
        print(f"ERROR: invalid case_config: {e}")
        sys.exit(1)

    out_dir = args.out_dir or f"run_{cfg['case_name']}"
    os.makedirs(out_dir, exist_ok=True)

    print(f"==> reading VTU: {cfg['vtu_path']}")
    mesh = read_vtu(cfg["vtu_path"])
    print(f"    {len(mesh['points'])} vertices")

    print(f"==> extracting features on {cfg['driver_fields']}")
    summary = compute_feature_summary(mesh, cfg["driver_fields"])
    summary_path = os.path.join(out_dir, "feature_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"    wrote {summary_path}")
    for field, s in summary["fields"].items():
        print(f"    {field}: {len(s['candidate_regions'])} candidate region(s), "
              f"gradient threshold {s['gradient_threshold_used']:.4g}")

    region_spec_path = os.path.join(out_dir, "region_spec.json")
    spec = _load_or_author_region_spec(region_spec_path, summary, cfg, args)

    result = None
    round_n = 0
    while True:
        result = build_size_field(mesh, spec)
        payload = payload_to_json_dict(build_whole_geometry_payload(mesh, result))
        viewer_path = os.path.join(out_dir, f"size_field_view_round{round_n}.html")
        write_standalone_viewer(payload, viewer_path,
                                 title=f"{cfg['case_name']} -- size field (round {round_n})")
        print(f"==> round {round_n}: wrote {viewer_path} -- open it in a browser and look at the size field")
        print(f"    iso-equivalent size range: "
              f"{result['iso_equivalent_size'].min():.4g} - {result['iso_equivalent_size'].max():.4g}")

        if args.yes:
            print("==> --yes: auto-approving")
            break
        if round_n >= args.max_rounds:
            print(f"==> hit --max-rounds ({args.max_rounds}), approving current size field")
            break

        answer = _prompt("Approve this size field? [y/n] ").strip().lower()
        if answer.startswith("y"):
            break

        feedback = _prompt("What should change? ")
        if args.auto_llm:
            from llm_region_spec import revise_region_spec
            print("==> sending feedback to the LLM for a revision (--auto-llm)")
            spec = revise_region_spec(spec, summary, cfg, feedback)
        else:
            spec.setdefault("human_feedback_log", []).append(
                {"turn": round_n + 1, "instruction": feedback, "applied_as": "pending manual edit"}
            )
            with open(region_spec_path, "w") as f:
                json.dump(spec, f, indent=2)
            _prompt(f"Edit {region_spec_path} to address that feedback, then press Enter to continue...")
            with open(region_spec_path) as f:
                spec = json.load(f)
            validate_region_spec(spec)
            spec.setdefault("human_feedback_log", [])

        with open(region_spec_path, "w") as f:
            json.dump(spec, f, indent=2)
        round_n += 1

    with open(region_spec_path, "w") as f:
        json.dump(spec, f, indent=2)

    size_field_prefix = os.path.join(out_dir, "size_field")
    write_size_field(size_field_prefix, mesh, result)

    print("\n==> running adaptation (apply_aniso_size_field + Fluent .cas export)...")
    try:
        adapt_result = run_adaptation(cfg, f"{size_field_prefix}.txt")
    except AdaptationError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    print(f"    adapted VTU: {adapt_result['adapted_vtu']}")
    print(f"    adapted Fluent case: {adapt_result['adapted_cas']}")
    print(f"    raw Simmetrix mesh: {adapt_result['adapted_sms']}")

    print("\n==> plan complete.")
    print(f"    next: python driver.py review --case-config {args.case_config} "
          f"--adapted-dir {adapt_result['output_dir']}")


def cmd_review(args):
    try:
        cfg = load_case_config(args.case_config)
    except CaseConfigError as e:
        print(f"ERROR: invalid case_config: {e}")
        sys.exit(1)

    adapted_vtu_path = os.path.join(args.adapted_dir, ADAPTED_VTU_NAME)
    adapted_cas_path = os.path.join(args.adapted_dir, ADAPTED_CAS_NAME)

    if not os.path.exists(adapted_vtu_path):
        print(f"ERROR: expected adapted VTU at {adapted_vtu_path} -- did `driver.py plan` "
              f"finish successfully? (--adapted-dir should be the output_dir it printed)")
        sys.exit(1)

    print(f"==> reading adapted VTU: {adapted_vtu_path}")
    mesh = read_vtu(adapted_vtu_path)
    print(f"    {len(mesh['points'])} vertices")

    payload = payload_to_json_dict(build_adapted_mesh_payload(mesh))
    out_path = os.path.join(args.adapted_dir, "adapted_mesh_view.html")
    write_standalone_viewer(payload, out_path, title=f"{cfg['case_name']} -- adapted mesh review")
    print(f"==> wrote {out_path} -- open it and check the adapted mesh "
          f"(colored/glyphed by actual local edge length, not a requested size)")

    if args.yes:
        approved = True
    else:
        approved = _prompt("Does this adapted mesh look right? [y/n] ").strip().lower().startswith("y")

    if approved:
        print(f"==> approved. Adapted Fluent case ready at: {adapted_cas_path}")
        if not os.path.exists(adapted_cas_path):
            print(f"    WARNING: that file doesn't actually exist yet at this path -- "
                  f"check run_on_scorec.sh's output (see logs/simmetrix.log in this same dir).")
    else:
        print("==> not approved. Go back to `driver.py plan` (edit region_spec.json to fix the "
              "regions responsible for the bad area) and re-run.")


def main():
    parser = argparse.ArgumentParser(description="LLM-enhanced anisotropic adaptation pipeline driver")
    sub = parser.add_subparsers(dest="command", required=True)

    p_plan = sub.add_parser(
        "plan",
        help="detect features -> region_spec -> size field -> visualize -> export -> run adaptation",
    )
    p_plan.add_argument("--case-config", required=True)
    p_plan.add_argument("--auto-llm", action="store_true",
                         help="use the Anthropic API to draft/revise region_spec.json instead of "
                              "pausing for a human/chat edit")
    p_plan.add_argument("--notes", default="", help="optional human notes to seed the first --auto-llm draft")
    p_plan.add_argument("--yes", action="store_true", help="auto-approve the first size field, skip the feedback loop")
    p_plan.add_argument("--max-rounds", type=int, default=3)
    p_plan.add_argument("--out-dir", default=None)
    p_plan.set_defaults(func=cmd_plan)

    p_review = sub.add_parser("review", help="visualize the adapted mesh and approve/reject it")
    p_review.add_argument("--case-config", required=True)
    p_review.add_argument("--adapted-dir", required=True,
                           help="output_dir that `driver.py plan` printed (run_adaptation's output)")
    p_review.add_argument("--yes", action="store_true")
    p_review.set_defaults(func=cmd_review)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
