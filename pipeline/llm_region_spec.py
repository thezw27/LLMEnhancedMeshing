"""
llm_region_spec.py -- have an LLM draft/revise region_spec.json, with **no
Cowork dependency**: this calls Claude directly through the plain Anthropic
API (`pip install anthropic`, set ANTHROPIC_API_KEY), so `driver.py
--auto-llm` works from any terminal, not just inside a Cowork chat.

Design choice, same spirit as size_field_builder.py: the LLM is *only* ever
asked to produce the "regions" list (shapes + falloff + target sizes) --
never the "defaults" block (hmin/hmax/background_size/growth_rate). Those
come from case_config/region_spec.defaults_from_case_config() deterministically,
set by the human once per case. So even in fully automated --auto-llm mode,
the LLM can misjudge *where* a feature is or how tight to make it, but it
structurally cannot violate the hmin<=background<=hmax numeric contract --
and region_spec.validate_region_spec() still runs on the assembled spec
afterward as a hard backstop regardless.

If you're driving this pipeline from inside a Cowork chat instead, you don't
need this module at all: skip --auto-llm in driver.py and just ask the
assistant you're talking to to write/edit region_spec.json directly at the
path driver.py tells you to -- that's the LLM's contribution in that mode,
same idea, just a different "LLM" doing the authoring.
"""

from __future__ import annotations

import json
import os

from region_spec import validate_region_spec, defaults_from_case_config, RegionSpecError

DEFAULT_MODEL = "claude-sonnet-4-5-20250929"

_SCHEMA_PRIMER = """You are drafting the "regions" list of a region_spec.json file used to \
drive anisotropic CFD mesh adaptation. You NEVER set hmin/hmax/background_size/growth_rate \
-- those are fixed by the human's case_config and given to you below as read-only context.

Output ONLY a JSON object of this exact shape, nothing else (no markdown fences, no commentary):

{
  "regions": [
    {
      "id": "<short_snake_case_id>",
      "description": "<one sentence, for humans>",
      "shape": {"type": "plane|box|sphere|cylinder|points", "...type-specific fields...": "..."},
      "influence": {"core_distance": 0.0, "falloff_distance": 0.0},
      "target_size": {
        "normal": 0.0, "tangential_1": 0.0, "tangential_2": 0.0,
        "direction_normal": null
      }
    }
  ]
}

Shape field requirements:
  sphere:   center [x,y,z], radius r
  box:      min [x,y,z], max [x,y,z]
  plane:    point [x,y,z], normal [nx,ny,nz]
  cylinder: point [x,y,z], axis [ax,ay,az], radius r
  points:   coordinates [[x,y,z], ...]

All target_size / influence values are lengths in the mesh's own coordinate units, and
should respect hmin <= value <= hmax (given below). "normal" should be near hmin for a
tight feature like a shock; tangential values can be larger, closer to background_size,
for the along-feature directions. core_distance/falloff_distance describe how far the
tight sizing extends and over what extra distance it relaxes back to background.
"""


def _extract_json_object(text: str) -> dict:
    start = text.find("{")
    if start == -1:
        raise ValueError(f"no JSON object found in LLM response: {text[:300]!r}")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i + 1])
    raise ValueError(f"unbalanced JSON object in LLM response: {text[:300]!r}")


def _client():
    try:
        import anthropic
    except ImportError as e:
        raise ImportError(
            "pip install anthropic to use --auto-llm (or omit --auto-llm and author "
            "region_spec.json yourself / via the assistant you're chatting with instead)."
        ) from e
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Set it to use --auto-llm, or omit --auto-llm and "
            "author region_spec.json by hand / through a chat instead."
        )
    return anthropic.Anthropic(api_key=api_key)


def _call_llm(client, model: str, system: str, user: str, max_tokens: int = 3000) -> str:
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return "".join(block.text for block in resp.content if getattr(block, "type", None) == "text")


def draft_region_spec(feature_summary: dict, case_config: dict, human_notes: str = "",
                       model: str = DEFAULT_MODEL, max_retries: int = 3) -> dict:
    """First-pass region_spec authoring straight from the feature summary
    (+ optional human notes), calling Claude directly via the Anthropic API."""
    client = _client()
    defaults, notes = defaults_from_case_config(case_config)
    for note in notes:
        print(f"    {note}")

    user_prompt = (
        f"case: {case_config['case_name']}\n"
        f"defaults (fixed, read-only): {json.dumps(defaults)}\n\n"
        f"feature summary from feature_extraction.compute_feature_summary:\n"
        f"{json.dumps(feature_summary, indent=2)}\n\n"
        + (f"human notes: {human_notes}\n\n" if human_notes else "")
        + "Draft the regions list now."
    )

    error_context = ""
    last_err = None
    for _ in range(max_retries):
        text = _call_llm(client, model, _SCHEMA_PRIMER, user_prompt + error_context)
        try:
            parsed = _extract_json_object(text)
            spec = {
                "defaults": defaults,
                "regions": parsed["regions"],
                "human_feedback_log": (
                    [{"turn": 0, "instruction": human_notes, "applied_as": "initial LLM draft"}]
                    if human_notes else []
                ),
            }
            validate_region_spec(spec)
            return spec
        except (RegionSpecError, ValueError, KeyError) as e:
            last_err = e
            error_context = (f"\n\nYour previous answer was invalid: {e}. "
                              f"Fix it and return the corrected JSON object only.")
    raise RuntimeError(f"LLM failed to produce a valid region_spec after {max_retries} attempts: {last_err}")


def revise_region_spec(spec: dict, feature_summary: dict, case_config: dict, human_feedback: str,
                        model: str = DEFAULT_MODEL, max_retries: int = 3) -> dict:
    """Revise an existing region_spec's regions in response to human feedback
    on the visualized size field (or on a previously adapted mesh)."""
    client = _client()
    defaults = spec["defaults"]  # keep the existing defaults untouched -- LLM never edits these
    turn = len(spec.get("human_feedback_log", []))

    user_prompt = (
        f"case: {case_config['case_name']}\n"
        f"defaults (fixed, read-only): {json.dumps(defaults)}\n\n"
        f"current regions: {json.dumps(spec['regions'], indent=2)}\n\n"
        f"feature summary: {json.dumps(feature_summary, indent=2)}\n\n"
        f"human feedback on the current size field: {human_feedback}\n\n"
        "Revise the regions list to address this feedback. Return the full updated regions list "
        "(not just the changed region)."
    )

    error_context = ""
    last_err = None
    for _ in range(max_retries):
        text = _call_llm(client, model, _SCHEMA_PRIMER, user_prompt + error_context)
        try:
            parsed = _extract_json_object(text)
            new_spec = {
                "defaults": defaults,
                "regions": parsed["regions"],
                "human_feedback_log": spec.get("human_feedback_log", []) + [
                    {"turn": turn, "instruction": human_feedback, "applied_as": "LLM revision"}
                ],
            }
            validate_region_spec(new_spec)
            return new_spec
        except (RegionSpecError, ValueError, KeyError) as e:
            last_err = e
            error_context = (f"\n\nYour previous answer was invalid: {e}. "
                              f"Fix it and return the corrected JSON object only.")
    raise RuntimeError(f"LLM failed to produce a valid revised region_spec after {max_retries} attempts: {last_err}")


if __name__ == "__main__":
    # This module calls out to the real Anthropic API, so its self-test is
    # necessarily limited to import/schema sanity rather than a live call --
    # driver.py --auto-llm is the integrated, end-to-end exercise of this code.
    from region_spec import EXAMPLE_REGION_SPEC
    validate_region_spec(EXAMPLE_REGION_SPEC)
    assert "regions" in _SCHEMA_PRIMER and "hmin" in _SCHEMA_PRIMER
    test_json = _extract_json_object('some preamble text {"regions": [{"a": 1}]} trailing text')
    assert test_json == {"regions": [{"a": 1}]}
    print("self-test OK (no live API call): schema primer present, JSON extraction works, "
          "region_spec module importable. Needs ANTHROPIC_API_KEY for a real draft/revise call.")
