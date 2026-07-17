# LLMEnhancedMeshing

Zachary Gordon
Emmet Whitehead
Soumyanil Sadhu Deep

## What this is

An LLM-driven pipeline for anisotropic mesh adaptation with Simmetrix: read a
CFD solution (VTU), extract solution features (shocks, boundary layers, etc.)
as a compact human-readable summary, have an LLM turn that (plus human
feedback in plain English) into a structured refinement spec, deterministically
build the per-vertex `[3][3]` anisoSize matrices from that spec, and hand them
to Simmetrix (`MSA_setAnisoVertexSize`) to adapt the mesh — producing an
adapted VTU (for review) and an adapted Fluent `.cas` file.

Everything runs on one machine — this repo is meant to be cloned and used
directly on `lore.scorec.rpi.edu` (or wherever the Simmetrix SimModSuite
license/modules live). There's no separate "remote SCOREC machine" hand-off
step; the Python pipeline invokes the Simmetrix C++ side as a direct
subprocess call.

See [`UPDATES.md`](UPDATES.md) for the detailed current architecture,
file-by-file breakdown, and open items.

## Getting started (new clone)

### 1. Python environment

The system Python here has no working `pip` and lacks `meshio`/`scipy`/
`numpy`. Set up the dedicated conda environment the pipeline needs:

```
bash pipeline/setup_env.sh      # creates (or updates) the "llmesh" env
conda activate llmesh
```

(Needs `conda` already on `PATH`. If you don't have it, install Miniconda
first: https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh)

### 2. Build the Simmetrix side

```
module use /opt/scorec/spack/rhel9/v0222_2/lmod/linux-rhel9-x86_64/Core/
module load gcc/13.2.0-4eahhas mpich/4.2.3-62uy3hd cmake gsl
export SIM_MPI="mpich4.2.3"
module load simmetrix-simmodsuite/2026.0-260411-3dtgxhh

cd simmetrix
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j4
```

This builds `apply_aniso_size_field`, the mesh-adaptation executable
`run_on_scorec.sh` calls (it also handles the Fluent `.cas` export via
SimModelerScript — no separate build step needed for that). Licensing is
handled automatically by the `simmetrix-simmodsuite` module on this system,
nothing to configure by hand.

`simmetrix/extrude/` (a separate tool that re-extrudes a 2D face into a 3D
volume mesh) needs a different toolchain/PUMI build — see the comment at
the top of `simmetrix/extrude/CMakeLists.txt` and `UPDATES.md` if you need it.

### 3. Set up a case

Write a `case_config.json` — see `pipeline/case_config.py`'s
`EXAMPLE_CASE_CONFIG` for the exact required fields, or
`testCases/ramp2/case_config.json` for a real filled-in example (paired
with `testCases/ramp2/driver_run/`, the output of an actual successful run,
if you want to see what a completed run looks like first).

You'll need: your input VTU (with a real solution on it), which fields to
detect features on, `hmin`/`hmax`, and the Simmetrix model/mesh/native-geometry
paths for your case.

### 4. Run it

```
conda activate llmesh
cd pipeline
python driver.py plan --case-config /path/to/case_config.json
```

The first run pauses and prints a feature summary — author `region_spec.json`
at the path it prints (by hand, or ask an LLM you're chatting with to write
it from the printed summary), then re-run the same command. It builds the
size field, runs the real adaptation, and prints where the adapted VTU and
Fluent `.cas` landed. Then:

```
python driver.py review --case-config /path/to/case_config.json --adapted-dir <output_dir>
```

to visualize the adapted mesh and approve/reject it.

Add `--auto-llm` to `plan` to have Claude draft/revise `region_spec.json`
automatically via the Anthropic API instead of pausing for a human (needs
`pip install anthropic` in the `llmesh` env, plus `ANTHROPIC_API_KEY` set).

## Repo layout

- `pipeline/` — the Python side: feature extraction, region spec schema,
  size field building, visualization, and adaptation orchestration
  (`driver.py` is the CLI entry point)
- `simmetrix/` — the Simmetrix C++ side: `apply_aniso_size_field`,
  `run_on_scorec.sh`, `translateToCas.py`, `extrude`
- `testCases/` — real example cases: model/mesh/native-geometry files,
  size fields, a filled-in `case_config.json`
- `simDocs/` — vendored Simmetrix API documentation (Doxygen HTML)
- `UPDATES.md` — current architecture and handoff notes, the most
  detailed/up-to-date doc in this repo
