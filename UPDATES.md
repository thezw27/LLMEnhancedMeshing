# UPDATES — current state of the SCOREC-side pipeline

Context: this whole repo now runs on one machine (`lore.scorec.rpi.edu`) —
there is no separate "local sandbox" that hands a script to a human to run
elsewhere. `pipeline/driver.py` runs feature extraction through mesh
adaptation as one flow, invoking the Simmetrix C++ side directly as a
subprocess. Read this before touching `case_config.json`, `pipeline/`, or
`simmetrix/`.

## 1. `simmetrix/` — the Simmetrix C++ side

- **`simmetrix/src/apply_aniso_size_field.cpp`** — reads a native model +
  `model.smd` + `mesh.sms`, maps each row of a size-field text file to its
  nearest mesh vertex via a nanoflann KD-tree (`simmetrix/src/nanoflann.hpp`,
  vendored, BSD-licensed), calls `MSA_setAnisoVertexSize`, runs `MSA_adapt`,
  then `VolumeMeshImprover` with `ShapeMetricType_VolLenRatio` target `0.3`
  (the only shape metric Simmetrix's docs say is properly supported on an
  anisotropic mesh), writes the adapted `.sms`, and writes `adapted.vtu`
  **directly** (no Simmetrix `meshExporter` plugin for VTU exists — this
  walks `M_regionIter`/`R_topoType`/`R_vertices` itself; supports
  tet/pyramid/wedge/hex, anything else is skipped with a warning).
  - Usage: `apply_aniso_size_field <native_model> <model.smd> <mesh.sms> <size_field.txt> <output_dir>`
  - `native_model` (e.g. a Parasolid `.x_t`) is required — `GM_load()` on
    this project's real Parasolid-backed `.smd` files fails outright
    without it ("Unable to read resource of type
    model.nonmanifold.parasolid"), confirmed against real test cases.
    Loaded via `ParasolidNM_createFromFile()` + `SimParasolid_start(1)`,
    same pattern as `exAnisoCyl_Parasolid.cc`.
  - Writes: `<output_dir>/adapted.vtu`, `<output_dir>/adapted_mesh/adapted.sms`, `<output_dir>/logs/simmetrix.log`
  - `size_field.txt` format: one line per point, `X Y Z m00 m01 m02 m10 m11
    m12 m20 m21 m22` (12 whitespace/comma-separated numbers; `#` comments
    and blank lines OK; no header) — matches what
    `pipeline/export_for_simmetrix.py` writes.
  - Builds against SimModSuite 2026.0-260411 (`simmetrix/CMakeLists.txt`,
    `SIM_PARASOLID ON`, same `FindSimModSuite.cmake` as `exampleCMakeList.txt`).

- **`simmetrix/run_on_scorec.sh`** — the single entry point everything else
  calls. Loads the module set `apply_aniso_size_field` was built against
  (`gcc/13.2.0-4eahhas`, `mpich/4.2.3-62uy3hd`,
  `simmetrix-simmodsuite/2026.0-260411-3dtgxhh`), runs it, then loads
  `simmetrix/simModeler/2026.0-260411` (confirmed correct for this
  SimModSuite version — not 2025.1) and runs `translateToCas.py` through
  `SimModelerScript` to produce `adapted.cas`.
  - Usage: `run_on_scorec.sh <native_model.x_t> <model.smd> <mesh.sms> <size_field.txt> <output_dir>`
  - Invoked directly as a subprocess by `pipeline/run_adaptation.py` — not
    over ssh, since everything is one machine now.

- **`simmetrix/translateToCas.py`** — Fluent .cas exporter via
  SimModelerScript's `simmetrix` Python module (`SimParasolidNativeModel` /
  `SimGModel` / `SimMesh` / `meshExporter("FLUENT")`). Takes the *adapted*
  mesh (not the original input) as its `mesh_file` arg.

- **`simmetrix/src/extrude.cc`** + **`simmetrix/extrude/CMakeLists.txt`** —
  a separate tool: takes an existing mesh + model, extracts the 2D
  triangulation classified on a given source face, and re-extrudes it into
  a fresh mixed-element volume mesh of a given thickness
  (`ExtrusionSizing_LayerSize`) between that face and a destination face.
  Usage: `extrude NAT_MODEL MODEL.smd MESH.sms SRC_FACE DST_FACE OUTMESH.sms THICKNESS`
  (face tags are model-specific — find them by enumerating `GM_faceIter`
  faces' centers/normals if not already known).
  - **Builds against a *different* core/PUMI checkout than everything
    else**: `apply_aniso_size_field` links raw SimModSuite directly, but
    `extrude.cc` also needs the SCOREC `core`/PUMI stack (`apf`, `gmi_sim`,
    etc.). The original prebuilt `core` at
    `/users/gordoz2/lore.scorec.rpi.edu/core` is on `master`, built against
    **SimModSuite 2025.1**/`mpich4.1.1` (see its `config.sh`) — genuinely
    cannot read this project's 2026-format `.smd` files (confirmed
    independently with a bare `GM_load()`, same error as above). So a
    **separate clone** exists at
    `/users/gordoz2/lore.scorec.rpi.edu/core-2026`, checked out to
    `develop` (merges PR #534 "support simModSuite 2026" — confirmed via
    `libSimParasolid381.a`), built against SimModSuite 2026.0-260411 to
    match everything else. `simmetrix/extrude/CMakeLists.txt` points at
    `core-2026`. **The original `core/` is untouched** (still `master`,
    still 2025.1) — don't repoint anything at it for this project's model
    files, and don't delete/rebuild it.
  - A real run (`testCases/ramp2`, the adapted mesh, faces 7/8, thickness
    0.006) succeeded: `Number of Region Extrusions identified: 1`,
    `Created 1569 volume extrusion elements.`, exit 0.

## 2. `pipeline/` — the Python side (runs on this same machine)

- **`pipeline/case_config.py`** — flat schema, no more `scorec`/`host`/
  `user` nesting (there's nothing remote to address anymore). Required:
  `case_name`, `vtu_path`, `driver_fields`, `hmin`, `hmax`, `native_model`,
  `model_smd`, `mesh_sms`, `run_on_scorec_script`, `work_dir` — all plain
  absolute local paths. `apply_aniso_exe` is optional (only needed to
  override `run_on_scorec.sh`'s own default build path).
  `ADAPTED_VTU_NAME`/`ADAPTED_CAS_NAME` are fixed constants
  (`"adapted.vtu"`/`"adapted.cas"`) matching what the C++/shell side
  actually hardcodes — not configurable, despite an earlier version of
  this file pretending they were.
- **`pipeline/run_adaptation.py`** (replaces the old
  `generate_scorec_run_script.py`, which is deleted) — runs
  `run_on_scorec_script` directly via `subprocess.run`, no ssh/scp. Returns
  `{"output_dir", "adapted_vtu", "adapted_cas", "adapted_sms"}`. Raises
  `AdaptationError` if the script fails or `adapted.vtu` doesn't show up.
- **`pipeline/driver.py`** — `plan` now runs adaptation directly at the end
  (feature extraction → region_spec → size field → **run_adaptation()** →
  prints where the adapted VTU/cas landed) instead of generating a script
  for a human to run elsewhere. `review` reads that same local directory.
  Both commands otherwise work the same as before.
- **Verified for real**: `run_adaptation()` invoked against
  `testCases/ramp2`'s real files (`ramp_nat.x_t`, `ramp.smd`,
  `ramp-initial.sms`, `size_field_final.txt`) → exit 0, produced real
  `adapted.vtu` (239 KB), `adapted.cas` (446 KB), `adapted_mesh/adapted.sms`
  (1.1 MB), `logs/simmetrix.log`. This is the actual code path
  `driver.py plan` now runs, not a mock.
- **Any `case_config.json` written before this refactor is invalid** under
  the new flat schema — needs `native_model`/`model_smd`/`mesh_sms`/
  `run_on_scorec_script`/`work_dir` at the top level, no `scorec` wrapper.

## 3. Feature detection redesign (`feature_extraction.py`)

**Bug that motivated this**: a first attempt at `testCases/wedgePaper`
placed a shock region at x=0.028, but the real wedge's leading-edge shock
(confirmed against the actual model geometry + theta-beta-M relations for
M=6/10°) is near x=0-0.01 — off by ~3x the wedge's own length. Root cause:
`compute_feature_summary` used **one global gradient-magnitude percentile
per field, over the whole mesh**. That's not a ramp-specific tuning
artifact (checked: it isn't — a separate ~600 lines of genuinely ramp2-
calibrated auto-build machinery in `size_field_builder.py` existed but
**was never called by `driver.py`**, so it wasn't the cause here; it's been
removed, see below). It's a structural flaw: once one feature dominates
the gradient distribution, a single percentile cutoff either merges
unrelated high-gradient regions into one bogus connected component, or
hides a weaker-but-real feature entirely.

**Fix — iterative "peel-off" multi-pass detection**: each pass computes its
percentile only over vertices no earlier pass already claimed as a region,
then removes what it finds before the next pass. Proven with a synthetic
self-test (`feature_extraction.py`'s `__main__`): a 1D chain with a strong
feature (slope 10, 300 vertices) and a weak one (slope 2, 100 vertices) —
a single 90th-percentile pass only ever finds the strong one (asserted
directly in the test); the new code finds both, in 2 passes.

**Second bug found immediately after, on real data**: unbounded peel-off
returned 40-56 "candidate regions" on `ramp2` — real solution fields are
never perfectly flat away from real features (turbulence/interpolation/
solver noise), so passes kept finding technically-above-percentile noise
indefinitely. Fixed with `min_relative_threshold` (default 0.01): stop once
a pass's threshold drops below 1% of that field's first (strongest) pass —
a broad two-orders-of-magnitude floor, not tuned to any one case. Brought
ramp2 down to 11/27 candidate regions across 2-3 passes; re-verified
`wedgePaper` still produces the *exact same* adapted mesh as before (418
pts/761 cells) — no regression to the actual adaptation output, since that
comes from the human-authored `region_spec.json`, not from
`feature_extraction`'s own region count.

**Configuration values now follow one consistent pattern** (per explicit
request: choose with stated reasoning, or let the user override) — each of
`gradient_percentile`, `min_region_size`/`min_region_fraction`,
`max_detection_passes`, `min_relative_threshold` (feature_extraction.py) and
`growth_rate`, `background_size` (region_spec.py) has a named default +
reasoning string constant; `driver.py` prints the reasoning for any value
NOT overridden in `case_config.json`, and every one is now a validated
optional `case_config.json` key (see `case_config.py`'s module docstring).
`region_spec.defaults_from_case_config` and `llm_region_spec.draft_region_spec`
changed signature accordingly (now return/consume `(defaults, notes)`).

**Dead code removed**: `size_field_builder.py`'s auto-build path
(`build_auto_size_field`, `detect_gradient_components`, centerline
extraction/densification/extension/smoothing, ~600 lines) — calibrated
against ramp2 specifically, never wired into `driver.py`. Recoverable from
git history if wanted again, but should be re-validated against more than
one case first. File went from 1012 → 348 lines.

**Known remaining limitation**: `_connected_components_on_mask` still
fragments a single physical feature into multiple disconnected components
whenever there's a detection gap (mesh resolution, threshold noise) —
11-27 regions for two fields on ramp2 is a big improvement over 40-56, but
still more than a truly "compact" summary; merging nearby same-direction
components is a reasonable next step if this becomes a problem in practice.

## Python environment for pipeline/

The system Python here has no working `pip` at all (`python3 -m pip` →
`No module named pip`, no `pip` binary on PATH, even under the `python`
spack module) and lacks `meshio`/`scipy`. Rather than fight that, a fresh
Miniconda was installed, self-contained, under this user's home dir (does
NOT touch the old `/opt/scorec/intel/intelpython3` conda 4.3.31 install,
which is separate and ancient):

```
/users/gordoz2/lore.scorec.rpi.edu/miniconda3            -- conda 26.5.3
/users/gordoz2/lore.scorec.rpi.edu/miniconda3/envs/llmesh -- python 3.11, meshio 5.3.5, scipy 1.17.1, numpy 2.4.6
```

Run pipeline/driver.py with:
```
/users/gordoz2/lore.scorec.rpi.edu/miniconda3/bin/conda run -n llmesh python3 driver.py plan --case-config ...
```
or `conda activate llmesh` first (needs `source .../miniconda3/etc/profile.d/conda.sh` in a non-interactive shell).

For anyone else cloning this repo (not this same home dir): `pipeline/environment.yml` + `pipeline/setup_env.sh` recreate the same env from scratch --
```
bash pipeline/setup_env.sh   # creates (or updates) the "llmesh" env from environment.yml
conda activate llmesh
```
(requires conda already on PATH; get Miniconda from
https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh if not).

### Full pipeline verified end-to-end on real CFD data

Ran `driver.py plan` + `driver.py review` against `testCases/ramp2/initEx.vtu`
(a real Fluent/CFD solution VTU for the ramp2 case — wedge/quad/triangle
cells, 3944 points, fields: density/pressure/mach_number/temperature/
velocity/etc, matching `ramp-initial.sms`'s vertex count) using
`case_config.json` in that same directory:

1. `plan` (first run, no region_spec.json yet): read the real VTU,
   extracted features on `["pressure", "mach_number"]` — found 2 and 3
   candidate regions respectively, all consistent with a single oblique
   shock (matching gradient directions across both fields) plus smaller
   secondary features. Wrote `feature_summary.json` and paused as designed.
2. Authored `region_spec.json` by hand from that real feature summary (one
   `plane`-shaped region at the dominant shock's centroid/gradient
   direction, tight normal size 0.001, background 0.03).
3. `plan` (second run, `--yes`): built the size field, ran the real
   adaptation (`MSA_adapt` + `VolumeMeshImprover`, exit 0), ran the real
   Fluent .cas export — wrote `adapted.vtu` (701 pts / 1883 tets),
   `adapted.cas`, `adapted_mesh/adapted.sms`.
4. `review --yes`: read the adapted VTU back, wrote the mesh-review HTML
   viewer, approved, reported the `.cas` path. Exit 0.

So this is no longer just structurally ready — the entire
VTU→features→region_spec→size_field→adapt→VTU/.cas loop has been run for
real, on real CFD data, on this machine, start to finish.

### Second real case (2D) found and fixed a real bug in the VTU writer

Ran the same full loop against `testCases/wedgePaper/wedge0.vtu` (a genuinely
2D case — all Z=0, triangle/line cells only, Mach 6 10° wedge). First attempt
produced `adapted.vtu` with **0 cells** — `writeAdaptedMeshVTU` in
`apply_aniso_size_field.cpp` only ever walked `M_regionIter` (3D volume
regions), so a surface-only mesh with no regions at all wrote points but no
connectivity. Fixed: if `M_regionIter` finds zero regions, it now falls back
to walking `M_faceIter`/`F_numEdges`/`F_vertices` and writes triangles/quads
instead (VTK_TRIANGLE=5/VTK_QUAD=9) — only in that no-regions case, since a
real volume mesh's faces include interior faces shared between two regions
that would double up the geometry if also written as cells. Re-ran after the
fix: `adapted.vtu` now has 572 points / 1043 real triangle cells,
connectivity internally consistent (verified with `xml.etree`).

## Still open
- The VTU writer's vertex ordering for pyramid/wedge/hex regions uses
  Simmetrix's `R_vertices(region, 1)` ordering as-is; tet ordering is
  proven correct end-to-end, but the other three haven't been exercised
  (test cases so far are pure tet) or visually verified in ParaView.
- `extrude.cc`'s `#ifndef NDEBUG` debug dump (`rasp_extrude_surf_mesh.sms`,
  hardcoded filename, written to cwd) is harmless but worth removing or
  namespacing if this gets used from more than one working directory at once.

## Directory layout

```
simmetrix/
  CMakeLists.txt              -- apply_aniso_size_field (SimModSuite 2026.0 only)
  run_on_scorec.sh             -- entry point: adapt + VTU + Fluent .cas
  translateToCas.py
  src/apply_aniso_size_field.cpp
  src/extrude.cc
  src/nanoflann.hpp
  extrude/CMakeLists.txt       -- extrude (SimModSuite 2026.0 + core-2026/PUMI)
  build/, extrude/build/       -- built artifacts, gitignored

pipeline/
  case_config.py       -- flat, single-machine schema
  driver.py             -- plan / review CLI, runs everything here
  run_adaptation.py     -- subprocess call into simmetrix/run_on_scorec.sh
  vtu_io.py, feature_extraction.py, region_spec.py, llm_region_spec.py,
  size_field_builder.py, export_for_simmetrix.py, visualization_data.py,
  visualize_standalone.py

core/          -- original prebuilt PUMI/core checkout, master, SimModSuite 2025.1 -- DO NOT TOUCH
core-2026/     -- new clone, develop branch, SimModSuite 2026.0 -- what extrude/ links against
```
