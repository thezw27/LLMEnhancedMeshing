# UPDATES — SCOREC-side Simmetrix work (this session)

Context: this session ran directly on `lore.scorec.rpi.edu` and built/extended
the C++ side of the pipeline (`simmetrix/`), then wired two new
`case_config.py` fields through `generate_scorec_run_script.py` so the
remote-run contract actually produces all three handoff files (adapted
mesh, VTU, Fluent .cas). Read this before touching any `case_config.json`
or the SCOREC remote-run scripts.

## What's new

### 1. `simmetrix/` — the actual Simmetrix C++ program

- **`simmetrix/src/apply_aniso_size_field.cpp`** — reads `model.smd` +
  `mesh.sms`, maps each row of a size-field text file to its nearest mesh
  vertex via a nanoflann KD-tree (`simmetrix/src/nanoflann.hpp`, vendored,
  BSD-licensed), calls `MSA_setAnisoVertexSize`, runs `MSA_adapt`, then runs
  `VolumeMeshImprover` with `ShapeMetricType_VolLenRatio` target `0.3` (the
  only shape metric Simmetrix's docs say is properly supported on an
  anisotropic mesh), writes the adapted `.sms`, and writes `adapted.vtu`
  **directly** — no Simmetrix `meshExporter` plugin for VTU was available,
  so this walks `M_regionIter`/`R_topoType`/`R_vertices` itself (supports
  tet/pyramid/wedge/hex; anything else is skipped with a warning, not a
  crash).
  - Usage: `apply_aniso_size_field <model.smd> <mesh.sms> <size_field.txt> <output_dir>`
  - Writes: `<output_dir>/adapted.vtu`, `<output_dir>/adapted_mesh/adapted.sms`, `<output_dir>/logs/simmetrix.log`
  - `size_field.txt` format: one line per point, `X Y Z m00 m01 m02 m10 m11
    m12 m20 m21 m22` (12 whitespace/comma-separated numbers; `#` comments
    and blank lines OK; no header) — this exactly matches what
    `pipeline/export_for_simmetrix.py` already writes.
  - Usage is actually `apply_aniso_size_field <native_model> <model.smd>
    <mesh.sms> <size_field.txt> <output_dir>` — a real end-to-end run (see
    "smoke test" below) found that `GM_load()` on this project's real
    Parasolid-backed `.smd` files fails outright ("Unable to read resource
    of type model.nonmanifold.parasolid") unless the native geometry
    (`.x_t`) is loaded via `ParasolidNM_createFromFile()` and passed in,
    bracketed by `SimParasolid_start(1)`/`SimParasolid_stop(1)` — exactly
    the pattern `exAnisoCyl_Parasolid.cc` uses. Fixed; `simmetrix/CMakeLists.txt`
    now also sets `SIM_PARASOLID ON` so the Parasolid kernel actually links.
  - Builds cleanly against SimModSuite 2026.0-260411
    (`simmetrix/CMakeLists.txt`, using the same `FindSimModSuite.cmake`
    module as `exampleCMakeList.txt`), and has now been **run for real** —
    see "smoke test" below.

- **`simmetrix/run_on_scorec.sh`** (new) — the actual remote entry point.
  Loads the module set `apply_aniso_size_field` was built against
  (`gcc/13.2.0-4eahhas`, `mpich/4.2.3-62uy3hd`,
  `simmetrix-simmodsuite/2026.0-260411-3dtgxhh`), runs
  `apply_aniso_size_field`, then loads `simmetrix/simModeler/2026.0-260411`
  (confirmed the correct module for this SimModSuite version — not 2025.1,
  which an older example elsewhere in this environment used) and runs
  `translateToCas.py` through `SimModelerScript` to produce `adapted.cas`.
  - Usage: `run_on_scorec.sh <native_model.x_t> <model.smd> <mesh.sms> <size_field.txt> <output_dir>`
  - This is what `generate_scorec_run_script.py` now invokes over ssh —
    **not** `apply_aniso_size_field` directly (the plain executable alone
    never produces the .cas).

- **`simmetrix/translateToCas.py`** — pre-existing, unchanged. Fluent .cas
  exporter via SimModelerScript's `simmetrix` Python module
  (`SimParasolidNativeModel` / `SimGModel` / `SimMesh` /
  `meshExporter("FLUENT")`). Needs the **adapted** mesh (not the original
  input mesh) as its `mesh_file` arg — `run_on_scorec.sh` passes
  `<output_dir>/adapted_mesh/adapted.sms`.

### 2. `pipeline/case_config.py` — two new required `scorec` fields

- **`native_model`** — absolute remote path to the CAD kernel's native
  geometry file (e.g. a Parasolid `.x_t`). Needed only for the .cas export
  step (`translateToCas.py`'s `nat_mod_file` arg) — `apply_aniso_size_field`
  itself doesn't need it (loads `model_smd` directly, no native model).
- **`run_on_scorec_script`** — absolute remote path to
  `simmetrix/run_on_scorec.sh`.
- Both validated the same way as `model_smd`/`mesh_sms`/etc (must be
  absolute paths). `EXAMPLE_CASE_CONFIG` updated to show both.
- **Any `case_config.json` written before this session is now invalid** —
  `validate_case_config()` will raise `CaseConfigError` for missing
  `native_model`/`run_on_scorec_script` until those are added.

### 3. `pipeline/generate_scorec_run_script.py` — updated to match

- The generated shell script now runs
  `${RUN_ON_SCOREC_SCRIPT} ${NATIVE_MODEL} ${MODEL_SMD} ${MESH_SMS}
  ${REMOTE_WORK_DIR}/size_field.txt ${REMOTE_WORK_DIR}/output` instead of
  calling `apply_aniso_exe` directly.
- Fixed an unrelated pre-existing self-test bug while in here: an assertion
  checked for a bare `ssh "${USER}@${HOST}"` substring the template never
  actually emits (`${SSH_OPTS[@]}` is always between them).
- `apply_aniso_exe` is still a valid field in case_config (still useful for
  manual testing, and `run_on_scorec.sh` accepts an `APPLY_ANISO_EXE` env
  override), it's just no longer what the generated script invokes
  directly.

All pipeline self-tests (`python3 <module>.py` for every file in
`pipeline/`) pass as of this update, plus a full `py_compile` sweep.

## Smoke test — actually run, end to end (not just compiled)

Ran the real chain against `testCases/ramp/` (`ramp_nat.x_t`, `ramp.smd`,
`ramp-initial.sms`, real license via `$SIM_LICENSE_FILE`):

1. Built a throwaway vertex-dump tool (same GM_load/M_load pattern) to get
   `ramp-initial.sms`'s actual 3926 vertex coordinates, then wrote a
   `size_field.txt` using those exact coordinates (isotropic, background
   size ≈ 2% of the model's bbox diagonal) — this is what caught the
   missing-native-model bug above on the first real attempt.
2. `apply_aniso_size_field ramp_nat.x_t ramp.smd ramp-initial.sms
   size_field.txt <out>` → **exit 0**. Nearest-vertex mapping distance
   min=0, max=5e-10 (floating-point noise — confirms exact coordinate
   correspondence). `MSA_adapt` ran 7 iterations (3926 → 21184 vertices,
   111126 tets), `VolumeMeshImprover` completed. Both `adapted.vtu` (valid
   XML, point/cell/connectivity counts internally consistent — verified
   with `xml.etree`) and `adapted_mesh/adapted.sms` written.
3. Full `run_on_scorec.sh ramp_nat.x_t ramp.smd ramp-initial.sms
   size_field.txt <out>` → **exit 0**. Ran the above, then
   `SimModelerScript -python translateToCas.py` produced `adapted.cas` — a
   valid Fluent case file whose cell count (`0x1b216` = 111126) matches the
   VTU exactly.

So the whole chain (adapt → VolumeMeshImprover → VTU → Fluent .cas) is
confirmed working on a real Parasolid-backed test case, not just compiled.
This test used a synthetic isotropic size field for coverage, not a real
CFD-driven anisotropic one — that's still untested end-to-end (needs the
Python side's `export_for_simmetrix.py` output, which needs `meshio`,
which isn't installed in this sandbox).

## Still open

- The VTU writer's vertex ordering for pyramid/wedge/hex regions uses
  Simmetrix's `R_vertices(region, 1)` ordering as-is; tet ordering is
  standard and now proven correct end-to-end above, but the other three
  haven't been exercised (the ramp test case is pure tet) or visually
  verified against VTK's expected node order in ParaView — worth checking
  on the first mixed-element mesh that goes through it.
- A real anisotropic size field (from the actual CFD/feature-extraction
  pipeline, not a synthetic isotropic one) hasn't been run through this
  yet.

## Directory layout this assumes

```
simmetrix/
  CMakeLists.txt
  run_on_scorec.sh
  translateToCas.py
  src/apply_aniso_size_field.cpp
  src/nanoflann.hpp
  build/apply_aniso_size_field   (built artifact, gitignored)
```
