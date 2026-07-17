#!/usr/bin/env bash
# run_on_scorec.sh -- runs entirely on this machine. Meant to be invoked
# remotely (e.g. by another machine that ssh's in after already scp'ing the
# size field file here): loads the module set apply_aniso_size_field was
# built against, runs it to adapt the mesh and write the adapted VTU, then
# uses SimModelerScript (translateToCas.py) to export the adapted Fluent
# .cas -- a separate step because that export goes through Simmetrix's
# Python scripting API (a self-contained SimModeler distribution), not the
# MeshSimAdapt C++ API apply_aniso_size_field is built against.
#
# The SimModeler version loaded for that second step MUST match the
# SimModSuite version apply_aniso_size_field was built against
# (2026.0-260411 here) -- an older SimModelerScript (e.g. 2025.1) reading a
# mesh/model written by a newer SimModSuite is exactly the kind of
# version mismatch Simmetrix warns against.
#
# Usage:
#   run_on_scorec.sh <native_model_file> <model_smd> <mesh_sms> <size_field_txt> <output_dir>
#
# Writes into <output_dir>:
#   adapted.vtu               -- written by apply_aniso_size_field
#   adapted_mesh/adapted.sms  -- written by apply_aniso_size_field (raw Simmetrix mesh, reference)
#   logs/simmetrix.log        -- written by apply_aniso_size_field
#   adapted.cas               -- written by translateToCas.py (adapted Fluent case)
set -euo pipefail

if [ "$#" -ne 5 ]; then
  echo "usage: $0 <native_model_file> <model_smd> <mesh_sms> <size_field_txt> <output_dir>" >&2
  exit 1
fi

NATIVE_MODEL="$1"
MODEL_SMD="$2"
MESH_SMS="$3"
SIZE_FIELD_TXT="$4"
OUTPUT_DIR="$5"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APPLY_EXE="${APPLY_ANISO_EXE:-${SCRIPT_DIR}/build/apply_aniso_size_field}"
TRANSLATE_TO_CAS_PY="${SCRIPT_DIR}/translateToCas.py"

# Must match the module set apply_aniso_size_field was built/linked against
# (see acConfig26.sh) -- loaded here too so this also works from a bare,
# non-interactive ssh session that hasn't sourced any module setup itself.
module use /opt/scorec/spack/rhel9/v0222_2/lmod/linux-rhel9-x86_64/Core/
module load gcc/13.2.0-4eahhas
module load mpich/4.2.3-62uy3hd
module load cmake gsl
export SIM_MPI="mpich4.2.3"
module load simmetrix-simmodsuite/2026.0-260411-3dtgxhh

# SimModelerScript's own bundled distribution -- version must match the
# SimModSuite module above (2026.0-260411), not just whatever's newest.
module load simmetrix/simModeler/2026.0-260411

echo "==> running apply_aniso_size_field"
"${APPLY_EXE}" "${NATIVE_MODEL}" "${MODEL_SMD}" "${MESH_SMS}" "${SIZE_FIELD_TXT}" "${OUTPUT_DIR}"

ADAPTED_SMS="${OUTPUT_DIR}/adapted_mesh/adapted.sms"
if [ ! -f "${ADAPTED_SMS}" ]; then
  echo "ERROR: expected adapted mesh at ${ADAPTED_SMS}; apply_aniso_size_field did not produce it" >&2
  exit 1
fi

echo "==> exporting adapted Fluent case via SimModelerScript"
SimModelerScript -python "${TRANSLATE_TO_CAS_PY}" \
  "${NATIVE_MODEL}" "${MODEL_SMD}" "${ADAPTED_SMS}" "${OUTPUT_DIR}/adapted.cas"

echo "==> done."
echo "    ${OUTPUT_DIR}/adapted.vtu              -- adapted mesh for visualization"
echo "    ${OUTPUT_DIR}/adapted.cas              -- adapted Fluent case"
echo "    ${OUTPUT_DIR}/adapted_mesh/adapted.sms -- raw Simmetrix mesh, reference only"
