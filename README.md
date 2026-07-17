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
to Simmetrix (`MSA_setAnisoVertexSize`) on the SCOREC remote machine.

See [`pipeline/PIPELINE.md`](pipeline/PIPELINE.md) for the full architecture,
file-by-file breakdown, open items, and how to run it once a real VTU is
available.
