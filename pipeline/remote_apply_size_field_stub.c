/* remote_apply_size_field_stub.c
 *
 * STUB — not runnable yet. Placeholder for the SCOREC-side program that
 * reads the .csv this pipeline exports and calls MSA_setAnisoVertexSize()
 * for every vertex, then drives Simmetrix's adaptation.
 *
 * TODO (pending your direction on SCOREC access / build setup):
 *   - confirm the exact Simmetrix headers/link libraries for this call
 *   - decide vertex correspondence strategy: row-index match (first pass
 *     on the solver's native mesh) vs. nearest-coordinate lookup (after
 *     at least one prior adaptation has renumbered vertices)
 *   - confirm how the adapted mesh gets written back out / SCP'd home
 *
 * Sketch of the loop once the above is settled:
 *
 *   FILE *f = fopen("size_field.csv", "r");
 *   char line[512];
 *   fgets(line, sizeof(line), f); // skip header
 *   while (fgets(line, sizeof(line), f)) {
 *       int vid; double x,y,z, m[9];
 *       sscanf(line, "%d,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf",
 *              &vid, &x,&y,&z, &m[0],&m[1],&m[2],&m[3],&m[4],&m[5],&m[6],&m[7],&m[8]);
 *
 *       pVertex v = lookup_vertex(vid, x, y, z);   // <-- correspondence strategy goes here
 *       double anisoSize[3][3] = {
 *           {m[0], m[1], m[2]},
 *           {m[3], m[4], m[5]},
 *           {m[6], m[7], m[8]},
 *       };
 *       MSA_setAnisoVertexSize(v, anisoSize);      // real Simmetrix call
 *   }
 *
 *   // then: run the adapter, write out the adapted mesh, done on SCOREC side
 */
