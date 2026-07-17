#include <cstdio>
#include <exception>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include <apf.h>
#include <apfSIM.h>
#include <gmi.h>
#include <gmi_sim.h>
#include <lionPrint.h>
#include <PCU.h>
#include <pcu_util.h>

#include <SimAdvMeshing.h>
#include <SimPartitionedMesh.h>

int main (int argc, char* argv[]) {
  SimPartitionedMesh_start(&argc, &argv);
  try {
    pcu::PCU PCU;
    if (argc < 8) {
      if (PCU.Self() == 0) {
        std::cerr << "USAGE: " << argv[0]
          << " NAT_MODEL MODEL.smd MESH.sms SRC_FACE DST_FACE OUTMESH.sms THICKNESS"
          << std::endl;
      }
      throw 1;
    }
    const char* nat_model_file = argv[1];
    const char* model_file = argv[2];
    const char* mesh_file = argv[3];
    int srcFaceTag = std::stoi(argv[4]);
    int dstFaceTag = std::stoi(argv[5]);
    const char* out_mesh = argv[6];
    double thickness = std::stod(argv[7]);
    lion_set_verbosity(1);
    Sim_logOn("sim.log");
    MS_init();
    SimAdvMeshing_start();
    gmi_register_sim();
    gmi_sim_start();
    Sim_readLicenseFile(0);
    gmi_model* gmodel = gmi_sim_load(nat_model_file, model_file);
    auto model = gmi_export_sim(gmodel);
    auto sim_mesh = PM_load(mesh_file, model, nullptr);
    auto apf_mesh = apf::createMesh(sim_mesh, &PCU);
    auto srcFace = GM_faceByTag(model, srcFaceTag),
      dstFace = GM_faceByTag(model, dstFaceTag);
    if (srcFace == nullptr || dstFace == nullptr)
      throw std::runtime_error("bad face tag");
    printf("Past load\n");
    // Create new empty mesh on the same model.
    auto new_mesh = M_new(0, model);
    pACase cs = MS_newMeshCase(model);
    std::map<apf::MeshEntity*, int> vertIds;
    // Loop through apf_mesh for vertices in closure of srcFace.
    auto apfSrcFace = reinterpret_cast<apf::ModelEntity*>(srcFace);
    apf::MeshIterator* it = apf_mesh->begin(0);
    printf("starting iter\n");
    for (apf::MeshEntity* vtx; (vtx = apf_mesh->iterate(it));) {
      apf::ModelEntity* vtxME = apf_mesh->toModel(vtx);
      if (apf_mesh->getModelType(vtxME) > 2) continue;
      if (apf_mesh->isInClosureOf(vtxME, apfSrcFace)) {
        apf::Vector3 xyz, uv;
        apf_mesh->getPoint(vtx, 0, xyz);
        apf_mesh->getParam(vtx, uv);
        int tag = vertIds.size() + 1;
        printf("At specify for %lf %lf %lf with uv %lf %lf %lf tag %d pointer %p\n", xyz[0], xyz[1], xyz[2], uv[0], uv[1], uv[2], tag, vtxME);
        MS_specifyVertex(new_mesh, &xyz[0], &uv[0], (pGEntity)vtxME, tag);
        printf("Past specify\n");
        vertIds[vtx] = tag;
      }
    }
    apf_mesh->end(it);
    printf("Past specifyVert\n");
    // Specify mesh edges on model edges in the closure of srcFace.
    int edgeTag = 0;
    it = apf_mesh->begin(1);
    for (apf::MeshEntity* e; (e = apf_mesh->iterate(it));) {
      apf::ModelEntity* me = apf_mesh->toModel(e);
      if (
        apf_mesh->getModelType(me) == 1
        && apf_mesh->isInClosureOf(me, apfSrcFace)
      ) {
        apf::Downward dw;
        apf_mesh->getDownward(e, 0, dw);
        int dwTags[2];
        dwTags[0] = vertIds[dw[0]], dwTags[1] = vertIds[dw[1]];
        MS_specifyEdge(new_mesh, dwTags, (pGEntity)me, edgeTag);
        ++edgeTag;
      }
    }
    apf_mesh->end(it);
    // Loop through apf_mesh for triangles in closure of srcFace.
    int faceTag = 0;
    it = apf_mesh->begin(2);
    for (apf::MeshEntity* e; (e = apf_mesh->iterate(it));) {
      if (apf_mesh->toModel(e) == apfSrcFace) {
        apf::Downward dw;
        int nd = apf_mesh->getDownward(e, 0, dw);
        std::vector<int> dwTags(nd);
        for (int i = 0; i < nd; ++i) dwTags[i] = vertIds[dw[i]];
        MS_specifyFace(new_mesh, nd, &dwTags[0], srcFace, faceTag);
        ++faceTag;
      }
    }
    apf_mesh->end(it);
    printf("Past downward\n");
#ifndef NDEBUG
    M_write(new_mesh, "rasp_extrude_surf_mesh.sms", 0, 0);
#endif
    // Extrude a single mixed-element layer of the given thickness between
    // srcFace and dstFace.
    MS_setGeneralExtrusion(
      cs, srcFace, -1, dstFace, ExtrusionSizing_LayerSize,
      thickness, 0, 0, 0, 0, 0, 0,
      ExtrusionOptions_MixedElements
    );
    auto surfMesher = SurfaceMesher_new(cs,new_mesh);
    SurfaceMesher_execute(surfMesher, nullptr);
    SurfaceMesher_delete(surfMesher);
    auto volMesher = VolumeMesher_new(cs, new_mesh);
    VolumeMesher_execute(volMesher, nullptr);
    VolumeMesher_delete(volMesher);
    M_write(new_mesh, out_mesh, 0, 0);
    M_release(new_mesh);
    apf::destroyMesh(apf_mesh);
    M_release(sim_mesh);
    gmi_destroy(gmodel);
    SimAdvMeshing_stop();
    gmi_sim_stop();
    Sim_unregisterAllKeys();
    MS_exit();
    Sim_logOff();
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    SimPartitionedMesh_stop();
    return 1;
  }
  SimPartitionedMesh_stop();
  return 0;
}
