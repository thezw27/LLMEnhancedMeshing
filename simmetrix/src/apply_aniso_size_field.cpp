/*
 * apply_aniso_size_field.cpp
 *
 * Reads a text file of per-point anisotropic size matrices (one line each:
 * X Y Z m00 m01 m02 m10 m11 m12 m20 m21 m22 -- rows are direction vectors
 * already scaled by the desired size in that direction, the convention
 * MSA_setAnisoVertexSize expects and the same one size_field_builder.py
 * already produces), maps each point to the nearest vertex of an existing
 * Simmetrix mesh by coordinate (option (b) from export_for_simmetrix.py's
 * docstring -- valid regardless of whether the mesh has been adapted
 * before and renumbered), sets that vertex's anisotropic size, then runs
 * MeshSimAdapt and writes the adapted mesh out.
 *
 * Serial only (single process, single-threaded pMesh) -- no partitioned
 * mesh / MPI adaptation calls are made even though the Simmetrix build
 * links an MPI-based bootstrap library.
 *
 * Usage:
 *   apply_aniso_size_field <model.smd> <mesh.sms> <size_field.txt> <output_dir>
 */

#include "MeshSimAdapt.h"
#include "MeshSim.h"
#include "SimModel.h"
#include "SimUtil.h"
#include "SimInfo.h"

#include "nanoflann.hpp"

#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace std;
using namespace nanoflann;

void messageHandler(int type, const char *msg)
{
  switch (type) {
  case Sim_InfoMsg:
    cout << "Info: " << msg << endl;
    break;
  case Sim_DebugMsg:
    cout << "Debug: " << msg << endl;
    break;
  case Sim_WarningMsg:
    cout << "Warning: " << msg << endl;
    break;
  case Sim_ErrorMsg:
    cout << "Error: " << msg << endl;
    break;
  }
}

namespace {

// Point cloud wrapper for nanoflann.
struct PointCloud {
  std::vector<std::array<double, 3>> pts;

  inline size_t kdtree_get_point_count() const { return pts.size(); }
  inline double kdtree_get_pt(const size_t idx, const size_t dim) const { return pts[idx][dim]; }
  template <class BBOX> bool kdtree_get_bbox(BBOX &) const { return false; }
};

typedef KDTreeSingleIndexAdaptor<
    L2_Simple_Adaptor<double, PointCloud>,
    PointCloud,
    3 /* dimension */
    > kd_tree_t;

struct SizeFieldEntry {
  double xyz[3];
  double m[3][3];
};

// Parses "X Y Z m00 m01 m02 m10 m11 m12 m20 m21 m22" per line (whitespace or
// comma separated). Blank lines and lines starting with '#' are skipped.
vector<SizeFieldEntry> readSizeFieldFile(const string &path) {
  ifstream in(path);
  if (!in) throw std::runtime_error("could not open size field file: " + path);

  vector<SizeFieldEntry> entries;
  string line;
  int lineNo = 0;
  while (getline(in, line)) {
    lineNo++;
    for (char &c : line) if (c == ',') c = ' ';
    istringstream iss(line);
    string first;
    if (!(iss >> first)) continue;             // blank line
    if (first[0] == '#') continue;              // comment line
    iss.seekg(0);

    SizeFieldEntry e;
    bool ok = static_cast<bool>(
        iss >> e.xyz[0] >> e.xyz[1] >> e.xyz[2] >> e.m[0][0] >> e.m[0][1] >> e.m[0][2] >>
        e.m[1][0] >> e.m[1][1] >> e.m[1][2] >> e.m[2][0] >> e.m[2][1] >> e.m[2][2]);
    if (!ok) {
      cerr << "Warning: skipping malformed line " << lineNo << " in " << path << endl;
      continue;
    }
    entries.push_back(e);
  }
  return entries;
}

}  // namespace

int main(int argc, char **argv)
{
  if (argc != 5) {
    cerr << "usage: apply_aniso_size_field <model.smd> <mesh.sms> <size_field.txt> <output_dir>" << endl;
    return 1;
  }
  string modelFile = argv[1];
  string meshFile = argv[2];
  string sizeFieldFile = argv[3];
  fs::path outRoot(argv[4]);

  fs::path logDir = outRoot / "logs";
  fs::path meshDir = outRoot / "adapted_mesh";
  fs::create_directories(logDir);
  fs::create_directories(meshDir);

  try {
    Sim_logOn((logDir / "simmetrix.log").string().c_str());
    MS_init();  // must be called before Sim_readLicenseFile
    // NOTE: Sim_readLicenseFile() is for internal testing only. For a
    // release product, register keys individually with Sim_registerKey().
    Sim_readLicenseFile(nullptr);

    Sim_setMessageHandler(messageHandler);
    pProgress progress = Progress_new();
    Progress_setDefaultCallback(progress);

    cout << "Loading model " << modelFile << " and mesh " << meshFile << endl;
    pGModel model = GM_load(modelFile.c_str(), nullptr, progress);
    pMesh mesh = M_load(meshFile.c_str(), model, progress);

    cout << "Building nearest-vertex KD-tree..." << endl;
    vector<pVertex> vertices;
    PointCloud cloud;
    VIter viter = M_vertexIter(mesh);
    pVertex vertex;
    while ((vertex = VIter_next(viter))) {
      double xyz[3];
      V_coord(vertex, xyz);
      vertices.push_back(vertex);
      cloud.pts.push_back({xyz[0], xyz[1], xyz[2]});
    }
    VIter_delete(viter);
    if (vertices.empty()) throw std::runtime_error("mesh has no vertices");

    kd_tree_t tree(3, cloud, KDTreeSingleIndexAdaptorParams(10 /* max leaf */));
    tree.buildIndex();

    cout << "Reading size field file " << sizeFieldFile << endl;
    vector<SizeFieldEntry> entries = readSizeFieldFile(sizeFieldFile);
    cout << "  " << entries.size() << " size field entries read" << endl;

    pMSAdapt adapter = MSA_new(mesh, 1);  // method=1: size-based adaptation

    double minDist = numeric_limits<double>::infinity();
    double maxDist = 0.0;
    double sumDist = 0.0;
    for (const auto &e : entries) {
      unsigned int idx;
      double distSq;
      tree.knnSearch(e.xyz, 1, &idx, &distSq);
      double dist = std::sqrt(distSq);
      MSA_setAnisoVertexSize(adapter, vertices[idx], e.m);
      minDist = min(minDist, dist);
      maxDist = max(maxDist, dist);
      sumDist += dist;
    }
    if (!entries.empty()) {
      cout << "Nearest-vertex mapping distance: min=" << minDist << " max=" << maxDist
           << " mean=" << (sumDist / entries.size()) << endl;
    }

    cout << "Running MSA_adapt..." << endl;
    MSA_adapt(adapter, progress);
    MSA_delete(adapter);

    // VolLenRatio is the shape metric the docs call out as properly
    // supported on an anisotropic mesh; other metrics assume isotropic
    // elements and can give undesirable results here.
    cout << "Running VolumeMeshImprover (VolLenRatio target 0.3)..." << endl;
    pVolumeMeshImprover vmi = VolumeMeshImprover_new(mesh);
    VolumeMeshImprover_setShapeMetric(vmi, ShapeMetricType_VolLenRatio, 0.3);
    VolumeMeshImprover_execute(vmi, progress);
    VolumeMeshImprover_delete(vmi);

    fs::path adaptedMeshPath = meshDir / "adapted.sms";
    M_write(mesh, adaptedMeshPath.string().c_str(), 0, progress);
    cout << "Wrote adapted mesh to " << adaptedMeshPath << endl;

    M_release(mesh);
    GM_release(model);
    Progress_delete(progress);

    Sim_unregisterAllKeys();
    MS_exit();
    Sim_logOff();
  } catch (pSimInfo err) {
    cerr << "SimModSuite error caught:" << endl;
    cerr << "  Error code: " << SimInfo_code(err) << endl;
    cerr << "  Error string: " << SimInfo_toString(err) << endl;
    SimInfo_delete(err);
    return 1;
  } catch (const std::exception &e) {
    cerr << "Error: " << e.what() << endl;
    return 1;
  } catch (...) {
    cerr << "Unhandled exception caught" << endl;
    return 1;
  }
  return 0;
}
