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
 *   apply_aniso_size_field <native_model> <model.smd> <mesh.sms> <size_field.txt> <output_dir>
 *
 * native_model is the CAD kernel's native geometry file (e.g. a Parasolid
 * .x_t) -- required here (not optional) because GM_load() on a real
 * Parasolid-backed .smd fails outright without it ("Unable to read
 * resource of type model.nonmanifold.parasolid"), confirmed against a real
 * test case. Loaded the same way exAnisoCyl_Parasolid.cc does: bracket
 * geometry access with SimParasolid_start()/_stop(), load the native model
 * via ParasolidNM_createFromFile(), pass it into GM_load().
 *
 * Writes <output_dir>/adapted.vtu (written directly by this program, not
 * through Simmetrix's meshExporter -- no VTU-format plugin was available
 * alongside it) and <output_dir>/adapted_mesh/adapted.sms. Fluent .cas
 * export is a separate step (see translateToCas.py) since that goes
 * through SimModelerScript's Python API, not this C++ program.
 */

#include "MeshSimAdapt.h"
#include "MeshSim.h"
#include "SimModel.h"
#include "SimUtil.h"
#include "SimInfo.h"
#include "SimParasolidKrnl.h"

#include "nanoflann.hpp"

#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>
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

// Writes the mesh currently in `mesh` out as a legacy-compatible XML VTU
// (ASCII UnstructuredGrid). Simmetrix's meshExporter plugins cover FLUENT,
// NASTRAN, Exodus, etc. but not VTU, so this walks the mesh directly via
// the MeshSim region/vertex API instead of shelling out to a converter.
// Only the standard solid element topologies are supported (matching what
// vtu_io.py reads on the Python side); any other region type is skipped
// with a warning rather than aborting the whole export.
void writeAdaptedMeshVTU(pMesh mesh, const string &path) {
  unordered_map<pVertex, int> vertexIndex;
  vector<array<double, 3>> points;

  VIter viter = M_vertexIter(mesh);
  pVertex vertex;
  while ((vertex = VIter_next(viter))) {
    array<double, 3> xyz;
    V_coord(vertex, xyz.data());
    vertexIndex[vertex] = static_cast<int>(points.size());
    points.push_back(xyz);
  }
  VIter_delete(viter);

  vector<int> connectivity, offsets, types;
  int runningOffset = 0;
  int skippedRegions = 0;

  RIter riter = M_regionIter(mesh);
  pRegion region;
  while ((region = RIter_next(riter))) {
    int vtkType = 0, numVerts = 0;
    switch (R_topoType(region)) {
    case Rtet:     vtkType = 10; numVerts = 4; break;  // VTK_TETRA
    case Rpyramid: vtkType = 14; numVerts = 5; break;  // VTK_PYRAMID
    case Rwedge:   vtkType = 13; numVerts = 6; break;  // VTK_WEDGE
    case Rhex:     vtkType = 12; numVerts = 8; break;  // VTK_HEXAHEDRON
    default:
      skippedRegions++;
      continue;
    }

    pPList regionVerts = R_vertices(region, 1);
    if (PList_size(regionVerts) != numVerts) {
      PList_delete(regionVerts);
      skippedRegions++;
      continue;
    }
    for (int i = 0; i < numVerts; i++) {
      pVertex v = static_cast<pVertex>(PList_item(regionVerts, i));
      connectivity.push_back(vertexIndex.at(v));
    }
    PList_delete(regionVerts);

    runningOffset += numVerts;
    offsets.push_back(runningOffset);
    types.push_back(vtkType);
  }
  RIter_delete(riter);

  if (skippedRegions > 0) {
    cerr << "Warning: skipped " << skippedRegions
         << " mesh region(s) with an unsupported topology while writing VTU" << endl;
  }

  // No 3D regions at all means this is a surface-only (2D) mesh, not a
  // volume mesh missing all its regions -- fall back to writing its faces
  // as the cells. (Only done when regions are entirely absent: a real
  // volume mesh's faces include interior faces shared between two
  // regions, which would double up the geometry if written as cells too.)
  if (types.empty()) {
    int skippedFaces = 0;
    FIter fiter = M_faceIter(mesh);
    pFace face;
    while ((face = FIter_next(fiter))) {
      int vtkType = 0, numVerts = F_numEdges(face);
      switch (numVerts) {
      case 3: vtkType = 5; break;  // VTK_TRIANGLE
      case 4: vtkType = 9; break;  // VTK_QUAD
      default:
        skippedFaces++;
        continue;
      }

      pPList faceVerts = F_vertices(face, 1);
      if (PList_size(faceVerts) != numVerts) {
        PList_delete(faceVerts);
        skippedFaces++;
        continue;
      }
      for (int i = 0; i < numVerts; i++) {
        pVertex v = static_cast<pVertex>(PList_item(faceVerts, i));
        connectivity.push_back(vertexIndex.at(v));
      }
      PList_delete(faceVerts);

      runningOffset += numVerts;
      offsets.push_back(runningOffset);
      types.push_back(vtkType);
    }
    FIter_delete(fiter);

    if (skippedFaces > 0) {
      cerr << "Warning: skipped " << skippedFaces
           << " mesh face(s) with an unsupported topology while writing VTU" << endl;
    }
  }

  ofstream out(path);
  if (!out) throw std::runtime_error("could not open VTU output file: " + path);
  out << "<?xml version=\"1.0\"?>\n"
      << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n"
      << "<UnstructuredGrid>\n"
      << "<Piece NumberOfPoints=\"" << points.size() << "\" NumberOfCells=\"" << types.size() << "\">\n"
      << "<Points><DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n";
  for (const auto &p : points) out << p[0] << " " << p[1] << " " << p[2] << "\n";
  out << "</DataArray></Points>\n"
      << "<Cells>\n"
      << "<DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n";
  for (int c : connectivity) out << c << " ";
  out << "\n</DataArray>\n"
      << "<DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n";
  for (int o : offsets) out << o << " ";
  out << "\n</DataArray>\n"
      << "<DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n";
  for (int t : types) out << t << " ";
  out << "\n</DataArray>\n"
      << "</Cells>\n"
      << "</Piece>\n</UnstructuredGrid>\n</VTKFile>\n";
}

}  // namespace

int main(int argc, char **argv)
{
  if (argc != 6) {
    cerr << "usage: apply_aniso_size_field <native_model> <model.smd> <mesh.sms> <size_field.txt> <output_dir>" << endl;
    return 1;
  }
  string nativeModelFile = argv[1];
  string modelFile = argv[2];
  string meshFile = argv[3];
  string sizeFieldFile = argv[4];
  fs::path outRoot(argv[5]);

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
    SimParasolid_start(1);

    Sim_setMessageHandler(messageHandler);
    pProgress progress = Progress_new();
    Progress_setDefaultCallback(progress);

    cout << "Loading native model " << nativeModelFile << ", model " << modelFile
         << ", and mesh " << meshFile << endl;
    pNativeModel nmodel = ParasolidNM_createFromFile(nativeModelFile.c_str(), 0);
    pGModel model = GM_load(modelFile.c_str(), nmodel, progress);
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

    fs::path adaptedVtuPath = outRoot / "adapted.vtu";
    writeAdaptedMeshVTU(mesh, adaptedVtuPath.string());
    cout << "Wrote adapted VTU to " << adaptedVtuPath << endl;

    M_release(mesh);
    GM_release(model);
    NM_release(nmodel);
    Progress_delete(progress);

    SimParasolid_stop(1);
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
