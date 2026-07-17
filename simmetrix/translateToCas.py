import simmetrix
import sys

if(len(sys.argv) < 5):
    print("Needs 5 args: translateToCas.py native_model_file model_file mesh_file out_cas_file_name")
else:
    nat_mod_file = sys.argv[1]
    mod_file = sys.argv[2]
    mesh_file = sys.argv[3]
    out_cas_file_name = sys.argv[4]


    native = simmetrix.SimParasolidNativeModel(nat_mod_file)
    model = simmetrix.SimGModel(mod_file, native)
    mesh = simmetrix.SimMesh(mesh_file, model)

    exporter = simmetrix.meshExporter("FLUENT")
    exporter.exportMesh(mesh, out_cas_file_name)