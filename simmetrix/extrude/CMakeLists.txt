cmake_minimum_required(VERSION 3.14)
project(Extrude CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# extrude.cc links against the SCOREC `core` (PUMI: apf/gmi/etc, via the
# Sim-bridging apf_sim/gmi_sim libs). This project's .smd model files are
# written by SimModeler 2026 -- the original prebuilt `core` at
# /users/.../core/build was configured against SIM_MPI=mpich4.1.1 and
# SimModSuite 2025.1-250602dev (see core/config.sh) and genuinely cannot
# read them (confirmed: "Unable to read smd part to create resource of
# type model", reproduced with a bare GM_load(), independent of any
# extrude.cc logic). So this points at a *separate* clone,
# /users/.../core-2026 (branch `develop`, which merges PR #534 "support
# simModSuite 2026" -- confirmed via `libSimParasolid381.a` in that PR),
# built against the same 2026.0-260411/mpich4.2.3 toolchain as the rest of
# simmetrix/. The original core/ checkout is untouched (still on master,
# still built against 2025.1) -- do not repoint this at it.
#
# Configure with the same module set as ../CMakeLists.txt / acConfig26.sh:
#   module use /opt/scorec/spack/rhel9/v0222_2/lmod/linux-rhel9-x86_64/Core/
#   module load gcc/13.2.0-4eahhas mpich/4.2.3-62uy3hd cmake gsl
#   export SIM_MPI="mpich4.2.3"
#   module load simmetrix-simmodsuite/2026.0-260411-3dtgxhh

set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};/users/gordoz2/lore.scorec.rpi.edu/adaptiveController/cmake/Modules/")

if(DEFINED ENV{SIMMETRIX_SIMMODSUITE_ROOT})
  set(SIMMETRIX_INCLUDE_DIR "$ENV{SIMMETRIX_SIMMODSUITE_ROOT}/include" CACHE PATH "Simmetrix SimModSuite include directory")
  set(SIMMETRIX_LIB_DIR "$ENV{SIMMETRIX_SIMMODSUITE_ROOT}/lib/x64_rhel8_gcc83" CACHE PATH "Simmetrix SimModSuite library directory")
endif()
set(SIM_PARASOLID ON CACHE BOOL "Use Parasolid through Simmetrix" FORCE)

find_package(SimModSuite MODULE REQUIRED)
find_package(MPI REQUIRED)

set(COREDIR /users/gordoz2/lore.scorec.rpi.edu/core-2026)
set(BUILDDIR /users/gordoz2/lore.scorec.rpi.edu/core-2026/build)

function(add_core_lib name path)
  add_library(${name} SHARED IMPORTED)
  set_target_properties(${name} PROPERTIES IMPORTED_LOCATION ${path})
endfunction()

foreach(lib apf apf_sim crv gmi gmi_sim lion ma mds mth parma pcu pumi ree sam spr)
  add_core_lib(${lib} ${BUILDDIR}/${lib}/lib${lib}.so)
endforeach()

add_executable(extrude ../src/extrude.cc)

target_include_directories(extrude PRIVATE
  ${SIMMODSUITE_INCLUDE_DIR}
  ${COREDIR} ${COREDIR}/apf ${COREDIR}/apf_sim ${COREDIR}/can ${COREDIR}/crv
  ${COREDIR}/dsp ${COREDIR}/gmi ${COREDIR}/gmi_sim ${COREDIR}/lion
  ${COREDIR}/ma ${COREDIR}/mds ${COREDIR}/mth ${COREDIR}/omega_h
  ${COREDIR}/parma ${COREDIR}/pcu ${COREDIR}/phasta ${COREDIR}/pumi
  ${COREDIR}/ree ${COREDIR}/sam ${COREDIR}/spr ${COREDIR}/stk ${COREDIR}/zoltan
  ${BUILDDIR}
)

# Order matters for the PUMI stack (each lib depends on the ones after it).
target_link_libraries(extrude PRIVATE
  MPI::MPI_CXX
  apf apf_sim crv gmi gmi_sim lion ma mds mth parma pcu pumi ree sam spr
  ${SIMMODSUITE_LIBS}
)

set_target_properties(extrude PROPERTIES
  BUILD_RPATH "${BUILDDIR}/apf;${BUILDDIR}/apf_sim;${BUILDDIR}/crv;${BUILDDIR}/gmi;${BUILDDIR}/gmi_sim;${BUILDDIR}/lion;${BUILDDIR}/ma;${BUILDDIR}/mds;${BUILDDIR}/mth;${BUILDDIR}/parma;${BUILDDIR}/pcu;${BUILDDIR}/pumi;${BUILDDIR}/ree;${BUILDDIR}/sam;${BUILDDIR}/spr"
)
