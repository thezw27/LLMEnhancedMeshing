cmake_minimum_required(VERSION 3.14)
project(ApplyAnisoSizeField CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};/users/gordoz2/lore.scorec.rpi.edu/adaptiveController/cmake/Modules/")

# Bridge the simmetrix-simmodsuite module's SIMMETRIX_SIMMODSUITE_ROOT env var
# into the FindSimModSuite.cmake inputs (it looks for SIMMETRIX_INCLUDE_DIR /
# SIMMETRIX_LIB_DIR, not the module's own env var name). x64_rhel8_gcc83 is
# hardcoded because that's the arch dir the loaded module's LD_LIBRARY_PATH
# points at and the one with a SimPartitionWrapper build matching SIM_MPI.
if(DEFINED ENV{SIMMETRIX_SIMMODSUITE_ROOT})
  set(SIMMETRIX_INCLUDE_DIR "$ENV{SIMMETRIX_SIMMODSUITE_ROOT}/include" CACHE PATH "Simmetrix SimModSuite include directory")
  set(SIMMETRIX_LIB_DIR "$ENV{SIMMETRIX_SIMMODSUITE_ROOT}/lib/x64_rhel8_gcc83" CACHE PATH "Simmetrix SimModSuite library directory")
endif()

find_package(SimModSuite MODULE REQUIRED)
find_package(MPI REQUIRED)

add_executable(apply_aniso_size_field src/apply_aniso_size_field.cpp)
target_include_directories(apply_aniso_size_field PRIVATE ${SIMMODSUITE_INCLUDE_DIR})
target_link_libraries(apply_aniso_size_field PRIVATE ${SIMMODSUITE_LIBS} MPI::MPI_CXX)
