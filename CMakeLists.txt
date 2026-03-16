cmake_minimum_required(VERSION 3.9)
project(OptCuts)

set(OPTCUTS_EXTERNAL "${CMAKE_CURRENT_SOURCE_DIR}/ext")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# Option to build Python bindings (pybind11)
option(OPTCUTS_BUILD_PYTHON "Build Python bindings with pybind11" OFF)

if(OPTCUTS_BUILD_PYTHON)
  # Prefer Python 3.9 (pyenv) when available - set before find_package
  if(NOT Python3_EXECUTABLE)
    set(_pyenv_root "$ENV{HOME}/.pyenv")
    if(DEFINED ENV{PYENV_ROOT})
      set(_pyenv_root "$ENV{PYENV_ROOT}")
    endif()
    # Try pyenv 3.9.x (e.g. 3.9.9)
    file(GLOB _py39_versions "${_pyenv_root}/versions/3.9.*/bin/python3")
    if(_py39_versions)
      list(GET _py39_versions 0 _py39_path)
      set(Python3_EXECUTABLE "${_py39_path}" CACHE STRING "Python 3.9 for optcuts")
      message(STATUS "Using Python 3.9: ${Python3_EXECUTABLE}")
    endif()
  endif()
  find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
  set(PYBIND11_PYTHON_VERSION ${Python3_VERSION})
endif()

include(${PROJECT_NAME}Dependencies)

# libigl
option(LIBIGL_USE_STATIC_LIBRARY "Use libigl as static library" OFF)
option(LIBIGL_WITH_ANTTWEAKBAR       "Use AntTweakBar"    OFF)
option(LIBIGL_WITH_CGAL              "Use CGAL"           OFF)
option(LIBIGL_WITH_COMISO            "Use CoMiso"         OFF)
option(LIBIGL_WITH_CORK              "Use Cork"           OFF)
option(LIBIGL_WITH_EMBREE            "Use Embree"         OFF)
option(LIBIGL_WITH_LIM               "Use LIM"            OFF)
option(LIBIGL_WITH_MATLAB            "Use Matlab"         OFF)
option(LIBIGL_WITH_MOSEK             "Use MOSEK"          OFF)
option(LIBIGL_WITH_OPENGL            "Use OpenGL"         ON)
option(LIBIGL_WITH_OPENGL_GLFW       "Use GLFW"           ON)
option(LIBIGL_WITH_OPENGL_GLFW_IMGUI "Use ImGui"          OFF)
option(LIBIGL_WITH_PNG               "Use PNG"            ON)
option(LIBIGL_WITH_PYTHON            "Use Python"         OFF)
option(LIBIGL_WITH_TETGEN            "Use Tetgen"         OFF)
option(LIBIGL_WITH_TRIANGLE          "Use Triangle"       ON)
option(LIBIGL_WITH_VIEWER            "Use OpenGL viewer"  ON)
option(LIBIGL_WITH_XML               "Use XML"            OFF)

find_package(LIBIGL REQUIRED QUIET)


# Intel TBB
#include(ext/tbb/cmake/TBBGet.cmake)
#tbb_get(TBB_ROOT tbb_root CONFIG_DIR TBB_DIR)
#find_package(TBB REQUIRED tbb)


file(GLOB SRCFILES
  src/main.cpp
  src/Optimizer.cpp
  src/TriMesh.cpp
  src/Scaffold.cpp

  src/Energy/Energy.cpp
  src/Energy/ARAPEnergy.cpp
  src/Energy/SymDirichletEnergy.cpp
  src/Energy/SigmaBoundEnergy.cpp
  src/Energy/SigmaSmoothEnergy.cpp

  src/Utils/IglUtils.cpp
  
  src/LinSysSolver/EigenLibSolver.cpp
)

include_directories(
  src
  src/Energy
  src/Utils
  src/LinSysSolver
  src/UnitTest
)

add_executable(${PROJECT_NAME}_bin ${SRCFILES})

target_link_libraries(${PROJECT_NAME}_bin igl::core igl::opengl_glfw igl::triangle igl::png)

if(TARGET TBB::tbb)
  target_link_libraries(${PROJECT_NAME}_bin TBB::tbb)
endif()

# Python bindings (optional): link main.cpp so all main.cpp functions/globals are available; use OPTCUTS_PYTHON to exclude int main().
if(OPTCUTS_BUILD_PYTHON)
  pybind11_add_module(optcuts
    src/python/optcuts_bindings.cpp
    src/main.cpp
    src/Optimizer.cpp
    src/TriMesh.cpp
    src/Scaffold.cpp
    src/Energy/Energy.cpp
    src/Energy/SymDirichletEnergy.cpp
    src/Energy/SigmaBoundEnergy.cpp
    src/Energy/SigmaSmoothEnergy.cpp
    src/Utils/IglUtils.cpp
    src/LinSysSolver/EigenLibSolver.cpp
  )
  target_compile_definitions(optcuts PRIVATE OPTCUTS_PYTHON)
  target_include_directories(optcuts PRIVATE
    src
    src/Energy
    src/Utils
    src/LinSysSolver
    src/UnitTest
  )
  set_target_properties(optcuts PROPERTIES CXX_STANDARD 14)
  target_link_libraries(optcuts PRIVATE
    igl::core
    igl::opengl_glfw
    igl::triangle
    igl::png
  )
  if(TARGET TBB::tbb)
    target_link_libraries(optcuts PRIVATE TBB::tbb)
  endif()
endif()