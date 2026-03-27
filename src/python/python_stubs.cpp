//
//  python_stubs.cpp
//  Stub definitions for globals used by OptCuts when built as Python module.
//  (These are normally defined in main.cpp)
//

#include "Types.hpp"
#include "Timer.hpp"
#include <fstream>
#include <vector>
#include <Eigen/Eigen>

#ifdef _WIN32
#define NULL_DEVICE "nul"
#else
#define NULL_DEVICE "/dev/null"
#endif

// Dummy log file (writes to null device)
std::ofstream logFile(NULL_DEVICE);

// Timers: timer needs 4 activities (0-3), timer_step needs 9 (0-8)
Timer timer(4);
Timer timer_step(9);

// Optimizer globals (extern for external linkage - const has internal linkage by default in C++)
OptCuts::MethodType methodType = OptCuts::MT_OPTCUTS;
extern const std::string outputFolderPath = "./";
extern const bool fractureMode = false;

// TriMesh topology optimization globals (written by TriMesh, read by Optimizer)
std::vector<std::pair<double, double>> energyChanges_bSplit, energyChanges_iSplit, energyChanges_merge;
std::vector<std::vector<int>> paths_bSplit, paths_iSplit, paths_merge;
std::vector<Eigen::MatrixXd> newVertPoses_bSplit, newVertPoses_iSplit, newVertPoses_merge;
double filterExp_in = 0.6;
int inSplitTotalAmt = 0;
