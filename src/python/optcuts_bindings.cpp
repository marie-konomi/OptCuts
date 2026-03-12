//
//  optcuts_bindings.cpp
//  OptCuts Python bindings: expose main.cpp entry and all main.cpp functions.
//  Build with main.cpp (OPTCUTS_PYTHON) so run_main and others call the real implementation.
//

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "Types.hpp"
#include "TriMesh.hpp"
#include "Optimizer.hpp"
#include "Energy/Energy.hpp"
#include "Energy/SymDirichletEnergy.hpp"
#include "Energy/SigmaBoundEnergy.hpp"
#include "Energy/SigmaSmoothEnergy.hpp"
#include <igl/opengl/glfw/Viewer.h>

#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

// ----- main.cpp entry and API (defined in main.cpp when linked) -----
extern int run_optcuts_main(int argc, char* argv[]);

extern void proceedOptimization(int proceedNum);
extern void updateViewerData(void);
extern void saveInfo(bool writePNG, bool writeGIF, bool writeMesh);
extern void saveInfoForPresent(const std::string fileName);
extern void toggleOptimization(void);
extern double updateLambda(double measure_bound, double lambda_SD, double kappa, double kappa2);
extern bool updateLambda_stationaryV(bool cancelMomentum, bool checkConvergence);
extern void converge_preDrawFunc(igl::opengl::glfw::Viewer& viewer);

// Globals from main.cpp (read/write as needed)
extern std::string outputFolderPath;
extern int iterNum;
extern int converged;
extern bool optimization_on;
extern bool outerLoopFinished;
extern OptCuts::Optimizer* optimizer;
extern std::vector<const OptCuts::TriMesh*> triSoup;

// Viewer is used by converge_preDrawFunc; we pass a reference from global.
extern igl::opengl::glfw::Viewer viewer;

// Result saved at end of run_optcuts_main (when OPTCUTS_PYTHON) for get_result_mesh after run returns.
#ifdef OPTCUTS_PYTHON
extern Eigen::MatrixXd result_V_after_run;
extern Eigen::MatrixXi result_F_after_run;
extern int iterNum_after_run;
#endif

PYBIND11_MODULE(optcuts, m) {
    m.doc() = "OptCuts: main.cpp entry and all optimization/API functions callable from Python";

    // ----- TriMesh -----
    py::class_<OptCuts::TriMesh>(m, "TriMesh")
        .def(py::init<>())
        .def(py::init([](const Eigen::MatrixXd& V_mesh,
                         const Eigen::MatrixXi& F_mesh,
                         const Eigen::MatrixXd& UV_mesh,
                         const Eigen::MatrixXi& FUV_mesh,
                         bool separateTri,
                         double initSeamLen,
                         double areaThres_AM) {
            Eigen::MatrixXi FUV = FUV_mesh;
            if (FUV.size() == 0 && UV_mesh.rows() > 0) FUV = F_mesh;
            return new OptCuts::TriMesh(V_mesh, F_mesh, UV_mesh, FUV,
                                        separateTri, initSeamLen, areaThres_AM);
        }),
             py::arg("V"), py::arg("F"), py::arg("UV"), py::arg("FUV") = Eigen::MatrixXi(),
             py::arg("separate_tri") = true, py::arg("init_seam_len") = 0.0, py::arg("area_thres_AM") = 0.0)
        .def("compute_features", &OptCuts::TriMesh::computeFeatures,
             py::arg("multi_comp") = false, py::arg("reset_fixed_v") = false)
        .def("save", [](OptCuts::TriMesh& self, const std::string& path) { self.save(path); }, py::arg("file_path"))
        .def("save_as_mesh", [](OptCuts::TriMesh& self, const std::string& path, bool scale_uv) {
            self.saveAsMesh(path, scale_uv);
        }, py::arg("file_path"), py::arg("scale_uv") = false)
        .def_readonly("V", &OptCuts::TriMesh::V)
        .def_readonly("V_rest", &OptCuts::TriMesh::V_rest)
        .def_readonly("F", &OptCuts::TriMesh::F)
        .def_readonly("cohE", &OptCuts::TriMesh::cohE);

    py::enum_<OptCuts::MethodType>(m, "MethodType")
        .value("OPTCUTS", OptCuts::MT_OPTCUTS)
        .value("EBCUTS", OptCuts::MT_EBCUTS)
        .value("OPTCUTS_NODUAL", OptCuts::MT_OPTCUTS_NODUAL)
        .value("DISTMIN", OptCuts::MT_DISTMIN)
        .export_values();

    // ----- Optimizer -----
    py::class_<OptCuts::Optimizer>(m, "Optimizer")
        .def(py::init([](const OptCuts::TriMesh& data0,
                         const std::vector<OptCuts::Energy*>& energyTerms,
                         const std::vector<double>& energyParams,
                         int propagateFracture, bool mute, bool scaffolding,
                         const Eigen::MatrixXd& UV_bnds, const Eigen::MatrixXi& E, const Eigen::VectorXi& bnd,
                         bool useDense) {
            return new OptCuts::Optimizer(data0, energyTerms, energyParams,
                                          propagateFracture, mute, scaffolding, UV_bnds, E, bnd, useDense);
        }),
             py::arg("data0"), py::arg("energy_terms"), py::arg("energy_params"),
             py::arg("propagate_fracture") = 1, py::arg("mute") = false, py::arg("scaffolding") = true,
             py::arg("UV_bnds") = Eigen::MatrixXd(), py::arg("E") = Eigen::MatrixXi(), py::arg("bnd") = Eigen::VectorXi(),
             py::arg("use_dense") = false)
        .def("precompute", &OptCuts::Optimizer::precompute)
        .def("solve", &OptCuts::Optimizer::solve, py::arg("max_iter") = 100)
        .def("get_result", &OptCuts::Optimizer::getResult, py::return_value_policy::reference)
        .def("get_last_energy_val", &OptCuts::Optimizer::getLastEnergyVal, py::arg("exclude_scaffold") = false)
        .def("create_fracture", [](OptCuts::Optimizer& self, double stress_thres, int prop_type, bool allow_prop, bool allow_in_split) {
            return self.createFracture(stress_thres, prop_type, allow_prop, allow_in_split);
        }, py::arg("stress_thres"), py::arg("prop_type") = 0, py::arg("allow_propagate") = true, py::arg("allow_in_split") = false);

    py::class_<OptCuts::Energy>(m, "Energy");
    py::class_<OptCuts::SymDirichletEnergy, OptCuts::Energy>(m, "SymDirichletEnergy").def(py::init<>());
    py::class_<OptCuts::SigmaSmoothEnergy, OptCuts::Energy>(m, "SigmaSmoothEnergy").def(py::init<>());
    py::class_<OptCuts::SigmaBoundEnergy, OptCuts::Energy>(m, "SigmaBoundEnergy")
        .def(py::init<double, double, double, double>(),
             py::arg("sigma1_max"), py::arg("sigma2_max"), py::arg("sigma1_min") = 0.0, py::arg("sigma2_min") = 0.0);

    // ----- main() equivalent: argv same as OptCuts_bin (argv[1]=progMode: 0=normal, 10=offline, 100=headless) -----
    m.def("run_main", [](py::list args) {
        std::vector<std::string> argv_str;
        for (auto& item : args)
            argv_str.push_back(item.cast<std::string>());
        std::vector<char*> argv_c;
        for (auto& s : argv_str)
            argv_c.push_back(const_cast<char*>(s.c_str()));
        return run_optcuts_main(static_cast<int>(argv_c.size()), argv_c.data());
    }, py::arg("argv"),
    "Run main.cpp entry. argv[0]=prog name, argv[1]=progMode (0/10/100), argv[2]=mesh path, ... Same as OptCuts_bin.");

    // ----- main.cpp functions exposed 1:1 -----
    m.def("proceed_optimization", &proceedOptimization, py::arg("proceed_num") = 1,
          "Run proceedNum steps of solve(1). Same as main proceedOptimization.");
    m.def("update_viewer_data", &updateViewerData, "Update viewer mesh/UV state. Same as main updateViewerData.");
    m.def("save_info", &saveInfo,
          py::arg("write_png") = true, py::arg("write_gif") = true, py::arg("write_mesh") = true,
          "Save screenshot and optional mesh. Same as main saveInfo.");
    m.def("save_info_for_present", &saveInfoForPresent, py::arg("file_name") = "info.txt",
          "Write timing/distortion info file. Same as main saveInfoForPresent.");
    m.def("toggle_optimization", &toggleOptimization, "Toggle optimization_on. Same as main toggleOptimization.");
    m.def("update_lambda", &updateLambda,
          py::arg("measure_bound"), py::arg("lambda_sd"), py::arg("kappa") = 1.0, py::arg("kappa2") = 1.0,
          "Same as main updateLambda.");
    m.def("update_lambda_stationary_v", &updateLambda_stationaryV,
          py::arg("cancel_momentum") = true, py::arg("check_convergence") = false,
          "Same as main updateLambda_stationaryV.");
    m.def("converge_pre_draw_func", [](void) { converge_preDrawFunc(viewer); },
          "Final converge step (sets optimization_on=false, outerLoopFinished=true). Same as main converge_preDrawFunc.");

    // ----- Globals: getters / setter -----
    m.def("get_iter_num", []() { return iterNum; });
    m.def("get_converged", []() { return converged; });
    m.def("get_optimization_on", []() { return optimization_on; });
    m.def("get_outer_loop_finished", []() { return outerLoopFinished; });
    m.def("set_output_folder", [](const std::string& path) { outputFolderPath = path; }, py::arg("path"));
    m.def("get_output_folder", []() { return outputFolderPath; });

    // ----- Result: from global optimizer while running, or from saved copy after run_main returns -----
    m.def("get_result_mesh", []() {
#ifdef OPTCUTS_PYTHON
        if (result_V_after_run.size() > 0)
            return py::make_tuple(result_V_after_run, result_F_after_run);
#endif
        if (optimizer == nullptr || triSoup.empty())
            throw std::runtime_error("get_result_mesh: no result (run run_main first, or run just finished)");
        const OptCuts::TriMesh& r = optimizer->getResult();
        return py::make_tuple(r.V, r.F);
    }, "Return (V, F) of result. After run_main() use saved copy; during step-by-step use current optimizer.");

    // ----- Standalone optimize_uv (no main.cpp state) -----
    m.def("optimize_uv", [](
            const Eigen::MatrixXd& V,
            const Eigen::MatrixXi& F,
            const Eigen::MatrixXd& UV_init,
            int max_iter, double lambda_init, bool scaffolding,
            double sigma1_bound, double sigma2_bound) {
        Eigen::MatrixXd V_cpy = V, UV_init_cpy = UV_init;
        Eigen::MatrixXi F_cpy = F;
        OptCuts::TriMesh triMesh(V_cpy, F_cpy, UV_init_cpy, F_cpy, false);
        std::vector<OptCuts::Energy*> energyTerms;
        std::vector<double> energyParams;
        energyParams.push_back(1.0 - lambda_init);
        energyTerms.push_back(new OptCuts::SymDirichletEnergy());
        energyParams.push_back(1.0 - lambda_init);
        energyTerms.push_back(new OptCuts::SigmaSmoothEnergy());
        if (sigma1_bound > 0.0 || sigma2_bound > 0.0) {
            energyParams.push_back(1.0 - lambda_init);
            energyTerms.push_back(new OptCuts::SigmaBoundEnergy(sigma1_bound, sigma2_bound));
        }
        OptCuts::Optimizer optimizer(triMesh, energyTerms, energyParams, 0, true, scaffolding);
        optimizer.precompute();
        int iter = optimizer.solve(max_iter);
        Eigen::MatrixXd UV_result = optimizer.getResult().V;
        Eigen::MatrixXi F_result = optimizer.getResult().F;
        for (auto* e : energyTerms) delete e;
        return py::make_tuple(UV_result, F_result, iter);
    },
    py::arg("V"), py::arg("F"), py::arg("UV_init"),
    py::arg("max_iter") = 100, py::arg("lambda_init") = 0.5, py::arg("scaffolding") = true,
    py::arg("sigma1_bound") = 0.0, py::arg("sigma2_bound") = 0.0,
    "Standalone UV optimization (does not use main.cpp state). Returns (UV_result, F_result, n_iter).");

    // ----- run_from_files: convenience wrapper that builds argv and calls run_main (headless) -----
    m.def("run_from_files", [](const std::string& mesh_path, const std::string& output_path,
            int max_iter, double lambda_init, bool scaffolding,
            double sigma1_bound, double sigma2_bound, int max_topo_iters,
            double upper_bound, double frac_thres, bool topo_line_search) {
        if (upper_bound <= 4.0) upper_bound = 4.1;
        size_t last = output_path.find_last_of("/\\");
        outputFolderPath = (last == std::string::npos) ? "./" : output_path.substr(0, last + 1);
        std::vector<std::string> argv_str = {"optcuts", "100", mesh_path};
        argv_str.push_back(std::to_string(lambda_init));
        argv_str.push_back("1.0");
        argv_str.push_back("0");
        argv_str.push_back(std::to_string(upper_bound));
        argv_str.push_back(scaffolding ? "1" : "0");
        argv_str.push_back(std::to_string(sigma1_bound));
        argv_str.push_back(std::to_string(sigma2_bound));
        std::vector<char*> argv_c;
        for (auto& s : argv_str) argv_c.push_back(const_cast<char*>(s.c_str()));
        int ret = run_optcuts_main(static_cast<int>(argv_c.size()), argv_c.data());
        if (ret != 0) throw std::runtime_error("run_from_files: run_optcuts_main returned " + std::to_string(ret));
#ifdef OPTCUTS_PYTHON
        return py::make_tuple(result_V_after_run, result_F_after_run, iterNum_after_run);
#else
        if (optimizer == nullptr || triSoup.empty())
            throw std::runtime_error("run_from_files: no result after run");
        const OptCuts::TriMesh& r = optimizer->getResult();
        return py::make_tuple(r.V, r.F, iterNum);
#endif
    },
    py::arg("mesh_path"), py::arg("output_path"),
    py::arg("max_iter") = 100, py::arg("lambda_init") = 0.999, py::arg("scaffolding") = true,
    py::arg("sigma1_bound") = 0.0, py::arg("sigma2_bound") = 0.0,
    py::arg("max_topo_iters") = 20, py::arg("upper_bound") = 4.1, py::arg("frac_thres") = 0.0, py::arg("topo_line_search") = true,
    "Convenience: set output folder, run main in headless mode (progMode=100), return (UV, F, iter_num). Output files go to output_path.");
}
