//
//  SigmaSmoothEnergy.cpp
//  OptCuts
//
//  E_smooth = Σ_edges w_ij * ( (σ1_i - σ1_j)² + (σ2_i - σ2_j)² )
//

#include "SigmaSmoothEnergy.hpp"
#include "SigmaBoundEnergy.hpp"
#include <Eigen/SVD>
#include <Eigen/Sparse>
#include <vector>
#include <set>
#include <tuple>
#include <cmath>
#include <iostream>

namespace OptCuts {

    SigmaSmoothEnergy::SigmaSmoothEnergy(void) : Energy(true) {}

    static void getSigmaAndGradientPerTri(const TriMesh& data, int triI,
        double& sigma1, double& sigma2,
        Eigen::Vector2d& g_s1_v0, Eigen::Vector2d& g_s1_v1, Eigen::Vector2d& g_s1_v2,
        Eigen::Vector2d& g_s2_v0, Eigen::Vector2d& g_s2_v1, Eigen::Vector2d& g_s2_v2)
    {
        Eigen::Matrix2d J;
        if (!SigmaBoundEnergy::computeJacobianAndSingularValues(data, triI, J, sigma1, sigma2)) {
            g_s1_v0 = g_s1_v1 = g_s1_v2 = g_s2_v0 = g_s2_v1 = g_s2_v2 = Eigen::Vector2d::Zero();
            return;
        }
        Eigen::JacobiSVD<Eigen::Matrix2d> svd(J, Eigen::ComputeFullU | Eigen::ComputeFullV);
        const Eigen::Matrix2d& U = svd.matrixU();
        const Eigen::Matrix2d& V = svd.matrixV();

        const double e0_len = std::sqrt(data.e0SqLen[triI]);
        const double twoA = 2.0 * data.triArea[triI];
        const double alpha = -data.e0dote1[triI] / (twoA * e0_len);
        const double beta = e0_len / twoA;

        Eigen::Matrix2d dS1_dJ = U.col(0) * V.col(0).transpose();
        Eigen::Matrix2d dS2_dJ = U.col(1) * V.col(1).transpose();

        auto gradFromDJD = [&](const Eigen::Matrix2d& dSigma_dJ) {
            Eigen::Vector2d gu0, gu1;
            gu0(0) = dSigma_dJ(0, 0) / e0_len + dSigma_dJ(0, 1) * alpha;
            gu0(1) = dSigma_dJ(1, 0) / e0_len + dSigma_dJ(1, 1) * alpha;
            gu1(0) = dSigma_dJ(0, 1) * beta;
            gu1(1) = dSigma_dJ(1, 1) * beta;
            Eigen::Vector2d gv0 = -(gu0 + gu1);
            Eigen::Vector2d gv1 = gu0;
            Eigen::Vector2d gv2 = gu1;
            return std::make_tuple(gv0, gv1, gv2);
        };

        Eigen::Vector2d g0_1, g1_1, g2_1, g0_2, g1_2, g2_2;
        std::tie(g0_1, g1_1, g2_1) = gradFromDJD(dS1_dJ);
        std::tie(g0_2, g1_2, g2_2) = gradFromDJD(dS2_dJ);
        g_s1_v0 = g0_1; g_s1_v1 = g1_1; g_s1_v2 = g2_1;
        g_s2_v0 = g0_2; g_s2_v1 = g1_2; g_s2_v2 = g2_2;
    }

    static void buildTriNeighbors(const TriMesh& data, std::vector<std::vector<int>>& triNeighbor)
    {
        triNeighbor.resize(data.F.rows());
        for (int i = 0; i < triNeighbor.size(); i++) triNeighbor[i].clear();
        std::set<std::pair<int,int>> done;
        for (int e = 0; e < data.cohE.rows(); e++) {
            int v0 = data.cohE(e, 0), v1 = data.cohE(e, 1), v2 = data.cohE(e, 2), v3 = data.cohE(e, 3);
            int ti = v0 / 3, tj = v2 / 3;
            if (ti == tj) continue;
            if (ti > tj) std::swap(ti, tj);
            if (done.count({ti, tj})) continue;
            done.insert({ti, tj});
            triNeighbor[ti].push_back(tj);
            triNeighbor[tj].push_back(ti);
        }
    }

    void SigmaSmoothEnergy::getEnergyValPerElem(const TriMesh& data, Eigen::VectorXd& energyValPerElem, bool uniformWeight) const
    {
        const double normalizer_div = data.surfaceArea;
        energyValPerElem.resize(data.F.rows());
        for (int i = 0; i < data.F.rows(); i++) energyValPerElem[i] = 0.0;

        std::vector<double> sigma1(data.F.rows(), 0.0), sigma2(data.F.rows(), 0.0);
        for (int triI = 0; triI < data.F.rows(); triI++) {
            Eigen::Matrix2d J;
            if (!SigmaBoundEnergy::computeJacobianAndSingularValues(data, triI, J, sigma1[triI], sigma2[triI]))
                sigma1[triI] = sigma2[triI] = 0.0;
        }

        std::vector<std::vector<int>> triNeighbor;
        buildTriNeighbors(data, triNeighbor);

        for (int ti = 0; ti < data.F.rows(); ti++) {
            for (int tj : triNeighbor[ti]) {
                if (tj < ti) continue;
                double w = (uniformWeight ? 1.0 : 0.5 * (data.triArea[ti] + data.triArea[tj]) / normalizer_div);
                double e = w * ((sigma1[ti] - sigma1[tj]) * (sigma1[ti] - sigma1[tj]) +
                               (sigma2[ti] - sigma2[tj]) * (sigma2[ti] - sigma2[tj]));
                energyValPerElem[ti] += e * 0.5;
                energyValPerElem[tj] += e * 0.5;
            }
        }
    }

    void SigmaSmoothEnergy::getEnergyValByElemID(const TriMesh& data, int elemI, double& energyVal, bool uniformWeight) const
    {
        Eigen::VectorXd ev;
        getEnergyValPerElem(data, ev, uniformWeight);
        energyVal = ev(elemI);
    }

    void SigmaSmoothEnergy::computeGradient(const TriMesh& data, Eigen::VectorXd& gradient, bool uniformWeight) const
    {
        const double normalizer_div = data.surfaceArea;
        gradient.resize(data.V.rows() * 2);
        gradient.setZero();

        const int nTri = data.F.rows();
        std::vector<double> sigma1(nTri), sigma2(nTri);
        std::vector<Eigen::Vector2d> g_s1_v0(nTri), g_s1_v1(nTri), g_s1_v2(nTri);
        std::vector<Eigen::Vector2d> g_s2_v0(nTri), g_s2_v1(nTri), g_s2_v2(nTri);

        for (int triI = 0; triI < nTri; triI++) {
            getSigmaAndGradientPerTri(data, triI, sigma1[triI], sigma2[triI],
                g_s1_v0[triI], g_s1_v1[triI], g_s1_v2[triI],
                g_s2_v0[triI], g_s2_v1[triI], g_s2_v2[triI]);
        }

        std::vector<std::vector<int>> triNeighbor;
        buildTriNeighbors(data, triNeighbor);

        std::vector<double> dE_ds1(nTri, 0.0), dE_ds2(nTri, 0.0);
        for (int ti = 0; ti < nTri; ti++) {
            for (int tj : triNeighbor[ti]) {
                double w = (uniformWeight ? 1.0 : 0.5 * (data.triArea[ti] + data.triArea[tj]) / normalizer_div);
                dE_ds1[ti] += 2.0 * w * (sigma1[ti] - sigma1[tj]);
                dE_ds2[ti] += 2.0 * w * (sigma2[ti] - sigma2[tj]);
            }
        }

        for (int triI = 0; triI < nTri; triI++) {
            const Eigen::Vector3i& triVInd = data.F.row(triI);
            double a = dE_ds1[triI], b = dE_ds2[triI];
            gradient.block<2, 1>(triVInd[0] * 2, 0) += a * g_s1_v0[triI] + b * g_s2_v0[triI];
            gradient.block<2, 1>(triVInd[1] * 2, 0) += a * g_s1_v1[triI] + b * g_s2_v1[triI];
            gradient.block<2, 1>(triVInd[2] * 2, 0) += a * g_s1_v2[triI] + b * g_s2_v2[triI];
        }

        for (const auto fixedVI : data.fixedVert) {
            gradient(2 * fixedVI) = 0.0;
            gradient(2 * fixedVI + 1) = 0.0;
        }
    }

    void SigmaSmoothEnergy::computeHessian(const TriMesh& data, Eigen::VectorXd* V,
                                          Eigen::VectorXi* I, Eigen::VectorXi* J, bool uniformWeight) const
    {
        const double normalizer_div = data.surfaceArea;
        std::vector<Eigen::Triplet<double>> triplets;
        const double scale = 1e-3;
        for (int triI = 0; triI < data.F.rows(); triI++) {
            const Eigen::Vector3i& triVInd = data.F.row(triI);
            double w = (uniformWeight ? 1.0 : data.triArea[triI] / normalizer_div);
            for (int v = 0; v < 3; v++) {
                int vi = triVInd(v);
                if (data.fixedVert.find(vi) != data.fixedVert.end()) continue;
                for (int d = 0; d < 2; d++)
                    triplets.push_back(Eigen::Triplet<double>(vi * 2 + d, vi * 2 + d, w * scale));
            }
        }
        Eigen::SparseMatrix<double> H(data.V.rows() * 2, data.V.rows() * 2);
        H.setFromTriplets(triplets.begin(), triplets.end());
        for (const auto fixedVI : data.fixedVert) {
            H.coeffRef(fixedVI * 2, fixedVI * 2) = 1.0;
            H.coeffRef(fixedVI * 2 + 1, fixedVI * 2 + 1) = 1.0;
        }
        int nnz = 0;
        for (int k = 0; k < H.outerSize(); k++)
            for (Eigen::SparseMatrix<double>::InnerIterator it(H, k); it; ++it) nnz++;
        V->resize(nnz); I->resize(nnz); J->resize(nnz);
        int idx = 0;
        for (int k = 0; k < H.outerSize(); k++)
            for (Eigen::SparseMatrix<double>::InnerIterator it(H, k); it; ++it, idx++) {
                (*V)(idx) = it.value();
                (*I)(idx) = it.row();
                (*J)(idx) = it.col();
            }
    }

    void SigmaSmoothEnergy::computeHessian(const TriMesh& data, Eigen::MatrixXd& Hessian, bool uniformWeight) const
    {
        const double normalizer_div = data.surfaceArea;
        Hessian.resize(data.V.rows() * 2, data.V.rows() * 2);
        Hessian.setZero();
        const double scale = 1e-3;
        for (int triI = 0; triI < data.F.rows(); triI++) {
            const Eigen::Vector3i& triVInd = data.F.row(triI);
            double w = (uniformWeight ? 1.0 : data.triArea[triI] / normalizer_div);
            for (int v = 0; v < 3; v++) {
                int vi = triVInd(v);
                if (data.fixedVert.find(vi) != data.fixedVert.end()) continue;
                for (int d = 0; d < 2; d++) Hessian(vi * 2 + d, vi * 2 + d) += w * scale;
            }
        }
        for (const auto fixedVI : data.fixedVert) {
            Hessian(fixedVI * 2, fixedVI * 2) = 1.0;
            Hessian(fixedVI * 2 + 1, fixedVI * 2 + 1) = 1.0;
        }
    }

    void SigmaSmoothEnergy::checkEnergyVal(const TriMesh& data) const
    {
        Eigen::VectorXd ev;
        getEnergyValPerElem(data, ev);
        std::cout << "SigmaSmoothEnergy total = " << ev.sum() << std::endl;
    }
}
