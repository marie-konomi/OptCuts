//
//  SigmaBoundEnergy.cpp
//  OptCuts
//
//  Penalty for σ1 ≤ sigma1_max, σ2 ≤ sigma2_max (and optional lower bounds).
//

#include "SigmaBoundEnergy.hpp"
#include <Eigen/SVD>
#include <Eigen/Sparse>
#include <cmath>
#include <cassert>

namespace OptCuts {

    SigmaBoundEnergy::SigmaBoundEnergy(double sigma1_max, double sigma2_max,
                                       double sigma1_min, double sigma2_min)
        : Energy(true)
        , sigma1_max_(sigma1_max)
        , sigma2_max_(sigma2_max)
        , sigma1_min_(sigma1_min)
        , sigma2_min_(sigma2_min)
    {
    }

    bool SigmaBoundEnergy::computeJacobianAndSingularValues(const TriMesh& data, int triI,
                                                             Eigen::Matrix2d& J, double& sigma1, double& sigma2)
    {
        const Eigen::Vector3i& triVInd = data.F.row(triI);
        const double A = data.triArea[triI];
        if (A <= 0.0) return false;

        const Eigen::RowVector3d& P1 = data.V_rest.row(triVInd[0]);
        const Eigen::RowVector3d& P2 = data.V_rest.row(triVInd[1]);
        const Eigen::RowVector3d& P3 = data.V_rest.row(triVInd[2]);
        Eigen::Vector3d e0 = (P2 - P1).transpose();
        Eigen::Vector3d e1 = (P3 - P1).transpose();

        const double e0_len = std::sqrt(data.e0SqLen[triI]);
        if (e0_len <= 0.0) return false;
        const double twoA = 2.0 * A;
        if (twoA <= 0.0) return false;

        const Eigen::RowVector2d& U1 = data.V.row(triVInd[0]);
        const Eigen::RowVector2d& U2 = data.V.row(triVInd[1]);
        const Eigen::RowVector2d& U3 = data.V.row(triVInd[2]);
        Eigen::Vector2d u0 = (U2 - U1).transpose();
        Eigen::Vector2d u1 = (U3 - U1).transpose();

        const double e0dote1 = data.e0dote1[triI];
        const double alpha = -e0dote1 / (twoA * e0_len);
        const double beta = e0_len / twoA;

        J.col(0) = u0 / e0_len;
        J.col(1) = alpha * u0 + beta * u1;

        Eigen::JacobiSVD<Eigen::Matrix2d> svd(J, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Vector2d sv = svd.singularValues();
        sigma1 = sv(0);
        sigma2 = sv(1);
        if (sigma1 < sigma2) std::swap(sigma1, sigma2);
        return true;
    }

    void SigmaBoundEnergy::getEnergyValPerElem(const TriMesh& data, Eigen::VectorXd& energyValPerElem, bool uniformWeight) const
    {
        const double normalizer_div = data.surfaceArea;
        energyValPerElem.resize(data.F.rows());

        for (int triI = 0; triI < data.F.rows(); triI++) {
            Eigen::Matrix2d J;
            double sigma1, sigma2;
            if (!computeJacobianAndSingularValues(data, triI, J, sigma1, sigma2)) {
                energyValPerElem[triI] = 0.0;
                continue;
            }

            double p = 0.0;
            if (sigma1_max_ > 0.0 && sigma1 > sigma1_max_)
                p += (sigma1 - sigma1_max_) * (sigma1 - sigma1_max_);
            if (sigma2_max_ > 0.0 && sigma2 > sigma2_max_)
                p += (sigma2 - sigma2_max_) * (sigma2 - sigma2_max_);
            if (sigma1_min_ > 0.0 && sigma1 < sigma1_min_)
                p += (sigma1_min_ - sigma1) * (sigma1_min_ - sigma1);
            if (sigma2_min_ > 0.0 && sigma2 < sigma2_min_)
                p += (sigma2_min_ - sigma2) * (sigma2_min_ - sigma2);

            const double w = (uniformWeight ? 1.0 : (data.triArea[triI] / normalizer_div));
            energyValPerElem[triI] = w * p;
        }
    }

    void SigmaBoundEnergy::getEnergyValByElemID(const TriMesh& data, int elemI, double& energyVal, bool uniformWeight) const
    {
        Eigen::VectorXd ev;
        getEnergyValPerElem(data, ev, uniformWeight);
        energyVal = ev(elemI);
    }

    void SigmaBoundEnergy::computeGradient(const TriMesh& data, Eigen::VectorXd& gradient, bool uniformWeight) const
    {
        const double normalizer_div = data.surfaceArea;
        gradient.resize(data.V.rows() * 2);
        gradient.setZero();

        for (int triI = 0; triI < data.F.rows(); triI++) {
            const Eigen::Vector3i& triVInd = data.F.row(triI);
            Eigen::Matrix2d J;
            double sigma1, sigma2;
            if (!computeJacobianAndSingularValues(data, triI, J, sigma1, sigma2)) continue;

            Eigen::JacobiSVD<Eigen::Matrix2d> svd(J, Eigen::ComputeFullU | Eigen::ComputeFullV);
            const Eigen::Matrix2d& U = svd.matrixU();
            const Eigen::Matrix2d& V = svd.matrixV();

            double dP_ds1 = 0.0, dP_ds2 = 0.0;
            if (sigma1_max_ > 0.0 && sigma1 > sigma1_max_) dP_ds1 += 2.0 * (sigma1 - sigma1_max_);
            if (sigma2_max_ > 0.0 && sigma2 > sigma2_max_) dP_ds2 += 2.0 * (sigma2 - sigma2_max_);
            if (sigma1_min_ > 0.0 && sigma1 < sigma1_min_) dP_ds1 -= 2.0 * (sigma1_min_ - sigma1);
            if (sigma2_min_ > 0.0 && sigma2 < sigma2_min_) dP_ds2 -= 2.0 * (sigma2_min_ - sigma2);

            Eigen::Matrix2d dP_dJ = dP_ds1 * U.col(0) * V.col(0).transpose() + dP_ds2 * U.col(1) * V.col(1).transpose();

            const double A = data.triArea[triI];
            const double e0_len = std::sqrt(data.e0SqLen[triI]);
            const double twoA = 2.0 * A;
            const double alpha = -data.e0dote1[triI] / (twoA * e0_len);
            const double beta = e0_len / twoA;

            const double w = (uniformWeight ? 1.0 : (data.triArea[triI] / normalizer_div));

            Eigen::Vector2d g_u0, g_u1;
            g_u0(0) = dP_dJ(0, 0) / e0_len + dP_dJ(0, 1) * alpha;
            g_u0(1) = dP_dJ(1, 0) / e0_len + dP_dJ(1, 1) * alpha;
            g_u1(0) = dP_dJ(0, 1) * beta;
            g_u1(1) = dP_dJ(1, 1) * beta;

            gradient.block<2, 1>(triVInd[0] * 2, 0) -= w * (g_u0 + g_u1);
            gradient.block<2, 1>(triVInd[1] * 2, 0) += w * g_u0;
            gradient.block<2, 1>(triVInd[2] * 2, 0) += w * g_u1;
        }

        for (const auto fixedVI : data.fixedVert) {
            gradient(2 * fixedVI) = 0.0;
            gradient(2 * fixedVI + 1) = 0.0;
        }
    }

    void SigmaBoundEnergy::computeHessian(const TriMesh& data, Eigen::VectorXd* V,
                                         Eigen::VectorXi* I, Eigen::VectorXi* J, bool uniformWeight) const
    {
        const double normalizer_div = data.surfaceArea;
        std::vector<Eigen::Triplet<double>> triplets;

        for (int triI = 0; triI < data.F.rows(); triI++) {
            const Eigen::Vector3i& triVInd = data.F.row(triI);
            Eigen::Matrix2d J;
            double sigma1, sigma2;
            if (!computeJacobianAndSingularValues(data, triI, J, sigma1, sigma2)) continue;

            const double w = (uniformWeight ? 1.0 : (data.triArea[triI] / normalizer_div));
            const double scale = w * 1e-2;
            for (int v = 0; v < 3; v++) {
                int vi = triVInd(v);
                if (data.fixedVert.find(vi) != data.fixedVert.end()) continue;
                for (int d = 0; d < 2; d++) {
                    int idx = vi * 2 + d;
                    triplets.push_back(Eigen::Triplet<double>(idx, idx, scale));
                }
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
        V->resize(nnz);
        I->resize(nnz);
        J->resize(nnz);
        int idx = 0;
        for (int k = 0; k < H.outerSize(); k++) {
            for (Eigen::SparseMatrix<double>::InnerIterator it(H, k); it; ++it, idx++) {
                (*V)(idx) = it.value();
                (*I)(idx) = it.row();
                (*J)(idx) = it.col();
            }
        }
    }

    void SigmaBoundEnergy::computeHessian(const TriMesh& data, Eigen::MatrixXd& Hessian, bool uniformWeight) const
    {
        const double normalizer_div = data.surfaceArea;
        Hessian.resize(data.V.rows() * 2, data.V.rows() * 2);
        Hessian.setZero();

        for (int triI = 0; triI < data.F.rows(); triI++) {
            const Eigen::Vector3i& triVInd = data.F.row(triI);
            Eigen::Matrix2d J;
            double sigma1, sigma2;
            if (!computeJacobianAndSingularValues(data, triI, J, sigma1, sigma2)) continue;

            const double w = (uniformWeight ? 1.0 : (data.triArea[triI] / normalizer_div));
            const double scale = w * 1e-2;
            for (int v = 0; v < 3; v++) {
                int vi = triVInd(v);
                if (data.fixedVert.find(vi) != data.fixedVert.end()) continue;
                for (int d = 0; d < 2; d++) {
                    int idx = vi * 2 + d;
                    Hessian(idx, idx) += scale;
                }
            }
        }
        for (const auto fixedVI : data.fixedVert) {
            Hessian(fixedVI * 2, fixedVI * 2) = 1.0;
            Hessian(fixedVI * 2 + 1, fixedVI * 2 + 1) = 1.0;
        }
    }

    void SigmaBoundEnergy::checkEnergyVal(const TriMesh& data) const
    {
        Eigen::VectorXd ev;
        getEnergyValPerElem(data, ev);
        double total = ev.sum();
    }

    void SigmaBoundEnergy::getMaxSingularValues(const TriMesh& data, double& max_sigma1, double& max_sigma2)
    {
        max_sigma1 = 0.0;
        max_sigma2 = 0.0;
        for (int triI = 0; triI < data.F.rows(); triI++) {
            Eigen::Matrix2d J;
            double sigma1, sigma2;
            if (!computeJacobianAndSingularValues(data, triI, J, sigma1, sigma2)) continue;
            if (sigma1 > max_sigma1) max_sigma1 = sigma1;
            if (sigma2 > max_sigma2) max_sigma2 = sigma2;
        }
    }
}